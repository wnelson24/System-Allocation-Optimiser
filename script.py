import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from io import StringIO

# === Optimisation Methods ===

def kelly_weights(returns):
    mean = returns.mean()
    cov = returns.cov()
    if np.linalg.det(cov.values) == 0:
        cov += np.eye(cov.shape[0]) * 1e-5
    weights = np.linalg.pinv(cov.values) @ mean.values
    weights = weights / np.sum(np.abs(weights))
    return pd.Series(weights, index=returns.columns)

def get_quasi_diag(link, labels):
    sort_ix = [int(link[-1, 0]), int(link[-1, 1])]
    num_items = link.shape[0] + 1
    while any(i >= num_items for i in sort_ix):
        new_sort_ix = []
        for i in sort_ix:
            if i < num_items:
                new_sort_ix.append(i)
            else:
                j = int(i - num_items)
                new_sort_ix.extend([int(link[j, 0]), int(link[j, 1])])
        sort_ix = new_sort_ix
    return labels[sort_ix].tolist()

def get_recursive_bisection(cov, ordered_assets):
    weights = pd.Series(1.0, index=ordered_assets)
    clusters = [ordered_assets]
    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            w_left = 1.0 / np.sum(1.0 / np.diag(cov.loc[left, left]))
            w_right = 1.0 / np.sum(1.0 / np.diag(cov.loc[right, right]))
            alpha = w_right / (w_left + w_right)
            weights[left] *= alpha
            weights[right] *= 1 - alpha
            new_clusters += [left, right]
        clusters = new_clusters
    return weights / weights.sum()

def hrp_weights(returns):
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    link = linkage(squareform(dist), method='single')
    sorted_idx = get_quasi_diag(link, returns.columns)
    cov = returns.cov()
    return get_recursive_bisection(cov, sorted_idx)

def cvar_weights(returns, alpha=0.95):
    T, N = returns.shape
    w = cp.Variable(N)
    z = cp.Variable(T)
    var = cp.Variable()
    loss = -returns.values @ w
    cvar = var + (1 / (1 - alpha)) * cp.sum(z) / T
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        z >= 0,
        z >= loss - var
    ]
    prob = cp.Problem(cp.Minimize(cvar), constraints)
    prob.solve()
    return pd.Series(w.value, index=returns.columns)

def combine_all_weights(kelly, hrp, cvar, blend=(0.5, 0.3, 0.2)):
    combined = (
        blend[0] * kelly +
        blend[1] * hrp +
        blend[2] * cvar
    )
    return combined / combined.sum()

# === Parser for QC-style Data ===

def parse_equity_series(raw_text):
    df = pd.read_csv(StringIO(raw_text), sep=r'\s+|\t+|,', engine='python', header=None)
    df[0] = pd.to_datetime(df[0])
    equity_series = pd.Series(df.iloc[:, 1].values, index=df[0])
    return equity_series

# === Main Workflow ===

def run_weightifier_with_date_alignment():
    print("📊 QuantConnect Portfolio Weightifier (Date-Aligned)")
    print("=====================================================")
    n = int(input("How many systems in your portfolio? "))

    equity_curves = {}
    for i in range(n):
        print(f"\n📋 Paste QC equity table for System_{i+1} (Date + 4 columns: 1M, 3M, 6M, 12M)")
        print("🔹 Press Enter TWICE to finish input:")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        raw_data = "\n".join(lines)
        series = parse_equity_series(raw_data)
        equity_curves[f"System_{i+1}"] = series

    df = pd.concat(equity_curves.values(), axis=1, join="inner")
    df.columns = equity_curves.keys()
    returns = df.pct_change().dropna()

    kelly_w = kelly_weights(returns)
    hrp_w = hrp_weights(returns)
    cvar_w = cvar_weights(returns)
    combo_w = combine_all_weights(kelly_w, hrp_w, cvar_w)

    result_df = pd.DataFrame({
        "Kelly": kelly_w,
        "HRP": hrp_w,
        "CVaR": cvar_w,
        "Composite": combo_w
    })

    print("\n✅ Final Portfolio Allocation Weights:\n")
    print(result_df.round(4).to_string())

if __name__ == "__main__":
    run_weightifier_with_date_alignment()
