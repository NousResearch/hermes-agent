"""
Markdown report generator for benchmark results.
Pure Python stdlib only.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


def _fmt(val, decimals: int = 3) -> str:
    """Format a float or return N/A."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _pct(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(val)


def _metric_label(key: str) -> str:
    labels = {
        "recall_at_1":       "Recall@1",
        "recall_at_3":       "Recall@3",
        "recall_at_5":       "Recall@5",
        "recall_at_10":      "Recall@10",
        "mrr":               "MRR",
        "ndcg_at_5":         "NDCG@5",
        "ndcg_at_10":        "NDCG@10",
        "average_precision": "Avg Precision",
        "exact_match":       "Exact Match",
        "token_f1":          "Token F1",
        "token_precision":   "Token Precision",
        "token_recall":      "Token Recall",
    }
    return labels.get(key, key.replace("_", " ").title())


def _retrieval_table_md(metrics: dict) -> str:
    """Render a markdown table from retrieval_metrics dict."""
    order = [
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "mrr", "ndcg_at_5", "ndcg_at_10", "average_precision",
        "exact_match", "token_f1", "token_precision", "token_recall",
    ]
    extra = [k for k in metrics if k not in order]
    keys = [k for k in order if k in metrics] + extra

    lines = ["| Metric | Value |", "| --- | --- |"]
    for k in keys:
        lines.append(f"| {_metric_label(k)} | {_fmt(metrics[k])} |")
    return "\n".join(lines)


def _category_table_md(per_cat_mean: dict, per_cat_std: dict = None,
                        runs_cats: dict = None) -> str:
    """Render a markdown table of per-category scores, sorted desc."""
    if not per_cat_mean:
        return "_No category data available._"

    per_cat_std = per_cat_std or {}

    # Build rows: (name, score, std, correct, total)
    rows = []
    for cat, score in per_cat_mean.items():
        std = per_cat_std.get(cat)
        correct = total = None
        if runs_cats and cat in runs_cats:
            correct = runs_cats[cat].get("correct")
            total   = runs_cats[cat].get("total")
        rows.append((cat, float(score), std, correct, total))

    rows.sort(key=lambda r: r[1], reverse=True)

    has_ci   = any(r[2] is not None and r[2] > 0 for r in rows)
    has_frac = any(r[3] is not None for r in rows)

    headers = ["Category", "Score"]
    if has_ci:
        headers.append("Std Dev")
    if has_frac:
        headers.append("Correct/Total")

    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep)    + " |",
    ]

    for cat, score, std, correct, total in rows:
        cols = [cat, _fmt(score)]
        if has_ci:
            cols.append(_fmt(std) if std is not None else "0.000")
        if has_frac:
            if correct is not None and total is not None:
                cols.append(f"{correct}/{total}")
            else:
                cols.append("N/A")
        lines.append("| " + " | ".join(cols) + " |")

    return "\n".join(lines)


def generate_report(result_json: dict, output_path: str = None) -> str:
    """
    Generate a full markdown report from a benchmark result dict.

    Parameters
    ----------
    result_json : dict
        The parsed JSON from a benchmark results file.
    output_path : str, optional
        If provided, write the markdown to this path.

    Returns
    -------
    str
        The full markdown report.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    backend   = result_json.get("backend", "unknown")
    profile   = result_json.get("profile", "")
    emb_model = result_json.get("embedding_model", "")
    overall   = result_json.get("mean_score", 0.0)
    std       = result_json.get("std", 0.0)
    ci        = result_json.get("ci_95", [overall, overall])
    num_runs  = result_json.get("num_runs", 1)
    per_cat   = result_json.get("per_category_mean", {})
    per_std   = result_json.get("per_category_std", {})
    ret_m     = result_json.get("retrieval_metrics", {})
    runs      = result_json.get("runs", [])

    # first-run details
    first_run = runs[0] if runs else {}
    cats_data = first_run.get("categories", {})
    wall_time = first_run.get("wall_time_seconds")
    token_use = first_run.get("token_usage", {})
    total_scenarios = None
    if cats_data and len(cats_data) == len(per_cat):
        total_scenarios = sum(c.get("total", 0) for c in cats_data.values())

    lines = []

    # ── Header ──────────────────────────────────────────────
    lines.append(f"# Benchmark Report: {backend}")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")

    # ── Run config ──────────────────────────────────────────
    lines.append("## Run Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Backend | `{backend}` |")
    if profile:
        lines.append(f"| Profile | `{profile}` |")
    if emb_model:
        lines.append(f"| Embedding Model | `{emb_model}` |")
    lines.append(f"| Num Runs | {num_runs} |")
    lines.append(f"| Total Scenarios | {total_scenarios if total_scenarios is not None else 'N/A (partial run details)'} |")
    lines.append(f"| Categories | {len(per_cat)} |")
    if wall_time is not None:
        lines.append(f"| Wall Time | {wall_time:.3f}s |")
    if token_use:
        lines.append(f"| Total Tokens | {token_use.get('recall_tokens', 'N/A')} |")
        lines.append(f"| Total Queries | {token_use.get('recall_queries', 'N/A')} |")
        avg = token_use.get("avg_recall_tokens_per_query")
        if avg is not None:
            lines.append(f"| Avg Tokens/Query | {avg} |")
    lines.append("")

    # ── Overall score ────────────────────────────────────────
    lines.append("## Overall Score")
    lines.append("")
    ci_str = ""
    if ci[0] != ci[1]:
        ci_str = f" (95% CI: [{_fmt(ci[0])}, {_fmt(ci[1])}])"
    std_str = f" ± {_fmt(std)}" if std > 0 else ""
    lines.append(f"**{_fmt(overall)} ({_pct(overall)})**{std_str}{ci_str}")
    lines.append("")

    # ── Per-category scores ──────────────────────────────────
    lines.append("## Per-Category Scores")
    lines.append("")
    lines.append(_category_table_md(per_cat, per_std, cats_data))
    lines.append("")

    # ── Retrieval metrics ────────────────────────────────────────
    if ret_m:
        lines.append("## Retrieval Metrics")
        lines.append("")
        lines.append(_retrieval_table_md(ret_m))
        lines.append("")

    # ── Cost Efficiency ──────────────────────────────────────────
    # Aggregate cost_metrics from individual runs (most accurate)
    cost_m = result_json.get("cost_metrics", {})
    if not cost_m and runs:
        # fall back to computing from per-run cost_metrics
        run_cms = [r.get("cost_metrics", {}) for r in runs if r.get("cost_metrics")]
        if run_cms:
            cost_m = {}
            for key in run_cms[0]:
                vals = [m[key] for m in run_cms if key in m]
                cost_m[key] = sum(vals) / len(vals) if vals else 0.0
    if cost_m:
        lines.append("## Cost Efficiency")
        lines.append("")
        lines.append("Token cost normalized by answer quality.")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        tpq = cost_m.get("tokens_per_query")
        tpc = cost_m.get("tokens_per_correct")
        eff = cost_m.get("cost_efficiency")
        score = cost_m.get("score")
        if tpq is not None:
            lines.append(f"| Tokens / Query | ~{tpq:.0f} |")
        if tpc is not None:
            lines.append(f"| Tokens / Correct Answer | ~{tpc:.0f} |")
        if eff is not None:
            lines.append(f"| Cost Efficiency (score / log2(tokens)) | {_fmt(eff)} |")
        if score is not None:
            lines.append(f"| Score (correct / total) | {_fmt(score)} ({_pct(score)}) |")
        lines.append("")
        lines.append("> **Note:** Efficiency = score / log2(tokens_per_query + 1). "
                     "Higher is better. Doubling tokens does not halve efficiency "
                     "due to logarithmic normalization.")
        lines.append("")

    # ── Weakest categories ───────────────────────────────────────
    if per_cat:
        sorted_cats = sorted(per_cat.items(), key=lambda x: x[1])
        bottom = sorted_cats[:3]
        lines.append("## Weakest Categories (Bottom 3)")
        lines.append("")
        lines.append("These categories have the most room for improvement:")
        lines.append("")
        for cat, score in bottom:
            correct = cats_data.get(cat, {}).get("correct")
            total   = cats_data.get(cat, {}).get("total")
            frac = f" ({correct}/{total})" if correct is not None else ""
            lines.append(f"- **{cat}**: {_fmt(score)} ({_pct(score)}){frac}")
        lines.append("")

    # ── Footer ──────────────────────────────────────────────
    lines.append("---")
    lines.append(f"_Report generated by hermes-agent benchmark suite_")

    md = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(md)

    return md


def generate_comparison_report(before_json: dict, after_json: dict) -> str:
    """
    Markdown report comparing two benchmark runs.

    Parameters
    ----------
    before_json : dict
        Benchmark result for the baseline run.
    after_json : dict
        Benchmark result for the updated run.

    Returns
    -------
    str
        Markdown comparison report.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    b_backend = before_json.get("backend", "before")
    a_backend = after_json.get("backend", "after")

    b_score   = before_json.get("mean_score", 0.0)
    a_score   = after_json.get("mean_score", 0.0)
    delta     = a_score - b_score
    delta_pct = delta * 100

    b_cat = before_json.get("per_category_mean", {})
    a_cat = after_json.get("per_category_mean",  {})
    b_ret = before_json.get("retrieval_metrics", {})
    a_ret = after_json.get("retrieval_metrics",  {})

    lines = []

    # ── Header ──────────────────────────────────────────────
    lines.append(f"# Benchmark Comparison Report")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append(f"| | Before (`{b_backend}`) | After (`{a_backend}`) | Delta |")
    lines.append("| --- | --- | --- | --- |")

    sign = "+" if delta >= 0 else ""
    arrow = " ▲" if delta > 0.001 else (" ▼" if delta < -0.001 else " →")
    lines.append(
        f"| **Overall Score** | {_fmt(b_score)} ({_pct(b_score)}) "
        f"| {_fmt(a_score)} ({_pct(a_score)}) "
        f"| {sign}{_fmt(delta)} ({sign}{delta_pct:.1f}pp){arrow} |"
    )
    lines.append("")

    # ── Per-category comparison ──────────────────────────────
    lines.append("## Per-Category Comparison")
    lines.append("")

    all_cats = sorted(set(b_cat.keys()) | set(a_cat.keys()))
    if all_cats:
        lines.append("| Category | Before | After | Delta | Direction |")
        lines.append("| --- | --- | --- | --- | --- |")

        for cat in sorted(all_cats, key=lambda c: a_cat.get(c, b_cat.get(c, 0)), reverse=True):
            bv = b_cat.get(cat)
            av = a_cat.get(cat)

            b_str = _fmt(bv) if bv is not None else "N/A"
            a_str = _fmt(av) if av is not None else "N/A"

            if bv is not None and av is not None:
                d = av - bv
                ds = "+" if d >= 0 else ""
                if abs(d) >= 0.1:
                    direction = "▲▲" if d > 0 else "▼▼"
                elif abs(d) >= 0.02:
                    direction = "▲" if d > 0 else "▼"
                elif abs(d) > 0.001:
                    direction = "↑" if d > 0 else "↓"
                else:
                    direction = "="
                lines.append(f"| {cat} | {b_str} | {a_str} | {ds}{_fmt(d)} | {direction} |")
            else:
                lines.append(f"| {cat} | {b_str} | {a_str} | N/A | - |")
    lines.append("")

    # ── Retrieval metrics comparison ─────────────────────────
    if b_ret or a_ret:
        lines.append("## Retrieval Metrics Comparison")
        lines.append("")

        all_ret = [k for k in [
            "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
            "mrr", "ndcg_at_5", "ndcg_at_10", "average_precision",
            "exact_match", "token_f1", "token_precision", "token_recall",
        ] if k in b_ret or k in a_ret]

        extra = [k for k in (set(b_ret) | set(a_ret)) if k not in all_ret]
        all_ret += extra

        if all_ret:
            lines.append("| Metric | Before | After | Delta |")
            lines.append("| --- | --- | --- | --- |")
            for key in all_ret:
                bv = b_ret.get(key)
                av = a_ret.get(key)
                b_str = _fmt(bv) if bv is not None else "N/A"
                a_str = _fmt(av) if av is not None else "N/A"
                if bv is not None and av is not None:
                    d = av - bv
                    ds = "+" if d >= 0 else ""
                    lines.append(f"| {_metric_label(key)} | {b_str} | {a_str} | {ds}{_fmt(d)} |")
                else:
                    lines.append(f"| {_metric_label(key)} | {b_str} | {a_str} | N/A |")
        lines.append("")

    # ── Notable changes ──────────────────────────────────────
    if b_cat and a_cat:
        common = [(c, a_cat[c] - b_cat[c]) for c in b_cat if c in a_cat]
        improved  = sorted([(c, d) for c, d in common if d >  0.001], key=lambda x: -x[1])
        regressed = sorted([(c, d) for c, d in common if d < -0.001], key=lambda x:  x[1])

        if improved:
            lines.append("## Most Improved Categories")
            lines.append("")
            for cat, d in improved[:5]:
                lines.append(f"- **{cat}**: +{_fmt(d)} ({_pct(abs(d))} gain)")
            lines.append("")

        if regressed:
            lines.append("## Regressed Categories")
            lines.append("")
            for cat, d in regressed[:5]:
                lines.append(f"- **{cat}**: {_fmt(d)} ({_pct(abs(d))} loss)")
            lines.append("")

    # ── Footer ──────────────────────────────────────────────
    lines.append("---")
    lines.append("_Comparison report generated by hermes-agent benchmark suite_")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# CLI self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib

    results_path = pathlib.Path(__file__).parent.parent / "results" / "baseline-flat.json"
    if not results_path.exists():
        print(f"Result file not found: {results_path}")
        raise SystemExit(1)

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    md = generate_report(data, output_path=None)
    print(md)
    print("\n\n" + "=" * 60 + "\n")

    # Create a fake 'before' by reducing scores slightly
    import copy
    before = copy.deepcopy(data)
    before["mean_score"] = max(0.0, data.get("mean_score", 0) - 0.05)
    for cat in before.get("per_category_mean", {}):
        before["per_category_mean"][cat] = max(
            0.0, before["per_category_mean"][cat] - 0.05
        )

    cmp_md = generate_comparison_report(before, data)
    print(cmp_md)
