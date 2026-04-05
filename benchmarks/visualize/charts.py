"""
ASCII/text-based chart generators for benchmark results.
Pure Python stdlib only - no matplotlib, no plotly.
"""

import os
import json
from datetime import datetime

# ANSI color codes - only use when stdout is a tty and NO_COLOR is unset
_USE_COLOR = (os.environ.get("NO_COLOR", "") == ""
              and os.environ.get("TERM", "") != "dumb")

_RESET  = "\033[0m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"


def _c(code: str) -> str:
    """Return ANSI code if colors are enabled, else empty string."""
    return code if _USE_COLOR else ""


def _score_color(score: float) -> str:
    if score >= 0.9:
        return _c(_GREEN)
    elif score >= 0.75:
        return _c(_YELLOW)
    else:
        return _c(_RED)


def _bar(score: float, width: int = 40, fill: str = "\u2588", empty: str = " ") -> str:
    """Render a filled bar of given width representing 0..1 score."""
    filled = round(score * width)
    filled = max(0, min(width, filled))
    return fill * filled + empty * (width - filled)


def radar_chart(categories: dict, title: str = "") -> str:
    """
    Bar-chart representation of per-category scores.

    Example:
      semantic_recall     |████████████████████| 1.000
      temporal_decay      |████████████████████| 0.956
    """
    lines = []
    bar_width = 40

    if title:
        lines.append(f"{_c(_BOLD)}{title}{_c(_RESET)}")
        lines.append("")

    if not categories:
        lines.append("(no categories)")
        return "\n".join(lines)

    max_name_len = max(len(k) for k in categories)
    max_name_len = max(max_name_len, 8)

    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)

    for name, score in sorted_cats:
        score = float(score)
        bar = _bar(score, bar_width)
        color = _score_color(score)
        name_col = name.ljust(max_name_len)
        lines.append(
            f"  {name_col} |{color}{bar}{_c(_RESET)}| {score:.3f}"
        )

    return "\n".join(lines)


def comparison_chart(before: dict, after: dict, title: str = "") -> str:
    """
    Side-by-side comparison of two result dicts (per_category_mean).

    Example:
      Category            Before    After     Delta
      semantic_recall      0.920     1.000    +0.080 ▲
    """
    lines = []

    if title:
        lines.append(f"{_c(_BOLD)}{title}{_c(_RESET)}")
        lines.append("")

    all_keys = sorted(set(before.keys()) | set(after.keys()))
    if not all_keys:
        lines.append("(no data)")
        return "\n".join(lines)

    max_name = max(len(k) for k in all_keys)
    max_name = max(max_name, 8)

    header = (
        f"  {'Category'.ljust(max_name)}  {'Before':>8}  {'After':>8}  {'Delta':>10}"
    )
    sep = "  " + "-" * (max_name + 34)
    lines.append(f"{_c(_BOLD)}{header}{_c(_RESET)}")
    lines.append(sep)

    for key in sorted(all_keys, key=lambda k: after.get(k, before.get(k, 0)), reverse=True):
        b_val = before.get(key)
        a_val = after.get(key)

        b_str = f"{b_val:.3f}" if b_val is not None else "  N/A "
        a_str = f"{a_val:.3f}" if a_val is not None else "  N/A "

        if b_val is not None and a_val is not None:
            delta = a_val - b_val
            sign = "+" if delta >= 0 else ""
            arrows = ""
            if abs(delta) >= 0.2:
                arrows = "\u25b2\u25b2" if delta > 0 else "\u25bc\u25bc"
            elif abs(delta) >= 0.05:
                arrows = "\u25b2" if delta > 0 else "\u25bc"
            elif abs(delta) > 0.001:
                arrows = "\u25b2" if delta > 0 else "\u25bc"

            if _USE_COLOR:
                color = _GREEN if delta > 0.001 else (_RED if delta < -0.001 else _DIM)
                delta_str = f"{_c(color)}{sign}{delta:.3f} {arrows}{_c(_RESET)}"
            else:
                delta_str = f"{sign}{delta:.3f} {arrows}"
        else:
            delta_str = "  ---"

        name_col = key.ljust(max_name)
        lines.append(f"  {name_col}  {b_str:>8}  {a_str:>8}  {delta_str}")

    return "\n".join(lines)


def metrics_table(metrics: dict, title: str = "") -> str:
    """
    Formatted table of retrieval metrics.

    Example:
      Metric          Value
      ─────────────── ──────
      Recall@1        0.513
    """
    lines = []

    if title:
        lines.append(f"{_c(_BOLD)}{title}{_c(_RESET)}")
        lines.append("")

    label_map = {
        "recall_at_1":         "Recall@1",
        "recall_at_3":         "Recall@3",
        "recall_at_5":         "Recall@5",
        "recall_at_10":        "Recall@10",
        "mrr":                 "MRR",
        "ndcg_at_5":           "NDCG@5",
        "ndcg_at_10":          "NDCG@10",
        "average_precision":   "Avg Precision",
        "exact_match":         "Exact Match",
        "token_f1":            "Token F1",
        "token_precision":     "Token Precision",
        "token_recall":        "Token Recall",
    }

    display_order = [
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "mrr", "ndcg_at_5", "ndcg_at_10", "average_precision",
        "exact_match", "token_f1", "token_precision", "token_recall",
    ]
    extra = [k for k in metrics if k not in display_order]
    ordered_keys = [k for k in display_order if k in metrics] + extra

    if not ordered_keys:
        lines.append("(no metrics)")
        return "\n".join(lines)

    max_label = max(len(label_map.get(k, k)) for k in ordered_keys)
    max_label = max(max_label, 6)

    header_metric = "Metric".ljust(max_label)
    header_value  = "Value"
    lines.append(f"  {_c(_BOLD)}{header_metric}  {header_value}{_c(_RESET)}")
    lines.append("  " + "\u2500" * max_label + "  " + "\u2500" * 7)

    for key in ordered_keys:
        label = label_map.get(key, key).ljust(max_label)
        val   = metrics[key]
        val_str = f"{val:.3f}"
        lines.append(f"  {label}  {val_str}")

    return "\n".join(lines)


def regression_chart(history: list, metric: str = "overall") -> str:
    """
    Show score history over time as horizontal bars.

    history : list of result dicts (or raw float values)
    metric  : 'overall' uses mean_score; pass any other key for that field

    Example:
      Score over time (overall):
      Run 1: ████████████████████████████████████████████████ 0.836
    """
    lines = []
    lines.append(f"{_c(_BOLD)}Score over time ({metric}):{_c(_RESET)}")

    if not history:
        lines.append("  (no history)")
        return "\n".join(lines)

    key_map = {
        "overall": "mean_score",
        "mean_score": "mean_score",
    }
    actual_key = key_map.get(metric, metric)

    scores = []
    for entry in history:
        if isinstance(entry, dict):
            val = entry.get(actual_key,
                  entry.get("mean_score",
                  entry.get("overall_score")))
        else:
            val = entry
        scores.append(float(val) if val is not None else 0.0)

    max_bar   = 60
    max_score = max(scores) if scores else 1.0
    max_score = max(max_score, 0.001)

    for i, score in enumerate(scores, start=1):
        filled = round((score / max_score) * max_bar)
        bar    = "\u2588" * filled
        color  = _score_color(score)
        label  = f"Run {i}:"
        lines.append(f"  {label:<7} {color}{bar}{_c(_RESET)} {score:.3f}")

    return "\n".join(lines)


def summary_dashboard(result_json: dict) -> str:
    """
    All-in-one summary combining radar + metrics + key stats.
    Returns the full dashboard as a string.
    """
    lines = []
    width = 54

    backend   = result_json.get("backend", "unknown")
    overall   = result_json.get("mean_score", 0.0)
    std       = result_json.get("std", 0.0)
    ci        = result_json.get("ci_95", [overall, overall])
    num_runs  = result_json.get("num_runs", 1)
    per_cat   = result_json.get("per_category_mean", {})
    ret_m     = result_json.get("retrieval_metrics", {})
    profile   = result_json.get("profile", "")

    total_scenarios = 0
    runs = result_json.get("runs", [])
    if runs:
        cats_data = runs[0].get("categories", {})
        for cat_info in cats_data.values():
            total_scenarios += cat_info.get("total", 0)

    border_h = "\u2550" * width
    border_s = "\u2500" * width

    lines.append(border_h)
    title_str = f" BENCHMARK DASHBOARD: {backend.upper()}"
    if profile:
        title_str += f" [{profile}]"
    lines.append(f"{_c(_BOLD)}{title_str}{_c(_RESET)}")
    lines.append(border_h)

    pct = overall * 100
    ci_str = (f"CI: [{ci[0]:.3f}, {ci[1]:.3f}]"
              if ci[0] != ci[1] else "")
    overall_color = _score_color(overall)
    lines.append(
        f" Overall: {overall_color}{overall:.3f} ({pct:.1f}%){_c(_RESET)}"
        + (f" \u00b1{std:.3f}" if std > 0 else "")
        + f"  |  Runs: {num_runs}"
    )
    if ci_str:
        lines.append(f" {ci_str}")
    lines.append(
        f" Categories: {len(per_cat)}  |  Scenarios: {total_scenarios}"
    )
    if profile:
        lines.append(f" Profile: {profile}")

    lines.append(border_s)
    lines.append(f"{_c(_BOLD)} [Category Scores]{_c(_RESET)}")
    lines.append("")
    lines.append(radar_chart(per_cat))
    lines.append("")

    if ret_m:
        lines.append(border_s)
        lines.append(f"{_c(_BOLD)} [Retrieval Metrics]{_c(_RESET)}")
        lines.append("")
        lines.append(metrics_table(ret_m))
        lines.append("")

    if runs:
        tu = runs[0].get("token_usage", {})
        wt = runs[0].get("wall_time_seconds")
        if tu or wt:
            lines.append(border_s)
            lines.append(f"{_c(_BOLD)} [Performance]{_c(_RESET)}")
            if wt is not None:
                lines.append(f"  Wall time:         {wt:.3f}s")
            if tu:
                lines.append(f"  Total tokens:      {tu.get('recall_tokens', 'N/A')}")
                lines.append(f"  Queries:           {tu.get('recall_queries', 'N/A')}")
                avg = tu.get("avg_recall_tokens_per_query")
                if avg is not None:
                    lines.append(f"  Avg tokens/query:  {avg}")
            lines.append("")

    # Cost efficiency section — aggregate across all runs
    all_cost_metrics = [r.get("cost_metrics", {}) for r in runs if r.get("cost_metrics")]
    if all_cost_metrics:
        avg_cm = {}
        for key in all_cost_metrics[0]:
            vals = [m[key] for m in all_cost_metrics if key in m]
            avg_cm[key] = sum(vals) / len(vals) if vals else 0.0
        lines.append(border_s)
        lines.append(f"{_c(_BOLD)} [Cost Efficiency]{_c(_RESET)}")
        tpq = avg_cm.get("tokens_per_query", 0)
        tpc = avg_cm.get("tokens_per_correct", 0)
        eff = avg_cm.get("cost_efficiency", 0)
        lines.append(f"  Tokens/query:    ~{tpq:.0f}")
        lines.append(f"  Tokens/correct:  ~{tpc:.0f}")
        lines.append(f"  Efficiency:      {eff:.3f}  (score / log2(tokens))")
        lines.append("")

    # Also use top-level cost_metrics if runs list is empty
    top_cm = result_json.get("cost_metrics", {})
    if not all_cost_metrics and top_cm:
        lines.append(border_s)
        lines.append(f"{_c(_BOLD)} [Cost Efficiency]{_c(_RESET)}")
        tpq = top_cm.get("tokens_per_query", 0)
        tpc = top_cm.get("tokens_per_correct", 0)
        eff = top_cm.get("cost_efficiency", 0)
        lines.append(f"  Tokens/query:    ~{tpq:.0f}")
        lines.append(f"  Tokens/correct:  ~{tpc:.0f}")
        lines.append(f"  Efficiency:      {eff:.3f}  (score / log2(tokens))")
        lines.append("")

    lines.append(border_h)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Quick self-test — run this file directly to preview all chart types
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib

    results_path = (
        pathlib.Path(__file__).parent.parent / "results" / "baseline-flat.json"
    )
    if not results_path.exists():
        print(f"Result file not found: {results_path}")
        raise SystemExit(1)

    with open(results_path) as f:
        data = json.load(f)

    per_cat = data.get("per_category_mean", {})
    ret_m   = data.get("retrieval_metrics", {})

    # ── radar chart ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RADAR CHART (bar representation)")
    print("=" * 60)
    print(radar_chart(per_cat, title="Category Scores"))

    # ── comparison chart ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON CHART")
    print("=" * 60)
    before_cat = {k: max(0.0, v - 0.05 - i * 0.01)
                  for i, (k, v) in enumerate(per_cat.items())}
    print(comparison_chart(before_cat, per_cat, title="Before vs After"))

    # ── metrics table ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("METRICS TABLE")
    print("=" * 60)
    print(metrics_table(ret_m, title="Retrieval Metrics"))

    # ── regression chart ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("REGRESSION CHART")
    print("=" * 60)
    history = [
        {"mean_score": 0.836},
        {"mean_score": 0.891},
        data,
    ]
    print(regression_chart(history, metric="overall"))

    # ── full dashboard ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY DASHBOARD")
    print("=" * 60)
    print(summary_dashboard(data))
