"""Compare read-tool eval result sets (baseline vs feature labels).

Usage:
  python3 evals/readtool/report.py --labels baseline feat-fifo-guard
  python3 evals/readtool/report.py --labels baseline feat-fifo-guard --model qwen_qwen3.8-max
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULTS = Path(__file__).resolve().parent / "results"

METRICS = ["score", "api_turns", "tool_calls", "read_file_calls", "total_tokens", "wall_s"]


def load_label(label: str, model_filter: str | None) -> dict:
    """-> {model: {task_id: {metric: [values across reps]}}}"""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    root = RESULTS / label
    if not root.is_dir():
        raise SystemExit(f"no results for label '{label}' under {root}")
    for model_dir in sorted(root.iterdir()):
        if model_filter and model_dir.name != model_filter:
            continue
        for rep_file in sorted(model_dir.glob("rep*.json")):
            data = json.loads(rep_file.read_text(encoding="utf-8-sig"))
            for rec in data["records"]:
                if rec.get("error"):
                    # count errored task-runs as score 0 but keep them in the
                    # denominator; efficiency metrics excluded (not comparable)
                    out[model_dir.name][rec["task_id"]]["score"].append(0.0)
                    out[model_dir.name][rec["task_id"]]["errors"].append(1)
                    continue
                for metric in METRICS:
                    if metric in rec and rec[metric] is not None:
                        out[model_dir.name][rec["task_id"]][metric].append(rec[metric])
    return out


def fmt(v: float, metric: str) -> str:
    if metric == "score":
        return f"{v:.3f}"
    if metric == "wall_s":
        return f"{v:.0f}s"
    return f"{v:,.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--model", default=None, help="model slug filter (dir name)")
    args = ap.parse_args()

    sets = {lbl: load_label(lbl, args.model) for lbl in args.labels}
    models = sorted({m for s in sets.values() for m in s})

    for model in models:
        print(f"\n=== {model} ===")
        task_ids = sorted(
            {t for lbl in args.labels for t in sets[lbl].get(model, {})}
        )
        # Per-task score table
        header = f"{'task':<22}" + "".join(f"{lbl:>24}" for lbl in args.labels)
        print(header)
        print("-" * len(header))
        for tid in task_ids:
            row = f"{tid:<22}"
            for lbl in args.labels:
                vals = sets[lbl].get(model, {}).get(tid, {})
                sc = vals.get("score", [])
                turns = vals.get("api_turns", [])
                tok = vals.get("total_tokens", [])
                cell = (
                    f"{mean(sc):.2f} ({len(sc)}r) "
                    f"t={mean(turns):.1f} " if turns else f"{mean(sc):.2f} ({len(sc)}r) t=? "
                ) if sc else "—"
                if sc and tok:
                    cell += f"tk={mean(tok)/1000:.0f}k"
                row += f"{cell:>24}"
            print(row)
        # Aggregates
        print()
        for metric in METRICS:
            row = f"{'MEAN ' + metric:<22}"
            base_val = None
            for lbl in args.labels:
                per_task = []
                for tid in task_ids:
                    vals = sets[lbl].get(model, {}).get(tid, {}).get(metric, [])
                    if vals:
                        per_task.append(mean(vals))
                if per_task:
                    v = mean(per_task)
                    delta = ""
                    if base_val is not None and base_val != 0:
                        pct = (v - base_val) / base_val * 100
                        delta = f" ({pct:+.0f}%)"
                    if base_val is None:
                        base_val = v
                    row += f"{fmt(v, metric) + delta:>24}"
                else:
                    row += f"{'—':>24}"
            print(row)
    print(
        "\nNote: efficiency means are per-task means over reps, then averaged "
        "across tasks (never sums). Errored runs score 0 but are excluded "
        "from efficiency means."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
