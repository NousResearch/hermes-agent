"""Aggregate results.jsonl into the scorecard tables.

Usage:
    python3 report.py [results/results.jsonl ...]

Groups by (model, arm): ok-rate, token mean/median, tool calls, wall clock,
and token delta vs the ``base`` arm of the same model when present.
"""
import json
import statistics
import sys
from collections import defaultdict


def main(paths):
    rows = []
    for p in paths:
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        print("no rows")
        return

    cells = defaultdict(list)
    for r in rows:
        cells[(r.get("model", "?"), r.get("arm", "?"))].append(r)

    base_tok = {}
    for (model, arm), rs in cells.items():
        if arm == "base":
            oks = [r for r in rs if r.get("ok")]
            if oks:
                base_tok[model] = statistics.mean(r.get("total_tokens", 0) for r in oks)

    hdr = f"{'model':<34} {'arm':<16} {'ok':>7} {'tok_mean':>9} {'tok_med':>8} {'calls':>6} {'wall_s':>7} {'vs base':>8}"
    print(hdr)
    print("-" * len(hdr))
    for (model, arm) in sorted(cells):
        rs = cells[(model, arm)]
        oks = [r for r in rs if r.get("ok")]
        n_ok, n = len(oks), len(rs)
        toks = [r.get("total_tokens", 0) for r in oks]
        calls = [r.get("tool_calls", 0) for r in oks]
        walls = [r.get("wall_s", 0) for r in oks]
        tok_mean = statistics.mean(toks) if toks else 0
        delta = ""
        if arm != "base" and model in base_tok and tok_mean:
            delta = f"{(tok_mean - base_tok[model]) / base_tok[model] * 100:+.0f}%"
        print(
            f"{model:<34} {arm:<16} {n_ok:>3}/{n:<3} {tok_mean:>9.0f} "
            f"{statistics.median(toks) if toks else 0:>8.0f} "
            f"{statistics.mean(calls) if calls else 0:>6.1f} "
            f"{statistics.mean(walls) if walls else 0:>7.1f} {delta:>8}"
        )

    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"\nerrors: {len(errs)}")
        for r in errs[:10]:
            print(f"  {r.get('model')}/{r.get('arm')}/{r.get('task')}/rep{r.get('rep')}: {r['error'][:120]}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/results.jsonl"])
