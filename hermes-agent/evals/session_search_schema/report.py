"""Summarize session_search schema A/B results.

Usage:
  python3 evals/session_search_schema/report.py [--label ab]
  python3 evals/session_search_schema/report.py results/ab/*.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent


def summarize(files):
    grand = collections.defaultdict(lambda: [0, 0, 0, 0])  # ok, n, tok, calls
    for f in sorted(files):
        agg = collections.defaultdict(
            lambda: dict(ok=0, n=0, calls=0, tok=0, bad=0))
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = (r["task"], r["arm"])
            a = agg[k]
            a["ok"] += r["ok"]
            a["n"] += 1
            a["calls"] += r["n_tool_calls"]
            a["tok"] += r["total_tokens"]
            a["bad"] += r["bad_calls"]
            g = grand[r["arm"]]
            g[0] += r["ok"]; g[1] += 1
            g[2] += r["total_tokens"]; g[3] += r["n_tool_calls"]
        tasks = sorted({k[0] for k in agg})
        arms = sorted({k[1] for k in agg})
        print("=" * 72)
        print(f)
        header = f"{'task':<14}" + "".join(f"{a + ' ok':<9}" for a in arms)
        header += "".join(f"{a + ' calls':<12}" for a in arms)
        header += "".join(f"{a + ' tok':<10}" for a in arms)
        print(header)
        for t in tasks:
            row = f"{t:<14}"
            for a in arms:
                c = agg.get((t, a), dict(ok=0, n=0))
                row += f"{str(c['ok']) + '/' + str(c['n']):<9}"
            for a in arms:
                c = agg.get((t, a), dict(calls=0, n=1))
                row += f"{c['calls'] / max(c['n'], 1):<12.1f}"
            for a in arms:
                c = agg.get((t, a), dict(tok=0, n=1))
                row += f"{c['tok'] // max(c['n'], 1):<10}"
            print(row)
    print("=" * 72)
    for arm, (ok, n, tok, calls) in sorted(grand.items()):
        if n:
            print(f"TOTAL {arm}: {ok}/{n} ok   "
                  f"avg tok/task {tok // n}   avg calls {calls / n:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=None)
    ap.add_argument("--label", default="ab")
    args = ap.parse_args()
    files = args.files or glob.glob(
        str(EVAL_DIR / "results" / args.label / "*.jsonl"))
    if not files:
        raise SystemExit("no result files found")
    summarize(files)


if __name__ == "__main__":
    main()
