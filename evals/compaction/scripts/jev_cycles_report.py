#!/usr/bin/env python3
"""Render the repeated-compaction table from jev_cycles.py outputs.

Usage: jev_cycles_report.py <cycles_json> [<cycles_json> ...]

One row per run: cycles reached, raw session consumed, freed-per-cycle at the
first and last cycle, text floor at the first and last cycle, and how the run
ended (still working / stuck / plugin fallback). Prints a markdown table.
"""
import json
import sys


def describe(d: dict) -> dict:
    cycles = d["cycles"]
    scored = [c for c in cycles if "freed_pct" in c]
    row = {
        "run": f"{d['transcript']} @{d['threshold'] // 1000}K",
        "cycles": len(scored),
        "raw": f"{d['raw_tokens_consumed'] / 1e6:.2f}M of {d['raw_total'] / 1e6:.1f}M",
        "freed": "—",
        "floor": "—",
        "end": "still working",
    }
    if scored:
        row["freed"] = f"{scored[0]['freed_pct']:.0f}% → {scored[-1]['freed_pct']:.0f}%"
        row["floor"] = f"{scored[0]['floor'] / 1000:.0f}K → {scored[-1]['floor'] / 1000:.0f}K"
        if scored[-1].get("stuck"):
            row["end"] = "STUCK (floor ≥ threshold, 0% freed)"
        elif scored[-1]["freed_pct"] < 10:
            row["end"] = f"degraded: {scored[-1]['before'] - scored[-1]['after']:,} tokens freed/cycle"
    if cycles and "fallback" in cycles[-1]:
        c = cycles[-1]
        row["end"] = f"fallback on cycle {c['cycle']} ({c['calls']} calls, state does not fit)"
    return row


def main() -> None:
    rows = [describe(json.load(open(p, encoding="utf-8"))) for p in sys.argv[1:]]
    print("| run | cycles | raw session consumed | freed per cycle | text floor | end state |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run']} | {r['cycles']} | {r['raw']} | {r['freed']} | {r['floor']} | {r['end']} |")


if __name__ == "__main__":
    main()
