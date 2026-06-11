#!/usr/bin/env python3
"""Compare enabled vs disabled runs and produce a readable report.

Reads scripts/out/_summary.json and the per-scenario JSONs, prints a side-by-
side comparison of what happened, and flags anomalies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def load_record(scenario_id: str, mode: str):
    path = OUT / f"{scenario_id}__{mode}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_tool_seq(calls):
    if not calls:
        return "(none)"
    return " → ".join(c["name"] for c in calls)


def fmt_bridge_seq(calls):
    if not calls:
        return "(none)"
    parts = []
    for c in calls:
        if c["name"] == "tool_call":
            inner = (c.get("args") or {}).get("name", "?")
            parts.append(f"tool_call→{inner}")
        elif c["name"] == "tool_search":
            q = (c.get("args") or {}).get("query", "?")
            parts.append(f"search('{q[:30]}')")
        elif c["name"] == "tool_describe":
            n = (c.get("args") or {}).get("name", "?")
            parts.append(f"describe({n})")
    return " → ".join(parts)


def main():
    if not OUT.exists():
        print("No output directory at", OUT)
        sys.exit(1)
    summary_path = OUT / "_summary.json"
    if not summary_path.exists():
        print("No _summary.json yet")
        sys.exit(1)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scenarios = sorted({row["scenario"] for row in summary})

    print(f"{'='*78}")
    print(f"  Live test results: tool_search ENABLED vs DISABLED")
    print(f"{'='*78}\n")

    fails = 0
    for sid in scenarios:
        en = load_record(sid, "enabled")
        di = load_record(sid, "disabled")
        if not en or not di:
            continue
        expected = set(en["expected_underlying_tools"])

        print(f"┌─ {sid}  ({en['scenario_description']})")
        print(f"│  Prompt: {en['prompt'][:120]}")
        print(f"│  Expected underlying tools: {sorted(expected) or '(none)'}")
        print(f"│")

        for label, rec in [("ENABLED ", en), ("DISABLED", di)]:
            called_under = [c["name"] for c in rec["underlying_tool_calls"]]
            called_set = set(called_under)
            missing = expected - called_set
            extra = called_set - expected - {"read_file", "search_files", "terminal", "todo", "memory"}

            mark = "✓" if (expected.issubset(called_set) and not rec["error"]) else "✗"
            if mark == "✗":
                fails += 1

            print(f"│  {label} {mark}  bridges={len(rec['bridge_calls']):2}  underlying={len(rec['underlying_tool_calls']):2}  "
                  f"iters={rec['n_iterations']:2}  elapsed={rec['elapsed_seconds']:5.1f}s  err={bool(rec['error'])}")
            print(f"│    underlying: {fmt_tool_seq(rec['underlying_tool_calls'])}")
            if rec["bridge_calls"]:
                print(f"│    bridges:    {fmt_bridge_seq(rec['bridge_calls'])}")
            if missing:
                print(f"│    ⚠ MISSING expected tools: {sorted(missing)}")
            if extra:
                print(f"│    ⓘ extra tools called: {sorted(extra)}")
            if rec["error"]:
                print(f"│    💥 error: {rec['error'][:200]}")
        # Bridge-trip count vs direct (interesting comparator)
        en_bridges = len(en["bridge_calls"])
        di_underlying = len(di["underlying_tool_calls"])
        en_underlying = len(en["underlying_tool_calls"])
        overhead = en_bridges + en_underlying - di_underlying
        print(f"│  Δ round-trip cost: enabled used {en_bridges + en_underlying} calls vs disabled {di_underlying}  →  +{overhead}")
        print(f"│  Final (enabled):  {(en.get('final_response') or '')[:140]}")
        print(f"│  Final (disabled): {(di.get('final_response') or '')[:140]}")
        print(f"└──")
        print()

    print(f"\nFails: {fails}/{2*len(scenarios)}")


if __name__ == "__main__":
    main()
