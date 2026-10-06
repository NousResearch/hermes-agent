#!/usr/bin/env python3
"""
run_eval.py — Scenario eval for the desired-state skill.

Runs realistic goal situations (from scenarios.jsonl) end-to-end through the
gap engine and checks expected progress / pace / summary. Complements the unit
tests: this is behavioral, dataset-shaped, and reports a scorecard — the form a
skill-optimization pass (e.g. GEPA) can score against. Exits nonzero on any
miss so it can gate CI.

    python3 run_eval.py [--json]

Stdlib-only. Python 3.11+.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from _common import GoalDoc  # noqa: E402  (path set above)
from gap import compute_gap  # noqa: E402


def _now(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _check(scenario: dict) -> tuple[bool, list[str]]:
    doc = GoalDoc(**scenario["goal"])
    res = compute_gap(doc, now=_now(scenario["now"]))
    expect = scenario["expect"]
    fails: list[str] = []
    if "progress_pct" in expect and res.progress_pct != expect["progress_pct"]:
        fails.append(f"progress_pct {res.progress_pct} != {expect['progress_pct']}")
    if "pace" in expect and res.pace != expect["pace"]:
        fails.append(f"pace {res.pace!r} != {expect['pace']!r}")
    for needle in expect.get("summary_contains", []):
        if needle not in res.summary:
            fails.append(f"summary missing {needle!r} (got {res.summary!r})")
    return (not fails), fails


def load_scenarios() -> list[dict]:
    lines = (HERE / "scenarios.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main(argv: list[str] | None = None) -> int:
    as_json = "--json" in (argv or sys.argv[1:])
    scenarios = load_scenarios()
    results = []
    for sc in scenarios:
        ok, fails = _check(sc)
        results.append({"id": sc["id"], "desc": sc["desc"], "pass": ok, "fails": fails})

    passed = sum(1 for r in results if r["pass"])
    total = len(results)

    if as_json:
        print(json.dumps({"passed": passed, "total": total, "results": results}, indent=2))
    else:
        print(f"desired-state eval — {passed}/{total} scenarios\n")
        for r in results:
            mark = "PASS" if r["pass"] else "FAIL"
            print(f"  [{mark}] {r['id']:<22} {r['desc']}")
            for f in r["fails"]:
                print(f"         ↳ {f}")
        print(f"\nscore: {passed}/{total} ({100 * passed // total}%)")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
