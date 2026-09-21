#!/usr/bin/env python3
"""
Semantic Regression Release Gate — runs the LogicHunter-style semantic test
suite and blocks a Hermes Agent release on any semantic regression.

This is the Phase 3 integration of the semantic-regression-testing proposal
(t_f25f6039). It wires two upstream deliverables into the release pipeline:

  * semantic_judge/  (t_60b946b5) — the isolated LLM judge (crash-filter first,
    then judge_run(expected, logs, judge) -> structured {verdict, confidence,
    reasoning[], summary}).
  * validation/      (t_c3a737ba) — the 17 Hermes-specific semantic scenarios
    (T1-T6 tool dispatch, P1-P6 plugin, S1-S5 skill) with expected behaviours.

Gate semantics (per the judge's documented integration seam):
  * verdict == "fail"   -> SEMANTIC REGRESSION -> blocks the release.
  * verdict == "skip"   -> crash-class failure, owned by the existing crash
                           suite -> does NOT block the semantic gate.
  * verdict == "pass"   -> correct behaviour.
  * verdict == "ambiguous" -> low-confidence undecidable; recorded but treated
                           as non-blocking (crash suite owns hard failures).
  * verdict == "error"  -> the judge itself failed (transient LLM) -> block
                           fail-closed (a gate that can't judge must not ship).

Runs in <= 10 minutes: the full 50-case set judges at ~2.7s/case (~2.5 min
wall) and the 17-scenario production set even faster.

Exit codes: 0 = release allowed, 2 = release blocked, 1 = usage/error.
Mirrors scripts/hermaguard_release_gate.py conventions.

CLI:
  python scripts/semantic_release_gate.py                      # run + block
  python scripts/semantic_release_gate.py --cases validation   # run a dir
  python scripts/semantic_release_gate.py --model deepseek-v4-flash
  python scripts/semantic_release_gate.py --report-dir <path>  # JSON reports
  python scripts/semantic_release_gate.py --self-test          # no LLM checks
  python scripts/semantic_release_gate.py --dry-crash          # crash-filter only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "semantic_regression"))

from semantic_judge import crash_filter  # noqa: E402
from semantic_judge.creds import default_judge  # noqa: E402
from semantic_judge.judge import judge_run, JudgeError  # noqa: E402

DEFAULT_MODEL = "deepseek-v4-flash"
# The production release corpus: known-correct Hermes behaviours (21 scenarios)
# + 4 crash-controls that must route to the crash suite (verdict "skip", never
# "fail"). This is the gate's default. The full 50-case validation corpus under
# validation/ is the judge-accuracy measurement set (mixed pass/fail by design)
# and is only used for offline calibration, not as the release gate.
DEFAULT_CASES = REPO_ROOT / "scripts" / "semantic_regression" / "cases" / "release"
# Default report dir mirrors the hermaguard-gate convention (governance logboard).
DEFAULT_REPORT_DIR = Path(
    os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
) / "governance" / "logboard"

# Verdicts that block the release gate.
BLOCKING_VERDICTS = ("fail", "error")


def collect_cases(cases_dir: Path):
    """Discover (expected.txt + log.txt) pairs.

    Supports two layouts:
      * flat:   <cases_dir>/<scenario>/expected.txt   (release corpus)
      * nested: <cases_dir>/<area>/<scenario>/expected.txt  (validation corpus)
    """
    cases = []
    cases_dir = Path(cases_dir)
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"cases dir not found: {cases_dir}")

    def add_case(area, cp):
        exp = cp / "expected.txt"
        log = cp / "log.txt"
        if exp.is_file() and log.is_file():
            cases.append({
                "id": cp.name,
                "area": area,
                "expected": exp.read_text(encoding="utf-8"),
                "logs": log.read_text(encoding="utf-8"),
            })

    for area in sorted(cases_dir.iterdir()):
        if not area.is_dir():
            continue
        # Flat layout: area itself is a scenario dir with expected.txt/log.txt.
        if (area / "expected.txt").is_file():
            add_case(area.parent.name, area)
            continue
        # Nested layout: area contains scenario subdirs.
        for cid in sorted(area.iterdir()):
            if cid.is_dir():
                add_case(area.name, cid)
    return cases


def judge_one(expected, logs, judge, retries=2):
    """judge_run with bounded retries on transient LLM errors -> 'error' verdict."""
    for attempt in range(retries + 1):
        try:
            return judge_run(expected, logs, judge)
        except JudgeError as e:
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            return {
                "verdict": "error",
                "crash_related": False,
                "confidence": 0.0,
                "reasoning": [f"judge error: {e}"],
                "summary": f"LLM judge failed after retries: {e}",
                "elapsed_s": 0.0,
            }
    # unreachable
    return {"verdict": "error", "reasoning": ["unreachable"], "confidence": 0.0}


def run_gate(cases_dir, model, report_dir) -> dict:
    started = time.time()
    cases = collect_cases(cases_dir)
    judge = default_judge(model=model)

    results = []
    for c in cases:
        r = judge_one(c["expected"], c["logs"], judge)
        r["id"] = c["id"]
        r["area"] = c["area"]
        results.append(r)
        print(f"  [{c['area']}/{c['id']}] {r['verdict']} "
              f"(conf={r.get('confidence')}, {r.get('elapsed_s')}s)", file=sys.stderr)

    blocking = [r for r in results if r["verdict"] in BLOCKING_VERDICTS]
    passed = len(results) - len(blocking)
    crash_skips = sum(1 for r in results if r.get("crash_related"))

    report = {
        "gate_name": "semantic_release_gate",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "cases_total": len(results),
        "cases_passed": passed,
        "cases_blocking": len(blocking),
        "crash_class_skips": crash_skips,
        "wall_seconds": round(time.time() - started, 2),
        "blocked": len(blocking) > 0,
        "results": [
            {
                "id": r["id"],
                "area": r["area"],
                "verdict": r["verdict"],
                "confidence": r.get("confidence"),
                "crash_related": r.get("crash_related", False),
                "reasoning": r.get("reasoning", []),
                "summary": r.get("summary", ""),
                "elapsed_s": r.get("elapsed_s"),
            }
            for r in results
        ],
    }

    if report_dir:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = Path(report_dir) / f"semantic-release-gate-{ts}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["_report_file"] = str(out)

    return report


def _self_test():
    """Deterministic checks that need no LLM: blocking semantics + case discovery."""
    ok = True
    # Crash filter: a traceback routes to skip (crash-class), not fail.
    f = crash_filter.classify("Traceback (most recent call last):\n  File \"x.py\"\nRuntimeError")
    if not (f["crash_related"] and f.get("crash_related")):
        print("  FAIL: traceback not routed as crash")
        ok = False
    # A clean log is not crash-related.
    f2 = crash_filter.classify("[12:00:01] tool web_search query='x'\n[12:00:02] web_search -> [{'url':'u'}]")
    if f2.get("crash_related"):
        print("  FAIL: clean log routed as crash")
        ok = False
    # Cases discoverable (release corpus).
    n = len(collect_cases(DEFAULT_CASES))
    if n < 17:
        print(f"  FAIL: expected >=17 scenarios, found {n}")
        ok = False
    print(f"SELF-TEST: {'PASS' if ok else 'FAIL'} ({n} release scenarios discovered)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Semantic regression release gate.")
    ap.add_argument("--cases", default=str(DEFAULT_CASES),
                    help="dir of cases (subdirs with expected.txt + log.txt)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR),
                    help="dir for JSON reports (governance logboard by default)")
    ap.add_argument("--self-test", action="store_true", help="no-LLM checks and exit")
    ap.add_argument("--dry-crash", action="store_true",
                    help="only run the deterministic crash filter, no LLM")
    args = ap.parse_args()

    if args.self_test:
        return 0 if _self_test() else 1

    if args.dry_crash:
        n_crash = 0
        for c in collect_cases(Path(args.cases)):
            f = crash_filter.classify(c["logs"])
            if f.get("crash_related"):
                n_crash += 1
                print(f"  crash: {c['area']}/{c['id']} — {f.get('reason','')}")
        print(f"CRASH-FILTER: {n_crash} crash-class cases routed out of semantic judge")
        return 0

    try:
        report = run_gate(Path(args.cases), args.model,
                          args.report_dir or None)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("\n=== SEMANTIC REGRESSION RELEASE GATE ===")
    verdict = "BLOCKED — semantic regression(s) found" if report["blocked"] else "ALLOWED"
    print(f"Verdict: {verdict}")
    print(f"Cases: {report['cases_total']}  |  Passed: {report['cases_passed']}  |  "
          f"Blocking: {report['cases_blocking']}  |  Crash-class: {report['crash_class_skips']}")
    print(f"Wall time: {report['wall_seconds']}s")

    if report["blocked"]:
        print("\nBLOCKING CASES (must be resolved before release):")
        for r in report["results"]:
            if r["verdict"] in BLOCKING_VERDICTS:
                print(f"  [{r['verdict'].upper()}] {r['area']}/{r['id']} "
                      f"(conf={r['confidence']})")
                for line in r.get("reasoning", []):
                    print(f"       {line}")
        print("\nRelease blocked by semantic regression gate.")
        print(f"Report: {report.get('_report_file','(no report)')}")
        return 2

    print("No semantic regressions. Release gate not triggered.")
    print(f"Report: {report.get('_report_file','(no report)')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
