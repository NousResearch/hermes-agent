#!/usr/bin/env python3
"""Validation harness: run the semantic judge over the 50-case labelled set and
measure accuracy against the acceptance criteria (>=95% on 25 known-correct and
25 known-regression runs).

Metrics:
  - exact_match_accuracy : judge verdict == expected label across all 50
  - correct_behaviour_accuracy : among the 25 correct-set cases (pass + skip),
    fraction where the judge did NOT falsely flag a semantic regression
    (verdict != "fail"). The skip/crash controls must yield "skip", never "fail".
  - regression_accuracy : among the 25 fail cases, fraction where judge said "fail".
  - fp / fn : false positives (correct case judged "fail") and false negatives
    (regression judged "pass").

Requires network LLM access (ollama-cloud). Run from repo root:
  python3 validation/harness.py [--model deepseek-v4-flash] [--out results.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_judge import crash_filter  # noqa: E402
from semantic_judge.creds import default_judge  # noqa: E402
from semantic_judge.judge import judge_run  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
CORRECT_LABELS = {"pass", "skip"}  # crash-controls belong to the correct set


def collect_cases():
    cases = []
    for area in sorted(os.listdir(ROOT)):
        ap = os.path.join(ROOT, area)
        if not os.path.isdir(ap):
            continue
        for cid in sorted(os.listdir(ap)):
            cp = os.path.join(ap, cid)
            exp = os.path.join(cp, "expected.txt")
            log = os.path.join(cp, "log.txt")
            lab = os.path.join(cp, "label.json")
            if not all(os.path.isfile(x) for x in (exp, log, lab)):
                continue
            with open(exp) as f:
                e = f.read()
            with open(log) as f:
                l = f.read()
            with open(lab) as f:
                label = json.load(f)["label"]
            cases.append({"id": cid, "area": area, "expected": e, "logs": l,
                          "label": label})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--out", default=os.path.join(ROOT, "results.json"))
    ap.add_argument("--dry-crash", action="store_true",
                    help="only test crash filter routing, no LLM")
    args = ap.parse_args()

    cases = collect_cases()
    assert len(cases) == 50, f"expected 50 cases, got {len(cases)}"

    # First, deterministic crash-filter routing check (no LLM).
    crash_stats = {"correct_routed": 0, "crash_cases": 0}
    for c in cases:
        if c["label"] == "skip":
            crash_stats["crash_cases"] += 1
            f = crash_filter.classify(c["logs"])
            if f["crash_related"]:
                crash_stats["correct_routed"] += 1
            else:
                print(f"  [WARN] crash control {c['id']} not routed as crash: "
                      f"{f['reason']}")

    if args.dry_crash:
        print(f"Crash-filter routing: {crash_stats['correct_routed']}/"
              f"{crash_stats['crash_cases']} crash controls correctly classified")
        return

    judge = default_judge(model=args.model)
    results = []

    def judge_one(c, attempt=0):
        try:
            return judge_run(c["expected"], c["logs"], judge)
        except Exception as e:  # noqa: BLE001 - retry transient LLM failures
            if attempt < 2:
                import time
                time.sleep(2)
                return judge_one(c, attempt + 1)
            return {
                "verdict": "error",
                "crash_related": False,
                "confidence": 0.0,
                "reasoning": [f"judge error: {e}"],
                "summary": f"LLM judge failed after retries: {e}",
            }

    for c in cases:
        print(f"[{c['id']}] ...", file=sys.stderr, flush=True)
        r = judge_one(c)
        r["id"] = c["id"]
        r["area"] = c["area"]
        r["expected_label"] = c["label"]
        results.append(r)
        print(f"  expected={c['label']} got={r['verdict']} "
              f"conf={r['confidence']} ({r.get('elapsed_s')}s)",
              file=sys.stderr)

    # ---- Metrics ----
    n = len(results)
    exact = sum(1 for r in results if r["verdict"] == r["expected_label"])

    correct_set = [r for r in results if r["expected_label"] in CORRECT_LABELS]
    regression_set = [r for r in results if r["expected_label"] == "fail"]

    # FP: correct case flagged as semantic regression (fail).
    fp = [r["id"] for r in correct_set if r["verdict"] == "fail"]
    # FN: regression judged pass (missed regression).
    fn = [r["id"] for r in regression_set if r["verdict"] == "pass"]
    # Crash-control specifically judged as semantic regression:
    crash_fp = [r["id"] for r in results
                if r["expected_label"] == "skip" and r["verdict"] == "fail"]

    correct_ok = len(correct_set) - len(fp)
    regression_ok = len(regression_set) - len(fn)

    report = {
        "model": args.model,
        "total_cases": n,
        "exact_match_accuracy": round(exact / n, 4),
        "correct_set_count": len(correct_set),
        "correct_behaviour_accuracy": round(correct_ok / len(correct_set), 4),
        "false_positives": fp,
        "crash_controls_flagged_as_regression": crash_fp,
        "regression_set_count": len(regression_set),
        "regression_accuracy": round(regression_ok / len(regression_set), 4),
        "false_negatives": fn,
        "crash_filter_routing": crash_stats,
        "passes_ac_threshold": (correct_ok / len(correct_set) >= 0.95
                                and regression_ok / len(regression_set) >= 0.95),
        "results": [
            {"id": r["id"], "area": r["area"], "expected": r["expected_label"],
             "verdict": r["verdict"], "confidence": r["confidence"],
             "crash_related": r.get("crash_related", False),
             "reasoning": r.get("reasoning", [])}
            for r in results
        ],
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== SEMANTIC JUDGE VALIDATION REPORT ===")
    print(f"model: {args.model}")
    print(f"total cases: {n}")
    print(f"exact-match accuracy: {report['exact_match_accuracy']:.1%}")
    print(f"correct-set ({len(correct_set)}): correct_behaviour_accuracy "
          f"{report['correct_behaviour_accuracy']:.1%}  FP={fp or 'none'}")
    print(f"  crash-controls flagged as regression: {crash_fp or 'none'}")
    print(f"regression-set ({len(regression_set)}): regression_accuracy "
          f"{report['regression_accuracy']:.1%}  FN={fn or 'none'}")
    print(f"crash-filter routing: {crash_stats}")
    print(f"AC >=95% both: {report['passes_ac_threshold']}")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
