#!/usr/bin/env python3
"""run_evals.py — run the information-gain skill on eval cases and adjudicate them.

For each case it runs the skill, then applies deterministic structural checks plus an
LLM adjudicator (a DIFFERENT model than the generation stages) to decide whether the
output is ACCEPTABLE. Prints a per-case report and exits non-zero if any case fails —
so it doubles as a CI gate.

Usage:
    python3 evals/run_evals.py                     # all cases
    python3 evals/run_evals.py --case usaw-calendar
    python3 evals/run_evals.py --gen-model fast --judge-model deepseek --json

Exit codes: 0 all acceptable · 1 one or more unacceptable · 2 Ollama unreachable.
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import pipeline  # noqa: E402
import infogain  # noqa: E402
import adjudicator  # noqa: E402


def load_cases(path):
    with open(path) as f:
        return json.load(f).get("cases", [])


def build_cfg(args):
    cfg = dict(infogain.DEFAULTS)
    # Use one (fast, local) model for all generation stages so evals are quick and
    # cheap; the JUDGE is a separate, stronger model for independence.
    for k in ("plan_model", "question_gen_model", "answer_model", "value_judge_model"):
        cfg[k] = args.gen_model
    cfg["max_rounds"] = args.max_rounds
    cfg["answers_per_question"] = args.answers_per_question
    return cfg


def run(args):
    cases = load_cases(args.cases)
    if args.case:
        cases = [c for c in cases if c.get("id") == args.case]
        if not cases:
            print(f"No case with id '{args.case}'", file=sys.stderr)
            return 1
    cfg = build_cfg(args)

    results = []
    for case in cases:
        if not args.json:
            print(f"\n▶ {case['id']} ({case['expectation']}): running skill …",
                  file=sys.stderr, flush=True)
        result = infogain.run(case["problem"], cfg)
        verdict = adjudicator.evaluate_case(case, result, args.judge_model, args.judge_timeout)
        verdict["_result"] = result if args.json else None
        results.append(verdict)
        if not args.json:
            _print_case(verdict)

    n_ok = sum(1 for v in results if v["acceptable"])
    if args.json:
        print(json.dumps({"cases": results, "passed": n_ok, "total": len(results)},
                         indent=2, default=str))
    else:
        print(f"\n{'=' * 60}\nACCEPTABLE: {n_ok}/{len(results)} cases "
              f"(judge={args.judge_model}, gen={args.gen_model})")
    return 0 if n_ok == len(results) else 1


def _print_case(v):
    mark = "✅" if v["acceptable"] else "❌"
    print(f"  {mark} {v['id']}: bucket={v['bucket_size']} "
          f"structural={'pass' if v['structural']['passed'] else 'FAIL'} "
          f"judge={'pass' if v['judged']['acceptable'] else 'FAIL'}")
    for f in v["structural"]["failures"]:
        print(f"      structural: {f}")
    if v["judged"].get("error"):
        print(f"      judge error: {v['judged']['error']}")
    for name, c in (v["judged"].get("criteria") or {}).items():
        flag = "" if c["score"] >= adjudicator.ACCEPT_FLOOR else "  ← low"
        print(f"      {name:18} {c['score']:.2f}{flag}  {c['reason'][:80]}")
    if v["judged"].get("summary"):
        print(f"      summary: {v['judged']['summary'][:160]}")


def build_parser():
    p = argparse.ArgumentParser(description="Adjudicated evals for the information-gain skill.")
    p.add_argument("--cases", default=os.path.join(_HERE, "cases.json"))
    p.add_argument("--case", help="Run only the case with this id.")
    p.add_argument("--gen-model", default="fast", help="Model alias for all generation stages.")
    p.add_argument("--judge-model", default="deepseek",
                   help="Adjudicator model alias (should differ from --gen-model).")
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--answers-per-question", type=int, default=4)
    p.add_argument("--judge-timeout", type=int, default=180)
    p.add_argument("--json", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not pipeline.ollama_reachable():
        print(f"Ollama not reachable at {pipeline.OLLAMA_URL}", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
