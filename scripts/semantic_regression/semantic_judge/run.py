#!/usr/bin/env python3
"""CLI: run the semantic regression judge.

Examples:
  # Judge one case (expected-behaviour + log files)
  python -m semantic_judge.run \\
      --expected scenarios/expected.md --log runs/run_T1.log

  # Judge all cases in a directory (each subdir: expected.txt + log.txt)
  python -m semantic_judge.run --dir validation/cases --out results.json
"""
import argparse
import json
import os
import sys

from . import crash_filter
from .creds import default_judge
from .judge import judge_run


def load_case(path):
    """Load a case directory or a single expected/log pair."""
    expected_path = os.path.join(path, "expected.txt")
    log_path = os.path.join(path, "log.txt")
    if not (os.path.exists(expected_path) and os.path.exists(log_path)):
        return None
    with open(expected_path) as f:
        expected = f.read()
    with open(log_path) as f:
        logs = f.read()
    return {"id": os.path.basename(path), "expected": expected, "logs": logs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="dir of cases (subdirs w/ expected.txt + log.txt)")
    ap.add_argument("--expected", help="expected behaviour file")
    ap.add_argument("--log", help="execution log file")
    ap.add_argument("--out", help="json output path (dir mode)")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--dry-crash", action="store_true",
                    help="only run the deterministic crash filter (no LLM)")
    args = ap.parse_args()

    cases = []
    if args.dir:
        for name in sorted(os.listdir(args.dir)):
            p = os.path.join(args.dir, name)
            if os.path.isdir(p):
                c = load_case(p)
                if c:
                    cases.append(c)
    elif args.expected and args.log:
        with open(args.expected) as f:
            expected = f.read()
        with open(args.log) as f:
            logs = f.read()
        cases = [{"id": os.path.basename(args.log), "expected": expected, "logs": logs}]
    else:
        ap.error("provide --dir OR --expected+--log")

    if args.dry_crash:
        results = []
        for c in cases:
            f = crash_filter.classify(c["logs"])
            results.append({"id": c["id"], "crash_related": f["crash_related"],
                            "reason": f["reason"], "markers": f["markers"]})
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
        return

    judge = default_judge(model=args.model)
    results = []
    for c in cases:
        print(f"[judge] {c['id']} ...", file=sys.stderr, flush=True)
        r = judge_run(c["expected"], c["logs"], judge)
        r["id"] = c["id"]
        results.append(r)
        print(f"  -> {r['verdict']} conf={r['confidence']} "
              f"({r.get('elapsed_s')}s) {r.get('summary','')[:80]}",
              file=sys.stderr)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
