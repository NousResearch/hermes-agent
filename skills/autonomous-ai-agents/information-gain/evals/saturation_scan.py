#!/usr/bin/env python3
"""saturation_scan.py — how wide should the initial breadth be? (Part 1a)

For each prompt, frame once, then draw the question generator at increasing breadth (gen_samples) and
measure how many DISTINCT targets (regions of uncertainty) the union covers. Breadth SATURATES where
more samples stop adding new distinct targets (marginal new-target rate → ~0). Generation-only (cheap:
no project/judge), so it isolates coverage. Use it to set the initial breadth (gen_samples /
questions_per_family) from evidence instead of a guess.

  OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes \
    python3 evals/saturation_scan.py --out /tmp/sat.json
"""

import argparse
import json
import os
import statistics as st
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import pipeline  # noqa: E402
import testbank  # noqa: E402

DEFAULT_IDS = ["buy-rent", "add-auth", "research-ratelimit", "gmail-triage", "deploy-app"]


def _targets(qs):
    return {(q.get("target") or q.get("question", "")).strip().lower() for q in qs if
            (q.get("target") or q.get("question"))}


def scan_prompt(pr, model, sweep, n, temperature, timeout):
    framing, _ = pipeline.frame_and_plan(pr["problem"], model, timeout)
    steps = []
    for s in sweep:
        qs, _ = pipeline.generate_questions(pr["problem"], framing, model, n, timeout=timeout,
                                            samples=s, temperature=temperature)
        steps.append({"samples": s, "n_questions": len(qs), "distinct_targets": len(_targets(qs))})
    return {"id": pr["id"], "cat": pr.get("cat", "?"), "steps": steps}


def _knee(steps):
    """First sample count where the marginal new-distinct-target gain drops below 1."""
    prev = 0
    for st_ in steps:
        gain = st_["distinct_targets"] - prev
        if prev and gain < 1:
            return st_["samples"] - 1
        prev = st_["distinct_targets"]
    return steps[-1]["samples"] if steps else 0


def render(rows, sweep):
    print(f"\n{'id':<18}{'cat':<14}" + "".join(f"s{s:>3}" for s in sweep) + "   knee")
    for r in rows:
        by = {x["samples"]: x["distinct_targets"] for x in r["steps"]}
        print(f"{r['id']:<18}{r['cat']:<14}" + "".join(f"{by.get(s, 0):>4}" for s in sweep)
              + f"   {_knee(r['steps'])}")
    print("\n— distinct targets vs breadth (samples). Saturation = where the row stops climbing. —")
    knees = [_knee(r["steps"]) for r in rows if r["steps"]]
    if knees:
        print(f"median saturation breadth (gen_samples): {st.median(knees)}  (range {min(knees)}–{max(knees)})")
    # average marginal gain per added sample (aggregate)
    for i, s in enumerate(sweep[1:], 1):
        gains = [r["steps"][i]["distinct_targets"] - r["steps"][i - 1]["distinct_targets"]
                 for r in rows if len(r["steps"]) > i]
        if gains:
            print(f"  samples {sweep[i-1]}→{s}: avg +{st.mean(gains):.1f} new distinct targets")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    p.add_argument("--gen-model", default="fast")
    p.add_argument("--sweep", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--n", type=int, default=6, help="questions per draw")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args(argv)

    model = pipeline.resolve_alias(args.gen_model)
    prompts = [testbank.BY_ID[i] for i in args.ids if i in testbank.BY_ID]
    rows, t0 = [], time.time()
    for pr in prompts:
        print(f"… {pr['id']} ({pr['cat']}): sweeping samples {args.sweep}", file=sys.stderr, flush=True)
        try:
            rows.append(scan_prompt(pr, model, args.sweep, args.n, args.temperature, args.timeout))
        except Exception as e:
            rows.append({"id": pr["id"], "cat": pr.get("cat"), "error": str(e), "steps": []})
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"rows": rows, "sweep": args.sweep, "n": args.n,
                           "elapsed_s": round(time.time() - t0, 1)}, f, indent=2, default=str)
    render([r for r in rows if not r.get("error")], args.sweep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
