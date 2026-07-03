#!/usr/bin/env python3
"""saturation_scan.py — how wide should the initial breadth be? (Part 1a)

Two modes, both swept over breadth (gen_samples):

  default (generation-only, cheap): frame once, draw the generator at increasing breadth and count
    how many DISTINCT targets (regions of uncertainty) the union covers. Coverage SATURATES where
    more samples stop adding new distinct targets. Isolates coverage; no project/judge.

  --scored (full pipeline, costly): run the WHOLE info-gain pipeline per breadth and track max(value)
    and the count of candidates above the discard floor. This is the breadth finding's stronger half:
    coverage (distinct targets) keeps climbing, but does the HIGH-VALUE signal saturate early? If the
    best value and the above-floor count plateau while distinct targets still grow, then breadth is
    bounded by value (modest breadth is right), not by coverage. Families off + max_rounds=1 so the
    one knob that varies is breadth.

  OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes \
    python3 evals/saturation_scan.py --out /tmp/sat.json                # coverage
    python3 evals/saturation_scan.py --scored --out /tmp/sat_scored.json # value saturation
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

import infogain  # noqa: E402
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


def scan_prompt_scored(pr, base_cfg, sweep, thr):
    """Full-pipeline value-saturation: run info-gain at each breadth (gen_samples=s) and record the
    high-value signal — max(value) and the count of scored candidates at/above the discard floor —
    alongside distinct-target coverage. Families off + max_rounds=1 (set by the caller) so breadth
    is the only varying knob."""
    steps = []
    for s in sweep:
        cfg = dict(base_cfg)
        cfg["gen_samples"] = s
        r = infogain.run(pr["problem"], cfg)
        sa = r.get("all_scored", [])
        vals = [q.get("value", 0.0) for q in sa]
        steps.append({
            "samples": s, "n_candidates": len(sa),
            "distinct_targets": len(_targets(sa)),
            "max_value": round(max(vals), 3) if vals else 0.0,
            "above_floor": sum(1 for v in vals if v >= thr),
            "bucket": len(r.get("bucket", [])),
        })
    return {"id": pr["id"], "cat": pr.get("cat", "?"), "steps": steps}


def _value_knee(steps, eps=0.03):
    """First breadth where max_value stops climbing by more than `eps` — the value plateau."""
    prev = None
    for s in steps:
        v = s.get("max_value", 0.0)
        if prev is not None and v - prev < eps:
            return s["samples"] - 1
        prev = v
    return steps[-1]["samples"] if steps else 0


def render_scored(rows, sweep, thr):
    print(f"\n=== SCORED value-saturation (discard_floor={thr}) ===")
    print(f"{'id':<18}{'cat':<14}" + "".join(f"  s{s}: maxV/≥flr" for s in sweep) + "   vknee")
    for r in rows:
        by = {x["samples"]: x for x in r["steps"]}
        cells = "".join(f"  {by[s]['max_value']:.2f}/{by[s]['above_floor']:>2}" if s in by
                        else "    -/- " for s in sweep)
        print(f"{r['id']:<18}{r['cat']:<14}{cells}   {_value_knee(r['steps'])}")
    print("\n— max(value) and #candidates ≥ floor, vs breadth. If these plateau while distinct\n"
          "  targets keep growing, the HIGH-VALUE signal saturates early → modest breadth is right. —")
    vknees = [_value_knee(r["steps"]) for r in rows if r["steps"]]
    if vknees:
        print(f"median VALUE-saturation breadth (gen_samples): {st.median(vknees)}  "
              f"(range {min(vknees)}–{max(vknees)})")
    for i, s in enumerate(sweep[1:], 1):
        dmaxv = [r["steps"][i]["max_value"] - r["steps"][i - 1]["max_value"]
                 for r in rows if len(r["steps"]) > i]
        dtgt = [r["steps"][i]["distinct_targets"] - r["steps"][i - 1]["distinct_targets"]
                for r in rows if len(r["steps"]) > i]
        if dmaxv:
            print(f"  samples {sweep[i-1]}→{s}: avg Δmax_value {st.mean(dmaxv):+.3f} | "
                  f"avg +{st.mean(dtgt):.1f} distinct targets (coverage still climbing if >0)")


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
    p.add_argument("--scored", action="store_true",
                   help="Full-pipeline value saturation (max_value + #candidates ≥ floor per breadth) "
                        "instead of generation-only coverage. Costly: one full info-gain run per "
                        "(prompt, breadth).")
    args = p.parse_args(argv)

    prompts = [testbank.BY_ID[i] for i in args.ids if i in testbank.BY_ID]
    rows, t0 = [], time.time()

    if args.scored:
        base_cfg = infogain.eval_cfg(args.gen_model)
        base_cfg["mode"] = "focus"
        base_cfg["max_rounds"] = 1                       # isolate breadth (no cross-round refill)
        base_cfg["gen_temperature"] = args.temperature   # >0 so extra samples actually diversify
        base_cfg["families"] = {"enabled": False}        # isolate breadth from the families layer
        thr = base_cfg["discard_threshold"]
        for pr in prompts:
            print(f"… SCORED {pr['id']} ({pr['cat']}): info-gain at breadths {args.sweep}",
                  file=sys.stderr, flush=True)
            try:
                rows.append(scan_prompt_scored(pr, base_cfg, args.sweep, thr))
            except Exception as e:
                rows.append({"id": pr["id"], "cat": pr.get("cat"), "error": str(e), "steps": []})
            if args.out:
                with open(args.out, "w") as f:
                    json.dump({"rows": rows, "sweep": args.sweep, "scored": True,
                               "discard_threshold": thr,
                               "elapsed_s": round(time.time() - t0, 1)}, f, indent=2, default=str)
        render_scored([r for r in rows if not r.get("error")], args.sweep, thr)
        return 0

    model = pipeline.resolve_alias(args.gen_model)
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
