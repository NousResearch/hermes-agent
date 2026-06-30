#!/usr/bin/env python3
"""validate_evsi.py — does the rating predict REALIZED improvement? (Phase 1: P1a + P1c)

For each prompt: run info-gain to get ranked questions with PROJECTED scores
(delta_plan/stakes/prob per answer, plus U / EVSI / value). Then for each
(question, answer) pair, inject the answer as an established fact, RE-DERIVE the
baseline response, and measure the REALIZED change vs the no-evidence baseline
(a strong judge rates 0..1). One row per pair.

Downstream analysis (done separately):
  P1a calibration — does projected `delta_plan` correlate with `realized_change`?
                    does a question's projected EVSI/value track its realized value
                    (Σ_a P(a)·realized_change(a))?
  P1c ablations   — re-rank questions per prompt under alternative formulas
                    (√(U·EVSI), EVSI-only, max-Δ, U-only) and see which projected
                    ranking best matches the realized ranking.

Run on the host (immune to hermes container restarts), incremental writes:
  OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes \
    python3 evals/validate_evsi.py --out /path/evsi_validation.json
"""

import argparse
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import infogain  # noqa: E402
import pipeline  # noqa: E402
import voi  # noqa: E402

# Prompts come from the shared categorized bank (evals/testbank.py): LIFE (generic control)
# + BANK (agentic/tool-access/coding — the real target domain). usaw-calendar is intentionally
# not in the bank (benchmark showed a niche-domain/model failure, not a rating problem).
import testbank  # noqa: E402

PROMPTS = testbank.ALL


def change_judge(prompt, baseline, new, model, timeout):
    """0..1: how much RESPONSE B differs in substance/recommendation from baseline A."""
    p = ("Rate how much RESPONSE B differs from RESPONSE A — in substance, recommendation, and "
         "emphasis — as answers to the same prompt. 0 = effectively identical; 1 = a materially "
         "different approach/conclusion.\n\n"
         f"PROMPT:\n{prompt}\n\nRESPONSE A (baseline):\n{baseline}\n\nRESPONSE B:\n{new}\n\n"
         'Return ONLY a JSON object: {"change": 0.0}.')
    obj, _ = pipeline._call_json(model, p, timeout, num_predict=120)
    return voi.clamp01(obj.get("change", 0.0)) if isinstance(obj, dict) else None


def run_prompt(pr, cfg, judge_model, max_answers, timeout, source="bucket"):
    result = infogain.run(pr["problem"], cfg)
    plan_model = pipeline.resolve_alias(cfg["plan_model"])
    baseline = (result.get("framing") or {}).get("baseline_plan", "")
    rows = []
    # source="all_scored" tests realized_change across the WHOLE value spectrum (incl. questions
    # below the discard threshold) — needed in the agentic domain where most fall below the
    # life-tuned cutoff, and it yields the improvement-vs-value (diminishing_floor) curve.
    for q in result.get(source, []):
        answers = sorted((q.get("answers") or []),
                         key=lambda a: -voi.clamp01(a.get("prob", 0)))[:max_answers]
        for a in answers:
            fact = f"{q['question']} -> {a.get('answer', '')}"
            new, _ = pipeline.frame_and_plan(pr["problem"], plan_model, timeout, evidence=[fact])
            new_resp = (new or {}).get("baseline_plan", "")
            realized = change_judge(pr["problem"], baseline, new_resp, judge_model, timeout)
            rows.append({
                "prompt": pr["id"], "question": q["question"][:120],
                "target": q.get("target"), "answer": (a.get("answer") or "")[:90],
                "prob": round(voi.clamp01(a.get("prob", 0)), 3),
                "projected_delta": round(voi.clamp01(a.get("delta_plan", 0)), 3),
                "stakes": round(voi.clamp01(a.get("stakes", 0)), 3),
                "realized_change": None if realized is None else round(realized, 3),
                "q_u": round(q.get("u", 0), 3), "q_evsi": round(q.get("evsi", 0), 3),
                "q_value": round(q.get("value", 0), 3),
            })
            print(f"    pair: {pr['id']} | Δproj={a.get('delta_plan')} realized={realized} "
                  f"| {q['question'][:45]}", file=sys.stderr, flush=True)
    return rows, baseline


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--prompt-ids", nargs="*")
    p.add_argument("--gen-model", default="fast", help="all info-gain stages (cheap, deterministic).")
    p.add_argument("--judge-model", default="deepseek", help="change judge (strong).")
    p.add_argument("--max-answers", type=int, default=3, help="top-N answers per question to test.")
    p.add_argument("--source", choices=["bucket", "all_scored"], default="bucket",
                   help="bucket = survivors only; all_scored = every scored candidate (full spectrum).")
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args(argv)

    cfg = dict(infogain.DEFAULTS)
    for k in ("plan_model", "question_gen_model", "answer_model"):
        cfg[k] = args.gen_model
    # value_judge_model stays at the shipped default (deepseek) so the projected_delta
    # we validate is the REAL judge's, not a cheap stand-in.
    cfg["max_rounds"] = 1
    cfg["mode"] = "focus"
    judge_model = pipeline.resolve_alias(args.judge_model)  # alias -> real model name

    prompts = [x for x in PROMPTS if not args.prompt_ids or x["id"] in args.prompt_ids]
    rows, t0 = [], time.time()
    for pr in prompts:
        print(f"… {pr['id']}: info-gain + realized-change per (question, answer)", file=sys.stderr, flush=True)
        try:
            prows, _ = run_prompt(pr, cfg, judge_model, args.max_answers, args.timeout, args.source)
        except Exception as e:
            prows = [{"prompt": pr["id"], "error": str(e)}]
        rows.extend(prows)
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"rows": rows, "n": len(rows), "partial": True,
                           "gen_model": args.gen_model, "judge_model": args.judge_model,
                           "elapsed_s": round(time.time() - t0, 1)}, f, indent=2, default=str)

    out = {"rows": rows, "n": len(rows), "partial": False,
           "gen_model": args.gen_model, "judge_model": args.judge_model,
           "elapsed_s": round(time.time() - t0, 1)}
    payload = json.dumps(out, indent=2, default=str)
    if args.out:
        with open(args.out, "w") as f:
            f.write(payload)
        print(f"wrote {args.out} ({len(rows)} pairs, {out['elapsed_s']}s)", file=sys.stderr)
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
