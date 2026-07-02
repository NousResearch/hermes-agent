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


def change_judge_graded(prompt, baseline, new, model, timeout):
    """Anchored variant of change_judge. The original names only the 0/1 endpoints and SATURATES
    (71% of realized_change lands exactly on 0 or 1 — coarse ground truth that caps every
    correlation). Mid-scale anchors force partial revisions to be graded. Same JSON contract;
    opt-in via --graded-change-judge. NOTE: a different instrument — its scores are NOT comparable
    with prior runs' realized_change; A/B it against the original on stored responses via
    evals/rejudge.py before adopting."""
    p = ("Rate how much RESPONSE B differs from RESPONSE A — in substance, recommendation, and "
         "emphasis — as answers to the same prompt. Use the FULL scale, not just the endpoints:\n"
         "  0.0 = effectively identical\n"
         "  0.2 = same plan; wording or emphasis shifts only\n"
         "  0.4 = same approach; one substantive detail changed (a step added, removed, or "
         "re-parameterized)\n"
         "  0.7 = approach kept, but the recommendation or priorities shift materially\n"
         "  1.0 = a different approach or conclusion\n\n"
         f"PROMPT:\n{prompt}\n\nRESPONSE A (baseline):\n{baseline}\n\nRESPONSE B:\n{new}\n\n"
         'First think in one short sentence, then return JSON: {"reason": "...", "change": 0.0}.')
    obj, _ = pipeline._call_json(model, p, timeout, num_predict=200)
    return voi.clamp01(obj.get("change", 0.0)) if isinstance(obj, dict) else None


def stakes_judge(prompt, baseline, new, model, timeout):
    """0..1: realized STAKES — how CONSEQUENTIAL the difference is, independent of its size.

    Measured separately from projected `stakes`, so realized EVSI (= realized_change ×
    realized_stakes) breaks the projected-stakes confound that nullified the P1a "validation".
    """
    p = ("Two responses A and B answer the same prompt and differ. IGNORING how large the "
         "difference is, rate how much it MATTERS for the user getting a good result — would a "
         "knowledgeable user care which one they received? Use the FULL range:\n"
         "  0.0 = wouldn't care, both serve the need equally well\n"
         "  0.3 = mild preference for the better one\n"
         "  0.6 = clearly wants the better one\n"
         "  1.0 = the worse one fails their actual need\n\n"
         f"PROMPT:\n{prompt}\n\nRESPONSE A:\n{baseline}\n\nRESPONSE B:\n{new}\n\n"
         'First think in one short sentence, then return JSON: {"reason": "...", "stakes": 0.0}.')
    obj, _ = pipeline._call_json(model, p, timeout, num_predict=200)
    return voi.clamp01(obj.get("stakes", 0.0)) if isinstance(obj, dict) else None


def _snapshot(records):
    """Freeze the PROJECTED scores of a record set keyed by object id, BEFORE any re-judging mutates
    the per-answer delta_plan/stakes (and before re-score_record overwrites q-level u/evsi/value).
    Returns {"q": {id(q): {u,evsi,value}}, "a": {id(q): {id(a): (delta, stakes)}}}."""
    q_scores, a_scores = {}, {}
    for q in records:
        q_scores[id(q)] = {"u": q.get("u", 0.0), "evsi": q.get("evsi", 0.0),
                           "value": q.get("value", 0.0)}
        a_scores[id(q)] = {id(a): (voi.clamp01(a.get("delta_plan", 0.0)),
                                   voi.clamp01(a.get("stakes", 0.0)))
                           for a in (q.get("answers") or [])}
    return {"q": q_scores, "a": a_scores}


def run_prompt(pr, cfg, judge_model, max_answers, timeout, source="bucket", ab=False,
               keep_responses=False, change_fn=None):
    """One prompt → realized-vs-projected rows. With ab=True, BOTH elicitation methods (absolute +
    pairwise) are scored on the SAME question/answer set and the realized measurement (re-derive +
    judges) is shared across them — so the per-method within-task ranking is a clean A/B and the
    only added cost is the pairwise re-judge calls, not a second realized pass."""
    result = infogain.run(pr["problem"], cfg)  # absolute judge (default cfg)
    plan_model = pipeline.resolve_alias(cfg["plan_model"])
    elicit_model = pipeline.resolve_alias(cfg["value_judge_model"])  # same model, both elicitations
    framing = result.get("framing") or {}
    baseline = framing.get("baseline_plan", "")
    records = result.get(source, [])

    # Snapshot ABSOLUTE projected scores first (the default run produced them), THEN, for ab, re-judge
    # the same records with the pairwise judge and snapshot those. Realized is measured once below.
    methods = {"absolute": _snapshot(records)}
    if ab:
        pipeline.judge_plan_change_pairwise_batch(pr["problem"], framing, baseline, records,
                                                  elicit_model, cfg["judge_timeout"])
        for q in records:
            voi.score_record(q)
        methods["pairwise"] = _snapshot(records)

    rows = []
    # source="all_scored" tests across the WHOLE value spectrum (incl. below-threshold questions) —
    # needed in the agentic domain and for the within-task ranking comparison (more questions/prompt).
    for q in records:
        answers = sorted((q.get("answers") or []),
                         key=lambda a: -voi.clamp01(a.get("prob", 0)))[:max_answers]
        for a in answers:
            fact = f"{q['question']} -> {a.get('answer', '')}"
            new, _ = pipeline.frame_and_plan(pr["problem"], plan_model, timeout, evidence=[fact])
            new_resp = (new or {}).get("baseline_plan", "")
            realized = (change_fn or change_judge)(pr["problem"], baseline, new_resp,
                                                   judge_model, timeout)
            r_stakes = stakes_judge(pr["problem"], baseline, new_resp, judge_model, timeout)
            regret = None if (realized is None or r_stakes is None) else realized * r_stakes
            shared = {
                "prompt": pr["id"], "cat": pr.get("cat"), "question": q["question"][:120],
                "target": q.get("target"), "answer": (a.get("answer") or "")[:90],
                # lens/family carried from the scored record so analyze_evsi can attribute realized
                # value per lens (e.g. premortem vs the rest). Flat generator leaves these empty.
                "lens": q.get("lens") or "", "family": q.get("family") or "",
                "prob": round(voi.clamp01(a.get("prob", 0)), 3),
                "realized_change": None if realized is None else round(realized, 3),
                "realized_stakes": None if r_stakes is None else round(r_stakes, 3),
                "realized_regret": None if regret is None else round(regret, 3),  # method-independent
            }
            if keep_responses:
                # full texts so judge experiments can RE-JUDGE offline (evals/rejudge.py) without
                # re-paying the frame_and_plan re-derivation — the expensive half of a realized pass
                shared["baseline_resp"] = baseline
                shared["new_resp"] = new_resp
            for method, snap in methods.items():
                qd, ad = snap["q"][id(q)], snap["a"][id(q)].get(id(a), (0.0, 0.0))
                rows.append({**shared, "method": method,
                             "projected_delta": round(ad[0], 3), "stakes": round(ad[1], 3),
                             "q_u": round(qd["u"], 3), "q_evsi": round(qd["evsi"], 3),
                             "q_value": round(qd["value"], 3)})
            print(f"    pair: {pr['id']} | realized={realized} r_stakes={r_stakes} | "
                  f"{q['question'][:40]}", file=sys.stderr, flush=True)
    return rows, baseline


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--prompt-ids", nargs="*")
    p.add_argument("--gen-model", default="fast", help="all info-gain stages (cheap, deterministic).")
    p.add_argument("--judge-model", default="deepseek", help="realized change/stakes judge (strong).")
    p.add_argument("--elicit-model", default=None,
                   help="override the value_judge_model used for ABSOLUTE + PAIRWISE elicitation "
                        "(default: keep the shipped deepseek). Set a host-local model (e.g. fast, "
                        "gpt-oss:20b) for a host A/B where cloud judges aren't reachable. Both arms "
                        "share it, so the A/B stays fair.")
    p.add_argument("--max-answers", type=int, default=3, help="top-N answers per question to test.")
    p.add_argument("--source", choices=["bucket", "all_scored"], default="bucket",
                   help="bucket = survivors only; all_scored = every scored candidate (full spectrum).")
    p.add_argument("--ab", action="store_true",
                   help="A/B both elicitation methods (absolute + pairwise) on the SAME question/answer "
                        "set, sharing the realized measurement. Emits two rows per pair tagged with "
                        "`method`; analyze_evsi reports per-method within-task ρ. (#24 gate.)")
    p.add_argument("--families", action="store_true",
                   help="run with the families layer on — rows carry lens/family for per-lens "
                        "attribution (default: flat generator, empty lens).")
    p.add_argument("--premortem", choices=["auto", "on", "off"], default="auto",
                   help="premortem lens setting when --families is on.")
    p.add_argument("--keep-responses", action="store_true",
                   help="store baseline + re-derived response TEXTS on each row so judge variants "
                        "can be A/B'd offline (evals/rejudge.py) without a second realized pass.")
    p.add_argument("--graded-change-judge", action="store_true",
                   help="use the anchored mid-scale change judge (de-saturated instrument). NOT "
                        "comparable with prior runs' realized_change — A/B via evals/rejudge.py "
                        "before adopting.")
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args(argv)

    cfg = dict(infogain.DEFAULTS)
    for k in ("plan_model", "question_gen_model", "answer_model"):
        cfg[k] = args.gen_model
    # value_judge_model (elicitation) defaults to the shipped deepseek so the projected_delta we
    # validate is the REAL judge's; --elicit-model overrides it (needed on a host where cloud judges
    # aren't reachable — both absolute and pairwise arms share it, keeping the A/B fair).
    if args.elicit_model:
        cfg["value_judge_model"] = args.elicit_model
    cfg["max_rounds"] = 1
    cfg["mode"] = "focus"
    if args.families:
        cfg["families"] = infogain.families_cfg(args.premortem, families_model=args.gen_model)
    judge_model = pipeline.resolve_alias(args.judge_model)  # alias -> real model name

    prompts = [x for x in PROMPTS if not args.prompt_ids or x["id"] in args.prompt_ids]
    rows, t0 = [], time.time()
    for pr in prompts:
        print(f"… {pr['id']}: info-gain + realized-change per (question, answer)", file=sys.stderr, flush=True)
        try:
            prows, _ = run_prompt(pr, cfg, judge_model, args.max_answers, args.timeout,
                                  args.source, ab=args.ab, keep_responses=args.keep_responses,
                                  change_fn=change_judge_graded if args.graded_change_judge else None)
        except Exception as e:
            prows = [{"prompt": pr["id"], "error": str(e)}]
        rows.extend(prows)
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"rows": rows, "n": len(rows), "partial": True,
                           "gen_model": args.gen_model, "judge_model": args.judge_model,
                           "change_judge": "graded" if args.graded_change_judge else "original",
                           "elapsed_s": round(time.time() - t0, 1)}, f, indent=2, default=str)

    out = {"rows": rows, "n": len(rows), "partial": False,
           "gen_model": args.gen_model, "judge_model": args.judge_model,
           "change_judge": "graded" if args.graded_change_judge else "original",
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
