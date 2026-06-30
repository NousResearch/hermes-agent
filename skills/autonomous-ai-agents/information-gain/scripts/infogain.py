#!/usr/bin/env python3
"""infogain.py — information-gain (value-of-information) analysis for a problem.

Given an underspecified problem, this orchestrates a research-grounded
Expected-Value-of-Sample-Information pipeline: it interrogates the prompt into
candidate questions, projects plausible answers, estimates how much each answer
would change the recommended plan (× stakes), scores each question's value, and
keeps generating fresh questions until a diverse bucket of genuinely high-value
questions is filled (or a round cap is hit). It then REPORTS the ranked questions
with recommendations (pre-answer / assume-default) — it does not act on them.

Usage:
    python3 infogain.py "Sync USAW events into our calendar"
    python3 infogain.py -p "Build an internal search tool" --json
    python3 infogain.py "<problem>" --dry-run        # show prompts, no model calls
    python3 infogain.py "<problem>" -o /tmp/report.md

Tunables: module defaults  ←  INFOGAIN_* env vars  ←  CLI flags.
Exit codes: 0 ok, 1 error, 2 Ollama unreachable, 3 no problem given.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline  # noqa: E402
import voi  # noqa: E402
from pipeline import resolve_alias  # noqa: E402

# ── config defaults (overridable via INFOGAIN_* env, then CLI) ────────────────
DEFAULTS = {
    "plan_model": "glm",
    "question_gen_model": "glm",
    "answer_model": "fast",
    "value_judge_model": "deepseek",
    "consolidate_model": "fast",
    "min_bucket_size": 3,
    "target_bucket_size": 5,
    "hard_cap": 7,
    "discard_threshold": 0.40,
    "pre_answer_threshold": 0.60,
    "refill_floor": 0.30,
    "questions_per_round": 6,
    "gen_samples": 1,
    "gen_temperature": 0.0,
    "answers_per_question": 5,
    "max_rounds": 3,
    "mmr_lambda": 0.4,
    "plan_timeout": 180,
    "gen_timeout": 180,
    "answer_timeout": 120,
    "judge_timeout": 150,
}
_INT = {"min_bucket_size", "target_bucket_size", "hard_cap", "questions_per_round",
        "answers_per_question", "max_rounds", "gen_samples", "plan_timeout", "gen_timeout",
        "answer_timeout", "judge_timeout"}
_MODEL_KEYS = ("plan_model", "question_gen_model", "answer_model", "value_judge_model",
               "consolidate_model")

# Named presets over the breadth knobs. "focus" = the (research-grounded) prioritized
# top-few default; "breadth" = wider coverage via more questions and more rounds, a
# bigger bucket, and a lower keep floor. Breadth needs no new generation logic — each
# round's "avoid the already-asked questions" instruction drives it toward new
# dimensions. Resolution order: DEFAULTS ← mode preset ← INFOGAIN_* env ← CLI flag.
MODES = {
    "focus": {},
    "breadth": {
        # Breadth comes from SAMPLING the model's question distribution (3 independent
        # draws at high temperature, unioned + deduped) — not from a seeded topic list.
        # More rounds + the avoid-list still add cross-round novelty.
        "gen_samples": 3,
        "gen_temperature": 0.9,
        "questions_per_round": 8,
        "max_rounds": 3,
        "target_bucket_size": 14,
        "hard_cap": 18,
        "discard_threshold": 0.30,
        "refill_floor": 0.25,
        "answers_per_question": 4,
    },
}


def _cast(key, val):
    if key in _INT:
        return int(val)
    if key in _MODEL_KEYS:
        return val
    return float(val)


def _env_default(key):
    val = os.environ.get("INFOGAIN_" + key.upper())
    return DEFAULTS[key] if val is None else _cast(key, val)


# ── orchestration ─────────────────────────────────────────────────────────────


def run(problem, cfg, progress=None, trace=False, evidence=None):
    """Run the full bucket-fill loop. Returns a result dict (see keys below).

    `evidence` is a list of already-established facts (the iterative loop): they are
    woven into framing, generation (don't re-ask), and answer-projection (resolved
    questions read as derivable and drop out).

    When trace=True, result['trace'] captures a 'show your work' record: each
    stage's prompt + raw model output, the per-question scoring arithmetic, and
    the per-round selection / stop decisions.
    """
    def log(msg):
        if progress:
            progress(msg)

    pipeline.reset_usage()
    _t0 = time.time()
    plan_model = resolve_alias(cfg["plan_model"])
    qg_model = resolve_alias(cfg["question_gen_model"])
    ans_model = resolve_alias(cfg["answer_model"])
    judge_model = resolve_alias(cfg["value_judge_model"])
    cons_model = resolve_alias(cfg["consolidate_model"])

    log(f"framing problem + baseline plan via {plan_model} ...")
    framing_sink = [] if trace else None
    framing, ferr = pipeline.frame_and_plan(problem, plan_model, cfg["plan_timeout"],
                                            sink=framing_sink, evidence=evidence)
    baseline_plan = framing.get("baseline_plan", "")
    trace_obj = ({"models": {"plan": plan_model, "question_gen": qg_model,
                             "answer": ans_model, "value_judge": judge_model,
                             "consolidate": cons_model},
                  "framing": (framing_sink[0] if framing_sink else None),
                  "rounds": []} if trace else None)

    seen, scored_all = [], []
    rounds_used = 0
    for _ in range(cfg["max_rounds"]):
        rounds_used += 1
        avoid = [r["question"] for r in seen]
        log(f"round {rounds_used}: generating {cfg['questions_per_round']} "
            f"questions via {qg_model} ...")
        gen_sink = [] if trace else None
        new_qs, _ = pipeline.generate_questions(
            problem, framing, qg_model, cfg["questions_per_round"], avoid,
            cfg["gen_timeout"], sink=gen_sink,
            samples=cfg["gen_samples"], temperature=cfg["gen_temperature"], evidence=evidence)
        cons_sink = None
        if cfg["gen_samples"] > 1 and len(new_qs) > 1:
            log(f"round {rounds_used}: consolidating {len(new_qs)} sampled candidates "
                f"via {cons_model} ...")
            cons_sink = [] if trace else None
            new_qs = pipeline.consolidate_questions(
                problem, new_qs, cons_model, cfg["gen_timeout"], sink=cons_sink)
        dropped = [q["question"] for q in new_qs if voi.is_duplicate(q, seen)]
        fresh = [q for q in new_qs if not voi.is_duplicate(q, seen)]
        seen.extend(fresh)
        if not fresh:
            log("round produced no new questions; stopping.")
            if trace:
                trace_obj["rounds"].append({
                    "round": rounds_used,
                    "generation": (gen_sink[0] if gen_sink else None),
                    "consolidation": (cons_sink[0] if cons_sink else None),
                    "dropped_as_duplicate": dropped, "questions": [],
                    "stop_reason": "no new questions"})
            break

        log(f"round {rounds_used}: projecting answers ({ans_model}) + "
            f"judging plan-change ({judge_model}) for {len(fresh)} questions ...")
        fresh = pipeline.project_answers_batch(
            problem, framing, fresh, ans_model, cfg["answers_per_question"],
            cfg["answer_timeout"], capture=trace, evidence=evidence)
        fresh = pipeline.judge_plan_change_batch(
            problem, framing, baseline_plan, fresh, judge_model, cfg["judge_timeout"],
            capture=trace)
        for r in fresh:
            voi.score_record(r)
        scored_all.extend(fresh)

        bucket, _ = voi.rank_and_select(
            scored_all, discard_threshold=cfg["discard_threshold"],
            pre_answer_threshold=cfg["pre_answer_threshold"],
            hard_cap=cfg["hard_cap"], mmr_lambda=cfg["mmr_lambda"])
        best_fresh = voi.best_value(fresh)
        log(f"round {rounds_used}: bucket={len(bucket)} "
            f"(target {cfg['target_bucket_size']}), best fresh value={best_fresh:.2f}")

        stop = None
        if len(bucket) >= cfg["target_bucket_size"]:
            stop = f"target bucket ({cfg['target_bucket_size']}) reached"
        elif len(bucket) >= cfg["min_bucket_size"] and best_fresh < cfg["refill_floor"]:
            stop = (f"min bucket ({cfg['min_bucket_size']}) reached and best fresh "
                    f"{best_fresh:.2f} < refill_floor ({cfg['refill_floor']})")

        if trace:
            trace_obj["rounds"].append({
                "round": rounds_used,
                "generation": (gen_sink[0] if gen_sink else None),
                "consolidation": (cons_sink[0] if cons_sink else None),
                "dropped_as_duplicate": dropped,
                "questions": [_trace_question(r) for r in fresh],
                "bucket_after_round": len(bucket),
                "best_fresh_value": round(best_fresh, 4),
                "stop_reason": stop or ("max_rounds reached"
                                        if rounds_used >= cfg["max_rounds"] else "continue"),
            })

        if stop is not None:
            log(stop + "; stopping.")
            break

    bucket, discarded = voi.rank_and_select(
        scored_all, discard_threshold=cfg["discard_threshold"],
        pre_answer_threshold=cfg["pre_answer_threshold"],
        hard_cap=cfg["hard_cap"], mmr_lambda=cfg["mmr_lambda"])

    usage = pipeline.get_usage()
    usage["wall_seconds"] = round(time.time() - _t0, 1)
    result = {
        "problem": problem,
        "evidence": list(evidence or []),
        "usage": usage,
        "framing": framing,
        "framing_error": ferr,
        "config": cfg,
        "rounds_used": rounds_used,
        "candidates_considered": len(scored_all),
        "bucket": bucket,
        "discarded_count": len(discarded),
        "min_met": len(bucket) >= cfg["min_bucket_size"],
        "pre_answer": [r for r in bucket if r.get("recommendation") == "PRE_ANSWER"],
    }
    if trace:
        result["trace"] = trace_obj
    return result


def _trace_question(r):
    """Compact per-question trace: inputs, the captured model calls, and the math."""
    return {
        "question": r.get("question"),
        "target": r.get("target"),
        "type": r.get("type"),
        "why": r.get("why"),
        "derivable_prob": r.get("derivable_prob"),
        "answers": [{"answer": a.get("answer"), "prob": a.get("prob"),
                     "delta_plan": a.get("delta_plan"), "stakes": a.get("stakes")}
                    for a in (r.get("answers") or [])],
        "breakdown": voi.score_breakdown(r),
        "gated_out": r.get("gated_out"),
        "value": r.get("value"),
        "recommendation": r.get("recommendation"),
        "answers_call": (r.get("_trace") or {}).get("project"),
        "judge_call": (r.get("_trace") or {}).get("judge"),
    }


# ── rendering ─────────────────────────────────────────────────────────────────


def _template():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "templates", "report.md")
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return _FALLBACK_TEMPLATE


_FALLBACK_TEMPLATE = """# Key Questions to Improve the Response

**Prompt:** {{problem}}
{{evidence}}
**Goal:** {{goal}}
**Response type:** {{decision}}

**Baseline response (best answer right now):**
{{baseline_plan}}

## Key questions, ranked by weight (answer these to improve the response)
weight = exploration value = answerability × √(uncertainty × value-of-answering)

{{ranked_list}}

{{discarded_note}}

<details><summary>Detailed scores</summary>

{{table}}

</details>

---
{{meta}}
"""


def _fmt_default(rec):
    m = rec.get("modal_answer")
    if not m:
        return "—"
    return f"{m.get('answer', '')[:80]} (p≈{voi.clamp01(m.get('prob', 0)):.2f})"


def _weight_clarification(rec):
    """Plain-language explanation of what a question's weight means — built from the
    score components, no extra model call."""
    ev = rec.get("evsi", 0.0)
    ans = voi.clamp01(rec.get("answerability", 1.0))
    modal = rec.get("modal_answer") or {}
    p = voi.clamp01(modal.get("prob", 0.0))
    change = "substantially" if ev >= 0.50 else "moderately" if ev >= 0.30 else "only slightly"
    how_answerable = ("readily answerable" if ans >= 0.7 else
                      "answerable with some effort" if ans >= 0.4 else "hard to resolve")
    default = (modal.get("answer", "") or "—")[:70]
    return (f"answering would **{change}** improve the response "
            f"(value-of-answering {ev:.2f}); otherwise you'd assume “{default}” "
            f"(~{(1.0 - p) * 100:.0f}% chance that's off in a way that matters); "
            f"{how_answerable} (answerability {ans:.2f}).")


def _ranked_list(bucket):
    """The headline output: key questions ranked by weight, each with a clarification
    of what its weight means."""
    if not bucket:
        return ("_No questions worth answering — the prompt is already specified well enough "
                "for a good response._")
    out = []
    for i, r in enumerate(bucket):
        out.append(
            f"{i + 1}. **[weight {r['value']:.2f}]** {r['question']}  "
            f"_({r.get('recommendation', '')})_\n"
            f"   - *what the weight means:* {_weight_clarification(r)}\n"
            f"   - *resolves:* {r.get('target', '') or '—'}")
    return "\n".join(out)


def render_markdown(result):
    fr = result["framing"]
    bucket = result["bucket"]

    if result["pre_answer"]:
        pre = "\n".join(
            f"{i + 1}. **{r['question']}**  \n   _why:_ {r.get('why', '')}  \n"
            f"   _assume if skipped:_ {_fmt_default(r)}"
            for i, r in enumerate(result["pre_answer"]))
    else:
        pre = "_None above the pre-answer threshold — the problem is well enough " \
              "specified to proceed (resolve any ASSUME-DEFAULT items if convenient)._"

    rows = ["| # | value | uncert | answer-value | answerable | rec | question | resolves | assume-if-skipped |",
            "|---|------:|------:|------:|------:|-----|----------|----------|-------------------|"]
    for i, r in enumerate(bucket):
        rows.append(
            f"| {i + 1} | {r['value']:.2f} | {r['u']:.2f} | {r['evsi']:.2f} | "
            f"{r.get('answerability', 1.0):.2f} | {r.get('recommendation', '')} | "
            f"{r['question']} | {r.get('target', '') or '—'} | {_fmt_default(r)} |")
    table = "\n".join(rows) if bucket else "_No valuable questions found._"

    note = (f"_{result['discarded_count']} lower-value/redundant question(s) "
            f"discarded._") if result["discarded_count"] else ""
    if not result["min_met"]:
        note += (f"\n\n> Bucket holds {len(bucket)} question(s), below the minimum of "
                 f"{result['config']['min_bucket_size']}, after "
                 f"{result['rounds_used']} round(s). This usually means the problem "
                 f"is already fairly well-specified.")

    u = result.get("usage") or {}
    meta = (f"_mode={result['config'].get('mode', 'focus')} · "
            f"models: plan={result['config']['question_gen_model']}, "
            f"answers={result['config']['answer_model']}, "
            f"judge={result['config']['value_judge_model']} · "
            f"rounds={result['rounds_used']} · candidates={result['candidates_considered']} · "
            f"{u.get('calls', 0)} calls · {u.get('input_tokens', 0)}+{u.get('output_tokens', 0)} tok · "
            f"{u.get('wall_seconds', 0)}s wall_")

    crit = fr.get("success_criteria") or []
    crit_str = "; ".join(crit) if isinstance(crit, list) else str(crit)

    ev = result.get("evidence") or []
    evidence_str = ("\n**Evidence folded in (already established):**\n"
                    + "\n".join(f"- {e}" for e in ev) + "\n") if ev else ""

    out = _template()
    for k, v in {
        "{{problem}}": result["problem"],
        "{{evidence}}": evidence_str,
        "{{goal}}": fr.get("goal", "") or "—",
        "{{decision}}": fr.get("decision", "") or "—",
        "{{success_criteria}}": crit_str or "—",
        "{{baseline_plan}}": fr.get("baseline_plan", "") or "—",
        "{{preanswer_list}}": pre,
        "{{ranked_list}}": _ranked_list(bucket),
        "{{table}}": table,
        "{{discarded_note}}": note,
        "{{meta}}": meta,
    }.items():
        out = out.replace(k, str(v))
    return out


def _clip(s, n=800):
    s = s or ""
    return s if len(s) <= n else s[:n] + f"\n…[+{len(s) - n} more chars]"


def _render_trace_question(q):
    b = q["breakdown"]
    lines = [f"\n#### Q: {q['question']}",
             f"_resolves: {q.get('target') or '—'} · type: {q.get('type') or '—'}_  ",
             f"_why: {q.get('why', '')}_\n",
             "| projected answer | P | Δplan | stakes | P·Δ·stakes |",
             "|---|---:|---:|---:|---:|"]
    for t in b["evsi_terms"]:
        lines.append(f"| {str(t['answer'])[:64]} | {t['p']:.2f} | {t['delta_plan']:.2f} "
                     f"| {t['stakes']:.2f} | {t['term']:.3f} |")
    ev = " + ".join(f"{t['term']:.3f}" for t in b["evsi_terms"]) or "0"
    rec = q.get("recommendation")
    if q.get("gated_out"):
        decision = "GATED OUT — discarded (no reducible uncertainty, or no expected plan-change)"
    elif rec in ("PRE_ANSWER", "ASSUME_DEFAULT"):
        decision = f"KEPT → {rec}"
    elif rec == "SKIP":
        decision = "discarded — value below the discard threshold"
    elif rec == "REDUNDANT":
        decision = "discarded — redundant with a higher-value question (same target)"
    elif rec == "OVERFLOW":
        decision = "discarded — valuable but beyond the bucket cap"
    else:
        decision = rec or "—"
    lines += [
        f"\nderivable_prob = {b['derivable_prob']:.2f} · answerability = {b['answerability']:.2f}\n",
        "**scoring (show your work):**  ",
        f"- uncertainty U = entropy {b['entropy']:.2f} × (1 − derivable {b['derivable_prob']:.2f}) "
        f"= **{b['u']:.3f}**  ",
        f"- value-of-answering (EVSI) = Σ P·Δplan·stakes = {ev} = **{b['evsi']:.3f}**  ",
        f"- worth-if-resolved = √(U × answer-value) = √({b['u']:.3f} × {b['evsi']:.3f}) "
        f"= **{b['intrinsic_value']:.3f}**  ",
        f"- exploration value = answerability {b['answerability']:.2f} × {b['intrinsic_value']:.3f} "
        f"= **{b['value']:.3f}**  ",
        f"- decision: **{decision}**\n",
    ]
    return "\n".join(lines)


def _render_bucket_table(bucket):
    rows = ["| # | value | rec | question | resolves |",
            "|---|------:|-----|----------|----------|"]
    for i, r in enumerate(bucket):
        rows.append(f"| {i + 1} | {r['value']:.2f} | {r.get('recommendation', '')} | "
                    f"{r['question']} | {r.get('target', '') or '—'} |")
    return "\n".join(rows)


def render_trace(result):
    """A 'show your work' document: prompts, raw model output, and scoring arithmetic."""
    t = result.get("trace")
    if not t:
        return "No trace captured — run with --trace."
    fr = result["framing"]
    m = t["models"]
    out = ["# Information-Gain — show your work\n",
           f"**Problem:** {result['problem']}\n",
           f"**Models:** plan/gen=`{m['question_gen']}` · answers=`{m['answer']}` · "
           f"judge=`{m['value_judge']}`\n",
           "\n## Stage 0 — frame the problem + baseline plan\n"]
    f = t.get("framing") or {}
    out += [
        "**Prompt sent:**\n```\n" + _clip(f.get("prompt", ""), 700) + "\n```",
        "**Raw model output:**\n```\n" + _clip(f.get("raw", ""), 700) + "\n```",
        f"**Parsed →**\n- goal: {fr.get('goal', '')}\n- decision: {fr.get('decision', '')}\n"
        f"- baseline plan: {fr.get('baseline_plan', '')}\n",
    ]
    for rd in t["rounds"]:
        out.append(f"\n## Round {rd['round']} — generate → project → judge → score\n")
        g = rd.get("generation") or {}
        out += [
            f"### Stage 1 — generate questions (`{g.get('model', '?')}`)",
            "**Prompt sent:**\n```\n" + _clip(g.get("prompt", ""), 650) + "\n```",
            "**Raw model output:**\n```\n" + _clip(g.get("raw", ""), 900) + "\n```",
        ]
        if rd.get("dropped_as_duplicate"):
            out.append("**Dropped as duplicate of an earlier round:** "
                       + "; ".join(rd["dropped_as_duplicate"]))
        if rd.get("questions"):
            out.append("\n### Stages 2-4 — per question (answers → Δplan/stakes → score)")
            for q in rd["questions"]:
                out.append(_render_trace_question(q))
        out.append(f"**Round {rd['round']} decision:** bucket now "
                   f"{rd.get('bucket_after_round', '?')}, best fresh value "
                   f"{rd.get('best_fresh_value', '?')} → **{rd.get('stop_reason', '?')}**\n")
    out.append("\n## Final ranked bucket\n")
    out.append(_render_bucket_table(result["bucket"]) if result["bucket"]
               else "_empty — the problem is already well-specified._")
    return "\n".join(out)


def _dry_run(problem, cfg, evidence=None):
    framing_stub = {"goal": "<goal from stage 0>", "decision": "<decision from stage 0>"}
    q_stub = {"question": "<a candidate question>"}
    a_stub = [{"answer": "<answer 1>"}, {"answer": "<answer 2>"}]
    sep = "\n" + "=" * 72 + "\n"
    print(sep.join([
        "DRY RUN — prompts only, no model calls.",
        "STAGE 0 — frame_and_plan:\n\n" + pipeline.frame_prompt(problem, evidence),
        "STAGE 1 — generate_questions:\n\n" + pipeline.questions_prompt(
            problem, framing_stub, cfg["questions_per_round"],
            avoid=["<already-considered question>"], evidence=evidence),
        "STAGE 2 — project_answers (per question, parallel):\n\n"
        + pipeline.answers_prompt(problem, framing_stub, q_stub["question"],
                                  cfg["answers_per_question"], evidence),
        "STAGE 3 — judge_plan_change (per question, parallel):\n\n"
        + pipeline.judge_prompt(problem, framing_stub, "<baseline plan from stage 0>",
                                q_stub["question"], a_stub),
    ]))


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser():
    p = argparse.ArgumentParser(
        description="Information-gain (value-of-information) analysis of a problem.")
    p.add_argument("problem_pos", nargs="*", help="The problem statement.")
    p.add_argument("-p", "--problem", help="Problem statement (alternative to positional).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON instead of markdown.")
    p.add_argument("--dry-run", action="store_true", help="Print stage prompts; make no model calls.")
    p.add_argument("--trace", action="store_true",
                   help="Show your work: capture every stage's prompt + raw model output and "
                        "the per-question scoring arithmetic, then print the full trace.")
    p.add_argument("-o", "--output", help="Write the report to this file.")
    p.add_argument("--evidence", nargs="*", default=None, metavar="FACT",
                   help="Already-established facts/answers to fold into the context (the "
                        "iterative loop). Pass several, e.g. --evidence \"budget: $0\" \"users: coaches\".")
    p.add_argument("--evidence-file", help="File with one established fact per line ('#' comments ok).")
    p.add_argument("--quiet", action="store_true", help="Suppress progress logging on stderr.")
    p.add_argument("--mode", choices=list(MODES), default=None,
                   help="Preset over the breadth knobs: 'focus' (default — prioritized top few) "
                        "or 'breadth' (wider coverage: more questions/rounds, bigger bucket). "
                        "Individual flags and INFOGAIN_* env vars override the preset.")
    # tunable overrides
    for key in DEFAULTS:
        flag = "--" + key.replace("_", "-")
        if key in _INT:
            p.add_argument(flag, type=int)
        elif key in _MODEL_KEYS:
            p.add_argument(flag, type=str)
        else:
            p.add_argument(flag, type=float)
    return p


def resolve_config(args):
    mode = getattr(args, "mode", None) or os.environ.get("INFOGAIN_MODE") or "focus"
    overrides = MODES.get(mode, {})
    cfg = {"mode": mode}
    for key in DEFAULTS:
        cli = getattr(args, key, None)
        env = os.environ.get("INFOGAIN_" + key.upper())
        if cli is not None:
            cfg[key] = cli
        elif env is not None:
            cfg[key] = _cast(key, env)
        elif key in overrides:
            cfg[key] = overrides[key]
        else:
            cfg[key] = DEFAULTS[key]
    return cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    problem = args.problem or " ".join(args.problem_pos).strip()
    if not problem:
        print("Error: no problem given. Pass it positionally or with --problem.",
              file=sys.stderr)
        return 3

    cfg = resolve_config(args)

    evidence = list(args.evidence or [])
    if args.evidence_file:
        with open(args.evidence_file) as f:
            evidence += [ln.strip() for ln in f
                         if ln.strip() and not ln.lstrip().startswith("#")]

    if args.dry_run:
        _dry_run(problem, cfg, evidence)
        return 0

    if not pipeline.ollama_reachable():
        print(f"Error: Ollama not reachable at {pipeline.OLLAMA_URL}. "
              "Is the daemon running? (override with OLLAMA_URL)", file=sys.stderr)
        return 2

    progress = None if args.quiet else (lambda m: print(f"… {m}", file=sys.stderr, flush=True))
    result = run(problem, cfg, progress=progress, trace=args.trace, evidence=evidence)

    if args.json:
        rendered = json.dumps(result, indent=2, default=str)
    elif args.trace:
        rendered = render_trace(result)
    else:
        rendered = render_markdown(result)

    if args.output:
        with open(args.output, "w") as f:
            f.write(rendered)
        print(f"✅ wrote {args.output} "
              f"({len(result['bucket'])} questions, {result['rounds_used']} rounds)",
              file=sys.stderr)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
