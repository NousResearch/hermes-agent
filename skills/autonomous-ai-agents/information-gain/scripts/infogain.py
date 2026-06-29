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
    "min_bucket_size": 3,
    "target_bucket_size": 5,
    "hard_cap": 7,
    "discard_threshold": 0.40,
    "pre_answer_threshold": 0.60,
    "refill_floor": 0.30,
    "questions_per_round": 6,
    "answers_per_question": 5,
    "max_rounds": 3,
    "mmr_lambda": 0.4,
    "plan_timeout": 180,
    "gen_timeout": 180,
    "answer_timeout": 120,
    "judge_timeout": 150,
}
_INT = {"min_bucket_size", "target_bucket_size", "hard_cap", "questions_per_round",
        "answers_per_question", "max_rounds", "plan_timeout", "gen_timeout",
        "answer_timeout", "judge_timeout"}


def _env_default(key):
    val = os.environ.get("INFOGAIN_" + key.upper())
    if val is None:
        return DEFAULTS[key]
    return int(val) if key in _INT else (float(val) if key not in (
        "plan_model", "question_gen_model", "answer_model", "value_judge_model"
    ) else val)


# ── orchestration ─────────────────────────────────────────────────────────────


def run(problem, cfg, progress=None):
    """Run the full bucket-fill loop. Returns a result dict (see keys below)."""
    def log(msg):
        if progress:
            progress(msg)

    plan_model = resolve_alias(cfg["plan_model"])
    qg_model = resolve_alias(cfg["question_gen_model"])
    ans_model = resolve_alias(cfg["answer_model"])
    judge_model = resolve_alias(cfg["value_judge_model"])

    log(f"framing problem + baseline plan via {plan_model} ...")
    framing, ferr = pipeline.frame_and_plan(problem, plan_model, cfg["plan_timeout"])
    baseline_plan = framing.get("baseline_plan", "")

    seen, scored_all = [], []
    rounds_used = 0
    for _ in range(cfg["max_rounds"]):
        rounds_used += 1
        avoid = [r["question"] for r in seen]
        log(f"round {rounds_used}: generating {cfg['questions_per_round']} "
            f"questions via {qg_model} ...")
        new_qs, _ = pipeline.generate_questions(
            problem, framing, qg_model, cfg["questions_per_round"], avoid,
            cfg["gen_timeout"])
        fresh = [q for q in new_qs if not voi.is_duplicate(q, seen)]
        seen.extend(fresh)
        if not fresh:
            log("round produced no new questions; stopping.")
            break

        log(f"round {rounds_used}: projecting answers ({ans_model}) + "
            f"judging plan-change ({judge_model}) for {len(fresh)} questions ...")
        fresh = pipeline.project_answers_batch(
            problem, framing, fresh, ans_model, cfg["answers_per_question"],
            cfg["answer_timeout"])
        fresh = pipeline.judge_plan_change_batch(
            problem, framing, baseline_plan, fresh, judge_model, cfg["judge_timeout"])
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

        if len(bucket) >= cfg["target_bucket_size"]:
            log("target bucket reached; stopping.")
            break
        if len(bucket) >= cfg["min_bucket_size"] and best_fresh < cfg["refill_floor"]:
            log("min bucket reached and fresh candidates below refill floor; stopping.")
            break

    bucket, discarded = voi.rank_and_select(
        scored_all, discard_threshold=cfg["discard_threshold"],
        pre_answer_threshold=cfg["pre_answer_threshold"],
        hard_cap=cfg["hard_cap"], mmr_lambda=cfg["mmr_lambda"])

    return {
        "problem": problem,
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


# ── rendering ─────────────────────────────────────────────────────────────────


def _template():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "templates", "report.md")
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return _FALLBACK_TEMPLATE


_FALLBACK_TEMPLATE = """# Information-Gain Analysis

**Problem:** {{problem}}

**Goal:** {{goal}}
**Decision:** {{decision}}

**Baseline plan (most-likely interpretation):**
{{baseline_plan}}

## Pre-answer these first
{{preanswer_list}}

## Ranked questions by value of information
{{table}}

{{discarded_note}}

---
{{meta}}
"""


def _fmt_default(rec):
    m = rec.get("modal_answer")
    if not m:
        return "—"
    return f"{m.get('answer', '')[:80]} (p≈{voi.clamp01(m.get('prob', 0)):.2f})"


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

    rows = ["| # | value | U | EVSI | rec | question | resolves | assume-if-skipped |",
            "|---|------:|----:|-----:|-----|----------|----------|-------------------|"]
    for i, r in enumerate(bucket):
        rows.append(
            f"| {i + 1} | {r['value']:.2f} | {r['u']:.2f} | {r['evsi']:.2f} | "
            f"{r.get('recommendation', '')} | {r['question']} | "
            f"{r.get('target', '') or '—'} | {_fmt_default(r)} |")
    table = "\n".join(rows) if bucket else "_No valuable questions found._"

    note = (f"_{result['discarded_count']} lower-value/redundant question(s) "
            f"discarded._") if result["discarded_count"] else ""
    if not result["min_met"]:
        note += (f"\n\n> Bucket holds {len(bucket)} question(s), below the minimum of "
                 f"{result['config']['min_bucket_size']}, after "
                 f"{result['rounds_used']} round(s). This usually means the problem "
                 f"is already fairly well-specified.")

    meta = (f"_models: plan={result['config']['question_gen_model']}, "
            f"answers={result['config']['answer_model']}, "
            f"judge={result['config']['value_judge_model']} · "
            f"rounds={result['rounds_used']} · "
            f"candidates={result['candidates_considered']} · "
            f"thresholds: discard={result['config']['discard_threshold']}, "
            f"pre-answer={result['config']['pre_answer_threshold']}_")

    crit = fr.get("success_criteria") or []
    crit_str = "; ".join(crit) if isinstance(crit, list) else str(crit)

    out = _template()
    for k, v in {
        "{{problem}}": result["problem"],
        "{{goal}}": fr.get("goal", "") or "—",
        "{{decision}}": fr.get("decision", "") or "—",
        "{{success_criteria}}": crit_str or "—",
        "{{baseline_plan}}": fr.get("baseline_plan", "") or "—",
        "{{preanswer_list}}": pre,
        "{{table}}": table,
        "{{discarded_note}}": note,
        "{{meta}}": meta,
    }.items():
        out = out.replace(k, str(v))
    return out


def _dry_run(problem, cfg):
    framing_stub = {"goal": "<goal from stage 0>", "decision": "<decision from stage 0>"}
    q_stub = {"question": "<a candidate question>"}
    a_stub = [{"answer": "<answer 1>"}, {"answer": "<answer 2>"}]
    sep = "\n" + "=" * 72 + "\n"
    print(sep.join([
        "DRY RUN — prompts only, no model calls.",
        "STAGE 0 — frame_and_plan:\n\n" + pipeline.frame_prompt(problem),
        "STAGE 1 — generate_questions:\n\n" + pipeline.questions_prompt(
            problem, framing_stub, cfg["questions_per_round"],
            avoid=["<already-considered question>"]),
        "STAGE 2 — project_answers (per question, parallel):\n\n"
        + pipeline.answers_prompt(problem, framing_stub, q_stub["question"],
                                  cfg["answers_per_question"]),
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
    p.add_argument("-o", "--output", help="Write the report to this file.")
    p.add_argument("--quiet", action="store_true", help="Suppress progress logging on stderr.")
    # tunable overrides
    for key in DEFAULTS:
        flag = "--" + key.replace("_", "-")
        if key in _INT:
            p.add_argument(flag, type=int)
        elif key in ("plan_model", "question_gen_model", "answer_model", "value_judge_model"):
            p.add_argument(flag, type=str)
        else:
            p.add_argument(flag, type=float)
    return p


def resolve_config(args):
    cfg = {}
    for key in DEFAULTS:
        cli = getattr(args, key, None)
        cfg[key] = cli if cli is not None else _env_default(key)
    return cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    problem = args.problem or " ".join(args.problem_pos).strip()
    if not problem:
        print("Error: no problem given. Pass it positionally or with --problem.",
              file=sys.stderr)
        return 3

    cfg = resolve_config(args)

    if args.dry_run:
        _dry_run(problem, cfg)
        return 0

    if not pipeline.ollama_reachable():
        print(f"Error: Ollama not reachable at {pipeline.OLLAMA_URL}. "
              "Is the daemon running? (override with OLLAMA_URL)", file=sys.stderr)
        return 2

    progress = None if args.quiet else (lambda m: print(f"… {m}", file=sys.stderr, flush=True))
    result = run(problem, cfg, progress=progress)

    if args.json:
        rendered = json.dumps(result, indent=2, default=str)
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
