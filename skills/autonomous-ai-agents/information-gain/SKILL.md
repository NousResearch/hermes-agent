---
name: information-gain
description: >
  Use when a problem or request is underspecified and you need to decide WHAT to clarify before
  doing the work. Interrogates the prompt into candidate questions, projects plausible answers with
  probabilities, estimates each question's value of information (how much the answer would change the
  recommended plan, weighted by likelihood and stakes), discards low-value/redundant ones, and keeps
  generating until a diverse bucket of high-value questions is filled. Reports a ranked list with
  recommendations (pre-answer / assume-default) using role-specialized local Ollama models. Reports
  only — it does not ask the user or answer the questions itself. Triggers: "what should I clarify",
  "what questions matter here", "is this spec complete", "what am I missing before I start".
version: 0.1.0
author: agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: autonomous-ai-agents
    tags: [information-gain, value-of-information, evsi, clarifying-questions, planning, decision-support, ollama]
    related_skills: [ask, advisors]
    config:
    - key: information-gain.question_gen_model
      description: Model alias that generates and frames candidate questions (strong)
      default: glm
      prompt: Which model should generate questions for information-gain?
    - key: information-gain.answer_model
      description: Fast model alias that projects plausible answers (runs in parallel)
      default: fast
      prompt: Which fast model should project answers?
    - key: information-gain.value_judge_model
      description: Model alias that judges per-answer plan-change and stakes (strong)
      default: deepseek
      prompt: Which model should judge question value?
    - key: information-gain.min_bucket_size
      description: Minimum number of high-value questions the report aims to contain
      default: 3
      prompt: Minimum bucket size for ranked questions?
    - key: information-gain.discard_threshold
      description: Value (0-1) below which a question is dropped as not valuable
      default: 0.30
      prompt: Discard threshold for low-value questions?
    - key: information-gain.max_rounds
      description: Max generation rounds while trying to fill the bucket
      default: 3
      prompt: Max generation rounds?
---

# Information-Gain — what to clarify before you start

## Overview

Given an underspecified problem, this skill estimates the **value of information** of clarifying
questions and reports the ones worth resolving first. It approximates the decision-theoretic
*Expected Value of Sample Information* (EVSI): a question is valuable only if its answer is genuinely
uncertain **and** would change the recommended plan **and** the stakes of guessing wrong are real.
It simulates plausible answers with local Ollama models, scores each question, discards the
low-value and redundant ones, and keeps generating until a diverse bucket is filled.

It is a **reporter / analysis primitive** — it ranks and recommends. Deciding to ask the user, or
researching an answer, is the caller's job (same primitive-vs-orchestrator discipline as `ask`).

## When to Use

**Use it** before committing to an approach on a vague brief, ambiguous ticket, or open-ended ask —
to surface the high-leverage unknowns and a safe default for each.

**Don't use it** for well-specified tasks (it will correctly report "nothing high-value to clarify"
and you've spent model calls to learn that), or when you just want a single model's opinion (use
`ask`), or for a multi-model debate (use `advisors`).

## How to run

The logic is in `scripts/`; run it with the terminal tool. Skill dir: `${HERMES_SKILL_DIR}`.

```bash
# Markdown report (default = focus mode: the prioritized top few)
python3 ${HERMES_SKILL_DIR}/scripts/infogain.py "Build a service to sync USAW events into our calendar"

# Breadth mode: wider coverage by SAMPLING the model's own question distribution
# (several high-temperature draws, unioned + deduped) — no seeded topic list.
# Heavier/slower — pair with a fast judge if needed.
python3 ${HERMES_SKILL_DIR}/scripts/infogain.py "<problem>" --mode breadth

# Structured JSON for programmatic use (read the bucket back into your reasoning)
python3 ${HERMES_SKILL_DIR}/scripts/infogain.py -p "Add SSO to the admin portal" --json

# See the exact stage prompts without any model calls
python3 ${HERMES_SKILL_DIR}/scripts/infogain.py "<problem>" --dry-run

# Show your work: every stage's prompt + raw model output + the per-question
# scoring arithmetic (U, the P·Δplan·stakes terms, √(U·EVSI)) + the loop decisions.
# Invaluable for comparing models or scaffolding a weaker one. Add --json for structured.
python3 ${HERMES_SKILL_DIR}/scripts/infogain.py "<problem>" --trace

# Iterative loop: fold answered evidence back into the same context; resolved questions drop
# out (they become derivable) and the next-best questions surface. You/the agent drive the loop.
python3 ${HERMES_SKILL_DIR}/scripts/infogain.py "<problem>" \
    --evidence "data residency: US-only, no third-party cloud" "budget: free tier only"
# ...or --evidence-file facts.txt (one established fact per line).

# Write to a file; quiet stderr
python3 ${HERMES_SKILL_DIR}/scripts/infogain.py "<problem>" -o /tmp/infogain.md --quiet
```

Use `--problem/-p` (not positional) when the text contains `--` or shell-special characters.

### What it returns

Given a prompt, a list of the **key questions ranked by weight** = exploration value =
`√(uncertainty × value-of-answering)` — *how much answering each would improve your
response to the prompt*. Each ranked question carries a **plain-language clarification of what its
weight means** (how much it would improve the response, and the assumption you'd otherwise make and
its chance of being wrong), plus the recommendation (**PRE_ANSWER** ≥ 0.60 /
**ASSUME_DEFAULT** ≥ 0.30). A detailed numeric table is included below the list. If nothing clears the
bar, it says so — the prompt is already specified well enough for a good response. See
`references/design-decisions.md` for the model.

## How it works (4 stages, looped)

1. **Frame + baseline plan** (`question_gen_model`) — restate goal/decision and the plan you'd give
   *right now* (folding in any `--evidence`); everything is scored as change from this baseline.
2. **Project answers** (`answer_model`, parallel) — plausible answers + probabilities, plus how
   *derivable* each question is (already inferable from the prompt?).
3. **Judge** (`value_judge_model`, parallel) — per-answer plan-change × stakes vs the baseline.
4. **Score / gate / diversify** (pure Python) — `value-of-answering (EVSI) = Σ P·Δplan·stakes`;
   `uncertainty U = entropy(answers)·(1−derivable)`; gate out the no-uncertainty/no-change cases;
   `exploration value = √(U · EVSI)`; collapse same-`target` duplicates; MMR-rank.
   If the bucket is under `min_bucket_size`, generate another round (deduped) up to `max_rounds`.

Full rationale + citations: `references/methodology.md`. Prompt contracts: `references/prompts.md`.

## Families (default ON — `--families`/`--no-families`)

Before generating individual questions, it first generates **families of questions** for *coverage*:
several **scoped** families (distinct regions of the unknowns) plus two lenses you'd otherwise miss —
a **contrarian** family (challenges the baseline approach itself: *"should we even build this?"*) and a
**vantage** family (questions whose answer is *access-relative* — differs by environment / server /
identity / credential / token; auto-enabled only for systems/access tasks). It then generates questions
within each family (tagged with `family`/`lens`).

Families are **domain exposure only** — there is **no family-level negation**. Every question is still
scored on its own merit (the same EVSI pipeline), so a low-average family can still surface the single
highest-value question (this is common for the contrarian/vantage lenses); irrelevant families
self-prune because their questions score low individually. The `family` tag adds a tier to the MMR
diversity kernel (`family > target > question`) so selection **spreads across families**, and the
report is **grouped by family** (contrarian/vantage labelled). Turn off with `--no-families` (or
`INFOGAIN_FAMILIES=off`) for the flat generator. Cost ≈ `questions_per_family × (n_scoped + contrarian +
vantage)` candidates (~12–15 at defaults), with a one-call fallback to the flat generator if family
generation yields nothing.

## Automating the loop — the `investigator` skill

This skill is **report-only** — it ranks what to clarify and stops. To *automate* the evidence loop
(research the top questions, fold facts back in, re-rank, and produce a final response), use the
sibling **`investigator`** skill (`autonomous-ai-agents/investigator/`), which calls this ranker and
owns the answering loop + a capability ladder. Keeping the answering/looping out of here preserves the
primitive-vs-orchestrator boundary. See that skill's `SKILL.md`.

## Tuning

**Mode presets:** `--mode focus` (default — prioritized top few) or `--mode breadth` (wider
coverage). Breadth works by **sampling the model's own distribution over questions** — `gen_samples`
independent draws at `gen_temperature`, unioned and deduped — so the breadth comes from the model's
uncertainty (the tail of its distribution), not a seeded topic list. Scoring stages stay
deterministic (temperature 0): explore stochastically, evaluate stably. When `gen_samples > 1` the
sampled candidates are then **semantically consolidated** — a `consolidate_model` (default `fast`)
clusters questions that resolve the same underlying unknown and keeps one canonical per cluster, so
high-temperature sampling doesn't pad the bucket with reworded duplicates (it never drops a distinct
unknown; on failure it falls back to the lexically-deduped set). Resolution order is
`DEFAULTS ← mode preset ← INFOGAIN_* env ← CLI flag`, so a preset sets the baseline and any explicit
flag/env still wins (e.g. `--gen-samples 5 --gen-temperature 1.0`).

Defaults live as module constants in `scripts/infogain.py`, overridable by `INFOGAIN_*` env vars
or CLI flags (e.g. `--min-bucket-size 5`, `--discard-threshold 0.5`, `--answer-model qwen`). Model
names are `ask`-style aliases (`glm`, `deepseek`, `fast`, `qwen`, …) resolved via `model_utils`.

## Dependency

Reuses the `ask` skill's dispatch helpers (`model_utils.py`). The scripts resolve it at runtime via
`HERMES_HOME` (default `~/.hermes` → `/opt/data` in-container) or an explicit `ASK_SCRIPTS_DIR`. If
the `ask` skill isn't installed, the scripts exit with a clear message.

## Common Pitfalls

- **Ollama must be reachable** at `host.docker.internal:11434` (override `OLLAMA_URL`). The script
  preflights `/api/tags` and exits 2 with a clear message if it's down.
- **`glm` defaults to Chinese** — handled automatically (`build_prompt` appends an English directive
  for `NON_ENGLISH_MODELS`); don't hand-roll one.
- **It reports, it doesn't act.** Don't expect it to ask the user or fill in answers — feed its
  shortlist into your own clarify/research step.
- **A short or empty bucket is a valid result**, not a failure — the problem is well-specified.
- **Cost scales with rounds × questions × 2 calls.** Cloud judge/gen models add latency; switch to
  local aliases (`--question-gen-model qwen --value-judge-model qwen`) for fast/cheap runs.

## Evaluating output quality (adjudicated evals)

`evals/` holds an adjudicator harness that decides whether the skill's output is *acceptable*,
not just well-formed. `evals/run_evals.py` runs the skill on `evals/cases.json` (a mix of
underspecified and well-specified problems) and applies two layers:

- **structural checks** (deterministic) — values in [0,1], ranked order, valid recommendations,
  no surviving duplicate `target`s, and per-case **calibration** (an underspecified problem must
  yield several questions; a well-specified one must yield few/none).
- **an LLM adjudicator** (`evals/adjudicator.py`) — a *different, stronger* model than the
  generation stages (default `deepseek` judging `fast`-generated output, to avoid self-judging
  bias) scores framing accuracy, question relevance, value justification, diversity, and calibration.

```bash
# all cases (exit non-zero if any unacceptable — usable as a CI gate)
python3 ${HERMES_SKILL_DIR}/evals/run_evals.py --gen-model fast --judge-model deepseek
python3 ${HERMES_SKILL_DIR}/evals/run_evals.py --case reverse-string --json
```

Weak runs are expected to fail (e.g. `fast` + 1 round on a hard problem); stronger generation
(`--gen-model glm --max-rounds 2`) should clear the bar. That two-sided behavior is the point.

## Verification Checklist

- [ ] `python3 -m py_compile scripts/*.py evals/*.py` passes.
- [ ] `uv run --with pytest python3 -m pytest tests/ -v -k "not live"` is green (pure logic + adjudicator logic).
- [ ] `--dry-run` prints all four stage prompts (confirms `model_utils` import + builders).
- [ ] With Ollama up, a live run on a vague problem returns a ranked bucket and a "pre-answer" list.
- [ ] A deliberately well-specified problem yields a small/empty bucket with the "well-specified" note.
- [ ] `evals/run_evals.py` rates the well-specified case acceptable (bucket≈0) and flags shallow runs.
- [ ] Frontmatter validates (starts at byte 0 with `---`, has `name` + `description`).
