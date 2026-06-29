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
      default: 0.40
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
# Markdown report (default)
python3 ${HERMES_SKILL_DIR}/scripts/infogain.py "Build a service to sync USAW events into our calendar"

# Structured JSON for programmatic use (read the bucket back into your reasoning)
python3 ${HERMES_SKILL_DIR}/scripts/infogain.py -p "Add SSO to the admin portal" --json

# See the exact stage prompts without any model calls
python3 ${HERMES_SKILL_DIR}/scripts/infogain.py "<problem>" --dry-run

# Write to a file; quiet stderr
python3 ${HERMES_SKILL_DIR}/scripts/infogain.py "<problem>" -o /tmp/infogain.md --quiet
```

Use `--problem/-p` (not positional) when the text contains `--` or shell-special characters.

### What it returns

A ranked table of questions with **value** = √(U · EVSI), the recommendation
(**PRE_ANSWER** ≥ 0.60 / **ASSUME_DEFAULT** ≥ 0.40), and an `assume-if-skipped` default (the
most-likely projected answer). The "Pre-answer these first" section is the actionable shortlist.
If the bucket can't reach `min_bucket_size`, it says so — that means the problem is already
well-specified.

## How it works (4 stages, looped)

1. **Frame + baseline plan** (`question_gen_model`) — restate goal/decision and the plan you'd give
   *right now*; everything is scored as change from this baseline.
2. **Project answers** (`answer_model`, parallel) — plausible answers + probabilities + how
   derivable each question is from the prompt.
3. **Judge** (`value_judge_model`, parallel) — per-answer plan-change × stakes vs the baseline.
4. **Score / gate / diversify** (pure Python) — `EVSI = Σ P·Δplan·stakes`, gate out the
   no-uncertainty/no-change cases, `value = √(U·EVSI)`, collapse same-`target` duplicates, MMR-rank.
   If the bucket is under `min_bucket_size`, generate another round (deduped) up to `max_rounds`.

Full rationale + citations: `references/methodology.md`. Prompt contracts: `references/prompts.md`.

## Tuning

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

## Verification Checklist

- [ ] `python3 -m py_compile scripts/*.py` passes.
- [ ] `uv run --with pytest python3 -m pytest tests/ -v -k "not live"` is green (pure logic).
- [ ] `--dry-run` prints all four stage prompts (confirms `model_utils` import + builders).
- [ ] With Ollama up, a live run on a vague problem returns a ranked bucket and a "pre-answer" list.
- [ ] A deliberately well-specified problem yields a small/empty bucket with the "well-specified" note.
- [ ] Frontmatter validates (starts at byte 0 with `---`, has `name` + `description`).
