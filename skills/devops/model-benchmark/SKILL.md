---
name: model-benchmark
description: "Use when benchmarking, comparing, or evaluating Hermes provider/model combinations for reasoning, instruction-following, format compliance, stability, and speed. Runs per-question independent hermes chat calls, grades answers automatically, and produces a comparison matrix. Triggers: benchmark, model comparison, model evaluation, model scoring, compare models, provider comparison."
version: 2.0.0
author: Community
license: MIT
metadata:
 hermes:
 tags: [benchmark, model-evaluation, comparison, testing, provider]
 related_skills: [hermes-provider-config]
---

# Hermes Native Model Benchmark

## Goal

This is the **Hermes-native** model benchmark skill. It replaces the legacy OpenClaw-based `model-benchmark`.

Key differences from the legacy version:

- ✅ Uses the local `hermes chat` CLI to invoke models
- ✅ **Per-question independent calls** (v2.0.0): each candidate × each question = one independent `hermes chat` call, so models cannot cross-pollinate answers
- ✅ No dependency on OpenClaw `sessions_spawn`
- ✅ Outputs land in `~/.hermes/benchmarks/<run_id>/`
- ✅ Supports multi provider/model sequential evaluation, parsing, automatic grading, and Markdown summary (including a model comparison matrix)

## When to use

Use this skill when the user wants to:

- benchmark / score / evaluate models
- compare the capability of multiple models
- evaluate whether a provider/model is suitable for cron, writing, reasoning, code, etc.
- run a uniform question set against models from different providers (Claude, GPT, Gemini, GLM, Qwen, DeepSeek, MiniMax, etc.)

## Quick start

Script path:

```bash
scripts/hermes_model_benchmark.py
```

One-shot benchmark (recommended):

```bash
python3 scripts/hermes_model_benchmark.py benchmark \
 --difficulty expert --count10 --seed601 \
 --candidate m1=provider-a::model-x \
 --candidate m2=provider-b::model-y \
 --candidate m3=provider-c::model-z \
 --timeout60
```

Candidate model format:

```text
NAME=PROVIDER::MODEL
```

Examples:

```text
m1=provider-a::model-x
m2=default::model-y # uses current provider/base_url from config.yaml
m3=custom::model-z # custom provider; the script injects it via an isolated HERMES_HOME
m4=default::model-w # uses the current config provider; --provider is omitted
```

> Note: `provider` must be a provider currently supported by the Hermes CLI, or a custom provider that this script supports internally (currently only `deepseek-v4`). `default` / `config` / `current` are script aliases. Do not casually use `auto` for formal benchmarks: `auto` follows the runtime's primary provider and may trigger fallback if the model name does not match.

## Subcommands

### 🔥 Main command: `benchmark` (v2.0.0, recommended)

Per-question independent evaluation. Each candidate model × each question = one independent `hermes chat` call.

```bash
python3 scripts/hermes_model_benchmark.py benchmark \
 --difficulty expert --count10 --seed601 \
 --candidate m1=provider-a::model-x \
 --candidate m2=provider-b::model-y \
 --timeout60
```

`--difficulty` supports: `easy` / `medium` / `mixed` / `instruction` / `hard` / `expert`.

`--timeout` is the **per-question timeout** (seconds). Total wall time ≈ candidates × questions × per-question time.

Output structure:

```text
~/.hermes/benchmarks/<run_id>/
├── questions.json # question set
├── summary.json # structured scoring report
├── summary.md # Markdown report (with comparison matrix)
├── raw/
│ ├── m1_expert_math_001.txt # one raw file per candidate × question
│ ├── m1_expert_code_001.txt
│ ├── m2_expert_math_001.txt
│ └── ...
└── answers/
 ├── m1.json # aggregated answers
 └── m2.json
```

#### Automated `benchmark` pipeline

```
benchmark = generate_questions() → for each candidate, for each question: run_hermes() → aggregate() → grade_run() → verify_run()
```

The entire process is automatic — no manual steps required.

### Supporting subcommands (used internally by `benchmark`)

The `benchmark` subcommand chains together several lower-level subcommands. You usually do not need to invoke them directly, but they are available for ad-hoc workflows.

#### `generate` — produce a question set

```bash
python3 scripts/hermes_model_benchmark.py generate --difficulty easy --count8
```

Output: `~/.hermes/benchmarks/<run_id>/questions.json`

#### `grade` — score a run

```bash
python3 scripts/hermes_model_benchmark.py grade --run-dir ~/.hermes/benchmarks/<run_id>
```

Output: `summary.json` + `summary.md`

#### `verify` — post-run identity check

```bash
python3 scripts/hermes_model_benchmark.py verify --run-dir ~/.hermes/benchmarks/<run_id>
```

Runs two automatic checks:
1. **MD5 identity check** — compares the MD5 of the first200 chars of each model's `raw/*.txt`
2. **Fallback contamination check** — searches for `Fallback activated`

The `benchmark` subcommand automatically invokes `verify` when it finishes.

## Model comparison matrix

The `summary.md` produced by the `benchmark` subcommand contains a comparison matrix that makes per-question differences between models immediately visible:

```text
## Model Comparison Matrix

|题目 | m1 | m2 | m3 |正确答案 |
|---|:---:|:---:|:---:|:---:|
| expert_math_001 | ✅下降10.9% | ✅下降10.9% | ✅下降10.9% |下降10.9% |
| expert_code_001 | ✅5 | ✅5 | ✅5 |5 |
| expert_reason_001 | ✅ A | ✅ A | ❌ C | A |
```

This is the core value of v2.0.0: **see real per-question model differences at a glance**.

## Multi-round stability testing

When you need to formally compare model capabilities, run at least `3×expert +1×hard` with different seeds:

```bash
SCRIPT=scripts/hermes_model_benchmark.py
C1='--candidate m1=provider-a::model-x'
C2='--candidate m2=provider-b::model-y'
C3='--candidate m3=provider-c::model-z'

python3 "$SCRIPT" benchmark --difficulty expert --count10 --seed601 $C1 $C2 $C3 --timeout60
python3 "$SCRIPT" benchmark --difficulty expert --count10 --seed602 $C1 $C2 $C3 --timeout60
python3 "$SCRIPT" benchmark --difficulty expert --count10 --seed603 $C1 $C2 $C3 --timeout60
python3 "$SCRIPT" benchmark --difficulty hard --count8 --seed701 $C1 $C2 $C3 --timeout60
```

### Cross-provider comparison (strongly recommended)

Each candidate model should route through a **different provider backend**:

```bash
SCRIPT=scripts/hermes_model_benchmark.py
python3 "$SCRIPT" benchmark --difficulty expert --count10 --seed777 \
 --candidate m1=provider-a::model-x \
 --candidate m2=provider-b::model-y \
 --candidate m3=provider-c::model-z \
 --timeout60
```

Pass criteria: **models on different providers have different MD5s and no fallback records appear**. If either condition is violated, that round's results are not trustworthy.

## Methodology limitations & verification

### Model identity verification (mandatory!)

When multiple candidate models use the same provider (especially `default::` / shared gateways), the backend API may **not distinguish model names** — all model names route to the same underlying model and return byte-identical outputs.

The `benchmark` subcommand automatically runs `verify` after completing, which detects MD5 collisions and fallback events.

Manual verification fallback:
```bash
for f in ~/.hermes/benchmarks/<run_id>/raw/*.txt; do
 echo "$(basename $f): $(head -c200 "$f" | md5)"
done
```

If multiple models have the same MD5 (only `session_id` differs), **the backend did not actually switch models and that candidate's results cannot be trusted**.

### Fallback contamination check

How to inspect:
```bash
grep "FALLBACK\|Fallback activated" ~/.hermes/logs/agent.log | grep <session_id>
```

When fallback is detected, the candidate's results are marked as "contaminated" and excluded from accuracy rankings.

### Answer-key error risk and review

**The `expected` values in the `expert` question set can themselves be wrong** (previous evaluations found answer-key errors in3 of10 questions). If every model gives the same answer but the answer key disagrees, manually verify that question's logic.

Known historical answer-key errors (already corrected in the bundled question set):

| Question ID | Old (incorrect) value | Correct value | Reason |
|-------------|----------------------|---------------|--------|
| `expert_math_001` |下降1% |下降10.9% |1×1.1×0.9×0.9 =0.891 |
| `expert_planning_001` |9 |8 | Critical path A→B→D→E =2+3+1+2 =8 |
| `expert_logic_001` | C | B | When B is guilty:3 true /1 false; when C is guilty:1 true /3 false |

> See `references/expert-expected-answer-errors.md` for details.

## Empirical conclusions

1. **The v1.x "one prompt, all questions" design is fundamentally flawed**: bundling every question into one prompt lets models cross-confirm answers, hiding per-question differences. Previous v1.x runs reported identical scores across three different backends (the identical incorrect expected answer masked the spread). Switching to v2.0.0 per-question independent evaluation immediately revealed specific questions where some models failed. **Every benchmark MUST invoke the model per question — never let the model see all questions at once.**

2. **Short-answer automatic grading is fine for fast, reproducible comparison**, but is not suitable for open-ended long-form reasoning, code review, or investment analysis. For those tasks, add a longform question set or introduce an LLM judge / human review.

3. **DeepSeek V4 Flash format stability**: even with the `reasoning_content` return-path fix applied, intermittent JSON format failures still occur. This model is better suited as a low-cost fallback than as a primary evaluation model.

4. **Temporary `HERMES_HOME` without user plugins → false negatives**: `prepare_candidate_env()` must copy the `plugins/` directory into the temporary `HERMES_HOME`, otherwise plugin-dependent provider initialization will crash (see `references/temp-env-plugin-gap.md`).

5. **The `expert` difficulty has demonstrated real discriminative power**: previous evaluations show strong models scoring10/10 on the expert set with notable speed differences, while weaker models miss specific questions.

## Evaluation scope

Default question types covered:

- Logical reasoning
- Arithmetic and ratios
- Instruction following
- Text extraction
- Simple code / data understanding

Default grading is deterministic exact / numeric / multiple-choice — no LLM judge is used.

## Interpreting the output

`summary.md` contains:

- **Model comparison matrix**: pass/fail status for every candidate × every question (new in v2.0.0)
- Each model's total score and accuracy
- Average elapsed time
- JSON format compliance
- Per-question correctness breakdown

## Constraints and notes

1. **Do not use the legacy OpenClaw flow**. If `~/.hermes/skills/openclaw-imports/model-benchmark` reappears, delete or migrate it first.
2. **Sequential execution by default**: avoids provider rate limits and local session contention. The v2.0.0 per-question design already guarantees each candidate runs sequentially.
3. **Use `--ignore-rules`**: the script skips persona/skill injection to reduce context pollution.
4. **Use `--max-turns1`**: prevents the model from calling tools.
5. **For formal model comparison, run at least3 rounds with different seeds**: a single round is only suitable for quick screening.
6. **Foreground command timeout ceiling is600 seconds**: the Hermes terminal foreground `timeout` cannot exceed600. If a multi-round benchmark is expected to exceed10 minutes, use `background=true` + `notify_on_complete=true`.
7. **Answer-key error check**: if every model gives the same answer but the answer key disagrees, manually verify (see "Answer-key error risk and review").
8. **Don't look at average score alone**: also examine question-type failures, speed, and the matrix table. Two models may have identical accuracy but one may be2× faster.
9. **Verify the actual route before trusting results**: after a run, check the `verify` output. If it flags identity or fallback warnings, that round's results are unreliable.
10. **The `benchmark` subcommand's `--timeout` is per-question**, not the total wall time. Total wall time ≈ candidates × questions × per-question time.

## FAQ

### Model does not emit valid JSON

The script tries to extract the first JSON object from the output. If that still fails, the model's `format_ok=false`.

### Provider does not exist

Run:

```bash
hermes chat --help
```

to confirm the provider name.

### Evaluating the current default model

Use `default` as the provider name:

```text
current=default::model-name
```

## Verification command

```bash
python3 scripts/hermes_model_benchmark.py benchmark \
 --difficulty easy --count2 --seed42 \
 --candidate test=provider-a::model-x \
 --timeout30
```

If this produces a `summary.md` containing the comparison matrix, the skill is working correctly.
