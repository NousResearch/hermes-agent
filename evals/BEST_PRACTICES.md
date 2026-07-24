# Hermes Eval Harness — Best Practices Guide

This guide is the practical companion to [`CONTRACT.md`](./CONTRACT.md). The
CONTRACT defines the machine-readable format (suite YAML, rubric function
signature, runner output JSON); this guide tells you *how to use it well* —
how to design suites that catch real regressions, how to run them locally and
in CI, how to manage baselines without fooling yourself, and how to keep
LLM-as-judge reliable and affordable.

If the CONTRACT and this guide ever disagree, the CONTRACT is the source of
truth for format; this guide is the source of truth for judgment.

---

## Table of Contents

1. [Getting Started in 5 Minutes](#1-getting-started-in-5-minutes)
2. [Adding a New Suite](#2-adding-a-new-suite)
3. [Running Suites Locally](#3-running-suites-locally)
4. [CI Integration — Tier 1 / 2 / 3 Gates](#4-ci-integration--tier-1--2--3-gates)
5. [Baseline Management](#5-baseline-management)
6. [LLM-as-Judge Rubric Design](#6-llm-as-judge-rubric-design)
7. [Windows-Specific Considerations](#7-windows-specific-considerations)
8. [Cost Management](#8-cost-management)
9. [Quick Reference](#9-quick-reference)

---

## 1. Getting Started in 5 Minutes

### Prerequisites

- Python 3.11+
- `pip install pyyaml`
- For live (Tier 2) runs: an API key for at least one provider
  (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY`)

### Run your first deterministic suite

From the repo root:

```bash
# Tier 1 — deterministic, no API key, ~2 seconds
python evals/runners/run_suite.py \
    --suite orchestration \
    --deterministic-only \
    --quiet
```

You should see a summary table with ✅/❌ per scenario and a pass rate. The
report JSON is written to `evals/reports/orchestration.json`.

### Run all Tier 1 suites at once (CI-style)

```bash
python scripts/ci/run_evals.py --tier 1
```

This runs `orchestration`, `cost_cache`, `subagent_verify`, and
`memory_recall` in deterministic mode, evaluates hard gates, compares
against baselines, and writes an aggregate report to
`evals/reports/latest.json`.

### Run a live suite (needs an API key)

```bash
export OPENROUTER_API_KEY=sk-or-...

python evals/runners/run_suite.py \
    --suite code_task \
    --provider openrouter \
    --model anthropic/claude-haiku-4.5 \
    --baseline evals/baselines/code_task_baseline.json
```

### Understand the output

Every run produces a JSON report with this shape (see CONTRACT.md for the
full schema):

```json
{
  "suite": "orchestration",
  "total": 5,
  "passed": 4,
  "failed": 1,
  "pass_rate": 0.80,
  "scenarios": [
    {"id": "O1_parallelizable", "pass": true, "score": 0.95, ...}
  ]
}
```

The runner exits `0` on success, `1` if pass rate < 0.5 or a baseline
regression is detected.

---

## 2. Adding a New Suite

A suite is three artifacts:

| Artifact | Location | Purpose |
|---|---|---|
| Suite YAML | `evals/suites/<suite_name>.yaml` | Scenario definitions + pass conditions |
| Rubric module | `evals/rubrics/<suite_name>.py` | Exports `grade(scenario, result) -> dict` |
| Baseline JSON | `evals/baselines/<suite_name>_baseline.json` | Stored pass-rate snapshot for regression comparison |

### 2.1 Suite YAML Format

The YAML is the single source of truth for what the suite tests. Every field
is documented in `CONTRACT.md`; here is the annotated template:

```yaml
name: my_suite                    # must match the filename (without .yaml)
description: >
  One-paragraph summary of what this suite tests and why. Include the
  pass threshold and whether it's deterministic or live.

scenarios:
  - id: S1_unique_id               # globally unique, descriptive
    description: "Human-readable explanation"
    user_message: "The exact prompt sent to the agent"
    system_message: "Optional system prompt override"
    config_overrides:
      delegation.max_concurrent_children: 8
      agent.max_iterations: 12
    enabled_toolsets: [terminal, file, delegation]
    skip_memory: true              # true = don't load MEMORY.md snapshot
    skip_context_files: true       # true = don't inject skill/context files
    pass_conditions:
      - type: delegate_call_count
        min: 2
      - type: plan_score
        min: 0.8
      - type: no_tool_error
      - type: response_contains
        value: "expected substring"
      - type: custom
        rubric: "rubrics.my_suite.grade"
```

**Naming conventions:**
- Suite name: `snake_case`, matches filename.
- Scenario IDs: `<suite_prefix><number>_<descriptive>` — e.g., `O1_parallelizable`,
  `E2_toolset_swap_forbidden`, `M3_stale_override`, `W1_encoding`. The prefix
  makes regressions immediately identifiable in CI output.
- Keep IDs stable across versions. Renaming a scenario invalidates its
  baseline entry and makes regression tracking impossible.

**Field guidance:**

| Field | When to use | Pitfall |
|---|---|---|
| `system_message` | When the scenario needs specific behavioral instructions the default system prompt doesn't provide | Over-specifying — if you script the exact answer, you're testing prompt compliance, not agent capability |
| `config_overrides` | Set `agent.max_iterations` generously (8–15) so the agent has room to work, but cap it to prevent runaway cost | Setting `max_iterations` too low causes false failures; too high burns API budget |
| `enabled_toolsets` | List exactly the tools the scenario needs — no more, no less | Including `delegation` when the scenario shouldn't delegate will create false positives for the `delegate_call_count` condition |
| `skip_memory` | `true` for almost all suites (prevents cross-test contamination). `false` only for `memory_recall` | Leaving `skip_memory: false` in a deterministic suite loads whatever MEMORY.md exists on the runner machine — non-reproducible |
| `skip_context_files` | `true` for almost all suites. `false` only if the scenario specifically tests skill/context-file loading | Same contamination risk as `skip_memory` |
| `_mock_messages` | Embed synthetic message transcripts for deterministic-mode testing of rubric logic (see `subagent_verify.yaml`) | These are test fixtures, not live expectations — keep them in sync with the rubric's parsing logic |

### 2.2 Rubric Contract

Every rubric module must export a `grade()` function with this exact
signature:

```python
# evals/rubrics/my_suite.py

def grade(scenario: dict, result: dict) -> dict:
    """Return {pass: bool, score: float 0-1, details: dict}"""
    ...
```

**Contract rules:**

1. **Return type is always a dict** with keys `"pass"` (bool), `"score"`
   (float in [0, 1]), and `"details"` (dict). Never raise — catch
   exceptions and return `{"pass": False, "score": 0.0, "details": {"error": str(e)}}`.

2. **The rubric is deterministic.** It must not call any external API, use
   `random` without a fixed seed, or depend on wall-clock time. The same
   `(scenario, result)` input must always produce the same output. This is
   what makes Tier 1 suites safe to run on every PR with no API key.

3. **The rubric inspects `result["messages"]`** (the full message
   transcript) and `result["final_response"]` (the agent's final text).
   It does not re-run the agent or make additional model calls.

4. **Pass conditions from the YAML are advisory.** The rubric can
   interpret `pass_conditions` directly (as `orchestration.py` does via its
   `_CONDITIONS` dispatch table), or it can implement its own logic and
   use the YAML conditions as documentation. Either way, the rubric's
   `"pass"` value is authoritative.

5. **`details` should be human-readable.** CI prints these on failure.
   Include the computed metrics (e.g., `delegate_subtask_count`,
   `verify_rate`, `cache_break_events`) so a developer can see *why* a
   scenario failed, not just *that* it failed.

6. **Include a self-test.** Add an `if __name__ == "__main__"` block with
   synthetic `result` dicts that exercise each pass/fail path. Run it with
   `python evals/rubrics/my_suite.py` — it should print per-scenario
   results and exit non-zero on any failure. See `orchestration.py` and
   `subagent_verify.py` for working examples.

### 2.3 Rubric design patterns

**Pattern 1: Metric computation + condition dispatch** (used by
`orchestration.py`)

Compute a metrics dict once from `result["messages"]`, then evaluate each
YAML `pass_condition` against those metrics via a dispatch table. This
separates *what to measure* from *what to require*, making thresholds
tunable in the YAML without touching the rubric.

```python
_CONDITIONS = {
    "delegate_call_count": _cond_delegate_call_count,
    "plan_score": _cond_plan_score,
    "no_tool_error": _cond_no_tool_error,
    ...
}

def grade(scenario, result):
    metrics = compute_metrics(result)
    conditions = scenario.get("pass_conditions", [])
    # evaluate each condition, aggregate pass/fail
```

**Pattern 2: Per-scenario grader dispatch** (used by `cost_cache.py`,
`memory_recall.py`)

When scenarios differ structurally (not just in thresholds), register a
per-scenario grader function:

```python
_GRADERS = {
    "E1_cache_stable": _grade_e1,
    "E2_toolset_swap_forbidden": _grade_e2,
    ...
}

def grade(scenario, result):
    grader = _GRADERS.get(scenario["id"])
    if grader is None:
        return {"pass": False, "score": 0.0, "details": {"error": "unknown scenario"}}
    return grader(analysis, result)
```

**Pattern 3: Fallback to YAML conditions** (used by the runner itself when
no rubric module exists)

If `evals/rubrics/<suite_name>.py` is missing, the runner evaluates
`pass_conditions` with its built-in evaluators (`delegate_call_count`,
`no_cache_break`, `response_contains`, `no_tool_error`). Unknown
condition types pass by default. **Do not rely on this for production
suites** — write a rubric module. The fallback exists for rapid
prototyping only.

### 2.4 Checklist for a new suite

- [ ] Suite YAML at `evals/suites/<name>.yaml` with `name:` matching filename
- [ ] Rubric at `evals/rubrics/<name>.py` exporting `grade(scenario, result) -> dict`
- [ ] Rubric is deterministic (no API calls, no unseeded randomness)
- [ ] Rubric has a self-test (`if __name__ == "__main__"`) that passes
- [ ] Scenarios have stable, descriptive IDs (prefix + number + descriptor)
- [ ] `skip_memory: true` and `skip_context_files: true` unless the suite
      specifically tests those features
- [ ] `config_overrides.agent.max_iterations` set to a reasonable cap (8–15)
- [ ] If deterministic: `_mock_messages` embedded for rubric self-test, or
      conditions that work with empty/structural-only result dicts
- [ ] If live: suite YAML header comments note the required API key
- [ ] Baseline JSON created at `evals/baselines/<name>_baseline.json` (copy
      the first successful report)
- [ ] Suite added to the appropriate tier list in `scripts/ci/run_evals.py`
      (`_TIER1_SUITES` or `_TIER2_SUITES`)
- [ ] If the suite emits a hard-gate metric, added to `HARD_GATES` in
      `scripts/ci/run_evals.py` and `metric_to_suites` mapping

---

## 3. Running Suites Locally

### 3.1 Deterministic mode (Tier 1)

Deterministic mode skips all live API calls. The runner produces a
result dict with empty `final_response` and `messages` (or the
`_mock_messages` from the YAML), then grades it with the rubric. This
tests the rubric logic and the suite structure — not the model.

```bash
python evals/runners/run_suite.py \
    --suite orchestration \
    --deterministic-only \
    --output evals/reports/orchestration.json \
    --quiet
```

**When to use deterministic mode:**
- On every PR (via CI — see §4)
- When developing a new rubric (iterate on logic without API cost)
- When debugging a rubric self-test failure
- When validating that suite YAML is well-formed

**What deterministic mode does NOT test:**
- Whether the agent actually produces the expected tool-call patterns
- Whether `pass_conditions` thresholds are correctly calibrated for a
  real model
- Whether the system prompt is byte-stable (that requires live API-call
  snapshots)

### 3.2 Live mode (Tier 2)

Live mode instantiates a real `AIAgent` with the specified provider and
model, sends the `user_message`, and captures the full transcript. This
is the only mode that exercises real agent behavior.

```bash
export OPENROUTER_API_KEY=sk-or-...

python evals/runners/run_suite.py \
    --suite code_task \
    --provider openrouter \
    --model anthropic/claude-haiku-4.5 \
    --output evals/reports/code_task.json \
    --baseline evals/baselines/code_task_baseline.json
```

**When to use live mode:**
- Before merging a change that could affect agent behavior
- When recalibrating pass-condition thresholds
- Nightly / release validation (Tier 3)
- When a deterministic suite starts failing and you need to confirm the
  rubric isn't the problem

**Cost awareness:** Each live scenario makes 1–15 API calls depending on
`max_iterations`. A 5-scenario suite with `max_iterations: 12` can make
up to 60 API calls. At ~$0.25/M input tokens with a haiku-class model,
expect $0.01–0.05 per scenario. See §8 for cost management strategies.

### 3.3 Comparing against a baseline

```bash
python evals/runners/run_suite.py \
    --suite orchestration \
    --deterministic-only \
    --baseline evals/baselines/orchestration_baseline.json
```

The runner prints a comparison line:

```
Baseline comparison: stable  (Δ=+0.00%)
```

If `delta < -0.05` (a 5-point regression), it prints:

```
Baseline comparison: regression  (Δ=-10.00%)
Regressions: O1_parallelizable, O3_no_spawn_trivial
```

and exits `1`. See §5 for baseline management philosophy.

### 3.4 Running the rubric self-test

Every rubric should be runnable standalone for a quick sanity check:

```bash
python evals/rubrics/orchestration.py
python evals/rubrics/subagent_verify.py
```

This runs the rubric against synthetic `result` dicts embedded in the
file's `__main__` block. It exercises every pass/fail path without any
API calls or YAML loading.

### 3.5 Common local workflows

| Task | Command |
|---|---|
| Quick rubric logic check | `python evals/rubrics/<suite>.py` |
| Full deterministic suite | `python evals/runners/run_suite.py --suite <name> --deterministic-only` |
| Full live suite | `python evals/runners/run_suite.py --suite <name> --provider openrouter --model <model>` |
| All Tier 1 suites (CI-style) | `python scripts/ci/run_evals.py --tier 1` |
| Single suite via CI runner | `python scripts/ci/run_evals.py --tier 1 --suite orchestration` |
| Compare to baseline | add `--baseline evals/baselines/<name>_baseline.json` |

---

## 4. CI Integration — Tier 1 / 2 / 3 Gates

The CI eval pipeline is defined in `.github/workflows/evals.yml` and
orchestrated by `scripts/ci/run_evals.py`. It uses a three-tier system
that separates fast, free, deterministic checks from slow, expensive,
live-model checks.

### 4.1 Tier architecture

| Tier | When | Suites | API key? | Runner | Blocks merge? |
|---|---|---|---|---|---|
| **Tier 1** | Every PR + push to main | `orchestration`, `cost_cache`, `subagent_verify`, `memory_recall` | No | `ubuntu-latest` | **Yes** (hard gates) |
| **Windows** | Every PR + push to main | `windows_reliability` | No | `windows-latest` | Yes |
| **Tier 2** | Push to main/master only | `code_task`, `research_citation` | Yes (`OPENROUTER_API_KEY`) | `ubuntu-latest` | No (`continue-on-error: true`) |
| **Tier 3** | Nightly cron (4 AM UTC) | All suites (Tier 1 + Tier 2) | Yes | `ubuntu-latest` | No (`continue-on-error: true`) |

### 4.2 Hard gates vs soft gates

**Hard gates** are metric thresholds that, if failed, cause the CI run to
exit non-zero. Branch protection rules can block the PR from merging.

| Metric | Operator | Threshold | Source suite | Meaning |
|---|---|---|---|---|
| `cache_break_events` | `==` | 0 | `cost_cache` | Prompt-cache prefix must never break (except compression) |
| `verify_rate` | `>=` | 0.90 | `subagent_verify` | Parent agent must verify ≥90% of delegate_task calls |
| `recall_at_3` | `>=` | 0.85 | `memory_recall` | Memory recall accuracy in top-3 results |
| `plan_score` | `>=` | 0.80 | `orchestration` | Decomposition quality heuristic |

Hard-gate evaluation logic:
- The CI runner extracts each metric from suite reports (suite-level field
  or worst-case across scenario `details`).
- For `==` gates (e.g., `cache_break_events`), the **max** value across
  scenarios is used (worst case).
- For `>=` gates (e.g., `verify_rate`), the **min** value across scenarios
  is used (worst case).
- If a suite ran but didn't emit the expected metric, the gate fails with
  status `"missing"` — this is a hard failure, not a skip.
- If the suite wasn't run at all (e.g., Tier 1 only), the gate is
  `"not_evaluated"` and doesn't fail.

**Soft gates** are pass-rate regressions against stored baselines. They
produce warnings but never fail the build:
- If `delta < -0.05` (5 percentage points), status is `"regression"`.
- If `delta > +0.05`, status is `"improvement"`.
- Otherwise, status is `"stable"`.
- Missing baseline → status `"no_baseline"` (informational).

### 4.3 CI workflow anatomy

The `evals.yml` workflow has four jobs:

```
tier1-deterministic    → runs on every PR/push, ubuntu-latest
windows-reliability    → runs on every PR/push, windows-latest
tier2-live             → runs on push to main/master only
tier3-nightly          → runs on schedule (cron)
```

**Triggers:**
```yaml
on:
  pull_request:
    paths:
      - 'agent/**'
      - 'tools/**'
      - 'hermes_cli/**'
      - 'run_agent.py'
      - 'toolsets.py'
      - 'evals/**'
      - 'scripts/ci/**'
  push:
    branches: [main, master]
  schedule:
    - cron: '0 4 * * *'
```

The `paths` filter ensures Tier 1 only runs when files that could affect
agent behavior or the eval harness itself are changed. Docs-only PRs
skip evals entirely.

**Tier 2 uses `continue-on-error: true`** so live-model failures don't
block the merge — they're informational and tracked via the aggregate
report artifact.

### 4.4 Adding a suite to CI

1. **Determine the tier.** If the suite is deterministic (no API key
   needed), it belongs in Tier 1. If it needs a live model, it belongs in
   Tier 2. The `windows_reliability` suite is special — it runs on a
   Windows runner and is triggered alongside Tier 1.

2. **Add to the tier list** in `scripts/ci/run_evals.py`:
   ```python
   _TIER1_SUITES = [
       "orchestration",
       "cost_cache",
       "subagent_verify",
       "memory_recall",
       "my_new_suite",       # ← add here
   ]
   ```

3. **Add a workflow step** in `.github/workflows/evals.yml`:
   ```yaml
   - name: Run my_new_suite suite (deterministic)
     run: |
       python evals/runners/run_suite.py \
         --suite my_new_suite \
         --deterministic-only \
         --output evals/reports/my_new_suite.json \
         --quiet
   ```

4. **If the suite emits a hard-gate metric**, register it:
   ```python
   HARD_GATES["my_metric"] = (">=", 0.75, "description of the metric")
   ```
   And add it to the `metric_to_suites` mapping:
   ```python
   metric_to_suites["my_metric"] = ["my_new_suite"]
   ```

5. **Create the baseline** by running the suite once and copying the
   report to `evals/baselines/my_new_suite_baseline.json`.

### 4.5 Exit codes

| Code | Meaning |
|---|---|
| `0` | All suites ran, all hard gates passed (soft-gate warnings are OK) |
| `1` | A hard gate failed, a suite errored, or a required secret was missing |
| `2` | Invalid arguments (argparse error) |

The per-suite runner (`run_suite.py`) exits `1` if `pass_rate < 0.5` or if
a baseline regression is detected. The CI runner (`run_evals.py`) exits
`1` if any hard gate fails or any suite produced an error without
scenarios.

---

## 5. Baseline Management

### 5.1 What a baseline is

A baseline is a frozen snapshot of a suite's report JSON, stored at
`evals/baselines/<suite_name>_baseline.json`. It records the `pass_rate`
and per-scenario `pass`/`score` values from a known-good run. The CI
runner compares every subsequent run against it.

The comparison logic (in `run_suite.py:compare_baseline`):

```python
delta = new_pass_rate - old_pass_rate
status = "regression" if delta < -0.05 else ("improvement" if delta > 0.05 else "stable")
```

Per-scenario regressions are also tracked: any scenario that was `pass:
true` in the baseline but `pass: false` in the current run is listed in
the `regressions` array.

### 5.2 When to rebaseline

**Rebaseline when:**

1. **You intentionally changed agent behavior** that legitimately affects
   pass rates — e.g., you tightened delegation rules, changed the system
   prompt, or improved the memory tool. The old baseline no longer
   represents the expected state.

2. **You added or removed scenarios** from a suite. Scenario IDs that
   don't exist in the baseline are ignored, but if you rename a
   scenario, its old baseline entry becomes orphaned and the new one has
   no comparison.

3. **You changed a rubric's scoring logic** in a way that changes what
   "pass" means. The baseline scores were computed with the old logic.

4. **The model was upgraded** and the new model performs differently.
   This is the most common reason — a new model version may be better or
   worse on certain scenarios.

**Do NOT rebaseline when:**

1. **A regression is a bug, not a feature.** If `cache_break_events`
   went from 0 to 2, that's a real regression — rebaselining hides it.
   Fix the bug, then the baseline will pass again.

2. **You're trying to make CI green.** Rebaselining to make a failing
   suite pass is the eval equivalent of deleting a failing test. It
   makes the number look good and removes the signal.

3. **The regression is in a hard gate.** Hard gates exist outside the
   baseline system — they're absolute thresholds, not relative
   comparisons. Rebaselining doesn't affect them.

### 5.3 How to rebaseline

```bash
# 1. Run the suite with the new code/model
python evals/runners/run_suite.py \
    --suite my_suite \
    --deterministic-only \
    --output evals/reports/my_suite.json

# 2. Verify the results are the new expected state
cat evals/reports/my_suite.json | python -m json.tool | head -20

# 3. Copy the report to the baseline location
cp evals/reports/my_suite.json evals/baselines/my_suite_baseline.json

# 4. Commit with a descriptive message
git add evals/baselines/my_suite_baseline.json
git commit -m "eval: rebaseline my_suite after <reason>"
```

For live suites, run the suite in live mode first:
```bash
python evals/runners/run_suite.py \
    --suite code_task \
    --provider openrouter \
    --model anthropic/claude-haiku-4.5 \
    --output evals/reports/code_task.json
cp evals/reports/code_task.json evals/baselines/code_task_baseline.json
```

### 5.4 Interpreting regressions

When the CI runner reports a regression, investigate in this order:

1. **Check the `regressions` array** — which scenario IDs flipped from
   pass to fail? These are the specific regressions, not the overall
   pass-rate drop.

2. **Read the `details` for each regressed scenario** — the rubric's
   `details` dict contains the computed metrics. For example:
   ```json
   "details": {
     "cache_break_events": 2,
     "system_prompt_changes": 1,
     "toolset_mutations": 1
   }
   ```
   This tells you *what* broke, not just *that* it broke.

3. **Determine if it's a real regression or a rubric issue:**
   - If the agent's behavior changed (e.g., it now mutates the system
     prompt), it's a real regression — investigate the code change.
   - If the agent's behavior is the same but the rubric now parses the
     transcript differently (e.g., a message format change), it's a
     rubric bug — fix the rubric, don't rebaseline.

4. **Check for flakiness** — live suites can be non-deterministic due to
   model sampling. Re-run the suite 2–3 times. If it passes sometimes
   and fails sometimes, the threshold may be too tight or the scenario
   may be inherently noisy. Consider:
   - Raising `max_iterations` to give the agent more room
   - Using a lower temperature (see §6)
   - Making the scenario more deterministic (more specific `user_message`)

5. **Check the delta magnitude:**
   - `Δ = -5%` (one scenario in a 20-scenario suite) → likely a real
     issue, investigate the specific scenario.
   - `Δ = -20%` with multiple regressions → systemic issue, check for a
     shared root cause (config change, tool schema change, system prompt
     change).
   - `Δ = +5%` → improvement, not a concern. Don't rebaseline upward
     unless you want to lock in the improvement as the new floor.

### 5.5 Baseline lifecycle

```
Create baseline  →  Run suite  →  Compare  →  (pass | regression | improvement)
      ↑                                                      |          |
      |______________________________________________  rebaseline  _________|
                     (only when intentional change)
```

Baselines should be reviewed at least once per release cycle. Stale
baselines (older than ~3 months or 2 model versions) should be
regenerated to ensure they reflect current expected behavior.

---

## 6. LLM-as-Judge Rubric Design

Several Hermes eval suites use the agent's own output as the grading
signal — the rubric inspects `result["final_response"]` and
`result["messages"]` to determine pass/fail. This is "LLM-as-judge" in
the broad sense: the model is being judged by deterministic rules applied
to its output, and in some cases (future suites) a second model may grade
the first model's response. This section covers best practices for both
patterns.

### 6.1 Temperature 0 for grading

**Always use temperature 0 (or the provider's equivalent deterministic
mode) for any LLM-as-judge call.** The grading model must produce the
same judgment for the same input every time. Temperature > 0 introduces
sampling noise that makes regression detection impossible.

For the existing Hermes evals, the rubrics are pure Python (no model
calls), so this is already satisfied. If you add a rubric that calls a
second model to grade the first model's response:

```python
# Bad — temperature defaults to 1.0, non-deterministic
judge_result = judge_agent.run_conversation(user_message=grading_prompt)

# Good — pin temperature to 0
judge_agent = AIAgent(
    provider="openrouter",
    model="anthropic/claude-haiku-4.5",
    temperature=0,          # deterministic grading
    quiet_mode=True,
    ...
)
```

If the provider doesn't support temperature 0, use the lowest available
value and run the grading 3 times, taking the majority vote.

### 6.2 Cross-validation for noisy metrics

Some scenarios are inherently noisy — the model may phrase the correct
answer differently across runs, or the `response_contains` check may
fail on a valid paraphrase. For these:

1. **Run the scenario N times** (3–5 is typical) and take the median
   score. This smooths out single-run noise.

2. **Use fuzzy matching** instead of exact substring checks. The
   `memory_recall.py` rubric demonstrates this with `_fuzzy_contains()`:
   ```python
   def _fuzzy_contains(haystack: str, needle: str) -> bool:
       """Case-insensitive, ignores extra whitespace, word-level proximity."""
       h = re.sub(r"\s+", " ", haystack.lower()).strip()
       n = re.sub(r"\s+", " ", needle.lower()).strip()
       if n in h:
           return True
       # Word-level proximity: all words appear within a window
       ...
   ```

3. **Use `match_mode: fuzzy`** in the `expected_recall` spec to enable
   fuzzy matching for specific facts:
   ```yaml
   expected_recall:
     - fact: "no prior memory"
       must_appear: true
       match_mode: fuzzy    # allows "don't have any memory"
   ```

4. **Use `match_mode: negative_claim`** for facts that must NOT appear as
   positive assertions (e.g., M4's hallucination check):
   ```yaml
   expected_recall:
     - fact: "Phantom"
       must_appear: false
       match_mode: negative_claim  # OK if negated, bad if asserted
   ```

### 6.3 Held-out sets

The `evals/datasets/golden_tasks.jsonl` file contains curated golden
tasks with expected tools and pass conditions. Use these as a held-out
set — tasks the agent has never seen during development — to detect
overfitting.

**Best practices for held-out sets:**

1. **Never include held-out tasks in the suite YAML.** The suite YAML
   is version-controlled and visible to developers. The held-out set
   should be evaluated separately, on a schedule (Tier 3 nightly), and
   the results should be reviewed by a human.

2. **Stratify by category and difficulty.** The golden tasks have
   `category` (code, orchestration, memory, delegation, research) and
   `difficulty` (easy, medium, hard). Sample proportionally to ensure
   coverage across all capabilities.

3. **Rotate the held-out set periodically.** Every few releases, retire
   some held-out tasks (they're no longer "held out" if developers have
   seen the results) and add new ones. Keep a rotation of ~30% of the
   set fresh per release cycle.

4. **Report held-out results separately from the main baseline.** A
   regression on held-out tasks is a stronger signal than a regression
   on in-suite tasks, because it rules out overfitting to the suite's
   specific scenarios.

### 6.4 Rubric calibration

When designing a new rubric:

1. **Start with deterministic checks.** Inspect `result["messages"]`
   for tool-call patterns (does the agent call `delegate_task`? does it
   call `read_file` after delegating? are there tool errors?). These
   are 100% reproducible and don't depend on model phrasing.

2. **Add response-content checks only when necessary.** `response_contains`
   is fragile — the model may use a synonym. Prefer structural checks
   (tool-call presence, message count, cache-break events) over content
   checks. When you must check content, use fuzzy matching.

3. **Set thresholds conservatively.** Start with `min: 0.85` for
   recall-style metrics and `min: 0.9` for verify-rate. Tighten only if
   you observe false positives (the suite passes when it shouldn't).
   Loosen only if you observe false negatives (the suite fails on correct
   behavior) AND you've confirmed the rubric logic is correct.

4. **Validate with the self-test.** Every rubric should have synthetic
   test cases that exercise both pass and fail paths. If your self-test
   only tests passing scenarios, you don't know if the rubric can detect
   failures.

5. **Avoid change-detector rubrics.** A rubric that freezes a specific
   model output ("the response must be exactly 42") is a change detector,
   not a behavior test. Test the *invariant* (the response must contain
   the correct answer) not the *snapshot* (the response must be a
   specific string). See AGENTS.md: "Behavior contracts over snapshots."

### 6.5 LLM-as-judge anti-patterns

| Anti-pattern | Why it's wrong | Do instead |
|---|---|---|
| Grading with temperature > 0 | Non-deterministic judgments make regression tracking impossible | Pin temperature to 0 |
| Single-run grading for noisy tasks | One bad sample looks like a regression | Run 3–5 times, take median |
| Exact-string matching on free-form responses | Model paraphrases validly | Use `match_mode: fuzzy` or regex patterns |
| Rubric that calls the agent under test | Circular grading — the model grades itself | Use a separate judge model or pure-Python structural checks |
| Threshold set to exactly the observed score | Any noise causes flaky failures | Leave a margin (e.g., if observed is 0.92, set threshold to 0.85) |
| No held-out set | Suite may overfit to known scenarios | Maintain `golden_tasks.jsonl` as a held-out set, evaluate nightly |

---

## 7. Windows-Specific Considerations

Hermes runs on Windows 10/11 with MSYS2 git-bash as the shell. The
`windows_reliability` suite tests the agent on a Windows runner and
catches platform-specific failures that Linux CI cannot detect.

### 7.1 The Windows reliability suite

Located at `evals/suites/windows_reliability.yaml` with rubric
`evals/rubrics/windows_reliability.py`. It tests four scenarios:

| ID | What it tests | Why it matters on Windows |
|---|---|---|
| `W1_encoding` | Non-ASCII paths and content (Arabic, CJK, emoji) | Windows console and file system encoding defaults (cp1252, not UTF-8) can corrupt non-ASCII text |
| `W2_longpath` | Paths > 260 characters | Windows MAX_PATH limit (260 chars) without long-path support enabled |
| `W3_home_spaces` | HERMES_HOME with spaces in path | Common on Windows (`C:\Users\First Last\`) — shell quoting bugs surface here |
| `W4_unicode_arg` | Emoji and RTL text in user goal | MSYS2 bash may mangle Unicode arguments passed to subprocesses |

### 7.2 Windows CI job

The `windows-reliability` job in `evals.yml` runs on `windows-latest`:

```yaml
windows-reliability:
  runs-on: windows-latest
  if: github.event_name == 'pull_request' || github.event_name == 'push'
  steps:
    - uses: actions/checkout@v4
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    - name: Install deps
      run: pip install pyyaml
    - name: Run Windows reliability suite
      run: |
        python evals/runners/run_suite.py `
          --suite windows_reliability `
          --deterministic-only `
          --output evals/reports/windows_reliability.json `
          --quiet
      shell: pwsh
```

Note the PowerShell-style line continuation (backtick) and `shell: pwsh`.
This is the only job that uses PowerShell — all others use bash.

### 7.3 Mojibake detection

The `windows_reliability.py` rubric includes a `_detect_mojibake()`
function that checks for common Windows encoding corruption patterns:

```python
def _detect_mojibake(text: str) -> bool:
    if "\ufffd" in text:        # Unicode replacement character
        return True
    mojibake_markers = [
        "Ã©", "Ã¨", "Ã¼",       # Latin-1 misinterpreted as UTF-8
        "â\u0080\u0099",          # Smart quote corruption
    ]
    ...
```

If mojibake is detected, the scenario fails immediately with `score: 0.0`
— this is a hard failure, not a partial-credit situation. Encoding bugs
on Windows are correctness issues, not quality issues.

### 7.4 Best practices for Windows-compatible suites

1. **Always use `--deterministic-only` for Windows CI.** Live-mode
   suites on Windows require the API key to be available on the Windows
   runner, which adds secret-management complexity. Keep Windows CI
   deterministic and run live Windows tests manually or nightly.

2. **Use UTF-8 everywhere.** The agent's terminal tool on Windows uses
   MSYS2 bash, which supports UTF-8. But the underlying Windows console
   may default to cp1252. Ensure all file I/O in rubrics uses
   `encoding="utf-8"`.

3. **Test paths with spaces.** `C:\Users\First Last\` is the default
   home directory format on Windows. Any path handling that doesn't
   quote properly will break here. The `W3_home_spaces` scenario
   catches this.

4. **Test long paths.** Windows has a 260-character path limit unless
   long-path support is enabled (`LongPathsEnabled` registry key). The
   `W2_longpath` scenario creates a 10-level nested directory to test
   this. If your suite creates files, keep paths under 260 characters or
   test explicitly for long-path support.

5. **Test Unicode in arguments.** Windows subprocess argument passing
   can mangle Unicode. The `W4_unicode_arg` scenario sends an Arabic +
   emoji user message. If your suite passes non-ASCII arguments to
   terminal commands, test this explicitly.

6. **Don't use Windows-specific paths in non-Windows suites.** The
   `orchestration` and `code_task` suites use `/tmp/` paths — these work
   on Linux CI but not on Windows. If a suite needs to run on both,
   use a relative path or a temp directory that works cross-platform.

7. **The `windows_reliability` suite does NOT need a baseline.** In
   deterministic mode, the scenarios pass by default (no live model
   calls). The baseline exists at
   `evals/baselines/windows_reliability_baseline.json` for consistency,
   but the real value of this suite is in live mode on a Windows
   developer machine, not in CI.

---

## 8. Cost Management

Live eval suites (Tier 2/3) make real API calls that cost money. This
section covers strategies to keep eval costs predictable and sustainable.

### 8.1 Understanding eval cost

Each live scenario makes between 1 and `max_iterations` API calls. The
cost per call depends on:
- **Model** — haiku-class models (~$0.25/M input, ~$1.25/M output) are
  ~10× cheaper than sonnet-class models (~$3/M input, ~$15/M output).
- **Context length** — grows with each turn as the conversation
  accumulates tool results. A 12-iteration scenario sends progressively
  larger prompts.
- **Tool results** — terminal and file tool outputs can be large
  (thousands of tokens). The `code_task` scenarios that create files and
  run tests produce significant tool-result content.

**Rough cost estimates** (haiku-class model, 2026 pricing):

| Suite | Scenarios | Avg calls/scenario | Est. cost per run |
|---|---|---|---|
| `code_task` | 4 | 8–12 | $0.05–$0.15 |
| `research_citation` | 4 | 6–10 | $0.03–$0.10 |
| Full Tier 2 | 8 | 7–11 | $0.08–$0.25 |
| Full Tier 3 (nightly) | 12+ | 5–11 | $0.15–$0.50 |

### 8.2 Sampling strategies

1. **Run Tier 2 only on merge to main, not on every PR.** The
   `evals.yml` workflow gates Tier 2 with
   `if: github.event_name == 'push' && (github.ref == 'refs/heads/main' ...)`.
   This limits live-model cost to once per merge, not once per PR commit.

2. **Use `continue-on-error: true` for Tier 2/3.** A live suite failure
   should not block the merge — it's informational. Blocking on live
   suites would pressure developers to rebaseline away real regressions.

3. **Run Tier 3 nightly, not hourly.** The cron schedule
   (`'0 4 * * *'`) runs once per day. This is sufficient for detecting
   drift without burning budget.

4. **Sample scenarios, don't run all.** For suites with many scenarios,
   consider running a random sample (e.g., 3 of 10) per night, rotating
   which 3 are run. This covers the full suite over ~3 nights while
   keeping nightly cost to 30% of a full run.

5. **Use the cheapest model that passes.** Tier 2 uses
   `anthropic/claude-haiku-4.5` by default. If the suite passes reliably
   with a cheaper model, don't upgrade. Only use sonnet-class models for
   suites where the haiku model can't complete the task (e.g., complex
   multi-step reasoning).

### 8.3 Model selection

| Use case | Recommended model | Why |
|---|---|---|
| Tier 1 (deterministic) | N/A — no model calls | Deterministic mode doesn't use a model |
| Tier 2 code_task | `anthropic/claude-haiku-4.5` | Sufficient for coding tasks; 10× cheaper than sonnet |
| Tier 2 research_citation | `anthropic/claude-haiku-4.5` | Web search + summarization is well within haiku's range |
| Nightly Tier 3 | `anthropic/claude-haiku-4.5` | Consistent with Tier 2; comparable results across runs |
| Release validation | `anthropic/claude-sonnet-4.5` (optional) | Higher-capability model for a one-time release gate |
| LLM-as-judge grading | `anthropic/claude-haiku-4.5` at temp 0 | Deterministic, cheap, sufficient for rubric-style grading |

**Switching models:**
```bash
# Override the default model for a single run
python evals/runners/run_suite.py \
    --suite code_task \
    --provider openrouter \
    --model anthropic/claude-sonnet-4.5 \
    --output evals/reports/code_task_sonnet.json
```

**When you switch models, rebaseline.** A different model produces
different pass rates. The baseline for `claude-haiku-4.5` is not valid for
`claude-sonnet-4.5`. Store per-model baselines if you run multiple models:

```
evals/baselines/code_task_baseline.json           # haiku (default)
evals/baselines/code_task_baseline_sonnet.json     # sonnet (release gate)
```

### 8.4 Token budget management

1. **Set `agent.max_iterations` conservatively.** Each iteration is one
   API call. For most suites, 8–12 is sufficient. Raise to 15 only for
   complex multi-step tasks (`code_task` refactoring scenarios).

2. **Use `skip_memory: true` and `skip_context_files: true`.** Loading
   memory snapshots and context files adds tokens to the system prompt,
   increasing cost per call. Unless the suite specifically tests those
   features, skip them.

3. **Monitor `api_calls` in reports.** Every scenario report includes an
   `api_calls` field. If a scenario consistently uses close to
   `max_iterations`, it may be stuck in a loop — investigate and either
   fix the agent behavior or raise the cap with awareness of the cost.

4. **Set per-suite timeouts.** The CI runner has a 1800-second (30-min)
   hard cap per suite. If a suite is consistently approaching this limit,
   it may be spending too long on a scenario. Consider splitting the
   suite or reducing `max_iterations`.

### 8.5 CI budget controls

The `evals.yml` workflow includes several implicit budget controls:

- **Path filtering:** Tier 1 only runs when `agent/**`, `tools/**`,
  `run_agent.py`, `toolsets.py`, or `evals/**` files change. Docs-only
  PRs skip evals entirely.
- **Tier 2 gating:** Only runs on push to main/master, not on PRs.
- **`continue-on-error`** on Tier 2/3: prevents retry loops from
  burning budget on flaky suites.
- **Per-suite timeout:** 30 minutes per suite, enforced by
  `subprocess.run(timeout=1800)` in the CI runner.

If you need tighter budget control, consider:
- Adding a monthly API spending cap at the provider level
- Gating Tier 2 behind a label (e.g., `run-evals`) on PRs
- Running Tier 3 weekly instead of nightly during low-activity periods

---

## 9. Quick Reference

### File layout

```
evals/
├── CONTRACT.md                    # Machine-readable format spec (read first)
├── BEST_PRACTICES.md              # This guide
├── suites/                        # Suite YAML definitions
│   ├── orchestration.yaml
│   ├── cost_cache.yaml
│   ├── subagent_verify.yaml
│   ├── memory_recall.yaml
│   ├── windows_reliability.yaml
│   ├── research_citation.yaml
│   └── code_task.yaml
├── rubrics/                       # Rubric modules (one per suite)
│   ├── orchestration.py
│   ├── cost_cache.py
│   ├── subagent_verify.py
│   ├── memory_recall.py
│   ├── windows_reliability.py
│   ├── research_citation.py
│   └── code_task.py
├── baselines/                     # Stored pass-rate snapshots
│   ├── orchestration_baseline.json
│   ├── cost_cache_baseline.json
│   ├── subagent_verify_baseline.json
│   ├── memory_recall_baseline.json
│   └── windows_reliability_baseline.json
├── reports/                       # Generated report JSONs (gitignored)
│   ├── latest.json
│   ├── orchestration.json
│   └── ...
├── datasets/
│   └── golden_tasks.jsonl         # Held-out golden tasks
└── runners/
    └── run_suite.py               # Suite runner CLI
```

### Command cheat sheet

```bash
# --- Tier 1 (deterministic, no API key) ---
# Run a single suite
python evals/runners/run_suite.py --suite orchestration --deterministic-only --quiet

# Run all Tier 1 suites (CI-style)
python scripts/ci/run_evals.py --tier 1

# Run a single suite via CI runner
python scripts/ci/run_evals.py --tier 1 --suite cost_cache

# --- Tier 2 (live, needs API key) ---
export OPENROUTER_API_KEY=sk-or-...
python evals/runners/run_suite.py --suite code_task --provider openrouter --model anthropic/claude-haiku-4.5
python scripts/ci/run_evals.py --tier 2

# --- Tier 3 (nightly comprehensive) ---
python scripts/ci/run_evals.py --tier 3

# --- Rubric self-test ---
python evals/rubrics/orchestration.py
python evals/rubrics/subagent_verify.py

# --- Baseline comparison ---
python evals/runners/run_suite.py --suite orchestration --deterministic-only \
    --baseline evals/baselines/orchestration_baseline.json

# --- Rebaseline ---
cp evals/reports/my_suite.json evals/baselines/my_suite_baseline.json
```

### Hard gate reference

| Metric | Op | Threshold | Suite | Meaning |
|---|---|---|---|---|
| `cache_break_events` | `==` | 0 | `cost_cache` | No prompt-cache breaks (except compression) |
| `verify_rate` | `>=` | 0.90 | `subagent_verify` | Parent verifies ≥90% of delegations |
| `recall_at_3` | `>=` | 0.85 | `memory_recall` | Memory recall accuracy |
| `plan_score` | `>=` | 0.80 | `orchestration` | Decomposition quality |

### Pass condition types

| Type | Fields | Evaluated by | Description |
|---|---|---|---|
| `delegate_call_count` | `min`, `max` | `orchestration.py` | Count of subagent children spawned |
| `plan_score` | `min` | `orchestration.py` | Heuristic decomposition quality (0–1) |
| `no_tool_error` | — | all rubrics | No tool result contains error indicators |
| `response_contains` | `value` | all rubrics | Final response contains substring (case-insensitive) |
| `no_cache_break` | — | `cost_cache.py` | `cache_break_events == 0` |
| `verify_rate` | `min` | `subagent_verify.py` | Fraction of delegations followed by verification |
| `recall_at_3` | `min` | `memory_recall.py` | Fraction of expected facts correctly recalled |
| `no_parallel_batch` | — | `orchestration.py` | No `delegate_task` call with >1 task |
| `concurrency_respected` | — | `orchestration.py` | All batches within `max_concurrent_children` cap |
| `no_cascade_delegation` | — | `orchestration.py` | No orchestrator-role delegation that exceeds depth |
| `depth_limit_respected` | — | `orchestration.py` | Orchestrator calls within `max_spawn_depth` |
| `custom` | `rubric` | dispatch | Import and call `module.function(scenario, metrics)` |

### Decision flowchart: "Is this a regression?"

```
CI reports regression (Δ < -5%)
    │
    ├─ Hard gate failed? → YES → Real bug. Fix the code. Do NOT rebaseline.
    │                    → NO ↓
    │
    ├─ Re-run suite 3×. Passes sometimes?
    │   ├─ YES → Flaky. Investigate threshold calibration (§6.4).
    │   └─ NO  → Consistent failure ↓
    │
    ├─ Did agent behavior intentionally change?
    │   ├─ YES → Rebaseline with descriptive commit message.
    │   └─ NO  → Real regression. Investigate the code change.
    │            Check regressed scenario `details` for root cause.
    │
    └─ Is the rubric parsing correctly?
        ├─ YES → Agent behavior issue. Fix agent code.
        └─ NO  → Rubric bug. Fix rubric. Do NOT rebaseline.
```