# Promptfoo Integration Guide — Hermes Eval Harness

This guide explains how to integrate [promptfoo](https://www.promptfoo.dev/) — an
open-source LLM evaluation framework — with the Hermes evals harness
(`evals/runners/run_suite.py`).  The integration lets you run Hermes agent
scenarios as promptfoo test cases, grade them with promptfoo's assertion
engine (`contains`, `contains-any`, `llm-rubric`, `javascript`, `python`),
and produce promptfoo-compatible JSON for dashboarding and CI.

---

## Why Integrate?

The existing Hermes harness (`run_suite.py`) is purpose-built for
deterministic suite scoring: it loads a suite YAML, runs each scenario
against `AIAgent`, grades with a Python rubric, and writes a JSON report.
It works, but it lacks:

- **Side-by-side model comparison** — promptfoo natively runs the same
  test cases against multiple providers/models and renders a comparison
  matrix.
- **Rich assertion types** — `llm-rubric` (LLM-as-judge), `contains-any`,
  `javascript` expressions, `python` script assertions, latency/cost
  thresholds.
- **Web dashboard** — `promptfoo view` serves a local UI for browsing
  results, diffs, and per-test detail.
- **CI integration** — `promptfoo eval` returns a non-zero exit code on
  failure and integrates with GitHub Actions
  ([promptfoo-action](https://github.com/promptfoo/promptfoo-action)).
- **Regression tracking** — promptfoo stores run history and can compare
  against a cached baseline.

The integration is **additive**: `run_suite.py` continues to work as-is.
Promptfoo sits *on top* of the harness as an alternative runner that calls
the same `run_scenario_live()` and rubric `grade()` functions, but adds
promptfoo's assertion engine, multi-provider comparison, and dashboard.

---

## Architecture

```
                         promptfoo eval
                              │
                   ┌──────────┴──────────┐
                   │  promptfooconfig.yaml │
                   │  (tests + assertions)  │
                   └──────────┬──────────┘
                              │
                   ┌──────────┴──────────┐
                   │  hermes_provider.py    │
                   │  (custom Python       │
                   │   provider)            │
                   └──────────┬──────────┘
                              │
                   ┌──────────┴──────────┐
                   │  run_suite.py         │
                   │  run_scenario_live()  │
                   │  + rubric grade()     │
                   └──────────┬──────────┘
                              │
                        AIAgent.run_conversation()
                              │
                        LLM provider API
```

**Data flow:**

1. **promptfoo** reads `promptfooconfig.yaml`, which defines `tests`
   (one per Hermes scenario) and `assert` blocks.
2. Each test's `vars` carry the Hermes scenario fields:
   `user_message`, `system_message`, `config_overrides`,
   `enabled_toolsets`, `suite`, `scenario_id`.
3. promptfoo calls the **custom provider** (`hermes_provider.py`) with the
   prompt + context.
4. The provider imports `run_suite.run_scenario_live()` and invokes
   `AIAgent.run_conversation()`, returning the agent's final response and
   the full message transcript as a JSON blob.
5. promptfoo runs the test's `assert` blocks against the provider output:
   - `contains` / `contains-any` — substring checks on the final response.
   - `llm-rubric` — an LLM-as-judge grades semantic quality.
   - `javascript` — inline expressions for numeric thresholds, regex, etc.
   - `python` (`file://`) — calls the existing Hermes rubric modules
     (e.g., `rubrics/orchestration.py`) for structural pass-condition
     checks like `delegate_call_count`, `plan_score`, `no_cache_break`.
6. promptfoo writes results to JSON (`--output results.json`) and can
   serve a web dashboard (`promptfoo view`).

---

## Prerequisites

1. **Node.js ≥ 18** — promptfoo is a Node CLI.
2. **promptfoo** — install globally:
   ```bash
   npm install -g promptfoo
   # or use npx without installing:
   # npx promptfoo@latest eval ...
   ```
3. **Python 3.10+** with the Hermes repo dependencies installed.
4. **LLM provider credentials** — set in `.env` (e.g.
   `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`) per the Hermes convention.

Verify the install:
```bash
cd /path/to/hermes-agent-source
promptfoo --version
python evals/runners/run_suite.py --suite orchestration --deterministic-only
```

---

## File Layout

```
evals/
├── promptfoo-integration.md      ← this guide
├── promptfoo/
│   ├── config.yaml               ← sample promptfoo config (orchestration suite)
│   ├── hermes_provider.py        ← custom Python provider bridge
│   └── hermes_assert.py          ← Python script assertion (calls Hermes rubrics)
├── runners/
│   └── run_suite.py              ← existing harness (unchanged)
├── rubrics/
│   └── orchestration.py          ← existing rubric (unchanged)
└── suites/
    └── orchestration.yaml        ← existing suite (unchanged)
```

---

## The Custom Provider

promptfoo supports custom providers via `exec:` (shell-out to a Python
script) or a JavaScript class.  We use the **`exec:` Python provider**
approach because it can directly import the Hermes evals harness.

The provider script (`evals/promptfoo/hermes_provider.py`) receives the
prompt as `sys.argv[1]` and test variables via a JSON file path in
`sys.argv[2]` (or environment variable).  It calls
`run_scenario_live()` and returns JSON to stdout:

```python
# Simplified — see evals/promptfoo/hermes_provider.py for the full version
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from evals.runners.run_suite import run_scenario_live

vars = json.loads(sys.argv[2])  # {user_message, system_message, ...}
scenario = {
    "user_message": vars["user_message"],
    "system_message": vars.get("system_message"),
    "config_overrides": vars.get("config_overrides", {}),
    "enabled_toolsets": vars.get("enabled_toolsets", ["terminal", "file", "delegation"]),
    "skip_memory": True,
    "skip_context_files": True,
}
result = run_scenario_live(scenario, vars.get("provider", "openrouter"),
                           vars.get("model", "anthropic/claude-haiku-4.5"))
# promptfoo expects {output: str}
print(json.dumps({"output": json.dumps(result)}))
```

In `config.yaml` the provider is declared as:

```yaml
providers:
  - exec: python evals/promptfoo/hermes_provider.py "{{prompt}}" "{{vars}}"
```

---

## Assertion Mapping

The Hermes harness uses `pass_conditions` in suite YAML.  The
integration maps these to promptfoo assertion types:

| Hermes pass_condition       | promptfoo assertion                          |
|-----------------------------|---------------------------------------------|
| `response_contains`         | `contains`                                  |
| *(multi-value contains)*    | `contains-any`                              |
| `no_tool_error`             | `python` (file://hermes_assert.py)          |
| `delegate_call_count`       | `python` (file://hermes_assert.py)           |
| `plan_score`                | `python` (file://hermes_assert.py)          |
| `no_cache_break`            | `python` (file://hermes_assert.py)          |
| `concurrency_respected`     | `python` (file://hermes_assert.py)          |
| `no_cascade_delegation`     | `python` (file://hermes_assert.py)          |
| `depth_limit_respected`     | `python` (file://hermes_assert.py)          |
| `verify_rate`               | `python` (file://hermes_assert.py)          |
| *(semantic quality)*        | `llm-rubric`                                |
| *(numeric threshold)*       | `javascript`                                |

### When to use which assertion type

- **`contains` / `contains-any`** — fast, deterministic substring checks.
  Use for `response_contains` conditions.  No LLM call needed.

- **`llm-rubric`** — LLM-as-judge.  Use for semantic quality checks that
  can't be expressed as substring matches (e.g., "the agent correctly
  explained why it did not delegate").  Requires a grading model
  (configured via `providers` in the assert block or the default test
  provider).  This is the assertion type that adds the most value over
  the existing harness — it enables qualitative grading.

- **`javascript`** — inline JS expression with access to `output`
  (string), `prompt`, `vars`.  Good for numeric thresholds, regex tests,
  and simple logic.  Example: `Math.max(0, 1 - (output.length - 500) / 5000)`
  to penalize overly long responses.

- **`python` (file://)** — runs a Python script that receives
  `output`, `prompt`, `vars`, and `context` via stdin.  Use this to call
  the **existing Hermes rubric modules** — it imports
  `rubrics/<suite>.py` and calls `grade()`, mapping the result to
  promptfoo's pass/fail + score format.  This is how structural checks
  (delegate_call_count, plan_score, no_cache_break) are preserved.

---

## The Python Script Assertion

`evals/promptfoo/hermes_assert.py` is a promptfoo Python assertion
script.  promptfoo calls it with the test output, prompt, and context
via stdin as JSON.  It imports the Hermes rubric module for the
requested suite, calls `grade()`, and returns a pass/fail + score:

```python
# Simplified — see evals/promptfoo/hermes_assert.py for the full version
import json, sys
from evals.rubrics.orchestration import grade

ctx = json.load(sys.stdin)       # {output, prompt, vars, ...}
result = json.loads(ctx["output"])  # the provider returned JSON
scenario = ctx["vars"]            # the original scenario fields
grade_result = grade(scenario, result)
# promptfoo expects {pass: bool, score: float, reason: str}
print(json.dumps({
    "pass": grade_result["pass"],
    "score": grade_result["score"],
    "reason": json.dumps(grade_result["details"]),
}))
```

In `config.yaml`:

```yaml
assert:
  - type: python
    value: file://evals/promptfoo/hermes_assert.py
```

The `config` key can pass parameters:

```yaml
assert:
  - type: python
    value: file://evals/promptfoo/hermes_assert.py
    config:
      suite: orchestration
      condition: delegate_call_count
      min: 2
```

---

## Running the Integration

### Basic eval

```bash
cd /path/to/hermes-agent-source

# Run the orchestration suite through promptfoo
promptfoo eval \
  -c evals/promptfoo/config.yaml \
  --output evals/promptfoo/results.json

# View the web dashboard
promptfoo view
```

### Multi-model comparison

The config defines multiple providers.  promptfoo runs all test cases
against each provider and produces a comparison matrix:

```yaml
providers:
  - id: openrouter-claude
    label: Claude Haiku (OpenRouter)
    ...
  - id: anthropic-sonnet
    label: Claude Sonnet (Anthropic)
    ...
```

```bash
promptfoo eval -c evals/promptfoo/config.yaml --output results.json
```

### CI integration

```bash
# Non-zero exit on any test failure
promptfoo eval -c evals/promptfoo/config.yaml

# Compare against a cached baseline (regression detection)
promptfoo eval -c evals/promptfoo/config.yaml --cache
```

For GitHub Actions, use
[promptfoo-action](https://github.com/promptfoo/promptfoo-action):

```yaml
# .github/workflows/eval.yml
- uses: promptfoo/promptfoo-action@v1
  with:
    config: evals/promptfoo/config.yaml
    output: evals/promptfoo/results.json
```

---

## Output Format

promptfoo writes results in its own JSON schema (see
[promptfoo output format](https://www.promptfoo.dev/docs/configuration/)).
Key fields:

```json
{
  "results": [
    {
      "providerId": "openrouter-claude",
      "promptId": "orchestration",
      "testCase": { "vars": { "scenario_id": "O1_parallelizable", ... } },
      "response": { "output": "{...agent result JSON...}" },
      "success": true,
      "score": 0.95,
      "assertionResults": [
        { "type": "contains", "pass": true, "value": "delegate", ... },
        { "type": "python", "pass": true, "score": 1.0, ... },
        { "type": "llm-rubric", "pass": true, "score": 0.9, ... }
      ],
      "latencyMs": 8400,
      "cost": 0.0021
    }
  ],
  "stats": { "successes": 4, "failures": 1, "errors": 0, "tokenUsage": {...} }
}
```

This is consumable by:
- **`promptfoo view`** — local web dashboard.
- **`promptfooaction`** — GitHub PR comments with before/after diffs.
- **Custom dashboards** — the JSON is self-describing; pipe into
  Elasticsearch, Datadog, or a Grafana panel.

### Converting to Hermes report format

If you need to feed promptfoo results back into the Hermes report
pipeline (baselines, regression comparison), a converter can map
promptfoo's per-test results to the Hermes report schema:

```python
# evals/promptfoo/to_hermes_report.py (example converter)
def convert(promptfoo_results: dict, suite: str) -> dict:
    scenarios = []
    for r in promptfoo_results["results"]:
        scenarios.append({
            "id": r["testCase"]["vars"].get("scenario_id", "?"),
            "pass": r["success"],
            "score": r.get("score", 0.0),
            "details": {a["type"]: a["pass"] for a in r.get("assertionResults", [])},
            "duration_s": r.get("latencyMs", 0) / 1000,
        })
    total = len(scenarios)
    passed = sum(1 for s in scenarios if s["pass"])
    return {
        "suite": suite,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "scenarios": scenarios,
    }
```

---

## Relationship to the Existing Harness

| Feature                        | `run_suite.py`      | promptfoo integration          |
|-------------------------------|---------------------|-------------------------------|
| Deterministic suite scoring    | ✅ built-in         | ✅ via `python` assertions    |
| Live agent runs                | ✅ built-in         | ✅ via custom provider         |
| Rubric modules                 | ✅ `grade()`        | ✅ called by `hermes_assert.py`|
| Multi-model comparison         | ❌                  | ✅ native                     |
| LLM-as-judge assertions        | ❌                  | ✅ `llm-rubric`               |
| Web dashboard                  | ❌                  | ✅ `promptfoo view`            |
| CI / GitHub Actions            | manual              | ✅ `promptfoo-action`         |
| Baseline regression comparison | ✅ `--baseline`     | ✅ `--cache`                  |
| Latency / cost tracking        | ❌                  | ✅ native assertions          |

**Recommendation:** Use `run_suite.py` for fast deterministic CI checks
(structural invariants, `--deterministic-only` mode).  Use promptfoo for
qualitative evaluation, multi-model comparison, and dashboarding.  The
two systems share the same rubric modules and scenario definitions, so
results are comparable.

---

## Adding a New Suite to promptfoo

1. **Create or reuse a suite YAML** in `evals/suites/<name>.yaml`
   (existing format — no changes needed).
2. **Add test cases to `evals/promptfoo/config.yaml`** — one `tests:`
   entry per scenario, with `vars` matching the scenario fields.
3. **Map pass_conditions to assertions** using the table above.  For
   structural conditions, use the `python` assertion pointing at
   `hermes_assert.py` with `config.suite` set to the suite name.  For
   semantic checks, add `llm-rubric` assertions.
4. **Run:** `promptfoo eval -c evals/promptfoo/config.yaml`.

See `evals/promptfoo/config.yaml` for a complete worked example using
the orchestration suite.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'evals'`**
The provider script and assertion script both add the repo root to
`sys.path`.  If running from a different directory, set
`PYTHONPATH=/path/to/hermes-agent-source` before `promptfoo eval`.

**`promptfoo: command not found`**
Install: `npm install -g promptfoo`, or prefix commands with `npx
promptfoo@latest`.

**Provider hangs / agent never returns**
Check that `agent.max_iterations` in `config_overrides` is reasonable
(12 is the default).  Long-running agents can exceed promptfoo's
default timeout — set `timeout` in the provider config:

```yaml
providers:
  - exec: python evals/promptfoo/hermes_provider.py "{{prompt}}" "{{vars}}"
    config:
      timeout: 120000  # ms
```

**`llm-rubric` assertions fail with empty reason**
The grading model needs API access.  Set `OPENAI_API_KEY` (or
`ANTHROPIC_API_KEY`) in `.env`, or configure a specific grading provider
in the assertion:

```yaml
assert:
  - type: llm-rubric
    value: "..."
    provider: openai:chat:gpt-4o-mini
```

**Deterministic mode (no API calls)**
To test the promptfoo wiring without spending tokens, point the provider
at the `--deterministic-only` path by setting `vars.deterministic_only:
true` in each test case.  The provider will use `_mock_messages` from
the suite YAML instead of calling the live agent.

---

## References

- [promptfoo documentation](https://www.promptfoo.dev/docs/)
- [promptfoo custom providers](https://www.promptfoo.dev/docs/providers/custom-script)
- [promptfoo assertions](https://www.promptfoo.dev/docs/configuration/expected-outputs)
- [promptfoo llm-rubric](https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/llm-rubric)
- [promptfoo GitHub Action](https://github.com/promptfoo/promptfoo-action)
- Hermes evals contract: `evals/CONTRACT.md`
- Hermes runner: `evals/runners/run_suite.py`