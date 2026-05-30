# Hermes Dynamic Workflow

This folder contains a local, optional Dynamic Workflow system for Hermes.
It is Python-native and keeps normal Hermes behavior unchanged unless the
`hermes workflow` command is used.

## Structure

```text
hermes_cli/workflow/
  orchestrator.py          Analyzer, planner, router, worker runner, evaluator, synthesizer
  runtime.py               Runtime, scheduler, agent executor, evaluator loop
  state.py                 SQLite run state and event history
  script_api.py            Reusable Python workflow script helpers
  cli.py                   `hermes workflow ...` parser and dispatcher
  config/workflow.yaml     Agent, execution, safety, and routing configuration
  prompts/*.md             Prompt files for agents, evaluator, and synthesizer
  examples/example_run.md  Dry-run and model-backed examples
```

## Usage

Run a foreground workflow:

```bash
hermes workflow run "user task here"
```

Exercise the workflow without model calls:

```bash
hermes workflow run --dry-run "Plan tests for a new Python CLI"
```

Run dependency-ready subtasks concurrently:

```bash
hermes workflow run --parallel --dry-run "Plan a release checklist"
```

Queue an explicit background run:

```bash
hermes workflow run --background --dry-run "Research options and summarize risks"
hermes workflow status <run_id>
hermes workflow logs <run_id>
```

Pause, resume, or cancel a persisted run:

```bash
hermes workflow pause <run_id>
hermes workflow resume <run_id>
hermes workflow cancel <run_id>
```

Format output for Codex Goal Mode:

```bash
hermes workflow run --codex-plan --dry-run "Break down the multi-role login rollout"
```

Run a contest harness with adversarial cross-review:

```bash
hermes workflow run --contest --cross-review --codex-plan \
  "Choose the best next plan for the multi-role login rollout"
```

Run a reusable Python workflow script:

```python
# workflow_example.py
research = task(
    "Compare workflow runtime options",
    worker="researcher",
    context="Keep this local and safe by default.",
    risk_level="low",
)
plan = task(
    "Turn findings into an implementation plan",
    worker="product_planner",
    depends_on=[research],
)
parallel([research], max_concurrency=2)
```

```bash
hermes workflow script --dry-run ./workflow_example.py
```

## Behavior

The runtime:

1. Analyzes task type and complexity.
2. Plans 1 to 8 structured subtasks.
3. Assigns worker types from routing rules and agent descriptions.
4. Persists run state, subtasks, outputs, evaluator scores, and events.
5. Executes sequentially by default, or in dependency-aware parallel mode with
   `--parallel`.
6. Optionally runs contest mode where multiple workers independently answer a
   subtask, reviewer/tester workers cross-review the candidates, and the best
   candidate is selected.
7. Evaluates each output and performs one bounded revision cycle if needed.
8. Synthesizes a concise final answer or a Codex Goal checklist.

JSONL logs are appended to `~/.hermes/logs/workflow_runs.jsonl`. Durable run
state is stored in `~/.hermes/workflow_runs.db` unless `execution.state_db` is
set in the config.

## Safety

No workflow runs automatically. Background execution requires `--background`.
Worker toolsets are disabled unless both the global safety config and the
specific agent allow file writes. Destructive commands still require explicit
confirmation. Logged payloads pass through Hermes redaction before being written
to JSONL or SQLite events.
