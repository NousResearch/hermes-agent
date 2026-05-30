# Example Run

```bash
hermes workflow run --dry-run "Create a README and tests for a small Python CLI"
```

The dry run does not call models. It exercises analyzer, planner, router,
evaluator, synthesizer, JSONL logging, and SQLite run state.

Run dependency-ready subtasks in parallel:

```bash
hermes workflow run --dry-run --parallel "Plan research, review, and tests for a release"
```

Queue a background run and inspect it:

```bash
hermes workflow run --dry-run --background "Plan a workflow status smoke test"
hermes workflow status <run_id>
hermes workflow logs <run_id> --tail 5
```

Generate a Codex Goal checklist:

```bash
hermes workflow run --dry-run --codex-plan "Break down the multi-role login rollout"
```

Run a contest harness with cross-review:

```bash
hermes workflow run --dry-run --contest --cross-review --codex-plan \
  "Choose the best next plan for homeowner, contractor, and worker adoption"
```

For a model-backed run, omit `--dry-run`:

```bash
hermes workflow run "Plan a migration for the billing webhook"
```

Runtime logs are written to:

```text
~/.hermes/logs/workflow_runs.jsonl
```

Durable state is written to:

```text
~/.hermes/workflow_runs.db
```
