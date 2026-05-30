# Hermes Dynamic Workflow MVP

This folder contains a local, optional Dynamic Workflow system for Hermes.

## Structure

```text
hermes_cli/workflow/
  orchestrator.py          Analyzer, planner, router, worker runner, evaluator, synthesizer
  cli.py                   `hermes workflow run` parser and dispatcher
  config/workflow.yaml     Agent and routing configuration
  prompts/*.md             Prompt files for agents, evaluator, and synthesizer
  examples/example_run.md  Dry-run and model-backed examples
```

## Usage

```bash
hermes workflow run "user task here"
```

Use `--dry-run` to exercise the workflow without model calls:

```bash
hermes workflow run --dry-run "Plan tests for a new Python CLI"
```

Use a custom config:

```bash
hermes workflow run --config ./workflow.yaml "research this migration"
```

## Behavior

The MVP runs sequentially:

1. Analyze task type and complexity.
2. Plan 1 to 8 subtasks.
3. Route each subtask to a worker type.
4. Run each worker with configured model/provider.
5. Evaluate acceptance criteria.
6. Run one revision cycle if evaluation fails.
7. Synthesize a concise final answer.

Logs are appended to `~/.hermes/logs/workflow_runs.jsonl`.

## Safety

The bundled config disables worker toolsets by default. That means model-backed
workers produce plans and analysis, not autonomous file changes. Destructive
commands, credential exposure, background execution, and cross-project access
remain out of scope for this MVP.
