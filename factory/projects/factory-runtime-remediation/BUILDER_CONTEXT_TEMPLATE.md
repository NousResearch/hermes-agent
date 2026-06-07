# Builder Context Template — Factory Runtime Remediation

Use this template when assigning a Factory task to Claude Code, Codex, OpenHands, or another builder engine.

## Required builder input

```yaml
project_id: factory-runtime-remediation
lane_id: factory-runtime-remediation-runtime-contract-v1
repo_path: /home/jean/workspace/hermes-factory-runtime-contract-v1
branch: factory/factory-runtime-contract-v1
documentation_index: factory/projects/factory-runtime-remediation/DOCUMENTATION_INDEX.md
project_global_vision: factory/projects/factory-runtime-remediation/PROJECT_GLOBAL_VISION.md
runtime_contract: factory/projects/factory-runtime-remediation/FACTORY_RUNTIME_CONTRACT_V1.md
active_task_id: <factory task id>
acceptance_criteria:
  - <criterion 1>
  - <criterion 2>
forbidden_actions:
  - Do not push, merge, or deploy unless Jean explicitly asks.
  - Do not edit another project's files unless the task says so.
  - Do not write directly to factory.* with psql; use Factory CLI/tools for gates/tasks.
  - Do not treat Notion/dashboard as source of truth.
expected_evidence:
  - tests run with exact command and result
  - files changed
  - doc updates
  - git status/diff summary
  - final STATE: DONE or STATE: BLOCKED
```

## Required reading order for builders

1. `DOCUMENTATION_INDEX.md`
2. `PROJECT_GLOBAL_VISION.md`
3. `FACTORY_RUNTIME_CONTRACT_V1.md`
4. `PRD.md`
5. `ADRS.md`
6. `TECHNICAL_BLUEPRINT.md`
7. `SPRINT_PLAN.md`
8. `TASK_GRAPH.md`
9. Task-specific prompt/acceptance criteria from Factory DB

## Expected run result shape

Future increment should make this a machine-validated `run_result.json`:

```json
{
  "project_id": "factory-runtime-remediation",
  "task_id": "<task-id>",
  "outcome": "done|blocked|failed|partial",
  "files_changed": [],
  "commands_run": [],
  "tests": [],
  "evidence_paths": [],
  "blockers": [],
  "doc_updates": [],
  "commit_sha": null,
  "summary": "",
  "final_state_marker": "STATE: DONE"
}
```

## Reviewer checklist

- Did the builder read the documentation index and current task docs?
- Do code changes match the PRD and runtime contract?
- Were tests written before production code when behavior changed?
- Are project-local artifacts indexed?
- Are project-local artifacts committed or explicitly waived?
- Is the final status based on tests/gates/evidence, not prose confidence?
