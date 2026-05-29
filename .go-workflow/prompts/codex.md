# go-workflow prompt for codex

You are working in a repository that uses **go-workflow**.

## Mandatory startup

1. Read `AGENTS.md`.
2. Read `.go-workflow/config.yaml`.
3. Read `.go-workflow/goals.yaml` and `.go-workflow/tasks.yaml`.
4. Validate: `python3 scripts/next_task.py --validate`.
5. Show the next tasks first: `python3 scripts/next_task.py --list --limit 5`.
6. Claim exactly one task: `python3 scripts/next_task.py --claim --agent codex`.
7. Read the generated handoff in `.go-workflow/runs/`.

## Workflow phases

SETUP → PLAN → ROUTE/CLAIM → BUILD → VERIFY → DOCS/LEDGER → DEVIL → ANTISLOP → SHIP

## Non-negotiables

- The repo is the source of truth.
- Stay inside `scope.modify` for the claimed task.
- Run the listed verification.
- Finish only with explicit evidence.
- Report changed files, verification, evidence, and git state.
