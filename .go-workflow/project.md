# go-workflow Project Contract — hermes-agent

This repository uses go-workflow as its local source of truth for agentic work.

## Canonical files

- `.go-workflow/config.yaml` — policy and paths.
- `.go-workflow/goals.yaml` — durable outcomes.
- `.go-workflow/tasks.yaml` — machine-readable task queue.
- `.go-workflow/gates.yaml` — phase gates.
- `.go-workflow/prompts/` — model-specific prompts.
- `.go-workflow/skills/go-workflow/SKILL.md` — portable root skill bundle.
- `.go-workflow/skills/go-workflow-<phase>/SKILL.md` — phase skills for setup, plan, route-claim, build, verify, docs-ledger, devil, antislop, and ship.
- `.go-workflow/skills/go-workflow-<support>/SKILL.md` — support routers for audit, git, release, interview, cancel, and better.
- `.go-workflow/runtime/go_workflow/` — embedded runtime used by `scripts/` wrappers, so target repos can run without pip install.
- `.hermes/skills/go-workflow*/SKILL.md` — compatibility view for Hermes-style skill discovery.
- `scripts/next_task.py`, `scripts/finish_task.py`, and `scripts/gate.py` — stable self-contained repo entrypoints.

## Workflow

SETUP → PLAN → ROUTE/CLAIM → BUILD → VERIFY → DOCS/LEDGER → DEVIL → ANTISLOP → SHIP
