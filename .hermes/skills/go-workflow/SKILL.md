---
name: go-workflow
description: Use when operating a repo-contained GO workflow with goals.yaml, tasks.yaml, gates, claim/finish lifecycle, phase skills, and agent prompts for Hermes, Codex, Claude, and Gemini.
version: 1.1.0
author: Viggo/Hermes
license: MIT
metadata:
  hermes:
    tags: [go-workflow, repo-local, orchestration, tasks, goals, agents, phase-gates]
---

# go-workflow Skill Bundle

## Core Rule

The repository is the source of truth. Chat state, Kanban boards, and model memory are runtime aids only.

## Workflow

SETUP → PLAN → ROUTE/CLAIM → BUILD → VERIFY → DOCS/LEDGER → DEVIL → ANTISLOP → SHIP

## Phase skills

- `go-workflow-setup` — SETUP: Establish repository context before planning or editing.
- `go-workflow-plan` — PLAN: Turn intent and repo state into bounded requirements and acceptance checks.
- `go-workflow-route-claim` — ROUTE/CLAIM: Select exactly one executable task and write an exclusive handoff.
- `go-workflow-build` — BUILD: Change only the claimed task's allowed modify scope.
- `go-workflow-verify` — VERIFY: Run the task and repository checks that prove the change.
- `go-workflow-docs-ledger` — DOCS/LEDGER: Update docs, task state, evidence, and run ledgers.
- `go-workflow-devil` — DEVIL: Run adversarial review for risky or multi-file changes.
- `go-workflow-antislop` — ANTISLOP: Remove sloppy artifacts before shipping.
- `go-workflow-ship` — SHIP: Commit, push or PR, and report final git evidence.

## Support skills / routers

- `go-workflow-audit` — AUDIT: Run selectable quality gates for code, tests, architecture, API, performance, accessibility, UX, and security-style checks.
- `go-workflow-git` — GIT ROUTER: Keep repository state safe: status hygiene, selective staging, ship, CI repair, PR/check inspection, and clean return to main.
- `go-workflow-release` — RELEASE: Separate release/version/tag/notes behavior from generic docs-ledger work and make final publish evidence explicit.
- `go-workflow-interview` — INTERVIEW: Ambiguity gate for work that cannot be safely planned without a small number of targeted questions.
- `go-workflow-cancel` — CANCEL: Gracefully stop, supersede, or cancel obsolete workflow tasks without corrupting queue or git state.
- `go-workflow-better` — BETTER: Post-run improvement loop: turn a completed task/session into better workflow rules, tasks, tests, docs, or skill updates.

Load the root skill first, then the phase or support skill matching the current gate/router when you need detailed inputs, outputs, allowed mutations, evidence, stop conditions, and handoff rules.

## Entry Points

```bash
python3 scripts/next_task.py --validate
python3 scripts/next_task.py --list --limit 5
python3 scripts/next_task.py --claim --agent <hermes|codex|claude|gemini|name>
python3 scripts/finish_task.py <TASK-ID> --evidence "<path-or-command>" --agent <name>
python3 scripts/gate.py --task-id <TASK-ID> --phase verify --evidence "task_verification_run=pytest passed"
```

Every workflow trigger must show the upcoming task preview before claiming or executing work. Use `python3 scripts/next_task.py --list --limit 5` and include that short list in the agent's first checkpoint/report.

## Rules

- Read `AGENTS.md` first, then `.go-workflow/config.yaml`, `goals.yaml`, `tasks.yaml`, and `gates.yaml`.
- Claim exactly one `ready` task before editing.
- Stay inside `scope.modify` unless the task is updated first.
- Do not claim `todo`, `waiting`, `review`, `done`, or `cancelled` work.
- Do not finish without evidence.
- For risky/multi-file changes, run a devil review before ship.
- Commit one task at a time unless tasks are inseparable.
