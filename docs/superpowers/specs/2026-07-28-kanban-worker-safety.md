# Kanban worker safety

## Problem

Assigned kanban tasks can start with an empty body, external coding workers can share one checkout, and the local `kanban-executor` skill permits destructive shortcuts. This produced overlapping T1/T5 edits and a green PR obtained by deleting tests.

## Scope

Protect future coding tasks and safely restart the current T1/T5 work. Keep the existing CPU quota and `max_in_progress: 3`. Do not add containers or a new orchestration subsystem.

## Design

1. Keep title-only cards valid as drafts, but make the dispatcher auto-block an assigned agent/user-created task when `body` is blank. This prevents both new and legacy rows from spawning a worker without changing the low-level card-creation contract.
2. For an existing PC repository, `kanban-executor` must give each task a unique git worktree derived from `HERMES_KANBAN_TASK`; workers must never edit a shared checkout.
3. Remove instructions that bypass destructive-command guards. Forbid `reset --hard` on an existing checkout, broad `git add -A`, and deleting or weakening tests merely to make a run green. Out-of-scope failures are reported and block completion.
4. Leave T2/PR #12 untouched. T1 already produced PR #13 and is blocked; run a fresh isolated audit task against that PR. T5/PR #11 is already merged; run a fresh correction task that restores test integrity and validates the merged behavior. Neither verification task may merge a PR automatically.

## Verification

- A focused dispatcher test proves an assigned agent/user-created blank task is blocked and never spawned; ordinary draft/triage cards remain valid.
- A direct skill-policy check proves the required per-task worktree and safety rules are present and the destructive bypass is absent.
- On the live host, T1/T5 restart with non-empty bodies, distinct worktrees, no shared checkout, at most three workers, and the existing 80% CPU quota.
