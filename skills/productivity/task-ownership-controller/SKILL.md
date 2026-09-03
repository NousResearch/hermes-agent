---
name: task-ownership-controller
description: Track task state, retries, and evidence-gated completion.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [tasks, state-machine, verification, retries, cron]
    category: productivity
    related_skills: [weekly-review-planning]
---

# Task-Ownership Controller Skill

Durable local tracking for a task you own end to end — not a TODO list, a
state machine. It records what state a task is in, what the next action
is, why it's blocked, how many times it's been retried, and — the
load-bearing rule — it refuses to let you mark a task DONE without
verification evidence on file. Backed by `hermes task`, a CLI over a local
SQLite DB (`~/.hermes/task_ownership.db`). Off by default; explicit
commands always work even while off (see Prerequisites).

## When to Use

- You're carrying a multi-step task across turns/sessions and need durable
  state (next action, blockers, decisions) that survives context loss.
- A task involves retries against a flaky dependency and you need a bounded
  retry count with a recorded fallback once it's exhausted.
- A task involves an external side effect (payment, email, API call) from
  an at-least-once caller and you need to record receipts idempotently so
  a retry doesn't double-book the effect.
- You are tempted to declare a task "done" and want a hard stop that
  forces you to record concrete verification evidence first.
- Not for: ephemeral in-session todos with no cross-session durability need
  — use the `todo` tool for those. Not for multi-agent/multi-worker
  coordination across a shared board — use `kanban` for that.

## Prerequisites

None — ships with Hermes core, no extra install. The controller has a
feature flag (`task_ownership.enabled` in `config.yaml`, default `false`)
that gates only the unattended `hermes task age-check` command (e.g. run
from cron): with the flag off, `age-check` is a strict no-op — no output,
no state mutation. Every other command (`create`, `update`, `outcome`,
`receipt`, `verify`, `done`, ...) works regardless of the flag, since
those are explicit actions you're deliberately taking, not passive
automation. Turn the flag on with `hermes task enable` once you also want
unattended aging sweeps; `hermes task disable` cleanly turns it back off.

## How to Run

```bash
hermes task create "Migrate customer export job" --next-action "write the batch script" --owner me
# -> t_ab12cd34ef56  [NEW]  Migrate customer export job

hermes task update t_ab12cd34ef56 --state WORKING
hermes task outcome t_ab12cd34ef56 --result failure --detail "rate limited" --retry --fallback "page on-call"
hermes task verify t_ab12cd34ef56 --evidence "output row count matches source (12,904 == 12,904)"
hermes task done t_ab12cd34ef56
hermes task show t_ab12cd34ef56 --json
```

## Quick Reference

| Task | Command |
|------|---------|
| Create a task | `hermes task create "<title>" [--next-action ...] [--owner ...] [--max-retries N] [--approval-required]` |
| List tasks | `hermes task list [--state STATE] [--all] [--json]` |
| Show one task | `hermes task show <id> [--json]` |
| Update fields / explicit transition | `hermes task update <id> [--next-action ...] [--blocker ...] [--decision ...] [--owner ...] [--fallback ...] [--state STATE]` |
| Record a worker attempt | `hermes task outcome <id> --result success\|failure\|partial [--detail ...] [--retry] [--fallback ...]` |
| Record an idempotent external receipt | `hermes task receipt <id> --receipt-id <ext-id> [--source ...] [--payload ...]` |
| Record verification evidence | `hermes task verify <id> --evidence "..."` |
| Mark DONE (requires evidence on file) | `hermes task done <id> [--evidence "..."]` |
| Record approval (approval-required tasks) | `hermes task approve <id> --by <name>` |
| Block / cancel | `hermes task block <id> --reason "..."` / `hermes task cancel <id> [--reason ...]` |
| Show a task's audit trail | `hermes task events <id> [--json]` |
| Evaluate aging (24h warn / 72h stale) | `hermes task age-check [--dry-run] [--json]` |
| Feature flag on/off | `hermes task enable` / `hermes task disable` |
| Overview: flag state + counts by state | `hermes task status [--json]` |

## Procedure

1. **Create** the task with `hermes task create`. It starts in `NEW` with
   a `next_action`.
2. **Work it.** Move to `WORKING` via `hermes task update <id> --state
   WORKING`. Update `next_action`/`blocker`/`decision` as you learn things
   — these are the fields to re-read after a context reset instead of
   re-deriving state from scratch.
3. **Record every attempt** with `hermes task outcome`. On failure with
   `--retry`, the controller bumps `retry_count` and moves to `RETRYING`;
   once `retry_count` exceeds `max_retries` it auto-transitions to
   `BLOCKED` and records whatever `--fallback` you gave on that call.
   `outcome` can never set a task to `DONE` — that's by design.
4. **Record receipts** for any external side effect via `hermes task
   receipt --receipt-id <external-idempotency-key>`. Calling it twice with
   the same `receipt_id` is a safe no-op (`duplicate: true` in the JSON) —
   this is what makes it safe to retry the surrounding operation.
5. **Verify before you claim done.** `hermes task verify --evidence "..."`
   records concrete evidence (a diff, a row count match, a URL, an output
   snippet — something checkable, not "looks right"). This moves the task
   to `VERIFYING`.
6. **Mark done.** `hermes task done` refuses unless verification evidence
   is on file (recorded in step 5, or passed inline via `--evidence`) and,
   for tasks created with `--approval-required`, unless `hermes task
   approve` has been recorded. There is no other path to `DONE`.
7. **(Optional) Wire aging into cron.** Once you've turned the flag on
   with `hermes task enable`, schedule a periodic sweep:
   ```bash
   cat > ~/.hermes/scripts/task_age_check.sh <<'SH'
   #!/usr/bin/env bash
   hermes task age-check
   SH
   chmod +x ~/.hermes/scripts/task_age_check.sh
   hermes cron add --no-agent --script task_age_check.sh --name task-aging "every 1h"
   ```
   `age-check` only prints when a task newly crosses the 24h/72h threshold
   (each tier fires once per aging window, not every run) — empty stdout
   is silent per the cron `--no-agent` contract, so this never spams.

## Pitfalls

- **Don't try to shortcut `done`.** There's no flag or force-mode to skip
  verification evidence — that's the entire point of the invariant. If a
  task genuinely can't be verified, record why in `--decision` and leave
  it in `VERIFYING`/`WORKING` rather than inventing evidence.
- **`outcome --result success` does not complete the task.** It logs the
  attempt only. You still need `verify` + `done`.
- **Aging only auto-progresses to `STALE`, never further** — the
  controller doesn't auto-cancel or auto-retry a stale task. A human/agent
  has to look at it and move it via `hermes task update --state WORKING`
  (or `cancel`).
- **`--max-retries` is set at creation** (or via `hermes task update
  --max-retries`); raising it after the fact does not clear an existing
  `BLOCKED` state — you still need `--state WORKING` to resume.

## Verification

- [ ] `hermes task show <id>` reflects the expected state, next action,
      and (if applicable) verification evidence.
- [ ] `hermes task events <id>` shows a `verification_recorded` event
      before any `completed` event — DONE never appears without it.
- [ ] `hermes task status` shows the feature flag state you expect before
      relying on `age-check` output in automation.
