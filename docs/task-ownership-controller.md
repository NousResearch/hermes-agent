# Task-Ownership Controller

Operator reference for `hermes task` — durable local state for tasks an
agent or operator owns end to end, with completion gated on recorded
verification evidence. Edge-layer feature (CLI + skill + optional cron
wiring); no new core model tool, no gateway/agent-loop changes.

## Files

- `hermes_cli/task_ownership_db.py` — storage engine: SQLite schema, state
  machine, CRUD, receipts, outcomes, verification, aging.
- `hermes_cli/task_ownership.py` — `hermes task` argparse subcommands
  (`register_cli`, wired in `hermes_cli/main.py`).
- `hermes_cli/config_defaults.py` — `task_ownership:` config section.
- `skills/productivity/task-ownership-controller/SKILL.md` — agent-facing
  usage skill.
- `tests/hermes_cli/test_task_ownership_db.py`,
  `tests/hermes_cli/test_task_ownership_cli.py` — test suite.

## Storage

SQLite DB at `<HERMES_HOME>/task_ownership.db` (profile-scoped via
`get_hermes_home()`; a fresh install creates it lazily on first `hermes
task` invocation, or explicitly via `hermes task init`). Four tables:
`tasks`, `task_events` (full audit trail of every state transition),
`task_outcomes` (worker attempt log), `task_receipts` (idempotent external
receipts, `UNIQUE(task_id, receipt_id)`).

## State machine

`NEW, WORKING, WAITING_FOR_USER, RETRYING, VERIFYING, DONE, BLOCKED,
STALE, CANCELLED`. Terminal: `DONE`, `CANCELLED`. Reachability is
enforced by a transition table in `task_ownership_db.TRANSITIONS`;
`set_state()` raises `InvalidTransitionError` outside it.

**No-false-completion invariant:** `mark_done()` is the *only* function
that can set `state=DONE`, and it refuses unless:

1. `verification_evidence` is on the task row (via `hermes task verify`,
   or inline `hermes task done --evidence ...`), and
2. for tasks created with `--approval-required`, `approved_by` is set
   (via `hermes task approve`), and
3. the task is currently in `VERIFYING`.

`record_outcome()` (worker attempt logging) cannot set `DONE` under any
result value — a worker self-reporting "success" is not sufficient to
complete a task.

## Bounded retries and fallback

`hermes task outcome <id> --result failure --retry [--fallback "..."]`
increments `retry_count`. While `retry_count <= max_retries` (set at
`create` time, default 3, configurable via
`task_ownership.default_max_retries`) it moves the task to `RETRYING`.
Once exceeded, it auto-transitions to `BLOCKED`, records the blocker
reason, and persists `--fallback` if given.

## Idempotent receipts

`hermes task receipt <id> --receipt-id <external-key> [--source ...]
[--payload ...]`. Same `(task_id, receipt_id)` recorded twice is a no-op
— the second call returns the original row with `duplicate: true` and
does not insert a second row. Safe for at-least-once external callers to
retry.

## Aging (24h / 72h) without notification spam

`hermes task age-check` scans every non-terminal task's
`state_changed_at`. Each tier fires **once per distinct
`state_changed_at` value** (tracked via `aged_24h_marker` /
`aged_72h_marker` columns) — a task that has been silent for a week
doesn't get reflagged on every run, only once per aging window. The 72h
tier also auto-transitions the task to `STALE` (state change resets the
anchor). Thresholds are configurable:
`task_ownership.aging_warn_hours` (default 24),
`task_ownership.aging_stale_hours` (default 72).

## Feature flag — default OFF / shadow mode

`task_ownership.enabled` in `config.yaml`, default `false`. Only
`age-check` is gated:

- **Disabled (default):** `age-check` is a strict no-op — no stdout, no DB
  mutation. `--dry-run` still works (pure read, prints a preview,
  regardless of the flag) so you can inspect what *would* happen before
  opting in.
- **Enabled:** `age-check` mutates markers/state and prints newly-crossed
  tasks (or JSON via `--json`).

Every other command (`create`, `update`, `outcome`, `receipt`, `verify`,
`done`, `approve`, `block`, `cancel`, `events`, `status`, `list`, `show`)
works regardless of the flag — those are explicit operator/worker
actions, not passive automation, so gating them would just get in the
way of the "real reversible system" the flag is meant to protect.

### Enable / disable / rollback

```bash
hermes task enable      # sets task_ownership.enabled: true
hermes task disable     # sets task_ownership.enabled: false
```

Disabling is a clean, complete rollback of the *automation surface*:
`age-check` goes immediately inert (no code path runs once the flag read
returns `false`), while all durable task state (the SQLite DB) is left
exactly as-is — nothing is deleted, nothing needs a migration to reverse.
To remove the feature entirely (e.g. in a test/throwaway profile), delete
`<HERMES_HOME>/task_ownership.db`; no other Hermes state references it.

## Optional cron wiring

Not a native hook — documented, standard `--no-agent` cron script (see
AGENTS.md § Cron):

```bash
cat > ~/.hermes/scripts/task_age_check.sh <<'SH'
#!/usr/bin/env bash
hermes task age-check
SH
chmod +x ~/.hermes/scripts/task_age_check.sh
hermes task enable
hermes cron add --no-agent --script task_age_check.sh --name task-aging "every 1h"
```

`--no-agent` delivers the script's stdout verbatim and skips the LLM
entirely; empty stdout is silent, matching `age-check`'s
already-deduplicated output.

## Command reference

See the Quick Reference table in
`skills/productivity/task-ownership-controller/SKILL.md`, or `hermes task
--help` / `hermes task <verb> --help`.
