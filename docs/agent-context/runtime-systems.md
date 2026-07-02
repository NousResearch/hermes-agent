# Hermes Runtime Systems Reference

This reference preserves runtime system orientation formerly embedded in `AGENTS.md`.

## Toolsets

Toolsets are defined in `toolsets.py` as a single `TOOLSETS` dict. `_HERMES_CORE_TOOLS` is the default bundle most platforms inherit from.

Enable/disable per platform with `hermes tools` or the `tools.<platform>.enabled` / `tools.<platform>.disabled` config lists.

Toolset changes take effect on a new session; do not mutate active toolsets mid-conversation because that breaks prompt caching.

## Delegation (`delegate_task`)

`tools/delegate_tool.py` spawns subagents with isolated context and terminal sessions.

Shapes:

- single: `goal`, optional `context`, optional `toolsets`
- batch: `tasks: [...]`, each child runs concurrently up to `delegation.max_concurrent_children`

Roles:

- `leaf` — focused worker; cannot re-delegate or call high-control tools
- `orchestrator` — can spawn workers, gated by config and max depth

Background delegation is process-local. For work that must survive restart, use cron jobs or terminal background processes with completion notification.

## Curator

Curator tracks agent-created skills, marks stale skills, and archives them with backups. It only touches skills with `created_by: "agent"` provenance.

Invariants:

- bundled and hub-installed skills are off-limits
- never deletes; max action is archive
- pinned skills are exempt from auto-transitions and LLM review

## Cron

Cron consists of `cron/jobs.py` and `cron/scheduler.py`.

Users and agents can schedule durable jobs via:

- `cronjob` tool
- `hermes cron <verb>`
- `/cron`

Supported schedules include durations, “every” phrases, 5-field cron, and ISO timestamps.

Hardening invariants:

- hard interrupt on runaway cron sessions
- file lock prevents duplicate ticks
- cron sessions skip memory by default
- deliveries are framed and not mirrored into the target session, preserving role alternation

## Kanban

Kanban is a durable SQLite work queue for multi-profile/multi-worker collaboration.

Users drive `hermes kanban <verb>`; workers get a focused `kanban_*` toolset gated to a specific board/task.

Isolation:

- board is the hard boundary
- tenant is a soft namespace within a board
- dispatcher can auto-block repeatedly failing tasks

## Background process notifications

Gateway background process notifications are controlled by `display.background_process_notifications`:

- `all`
- `result`
- `error`
- `off`

Use `terminal(background=True, notify_on_complete=True)` for bounded long-running commands.

## Project context files

Hermes reads one project context source per session, first match wins:

1. `.hermes.md` / `HERMES.md` — parent walk to git root
2. `AGENTS.md` / `agents.md` — cwd only
3. `CLAUDE.md` / `claude.md` — cwd only
4. Cursor rules — cwd only

Each context file is capped at 20,000 characters and head+tail truncated when longer. Keep always-on project context short; move detail to skills or docs.
