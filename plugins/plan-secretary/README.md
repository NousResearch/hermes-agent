# plan-secretary

Turn assistant future-commitments into human-confirmed, session-scoped task plans.

## Problem

Hermes frequently replies with future-looking commitments in ordinary conversation:

> 我接下来会检查 logs/xxx
> I'll fix that after this
> 下一步我会启动 …

There is no durable executor behind these promises. The human reads them as
"work is scheduled", but nothing wakes the agent up, nothing confirms a due
time, and the commitment drifts or gets double-handled.

## Design: capture → confirm → remind → resolve

1. **Capture (precise filter).** Assistant messages are scanned for explicit
   future-commitment sentences. A sentence only becomes a *pending capture*
   when all three gates pass:
   - **actor** — the assistant/agent (小墨 / me / I) is the one committing;
   - **action verb** — 检查/启动/修复/验证/写/跑/check/start/fix/verify/write/run …;
   - **object** — a concrete file, script, process, watcher, log, plan, registry…
   
   Documentation notes, user advice, session-transition suggestions and
   report summaries are rejected (`NON_COMMITMENT_PATTERNS`).

2. **Confirm.** The pending capture is surfaced to the human: **register /
   ignore**. If registering, a **due time** is set (`10m`, `2h`, `1d`,
   `today 17:00`, ISO-8601). If the session already has unfinished plans, the
   human is asked: **replace previous / add in parallel / defer after
   previous**.

3. **Remind (session-scoped).** At the agreed time the reminder is emitted
   **in the same session** where the commitment was made. Each capture/plan
   carries `source_session_id`, and the notifier filters strictly by it — no
   cross-session leakage.

4. **Resolve.** If a previous task is still unfinished at reminder time, the
   human chooses: **run in parallel / defer this / defer other**. Plans can be
   `complete`, `defer`, `block`, or `cancel`.

## State

All state lives under `$HERMES_HOME/state/plan_secretary/` (per-session
isolated files):

```
pending_captures.json   # status: pending|confirmed|ignored|completed
plan_registry.json      # status: active|deferred|blocked|completed|cancelled
plan_status.json        # aggregate summary
watcher_cursor_<sid>.json          # per-session scan cursor
notification_state_<sid>.json      # per-session reminder dedupe
```

## CLI

```
python -m plugins.plan_secretary scan-text --text "..." [--source ... --source-id ... --source-session-id ...]
python -m plugins.plan_secretary list-captures [--all]
python -m plugins.plan_secretary confirm-capture <id> --due 10m --mode parallel --owner 小墨 --worker agent
python -m plugins.plan_secretary ignore-capture <id> --reason "..."
python -m plugins.plan_secretary add --title "..." --due 10m --owner 小墨
python -m plugins.plan_secretary list [--all]
python -m plugins.plan_secretary complete|cancel|block|defer <id> ...
python -m plugins.plan_secretary check [--prereq-defer-minutes 30]
python -m plugins.plan_secretary notify --session-id <sid> [--loop --interval 60]
```

## Resident loop

A per-session resident entry (`resident.py <session-id>`) can be driven by the
existing Hermes cron scheduler: each tick runs an incremental scan (window:
last N minutes; the cursor advances forever) plus a session-scoped reminder
pass. This gives cross-restart durability without a new daemon.

## Testing

```
python -m pytest tests/plugins/plan_secretary/ -q
```
