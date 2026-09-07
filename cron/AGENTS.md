# cron/ (+ kanban) — scheduled jobs and the multi-agent work queue

Applies on top of the root `AGENTS.md`. Long-form: `website/docs/developer-guide/cron-internals.md`;
user docs `website/docs/user-guide/features/cron.md`, `kanban.md`.

## Cron

`cron/jobs.py` (job store) + `cron/scheduler.py` (tick loop; `scheduler_*.py` siblings). Agents
schedule via the `cronjob` tool; users via `hermes cron list|add|edit|pause|resume|run|remove` or
`/cron`. Schedules: duration (`"30m"`, `"2h"`, `"1d"`), "every" phrase (`"every 2h"`, `"every monday
9am"`), 5-field cron (`"0 9 * * *"`), ISO one-shot (`"2026-06-01T09:00:00Z"`). Per-job fields:
`skills`, `model`/`provider` overrides, `script` (pre-run data-collection script whose stdout is
injected into the prompt; `no_agent=True` makes the script the whole job), `context_from` (chain job
A's last output into job B's prompt), `workdir` (run with that directory's `AGENTS.md`/`CLAUDE.md`
loaded), multi-platform delivery.

Hardening invariants — each guards a real failure; don't weaken without answering for it:
- **3-minute hard interrupt** on cron sessions: runaway loops cannot monopolise the scheduler.
- Catch-up window = half the period, clamped to 120s–2h; 120s grace for missed one-shots.
- File lock `~/.hermes/cron/.tick.lock` prevents duplicate ticks across processes.
- Cron sessions pass `skip_memory=True`; memory providers intentionally do not run during cron.
- Deliveries are **not mirrored** into the target gateway session — they land in their own cron
  session with a header/footer frame so the main conversation's role alternation stays intact.
- The cron ticker runs in the desktop-spawned backend when `HERMES_DESKTOP=1` — that env var means
  "spawned by the app", not "a GUI is watching" (root: capability is a property of the session).
- Background `delegate_task` is process-local; work that must survive restarts is a cron job or a
  `terminal(background=True, notify_on_complete=True)` process.

## Kanban (multi-agent work queue)

Durable SQLite-backed board letting multiple profiles/workers collaborate. Users: `hermes kanban
<verb>`; dispatcher-spawned workers use a dedicated `kanban_*` toolset so their schema footprint is
zero outside a kanban task (footprint ladder rung 3).

- **CLI:** `hermes_cli/kanban.py` facade + 14 `kanban_*.py` siblings (`boards`, `db`, `db_connect`,
  `db_dispatch`, `db_notify`, `workspace`, ...). Verbs: `init, create, list (ls), show, assign, link,
  unlink, comment, attach, attachments, attach-rm, complete, request-review, request-changes,
  reopen-review, block, unblock, archive, tail`, plus `watch, stats, runs, log, assignees, heartbeat,
  notify-*, dispatch, daemon, gc`. Argparse alias dispatch must accept both `list` and `ls` (root).
- **Toolset:** `tools/kanban_tools.py` — `kanban_show, kanban_complete, kanban_request_review,
  kanban_request_changes, kanban_block, kanban_heartbeat, kanban_comment, kanban_create, kanban_link,
  kanban_attach, kanban_attach_url, kanban_attachments`; profiles enabling `kanban` outside a
  dispatched task also get `kanban_list` and `kanban_unblock` for board routing.
- **Dispatcher:** long-lived loop (default 60s) that reclaims stale claims, promotes ready tasks,
  atomically claims, and spawns assigned profiles. Runs **inside the gateway** by default
  (`kanban.dispatch_in_gateway: true`). Standalone: `plugins/kanban/systemd/hermes-kanban-dispatcher.service`.
- **Plugin assets:** `plugins/kanban/dashboard/` (web UI) + systemd unit. `kanban_db.connect` is its
  own connection helper — do not alias it to `projects_db.connect` (a path-proximity generator did).

Isolation: **board** is the hard boundary — workers get `HERMES_KANBAN_BOARD` pinned in their env and
cannot see other boards; **tenant** is a soft namespace within a board (workspace-path + memory-key
isolation, one fleet serving several businesses). After `kanban.failure_limit` consecutive
non-success attempts on a task (default 2) the dispatcher auto-blocks it to stop spin loops.
Process-identity note: `kanban --preserve-cache` contains "serve" — never classify processes by argv
substring (root).

## Tests

`tests/cron/`, `tests/hermes_cli/test_kanban*.py`, `tests/tools/test_kanban*.py`. Schedule parsing
and catch-up windows are pure functions — test them as data. Never assert on the verb list or
toolset size (root: no change-detectors). Time-based tests use loose bounds (≥ 2s) and event sync.
