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
- Every recurring occurrence is accounted for: `tick()` advances `next_run_at` BEFORE dispatch
  (at-most-once across a mid-run crash) and stamps `pending_slot` in the same save; a scan that
  finds the stamp with a dead owner restores the instant ONCE (`cron/occurrences.py`), the
  executions ledger's `scheduled_instant` blocks a second fire, `cron.catch_up_missed: false`
  skips past-grace misses with a logged reason. Never drop a slot silently (#107485).
- Per-home tick lock `<home>/cron/.tick.lock` prevents duplicate ticks across processes for
  that profile's store; never a `~/.hermes/...` literal.
- **The ticker binds each served profile's scope for the whole tick, including pre-loop code.**
  `scheduler_provider.py::_start_multiplex` is ONE ticker iterating `profiles_to_serve()`
  sequentially under `_profile_cron_scope(home)` (home + secret scope + terminal scope) — never N
  threads (module globals race). Everything a tick touches lives inside that guard: store open,
  lock path, backoff/failure counters (`_note_tick_failure`), job env construction, and the
  `on_session_end` flush of a finished job. Supervision (`scheduler_thread.py::
  SupervisedTickerThread`; start on gateway boot, stand down for homes another gateway already
  serves, re-enumerate when a profile dir appears or is tombstoned) is per served home, not per
  process. Why: a store opened before the scope was entered wrote a secondary profile's run
  records into the launch profile's `jobs.json`.
- Cron sessions pass `skip_memory=True`; memory providers intentionally do not run during cron.
- Cron execution has its own session. Eligible continuable deliveries may mirror or seed the
  reply-facing conversation: origin, origin-less home fallback, user-written bare-platform home,
  or opted-in explicit targets. `all` expansions do not gain home mirror eligibility. Mirrored
  briefs are labelled user turns appended at a turn boundary, preserving role alternation.
- The cron ticker runs in the desktop-spawned backend when `HERMES_DESKTOP=1` — that env var means
  "spawned by the app", not "a GUI is watching" (root: capability is a property of the session).
- Background `delegate_task` is process-local; work that must survive restarts is a cron job or a
  `terminal(background=True, notify_on_complete=True)` process.

## Kanban (multi-agent work queue)

Durable SQLite-backed board letting multiple profiles/workers collaborate. Users: `hermes kanban
<verb>`; dispatcher-spawned workers use a dedicated `kanban_*` toolset so their schema footprint is
zero outside a kanban task (footprint ladder rung 3).

- **CLI:** `hermes_cli/kanban.py` facade + `kanban_*.py` siblings (`boards`, `db`, `db_connect`,
  `db_dispatch`, `db_notify`, `db_graph` (task initialization and decomposition), `db_skills`
  (forced-skill validation), `workspace`, ...). Verbs: `init, create, list (ls), show, assign, link,
  unlink, comment, attach, attachments, attach-rm, complete, request-review, request-changes,
  reopen-review, block, unblock, archive, tail`, plus `watch, stats, runs, log, assignees, heartbeat,
  notify-*, dispatch, daemon, gc`. Argparse alias dispatch must accept both `list` and `ls` (root).
- **Toolset:** `tools/kanban_tools.py` — `kanban_show, kanban_complete, kanban_request_review,
  kanban_request_changes, kanban_block, kanban_heartbeat, kanban_comment, kanban_create, kanban_link,
  kanban_attach, kanban_attach_url, kanban_attachments`; platforms whose saved selection enables
  `kanban` (`hermes tools enable kanban --platform <p>`; default-off, in `CONFIGURABLE_TOOLSETS`) get
  the full set plus `kanban_list`/`kanban_unblock` for board routing. The check_fn reads the schema
  build's own selection (`tools/kanban_toolset_context.py`), never the legacy top-level `toolsets`
  key alone.
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
substring (root). Worker liveness is `(worker_pid, worker_started_at)` — the start-time fingerprint
(`gateway.status.get_process_start_time`) recorded at claim time — never bare PID existence, or a
recycled PID gets killed on reclaim.

- **Notifications leave through the task's owning profile.** `hermes_cli/kanban_db_notify.py`
  subscriptions carry the profile; `gateway/kanban_watchers_notifier.py` delivers via THAT
  profile's adapter under its scope (`_notify_profile_filter`), never the multiplexer's launch
  adapter; a fail-closed skip logs once at WARNING with the remedy, never a bare `continue`.
  Dispatched workers get `HERMES_KANBAN_BOARD` and the assignee's `HERMES_HOME` pinned in a
  scrubbed child env (`build_subprocess_env` + `strip_launch_profile_env`); they never inherit the
  default profile's `.env`.
- **A card's forced skills are validated against the ASSIGNEE's profile, never the creator's.**
  `kanban_db_skills.check_forced_skills` binds the assignee's home + secret scope for the scan
  (one resolver; `create_task` calls it for every surface, the dispatcher again at the spawn
  boundary once the workspace — hence project-local skills — is known). Force-loading is
  fail-soft downstream: `build_preloaded_skills_prompt` reports an unresolvable name *missing*
  and runs anyway, so an unvalidated typo costs a whole worker run in the wrong context.
  Creation-time checking fails OPEN (no profile dir, scan raised); the dispatcher backstop
  `block_task`s instead of `_record_task_failure` — a malformed card is not a flaky one and must
  not spend the breaker's budget or read as "gave up after N failures".
- **`crashed`/`timed_out` are emitted BEFORE the breaker is accounted** (reclaim txn writes the
  event; `_account_crashes` decides after), so they cannot know whether a retry follows: their copy
  states the condition, never a guarantee, and terminal `gave_up` names its real `trigger_outcome`.
  Gateway notifier and TUI poller both render from `gateway/kanban_watchers_common.py`
  (`CRASH_RETRY_NOTE`, `gave_up_cause`) — the TUI copy had drifted to blaming every trip on spawns.
- **Descendant fence is a path, not a flag.** A delegated child's Kanban marker
  (`agent/delegation_context.py::DELEGATED_CHILD_ENV_MARKER`) carries the fenced board ROOT;
  `kanban_path_is_fenced(path)` denies mutations only on the dispatcher-pinned `HERMES_KANBAN_DB`
  or under that root, so a child working against a scratch `HERMES_HOME` keeps a writable board.

## Tests

`tests/cron/`, `tests/hermes_cli/test_kanban*.py`, `tests/tools/test_kanban*.py`. Schedule parsing
and catch-up windows are pure functions — test them as data. Never assert on the verb list or
toolset size (root: no change-detectors). Time-based tests use loose bounds (≥ 2s) and event sync.
