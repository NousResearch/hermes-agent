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
- **Inactivity watchdog** on cron agent sessions (`_cron_inactivity_seconds()`): default 600s idle,
  `HERMES_CRON_TIMEOUT` overrides, `0` = unlimited. It is idle time, not wall-clock — a stalled
  session is hard-interrupted so it cannot monopolise the scheduler, while a long-but-active job
  is never cut off. Attached scripts (pre-run or `no_agent`) are bounded separately by the script
  timeout (`_DEFAULT_SCRIPT_TIMEOUT`, 3600s).
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
- **Cron ownership is not gated on `gateway.multiplex_profiles`.** That flag gates ADAPTERS; one
  host gateway process ticks EVERY profile's store either way (`run.py::_cron_tick_profile_homes`).
  Gating the tick set on it left every non-launch profile's jobs in a store no ticker visited.
- **Per-profile process assumptions are the bug class.** One process ticks N homes, so anything
  keyed on "this process's profile" is wrong: in-flight state (`_running_job_ids`,
  `_running_since`, `_running_futures`, `_running_worker_pids`, `_running_fire_owners`,
  `_interrupted_job_ids`) is keyed by `_inflight_key(job_id)` = `(home key, job id)` — two
  profiles legitimately carry a `daily-brief`; the parallel pool is keyed by home
  (`cron.max_parallel_jobs` is per profile); and the stale-code yield gate asks
  `scheduler_ownership.owns_cron_tick_for(home)` / `live_gateway_ticking(home)` instead of the
  process-global runtime-lock boolean. Public accessors (`get_running_job_ids`,
  `get_running_job_details`, `get_wedged_job_ids`) still report the host-wide union of bare job
  ids for the shutdown drain, but LIVENESS consumers (`jobs.py::_job_running_in_this_process`,
  `tools/cronjob_tools`) ask `is_job_running(job_id, home=...)` — the union made profile A's
  running `daily-brief` answer for profile B's idle one.
- **A claim is released under the key it was registered with.** The cron scope is a ContextVar:
  `try_register_running_job` runs on the ticker thread inside `_profile_cron_scope`, while the
  pool worker's `finally` sits OUTSIDE `ctx.run` and resolves the LAUNCH home. Pass the
  registering home (`release_running_job(job_id, home=...)`), or every secondary profile's claim
  leaks — the job skips a fire window until the force-release backstop sweeps it, and the drain
  sees phantom work. Never rebuild a home from a key half (`Path(key[0])`): `hermes_home_key`
  normcases, so use `_inflight_home_path`.
- **Ticked-home state is reclaimed when a home leaves the set.** `register_ticked_homes` is
  republished every cycle and reaps the departed homes' parallel pools; pools used to live until
  `atexit`, so each home ever ticked kept a ThreadPoolExecutor and its worker threads forever.
- **The host gateway stands down for a profile that runs its OWN gateway.** `run.py::
  _cron_profile_gate` (the same gate `hermes_cli/web_server.py` passes) keeps the launch process
  and a per-profile gateway off one store: the tick lock stops a simultaneous double-run but not
  the race, and when the launch process wins, delivery goes through `SharedRouteAdapters`/
  fail-closed instead of that profile's live adapters. The gate compares the liveness PID against
  `os.getpid()` — this process holds the launch `gateway.pid` AND publishes every served profile
  in `served_profiles`, so a bare liveness answer would stand cron down host-wide.
- **The restart-safe external worker boots itself.** `_launch_external_cron_worker` pins the
  checkout — and, on a PM install, the committed generation's `site-packages` — on the child's
  `PYTHONPATH` and marks it `HERMES_CRON_EXTERNAL_WORKER`; the child's own entry
  (`cron/scheduler.py` → `cron/worker_bootstrap.py`) then runs PM's `activate_dependencies`,
  which leases that generation for the worker's lifetime and activates it before the first
  application import. Neither half is redundant (the pin is what the child can import through,
  the boot is what keeps the generation alive and its `.pth` files executed) and the gateway
  never re-runs the boot — `hermes_bootstrap` already did at its own launch (#122222).
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

- **CLI:** `hermes_cli/kanban.py` facade + 14 `kanban_*.py` siblings (`boards`, `db`, `db_connect`,
  `db_dispatch`, `db_notify`, `db_graph` (task initialization and decomposition), `workspace`, ...). Verbs: `init, create, list (ls), show, assign, link,
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
non-success attempts on a task (default 2) the dispatcher auto-blocks it to stop spin loops; a
worker exit of `KANBAN_TERMINAL_PROVIDER_EXIT_CODE` (78 — credential revoked, model gone; the
worker's own `failure_reason` classification via `cli._TERMINAL_PROVIDER_REASONS`) trips it on
the first attempt, sticky, because no retry can heal it (#114587).
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
- **Prompt injection sites gate on ownership, not tool access.** Tool access (`kanban_show` visible
  via a profile's toolset) and an inherited `HERMES_KANBAN_TASK` (delegate children, cron runs beside
  a worker) are not ownership. The kanban guidance (`agent_init`, `system_prompt` fallback) and the
  stop nudge resolve the task via `agent/delegation_context.py::owned_kanban_task()`; other readers
  pair their env read with `is_dispatcher_owned_worker_context()`.
- **Descendant fence is a path, not a flag.** A delegated child's Kanban marker
  (`agent/delegation_context.py::DELEGATED_CHILD_ENV_MARKER`) carries the fenced board ROOT;
  `kanban_path_is_fenced(path)` denies mutations only on the dispatcher-pinned `HERMES_KANBAN_DB`
  or under that root, so a child working against a scratch `HERMES_HOME` keeps a writable board.

## Tests

`tests/cron/`, `tests/hermes_cli/test_kanban*.py`, `tests/tools/test_kanban*.py`. Schedule parsing
and catch-up windows are pure functions — test them as data. Never assert on the verb list or
toolset size (root: no change-detectors). Time-based tests use loose bounds (≥ 2s) and event sync.
