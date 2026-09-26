# Cron and Kanban

The root guidance applies. Read `website/docs/developer-guide/cron-internals.md`, `website/docs/user-guide/features/cron.md`, and `website/docs/user-guide/features/kanban.md` for commands, fields, and architecture.

## Cron execution

`cron/jobs.py` owns the store. `cron/scheduler.py` and its topical siblings own ticking, occurrence recovery, supervision, and delivery.

- The inactivity watchdog measures idle time, not job wall time. Script execution has its own timeout. Preserve unlimited mode and active long-running jobs.
- Advance `next_run_at` and persist `pending_slot` before dispatch. `cron/occurrences.py` may restore a dead owner's slot once. The execution ledger's `scheduled_instant` prevents a second fire. A skipped missed occurrence records why.
- The per-home tick lock prevents two processes from ticking one profile store. All paths use the resolved profile home, never a literal home directory.
- One host ticker iterates every served profile sequentially under `_profile_cron_scope(home)`. The scope covers store open, lock selection, counters, environment construction, execution, and session-end flush. Do not create one ticker thread per profile.
- Cron ownership is independent of `gateway.multiplex_profiles`, which controls adapters rather than stores. Every served profile must have a ticker owner.
- All in-flight keys, pools, claims, and liveness checks include the home. Public host-wide summaries may return bare IDs, but decisions call `is_job_running(job_id, home=...)`.
- Release a claim with the exact home used to register it. A worker `finally` may run outside the copied context. Resolve normalized keys through the scheduler's home map rather than rebuilding a path from the key.
- Reclaim per-home pools when a home leaves the ticked set. Stand down for a profile that has its own live gateway. Compare the liveness PID with this process because the host publishes every served profile.
- Cron sessions use `skip_memory=True` and their own session. Mirroring into a reply-facing conversation is limited to eligible routes and appends a labeled user turn at a turn boundary.
- `HERMES_DESKTOP` means the backend was spawned by Desktop. It does not prove that a GUI watches the session.
- Process-local delegation does not survive restart. Durable scheduled work remains a cron job or a supervised background process with completion notification.

## Kanban boundary

Kanban is a durable SQLite work queue. The board is the hard isolation boundary. Tenant is a namespace inside a board. Workers receive a pinned `HERMES_KANBAN_BOARD` and the assignee's scrubbed `HERMES_HOME`.

- Worker tools live in the gated `kanban` toolset. Do not add them to the default schema.
- Dispatcher claims are atomic. Reclaim uses `(worker_pid, worker_started_at)`, never PID existence alone. Repeated non-success attempts auto-block. A terminal provider failure blocks on its first attempt.
- `kanban_db.connect` owns Kanban's connection and must not alias another subsystem's helper.
- Notifications leave through the task owner's profile adapter under that profile's scope. Missing delivery ownership fails closed and logs a remedy.
- Prompt injection sites use `agent/delegation_context.py::owned_kanban_task()`. Tool visibility or inherited task-shaped environment variables do not prove dispatcher ownership.
- A delegated-child marker carries the fenced board-root path. `kanban_path_is_fenced` blocks mutations only to the pinned board or its descendants, not unrelated scratch homes.
- Child environments come from `build_subprocess_env` and `strip_launch_profile_env`. They do not inherit the launch profile's credentials.

The CLI implementation is the `hermes_cli/kanban.py` facade with topical siblings. The dispatcher runs in the gateway by default. The standalone service remains an explicit deployment. Treat command and tool inventories as discoverable data rather than copying them here.

## Tests

Run `tests/cron/`, `tests/hermes_cli/test_kanban*.py`, and `tests/tools/test_kanban*.py` through `scripts/run_tests.sh`. Test schedule parsing and catch-up as data, profile-home separation with duplicate job IDs, claim release across context boundaries, ownership-gated prompt injection, and durable occurrence recovery. Use event synchronization and generous timing bounds.