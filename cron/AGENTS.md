# cron governance

Read repository-root `AGENTS.md` and its required routing table before acting.
These are supplemental current-lineage rules, not exceptions to global rules.
Read this file completely; automatic hints may truncate it. Scope follows the
behavior affected, including tests, docs and callers outside this directory.

## Cron (scheduled jobs)

`cron/jobs.py` (job store) + `cron/scheduler.py` (tick loop). Agents
schedule jobs via the `cronjob` tool; users via `hermes cron <verb>`
(`list`, `add`, `edit`, `pause`, `resume`, `run`, `remove`) or the
`/cron` slash command.

Supported schedule formats:
- Duration: `"30m"`, `"2h"`, `"1d"`
- "every" phrase: `"every 2h"`, `"every monday 9am"`
- 5-field cron expression: `"0 9 * * *"`
- ISO timestamp (one-shot): `"2026-06-01T09:00:00Z"`

Per-job fields include `skills` (load specific skills), `model` /
`provider` overrides, `script` (pre-run data-collection script whose
stdout is injected into the prompt; `no_agent=True` turns the script
into the entire job), `context_from` (chain job A's last output into
job B's prompt), `workdir` (run in a specific directory with its
`AGENTS.md`/`CLAUDE.md` loaded), and multi-platform delivery.

Hardening invariants:
- **3-minute hard interrupt** on cron sessions — runaway agent loops
  cannot monopolize the scheduler.
- Catchup window: half the job's period, clamped to 120s–2h.
- Grace window: 120s for one-shot jobs whose fire time was missed.
- File lock at `~/.hermes/cron/.tick.lock` prevents duplicate ticks
  across processes.
- Cron sessions pass `skip_memory=True` by default; memory providers
  intentionally do not run during cron.

Cron deliveries are **not** mirrored into the target gateway session —
they land in their own cron session with a header/footer frame so the
main conversation's message-role alternation stays intact.

---

## Kanban (multi-agent work queue)

Durable SQLite-backed board that lets multiple profiles / workers
collaborate on shared tasks. Users drive it via `hermes kanban <verb>`;
workers spawned by the dispatcher drive it via a dedicated `kanban_*`
toolset so their schema footprint is zero when they're not inside a
kanban task.

- **CLI:** `hermes_cli/kanban.py` wires `hermes kanban` with verbs
  `init`, `create`, `list` (alias `ls`), `show`, `assign`, `link`,
  `unlink`, `comment`, `attach`, `attachments`, `attach-rm`, `complete`,
  `request-review`, `request-changes`, `reopen-review`, `block`, `unblock`, `archive`,
  `tail`, plus less-commonly-used `watch`, `stats`, `runs`, `log`,
  `assignees`, `heartbeat`, `notify-*`, `dispatch`, `daemon`, `gc`.
- **Worker/orchestrator toolset:** `tools/kanban_tools.py` exposes
  `kanban_show`, `kanban_complete`, `kanban_request_review`,
  `kanban_request_changes`, `kanban_block`,
  `kanban_heartbeat`, `kanban_comment`, `kanban_create`, `kanban_link`,
  `kanban_attach`, `kanban_attach_url`, `kanban_attachments`; profiles that
  explicitly enable the `kanban` toolset outside a dispatcher-spawned
  task also get `kanban_list` and `kanban_unblock` for board routing.
- **Dispatcher:** long-lived loop that (default every 60s) reclaims
  stale claims, promotes ready tasks, atomically claims, and spawns
  assigned profiles. Runs **inside the gateway** by default via
  `kanban.dispatch_in_gateway: true`.
- **Plugin assets:** `plugins/kanban/dashboard/` (web UI) +
  `plugins/kanban/systemd/` (`hermes-kanban-dispatcher.service` for
  standalone dispatcher deployment).

Isolation model:
- **Board** is the hard boundary — workers are spawned with
  `HERMES_KANBAN_BOARD` pinned in their env so they can't see other
  boards.
- **Tenant** is a soft namespace *within* a board — one specialist
  fleet can serve multiple businesses with workspace-path + memory-key
  isolation.
- After `kanban.failure_limit` consecutive non-success attempts on the
  same task (default: 2), the dispatcher auto-blocks it to prevent spin
  loops.

Full user-facing docs: `website/docs/user-guide/features/kanban.md`.

---

<!-- End of relocated baseline guidance. -->
