# Workflow-Engine Hermes Plugin — Refactor Plan

Target repo: `Interstellar-code/hermes-agent` (working tree at `~/.hermes/hermes-agent/`)
Target path: `plugins/workflow-engine/`
Reference plugin: `plugins/kanban/`
Out of scope: `hermes-switchui` (TS engine, frontend toggle), upstream `PluginContext` API changes.

---

## 1. Requirements Summary

The workflow-engine plugin currently conflates Hermes' two distinct plugin contracts into a single broken `register(host)` and depends on capabilities the plugin loader does not provide (`include_router`, running event loop). Refactor so:

- Dashboard HTTP surface is mounted **only** by the dashboard plugin loader (`_mount_plugin_api_routes` in `hermes_cli/web_server.py:4340`).
- Agent-side surface registers **only** what `PluginContext` supports: `register_tool`, `register_hook`, `register_command`, `register_cli_command`.
- Engine is constructed **exactly once** per process and shared between the dashboard router and any agent tools.
- Background schedulers (`CronPoller`, `KanbanDispatcher`) own a real lifecycle — not `asyncio.create_task` at synchronous register time.
- The `sys.path` hack survives in **at most one location**; nothing in the agent-side entry point pollutes import paths.

## 2. Decision: Background-Task Pattern

Two patterns considered for `CronPoller` + `KanbanDispatcher`:

### Option 1 — Embedded in gateway via config gate

Mirror `kanban.dispatch_in_gateway` precedent: add a `_workflow_watcher` task in `gateway/run.py` (around line 3813 next to `_kanban_dispatcher_watcher`), gated by `workflow.dispatch_in_gateway` (default `true`).

- ✅ Same UX as kanban, single gateway process owns lifecycle, clean shutdown via existing teardown.
- ❌ Modifies Hermes core (`gateway/run.py`, `hermes_cli/config.py`). Out of scope per refactor brief ("we adapt to Hermes, not extend Hermes").

### Option 2 — Standalone CLI daemon via `register_cli_command` (RECOMMENDED)

Register a `hermes workflow daemon` CLI command via `ctx.register_cli_command()` in `__init__.py`. Provide a systemd unit at `plugins/workflow-engine/systemd/hermes-workflow-dispatcher.service`. The daemon owns its own `asyncio.run()`, so it has a real loop and clean signal handling — exactly the pattern `hermes kanban daemon` followed before the embedded migration.

- ✅ Zero Hermes core changes. Plugin-local lifecycle. Trivial rollback (`systemctl disable`).
- ✅ Re-uses existing `CronPoller.run_forever()` / `KanbanDispatcher.run_forever()` — they already expect to be awaited.
- ⚠️ Requires user to install the systemd unit manually (one-time, documented in `after-install.md`).
- ⚠️ Two processes (gateway + workflow daemon) — fine for now, mirrors kanban's pre-embed era.

### Option 3 — Hybrid (FOLLOW-UP)

Ship Option 2 now. File a follow-up upstream PR to add `workflow.dispatch_in_gateway` mirroring kanban. The CLI daemon then becomes the documented escape hatch (as kanban's systemd unit is today).

**Decision: Option 2 (+ Option 3 as follow-up).**

## 3. Architecture After Refactor

```
plugins/workflow-engine/
├── __init__.py                     # Agent-side: register_tool x5, register_cli_command x1, NO router, NO asyncio.create_task
├── plugin.yaml                     # unchanged
├── README.md                       # update to describe split + daemon install
├── after-install.md                # add systemd install instructions
├── _shared.py                      # NEW: single engine factory + sys.path bootstrap (called by both __init__ and dashboard)
├── daemon.py                       # NEW: `hermes workflow daemon` entrypoint — owns asyncio.run + signal handlers
├── tools/                          # NEW: agent tool handlers (one file per tool, thin wrappers around engine)
│   ├── __init__.py
│   ├── list_workflows.py
│   ├── run_workflow.py
│   ├── workflow_status.py
│   ├── approve_workflow.py
│   └── cancel_workflow.py
├── dashboard/
│   ├── manifest.json               # unchanged (already wires "api": "plugin_api.py")
│   └── plugin_api.py               # router only; imports engine from _shared
├── systemd/                        # NEW
│   └── hermes-workflow-dispatcher.service
├── defaults/                       # unchanged
├── engine/                         # unchanged (kept as top-level package; only _shared.py mutates sys.path)
└── tests/                          # unchanged path; conftest absorbs sys.path setup
```

Key invariants:

- `__init__.py` never imports `from engine.*` and never touches `sys.path`. It defers everything to `_shared.get_engine()` lazily inside tool handlers.
- `dashboard/plugin_api.py` imports `from ._shared import get_engine` only.
- `daemon.py` imports `from _shared import get_engine` and constructs `CronPoller(engine)` + `KanbanDispatcher(engine)` inside `asyncio.run(main())`.
- Only `_shared.py` performs the sys.path injection — once, idempotent, guarded.

## 4. Phased Implementation

Each phase is independently testable. Run `pytest plugins/workflow-engine/tests/` after every phase. Run `hermes gateway restart` and tail `~/.hermes/logs/agent.log` for `WARNING workflow.plugin` to verify warnings disappear.

### Phase 0 — Branch + baseline

- Cut branch `refactor/workflow-plugin-contract` from `main` in `hermes-agent` repo.
- Capture baseline: `pytest plugins/workflow-engine/tests/ -x` count, gateway-restart agent.log workflow lines.
- **Risk**: none. **Rollback**: discard branch.

### Phase 1 — Extract shared engine bootstrap

**Files**: new `plugins/workflow-engine/_shared.py`; modify `dashboard/plugin_api.py`.

`_shared.py` contents:

```python
"""Shared engine factory — single sys.path injection, single create_engine call."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from engine import WorkflowEngine, create_engine  # noqa: E402

_engine: Optional[WorkflowEngine] = None

def get_engine() -> WorkflowEngine:
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine
```

Modify `dashboard/plugin_api.py` (line 23 area): remove the `_PLUGIN_DIR` sys.path block and `from engine import` line; replace with `from .._shared import get_engine` and lazy `_engine = get_engine()` at module top.

- **Acceptance**:
  - `python -c "import sys; sys.path.insert(0, 'plugins/workflow-engine'); from dashboard.plugin_api import _engine; print(_engine.__class__.__name__)"` prints `WorkflowEngine`.
  - All existing tests still pass.
- **Risk**: relative import `from .._shared` requires `dashboard` to be a package. Add `dashboard/__init__.py` if missing.
- **Rollback**: revert two files.

### Phase 2 — Strip __init__.py to a no-op skeleton

**Files**: rewrite `plugins/workflow-engine/__init__.py`.

> ⚠️ **Regression window — bundle Phases 2 + 3 + 4 in a single PR**. Today's broken `__init__.py:48,:57` *attempts* `asyncio.create_task` (and fails); the no-op stub drops that attempt entirely. If Phase 2 ships alone, schedulers are silently absent until Phase 4 adds `hermes workflow daemon`. **Rule**: Phases 2, 3, and 4 land together in one merged PR. Internal commit boundaries still mirror the phase split for review clarity, but no intermediate state is ever deployed.

New body (no register in this phase; tools wired in Phase 3, schedulers picked up in Phase 4, all in same PR):

```python
"""workflow-engine plugin.

Dashboard router: dashboard/plugin_api.py (auto-mounted by web_server.py).
Agent-side tools/hooks/CLI: registered in register() below.
Background scheduler: hermes workflow daemon (see systemd/).
"""
from __future__ import annotations
import logging

logger = logging.getLogger("workflow.plugin")

def register(ctx) -> None:  # noqa: ANN001
    """Register agent-side surface with Hermes PluginContext.

    Phase 2 stub: no-op. Tools wired in Phase 3, CLI command in Phase 4.
    """
    logger.info("workflow-engine plugin loaded (agent-side stub; dashboard auto-mounts router)")

def disable() -> None:
    pass
```

- **Acceptance**:
  - `hermes gateway restart` → `grep -i 'workflow' ~/.hermes/logs/agent.log | tail` shows the new INFO line and no `lacks include_router` or `no running asyncio loop` warnings.
  - Dashboard still serves `/api/plugins/workflow-engine/*` (returns 401 JSON, not 404 HTML) — confirms the dashboard loader path is independent.
  - Browser /workflows page still works.
- **Risk**: dashboard plugin loader may also try to import the root `__init__.py` and trip on something; unlikely but verify. **Rollback**: restore old `__init__.py`.

#### Phase 1/2 test-import audit (do before Phase 2 merges)

The repo's `plugins/workflow-engine/tests/conftest.py` (currently at lines ~21 and ~39 per Codex review) injects the plugin dir into `sys.path` and aliases the package as `plugins.workflow_engine`. Files such as `tests/test_api_definitions.py:10` import top-level `engine.wiring` directly. After Phase 1 removes the sys.path block from `dashboard/plugin_api.py`, run:

```bash
pytest plugins/workflow-engine/tests/ -x -q
```

If any test fails on `ModuleNotFoundError: engine`, update `conftest.py` to ensure `_shared.py` is imported once at session-start (which performs the single sys.path injection). Do not push the path responsibility into individual test files.

### Phase 3 — Register agent tools

**Files**: new `plugins/workflow-engine/tools/__init__.py` + 5 tool modules; modify `__init__.py`.

Each tool handler is a thin sync or async function. Use `is_async=True` where the engine call awaits I/O. Tool schemas follow the JSON Schema shape used elsewhere in `tools.registry` (see `tools/kanban_tools.py` as reference; do not copy from this plan — read the file).

Tools to register (toolset: `workflow`):

| Name | Args | Returns | Engine call |
|---|---|---|---|
| `workflow_list` | `tags?:string[], source?:string` | list of `{id,label,source}` | `engine.list_definitions(...)` |
| `workflow_run` | `id:string, working_path?:string, inputs?:object` | `{run_id,status}` | `engine.start_run(...)` |
| `workflow_status` | `run_id:string` | `{status, current_node, events:[...]}` (events capped 50) | `engine.get_run(...) + listRecentWorkflowEvents` |
| `workflow_approve` | `run_id:string, node_id:string, decision:"approve"\|"reject", note?:string` | `{ok:bool}` | `engine.approve_node(...)` |
| `workflow_cancel` | `run_id:string, reason?:string` | `{ok:bool}` | `engine.cancel_run(...)` |

In `__init__.py.register()`:

```python
from ._shared import get_engine
from .tools.list_workflows import handler as list_handler, SCHEMA as list_schema, check as list_check
# ... etc, lazy imports inside register to keep import-time cheap
for name, schema, handler, is_async, check_fn in (
    ("workflow_list",    list_schema,    list_handler,    True, list_check),
    ("workflow_run",     run_schema,     run_handler,     True, run_check),
    ("workflow_status",  status_schema,  status_handler,  True, status_check),
    ("workflow_approve", approve_schema, approve_handler, True, mutating_check),
    ("workflow_cancel",  cancel_schema,  cancel_handler,  True, mutating_check),
):
    ctx.register_tool(
        name=name, toolset="workflow", schema=schema, handler=handler,
        check_fn=check_fn,
        is_async=is_async, description=schema.get("description",""), emoji="🔁",
    )
```

#### Phase 3 — Auth and scoping (was Codex BLOCKER)

Mutating tools (`workflow_run`, `workflow_approve`, `workflow_cancel`) MUST NOT ship without a `check_fn` gate. The dashboard router validates `working_path` (`dashboard/plugin_api.py:225` rejects relative / `..`) and approval can resume a terminated run (`:317`) — neither is enforced at tool-dispatch time. Without gating, any authenticated agent session can flip terminal state, point a run at an arbitrary working_path, or hijack a conversation_id.

Required check functions (modeled on `tools/kanban_tools.py:46,76,1081`):

| Tool | check_fn enforces |
|---|---|
| `workflow_list` | always allow (read-only) |
| `workflow_status` | always allow (read-only) |
| `workflow_run` | (a) `working_path` resolved against an allowlist root (`config.workflow.allowed_roots`, default `~/.hermes` + cwd); reject `..`, symlinks escaping the root, and absolute paths outside the allowlist. (b) `conversation_id` either omitted (engine assigns) or matches the current session's conversation id. (c) per-session run-rate cap (`workflow.run_rate_per_session`, default 5/min). |
| `workflow_approve` | (a) caller's session must own the run (`engine.get_run(run_id).owner_session == ctx.session_key`) OR caller has `workflow.approve_any` capability flag in config. (b) reject if run is in terminal state (`completed`/`failed`/`cancelled`) — no resurrecting finished runs. |
| `workflow_cancel` | same ownership check as approve; allow on non-terminal states only. |

Add new config keys (additive, non-breaking):

```yaml
workflow:
  allowed_roots: ["~", "${HERMES_HOME}"]   # working_path must resolve under one of these
  run_rate_per_session: 5                  # per-minute cap on workflow_run
  approve_any: false                       # if true, anyone can approve any run (dev only)
```

Update Section 8 rollback note: this DOES add config keys. Defaults are restrictive (run_rate=5, approve_any=false) so existing installs gain protection without manual config.

- **Acceptance**:
  - `hermes gateway restart` → log shows `Plugin workflow-engine registered tool: workflow_list` (etc, debug level — bump to verbose to see).
  - From a Telegram/CLI agent session: `hermes` agent can call `workflow_list` and `workflow_run` and see results.
  - Negative test: agent in session A cannot `workflow_approve` a run started by session B (unless `approve_any=true`).
  - Negative test: `workflow_run` with `working_path: /etc` returns a permission error, not a 500.
  - Existing tests pass; add `tests/test_agent_tools.py` covering schema validation + happy-path handler dispatch + each check_fn rejection path using a fake engine + fake session_key fixture.
- **Risk**: tool schema mismatch → registry rejects. Validate against `tools.registry.register` signature before phase end. Tighter risk: ownership check requires `WorkflowRun` to carry an `owner_session` column — verify the engine schema. If absent, add a Phase 3a migration (column with NULL default; tools refuse to approve runs with NULL owner unless `approve_any=true`).
- **Rollback**: revert `__init__.py` to Phase 2 stub. Config keys are read with `.get(..., default)` so removing them leaves callers on the secure default.

### Phase 4 — Standalone daemon + CLI command

**Files**: new `plugins/workflow-engine/daemon.py`; new `plugins/workflow-engine/systemd/hermes-workflow-dispatcher.service`; modify `__init__.py`.

`daemon.py` skeleton:

```python
"""hermes workflow daemon — runs CronPoller + KanbanDispatcher."""
from __future__ import annotations
import argparse, asyncio, logging, signal, sys
from ._shared import get_engine
from engine.cron.poller import CronPoller
from engine.dispatcher.kanban import KanbanDispatcher

log = logging.getLogger("workflow.daemon")

async def _main(args) -> int:
    engine = get_engine()
    poller = CronPoller(engine, interval_s=args.interval)
    dispatcher = KanbanDispatcher(engine, interval_s=args.interval)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    tasks = [asyncio.create_task(poller.run_forever()),
             asyncio.create_task(dispatcher.run_forever())]
    await stop.wait()
    for t in tasks: t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    return 0

def _setup(sub) -> None:
    sub.add_argument("--interval", type=float, default=60.0)
    sub.add_argument("--pidfile", default=None)
    sub.set_defaults(func=lambda ns: sys.exit(asyncio.run(_main(ns))))
```

In `__init__.py.register()`:

```python
from .daemon import _setup as daemon_setup
ctx.register_cli_command(
    name="workflow",
    help="Workflow engine subcommands",
    setup_fn=daemon_setup,  # adds `daemon` subcommand
    description="Run the workflow scheduler (cron + kanban dispatcher).",
)
```

`systemd/hermes-workflow-dispatcher.service`: copy `plugins/kanban/systemd/hermes-kanban-dispatcher.service` verbatim, swap names + `ExecStart=hermes workflow daemon --interval 60 --pidfile %t/hermes-workflow-dispatcher.pid`. Mark with same DEPRECATED warning header pointing at the future gateway-embedded mode.

`after-install.md` addendum (THREE OS targets, not just Linux):

- **Linux (systemd)**: `cp systemd/hermes-workflow-dispatcher.service ~/.config/systemd/user/ && systemctl --user enable --now hermes-workflow-dispatcher.service`.
- **macOS (launchd)**: ship `launchd/ai.hermes.workflow-dispatcher.plist` alongside the systemd unit. Install: `cp launchd/ai.hermes.workflow-dispatcher.plist ~/Library/LaunchAgents/ && launchctl load -w ~/Library/LaunchAgents/ai.hermes.workflow-dispatcher.plist`. The plist mirrors the kanban dispatcher pattern (KeepAlive true, StandardOutPath / StandardErrorPath to `~/.hermes/logs/`). Find an existing macOS launchd plist in hermes-agent for the template (e.g. `~/.hermes/launchd/` or search the repo) and copy its structure.
- **Foreground / dev**: `hermes workflow daemon --interval 30` — sufficient for laptops where no supervisor is wanted. Document the lack of auto-restart explicitly.

> **Codex note**: a crashed daemon on a dev macOS box with no plist installed has no restart path. The plan accepts this: dev mode is foreground; production is launchd or systemd. README must warn users not to assume background dispatch is running if they haven't installed a supervisor.

#### CLI vs slash namespace (do before Phase 4 merges)

`engine/nodes/approval.py:44` and `engine/nodes/loop.py:166` emit `/workflow approve|reject` slash-style instructions for chat users. These live in the **slash command** namespace, not the **CLI** namespace. `ctx.register_cli_command(name="workflow", ...)` claims the `hermes workflow ...` terminal namespace — distinct surface, no actual collision. Still document the distinction in README to prevent future confusion. If someone later wants to `ctx.register_command(name="workflow", ...)` (slash command), that WOULD collide and must be rejected; add a guard or rename early.

- **Acceptance**:
  - `hermes workflow daemon --interval 5` runs in foreground, logs `cron poller tick` / `kanban dispatcher tick` every 5s, exits cleanly on Ctrl-C.
  - With daemon running, a kanban task tagged with a workflow trigger fires a run within 60s.
  - `systemd-analyze verify systemd/hermes-workflow-dispatcher.service` passes.
- **Risk**: signal-handler race on macOS. Mitigate by `loop.add_signal_handler` guarded inside `try/except NotImplementedError` (Windows fallback to `signal.signal`).
- **Rollback**: remove daemon.py + systemd unit; drop the `register_cli_command` line.

### Phase 5 — Tests + cleanup

**Files**: `tests/conftest.py` (verify sys.path bootstrap still works without root `__init__.py` injection); new `tests/test_register_contract.py`.

New test: import `plugins.workflow_engine` (use a fake ctx) and assert `register(ctx)` registered exactly 5 tools, 0 hooks, 1 CLI command, and did not call `ctx.include_router` (the fake ctx will not have it; absence of AttributeError = pass).

- **Acceptance**: full `pytest plugins/workflow-engine/tests/` green, including new contract test.
- **Risk**: low. **Rollback**: revert tests only.

### Phase 6 — Documentation + PR

- Update `plugins/workflow-engine/README.md`: architecture section describing the split (dashboard router vs agent tools vs daemon).
- PR description references this plan, lists the 5 new agent tools, calls out follow-up Option 3 (gateway-embedded `workflow.dispatch_in_gateway` for a future PR).

## 5. Risks (Cross-Phase) and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Relative-import refactor breaks tests that do `from engine.X` | High | Keep `engine/` top-level. Centralize sys.path injection in `_shared.py`. Tests import via existing conftest. |
| `register_cli_command` collides with another plugin's `workflow` | Low | Search Hermes for `name="workflow"` before claiming. Fall back to `wfengine` if conflict. |
| Daemon runs twice (user enables systemd AND a future embedded mode) | Med (future) | Document at top of service unit + check `~/.hermes/dispatch_in_gateway` style guard before starting. |
| Engine constructed twice if `_shared.get_engine()` imported from two threads simultaneously at import time | Low | Use `threading.Lock` inside `get_engine`. |
| Cross-process SQLite schema race: dashboard process + workflow daemon process both call `engine/wiring.py:60` → `ensure_schema()` (`engine/db/migrate.py:64`) without a cross-process lock. First-boot or after-upgrade race can double-apply pending migrations or partially write. | Med | (a) Wrap `ensure_schema()` body in `sqlite3` `BEGIN EXCLUSIVE` against the migrations table. (b) Add an OS file lock (`fcntl.flock` on `~/.hermes/switchui-workflows.db.migrate.lock`) acquired around schema check + apply. (c) Make `ensure_schema()` idempotent on partial state. Implement as a Phase 1.5 preflight patch in `engine/db/migrate.py` before any Phase 4 daemon ships. |
| Frontend toggle expects engine to be available via the dashboard endpoint immediately | Already shipped; the dashboard router is independent of this refactor; verified Phase 2 |
| Daemon crashes on dev macOS without launchd plist installed → silent stop | Med | README + after-install.md must call this out explicitly; recommend foreground mode for dev, plist for "always-on" macOS users. |

## 6. Acceptance Criteria (End-State)

1. `~/.hermes/logs/agent.log` contains no `workflow.plugin` `WARNING` lines after `hermes gateway restart`.
2. `/api/plugins/workflow-engine/*` returns 401 JSON (or 200 with a session token) — not 404 SPA HTML.
3. `hermes` agent can call `workflow_list`, `workflow_run`, `workflow_status`, `workflow_approve`, `workflow_cancel` from a chat session.
4. `hermes workflow daemon --interval 5` runs, ticks both poller and dispatcher, exits cleanly on SIGINT.
5. `systemctl --user enable --now hermes-workflow-dispatcher.service` (on Linux) brings up the dispatcher.
6. `pytest plugins/workflow-engine/tests/` is green.
7. `grep -r "include_router" plugins/workflow-engine/__init__.py` returns nothing.
8. `grep -r "sys.path.insert" plugins/workflow-engine/` matches **only** `_shared.py`.

## 7. Verification Steps (Run in Order)

```bash
# baseline
cd ~/.hermes/hermes-agent
git checkout -b refactor/workflow-plugin-contract
pytest plugins/workflow-engine/tests/ -x

# after each phase
pytest plugins/workflow-engine/tests/ -x
hermes gateway restart
sleep 5
grep -i workflow ~/.hermes/logs/agent.log | tail -20
python3 -c "import urllib.request as r; print(r.urlopen('http://127.0.0.1:9119/api/plugins/workflow-engine/health',timeout=3).status)" 2>&1 | tail -5

# after Phase 4
hermes workflow daemon --interval 5 &
DAEMON_PID=$!
sleep 12
kill -INT $DAEMON_PID
wait $DAEMON_PID  # expect exit 0
```

## 8. Rollback Path

Each phase reverts independently (file list above). Full rollback = `git reset --hard main` on the refactor branch.

**DB schema is one-way.** `engine/wiring.py:60` calls `engine.db.migrate.ensure_schema()` on every engine boot (`engine/db/migrate.py:64,:69`). Any pending migrations apply on first daemon or dashboard startup after this refactor. Rolling back the **code** to pre-refactor does NOT undo the schema. Acceptable because:

- New migrations introduced by this refactor (if any — verify with `git diff main -- plugins/workflow-engine/engine/db/migrations/`) are additive (new columns / new tables with NULL-safe defaults). Old code continues to work against the new schema.
- If a destructive migration sneaks in, ship a paired down-migration before merging. Verify via `pytest plugins/workflow-engine/tests/test_db_compat.py` against a fresh DB on the refactor branch AND against a DB that already ran the new schema.

**Config keys ARE added** (`workflow.allowed_roots`, `workflow.run_rate_per_session`, `workflow.approve_any`). All have safe defaults and are read with `.get(..., default)`, so removing them in a rollback returns to the secure-by-default behavior; no config-file edits required to roll back.

No removed public surface in the dashboard router. `/api/plugins/workflow-engine/*` HTTP contract is preserved across the refactor — the switchui frontend can stay on the same URLs throughout.

## 9. Follow-Up (Out of Scope for This PR)

1. Upstream PR: add `workflow.dispatch_in_gateway` gate + `_workflow_watcher` task in `gateway/run.py` (mirrors kanban). Marks the systemd unit DEPRECATED.
2. Restore `engine/` as `wf_engine/` (proper package name) to retire the sys.path injection entirely. Bundle a `pyproject.toml` so tests import via the installed package.
3. Agent-callable `workflow_define` tool (upload YAML from chat) once auth scoping for plugin tools is sorted.
