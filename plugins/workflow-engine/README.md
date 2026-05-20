# workflow-engine plugin

A DAG workflow engine for [hermes-agent](https://github.com/Interstellar-code/hermes-agent), ported from the Switch UI TypeScript implementation. Runs YAML-defined multi-node workflows with conditional branching, parallel execution, bash nodes, approval gates, and Kanban task dispatch.

## What it is

The workflow-engine plugin exposes a REST API that lets you define, trigger, monitor, and approve DAG-based workflows. Each workflow is a YAML file describing nodes (steps), their dependencies, providers (claude, codex, hermes-kanban), and conditional edges. The engine stores state in SQLite, emits SSE events for live progress, and integrates with the Hermes Kanban dispatcher for agent task routing.

## Install

The plugin ships bundled with hermes-agent. If you are running a standalone install:

```bash
hermes plugins install Interstellar-code/hermes-workflow-engine
```

## Enable

```bash
hermes plugins enable workflow-engine
hermes dashboard restart
```

Or set in your hermes config:

```yaml
plugins:
  workflow-engine:
    enabled: true
```

## Config

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `WORKFLOW_DB_PATH` | `~/.hermes/switchui/workflow-engine.db` | SQLite database path |
| `WORKFLOW_DEFAULTS_DIR` | `<plugin>/defaults/` | Directory of bundled default YAML files |
| `WORKFLOW_YAML_DIR` | `~/.hermes/switchui/workflows/` | User workflow YAML search path |
| `TOOL_CATALOG_ROOT` | (none) | Root path for tool-catalog-write workflow |
| `WORKFLOW_POLL_INTERVAL` | `60` | Cron poller interval in seconds |

**DB location:** `~/.hermes/switchui/workflow-engine.db` (SQLite, auto-migrated on startup).

**Default YAML dir:** YAML files in `plugins/workflow-engine/defaults/` are copied into the user's workflow store on first enable.

## API endpoints

All endpoints are mounted under `/api/workflows` (relative to the hermes-agent gateway, default `http://localhost:8642`).

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/workflows/health` | Plugin health check — returns `{"status":"ok"}` |
| `GET` | `/api/workflows/definitions` | List all workflow definitions |
| `POST` | `/api/workflows/definitions` | Register a new workflow definition (YAML body or `{slug, yaml}`) |
| `GET` | `/api/workflows/definitions/{def_id}` | Get a single definition by ID or slug |
| `GET` | `/api/workflows/definitions/{def_id}/parsed` | Get the parsed (validated) DAG structure for a definition |
| `DELETE` | `/api/workflows/definitions/{def_id}` | Delete a definition |
| `GET` | `/api/workflows/runs` | List workflow runs (filterable by status, definition) |
| `GET` | `/api/workflows/runs/active` | List currently active (running) workflow runs |
| `GET` | `/api/workflows/runs/by-conversation/{conv_id}` | List runs associated with a conversation ID |
| `POST` | `/api/workflows/runs` | Trigger a new workflow run |
| `GET` | `/api/workflows/runs/{run_id}` | Get a single run by ID |
| `GET` | `/api/workflows/runs/{run_id}/nodes` | List all node-runs for a workflow run |
| `GET` | `/api/workflows/runs/{run_id}/events` | List stored events for a run |
| `POST` | `/api/workflows/runs/{run_id}/events` | Append an event to a run (internal use) |
| `POST` | `/api/workflows/runs/{run_id}/approve` | Approve a paused approval-gate node |
| `POST` | `/api/workflows/runs/{run_id}/resume` | Resume a paused run |
| `POST` | `/api/workflows/runs/{run_id}/phase-transitions` | Record a phase transition (internal use) |
| `GET` | `/api/workflows/runs/{run_id}/phase-transitions` | List phase transitions for a run |
| `GET` | `/api/workflows/events` | SSE stream — subscribe to live workflow events |
| `GET` | `/api/workflows/node-runs/{node_run_id}` | Get a single node-run by ID |
| `POST` | `/api/workflows/runs/{run_id}/approval-claim` | Claim an approval gate (idempotent; used by UI) |

## Switch UI integration

The [hermes-switchui](https://github.com/Interstellar-code/hermes-switchui) UI exposes a **backend toggle** on the `/workflows` settings panel:

- **native** — uses the TypeScript workflow engine built into Switch UI (no plugin required).
- **plugin** — proxies all workflow API calls to this plugin via the hermes-agent gateway.

Toggle location: Settings → Workflows → Backend. The setting is persisted in `localStorage` and sent as `?backend=plugin` on all workflow API calls.

## Architecture (after refactor)

The plugin uses two distinct Hermes extension surfaces that must not be confused:

### 1. Dashboard router (HTTP)

`dashboard/plugin_api.py` exports a FastAPI `APIRouter`. The Hermes web server auto-mounts it via `_mount_plugin_api_routes` (no `include_router` call in `__init__.py`). All REST endpoints listed above are served here. The router imports from `_shared.get_engine()` — the same singleton used by agent tools.

### 2. Agent tools (5 tools)

`__init__.py:register(ctx)` registers 5 tools via `ctx.register_tool`. These are callable from any authenticated Hermes chat session:

| Tool | Description |
|------|-------------|
| `workflow_list` | List workflow definitions (read-only, always allowed) |
| `workflow_run` | Start a run (rate-capped, working_path allowlist) |
| `workflow_status` | Get run status + recent events (read-only) |
| `workflow_approve` | Approve/reject a paused approval node (ownership-gated) |
| `workflow_cancel` | Cancel an active run (ownership-gated) |

Auth gates (configured via `workflow.*` config keys):

```yaml
workflow:
  allowed_roots: ["~", "${HERMES_HOME}"]  # working_path must resolve here
  run_rate_per_session: 5                 # max workflow_run calls per minute per session
  approve_any: false                      # if true, any session may approve any run
```

### 3. Background scheduler (daemon process)

`CronPoller` and `KanbanDispatcher` run in a **separate process** — not inside the gateway. Start with:

```bash
hermes workflow daemon --interval 60
```

The daemon owns its own `asyncio.run()` loop and handles `SIGINT`/`SIGTERM` cleanly.

> **Note**: Without a supervisor installed, the daemon does **not** auto-restart on crash. Use foreground mode (`hermes workflow daemon`) for dev; install the systemd unit or launchd plist for always-on deployments.

### Key invariant: `_shared.py` is the only sys.path mutator

```
_shared.py          ← sys.path injection (once, idempotent, thread-safe)
  └─ get_engine()   ← singleton WorkflowEngine, shared across dashboard + tools

dashboard/plugin_api.py   ← imports from .._shared; never touches sys.path
__init__.py               ← imports from ._shared; never touches sys.path
daemon.py                 ← imports from ._shared; never touches sys.path
tools/*.py                ← import from .._shared inside handler functions
```

### Follow-up (out of scope for this PR)

A future upstream PR will add `workflow.dispatch_in_gateway` (mirroring `kanban.dispatch_in_gateway`) so the scheduler runs embedded in the gateway process. The systemd/launchd units will then be marked DEPRECATED.

## Cron integration

The plugin ships a built-in cron poller (`engine/cron/poller.py`). Workflows with a `cron:` field in their YAML are triggered automatically at the configured interval. The poller runs in the **workflow daemon process** (`hermes workflow daemon`) — not inside the hermes-agent gateway. See the Background scheduler section above.

```yaml
# Example: run every hour
name: my-workflow
cron: "0 * * * *"
nodes: ...
```

## Kanban integration

Nodes with `provider: hermes-kanban` (or aliases `claude`, `codex`) are dispatched as Kanban tasks. The Kanban dispatcher writes tasks to the hermes-agent Kanban DB; workers pick them up and execute. This integrates seamlessly with the existing Hermes worker pool.

## Bundled default workflows

See [`defaults/README.md`](defaults/README.md) for the list of workflows bundled with this plugin.
