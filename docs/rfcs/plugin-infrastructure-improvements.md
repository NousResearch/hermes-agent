# Plugin Infrastructure Improvements — Design Proposal

**Branch:** `feat/plugin-infrastructure-improvements`
**Target:** `hermes_cli/plugins.py`, `plugin.yaml` schema
**Concrete consumer:** kanban-advanced (dashboard sidecar, version gating, persistent state)

---

## Summary

Three infrastructure improvements that reduce plugin boilerplate and improve
the developer experience — driven by patterns observed across the plugin
ecosystem and specifically in kanban-advanced's architecture.

---

## Proposal 1: Plugin API backend support

### Current state (GHSA-5qr3-c538-wm9j workaround)

Non-bundled plugins cannot import Python API backends. This forces any plugin
with a dashboard or HTTP API to run a **standalone sidecar server** on a
separate port, managed by the plugin itself:
- PID file locking to prevent duplicate instances
- Keepalive crons to restart crashed sidecars
- Port conflict management
- Health check endpoints

kanban-advanced's dashboard runs as `scripts/dashboard_server.py` on
`127.0.0.1:18900` — completely outside Hermes' plugin infrastructure.

### Proposed: `api_backend` in `plugin.yaml`

```yaml
# plugin.yaml
kind: standalone
api_backend:
  enabled: false           # default: off for security
  module: dashboard.api    # Python module with a create_app() factory
  require_user_approval: true  # prompt on first install
```

When enabled (and approved by the user), Hermes imports the module and mounts
it at `/_plugin/<plugin_id>/` on the existing gateway HTTP server. The module
must expose:

```python
# plugin/dashboard/api.py
def create_app() -> "FastAPI | Flask | ASGI app":
    """Return an ASGI/WSGI application. Called once at gateway startup."""
```

### Benefits
- No separate port, no PID management, no keepalive crons
- Inherits Hermes' existing auth (if gateway has auth configured)
- Clean lifecycle: starts/stops with the gateway
- User must explicitly opt in via `hermes plugins enable --with-api <name>`

### Security
- `require_user_approval: true` (default) — prompts on first install
- API backend module is imported in a restricted context (no file system write access by default)
- `plugins.entries.<id>.api_backend.enabled` in config.yaml for persistent toggle

---

## Proposal 2: Plugin version & dependency declaration

### Current state

Plugin manifests have `version` and `requires_env` but no way to declare:
- Minimum Hermes version required
- Dependencies on other plugins
- Python package dependencies

We document "Hermes ≥ 0.16.0" in 8 different files. Version upgrade audits
are manual.

### Proposed manifest fields

```yaml
# plugin.yaml
version: 1.0.0
requires_hermes: ">=0.17.0"          # semver constraint
requires_plugins:                     # optional
  kanban: ">=0.1.0"
requires_python: ">=3.10"
requires_packages:                    # pip packages
  - "httpx>=0.24"
  - "pydantic>=2.0"
```

Behavior:
- **`requires_hermes`** — `hermes plugins install` warns on mismatch.
  `hermes doctor` flags it. Plugin still loads (forward compat) but with
  a warning in `hermes plugins list`.
- **`requires_plugins`** — install blocks if dependency not present.
  `hermes plugins install <name>` auto-installs missing deps with confirmation.
- **`requires_packages`** — `hermes plugins install` offers to pip install.
  Refused in gateway context (must be pre-installed).

---

## Proposal 3: `ctx.data_dir` — plugin-owned persistent state

### Current state

Plugins that need persistent state (logs, caches, databases) must resolve
paths manually using `HERMES_HOME` or `~/.hermes/`. Paths aren't namespaced
to the plugin, leading to collisions and cleanup difficulty.

kanban-advanced writes:
- `~/.hermes/logs/kanban/board_events.jsonl`
- `.hermes/kanban-overrides/kanban-config.yaml`
- `.hermes/kanban/preflight_cache.json`

### Proposed API

```python
class PluginContext:
    @property
    def data_dir(self) -> Path:
        """Return this plugin's persistent data directory.

        Returns ~/.hermes/plugins/<plugin_id>/data/ — created on first access.
        Guaranteed to exist and be writable. Cleaned up on plugin removal.
        """
        if self._data_dir is None:
            plugin_id = self.manifest.key or self.manifest.name
            safe_id = plugin_id.replace("/", "__")
            self._data_dir = get_hermes_home() / "plugins" / safe_id / "data"
            self._data_dir.mkdir(parents=True, exist_ok=True)
        return self._data_dir
```

### Benefits
- Automatic cleanup on `hermes plugins remove`
- No path collision between plugins
- Predictable location for debugging (`hermes plugins data-dir <name>`)

---

## Non-goals

- **Plugin sandboxing/containerization** — out of scope. Hermes plugins run
  with the same trust level as the agent.
- **Plugin marketplace/discovery** — out of scope. This is about the
  technical interface, not distribution.
- **Hot-reload of plugin code** — out of scope. Plugin code changes require
  a gateway restart (same as today).

---

## Related

- PR #58541 — kanban lifecycle hooks
- PR #58542 — plugin config & state bridge
- PR #58547 — context injection hooks
- PR #58548 — observability hooks
- GHSA-5qr3-c538-wm9j — current API backend restriction
