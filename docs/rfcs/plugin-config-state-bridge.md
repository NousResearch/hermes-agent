# Plugin Config & State Bridge — Design Proposal

**Branch:** `feat/plugin-config-state-bridge`
**Target:** `hermes_cli/plugins.py` (PluginContext), `hermes_cli/config.py`, `cron/scheduler.py`
**Concrete consumer:** kanban-advanced plugin (config overlay, cron provisioning, dashboard)

---

## Summary

Today plugins that need to read/write Hermes configuration or manage cron jobs
must shell out to CLI commands (`hermes config set`, `hermes cron create`) or
manipulate YAML/config files directly. This is fragile across platform
differences (Windows path separators, MSYS vs native Python), Hermes version
bumps, and concurrent access.

This proposal adds four capabilities to the `PluginContext` API so plugins can
integrate with Hermes' config and cron systems through stable, typed interfaces
without reaching into core internals.

---

## Proposal 1: `ctx.get_config()` / `ctx.set_config()`

### Current state
Plugins read `config.yaml` via `hermes_cli.config.load_config()`, parse it
manually, and write back via direct file manipulation. This bypasses Hermes'
own config manager — no schema validation, no atomic writes, no migration
compatibility.

### Proposed API

```python
class PluginContext:
    def get_config(self, key: str, default: Any = None) -> Any:
        """Read a config value by dotted key (e.g. 'kanban.dispatch_stale_timeout_seconds').

        Returns the default if the key is unset or the config file is missing.
        Reads through Hermes' own config loader so layered configs (env overrides,
        profile-specific merges) are respected.
        """

    def set_config(self, key: str, value: Any) -> None:
        """Write a config value by dotted key.

        Writes through Hermes' config manager — atomic file write, schema
        validation for known keys, migration compatibility. Raises ValueError
        if the key path is invalid or the value fails schema validation.
        """
```

### Implementation sketch

```python
# hermes_cli/plugins.py — PluginContext additions
def get_config(self, key: str, default: Any = None) -> Any:
    from hermes_cli.config import load_config, cfg_get
    try:
        config = load_config()
        return cfg_get(config, key, default=default)
    except Exception:
        return default

def set_config(self, key: str, value: Any) -> None:
    from hermes_cli.config import load_config, save_config, cfg_set
    config = load_config() or {}
    cfg_set(config, key, value)
    save_config(config)  # handles atomic write + schema validation
```

### Concrete use case (kanban-advanced)
Our `config_overlay.py` currently does:
```python
# Current: fragile YAML manipulation
config = yaml.safe_load(Path(config_path).read_text())
config["kanban"]["dispatch_stale_timeout_seconds"] = 14400
Path(config_path).write_text(yaml.dump(config))
```

With this API:
```python
ctx.set_config("kanban.dispatch_stale_timeout_seconds", 14400)
ctx.set_config("kanban.auto_decompose", False)
```

### Platform safety
On Windows, our config writes hit `re.sub` backslash escape bugs because
paths like `C:\Users\Owner` contain `\U` which Python interprets as a
Unicode escape in replacement strings. Going through Hermes' own config
manager eliminates this class of bug.

---

## Proposal 2: `ctx.register_config_schema()`

### Current state
Plugins with configuration (like kanban-advanced's `kanban-config.yaml`) have
JSON Schemas that are invisible to `hermes doctor`, `hermes config check`, and
the setup wizard. Users only discover configuration errors at runtime.

### Proposed API

```python
class PluginContext:
    def register_config_schema(self, schema: dict) -> None:
        """Register a JSON Schema for this plugin's config namespace.

        The schema validates keys under plugins.entries.<plugin_id>.config.
        After registration, `hermes config check` validates the plugin's
        config against this schema, and `hermes setup` can walk plugin
        config interactively.
        """
```

### Manifest support (plugin.yaml)

```yaml
# plugin.yaml — optional schema declaration
provides_config_schema: schema/kanban-config.schema.json
```

When present, the schema is loaded and registered automatically during plugin
init — no explicit `register_config_schema()` call needed in `__init__.py`.

### Concrete use case (kanban-advanced)
Our kanban-config has 30+ keys with validation rules. Currently validation is
done in our own scripts only. With this, `hermes doctor` would catch
misconfigurations before they cause runtime failures.

---

## Proposal 3: `ctx.cron` — Cron API access

### Current state
Plugins that manage cron jobs (kanban-advanced provisions 5+ crons for
auto_unblock, board_keeper, lifecycle, dashboard keepalive) must shell out to
`hermes cron create/list/remove` via `subprocess.run()`. This is fragile:
- Platform-dependent (bash vs cmd, path resolution)
- No structured error handling
- No idempotency guarantees
- Cannot inspect job state programmatically

### Proposed API

```python
class PluginContext:
    @property
    def cron(self) -> "PluginCronFacade":
        """Return a facade for managing cron jobs owned by this plugin.

        All jobs created through this facade are tagged with the plugin's
        name, enabling bulk operations (list plugin jobs, remove all on
        uninstall).
        """
        if self._cron is None:
            from hermes_cli.plugins import PluginCronFacade
            self._cron = PluginCronFacade(
                plugin_id=self.manifest.key or self.manifest.name
            )
        return self._cron


class PluginCronFacade:
    def create(self, name: str, schedule: str, command: str, *,
               deliver: str = "local", skills: list[str] | None = None,
               idempotency_key: str | None = None) -> str:
        """Create a cron job. Returns the job ID.
        When idempotency_key is set, returns existing job ID if a match exists.
        """

    def list(self) -> list[dict]:
        """List all cron jobs owned by this plugin."""

    def remove(self, job_id: str) -> bool:
        """Remove a cron job by ID."""

    def get(self, job_id: str) -> dict | None:
        """Get a single job's details."""

    def pause(self, job_id: str) -> bool: ...
    def resume(self, job_id: str) -> bool: ...
```

### Implementation sketch

The `PluginCronFacade` delegates to the existing `CronManager` that powers
`hermes cron` CLI — same validation, same scheduler integration. It adds
plugin-namespacing via a `plugin_id` column on the jobs table (or a
`plugin_id` tag in the job metadata).

### Concrete use case (kanban-advanced)
Our `kanban_handoff.py` currently does:
```bash
hermes cron create "30s" --name "auto_unblock" \
  --script scripts/auto_unblock.sh --deliver local
```

With this API:
```python
ctx.cron.create(
    name="auto_unblock",
    schedule="30s",
    command="bash scripts/auto_unblock.sh",
    deliver="local",
    idempotency_key="kanban-advanced-auto_unblock",
)
```

---

## Proposal 4: Config defaults in `plugin.yaml`

### Current state
Plugins that need non-default Hermes config values (kanban-advanced needs
`dispatch_stale_timeout_seconds: 14400`, `auto_decompose: false`,
`BLOCK_RECURRENCE_LIMIT: 5`) must apply them via bootstrap scripts. If the
user forgets to run bootstrap, the system runs with unsafe defaults.

### Proposed manifest field

```yaml
# plugin.yaml
provides_config_defaults:
  kanban.dispatch_stale_timeout_seconds: 14400
  kanban.auto_decompose: false
```

On `hermes plugins install`, Hermes prompts the user with a diff of proposed
config changes. On `hermes plugins update`, new defaults merge in (existing
user overrides preserved). The prompt shows:
```
Plugin 'kanban-advanced' recommends these config changes:

  kanban.dispatch_stale_timeout_seconds: 900 → 14400
  kanban.auto_decompose: true → false

Apply? [Y/n]
```

### Design constraints
- **Never silently override user config** — always prompt
- **First-install vs update** — first install applies all defaults; update
  only applies NEW keys that don't exist in user config
- **Opt-out per key** — users can add keys to a `plugins.entries.<id>.config_defaults_skip`
  list to reject specific defaults permanently

---

## Cross-cutting concerns

### Why these belong in PluginContext and not as separate CLI commands

1. **Atomicity** — config writes go through the same config manager that
   handles migrations, schema validation, and concurrent access
2. **Platform safety** — bypasses shell-level path issues on Windows
3. **Plugin lifecycle** — cron jobs tagged by plugin can be bulk-removed on
   `hermes plugins remove`
4. **No new env vars** — follows the AGENTS.md rule: behavioral settings in
   config.yaml, not env vars

### Backward compatibility

All four additions are purely additive:
- Existing plugins that shell out to CLI continue to work
- New methods are opt-in
- Config schema registration doesn't affect existing config validation

---

## Related

- kanban-advanced planned features: `plugin/data/references/planned-features.md`
- Hermes AGENTS.md: "The core is a narrow waist; capability lives at the edges"
- Discord plugin interface expansion discussion
