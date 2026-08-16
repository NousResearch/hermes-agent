# Plugin Config & State Bridge

**Status:** config + state slice implemented by #64227

**Original design:** Topher Ross (@thebizfixer), RFC PR #58542

**Concrete consumer:** kanban-advanced

## Scope

This slice adds two native `PluginContext` capabilities:

- typed, namespace-jailed settings via `ctx.get_config()` and `ctx.set_config()`;
- atomic, profile-scoped runtime data via `ctx.state`.

Config schema registration, config defaults, and the cron facade from the
original RFC remain separate follow-up work. No core model tool is added.

## Config API

```python
def register(ctx):
    endpoint = ctx.get_config("api_url", default="https://example.invalid")
    retries = ctx.get_config("retry.attempts", default=3)

    ctx.set_config("api_url", "https://api.example.com")
    ctx.set_config("retry.attempts", 5)
```

Keys are **relative to the calling plugin**. The example above reads and writes:

```yaml
plugins:
  entries:
    <effective-plugin-id>:
      settings:
        api_url: https://api.example.com
        retry:
          attempts: 5
```

`<effective-plugin-id>` is `manifest.key` when present, otherwise
`manifest.name`. `settings` is the canonical namespace chosen after the issue
discussion in #64227/#67531. For migration safety, reads fall back to the former
`plugins.entries.<id>.config.*` subtree only when the canonical value is absent.
Writes always target `settings`; they do not rewrite or delete legacy values.

### Namespace jail

The API does not accept full config paths. A plugin can never use it to inspect
or change arbitrary Hermes configuration.

Accepted:

```python
ctx.get_config("endpoint")
ctx.set_config("retry.policy", {"attempts": 3})
```

Rejected with `ValueError` and a warning log:

```python
ctx.get_config("security.approval_mode")
ctx.set_config("model.provider", "attacker-proxy")
ctx.set_config("plugins.entries.other.settings.token", "...")
ctx.set_config("../../security.approval_mode", "always_allow")
ctx.set_config(r"..\..\model.provider", "attacker-proxy")
```

There is no global read allowlist: `ctx.profile_name` already exposes the only
small host fact requested by the RFC. Settings writes use Hermes'
profile-aware config loader/saver and atomic YAML replacement. The bridge
validates the existing YAML before writing so malformed config is never
silently replaced. Every operation resolves the active context-local
`HERMES_HOME`, so one globally loaded plugin context follows multiplexed
profile turns without crossing profile data.

## Durable state API

Use state for plugin-owned runtime data such as cursors, dedupe sets, and
caches. Do not put those values in user-owned config.

```python
def register(ctx):
    cursor = ctx.state.get("cursor", default={"page": 0})
    ctx.state.set("cursor", {"page": cursor["page"] + 1})
```

The facade stores one JSON object at:

```text
<HERMES_HOME>/plugin-data/<plugin-data-namespace>/state.json
```

Portable Agent Plugins use their existing `PLUGIN_DATA` namespace exactly.
Native and nested plugin ids use the same collision-resistant, Windows-safe
namespace algorithm. `ctx.state.data_dir` exposes the directory and
`ctx.state.path` exposes the JSON file when a plugin needs to inspect its own
location.

### State guarantees

- **Profile isolation:** the data root resolves from the active context-local
  Hermes home on every operation.
- **Atomic replacement:** state writes use temp-file + `fsync` + `os.replace`.
- **Concurrent updates:** a sibling lock file serializes read-modify-write across
  threads and processes (`fcntl` on POSIX, `msvcrt` on Windows).
- **Quota:** the complete serialized state is limited to 10 MiB per plugin. A
  rejected update leaves the previous file untouched.
- **Fail closed:** malformed/non-object JSON is reported and never overwritten.
- **Typed values:** values must be JSON-serializable.

State keys are 1–128 characters and may contain letters, numbers, `_`, `-`,
`.`, or `:`. Path separators and `..` are rejected.

## State vs. config

| Data | API | Ownership | Example |
|---|---|---|---|
| User-visible behavior | `ctx.get_config` / `ctx.set_config` | User/plugin settings in `config.yaml` | endpoint, timeout, feature mode |
| Runtime bookkeeping | `ctx.state.get` / `ctx.state.set` | Plugin data under `plugin-data/` | cursor, cache, dedupe ids |

Both APIs are additive. Existing plugins that perform their own file I/O keep
working, but new plugins should use this bridge for stable profile and Windows
semantics.

## Verification contract

The implementation is covered with real temporary-Hermes-home tests for:

- fixture-plugin discovery and config/state round trips;
- canonical `settings` writes and legacy `config` read fallback;
- direct global, cross-plugin, POSIX traversal, and Windows traversal rejection;
- concurrent settings writes without lost siblings;
- cross-thread and cross-process state updates;
- atomic quota rejection and malformed-state/config preservation;
- two-profile isolation after the ambient profile changes;
- Unicode and Windows-style path values.

## Related

- [Issue #64227](https://github.com/NousResearch/hermes-agent/issues/64227)
- [RFC PR #58542](https://github.com/NousResearch/hermes-agent/pull/58542) by Topher Ross
- #67531 — standalone plugin settings namespace discussion
