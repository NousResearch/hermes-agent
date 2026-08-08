# Hermes Plugin Hook Taxonomy

This document is the canonical catalog of plugin lifecycle hooks: their
contract, return-value semantics, and naming grammar. It is the reference
used by the #64231 batch disposition of pending hook PRs and by plugin
authors deciding which hook to register.

Hook registration is additive and never breaks existing plugins. A plugin
registers callbacks from `register(ctx)`:

```python
def register(ctx):
    ctx.register_hook("footer", on_footer)
    ctx.register_hook("usage_extra", on_usage_extra)
```

Every hook callback receives keyword arguments. Plugins should accept
`**kwargs` so additive fields remain backward-compatible:

```python
def on_footer(**kwargs):
    model = kwargs.get("model")
    context_tokens = kwargs.get("context_tokens")
```

The plugin manager injects a schema-version field into every hook payload:

```text
telemetry_schema_version = "hermes.observer.v1"
```

Hook callbacks are fail-open. Hermes catches callback exceptions, logs a
warning, and keeps the agent loop running. Call sites short-circuit via
`has_hook(name)` so a hook with no registered callback costs nothing.

This document covers **observer / render-extension** hooks (the footer and
usage surfaces). Mutating-contract hooks (block signals, first-valid-wins
middleware) are covered by their own design notes and share the same
registration surface.

## Naming grammar

Hook names follow `<subsystem>_<noun>_<verb-past>` where possible, but the
two render-extension hooks below use short, discoverable ids (`footer`,
`usage_extra`) because they map 1:1 to a user-visible surface rather than a
subsystem event. Both are observer-only: they return a string block that the
call site appends; they never mutate the message or the runtime state.

## Render-extension hooks (observer)

These hooks let a plugin append an extra block to a rendered surface without
any core special-casing of a feature. Both are fail-open and short-circuited
when nothing is registered.

### `footer`

Fired after the runtime footer line is built, before it is appended to the
final response in `gateway/run.py`. A plugin returns a string block that is
appended after the footer line.

| Field | Type | Meaning |
| --- | --- | --- |
| `model` | `str \| None` | Bare model id (same value the footer renders). |
| `context_tokens` | `int` | Last-prompt token count. |
| `context_length` | `int \| None` | Model context window length. |
| `cwd` | `str` | Working directory (home-collapsed). |
| `turn_seconds` | `float \| None` | Wall-clock turn duration. |
| `platform_key` | `str \| None` | Gateway platform (discord/telegram/slack/…). |
| `user_config` | `dict \| None` | Effective `~/.hermes/config.yaml`. |

Return behavior: return a string block (may be multi-line). The call site
joins non-empty blocks with blank lines after the footer line. Return `None`
(or `""`) to contribute nothing.

### `usage_extra`

Fired at the end of the `/usage` command render in `cli.py`, after the
account/credits blocks. A plugin returns an extra section appended after
those blocks.

| Field | Type | Meaning |
| --- | --- | --- |
| `provider` | `str \| None` | Active provider (if any). |
| `base_url` | `str \| None` | Provider base URL. |
| `api_key` | `str \| None` | Provider API key (present only in-process; do not log). |
| `session_id` | `str \| None` | Current session id. |

Return behavior: return a string block (may be multi-line). The call site
prints non-empty blocks after the account/credits blocks. Return `None` (or
`""`) to contribute nothing.

## Example: per-provider quota block

The `hermes-quota-plugin` (https://github.com/rarf/hermes-quota-plugin)
registers both hooks. It reads a precomputed `quota_cache.json` (written by a
scheduled refresh) so the footer never does network I/O on the hot path, and
is fail-open per provider:

```python
def footer_segment(**kwargs):
    if quota_cache_age_seconds() <= CACHE_MAX_AGE_S:
        return _format_quota_block(read_quota_cache())
    return None

def register(ctx):
    ctx.register_hook("footer", footer_segment)
    ctx.register_hook("usage_extra", usage_extra)
```

## Acceptance criteria for new hooks

A new hook is accepted when it satisfies the plugin-interface expansion ground
rules:

1. **Additive** — no change to existing plugin behavior; new hook id in
   `VALID_HOOKS`.
2. **Observer-first** — where observe/mutate is a choice, ship the observer
   version first.
3. **Fail-open** — a broken plugin cannot abort the response, `/usage`, or any
   hot path.
4. **No prompt-cache violation** — render-extension hooks append text only;
   they never mutate past context or the system prompt.
5. **Documented** — listed in this catalog with kwargs and return behavior.
