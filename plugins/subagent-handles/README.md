# subagent-handles

Tracks in-flight `delegate_task` children as durable handles and exposes a
read-only registry introspection tool to the parent agent.

## What it does

- Registers a `subagent_start` hook → builds a `SubagentHandle` in a
  thread-safe registry keyed by `subagent_id` + `session_id`, and checkpoints
  it to disk under `HERMES_HOME/state/subagent-handles/`.
- Registers a `subagent_stop` hook → transitions the handle to `"done"`
  (keeps it for post-run inspection, persisted).
- Restores persisted handles on plugin load, reconciling any handle still
  marked `"running"` to `"failed"` — a child is a subprocess of its owner, so
  after a restart a "running" handle belongs to a dead process.
- Exposes `subagent_handles` (read-only): list every tracked handle, or
  resolve one by `subagent_id` — state, role, session, parent, goal.

Live steering is intentionally NOT in this plugin: the platform delegation
toolset already ships `subagent_send` and `cancel_subagent`. Registering
those names from a plugin would be rejected by `register_tool` as shadowing
a built-in without `allow_tool_override`, so this plugin provides the
durable registry layer and lets the platform tools do the steering.

All pieces share one module-level `registry` singleton, so handles
registered by the hooks are immediately visible to the tool.

## Install

Drop the `subagent-handles/` directory under `plugins/` in your hermes-agent
checkout. The plugin is enabled automatically on load.

## Tests

```
python -m pytest plugins/subagent-handles/tests/ -q
```

27 tests covering registry, hooks, persistence, status tool, and
integration (hook-registered handle is resolvable by the status tool),
including a regression guard asserting `register_tools` uses the correct
`PluginContext.register_tool(name, toolset, schema, handler)` signature and
does NOT shadow the platform `subagent_send` / `cancel_subagent` tools.

## Attribution

Derived from in-session work by Michael Anselmi, reconciled 2026-08-11.
