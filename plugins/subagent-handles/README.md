# subagent-handles

Tracks in-flight `delegate_task` children as lightweight handles and exposes
mid-flight steering tools to the parent agent.

## What it does

- Registers a `subagent_start` hook → builds a `SubagentHandle` in a
  thread-safe registry keyed by `subagent_id` + `session_id`.
- Registers a `subagent_stop` hook → transitions the handle to `"done"`
  (keeps it for post-run inspection).
- Exposes `subagent_send(subagent_id, text)` to queue a follow-up message
  to a still-running child.
- Exposes `cancel_subagent(subagent_id)` to abort a running child.

All four pieces share one module-level `registry` singleton, so handles
registered by the hooks are immediately visible to the tools.

## Install

Drop the `subagent-handles/` directory under `plugins/` in your hermes-agent
checkout. The plugin is enabled automatically on load.

## Tests

```
python -m pytest plugins/subagent-handles/tests/ -q
```

32 tests covering registry, hooks, tools, and integration (hook-registered
handle is resolvable by sender), including a regression guard asserting
`register_tools` uses the correct `PluginContext.register_tool(name, toolset,
schema, handler)` signature.

## Attribution

Derived from in-session work by Michael Anselmi, reconciled 2026-08-11.
