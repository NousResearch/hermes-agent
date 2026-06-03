# Achilli Tendon

Subagent orchestration health monitor for Hermes Agent.

Tendon hooks into the `subagent_stop` lifecycle event to track every child
agent's exit — normal completion, timeout, or failure — and maintains a
running ledger. It exposes that telemetry through the same tools the core
`tendon_health` and `monitor_subagents` already provide, enriched with
historical data from the current session.

## What It Does

1. **Child lifecycle tracking**: Every `subagent_stop` event is recorded with
   role, status, duration, and session ID. Maintains an in-memory ledger
   for the session.

2. **Health dashboard via `tendon_health`**: Tendon wraps the core
   `tendon_health` tool. When called, it supplements live data with
   historical aggregations — average duration, failure rate, longest-running
   child this session.

3. **Live monitoring via `monitor_subagents`**: Wraps the core
   `monitor_subagents` tool. Adds session-level context (how many children
   have completed, how many are still running, time since last completion).

4. **End-of-session summary via `on_session_end`**: On every turn boundary,
   checks if any children have been running longer than a configurable
   threshold (default 300s). Flags stuck children.

## What It Cannot Do

`pre_tool_call` hooks do **not** fire for `delegate_task` in the current
Hermes core. Delegate task is intercepted before plugin hook dispatch. This
means Tendon **cannot** intercept or block subagent spawns. It can only
observe them after they complete via `subagent_stop`.

This is a core Hermes limitation, not a plugin bug. If Hermes adds plugin
hook dispatch for `delegate_task` in a future version, Tendon will
automatically gain pre-spawn interception.

## Enabling

```bash
hermes plugins enable achilli-tendon
# or edit ~/.hermes/config.yaml:
plugins:
  enabled:
    - achilli-tendon
```

## Dependencies

- Requires Hermes Agent >= 0.15.1 (for `subagent_stop` hook)
- No external packages required
- Compatible with all other Achilli plugins

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `ACHILLI_TENDON_STUCK_THRESHOLD` | `300` | Seconds before a child is flagged as stuck |
| `ACHILLI_TENDON_MAX_LEDGER` | `1000` | Maximum child records to retain in-memory |

## Architecture

```
delegate_task() -> child runs -> child exits
                              |
                              v
                     subagent_stop hook
                              |
                              v
                     Tendon._on_subagent_stop()
                              |
                              +-- Update ledger
                              +-- Check stuck threshold
                              +-- Emit warning if needed
```

The `tendon_health` and `monitor_subagents` tools are registered by Tendon
with `override=True` to extend the core implementations. If the core tools
are already loaded, Tendon's versions wrap them and add session telemetry.
