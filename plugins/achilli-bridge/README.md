# Achilli Bridge

MCP tool chain resilience, circuit breaker, and format translation for Hermes Agent.

## STATUS: Partially Functional -- Core Hook Missing

**The `post_tool_call` hook that Bridge requires is declared in Hermes
`VALID_HOOKS` but is never invoked anywhere in the Hermes Python runtime.**
This was confirmed by source code audit (grep across `conversation_loop.py`,
`tool_executor.py`, `agent_runtime_helpers.py` -- zero call sites).

This means Bridge **loads without errors** but its core interception logic
(format translation between tool calls, failure detection, circuit breaking)
**cannot function**.

This is a known Hermes core bug. Bridge is published as-is so it's ready
when the hook is implemented.

## What It Will Do (When post_tool_call Is Fixed)

1. **Circuit breaker per MCP server**: Track consecutive failures per MCP
   server. Open circuit after N failures, return cached "unavailable"
   message instead of hammering a dead server.

2. **Tool failure detection**: Hook into `post_tool_call` to inspect every
   tool result. Detect timeouts, malformed outputs, error patterns.

3. **Bridge status dashboard**: Per-MCP-server health: success/failure rates,
   average latency, circuit breaker state.

4. **Format translation** (future): Auto-detect input requirements of the
   next tool in a chain and translate (JSON <-> CSV, Markdown <-> plain text).

5. **Retry with fallback** (future): Retry failed tools with jittered backoff.
   Fall back to alternative tools on persistent failure.

## What Works Now

Without `post_tool_call`, Bridge can only:
- Track subagent_stop events (for delegation-related MCP calls)
- Report bridge_status (static info about loaded MCP servers)

## Enabling

```bash
hermes plugins enable achilli-bridge
# or edit ~/.hermes/config.yaml:
plugins:
  enabled:
    - achilli-bridge
```

## Tracking the Core Fix

The `post_tool_call` hook needs to be invoked in the tool dispatch path
after a tool returns but before the result is appended to the conversation.
See the Hermes source at `hermes_cli/plugins.py` -- `VALID_HOOKS` declares
it at line 129, but no `invoke_hook("post_tool_call", ...)` call exists.

## Dependencies

- Requires Hermes Agent >= 0.15.1
- Requires MCP server connections (for circuit breaking to be meaningful)
- No external packages required

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `ACHILLI_BRIDGE_FAILURE_THRESHOLD` | `3` | Consecutive failures before circuit opens |
| `ACHILLI_BRIDGE_COOLDOWN` | `60` | Seconds before circuit goes half-open |
| `ACHILLI_BRIDGE_MAX_LATENCY_MS` | `5000` | Log warning if MCP call exceeds this |
