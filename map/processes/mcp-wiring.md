---
id: mcp-wiring
kind: process
universe: mcp
name: MCP Wiring
summary: >
  Expose Hermes messaging sessions as an MCP stdio server: build tool surface,
  load session index, bridge events, and serve `run_stdio_async`.
aliases: []
tags: [mcp, stdio, bridge]
shape: process
steps:
  - id: step.1
    summary: >
      Lazy-import `MCPServer`, falling back to `FastMCP`, and construct the
      server instance with an `EventBridge` backend.
  - id: step.2
    summary: >
      Load the session index from `state.db` first, falling back to
      `sessions.json` for legacy databases.
  - id: step.3
    summary: >
      Register nine tools: `conversations_list`, `conversation_get`,
      `messages_read`, `attachments_fetch`, `events_poll`, `events_wait`,
      `messages_send`, `permissions_list_open`, `permissions_respond`.
  - id: step.4
    summary: >
      In `main()`, start the bridge, call `await server.run_stdio_async()`,
      and ensure `bridge.stop()` runs on exit.
entrypoints: [step.1]
produces: [mcp:hermes-mcp-server]
consumes: [repo:mcp_serve.py, repo:hermes_state.py]
---

# MCP Wiring

1. Lazy-import `MCPServer` from `mcp.server`, falling back to `mcp.server.fastmcp.FastMCP` for SDK 1.x (`mcp_serve.py:55-60`).
2. Initialize `EventBridge` with an in-memory queue for `message`, `approval_requested`, and `approval_resolved` events (`mcp_serve.py:336-344`).
3. Build the MCP server via `create_mcp_server(event_bridge)` and register tools (`mcp_serve.py:624-641`).
4. Load the session index from `state.db` first, falling back to `sessions.json` for legacy databases (`mcp_serve.py:102-116`).
5. Register at least these tools:
   - `conversations_list`: enumerate sessions with platform, chat type, display name, updated_at (`mcp_serve.py:646-698`).
   - `conversation_get`: return one session detail by `session_key` (`mcp_serve.py:703-731`).
   - `messages_read`: return recent `user`/`assistant` messages with row ids and timestamps (`mcp_serve.py:736-783`).
   - `attachments_fetch`: list non-text attachments for a message id (`mcp_serve.py:788-830`).
   - `events_poll`: poll bridge events after a cursor (`mcp_serve.py:835-859`).
   - `events_wait`: long-poll for one event with `timeout_ms` cap (`mcp_serve.py:864-893`).
   - `messages_send`: send to `platform:chat_id` target (`mcp_serve.py:898-1019`).
   - `permissions_list_open`: list pending approval requests (`mcp_serve.py:989-1019`).
   - `permissions_respond`: respond to approval id with `approve`/`deny` (`mcp_serve.py:1020+`).
6. In `main()`, create the bridge, start it, call `await server.run_stdio_async()`, and ensure `bridge.stop()` runs on exit (`mcp_serve.py:1045-1061`).

## Human check

Confirm `hermes mcp serve` still launches `create_mcp_server(...).run_stdio_async()` and that `messages_send` target parsing uses `platform:chat_id` format.

## Deterministic validation

```bash
grep -n "def create_mcp_server" mcp_serve.py
grep -n "@mcp.tool()" mcp_serve.py
grep -n "run_stdio_async" mcp_serve.py
grep -n "def main" mcp_serve.py
```

Expected: `create_mcp_server` at 624, nine `@mcp.tool()` decorators, `run_stdio_async` around 1054, and `main` around 1045.
