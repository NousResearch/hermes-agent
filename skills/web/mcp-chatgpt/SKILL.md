---
name: mcp-chatgpt
description: Run Hermes as an MCP server for ChatGPT Desktop and Codex.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [mcp, chatgpt, desktop, codex]
    category: web
    related_skills: [chatgpt-actions, hermes-bridge]
    config: []
---

# MCP ChatGPT Skill

Run Hermes as a Model Context Protocol (MCP) server (`python -m tools.mcp_server`) so
ChatGPT Desktop, Claude Desktop, Codex, Cursor, or any MCP client can call Hermes functions
directly: `read_file`, `write_file`, `patch`, `search_files`, `terminal`, `web_search`,
`web_extract`, `delegate_task`, `memory`, `todo_list`, `vision_analyze`, `image_generate`,
`execute_code`, and more.

## When to Use

- You want ChatGPT Desktop / Codex to call Hermes functions natively via MCP.
- You prefer a local stdio MCP connection over the remote OpenAI Actions path
  (`chatgpt-actions` skill).
- You are wiring Hermes into an MCP host that already speaks the protocol.

## Prerequisites

- Hermes agent source available locally (the `tools/mcp_server.py` module).
- The `mcp` Python package only needed for the optional `--http` transport; stdio works
  with the standard library alone.
- Tool secrets (e.g. `OPENAI_API_KEY`, `API_SERVER_KEY`) set so the tools you want to call
  pass their `check_fn`.

## How to Run

```bash
cd <HERMES_AGENT_DIR>
python -m tools.mcp_server
```

Then add the hermes server to your MCP client config (see `openapi/mcp_client_config.json`),
replacing `<HERMES_AGENT_DIR>` and the secrets.

## Quick Reference

| MCP method | Purpose |
|------------|---------|
| `tools/list` | List every registered Hermes tool |
| `tools/call` | Call one Hermes tool by name with arguments |
| `ping` | Liveness check |
| `initialize` | Capability negotiation |

## Procedure

1. Start the MCP server: `python -m tools.mcp_server` (stdio on stdout/stdin).
2. Add the `hermes` MCP server entry to your client's config.
3. Restart the client so it discovers the server.
4. Ask the client to use a Hermes tool, e.g. "read the file README.md" or "search files for TODO".

## Pitfalls

- MCP stdio carries no auth; trust the client↔server process boundary. Tool-level secrets gate
  what can actually run.
- Only tools whose `check_fn()` passes are callable; a tool missing its secret returns a clean
  error instead of failing silently.
- The `--http` flag requires `pip install mcp`; stdio is the default and recommended transport.

## Verification

- [ ] `python -m tools.mcp_server` starts and logs to stderr.
- [ ] MCP client lists the `hermes` server and its tools.
- [ ] Calling a no-secret tool (e.g. `search_files`) returns content.
- [ ] Calling a tool whose secret is missing returns a clear error, not a crash.
