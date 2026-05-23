# Hermes MCP Server for jcode

This bridge exposes selected Hermes services as a stdio MCP server that jcode
can connect to with its built-in MCP manager.

It is intentionally dependency-free. It implements the small MCP surface jcode
uses today:

- `initialize`
- `tools/list`
- `tools/call`
- `notifications/initialized`
- `shutdown`

The server does not dispatch directly into arbitrary Hermes internals. Every
tool call is converted into the versioned `hermes-service.v1` request envelope
and handled by the jcode bridge service layer. That keeps the Rust hot path on
the jcode side while preserving Hermes' orchestration, allowlist, and safety
checks.

## Run

From a Hermes checkout:

```bash
python3 bridges/hermes-mcp-server/hermes_mcp_server.py
```

Minimal jcode `.jcode/mcp.json` entry:

```json
{
  "servers": {
    "hermes": {
      "command": "python3",
      "args": [
        "/absolute/path/to/hermes/bridges/hermes-mcp-server/hermes_mcp_server.py"
      ],
      "env": {
        "JCODE_BRIDGE_ROOT": "/absolute/path/to/hermes"
      },
      "shared": true
    }
  }
}
```

After jcode reloads MCP servers, it should expose tools such as:

- `mcp__hermes__hermes_tool`
- `mcp__hermes__hermes_web_search`
- `mcp__hermes__hermes_web_extract`
- `mcp__hermes__hermes_session_search`
- `mcp__hermes__hermes_memory`

## Tool Policy

Default allowlist:

- `web_search`
- `web_extract`
- `session_search`
- `memory`

Add more tools explicitly:

```bash
python3 bridges/hermes-mcp-server/hermes_mcp_server.py \
  --allow-tool web_search \
  --allow-tool web_extract \
  --allow-tool send_message
```

Side-effect tools such as `send_message` still flow through the service safety
checks and require confirmation fields.

## Smoke

The bridge smoke test uses mock dispatch so it does not need network access or
Hermes credentials:

```bash
scripts/jcode_bridge_smoke.py
```
