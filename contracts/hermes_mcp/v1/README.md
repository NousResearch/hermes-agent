# Hermes MCP Bridge Contract v1

This contract covers the stdio MCP wrapper that exposes selected Hermes
services to jcode's MCP manager.

The wrapper lives at:

```text
bridges/hermes-mcp-server/hermes_mcp_server.py
```

It intentionally exposes a narrow MCP surface:

- `initialize`
- `tools/list`
- `tools/call`
- `notifications/initialized`
- `shutdown`

Tool calls are translated into the `hermes-service.v1` envelope before Hermes
dispatch. This keeps jcode integration on a standard MCP transport while
preserving Hermes-side allowlists and safety checks.

Validate the contract with:

```bash
python3 bridges/hermes-mcp-server/hermes_mcp_server.py --check --live
```
