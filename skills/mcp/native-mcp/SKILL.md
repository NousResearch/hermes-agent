---
name: native-mcp
description: Configure and use native MCP (Model Context Protocol) client integration to connect Hermes to external tool servers (GitHub, filesystem, databases, etc.) without any external CLI tools.
version: 1.0.0
author: community
license: MIT
metadata:
  hermes:
    tags: [MCP, Tools, Integrations, Protocol, Native]
    homepage: https://modelcontextprotocol.io
---

# Native MCP Client Integration

Hermes Agent has built-in MCP (Model Context Protocol) client support. Connect to any MCP-compatible server and use its tools natively -- no external CLI required.

## Quick Start

### 1. Configure Servers

Add MCP servers to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]

  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxxx"
```

### 2. Launch Hermes

```bash
hermes --toolsets mcp -q "What MCP tools are available?"
```

MCP servers connect automatically on startup. Their tools appear as native hermes tools.

## Configuration Reference

### Stdio Server (subprocess)

```yaml
mcp_servers:
  my_server:
    command: "npx"              # Command to run
    args: ["-y", "package"]     # Command arguments
    env:                        # Environment variables
      API_KEY: "xxx"
    cwd: "/path/to/dir"         # Working directory (optional)
    enabled: true               # Enable/disable (default: true)
    auto_connect: true          # Connect on startup (default: true)
```

### HTTP Server (remote)

```yaml
mcp_servers:
  remote_api:
    url: "https://mcp.example.com/api"
    headers:
      Authorization: "Bearer sk-xxxx"
```

## Tool Naming

MCP tools are registered with the pattern `mcp_{server}_{tool}`:

| Server | MCP Tool | Hermes Tool Name |
|--------|----------|-----------------|
| github | create_issue | `mcp_github_create_issue` |
| filesystem | read_file | `mcp_filesystem_read_file` |

## Management Commands

Use the `mcp` tool to manage connections:

```
mcp(action="status")       -- Show all servers and connection state
mcp(action="list_tools")   -- List all available MCP tools
mcp(action="reconnect", server_name="github")  -- Reconnect a server
```

## Popular MCP Servers

| Server | Install | Purpose |
|--------|---------|---------|
| Filesystem | `@modelcontextprotocol/server-filesystem` | Read/write local files |
| GitHub | `@modelcontextprotocol/server-github` | Issues, PRs, repos |
| PostgreSQL | `@modelcontextprotocol/server-postgres` | Database queries |
| Brave Search | `@modelcontextprotocol/server-brave-search` | Web search |
| Memory | `@modelcontextprotocol/server-memory` | Persistent memory |

Browse more at [mcpservers.org](https://mcpservers.org/).

## Troubleshooting

### Server won't connect
- Check that the command is installed: `npx -y @modelcontextprotocol/server-xxx --help`
- Verify environment variables are set correctly
- Check `hermes --debug` output for connection errors

### Tools not appearing
- Ensure the server is in `~/.hermes/config.yaml` under `mcp_servers`
- Check `mcp(action="status")` for connection errors
- Try `mcp(action="reconnect", server_name="xxx")`

### Timeout errors
- Some servers take time to start (especially npm packages on first run)
- Increase system timeout or retry with `mcp(action="reconnect")`

## Notes

- MCP tools are sandboxed per-server -- each tool call routes to the correct server
- If a server disconnects, its tools automatically hide from the LLM
- Auto-reconnect is attempted on the next tool call to a disconnected server
- Zero external Python dependencies -- uses only stdlib (subprocess, urllib, json, threading)
