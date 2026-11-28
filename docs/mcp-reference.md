# Hermes MCP Reference

Complete Model Context Protocol (MCP) reference for Hermes Agent — both as an MCP **client** (connecting to external MCP servers for extra tools) and as an MCP **server** (exposing Hermes conversations to other agents).

## Table of Contents

1. [Hermes as MCP Client](#hermes-as-mcp-client)
   - [Stdio Transport](#stdio-transport)
   - [HTTP / StreamableHTTP Transport](#http--streamablehttp-transport)
   - [SSE Transport](#sse-transport)
   - [OAuth Authentication](#oauth-authentication)
   - [Tool Selection](#tool-selection)
   - [Sampling (Server-Initiated LLM Requests)](#sampling)
   - [Common MCP Server Examples](#common-mcp-server-examples)
2. [Hermes as MCP Server](#hermes-as-mcp-server)
   - [Tools Reference](#tools-reference)
   - [Client Integration Examples](#client-integration-examples)
3. [Catalog Servers](#catalog-servers)
4. [Troubleshooting](#troubleshooting)
5. [Developer Guide](#developer-guide)

---

## Hermes as MCP Client

Hermes connects to external MCP servers to extend its toolset at runtime. Configuration lives in `~/.hermes/config.yaml` under the `mcp_servers` key.

### Stdio Transport

Spawns a local process and communicates via stdin/stdout JSON-RPC.

```yaml
# ~/.hermes/config.yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    env: {}
    timeout: 120          # per-tool-call timeout (seconds, default: 300)
    connect_timeout: 60   # initial connection timeout (default: 60)
    keepalive_interval: 10  # liveness ping cadence (default: 180s, floored at 5s)
    supports_parallel_tool_calls: true  # allow concurrent tool calls

  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."
```

**Key parameters:**

| Parameter                | Type    | Default | Description                                                 |
|--------------------------|---------|---------|-------------------------------------------------------------|
| `command`                | string  | —       | Binary or command to execute (required for stdio)           |
| `args`                   | list    | `[]`    | Arguments passed to the command                             |
| `env`                    | dict    | `{}`    | Environment variables for the subprocess                    |
| `timeout`                | int     | `300`   | Maximum seconds per tool call                               |
| `connect_timeout`        | int     | `60`    | Maximum seconds to wait for initial JSON-RPC handshake      |
| `keepalive_interval`     | int     | `180`   | Seconds between liveness pings; lower for servers with short TTLs (e.g. Unreal Engine editor, ~15s) |
| `supports_parallel_tool_calls` | bool | `false` | When `true`, tools from this server may run concurrently |

**CLI commands:**

```bash
hermes mcp add filesystem --command npx --args -y @modelcontextprotocol/server-filesystem /tmp
hermes mcp list          # show all configured servers
hermes mcp test <name>   # test connection to a server
hermes mcp remove <name> # remove a server
```

### HTTP / StreamableHTTP Transport

Connects to a remote MCP endpoint over HTTP. Hermes uses StreamableHTTP by default when the transport isn't explicitly set and the server endpoint negotiates it.

```yaml
mcp_servers:
  remote_api:
    url: "https://my-mcp-server.example.com/mcp"
    headers:
      Authorization: "Bearer sk-..."
    timeout: 180
    connect_timeout: 30
```

**Key parameters:**

| Parameter         | Type   | Default  | Description                                    |
|-------------------|--------|----------|------------------------------------------------|
| `url`             | string | —        | MCP endpoint URL (required for HTTP transport) |
| `headers`         | dict   | `{}`     | Custom HTTP headers (auth tokens, etc.)        |
| `timeout`         | int    | `300`    | Per-tool-call timeout                          |
| `connect_timeout` | int    | `60`     | Initial HTTP connection timeout                |

### SSE Transport

For MCP servers using the Server-Sent Events protocol:

```yaml
mcp_servers:
  searxng:
    url: "http://localhost:8000/sse"
    transport: sse
    timeout: 180
    connect_timeout: 10
```

Hybrid servers (SSE endpoint + stdio command for analysis) are supported:

```yaml
mcp_servers:
  analysis:
    url: "http://localhost:8000/sse"
    transport: sse
    command: "npx"
    args: ["-y", "analysis-server"]
    timeout: 180
```

### OAuth Authentication

For MCP servers requiring OAuth 2.0 (e.g. Linear, Google APIs):

```yaml
mcp_servers:
  linear:
    url: "https://api.linear.app/mcp"
    auth: oauth
    oauth:
      client_id: "your-client-id"
      client_secret: "${LINEAR_CLIENT_SECRET}"
```

CLI commands:

```bash
hermes mcp add linear --url https://api.linear.app/mcp --auth oauth
hermes mcp login linear     # trigger OAuth flow
hermes mcp reauth --all     # re-authenticate all OAuth servers
```

### Tool Selection

Toggle individual tools on/off per server:

```bash
hermes mcp configure <name>   # interactive tool picker
```

Enable only specific tools:

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    enabled_tools:
      - read_file
      - list_directory
```

### Sampling

Some MCP servers can request LLM completions during tool execution (Sampling). Configure per-server:

```yaml
mcp_servers:
  smart_server:
    url: "https://..."
    sampling:
      enabled: true                # default: true
      model: "gemini-3-flash"     # override model
      max_tokens_cap: 4096        # max tokens per request
      timeout: 30                  # LLM call timeout (seconds)
      max_rpm: 10                  # max requests per minute
      allowed_models: []           # model whitelist (empty = all)
      max_tool_rounds: 5           # tool loop limit (0 = disable)
      log_level: "info"            # audit verbosity
```

### Common MCP Server Examples

**Filesystem** — read/write files and directories:
```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
```

**GitHub** — manage issues, PRs, repos:
```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."
```

**SQLite** — query local databases:
```yaml
mcp_servers:
  sqlite:
    command: "uvx"
    args: ["mcp-server-sqlite", "--db-path", "/path/to/database.db"]
```

**Playwright** — browser automation:
```yaml
mcp_servers:
  playwright:
    command: "npx"
    args: ["-y", "@playwright/mcp"]
```

**PostgreSQL** — query remote databases:
```yaml
mcp_servers:
  postgres:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-postgres"]
    env:
      DATABASE_URL: "postgresql://user:pass@localhost:5432/mydb"
```

**Brave Search** — web search:
```yaml
mcp_servers:
  brave:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-brave-search"]
    env:
      BRAVE_API_KEY: "your-api-key"
```

---

## Hermes as MCP Server

Hermes exposes its messaging conversations as MCP tools, letting any MCP client (Claude Code, Cursor, Codex, VS Code, etc.) read and send messages across all connected platforms.

### Quick Start

```bash
# Start the MCP server on stdio
hermes mcp serve

# With verbose logging to stderr
hermes mcp serve --verbose
```

Requirements: the `mcp` Python package (`pip install mcp`).

### Tools Reference

Hermes exposes **10 tools**, matching OpenClaw's MCP channel bridge surface:

#### `conversations_list`

List active messaging conversations across connected platforms.

| Parameter   | Type    | Default | Description                                      |
|-------------|---------|---------|--------------------------------------------------|
| `platform`  | string  | —       | Filter by platform (telegram, discord, slack…)   |
| `limit`     | int     | 50      | Max conversations (1–200)                        |
| `search`    | string  | —       | Filter conversations by name                     |

Returns: `count`, `conversations[]` with `session_key`, `platform`, `display_name`, `updated_at`.

#### `conversation_get`

Get detailed info about one conversation.

| Parameter     | Type   | Required | Description                                 |
|---------------|--------|----------|---------------------------------------------|
| `session_key` | string | ✅       | Session key from `conversations_list`        |

Returns: platform, chat type, names, token counts, timestamps.

#### `messages_read`

Read recent messages from a conversation in chronological order.

| Parameter     | Type   | Default | Description                                 |
|---------------|--------|---------|---------------------------------------------|
| `session_key` | string | ✅      | Session key from `conversations_list`        |
| `limit`       | int    | 50      | Max messages (1–200, most recent first)      |

Returns: `messages[]` with `id`, `role`, `content` (truncated at 2000 chars), `timestamp`.

#### `attachments_fetch`

List non-text attachments (images, media files) for a message.

| Parameter     | Type   | Required | Description                        |
|---------------|--------|----------|------------------------------------|
| `session_key` | string | ✅       | Session key                        |
| `message_id`  | string | ✅       | Message ID from `messages_read`    |

#### `events_poll`

Poll for new conversation events since a cursor position (non-blocking).

| Parameter      | Type    | Default | Description                               |
|----------------|---------|---------|-------------------------------------------|
| `after_cursor` | int     | 0       | Return events after this cursor           |
| `session_key`  | string  | —       | Filter to one conversation                |
| `limit`        | int     | 20      | Max events (1–200)                        |

Event types: `message`, `approval_requested`, `approval_resolved`.

#### `events_wait`

Wait for the next event (long-poll, blocking up to timeout).

| Parameter      | Type    | Default  | Description                               |
|----------------|---------|----------|-------------------------------------------|
| `after_cursor` | int     | 0        | Wait for events after this cursor         |
| `session_key`  | string  | —        | Filter to one conversation                |
| `timeout_ms`   | int     | 30000    | Max wait in milliseconds (max 300000)     |

Returns `null` + `"reason": "timeout"` when no event arrives.

#### `messages_send`

Send a message to a platform conversation. Target format: `"platform:identifier"`.

| Parameter  | Type   | Required | Description                                                |
|------------|--------|----------|------------------------------------------------------------|
| `target`   | string | ✅       | `"telegram:6308981865"`, `"discord:#general"`, `"slack:#engineering"` |
| `message`  | string | ✅       | Message text                                               |

#### `channels_list`

List available messaging channels and target strings for `messages_send`.

| Parameter  | Type    | Description                                  |
|------------|---------|----------------------------------------------|
| `platform` | string  | Filter by platform (telegram, discord, etc.) |

#### `permissions_list_open`

List pending approval requests observed during this bridge session.

No parameters. Returns exec and plugin approval requests.

#### `permissions_respond`

Respond to a pending approval request.

| Parameter  | Type   | Required | Description                                        |
|------------|--------|----------|----------------------------------------------------|
| `id`       | string | ✅       | Approval ID from `permissions_list_open`            |
| `decision` | string | ✅       | One of: `"allow-once"`, `"allow-always"`, `"deny"` |

### Client Integration Examples

**Claude Code:**
```bash
claude mcp add --transport stdio hermes -- hermes mcp serve
```

**Cursor / Windsurf / VS Code (`mcp.json`):**
```json
{
  "mcpServers": {
    "hermes": {
      "command": "hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

**Claude Desktop (`claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "hermes": {
      "command": "hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

**Zed / JetBrains (ACP native):**
```
Add stdio MCP server: command=hermes, args=["mcp", "serve"]
```

---

## Catalog Servers

Hermes maintains a curated catalog of one-click-install MCP servers:

```bash
hermes mcp catalog          # list available catalog servers
hermes mcp install linear   # install a catalog server
```

Current catalog entries: **Linear**, **n8n**, **Unreal Engine**.

To submit a server for the catalog, open a PR against the Hermes agent repo adding a `mcp_catalog.yaml` entry.

---

## Troubleshooting

### Connection fails with "command not found"

Ensure `npx`/`uvx`/`node` is on the PATH visible to Hermes. For explicit paths:

```yaml
mcp_servers:
  myserver:
    command: "/usr/local/bin/npx"     # absolute path
    args: ["-y", "@scope/server"]
    env:
      PATH: "/usr/local/bin:/usr/bin"  # or add PATH
```

### "ECONNREFUSED" / timeout on HTTP servers

- Verify the server is running and reachable
- Check firewall rules
- Increase `connect_timeout` for slow-starting servers
- The URL must point to an MCP endpoint, not a plain web app

### Credentials leaking in error messages

Hermes automatically strips credential-like values (API keys, tokens, passwords) from error messages returned to the LLM. This is on by default.

### SSE server stops responding

SSE connections can time out. Increase `keepalive_interval` or switch to StreamableHTTP if the server supports it.

### "package 'mcp' not found" when running `hermes mcp serve`

```bash
pip install mcp
```

The `mcp` Python package provides the FastMCP SDK used by Hermes' MCP server.

---

## Developer Guide

### Building an MCP Server for Hermes

Hermes supports three MCP transport protocols:

| Transport         | MIME Type                     | Best For                                    |
|-------------------|-------------------------------|---------------------------------------------|
| **stdio**         | —                             | Local tools, CLI wrappers                   |
| **StreamableHTTP**| `application/json`            | Remote servers, REST-style backends         |
| **SSE**           | `text/event-stream` (SSE)     | Long-lived event streams, push notifications|

### Stdio Server (Python, FastMCP)

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
```

### HTTP Server (Python, FastMCP)

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-remote-server")

@mcp.tool()
def search_db(query: str) -> str:
    """Search the database."""
    return f"Results for: {query}"

if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_streamable_http_async())
```

### Best Practices

1. **Tool descriptions are critical** — the LLM decides when to call your tool based on its name and docstring. Write clear, specific docstrings.

2. **Return structured JSON** — primitive strings work but structured JSON lets the LLM parse results correctly.

3. **Validate inputs** — MCP clients can send any value type; coerce and validate at the tool boundary.

4. **Handle errors gracefully** — return error messages as strings (don't raise exceptions that break the JSON-RPC connection).

5. **Test with `hermes mcp test`** — verify your server works end-to-end before relying on it.

6. **Registry submission** — to get your server into the Hermes catalog (`hermes mcp install`), open a PR adding your entry to the catalog in the agent repo.

---

## Related Docs

- [IDE Integration Guide](ide-integration.md) — connecting Hermes to IDEs via MCP and ACP
