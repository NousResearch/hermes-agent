# MCP (Model Context Protocol) Support

MCP lets Hermes Agent connect to external tool servers — giving the agent access to databases, APIs, filesystems, and more without any code changes.

## Overview

The [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) is an open standard for connecting AI agents to external tools and data sources. MCP servers expose tools over a lightweight RPC protocol, and Hermes Agent can connect to any compliant server automatically.

What this means for you:

- **Thousands of ready-made tools** — browse the [MCP server directory](https://github.com/modelcontextprotocol/servers) for servers covering GitHub, Slack, databases, file systems, web scraping, and more.
- **No code changes needed** — add a few lines to `~/.hermes/config.yaml` and the tools appear alongside built-in ones.
- **Mix and match** — run multiple MCP servers simultaneously, combining stdio-based and HTTP-based servers.
- **Secure by default** — environment variables are filtered and credentials are stripped from error messages returned to the LLM.

## Prerequisites

Install MCP support as an optional dependency:

```bash
pip install hermes-agent[mcp]
```

Depending on which MCP servers you want to use, you may need additional runtimes:

| Server Type | Runtime Needed | Example |
|-------------|---------------|---------|
| HTTP/remote | Nothing extra | `url: "https://mcp.example.com"` |
| npm-based (npx) | Node.js 18+ | `command: "npx"` |
| Python-based | uv (recommended) | `command: "uvx"` |

Most popular MCP servers are distributed as npm packages and launched via `npx`. Python-based servers typically use `uvx` (from the [uv](https://docs.astral.sh/uv/) package manager).

## Configuration

MCP servers are configured in `~/.hermes/config.yaml` under the `mcp_servers` key. Each entry is a named server with its connection details.

### Stdio Servers (command + args + env)

Stdio servers run as local subprocesses. Communication happens over stdin/stdout.

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
    env: {}

  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxxxxxxxxxxx"
```

| Key | Required | Description |
|-----|----------|-------------|
| `command` | Yes | Executable to run (e.g., `npx`, `uvx`, `python`) |
| `args` | No | List of command-line arguments |
| `env` | No | Environment variables to pass to the subprocess |

**Note:** Only explicitly listed `env` variables plus a safe baseline (PATH, HOME, USER, LANG, SHELL, TMPDIR, XDG_*) are passed to the subprocess. Your shell's API keys, tokens, and secrets are **not** leaked. See [Security](#security) for details.

### HTTP Servers (url + headers)

HTTP servers run remotely and are accessed over HTTP/StreamableHTTP.

```yaml
mcp_servers:
  remote_api:
    url: "https://my-mcp-server.example.com/mcp"
    headers:
      Authorization: "Bearer sk-xxxxxxxxxxxx"
```

| Key | Required | Description |
|-----|----------|-------------|
| `url` | Yes | Full URL of the MCP HTTP endpoint |
| `headers` | No | HTTP headers to include (e.g., auth tokens) |

### Per-Server Timeouts

Each server can have custom timeouts:

```yaml
mcp_servers:
  slow_database:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-postgres"]
    env:
      DATABASE_URL: "postgres://user:pass@localhost/mydb"
    timeout: 300          # Tool call timeout in seconds (default: 120)
    connect_timeout: 90   # Initial connection timeout in seconds (default: 60)
```

| Key | Default | Description |
|-----|---------|-------------|
| `timeout` | 120 | Maximum seconds to wait for a single tool call to complete |
| `connect_timeout` | 60 | Maximum seconds to wait for the initial connection and tool discovery |

### Mixed Configuration Example

You can combine stdio and HTTP servers freely:

```yaml
mcp_servers:
  # Local filesystem access via stdio
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

  # GitHub API via stdio with auth
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxxxxxxxxxxx"

  # Remote database via HTTP
  company_db:
    url: "https://mcp.internal.company.com/db"
    headers:
      Authorization: "Bearer sk-xxxxxxxxxxxx"
    timeout: 180

  # Python-based server via uvx
  memory:
    command: "uvx"
    args: ["mcp-server-memory"]
```

## Config Translation (Claude/Cursor JSON → Hermes YAML)

Many MCP server docs show configuration in Claude Desktop JSON format. Here's how to translate:

**Claude Desktop JSON** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxxxxxxxxx"
      }
    }
  }
}
```

**Hermes Agent YAML** (`~/.hermes/config.yaml`):

```yaml
mcp_servers:                          # mcpServers → mcp_servers (snake_case)
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    env: {}
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxxxxxxxxxxx"
```

Translation rules:

1. **Key name**: `mcpServers` → `mcp_servers` (snake_case)
2. **Format**: JSON → YAML (remove braces/brackets, use indentation)
3. **Arrays**: `["a", "b"]` stays the same in YAML flow style, or use block style with `- a`
4. **Everything else**: Keys (`command`, `args`, `env`) are identical

## How It Works

### Startup & Discovery

When Hermes Agent starts, the tool discovery system calls `discover_mcp_tools()`:

1. **Config loading** — Reads `mcp_servers` from `~/.hermes/config.yaml`
2. **Background loop** — Spins up a dedicated asyncio event loop in a daemon thread for MCP connections
3. **Connection** — Connects to each configured server (stdio subprocess or HTTP)
4. **Session init** — Initializes the MCP client session (protocol handshake)
5. **Tool discovery** — Calls `list_tools()` on each server to get available tools
6. **Registration** — Registers each MCP tool into the Hermes tool registry with a prefixed name

### Tool Registration

Each discovered MCP tool is registered with a prefixed name following this pattern:

```
mcp_{server_name}_{tool_name}
```

Hyphens and dots in both server and tool names are replaced with underscores for API compatibility. For example:

| Server Name | MCP Tool Name | Registered As |
|-------------|--------------|---------------|
| `filesystem` | `read_file` | `mcp_filesystem_read_file` |
| `github` | `create-issue` | `mcp_github_create_issue` |
| `my-api` | `query.data` | `mcp_my_api_query_data` |

Tools appear alongside built-in tools — the agent sees them in its tool list and can call them like any other tool.

### Tool Calling

When the agent calls an MCP tool:

1. The handler is invoked by the tool registry (sync interface)
2. The handler schedules the actual MCP `call_tool()` RPC on the background event loop
3. The call blocks (with timeout) until the MCP server responds
4. Response content blocks are collected and returned as JSON
5. Errors are sanitized to strip credentials before returning to the LLM

### Shutdown

On agent exit, `shutdown_mcp_servers()` is called:

1. All server tasks are signalled to exit via their shutdown events
2. Each server's `async with` context manager exits, cleaning up transports
3. The background event loop is stopped and its thread is joined
4. All server state is cleared

## Security

### Environment Variable Filtering

When launching stdio MCP servers, Hermes does **not** pass your full shell environment to the subprocess. The `_build_safe_env()` function constructs a minimal environment:

**Always passed through** (from your current environment):
- `PATH`, `HOME`, `USER`, `LANG`, `LC_ALL`, `TERM`, `SHELL`, `TMPDIR`
- Any variable starting with `XDG_`

**Explicitly added**: Any variables you list in the server's `env` config.

**Everything else is excluded** — your `OPENAI_API_KEY`, `AWS_SECRET_ACCESS_KEY`, database passwords, and other secrets are never leaked to MCP server subprocesses unless you explicitly add them.

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      # Only this token is passed — nothing else from your shell
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxxxxxxxxxxx"
```

### Credential Stripping in Errors

If an MCP tool call fails, the error message is sanitized by `_sanitize_error()` before being returned to the LLM. The following patterns are replaced with `[REDACTED]`:

- GitHub PATs (`ghp_...`)
- OpenAI-style keys (`sk-...`)
- Bearer tokens (`Bearer ...`)
- Query parameters (`token=...`, `key=...`, `API_KEY=...`, `password=...`, `secret=...`)

This prevents accidental credential exposure through error messages in the conversation.

## Transport Types

### Stdio Transport

The default transport for locally-installed MCP servers. The server runs as a subprocess and communicates over stdin/stdout.

```yaml
mcp_servers:
  my_server:
    command: "npx"           # or "uvx", "python", any executable
    args: ["-y", "package"]
    env:
      MY_VAR: "value"
```

**Pros:** Simple setup, no network needed, works offline.
**Cons:** Server must be installed locally, one process per server.

### HTTP / StreamableHTTP Transport

For remote MCP servers accessible over HTTP. Uses the StreamableHTTP protocol from the MCP SDK.

```yaml
mcp_servers:
  my_remote:
    url: "https://mcp.example.com/endpoint"
    headers:
      Authorization: "Bearer token"
```

**Pros:** No local installation needed, shared servers, cloud-hosted.
**Cons:** Requires network, slightly higher latency, needs `mcp` package with HTTP support.

**Note:** If HTTP transport is not available in your installed `mcp` package version, Hermes will log a clear error and skip that server.

## Reconnection

If an MCP server connection drops after initial setup (e.g., process crash, network hiccup), Hermes automatically attempts to reconnect with exponential backoff:

| Attempt | Delay Before Retry |
|---------|--------------------|
| 1 | 1 second |
| 2 | 2 seconds |
| 3 | 4 seconds |
| 4 | 8 seconds |
| 5 | 16 seconds |

- Maximum of **5 retry attempts** before giving up
- Backoff is capped at **60 seconds** (relevant if the formula exceeds this)
- Reconnection only triggers for **established connections** that drop — initial connection failures are reported immediately without retries
- If shutdown is requested during reconnection, the retry loop exits cleanly

## Troubleshooting

### Common Errors

**"mcp package not installed"**

```
MCP SDK not available -- skipping MCP tool discovery
```

Solution: Install the MCP optional dependency:

```bash
pip install hermes-agent[mcp]
```

---

**"command not found" or server fails to start**

The MCP server command (`npx`, `uvx`, etc.) is not on PATH.

Solution: Install the required runtime:

```bash
# For npm-based servers
npm install -g npx    # or ensure Node.js 18+ is installed

# For Python-based servers
pip install uv        # then use "uvx" as the command
```

---

**"MCP server 'X' has no 'command' in config"**

Your stdio server config is missing the `command` key.

Solution: Check your `~/.hermes/config.yaml` indentation and ensure `command` is present:

```yaml
mcp_servers:
  my_server:
    command: "npx"        # <-- required for stdio servers
    args: ["-y", "package-name"]
```

---

**Server connects but tools fail with authentication errors**

Your API key or token is missing or invalid.

Solution: Ensure the key is in the server's `env` block (not your shell env):

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_your_actual_token"  # <-- check this
```

---

**"MCP server 'X' is not connected"**

The server disconnected and reconnection failed (or was never established).

Solution:
1. Check the Hermes logs for connection errors (`hermes --verbose`)
2. Verify the server works standalone (e.g., run the `npx` command manually)
3. Increase `connect_timeout` if the server is slow to start

---

**Connection timeout during discovery**

```
Failed to connect to MCP server 'X': TimeoutError
```

Solution: Increase the `connect_timeout` for slow-starting servers:

```yaml
mcp_servers:
  slow_server:
    command: "npx"
    args: ["-y", "heavy-server-package"]
    connect_timeout: 120   # default is 60
```

---

**HTTP transport not available**

```
mcp.client.streamable_http is not available
```

Solution: Upgrade the `mcp` package to a version that includes HTTP support:

```bash
pip install --upgrade mcp
```

## Popular MCP Servers

Here are some popular free MCP servers you can use immediately:

| Server | Package | Description |
|--------|---------|-------------|
| Filesystem | `@modelcontextprotocol/server-filesystem` | Read/write/search local files |
| GitHub | `@modelcontextprotocol/server-github` | Issues, PRs, repos, code search |
| Git | `@modelcontextprotocol/server-git` | Git operations on local repos |
| Fetch | `@modelcontextprotocol/server-fetch` | HTTP fetching and web content extraction |
| Memory | `@modelcontextprotocol/server-memory` | Persistent key-value memory |
| SQLite | `@modelcontextprotocol/server-sqlite` | Query SQLite databases |
| PostgreSQL | `@modelcontextprotocol/server-postgres` | Query PostgreSQL databases |
| Brave Search | `@modelcontextprotocol/server-brave-search` | Web search via Brave API |
| Puppeteer | `@modelcontextprotocol/server-puppeteer` | Browser automation |
| Sequential Thinking | `@modelcontextprotocol/server-sequential-thinking` | Step-by-step reasoning |

### Example Configs for Popular Servers

```yaml
mcp_servers:
  # Filesystem — no API key needed
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]

  # Git — no API key needed
  git:
    command: "uvx"
    args: ["mcp-server-git", "--repository", "/home/user/my-repo"]

  # GitHub — requires a personal access token
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxxxxxxxxxxx"

  # Fetch — no API key needed
  fetch:
    command: "uvx"
    args: ["mcp-server-fetch"]

  # SQLite — no API key needed
  sqlite:
    command: "uvx"
    args: ["mcp-server-sqlite", "--db-path", "/home/user/data.db"]

  # Brave Search — requires API key (free tier available)
  brave_search:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-brave-search"]
    env:
      BRAVE_API_KEY: "BSA_xxxxxxxxxxxx"
```

## Advanced

### Multiple Servers

You can run as many MCP servers as you want simultaneously. Each server gets its own subprocess (stdio) or HTTP connection, and all tools are registered into a single unified namespace.

Servers are connected sequentially during startup. If one server fails to connect, the others still work — failed servers are logged as warnings and skipped.

### Tool Naming Convention

All MCP tools follow the naming pattern:

```
mcp_{server_name}_{tool_name}
```

Both the server name and tool name are sanitized: hyphens (`-`) and dots (`.`) are replaced with underscores (`_`). This ensures compatibility with LLM function-calling APIs that restrict tool name characters.

If you configure a server named `my-api` that exposes a tool called `query.users`, the agent will see it as `mcp_my_api_query_users`.

### Configurable Timeouts

Fine-tune timeouts per server based on expected response times:

```yaml
mcp_servers:
  fast_cache:
    command: "npx"
    args: ["-y", "mcp-server-redis"]
    timeout: 30            # Fast lookups — short timeout
    connect_timeout: 15

  slow_analysis:
    url: "https://analysis.example.com/mcp"
    timeout: 600           # Long-running analysis — generous timeout
    connect_timeout: 120
```

### Idempotent Discovery

`discover_mcp_tools()` is idempotent — calling it multiple times only connects to servers that aren't already running. Already-connected servers keep their existing connections and tool registrations.

### Custom Toolsets

Each MCP server's tools are automatically grouped into a toolset named `mcp-{server_name}`. These toolsets are also injected into all `hermes-*` platform toolsets, so MCP tools are available in CLI, Telegram, Discord, and other platforms.

### Thread Safety

The MCP subsystem is fully thread-safe. A dedicated background event loop runs in a daemon thread, and all server state is protected by a lock. This works correctly even with Python 3.13+ free-threading builds.
