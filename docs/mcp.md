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

## Sampling (Server-Initiated LLM Requests)

MCP's `sampling/createMessage` capability allows MCP servers to request LLM completions through the Hermes agent. This is a powerful feature that enables agent-in-the-loop workflows where servers can leverage the LLM for tasks like:

- **Data analysis**: A database server asks the LLM to interpret query results
- **Content generation**: A CMS server requests the LLM to generate or summarize content
- **Decision making**: A workflow server asks the LLM to decide the next step
- **Code review**: A code analysis server asks the LLM to review findings

### How It Works

```
MCP Server --[sampling/createMessage]--> ClientSession
  --> sampling callback (async, on MCP background loop)
    --> asyncio.to_thread(sync LLM call)   # non-blocking
    --> CreateMessageResult | ErrorData returned to server
```

The sampling callback:

1. Validates the request against configurable rate limits
2. Resolves the model to use (config override > server hint > default)
3. Converts MCP messages to OpenAI-compatible format (text + images)
4. Adds system prompt if provided by the server
5. Offloads the LLM call to a thread via `asyncio.to_thread()` (non-blocking)
6. Applies a configurable timeout to the LLM call
7. Sanitizes the response (credential stripping)
8. Returns a `CreateMessageResult` on success or `ErrorData` on failure

### Configuration

Sampling is **enabled by default** for all configured MCP servers. No additional setup is needed — if you have an auxiliary LLM client configured (OpenRouter, Nous Portal, or custom endpoint), sampling works automatically.

#### Per-Server Sampling Config

```yaml
mcp_servers:
  analysis_server:
    command: "npx"
    args: ["-y", "my-analysis-server"]
    sampling:
      enabled: true           # default: true
      model: "gemini-3-flash" # override model (optional)
      max_tokens_cap: 4096    # max tokens per request (default: 4096)
      timeout: 30             # LLM call timeout in seconds (default: 30)
      max_rpm: 10             # max requests per minute (default: 10)
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sampling.enabled` | bool | `true` | Enable or disable sampling for this server |
| `sampling.model` | string | (auto) | Force a specific model for sampling requests |
| `sampling.max_tokens_cap` | int | `4096` | Maximum tokens a server can request |
| `sampling.timeout` | int | `30` | Timeout in seconds for each LLM call |
| `sampling.max_rpm` | int | `10` | Maximum sampling requests per minute |
| `sampling.allowed_models` | list | `[]` | Model whitelist (empty = allow all) |
| `sampling.max_tool_rounds` | int | `5` | Max consecutive tool use rounds (0 = disable tool loops) |
| `sampling.log_level` | string | `"info"` | Audit log verbosity: `"debug"`, `"info"`, or `"warning"` |

#### Model Resolution

The model used for sampling is resolved in this order:

1. **Config override** (`sampling.model`) — always used if set
2. **Server hint** (`modelPreferences.hints[].name`) — from the server's request
3. **Default** — the auxiliary client's default model

### Security

Sampling introduces a vector where MCP servers can trigger LLM calls, so several safeguards are in place:

#### Non-Blocking Execution

The sync LLM client call is offloaded to a thread via `asyncio.to_thread()`. This prevents blocking the MCP background event loop, which also serves tool calls and other server connections.

#### LLM Call Timeout

Each LLM call has a configurable timeout (default: 30 seconds) enforced via `asyncio.wait_for()`. If the LLM provider is slow or unresponsive, the call is cancelled and an `ErrorData` is returned to the server.

#### Rate Limiting

Each server is limited to `max_rpm` sampling requests per minute (default: 10). Exceeding this limit returns a structured `ErrorData` to the server. This prevents a runaway or malicious server from draining LLM credits.

```yaml
# Increase rate limit for a trusted high-throughput server
sampling:
  max_rpm: 30
```

#### Token Cap

The `max_tokens_cap` config (default: 4096) limits how many tokens a server can request per sampling call. Even if a server requests `maxTokens: 100000`, it will be capped to the configured limit.

#### Credential Stripping

LLM responses are sanitized through `_sanitize_error()` before being returned to the MCP server, removing any credential-like patterns (API keys, tokens, passwords) that the LLM might accidentally include.

#### Typed Error Responses

All error conditions (rate limit, timeout, no provider, LLM errors) return structured `ErrorData` objects per the MCP specification, rather than raw exceptions. This gives servers a predictable error format they can handle gracefully.

#### Backward Compatibility

Sampling types (`CreateMessageResult`, `TextContent`, `ErrorData`) are imported in a separate `try/except` block. If the installed MCP SDK version doesn't have these types, sampling is silently disabled while all other MCP features continue working normally.

#### Human-in-the-Loop Considerations

The MCP specification recommends that clients provide human review of sampling requests. Currently, Hermes Agent auto-approves sampling within the configured safety limits. For untrusted servers, the recommended approach is to disable sampling entirely:

```yaml
mcp_servers:
  untrusted:
    command: "npx"
    args: ["-y", "untrusted-server"]
    sampling:
      enabled: false    # Server cannot make LLM requests
```

### Tool Use in Sampling

MCP servers can include `tools` and `toolChoice` in their sampling requests, enabling multi-turn tool-augmented workflows within a single sampling session.

#### How Tool Use Works

1. The server sends a `sampling/createMessage` request with `tools` (tool definitions) and optionally `toolChoice` (auto/required/none)
2. The sampling callback forwards these to the LLM as OpenAI-compatible function definitions
3. If the LLM responds with `tool_calls`, the callback returns a `CreateMessageResult` with `stopReason: "toolUse"` and tool use content blocks
4. The server executes the tools and sends another sampling request with the results
5. This loop continues until the LLM returns a text response or the `max_tool_rounds` limit is reached

#### Tool Loop Governance

To prevent runaway tool loops (a server and LLM endlessly bouncing tool calls), the `max_tool_rounds` config limits consecutive tool use rounds per server:

```yaml
sampling:
  max_tool_rounds: 5   # default: 5 consecutive tool rounds max
```

- **max_tool_rounds: 5** (default) — allows up to 5 consecutive rounds where the LLM requests tool use
- **max_tool_rounds: 0** — disables tool loops entirely; tool use responses are rejected
- The counter resets when the LLM returns a normal text response (endTurn)
- Exceeding the limit returns an `ErrorData` to the server

#### Content Block Conversion

The sampling callback handles all MCP content block types:

| MCP Content Block | OpenAI Format |
|------------------|---------------|
| `TextContent` | `{"role": "...", "content": "text"}` |
| `ImageContent` | `{"type": "image_url", ...}` |
| `ToolUseContent` | `{"tool_calls": [...]}` on assistant message |
| `ToolResultContent` | `{"role": "tool", "tool_call_id": "..."}` |

### Per-Server Sampling Policy

Different MCP servers have different trust levels. Use per-server sampling config to enforce appropriate policies:

```yaml
mcp_servers:
  # Trusted internal server: full access
  internal_analytics:
    command: "npx"
    args: ["-y", "analytics-server"]
    sampling:
      enabled: true
      max_rpm: 30
      max_tool_rounds: 10
      log_level: "debug"        # verbose audit trail

  # Semi-trusted: restricted model access
  community_server:
    command: "npx"
    args: ["-y", "community-server"]
    sampling:
      enabled: true
      allowed_models: ["gemini-3-flash"]  # only cheap models
      max_tool_rounds: 3
      max_tokens_cap: 2048

  # Untrusted: no sampling at all
  experimental:
    command: "npx"
    args: ["-y", "experimental-server"]
    sampling:
      enabled: false
```

#### Audit Metrics

Each server's sampling activity is tracked with structured metrics exposed via `get_mcp_status()`:

- **requests**: total successful sampling requests
- **errors**: total failed requests (rate limit, timeout, model blocked, etc.)
- **tokens_used**: total tokens consumed across all sampling calls
- **tool_use_count**: total responses where the LLM requested tool use

### Troubleshooting

**"No LLM provider available for sampling"**

No auxiliary LLM client is configured. Set up one of:
- `OPENROUTER_API_KEY` environment variable
- Nous Portal authentication (`hermes auth`)
- Custom endpoint (`OPENAI_BASE_URL` + `OPENAI_API_KEY`)

**"Sampling rate limit exceeded"**

The server is making too many sampling requests. Increase `max_rpm` if this is expected behavior:

```yaml
sampling:
  max_rpm: 30   # Increase from default 10
```

**"Sampling LLM call timed out"**

The LLM provider is too slow. Increase the timeout:

```yaml
sampling:
  timeout: 60   # Increase from default 30
```

**Server receives error responses**

Check the Hermes logs (`hermes --verbose`) for sampling request/response details. Common issues:
- LLM API errors (auth, quota)
- Invalid message format from the server
- Model not available
