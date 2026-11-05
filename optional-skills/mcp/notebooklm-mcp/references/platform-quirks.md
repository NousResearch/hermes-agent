# MCP platform config quirks — cross-platform reference

Each MCP client platform has its own config format with subtle differences. Getting these wrong causes servers to be **silently ignored** (no error, just no tools). This reference documents every platform-specific quirk.

## Quick reference matrix

| Platform | Config file | Root key | Remote URL field | Stdio command | Env field | Transport marker |
|----------|-------------|----------|------------------|---------------|-----------|------------------|
| Claude Code | `~/.claude.json` | `mcpServers` | `url` | `command` + `args` | `env` | `type: "http"` / `"stdio"` |
| Claude Desktop | `claude_desktop_config.json` | `mcpServers` | `url` | `command` + `args` | `env` | — (inferred) |
| Cursor | `~/.cursor/mcp.json` | `mcpServers` | `url` | `command` + `args` | `env` | `type: "stdio"` required for stdio |
| Devin CLI | `~/.config/devin/mcp_config.json` | `mcpServers` | `url` | `command` + `args` | `env` | `transport: "http"` (optional) |
| Devin Desktop | `~/.devin/mcp_config.json` | `mcpServers` | **`serverUrl`** | `command` + `args` | `env` | — |
| OpenCode | `~/.config/opencode/opencode.json` | **`mcp`** | `url` | **`command` (single array)** | **`environment`** | `type: "remote"` / `"local"` |
| Antigravity IDE | `~/.gemini/config/mcp_config.json` | `mcpServers` | **`serverUrl`** | `command` + `args` | `env` | — |
| Antigravity CLI (agy) | `~/.gemini/config/mcp_config.json` | `mcpServers` | **`serverUrl`** | `command` + `args` | `env` | — |
| OpenClaw | OpenClaw config | **`mcp.servers`** | `url` + `transport` | `command` + `args` | `env` | `transport: "streamable-http"` / `"stdio"` |

## Critical gotchas (silent failure if wrong)

### 1. `serverUrl` vs `url` for remote servers

**Devin Desktop** and **Antigravity** (IDE + CLI) use `serverUrl` for remote HTTP/SSE servers. Using `url` causes the server to be **silently ignored** — no error, just no tools appear.

```json
// ✅ Devin Desktop / Antigravity
{ "serverUrl": "https://connect.composio.dev/mcp" }

// ❌ silently ignored
{ "url": "https://connect.composio.dev/mcp" }
```

All other platforms (Claude Code, Cursor, Devin CLI, OpenCode, OpenClaw) use `url`.

### 2. OpenCode: `mcp` not `mcpServers`, `environment` not `env`, `command` is an array

OpenCode is the most divergent platform:

```json
// ✅ OpenCode
{
  "mcp": {
    "my-server": {
      "type": "remote",
      "url": "https://...",
      "headers": { "x-api-key": "..." },
      "enabled": true
    }
  }
}

// ❌ silently ignored — wrong root key
{
  "mcpServers": { ... }
}
```

- Root key is `mcp` (not `mcpServers`). OpenCode v2 nests under `mcp.servers`.
- `type` is required on every entry: `"local"` (stdio) or `"remote"` (HTTP).
- `command` is a **single array** containing the binary and all args — no separate `args` field.
  - ✅ `"command": ["npx", "-y", "my-server"]`
  - ❌ `"command": "npx", "args": ["-y", "my-server"]`
- Env vars go in `environment` (not `env`). An `env` block copied from another client is silently ignored.
- `enabled` is a boolean (default true).

### 3. OpenClaw: `mcp.servers` with `transport` field

OpenClaw uses `mcp.servers` (dotted, not `mcpServers`) and `transport: "streamable-http"` (not `type: "http"`):

```json5
// ✅ OpenClaw
{
  mcp: {
    servers: {
      "my-server": {
        url: "https://...",
        transport: "streamable-http",
        headers: { "x-api-key": "..." }
      }
    }
  }
}
```

Managed via CLI: `openclaw mcp add/set/unset`. The `type: "http"` alias is normalized to `transport` by `openclaw mcp set` and `openclaw doctor --fix`.

### 4. Antigravity: MCP cache must be cleared on uninstall

Antigravity caches MCP servers in:
- `~/.gemini/antigravity/mcp/<name>` (AGY)
- `~/.gemini/antigravity-ide/mcp/<name>` (AGY IDE)
- `~/.gemini/antigravity-cli/mcp/<name>` (AGY CLI)

Removing the config entry is not enough — the cached directory must also be deleted:

```bash
rm -rf ~/.gemini/antigravity*/mcp/<server-name>
```

### 5. Devin CLI: config file location changed in v3000.3

- **v3000.3+**: `~/.config/devin/mcp_config.json` (dedicated file)
- **Older**: `mcpServers` key in `~/.config/devin/config.json` (migrated automatically on startup)

Both locations are checked. The `devin mcp add` CLI handles this transparently.

### 6. Cursor: `type: "stdio"` is required for stdio servers

Cursor requires an explicit `type` field on stdio servers:

```json
// ✅ Cursor stdio
{ "type": "stdio", "command": "npx", "args": ["-y", "my-server"] }

// ❌ may not be detected
{ "command": "npx", "args": ["-y", "my-server"] }
```

For remote servers, `type` is optional (inferred from the presence of `url`).

## Env var substitution syntax

| Platform | Syntax | Example |
|----------|--------|---------|
| Claude Code | `${VAR}` | `"${COMPOSIO_CONSUMER_KEY}"` |
| Claude Desktop | `${VAR}` | `"${COMPOSIO_CONSUMER_KEY}"` |
| Cursor | `${VAR}` | `"${COMPOSIO_CONSUMER_KEY}"` |
| Devin CLI | `${VAR}` | `"${COMPOSIO_CONSUMER_KEY}"` |
| Devin Desktop | `${VAR}` | `"${COMPOSIO_CONSUMER_KEY}"` |
| OpenCode | `{env:VAR}` | `"{env:COMPOSIO_CONSUMER_KEY}"` |
| Antigravity | `${VAR}` | `"${COMPOSIO_CONSUMER_KEY}"` |
| OpenClaw | `${VAR}` | `"${COMPOSIO_CONSUMER_KEY}"` |

> **OpenCode** is the outlier — it uses `{env:VAR}` (curly braces with `env:` prefix), not `${VAR}`.

## Per-platform CLI helpers

Most platforms offer a CLI to add/remove MCP servers without editing JSON:

| Platform | Add | Remove | List |
|----------|-----|--------|------|
| Claude Code | `claude mcp add` | `claude mcp remove` | `claude mcp list` |
| Cursor | (UI only or edit `mcp.json`) | (UI or edit file) | (UI) |
| Devin CLI | `devin mcp add` | `devin mcp remove` | `devin mcp list` |
| OpenCode | (edit `opencode.json`) | (edit file) | `opencode mcp list` |
| Antigravity | (Interactive MCP Manager or edit file) | (edit file + clear cache) | (UI) |
| OpenClaw | `openclaw mcp add` | `openclaw mcp unset` | `openclaw mcp list` |
| NotebookLM | `nlm setup add <platform>` | `nlm setup remove <platform>` | `nlm setup list` |

## OpenClaw browser CDP (for cookie-based auth)

OpenClaw runs a managed browser with Chrome DevTools Protocol (CDP) on port **18800** by default. This is the recommended headless auth path for NotebookLM — `nlm` can read cookies from the OpenClaw browser session without launching a second browser:

```bash
nlm login --provider openclaw --cdp-url http://127.0.0.1:18800
```

Custom OpenClaw browser profiles may use different CDP ports. Check with:

```bash
openclaw config get browser.profiles
# Look for cdpPort in each profile
```

Common OpenClaw browser profiles:

| Profile | Default CDP port | Use case |
|---------|------------------|----------|
| `openclaw` (default) | 18800 | Isolated OpenClaw browser |
| `work` | 18801 | Custom profile (example) |
| `user` | — | Existing Chrome session (attach-only) |
| `brave` | — | Existing Brave session (attach-only) |

## Sources

- [Devin CLI MCP configuration](https://docs.devin.ai/cli/extensibility/mcp/configuration)
- [Devin MCP server](https://docs.devin.ai/work-with-devin/devin-mcp) (serverUrl vs url)
- [Antigravity MCP docs](https://antigravity.google/docs/mcp/)
- [Antigravity CLI MCP](https://antigravity.google/docs/cli/mcp/)
- [Antigravity migration guide](https://antigravity.google/docs/cli/gcli-migration/)
- [OpenCode MCP servers](https://opencode.ai/docs/mcp-servers/)
- [OpenCode v2 MCP servers](https://opencode.ai/v2/docs/mcp-servers)
- [OpenClaw MCP tools](https://docs.openclaw.ai/tools/mcp)
- [OpenClaw configuration reference](https://docs.openclaw.ai/gateway/configuration-reference)
- [OpenClaw browser](https://docs.openclaw.ai/browser)
- [Cursor MCP docs](https://cursor.com/docs/mcp)
- [Composio Connect](https://docs.composio.dev/docs/composio-connect)
- [NotebookLM MCP guide](https://github.com/jacob-bd/notebooklm-mcp-cli/blob/main/docs/MCP_GUIDE.md)
