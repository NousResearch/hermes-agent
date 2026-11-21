# MCP platform config quirks — cross-platform reference

Each MCP client platform has its own config format with subtle differences. Getting these wrong causes servers to be **silently ignored** (no error, just no tools). This reference documents every platform-specific quirk for WordPress MCP.

## Quick reference matrix

| Platform | Config file | Root key | Remote URL field | Auth field | Transport marker |
|----------|-------------|----------|------------------|------------|------------------|
| Claude Code | `~/.claude.json` | `mcpServers` | `url` | `headers` | `type: "http"` |
| Devin CLI | `~/.config/devin/mcp_config.json` | `mcpServers` | `url` | `headers` | — (HTTP auto-detected) |
| OpenCode | `~/.config/opencode/opencode.json` | **`mcp`** | `url` | `headers` | `type: "remote"` |
| Gemini CLI | `~/.gemini/settings.json` | `mcpServers` | **`httpUrl`** | `headers` | — |
| AGY | `~/.gemini/antigravity-cli/settings.json` | `mcpServers` | **`httpUrl`** | `headers` | — |
| Codex | `~/.codex/config.toml` | `[mcp_servers.<name>]` | `url` | **sub-table `headers`** | — |
| OpenClaw | `~/.openclaw/openclaw.json` | `mcp.servers` | `url` | `--header` flag | `transport: "streamable-http"` |

## Critical gotchas (silent failure if wrong)

### 1. `httpUrl` vs `url` for HTTP servers (Gemini CLI / AGY)

**Gemini CLI** and **AGY** (Antigravity CLI) use `httpUrl` for remote HTTP/SSE servers. Using `url` causes the server to be **silently ignored** — no error, just no tools appear.

```json
// ✅ Gemini CLI / AGY
{ "httpUrl": "https://yourdomain.com/wp-json/mcp/v1/http" }

// ❌ silently ignored
{ "url": "https://yourdomain.com/wp-json/mcp/v1/http" }
```

All other platforms (Claude Code, Devin CLI, OpenCode, Codex, OpenClaw) use `url`.

### 2. OpenCode: `mcp` not `mcpServers`

OpenCode uses `mcp` as the root key (not `mcpServers`), and requires `type: "remote"` and `enabled: true` on every entry:

```json
// ✅ OpenCode
{
  "mcp": {
    "wordpress-ai-engine": {
      "type": "remote",
      "url": "https://yourdomain.com/wp-json/mcp/v1/http",
      "headers": { "Authorization": "Bearer <token>" },
      "enabled": true
    }
  }
}

// ❌ silently ignored (wrong root key)
{
  "mcpServers": { ... }
}
```

### 3. Codex: headers in TOML sub-table

Codex uses TOML, not JSON. Headers go in a separate sub-table, not inline:

```toml
# ✅ Codex
[mcp_servers.wordpress-ai-engine]
url = "https://yourdomain.com/wp-json/mcp/v1/http"

[mcp_servers.wordpress-ai-engine.headers]
Authorization = "Bearer <token>"
```

```toml
# ❌ invalid TOML (headers inline)
[mcp_servers.wordpress-ai-engine]
url = "https://yourdomain.com/wp-json/mcp/v1/http"
headers = { Authorization = "Bearer <token>" }  # may work but non-standard
```

The `codex mcp add` CLI does **not** support `--header`. You must add the headers sub-table manually after adding the server.

### 4. OpenClaw: CLI-managed, not JSON patching

OpenClaw config is at `~/.openclaw/openclaw.json` but should be managed via the `openclaw mcp add` CLI, not manual JSON editing:

```bash
# ✅ OpenClaw
openclaw mcp add wordpress-ai-engine \
  --url "https://yourdomain.com/wp-json/mcp/v1/http" \
  --header "Authorization=Bearer <token>" \
  --transport streamable-http \
  --no-probe
```

Note: OpenClaw uses `=` (not `:`) for header key-value in the CLI flag.

### 5. Devin CLI: `devin mcp add` is the cleanest path

Devin CLI supports both JSON editing and the `devin mcp add` CLI. The CLI is preferred:

```bash
devin mcp add -s user wordpress-mcp "https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server" \
  --header "Authorization: Basic <base64>"
```

`-s user` makes it global (all projects). Without `-s`, it's project-scoped.

### 6. Claude Code: `type: "http"` required

Claude Code requires `type: "http"` for HTTP servers (not inferred):

```json
// ✅ Claude Code
{
  "mcpServers": {
    "wordpress-ai-engine": {
      "type": "http",
      "url": "...",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}

// ❌ may be treated as stdio (no url field recognized)
{
  "mcpServers": {
    "wordpress-ai-engine": {
      "url": "...",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

## Session management: mcp-adapter vs AI Engine

| Server | Session required | Flow |
|--------|-----------------|------|
| mcp-adapter | **Yes** | `initialize` → capture `Mcp-Session-Id` header → `notifications/initialized` (HTTP 202) → `tools/list` with session header |
| AI Engine | **No** | Each request is independent — `initialize` then `tools/list` in separate calls, no session header needed |

> **Impact on MCP clients:** Most MCP clients (Claude Code, Devin, OpenCode, etc.) handle session management automatically. The manual curl flow requires the `Mcp-Session-Id` header for mcp-adapter. If you're testing with curl and `tools/list` returns empty, you likely skipped `notifications/initialized`.

## Env substitution by platform

| Platform | Syntax | Example |
|----------|--------|---------|
| Claude Code | `${VAR}` | `"Authorization": "Basic ${WP_BASIC_AUTH}"` |
| Devin CLI | `${VAR}` | `"Authorization": "Bearer ${WP_AI_ENGINE_TOKEN}"` |
| OpenCode | `{env:VAR}` | `"Authorization": "Bearer {env:WP_AI_ENGINE_TOKEN}"` |
| Gemini CLI | Not supported | Use literal values |
| AGY | Not supported | Use literal values |
| Codex | Not supported | Use literal values |
| OpenClaw | `${VAR}` | Via `--header` flag |

> For platforms without env substitution, the setup script writes literal values. Protect the config files with `chmod 600` and never commit them.
