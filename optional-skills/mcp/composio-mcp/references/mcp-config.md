# Composio MCP — per-platform config reference

The MCP server is a streamable HTTP server at `https://connect.composio.dev/mcp`. It requires the `x-consumer-api-key` header containing a `ck_*` consumer key (obtained from the dashboard → Connect Settings → Sessions & API Key).

## Platform config matrix

| Platform | Config file | Root key | URL field | Env field | Transport field |
|----------|-------------|----------|-----------|-----------|-----------------|
| Claude Code | `~/.claude.json` | `mcpServers` | `url` | `env` | `type: "http"` |
| Claude Desktop | `claude_desktop_config.json` | `mcpServers` | `url` | `env` | — |
| Cursor | `~/.cursor/mcp.json` | `mcpServers` | `url` | `env` | — |
| Devin CLI | `~/.config/devin/mcp_config.json` | `mcpServers` | `url` | `env` | `transport: "http"` (optional) |
| Devin Desktop | `~/.devin/mcp_config.json` | `mcpServers` | **`serverUrl`** | `env` | — |
| OpenCode | `~/.config/opencode/opencode.json` | **`mcp`** | `url` | **`environment`** | `type: "remote"` |
| Antigravity IDE/CLI | `~/.gemini/config/mcp_config.json` | `mcpServers` | **`serverUrl`** | `env` | — |
| OpenClaw | OpenClaw config | **`mcp.servers`** | `url` | `env` | `transport: "streamable-http"` |

> **Critical gotchas:**
> - **Devin Desktop** and **Antigravity** use `serverUrl` (not `url`) for remote servers. Using `url` causes the server to be silently ignored.
> - **OpenCode** uses `mcp` (not `mcpServers`) as the root key, `environment` (not `env`) for env vars, and `type: "remote"`/`"local"` on every entry. A `command` is a single array — no separate `args`.
> - **OpenClaw** uses `mcp.servers` with `transport: "streamable-http"` and is managed via `openclaw mcp add/set` CLI commands.

---

## Claude Code

File: `~/.claude.json`

```json
{
  "mcpServers": {
    "composio": {
      "type": "http",
      "url": "https://connect.composio.dev/mcp",
      "headers": {
        "x-consumer-api-key": "ck_your_consumer_key"
      }
    }
  }
}
```

Or via the CLI:

```bash
claude mcp add --scope user --transport http composio https://connect.composio.dev/mcp \
  --header "x-consumer-api-key: ck_your_consumer_key"
```

## Claude Desktop

macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "composio": {
      "url": "https://connect.composio.dev/mcp",
      "headers": {
        "x-consumer-api-key": "ck_your_consumer_key"
      }
    }
  }
}
```

## Cursor

File: `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "composio": {
      "url": "https://connect.composio.dev/mcp",
      "headers": {
        "x-consumer-api-key": "ck_your_consumer_key"
      }
    }
  }
}
```

## Devin CLI

File: `~/.config/devin/mcp_config.json` (v3000.3+; older versions use `mcpServers` key in `~/.config/devin/config.json` — migrated automatically on startup)

```json
{
  "mcpServers": {
    "composio": {
      "url": "https://connect.composio.dev/mcp",
      "transport": "http",
      "headers": {
        "x-consumer-api-key": "ck_your_consumer_key"
      }
    }
  }
}
```

Or via the CLI:

```bash
devin mcp add --scope user --transport http composio https://connect.composio.dev/mcp \
  --header "x-consumer-api-key: ck_your_consumer_key"
```

## Devin Desktop

File: `~/.devin/mcp_config.json`

> **Note:** Devin Desktop uses `serverUrl` (not `url`) for remote HTTP servers. Using `url` causes the server to be silently ignored.

```json
{
  "mcpServers": {
    "composio": {
      "serverUrl": "https://connect.composio.dev/mcp",
      "headers": {
        "x-consumer-api-key": "ck_your_consumer_key"
      }
    }
  }
}
```

## OpenCode

File: `~/.config/opencode/opencode.json` (user scope) or `opencode.json` (project scope)

> **Note:** OpenCode uses `mcp` (not `mcpServers`) as the root key, `environment` (not `env`) for env vars, and requires `type: "remote"` on every remote entry.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "composio": {
      "type": "remote",
      "url": "https://connect.composio.dev/mcp",
      "headers": {
        "x-consumer-api-key": "ck_your_consumer_key"
      },
      "enabled": true
    }
  }
}
```

> **OpenCode v2** nests under `mcp.servers` instead of `mcp` directly:
> ```json
> { "mcp": { "servers": { "composio": { "type": "remote", "url": "...", "headers": {...} } } } }
> ```

## Antigravity IDE / Antigravity CLI (agy)

File: `~/.gemini/config/mcp_config.json` (global) or `.agents/mcp_config.json` (workspace)

> **Note:** Antigravity uses `serverUrl` (not `url`) for remote servers. Legacy `url`/`httpUrl` fields are not supported.

```json
{
  "mcpServers": {
    "composio": {
      "serverUrl": "https://connect.composio.dev/mcp",
      "headers": {
        "x-consumer-api-key": "ck_your_consumer_key"
      }
    }
  }
}
```

After editing, Antigravity caches MCP servers in:
- `~/.gemini/antigravity/mcp/composio` (AGY)
- `~/.gemini/antigravity-ide/mcp/composio` (AGY IDE)
- `~/.gemini/antigravity-cli/mcp/composio` (AGY CLI)

To fully uninstall, remove the config entry **and** delete the cached directory.

## OpenClaw

OpenClaw manages MCP servers under `mcp.servers` in its config, via the `openclaw mcp` CLI or the Control UI (Settings → MCP).

> **Note:** OpenClaw uses `transport: "streamable-http"` (not `type: "http"`) and `mcp.servers` (not `mcpServers`) as the root key.

### Via CLI

```bash
openclaw mcp add composio --transport streamable-http --url https://connect.composio.dev/mcp \
  --header "x-consumer-api-key: ck_your_consumer_key"
```

### Via config file

```json5
{
  mcp: {
    servers: {
      composio: {
        url: "https://connect.composio.dev/mcp",
        transport: "streamable-http",
        headers: {
          "x-consumer-api-key": "ck_your_consumer_key"
        }
      }
    }
  }
}
```

### Via Control UI

1. Open the Control UI and go to **Settings → MCP**.
2. Under **Configured servers**, select **Add server**.
3. Name it `composio`, pick **Streamable HTTP**.
4. Enter `https://connect.composio.dev/mcp`.
5. Add header `x-consumer-api-key: ck_your_consumer_key`.
6. Select **Add server**.

Verify:

```bash
openclaw mcp status --verbose
openclaw mcp probe composio   # live connection test
```

---

## Env-substituted variant (recommended)

To avoid committing the key, use `${COMPOSIO_CONSUMER_KEY}` and export it in your shell profile:

```bash
# ~/.bashrc or ~/.zshrc
export COMPOSIO_CONSUMER_KEY="ck_your_consumer_key"
```

Then in any config above:

```json
"headers": { "x-consumer-api-key": "${COMPOSIO_CONSUMER_KEY}" }
```

> **OpenCode** uses `{env:COMPOSIO_CONSUMER_KEY}` syntax (curly braces with `env:` prefix), not `${...}`:
> ```json
> "headers": { "x-consumer-api-key": "{env:COMPOSIO_CONSUMER_KEY}" }
> ```

## Optional: enforce API key at the org level

When `require_mcp_api_key` is enabled on the project, MCP requests must also carry the `ak_*` project API key in an `x-api-key` header:

```bash
curl -X PATCH "https://backend.composio.dev/api/v3/org/project/config" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $COMPOSIO_API_KEY" \
  -d '{"require_mcp_api_key": true}'
```

Config with both headers:

```json
"headers": {
  "x-consumer-api-key": "ck_your_consumer_key",
  "x-api-key": "ak_your_project_key"
}
```
