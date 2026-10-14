# NotebookLM MCP — per-platform config reference

The server binary is `notebooklm-mcp` (stdio by default). Recommended server name: `notebooklm-mcp` (or `gemini-notebook-mcp` to avoid clashing with legacy servers).

## Platform config matrix

| Platform | Config file | Root key | Stdio command field | Env field | Notes |
|----------|-------------|----------|---------------------|-----------|-------|
| Claude Code | `~/.claude.json` | `mcpServers` | `command` + `args` | `env` | `type: "stdio"` |
| Claude Desktop | `claude_desktop_config.json` | `mcpServers` | `command` + `args` | `env` | — |
| Cursor | `~/.cursor/mcp.json` | `mcpServers` | `command` + `args` | `env` | `type: "stdio"` required |
| Devin CLI | `~/.config/devin/mcp_config.json` | `mcpServers` | `command` + `args` | `env` | `devin mcp add` CLI |
| Devin Desktop | `~/.devin/mcp_config.json` | `mcpServers` | `command` + `args` | `env` | — |
| OpenCode | `~/.config/opencode/opencode.json` | **`mcp`** | **`command` (single array)** | **`environment`** | `type: "local"` |
| Antigravity IDE/CLI | `~/.gemini/config/mcp_config.json` | `mcpServers` | `command` + `args` | `env` | Cache in `~/.gemini/antigravity-cli/mcp/` |
| OpenClaw | OpenClaw config | **`mcp.servers`** | `command` + `args` | `env` | `openclaw mcp add` CLI |

> **Critical gotchas:**
> - **OpenCode** uses `mcp` (not `mcpServers`) as the root key, `environment` (not `env`) for env vars, `type: "local"` on every entry, and `command` as a single array (binary + args merged — no separate `args` field).
> - **OpenClaw** uses `mcp.servers` as the root key and is managed via `openclaw mcp add/set` CLI commands.
> - **Antigravity** caches MCP servers in `~/.gemini/antigravity{,-ide,-cli}/mcp/` — remove the cache dir to fully uninstall.

---

## Claude Code

File: `~/.claude.json`

```json
{
  "mcpServers": {
    "notebooklm-mcp": {
      "type": "stdio",
      "command": "notebooklm-mcp"
    }
  }
}
```

Or via the CLI:

```bash
nlm setup add claude-code
# or
claude mcp add --scope user notebooklm-mcp notebooklm-mcp
```

## Claude Desktop

macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "notebooklm-mcp": {
      "command": "notebooklm-mcp"
    }
  }
}
```

Or: `nlm setup add claude-desktop`

## Cursor

File: `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "notebooklm-mcp": {
      "type": "stdio",
      "command": "notebooklm-mcp"
    }
  }
}
```

Or: `nlm setup add cursor`

## Devin CLI

File: `~/.config/devin/mcp_config.json` (v3000.3+; older versions use `mcpServers` key in `~/.config/devin/config.json` — migrated automatically on startup)

> **Note:** The Devin CLI config uses `nlm mcp start` (the wrapper) rather than the bare `notebooklm-mcp` binary. Both work; `nlm mcp start` ensures the profile/auth layer is initialized.

```json
{
  "mcpServers": {
    "notebooklm-mcp": {
      "command": "nlm",
      "args": ["mcp", "start"]
    }
  }
}
```

Or via the CLI:

```bash
devin mcp add --scope user notebooklm-mcp -- nlm mcp start
```

## Devin Desktop

File: `~/.devin/mcp_config.json`

```json
{
  "mcpServers": {
    "notebooklm-mcp": {
      "command": "nlm",
      "args": ["mcp", "start"]
    }
  }
}
```

## OpenCode

File: `~/.config/opencode/opencode.json` (user scope) or `opencode.json` (project scope)

> **Note:** OpenCode uses `mcp` (not `mcpServers`) as the root key, `environment` (not `env`) for env vars, and `command` as a single array (binary + args merged). `type: "local"` for stdio.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "notebooklm-mcp": {
      "type": "local",
      "command": ["nlm", "mcp", "start"],
      "enabled": true
    }
  }
}
```

> **OpenCode v2** nests under `mcp.servers`:
> ```json
> { "mcp": { "servers": { "notebooklm-mcp": { "type": "local", "command": ["nlm", "mcp", "start"] } } } }
> ```

## Antigravity IDE / Antigravity CLI (agy)

File: `~/.gemini/config/mcp_config.json` (global) or `.agents/mcp_config.json` (workspace)

```json
{
  "mcpServers": {
    "notebooklm-mcp": {
      "command": "nlm",
      "args": ["mcp", "start"]
    }
  }
}
```

After editing, Antigravity caches MCP servers in:
- `~/.gemini/antigravity/mcp/notebooklm-mcp` (AGY)
- `~/.gemini/antigravity-ide/mcp/notebooklm-mcp` (AGY IDE)
- `~/.gemini/antigravity-cli/mcp/notebooklm-mcp` (AGY CLI)

To fully uninstall, remove the config entry **and** delete the cached directory:

```bash
rm -rf ~/.gemini/antigravity*/mcp/notebooklm-mcp
```

## OpenClaw

OpenClaw manages MCP servers under `mcp.servers` in its config, via the `openclaw mcp` CLI or the Control UI (Settings → MCP).

### Via CLI

```bash
openclaw mcp add notebooklm-mcp --transport stdio --command nlm --args mcp,start
```

### Via config file

```json5
{
  mcp: {
    servers: {
      "notebooklm-mcp": {
        command: "nlm",
        args: ["mcp", "start"]
      }
    }
  }
}
```

### Via Control UI

1. Open the Control UI and go to **Settings → MCP**.
2. Under **Configured servers**, select **Add server**.
3. Name it `notebooklm-mcp`, pick **Stdio**.
4. Enter command `nlm` with args `mcp start`.
5. Select **Add server**.

Verify:

```bash
openclaw mcp status --verbose
openclaw mcp probe notebooklm-mcp   # live connection test
```

### OpenClaw browser CDP for NotebookLM auth

OpenClaw runs a managed browser with CDP on port 18800 by default. This is the recommended headless auth path for NotebookLM — `nlm` can read cookies from the OpenClaw browser session without launching a second browser:

```bash
nlm login --provider openclaw --cdp-url http://127.0.0.1:18800
```

If you configured a custom OpenClaw browser profile with a different CDP port:

```bash
# Check your OpenClaw browser config
openclaw config get browser.profiles

# Use the matching CDP port
nlm login --provider openclaw --cdp-url http://127.0.0.1:<port>
```

---

## HTTP / SSE transport (remote)

```bash
notebooklm-mcp --transport http --port 8000
notebooklm-mcp --transport sse  --port 8000
```

Config (HTTP) — Claude Code / Cursor / Devin CLI:

```json
{
  "mcpServers": {
    "notebooklm-mcp": {
      "type": "http",
      "url": "http://your-server:8000/mcp"
    }
  }
}
```

> **Devin Desktop** and **Antigravity** use `serverUrl` instead of `url`:
> ```json
> { "mcpServers": { "notebooklm-mcp": { "serverUrl": "http://your-server:8000/mcp" } } }
> ```

> **OpenCode** uses `type: "remote"`:
> ```json
> { "mcp": { "notebooklm-mcp": { "type": "remote", "url": "http://your-server:8000/mcp", "enabled": true } } }
> ```

> **Warning:** HTTP transport does not provide HTTPS, caller authentication, per-user NotebookLM accounts, or remote file transfer. Do not expose it publicly without a reverse proxy adding TLS + auth.

## Env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `NOTEBOOKLM_MCP_TRANSPORT` | stdio | Transport type (stdio/http/sse) |
| `NOTEBOOKLM_MCP_PORT` | 8000 | HTTP/SSE port |
| `NOTEBOOKLM_MCP_DEBUG` | false | Enable verbose logging |
| `NOTEBOOKLM_HL` | en | Interface language / locale (e.g. `pt-BR`, `es-419`) |
| `NOTEBOOKLM_QUERY_TIMEOUT` | — | Query timeout (seconds) |
| `NOTEBOOKLM_BASE_URL` | `https://notebooklm.google.com` | Override for Enterprise/Workspace |

Pass env vars in the config (example for Claude Code):

```json
{
  "mcpServers": {
    "notebooklm-mcp": {
      "type": "stdio",
      "command": "nlm",
      "args": ["mcp", "start"],
      "env": {
        "NOTEBOOKLM_HL": "pt-BR",
        "NOTEBOOKLM_MCP_DEBUG": "true"
      }
    }
  }
}
```

> **OpenCode** uses `environment` (not `env`):
> ```json
> { "mcp": { "notebooklm-mcp": { "type": "local", "command": ["nlm", "mcp", "start"], "environment": { "NOTEBOOKLM_HL": "pt-BR" } } } }
> ```

## Removing the server

```bash
# Per-platform CLI helpers
nlm setup remove claude-code
nlm setup remove claude-desktop
nlm setup remove cursor
nlm setup remove gemini
# etc.

# Or via the platform's own CLI
devin mcp remove notebooklm-mcp
claude mcp remove notebooklm-mcp
openclaw mcp unset notebooklm-mcp

# Antigravity: also clear the cache
rm -rf ~/.gemini/antigravity*/mcp/notebooklm-mcp
```
