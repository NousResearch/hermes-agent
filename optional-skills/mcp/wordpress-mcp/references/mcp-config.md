# WordPress MCP — per-platform config reference

Three MCP servers can be configured. Each has its own endpoint and auth.

| Server | Endpoint path | Auth header | Transport |
|--------|---------------|-------------|-----------|
| `wordpress-mcp` (mcp-adapter) | `/wp-json/mcp/mcp-adapter-default-server` | `Authorization: Basic <base64(user:app_password)>` | HTTP + STDIO |
| `wordpress-ai-engine` (AI Engine) | `/wp-json/mcp/v1/http` | `Authorization: Bearer <token>` | HTTP only |
| `wordpress-ultimate` (wp-mcp-ultimate) | `/wp-json/mcp-ultimate/v1` | `Authorization: Basic <base64(user:app_password)>` | HTTP only |

Replace `https://yourdomain.com` with your actual WordPress URL.

## Platform config matrix

| Platform | Config file | Root key | URL field | Auth field | Transport field |
|----------|-------------|----------|-----------|------------|-----------------|
| Claude Code | `~/.claude.json` | `mcpServers` | `url` | `headers` | `type: "http"` |
| Devin CLI | `~/.config/devin/mcp_config.json` | `mcpServers` | `url` | `headers` | — (HTTP auto) |
| OpenCode | `~/.config/opencode/opencode.json` | `mcp` | `url` | `headers` | `type: "remote"` |
| Gemini CLI | `~/.gemini/settings.json` | `mcpServers` | **`httpUrl`** | `headers` | — |
| AGY | `~/.gemini/antigravity-cli/settings.json` | `mcpServers` | **`httpUrl`** | `headers` | — |
| Codex | `~/.codex/config.toml` | `[mcp_servers.<name>]` | `url` | sub-table `headers` | — |
| OpenClaw | `~/.openclaw/openclaw.json` | `mcp.servers` | `url` | `--header` flag | `transport: "streamable-http"` |

> **Critical gotchas:**
> - **Gemini CLI** and **AGY** use `httpUrl` (not `url`) for HTTP servers. Using `url` causes the server to be silently ignored.
> - **OpenCode** uses `mcp` (not `mcpServers`) as the root key.
> - **Codex** stores headers in a TOML sub-table `[mcp_servers.<name>.headers]`, not inline.
> - **OpenClaw** is managed via `openclaw mcp add` CLI, not JSON patching.

---

## Claude Code

File: `~/.claude.json`

```json
{
  "mcpServers": {
    "wordpress-mcp": {
      "type": "http",
      "url": "https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server",
      "headers": {
        "Authorization": "Basic <base64>"
      }
    },
    "wordpress-ai-engine": {
      "type": "http",
      "url": "https://yourdomain.com/wp-json/mcp/v1/http",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    },
    "wordpress-ultimate": {
      "type": "http",
      "url": "https://yourdomain.com/wp-json/mcp-ultimate/v1",
      "headers": {
        "Authorization": "Basic <base64>"
      }
    }
  }
}
```

> For `wordpress-ultimate`, the Basic Auth value is the same format as `wordpress-mcp` (Application Password).

---

## Devin CLI

File: `~/.config/devin/mcp_config.json`

```json
{
  "mcpServers": {
    "wordpress-mcp": {
      "url": "https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server",
      "headers": {
        "Authorization": "Basic <base64>"
      }
    },
    "wordpress-ai-engine": {
      "url": "https://yourdomain.com/wp-json/mcp/v1/http",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

Or via CLI:

```bash
devin mcp add -s user wordpress-mcp "https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server" \
  --header "Authorization: Basic <base64>"

devin mcp add -s user wordpress-ai-engine "https://yourdomain.com/wp-json/mcp/v1/http" \
  --header "Authorization: Bearer <token>"
```

---

## OpenCode

File: `~/.config/opencode/opencode.json`

```json
{
  "mcp": {
    "wordpress-mcp": {
      "type": "remote",
      "url": "https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server",
      "headers": {
        "Authorization": "Basic <base64>"
      },
      "enabled": true
    },
    "wordpress-ai-engine": {
      "type": "remote",
      "url": "https://yourdomain.com/wp-json/mcp/v1/http",
      "headers": {
        "Authorization": "Bearer <token>"
      },
      "enabled": true
    }
  }
}
```

Or via CLI:

```bash
opencode mcp add wordpress-mcp \
  --url "https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server" \
  --header "Authorization=Basic <base64>"

opencode mcp add wordpress-ai-engine \
  --url "https://yourdomain.com/wp-json/mcp/v1/http" \
  --header "Authorization=Bearer <token>"
```

---

## Gemini CLI

File: `~/.gemini/settings.json`

```json
{
  "mcpServers": {
    "wordpress-mcp": {
      "httpUrl": "https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server",
      "headers": {
        "Authorization": "Basic <base64>"
      }
    },
    "wordpress-ai-engine": {
      "httpUrl": "https://yourdomain.com/wp-json/mcp/v1/http",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

> **Gotcha:** Gemini uses `httpUrl`, NOT `url`. Using `url` will silently fail.

---

## AGY (Antigravity CLI)

File: `~/.gemini/antigravity-cli/settings.json`

```json
{
  "mcpServers": {
    "wordpress-mcp": {
      "httpUrl": "https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server",
      "headers": {
        "Authorization": "Basic <base64>"
      }
    },
    "wordpress-ai-engine": {
      "httpUrl": "https://yourdomain.com/wp-json/mcp/v1/http",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

> **Gotcha:** Same as Gemini — use `httpUrl`, not `url`.

---

## Codex

File: `~/.codex/config.toml`

```toml
[mcp_servers.wordpress-mcp]
url = "https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server"

[mcp_servers.wordpress-mcp.headers]
Authorization = "Basic <base64>"

[mcp_servers.wordpress-ai-engine]
url = "https://yourdomain.com/wp-json/mcp/v1/http"

[mcp_servers.wordpress-ai-engine.headers]
Authorization = "Bearer <token>"
```

Or via CLI (then add headers manually):

```bash
codex mcp add wordpress-mcp --url "https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server"
codex mcp add wordpress-ai-engine --url "https://yourdomain.com/wp-json/mcp/v1/http"
# Then edit ~/.codex/config.toml to add [mcp_servers.<name>.headers] sub-tables
```

> **Gotcha:** Codex does not support `--header` on the CLI. Headers must be added as a TOML sub-table.

---

## OpenClaw

Managed via CLI (not JSON patching):

```bash
openclaw mcp add wordpress-mcp \
  --url "https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server" \
  --header "Authorization=Basic <base64>" \
  --transport streamable-http \
  --no-probe

openclaw mcp add wordpress-ai-engine \
  --url "https://yourdomain.com/wp-json/mcp/v1/http" \
  --header "Authorization=Bearer <token>" \
  --transport streamable-http \
  --no-probe
```

Config is saved to `~/.openclaw/openclaw.json`.

---

## Generating the Basic Auth value

```bash
echo -n "admin:xxxx xxxx xxxx xxxx xxxx xxxx" | base64
# Output: YWRtaW46eHh4eCB4eHh4IHh4eHggeHh4eCB4eHh4IHh4eHg=
```

Use this as the `Authorization: Basic <value>` header.

---

## STDIO transport (mcp-adapter only, local sites)

The mcp-adapter supports STDIO via WP-CLI. This works on all MCP clients that support `command`-based servers (Claude Code, Claude Desktop, Cursor, Devin, OpenCode, Gemini CLI, Codex).

### Direct STDIO (wp binary on same machine)

```jsonc
// Claude Code / Cursor / Devin / OpenCode
{
  "mcpServers": {
    "wordpress-stdio": {
      "command": "wp",
      "args": [
        "--path=/path/to/wordpress",
        "mcp-adapter",
        "serve",
        "--server=mcp-adapter-default-server",
        "--user=admin"
      ]
    }
  }
}
```

```toml
# Codex (~/.codex/config.toml)
[mcp_servers.wordpress-stdio]
command = "wp"
args = ["--path=/path/to/wordpress", "mcp-adapter", "serve", "--user=admin"]
```

> **Gemini CLI / AGY:** STDIO servers use `command` + `args` (not `httpUrl`). Gemini supports both `command` (STDIO) and `httpUrl` (HTTP) — use the right one for your transport.

### Remote HTTP via STDIO proxy (@automattic/mcp-wordpress-remote)

For MCP clients that only support STDIO but the WordPress site is remote:

```jsonc
{
  "mcpServers": {
    "wordpress-remote": {
      "command": "npx",
      "args": ["-y", "@automattic/mcp-wordpress-remote@latest"],
      "env": {
        "WP_API_URL": "https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server",
        "WP_API_USERNAME": "admin",
        "WP_API_PASSWORD": "xxxx xxxx xxxx xxxx xxxx xxxx",
        "LOG_FILE": "/tmp/mcp-adapter.log"
      }
    }
  }
}
```

```toml
# Codex
[mcp_servers.wordpress-remote]
command = "npx"
args = ["-y", "@automattic/mcp-wordpress-remote@latest"]

[mcp_servers.wordpress-remote.env]
WP_API_URL = "https://yourdomain.com/wp-json/mcp/mcp-adapter-default-server"
WP_API_USERNAME = "admin"
WP_API_PASSWORD = "xxxx xxxx xxxx xxxx xxxx xxxx"
```

> **Env substitution:** For platforms that support it (Claude Code, Devin, OpenCode), use `${WP_API_PASSWORD}` instead of the literal value. For Codex/Gemini, use literal values and protect the file with `chmod 600`.
