# Platform Quirks

Reference for `create-agent-harness` Phase 3 — platform-specific harness directories and MCP configuration gotchas.

## Platform directories

Generate these only for platforms the repo explicitly targets. Each mirrors `.claude/` structure where needed:

| Platform | Directory | Key files |
|---|---|---|
| OpenCode | `.opencode/` | `skills/`, `hooks/`, `config.json` (MCP under `mcp` key, not `mcpServers`) |
| Cursor | `.cursor/` | `skills/`, `hooks/`, `mcp.json` (MCP under `mcpServers` key) |
| Gemini CLI | `.gemini/` | `skills/`, `hooks/`, `settings.json`, `config/mcp_config.json` |
| Antigravity IDE | `.gemini/` | Same as Gemini CLI (shared directory) |
| Antigravity CLI (agy) | `.gemini/antigravity-cli/` | `skills/`, `hooks/` (separate from IDE) |

## MCP config gotchas

| Client / platform | MCP key | Notes |
|---|---|---|
| Claude Code | `.mcp.json` or `mcpServers` in `.claude/settings.json` | Uses `mcpServers` array |
| Devin Desktop | `.devin/mcp.json` | Uses `serverUrl` (not `url`) for remote servers |
| OpenCode | `.opencode/config.json` | MCP under `mcp` key, not `mcpServers` |
| Cursor | `.cursor/mcp.json` | MCP under `mcpServers` key |
| Gemini CLI | `.gemini/config/mcp_config.json` | Follow Google's MCP schema |
| Antigravity IDE | `.gemini/config/mcp_config.json` | Shared with Gemini CLI |

For full details on a specific MCP setup, see the corresponding skill:

- `composio-mcp`
- `notebooklm-mcp`

## Devin CLI import

```json
{
  "read_config_from": { "claude": true }
}
```

> Do not duplicate `permissions` or `hooks` in `.devin/config.json` — the source of truth is `.claude/settings.json`. Do not create `AGENTS.md`, `DEVIN.md` or `.devin/agents/` as source-of-truth files.

> ⚠️ `read_config_from: { claude: true }` is REQUIRED — without it, Devin CLI will not import Claude Code's rules, skills, and subagents.
