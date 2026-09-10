# IDE Integration Guide

Connect Hermes Agent to your favorite IDE using MCP (Model Context Protocol) or ACP (Agent Client Protocol). Once connected, Hermes becomes your in-editor coding assistant with full terminal access, file operations, web search, and delegation.

## Quick Reference

| IDE | Protocol | Config File | Setup Effort |
|-----|----------|-------------|:---:|
| Cursor | MCP stdio | `~/.cursor/mcp.json` | ⚡ 2 min |
| Windsurf | MCP stdio | `~/.codeium/windsurf/mcp_config.json` | ⚡ 2 min |
| VS Code (Continue) | MCP stdio | `~/.continue/config.json` | ⚡ 3 min |
| VS Code (Cline) | MCP stdio | Cline Settings UI | ⚡ 2 min |
| Claude Code | MCP stdio | `claude mcp add` | ⚡ 1 min |
| JetBrains | ACP | IDE Plugin Settings | ⚡ 3 min |
| Zed | ACP | `zed: open settings` | ⚡ 2 min |
| VS Code (Native) | ACP | Extension | ⚡ 2 min |

> **Hermes binary location**: run `which hermes` to find your path. Common locations:
> - Linux/macOS (uv): `~/.local/bin/hermes`
> - macOS (Homebrew): `/opt/homebrew/bin/hermes`
> - Windows: `%USERPROFILE%\.local\bin\hermes.exe` or `%USERPROFILE%\AppData\Roaming\Python\Scripts\hermes.exe`
>
> Use the **full path** to avoid PATH resolution issues.

---

## Cursor

Cursor uses `~/.cursor/mcp.json` for MCP server configuration.

### Setup

Create or edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "hermes": {
      "command": "/Users/YOUR_USERNAME/.local/bin/hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

Replace `/Users/YOUR_USERNAME/.local/bin/hermes` with your actual path from `which hermes`.

### Troubleshooting

- **"hermes: command not found"** — use the full path from `which hermes`
- **MCP tools not showing** — restart Cursor (`Cmd+Shift+P` → "Reload Window")
- **Permission denied** — ensure `hermes` is executable: `chmod +x ~/.local/bin/hermes`

---

## Windsurf

Windsurf uses `~/.codeium/windsurf/mcp_config.json`.

### Setup

Create or edit `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "hermes": {
      "command": "/Users/YOUR_USERNAME/.local/bin/hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

### Troubleshooting

- **Tools not loading** — open Windsurf's MCP panel and click "Refresh"
- **Connection refused** — verify the full path is correct

---

## VS Code + Continue

[Continue](https://continue.dev) uses `~/.continue/config.json`.

### Setup

Add Hermes to the `mcpServers` array in `~/.continue/config.json`:

```json
{
  "models": [ /* your existing models */ ],
  "mcpServers": [
    {
      "name": "hermes",
      "command": "/Users/YOUR_USERNAME/.local/bin/hermes",
      "args": ["mcp", "serve"]
    }
  ]
}
```

### Using Hermes in Continue

1. Open Continue sidebar (`Cmd+L` / `Ctrl+L`)
2. Select a chat model
3. Hermes MCP tools are automatically available: ask to read files, run terminal commands, search the web

### Troubleshooting

- **"hermes not found"** — use the full absolute path from `which hermes`
- **MCP not connecting** — check Continue logs: `Cmd+Shift+P` → "Continue: View Logs"

---

## VS Code + Cline

[Cline](https://github.com/cline/cline) has a built-in MCP settings UI.

### Setup

1. Open Cline extension
2. Click the **MCP Servers** icon (plug icon in the Cline sidebar)
3. Click **"Installed"** tab
4. Click **"Configure MCP Servers"**
5. Add a new entry:

```json
{
  "mcpServers": {
    "hermes": {
      "command": "/Users/YOUR_USERNAME/.local/bin/hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

6. Click **"Done"** — Cline auto-reconnects

### Using Hermes in Cline

Cline will list Hermes tools alongside its built-in tools. You can ask Cline to use Hermes for terminal operations, file manipulation, or web searches — Cline delegates these to the Hermes MCP server transparently.

---

## Claude Code

Anthropic's [Claude Code](https://claude.ai/code) CLI supports MCP servers natively.

### Setup

```bash
# Add Hermes as an MCP server
claude mcp add --transport stdio hermes -- hermes mcp serve

# Or with full path:
claude mcp add --transport stdio hermes -- /Users/YOUR_USERNAME/.local/bin/hermes mcp serve
```

To verify:

```bash
claude mcp list
# Should show "hermes" with status "connected"
```

### Using Hermes in Claude Code

Once connected, tell Claude to use Hermes tools:

```
> Use the hermes terminal tool to run `npm test` in this project.
> Use hermes web_search to find the latest React documentation on Server Components.
```

### Troubleshooting

- **"Failed to connect"** — run `hermes mcp serve` in a terminal first to verify it works standalone
- **Remove and re-add**: `claude mcp remove hermes && claude mcp add --transport stdio hermes -- hermes mcp serve`
- **PATH issues**: use the full absolute path

---

## JetBrains IDEs (IntelliJ, PyCharm, WebStorm, GoLand)

Hermes supports [ACP (Agent Client Protocol)](https://github.com/agentclientprotocol/agent-client-protocol) for JetBrains integration.

### Setup

1. **Install the ACP plugin**:
   - Open Settings → Plugins → Marketplace
   - Search for "Agent Client Protocol" or "ACP"
   - Install and restart

2. **Configure Hermes as ACP server**:
   - Open Settings → Tools → Agent Client Protocol
   - Add new agent:
     - **Name**: Hermes
     - **Command**: `/Users/YOUR_USERNAME/.local/bin/hermes acp`
     - **Working Directory**: your project root

3. **Verify**: The Hermes agent should appear in the ACP sidebar with status "Connected"

### Features

- Chat with Hermes directly in the IDE
- Attach files, selections, and error messages as context
- Hermes can read project files, run tests, and suggest fixes

---

## Zed

[Zed](https://zed.dev) supports ACP natively.

### Setup

1. Open Zed Settings (`Cmd+,` / `Ctrl+,`)
2. Add to `settings.json`:

```json
{
  "agent_servers": {
    "Hermes": {
      "command": "/Users/YOUR_USERNAME/.local/bin/hermes",
      "args": ["acp"]
    }
  }
}
```

3. Open the agent panel (`Cmd+Shift+A`) — Hermes appears as an available agent

---

## VS Code (Native ACP Extension)

For native VS Code integration without Continue or Cline:

### Setup

1. Install the "Agent Client Protocol" extension from VS Code Marketplace
2. Open Settings (`Cmd+,`) → search "ACP"
3. Add agent configuration:

```json
{
  "acp.agents": [
    {
      "name": "Hermes",
      "command": "/Users/YOUR_USERNAME/.local/bin/hermes",
      "args": ["acp"]
    }
  ]
}
```

4. The Hermes agent panel opens with `Cmd+Shift+P` → "ACP: Open Chat"

---

## What Hermes Brings to Your IDE

When connected via MCP or ACP, Hermes provides these tools to your IDE's agent:

| Tool | Description |
|------|-------------|
| `terminal` | Run shell commands in a persistent environment |
| `read_file` | Read files with line numbers |
| `write_file` | Create or overwrite files |
| `patch` | Targeted find-and-replace edits |
| `search_files` | Search file contents (ripgrep) or find files by pattern |
| `web_search` | Real-time web search |
| `web_fetch` | Fetch and extract web page content |
| `delegate_task` | Spawn sub-agents for parallel work |
| `memory` | Persistent cross-session memory |
| `session_search` | Search past conversations |
| `image_generate` | AI image generation |
| `cronjob` | Schedule recurring tasks |

All tools work the same across every IDE — learn them once, use them everywhere.

---

## Security Notes

- **MCP mode is read-write** — Hermes can read, write, and execute files in your system. Only connect to trusted IDE sessions.
- **Environment isolation** — MCP subprocesses do NOT inherit your full shell environment; only safe baseline variables are passed.
- **Approval prompts** — dangerous commands (`rm -rf`, `git reset --hard`) trigger user approval before execution (configurable via `approvals.mode` in `~/.hermes/config.yaml`).

---

## Verifying Your Setup

To confirm Hermes MCP is working before connecting to an IDE:

```bash
# Test MCP server mode
hermes mcp serve --verbose

# Should output:
# MCP server starting on stdio...
# Discovering tools...
# Ready
```

Press `Ctrl+C` to stop.

For ACP mode:

```bash
# Test ACP server mode
hermes acp --check

# Should output:
# ACP dependencies OK
# Adapter imports OK
```
