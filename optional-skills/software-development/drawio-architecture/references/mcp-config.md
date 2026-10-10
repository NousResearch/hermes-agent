# MCP Server Configuration — `@drawio/mcp`

> **Security note:** `npx` downloads and executes a remote package. Always pin an explicit version (`@drawio/mcp@<VERSION>`) verified on [npm](https://www.npmjs.com/package/@drawio/mcp) and confirm the publisher is `drawio` / JGraph. Do not run bare `npx -y @drawio/mcp` (without an explicit `@<VERSION>`) because it resolves to the latest remote version at runtime. The hosted endpoint `https://mcp.draw.io/mcp` sends diagram XML to the draw.io vendor's servers — only use it for non-sensitive diagrams and when you trust the vendor.

The official draw.io MCP server is an **stdio** server distributed via npm. The universal invocation is:

```bash
npx -y @drawio/mcp@<VERSION>
```

Replace `<VERSION>` with the stable release you have verified. Add it under `mcpServers.drawio` in your client's config. Below are the exact blocks per platform. For an automated, platform-detecting writer, use `scripts/setup_drawio_mcp.py` and ensure it pins the version.

### All supported platforms (quick reference)

| Target (`--target`) | Agent / IDE | Config file | JSON key |
|---------------------|-------------|-------------|----------|
| `claude-desktop` | Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` (mac) / `%APPDATA%\Claude\...` (win) / `~/.config/Claude/...` (linux) | `mcpServers` |
| `claude-code` | Claude Code | `~/.claude/settings.json` (or `claude mcp add drawio -- npx -y @drawio/mcp@<VERSION>`) | `mcpServers` |
| `vscode` | VS Code / GitHub Copilot | `.vscode/mcp.json` (workspace) or `~/.config/Code/User/mcp.json` (global) | `servers` |
| `cursor` | Cursor | `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project) | `mcpServers` |
| `opencode` | OpenCode | `~/.config/opencode/opencode.json` (or `opencode mcp add drawio -- npx -y @drawio/mcp@<VERSION>`) | `mcpServers` |
| `windsurf` | Windsurf | `~/.codeium/windsurf/mcp.json` (global) or `.windsurf/mcp.json` (project) | `mcpServers` |
| `devin` | Devin Desktop | `~/.devin/mcp.json` (linux) — also via Devin app/UI Integrations | `mcpServers` |
| `devin-cli` | Devin CLI | `~/.config/devin/mcp.json` — also via Devin app/UI | `mcpServers` |
| `agy` | AGY (Antigravity CLI) | `~/.gemini/antigravity-cli/mcp.json` — also via Antigravity UI | `mcpServers` |
| `antigravity` | Antigravity IDE | `~/.gemini/settings.json` (reuses Gemini CLI) | `mcpServers` |
| `gemini` | Gemini CLI | `~/.gemini/settings.json` | `mcpServers` |

The universal block for **every** stdio client (Claude, Cursor, OpenCode, Windsurf, Devin, AGY, Antigravity, Gemini, …):

```json
{
  "mcpServers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "@drawio/mcp@<VERSION>"]
    }
  }
}
```

> VS Code/Copilot uses `"servers"` instead of `"mcpServers"`; everything else uses `"mcpServers"`. Some agents (Devin, AGY) also expose MCP through their **UI/Integrations** settings — use the same stdio command there if a config file isn't picked up.

---

## Claude Desktop

Config file:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "@drawio/mcp@<VERSION>"]
    }
  }
}
```

---

## Claude Code

Command line:
```bash
claude mcp add drawio -- npx -y @drawio/mcp@<VERSION>
```

Or manually in `.claude/settings.json` (project or user):
```json
{
  "mcpServers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "@drawio/mcp@<VERSION>"]
    }
  }
}
```

---

## VS Code (GitHub Copilot)

Add to `.vscode/mcp.json` in your workspace (or run **MCP: Open User Configuration** for global). Then click **Start**, **trust** the server, switch Copilot Chat to **Agent mode**, and enable the drawio tools under **Configure Tools** (🔧).

```json
{
  "servers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "@drawio/mcp@<VERSION>"]
    }
  }
}
```

> Copilot does **not** yet support the MCP Apps protocol, so use this **stdio** server (opens diagrams in the browser), not the hosted `https://mcp.draw.io/mcp` endpoint. Other stdio clients (Windsurf, etc.) use the same shape.

---

## Cursor

One-click: `https://cursor.com/en/install-mcp?name=drawio&config=eyJjb21tYW5kIjoibnB4IiwiYXJncyI6WyIteSIsIkBkcmF3aW8vbWNwIl19`

Manual — global `~/.cursor/mcp.json` or project `.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "@drawio/mcp@<VERSION>"]
    }
  }
}
```
Enable under **Cursor Settings → MCP**. (Cursor *also* supports the hosted MCP App Server for inline rendering.)

---

## OpenCode

OpenCode reads MCP servers from its config (`opencode.json` / `opencode.jsonc` or `~/.config/opencode/`). Use the stdio shape:
```json
{
  "mcpServers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "@drawio/mcp@<VERSION>"]
    }
  }
}
```
(Equivalent to running `opencode mcp add drawio -- npx -y @drawio/mcp@<VERSION>` if your CLI supports it.)

---

## Windsurf / other stdio clients

```json
{
  "mcpServers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "@drawio/mcp@<VERSION>"]
    }
  }
}
```

---

## Self-hosted draw.io

To open diagrams in your own draw.io instance, set `DRAWIO_BASE_URL` (default `https://app.diagrams.net/`):

```json
{
  "mcpServers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "@drawio/mcp@<VERSION>"],
      "env": {
        "DRAWIO_BASE_URL": "https://drawio.example.com/"
      }
    }
  }
}
```

---

## Hosted MCP App Server (inline rendering)

> **Security note:** This remote endpoint is operated by the draw.io vendor. It receives your diagram XML and renders it inline. Only use it for non-sensitive diagrams and when you trust the vendor and the TLS channel.

No install — renders diagrams *inline* in chat via the MCP Apps protocol. Add the remote endpoint as a remote MCP server for Claude.ai / VS Code / Cursor:

```
https://mcp.draw.io/mcp
```

This is a **different** server type than the stdio one above (inline vs. opens-a-browser-tab). For sensitive or regulated architecture data, prefer the local stdio server with a pinned `@drawio/mcp@<VERSION>` package.

---

## Devin Desktop

Conventional file path (linux); Devin also configures MCP through its **app/UI → Integrations** — use the same stdio command there if a file isn't picked up.

```json
{
  "mcpServers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "@drawio/mcp@<VERSION>"]
    }
  }
}
```

- Linux: `~/.devin/mcp.json`
- macOS: `~/Library/Application Support/Devin/mcp.json`
- Windows: `%APPDATA%\Devin\mcp.json`

## Devin CLI

```json
{
  "mcpServers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "@drawio/mcp@<VERSION>"]
    }
  }
}
```

- Config path: `~/.config/devin/mcp.json`
- If your Devin build prefers the UI, add the server under Devin's MCP/Integrations settings with command `npx -y @drawio/mcp@<VERSION>`.

## AGY (Antigravity CLI)

```json
{
  "mcpServers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "@drawio/mcp@<VERSION>"]
    }
  }
}
```

- Config path: `~/.gemini/antigravity-cli/mcp.json`
- Also configurable via the Antigravity UI's MCP settings.

## Antigravity IDE / Gemini CLI

Both reuse the Gemini CLI settings file:

```json
{
  "mcpServers": {
    "drawio": {
      "command": "npx",
      "args": ["-y", "@drawio/mcp@<VERSION>"]
    }
  }
}
```

- Config path: `~/.gemini/settings.json`

---

## Verifying the install

1. Restart the client / re-read config.
2. Ask the agent: *"Use the draw.io MCP to make a simple flowchart of A → B → C."*
3. The agent should return a `app.diagrams.net/#create=…` URL or open the editor. If it instead writes a `.drawio` file by hand, add a system instruction: *"Always use the draw.io MCP tools to create diagrams."*

---

## Local CLI fallback (no MCP / need image deliverables)

Requires the draw.io **desktop** CLI. Resolve the binary name first:

```bash
if command -v drawio &>/dev/null; then DRAWIO="drawio"
elif command -v draw.io &>/dev/null; then DRAWIO="draw.io"
elif [ -f "/Applications/draw.io.app/Contents/MacOS/draw.io" ]; then DRAWIO="/Applications/draw.io.app/Contents/MacOS/draw.io"
elif grep -qi microsoft /proc/version 2>/dev/null && [ -f "/mnt/c/Program Files/draw.io/draw.io.exe" ]; then DRAWIO="/mnt/c/Program Files/draw.io/draw.io.exe"
else echo "drawio not found"; fi
```

Export:
```bash
# Preview (NO -e, width-capped under 2576px for vision self-check)
$DRAWIO -x -f png --width 2000 -o diagram.png input.drawio
# Final (WITH -e, double extension, then repair IEND)
$DRAWIO -x -f png -e -s 2 -o diagram.drawio.png input.drawio
python3 scripts/validate_drawio.py diagram.drawio.png --repair-iend
# SVG / PDF
$DRAWIO -x -f svg -e --embed-svg-images -o diagram.svg input.drawio
$DRAWIO -x -f pdf -e -o diagram.pdf input.drawio
```

Linux headless (server/CI):
```bash
export HOME=${HOME:-/tmp}
xvfb-run -a --server-args="-screen 0 1280x1024x24" \
  drawio -x -f png -e -s 2 -o diagram.drawio.png input.drawio --disable-gpu
# Running as root? Append --no-sandbox AT THE END of the command.
```

Browser fallback (no CLI): `python3 scripts/setup_drawio_mcp.py --viewer-url input.drawio` (read-only) or `--edit` (editable). The XML is deflate-compressed + base64 in the `#` fragment, so nothing is uploaded.

### Fallback chain

| Scenario | Behavior |
|----------|----------|
| MCP available | Prefer `open_drawio_xml` / `_mermaid` / `_csv` |
| CLI missing, Python available | Browser fallback URL |
| CLI missing, Python missing | Emit `.drawio` XML; tell user to open in draw.io manually |
| CLI crashes in macOS sandbox | Treat as unavailable; use browser fallback / XML only |
| Vision unavailable | Skip self-check; show the PNG directly |
| Linux headless export fails | `xvfb-run -a` → `--no-sandbox` (root) → `--disable-gpu` → `export HOME=/tmp` → drawio-renderer Docker |
