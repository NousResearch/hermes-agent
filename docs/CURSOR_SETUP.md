# Hermes + Cursor Integration Setup

## ✅ What's Been Done

1. **Hermes environment ready**
   - venv: `~/.hermes/hermes-agent/.venv/`
   - Dependencies: all installed
   - MCP package: installed

2. **MCP launch script created**
   - Script: `hermes-mcp-serve`
   - Symlink from setup: `~/.local/bin/hermes-mcp-serve`
   - Auto-activates venv, starts MCP server

3. **MCP config template created**
   - Location: `.cursor/mcp.json`

---

## 📋 Next Steps: Configure Cursor

### Step 1: Open Cursor Settings

In Cursor IDE:
- macOS: `Cursor → Settings → Settings` or `Cmd+,`
- Linux: `File → Preferences → Settings`
- Windows: `File → Preferences → Settings`

### Step 2: Add Hermes MCP Server

Search for **"MCP"** in settings. 

Under **MCP > Servers**, add this JSON:

```json
{
  "hermes": {
    "command": "${workspaceFolder}/hermes-mcp-serve",
    "args": []
  }
}
```

**Or** if Cursor uses the newer format, go to Settings JSON and add:

```json
{
  "mcpServers": {
    "hermes": {
      "command": "${workspaceFolder}/hermes-mcp-serve",
      "args": []
    }
  }
}
```

### Step 3: Restart Cursor

Close and reopen Cursor completely.

### Step 4: Verify Connection

1. Open Cursor Composer
2. Look for the "Hermes" MCP icon/indicator in the bottom toolbar
3. Should show **Connected** or a checkmark
4. If it says "Building MCP" or loading, wait 10-30s for venv startup

---

## 🧪 Manual Testing (Optional)

Test the MCP server directly:

```bash
./hermes-mcp-serve
```

Should start with no errors and wait for requests. Press `Ctrl+C` to stop.

---

## 🎯 What You Get in Cursor

Once connected, Cursor Composer can call **17 MCP tools** from Hermes (see `website/docs/user-guide/features/mcp.md`):

### Messaging (10 tools)
- List conversations and read message history across connected platforms
- Poll or wait for live events; send messages (gateway must be running for sends)
- List channels and respond to approval prompts seen by the bridge

### Skills & knowledge (7 tools, read-only)
- List/read agent `SOUL.md` files and repo skills
- Browse agent registry, knowledge-layer artifacts, learnings, and artifacts

This MCP bridge does **not** expose the full interactive Hermes agent tool surface (terminal, browser, delegation, etc.). For editor-native Hermes as the coding agent, use `hermes acp` instead (see ACP docs).

### Example Prompt in Cursor

```
Using Hermes memory and skills, create a GitHub workflow for my project.
Then use the GitHub PR workflow skill to open a PR with the changes.
```

Cursor will access Hermes' tools and skills to execute this end-to-end.

---

## 🔗 Integration Points

- **Cursor Composer**: Full access to all Hermes tools
- **Cursor Editor Tools**: Can invoke Hermes for specific tasks
- **Hermes from Cursor**: Run complex multi-step workflows
- **Persistent State**: Hermes memory carries across Cursor sessions

---

## ❌ Troubleshooting

### "MCP server failed to start"
- Check: `hermes-mcp-serve` is executable
  ```bash
  ls -l hermes-mcp-serve
  ```
- Check venv exists: `ls -la .venv/ venv/`
- Try manual start: `./hermes-mcp-serve`

### "Connection refused / timeout"
- Hermes MCP server takes 5-10s to start on first run (venv warmup)
- Wait longer before refreshing
- Check Cursor logs: `Help → Toggle Developer Tools`

### "Command not found"
- Use the project path: `${workspaceFolder}/hermes-mcp-serve`
- Or run `./setup-hermes.sh` and add `~/.local/bin` to PATH

### "Unknown tools/skills"
- Refresh Cursor MCP connection: 
  - Settings → MCP → click refresh icon next to Hermes
  - Or restart Cursor

---

## 📁 Files Created

```
hermes-mcp-serve                 # Launch script
.cursor/mcp.json                 # Project-level Cursor MCP config
cursor-mcp-config.json           # Copyable config template
docs/CURSOR_SETUP.md             # This file
```

---

## 🚀 Done!

Your Cursor is now ready to use Hermes as an MCP server. Start asking complex questions in Composer — Cursor will route them through Hermes' full toolkit.
