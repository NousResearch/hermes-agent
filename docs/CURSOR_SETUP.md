# Hermes + Cursor Integration Setup

## ✅ What's Been Done

1. **Hermes environment ready**
   - venv: `~/.hermes/hermes-agent/.venv/`
   - Dependencies: all installed
   - MCP package: installed

2. **MCP launch script created**
   - Script: `~/.hermes/hermes-agent/hermes-mcp-serve`
   - Symlink: `~/.hermes/bin/hermes-mcp-serve`
   - Auto-activates venv, starts MCP server

3. **MCP config template created**
   - Location: `~/.hermes/hermes-agent/cursor-mcp-config.json`

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
    "command": "/home/arrenchulz/.hermes/bin/hermes-mcp-serve",
    "args": []
  }
}
```

**Or** if Cursor uses the newer format, go to Settings JSON and add:

```json
{
  "mcpServers": {
    "hermes": {
      "command": "/home/arrenchulz/.hermes/bin/hermes-mcp-serve",
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
~/.hermes/bin/hermes-mcp-serve
```

Should start with no errors and wait for requests. Press `Ctrl+C` to stop.

---

## 🎯 What You Get in Cursor

Once connected, use Cursor Composer with:

### Memory & Sessions
- Access persistent Hermes memory
- Browse past sessions
- Search conversation history

### Skills (45+ agents)
- Claude Code, GitHub workflows, Jupyter, Design tools
- Creative (ASCII art, video, infographics, etc.)
- MLOps (training, inference, vector DBs)
- Software Dev (debugging, testing, planning)
- And 30+ more categories

### Tools
- Terminal execution
- File read/write/patch
- Web browsing
- Vision/image analysis
- Todo management
- Custom skills

### Messaging
- Send to Telegram, Discord, Slack, etc.
- Route outputs to messaging platforms
- Multi-platform delivery

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
- Check: `~/.hermes/bin/hermes-mcp-serve` is executable
  ```bash
  ls -l ~/.hermes/bin/hermes-mcp-serve
  ```
- Check venv exists: `ls -la ~/.hermes/hermes-agent/.venv/`
- Try manual start: `~/.hermes/bin/hermes-mcp-serve`

### "Connection refused / timeout"
- Hermes MCP server takes 5-10s to start on first run (venv warmup)
- Wait longer before refreshing
- Check Cursor logs: `Help → Toggle Developer Tools`

### "Command not found"
- Use full path: `/home/arrenchulz/.hermes/bin/hermes-mcp-serve`
- Or add to PATH: `export PATH="$HOME/.hermes/bin:$PATH"` in `~/.bashrc`

### "Unknown tools/skills"
- Refresh Cursor MCP connection: 
  - Settings → MCP → click refresh icon next to Hermes
  - Or restart Cursor

---

## 📁 Files Created

```
~/.hermes/hermes-agent/hermes-mcp-serve          # Launch script
~/.hermes/bin/hermes-mcp-serve                   # Symlink for easy access
~/.hermes/hermes-agent/cursor-mcp-config.json    # Config template
~/.hermes/CURSOR_SETUP.md                        # This file
```

---

## 🚀 Done!

Your Cursor is now ready to use Hermes as an MCP server. Start asking complex questions in Composer — Cursor will route them through Hermes' full toolkit.
