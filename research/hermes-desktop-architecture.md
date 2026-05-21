# Hermes Desktop Architecture Research Report

**Date:** 2026-05-21
**Version:** 0.4.5
**Source:** https://github.com/fathah/hermes-desktop

---

## 1. Does the App Start Hermes Agent While Running?

**Short answer: No, it does NOT keep a persistent Hermes agent process running.** Instead, it manages a **gateway process** (which includes the API server) and uses a **two-tier communication strategy**.

### Architecture Overview

```
Desktop App (Electron) --IPC--> Main Process (Node.js)
                                    |
                                    +---> HTTP API (port 8642) via gateway process
                                    +---> CLI subprocess (per-message spawn)
```

### The Gateway Process (Background, Persistent)

The desktop app starts the Hermes **gateway** as a detached background process when you first send a message (lazy start). This gateway includes the **API server** on `127.0.0.1:8642`.

From `src/main/hermes.ts`:
```typescript
// Started lazily on first chat message or explicit gateway start
gatewayProcess = spawn(HERMES_PYTHON, hermesCliArgs(["gateway"]), {
  cwd: HERMES_REPO,
  env: gatewayEnv,
  stdio: "ignore",
  detached: true,    // <-- runs independently of the desktop app
});
gatewayProcess.unref();  // <-- desktop app doesn't block on it
```

The gateway stays running and the desktop app communicates with it via HTTP REST API (OpenAI-compatible endpoints) using SSE streaming.

### The CLI Fallback (Per-Message, Ephemeral)

If the gateway/API server is not available, the app falls back to spawning a **new Hermes CLI process for each message**:

```typescript
// sendMessage() auto-routes: API first, CLI fallback
export async function sendMessage(message, cb, profile?) {
  if (apiServerAvailable) {
    return sendMessageViaApi(message, cb, profile, ...);  // HTTP to port 8642
  }
  return sendMessageViaCli(message, cb, profile, ...);     // spawn per message
}
```

### Connection Modes

1. **Local mode** (default) - App starts the gateway on `127.0.0.1:8642`
2. **Remote mode** - App connects to an existing remote Hermes API server
3. **SSH mode** - App creates an SSH tunnel to a remote server, then connects via the tunnel

### Health Checking

The app polls the gateway every 15 seconds to check if the API server is alive:
```typescript
function startHealthPolling(): void {
  _healthCheckInterval = setInterval(async () => {
    apiServerAvailable = await isApiServerReady();
    if (apiServerAvailable) {
      clearInterval(_healthCheckInterval);  // Stop once confirmed
    }
  }, 15000);
}
```

---

## 2. When Are Scripts Reloaded?

**Short answer: Scripts are NOT hot-reloaded. Python code changes are snapshotted at import time.**

### Key Facts

1. **No hot-reload** - The desktop app does NOT monitor source files for changes. When you patch Python code in `~/.hermes/hermes-agent/`, the changes are NOT picked up until the gateway process is restarted.

2. **Import-time snapshotting** - Hermes Agent reads configuration, tools, skills, and toolsets at Python import time. Changes to these are only visible after a full process restart.

3. **Config values snapshotted on startup** - Settings like `model.context_length`, `delegation.child_timeout_seconds`, `security.redact_secrets`, etc. are read once when the process starts.

### What Triggers a Reload

| Change Type | Takes Effect When |
|---|---|
| **Python source code** | Gateway restart or CLI relaunch |
| **config.yaml** | Gateway restart or new CLI session |
| **.env file** | Gateway restart or new CLI session |
| **Skills** | New session (`/reset` in chat, or restart CLI/gateway) |
| **Tools enable/disable** | New session (`/reset` in chat, or restart CLI/gateway) |
| **Memory entries** | Immediate (read from disk on demand) |
| **Cron jobs** | Immediate (jobs.json read on each tick) |
| **SOUL.md (persona)** | New session (read at session start) |

---

## 3. How to Patch Hermes Scripts with Custom Logic

### Source Location

Hermes Agent source code lives at:
```
~/.hermes/hermes-agent/          (default on Linux/Mac)
%LOCALAPPDATA%\hermes\hermes-agent\  (default on Windows)
```

### Project Structure

```
hermes-agent/
├── run_agent.py          # Core conversation loop
├── model_tools.py        # Tool discovery and dispatch
├── toolsets.py           # Toolset definitions
├── cli.py                # Interactive CLI (HermesCLI)
├── hermes_state.py       # SQLite session store
├── agent/                # Prompt builder, context compression, memory, model routing
├── hermes_cli/           # CLI subcommands, config, setup, slash commands
│   ├── commands.py       # Slash command registry
│   ├── config.py         # DEFAULT_CONFIG, env var definitions
│   └── main.py           # CLI entry point
├── tools/                # One file per tool
│   └── registry.py       # Central tool registry
├── gateway/              # Messaging gateway
│   └── platforms/        # Platform adapters
├── cron/                 # Job scheduler
└── plugins/              # Plugin system
```

### Patching Approach

1. **Edit files directly** in `~/.hermes/hermes-agent/`
2. **Restart the gateway** to pick up changes (see section 4)

### Adding a Custom Tool

Create `~/.hermes/hermes-agent/tools/your_tool.py`:
```python
import json, os
from tools.registry import registry

def check_requirements() -> bool:
    return True  # or check for dependencies

async def your_tool(param: str, task_id: str = None) -> str:
    return json.dumps({"success": True, "data": "..."})

registry.register(
    name="your_tool",
    toolset="example",
    schema={"name": "your_tool", "description": "...", "parameters": {...}},
    handler=lambda args, **kw: your_tool(
        param=args.get("param", ""), task_id=kw.get("task_id")),
    emoji="🔧",
)
```

Then add the tool to `toolsets.py` - this is REQUIRED for the tool to be exposed.

### Adding a Slash Command

1. Add `CommandDef` to `COMMAND_REGISTRY` in `hermes_cli/commands.py`
2. Add handler in `cli.py` -> `process_command()`

### Custom Providers (Adding Models)

Edit `~/.hermes/config.yaml` under `custom_providers`:
```yaml
custom_providers:
- name: llama.cpp
  base_url: http://100.66.170.3:8080/v1
  model: your-model-name
```

---

## 4. How to Reload Hermes

### From the Desktop App UI

1. **Restart Gateway:**
   - Go to Settings -> Gateway section
   - Click "Stop Gateway" then "Start Gateway"
   - Or use the "Restart Gateway" button if available

2. **Update Hermes:**
   - Go to Settings -> Hermes section
   - Click "Update Hermes" (runs `hermes update` internally)

3. **Change Model/Provider:**
   - Settings -> Models section
   - Changing model/provider automatically restarts the gateway

4. **Change Toolsets:**
   - Tools screen -> toggle toolsets on/off
   - Changing toolsets automatically restarts the gateway

5. **Change Platform Settings:**
   - Settings -> toggle platforms
   - Automatically restarts the gateway

### From the CLI (if running in terminal)

```bash
# In gateway mode
/restart          # Slash command to restart gateway
/hermes update    # Update to latest version

# In CLI mode
# Just exit and relaunch - changes are picked up on new session
hermes            # Start fresh
hermes chat -q "..."  # One-shot mode
```

### From the Terminal (directly)

```bash
# Stop the gateway
kill $(cat ~/.hermes/gateway.pid) 2>/dev/null

# Or use hermes CLI
hermes gateway stop

# Start the gateway
hermes gateway run    # foreground
hermes gateway start  # background service

# Update hermes
hermes update
```

### After Patching Source Code

```bash
# 1. Edit files in ~/.hermes/hermes-agent/
# 2. Restart the gateway
hermes gateway restart

# Or from the desktop app:
# Settings -> Stop Gateway -> Start Gateway
```

---

## 5. Desktop App IPC Surface

The desktop app communicates with Hermes through Electron IPC. Key handlers:

### Chat
- `send-message` - Send a chat message (auto-starts gateway if needed)
- `abort-chat` - Abort current chat

### Gateway
- `start-gateway` - Start the gateway process
- `stop-gateway` - Stop the gateway process
- `gateway-status` - Check if gateway is running

### Configuration
- `get-env` / `set-env` - Read/write .env variables
- `get-config` / `set-config` - Read/write config.yaml
- `get-hermes-home` - Get the HERMES_HOME path
- `get-model-config` / `set-model-config` - Model/provider settings

### Updates
- `get-hermes-version` - Get installed version
- `refresh-hermes-version` - Refresh version cache
- `run-hermes-update` - Run `hermes update`
- `run-hermes-doctor` - Run `hermes doctor`

### Profiles
- `list-profiles` / `create-profile` / `delete-profile` / `set-active-profile`

### Sessions
- `list-sessions` / `get-session-messages` / `search-sessions` / `delete-session`

### Skills, Memory, Tools, Soul, Cron - all via IPC handlers

---

## 6. Key File Paths

| What | Path |
|---|---|
| Hermes home | `~/.hermes/` |
| Config | `~/.hermes/config.yaml` |
| API keys | `~/.hermes/.env` |
| Source code | `~/.hermes/hermes-agent/` |
| Python venv | `~/.hermes/hermes-agent/venv/` |
| Gateway PID | `~/.hermes/gateway.pid` |
| Active profile | `~/.hermes/active_profile` |
| Sessions DB | `~/.hermes/state.db` |
| Cron jobs | `~/.hermes/cron/jobs.json` |
| Skills | `~/.hermes/skills/` |
| Profile dir | `~/.hermes/profiles/<name>/` |
| Gateway logs | `~/.hermes/logs/gateway.log` |
| Auth/credentials | `~/.hermes/auth.json` |

Profile-specific files mirror the above under `~/.hermes/profiles/<name>/`.

---

## 7. Important Caveats

1. **No hot-reload** - Always restart the gateway after patching Python source code
2. **Gateway is lazy-started** - The gateway only starts when you first send a message (or explicitly via Settings)
3. **Python imports are snapshotted** - Config, tools, skills, toolsets are all read at import time
4. **API keys injected at spawn time** - The gateway process gets API keys from `.env` when it starts; changing `.env` requires a gateway restart
5. **Tool changes need `/reset`** - Enabling/disabling tools takes effect on new sessions, not mid-conversation
6. **The app does NOT manage the agent loop** - The desktop app is a thin HTTP client. The actual agent conversation loop runs inside the Hermes gateway (Python process)
7. **Health polling stops after first success** - Once the API server is confirmed available, the 15-second health polling stops. It restarts if the gateway dies.
