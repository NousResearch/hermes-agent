# Hermes Multiplexer Patches

Patches applied to the Hermes agent codebase to enable **stable multi-profile multiplexing with separate Telegram channels**.

## Context

Hermes runs as a single systemd gateway serving multiple agent profiles. Each profile has its own `config.yaml`, `SOUL.md`, and MCP server definitions, and **its own Telegram channel** — allowing multiple independent bots to run simultaneously on one gateway instance. All profiles share the same underlying MCP tool servers (gbrain, hindsight, instantly, transcriptor, open-design).

Without these patches, multiplexed profiles experienced frequent MCP server disconnects, orphaned subprocesses on restart, and broken Hindsight daemon connections — making separate Telegram channels unreliable and unusable for production use.

The following patches fix these issues and enable bug-free operation of multiple independent agent profiles on a single Hermes gateway.

---

## 1. MCP Reconnect Resilience (`mcp_tool.py`)

**File:** `$HERMES_HOME/hermes-agent/tools/mcp_tool.py`
**Lines:** 279, 281

### Problem
The default reconnect logic gave up after only 5 retries with exponential backoff capped at 60s. When an SSE stream broke (e.g. gbrain TaskGroup errors), the MCP client would permanently lose the server after ~31 seconds of attempts, leaving all 81 gbrain tools unavailable for the rest of the session.

### Patch

```diff
-_MAX_RECONNECT_RETRIES = 5
-_MAX_BACKOFF_SECONDS = 60
+_MAX_RECONNECT_RETRIES = 20
+_MAX_BACKOFF_SECONDS = 30
```

### Rationale
- **20 retries** gives the server ~8 minutes to recover (enough for transient SSE drops, Ollama model loads, or network blips).
- **30s backoff cap** (down from 60s) ensures retries happen frequently enough to catch a quick recovery, without the exponential tail making waits absurdly long.

---

## 2. MCP Keepalive & Timeout Tuning (all profiles)

**Files:**
- `$HERMES_HOME/config.yaml` (main profile)
- `$HERMES_HOME/profiles/<profile-a>/config.yaml`
- `$HERMES_HOME/profiles/<profile-b>/config.yaml`

### Problem
The default keepalive interval for HTTP MCP servers is 180s (code default in `mcp_tool.py:288`). gbrain and hindsight servers have shorter SSE session TTLs. With 180s keepalive, idle sessions would expire between pings, causing `TaskGroup` errors and connection drops. The tool-call timeout of 60s was also too short for gbrain operations like `mcp_gbrain_think` or `mcp_gbrain_submit_job`.

### Patch

For **gbrain** and **hindsight** in all profile configs:

```diff
 mcp_servers:
   gbrain:
     url: http://localhost:7777/mcp
+    keepalive_interval: 60
     # ... auth headers ...
     connect_timeout: 30
-    timeout: 60
+    timeout: 120
   hindsight:
     url: http://localhost:8888/mcp
+    keepalive_interval: 60
     connect_timeout: 30
-    timeout: 60
+    timeout: 120
```

### Rationale
- **60s keepalive** (down from 180s default) ensures SSE sessions stay alive before any reasonable server-side TTL expires.
- **120s timeout** (up from 60s) accommodates long-running gbrain tool calls (graph traversal, job submission, schema operations).
- Applied to **all profiles** so multiplexed agents share the same stability guarantees.

---

## 3. Hindsight Ollama Endpoint Fix

**Files:**
- `$HERMES_HOME/.env`
- `$HINDSIGHT_HOME/profiles/<profile>.env` (one per profile)

### Problem
The Hindsight embedded daemon uses `dotenv` with `find_dotenv(usecwd=True, override=True)` in `hindsight_api/config.py:23`. Since the gateway's `WorkingDirectory` is `~/.hermes`, the daemon picks up `~/.hermes/.env` and **overrides** any environment variables set by the embedded manager. The URL was set to the WSL Windows-host IP, which is unreachable from within WSL when Ollama runs natively on Linux.

### Patch

```diff
 # $HERMES_HOME/.env
-HINDSIGHT_API_LLM_BASE_URL=http://<host-ip>:11434/v1
+HINDSIGHT_API_LLM_BASE_URL=http://localhost:11434/v1

 # $HINDSIGHT_HOME/profiles/<profile>.env  (repeat for each profile)
-HINDSIGHT_API_LLM_BASE_URL=http://<host-ip>:11434/v1
+HINDSIGHT_API_LLM_BASE_URL=http://localhost:11434/v1
```

### Rationale
- Ollama runs natively on `localhost:11434` inside WSL.
- The Windows-host IP is only reachable via WSL's mirrored networking, which is unreliable and adds latency.
- All env files must be patched because `dotenv(override=True)` in the daemon process will overwrite any env vars set by the embedded manager from the profile `.env`.

---

## 4. Orphaned stdio MCP Process Cleanup

**File:** `$HERMES_HOME/scripts/kill-mcp-orphans.sh` (new)

### Problem
When the gateway restarts, stdio MCP subprocesses (transcriptor, open-design) are not always killed cleanly. `KillMode=mixed` in the systemd service sends SIGTERM to the main process but leaves orphaned stdio subprocesses consuming memory and holding ports.

### Patch

Created `$HERMES_HOME/scripts/kill-mcp-orphans.sh`:

```bash
#!/bin/bash
# Kill orphaned stdio MCP subprocesses from previous gateway runs
# Only targets transcriptor wrapper and open-design MCP stdio clients (not the daemon)

pkill -f 'transcriptor-mcp-wrapper.sh' 2>/dev/null || true
pkill -f 'open-design.*cli.js.*mcp.*daemon-url' 2>/dev/null || true

exit 0
```

### Rationale
- Targets only the specific stdio MCP wrapper processes, not all child processes.
- Safe to run before gateway startup (`ExecStartPre`) or manually.
- The `|| true` ensures the script doesn't fail if no orphans exist.

### Note
The `ExecStartPre` directive was prepared but is **not yet wired** into the systemd service file. To activate, add to `hermes-gateway.service`:

```ini
[Service]
ExecStartPre=$HERMES_HOME/scripts/kill-mcp-orphans.sh
```

---

## 5. systemd Service Hardening (drop-in)

**Files:**
- `~/.config/systemd/user/hermes-gateway.service.d/autostart.conf`
- `~/.config/systemd/user/hermes-gateway.service.d/hardening.conf`

*(Paths use `~` for user home. Replace `$HERMES_HOME` with your Hermes installation path.)*

### autostart.conf
Ensures the gateway starts after Ollama is available:

```ini
[Unit]
After=network-online.target ollama.service
Wants=network-online.target
```

### hardening.conf
Prevents crash loops and ensures clean process group kills:

```ini
[Unit]
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Restart=on-failure
RestartSec=15
KillMode=control-group
RuntimeMaxSec=43200
```

### Rationale
- `After=ollama.service` ensures Ollama is up before the gateway tries to connect MCP servers that depend on it.
- `KillMode=control-group` (overriding the main service's `mixed`) ensures all child processes including stdio MCP subprocesses are killed on stop.
- `StartLimitBurst=5` prevents infinite restart loops if the gateway crashes on startup.
- `RuntimeMaxSec=43200` (12h) forces a periodic clean restart to prevent memory leaks in long-running sessions.

---

## Summary Table

| # | File | Change | Impact |
|---|------|--------|--------|
| 1 | `tools/mcp_tool.py:279,281` | Reconnect retries 5→20, backoff cap 60→30s | MCP servers survive transient SSE drops |
| 2 | `config.yaml` (all profiles) | keepalive 180→60s, timeout 60→120s for gbrain/hindsight | No more idle-session expiry; long tool calls don't timeout |
| 3 | `.env` (per profile + main) | Ollama URL `<host-ip>`→`localhost` | Hindsight daemon can reach Ollama for LLM verification |
| 4 | `scripts/kill-mcp-orphans.sh` | New orphan cleanup script | Prevents zombie stdio MCP processes after restarts |
| 5 | systemd drop-ins | After=ollama, KillMode=control-group, StartLimitBurst | Clean startup ordering, no crash loops, no orphans |

## Verification

After applying all patches and restarting the gateway:

```
MCP: registered 190 tool(s) from 5 server(s)
  gbrain (HTTP):     81 tools
  hindsight (HTTP):  36 tools
  instantly (HTTP):  42 tools
  open-design (stdio): 20 tools
  transcriptor (stdio): 11 tools
```

- No gbrain connection drops in logs post-restart
- No hindsight MCP connection drops in logs post-restart
- All profiles have identical keepalive/timeout settings
