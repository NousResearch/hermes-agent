# Hermes Selfhost v2.0 — Docker + MCP + Agent Swarm

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Docker migration, wire Hermes into Claude Code via MCP, write a health-check runbook, tag selfhost-v2.0, and activate the 17-agent swarm.

**Architecture:** Single Docker container runs gateway + dashboard via s6-overlay on `network_mode: host` (WSL2). Claude Code connects to Hermes over MCP stdio (`hermes mcp serve`). Agents are loaded from `~/.hermes/skills/`. Tailscale exposes the stack at `100.108.75.69:{8642,9119}`.

**Tech Stack:** Docker / s6-overlay, Hermes Agent v0.15.1 (local build), Python 3.11, Claude Code CLI, LM Studio on `127.0.0.1:1234`, Tailscale, PowerShell + WSL2.

---

## File Map

| File | Action | Why |
|------|--------|-----|
| `~/.hermes/gateway.pid` | delete before restart | stale PID blocks api_server init |
| `~/.hermes/gateway.lock` | delete before restart | same — false "already in use" |
| `~/.hermes/gateway_state.json` | inspect only | shows api_server state |
| `docker-compose.yml` | modify — uncomment DASHBOARD vars | re-enable dashboard after api_server fix |
| `~/.claude/settings.json` | modify — add `mcpServers` key | wires Hermes into Claude Code MCP |
| `C:\Users\calumai\AppData\Roaming\Claude\claude_desktop_config.json` | modify — add hermes MCP entry | wires Hermes into Claude Desktop app |
| `~/.hermes/skills/` | create subdir + copy `.md` files | loads the 17-agent swarm profiles |
| `README.md` | update — selfhost-v2.0 instructions | release documentation |
| `docs/SELFHOST-ARCHITECTURE.md` | create | open design document |

---

## Task 1: Kill Stale Locks — Fix the api_server Port 8642 False Positive

**Files:**
- Modify: `~/.hermes/gateway.pid` (delete)
- Modify: `~/.hermes/gateway.lock` (delete)
- Read: `~/.hermes/gateway_state.json` (diagnose)

The gateway logs "Port 8642 already in use" even when no socket holds that port. The root cause is Hermes reading `gateway.pid` or `gateway.lock` from a previous container run and short-circuiting the start before the kernel bind ever fires.

- [ ] **Step 1: Stop the running container**

In PowerShell:
```powershell
docker compose -f C:\dev\ai\hermes-agent\docker-compose.yml stop
```
Expected output: `Container hermes  Stopped`

- [ ] **Step 2: Inspect the state file to confirm the symptom**

```powershell
Get-Content "$env:USERPROFILE\.hermes\gateway_state.json" | ConvertFrom-Json | Select-Object -ExpandProperty api_server
```
Expected: `state = "retrying"`, `error_message` contains "already in use" or "failed to reconnect".

- [ ] **Step 3: Remove stale lock and PID files**

```powershell
Remove-Item "$env:USERPROFILE\.hermes\gateway.pid"   -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\.hermes\gateway.lock"  -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\.hermes\gateway_state.json" -ErrorAction SilentlyContinue
```
No output means success (files removed or didn't exist).

- [ ] **Step 4: Verify port 8642 is free before restarting**

```powershell
netstat -ano | Select-String ":8642"
```
Expected: no output (port is free).

- [ ] **Step 5: Start container and tail logs for 30 seconds**

```powershell
docker compose -f C:\dev\ai\hermes-agent\docker-compose.yml up -d; docker logs -f hermes --tail 40
```
Expected in logs:
```
[api_server] Listening on 0.0.0.0:8642
```
NOT expected: `Port 8642 already in use`

Press `Ctrl+C` to stop tailing after confirming.

- [ ] **Step 6: Confirm api_server responds**

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8642/health" -Headers @{ "Authorization" = "Bearer d6cabf77333778d37a8d1f483c66b6e6" } | Select-Object StatusCode, Content
```
Expected: `StatusCode 200`

If Step 5 still shows the error — the lock isn't the only cause. Run the source search below and skip ahead to investigate:
```powershell
docker exec hermes find /opt/hermes/.venv/lib -name "*.py" | xargs grep -l "already in use" 2>/dev/null
```
Then read the matching file to find the exact guard condition.

- [ ] **Step 7: Commit the docker-compose state**

```powershell
cd C:\dev\ai\hermes-agent
git add docker-compose.yml
git commit -m "fix: remove stale gateway lock on Docker restart"
```

---

## Task 2: Re-enable Dashboard in Docker

**Files:**
- Modify: `C:\dev\ai\hermes-agent\docker-compose.yml` (uncomment DASHBOARD vars)

Dashboard was disabled during port 8642 isolation testing. Now that api_server is fixed, re-enable it.

- [ ] **Step 1: Open and edit docker-compose.yml**

In `C:\dev\ai\hermes-agent\docker-compose.yml`, find the block:
```yaml
      # HERMES_DASHBOARD temporarily disabled for port 8642 isolation test
      # - HERMES_DASHBOARD=1
      # - HERMES_DASHBOARD_HOST=0.0.0.0
      # - HERMES_DASHBOARD_INSECURE=1
```

Replace with:
```yaml
      - HERMES_DASHBOARD=1
      - HERMES_DASHBOARD_HOST=0.0.0.0
      - HERMES_DASHBOARD_INSECURE=1
```

- [ ] **Step 2: Restart the container**

```powershell
docker compose -f C:\dev\ai\hermes-agent\docker-compose.yml up -d
```
Expected: `Container hermes  Started`

- [ ] **Step 3: Wait 10 seconds, then check dashboard**

```powershell
Start-Sleep 10
Invoke-WebRequest -Uri "http://127.0.0.1:9119/api/status" | Select-Object StatusCode
```
Expected: `StatusCode 200`

- [ ] **Step 4: Open dashboard in browser**

```powershell
Start-Process "http://127.0.0.1:9119"
```
Expected: Hermes dashboard loads. Profiles list is populated.

- [ ] **Step 5: Commit**

```powershell
cd C:\dev\ai\hermes-agent
git add docker-compose.yml
git commit -m "feat: re-enable dashboard in Docker after api_server fix"
```

---

## Task 3: Fix MCP Install — Wire Hermes into Claude Code

**Files:**
- Modify: `C:\Users\calumai\.claude\settings.json` (add `mcpServers` key)
- Modify: `C:\Users\calumai\AppData\Roaming\Claude\claude_desktop_config.json` (add hermes MCP entry)

The `hermes mcp serve` command exposes Hermes conversations as MCP tools to any MCP client. Neither `~/.claude/settings.json` nor `claude_desktop_config.json` currently have a `hermes` entry under `mcpServers`. That's the install gap.

### Diagnosis

- [ ] **Step 1: Confirm `hermes` is on PATH**

```powershell
Get-Command hermes | Select-Object Source
```
Expected: `C:\Users\calumai\AppData\Local\hermes\hermes-agent\hermes.exe` (or similar).

If not found, the PATH fix is:
```powershell
$env:PATH += ";$env:LOCALAPPDATA\hermes\hermes-agent"
```
Add permanently via System Properties → Environment Variables → User PATH.

- [ ] **Step 2: Confirm `hermes mcp serve` starts cleanly (manual test)**

Open a NEW terminal (to test the environment Claude Code sees):
```powershell
hermes mcp serve --verbose
```
Expected within 3 seconds:
```
[MCP] stdio server started
[MCP] Waiting for gateway connection at http://127.0.0.1:8642
[MCP] Gateway connected
```
If it logs `Gateway not reachable` — the gateway container isn't running. Go back to Task 1.

Press `Ctrl+C` to stop.

- [ ] **Step 3: Add Hermes MCP to Claude Code settings**

Read the current `~/.claude/settings.json` first. Then add the `mcpServers` block at the top level. The final relevant section should look like:

```json
{
  "mcpServers": {
    "hermes": {
      "command": "hermes",
      "args": ["mcp", "serve"]
    }
  }
}
```

If the file already has a top-level `mcpServers` key, merge — do NOT replace the whole file.

- [ ] **Step 4: Add Hermes MCP to Claude Desktop config**

Edit `C:\Users\calumai\AppData\Roaming\Claude\claude_desktop_config.json`. Add the hermes entry inside the existing `mcpServers` object (alongside `desktop-commander` and `MCP_DOCKER`):

```json
"hermes": {
  "command": "hermes",
  "args": ["mcp", "serve"]
}
```

The full `mcpServers` section becomes:
```json
"mcpServers": {
  "desktop-commander": { ... },
  "MCP_DOCKER": { ... },
  "hermes": {
    "command": "hermes",
    "args": ["mcp", "serve"]
  }
}
```

- [ ] **Step 5: Restart Claude Desktop app**

```powershell
Stop-Process -Name "claude" -ErrorAction SilentlyContinue
Start-Sleep 2
Start-Process "$env:LOCALAPPDATA\AnthropicClaude\Claude.exe"
```

- [ ] **Step 6: Verify Hermes MCP shows up in Claude Code**

In a Claude Code session, run:
```
/mcp
```
Expected: `hermes` appears in the list with status `connected`.

If status is `error` — run `hermes mcp serve --verbose` in a terminal while retrying to see the error message. Common issues:
- `hermes: command not found` → PATH not set for the process Claude Code spawns
- `Gateway not reachable` → container not running, check `docker ps`
- `Auth failed` → check `API_SERVER_KEY` in `~/.hermes/.env` matches

- [ ] **Step 7: Test a Hermes MCP tool call**

In Claude Code, ask:
```
Use the hermes MCP to list conversations.
```
Expected: `conversations_list` tool call succeeds and returns a list (even if empty).

- [ ] **Step 8: Commit nothing (config files only — do not commit `.env` or API keys)**

Document the MCP setup was completed:
```powershell
cd C:\dev\ai\hermes-agent
git commit --allow-empty -m "chore: Claude Code MCP integration configured and verified"
```

---

## Task 4: Hermes Health Check Runbook (Post-Docker-Restart)

**Files:**
- Create: `C:\dev\ai\hermes-agent\docs\HERMES-HEALTH-CHECK.md`

Write and execute a systematic health check you can run any time Docker restarts. The goal is to catch failures early with exact expected outputs so future issues are diagnosed in under 2 minutes.

- [ ] **Step 1: Run all health checks now (capture expected outputs)**

```powershell
# 1. Docker container status
docker ps --filter name=hermes --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 2. Gateway API health
Invoke-WebRequest -Uri "http://127.0.0.1:8642/health" `
  -Headers @{ "Authorization" = "Bearer d6cabf77333778d37a8d1f483c66b6e6" } `
  | Select-Object StatusCode, Content

# 3. Dashboard status
Invoke-WebRequest -Uri "http://127.0.0.1:9119/api/status" | Select-Object StatusCode, Content

# 4. LM Studio models available
Invoke-WebRequest -Uri "http://127.0.0.1:1234/v1/models" | ConvertFrom-Json | Select-Object -ExpandProperty data | Select-Object id

# 5. Tailscale reachability
Test-NetConnection -ComputerName "100.108.75.69" -Port 8642 | Select-Object TcpTestSucceeded

# 6. Profiles loaded count
docker exec hermes ls /opt/data/profiles/ | Measure-Object | Select-Object Count
```

Record the expected output for each. These become the "green baseline" in the doc.

- [ ] **Step 2: Write the health check doc**

Create `C:\dev\ai\hermes-agent\docs\HERMES-HEALTH-CHECK.md`:

```markdown
# Hermes Health Check — selfhost-v2.0

Run this after any Docker restart or system reboot. All checks should pass before
opening the workspace or starting agent work.

## Quick Check (< 1 min)

```powershell
# Full health check — paste and run
docker ps --filter name=hermes --format "table {{.Names}}\t{{.Status}}"
Invoke-WebRequest http://127.0.0.1:8642/health -Headers @{ Authorization="Bearer d6cabf77333778d37a8d1f483c66b6e6" } | Select StatusCode
Invoke-WebRequest http://127.0.0.1:9119/api/status | Select StatusCode
Test-NetConnection 100.108.75.69 -Port 8642 | Select TcpTestSucceeded
```

Expected all green:
| Check | Expected |
|-------|----------|
| Container status | `hermes   Up X minutes` |
| Gateway /health | `StatusCode 200` |
| Dashboard /api/status | `StatusCode 200` |
| Tailscale 8642 | `TcpTestSucceeded : True` |

## If Gateway Is Down

```powershell
# 1. Remove stale locks
Remove-Item "$env:USERPROFILE\.hermes\gateway.pid" -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\.hermes\gateway.lock" -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\.hermes\gateway_state.json" -ErrorAction SilentlyContinue

# 2. Restart
docker compose -f C:\dev\ai\hermes-agent\docker-compose.yml restart

# 3. Tail logs for 20s
docker logs -f hermes --tail 30
```

## If LM Studio Is Not Loaded

Open LM Studio → Load any model → ensure server is running on port 1234.

## If MCP Shows Error in Claude Code

```powershell
hermes mcp serve --verbose
```
Check the error line — "Gateway not reachable" means container is down. "Auth failed" means
API_SERVER_KEY mismatch between ~/.hermes/.env and settings.json.
```

- [ ] **Step 3: Commit**

```powershell
cd C:\dev\ai\hermes-agent
git add docs/HERMES-HEALTH-CHECK.md
git commit -m "docs: add Hermes health check runbook for selfhost-v2.0"
```

---

## Task 5: Update README, Git Tag, Activate Swarm, Open Design Doc

**Files:**
- Modify: `C:\dev\ai\hermes-agent\README.md`
- Create: `~/.hermes/skills/swarm/` (copy agent .md files here)
- Create: `C:\dev\ai\hermes-agent\docs\SELFHOST-ARCHITECTURE.md`
- Tag: `selfhost-v2.0`

### 5a — Update README

- [ ] **Step 1: Open README.md and add a selfhost section**

Find the section in `README.md` nearest to "Installation" or "Getting Started". Add a new section immediately after it:

```markdown
## Self-Hosting with Docker (selfhost-v2.0)

Run Hermes as a single Docker container — gateway + dashboard together via s6-overlay.

**Prerequisites:** Docker Desktop (WSL2 backend), `~/.hermes/` config exists.

```powershell
# Clone and build
git clone https://github.com/reevesc88/hermes-agent
cd hermes-agent

# Set host UID/GID so files inside container stay writable on host
$env:HERMES_UID = "10000"; $env:HERMES_GID = "10000"

# Start (WSL2 / Linux — uses network_mode: host)
docker compose up -d

# OR — Docker Desktop on Windows (explicit port mapping)
docker compose -f docker-compose.windows.yml up -d
```

**Health check:**
```powershell
Invoke-WebRequest http://127.0.0.1:8642/health -Headers @{Authorization="Bearer <your-key>"}
Invoke-WebRequest http://127.0.0.1:9119/api/status
```

See [docs/HERMES-HEALTH-CHECK.md](docs/HERMES-HEALTH-CHECK.md) for full runbook.

**Claude Code MCP integration:**
```json
// Add to ~/.claude/settings.json under mcpServers:
"hermes": { "command": "hermes", "args": ["mcp", "serve"] }
```
```

- [ ] **Step 2: Stage README**

```powershell
cd C:\dev\ai\hermes-agent
git add README.md
```

### 5b — Activate the 17-Agent Swarm

- [ ] **Step 3: Create swarm skills directory**

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.hermes\skills\swarm"
```

- [ ] **Step 4: Copy agent profiles to Hermes skills**

```powershell
# Core orchestrator
Copy-Item "C:\dev\ai\hermes-agent\hermes-multiagent\agent\core\hermes.md" `
  "$env:USERPROFILE\.hermes\skills\"

# All subagent groups
Get-ChildItem "C:\dev\ai\hermes-agent\hermes-multiagent\agent\subagents" -Recurse -Filter "*.md" |
  ForEach-Object { Copy-Item $_.FullName "$env:USERPROFILE\.hermes\skills\swarm\" }
```

- [ ] **Step 5: Verify skills are visible in Hermes**

```powershell
hermes skills list 2>$null
# OR check the file count
Get-ChildItem "$env:USERPROFILE\.hermes\skills\" -Recurse -Filter "*.md" | Measure-Object | Select Count
```
Expected: at least 18 skill files (1 core + 17 subagents).

- [ ] **Step 6: Test the orchestrator skill loads**

```powershell
hermes chat --profile cli -q "/skill hermes"
```
Expected: Hermes confirms the skill is loaded and displays the orchestrator role description. It should mention delegation to subagents.

Press `Ctrl+C` after confirming.

### 5c — Write the Open Design Document

- [ ] **Step 7: Create docs/SELFHOST-ARCHITECTURE.md**

Create `C:\dev\ai\hermes-agent\docs\SELFHOST-ARCHITECTURE.md`:

```markdown
# Hermes Selfhost Architecture — v2.0

## Overview

This document describes the selfhost-v2.0 architecture: a single-node Hermes deployment
on Windows 11 with WSL2, Docker, LM Studio, and Tailscale for remote access.

## Stack

```
┌─────────────────────────────────────────────────────┐
│  Windows 11 (host)                                  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  WSL2 / Docker                                │  │
│  │                                               │  │
│  │  hermes (container)                           │  │
│  │  ├─ gateway (port 8642) ─── api_server        │  │
│  │  └─ dashboard (port 9119)                     │  │
│  │         ↑ network_mode: host                  │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  LM Studio ─────── 127.0.0.1:1234                   │
│  Claude Code ──── MCP stdio (hermes mcp serve)      │
│  Tailscale ──────── 100.108.75.69:{8642,9119}       │
└─────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Role | Config |
|-----------|------|--------|
| Hermes gateway | Agent orchestration, platform routing | `~/.hermes/config.yaml` |
| api_server | HTTP REST + webhook endpoint | `~/.hermes/.env` (API_SERVER_KEY) |
| Dashboard | Web UI for profiles, conversations, state | port 9119 |
| LM Studio | Local LLM inference (13 models) | `127.0.0.1:1234` |
| MCP server | Claude Code ↔ Hermes bridge | `hermes mcp serve` (stdio) |
| Tailscale | Secure remote access | IP: 100.108.75.69 |

## s6-overlay Process Tree

```
/init  (PID 1 — s6-overlay)
├── cont-init.d/  (chown, profile reconcile, dashboard toggle)
└── services/
    ├── hermes-dashboard  (if HERMES_DASHBOARD=1)
    └── gateway runs as Docker CMD (not s6-supervised child)
        └── per-profile sub-agents registered at runtime in /run/service/
```

## Key Design Decisions

1. **Single container** — gateway + dashboard in one container avoids the shared-volume
   lock conflicts that occur when two containers both access `~/.hermes` and register
   the same gateway-profile s6-services.

2. **HERMES_GATEWAY_NO_SUPERVISE=1** — prevents CMD from delegating to an s6-supervised
   child, which would race the CMD for port 8642 and cause a deadlock.

3. **Stale lock cleanup on restart** — `gateway.pid` and `gateway.lock` from a previous
   run must be removed before starting a new container, or the api_server init incorrectly
   reports "Port 8642 already in use." The health-check runbook handles this.

4. **MCP over stdio** — `hermes mcp serve` connects to the gateway over HTTP
   (`127.0.0.1:8642`) then speaks MCP stdio to any MCP client (Claude Code, Claude Desktop,
   Cursor). No separate port needed for MCP.

## Agent Swarm

The 17-agent swarm (`hermes-multiagent/`) is loaded as skills in `~/.hermes/skills/swarm/`.
Each agent is a `.md` system-prompt file. The master orchestrator (`hermes.md`) delegates
to subagents via the `delegate_task` tool using `kimi-k2.6:cloud` for orchestration and
lighter models (Qwen3-Coder, GLM-5.1, Gemma-4) for worker roles.

To activate: `hermes chat --profile cli` then `/skill hermes`.
```

### 5d — Git Tag and Final Commit

- [ ] **Step 8: Stage remaining files and commit**

```powershell
cd C:\dev\ai\hermes-agent
git add docs/SELFHOST-ARCHITECTURE.md README.md
git commit -m "docs: add selfhost-v2.0 architecture doc and README selfhost section"
```

- [ ] **Step 9: Tag the release**

```powershell
git tag -a selfhost-v2.0 -m "selfhost-v2.0: Docker migration complete, MCP wired, agent swarm active"
```

- [ ] **Step 10: Push tag to origin**

```powershell
git push origin selfhost-v2.0
```
Expected:
```
To https://github.com/reevesc88/hermes-agent.git
 * [new tag]         selfhost-v2.0 -> selfhost-v2.0
```

- [ ] **Step 11: Verify tag on GitHub**

```powershell
gh release list --repo reevesc88/hermes-agent 2>$null
# OR check tags
gh api repos/reevesc88/hermes-agent/tags --jq '.[0].name'
```
Expected: `selfhost-v2.0`

- [ ] **Step 12: Final smoke test — run the full stack**

```powershell
# 1. Container running?
docker ps --filter name=hermes

# 2. API up?
Invoke-WebRequest http://127.0.0.1:8642/health -Headers @{Authorization="Bearer d6cabf77333778d37a8d1f483c66b6e6"} | Select StatusCode

# 3. Dashboard up?
Start-Process "http://127.0.0.1:9119"

# 4. MCP in Claude Code → ask: "Use hermes MCP to list conversations"

# 5. Swarm loaded?
hermes chat --profile cli -q "/skill hermes" 2>&1 | Select-Object -First 5
```

All 5 checks green = selfhost-v2.0 complete.

---

## Summary

| Task | What It Fixes | Key Output |
|------|---------------|------------|
| 1 | api_server "port in use" false positive | Gateway responds on 8642 |
| 2 | Dashboard disabled during isolation test | Dashboard on 9119 |
| 3 | Hermes not wired to Claude Code MCP | `/mcp` shows `hermes: connected` |
| 4 | No documented restart procedure | `docs/HERMES-HEALTH-CHECK.md` |
| 5 | Release packaging + swarm + design doc | tag `selfhost-v2.0` pushed |
