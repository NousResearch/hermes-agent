# Persistent `ClosedResourceError` Despite Alive MCP and Valid Token

Session: 2026-04-30. Environment: Hetzner VPS (Ubuntu), Hermes gateway v2.x, Kimi K2.6 via Ollama Cloud Pro.

---

## Failure Pattern (Critical Distinction)

This is **NOT** the classic expired-token or stale-OAuth-listener failure. All three standard indicators are GREEN, yet the error persists.

1. MCP process IS alive: `ps aux | grep workspace-mcp` shows PID running
2. Token IS valid: `~/.google_workspace_mcp/credentials/<email>.json` has `refresh_token`, `access_token`, scopes, and `expiry` within past 7 days
3. MCP server log IS clean: `mcp_server_debug.log` shows only normal startup lines — `ListToolsRequest` processed, no auth errors, no crashes
4. **BUT** every `mcp_google_workspace_*` call returns `ClosedResourceError` (or escalates through `12 consecutive failures` / `Auto-retry available`)
5. Gateway restart does **NOT** fix it
6. `~/.hermes/logs/errors.log` shows `tools.mcp_tool` failed entries with **empty message** — no exception text, no traceback, just the tool name

**This is a transport-layer desync, not an auth-layer failure.**

---

## Root-Cause Hypothesis

Hermes gateway's stdio transport maintains per-tool-call session state. When earlier sessions experienced JSON-RPC `session binding conflict` errors (visible in old logs from gateway restarts without full cleanup), the internal MCP session registry may have become desynchronized with the actual OS stdio pipes. The MCP process reads from pipe fd 0 and writes to fd 1, but the gateway's Python-side `asyncio` stream reader has been closed or re-bound, so the gateway sees `ClosedResourceError` even though the OS pipes are still open.

Evidence from `errors.log` (earlier session traces):
- `RuntimeError: aclose(): asynchronous generator is already running` — async generator cleanup failure during previous session termination
- Multiple gateway processes running simultaneously (`pts/0`, `pts/1`) competing for the same Telegram session — this corrupts gateway state
- `gateway.run: Shutdown (final-cleanup): killed 1 tool subprocess(es)` — previous shutdown did not fully clean stdio session registry

After enough corrupted restarts, the gateway's internal MCP client session for `google_workspace` points to a stale or closed stream while the OS process is still alive.

---

## Diagnosis Checklist

Run these in order. If step N fails, the issue is already known (expired token, dead process, etc.). Only proceed to step N+1 if step N passes.

### Step 1: Is the MCP process alive?
```bash
ps aux | grep -i workspace | grep -v grep
# Should show: uv tool uvx workspace-mcp  AND  python workspace-mcp
```
**If NO** → classic "dead MCP" failure. Restart gateway: `sudo hermes gateway restart --system`.

### Step 2: Is the token recent and complete?
```bash
cat ~/.google_workspace_mcp/credentials/<email>.json | grep -E '"token"|"refresh_token"|"expiry"'
```
**If NO** → expired token. Use `references/manual-token-refresh.md` or `references/oauth-callback-vps.md`.

### Step 3: Is the MCP server log clean?
```bash
# Check timestamps match
ls -la /home/kensei/.google_workspace_mcp/logs/mcp_server_debug.log
# Should show recent timestamp matching ps start time
tail -20 /home/kisei/.google_workspace_mcp/logs/mcp_server_debug.log
# Should show ONLY startup lines, NO errors after ListToolsRequest
```
**If ERRORS present** → investigate those errors specifically (auth, scope, crash).

### Step 4: Are the OS stdio pipes open?
```bash
ls -la /proc/$(pgrep -f "workspace-mcp" | tail -1)/fd/
# Look for:
# fd 0 -> pipe:[NNNNNN]   (read end — from gateway)
# fd 1 -> pipe:[NNNNNN]   (write end — to gateway)
# fd 2 -> .../mcp-stderr.log
# fd 10 -> .../mcp_server_debug.log
```
**If pipes missing** → MCP process detached from stdio. Restart gateway.

### Step 5: Is the gateway writing to the pipe?
This requires `strace` on the MCP process (needs elevated privileges):
```bash
sudo timeout 5 strace -e trace=read,write -p $(pgrep -f "workspace-mcp" | tail -1) 2>&1
```
During strace, trigger a tool call from another terminal/session. If MCP process does NOT receive data (no `read()` calls during the 5s window), the gateway is NOT writing to the pipe. This **confirms transport desync**.

**If strace shows reads but no writes, or vice versa** → the pipe is open but the gateway is not using it for tool calls.

**If strace shows no activity on either fd 0 or fd 1 during tool call** → transport desync confirmed.

---

## What Does NOT Fix This

| Action | Result | Why |
|--------|--------|-----|
| Gateway restart (`sudo hermes gateway restart --system`) | Stays broken | Respawns MCP with same corrupted session registry |
| Token refresh (manual script) | No effect | Token was already valid |
| Kill MCP PID, let gateway respawn | No effect | New PID, same registry mapping |
| Retry same tool call after cooldown | No effect | Registry entry still points to closed stream |

---

## What MIGHT Fix This (Untested but Theoretically Sound)

The underlying issue is in Hermes gateway's internal MCP client session state, which is NOT cleared by a simple restart because session data may be persisted across restarts or cached in memory by the Python asyncio event loop.

### Fix A: Full Hermes Stop + Start (Recommended)

Stop the gateway **completely**, ensure no python gateway processes remain, then start fresh:

```bash
# 1. Stop gateway service
sudo systemctl stop hermes-gateway

# 2. Kill any orphaned gateway processes
pkill -f "hermes.*gateway"
pkill -f "python.*hermes_cli"

# 3. Verify clean slate
ps aux | grep hermes | grep -v grep
# Should show NO python hermes processes
# (ms-365-mcp-server child processes may remain if Hermes didn't clean them up)
pkill -f "workspace-mcp"
pkill -f "ms-365-mcp-server"

# 4. Clear any cached gateway state
rm -f ~/.hermes/sessions/* 2>/dev/null
# Do NOT delete sessions you need — just the corrupted ones
# Or more targeted: find sessions with failed MCP calls and delete those specifically

# 5. Start gateway fresh
sudo systemctl start hermes-gateway
sleep 5
hermes gateway status
```

### Fix B: Hermes Reinstall / Update

If full stop+start fails, the corruption may be in Hermes' internal compiled state:
```bash
pip install --upgrade hermes-agent  # or equivalent reinstall
# Then restart gateway as in Fix A
```

### Fix C: Kill ALL Gateway Processes on All Terminals

Multiple Hermes agent sessions on multiple PTYs (pts/0, pts/1, etc.) can interfere:
```bash
# Kill ALL hermes processes (except the one you're running this from)
ps aux | grep "hermes-agent/venv" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sudo systemctl stop hermes-gateway
sudo systemctl start hermes-gateway
```

### Fix D: Wait for Background Process Cleanup

Hermes gateway `Recovered X background process(es) from previous run` on startup means it inherited child processes. This inheritance may carry corrupted state. Kill those children BEFORE starting gateway:
```bash
pkill -f "workspace-mcp"
pkill -f "ms-365-mcp-server"
# Wait 2 seconds, then start gateway
```

---

## Session-Specific Findings

From 2026-04-30 debugging session:

- **Token expiry:** 2026-04-30T11:52:12Z — expired roughly 20 minutes before tool calls. However token file had access_token present. The MCP might have auto-refreshed successfully (no log errors) but the refreshed token still failed.
- **Hermes errors.log entries:** All `google_workspace/*` tool failures had **empty error messages** — no traceback, no exception text. This is consistent with a `ClosedResourceError` bubbling from the transport layer before any tool-specific logic can report an error.
- **Gateway.log:** No errors for google_workspace MCP. It started cleanly: `Starting MCP server 'google_workspace'... ListToolsRequest processed`.
- **Outlook MCP:** Running in parallel, working fine. This proves the gateway CAN talk to MCP servers — the problem is specific to the google_workspace session binding.
- **Earlier session artifacts:** `RuntimeError: aclose(): asynchronous generator is already running` in errors.log from ~01:56 — this async cleanup failure may have corrupted the gateway's session registry.

**Conclusion:** The most likely cause is accumulated async session binding corruption from gateway restarts without full process cleanup. The fix is a complete stop + process kill + fresh start.

---

## Prevention

1. **Avoid rapid gateway restarts.** Each restart risks session binding corruption. If you must restart, do a full `systemctl stop` + verify no processes remain + `systemctl start`.

2. **Kill sibling gateway processes.** Multiple `hermes` CLI sessions on different terminals compete for the same Telegram session and can corrupt gateway state. Before restarting, check `ps aux | grep hermes | grep -v grep`.

3. **Monitor for empty error messages.** In `~/.hermes/logs/errors.log`, `mcp_google_workspace_* call failed:` with nothing after the colon is the hallmark of transport desync. If you see this, stop and do full restart.

4. **Check mcp_server_debug.log for tool call entries.** A healthy MCP server logs `Processing request of type ListToolsRequest` and subsequent `ListToolsResult` on tool list, and similar for tool calls. If the log shows startup but NO tool call processing entries, the gateway is not sending tool calls to this MCP.

---

## Related

- `SKILL.md` — Auth troubleshooting section on `ClosedResourceError`
- `references/oauth-callback-vps.md` — Stale OAuth listener (different failure mode)
- `references/manual-token-refresh.md` — Expired token (different failure mode)
- `native-mcp` skill — MCP server lifecycle and state management
