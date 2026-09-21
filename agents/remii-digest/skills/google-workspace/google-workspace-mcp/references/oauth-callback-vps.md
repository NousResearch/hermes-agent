# OAuth Callback Failures on Remote VPS — Session-Proven Debug Trace

Session: 2026-04-30. Environment: Hetzner VPS (Ubuntu), Hermes gateway running as systemd service, user on separate machine.

---

## Failure Pattern

1. `mcp_google_workspace_start_google_auth` generates an auth URL
2. User clicks it in their browser (on their local machine)
3. Redirect URI is `http://localhost:8000/oauth2callback` — resolves to the USER'S localhost, not the VPS
4. Error received: "Authentication Processing Error — (invalid_grant) code_verifier or verifier is not needed"
5. Retry produces same error
6. All subsequent Google Workspace calls fail with `ClosedResourceError`

**Root cause:** The auth code never reaches the VPS. It goes to the user's localhost:8000. The VPS holds the PKCE verifier in its `workspace-mcp` process. The verifier never sees the code. Eventually the MCP process crashes from accumulated auth failures.

---

## Diagnosis Steps

### 1. Check for stale listener
```bash
ss -tlnp | grep -w 8000
# or
lsof -i:8000
```
Symptom: A python (uvx workspace-mcp) process listening on port 8000 even when NO active auth is in progress. This is stale state from a previous failed or abandoned auth.

### 2. Check MCP process health
```bash
ps aux | grep -i "workspace-mcp\|uvx" | grep -v grep
```
Symptom of crash: Process was there, now gone. Gateway child process (`ms-365-mcp-server`) still running — only Google Workspace MCP died.

### 3. Check MCP stderr log
```bash
tail -30 ~/.hermes/logs/mcp-stderr.log
```
Look for:
- `OAuth token exchange rejected PKCE verifier`
- `Error processing OAuth callback: (invalid_grant) code_verifier or verifier is not needed`
- `OAuth callback missing state parameter; using most recent stored state`
- `Error handling auth callback: Missing OAuth state parameter and no stored state available`

Repeated `Missing OAuth state parameter` means the listener has consumed all stored states and is now empty. It's dead.

### 4. Check token cache state
```bash
ls -la ~/.google_workspace_mcp/credentials/
```
Look for:
- `oauth_states.json` — non-empty means stale states
- Token files for affected accounts — may be missing (symlink to nonexistent file) or present but scope-insufficient
- **Google normalizes dots in Gmail addresses for token filenames.** See memory: `sahil.saghirss9@gmail.com` → stored as `sahilsaghir.ss9@gmail.com.json`. Symlink created manually.

---

## Repair Sequence

Order matters. Do not skip.

```bash
# Step 1: Kill stale OAuth listener
pkill -f "python.*8000"
ss -tlnp | grep -w 8000   # confirm port is free

# Step 2: Clear stale state
> ~/.google_workspace_mcp/credentials/oauth_states.json

# Step 3: Remove broken token for affected account
rm ~/.google_workspace_mcp/credentials/<email>.json
# Note: account for dot normalization. Check ls -la first.

# Step 4: Restart gateway (respawns MCP child processes)
sudo hermes gateway restart --system
sleep 5
hermes gateway status      # confirm active

# Step 5: Verify MCP is alive again
curl -s http://localhost:8000/   # should get connection refused (no listener until auth starts)
# Or try a lightweight tool call on a known-good account to confirm MCP is back
```

---

## Authorization Options for Remote VPS

### Option A: SSH Port Forwarding (works but brittle)

On user's local machine:
```bash
ssh -L 8000:localhost:8000 user@<hetzner-ip> -N
```

**Requirements:**
- Tunnel must be open BEFORE auth starts
- Browser must run on the SAME machine as the SSH tunnel
- User must click through quickly (5-minute timeout)
- Port 8000 must be completely free on user's local machine
- If anything else is on user's port 8000, the auth code goes to the wrong place

**Failures observed in session:**
- User got `invalid_grant code_verifier` even with tunnel active
- Reason: a previous stale listener was still holding old PKCE state
- Fix: ALWAYS kill stale listener + clear oauth_states.json before any retry

### Option B: Run Browser on VPS Itself

If the VPS has a desktop environment or X11 forwarding:
```bash
# From user's machine
ssh -X user@<hetzner-ip>
# Then on VPS
firefox "<auth_url>"
```
This guarantees browser and callback server share localhost.

### Option C: Manual Token Exchange Script

For emergency recovery when the browser flow is completely blocked:
1. Generate the auth URL manually
2. Open it in a browser
3. After consent, the browser shows an error (expected — no listener on YOUR localhost)
4. Copy the `code=` parameter from the browser's address bar
5. On the VPS, run a curl POST to exchange the code for tokens

This requires inspecting the MCP's internal OAuth client credentials. More fragile than Option A.

### Option D: Use a Public Redirect URI

This is the proper fix but requires reconfiguring the OAuth app in Google Cloud Console:
1. Add `https://your-vps-domain.com/oauth2callback` as an authorized redirect URI
2. Update the MCP to use that redirect URI instead of localhost
3. Use a simple reverse proxy (nginx, traefik, or even `python -m http.server`) on port 80/443 on the VPS

This eliminates the SSH tunnel entirely but is a one-time infrastructure change.

---

## Key Lessons

1. **Kill stale listener FIRST.** The most common cause of repeated `code_verifier` errors is an old OAuth server holding stale PKCE state.
2. **Port 8000 conflicts are invisible.** Check both client and server sides. `lsof -i:8000` on both machines.
3. **MCP crashes are a symptom, not the root cause.** Google Workspace MCP crashes after repeated auth failures. Restart gateway after clearing state.
4. **Google normalizes dots in token filenames.** `sahil.saghirss9@gmail.com` → `sahilsaghir.ss9@gmail.com.json`. The symlink fix is documented in memory but easy to forget.
5. **OAuth 7-day expiry in Testing mode.** Tokens last one week. Every month: check `~/.google_workspace_mcp/credentials/`, look at file ages, proactively refresh before expiry.

---

## Related

- `../SKILL.md` — Auth troubleshooting section
- `add-gmail-account` skill — full OAuth walkthrough for new accounts
- `native-mcp` skill — MCP server lifecycle and state management
