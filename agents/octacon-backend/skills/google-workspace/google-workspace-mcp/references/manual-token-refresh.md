# Manual Token Refresh for Google Workspace MCP — Headless VPS

Session: 2026-04-30. Environment: Hetzner VPS, Hermes gateway as systemd service.

---

## Failure Pattern: Expired Access Token, Auto-Refresh Blocked

1. `mcp_google_workspace_*` tools fail with `ClosedResourceError`
2. MCP server stderr: `Google API Error (401): Unable to retrieve access token` or no response (auth hangs)
3. Access token in `~/.google_workspace_mcp/credentials/<email>.json` is expired (check `expiry` field)
4. MCP server tries to auto-refresh but stalls — headless VPS cannot open browser for re-auth
5. Hermes times out, closes stdio transport, next call gets `ClosedResourceError`
6. Gateway may crashloop (TEMPFAIL), respawning MCP but without fixing token

**Root cause:** Testing-mode OAuth tokens expire every 7 days. Auto-refresh requires user interaction (browser consent). On a headless VPS with no browser, the refresh flow hangs indefinitely.

---

## Diagnosis

### 1. Check token expiry
```bash
jq '.token.expiry' ~/.google_workspace_mcp/credentials/<email>.json
# or just look at the file
```

### 2. Verify the refresh_token is still valid
```bash
credentials_path="~/.google_workspace_mcp/credentials/<email>.json"
client_id=$(jq -r '.token["client_id"]')
client_secret=$(jq -r '.token["client_secret"]')
refresh_token=$(jq -r '.token["refresh_token"]')

curl -s -X POST https://oauth2.googleapis.com/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=$client_id&client_secret=$client_secret&refresh_token=$refresh_token&grant_type=refresh_token"
```

Good response: `{"access_token": "ya29...", "expires_in": 3599, ...}`
Bad response: `(invalid_grant) Token has been expired or revoked.` means re-auth required.

### 3. Check MCP process health
```bash
ps aux | grep -i "workspace-mcp\|uvx" | grep -v grep
```
Alive does not mean healthy — check stderr log for auth errors.

### 4. Check gateway status
```bash
hermes gateway status
sudo systemctl status hermes-gateway
```
Crashlooping after repeated auth failures is common.

---

## Before You Refresh: Verify First

Not every `ACTION REQUIRED` error means the refresh token is dead. A prior `workspace-mcp` process may have orphaned itself, or multiple accounts may have staggered expiry times. Run a **live verification** before assuming the refresh token is invalid:

```bash
# Test with a lightweight live call (preferred)
mcp_google_workspace_list_gmail_labels user_google_email=<email>
```

If the live call **returns labels**, the access token is still valid and the `ACTION REQUIRED` was a false positive (orphaned MCP transport or wrong-process state). If it **returns the auth URL again**, the token is genuinely dead. Then decide based on the next section.

## Repair: Script Token Refresh (Only If Live Call Fails)

When the refresh_token is still valid but the cached access_token is expired, manually refresh and write back to the credentials file.

When the refresh_token is still valid but the cached access_token is expired, manually refresh and write back to the credentials file.

### Python script (recommended)
```python
import json, requests, datetime, sys

def refresh_token_for_email(email):
    creds_path = f"/home/kensei/.google_workspace_mcp/credentials/{email}.json"
    with open(creds_path) as f:
        data = json.load(f)

    client_id = data["token"]["client_id"]
    client_secret = data["token"]["client_secret"]
    refresh_token = data["token"]["refresh_token"]

    resp = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    })

    if resp.ok:
        new_data = resp.json()
        data["token"]["access_token"] = new_data["access_token"]
        # expiry_in is seconds; add to current UTC time
        expiry = datetime.datetime.utcnow() + datetime.timedelta(seconds=new_data.get("expires_in", 3599))
        data["token"]["expiry"] = expiry.strftime("%Y-%m-%dT%H:%M:%S")
        with open(creds_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"SUCCESS: {email} -> expires {data['token']['expiry']}")
    else:
        print(f"FAIL: {email} -> {resp.status_code}: {resp.text}")

# Use: refresh_token_for_email("saghir.sahil@gmail.com")
```

### After updating tokens
1. Restart gateway: `sudo hermes gateway restart --system`
2. Wait 5 seconds, check `hermes gateway status`
3. Test with lightweight tool call: `list_gmail_labels` on one account

---

## If Refresh Token Is Also Expired

Only fix: complete OAuth from scratch. On a headless VPS, options:

### Option A: SSH Port Forwarding (brittle but works)
```bash
# User's local machine
ssh -L 8000:localhost:8000 user@<hetzner-ip> -N
```
Then trigger auth from agent. User clicks URL in local browser, callback lands on local:8000 forwarded to VPS:8000. See `references/oauth-callback-vps.md` for details and failure modes.

### Option B: Manual PKCE Exchange (expert)
Extract PKCE `code_verifier` from `oauth_states.json`, generate auth URL manually, have user complete consent, capture `code=` from redirect manually, then POST to token endpoint. Fragile — only if Option A completely fails.

### Option C: Run Browser on VPS (requires GUI)
If VPS has X11 or desktop, `ssh -X user@<hetzner-ip>` then `firefox <auth_url>`.

---

## Prevention: Proactive Token Age Monitoring

Add a weekly cron that flags tokens expiring within 24 hours:
```bash
#!/bin/bash
now=$(date +%s)
for f in ~/.google_workspace_mcp/credentials/*.json; do
  [[ "$f" == *"oauth_states.json" ]] && continue
  expiry=$(jq -r '.token.expiry' "$f")
  expiry_ts=$(date -d "$expiry" +%s 2>/dev/null || echo 0)
  hours_left=$(( (expiry_ts - now) / 3600 ))
  email=$(basename "$f" .json)
  echo "$email: $hours_left hours left"
done
```

---

## Key Lessons

1. **Expired token causes `ClosedResourceError`, not a clear auth error.** The MCP freezes on refresh, Hermes closes the pipe.
2. **Always test refresh_token validity with curl first.** Saves time vs debugging MCP transport issues.
3. **Gateway crashlooping is a symptom of auth failures.** Restarting gateway alone does NOT fix expired tokens.
4. **Scripted refresh is faster than full re-auth.** Only viable while refresh_token is still valid (7-day Testing mode).
5. **Account for dot normalization.** `sahil.saghirss9@gmail.com` token file is `sahilsaghir.ss9@gmail.com.json`.

---

## Related

- `SKILL.md` — Auth troubleshooting, token refresh policy
- `references/oauth-callback-vps.md` — SSH tunnel auth flow details
- `add-gmail-account` — full OAuth setup for new accounts
