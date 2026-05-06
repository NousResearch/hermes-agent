---
name: self-restart
description: "Trigger a Railway redeploy of the hermes-agent service (container exit + restart on same image)."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [hermes, railway, deployment, restart]
    homepage: https://github.com/NousResearch/hermes-agent
---

# self-restart

Trigger a Railway redeploy of the hermes-agent service. The container exits and a new container starts on the same image; the entrypoint refreshes `/opt/data/repo`, and daemons + gateway pick up the new code via symlinks.

## When to Invoke

- You just edited a **Bucket-2** file: `tools/**`, `gateway/**`, `scripts/source_watcher.py`, `scripts/cost_report_daemon.py`, top-level `*.py` (anything imported into the long-running gateway or long-running daemon scripts).
- AND you want the change live (not just persisted to git).
- AND it is **not** a rebuild-class change (not a Bucket-3 file: no Dockerfile, docker/**, pyproject.toml, package*.json, requirements*.txt, uv.lock).

**Bucket quick-ref:**
- Bucket 1 (no action): `skills/**`, `*.md`, `docs/**`, `scripts/cost_report.py`
- Bucket 2 (self-restart): `tools/**`, `gateway/**`, `scripts/source_watcher.py`, `scripts/cost_report_daemon.py`, top-level `*.py`
- Bucket 3 (self-rebuild): `Dockerfile`, `docker/**`, `pyproject.toml`, `package*.json`, `requirements*.txt`, `uv.lock`

## Action

POST the `serviceInstanceRedeploy` mutation to Railway's GraphQL API.

**Caution:** The gateway process exits when the container is told to redeploy — the current conversation will be cut short mid-reply. Do NOT send any pre-restart message to the user. Create the flag file, call the API, and let the new container send the notification.

Pending uncommitted edits in `/opt/data/repo` survive the restart (entrypoint preserves them); the watcher commits them on next boot. You can trigger restart immediately after editing without waiting for the watcher.

No user confirmation required — restart is cheap and routine.

## Implementation

```python
import os, subprocess, urllib.request, json

RAILWAY_API_TOKEN = os.environ.get("RAILWAY_API_TOKEN")
PROJECT_ID = "c49b3e8b-a36d-4d24-a972-eab5e05b881d"
ENVIRONMENT_ID = "38eea0f3-0bd3-48f4-abaf-ec3de09174de"
SERVICE_ID = "c32be0a9-9d43-49a8-bf43-764915360dfb"

mutation = """
mutation serviceInstanceRedeploy($serviceId: String!, $environmentId: String!) {
  serviceInstanceRedeploy(serviceId: $serviceId, environmentId: $environmentId)
}
"""

variables = {
    "serviceId": SERVICE_ID,
    "environmentId": ENVIRONMENT_ID,
}

body = json.dumps({"query": mutation, "variables": variables}).encode()
req = urllib.request.Request(
    "https://backboard.railway.app/graphql/v2",
    data=body,
    headers={"Authorization": f"Bearer {RAILWAY_API_TOKEN}", "Content-Type": "application/json", "User-Agent": "railway-cli/4.44.0"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as resp:
    result = json.loads(resp.read())

print(json.dumps(result, indent=2))
```

## Known Failure Modes

### Mode A — HTTP 403 from Cloudflare (code 1010 "Access denied")
... (already patched) ...

### Mode B — "Cannot query field"
The mutation name was renamed (e.g. `serviceInstanceRedeploy` → `deploymentRedeploy`).

### Mode C — "Not Authorized" (INTERNAL_SERVER_ERROR)
The GraphQL API accepts the request but the token lacks the specific `serviceInstanceRedeploy` permission.
- Symptom: HTTP 200 response with `{"errors": [{"message": "Not Authorized", "extensions": {"code": "INTERNAL_SERVER_ERROR"}}]}`
- Distinguishes from Mode A (403 direct) and Mode B ("Cannot query field")
- Three distinct sub-causes — check in order:

  **C1 — Wrong project token**: Token belongs to a different Railway project.
  - Verify: introspection works (164 mutations visible) but `project(id:)` query returns Not Authorized
  - Fix: get a token from **Settings → API Tokens** of the **correct** project (`pretty-amazement`)

  **C2 — Account not a project member**: Owning account is not a collaborator on the project.
  - Verify: project query returns Not Authorized even with the correct project token
  - Fix: **Settings → Members** → invite the account that owns the token; acceptance required; role must be Admin or Developer (not Viewer). A "pending" invite means the account is NOT yet a member.
  - **Confirmed fix this session**: invite the same account that owns the token; accept fully; role confirmed as Developer.

  **C3 — Token missing scopes**: Token is from the right project/account but lacks permissions.
  - Verify: token belongs to correct project AND account is a member with Admin/Developer role, but still fails
  - Fix: regenerate token at https://railway.app/account with `read` + `write` scopes

- **NEW env var value requires dashboard redeploy**: Railway injects env vars at container **startup**, not dynamically. Updating `RAILWAY_API_TOKEN` in the Railway dashboard while the container is running has NO effect — the running container still has the old value. Must manually redeploy via the Railway dashboard Deployments tab (Restart or Redeploy button).

### Mode D — Env var changed but container hasn't restarted
Same as C3 note above — Railway injects env vars at container **startup**. Updating `RAILWAY_API_TOKEN` in the dashboard while the container is running has NO effect. Must manually redeploy via the Railway dashboard.

### Diagnosing Which Failure Mode
## References

- `references/railway-restart-this-container.md` — IDs, Python restart code, "I'm back!" flag file trick, failure modes for this specific Railway deployment.

## Pitfalls

### Programmatic restarts don't fire the startup message
`_send_restart_notification()` only fires when `/restart` was triggered from Telegram (the flag file is written by the command handler, not by the API call). Before calling the Railway API from inside the container, pre-create the flag:
```python
import json, os
data = {'platform': 'telegram', 'chat_id': '8746106424'}
path = os.path.join(os.environ.get('HERMES_HOME', '/opt/data'), '.restart_notify.json')
with open(path, 'w') as f:
    json.dump(data, f)
```

### Wrong Railway IDs in skill
The skill may contain stale IDs. Cross-check against the Railway dashboard URL: `railway.app/project/<PROJECT_ID>/service/<SERVICE_ID>`. Current correct IDs are in `references/railway-restart-this-container.md`.

### No curl/wget — Python only
This container has no `curl` or `wget`. Use Python stdlib `urllib` with `User-Agent: railway-cli/4.44.0` header.

## Verification

Print the raw API response verbatim. Tell the user clearly: **"Restarting now — your message will be answered by the new container in ~30s."**

## Mode C — Token from Wrong Project (Not Authorized on data queries)

The token can reach the API (introspection works) but returns "Not Authorized" on any data mutation or query.

**Diagnosis:** Run introspection — if it succeeds, the token is valid but lacks project access:
```python
# Introspection succeeds: shows 164 mutation fields including serviceInstanceRedeploy
# But project queries/mutations fail: "Not Authorized"
```

**Root cause:** The Railway account the token belongs to is not a member of the `pretty-amazement` project — even if the invite was sent and "accepted", if it shows as "pending" in project settings, the token has no access.

**Fix:**
1. Go to Railway dashboard → project → **Settings → Members**
2. Verify your account is listed as **Admin** or **Developer** (not Viewer, not Pending)
3. If invite shows as pending: revoke and re-invite, accept while logged into the correct account
4. Redeploy/restart the container from Railway dashboard so it picks up fresh env vars

**Why restart is needed:** Railway injects env vars at container start. Changing the token in the dashboard doesn't affect a running container.

## If the mutation fails

Railway has renamed GraphQL mutations historically. Three distinct failure modes are known:

**Mode A — HTTP 403 from Cloudflare (code 1010 "Access denied"):**
The token lacks `backboard` scope, or is a deploy-token rather than an account-level API token. Cloudflare returns this before Railway's API sees the request.
- Verify: the introspection query also fails with 403
- Fix: generate a new token at https://railway.app/account with `read` + `write` scopes; update `RAILWAY_API_TOKEN` in Railway project settings
- **Also**: Cloudflare requires a `User-Agent` header that looks like a real browser/CLI client. Python's urllib sends no User-Agent by default and gets 403. Always include `User-Agent: railway-cli/4.44.0` (or similar) in the request headers.

**Mode B — "Cannot query field":**
The mutation name was renamed (e.g. `serviceInstanceRedeploy` → `deploymentRedeploy`).

### Diagnosing Which Failure Mode

1. **Try the redeploy mutation first** (the code above).
2. **If it fails with 403**, run the introspection probe:
```bash
python3 -c "
import os, urllib.request, json
token = os.environ.get('RAILWAY_API_TOKEN')
query = '{\"query\":\"{ __schema { mutationType { fields { name } } } }\"}'
body = query.encode()
req = urllib.request.Request(
    'https://backboard.railway.app/graphql/v2',
    data=body,
    headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
    method='POST',
)
with urllib.request.urlopen(req, timeout=30) as resp:
    result = json.loads(resp.read())
for f in result.get('data', {}).get('__schema', {}).get('mutationType', {}).get('fields', []):
    if 'redeploy' in f['name'].lower():
        print(f['name'])
"
```

3. **Interpret the result:**
   - **Introspection also 403s** → **Mode A** (bad token). The introspection probe itself will 403, so you cannot use it to discover a renamed mutation. Generate a new account-level token.
   - **Introspection returns fields but redeploy fails with "Cannot query field"** → **Mode B** (mutation renamed). Use the name from introspection.
   - **Introspection returns 200 but no redeploy field** → Railway may have removed the redeploy mutation entirely; check Railway docs or use the Railway CLI from outside the container instead.
   - **Introspection returns 200 with redeploy field, project queries return "Not Authorized"** → **Mode C** (token not a member of the target project). See Mode C section.

### Redeploy vs Restart — What's the Difference

The Railway dashboard **Redeploy** button and `serviceInstanceRedeploy` mutation do the same thing: stop the current container and start a new one on the **same image**. They do NOT re-run the entrypoint.

**What survives a redeploy:**
- Env vars (they're baked into the new container at start, so changing them in Railway dashboard has no effect until next container start)
- Volume data (`/opt/data/`)

**What is reset on redeploy:**
- Entry point runs fresh: re-links `scripts/`, `tools/`, `gateway/` from the git checkout at `origin/main` as of the **original boot time** — not current HEAD
- Self-edits to `/opt/data/repo/` made after container start are **LOST** on redeploy

**What requires a restart (self-restart or dashboard restart):**
- Picking up edits to `/opt/data/repo/` that were pushed to GitHub
- Picking up new env var values
- Re-running the entrypoint fresh

**What requires a rebuild (push to release-pin):**
- Changes to Bucket-3 files: `Dockerfile`, `docker/**`, `pyproject.toml`, `package*.json`, `requirements*.txt`, `uv.lock`

## Container tool limitations

This Railway container has **no `curl` or `wget`** — only Python stdlib `urllib`. The 403 cannot be bypassed with alternate HTTP clients. If stdlib urllib is blocked by Cloudflare and a token fix is not possible, fall back to:
- Railway CLI run from outside the container: `railway redeploy`
- Railway dashboard: manual redeploy button at railway.app
