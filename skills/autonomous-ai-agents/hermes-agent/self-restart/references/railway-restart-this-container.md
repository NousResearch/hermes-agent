# Railway Restart — This Container

## IDs (correct as of May 2026)

- **SERVICE_ID**: `c32be0a9-9d43-49a8-bf43-764915360dfb`
- **ENVIRONMENT_ID**: `38eea0f3-0bd3-48f4-abaf-ec3de09174de`
- **PROJECT_ID**: `c49b3e8b-a36d-4d24-a972-eab5e05b881d`

## How to Restart

Python only — no `curl` or `wget` in this container:

```python
import json, os, urllib.request

# Create flag file so new container sends "👋 I'm back!" via _send_restart_notification
data = {'platform': 'telegram', 'chat_id': '8746106424'}
path = os.path.join(os.environ.get('HERMES_HOME', '/opt/data'), '.restart_notify.json')
with open(path, 'w') as f:
    json.dump(data, f)

# Restart this container
mutation = 'mutation { serviceInstanceRedeploy(serviceId: "c32be0a9-9d43-49a8-bf43-764915360dfb", environmentId: "38eea0f3-0bd3-48f4-abaf-ec3de09174de") }'
body = json.dumps({'query': mutation}).encode()
req = urllib.request.Request(
    'https://backboard.railway.app/graphql/v2',
    data=body,
    headers={
        'Authorization': f'Bearer {os.getenv("RAILWAY_API_TOKEN")}',
        'Content-Type': 'application/json',
        'User-Agent': 'railway-cli/4.44.0',
    },
    method='POST',
)
with urllib.request.urlopen(req, timeout=30) as resp:
    print(json.loads(resp.read()))
```

## Sending "I'm back!" on Startup

The gateway's `_send_restart_notification()` (gateway/run.py ~line 8727) reads `~/.hermes/.restart_notify.json` — it only fires when the **previous** container session triggered a `/restart` command from Telegram. The flag file is written by the `/restart` command handler.

**For programmatic restarts** (triggered via Railway API from inside the container, not from a Telegram `/restart` command), pre-create the flag file before calling the API:

```python
import json, os

data = {'platform': 'telegram', 'chat_id': '8746106424'}
path = os.path.join(os.environ.get('HERMES_HOME', '/opt/data'), '.restart_notify.json')
with open(path, 'w') as f:
    json.dump(data, f)
# Then call the redeploy mutation
```

The message sent is `"I'm back!"` (plain text) — configured in `gateway/run.py` line 8754.

## Known Failure Modes

### Mode C — "Not Authorized" on redeploy mutation
The GraphQL mutation returns `{"errors": [{"message": "Not Authorized", ...}]}`.

**Cause**: Token lacks `serviceInstanceRedeploy` permission, OR wrong service/environment IDs.

**Fix**: 
1. Verify token has `read` + `write` scopes at railway.app/account
2. Verify IDs match those above (skill may have stale values — cross-check against Railway dashboard URL)

### Mode A — HTTP 403 from Cloudflare
Cloudflare WAF blocks `urllib` requests without a `User-Agent` header. Always include `User-Agent: railway-cli/4.44.0`.

## What Survives a Restart

- `/opt/data/` volume (sessions, skills, memories, logs, `.restart_notify.json`)
- Env vars (baked in at container start — changes in Railway dashboard don't affect running container until next restart)
- Gateway code in `/opt/data/repo/gateway/` (symlinked from git checkout at `origin/main` as of container start time)
