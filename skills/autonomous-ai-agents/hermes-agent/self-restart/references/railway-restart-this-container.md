# Railway Restart — This Container

## IDs (correct as of May 2026)

- **PROJECT_ID** (`pretty-amazement`): `c49b3e8b-a36d-4d24-a972-eab5e05b881d`
- **SERVICE_ID** (`hermes-agent`):     `c32be0a9-9d43-49a8-bf43-764915360dfb`
- **ENVIRONMENT_ID** (`production`):   `38eea0f3-0bd3-48f4-abaf-ec3de09174de`

Cross-check via the Railway dashboard URL: `railway.app/project/<PROJECT_ID>/service/<SERVICE_ID>`.

## How to Restart

`curl` is available in this container (added in the Dockerfile alongside `git`/`ripgrep`/etc). The full recipe lives in the parent `SKILL.md`. Quick form:

```bash
HERMES_HOME="${HERMES_HOME:-/opt/data}"
touch "$HERMES_HOME/.planned_redeploy"

curl -s -X POST https://backboard.railway.app/graphql/v2 \
  -H "Authorization: Bearer $RAILWAY_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "User-Agent: railway-cli/4.44.0" \
  -d '{"query":"mutation { serviceInstanceRedeploy(serviceId: \"c32be0a9-9d43-49a8-bf43-764915360dfb\", environmentId: \"38eea0f3-0bd3-48f4-abaf-ec3de09174de\") }"}'
```

The `touch` writes a marker file that the gateway's SIGTERM handler reads to recognize the upcoming shutdown as planned (exit 0, no "Deploy Crashed!" notification).

The `User-Agent` header is mandatory — Cloudflare's WAF blocks default UAs (curl/X.Y.Z is fine; Python urllib's default is not) before the token is checked.

## Sending "I'm back!" on Startup

Two distinct messages, two sources — keep them straight:

1. **`docker/entrypoint.sh`** — sends `"👋 I'm back!"` to `TELEGRAM_HOME_CHANNEL` on **every** container start (Railway redeploy, restart, or rebuild). Always fires.
2. **`gateway/run.py:_send_restart_notification`** — sends `"♻ Gateway restarted successfully. Your session continues."` only when `~/.hermes/.restart_notify.json` exists. The chat handler writes that file when a user sends `/restart` from Telegram.

For programmatic restarts (API call from inside the container, not a chat `/restart`), if you want the chat-side notification too, pre-create the flag *before* calling the API:

```bash
HERMES_HOME="${HERMES_HOME:-/opt/data}"
printf '{"platform":"telegram","chat_id":"%s"}' "$TELEGRAM_HOME_CHANNEL" \
  > "$HERMES_HOME/.restart_notify.json"
```

The entrypoint message fires regardless.

## Known Failure Modes

See parent `SKILL.md` "Known Failure Modes" — Modes A (Cloudflare 403), B (renamed mutation), C (Not Authorized: wrong project / pending member / missing scopes).

## What Survives a Restart

- `/opt/data/` volume (sessions, skills, memories, logs, `.restart_notify.json`)
- Env vars (baked in at container start — changes in Railway dashboard don't affect running container until next restart)
- Repo code in `/opt/data/repo/` is reset to `origin/main` on container start (the entrypoint pulls fresh)
