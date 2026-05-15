---
name: self-restart
description: "Trigger a Railway redeploy of the hermes-agent service. Despite the mutation name implying a fast same-image restart, observed 2026-05-15 to take ~8-10 minutes (full rebuild from release-pin), comparable to a release-pin push. Treat as rebuild-class for downtime planning."
version: 1.2.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [hermes, railway, deployment, restart]
    homepage: https://github.com/NousResearch/hermes-agent
---

# self-restart

Trigger a Railway redeploy of the hermes-agent service. The container exits and a new one starts; the entrypoint refreshes `/opt/data/repo` and re-symlinks workspace skills, and daemons + gateway pick up the new code.

**Observed duration (2026-05-15):** ~8 minutes BUILDING + ~1 minute DEPLOYING ≈ ~10 minutes total. Despite the `serviceInstanceRedeploy` mutation name, Railway appears to perform a full rebuild from `release-pin`, not a fast same-image restart. Either Railway changed behavior or earlier-documented "~30s same-image" claim was always wrong. **Treat as rebuild-class for planning** until proven otherwise: pre-warn the user of ~10 min downtime, don't chain rapid restarts.

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

**Caution:** The gateway process exits when the container is told to redeploy — the current conversation will be cut short mid-reply. **Tell the user "Restarting now — the new container will be ready in ~10 minutes (full rebuild)" before triggering.** Matches GordonClaw's pattern: the heads-up gets cut off, but the user knows what to expect; the entrypoint's `"👋 I'm back!"` Telegram message in the next container start is the resumption signal.

Pending uncommitted edits in `/opt/data/repo` survive the restart (entrypoint preserves them); the watcher commits them on next boot. You can trigger restart immediately after editing without waiting for the watcher.

User confirmation **IS** required — restart is rebuild-class (~10 min, costs the same as a release-pin push). It is no longer a "cheap" operation worth taking unilaterally.

## Implementation

Use `curl` (matches GordonClaw's `gordonclaw-self-restart` skill — same recipe, same failure modes, shared notes). The `User-Agent` header is mandatory: Cloudflare's WAF blocks default UAs (Python urllib, etc.) with 403 before the request reaches Railway.

**Write the planned-redeploy marker first.** The gateway's SIGTERM handler reads this marker and exits 0 instead of 1, so Railway doesn't fire a spurious "Deploy Crashed!" notification when it tears down the container for the redeploy.

```bash
HERMES_HOME="${HERMES_HOME:-/opt/data}"
touch "$HERMES_HOME/.planned_redeploy"

curl -s -X POST https://backboard.railway.app/graphql/v2 \
  -H "Authorization: Bearer $RAILWAY_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "User-Agent: railway-cli/4.44.0" \
  -d '{
    "query": "mutation { serviceInstanceRedeploy(serviceId: \"c32be0a9-9d43-49a8-bf43-764915360dfb\", environmentId: \"38eea0f3-0bd3-48f4-abaf-ec3de09174de\") }"
  }'
```

The IDs are baked in (not secrets):
- Project: `pretty-amazement` (`c49b3e8b-a36d-4d24-a972-eab5e05b881d`)
- Service: `c32be0a9-9d43-49a8-bf43-764915360dfb`
- Environment (production): `38eea0f3-0bd3-48f4-abaf-ec3de09174de`

`RAILWAY_API_TOKEN` is set as a service env var in Railway. If unset, the curl will fail with a 401; tell the user the token isn't configured and stop.

### Programmatic restarts and the "I'm back!" flag file

`gateway/run.py:_send_restart_notification()` only fires when `/restart` was triggered from chat (the flag file is written by the chat handler, not by the API call). To get the chat-side restart message after a programmatic restart, pre-create the flag *before* curl:

```bash
HERMES_HOME="${HERMES_HOME:-/opt/data}"
printf '{"platform":"telegram","chat_id":"%s"}' "$TELEGRAM_HOME_CHANNEL" \
  > "$HERMES_HOME/.restart_notify.json"
```

The entrypoint's `"👋 I'm back!"` Telegram message fires on every container start regardless — that one doesn't need this flag.

## Known Failure Modes

### Mode A — HTTP 403 from Cloudflare (code 1010 "Access denied")
Almost always the User-Agent header is missing or set to a default urllib/python UA. Cloudflare blocks before the token is checked. Verify the request includes `User-Agent: railway-cli/4.44.0` (or any non-default UA).

If introspection ALSO 403s, it's a real auth problem — generate a new account-level token at https://railway.app/account with `read` + `write` scopes.

### Mode B — "Cannot query field"
The mutation name was renamed (e.g. `serviceInstanceRedeploy` → `deploymentRedeploy`). Introspect to find the current name:

```bash
curl -s -X POST https://backboard.railway.app/graphql/v2 \
  -H "Authorization: Bearer $RAILWAY_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "User-Agent: railway-cli/4.44.0" \
  -d '{"query":"{ __schema { mutationType { fields { name } } } }"}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('\n'.join(f['name'] for f in d['data']['__schema']['mutationType']['fields'] if 'edeploy' in f['name'].lower() or 'estart' in f['name'].lower()))"
```

Update this skill with the new mutation name.

### Mode C — "Not Authorized" (INTERNAL_SERVER_ERROR)
HTTP 200 with `{"errors":[{"message":"Not Authorized","extensions":{"code":"INTERNAL_SERVER_ERROR"}}]}`. Three sub-causes:

- **C1 — Wrong project token**: token belongs to a different Railway project. Get a token from the **correct** project's Settings → API Tokens.
- **C2 — Account not a project member**: invite the token's account to **Settings → Members** as Admin or Developer (not Viewer); accept the invite fully (no "pending").
- **C3 — Token missing scopes**: regenerate at https://railway.app/account with `read` + `write`.

Note: Railway injects env vars at container **startup**, not dynamically. Updating `RAILWAY_API_TOKEN` in the dashboard while the container is running has NO effect — must trigger a manual redeploy from the dashboard so the new token takes effect.

## Required env

- `RAILWAY_API_TOKEN` — workspace-scoped Railway API token, set as a service env var in Railway dashboard. NOT a GitHub secret.
- `TELEGRAM_HOME_CHANNEL` — only required if you want the chat-side "I'm back!" notification (the flag-file trick above).

## What never to do

- Never use the `railway` CLI from the container — the CLI requires a login flow we haven't set up. Use the GraphQL API directly via curl.
- Never call this skill without telling the user first. Restart cuts off any in-flight conversation.
- Never call this skill for Bucket 1 changes (skills, brief scripts, docs). They're already live; a restart is wasted.
- Never strip the `User-Agent` header thinking it's optional. It isn't.

## Cross-reference

GordonClaw uses the identical recipe in its `gordonclaw-self-restart` skill (different project/service/environment IDs and token, same curl invocation). Keep the two skills in lockstep — when you fix or extend one, mirror the change to the other.

## Verification

Print the raw API response verbatim. Tell the user clearly: **"Restarting now — the new container will be ready in ~10 minutes (full rebuild from release-pin)."** Do not promise ~30s; the observed behavior is rebuild-class.
