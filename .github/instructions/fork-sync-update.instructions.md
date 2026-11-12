---
description: "Use after git pull or git merge on this fork to apply pending runtime fixes. Covers the weixin base_url config pin, data/config.yaml patching, GHCR image refresh, and post-sync verification. Automatically triggers when the user says 'pull', 'sync', 'update', or 'fix weixin' after a fork sync."
name: "Fork Sync Update & Fix"
applyTo:
  - "docker/deploy.sh"
  - "docker-compose.upstream.yml"
  - "docker-compose.yml"
  - "docker/hermes-config.yaml"
  - "data/.env"
  - "data/config.yaml"
  - "INSTALL.md"
  - "gateway/platforms/weixin_qr_session.py"
  - "plugins/platforms/weixin/adapter.py"
---

# Fork Sync Update & Fix

After pulling or merging changes from `origin/main`, run these steps to
apply pending runtime fixes that can't be carried by git alone (because
`data/.env` and `data/config.yaml` are gitignored local state).

## When to use

- After `git pull origin main` on any machine
- After merging upstream into the fork
- When the user says "sync", "update", "pull latest", "fix weixin"
- When weixin is not connecting after a code update

## Step 1 — Pull latest

```bash
cd hermes-agent
git pull origin main
```

## Step 2 — Refresh the GHCR image and recreate containers

```bash
docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans
```

This pulls the latest `ghcr.io/jzkk720/hermes-agent:latest` (auto-built
from the fork's `main` branch by the `fork-ghcr-publish.yml` workflow)
and recreates all containers while preserving `data/.env`,
`data/config.yaml`, sessions, memories, and the PostgreSQL volume.

## Step 3 — Pin weixin base_url in data/config.yaml

The gateway reads its runtime config from `data/config.yaml` (gitignored,
generated locally). A stale `WEIXIN_BASE_URL` in `data/.env` can override
the correct iLink endpoint and cause silent "Session expired" errors.

The deploy script does this automatically, but for manual syncs:

```bash
docker compose -f docker-compose.upstream.yml run --rm --no-deps \
    --entrypoint "" hermes-gateway python3 -c "
import yaml
with open('/opt/data/config.yaml') as f:
    cfg = yaml.safe_load(f)
wx = cfg.setdefault('platforms', {}).setdefault('weixin', {})
wx.setdefault('extra', {})['base_url'] = 'https://ilinkai.weixin.qq.com'
with open('/opt/data/config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print('weixin base_url pinned')
"
```

This is idempotent — safe to run even if the pin is already present.

Alternatively, just run the deploy script which does this automatically:

```bash
bash docker/deploy.sh
```

## Step 4 — Restart the gateway

```bash
docker restart hermes-gateway
```

## Step 5 — Verify

```bash
# Wait for the first long-poll cycle (35s)
sleep 40

# Check for weixin errors — no output means healthy
docker logs hermes-gateway --since 1m 2>&1 | grep -iE "weixin|ilink|expired|errcode"

# Check dashboard status
docker exec hermes-gateway sh -c "curl -s http://hermes-web:9119/api/status | python3 -m json.tool" | grep -A5 weixin
```

Expected: `weixin.state: "connected"`, `error_code: null`, no
`Session expired` lines in the logs.

## Step 6 — QR login (only if token is expired)

If the logs show `Session expired; pausing for 10 minutes` even after
the base_url pin, the iLink bot token itself has expired (errcode=-14).
Each machine needs its own QR login:

1. Open `http://localhost:9119` → **Channels** → **Weixin / WeChat** → **Set up with QR**
2. Scan with WeChat on your phone → confirm login
3. The dashboard auto-saves credentials and restarts the gateway

Or via CLI:
```bash
docker compose -f docker-compose.upstream.yml run --rm hermes-gateway hermes gateway setup
```
Pick **Weixin / WeChat** (item 13).

## Step 7 — Validate that every configured model actually exists

`data/config.yaml` is gitignored, so a model can be configured and then never
pulled. Ollama answers with `model '<name>' not found`, and because the
**auxiliary** models (compression, vision) run on their own schedule, the agent
shows inference errors even when the main model works.

```bash
# every model the config names
grep -E '^\s*(default|model):' data/config.yaml | sed 's/.*: *//' | sort -u
```

```bash
# what is actually pulled
curl -s http://127.0.0.1:11434/api/tags | python3 -c "import json,sys;print('\n'.join(m['name'] for m in json.load(sys.stdin)['models']))"
```

Any name in the first list that is missing from the second is a live bug. Confirm
a candidate really generates before switching to it — `200` on `/v1/models` proves
the endpoint, not that the tag exists:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<tag>","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'
```

## Step 8 — Make sure no host process is shadowing the dashboard port

`compose` binds the dashboard to `127.0.0.1:9119`. A host-native
`hermes dashboard` (for example one started from the repo `.venv`) takes that
address first, so the browser reaches the **old** code while the container is
healthy and unreachable. The symptom is a dashboard reporting a stale
`version`/`config_version` — e.g. `0.21.5` / `40` when the container serves
`53`.

```bash
# who owns the port? Docker's proxy, or a stray python?
curl -s http://127.0.0.1:9119/api/status | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['version'], d['config_version'], d['latest_config_version'])"
```

If `version` is `unknown` you are on the container (fine — the image reports the
git stamp). If it names an old release, a host process owns the port: stop it and
then `docker compose ... up -d --force-recreate hermes-web` so Docker rebinds.

> A gateway reporting `DEGRADED` with `weixin: WEIXIN_TOKEN is required` is not a
> broken upgrade: weixin has no credentials. Either scan the QR code or set
> `platforms.weixin.enabled: false`; the dashboard QR apply handler re-enables it
> on its own.

## What the fixes cover

| Fix | Layer | How it propagates |
|-----|-------|-------------------|
| `weixin_qr_session.py` normalisation | Source (baked into GHCR image) | `git pull` → GHCR auto-rebuild → `docker compose pull` |
| `base_url` pin in `docker/hermes-config.yaml` | Template (bind-mounted as `cli-config.yaml.example`) | `git pull` — used for fresh installs |
| `base_url` pin in `data/config.yaml` | Runtime config (gitignored) | Deploy script patches it, or run the python one-liner above |
| `data/.env` cleanup | Local secrets (gitignored) | Manual — remove any `WEIXIN_BASE_URL=https://ilinkai.wechat.com` line |

## Quick reference: one-command sync

```bash
git pull origin main && bash docker/deploy.sh
```

The deploy script handles steps 2–3 automatically. Only step 4 (restart)
and step 6 (QR login) may be needed manually.

## Post-sync port consistency check

After pulling, verify that `docker-compose.yml` and
`docker-compose.upstream.yml` expose the same gateway ports. The
canonical set is `8642` (API server), `8789` (health), `8644` (webhook).
If they diverge, sync `docker-compose.yml` to match
`docker-compose.upstream.yml`. See
[api-gateway-ports.instructions.md](api-gateway-ports.instructions.md)
for the full port map.