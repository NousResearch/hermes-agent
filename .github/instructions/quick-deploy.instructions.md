---
description: "Use when the user asks to install, deploy, or spin up the Hermes stack on a new machine from the fork. Provides the canonical one-liner and post-install WeChat onboarding steps so the local agent can execute the deployment without manual lookup."
name: "Quick Deploy from Fork"
applyTo:
  - "docker/deploy.sh"
  - "docker-compose.upstream.yml"
  - "INSTALL.md"
---

# Quick Deploy from Fork

This fork's canonical deployment path is the fork's published GHCR image
(`ghcr.io/jzkk720/hermes-agent:latest`) — built from the fork's source
(upstream + Weixin QR onboarding + dashboard auth) via the
`fork-ghcr-publish.yml` GitHub Actions workflow.
Do **not** use `docker compose up -d --build` for routine installs — that builds
from local source and is only for testing fork code changes.

## One-liner (fresh machine)

```bash
curl -fsSL https://raw.githubusercontent.com/JZKK720/hermes-agent/main/docker/deploy.sh | bash
```

Prerequisites on the target machine:
- Docker + Docker Compose v2
- Ollama running on host port 11434 with a model pulled:
  ```bash
  ollama pull gemma4:e4b-it-q8_0
  ```

## What the deploy script does

1. Clones `JZKK720/hermes-agent` (skipped if already cloned)
2. Creates `data/` and seeds `data/.env` from `docker/hermes-env.example`
3. Runs `docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans`

## Post-install: connect WeChat

Open `http://localhost:9119` in a browser → **Channels** page → click
**"Set up with QR"** on the WeChat card → scan the QR with the WeChat phone app →
confirm. Credentials are saved automatically; the gateway restarts.

No terminal, CLI, or TTY required — the web QR onboarding flow handles everything.

## Routine updates (existing machine)

```bash
cd hermes-agent
git pull origin main
docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans
```

This preserves `data/.env`, `data/config.yaml`, sessions, memories, and the
PostgreSQL volume while refreshing the published upstream image.

## Services after deploy

| Service | URL / Port | Notes |
|---------|-----------|-------|
| Hermes Web UI | `http://localhost:9119` | Dashboard + chat |
| WeChat gateway | outbound only | Long-poll to Tencent iLink, no host port |
| PostgreSQL | `localhost:5433` | Internal database |

## Smoke test

```bash
docker compose ps
curl -s http://127.0.0.1:9119/api/status | head -5
```

All three containers should show `(healthy)`.