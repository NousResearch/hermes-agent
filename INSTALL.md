# Hermes-Agent — Installation Guide

Deploy the JZKK720/hermes-agent fork with Ollama, PostgreSQL, and WeChat personal-account chat using Docker Compose.

## Prerequisites

| Requirement | Install |
|---|---|
| Docker + Docker Compose v2 | https://docs.docker.com/get-docker/ |
| Git | https://git-scm.com/ |
| Ollama | https://ollama.com/ |

### Pull the default model

```bash
ollama pull gemma4:e4b-it-q8_0
```

To use a different model, follow the [Change the model](#change-the-model) section after install.

---

## Quick Install (one command)

```bash
curl -fsSL https://raw.githubusercontent.com/JZKK720/hermes-agent/main/docker/deploy.sh | bash
```

The script clones the repo, seeds `data/.env`, and recreates all services from the fork's GHCR-published `ghcr.io/jzkk720/hermes-agent:latest` image with the upstream-image compose file.

Routine updates on an existing machine stay pinned to the fork's GHCR image:

```bash
git pull origin main
docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans
```

To fall back to the upstream Docker Hub image, set `HERMES_FORK_IMAGE=docker.io/nousresearch/hermes-agent:latest` before running compose.

---

## Manual Install

### 1. Clone the fork

```bash
git clone https://github.com/JZKK720/hermes-agent.git
cd hermes-agent
```

### 2. Create the data directory and env file

```bash
mkdir -p data
cp docker/hermes-env.example data/.env
```

Edit `data/.env` if needed (see [Configuration](#configuration) below). For a plain Ollama setup no changes are required.

### 3. Start all services

```bash
docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans
```

This keeps the local `data/.env`, `data/config.yaml`, sessions, memories, and PostgreSQL data while pulling the fork's GHCR-published image. Use this pulled-image compose path for routine refreshes on this fork; do not switch the machine to a fork-owned build lane for normal updates.

---

## Services

| Service | URL | Description |
|---|---|---|
| Hermes Web UI | http://localhost:9119 | Chat interface |
| WeChat gateway | outbound only | `hermes-gateway` runs the gateway with `weixin` (WeChat personal) enabled. No host port — uses outbound long-poll to Tencent iLink. |
| PostgreSQL | localhost:5433 | Internal database (host port 5433) |

### Connect WeChat (one-shot)

The `hermes gateway setup` subcommand does **not** accept a platform argument — it opens an interactive menu. From the menu, pick `Weixin / WeChat` (item **13** in the default list).

```bash
docker compose -f docker-compose.upstream.yml run --rm hermes-gateway \
    hermes gateway setup
```

When prompted `Select [1-27] (27):`, type `13` and press Enter.

The wizard prints an ASCII QR code in the terminal. Scan it with the WeChat app on your phone (Settings → Plugins → search "iLink Bot" if not visible), confirm in WeChat, and the wizard saves credentials to `data/weixin/accounts/<account_id>.json` automatically.

Then bring the gateway up:

```bash
docker compose -f docker-compose.upstream.yml up -d hermes-gateway
```

You only need to run the wizard once per WeChat account. To re-bind a different account, repeat the command and the wizard will save a separate JSON file.

### Interactive CLI

```bash
docker exec -it hermes-web hermes
```

### Useful commands

```bash
docker compose -f docker-compose.upstream.yml logs -f                                                # stream logs from all services
docker compose -f docker-compose.upstream.yml logs -f hermes-web                                     # web UI logs only
docker compose -f docker-compose.upstream.yml logs -f hermes-gateway                                 # WeChat gateway logs only
docker compose -f docker-compose.upstream.yml down                                                   # stop all services
docker compose -f docker-compose.upstream.yml down -v                                                # stop + delete volumes (resets data!)
docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans # start or refresh from the fork GHCR image
```

### Smoke test

```bash
docker compose ps
curl -fsS http://127.0.0.1:9119/api/status
```

The web UI healthcheck on `:9119` is the primary smoke signal. The WeChat gateway has no public HTTP port — verify with `docker compose logs hermes-gateway` and look for a successful iLink `getupdates` long-poll connection.

---

## Configuration

### `data/.env` — secrets (never committed to git)

| Variable | Purpose | Required |
|---|---|---|
| `POSTGRES_PASSWORD` | PostgreSQL container password (matches compose default `changeme`) | No — defaults to `changeme` |
| `TELEGRAM_BOT_TOKEN` | Telegram gateway | Optional |
| `DISCORD_BOT_TOKEN` | Discord gateway | Optional |
| `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | Slack gateway | Optional |
| `EXA_API_KEY` | AI-native web search | Optional |

WeChat personal accounts (`weixin`) do not use tokens — credentials are obtained via the QR wizard and saved to `data/weixin/accounts/`.

> **Weixin "Session expired" fix:** If the gateway logs `Session expired; pausing for 10 minutes`, check `data/.env` for a wrong `WEIXIN_BASE_URL=https://ilinkai.wechat.com` (missing `.qq.com`). The correct URL is `https://ilinkai.weixin.qq.com`. To pin it permanently in `data/config.yaml` (wins over any stale env var):
> ```bash
> docker compose -f docker-compose.upstream.yml run --rm --no-deps --entrypoint "" hermes-gateway python3 -c "
> import yaml
> with open('/opt/data/config.yaml') as f:
>     cfg = yaml.safe_load(f)
> wx = cfg.setdefault('platforms', {}).setdefault('weixin', {})
> wx.setdefault('extra', {})['base_url'] = 'https://ilinkai.weixin.qq.com'
> with open('/opt/data/config.yaml', 'w') as f:
>     yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
> print('weixin base_url pinned')
> "
> docker restart hermes-gateway
> ```

### `data/config.yaml` — runtime settings

Auto-created from `docker/hermes-config.yaml` on first start.  
Edit directly; no rebuild needed — takes effect on next container restart.

### Change the model

Edit `data/config.yaml`:

```yaml
model:
  default: "llama3.3:70b"        # any model pulled in Ollama
  context_length: 131072
```

Then restart the containers:

```bash
docker compose restart hermes-web hermes-gateway
```

---

## Keeping Up to Date

### Pull your own fork changes

```bash
git pull origin main
docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans
```

This is the normal update path for this fork. Keep regular refreshes pinned to the published upstream Docker Hub image through the fork wrapper; do not treat `upstream/main` merges as part of the routine local update flow for this machine.

### Refresh containers from the published upstream image

```bash
docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans
```

This updates the Hermes containers from the published Docker Hub image while keeping your local `data/.env`, `data/config.yaml`, sessions, and other persisted runtime data.
---

## Troubleshooting

### Web UI not reachable

```bash
docker compose ps              # check service state
docker compose logs hermes-web # read startup output
```

### Model calls fail / "unknown provider"

Ensure Ollama is running on the host and the model is pulled:

```bash
ollama list
ollama pull gemma4:e4b-it-q8_0
```

Check `data/config.yaml` has `provider: "custom"` (not `"ollama"`).

### `hermes-web` and `hermes-gateway` drift to different image digests

Recreate both services together so the dashboard and gateway stay on the same upstream image generation:

```bash
docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans
```

### Pull fails resolving `ghcr.io/jzkk720/hermes-agent:latest`

That reference is the fork's GHCR image source. The fork publishes its own image (with Weixin QR onboarding + dashboard auth) to GHCR via the `fork-ghcr-publish.yml` workflow. If Docker reports a timeout or `failed to do request: Head ...`, treat it as a GHCR connectivity or auth problem on the machine.

```bash
docker login ghcr.io
docker pull ghcr.io/jzkk720/hermes-agent:latest
docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans
```

### `8644` does not respond

Webhook platform is no longer pre-published in this stack. To enable inbound webhooks, bind `:8644` in `docker-compose.yml` on the `hermes-gateway` service and set `platforms.webhook.enabled: true` in `data/config.yaml`.

### Reset everything (start fresh)

```bash
docker compose down -v   # removes named volumes (including postgres_data — all DB rows gone)
rm -rf data/             # removes persisted data (config, sessions, memories, weixin/accounts)
docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans
```

---

## Ports Reference

| Port (host) | Port (container) | Service |
|---|---|---|
| 9119 | 9119 | Hermes Web UI (dashboard) |
| 8642 | 8642 | API server (OpenAI-compatible `/v1` endpoint) — enable via `API_SERVER_KEY` in `data/.env` |
| 8789 | 8789 | Gateway health endpoint (`/health`) — used by dashboard for cross-container status |
| 8644 | 8644 | Webhook inbound — disabled by default; enable in `data/config.yaml` under `platforms.webhook` |
| 5433 | 5432 | PostgreSQL |

> **WeChat (weixin) gateway**: uses outbound long-poll to Tencent iLink — no inbound host port needed.
> **API server**: disabled by default. Set `API_SERVER_KEY` in `data/.env` to enable the OpenAI-compatible endpoint at `http://localhost:8642/v1`. See `.github/instructions/api-gateway-ports.instructions.md` for full configuration details.
