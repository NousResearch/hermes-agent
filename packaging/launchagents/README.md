# macOS LaunchAgents (deployment snapshots)

These plist files are **committed snapshots** of the deployed macOS
LaunchAgents. They are the single source of truth for how the three
Hermes services are launched by `launchd`:

| Label | Purpose | ProgramArguments |
|-------|---------|------------------|
| `ai.hermes.gateway` | Gateway process (FastAPI/aiohttp on `localhost:8642`) | `python -m hermes_cli.main gateway run --replace` |
| `com.hermes.tunnel` | Cloudflare Quick Tunnel supervisor | `bash scripts/run-tunnel.sh` |
| `ai.hermes.monitor` | macOS menu-bar monitor (rumps) | `python dashboard/hermes-monitor-mac.py` |

## Deployment philosophy (2026-04-12 reorganization)

The running agents must reference **this repo's paths** — never `~/.hermes/`
for code. Drift caused real outages: on 2026-04-12 a gateway restart left
`cloudflared` with stale origin connections because `~/.hermes/dashboard/*`
had diverged from the fork and nobody could tell which copy was authoritative.

After this reorganization:

- `~/.hermes/` holds **runtime state only**: `.env`, `logs/`, `sessions/`,
  `state.db`, `cache/`, `gateway.pid`, `venv/` (pip-installed deps for monitor).
- `~/workspace/hermes-agent/` holds **all code**: `dashboard/api.py`,
  `dashboard/index.html`, `dashboard/hermes-monitor-mac.py`, `scripts/run-tunnel.sh`,
  `scripts/sync-cloudflare-worker.sh`, `gateway/platforms/api_server.py`.

## Install / update

```bash
# One-time: install plists into ~/Library/LaunchAgents
cp packaging/launchagents/*.plist ~/Library/LaunchAgents/

# Apply plist changes (path/env edits): bootout + bootstrap, NOT kickstart
UID=$(id -u)
launchctl bootout   gui/$UID ~/Library/LaunchAgents/ai.hermes.gateway.plist
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/ai.hermes.gateway.plist

# Apply code changes (api_server.py, run-tunnel.sh, monitor): kickstart is enough
launchctl kickstart -k gui/$UID/ai.hermes.gateway
launchctl kickstart -k gui/$UID/com.hermes.tunnel
launchctl kickstart -k gui/$UID/ai.hermes.monitor
```

## Environment variables (gateway)

Set in `ai.hermes.gateway.plist` under `EnvironmentVariables`:

| Key | Purpose |
|-----|---------|
| `PATH` | Must include `venv/bin` and homebrew paths |
| `VIRTUAL_ENV` | Points at `hermes-agent/venv` |
| `HERMES_HOME` | `/Users/mong-e/.hermes` (runtime state) |
| `HERMES_FC_RAG_ENABLED` | `1` — gates `/api/fc-rag-log` route registration |

Add `HERMES_DASHBOARD_DIR` here only if you want to override the default
(repo-relative `dashboard/`) with a detached copy.

## Verification

```bash
curl -s -o /dev/null -w "gateway=%{http_code}\n"   http://localhost:8642/health
curl -s -o /dev/null -w "fc-rag-log=%{http_code}\n" \
     -X POST http://localhost:8642/api/fc-rag-log \
     -H "Authorization: Bearer lexdiff-hermes-local" \
     -H "Content-Type: application/json" -d '{}'
curl -s -o /dev/null -w "tunnel=%{http_code}\n" \
     https://openclaw-bridge.ryuseungin.workers.dev/health \
     -H "Authorization: Bearer lexdiff-hermes-local"
```

Expected: `gateway=200`, `fc-rag-log=202`, `tunnel=200`.
