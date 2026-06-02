# Kevin Local Hermes Operations

This runbook records the local-only operating surface for Kevin's Docker-based
Hermes setup. It is not an upstream installation guide.

## Current Local Shape

| Surface | Current policy |
|---|---|
| Primary runtime | Docker compose |
| Local compose file | `compose.hermes.local.yml`, ignored by Git |
| Container name | `hermes-kevin` |
| Runtime data | `.hermes-docker/`, ignored by Git |
| Dashboard | `http://127.0.0.1:9119/` |
| API server | off unless explicitly enabled |
| Global install | do not create `~/.hermes` or global `hermes` without approval |

## Git Remote Policy

This repository may have more than one Kevin-owned fork remote. That is useful,
but it can also make push/PR targeting ambiguous.

| Remote | Intended use |
|---|---|
| `upstream` | Read from official `NousResearch/hermes-agent`; push must stay disabled |
| `origin` | Kevin fork used by some local Hermes workflows |
| `fork` | Kevin fork used by dashboard stability branches |

Before any push or PR:

```bash
git status --short --branch --untracked-files=all
git remote -v
git branch -vv
scripts/check-git-remote-safety.sh
```

If the safety check fails, stop and choose the target remote/branch explicitly
before pushing.

## Dashboard Watchdog

The watchdog is a local helper for Kevin's Mac. It should not call an LLM or
consume model tokens.

Files:

- `scripts/hermes-dashboard-watchdog.sh`
- `scripts/com.kevin.hermes.dashboard.watchdog.plist`

Checks:

```bash
zsh -n scripts/hermes-dashboard-watchdog.sh
plutil -lint scripts/com.kevin.hermes.dashboard.watchdog.plist
```

## One-Shot Health Check

Use this for a manual local runtime check. It reports status only; it does not
start, restart, stop, or mutate the Docker service.

```bash
scripts/check-kevin-local-hermes-health.sh
```

The script checks:

- `docker info`
- `docker compose -f compose.hermes.local.yml ps`
- dashboard HTTP `200` at `http://127.0.0.1:9119/chat`
- dashboard WebSocket `/api/pty` when local Python has the `websockets` package
- `scripts/check-git-remote-safety.sh`

Expected dashboard response is `200`. WebSocket can be reported as skipped when
the optional Python package is missing or the dashboard does not expose a local
session token. The API server on `127.0.0.1:8642` may be off by design.

If you only need the runtime check and do not want the Git remote safety gate:

```bash
CHECK_GIT_SAFETY=0 scripts/check-kevin-local-hermes-health.sh
```

## Secrets And Runtime Files

Never commit:

- `.hermes-docker/.env`
- `.hermes-docker/auth.json`
- `.hermes-docker/logs/`
- `.hermes-docker/sessions/`
- `.notebooklm-home/`
- `.notebooklm-cli-venv/`
- `.notebooklm-playwright/`
- `.pip-cache/`
- `.uv-cache/`
- Discord tokens, OAuth tokens, Google cookies, or OpenAI auth files

## Rollback

For docs/scripts in Git:

```bash
git status --short
git restore --staged <file>
git restore <file>
```

For the one-shot health check addition only:

```bash
git restore docs/kevin-local-operations.md scripts/check-kevin-local-hermes-health.sh
```

For local runtime only, prefer stopping the compose service rather than deleting
data:

```bash
docker compose -f compose.hermes.local.yml down
```

Do not remove `.hermes-docker/` or `.notebooklm-home/` unless Kevin explicitly
approves that data cleanup.
