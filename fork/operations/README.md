# Fork Operations (Windows)

Day-to-day service control for this workstation: gateway, harness, dashboard,
desktop, Tailscale Serve, and cron-sidecar scripts.

## Hermes stack scripts

| Script | Purpose |
|--------|---------|
| `scripts/windows/restart-hermes-stack.ps1` | Idempotent restart; **skips llama** unless `-StartLlama` |
| `scripts/windows/start-hermes-gateway.ps1` | Messaging gateway (:8787) |
| `scripts/windows/start-hermes-dashboard.ps1` | Dashboard UI (:9120) |
| `scripts/windows/start-hermes-desktop.ps1` | Launch desktop (`hermes desktop --source`) |
| `scripts/windows/check-local-llm.ps1` | llama.cpp health on :8080/:8081 |

## Headless backend (desktop / remote)

```powershell
py -3 -m hermes_cli.main serve          # JSON-RPC/WS on :9119, no browser
py -3 -m hermes_cli.main serve --status
py -3 -m hermes_cli.main serve --stop
```

## Tailscale Serve

Config script: `%LOCALAPPDATA%\HermesWebUI\Update-HermesTailscaleServe.ps1`

Typical routes on `https://<machine>.ts.net`:

- `/` → gateway `127.0.0.1:8787`
- `/line` → LINE webhook `127.0.0.1:8646`
- `/v1`, `/llama/v1` → local LLM proxy (often `127.0.0.1:8081`)

## Cron / daily automation (`scripts/`)

| Script | Role |
|--------|------|
| `scripts/daily_vrchat_post.py` | VRChat photo + Irodori TTS + LM Twitterer |
| `scripts/daily_moa_provider_selector.py` | MOA free-provider evaluation + config preset |
| `scripts/daily_vrchat_post_voicevox.py` | Alternate voice path (gitignored variant) |

Register via `hermes cron` with adequate `script_timeout_seconds` (900+ for twitterer jobs).

## Python environment

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[web]"
py -3 -m hermes_cli.main gateway status
```

Use `scripts/run_tests.sh` for CI-parity tests — not raw `pytest`.

See [`AGENTS.md`](AGENTS.md) for agent troubleshooting order.
