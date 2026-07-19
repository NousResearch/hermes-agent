# Agent Harness — Instructions for Hermes AI

This folder documents how **Hermes Agent** uses the **Hypura agent harness**
(HTTP actuator daemon). It is **documentation only**.

Do **not** confuse this with [`../harness/`](../harness/) — that guide is for
**upstream merge / overlay** (`scripts/merge_tools/`). Wrong guide → wrong
actions (merge policy edits vs runtime daemon ops).

## Decision: which harness?

| Need | Read |
|------|------|
| Start/stop daemon, call `harness_*` tools, VRChat/voice via harness | **This file** |
| Merge `upstream/main`, overlay `toolsets.py`, vendor pins | [`../harness/AGENTS.md`](../harness/AGENTS.md) |

## Runtime map (do not relocate)

| Path | Role |
|------|------|
| `vendor/openclaw-mirror/extensions/hypura-harness/scripts/harness_daemon.py` | FastAPI daemon (default `127.0.0.1:18794`) |
| `vendor/openclaw-mirror/extensions/hypura-harness/config/` | Daemon config |
| `hermes_cli/harness.py` | `hermes harness start\|stop\|restart\|status` |
| `tools/openclaw/harness_client.py` | HTTP client used by tools |
| `tools/harness_tools.py` | Model tools: `harness_scavenge`, `harness_wisdom`, `harness_evolve`, `harness_speak`, `harness_osc`, `harness_status` |

`hermes_cli/harness.py` pins the daemon relative path. Moving the vendor tree
or renaming these modules without updating that pin **breaks** `hermes harness`.

## Operator / agent procedure

1. **Health** — `GET http://127.0.0.1:18794/health` (or `hermes harness status`).
2. **Start if down** — from repo root:
   ```powershell
   py -3 -m hermes_cli.main harness start
   ```
   Or ensure `harness.auto_start` / env is enabled per `config.yaml` (`harness:` section).
3. **Expose tools** — toolset `harness` is **opt-in** (not in default core). Enable via
   `hermes tools` / `config.yaml` `tools.<platform>.enabled` including `harness`.
4. **Call tools** — use `harness_status` first; then scavenge/wisdom/evolve/speak/osc as needed.
5. **Optional skill sync** — `hermes openclaw-vendor install --extension hypura-harness`
   links vendor skills into `~/.hermes/skills/` (does not move vendor files).

## Config and secrets

- Behaviour: `harness:` in `~/.hermes/config.yaml` (host, port, `enabled`, `auto_start`).
- Secrets stay in `~/.hermes/.env` only. Do not invent new non-secret `HERMES_*` env vars.
- Override script/python only via documented `HYPURA_HARNESS_*` when the operator already uses them.

## Safety

- VRChat move/speak via harness or `vrchat-autonomy` requires explicit user ACK before
  `dry_run=false` — see [`../extensions/AGENTS.md`](../extensions/AGENTS.md).
- Do not expose the harness HTTP port beyond localhost / operator Tailscale without review.
- Root scratch (`output/`, `tmp/`) must not be wired into merge overlay paths.

## Stack restart

For gateway / desktop / watchdog restarts (not harness-only), use
[`../operations/AGENTS.md`](../operations/AGENTS.md) and
`scripts/windows/restart-hermes-stack.ps1` (llama only with `-StartLlama`).

## What not to do

- Do not move `vendor/.../hypura-harness/` or `tools/harness_tools.py` “for cleanliness”.
- Do not edit `scripts/merge_tools/` while following this guide.
- Do not treat empty `/` responses on `:18794` as down if `/health` returns 200.
