# Ops — OpenClaw cron jobs (audit + safety)

## Goal
Maintain a clear inventory of scheduled jobs (cron/timers) and ensure:
- no WalletDB job reads NGMI-terminal databases implicitly
- Telegram delivery uses the correct sender bot (`@Sathyrnbot`)

## Where schedules live
Depending on how you run OpenClaw, schedules may be implemented as:
- OpenClaw Gateway cron jobs (managed by the daemon)
- `systemd --user` timers
- host `crontab`

## Audit checklist
### 1) Find WalletDB schedules
- `systemctl --user list-timers | rg walletdb`
- `crontab -l | rg walletdb`

### 2) Verify DB isolation
No always-on job should reference NGMI-terminal DB paths.

Red flags:
- `/home/yeqiuqiu/projects/ngmi-terminal/data/ngmi_terminal.db`
- `WALLETDB_NGMI_DB_PATH`
- `NGMI_DB_PATH`

Quick search (WalletDB repo):
```bash
cd /home/yeqiuqiu/projects/walletdb
rg -n -- "ngmi_terminal\.db|WALLETDB_NGMI_DB_PATH|NGMI_DB_PATH|WALLETDB_NGMI_DB" src scripts docs
```

### 3) Verify Telegram sender
- Sender bot should be `@Sathyrnbot`
- Ensure `TELEGRAM_BOT_TOKEN` corresponds to `@Sathyrnbot` in the env files used by the service.

## Recommended pattern
- WalletDB scheduled runners should execute **inside** `/home/yeqiuqiu/projects/walletdb` using `scripts/run_walletdb.sh`.
- NGMI-terminal scheduled runners should execute **inside** `/home/yeqiuqiu/projects/ngmi-terminal` using its `.venv` and CLI.

Avoid “wrapper repos” that vendor/copy code from the other system.

## No-nudge continuity reminder hardening (internal-only)
Reminder rails intended to be silent-on-success must not run as `sessionTarget=main` + `payload.kind=systemEvent`.

Hard requirements for enabled `continuity:*` reminder jobs:
- `sessionTarget = isolated`
- `payload.kind = agentTurn`
- `delivery.mode = none`
- deterministic contract command path present in message
- message enforces `reply exactly: NO_REPLY`
- message must **not** contain model-side `BLOCKER` forwarding instructions

Operational commands:
```bash
# Legacy continuity-only policy guard
bash /home/yeqiuqiu/clawd-architect/ops/openclaw/no_nudge_continuity_cron_guard.sh --strict --json

# XE-304 authority guard (cross watchdog/canary/checkpoint/scheduler-governance set)
bash /home/yeqiuqiu/clawd-architect/ops/openclaw/no_llm_watchdog_cron_authority_guard.sh --strict --json

# Continuity-only hardener wrapper (delegates to XE-304 hardener)
bash /home/yeqiuqiu/clawd-architect/ops/openclaw/harden_no_nudge_continuity_reminders.sh
```

## Routing boundary for recurring control jobs
Recurring watchdog/canary/checkpoint/scheduler-governance jobs are deterministic-authority surfaces.

Hard requirements:
- authoritative route class = `NO_LLM`
- enabled authority jobs must run deterministic contract scripts first
- `cron_protocol_outcome.sh` handles blocker side-effect routing through `event_router.sh`; model wrappers do **not** decide/forward blocker verdicts
- authority job payload messages must always end with `reply exactly: NO_REPLY`
- optional operator summaries/digests must be separate downstream jobs and cannot share authority channels

Practical implication:
- `continuity:*`, watchdog, canary, checkpoint-health, and scheduler-governance rails are deterministic producers first
- any model involvement is secondary narration/execution wrapper only, never the governing runtime verdict

Operational commands:
```bash
# Enforce no-llm authority contract for recurring control rails
bash /home/yeqiuqiu/clawd-architect/ops/openclaw/harden_no_llm_watchdog_cron_authority.sh

# Control-plane subset wrapper (delegates to XE-304 hardener)
bash /home/yeqiuqiu/clawd-architect/ops/openclaw/harden_control_plane_watchdog_contracts.sh
```

Guard blocker classifications (first protocol line + JSON `classification` field):
- `no_llm_watchdog_cron_authority_gateway_connectivity_failure` / `gateway_connectivity_failure`:
  `openclaw cron list --json` could not reach gateway/daemon (transient infra/connectivity path).
- `no_llm_watchdog_cron_authority_contract_drift` / `cron_contract_drift`:
  cron payload contract could not be evaluated (invalid/shifted cron-list contract).
- `no_llm_watchdog_cron_authority_policy_failed` / `cron_policy_failed`:
  cron payload parsed but one or more authority rails violate deterministic no-llm contract.

## Context watch output contract
`ops/openclaw/context_runtime_local_watch.sh` now defaults to blocker-only chat protocol behavior:
- BLOCKER conditions still emit `BLOCKER:` lines.
- Non-blocker status is emitted as `INTERNAL_STATUS:` (not `READY:`/`PROGRESS:`), so cron forwarders that key off protocol prefixes stay silent.
- Legacy `READY:`/`PROGRESS:` lines can be re-enabled by setting:
  - `OPENCLAW_CONTEXT_WATCH_EMIT_PROTOCOL_LINES=1`
