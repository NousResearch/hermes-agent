# Kanban low-noise cron/watchdog operating model

This is the operator contract for Gibs' local Hermes Kanban notifications and watchdogs. The goal is event-driven help without babysitting noise.

## Ownership map

| Role | Owner/profile/job | Noise contract | Notes |
|---|---|---|---|
| Completion bot | Gateway built-in Kanban notifier and default-profile fallback cron `da2b6b08284a` (`Legacy completion-only Kanban Telegram fallback`, script `kanban_telegram_terminal_watch.py`) | Completion-only human-facing messages. No started/ready/blocked/error/decomposition spam. | The fallback script is no-agent and should stay silent on no-op. If the gateway notifier is healthy, avoid adding another broad completion notifier unless duplicate delivery is explicitly wanted. |
| Blocked-count reminders | Vicky cron `f7b7ce77ad03` (`Kanban monitor watchdog`, script `kanban_monitor.py`) | Sparse count-level reminders only when blocked count changes or the low-frequency reminder interval is due. Empty stdout on no-op. | It reports aggregate counts by board. It does not report individual task events or completions. |
| Auto-recovery watchdog | Vicky cron `8644064e8f60` (`Kanban auto-unblock watchdog`, script `kanban_auto_unblock.py`) | Mechanical recovery only, silent on no-op and successful recovery. Emit stdout only for actionable operator errors. | It may reclaim dead/stale/expired running claims and run a bounded dispatch pass. It must not unblock human decisions, approve review-required cards, install software, delete files, edit credentials/configs, or make product/art decisions. |
| Triage proposals | Default cron `18e75c8d7e9e` (`kanban-triage-proposal-watch`, script `kanban_triage_watch.py`, skill `kanban-orchestrator`, deliver `origin`) | Origin-only proposal context for an LLM job. No board mutations from the cron job itself. | The script may emit `NO_NEW_TRIAGE` for no-op runs so the LLM job can stay silent. New cards are proposals only; a human/orchestrator should perform actual board changes. |

## Guardrails

- Do not combine roles. Completion delivery, blocked reminders, mechanical recovery, and triage proposals are separate lanes.
- No-agent watchdog scripts must print nothing on healthy no-op runs. In no-agent cron mode, empty stdout means no user notification.
- Successful auto-recovery should normally stay quiet; only failures needing operator action should print.
- Blocked task reminders should be sparse and aggregate. Do not reintroduce per-event blocked/start/ready spam.
- Triage proposal jobs may use the LLM, but they must be origin-delivered and proposal-only. They should not mutate the board directly.
- Cron job names and script headers must describe the actual role. Avoid names like `blocked/done notifier` for a job that sends more than completions.

## Active-job verification

Run these after changing any Kanban notification/watchdog cron job:

```bash
hermes -p default cron list --all
hermes -p vicky cron list --all
```

Expected jobs:

- default `da2b6b08284a`: `Legacy completion-only Kanban Telegram fallback`, `every 1m`, `no-agent`, script `kanban_telegram_terminal_watch.py`. It may remain paused when the gateway completion notifier is the active completion lane; if resumed, it must stay completion-only.
- default `18e75c8d7e9e`: active `kanban-triage-proposal-watch`, `every 15m`, origin delivery, script `kanban_triage_watch.py`, LLM proposal job.
- vicky `f7b7ce77ad03`: active `Kanban monitor watchdog`, `every 5m`, origin delivery, no-agent, script `kanban_monitor.py`.
- vicky `8644064e8f60`: active `Kanban auto-unblock watchdog`, `every 5m`, origin delivery, no-agent, script `kanban_auto_unblock.py`.

Paused jobs such as old status digests and fallback notifiers can remain paused, but should not be resumed as recurring noise unless Gibs explicitly asks for that pattern again.

## Script no-op/dry-run checks

Each active script has a check mode intended for operator verification. It exits 0 and prints empty stdout when the script can perform its read-only checks successfully.

```bash
python /home/john/.hermes/scripts/kanban_telegram_terminal_watch.py --check-noop
python /home/john/.hermes/scripts/kanban_triage_watch.py --check-noop
python /home/john/.hermes/profiles/vicky/scripts/kanban_monitor.py --check-noop
python /home/john/.hermes/profiles/vicky/scripts/kanban_auto_unblock.py --check-noop
```

For the no-agent watchdogs, treat any stdout from these check commands as actionable: fix the script/board access problem before relying on the cron job.

## Out-of-scope ownership notes

- Product or human-review decisions are out of scope for `kanban_auto_unblock.py`; those stay blocked until a human or assigned reviewer unblocks them.
- Triage proposal content is out of scope for no-agent watchdog scripts; keep proposal/routing reasoning in the default-profile triage job.
- Gateway completion-notifier behavior lives in the Hermes gateway code. The cron fallback should not grow into a second broad board-event notifier.
