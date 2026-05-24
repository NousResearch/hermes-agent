# Two-Channel Alert Design — Operator Spec (2026-05-07)
# Absorbed from: in-code-operator-alerts skill

Formalized by operator during implementation session. Canonical decision record
for how alerts are routed in the biotech screener.

## Channel 1: Hermes cron `deliver: telegram`

Use for:
- Scheduled daily/weekly job summaries
- Final PASS / WARN / FAIL receipts
- Non-conditional "tell me how the run ended" messages

Best for: daily production summary, weekly agent fleet summary, audit digest, ruleset health packet.

How: set `deliver="telegram"` on the Hermes cronjob at creation time.

## Channel 2: `common/alerts.py send_operator_alert()`

Use for:
- Condition-triggered alerts discovered during execution
- Hard failures that should interrupt operator attention
- Cases where the job may fail before Hermes can produce a clean final response

Best for: missing production snapshot, stale lock / WSL uptime failure, missing required artifact,
ruleset mismatch, financial consistency FAIL, classifier pool FAIL, B0/B1 bioshort producer failure.

How: `from common.alerts import send_operator_alert` inside the ops/QA wrapper,
called after deterministic verdict is known.

## Rules

1. Use BOTH, but keep them separated.
2. Do NOT scatter raw Telegram calls through the pipeline.
3. Add ONE wrapper (`common/alerts.py`), call it from ops/QA layer only.
4. Never alert from scoring math itself.
5. Alert from ops/QA wrappers after deterministic verdicts are known.

## Implementation rules

- TOKEN from env only, never committed
- chat_id from env/config, not hardcoded in source
- dry-run mode for tests (`ALERTS_DRY_RUN=1`)
- rate limit / dedupe repeated alerts (4h window, 10/hr cap)
- never alert from scoring math itself

## Bioshort hook scope (current)

In-code alerts ONLY for:
- Daily production wrapper failed to produce snapshot
- B0 latest_status.json missing after production run
- Hedge_report producer cron fails after B1b
- Hedge_report becomes stale beyond threshold after restoration

NOT sending alerts for normal STALE during the suppressed/orphaned state —
that condition is already known and governed.

## Env vars (repo .env, gitignored)

```
TELEGRAM_BOT_TOKEN=<bot token from BotFather>
TELEGRAM_CHAT_ID=6118239110   # operator personal user ID
```
