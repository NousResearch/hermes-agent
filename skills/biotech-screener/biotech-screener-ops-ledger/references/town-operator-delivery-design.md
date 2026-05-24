# Town Operator Delivery — Design Notes (Spec 090, Phase A)
# Absorbed from: in-code-operator-alerts skill

## Purpose

Bridge between Hermes/biotech-screener and Town task manager.
Town does NOT expose a native inbound webhook. Email trigger is the integration path.

## Integration path (confirmed 2026-05-07)

```
Hermes / biotech-screener pipeline
  → common/operator_delivery.py:send_operator_event(channel="town", ...)
    → _send_town_email()
      → SMTP (djschulz@gmail.com via existing credentials)
        → Gmail inbox
          → Town routine (filters on subject prefix "[Hermes]")
            → Town creates task
```

## Email format

Subject: `[Hermes] {SEVERITY} | {event_type} | {original_title}`
Body:
```
=== Hermes Operator Event ===
Title: {prefixed_title}
Severity: {severity}
Event Type: {event_type}
Summary: {summary}
Artifact: {artifact}
Next Action: {next_operator_action}
Timestamp: {as_of}

--- JSON payload ---
{
  "data": {
    "source": "hermes",
    "event_type": "held_spec_ledger",
    "severity": "INFO",
    "title": "🔵 INFO | held_spec_ledger | Held-spec ledger updated",
    "summary": "...",
    "artifact": "artifacts/ops/held_spec_ledger/latest.md",
    "next_operator_action": "...",
    "as_of": "2026-05-07T17:00:00Z"
  }
}
```

## SMTP Credentials

Shared with `common/alert_email.py` (already in repo .env):
```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=djschulz@gmail.com
SMTP_PASSWORD=<app password>
TOWN_EMAIL=djschulz@gmail.com
```

## Phase Plan

Phase A (COMPLETE as of 2026-05-07):
- `common/operator_delivery.py` implemented with dry-run default
- `OPERATOR_DELIVERY_DRY_RUN=1` set in env (logs only, no live sends)
- 29 unit tests pass in `tests/test_operator_delivery.py`

Phase B (next): Set `OPERATOR_DELIVERY_DRY_RUN=0` and verify Town picks up email from hermes-held-spec-ledger cron.

Phase C: Wire hard-failure event types (first_fire_fail, snapshot_missing, ruleset_mismatch, cron_missed).

## Town Auth Restriction

Town auth does NOT allow external POST webhooks from unknown callers. Email is the
only reliable inbound trigger without additional configuration.

Future: if Town adds native webhook, add `_send_town_webhook()` alongside
`_send_town_email()`. Callers (send_operator_event) unchanged — just add channel path.

## Guardrails — What Town MAY and MAY NOT Do

Town MAY:
- Create tasks from Hermes email events
- Surface action items to the operator
- Organize by severity / event_type

Town MAY NOT:
- Infer approval from a stale task entry
- Approve spec changes or cron modifications
- Change production state based on Town task content alone

## Pitfalls

- **Subject emoji doubling:** `send_operator_event()` prefixes `title` with emoji+severity
  before passing to `_send_town_email()`. Fix: pass `subject_title=original_title` to
  the email function. The JSON payload carries the prefixed title; the email subject uses
  the original.
- **ensure_ascii:** Always `json.dumps(..., ensure_ascii=False)` so emoji survive as UTF-8
  in log output and email body.
- **Script file for smoke test:** Never run `python3 -c "... extra={'key': val} ..."` —
  f-strings with dict literals cause `ValueError: Invalid format specifier`.
  Write `tools/smoke_operator_delivery.py` and add `sys.path.insert` for repo root.
- **DRY_RUN=1 is the default Phase A state.** Don't assume live sends until Phase B is
  explicitly activated.
