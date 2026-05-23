---
name: hermes-final-report
description: Use when the Hermes optimization campaign is fully implemented and needs a final redacted operator report, how-to guide, system anatomy, validation summary, and gated Telegram delivery.
---

# Hermes Final Report

Use this only after the Hermes optimization campaign reaches Phase 8 final
integration or a clearly documented final stopping point.

## Preconditions

- All active phases are documented in `docs/HERMES_EXECUTION_PLAN.md`.
- `docs/HERMES_BUILD_LOG.md` has current validation and judge results.
- Architecture, Reliability/Security, and Tooling/UX judges have passed, or a
  blocker is explicitly documented.
- `hermes ops status --markdown`, `hermes gateway status`, and `hermes doctor`
  have been run for final evidence.

## Final Report Anatomy

Include these sections:

1. Executive summary: what Hermes is now capable of post-upgrade.
2. Upgrade map: phases completed, files/systems changed, and why.
3. System anatomy: CLI, gateway, agent runtime, tools, memory, security,
   reliability, ops, skills, docs, tests, and runtime boundaries.
4. How-to guide: start, inspect, test, debug, capture receipts, recover safely,
   and continue future upgrade slices.
5. Validation evidence: focused tests, smoke checks, secret scans, rollback
   checks, gateway status, doctor results, and judge results.
6. Known limitations: pre-existing issues, optional-provider warnings, deferred
   work, and unsafe/blocked capabilities.
7. Operator commands: exact commands for status, doctor, tests, logs, incident
   bundles, and skills.
8. Next actions: remaining roadmap items, if any.

## Required Commands

```bash
./venv/bin/python -m hermes_cli.main ops status --markdown > /tmp/hermes-final-ops-status.md
./venv/bin/python -m hermes_cli.main gateway status > /tmp/hermes-final-gateway-status.txt
./venv/bin/python -m hermes_cli.main doctor > /tmp/hermes-final-doctor.txt
./venv/bin/python -m hermes_cli.main send --to telegram --file docs/HERMES_FINAL_REPORT.md --dry-run --json --output /tmp/hermes-final-telegram-preflight.json
```

For tests, use the smallest representative set that covers the final changed
surfaces, then document any intentionally skipped broad suites.

## Redaction Rules

- Do not include raw `.env`, auth files, Keychain values, launchd environment,
  private memory, raw logs, cron prompt bodies, provider facts, credentials, or
  private user data.
- Use redacted ops-status receipts and summarized command outcomes.
- Secret-scan the final report draft before any delivery.

## Telegram Delivery Gate

Telegram delivery is an external action. Send only after:

- Phase 8 final integration passes or a final blocker is documented.
- The final report draft passes a targeted secret scan.
- A `hermes send --dry-run` Telegram preflight passes and writes a redacted
  receipt before any live delivery.
- The user has requested final Telegram delivery for the completed campaign.
- Delivery uses the existing Hermes/Operator Telegram path without printing or copying bot tokens.

If the report is long, split it into safe chunks around 3000-3600 characters.
Record delivery result and message IDs in `docs/HERMES_BUILD_LOG.md` without
recording tokens, chat-private content beyond the report itself, or raw API
responses.
