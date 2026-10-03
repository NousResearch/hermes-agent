---
name: gmail-triage
description: "Triage authenticated Gmail into calendar and memory."
version: 0.1.0
author: OpenAI Codex
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [Gmail, Calendar, Memory, Automation]
    related_skills: [email-inbox-triage, google-workspace]
---

# Gmail triage

Use the bundled `scripts/gmail_triage.py` as a no-agent Hermes cron script.
It is intentionally self-contained: Gmail content is sent to the existing local
Hindsight `reflect` structured-output classifier in a dedicated empty bank. The
bank mission holds the trusted classifier rules; every run verifies the mission,
empty bank, and zero memory provenance before a closed validator permits only
Apple Calendar and Hindsight writes.

## Setup

1. Copy `scripts/gmail_triage.py` to
   `~/.hermes/scripts/gmail-triage/gmail_triage.py`.
2. Copy `config.example.json` to `~/.hermes/gmail-triage.json`, set
   `cutover_at` to the deployment timestamp, set `script_sha256` to the
   installed script's SHA-256, and set the file mode to `0600`.
3. Set top-level `timezone: America/Sao_Paulo` in `~/.hermes/config.yaml`.
4. Provision the dedicated Hindsight bank `gmail-triage-classifier` with the
   script's exact `CLASSIFIER_MISSION`, and verify it has zero memories, mental
   models, and directives. Never retain content into this bank.
5. Validate with the Hermes venv Python:
   `~/.hermes/hermes-agent/venv/bin/python ~/.hermes/scripts/gmail-triage/gmail_triage.py doctor`,
   then run `synthetic` and `dry-run` with the same interpreter.
6. Schedule it with native cron in no-agent mode.

```bash
hermes cron create "0 10,22 * * *" --name "Gmail triage" \
  --script gmail-triage/gmail_triage.py --no-agent --deliver local
```

The job never sends or replies to mail, follows links, executes content, or
uses any capability beyond calendar, memory, and its two audit labels.
`review` is an audited Gmail queue (`Jarvis/Review`) and output includes the
opaque Gmail message ID plus a reason code; uncertain messages are not acted
on silently.

## Production gate and rollback

Before enabling the job, require the focused tests and skill-authoring checks
to pass, deploy the merged commit's exact script, verify its checksum, and run
`doctor`, `synthetic`, and `dry-run` with the runtime venv. After creating the
job, confirm it is enabled, its next run is 10:00 or 22:00 in
`America/Sao_Paulo`, trigger one manual run, and inspect `hermes cron runs` and
`hermes cron incidents` plus gateway health.

Rollback by pausing the job first, restoring the prior script/config pair and
checksum, then running `doctor`. Preserve the SQLite ledger so a rollback or
redeploy cannot duplicate completed effects; remove the job only after its ID
has been recorded for audit.
