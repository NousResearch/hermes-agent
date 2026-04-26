---
name: wrap-up
description: Use when someone asks to safely close Hermes CLI sessions, wrap up running Hermes terminal sessions, save current Hermes conversations before shutdown, or drain and close Hermes CLI work.
disable-model-invocation: true
---

# Wrap Up Hermes CLI Sessions

Safely drains, saves, and closes live Hermes CLI sessions only.

## Scope

This skill must affect only Hermes CLI sessions. Do not stop or restart:

- Hermes gateway
- Telegram/Discord/Slack/other platform sessions
- cron jobs
- unrelated terminals or shells

## Workflow

1. State that the skill will wrap up Hermes CLI sessions only.
2. Run the helper from the Hermes agent repo:

   ```bash
   PYTHONPATH=/home/anybody/.hermes/hermes-agent /home/anybody/.hermes/hermes-agent/venv/bin/python3 -m hermes_cli.wrapup wrap-up --timeout 600
   ```

3. Report the helper summary exactly enough for the user to know:
   - manifest path
   - which sessions were closed
   - which sessions were skipped
   - whether any session fell back to closing only Hermes rather than the actual terminal window/tab

## Safety rules

- Sessions without a confident `session_id` are skipped.
- Active sessions are waited on for up to 10 minutes.
- After timeout, the helper exports the current transcript and closes the Hermes CLI process/session.
- Closing the actual terminal window/tab is best effort. If it is too brittle or unsafe, closing only Hermes is acceptable and must be reported.
- Never use broad `kill all` terminal commands.

## Output format

Keep the response concise:

```text
Wrap-up complete.
Manifest: <path>
Closed: <n>
Skipped: <n>
Manual notes: <if any>
```
