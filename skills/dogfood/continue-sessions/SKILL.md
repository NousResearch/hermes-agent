---
name: continue-sessions
description: Use when someone asks to resume Hermes CLI sessions closed by wrap-up, continue wrapped Hermes terminal sessions, or reopen the previous Hermes CLI work set.
disable-model-invocation: true
---

# Continue Wrapped Hermes CLI Sessions

Resume the Hermes CLI sessions recorded by the latest `/wrap-up` manifest.

## Workflow

1. Run the helper from the Hermes agent repo:

   ```bash
   PYTHONPATH=/home/anybody/.hermes/hermes-agent /home/anybody/.hermes/hermes-agent/venv/bin/python3 -m hermes_cli.wrapup continue-sessions --max-auto-open 5
   ```

2. Report:
   - manifest path used
   - sessions opened automatically
   - any manual resume commands printed by the helper

## Behavior

- Reads `~/.hermes/wrap-up/latest.json`.
- Opens new Windows Terminal/WSL windows or tabs automatically when available.
- Opens at most 5 sessions automatically per invocation.
- If the manifest contains more than 5 resumable sessions, the remaining sessions are printed as manual `hermes --resume <session_id>` commands.
- If automatic opening fails, print the commands instead of silently failing.

## Output format

Keep the response concise:

```text
Continue-sessions complete.
Manifest: <path>
Opened automatically: <n>
Manual commands: <n>
```
