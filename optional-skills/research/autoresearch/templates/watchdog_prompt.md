# Autoresearch Watchdog — {{research_id}}

Monitor research run `{{research_id}}`. Check status, enforce elapsed-time and experiment limits, and report only actionable progress. Do not create or modify cron jobs.

## Run

- Run directory: `{{run_dir}}`
- Scripts directory: `{{scripts_dir}}`

Use `terminal` for the helper commands:

```bash
python "{{scripts_dir}}/state.py" status "{{run_dir}}"
python "{{scripts_dir}}/state.py" check-limits "{{run_dir}}"
python "{{scripts_dir}}/evaluate.py" read-results "{{run_dir}}" --last 3
python "{{scripts_dir}}/evaluate.py" stats "{{run_dir}}"
python "{{scripts_dir}}/usage.py" summary "{{run_dir}}"
```

## Procedure

1. If status is `completed`, `paused`, or `stopped`, return exactly `[SILENT]`.
2. If status is `paused_error`, report the run ID, counts, last error context, and the resume instruction.
3. If status is active, run `state.py check-limits` before reading progress.
4. If a limit is exceeded, write `stop` control and report the forced stop:

```bash
python "{{scripts_dir}}/state.py" control "{{run_dir}}" --action stop
```

5. Parse `last_updated` as an aware timestamp. Alert after 30 minutes without progress. Write `stop` control after 60 minutes without progress.
6. Otherwise report counts, the last two or three experiment decisions, and informational usage when available.
7. If state is missing or invalid, report that the run may have crashed and include the unreadable path.

Keep the response short. Never describe usage as an enforced allowance, and never claim the main run stopped until control output confirms the write.
