# Hermes Cronjob Tool Reference

Quick reference for the `cronjob` tool in Hermes Agent sessions.

## Basic Usage

```
cronjob action='create' name='<name>' schedule='<cron>' prompt='<self-contained prompt>'
```

## Actions

| Action | Description |
|--------|-------------|
| `create` | Schedule a new job (requires `schedule` + `prompt`) |
| `list` | Show all jobs |
| `update` | Modify an existing job (pass `job_id`) |
| `remove` | Delete a job (pass `job_id`) |
| `run` | Trigger immediately (pass `job_id`) |
| `pause` | Pause a job |
| `resume` | Resume a paused job |

## Key Parameters

- **schedule**: `'30m'`, `'every 2h'`, `'0 9 * * *'`, or ISO timestamp for one-shot
- **deliver**: `'local'` (save only), `'origin'` (same chat), `'all'` (fan-out), or `platform:chat_id:thread_id`
- **no_agent**: `true` to run a script directly (no LLM), requires `script`
- **script**: Path to a script file under `~/.hermes/scripts/` (`.sh`/`.bash` via bash, others via Python)
- **attach_to_session**: `true` to make the job continuable (user can reply)
- **context_from**: Array of job IDs whose output is injected as context
- **workdir**: Absolute path for the job's working directory

## Prompt Guidelines

- Must be **self-contained** — no prior conversation context available
- Include all necessary file paths, tool names, and acceptance criteria
- Specify output format expected
- Use `deliver='local'` for data-collection jobs, explicit targets for notifications

## Example: Knowledge Morning Report

```
cronjob action='create'
  name='knowledge-morning-report'
  schedule='0 6 * * *'
  deliver='local'
  prompt='Execute the knowledge morning report workflow:
  1. Check ${KNOWLEDGE_BASE}/logs/ for overnight activity
  2. Summarize new Discord threads, YouTube insights, and Git commits
  3. Output a concise daily briefing'
```
