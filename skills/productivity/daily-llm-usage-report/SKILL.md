---
name: daily-llm-usage-report
description: "Gordon's daily LLM usage report — request metrics sent to Telegram each morning at 6 AM PT. Uses a daemon + cron, no OpenRouter dependency."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [telegram, usage, metrics, cron, daily-routine]
    homepage: https://github.com/NousResearch/hermes-agent
---

# daily-llm-usage-report

Gordon's daily LLM usage report — sends request metrics to Telegram at 6 AM PT every morning.

## Metrics Reported

- User Interactions (24h)
- API Requests (24h)
- Max Req / 5H Window (24h)
- Max Req / 5H Window (30d)

## Components

### `/opt/hermes/scripts/cost_report.py`
Core report script. Reads request log, computes metrics, sends Telegram message.
No OpenRouter dependency — Gordon is on subscription.

**Usage:**
```bash
# Test/send report
python3 /opt/hermes/scripts/cost_report.py

# Take baseline (no send)
python3 /opt/hermes/scripts/cost_report.py --baseline
```

### `/opt/hermes/scripts/cost_daemon_wrapper.py`
Daemon wrapper that reads Telegram tokens from `/opt/data/.env.tokens` and execs the daemon with those tokens injected into the environment.

**Always start the daemon via this wrapper**, not directly. The daemon reads tokens from its process env, and cron/the subprocess don't have the container's env vars.

### `/opt/hermes/scripts/cost_report_daemon.py`
Sleeps until next 6 AM PT, runs `cost_report.py`, repeats. Daemon is long-lived.

### `/opt/data/request-log.jsonl`
JSONL log of API calls. Each line: `{"ts": unix_timestamp, "api_calls": int, "platform": "telegram", ...}`

### `/opt/data/cost-state.json`
State file (keeps prior metrics for comparison).

## Token Setup

Tokens live in `/opt/data/.env.tokens`:
```
TELEGRAM_BOT_TOKEN=<bot_token>
TELEGRAM_HOME_CHANNEL=<chat_id>
```

## Starting / Restarting the Daemon

Always use the wrapper to get tokens:
```bash
python3 /opt/hermes/scripts/cost_daemon_wrapper.py &
```

To verify it's running:
```bash
ps aux | grep cost_report_daemon
tail /opt/data/cost-report-daemon.log
```

## Pitfalls
The daemon must be started via `cost_daemon_wrapper.py` (not directly), which reads tokens from `/opt/data/.env.tokens`. Starting it directly causes silent failures — the daemon sleeps and never sends.

### Old daemon still running with stale tokens
If you update `/opt/data/.env.tokens`, kill and restart the daemon:
```bash
pkill -f cost_report_daemon
python3 /opt/hermes/scripts/cost_daemon_wrapper.py &
```

### Cron schedule is UTC
Hermes cron uses UTC. To fire at 6 AM PT, schedule as `5 13 * * *` (13:05 UTC). Always verify UTC equivalent.

### Request log may be empty on first days
Metrics will be 0 until the log has at least 24h of data.
