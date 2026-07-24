---
name: system-health
description: VPS system health monitoring — collect memory, CPU, disk stats and optionally email reports.
version: 1.0.0
author: Vijay Selvaraj (vijays365), Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [devops, monitoring, health, system, vps, cron]
    category: devops
    requires_toolsets: [terminal, cronjob]
    related_skills: [hermes-diag, config-validator]
---

# System Health Monitoring

Monitor VPS resource usage (memory, CPU, disk) and optionally email reports via the AgentMail API.

## When to Use

- User asks about server health or resource usage
- User wants automated health reports emailed periodically
- Setting up monitoring for a new VPS or server
- Troubleshooting performance issues (checking memory/CPU/disk)

## Quick check (ad-hoc)

```bash
# All in one
free -h
cat /proc/loadavg
df -h /
uptime
ps aux --sort=-%mem | head -6
```

## Automated reporting script

A standalone Python script at `$HERMES_HOME/skills/devops/system-health/scripts/system-health-report.py` collects all stats and sends an email report.

### Metrics and thresholds

| Metric | Source | Default threshold |
|--------|--------|-------------------|
| Memory usage | `free -m` | Warn if >85% used |
| Swap usage | `free -m` | Warn if >20% used |
| CPU load (1/5/15min) | `/proc/loadavg` | Warn if 1min > nproc×0.8 |
| Disk usage | `df -h` for `/` | Warn if >80% |
| Uptime | `uptime` | Info only |
| Top 5 memory consumers | `ps` | Info only |

### Configuration

Edit the top of the script to set:

- `TO_EMAIL` — recipient email address
- `FROM_EMAIL` — your AgentMail sender inbox
- `API_KEY_FILE` — path to your AgentMail API key
- `THRESHOLDS` — warning level overrides

### Cron usage

```bash
cronjob(action='create', script='$HERMES_HOME/skills/devops/system-health/scripts/system-health-report.py', no_agent=True, schedule='0 8 * * 2,5')
```

The script is self-contained — no LLM tokens needed on each run.

## Output format

When all metrics are within thresholds, the subject starts with ✅. If any threshold is exceeded, it starts with ⚠️.

Example output (body):

```
System Health Report — myserver
================================
Uptime: up 14 days, 3 hours

Memory:  3200MB / 7746MB (41.3%)
Swap:    0MB / 0MB (0%)
CPU Load: 0.45 (1m) / 0.52 (5m) / 0.48 (15m) — 4 cores
Disk /:  30G / 75G used (40%) — 45G avail

All metrics within thresholds.

Top 5 memory consumers:
USER       PID %CPU %MEM ...
...
```

## Common Pitfalls

1. **API key not set.** The script reads the API key from `API_KEY_FILE`. Store it outside version control.
2. **Recipient email not configured.** Update `TO_EMAIL` at the top of the script before first use.
3. **AgentMail not available.** Modify the `send_email()` function to use any other email API (SMTP, SendGrid, etc.).

## Verification Checklist

- [ ] Run `python3 $HERMES_HOME/skills/devops/system-health/scripts/system-health-report.py` and confirm stats are printed
- [ ] Check an email arrives at the configured recipient
- [ ] Verify the subject line includes ✅ (normal) or ⚠️ (warning)
