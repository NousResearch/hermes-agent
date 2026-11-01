---
name: auto-fetch
description: Auto-fetch emails (Himalaya CLI), GitHub notifications (gh CLI), and other routine data pulls every 20 minutes via Hermes cron. Summarize and inject into memory for proactive context. Use this whenever the user wants automated background data syncing, daily briefings, email digests, or Github notification monitoring without manual prompts. Also use when the user says "auto-fetch", "automate email checking", "background sync", or "cron fetch".
---

# Auto-Fetch Pipeline

Periodic background pipeline: fetch → compress → summarize → memory injection. Based on openhuman's 20-min auto-fetch architecture.

## Overview

```
Cron (every 20min) → fetch_script.py → TokenJuice compress → LLM summarize → Memory save
```

Sources:
- **Email**: Himalaya CLI (`himalaya search`) → new emails since last check
- **GitHub**: `gh api notifications` → unread notifications
- **Calendar**: Google Calendar via `gws` CLI (future)
- **Custom**: any script in `~/.hermes/scripts/auto-fetch/`

## Setup (one-time)

### 1. Create the fetch script

```bash
mkdir -p ~/.hermes/scripts/auto-fetch
```

Place or reference any fetch scripts here. The main aggregator:

```bash
# ~/.hermes/scripts/auto-fetch/fetch.sh
export PATH="$HOME/.local/bin:$PATH"

# Email - Himalaya
echo "=== EMAIL ==="
himalaya search "newer_than:1d" 2>/dev/null | head -30 || echo "no email"

# GitHub
echo "=== GITHUB ==="
gh api notifications --jq '.[] | "\(.subject.title) — \(.repository.full_name)"' 2>/dev/null | head -10 || echo "no gh notifications"
```

### 2. Create the cron job

```python
# Hermes will run this via cron tool:
# cronjob action='create'
#   name='auto-fetch-pipeline'
#   schedule='every 20m'
#   script='auto-fetch/fetch.sh'
#   skills=['auto-fetch','token-juice']
#   prompt='Process the auto-fetch output:
# 1. Summarize each section (email, github) in 2-3 bullets
# 2. Flag anything urgent (deadlines, mentions, PR reviews needed)
# 3. Skip if empty/unchanged from last run
# 4. Save key facts to memory'
```

### 3. Optional: GitHub auth check

```bash
gh auth status 2>/dev/null || gh auth login
```

## Flow per tick

1. Script runs → raw output from all sources
2. TokenJuice compresses output (dedup, trim)
3. LLM reads compressed output, produces summary
4. Summary saved to memory for proactive context
5. Next session: "gm!" → agent already knows new emails + PRs

## Anti-Patterns
- Don't run on short intervals (<10min) — API rate limits
- Don't save raw data to memory — save summaries only
- Don't alert on every tick — only flag urgent items
- Himalayas folder: use `[Gmail]/Inbox` alias if configured
