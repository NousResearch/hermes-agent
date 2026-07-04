---
name: bilibili-hermes-agent-weekly-top-feishu
description: Configure and run a scheduled Bilibili weekly top-video digest for Hermes Agent videos to Feishu/Lark. Use this skill when the user asks to crawl Bilibili/B站/哔哩哔哩 for this week's top Hermes Agent videos, schedule a daily 23:40 report, send Hermes Agent video links to Feishu, or test/repair this cron job.
---

# Bilibili Hermes Agent Weekly Top Feishu

Create or maintain a Hermes cron job that reports this week's highest-play Bilibili videos about Hermes Agent.

## Install Or Update The Job

Run from the repository root:

```bash
source venv/bin/activate
python skills/research/bilibili-hermes-agent-weekly-top-feishu/scripts/setup_daily_bilibili_hermes_agent_top.py
```

Defaults:

- Schedule: `40 23 * * *` (daily at 23:40 in the Hermes configured timezone)
- Delivery target: the Feishu DM target passed to the setup script, or `feishu` if none is provided
- Job name: `Daily Bilibili Hermes Agent Weekly Top Videos to Feishu`
- Runtime crawler: copied to `HERMES_HOME/scripts/bilibili_hermes_agent_weekly_top.py`

Use the known Feishu DM target explicitly when available:

```bash
python skills/research/bilibili-hermes-agent-weekly-top-feishu/scripts/setup_daily_bilibili_hermes_agent_top.py --deliver 'feishu:oc_xxx'
```

Use `--trigger-now` only when the user explicitly wants an immediate test delivery.

## Runtime Behavior

The crawler uses Bilibili's public web search API and searches the default keywords:

- `HermesAgent`
- `Hermes Agent`

For each keyword it queries both play-sort (`order=click`) and publish-time-sort (`order=pubdate`) result pages, then filters locally to the current Asia/Shanghai week, deduplicates by `bvid`/`aid`, and ranks by play count.

The script returns up to 10 videos with:

- Bilibili link
- title
- UP author
- play count
- likes, favorites, comments, danmaku
- publish time
- category, duration, and tags

The cron prompt formats the JSON into a concise Chinese Feishu message. Do not call `send_message` inside the cron prompt; Hermes cron automatically delivers the final response.

## Useful Commands

Run the crawler locally:

```bash
source venv/bin/activate
python skills/research/bilibili-hermes-agent-weekly-top-feishu/scripts/fetch_bilibili_hermes_agent_weekly_top.py --limit 10
```

Create the job but deliver locally for debugging:

```bash
python skills/research/bilibili-hermes-agent-weekly-top-feishu/scripts/setup_daily_bilibili_hermes_agent_top.py --deliver local
```

List the installed job:

```bash
source venv/bin/activate
python - <<'PY'
from cron.jobs import list_jobs
for job in list_jobs(include_disabled=True):
    if job.get("name") == "Daily Bilibili Hermes Agent Weekly Top Videos to Feishu":
        print(job)
PY
```

## Troubleshooting

- If `deliver=feishu` cannot resolve, set `FEISHU_HOME_CHANNEL` in `HERMES_HOME/.env` or recreate the job with `--deliver 'feishu:<chat_id>'`.
- If the gateway is not running, cron jobs will not tick. Start it with `hermes gateway start`.
- If Bilibili returns HTTP 412, reduce `--pages`, set a valid `BILIBILI_COOKIE` or `BILI_COOKIE`, or retry later. The crawler records source errors in JSON instead of inventing videos.
- If fewer than 10 matching videos exist for the current week, report the actual count instead of inventing links.
