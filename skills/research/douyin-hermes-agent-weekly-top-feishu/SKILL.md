---
name: douyin-hermes-agent-weekly-top-feishu
description: Configure and run a scheduled Douyin weekly top-video digest for Hermes Agent videos to Feishu/Lark. Use this skill when the user asks to crawl 抖音/Douyin for this week's top Hermes Agent videos, schedule a daily 23:10 report, send Hermes Agent Douyin video links to Feishu, or test/repair this cron job.
---

# Douyin Hermes Agent Weekly Top Feishu

Create or maintain a Hermes cron job that reports this week's highest-play Douyin videos about Hermes Agent.

## Install Or Update The Job

Run from the repository root:

```bash
source venv/bin/activate
python skills/research/douyin-hermes-agent-weekly-top-feishu/scripts/setup_daily_douyin_hermes_agent_top.py
```

Defaults:

- Schedule: `10 23 * * *` (daily at 23:10 in the Hermes configured timezone)
- Delivery target: the Feishu DM target passed to the setup script, the single Feishu channel in `HERMES_HOME/channel_directory.json`, or `feishu` if none is resolved
- Job name: `Daily Douyin Hermes Agent Weekly Top Videos to Feishu`
- Runtime crawler: copied to `HERMES_HOME/scripts/douyin_hermes_agent_weekly_top.py`

Use the known Feishu DM target explicitly when available:

```bash
python skills/research/douyin-hermes-agent-weekly-top-feishu/scripts/setup_daily_douyin_hermes_agent_top.py --deliver 'feishu:oc_xxx'
```

Use `--trigger-now` only when the user explicitly wants an immediate test delivery.

## Runtime Behavior

The crawler uses Douyin's web video-search endpoint and searches the default keywords:

- `HermesAgent`
- `Hermes Agent`
- `hermes-agent`

For each keyword it queries several Douyin `sort_type` values with the web one-week `publish_time` filter, then filters locally to the current Asia/Shanghai week, deduplicates by `aweme_id`/URL, and ranks by `statistics.play_count`. Once enough matching videos are collected for the requested Top N, the crawler stops additional keyword/sort/page requests to reduce Douyin verification triggers.

The crawler sends the full PC Web browser parameter set (`pc_client_type`, `version_code`, browser/engine/OS/screen fields) and dynamic tracking parameters. It reads `msToken` from cookies or `DOUYIN_MSTOKEN` / `DOUYIN_MS_TOKEN`, falls back to a syntactically valid generated token, and generates `webid` through Douyin's webid endpoint unless `DOUYIN_WEBID` is set. These parameters are required; the minimal search query often returns `search_nil_type=verify_check`. When Douyin still returns a verification or anti-spam empty page, the crawler refreshes `webid`/`msToken` and retries before recording the page as failed.

Douyin search currently requires a logged-in browser cookie. The crawler resolves login state in this order:

1. Explicit cookie variables in `HERMES_HOME/.env`: `DOUYIN_COOKIE`, `DOUYIN_SEARCH_COOKIE`, or `DOUYIN_WEB_COOKIE`
2. The default Firefox profile under `~/.mozilla/firefox`, Snap Firefox, or Flatpak Firefox
3. A profile or cookie database path specified with `DOUYIN_FIREFOX_PROFILE`, `FIREFOX_PROFILE`, or `DOUYIN_FIREFOX_COOKIES_SQLITE`

The script loads `HERMES_HOME/.env` itself because Hermes cron executes pre-run scripts before the scheduler reloads `.env` for the agent turn.

For most local installs, log in to Douyin in Firefox and rerun the crawler. If Firefox uses a non-default profile, set:

```bash
DOUYIN_FIREFOX_PROFILE='/path/to/firefox/profile'
```

You can still force an explicit cookie:

```bash
DOUYIN_COOKIE='passport_csrf_token=...; sessionid=...; ...'
```

The script returns up to 10 videos with:

- Douyin link
- title/description
- author
- play count
- likes, comments, shares, favorites
- publish time
- hashtags

The cron prompt formats the JSON into a concise Chinese Feishu message. Do not call `send_message` inside the cron prompt; Hermes cron automatically delivers the final response.

## Useful Commands

Run the crawler locally:

```bash
source venv/bin/activate
python skills/research/douyin-hermes-agent-weekly-top-feishu/scripts/fetch_douyin_hermes_agent_weekly_top.py --limit 10
```

Create the job but deliver locally for debugging:

```bash
python skills/research/douyin-hermes-agent-weekly-top-feishu/scripts/setup_daily_douyin_hermes_agent_top.py --deliver local
```

List the installed job:

```bash
source venv/bin/activate
python - <<'PY'
from cron.jobs import list_jobs
for job in list_jobs(include_disabled=True):
    if job.get("name") == "Daily Douyin Hermes Agent Weekly Top Videos to Feishu":
        print(job)
PY
```

## Troubleshooting

- If `deliver=feishu` cannot resolve, set `FEISHU_HOME_CHANNEL` in `HERMES_HOME/.env` or recreate the job with `--deliver 'feishu:<chat_id>'`.
- If the gateway is not running, cron jobs will not tick. Start it with `hermes gateway start`.
- If Douyin returns `请先登录，再继续搜索吧`, log in to Douyin in Firefox, or set `DOUYIN_FIREFOX_PROFILE` / `DOUYIN_FIREFOX_COOKIES_SQLITE` if the crawler cannot find the correct profile.
- If Douyin returns `search_nil_type=verify_check`, first verify the runtime crawler is the updated version that includes PC browser parameters, `webid`/`msToken`, and retry support. The retry count defaults to 3 and can be adjusted with `DOUYIN_VERIFY_RETRIES`. If it still happens, refresh Firefox login/verification state or set `DOUYIN_WEBID` and `DOUYIN_MSTOKEN` / `DOUYIN_MS_TOKEN` explicitly.
- If Douyin does not expose `play_count` for a result, the crawler keeps the video but ranks its play count as 0 and records a note.
- If fewer than 10 matching videos exist for the current week, report the actual count instead of inventing links.
