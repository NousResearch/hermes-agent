---
name: a-share-optical-module-pcb-limitups-feishu
description: Configure and run a scheduled A-share optical-module and PCB concept limit-up digest for Feishu/Lark. Use this skill when the user asks to crawl A股光模块、光通信模块、CPO、PCB概念涨停股, schedule a daily 23:30 stock report, send 光模块PCB概念涨停股 to Feishu, or test/repair this cron job.
---

# A-share Optical Module PCB Limit-ups Feishu

Create or maintain a Hermes cron job that reports current-trading-day A-share limit-up stocks related to optical module and PCB concept boards.

## Install Or Update The Job

Run from the repository root:

```bash
source venv/bin/activate
python skills/research/a-share-optical-module-pcb-limitups-feishu/scripts/setup_daily_optical_module_pcb_limitups.py
```

Defaults:

- Schedule: `30 23 * * *` (daily at 23:30 in the Hermes configured timezone)
- Delivery target: the Feishu DM target passed to the setup script, or `feishu` if none is provided
- Job name: `Daily A-share Optical Module PCB Limit-ups to Feishu`
- Runtime crawler: copied to `HERMES_HOME/scripts/a_share_optical_module_pcb_limitups.py`

Use the known Feishu DM target explicitly when available:

```bash
python skills/research/a-share-optical-module-pcb-limitups-feishu/scripts/setup_daily_optical_module_pcb_limitups.py --deliver 'feishu:oc_xxx'
```

Use `--trigger-now` only when the user explicitly wants an immediate test delivery.

## Runtime Behavior

The crawler uses Eastmoney public endpoints:

- Search API to resolve concept board codes
- H5 `ZJLX/getZDYLBData` with `fs=b:<board_code>` to fetch board constituents, with Quote `clist/get` as fallback
- `push2ex.eastmoney.com/getTopicZTPool` to fetch the limit-up pool for the latest available trading day

Default concept boards:

- `光通信模块` (`BK1136`)
- `CPO概念` (`BK1128`)
- `PCB` (`BK0877`)

The script unions board constituents, intersects them with the limit-up pool, and returns up to 10 stocks sorted by relevance and limit-up strength:

1. More matched concept boards
2. Higher consecutive limit-up count (`lbc`)
3. Higher sealed-order fund (`fund`)
4. Earlier first limit-up time (`fbt`)

The cron prompt formats the JSON into a concise Chinese Feishu message. Do not call `send_message` inside the cron prompt; Hermes cron automatically delivers the final response.

## Useful Commands

Run the crawler locally:

```bash
source venv/bin/activate
python skills/research/a-share-optical-module-pcb-limitups-feishu/scripts/fetch_optical_module_pcb_limitups.py --limit 10
```

Create the job but deliver locally for debugging:

```bash
python skills/research/a-share-optical-module-pcb-limitups-feishu/scripts/setup_daily_optical_module_pcb_limitups.py --deliver local
```

List the installed job:

```bash
source venv/bin/activate
python - <<'PY'
from cron.jobs import list_jobs
for job in list_jobs(include_disabled=True):
    if job.get("name") == "Daily A-share Optical Module PCB Limit-ups to Feishu":
        print(job)
PY
```

## Troubleshooting

- If `deliver=feishu` cannot resolve, set `FEISHU_HOME_CHANNEL` in `HERMES_HOME/.env` or recreate the job with `--deliver 'feishu:<chat_id>'`.
- If the gateway is not running, cron jobs will not tick. Start it with `hermes gateway start`.
- If Eastmoney temporarily closes connections, the crawler retries and falls back to `curl`; make sure `curl` is available on the host.
- If fewer than 10 matching stocks exist for the current trading day, report the actual count instead of inventing stocks.
