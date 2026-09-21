---
name: huangli-lunar-almanac
description: 中国农历黄历吉凶 · Zhongguo Nongli Huangli Jixiong · China Lunar Almanac. Query api.nongli.skill.4glz.com by date, range, or keyword. Use when users ask 宜忌/吉时/吉日 and need CLI-safe token setup with explicit persistence behavior.
version: 1.0.0
author: Leocdchina, adapted for Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [中国农历黄历吉凶, Zhongguo Nongli Huangli Jixiong, China Lunar Almanac, huangli, lunar, almanac, research, api]
required_environment_variables:
  - name: HUANGLI_TOKEN
    prompt: "Huangli API token"
    help: "Get token from https://nongli.skill.4glz.com/dashboard"
    required_for: "Calling Huangli API endpoints"
  - name: HUANGLI_BASE
    prompt: "Huangli API base (optional)"
    help: "Default is https://api.nongli.skill.4glz.com"
    required_for: "Override API endpoint"
---

# 中国农历黄历吉凶 · Zhongguo Nongli Huangli Jixiong · China Lunar Almanac

Query Chinese lunar almanac data using the official API for 中国农历黄历吉凶 · Zhongguo Nongli Huangli Jixiong · China Lunar Almanac.

## API endpoints

- `GET /api/lunar/date/{YYYY-MM-DD}`
- `POST /api/lunar/batch`

Base URL default: `https://api.nongli.skill.4glz.com`

## Runtime transparency

- This skill needs `HUANGLI_TOKEN` for API calls.
- Optional `HUANGLI_BASE` overrides the default base URL.
- If you use `scripts/huangli_auth.py`, it writes:
  - `~/.huangli_token.json`
  - `~/.huangli.env`
- `~/.zshrc` is modified only when `--append-zshrc` is explicitly provided.

## Quick use

```bash
python3 SKILL_DIR/scripts/huangli_toolkit.py by-date 2027-08-08
python3 SKILL_DIR/scripts/huangli_toolkit.py batch 2027-08-01 2027-08-31 --filter 搬家
python3 SKILL_DIR/scripts/huangli_toolkit.py search 甲子日 --year 2027
```

## Token setup options

### Option A: Manual token (no local token file writes)

```bash
export HUANGLI_TOKEN="your_token_here"
export HUANGLI_BASE="https://api.nongli.skill.4glz.com"
```

### Option B: CLI auth helper (writes local files)

```bash
python3 SKILL_DIR/scripts/huangli_auth.py login --username=<name> --password=<password>
source ~/.huangli.env
```

New user registration:

```bash
python3 SKILL_DIR/scripts/huangli_auth.py register --username=<name> --email=<mail>
source ~/.huangli.env
```

Status check:

```bash
python3 SKILL_DIR/scripts/huangli_auth.py status
```

## When to use which command

- `by-date`: one specific date
- `batch`: date range or multiple dates
- `search`: keyword filtering over a year or date range

## Notes

- Prefer `batch` instead of repeated single-date requests.
- On `429`, tell user to reset quota in dashboard.
- Logout/device unbind is handled on web dashboard.
