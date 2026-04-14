#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
exec "$ROOT/ops/openclaw/cron_protocol_outcome.sh" \
  --task "obsidian_hourly_canary_watchdog" \
  -- "$ROOT/ops/openclaw/run_obsidian_hourly_canary_watchdog.sh"
