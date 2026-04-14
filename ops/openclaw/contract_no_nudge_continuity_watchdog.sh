#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
exec "$ROOT/ops/openclaw/cron_protocol_outcome.sh" \
  --task "no_nudge_continuity_watchdog" \
  -- "$ROOT/ops/openclaw/run_no_nudge_continuity_watchdog.sh"
