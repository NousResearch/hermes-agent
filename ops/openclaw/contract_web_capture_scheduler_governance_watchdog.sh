#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
exec "$ROOT/ops/openclaw/cron_protocol_outcome.sh" \
  --task "web_capture_scheduler_governance_watchdog" \
  -- "$ROOT/ops/openclaw/web_capture_scheduler_governance_guard.sh"
