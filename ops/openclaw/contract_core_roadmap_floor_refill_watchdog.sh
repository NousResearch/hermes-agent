#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
exec "$ROOT/ops/openclaw/cron_protocol_outcome.sh" \
  --task "core_roadmap_floor_refill_watchdog" \
  -- "$ROOT/ops/openclaw/run_core_roadmap_floor_refill_loop.sh"
