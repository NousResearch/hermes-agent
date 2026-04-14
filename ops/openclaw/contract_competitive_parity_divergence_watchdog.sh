#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
exec "$ROOT/ops/openclaw/cron_protocol_outcome.sh" \
  --task "competitive_parity_divergence_watchdog" \
  -- "$ROOT/ops/openclaw/run_competitive_parity_harness.sh"
