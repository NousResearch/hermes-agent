#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
exec "$ROOT/ops/openclaw/cron_protocol_outcome.sh" \
  --task "context_runtime_local_watch" \
  -- "$ROOT/ops/openclaw/context_runtime_local_watch.sh"
