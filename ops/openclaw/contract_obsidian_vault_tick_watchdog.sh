#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
exec "$ROOT/ops/openclaw/cron_protocol_outcome.sh" \
  --task "obsidian_vault_tick_watchdog" \
  -- "$ROOT/ops/obsidian/vault_tick.sh"
