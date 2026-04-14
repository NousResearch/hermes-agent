#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
HARDENER="$ROOT/ops/openclaw/harden_no_llm_watchdog_cron_authority.sh"
EXPECTED_NAMES="web-capture-scheduler-governance-watchdog,obsidian:hourly-canary,obsidian:vault-tick-hourly"

if [[ ! -x "$HARDENER" ]]; then
  echo "BLOCKER: missing hardener script: $HARDENER"
  exit 1
fi

exec "$HARDENER" --expected-names "$EXPECTED_NAMES" "$@"
