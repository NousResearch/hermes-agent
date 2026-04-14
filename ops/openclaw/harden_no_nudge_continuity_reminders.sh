#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
DELEGATE="$ROOT/ops/openclaw/harden_no_llm_watchdog_cron_authority.sh"
EXPECTED_NAMES_CSV="${OPENCLAW_NO_NUDGE_REMINDER_NAMES:-continuity:backup-checkpoint-90m,continuity:stale-progress-45m}"

if [[ ! -x "$DELEGATE" ]]; then
  echo "BLOCKER: missing delegate hardener: $DELEGATE"
  exit 1
fi

exec "$DELEGATE" --expected-names "$EXPECTED_NAMES_CSV" "$@"
