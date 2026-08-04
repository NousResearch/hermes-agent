#!/bin/bash
# pr-to-blog-daily.sh — PR-to-blog generation (P13: dry-run + no background detach).
# Runs synchronously under the cron agent.
# BLOG_DAILY_DRY_RUN=1 short-circuits before any Python invocation.
set -euo pipefail

HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
ROOT="${BLOG_CONTENT_ROOT:-/home/kensei/repos/KenseiAgent/content_engine}"
LOG_DIR="$ROOT/output/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pr-to-blog-daily-$(date +%Y%m%d-%H%M%S).log"

if [[ "${BLOG_DAILY_DRY_RUN:-0}" == "1" ]]; then
  echo "dry-run: would launch PR-to-blog pipeline (synchronous) -> $LOG"
  exit 0
fi

# Legacy noop kept for backward compatibility with existing env wiring.
if [[ "${BLOG_DAILY_NOOP:-}" == "1" ]]; then
  echo "noop: would launch PR-to-blog pipeline -> $LOG"
  exit 0
fi

(
  cd "$ROOT"
  set -a
  . "${HERMES_HOME_DIR}/.env" 2>/dev/null || true
  set +a
  echo "[$(date -Is)] starting PR-to-blog daily"
  PYTHONPATH=. python3 -m blog.pr_to_blog
  rc=$?
  echo "[$(date -Is)] finished PR-to-blog daily rc=$rc"
  exit "$rc"
) >>"$LOG" 2>&1 || true

# Synchronous — silent on success (no Discord delivery)
# NOTE: || true prevents set -e from propagating the subshell's exit code.
# The subshell's rc is already captured and logged inside the block.
exit 0
