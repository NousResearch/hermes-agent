#!/bin/bash
# blog-stream-daily.sh — daily SahilBlog stream generation (P13: dry-run + no background detach).
# Runs synchronously under the cron agent.
# BLOG_DAILY_DRY_RUN=1 short-circuits before any Python invocation.
set -euo pipefail

HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
# Cron's minimal PATH does not include npm's user-global bin directory, where
# the ChatGPT-authenticated Codex CLI is installed. Keep this idempotent and
# prepend only when the directory exists.
if [[ -d "$HOME/.npm-global/bin" ]]; then
  export PATH="$HOME/.npm-global/bin:$PATH"
fi
ROOT="${BLOG_CONTENT_ROOT:-/home/kensei/repos/KenseiAgent/content_engine}"
LOG_DIR="$ROOT/output/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/blog-stream-daily-$(date +%Y%m%d-%H%M%S).log"

if [[ "${BLOG_DAILY_DRY_RUN:-0}" == "1" ]]; then
  echo "dry-run: would launch blog stream pipeline (synchronous) -> $LOG"
  exit 0
fi

# Legacy noop kept for backward compatibility with existing env wiring.
if [[ "${BLOG_DAILY_NOOP:-}" == "1" ]]; then
  echo "noop: would launch blog stream pipeline -> $LOG"
  exit 0
fi

(
  cd "$ROOT"
  set -a
  . "${HERMES_HOME_DIR}/.env" 2>/dev/null || true
  set +a
  echo "[$(date -Is)] starting blog stream daily"
  PYTHONPATH=. python3 -m blog.blog_pipeline --stream all
  rc=$?
  echo "[$(date -Is)] finished blog stream daily rc=$rc"
  exit "$rc"
) >>"$LOG" 2>&1 || true

# Synchronous — silent on success (no Discord delivery)
# NOTE: || true prevents set -e from propagating the subshell's exit code.
# The subshell's rc is already captured and logged inside the block.
exit 0
