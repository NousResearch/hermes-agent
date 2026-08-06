#!/bin/bash
# blog-backlog-pregen.sh — backlog pre-generation (P13: dry-run + no background detach).
# Generates ONE ready-to-approve post per run (rotating ai/pm/builder), pulling
# from the backlog queues. Runs synchronously under the cron agent.
# BLOG_DAILY_DRY_RUN=1 short-circuits before any Python invocation.
# One post = ~3 Codex images, well under the usage cap.
# Posts accrue as approved:false drafts + approval cards.
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
LOG="$LOG_DIR/blog-backlog-pregen-$(date +%Y%m%d-%H%M%S).log"

if [[ "${BLOG_DAILY_DRY_RUN:-0}" == "1" ]]; then
  echo "dry-run: would launch backlog pregen (synchronous) -> $LOG"
  exit 0
fi

# Legacy noop kept for backward compatibility with existing env wiring.
if [[ "${BLOG_DAILY_NOOP:-}" == "1" ]]; then
  echo "noop: would launch backlog pregen -> $LOG"
  exit 0
fi

(
  cd "$ROOT"
  set -a
  . "${HERMES_HOME_DIR}/.env" 2>/dev/null || true
  set +a
  echo "[$(date -Is)] starting backlog pregen"
  PYTHONPATH=. python3 -m blog.backlog_pregen
  rc=$?
  echo "[$(date -Is)] finished backlog pregen rc=$rc"
  exit "$rc"
) >>"$LOG" 2>&1 || rc=$?

# Synchronous — silent on success (no Discord delivery).
# NOTE: prior version always exited 0, masking the pipeline's rc=1 from the
# scheduler (last_status stayed 'ok' even when the run ended failed_images),
# so system-health could never report this job's failures. Propagate the real
# rc so a failed run surfaces as last_status=error and the alert chain works.
if [[ "$rc" -ne 0 ]]; then
  echo "[$(date -Is)] backlog pregen FAILED rc=$rc (log: $LOG)" >&2
fi
exit "$rc"
