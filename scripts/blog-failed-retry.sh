#!/bin/bash
# blog-failed-retry.sh — holistic recovery for held (failed-image) blog posts.
# Retries EVERY post in failed_images.jsonl. Codex-cap-aware: if Codex reports
# its usage cap the whole run defers (no attempts burned, drafts preserved) and
# tries again next window.
#
# Writes a status JSON (blog-failed-retry-status.json) on completion so the
# pipeline-audit cron can report the outcome to #blog-management. Previously
# this cron ran silent — retries happened with zero visibility.
#
# Env overrides (P13 isolation / local disposable-tree runs):
#   BLOG_RETRY_ENGINE_ROOT  — content_engine dir (default: production path)
#   BLOG_RETRY_PIPELINE_CMD — full command to run instead of the default
#                             blog.blog_pipeline --retry invocation
#
# Runs SYNCHRONOUSLY (the previous detached-background design made the cron
# return before the retry finished, so the audit cron could not rely on the
# status file). The pipeline exit code is captured without errexit aborting
# the wrapper and propagated as the wrapper exit code.
set -uo pipefail
# NOTE: errexit (set -e) is intentionally OFF for the wrapper so a non-zero
# pipeline exit is captured and written to the status file rather than
# aborting the wrapper before the status is persisted.

# Cron's minimal PATH does not include npm's user-global bin directory, where
# the ChatGPT-authenticated Codex CLI is installed. Keep this idempotent and
# prepend only when the directory exists.
if [[ -d "$HOME/.npm-global/bin" ]]; then
  export PATH="$HOME/.npm-global/bin:$PATH"
fi

# Load API keys from the Hermes env (same pattern as blog-backlog-pregen.sh).
# Without this the art-director LLM chain runs with empty keys: commandcode
# 401/403, ollama 429 (quota), and gemini 400 "Please pass a valid API key" —
# so every retry dies at the art brief even when a provider (gemini) is live.
HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
if [[ -f "$HERMES_HOME_DIR/.env" ]]; then
  set -a
  . "$HERMES_HOME_DIR/.env" 2>/dev/null || true
  set +a
fi

ROOT=${BLOG_RETRY_ENGINE_ROOT:-/home/kensei/repos/KenseiAgent/content_engine}
LOG_DIR=$ROOT/output/logs
STATUS=$LOG_DIR/blog-failed-retry-status.json

# Noop guard BEFORE mkdir so a noop run touches nothing on disk.
if [[ ${BLOG_RETRY_NOOP:-} == "1" ]]; then
  echo "noop: would run blog failed-image retry -> $STATUS"
  exit 0
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/blog-failed-retry-$(date +%Y%m%d-%H%M%S).log"

PIPELINE_CMD=${BLOG_RETRY_PIPELINE_CMD:-PYTHONPATH=. ../.venv/bin/python -m blog.blog_pipeline --retry}

# Run synchronously. Capture output + rc without errexit aborting the wrapper.
OUT=$(cd "$ROOT" && eval "$PIPELINE_CMD" 2>&1)
rc=$?
{
  echo "[$(date -Is)] starting blog failed-image retry"
  echo "$OUT"
  echo "[$(date -Is)] finished blog failed-image retry rc=$rc"
} >> "$LOG"

# Persist machine-readable status for the audit cron to surface. The rc,
# status path, and captured output are passed as explicit env vars to a
# separate Python helper to avoid shell escaping issues with inline Python.
BLOG_STATUS_RC="$rc" BLOG_STATUS_PATH="$STATUS" BLOG_STATUS_RAW="$OUT" \
  python3 "$(dirname "$0")/blog_retry_status_writer.py"

exit "$rc"
