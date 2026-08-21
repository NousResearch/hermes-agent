#!/bin/bash
# blog-failed-retry with two-account Codex fallback (added 22/08/2026).
# Primary: ~/.codex/auth.json (active ChatGPT account)
# Secondary: ~/.codex/auth.json.secondary (second account; populated by Sahil re-login)
# Flow: run retry. If capped AND secondary exists -> swap auth, rerun once, swap back.
# Never leaves the secondary in place after the run; never touches auth when no cap hit.

set -uo pipefail

CODEX_DIR="$HOME/.codex"
AUTH="$CODEX_DIR/auth.json"
SECONDARY="$CODEX_DIR/auth.json.secondary"
LOCK="$CODEX_DIR/.fallback_in_use"

if [[ -d "$HOME/.npm-global/bin" ]]; then
  export PATH="$HOME/.npm-global/bin:$PATH"
fi

HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
if [[ -f "$HERMES_HOME_DIR/.env" ]]; then
  set -a; . "$HERMES_HOME_DIR/.env" 2>/dev/null || true; set +a
fi

ROOT=${BLOG_RETRY_ENGINE_ROOT:-/home/kensei/repos/KenseiAgent/content_engine}
LOG_DIR=$ROOT/output/logs
STATUS=$LOG_DIR/blog-failed-retry-status.json

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/blog-failed-retry-$(date +%Y%m%d-%H%M%S).log"

PIPELINE_CMD=${BLOG_RETRY_PIPELINE_CMD:-PYTHONPATH=. ../.venv/bin/python -m blog.blog_pipeline --retry}

run_pipeline() {
  OUT=$(cd "$ROOT" && eval "$PIPELINE_CMD" 2>&1)
  echo "$OUT"
}

OUT=$(run_pipeline)
rc=$?

# Cap detection: deferred/capped signal in output means primary exhausted.
if grep -qE "codex_capped|usage limit|CodexCapExceeded|deferred.*cap" <<<"$OUT" \
   && [[ -f "$SECONDARY" ]] && [[ ! -f "$LOCK" ]]; then
  {
    echo "[$(date -Is)] primary Codex capped - switching to secondary account"
  } >> "$LOG"
  cp "$AUTH" "$AUTH.primary-stash"
  cp "$SECONDARY" "$AUTH"
  touch "$LOCK"
  trap 'cp "$AUTH.primary-stash" "$AUTH" 2>/dev/null; rm -f "$LOCK" "$AUTH.primary-stash"' EXIT

  OUT=$(run_pipeline)
  rc=$?
  echo "[$(date -Is)] secondary-account run finished rc=$rc" >> "$LOG"
fi

{
  echo "[$(date -Is)] starting blog failed-image retry"
  echo "$OUT"
  echo "[$(date -Is)] finished blog failed-image retry rc=$rc"
} >> "$LOG"

BLOG_STATUS_RC="$rc" BLOG_STATUS_PATH="$STATUS" BLOG_STATUS_RAW="$OUT" \
  python3 "$(dirname "$0")/blog_retry_status_writer.py"

exit "$rc"
