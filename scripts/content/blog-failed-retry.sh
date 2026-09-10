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

# Hermetic P13 guard: no directory, log, auth, or pipeline mutation.
if [[ "${BLOG_RETRY_NOOP:-0}" == "1" ]]; then
  echo "noop: blog failed-image retry disabled"
  exit 0
fi

ROOT=${BLOG_RETRY_ENGINE_ROOT:-/home/kensei/repos/KenseiAgent/content_engine}
LOG_DIR=$ROOT/output/logs
STATUS=$LOG_DIR/blog-failed-retry-status.json

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/blog-failed-retry-$(date +%Y%m%d-%H%M%S).log"

PIPELINE_CMD=${BLOG_RETRY_PIPELINE_CMD:-PYTHONPATH=. ../.venv/bin/python -m blog.blog_pipeline --retry}

run_pipeline() {
  local output pipeline_rc
  output=$(cd "$ROOT" && eval "$PIPELINE_CMD" 2>&1)
  pipeline_rc=$?
  printf '%s\n' "$output"
  return "$pipeline_rc"
}

OUT=$(run_pipeline)
rc=$?

# Cap detection: deferred/capped signal in output means primary exhausted.
if grep -qE "codex_capped|usage limit|CodexCapExceeded|deferred.*cap" <<<"$OUT" \
   && [[ -f "$SECONDARY" ]] && [[ ! -f "$LOCK" ]]; then
  {
    echo "[$(date -Is)] primary Codex capped - switching to secondary account"
  } >> "$LOG"
  install -m 0600 "$AUTH" "$AUTH.primary-stash"
  install -m 0600 "$SECONDARY" "$AUTH"
  touch "$LOCK"

  restore_codex_auths() {
    local trap_rc=$?
    # Codex rotates single-use refresh tokens in auth.json. Persist the
    # refreshed secondary before restoring the primary or the next fallback
    # run will replay an already-consumed token and fail authentication.
    if [[ -f "$AUTH" ]] \
       && python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$AUTH" 2>/dev/null; then
      install -m 0600 "$AUTH" "$SECONDARY"
    fi
    if [[ -f "$AUTH.primary-stash" ]]; then
      install -m 0600 "$AUTH.primary-stash" "$AUTH"
    fi
    rm -f "$LOCK" "$AUTH.primary-stash"
    return "$trap_rc"
  }
  trap restore_codex_auths EXIT

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
