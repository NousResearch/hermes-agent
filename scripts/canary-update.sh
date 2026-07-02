#!/usr/bin/env bash
# canary-update.sh — safe rolling code update for the Master Console fleet.
#
# Replaces a bare `hermes update --yes` for fleet-wide rollouts. Instead of
# restarting every profile simultaneously, this script:
#   1. Pulls origin/main + reinstalls Python deps (no restarts yet)
#   2. Restarts ONE canary profile and waits for its API to respond
#   3. Smoke-tests the canary's fork-specific endpoints (actions, config, kanban)
#   4. On PASS: rolls out to the remaining profiles in a staggered order
#   5. On FAIL: halts — remaining profiles keep running the previous code
#
# Usage:
#   scripts/canary-update.sh              # normal rollout
#   scripts/canary-update.sh --dry-run   # check for updates without touching anything
#   scripts/canary-update.sh --force     # skip the "already up to date" guard
#
# Exit codes:
#   0 — rollout complete (or no update needed)
#   1 — update failed (git/pip error, or canary smoke test failed)
#
# Design notes:
# - Uses uv (preferred) or pip for dependency install, matching hermes update's own logic.
# - Smoke test treats HTTP 200 and 401 as "endpoint exists"; 404/5xx = feature missing/crash.
# - property-ops is restarted last because it carries time-sensitive cron jobs.
# - The default (no --profile) gateway restarts before property-ops; it runs the
#   global cron scheduler but has no per-agent jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"
LAUNCHD_DOMAIN="gui/$(id -u)"

# Canary profile: first to restart, smoke-tested before the rest of the fleet.
# browser-specialist chosen because it has no time-sensitive cron jobs.
CANARY_LABEL="ai.hermes.gateway-browser-specialist"
CANARY_PORT=8644

# Profiles restarted after canary passes, in order.
# property-ops is last: it carries the revital-email-sweep and other time-sensitive jobs.
REST_LABELS=(
  "ai.hermes.gateway-skills-consultant"
  "ai.hermes.gateway-visual-analysis"
  "ai.hermes.gateway"
  "ai.hermes.gateway-property-ops"
)

# Fork-specific API endpoints to smoke-test on the canary after restart.
# Expected HTTP responses: 200 or 401 (endpoint exists; auth required).
# 404 = fork feature was dropped by the sync.
# 5xx / 000 = gateway crashed or endpoint panicked.
SMOKE_ENDPOINTS=(
  "/v1/models"          # baseline: core API server is alive
  "/v1/actions"         # actions_api — tool-gate approval surface (fork-only)
  "/api/config"         # config_api  — Master Console config R/W (fork-only)
  "/api/kanban/tasks"   # kanban_api  — Master Console Kanban dispatch (fork-only)
)

SMOKE_WAIT_SECS=60      # seconds to wait for the canary API after kickstart
RESTART_GAP_SECS=2      # brief stagger between non-canary profile restarts

DRY_RUN=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --force)   FORCE=1 ;;
  esac
done

# ── Helpers ────────────────────────────────────────────────────────────────

log()  { echo "canary-update: $*"; }
err()  { echo "canary-update: ERROR — $*" >&2; }
fail() { err "$*"; exit 1; }

launchd_loaded() {
  # Returns 0 (true) if the launchd label is currently loaded; 1 otherwise.
  launchctl list "$1" >/dev/null 2>&1
}

kickstart() {
  local label="$1"
  launchctl kickstart -k "$LAUNCHD_DOMAIN/$label"
}

# ── 1. Check for new commits ───────────────────────────────────────────────

log "Fetching origin..."
git -C "$PROJECT_DIR" fetch origin --quiet

if [[ $FORCE -eq 0 ]] && git -C "$PROJECT_DIR" merge-base --is-ancestor origin/main HEAD; then
  log "Already up to date; nothing to do."
  exit 0
fi

new_count=$(git -C "$PROJECT_DIR" rev-list --count HEAD..origin/main 2>/dev/null || echo "?")
log "New commits on origin/main: ${new_count}. Beginning canary rollout."

[[ $DRY_RUN -eq 1 ]] && { log "DRY RUN — stopping here (not touching code or services)."; exit 0; }

# ── 2. Pull + reinstall deps (no restarts yet) ────────────────────────────

log "Pulling origin/main..."
if ! git -C "$PROJECT_DIR" pull origin main --ff-only --quiet; then
  fail "git pull --ff-only failed. Is the local branch diverged? Check git log."
fi

log "Reinstalling Python dependencies (.[all])..."
# Mirror hermes update's own logic: prefer uv in the activated venv environment.
UUV=$(command -v uv 2>/dev/null || true)
if [[ -n "$UUV" ]]; then
  VIRTUAL_ENV="$VENV_DIR" "$UUV" pip install -e "$PROJECT_DIR[all]" --quiet 2>&1 \
    | grep -v "^Resolved\|^Prepared\|^Installed\|^Uninstalled\|^Audited" || true
else
  "$VENV_PY" -m pip install -e "$PROJECT_DIR[all]" --quiet --disable-pip-version-check 2>&1 \
    | grep -v "^Obtaining\|^  Preparing\|^  Running\|^Successfully" || true
fi

# ── 3. Restart canary profile ─────────────────────────────────────────────

if ! launchd_loaded "$CANARY_LABEL"; then
  fail "Canary service '$CANARY_LABEL' is not loaded in launchd. Was it uninstalled?"
fi

log "Restarting canary: $CANARY_LABEL..."
kickstart "$CANARY_LABEL"

# ── 4. Wait for canary API ────────────────────────────────────────────────

log "Waiting up to ${SMOKE_WAIT_SECS}s for canary API on port $CANARY_PORT..."
deadline=$(( $(date +%s) + SMOKE_WAIT_SECS ))
last_code="000"
while [[ $(date +%s) -lt $deadline ]]; do
  last_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 \
    "http://localhost:$CANARY_PORT/v1/models" 2>/dev/null || echo "000")
  if [[ "$last_code" == "200" || "$last_code" == "401" ]]; then
    break
  fi
  sleep 3
done

if [[ "$last_code" != "200" && "$last_code" != "401" ]]; then
  fail "Canary gateway did not respond within ${SMOKE_WAIT_SECS}s (last HTTP code: $last_code). " \
       "Halting rollout — remaining profiles still on previous code."
fi

log "Canary API is up (HTTP $last_code). Running smoke test..."

# ── 5. Smoke test ─────────────────────────────────────────────────────────

smoke_fail=0
for endpoint in "${SMOKE_ENDPOINTS[@]}"; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    "http://localhost:$CANARY_PORT$endpoint" 2>/dev/null || echo "000")
  if [[ "$code" == "200" || "$code" == "401" ]]; then
    log "  PASS $endpoint → HTTP $code"
  else
    err "  FAIL $endpoint → HTTP $code (expected 200 or 401; 404 = feature dropped, 5xx = crash)"
    smoke_fail=1
  fi
done

if [[ $smoke_fail -ne 0 ]]; then
  fail "Smoke test FAILED on canary ($CANARY_LABEL). " \
       "Halting rollout — remaining profiles still on previous code."
fi

log "Smoke test passed."

# ── 6. Roll out to remaining profiles ─────────────────────────────────────

for label in "${REST_LABELS[@]}"; do
  if launchd_loaded "$label"; then
    log "Restarting $label..."
    kickstart "$label"
    sleep $RESTART_GAP_SECS
  else
    log "Skipping $label (not loaded in launchd)."
  fi
done

log "Canary rollout complete. All loaded fleet profiles updated."
