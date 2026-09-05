#!/usr/bin/env bash
# Daily review-only upstream monitor.
#
# This job fetches and reviews upstream, then writes a human-readable report to
# stdout for Hermes cron to deliver to Telegram. It never applies, restarts, or
# pushes. A real update requires an explicit user approval for the run_id.
set -euo pipefail

export TZ=Asia/Taipei
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
REPO="$HERMES_HOME/hermes-agent"
LOG_DIR="$HERMES_HOME/logs"
STATE_DIR="$HERMES_HOME/hermes-upstream-state"
CRON_STATE_DIR="$HERMES_HOME/cron"
LOG="$LOG_DIR/hermes_upstream_update_guard.log"
LOCK="$CRON_STATE_DIR/.upstream_update.lock"
IN_PROGRESS_MARKER="$CRON_STATE_DIR/.upstream_apply_in_progress"

mkdir -p "$LOG_DIR" "$CRON_STATE_DIR" "$STATE_DIR"

if [[ -f "$IN_PROGRESS_MARKER" ]]; then
  printf '⏳ Hermes upstream 每日檢查暫停：目前有 update apply 正在執行。\n'
  exit 0
fi

exec 9>"$LOCK"
if ! flock -n 9; then
  printf '⏳ Hermes upstream 每日檢查略過：另一個 updater 正在執行。\n'
  exit 0
fi

cd "$REPO"

PREFLIGHT_SCRIPT="$HERMES_HOME/scripts/hermes_upstream_preflight.py"
[[ -x "$PREFLIGHT_SCRIPT" ]] || PREFLIGHT_SCRIPT="$REPO/scripts/hermes_upstream_preflight.py"
REVIEW_SCRIPT="$HERMES_HOME/scripts/hermes_upstream_review.py"
[[ -x "$REVIEW_SCRIPT" ]] || REVIEW_SCRIPT="$REPO/scripts/hermes_upstream_review.py"
REPORT_SCRIPT="$HERMES_HOME/scripts/hermes_upstream_report.py"
[[ -x "$REPORT_SCRIPT" ]] || REPORT_SCRIPT="$REPO/scripts/hermes_upstream_report.py"

now() { date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { printf '[%s] %s\n' "$(now)" "$*" >> "$LOG"; }

if [[ ! -x "$PREFLIGHT_SCRIPT" || ! -x "$REVIEW_SCRIPT" || ! -x "$REPORT_SCRIPT" ]]; then
  message="❌ Hermes upstream 每日檢查失敗：找不到 preflight、review 或 report script；為安全起見未執行更新。"
  printf '%s\n' "$message"
  log "$message"
  exit 0
fi

preflight_output=""
if ! preflight_output="$("$PREFLIGHT_SCRIPT" \
    --repo "$REPO" \
    --state-dir "$STATE_DIR" \
    --mode review \
    --json 2>&1)"; then
  printf '⚠️ Hermes upstream 每日檢查被 gate 阻擋（未 deploy、未 restart、未 push）。\n%s\n' "$preflight_output"
  log "preflight blocked/failed"
  log "$preflight_output"
  exit 0
fi

run_id="$(date -u '+%Y%m%d-%H%M%S')"
review_output=""
if ! review_output="$("$REVIEW_SCRIPT" \
    --repo "$REPO" \
    --state-dir "$STATE_DIR" \
    --run-id "$run_id" \
    --json 2>&1)"; then
  printf '⚠️ Hermes upstream review 未通過（未 deploy、未 restart、未 push）。\n%s\n' "$review_output"
  log "review blocked/failed for run_id=$run_id"
  log "$review_output"
  exit 0
fi

candidate_path="$STATE_DIR/candidates/$run_id.json"
if [[ ! -f "$candidate_path" ]]; then
  printf '❌ Hermes upstream review 沒有產生 candidate metadata（run_id=%s）；未 deploy。\n' "$run_id"
  log "missing candidate metadata for run_id=$run_id"
  exit 0
fi

log "review completed for run_id=$run_id"
log "$review_output"

# The report helper reads only candidate metadata and Git history/diff. Its
# stdout is intentionally the Telegram payload for this no-agent cron job.
if ! "$REPORT_SCRIPT" --repo "$REPO" --candidate "$candidate_path"; then
  printf '⚠️ Hermes upstream report 產生失敗（run_id=%s）；未 deploy。\n' "$run_id"
  log "report failed for run_id=$run_id"
  exit 0
fi
