#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
NOW_SCRIPT="$ROOT/ops/openclaw/continuity/continuity_now.sh"
CURRENT_SCRIPT="$ROOT/ops/openclaw/continuity/continuity_current.sh"
HANDOVER_SCRIPT="$ROOT/ops/openclaw/continuity/handover_latest.sh"
NO_LLM_GUARD_SCRIPT="$ROOT/ops/openclaw/no_llm_watchdog_cron_authority_guard.sh"
LEGACY_GUARD_SCRIPT="$ROOT/ops/openclaw/no_nudge_continuity_cron_guard.sh"
EVENT_ROUTER="$ROOT/ops/openclaw/event_router.sh"
EVENT_ROUTER_STATE_FILE="${OPENCLAW_EVENT_ROUTER_STATE_FILE:-$ROOT/state/cron_watchdog/event_fingerprints.json}"
BLOCKER_ROUTING_LIB="$ROOT/ops/openclaw/lib/blocker_routing.sh"
NO_NUDGE_GUARD_PROTOCOL_LIB="$ROOT/ops/openclaw/lib/no_nudge_guard_protocol.sh"
NO_NUDGE_EXPECTED_NAMES_CSV="${OPENCLAW_NO_NUDGE_REMINDER_NAMES:-continuity:backup-checkpoint-90m,continuity:stale-progress-45m}"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_MODE_RAW="${OPENCLAW_NO_NUDGE_CRON_SESSION_CARD_LIFECYCLE_RECONCILE_MODE:-disabled}"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_APPLY_ENABLED_RAW="${OPENCLAW_NO_NUDGE_CRON_SESSION_CARD_LIFECYCLE_RECONCILE_APPLY_ENABLED:-0}"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_AUTO_APPLY_ENABLED_RAW="${OPENCLAW_NO_NUDGE_CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_AUTO_APPLY_ENABLED:-0}"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_MAX_CANDIDATES="${OPENCLAW_NO_NUDGE_CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_MAX_CANDIDATES:-3}"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_MIN_AGE_SEC="${OPENCLAW_NO_NUDGE_CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_MIN_AGE_SEC:-172800}"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER="${OPENCLAW_NO_NUDGE_CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER:-$ROOT/ops/openclaw/run_cron_session_card_lifecycle_reconcile.py}"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_OPENCLAW_BIN="${OPENCLAW_NO_NUDGE_CRON_SESSION_CARD_LIFECYCLE_RECONCILE_OPENCLAW_BIN:-openclaw}"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_PYTHON_BIN="${OPENCLAW_NO_NUDGE_CRON_SESSION_CARD_LIFECYCLE_RECONCILE_PYTHON_BIN:-python3}"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TIMEOUT_SEC="${OPENCLAW_NO_NUDGE_CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TIMEOUT_SEC:-120}"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TRACE_PATH="${OPENCLAW_NO_NUDGE_CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TRACE_PATH:-$ROOT/state/cron_watchdog/no_nudge_cron_session_card_lifecycle_reconcile_latest.json}"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="not_run"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MODE="disabled"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY="not_applicable"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_STATE="disabled"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_REASON="mode_disabled"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_EVIDENCE=""
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_RESULT_MODE=""
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_CANDIDATE_COUNT="0"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MUTATED_COUNT="0"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STATE="not_applicable"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFIED_MUTATION_COUNT="0"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_MISMATCH_COUNT="0"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STORE_LOAD_ERROR_COUNT="0"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MUTATED_SAMPLE_SESSION_KEYS=""
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_ACTIVE_RUNNING_COUNT="0"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_NON_FAILED_LIKE_COUNT="0"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_MISSING_SESSION_COUNT="0"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_MISSING_STORE_COUNT="0"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STORE_LOAD_ERROR_COUNT="0"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_ROLLUP=""
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_STATUS_SUFFIX=""

MAX_AGE_SEC="${OPENCLAW_NO_NUDGE_CONTINUITY_MAX_AGE_SEC:-10800}"           # 3h hard fail
REFRESH_AFTER_SEC="${OPENCLAW_NO_NUDGE_CONTINUITY_REFRESH_AFTER_SEC:-5400}" # 90m refresh threshold
BLOCKER_COOLDOWN_SEC="${OPENCLAW_NO_NUDGE_CONTINUITY_BLOCKER_COOLDOWN_SEC:-10800}"
CRON_SESSION_SURFACE_RECONCILIATION_COOLDOWN_SEC="${OPENCLAW_NO_NUDGE_CRON_SESSION_SURFACE_RECONCILIATION_COOLDOWN_SEC:-21600}"
CRON_RUNTIME_FAILURE_INCIDENT_GEN_PATH="${OPENCLAW_NO_NUDGE_CRON_RUNTIME_FAILURE_INCIDENT_GEN_PATH:-$ROOT/state/cron_watchdog/no_nudge_runtime_failure_incident_generation.txt}"
AUTO_FRESHNESS_REPAIR="${OPENCLAW_NO_NUDGE_AUTO_FRESHNESS_REPAIR:-1}"
RESTORE_DRILL_REFRESH_SCRIPT="$ROOT/ops/openclaw/continuity/restore_drill_refresh.sh"
RESTORE_DRILL_AUTO_REFRESH_ENABLED="${OPENCLAW_NO_NUDGE_RESTORE_DRILL_AUTO_REFRESH_ENABLED:-1}"
RESTORE_DRILL_REFRESH_AFTER_SEC="${OPENCLAW_NO_NUDGE_RESTORE_DRILL_REFRESH_AFTER_SEC:-518400}"
RESTORE_DRILL_MAX_AGE_SEC="${OPENCLAW_NO_NUDGE_RESTORE_DRILL_MAX_AGE_SEC:-604800}"
CONTINUITY_DISPATCH="$ROOT/ops/openclaw/continuity.sh"
EXECUTION_FRONTIER_LEDGER_PATH="$ROOT/state/continuity/latest/execution_frontier_ledger.json"
EXECUTION_FRONTIER_CONTROLLER_ENABLED="${OPENCLAW_NO_NUDGE_EXECUTION_FRONTIER_CONTROLLER_ENABLED:-1}"
EXECUTION_FRONTIER_CONTROLLER_REASON="${OPENCLAW_NO_NUDGE_EXECUTION_FRONTIER_CONTROLLER_REASON:-watchdog_no_nudge_controller_tick}"
EXECUTION_FRONTIER_CONTROLLER_TRACE_PATH="${OPENCLAW_NO_NUDGE_EXECUTION_FRONTIER_CONTROLLER_TRACE_PATH:-$ROOT/state/continuity/latest/no_nudge_execution_frontier_controller_tick_latest.json}"
EXECUTION_FRONTIER_CONTROLLER_HISTORY_PATH="${OPENCLAW_NO_NUDGE_EXECUTION_FRONTIER_CONTROLLER_HISTORY_PATH:-$ROOT/state/continuity/history/no_nudge_execution_frontier_controller_ticks.jsonl}"
EXECUTION_FRONTIER_ENFORCEMENT_LATCH_PATH="${OPENCLAW_NO_NUDGE_EXECUTION_FRONTIER_ENFORCEMENT_LATCH_PATH:-$ROOT/state/continuity/latest/execution_frontier_post_completion_enforcement_latch.json}"
EXECUTION_FRONTIER_ENFORCEMENT_LATCH_HISTORY_PATH="${OPENCLAW_NO_NUDGE_EXECUTION_FRONTIER_ENFORCEMENT_LATCH_HISTORY_PATH:-$ROOT/state/continuity/history/execution_frontier_post_completion_enforcement_latch.jsonl}"
AUTONOMOUS_EXECUTION_INTENT_PATH="${OPENCLAW_NO_NUDGE_AUTONOMOUS_EXECUTION_INTENT_PATH:-$ROOT/state/continuity/latest/autonomous_execution_intent_latest.json}"
AUTONOMOUS_EXECUTION_INTENT_HISTORY_PATH="${OPENCLAW_NO_NUDGE_AUTONOMOUS_EXECUTION_INTENT_HISTORY_PATH:-$ROOT/state/continuity/history/autonomous_execution_intent_history.jsonl}"
EXECUTION_FRONTIER_CONTROLLER_COOLDOWN_AFTER="${OPENCLAW_NO_NUDGE_EXECUTION_FRONTIER_CONTROLLER_COOLDOWN_AFTER:-3}"
EXECUTION_FRONTIER_CONTROLLER_COOLDOWN_SEC="${OPENCLAW_NO_NUDGE_EXECUTION_FRONTIER_CONTROLLER_COOLDOWN_SEC:-900}"
EXECUTION_FRONTIER_CONTROLLER_RETRY_BUDGET="${OPENCLAW_NO_NUDGE_EXECUTION_FRONTIER_CONTROLLER_RETRY_BUDGET:-1}"

AUTOPILOT_STATE_FILE="$ROOT/ops/autopilot/state/hl_terminal_v1.json"
AUTOPILOT_TICK_SCRIPT="$ROOT/ops/autopilot/bin/hl_autopilot_tick.sh"
IDLE_AUTOSPAWN_TRACE_PATH="$ROOT/state/continuity/latest/no_nudge_idle_lane_autospawn_latest.json"
IDLE_AUTOSPAWN_ENABLED="${OPENCLAW_NO_NUDGE_IDLE_LANE_AUTOSPAWN_ENABLED:-1}"
IDLE_AUTOSPAWN_IDLE_SEC="${OPENCLAW_NO_NUDGE_IDLE_LANE_AUTOSPAWN_IDLE_SEC:-1800}"
IDLE_AUTOSPAWN_COOLDOWN_SEC="${OPENCLAW_NO_NUDGE_IDLE_LANE_AUTOSPAWN_COOLDOWN_SEC:-900}"
IDLE_AUTOSPAWN_IMPLEMENTATION_STEP_IDS="${OPENCLAW_NO_NUDGE_IDLE_LANE_IMPLEMENTATION_STEP_IDS:-apply_fixes}"
IDLE_AUTOSPAWN_CONTRADICTION_LATCH_PATH="${OPENCLAW_NO_NUDGE_IDLE_LANE_CONTRADICTION_LATCH_PATH:-$ROOT/state/continuity/latest/no_nudge_idle_lane_autospawn_contradiction_latch.json}"
IDLE_AUTOSPAWN_CONTRADICTION_LATCH_AFTER="${OPENCLAW_NO_NUDGE_IDLE_LANE_CONTRADICTION_LATCH_AFTER:-3}"
IDLE_AUTOSPAWN_CONTRADICTION_ABORT_SEC="${OPENCLAW_NO_NUDGE_IDLE_LANE_CONTRADICTION_ABORT_SEC:-7200}"
# Optional hard guard for contradiction latch leases. Default (0) derives a bounded max from abort_sec.
IDLE_AUTOSPAWN_CONTRADICTION_MAX_REMAINING_SEC="${OPENCLAW_NO_NUDGE_IDLE_LANE_CONTRADICTION_MAX_REMAINING_SEC:-0}"

ROUTED_BLOCKER_ENABLED=0
if [[ -f "$BLOCKER_ROUTING_LIB" ]]; then
  # shellcheck source=ops/openclaw/lib/blocker_routing.sh
  source "$BLOCKER_ROUTING_LIB"
  if [[ -x "$EVENT_ROUTER" ]] && declare -F openclaw_route_blocker >/dev/null 2>&1; then
    ROUTED_BLOCKER_ENABLED=1
  fi
fi

emit_blocker() {
  local key="${1:-unknown_blocker}"
  local msg="${2:-unknown_blocker}"
  local evidence_ref="${3:-$ROOT/state/continuity/latest/handover_latest.json}"
  local fingerprint_input="${4:-task=run_no_nudge_continuity_watchdog;key=${key}}"

  if [[ "$ROUTED_BLOCKER_ENABLED" -eq 1 ]] && declare -F openclaw_route_blocker >/dev/null 2>&1; then
    openclaw_route_blocker \
      --event-router "$EVENT_ROUTER" \
      --source "watchdog.no_nudge_continuity" \
      --key "$key" \
      --severity "critical" \
      --summary "$msg" \
      --evidence-ref "$evidence_ref" \
      --cooldown-sec "$BLOCKER_COOLDOWN_SEC" \
      --fingerprint-input "$fingerprint_input"
  else
    echo "BLOCKER: ${msg}"
  fi
}

emit_operator_event() {
  local key="${1:-unknown_event}"
  local severity="${2:-info}"
  local msg="${3:-unknown_event}"
  local evidence_ref="${4:-$ROOT/state/continuity/latest/handover_latest.json}"
  local cooldown_sec="${5:-$BLOCKER_COOLDOWN_SEC}"
  local fingerprint_input="${6:-task=run_no_nudge_continuity_watchdog;key=${key}}"

  if [[ "$ROUTED_BLOCKER_ENABLED" -eq 1 ]] && declare -F openclaw_route_event >/dev/null 2>&1; then
    openclaw_route_event \
      --event-router "$EVENT_ROUTER" \
      --source "watchdog.no_nudge_continuity" \
      --key "$key" \
      --severity "$severity" \
      --summary "$msg" \
      --evidence-ref "$evidence_ref" \
      --cooldown-sec "$cooldown_sec" \
      --fingerprint-input "$fingerprint_input" >/dev/null 2>&1 || true
  fi
}

normalize_non_negative_int() {
  local raw="${1:-0}"
  if [[ "$raw" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$raw"
  else
    printf '0\n'
  fi
}

build_cron_session_card_lifecycle_single_run_promotion_evidence() {
  local requested_mode="${1:-unknown}"
  local state="${2:-unknown}"
  local reason="${3:-unknown}"
  local candidate_count
  candidate_count="$(normalize_non_negative_int "${4:-0}")"
  local max_candidates
  max_candidates="$(normalize_non_negative_int "${5:-0}")"
  local min_age_sec
  min_age_sec="$(normalize_non_negative_int "${6:-0}")"
  local age_known_count
  age_known_count="$(normalize_non_negative_int "${7:-0}")"
  local age_unknown_count
  age_unknown_count="$(normalize_non_negative_int "${8:-0}")"
  local oldest_age_sec="${9:-unknown}"
  local newest_age_sec="${10:-unknown}"
  local apply_enabled="${11:-0}"

  if [[ "$oldest_age_sec" != "unknown" ]]; then
    oldest_age_sec="$(normalize_non_negative_int "$oldest_age_sec")"
  fi
  if [[ "$newest_age_sec" != "unknown" ]]; then
    newest_age_sec="$(normalize_non_negative_int "$newest_age_sec")"
  fi

  printf 'v1|requested=%s|state=%s|reason=%s|candidate=%s|max_candidates=%s|min_age_sec=%s|age_known=%s|age_unknown=%s|oldest_age_sec=%s|newest_age_sec=%s|apply_enabled=%s\n' \
    "$requested_mode" \
    "$state" \
    "$reason" \
    "$candidate_count" \
    "$max_candidates" \
    "$min_age_sec" \
    "$age_known_count" \
    "$age_unknown_count" \
    "$oldest_age_sec" \
    "$newest_age_sec" \
    "$apply_enabled"
}

build_cron_session_card_lifecycle_reconcile_rollup() {
  local status="${1:-unknown}"
  local mode="${2:-unknown}"
  local apply_enable_policy="${3:-unknown}"
  local result_mode="${4:-unknown}"
  local verification_state="${5:-unknown}"
  local candidate_count
  candidate_count="$(normalize_non_negative_int "${6:-0}")"
  local mutated_count
  mutated_count="$(normalize_non_negative_int "${7:-0}")"
  local verified_mutation_count
  verified_mutation_count="$(normalize_non_negative_int "${8:-0}")"
  local verification_mismatch_count
  verification_mismatch_count="$(normalize_non_negative_int "${9:-0}")"
  local verification_store_load_error_count
  verification_store_load_error_count="$(normalize_non_negative_int "${10:-0}")"
  local skipped_active_running_count
  skipped_active_running_count="$(normalize_non_negative_int "${11:-0}")"
  local skipped_non_failed_like_count
  skipped_non_failed_like_count="$(normalize_non_negative_int "${12:-0}")"
  local skipped_missing_session_count
  skipped_missing_session_count="$(normalize_non_negative_int "${13:-0}")"
  local skipped_missing_store_count
  skipped_missing_store_count="$(normalize_non_negative_int "${14:-0}")"
  local store_load_error_count
  store_load_error_count="$(normalize_non_negative_int "${15:-0}")"

  local skipped_total=$((
    skipped_active_running_count +
    skipped_non_failed_like_count +
    skipped_missing_session_count +
    skipped_missing_store_count
  ))
  local error_total=$((
    verification_mismatch_count +
    verification_store_load_error_count +
    store_load_error_count
  ))

  local outcome="unknown"
  case "$status" in
    disabled)
      outcome="disabled"
      ;;
    dry_run_ok)
      outcome="dry_run_projected"
      ;;
    apply_ok)
      if [[ "$mutated_count" -gt 0 ]]; then
        outcome="apply_verified_mutated"
      else
        outcome="apply_verified_noop"
      fi
      ;;
    apply_policy_blocked)
      outcome="apply_policy_blocked"
      ;;
    dry_run_failed|apply_failed|mode_invalid)
      outcome="reconcile_failed"
      ;;
    *)
      outcome="$status"
      ;;
  esac

  printf 'v1|status=%s|mode=%s|policy=%s|result=%s|verification=%s|outcome=%s|candidate=%s|mutated=%s|verified=%s|skipped_total=%s|error_total=%s\n' \
    "$status" \
    "$mode" \
    "$apply_enable_policy" \
    "$result_mode" \
    "$verification_state" \
    "$outcome" \
    "$candidate_count" \
    "$mutated_count" \
    "$verified_mutation_count" \
    "$skipped_total" \
    "$error_total"
}

# Always project a baseline reconcile policy tuple in watchdog first-line output,
# even when no historical residue rows are present (default safe mode).
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_EVIDENCE="$(build_cron_session_card_lifecycle_single_run_promotion_evidence \
  "disabled" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_STATE:-disabled}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_REASON:-mode_disabled}" \
  "0" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_MAX_CANDIDATES:-3}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_MIN_AGE_SEC:-172800}" \
  "0" \
  "0" \
  "unknown" \
  "unknown" \
  "0")"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_ROLLUP="$(build_cron_session_card_lifecycle_reconcile_rollup \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS:-disabled}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MODE:-disabled}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY:-not_applicable}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_RESULT_MODE:-disabled}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STATE:-disabled}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_CANDIDATE_COUNT:-0}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MUTATED_COUNT:-0}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFIED_MUTATION_COUNT:-0}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_MISMATCH_COUNT:-0}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STORE_LOAD_ERROR_COUNT:-0}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_ACTIVE_RUNNING_COUNT:-0}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_NON_FAILED_LIKE_COUNT:-0}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_MISSING_SESSION_COUNT:-0}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_MISSING_STORE_COUNT:-0}" \
  "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STORE_LOAD_ERROR_COUNT:-0}")"
CRON_SESSION_CARD_LIFECYCLE_RECONCILE_STATUS_SUFFIX="; cron_session_card_lifecycle_reconcile_status=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS:-disabled}; cron_session_card_lifecycle_reconcile_mode=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MODE:-disabled}; cron_session_card_lifecycle_reconcile_apply_enable_policy=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY:-not_applicable}; cron_session_card_lifecycle_reconcile_single_run_promotion_state=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_STATE:-disabled}; cron_session_card_lifecycle_reconcile_single_run_promotion_reason=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_REASON:-mode_disabled}; cron_session_card_lifecycle_reconcile_single_run_promotion_evidence=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_EVIDENCE}; cron_session_card_lifecycle_reconcile_result_mode=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_RESULT_MODE:-disabled}; cron_session_card_lifecycle_reconcile_candidate_count=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_CANDIDATE_COUNT:-0}; cron_session_card_lifecycle_reconcile_mutated_count=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MUTATED_COUNT:-0}; cron_session_card_lifecycle_reconcile_verification_state=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STATE:-disabled}; cron_session_card_lifecycle_reconcile_verified_mutation_count=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFIED_MUTATION_COUNT:-0}; cron_session_card_lifecycle_reconcile_verification_mismatch_count=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_MISMATCH_COUNT:-0}; cron_session_card_lifecycle_reconcile_verification_store_load_error_count=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STORE_LOAD_ERROR_COUNT:-0}; cron_session_card_lifecycle_reconcile_skipped_active_running_count=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_ACTIVE_RUNNING_COUNT:-0}; cron_session_card_lifecycle_reconcile_skipped_non_failed_like_count=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_NON_FAILED_LIKE_COUNT:-0}; cron_session_card_lifecycle_reconcile_skipped_missing_session_count=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_MISSING_SESSION_COUNT:-0}; cron_session_card_lifecycle_reconcile_skipped_missing_store_count=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_MISSING_STORE_COUNT:-0}; cron_session_card_lifecycle_reconcile_store_load_error_count=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STORE_LOAD_ERROR_COUNT:-0}; cron_session_card_lifecycle_reconcile_rollup=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_ROLLUP}; cron_session_card_lifecycle_reconcile_trace_path=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TRACE_PATH}"

run_cron_session_card_lifecycle_reconcile() {
  local residue_names="${1:-}"
  local mode_raw="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_MODE_RAW:-disabled}"
  local mode_norm=""
  mode_norm="$(printf '%s' "$mode_raw" | tr '[:upper:]' '[:lower:]')"

  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="not_run"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MODE="disabled"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY="not_applicable"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_STATE="disabled"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_REASON="mode_disabled"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_EVIDENCE="$(build_cron_session_card_lifecycle_single_run_promotion_evidence \
    "disabled" \
    "disabled" \
    "mode_disabled" \
    "0" \
    "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_MAX_CANDIDATES:-3}" \
    "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_MIN_AGE_SEC:-172800}" \
    "0" \
    "0" \
    "unknown" \
    "unknown" \
    "0")"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_RESULT_MODE=""
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_CANDIDATE_COUNT="0"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MUTATED_COUNT="0"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STATE="not_applicable"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFIED_MUTATION_COUNT="0"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_MISMATCH_COUNT="0"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STORE_LOAD_ERROR_COUNT="0"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MUTATED_SAMPLE_SESSION_KEYS=""
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_ACTIVE_RUNNING_COUNT="0"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_NON_FAILED_LIKE_COUNT="0"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_MISSING_SESSION_COUNT="0"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_MISSING_STORE_COUNT="0"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STORE_LOAD_ERROR_COUNT="0"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_ROLLUP=""

  local apply_enabled_raw="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_APPLY_ENABLED_RAW:-0}"
  local apply_enabled_norm=""
  local apply_enabled_flag="0"
  apply_enabled_norm="$(printf '%s' "$apply_enabled_raw" | tr '[:upper:]' '[:lower:]')"
  case "$apply_enabled_norm" in
    "1"|"on"|"true"|"yes"|"enabled")
      apply_enabled_flag="1"
      ;;
  esac

  local invoke_mode="disabled"
  case "$mode_norm" in
    ""|"0"|"off"|"false"|"no"|"disabled")
      invoke_mode="disabled"
      ;;
    "dry_run"|"dry-run"|"dryrun"|"plan")
      invoke_mode="dry_run"
      ;;
    "apply"|"write"|"on"|"true"|"yes"|"1"|"enabled")
      invoke_mode="apply"
      ;;
    *)
      CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="mode_invalid"
      local mode_invalid_rollup=""
      mode_invalid_rollup="$(build_cron_session_card_lifecycle_reconcile_rollup \
        "mode_invalid" \
        "invalid" \
        "not_applicable" \
        "invalid" \
        "unknown" \
        "0" \
        "0" \
        "0" \
        "0" \
        "0" \
        "0" \
        "0" \
        "0" \
        "0" \
        "0")"
      emit_blocker \
        "cron_session_card_lifecycle_reconcile_mode_invalid" \
        "cron_session_card_lifecycle_reconcile_mode_invalid; mode=${mode_raw}; cron_session_card_lifecycle_reconcile_rollup=${mode_invalid_rollup}" \
        "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER" \
        "task=run_no_nudge_continuity_watchdog;key=cron_session_card_lifecycle_reconcile_mode_invalid;mode=${mode_raw};rollup=${mode_invalid_rollup}"
      return 1
      ;;
  esac

  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MODE="$invoke_mode"

  if [[ "$invoke_mode" == "apply" ]]; then
    case "$apply_enabled_flag" in
      "1")
        CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY="allowed"
        ;;
      *)
        CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY="blocked"
        CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="apply_policy_blocked"
        CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_RESULT_MODE="policy_blocked"
        CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STATE="policy_blocked"
        local policy_blocked_rollup=""
        policy_blocked_rollup="$(build_cron_session_card_lifecycle_reconcile_rollup \
          "apply_policy_blocked" \
          "apply" \
          "blocked" \
          "policy_blocked" \
          "policy_blocked" \
          "0" \
          "0" \
          "0" \
          "0" \
          "0" \
          "0" \
          "0" \
          "0" \
          "0" \
          "0")"
        emit_blocker \
          "cron_session_card_lifecycle_reconcile_apply_policy_blocked" \
          "cron_session_card_lifecycle_reconcile_apply_policy_blocked; cron_session_card_lifecycle_reconcile_status=apply_policy_blocked; cron_session_card_lifecycle_reconcile_mode=apply; cron_session_card_lifecycle_reconcile_result_mode=policy_blocked; cron_session_card_lifecycle_reconcile_verification_state=policy_blocked; cron_session_card_lifecycle_reconcile_apply_enable_policy=blocked; cron_session_card_lifecycle_reconcile_rollup=${policy_blocked_rollup}; apply_enable=${apply_enabled_raw}; required=OPENCLAW_NO_NUDGE_CRON_SESSION_CARD_LIFECYCLE_RECONCILE_APPLY_ENABLED=1" \
          "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER" \
          "task=run_no_nudge_continuity_watchdog;key=cron_session_card_lifecycle_reconcile_apply_policy_blocked;mode=apply;apply_enable_policy=blocked;apply_enable=${apply_enabled_raw};rollup=${policy_blocked_rollup}"
        return 1
        ;;
    esac
  fi

  if [[ "$invoke_mode" == "disabled" ]]; then
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="disabled"
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_RESULT_MODE="disabled"
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STATE="disabled"
    return 0
  fi

  if [[ ! -f "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER" ]]; then
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="${invoke_mode}_failed"
    local driver_missing_rollup=""
    driver_missing_rollup="$(build_cron_session_card_lifecycle_reconcile_rollup \
      "${invoke_mode}_failed" \
      "$invoke_mode" \
      "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY:-not_applicable}" \
      "driver_missing" \
      "unknown" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0")"
    emit_blocker \
      "cron_session_card_lifecycle_reconcile_driver_missing" \
      "cron_session_card_lifecycle_reconcile_driver_missing; path=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER}; cron_session_card_lifecycle_reconcile_rollup=${driver_missing_rollup}" \
      "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER" \
      "task=run_no_nudge_continuity_watchdog;key=cron_session_card_lifecycle_reconcile_driver_missing;path=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER};rollup=${driver_missing_rollup}"
    return 1
  fi

  local -a reconcile_cmd
  reconcile_cmd=(
    "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_PYTHON_BIN"
    "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER"
    "--collect-inputs"
    "--guard-script"
    "$GUARD_SCRIPT"
    "--expected-names"
    "$NO_NUDGE_EXPECTED_NAMES_CSV"
    "--openclaw-bin"
    "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_OPENCLAW_BIN"
    "--timeout-sec"
    "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TIMEOUT_SEC"
  )
  if [[ "$invoke_mode" == "apply" ]]; then
    reconcile_cmd+=("--apply")
  elif [[ "$invoke_mode" == "dry_run" ]]; then
    local single_run_auto_apply_norm=""
    single_run_auto_apply_norm="$(printf '%s' "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_AUTO_APPLY_ENABLED_RAW:-0}" | tr '[:upper:]' '[:lower:]')"
    case "$single_run_auto_apply_norm" in
      "1"|"on"|"true"|"yes"|"enabled")
        reconcile_cmd+=(
          "--single-run-auto-apply"
          "--single-run-max-candidates"
          "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_MAX_CANDIDATES"
          "--single-run-min-age-sec"
          "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_MIN_AGE_SEC"
        )
        if [[ "$apply_enabled_flag" == "1" ]]; then
          reconcile_cmd+=("--single-run-apply-enabled")
        fi
        ;;
    esac
  fi

  set +e
  local reconcile_json=""
  reconcile_json="$("${reconcile_cmd[@]}" 2>/tmp/run_no_nudge_continuity_watchdog_cron_session_card_lifecycle_reconcile.err)"
  local reconcile_rc=$?
  set -e

  if [[ "$reconcile_rc" -ne 0 || -z "$reconcile_json" ]]; then
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="${invoke_mode}_failed"
    local err=""
    err="$(cat /tmp/run_no_nudge_continuity_watchdog_cron_session_card_lifecycle_reconcile.err 2>/dev/null || true)"
    local failed_rollup=""
    failed_rollup="$(build_cron_session_card_lifecycle_reconcile_rollup \
      "${invoke_mode}_failed" \
      "$invoke_mode" \
      "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY:-not_applicable}" \
      "driver_error" \
      "unknown" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0")"
    emit_blocker \
      "cron_session_card_lifecycle_reconcile_failed" \
      "cron_session_card_lifecycle_reconcile_failed; mode=${invoke_mode}; residue_names=${residue_names}; rc=${reconcile_rc}; err=${err:0:180}; cron_session_card_lifecycle_reconcile_rollup=${failed_rollup}" \
      "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER" \
      "task=run_no_nudge_continuity_watchdog;key=cron_session_card_lifecycle_reconcile_failed;mode=${invoke_mode};rc=${reconcile_rc};rollup=${failed_rollup}"
    return 1
  fi

  mkdir -p "$(dirname "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TRACE_PATH")"
  printf '%s\n' "$reconcile_json" > "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TRACE_PATH" 2>/dev/null || true

  set +e
  local parse_output=""
  parse_output="$(python3 - "$reconcile_json" 2>/tmp/run_no_nudge_continuity_watchdog_cron_session_card_lifecycle_reconcile_parse.err <<'PY'
import json
import sys

obj = json.loads(sys.argv[1])
if not isinstance(obj, dict):
    raise SystemExit(2)

result = obj.get("result") if isinstance(obj.get("result"), dict) else {}
ok = obj.get("ok") is True
error = str(obj.get("error") or "").strip()
mode = str(result.get("mode") or "").strip()
mutated_count = int(result.get("mutated_count") or 0)
candidate_count = int(result.get("candidate_count") or 0)
verification_state = str(result.get("verification_state") or "").strip()
if not verification_state:
    verification_state = "not_applicable_dry_run" if mode == "dry_run" else "unknown"
verified_mutation_count = int(result.get("verified_mutation_count") or 0)
verification_mismatch_count = int(result.get("verification_mismatch_count") or 0)
verification_store_load_error_count = int(result.get("verification_store_load_error_count") or 0)
mutated_session_keys = result.get("mutated_session_keys") if isinstance(result.get("mutated_session_keys"), list) else []
mutated_sample_session_keys = []
for item in mutated_session_keys:
    txt = str(item or "").strip()
    if not txt:
        continue
    mutated_sample_session_keys.append(txt)
    if len(mutated_sample_session_keys) >= 3:
        break
skipped_active_running_count = int(result.get("skipped_active_running_count") or 0)
skipped_non_failed_like_count = int(result.get("skipped_non_failed_like_count") or 0)
skipped_missing_session_count = int(result.get("skipped_missing_session_count") or 0)
skipped_missing_store_count = int(result.get("skipped_missing_store_count") or 0)
store_load_error_count = int(result.get("store_load_error_count") or 0)
single_run_promotion_state = str(result.get("single_run_promotion_state") or "").strip() or "unknown"
single_run_promotion_reason = str(result.get("single_run_promotion_reason") or "").strip() or "unknown"
single_run_promotion_evidence = str(result.get("single_run_promotion_evidence") or "").strip()
rollup = str(result.get("rollup") or "").strip()

print("1" if ok else "0")
print(error)
print(mode)
print(str(candidate_count))
print(str(mutated_count))
print(verification_state)
print(str(verified_mutation_count))
print(str(verification_mismatch_count))
print(str(verification_store_load_error_count))
print(",".join(mutated_sample_session_keys))
print(str(skipped_active_running_count))
print(str(skipped_non_failed_like_count))
print(str(skipped_missing_session_count))
print(str(skipped_missing_store_count))
print(str(store_load_error_count))
print(single_run_promotion_state)
print(single_run_promotion_reason)
print(single_run_promotion_evidence)
print(rollup)
PY
)"
  local parse_rc=$?
  set -e

  if [[ "$parse_rc" -ne 0 ]]; then
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="${invoke_mode}_failed"
    local err=""
    err="$(cat /tmp/run_no_nudge_continuity_watchdog_cron_session_card_lifecycle_reconcile_parse.err 2>/dev/null || true)"
    local parse_failed_rollup=""
    parse_failed_rollup="$(build_cron_session_card_lifecycle_reconcile_rollup \
      "${invoke_mode}_failed" \
      "$invoke_mode" \
      "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY:-not_applicable}" \
      "invalid_json" \
      "unknown" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0" \
      "0")"
    emit_blocker \
      "cron_session_card_lifecycle_reconcile_failed" \
      "cron_session_card_lifecycle_reconcile_failed; mode=${invoke_mode}; residue_names=${residue_names}; invalid_json=1; err=${err:0:180}; cron_session_card_lifecycle_reconcile_rollup=${parse_failed_rollup}" \
      "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER" \
      "task=run_no_nudge_continuity_watchdog;key=cron_session_card_lifecycle_reconcile_failed;mode=${invoke_mode};invalid_json=1;rollup=${parse_failed_rollup}"
    return 1
  fi

  readarray -t parsed_fields <<<"$parse_output"
  local parsed_ok="${parsed_fields[0]:-0}"
  local parsed_error="${parsed_fields[1]:-}"
  local parsed_mode="${parsed_fields[2]:-}"
  local parsed_candidate_count="${parsed_fields[3]:-0}"
  local parsed_mutated_count="${parsed_fields[4]:-0}"
  local parsed_verification_state="${parsed_fields[5]:-unknown}"
  local parsed_verified_mutation_count="${parsed_fields[6]:-0}"
  local parsed_verification_mismatch_count="${parsed_fields[7]:-0}"
  local parsed_verification_store_load_error_count="${parsed_fields[8]:-0}"
  local parsed_mutated_sample_session_keys="${parsed_fields[9]:-}"
  local parsed_skipped_active_running_count="${parsed_fields[10]:-0}"
  local parsed_skipped_non_failed_like_count="${parsed_fields[11]:-0}"
  local parsed_skipped_missing_session_count="${parsed_fields[12]:-0}"
  local parsed_skipped_missing_store_count="${parsed_fields[13]:-0}"
  local parsed_store_load_error_count="${parsed_fields[14]:-0}"
  local parsed_single_run_promotion_state="${parsed_fields[15]:-unknown}"
  local parsed_single_run_promotion_reason="${parsed_fields[16]:-unknown}"
  local parsed_single_run_promotion_evidence="${parsed_fields[17]:-}"
  local parsed_rollup="${parsed_fields[18]:-}"

  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MODE="${parsed_mode:-$invoke_mode}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_RESULT_MODE="${parsed_mode:-unknown}"
  if [[ "${parsed_mode:-}" == "apply" ]]; then
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY="allowed"
  fi
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_CANDIDATE_COUNT="${parsed_candidate_count:-0}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MUTATED_COUNT="${parsed_mutated_count:-0}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STATE="${parsed_verification_state:-unknown}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFIED_MUTATION_COUNT="${parsed_verified_mutation_count:-0}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_MISMATCH_COUNT="${parsed_verification_mismatch_count:-0}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STORE_LOAD_ERROR_COUNT="${parsed_verification_store_load_error_count:-0}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MUTATED_SAMPLE_SESSION_KEYS="${parsed_mutated_sample_session_keys:-}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_ACTIVE_RUNNING_COUNT="${parsed_skipped_active_running_count:-0}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_NON_FAILED_LIKE_COUNT="${parsed_skipped_non_failed_like_count:-0}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_MISSING_SESSION_COUNT="${parsed_skipped_missing_session_count:-0}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_MISSING_STORE_COUNT="${parsed_skipped_missing_store_count:-0}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STORE_LOAD_ERROR_COUNT="${parsed_store_load_error_count:-0}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_STATE="${parsed_single_run_promotion_state:-unknown}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_REASON="${parsed_single_run_promotion_reason:-unknown}"
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_EVIDENCE="${parsed_single_run_promotion_evidence:-}"
  if [[ -z "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_EVIDENCE" ]]; then
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_EVIDENCE="$(build_cron_session_card_lifecycle_single_run_promotion_evidence \
      "$invoke_mode" \
      "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_STATE:-unknown}" \
      "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_REASON:-unknown}" \
      "${parsed_candidate_count:-0}" \
      "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_MAX_CANDIDATES:-3}" \
      "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_SINGLE_RUN_MIN_AGE_SEC:-172800}" \
      "0" \
      "0" \
      "unknown" \
      "unknown" \
      "$apply_enabled_flag")"
  fi
  CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_ROLLUP="${parsed_rollup:-}"

  if [[ "$parsed_ok" != "1" ]]; then
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="${invoke_mode}_failed"
    local failed_rollup="${parsed_rollup:-}"
    if [[ -z "$failed_rollup" ]]; then
      failed_rollup="$(build_cron_session_card_lifecycle_reconcile_rollup \
        "${invoke_mode}_failed" \
        "$invoke_mode" \
        "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY:-not_applicable}" \
        "${parsed_mode:-unknown}" \
        "${parsed_verification_state:-unknown}" \
        "${parsed_candidate_count:-0}" \
        "${parsed_mutated_count:-0}" \
        "${parsed_verified_mutation_count:-0}" \
        "${parsed_verification_mismatch_count:-0}" \
        "${parsed_verification_store_load_error_count:-0}" \
        "${parsed_skipped_active_running_count:-0}" \
        "${parsed_skipped_non_failed_like_count:-0}" \
        "${parsed_skipped_missing_session_count:-0}" \
        "${parsed_skipped_missing_store_count:-0}" \
        "${parsed_store_load_error_count:-0}")"
    fi
    emit_blocker \
      "cron_session_card_lifecycle_reconcile_failed" \
      "cron_session_card_lifecycle_reconcile_failed; mode=${invoke_mode}; residue_names=${residue_names}; error=${parsed_error:-driver_reported_failure}; cron_session_card_lifecycle_reconcile_rollup=${failed_rollup}" \
      "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER" \
      "task=run_no_nudge_continuity_watchdog;key=cron_session_card_lifecycle_reconcile_failed;mode=${invoke_mode};error=${parsed_error:-driver_reported_failure};rollup=${failed_rollup}"
    return 1
  fi

  if [[ "$invoke_mode" == "dry_run" && "${parsed_mode:-}" == "apply" && "${parsed_single_run_promotion_state:-unknown}" != "promoted" ]]; then
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="dry_run_failed"
    local promotion_mismatch_rollup="${parsed_rollup:-}"
    if [[ -z "$promotion_mismatch_rollup" ]]; then
      promotion_mismatch_rollup="$(build_cron_session_card_lifecycle_reconcile_rollup \
        "dry_run_failed" \
        "dry_run" \
        "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY:-not_applicable}" \
        "${parsed_mode:-unknown}" \
        "${parsed_verification_state:-unknown}" \
        "${parsed_candidate_count:-0}" \
        "${parsed_mutated_count:-0}" \
        "${parsed_verified_mutation_count:-0}" \
        "${parsed_verification_mismatch_count:-0}" \
        "${parsed_verification_store_load_error_count:-0}" \
        "${parsed_skipped_active_running_count:-0}" \
        "${parsed_skipped_non_failed_like_count:-0}" \
        "${parsed_skipped_missing_session_count:-0}" \
        "${parsed_skipped_missing_store_count:-0}" \
        "${parsed_store_load_error_count:-0}")"
    fi
    emit_blocker \
      "cron_session_card_lifecycle_reconcile_failed" \
      "cron_session_card_lifecycle_reconcile_failed; mode=${invoke_mode}; residue_names=${residue_names}; expected_mode=dry_run_or_promoted_apply; driver_mode=${parsed_mode:-unknown}; single_run_promotion_state=${parsed_single_run_promotion_state:-unknown}; single_run_promotion_reason=${parsed_single_run_promotion_reason:-unknown}; cron_session_card_lifecycle_reconcile_rollup=${promotion_mismatch_rollup}" \
      "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER" \
      "task=run_no_nudge_continuity_watchdog;key=cron_session_card_lifecycle_reconcile_failed;mode=${invoke_mode};expected_mode=dry_run_or_promoted_apply;driver_mode=${parsed_mode:-unknown};single_run_promotion_state=${parsed_single_run_promotion_state:-unknown};rollup=${promotion_mismatch_rollup}"
    return 1
  fi

  if [[ "$invoke_mode" == "dry_run" && "${parsed_mode:-}" != "dry_run" && "${parsed_mode:-}" != "apply" ]]; then
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="dry_run_failed"
    local mode_mismatch_rollup="${parsed_rollup:-}"
    if [[ -z "$mode_mismatch_rollup" ]]; then
      mode_mismatch_rollup="$(build_cron_session_card_lifecycle_reconcile_rollup \
        "dry_run_failed" \
        "dry_run" \
        "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY:-not_applicable}" \
        "${parsed_mode:-unknown}" \
        "${parsed_verification_state:-unknown}" \
        "${parsed_candidate_count:-0}" \
        "${parsed_mutated_count:-0}" \
        "${parsed_verified_mutation_count:-0}" \
        "${parsed_verification_mismatch_count:-0}" \
        "${parsed_verification_store_load_error_count:-0}" \
        "${parsed_skipped_active_running_count:-0}" \
        "${parsed_skipped_non_failed_like_count:-0}" \
        "${parsed_skipped_missing_session_count:-0}" \
        "${parsed_skipped_missing_store_count:-0}" \
        "${parsed_store_load_error_count:-0}")"
    fi
    emit_blocker \
      "cron_session_card_lifecycle_reconcile_failed" \
      "cron_session_card_lifecycle_reconcile_failed; mode=${invoke_mode}; residue_names=${residue_names}; expected_mode=dry_run; driver_mode=${parsed_mode:-unknown}; cron_session_card_lifecycle_reconcile_rollup=${mode_mismatch_rollup}" \
      "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER" \
      "task=run_no_nudge_continuity_watchdog;key=cron_session_card_lifecycle_reconcile_failed;mode=${invoke_mode};expected_mode=dry_run;driver_mode=${parsed_mode:-unknown};rollup=${mode_mismatch_rollup}"
    return 1
  fi

  if [[ "$invoke_mode" == "apply" && "${parsed_mode:-}" != "apply" ]]; then
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="apply_failed"
    local mode_mismatch_rollup="${parsed_rollup:-}"
    if [[ -z "$mode_mismatch_rollup" ]]; then
      mode_mismatch_rollup="$(build_cron_session_card_lifecycle_reconcile_rollup \
        "apply_failed" \
        "apply" \
        "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY:-not_applicable}" \
        "${parsed_mode:-unknown}" \
        "${parsed_verification_state:-unknown}" \
        "${parsed_candidate_count:-0}" \
        "${parsed_mutated_count:-0}" \
        "${parsed_verified_mutation_count:-0}" \
        "${parsed_verification_mismatch_count:-0}" \
        "${parsed_verification_store_load_error_count:-0}" \
        "${parsed_skipped_active_running_count:-0}" \
        "${parsed_skipped_non_failed_like_count:-0}" \
        "${parsed_skipped_missing_session_count:-0}" \
        "${parsed_skipped_missing_store_count:-0}" \
        "${parsed_store_load_error_count:-0}")"
    fi
    emit_blocker \
      "cron_session_card_lifecycle_reconcile_failed" \
      "cron_session_card_lifecycle_reconcile_failed; mode=${invoke_mode}; residue_names=${residue_names}; expected_mode=apply; driver_mode=${parsed_mode:-unknown}; cron_session_card_lifecycle_reconcile_rollup=${mode_mismatch_rollup}" \
      "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER" \
      "task=run_no_nudge_continuity_watchdog;key=cron_session_card_lifecycle_reconcile_failed;mode=${invoke_mode};expected_mode=apply;driver_mode=${parsed_mode:-unknown};rollup=${mode_mismatch_rollup}"
    return 1
  fi

  if [[ ( "$invoke_mode" == "apply" || ( "$invoke_mode" == "dry_run" && "${parsed_mode:-}" == "apply" && "${parsed_single_run_promotion_state:-unknown}" == "promoted" ) ) && "${parsed_verification_state:-unknown}" != "verified" ]]; then
    if [[ "$invoke_mode" == "apply" ]]; then
      CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="apply_failed"
    else
      CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="dry_run_failed"
    fi
    local verification_failed_rollup="${parsed_rollup:-}"
    if [[ -z "$verification_failed_rollup" ]]; then
      verification_failed_rollup="$(build_cron_session_card_lifecycle_reconcile_rollup \
        "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS:-apply_failed}" \
        "${parsed_mode:-apply}" \
        "${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY:-not_applicable}" \
        "${parsed_mode:-unknown}" \
        "${parsed_verification_state:-unknown}" \
        "${parsed_candidate_count:-0}" \
        "${parsed_mutated_count:-0}" \
        "${parsed_verified_mutation_count:-0}" \
        "${parsed_verification_mismatch_count:-0}" \
        "${parsed_verification_store_load_error_count:-0}" \
        "${parsed_skipped_active_running_count:-0}" \
        "${parsed_skipped_non_failed_like_count:-0}" \
        "${parsed_skipped_missing_session_count:-0}" \
        "${parsed_skipped_missing_store_count:-0}" \
        "${parsed_store_load_error_count:-0}")"
    fi
    emit_blocker \
      "cron_session_card_lifecycle_reconcile_failed" \
      "cron_session_card_lifecycle_reconcile_failed; mode=${invoke_mode}; residue_names=${residue_names}; expected_verification_state=verified; driver_mode=${parsed_mode:-unknown}; driver_verification_state=${parsed_verification_state:-unknown}; single_run_promotion_state=${parsed_single_run_promotion_state:-unknown}; verification_mismatch_count=${parsed_verification_mismatch_count:-0}; verification_store_load_error_count=${parsed_verification_store_load_error_count:-0}; cron_session_card_lifecycle_reconcile_rollup=${verification_failed_rollup}" \
      "$CRON_SESSION_CARD_LIFECYCLE_RECONCILE_DRIVER" \
      "task=run_no_nudge_continuity_watchdog;key=cron_session_card_lifecycle_reconcile_failed;mode=${invoke_mode};expected_verification_state=verified;driver_mode=${parsed_mode:-unknown};driver_verification_state=${parsed_verification_state:-unknown};single_run_promotion_state=${parsed_single_run_promotion_state:-unknown};rollup=${verification_failed_rollup}"
    return 1
  fi

  if [[ "$invoke_mode" == "apply" || ( "$invoke_mode" == "dry_run" && "${parsed_mode:-}" == "apply" && "${parsed_single_run_promotion_state:-unknown}" == "promoted" ) ]]; then
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="apply_ok"
  else
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS="dry_run_ok"
  fi

  return 0
}

read_runtime_failure_incident_generation() {
  local raw=""
  if [[ -f "$CRON_RUNTIME_FAILURE_INCIDENT_GEN_PATH" ]]; then
    raw="$(cat "$CRON_RUNTIME_FAILURE_INCIDENT_GEN_PATH" 2>/dev/null || true)"
  fi
  if [[ "$raw" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$raw"
  else
    printf '0\n'
  fi
}

bump_runtime_failure_incident_generation() {
  local current="0"
  current="$(read_runtime_failure_incident_generation)"
  if [[ ! "$current" =~ ^[0-9]+$ ]]; then
    current="0"
  fi
  local next=$((current + 1))
  mkdir -p "$(dirname "$CRON_RUNTIME_FAILURE_INCIDENT_GEN_PATH")"
  printf '%s\n' "$next" > "$CRON_RUNTIME_FAILURE_INCIDENT_GEN_PATH"
  printf '%s\n' "$next"
}

session_surface_reconciliation_clear_transition_pending() {
  python3 - "$EVENT_ROUTER_STATE_FILE" <<'PY'
import json
import pathlib
import sys

state_path = pathlib.Path(sys.argv[1])
if not state_path.exists():
    print("0")
    raise SystemExit(0)

try:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
except Exception:
    print("0")
    raise SystemExit(0)

events = payload.get("events") if isinstance(payload, dict) else {}
if not isinstance(events, dict):
    print("0")
    raise SystemExit(0)

def last_emitted_epoch(route_key: str) -> int:
    row = events.get(route_key)
    if not isinstance(row, dict):
        return 0
    try:
        return int(row.get("last_emitted_epoch") or 0)
    except Exception:
        return 0

projected_epoch = max(
    last_emitted_epoch("watchdog.no_nudge_continuity|cron_session_surface_reconciliation_projected"),
    last_emitted_epoch("watchdog.no_nudge_continuity|cron_session_surface_reconciliation_status_unknown"),
    last_emitted_epoch("watchdog.no_nudge_continuity|cron_session_surface_reconciliation_resolved_historical_cards"),
    last_emitted_epoch("watchdog.no_nudge_continuity|cron_session_surface_reconciliation_retired_historical_cards"),
)
cleared_epoch = last_emitted_epoch("watchdog.no_nudge_continuity|cron_session_surface_reconciliation_cleared")

if projected_epoch > 0 and projected_epoch > cleared_epoch:
    print("1")
else:
    print("0")
PY
}

extract_historical_retired_card_rows() {
  local raw_per_rail="${1:-}"
  python3 - "$raw_per_rail" <<'PY'
import sys

raw = str(sys.argv[1] or "").strip()
if not raw:
    print("")
    raise SystemExit(0)

rows = []
seen = set()
for token in raw.split(','):
    item = str(token or "").strip()
    if not item or '=' not in item:
        continue
    name, state = item.split('=', 1)
    name = str(name or "").strip()
    state = str(state or "").strip()
    if not name or state != "resolved_historical_retired":
        continue
    row = f"{name}={state}"
    if row in seen:
        continue
    seen.add(row)
    rows.append(row)

print(','.join(rows[:20]))
PY
}

extract_resolved_historical_card_rows() {
  local raw_per_rail="${1:-}"
  python3 - "$raw_per_rail" <<'PY'
import sys

raw = str(sys.argv[1] or "").strip()
if not raw:
    print("")
    raise SystemExit(0)

rows = []
seen = set()
for token in raw.split(','):
    item = str(token or "").strip()
    if not item or '=' not in item:
        continue
    name, state = item.split('=', 1)
    name = str(name or "").strip()
    state = str(state or "").strip()
    if not name or not state.startswith("resolved_historical_"):
        continue
    row = f"{name}={state}"
    if row in seen:
        continue
    seen.add(row)
    rows.append(row)

print(','.join(rows[:20]))
PY
}

autospawn_contradiction_latch_snapshot() {
  python3 - "$IDLE_AUTOSPAWN_CONTRADICTION_LATCH_PATH" "$IDLE_AUTOSPAWN_CONTRADICTION_ABORT_SEC" "$IDLE_AUTOSPAWN_CONTRADICTION_MAX_REMAINING_SEC" <<'PY'
import json
import pathlib
import time
import os
import sys

path = pathlib.Path(sys.argv[1])
try:
    configured_abort_sec = max(0, int(sys.argv[2]))
except Exception:
    configured_abort_sec = 7200
try:
    configured_max_remaining_sec = max(0, int(sys.argv[3]))
except Exception:
    configured_max_remaining_sec = 0

fixed_now_raw = os.environ.get("OPENCLAW_AUTOPILOT_FIXED_NOW_TS")
if fixed_now_raw is not None and str(fixed_now_raw).strip() != "":
    try:
        now_ts = int(str(fixed_now_raw).strip())
    except Exception:
        now_ts = int(time.time())
else:
    now_ts = int(time.time())
active = "0"
remaining = "0"
consecutive = "0"
total = "0"
last_kind = ""
last_reason = ""
issue = ""

if path.exists():
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            consecutive_count = int(payload.get("consecutive_count") or 0)
            total_count = int(payload.get("total_count") or 0)
            threshold = max(1, int(payload.get("threshold") or 1))
            consecutive = str(consecutive_count)
            total = str(total_count)
            last_kind = str(payload.get("last_kind") or "")
            last_reason = str(payload.get("last_reason") or "")
            latched = bool(payload.get("latched") is True or consecutive_count >= threshold)
            abort_until_ts = int(payload.get("abort_until_ts") or 0)
            payload_abort_sec = int(payload.get("abort_sec") or 0)
            effective_abort_sec = payload_abort_sec if payload_abort_sec > 0 else configured_abort_sec
            if configured_max_remaining_sec > 0:
                max_remaining_sec = configured_max_remaining_sec
            else:
                max_remaining_sec = max(effective_abort_sec * 2, effective_abort_sec, 7200)

            if abort_until_ts > now_ts:
                remaining_val = max(0, abort_until_ts - now_ts)
                if not latched:
                    issue = "abort_until_present_without_latched_state"
                elif max_remaining_sec > 0 and remaining_val > max_remaining_sec:
                    issue = f"abort_remaining_exceeds_guard:{remaining_val}>{max_remaining_sec}"
                else:
                    active = "1"
                    remaining = str(remaining_val)
    except Exception:
        issue = "latch_payload_invalid_json"

print(active)
print(remaining)
print(consecutive)
print(total)
print(last_kind)
print(last_reason)
print(issue)
PY
}

record_autospawn_contradiction_latch() {
  local contradiction_kind="${1:-unknown}"
  local contradiction_reason="${2:-unknown}"
  local trace_path="${3:-$IDLE_AUTOSPAWN_TRACE_PATH}"

  # Nounset-safe forwarding: avoid watchdog hard-fail even if caller/local naming drifts.
  python3 - "$IDLE_AUTOSPAWN_CONTRADICTION_LATCH_PATH" "$IDLE_AUTOSPAWN_CONTRADICTION_LATCH_AFTER" "$IDLE_AUTOSPAWN_CONTRADICTION_ABORT_SEC" "${contradiction_kind:-${1:-unknown}}" "${contradiction_reason:-${2:-unknown}}" "${trace_path:-${3:-$IDLE_AUTOSPAWN_TRACE_PATH}}" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
try:
    threshold = max(1, int(sys.argv[2]))
except Exception:
    threshold = 3
try:
    abort_sec = max(0, int(sys.argv[3]))
except Exception:
    abort_sec = 7200
kind = str(sys.argv[4] or "unknown")
reason = str(sys.argv[5] or "unknown")
trace_path = str(sys.argv[6] or "")

now_ts = int(time.time())
now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts))
payload = {}
if path.exists():
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded
    except Exception:
        payload = {}

consecutive_count = int(payload.get("consecutive_count") or 0) + 1
total_count = int(payload.get("total_count") or 0) + 1
latched = consecutive_count >= threshold
abort_until_ts = (now_ts + abort_sec) if latched and abort_sec > 0 else 0
abort_until_at = (
    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(abort_until_ts))
    if abort_until_ts > 0
    else ""
)

updated = {
    "schema": "claw.no_nudge_idle_lane_autospawn_contradiction_latch.v1",
    "updated_at": now_iso,
    "updated_ts": now_ts,
    "threshold": threshold,
    "abort_sec": abort_sec,
    "latched": bool(latched),
    "abort_until_ts": abort_until_ts,
    "abort_until_at": abort_until_at,
    "consecutive_count": consecutive_count,
    "total_count": total_count,
    "last_kind": kind,
    "last_reason": reason,
    "trace_path": trace_path,
}

path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print("1" if latched else "0")
print(str(consecutive_count))
print(str(total_count))
print(str(abort_until_ts))
print(abort_until_at)
PY
}

clear_autospawn_contradiction_latch() {
  rm -f "$IDLE_AUTOSPAWN_CONTRADICTION_LATCH_PATH" 2>/dev/null || true
}

GUARD_SCRIPT=""
GUARD_EVIDENCE_REF=""
GUARD_ARGS=()
if [[ -x "$NO_LLM_GUARD_SCRIPT" ]]; then
  GUARD_SCRIPT="$NO_LLM_GUARD_SCRIPT"
  GUARD_EVIDENCE_REF="$NO_LLM_GUARD_SCRIPT"
  GUARD_ARGS=(--expected-names "$NO_NUDGE_EXPECTED_NAMES_CSV")
elif [[ -x "$LEGACY_GUARD_SCRIPT" ]]; then
  GUARD_SCRIPT="$LEGACY_GUARD_SCRIPT"
  GUARD_EVIDENCE_REF="$LEGACY_GUARD_SCRIPT"
  GUARD_ARGS=()
elif [[ -e "$NO_LLM_GUARD_SCRIPT" ]]; then
  GUARD_SCRIPT="$NO_LLM_GUARD_SCRIPT"
  GUARD_EVIDENCE_REF="$NO_LLM_GUARD_SCRIPT"
  GUARD_ARGS=(--expected-names "$NO_NUDGE_EXPECTED_NAMES_CSV")
else
  GUARD_SCRIPT="$LEGACY_GUARD_SCRIPT"
  GUARD_EVIDENCE_REF="$LEGACY_GUARD_SCRIPT"
  GUARD_ARGS=()
fi

for dep in "$NOW_SCRIPT" "$CURRENT_SCRIPT" "$HANDOVER_SCRIPT" "$NO_NUDGE_GUARD_PROTOCOL_LIB" "$GUARD_SCRIPT"; do
  if [[ ! -e "$dep" ]]; then
    emit_blocker "missing_dependency" "missing_dependency=${dep}" "$dep" "task=run_no_nudge_continuity_watchdog;key=missing_dependency;dep=${dep}"
    exit 0
  fi
done

for dep in "$NOW_SCRIPT" "$CURRENT_SCRIPT" "$HANDOVER_SCRIPT" "$GUARD_SCRIPT"; do
  if [[ ! -x "$dep" ]]; then
    emit_blocker "missing_dependency" "missing_dependency=${dep}" "$dep" "task=run_no_nudge_continuity_watchdog;key=missing_dependency;dep=${dep}"
    exit 0
  fi
done

# shellcheck source=ops/openclaw/lib/no_nudge_guard_protocol.sh
source "$NO_NUDGE_GUARD_PROTOCOL_LIB"
if ! declare -F openclaw_no_nudge_guard_first_line >/dev/null 2>&1; then
  emit_blocker "missing_dependency" "missing_dependency=openclaw_no_nudge_guard_first_line" "$NO_NUDGE_GUARD_PROTOCOL_LIB" "task=run_no_nudge_continuity_watchdog;key=missing_dependency;dep=openclaw_no_nudge_guard_first_line"
  exit 0
fi

# Guard against drift back to chat-facing reminder cron rails.
guard_line="$(openclaw_no_nudge_guard_first_line "$GUARD_SCRIPT" "/tmp/run_no_nudge_continuity_watchdog_guard.err" "${GUARD_ARGS[@]}")"
if [[ "$guard_line" == BLOCKER:* ]]; then
  guard_detail="${guard_line#BLOCKER: }"
  guard_event_key="cron_contract_drift_detected"

  # Preserve guard taxonomy through watchdog/event-router keys.
  # - connectivity failures from cron list path
  # - contract drift/protocol/schema/runtime shape issues
  # - runtime failures from enabled authority rails (operator recovery required)
  # - policy failures (chat-facing/noisy reminder rails)
  case "$guard_detail" in
    no_nudge_cron_guard_gateway_connectivity_failure*|no_llm_watchdog_cron_authority_gateway_connectivity_failure*)
      guard_event_key="gateway_connectivity_failure_detected"
      ;;
    no_nudge_cron_guard_runtime_failed*|no_llm_watchdog_cron_authority_runtime_failed*)
      guard_event_key="cron_runtime_failure_detected"
      ;;
    no_nudge_cron_guard_policy_failed*|no_llm_watchdog_cron_authority_policy_failed*)
      guard_event_key="cron_policy_failure_detected"
      ;;
    *)
      guard_event_key="cron_contract_drift_detected"
      ;;
  esac

  guard_fingerprint="task=run_no_nudge_continuity_watchdog;key=${guard_event_key};detail=${guard_detail}"
  if [[ "$guard_event_key" == "cron_runtime_failure_detected" ]]; then
    runtime_failure_incident_generation="$(read_runtime_failure_incident_generation)"
    guard_fingerprint="${guard_fingerprint};runtime_failure_incident_generation=${runtime_failure_incident_generation}"
  fi

  emit_blocker "$guard_event_key" "${guard_event_key}; detail=${guard_detail}" "$GUARD_EVIDENCE_REF" "$guard_fingerprint"
  exit 0
fi

if [[ "$guard_line" == READY:* && "$guard_line" == *"recovered_historical_names="* ]]; then
  guard_recovered_names="$(printf '%s' "$guard_line" | sed -n 's/.*recovered_historical_names=\([^;[:space:]]*\).*/\1/p')"
  if [[ -n "$guard_recovered_names" ]]; then
    bump_runtime_failure_incident_generation >/dev/null
    emit_operator_event \
      "cron_runtime_failure_recovered" \
      "info" \
      "cron_runtime_failure_recovered; recovered_historical_names=${guard_recovered_names}" \
      "$GUARD_EVIDENCE_REF" \
      "$BLOCKER_COOLDOWN_SEC" \
      "task=run_no_nudge_continuity_watchdog;key=cron_runtime_failure_recovered;recovered_historical_names=${guard_recovered_names}"
  fi
fi

if [[ "$guard_line" == READY:* && "$guard_line" == *"historical_failed_session_residue_names="* ]]; then
  guard_session_residue_names="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_names=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_ownership_per_rail="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_ownership_per_rail=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_playbook_per_rail="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_playbook_per_rail=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_current_health_per_rail="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_current_health_per_rail=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_decay_state="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_decay_state=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_decay_urgency="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_decay_urgency=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_staleness_per_rail="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_staleness_per_rail=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_card_scope="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_card_scope=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_card_per_rail="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_card_per_rail=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_card_status_scope="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_card_status_scope=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_card_status_per_rail="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_card_status_per_rail=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_card_severity_scope="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_card_severity_scope=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_card_severity_per_rail="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_card_severity_per_rail=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_card_retirement_scope="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_card_retirement_scope=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_card_retirement_per_rail="$(printf '%s' "$guard_line" | sed -n 's/.*historical_failed_session_residue_card_retirement_per_rail=\([^;[:space:]]*\).*/\1/p')"
  guard_session_residue_card_resolved_historical_per_rail="$(extract_resolved_historical_card_rows "$guard_session_residue_card_retirement_per_rail")"
  guard_session_residue_card_retired_per_rail="$(extract_historical_retired_card_rows "$guard_session_residue_card_retirement_per_rail")"
  if [[ -n "$guard_session_residue_names" ]]; then
    if ! run_cron_session_card_lifecycle_reconcile "$guard_session_residue_names"; then
      exit 0
    fi

    reconcile_status="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STATUS:-not_run}"
    reconcile_mode="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MODE:-disabled}"
    reconcile_apply_enable_policy="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_APPLY_ENABLE_POLICY:-not_applicable}"
    reconcile_single_run_promotion_state="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_STATE:-disabled}"
    reconcile_single_run_promotion_reason="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_REASON:-mode_disabled}"
    reconcile_single_run_promotion_evidence="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SINGLE_RUN_PROMOTION_EVIDENCE:-}"
    reconcile_result_mode="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_RESULT_MODE:-unknown}"
    reconcile_candidate_count="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_CANDIDATE_COUNT:-0}"
    reconcile_mutated_count="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MUTATED_COUNT:-0}"
    reconcile_verification_state="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STATE:-unknown}"
    reconcile_verified_mutation_count="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFIED_MUTATION_COUNT:-0}"
    reconcile_verification_mismatch_count="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_MISMATCH_COUNT:-0}"
    reconcile_verification_store_load_error_count="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_VERIFICATION_STORE_LOAD_ERROR_COUNT:-0}"
    reconcile_mutated_sample_session_keys="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_MUTATED_SAMPLE_SESSION_KEYS:-}"
    reconcile_skipped_active_running_count="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_ACTIVE_RUNNING_COUNT:-0}"
    reconcile_skipped_non_failed_like_count="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_NON_FAILED_LIKE_COUNT:-0}"
    reconcile_skipped_missing_session_count="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_MISSING_SESSION_COUNT:-0}"
    reconcile_skipped_missing_store_count="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_SKIPPED_MISSING_STORE_COUNT:-0}"
    reconcile_store_load_error_count="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_STORE_LOAD_ERROR_COUNT:-0}"
    reconcile_rollup="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_LAST_ROLLUP:-}"
    if [[ -z "$reconcile_rollup" ]]; then
      reconcile_rollup="$(build_cron_session_card_lifecycle_reconcile_rollup \
        "$reconcile_status" \
        "$reconcile_mode" \
        "$reconcile_apply_enable_policy" \
        "$reconcile_result_mode" \
        "$reconcile_verification_state" \
        "$reconcile_candidate_count" \
        "$reconcile_mutated_count" \
        "$reconcile_verified_mutation_count" \
        "$reconcile_verification_mismatch_count" \
        "$reconcile_verification_store_load_error_count" \
        "$reconcile_skipped_active_running_count" \
        "$reconcile_skipped_non_failed_like_count" \
        "$reconcile_skipped_missing_session_count" \
        "$reconcile_skipped_missing_store_count" \
        "$reconcile_store_load_error_count")"
    fi
    CRON_SESSION_CARD_LIFECYCLE_RECONCILE_STATUS_SUFFIX="; cron_session_card_lifecycle_reconcile_status=${reconcile_status}; cron_session_card_lifecycle_reconcile_mode=${reconcile_mode}; cron_session_card_lifecycle_reconcile_apply_enable_policy=${reconcile_apply_enable_policy}; cron_session_card_lifecycle_reconcile_single_run_promotion_state=${reconcile_single_run_promotion_state}; cron_session_card_lifecycle_reconcile_single_run_promotion_reason=${reconcile_single_run_promotion_reason}; cron_session_card_lifecycle_reconcile_single_run_promotion_evidence=${reconcile_single_run_promotion_evidence}; cron_session_card_lifecycle_reconcile_result_mode=${reconcile_result_mode}; cron_session_card_lifecycle_reconcile_candidate_count=${reconcile_candidate_count}; cron_session_card_lifecycle_reconcile_mutated_count=${reconcile_mutated_count}; cron_session_card_lifecycle_reconcile_verification_state=${reconcile_verification_state}; cron_session_card_lifecycle_reconcile_verified_mutation_count=${reconcile_verified_mutation_count}; cron_session_card_lifecycle_reconcile_verification_mismatch_count=${reconcile_verification_mismatch_count}; cron_session_card_lifecycle_reconcile_verification_store_load_error_count=${reconcile_verification_store_load_error_count}; cron_session_card_lifecycle_reconcile_skipped_active_running_count=${reconcile_skipped_active_running_count}; cron_session_card_lifecycle_reconcile_skipped_non_failed_like_count=${reconcile_skipped_non_failed_like_count}; cron_session_card_lifecycle_reconcile_skipped_missing_session_count=${reconcile_skipped_missing_session_count}; cron_session_card_lifecycle_reconcile_skipped_missing_store_count=${reconcile_skipped_missing_store_count}; cron_session_card_lifecycle_reconcile_store_load_error_count=${reconcile_store_load_error_count}; cron_session_card_lifecycle_reconcile_rollup=${reconcile_rollup}; cron_session_card_lifecycle_reconcile_trace_path=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TRACE_PATH}"
    if [[ -n "$reconcile_mutated_sample_session_keys" ]]; then
      CRON_SESSION_CARD_LIFECYCLE_RECONCILE_STATUS_SUFFIX+="; cron_session_card_lifecycle_reconcile_mutated_sample_session_keys=${reconcile_mutated_sample_session_keys}"
    fi

    reconciliation_summary="cron_session_surface_reconciliation_projected; historical_failed_session_residue_names=${guard_session_residue_names}; authority_runtime_status=healthy"
    reconciliation_fingerprint="task=run_no_nudge_continuity_watchdog;key=cron_session_surface_reconciliation_projected;historical_failed_session_residue_names=${guard_session_residue_names}"
    reconciliation_summary+="; cron_session_card_lifecycle_reconcile_status=${reconcile_status}; cron_session_card_lifecycle_reconcile_mode=${reconcile_mode}; cron_session_card_lifecycle_reconcile_apply_enable_policy=${reconcile_apply_enable_policy}; cron_session_card_lifecycle_reconcile_single_run_promotion_state=${reconcile_single_run_promotion_state}; cron_session_card_lifecycle_reconcile_single_run_promotion_reason=${reconcile_single_run_promotion_reason}; cron_session_card_lifecycle_reconcile_single_run_promotion_evidence=${reconcile_single_run_promotion_evidence}; cron_session_card_lifecycle_reconcile_result_mode=${reconcile_result_mode}; cron_session_card_lifecycle_reconcile_candidate_count=${reconcile_candidate_count}; cron_session_card_lifecycle_reconcile_mutated_count=${reconcile_mutated_count}; cron_session_card_lifecycle_reconcile_verification_state=${reconcile_verification_state}; cron_session_card_lifecycle_reconcile_verified_mutation_count=${reconcile_verified_mutation_count}; cron_session_card_lifecycle_reconcile_verification_mismatch_count=${reconcile_verification_mismatch_count}; cron_session_card_lifecycle_reconcile_verification_store_load_error_count=${reconcile_verification_store_load_error_count}; cron_session_card_lifecycle_reconcile_skipped_active_running_count=${reconcile_skipped_active_running_count}; cron_session_card_lifecycle_reconcile_skipped_non_failed_like_count=${reconcile_skipped_non_failed_like_count}; cron_session_card_lifecycle_reconcile_skipped_missing_session_count=${reconcile_skipped_missing_session_count}; cron_session_card_lifecycle_reconcile_skipped_missing_store_count=${reconcile_skipped_missing_store_count}; cron_session_card_lifecycle_reconcile_store_load_error_count=${reconcile_store_load_error_count}; cron_session_card_lifecycle_reconcile_rollup=${reconcile_rollup}; cron_session_card_lifecycle_reconcile_trace_path=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TRACE_PATH}"
    if [[ -n "$reconcile_mutated_sample_session_keys" ]]; then
      reconciliation_summary+="; cron_session_card_lifecycle_reconcile_mutated_sample_session_keys=${reconcile_mutated_sample_session_keys}"
    fi
    reconciliation_fingerprint+=";cron_session_card_lifecycle_reconcile_status=${reconcile_status};cron_session_card_lifecycle_reconcile_mode=${reconcile_mode};cron_session_card_lifecycle_reconcile_apply_enable_policy=${reconcile_apply_enable_policy};cron_session_card_lifecycle_reconcile_single_run_promotion_state=${reconcile_single_run_promotion_state};cron_session_card_lifecycle_reconcile_single_run_promotion_reason=${reconcile_single_run_promotion_reason};cron_session_card_lifecycle_reconcile_single_run_promotion_evidence=${reconcile_single_run_promotion_evidence};cron_session_card_lifecycle_reconcile_result_mode=${reconcile_result_mode};cron_session_card_lifecycle_reconcile_verification_state=${reconcile_verification_state};cron_session_card_lifecycle_reconcile_verified_mutation_count=${reconcile_verified_mutation_count};cron_session_card_lifecycle_reconcile_verification_mismatch_count=${reconcile_verification_mismatch_count};cron_session_card_lifecycle_reconcile_verification_store_load_error_count=${reconcile_verification_store_load_error_count};cron_session_card_lifecycle_reconcile_rollup=${reconcile_rollup};cron_session_card_lifecycle_reconcile_trace_path=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TRACE_PATH}"
    if [[ -n "$guard_session_residue_ownership_per_rail" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_ownership_per_rail=${guard_session_residue_ownership_per_rail}"
      reconciliation_fingerprint+=";historical_failed_session_residue_ownership_per_rail=${guard_session_residue_ownership_per_rail}"
    fi
    if [[ -n "$guard_session_residue_playbook_per_rail" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_playbook_per_rail=${guard_session_residue_playbook_per_rail}"
      reconciliation_fingerprint+=";historical_failed_session_residue_playbook_per_rail=${guard_session_residue_playbook_per_rail}"
    fi
    if [[ -n "$guard_session_residue_current_health_per_rail" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_current_health_per_rail=${guard_session_residue_current_health_per_rail}"
      reconciliation_fingerprint+=";historical_failed_session_residue_current_health_per_rail=${guard_session_residue_current_health_per_rail}"
    fi
    if [[ -n "$guard_session_residue_decay_state" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_decay_state=${guard_session_residue_decay_state}"
      reconciliation_fingerprint+=";historical_failed_session_residue_decay_state=${guard_session_residue_decay_state}"
    fi
    if [[ -n "$guard_session_residue_decay_urgency" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_decay_urgency=${guard_session_residue_decay_urgency}"
      reconciliation_fingerprint+=";historical_failed_session_residue_decay_urgency=${guard_session_residue_decay_urgency}"
    fi
    if [[ -n "$guard_session_residue_staleness_per_rail" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_staleness_per_rail=${guard_session_residue_staleness_per_rail}"
      reconciliation_fingerprint+=";historical_failed_session_residue_staleness_per_rail=${guard_session_residue_staleness_per_rail}"
    fi
    if [[ -n "$guard_session_residue_card_scope" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_card_scope=${guard_session_residue_card_scope}"
      reconciliation_fingerprint+=";historical_failed_session_residue_card_scope=${guard_session_residue_card_scope}"
    fi
    if [[ -n "$guard_session_residue_card_per_rail" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_card_per_rail=${guard_session_residue_card_per_rail}"
      reconciliation_fingerprint+=";historical_failed_session_residue_card_per_rail=${guard_session_residue_card_per_rail}"
    fi
    if [[ -n "$guard_session_residue_card_status_scope" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_card_status_scope=${guard_session_residue_card_status_scope}"
      reconciliation_fingerprint+=";historical_failed_session_residue_card_status_scope=${guard_session_residue_card_status_scope}"
    fi
    if [[ -n "$guard_session_residue_card_status_per_rail" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_card_status_per_rail=${guard_session_residue_card_status_per_rail}"
      reconciliation_fingerprint+=";historical_failed_session_residue_card_status_per_rail=${guard_session_residue_card_status_per_rail}"
    fi
    if [[ -n "$guard_session_residue_card_severity_scope" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_card_severity_scope=${guard_session_residue_card_severity_scope}"
      reconciliation_fingerprint+=";historical_failed_session_residue_card_severity_scope=${guard_session_residue_card_severity_scope}"
    fi
    if [[ -n "$guard_session_residue_card_severity_per_rail" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_card_severity_per_rail=${guard_session_residue_card_severity_per_rail}"
      reconciliation_fingerprint+=";historical_failed_session_residue_card_severity_per_rail=${guard_session_residue_card_severity_per_rail}"
    fi
    if [[ -n "$guard_session_residue_card_retirement_scope" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_card_retirement_scope=${guard_session_residue_card_retirement_scope}"
      reconciliation_fingerprint+=";historical_failed_session_residue_card_retirement_scope=${guard_session_residue_card_retirement_scope}"
    fi
    if [[ -n "$guard_session_residue_card_retirement_per_rail" ]]; then
      reconciliation_summary+="; historical_failed_session_residue_card_retirement_per_rail=${guard_session_residue_card_retirement_per_rail}"
      reconciliation_fingerprint+=";historical_failed_session_residue_card_retirement_per_rail=${guard_session_residue_card_retirement_per_rail}"
    fi
    emit_operator_event \
      "cron_session_surface_reconciliation_projected" \
      "info" \
      "$reconciliation_summary" \
      "$GUARD_EVIDENCE_REF" \
      "$CRON_SESSION_SURFACE_RECONCILIATION_COOLDOWN_SEC" \
      "$reconciliation_fingerprint"

    if [[ -n "$guard_session_residue_card_resolved_historical_per_rail" ]]; then
      emit_operator_event \
        "cron_session_surface_reconciliation_resolved_historical_cards" \
        "info" \
        "cron_session_surface_reconciliation_resolved_historical_cards; historical_failed_session_residue_names=${guard_session_residue_names}; historical_failed_session_residue_card_retirement_scope=${guard_session_residue_card_retirement_scope:-projected}; historical_failed_session_residue_card_resolved_historical_per_rail=${guard_session_residue_card_resolved_historical_per_rail}; cron_session_card_lifecycle_reconcile_rollup=${reconcile_rollup}; cron_session_card_lifecycle_reconcile_trace_path=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TRACE_PATH}; authority_runtime_status=healthy" \
        "$GUARD_EVIDENCE_REF" \
        "$CRON_SESSION_SURFACE_RECONCILIATION_COOLDOWN_SEC" \
        "task=run_no_nudge_continuity_watchdog;key=cron_session_surface_reconciliation_resolved_historical_cards;historical_failed_session_residue_card_resolved_historical_per_rail=${guard_session_residue_card_resolved_historical_per_rail};cron_session_card_lifecycle_reconcile_rollup=${reconcile_rollup};cron_session_card_lifecycle_reconcile_trace_path=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TRACE_PATH}"
    fi

    if [[ -n "$guard_session_residue_card_retired_per_rail" ]]; then
      emit_operator_event \
        "cron_session_surface_reconciliation_retired_historical_cards" \
        "info" \
        "cron_session_surface_reconciliation_retired_historical_cards; historical_failed_session_residue_names=${guard_session_residue_names}; historical_failed_session_residue_card_retirement_scope=${guard_session_residue_card_retirement_scope:-projected}; historical_failed_session_residue_card_retired_per_rail=${guard_session_residue_card_retired_per_rail}; cron_session_card_lifecycle_reconcile_rollup=${reconcile_rollup}; cron_session_card_lifecycle_reconcile_trace_path=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TRACE_PATH}; authority_runtime_status=healthy" \
        "$GUARD_EVIDENCE_REF" \
        "$CRON_SESSION_SURFACE_RECONCILIATION_COOLDOWN_SEC" \
        "task=run_no_nudge_continuity_watchdog;key=cron_session_surface_reconciliation_retired_historical_cards;historical_failed_session_residue_card_retired_per_rail=${guard_session_residue_card_retired_per_rail};cron_session_card_lifecycle_reconcile_rollup=${reconcile_rollup};cron_session_card_lifecycle_reconcile_trace_path=${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_TRACE_PATH}"
    fi
  fi
fi

if [[ "$guard_line" == READY:* && "$guard_line" == *"historical_failed_session_residue_state=unknown_status_metadata_unavailable"* ]]; then
  emit_operator_event \
    "cron_session_surface_reconciliation_status_unknown" \
    "info" \
    "cron_session_surface_reconciliation_status_unknown; historical_failed_session_residue_state=unknown_status_metadata_unavailable; authority_runtime_status=healthy" \
    "$GUARD_EVIDENCE_REF" \
    "$CRON_SESSION_SURFACE_RECONCILIATION_COOLDOWN_SEC" \
    "task=run_no_nudge_continuity_watchdog;key=cron_session_surface_reconciliation_status_unknown;historical_failed_session_residue_state=unknown_status_metadata_unavailable"
fi

if [[ "$guard_line" == READY:* && "$guard_line" == *"historical_failed_session_residue_state=none_observed"* && "$guard_line" == *"historical_failed_session_residue_clear_state=reconciled_now"* ]]; then
  if [[ "$(session_surface_reconciliation_clear_transition_pending)" == "1" ]]; then
    emit_operator_event \
      "cron_session_surface_reconciliation_cleared" \
      "info" \
      "cron_session_surface_reconciliation_cleared; historical_failed_session_residue_state=none_observed; historical_failed_session_residue_clear_state=reconciled_now; authority_runtime_status=healthy" \
      "$GUARD_EVIDENCE_REF" \
      "$CRON_SESSION_SURFACE_RECONCILIATION_COOLDOWN_SEC" \
      "task=run_no_nudge_continuity_watchdog;key=cron_session_surface_reconciliation_cleared;historical_failed_session_residue_state=none_observed;historical_failed_session_residue_clear_state=reconciled_now"
  fi
fi

project_now_payload() {
  local payload_json="${1:-}"
  python3 - "$payload_json" "$ROOT" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

obj = json.loads(sys.argv[1])
root = pathlib.Path(sys.argv[2]).resolve()
verify_status = str(((obj.get("verify") or {}).get("status") or "UNKNOWN")).upper()

sys.path.insert(0, str((root / "ops" / "openclaw" / "continuity").resolve()))
try:
    from continuity_policy import DRIFT_REASON_SET as _DRIFT_REASON_SET
except Exception:
    _DRIFT_REASON_SET = {
        "pointer_drift",
        "ground_truth_capture_drift",
        "connector_freshness_drift",
        "policy_freshness_drift",
    }

def normalize_reasons(values):
    unique = []
    seen = set()
    for raw in (values or []):
        reason = str(raw).strip()
        if not reason or reason in seen:
            continue
        seen.add(reason)
        unique.append(reason)
    unique.sort()
    return unique

not_ready_reasons = normalize_reasons(obj.get("not_ready_reasons") or [])
warning_reasons = normalize_reasons(obj.get("warning_reasons") or [])
raw_blocker_reasons = normalize_reasons(obj.get("blocker_reasons") or [])
drift_reasons = set(_DRIFT_REASON_SET)
if raw_blocker_reasons:
    blocker_reasons = raw_blocker_reasons
else:
    blocker_reasons = normalize_reasons([reason for reason in not_ready_reasons if reason not in drift_reasons])
generated_at = obj.get("generated_at")

age_sec = None
if isinstance(generated_at, str) and generated_at.strip():
    try:
        dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        age_sec = int(max(0.0, (datetime.now(timezone.utc) - dt).total_seconds()))
    except Exception:
        age_sec = None

print(verify_status)
print(json.dumps(not_ready_reasons, ensure_ascii=False))
print(json.dumps(blocker_reasons, ensure_ascii=False))
print(json.dumps(warning_reasons, ensure_ascii=False))
print("" if age_sec is None else str(age_sec))
PY
}

freshness_repair_flags() {
  local not_ready_json="${1:-[]}"
  python3 - "$not_ready_json" "$ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[2]).resolve()
sys.path.insert(0, str((root / "ops" / "openclaw" / "continuity").resolve()))
try:
    from continuity_policy import DRIFT_REASON_SET as _DRIFT_REASON_SET
except Exception:
    _DRIFT_REASON_SET = {
        "pointer_drift",
        "ground_truth_capture_drift",
        "connector_freshness_drift",
        "policy_freshness_drift",
    }

drift_reasons = set(_DRIFT_REASON_SET)
verify_stale_tokens = {"verify_status_evidence_stale"}

try:
    raw = json.loads(sys.argv[1])
except Exception:
    raw = []

if not isinstance(raw, list):
    raw = []

reasons = {str(item).strip() for item in raw if str(item).strip()}
print("1" if reasons.intersection(verify_stale_tokens) else "0")
print("1" if reasons.intersection(drift_reasons) else "0")
PY
}

extract_action_token() {
  local payload_json="${1:-}"
  python3 - "$payload_json" <<'PY'
import json
import sys

try:
    obj = json.loads(sys.argv[1])
except Exception:
    obj = {}

if isinstance(obj, dict):
    print(str(obj.get("action_token") or "").strip())
else:
    print("")
PY
}

run_execution_frontier_controller_tick() {
  python3 - "$ROOT" "$CONTINUITY_DISPATCH" "$CURRENT_SCRIPT" "$EXECUTION_FRONTIER_LEDGER_PATH" "$EXECUTION_FRONTIER_CONTROLLER_ENABLED" "$EXECUTION_FRONTIER_CONTROLLER_REASON" "$EXECUTION_FRONTIER_CONTROLLER_TRACE_PATH" "$EXECUTION_FRONTIER_CONTROLLER_HISTORY_PATH" "$EXECUTION_FRONTIER_ENFORCEMENT_LATCH_PATH" "$EXECUTION_FRONTIER_ENFORCEMENT_LATCH_HISTORY_PATH" "$AUTONOMOUS_EXECUTION_INTENT_PATH" "$AUTONOMOUS_EXECUTION_INTENT_HISTORY_PATH" "$EXECUTION_FRONTIER_CONTROLLER_COOLDOWN_AFTER" "$EXECUTION_FRONTIER_CONTROLLER_COOLDOWN_SEC" "$EXECUTION_FRONTIER_CONTROLLER_RETRY_BUDGET" <<'PY'
import datetime as dt
import json
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Optional

root = pathlib.Path(sys.argv[1]).resolve()
continuity_dispatch = pathlib.Path(sys.argv[2]).resolve()
current_script = pathlib.Path(sys.argv[3]).resolve()
frontier_ledger_path = pathlib.Path(sys.argv[4]).resolve()
controller_enabled_raw = str(sys.argv[5] or "1").strip().lower()
controller_reason = str(sys.argv[6] or "watchdog_no_nudge_controller_tick").strip() or "watchdog_no_nudge_controller_tick"
trace_path = pathlib.Path(sys.argv[7]).resolve()
history_path = pathlib.Path(sys.argv[8]).resolve()
latch_path = pathlib.Path(sys.argv[9]).resolve()
latch_history_path = pathlib.Path(sys.argv[10]).resolve()
intent_path = pathlib.Path(sys.argv[11]).resolve()
intent_history_path = pathlib.Path(sys.argv[12]).resolve()
try:
    cooldown_after = max(1, int(sys.argv[13]))
except Exception:
    cooldown_after = 3
try:
    cooldown_sec = max(0, int(sys.argv[14]))
except Exception:
    cooldown_sec = 900
try:
    retry_budget = max(0, int(sys.argv[15]))
except Exception:
    retry_budget = 1


def now_dt() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_from_dt(value: dt.datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def now_iso() -> str:
    return iso_from_dt(now_dt())


def truthy(raw: str) -> bool:
    return raw not in {"", "0", "false", "off", "no", "disabled"}


def dedupe(values: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in (values or []):
        txt = str(raw or "").strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


def to_nonnegative_int(value: Any, *, default: int = 0) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return max(0, parsed)


def summarize_frontier_queue(frontier_queue_obj: Any) -> Dict[str, Any]:
    queue_obj = frontier_queue_obj if isinstance(frontier_queue_obj, dict) else {}

    ready_rows = queue_obj.get("ready_candidates") if isinstance(queue_obj.get("ready_candidates"), list) else []
    blocked_rows = (
        queue_obj.get("dependency_blocked_candidates")
        if isinstance(queue_obj.get("dependency_blocked_candidates"), list)
        else []
    )

    ready_candidate_ids = dedupe(
        [
            str((row.get("task_id") if isinstance(row, dict) else row) or "").strip()
            for row in ready_rows
            if str((row.get("task_id") if isinstance(row, dict) else row) or "").strip()
        ]
    )
    blocked_candidate_ids = dedupe(
        [
            str((row.get("task_id") if isinstance(row, dict) else row) or "").strip()
            for row in blocked_rows
            if str((row.get("task_id") if isinstance(row, dict) else row) or "").strip()
        ]
    )

    next_candidates = dedupe(
        (queue_obj.get("next_candidates") if isinstance(queue_obj.get("next_candidates"), list) else [])
        + ready_candidate_ids
    )

    return {
        "present": bool(queue_obj),
        "ready_count": max(to_nonnegative_int(queue_obj.get("ready_count"), default=0), len(ready_candidate_ids)),
        "dependency_blocked_count": max(
            to_nonnegative_int(queue_obj.get("dependency_blocked_count"), default=0),
            len(blocked_candidate_ids),
        ),
        "ready_candidate_ids": ready_candidate_ids,
        "dependency_blocked_candidate_ids": blocked_candidate_ids,
        "next_candidates": next_candidates,
    }


def requires_post_completion_enforcement(
    *,
    selector_state: Optional[str],
    close_condition_met: Optional[bool],
    next_candidate: Optional[str],
    frontier_queue: Optional[Dict[str, Any]],
    block_reasons: Optional[List[str]] = None,
) -> bool:
    if close_condition_met is not True:
        return False

    selector = str(selector_state or "").strip()
    queue_obj = frontier_queue if isinstance(frontier_queue, dict) else {}
    ready_count = to_nonnegative_int(queue_obj.get("ready_count"), default=0)
    dependency_blocked_count = to_nonnegative_int(queue_obj.get("dependency_blocked_count"), default=0)
    next_candidate_txt = str(next_candidate or "").strip()
    reason_set = {str(item or "").strip() for item in (block_reasons or []) if str(item or "").strip()}
    dependency_blocked_signal = bool(
        dependency_blocked_count > 0
        or "frontier_queue_only_dependency_blocked_candidates" in reason_set
        or "next_candidate_dependency_blocked" in reason_set
    )

    if selector == "ready_for_dispatch":
        return bool(next_candidate_txt or ready_count > 0)
    if selector in {"closed_blocked", "idle_no_candidate"}:
        return bool(next_candidate_txt or ready_count > 0 or dependency_blocked_signal)
    return False


def write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(raw or "")
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except Exception:
        return str(path)


def parse_iso(raw: Any) -> Optional[dt.datetime]:
    txt = str(raw or "").strip()
    if not txt:
        return None
    normalized = txt.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def derive_loop_state(*, required: bool, status: str, selector_state: Optional[str], block_reasons: List[str]) -> Optional[str]:
    if not required:
        return None
    if status == "applied":
        return "CLEAR"
    reason_set = {str(r).strip() for r in block_reasons if str(r).strip()}
    if (
        "stalled_detection_active" in reason_set
        or "post_completion_no_next_candidate" in reason_set
        or str(selector_state or "").strip() == "idle_no_candidate"
    ):
        return "STALLED_LOOP"
    return "BLOCKED_LOOP"


controller_enabled = truthy(controller_enabled_raw)
recorded = now_dt()
recorded_iso = iso_from_dt(recorded)
recorded_ts = int(recorded.timestamp())

previous_latch = load_json(latch_path)
previous_retry_contract = (
    previous_latch.get("retry_contract") if isinstance(previous_latch.get("retry_contract"), dict) else {}
)
previous_latch_active = bool(previous_latch.get("latched") is True)
previous_loop_state = str(previous_latch.get("loop_state") or "").strip() or None
previous_first_seen_at = str(previous_latch.get("first_seen_at") or "").strip() or None
previous_cooldown_until_ts = to_nonnegative_int(previous_latch.get("cooldown_until_ts"), default=0)
active_cooldown_remaining_sec = 0
if previous_cooldown_until_ts > recorded_ts:
    active_cooldown_remaining_sec = previous_cooldown_until_ts - recorded_ts

result: Dict[str, Any] = {
    "schema": "claw.no_nudge_execution_frontier_controller_tick.v1",
    "recorded_at": recorded_iso,
    "recorded_ts": recorded_ts,
    "controller_enabled": controller_enabled,
    "controller_reason": controller_reason,
    "status": "skipped",
    "decision": "SKIP",
    "skip_reason": None,
    "block_reason": None,
    "block_reasons": [],
    "error": None,
    "action_token_present": False,
    "execution_frontier": {
        "source_present": False,
        "selector_state": None,
        "close_condition_met": None,
        "next_candidate": None,
        "next_candidate_wave": None,
        "next_candidate_source": None,
        "next_candidate_resolution": None,
        "dispatch_next_candidate": None,
        "supervisor_state": None,
        "autonomous_dispatch_eligible": None,
        "autonomous_dispatch_block_reasons": [],
        "frontier_queue": {
            "present": False,
            "ready_count": 0,
            "dependency_blocked_count": 0,
            "ready_candidate_ids": [],
            "dependency_blocked_candidate_ids": [],
            "next_candidates": [],
        },
        "ledger_path": rel(frontier_ledger_path),
    },
    "dispatch_attempt": {
        "executed": False,
        "returncode": None,
        "decision": None,
        "advance_applied": False,
        "block_reason": None,
        "block_reasons": [],
        "error": None,
        "attempt_evidence": None,
    },
    "trace_path": rel(trace_path),
    "history_path": rel(history_path),
}

if not controller_enabled:
    result["skip_reason"] = "controller_disabled"
elif not continuity_dispatch.exists() or not continuity_dispatch.is_file() or not continuity_dispatch.stat().st_mode & 0o111:
    result["skip_reason"] = "continuity_dispatch_unavailable"
else:
    show_cp = subprocess.run(
        [str(continuity_dispatch), "execution-frontier", "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    show_payload = parse_json_object(show_cp.stdout)
    if show_payload is None:
        result["status"] = "error"
        result["decision"] = "ERROR"
        result["error"] = "execution_frontier_show_invalid_json"
        result["dispatch_attempt"]["error"] = (show_cp.stderr or show_cp.stdout or "")[:240]
    elif show_cp.returncode != 0:
        show_error = str(show_payload.get("error") or "").strip()
        if show_error == "execution_frontier_ledger_missing":
            result["skip_reason"] = "execution_frontier_ledger_missing"
        else:
            result["status"] = "error"
            result["decision"] = "ERROR"
            result["error"] = show_error or "execution_frontier_show_failed"
            result["dispatch_attempt"]["error"] = (show_cp.stderr or show_cp.stdout or "")[:240]
    else:
        transition_obj = show_payload.get("transition") if isinstance(show_payload.get("transition"), dict) else {}
        supervisor_obj = show_payload.get("supervisor_state") if isinstance(show_payload.get("supervisor_state"), dict) else {}
        frontier_queue_obj = show_payload.get("frontier_queue") if isinstance(show_payload.get("frontier_queue"), dict) else {}
        frontier_queue_summary = summarize_frontier_queue(frontier_queue_obj)
        selector_state = str(transition_obj.get("selector_state") or "").strip() or None
        close_condition_met = transition_obj.get("close_condition_met") if isinstance(transition_obj.get("close_condition_met"), bool) else None
        next_candidate = str(show_payload.get("next_candidate") or "").strip() or None
        next_candidate_wave = show_payload.get("next_candidate_wave")
        next_candidate_source = str(show_payload.get("next_candidate_source") or "").strip() or None
        supervisor_state = str(supervisor_obj.get("state") or "").strip() or None
        autonomous_dispatch_eligible = supervisor_obj.get("autonomous_dispatch_eligible") if isinstance(supervisor_obj.get("autonomous_dispatch_eligible"), bool) else None
        autonomous_dispatch_block_reasons = dedupe(supervisor_obj.get("autonomous_dispatch_block_reasons") or [])

        post_completion_enforcement_required = requires_post_completion_enforcement(
            selector_state=selector_state,
            close_condition_met=close_condition_met,
            next_candidate=next_candidate,
            frontier_queue=frontier_queue_summary,
            block_reasons=autonomous_dispatch_block_reasons,
        )

        result["execution_frontier"] = {
            "source_present": True,
            "selector_state": selector_state,
            "close_condition_met": close_condition_met,
            "next_candidate": next_candidate,
            "next_candidate_wave": next_candidate_wave,
            "next_candidate_source": next_candidate_source,
            "next_candidate_resolution": None,
            "dispatch_next_candidate": None,
            "supervisor_state": supervisor_state,
            "autonomous_dispatch_eligible": autonomous_dispatch_eligible,
            "autonomous_dispatch_block_reasons": autonomous_dispatch_block_reasons,
            "post_completion_enforcement_required": post_completion_enforcement_required,
            "frontier_queue": frontier_queue_summary,
            "ledger_path": rel(frontier_ledger_path),
        }

        needs_dispatch_attempt = autonomous_dispatch_eligible is True
        if post_completion_enforcement_required and selector_state == "ready_for_dispatch":
            # Hard post-completion rule: once a wave closes with a selected frontier candidate,
            # we must attempt canonical autonomous dispatch immediately and fail closed on block.
            needs_dispatch_attempt = True

        if post_completion_enforcement_required and active_cooldown_remaining_sec > 0:
            result["status"] = "blocked"
            result["decision"] = "BLOCK"
            result["block_reason"] = "controller_cooldown_active"
            result["block_reasons"] = ["controller_cooldown_active"]
            result["skip_reason"] = "controller_cooldown_active"
        elif not needs_dispatch_attempt:
            if post_completion_enforcement_required:
                derived_block_reasons = dedupe(autonomous_dispatch_block_reasons)
                if not derived_block_reasons:
                    if selector_state == "closed_blocked":
                        derived_block_reasons = ["post_completion_closed_blocked"]
                    elif selector_state == "idle_no_candidate":
                        derived_block_reasons = ["post_completion_no_next_candidate"]
                    else:
                        derived_block_reasons = ["post_completion_dispatch_blocked"]
                result["status"] = "blocked"
                result["decision"] = "BLOCK"
                result["block_reason"] = derived_block_reasons[0]
                result["block_reasons"] = derived_block_reasons
            else:
                result["skip_reason"] = "autonomous_dispatch_not_eligible"
                result["block_reasons"] = autonomous_dispatch_block_reasons
        else:
            current_cp = subprocess.run(
                [str(current_script), "--json"],
                text=True,
                capture_output=True,
                check=False,
            )
            current_payload = parse_json_object(current_cp.stdout)
            if current_cp.returncode != 0 or current_payload is None:
                result["status"] = "error"
                result["decision"] = "ERROR"
                result["error"] = "continuity_current_token_probe_failed"
                result["dispatch_attempt"]["error"] = (current_cp.stderr or current_cp.stdout or "")[:240]
            else:
                action_token = str(current_payload.get("action_token") or "").strip()
                if not action_token:
                    result["status"] = "blocked"
                    result["decision"] = "BLOCK"
                    result["block_reason"] = "continuity_action_token_missing"
                    result["block_reasons"] = ["continuity_action_token_missing"]
                else:
                    result["action_token_present"] = True
                    dispatch_cp = subprocess.run(
                        [
                            str(continuity_dispatch),
                            "--action-token",
                            action_token,
                            "execution-frontier",
                            "supervisor-autonomous-dispatch",
                            "--reason",
                            controller_reason,
                            "--json",
                        ],
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    dispatch_payload = parse_json_object(dispatch_cp.stdout)
                    dispatch_decision = str((dispatch_payload or {}).get("decision") or "").strip() or None
                    dispatch_applied = bool((dispatch_payload or {}).get("advance_applied") is True)
                    dispatch_block_reason = str((dispatch_payload or {}).get("block_reason") or "").strip() or None
                    dispatch_block_reasons = dedupe((dispatch_payload or {}).get("block_reasons") or [])
                    dispatch_error = str((dispatch_payload or {}).get("error") or "").strip() or None
                    dispatch_attempt_evidence = (dispatch_payload or {}).get("attempt_evidence")
                    dispatch_next_candidate = str((dispatch_payload or {}).get("next_candidate") or "").strip() or None
                    dispatch_next_candidate_resolution = (
                        str((dispatch_payload or {}).get("next_candidate_resolution") or "").strip() or None
                    )

                    result["dispatch_attempt"] = {
                        "executed": True,
                        "returncode": int(dispatch_cp.returncode),
                        "decision": dispatch_decision,
                        "advance_applied": dispatch_applied,
                        "next_candidate": dispatch_next_candidate,
                        "next_candidate_resolution": dispatch_next_candidate_resolution,
                        "block_reason": dispatch_block_reason,
                        "block_reasons": dispatch_block_reasons,
                        "error": dispatch_error,
                        "attempt_evidence": dispatch_attempt_evidence,
                    }

                    execution_frontier_obj = result.get("execution_frontier") if isinstance(result.get("execution_frontier"), dict) else {}
                    execution_frontier_obj["dispatch_next_candidate"] = dispatch_next_candidate
                    execution_frontier_obj["next_candidate_resolution"] = dispatch_next_candidate_resolution
                    result["execution_frontier"] = execution_frontier_obj

                    if dispatch_cp.returncode == 0 and dispatch_decision == "APPLY" and dispatch_applied:
                        result["status"] = "applied"
                        result["decision"] = "APPLY"
                    elif dispatch_decision == "BLOCK" or dispatch_cp.returncode == 3:
                        result["status"] = "blocked"
                        result["decision"] = "BLOCK"
                        result["block_reason"] = dispatch_block_reason or "execution_frontier_transition_blocked"
                        result["block_reasons"] = dispatch_block_reasons or [result["block_reason"]]
                        if dispatch_error:
                            result["error"] = dispatch_error
                    else:
                        result["status"] = "error"
                        result["decision"] = "ERROR"
                        result["error"] = dispatch_error or "execution_frontier_dispatch_failed"
                        if dispatch_payload is None:
                            result["dispatch_attempt"]["error"] = (dispatch_cp.stderr or dispatch_cp.stdout or "")[:240]

result["block_reasons"] = dedupe(result.get("block_reasons") or [])

execution_frontier_obj = result.get("execution_frontier") if isinstance(result.get("execution_frontier"), dict) else {}
execution_frontier_source_present = bool(execution_frontier_obj.get("source_present") is True)
post_completion_enforcement_required = bool(execution_frontier_obj.get("post_completion_enforcement_required") is True)
selector_state = str(execution_frontier_obj.get("selector_state") or "").strip() or None

latch_active = bool(post_completion_enforcement_required and result.get("status") != "applied")
loop_state = derive_loop_state(
    required=post_completion_enforcement_required,
    status=str(result.get("status") or "missing"),
    selector_state=selector_state,
    block_reasons=list(result.get("block_reasons") or []),
)
carry_forward_from_previous = False
if not execution_frontier_source_present and previous_latch_active:
    latch_active = True
    carry_forward_from_previous = True
    if not loop_state:
        loop_state = previous_loop_state or "BLOCKED_LOOP"

if latch_active:
    first_seen_at = previous_first_seen_at if previous_latch_active and previous_first_seen_at else recorded_iso
else:
    first_seen_at = None

if latch_active:
    consecutive_latched_ticks = to_nonnegative_int(previous_latch.get("consecutive_latched_ticks"), default=0) + (1 if previous_latch_active else 1)
else:
    consecutive_latched_ticks = 0

latest_status = str(result.get("status") or "missing")
if latest_status == "blocked":
    blocked_streak = to_nonnegative_int(previous_latch.get("blocked_streak"), default=0) + 1
else:
    blocked_streak = 0
if latest_status == "error":
    error_streak = to_nonnegative_int(previous_latch.get("error_streak"), default=0) + 1
else:
    error_streak = 0

retry_attempts = 0
if latest_status == "error":
    retry_attempts = to_nonnegative_int(previous_retry_contract.get("attempts"), default=0) + 1
retry_exhausted = bool(latest_status == "error" and retry_attempts > retry_budget)
retry_state = "idle"
if latest_status == "error":
    retry_state = "retry_exhausted" if retry_exhausted else "retry_scheduled"

retry_contract = {
    "schema_version": "continuity.execution_frontier_post_completion_retry_contract.v1",
    "policy": "single_rerun_then_block",
    "max_attempts": retry_budget,
    "attempts": retry_attempts,
    "state": retry_state,
    "retry_due": bool(latest_status == "error" and not retry_exhausted),
    "retry_exhausted": retry_exhausted,
    "error": str(result.get("error") or "") if latest_status == "error" else "",
}

parity_issues: List[str] = []
if post_completion_enforcement_required and result.get("status") == "skipped":
    parity_issues.append("post_completion_required_but_skipped")
if result.get("status") == "applied" and bool((result.get("dispatch_attempt") or {}).get("advance_applied") is not True):
    parity_issues.append("applied_without_dispatch_advance")
if (
    str((result.get("dispatch_attempt") or {}).get("decision") or "").strip() == "APPLY"
    and result.get("status") != "applied"
):
    parity_issues.append("dispatch_apply_decision_without_applied_status")

loop_status_for_cooldown = str(result.get("status") or "")
cooldown_triggered = False
cooldown_until_ts = previous_cooldown_until_ts if previous_cooldown_until_ts > recorded_ts else 0
if loop_status_for_cooldown in {"blocked", "error"} and latch_active and cooldown_sec > 0:
    repeated_count = max(blocked_streak, error_streak)
    if repeated_count >= cooldown_after and cooldown_until_ts <= recorded_ts:
        cooldown_until_ts = recorded_ts + cooldown_sec
        cooldown_triggered = True

if not latch_active:
    cooldown_until_ts = 0
    cooldown_triggered = False

cooldown_active = cooldown_until_ts > recorded_ts
cooldown_remaining_sec = max(0, cooldown_until_ts - recorded_ts) if cooldown_active else 0
cooldown_until_iso = iso_from_dt(dt.datetime.fromtimestamp(cooldown_until_ts, tz=dt.timezone.utc)) if cooldown_active else None

if retry_exhausted:
    exhausted_reason = "execution_frontier_retry_budget_exhausted"
    reasons = dedupe(list(result.get("block_reasons") or []) + [exhausted_reason])
    result["status"] = "blocked"
    result["decision"] = "BLOCK"
    result["block_reason"] = exhausted_reason
    result["block_reasons"] = reasons

loop_state = derive_loop_state(
    required=post_completion_enforcement_required or carry_forward_from_previous,
    status=str(result.get("status") or "missing"),
    selector_state=selector_state,
    block_reasons=list(result.get("block_reasons") or []),
)
if carry_forward_from_previous and not loop_state:
    loop_state = previous_loop_state or "BLOCKED_LOOP"

latch_payload: Dict[str, Any] = {
    "schema": "clawd.execution_frontier_post_completion_enforcement_latch.v1",
    "recorded_at": recorded_iso,
    "recorded_ts": recorded_ts,
    "latched": bool(latch_active),
    "loop_state": loop_state,
    "first_seen_at": first_seen_at,
    "last_seen_at": recorded_iso if latch_active else None,
    "last_status": str(result.get("status") or "missing"),
    "last_decision": str(result.get("decision") or "SKIP"),
    "last_skip_reason": str(result.get("skip_reason") or ""),
    "last_block_reason": str(result.get("block_reason") or ""),
    "last_block_reasons": dedupe(result.get("block_reasons") or []),
    "last_error": str(result.get("error") or ""),
    "consecutive_latched_ticks": int(consecutive_latched_ticks),
    "blocked_streak": int(blocked_streak),
    "error_streak": int(error_streak),
    "post_completion_enforcement_required": bool(post_completion_enforcement_required),
    "selector_state": selector_state,
    "close_condition_met": execution_frontier_obj.get("close_condition_met") if isinstance(execution_frontier_obj.get("close_condition_met"), bool) else None,
    "next_candidate": str(execution_frontier_obj.get("next_candidate") or "") or None,
    "next_candidate_wave": execution_frontier_obj.get("next_candidate_wave") if isinstance(execution_frontier_obj.get("next_candidate_wave"), int) else None,
    "frontier_queue": execution_frontier_obj.get("frontier_queue") if isinstance(execution_frontier_obj.get("frontier_queue"), dict) else {},
    "carry_forward_from_previous": carry_forward_from_previous,
    "source_degraded": not execution_frontier_source_present,
    "retry_contract": retry_contract,
    "cooldown_until_ts": int(cooldown_until_ts) if cooldown_active else None,
    "cooldown_until_iso": cooldown_until_iso,
    "cooldown_policy": {
        "schema_version": "continuity.execution_frontier_post_completion_cooldown_policy.v1",
        "threshold": int(cooldown_after),
        "cooldown_sec": int(cooldown_sec),
        "active": bool(cooldown_active),
        "remaining_sec": int(cooldown_remaining_sec),
        "until_ts": int(cooldown_until_ts) if cooldown_active else None,
        "until_iso": cooldown_until_iso,
        "triggered_this_tick": bool(cooldown_triggered),
    },
    "queue_truth_vs_narrative_parity": {
        "schema_version": "continuity.execution_frontier_queue_truth_vs_narrative_parity.v1",
        "status": "ok" if not parity_issues else "mismatch",
        "issues": parity_issues,
    },
    "trace_path": rel(trace_path),
    "history_path": rel(history_path),
    "latch_path": rel(latch_path),
    "latch_history_path": rel(latch_history_path),
}

intent_payload: Dict[str, Any] = {
    "schema": "clawd.autonomous_execution_intent.v1",
    "recorded_at": recorded_iso,
    "recorded_ts": recorded_ts,
    "active": bool(latch_payload.get("latched") is True),
    "status": str(result.get("status") or "missing"),
    "decision": str(result.get("decision") or "SKIP"),
    "post_completion_enforcement_required": bool(post_completion_enforcement_required),
    "post_completion_enforcement_latched": bool(latch_payload.get("latched") is True),
    "loop_state": loop_state,
    "queue_position": {
        "selector_state": selector_state,
        "close_condition_met": execution_frontier_obj.get("close_condition_met") if isinstance(execution_frontier_obj.get("close_condition_met"), bool) else None,
        "next_candidate": str(execution_frontier_obj.get("next_candidate") or "") or None,
        "next_candidate_wave": execution_frontier_obj.get("next_candidate_wave") if isinstance(execution_frontier_obj.get("next_candidate_wave"), int) else None,
        "next_candidate_source": str(execution_frontier_obj.get("next_candidate_source") or "") or None,
        "ready_count": to_nonnegative_int(((execution_frontier_obj.get("frontier_queue") if isinstance(execution_frontier_obj.get("frontier_queue"), dict) else {}).get("ready_count")), default=0),
        "dependency_blocked_count": to_nonnegative_int(((execution_frontier_obj.get("frontier_queue") if isinstance(execution_frontier_obj.get("frontier_queue"), dict) else {}).get("dependency_blocked_count")), default=0),
    },
    "block_reason": str(result.get("block_reason") or ""),
    "block_reasons": dedupe(result.get("block_reasons") or []),
    "error": str(result.get("error") or ""),
    "retry_contract": retry_contract,
    "cooldown": dict(latch_payload.get("cooldown_policy") or {}),
    "parity": dict(latch_payload.get("queue_truth_vs_narrative_parity") or {}),
    "trace_path": rel(trace_path),
    "history_path": rel(history_path),
    "latch_path": rel(latch_path),
    "latch_history_path": rel(latch_history_path),
    "intent_path": rel(intent_path),
    "intent_history_path": rel(intent_history_path),
}

result["post_completion_latch"] = {
    "latched": bool(latch_payload.get("latched") is True),
    "loop_state": latch_payload.get("loop_state"),
    "first_seen_at": latch_payload.get("first_seen_at"),
    "blocked_streak": int(latch_payload.get("blocked_streak") or 0),
    "error_streak": int(latch_payload.get("error_streak") or 0),
    "retry_contract": retry_contract,
    "cooldown_policy": dict(latch_payload.get("cooldown_policy") or {}),
    "queue_truth_vs_narrative_parity": dict(latch_payload.get("queue_truth_vs_narrative_parity") or {}),
    "latch_path": rel(latch_path),
    "latch_history_path": rel(latch_history_path),
}
result["autonomous_execution_intent"] = {
    "active": bool(intent_payload.get("active") is True),
    "status": str(intent_payload.get("status") or "missing"),
    "loop_state": intent_payload.get("loop_state"),
    "intent_path": rel(intent_path),
    "intent_history_path": rel(intent_history_path),
}

write_json(trace_path, result)
append_jsonl(history_path, result)
write_json(latch_path, latch_payload)
append_jsonl(latch_history_path, latch_payload)
write_json(intent_path, intent_payload)
append_jsonl(intent_history_path, intent_payload)
print(json.dumps(result, ensure_ascii=False))
PY
}

set +e
now_json="$($NOW_SCRIPT --json 2>/tmp/run_no_nudge_continuity_watchdog_now.err)"
now_rc=$?
set -e
if [[ "$now_rc" -ne 0 ]]; then
  err="$(cat /tmp/run_no_nudge_continuity_watchdog_now.err 2>/dev/null || true)"
  emit_blocker "continuity_now_failed" "continuity_now_failed; err=${err:0:180}" "$NOW_SCRIPT"
  exit 0
fi

set +e
parsed_output="$(project_now_payload "$now_json" 2>/tmp/run_no_nudge_continuity_watchdog_now_parse.err)"
parsed_rc=$?
set -e
if [[ "$parsed_rc" -ne 0 ]]; then
  err="$(cat /tmp/run_no_nudge_continuity_watchdog_now_parse.err 2>/dev/null || true)"
  emit_blocker "continuity_now_payload_invalid_json" "continuity_now_payload_invalid_json; err=${err:0:180}" "$NOW_SCRIPT"
  exit 0
fi

readarray -t parsed <<<"$parsed_output"

verify_status="${parsed[0]:-UNKNOWN}"
not_ready_json="${parsed[1]:-[]}"
blocker_json="${parsed[2]:-[]}"
warning_json="${parsed[3]:-[]}"
age_sec_raw="${parsed[4]:-}"

restore_drill_refresh_suffix=""

if [[ -z "$age_sec_raw" ]]; then
  emit_blocker "continuity_now_missing_generated_at" "continuity_now_missing_generated_at" "$NOW_SCRIPT"
  exit 0
fi

age_sec=$((age_sec_raw))

refreshed=0
if (( age_sec > REFRESH_AFTER_SEC )); then
  set +e
  refreshed_now_json="$($NOW_SCRIPT --refresh --json 2>/tmp/run_no_nudge_continuity_watchdog_refresh.err)"
  refresh_rc=$?
  set -e
  if [[ "$refresh_rc" -ne 0 ]]; then
    err="$(cat /tmp/run_no_nudge_continuity_watchdog_refresh.err 2>/dev/null || true)"
    emit_blocker "continuity_refresh_failed" "continuity_refresh_failed; err=${err:0:180}" "$NOW_SCRIPT"
    exit 0
  fi

  # Keep successor surfaces aligned after refresh.
  "$CURRENT_SCRIPT" --json >/dev/null 2>/tmp/run_no_nudge_continuity_watchdog_current.err || {
    err="$(cat /tmp/run_no_nudge_continuity_watchdog_current.err 2>/dev/null || true)"
    emit_blocker "continuity_current_refresh_failed" "continuity_current_refresh_failed; err=${err:0:180}" "$CURRENT_SCRIPT"
    exit 0
  }

  "$HANDOVER_SCRIPT" --refresh --json >/dev/null 2>/tmp/run_no_nudge_continuity_watchdog_handover.err || {
    err="$(cat /tmp/run_no_nudge_continuity_watchdog_handover.err 2>/dev/null || true)"
    emit_blocker "handover_refresh_failed" "handover_refresh_failed; err=${err:0:180}" "$HANDOVER_SCRIPT"
    exit 0
  }

  set +e
  age_sec="$(python3 - "$refreshed_now_json" 2>/tmp/run_no_nudge_continuity_watchdog_refresh_parse.err <<'PY'
import json
import sys
from datetime import datetime, timezone
obj = json.loads(sys.argv[1])
generated_at = obj.get("generated_at")
if not isinstance(generated_at, str) or not generated_at.strip():
    print("")
    raise SystemExit(0)
try:
    dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    print(int(max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())))
except Exception:
    print("")
PY
)"
  age_parse_rc=$?
  set -e
  if [[ "$age_parse_rc" -ne 0 ]]; then
    err="$(cat /tmp/run_no_nudge_continuity_watchdog_refresh_parse.err 2>/dev/null || true)"
    emit_blocker "continuity_refresh_payload_invalid_json" "continuity_refresh_payload_invalid_json; err=${err:0:180}" "$NOW_SCRIPT"
    exit 0
  fi

  if [[ -z "$age_sec" ]]; then
    emit_blocker "continuity_refresh_missing_generated_at" "continuity_refresh_missing_generated_at" "$NOW_SCRIPT"
    exit 0
  fi

  refreshed=1
  set +e
  refreshed_parsed_output="$(project_now_payload "$refreshed_now_json" 2>/tmp/run_no_nudge_continuity_watchdog_refresh_projection.err)"
  refreshed_parse_rc=$?
  set -e
  if [[ "$refreshed_parse_rc" -ne 0 ]]; then
    err="$(cat /tmp/run_no_nudge_continuity_watchdog_refresh_projection.err 2>/dev/null || true)"
    emit_blocker "continuity_refresh_payload_invalid_json" "continuity_refresh_payload_invalid_json; err=${err:0:180}" "$NOW_SCRIPT"
    exit 0
  fi
  readarray -t refreshed_parsed <<<"$refreshed_parsed_output"
  verify_status="${refreshed_parsed[0]:-UNKNOWN}"
  not_ready_json="${refreshed_parsed[1]:-[]}"
  blocker_json="${refreshed_parsed[2]:-[]}"
  warning_json="${refreshed_parsed[3]:-[]}"
fi

if (( age_sec > MAX_AGE_SEC )); then
  emit_blocker "continuity_surface_stale_after_refresh" "continuity_surface_stale_after_refresh; age_sec=${age_sec}; max_age_sec=${MAX_AGE_SEC}" "$NOW_SCRIPT"
  exit 0
fi

if [[ "$RESTORE_DRILL_AUTO_REFRESH_ENABLED" == "1" && -x "$RESTORE_DRILL_REFRESH_SCRIPT" ]]; then
  set +e
  restore_drill_refresh_json="$(OPENCLAW_RESTORE_DRILL_REFRESH_AFTER_SEC="$RESTORE_DRILL_REFRESH_AFTER_SEC" OPENCLAW_SLO_RESTORE_DRILL_MAX_AGE_SEC="$RESTORE_DRILL_MAX_AGE_SEC" "$RESTORE_DRILL_REFRESH_SCRIPT" --trigger "watchdog.no_nudge_continuity" --json 2>/tmp/run_no_nudge_continuity_watchdog_restore_refresh.err)"
  restore_drill_refresh_rc=$?
  set -e
  if [[ "$restore_drill_refresh_rc" -ne 0 ]]; then
    err="$(cat /tmp/run_no_nudge_continuity_watchdog_restore_refresh.err 2>/dev/null || true)"
    emit_blocker "restore_drill_auto_refresh_failed" "restore_drill_auto_refresh_failed; err=${err:0:180}" "$RESTORE_DRILL_REFRESH_SCRIPT" "task=run_no_nudge_continuity_watchdog;key=restore_drill_auto_refresh_failed"
    exit 0
  fi

  set +e
  restore_drill_fields="$(python3 - "$restore_drill_refresh_json" 2>/tmp/run_no_nudge_continuity_watchdog_restore_refresh_parse.err <<'PY'
import json
import sys

obj = json.loads(sys.argv[1])
if not isinstance(obj, dict):
    raise SystemExit(2)

decision = str(obj.get("decision") or "").strip()
drill_status = str(obj.get("drill_status") or "").strip()
checkpoint_id = str(obj.get("checkpoint_id") or "").strip()
report_ref = str(obj.get("report_ref") or "").strip()

print(decision)
print(drill_status)
print(checkpoint_id)
print(report_ref)
PY
)"
  restore_drill_parse_rc=$?
  set -e
  if [[ "$restore_drill_parse_rc" -ne 0 ]]; then
    err="$(cat /tmp/run_no_nudge_continuity_watchdog_restore_refresh_parse.err 2>/dev/null || true)"
    emit_blocker "restore_drill_auto_refresh_invalid_json" "restore_drill_auto_refresh_invalid_json; err=${err:0:180}" "$RESTORE_DRILL_REFRESH_SCRIPT" "task=run_no_nudge_continuity_watchdog;key=restore_drill_auto_refresh_invalid_json"
    exit 0
  fi

  readarray -t restore_drill_parsed <<<"$restore_drill_fields"
  restore_drill_decision="${restore_drill_parsed[0]:-}"
  restore_drill_status="${restore_drill_parsed[1]:-}"
  restore_drill_checkpoint_id="${restore_drill_parsed[2]:-}"
  restore_drill_report_ref="${restore_drill_parsed[3]:-}"

  if [[ "$restore_drill_decision" == refreshed_* ]]; then
    set +e
    restore_refreshed_now_json="$($NOW_SCRIPT --json 2>/tmp/run_no_nudge_continuity_watchdog_restore_refresh_now.err)"
    restore_refreshed_now_rc=$?
    set -e
    if [[ "$restore_refreshed_now_rc" -ne 0 ]]; then
      err="$(cat /tmp/run_no_nudge_continuity_watchdog_restore_refresh_now.err 2>/dev/null || true)"
      emit_blocker "continuity_now_failed_after_restore_refresh" "continuity_now_failed_after_restore_refresh; err=${err:0:180}" "$NOW_SCRIPT"
      exit 0
    fi

    set +e
    restore_refreshed_parsed_output="$(project_now_payload "$restore_refreshed_now_json" 2>/tmp/run_no_nudge_continuity_watchdog_restore_refresh_now_parse.err)"
    restore_refreshed_parse_rc=$?
    set -e
    if [[ "$restore_refreshed_parse_rc" -ne 0 ]]; then
      err="$(cat /tmp/run_no_nudge_continuity_watchdog_restore_refresh_now_parse.err 2>/dev/null || true)"
      emit_blocker "continuity_now_payload_invalid_json_after_restore_refresh" "continuity_now_payload_invalid_json_after_restore_refresh; err=${err:0:180}" "$NOW_SCRIPT"
      exit 0
    fi

    readarray -t restore_refreshed_parsed <<<"$restore_refreshed_parsed_output"
    verify_status="${restore_refreshed_parsed[0]:-UNKNOWN}"
    not_ready_json="${restore_refreshed_parsed[1]:-[]}"
    blocker_json="${restore_refreshed_parsed[2]:-[]}"
    warning_json="${restore_refreshed_parsed[3]:-[]}"
    age_sec_raw="${restore_refreshed_parsed[4]:-}"
    if [[ -n "$age_sec_raw" ]]; then
      age_sec=$((age_sec_raw))
    fi

    restore_drill_refresh_suffix="; restore_drill_auto_refresh=1; restore_drill_status=${restore_drill_status:-unknown}; restore_drill_checkpoint=${restore_drill_checkpoint_id:-unknown}; restore_drill_report=${restore_drill_report_ref:-none}"
  fi
fi

if [[ "$verify_status" != "READY" ]]; then
  emit_blocker "verify_status_not_ready" "verify_status=${verify_status}; not_ready_reasons=${not_ready_json}" "$NOW_SCRIPT" "task=run_no_nudge_continuity_watchdog;key=verify_status_not_ready;status=${verify_status};reasons=${not_ready_json}"
  exit 0
fi

if [[ "$AUTO_FRESHNESS_REPAIR" == "1" && -x "$CONTINUITY_DISPATCH" ]]; then
  readarray -t freshness_flags < <(freshness_repair_flags "$not_ready_json")
  needs_verify_evidence_refresh="${freshness_flags[0]:-0}"
  needs_drift_reconcile="${freshness_flags[1]:-0}"

  if [[ "$needs_verify_evidence_refresh" == "1" || "$needs_drift_reconcile" == "1" ]]; then
    repair_action_token=""

    set +e
    current_refresh_json="$($CURRENT_SCRIPT --refresh --json 2>/tmp/run_no_nudge_continuity_watchdog_current_refresh.err)"
    current_refresh_rc=$?
    set -e
    if [[ "$current_refresh_rc" -eq 0 ]]; then
      repair_action_token="$(extract_action_token "$current_refresh_json")"
    fi

    if [[ "$needs_verify_evidence_refresh" == "1" ]]; then
      set +e
      "$CONTINUITY_DISPATCH" verify --skip-baseline-checks >/tmp/run_no_nudge_continuity_watchdog_verify_repair.out 2>/tmp/run_no_nudge_continuity_watchdog_verify_repair.err
      set -e
    fi

    if [[ "$needs_drift_reconcile" == "1" && -n "$repair_action_token" ]]; then
      set +e
      "$CONTINUITY_DISPATCH" --action-token "$repair_action_token" reconcile --json >/tmp/run_no_nudge_continuity_watchdog_reconcile_repair.out 2>/tmp/run_no_nudge_continuity_watchdog_reconcile_repair.err
      set -e
    fi

    set +e
    repaired_now_json="$($NOW_SCRIPT --json 2>/tmp/run_no_nudge_continuity_watchdog_post_repair.err)"
    repaired_now_rc=$?
    set -e
    if [[ "$repaired_now_rc" -ne 0 ]]; then
      err="$(cat /tmp/run_no_nudge_continuity_watchdog_post_repair.err 2>/dev/null || true)"
      emit_blocker "continuity_freshness_repair_failed" "continuity_freshness_repair_failed; err=${err:0:180}" "$NOW_SCRIPT"
      exit 0
    fi

    set +e
    repaired_parsed_output="$(project_now_payload "$repaired_now_json" 2>/tmp/run_no_nudge_continuity_watchdog_post_repair_parse.err)"
    repaired_parse_rc=$?
    set -e
    if [[ "$repaired_parse_rc" -ne 0 ]]; then
      err="$(cat /tmp/run_no_nudge_continuity_watchdog_post_repair_parse.err 2>/dev/null || true)"
      emit_blocker "continuity_freshness_repair_failed" "continuity_freshness_repair_failed; err=${err:0:180}" "$NOW_SCRIPT"
      exit 0
    fi

    readarray -t repaired_parsed <<<"$repaired_parsed_output"
    verify_status="${repaired_parsed[0]:-UNKNOWN}"
    not_ready_json="${repaired_parsed[1]:-[]}"
    blocker_json="${repaired_parsed[2]:-[]}"
    warning_json="${repaired_parsed[3]:-[]}"
    age_sec_raw="${repaired_parsed[4]:-}"

    if [[ -n "$age_sec_raw" ]]; then
      age_sec=$((age_sec_raw))
    fi
  fi
fi

if [[ "$verify_status" != "READY" ]]; then
  emit_blocker "verify_status_not_ready" "verify_status=${verify_status}; not_ready_reasons=${not_ready_json}" "$NOW_SCRIPT" "task=run_no_nudge_continuity_watchdog;key=verify_status_not_ready;status=${verify_status};reasons=${not_ready_json}"
  exit 0
fi

if [[ "$blocker_json" != "[]" ]]; then
  emit_blocker "continuity_not_ready" "continuity_not_ready; blocker_reasons=${blocker_json}; all_not_ready_reasons=${not_ready_json}" "$NOW_SCRIPT" "task=run_no_nudge_continuity_watchdog;key=continuity_not_ready;blocker_reasons=${blocker_json};all_reasons=${not_ready_json}"
  exit 0
fi

if [[ "$not_ready_json" != "[]" ]]; then
  echo "PROGRESS: continuity_reconcile_recommended; drift_reasons=${not_ready_json}${restore_drill_refresh_suffix}"
  exit 0
fi

set +e
execution_frontier_tick_json="$(run_execution_frontier_controller_tick 2>/tmp/run_no_nudge_continuity_watchdog_frontier_tick.err)"
execution_frontier_tick_rc=$?
set -e
if [[ "$execution_frontier_tick_rc" -ne 0 || -z "$execution_frontier_tick_json" ]]; then
  err="$(cat /tmp/run_no_nudge_continuity_watchdog_frontier_tick.err 2>/dev/null || true)"
  emit_blocker "execution_frontier_controller_tick_failed" "execution_frontier_controller_tick_failed; err=${err:0:180}" "$CONTINUITY_DISPATCH"
  exit 0
fi

set +e
execution_frontier_tick_parsed="$(python3 - "$execution_frontier_tick_json" 2>/tmp/run_no_nudge_continuity_watchdog_frontier_tick_parse.err <<'PY'
import json
import sys

obj = json.loads(sys.argv[1])
if not isinstance(obj, dict):
    raise SystemExit(3)

status = str(obj.get("status") or "").strip().lower()
skip_reason = str(obj.get("skip_reason") or "").strip()
block_reason = str(obj.get("block_reason") or "").strip()
block_reasons = obj.get("block_reasons") if isinstance(obj.get("block_reasons"), list) else []
error = str(obj.get("error") or "").strip()
trace_path = str(obj.get("trace_path") or "").strip()
history_path = str(obj.get("history_path") or "").strip()
dispatch_obj = obj.get("dispatch_attempt") if isinstance(obj.get("dispatch_attempt"), dict) else {}
dispatch_decision = str(dispatch_obj.get("decision") or "").strip()
dispatch_returncode = dispatch_obj.get("returncode")
dispatch_returncode_txt = "" if dispatch_returncode is None else str(dispatch_returncode)

print(status)
print(skip_reason)
print(block_reason)
print(json.dumps(block_reasons, ensure_ascii=False, separators=(",", ":")))
print(error)
print(trace_path)
print(history_path)
print(dispatch_decision)
print(dispatch_returncode_txt)
PY
)"
execution_frontier_tick_parse_rc=$?
set -e
if [[ "$execution_frontier_tick_parse_rc" -ne 0 ]]; then
  err="$(cat /tmp/run_no_nudge_continuity_watchdog_frontier_tick_parse.err 2>/dev/null || true)"
  emit_blocker "execution_frontier_controller_tick_invalid_json" "execution_frontier_controller_tick_invalid_json; err=${err:0:180}" "$CONTINUITY_DISPATCH"
  exit 0
fi

readarray -t execution_frontier_tick_fields <<<"$execution_frontier_tick_parsed"
execution_frontier_tick_status="${execution_frontier_tick_fields[0]:-}"
execution_frontier_tick_skip_reason="${execution_frontier_tick_fields[1]:-}"
execution_frontier_tick_block_reason="${execution_frontier_tick_fields[2]:-}"
execution_frontier_tick_block_reasons_json="${execution_frontier_tick_fields[3]:-[]}"
execution_frontier_tick_error="${execution_frontier_tick_fields[4]:-}"
execution_frontier_tick_trace_path="${execution_frontier_tick_fields[5]:-}"
execution_frontier_tick_history_path="${execution_frontier_tick_fields[6]:-}"
execution_frontier_tick_dispatch_decision="${execution_frontier_tick_fields[7]:-}"
execution_frontier_tick_dispatch_returncode="${execution_frontier_tick_fields[8]:-}"

case "$execution_frontier_tick_status" in
  applied)
    echo "PROGRESS: execution_frontier_autonomous_dispatch_applied; decision=${execution_frontier_tick_dispatch_decision:-APPLY}; returncode=${execution_frontier_tick_dispatch_returncode:-0}; trace_path=${execution_frontier_tick_trace_path}; history_path=${execution_frontier_tick_history_path}${restore_drill_refresh_suffix}"
    exit 0
    ;;
  blocked)
    if [[ -z "$execution_frontier_tick_block_reason" ]]; then
      execution_frontier_tick_block_reason="execution_frontier_transition_blocked"
    fi
    emit_blocker "execution_frontier_autonomous_dispatch_blocked" "execution_frontier_autonomous_dispatch_blocked; reason=${execution_frontier_tick_block_reason}; block_reasons=${execution_frontier_tick_block_reasons_json}; dispatch_decision=${execution_frontier_tick_dispatch_decision}; returncode=${execution_frontier_tick_dispatch_returncode}; trace_path=${execution_frontier_tick_trace_path}" "$CONTINUITY_DISPATCH" "task=run_no_nudge_continuity_watchdog;key=execution_frontier_autonomous_dispatch_blocked;reason=${execution_frontier_tick_block_reason};reasons=${execution_frontier_tick_block_reasons_json}"
    exit 0
    ;;
  error)
    if [[ -z "$execution_frontier_tick_error" ]]; then
      execution_frontier_tick_error="execution_frontier_autonomous_dispatch_error"
    fi
    emit_blocker "execution_frontier_autonomous_dispatch_error" "execution_frontier_autonomous_dispatch_error; error=${execution_frontier_tick_error}; trace_path=${execution_frontier_tick_trace_path}; returncode=${execution_frontier_tick_dispatch_returncode}" "$CONTINUITY_DISPATCH"
    exit 0
    ;;
  skipped)
    :
    ;;
  *)
    emit_blocker "execution_frontier_controller_tick_invalid_status" "execution_frontier_controller_tick_invalid_status; status=${execution_frontier_tick_status}; skip_reason=${execution_frontier_tick_skip_reason}; trace_path=${execution_frontier_tick_trace_path}" "$CONTINUITY_DISPATCH"
    exit 0
    ;;
esac

readarray -t contradiction_latch_snapshot < <(autospawn_contradiction_latch_snapshot)
autospawn_contradiction_abort_active="${contradiction_latch_snapshot[0]:-0}"
autospawn_contradiction_abort_remaining_sec="${contradiction_latch_snapshot[1]:-0}"
autospawn_contradiction_consecutive_seen="${contradiction_latch_snapshot[2]:-0}"
autospawn_contradiction_total_seen="${contradiction_latch_snapshot[3]:-0}"
autospawn_contradiction_last_kind="${contradiction_latch_snapshot[4]:-}"
autospawn_contradiction_last_reason="${contradiction_latch_snapshot[5]:-}"
autospawn_contradiction_latch_issue="${contradiction_latch_snapshot[6]:-}"
autospawn_contradiction_latch_repaired="0"
autospawn_contradiction_latch_repair_reason=""

if [[ -n "$autospawn_contradiction_latch_issue" ]]; then
  # Invalid/stale contradiction latch leases must not freeze idle-lane progress forever.
  # Reset residue so watchdog evaluates current truth on the next branch.
  clear_autospawn_contradiction_latch
  autospawn_contradiction_abort_active="0"
  autospawn_contradiction_abort_remaining_sec="0"
  autospawn_contradiction_latch_repaired="1"
  autospawn_contradiction_latch_repair_reason="$autospawn_contradiction_latch_issue"
fi

set +e
autospawn_summary_json="$(python3 - "$AUTOPILOT_STATE_FILE" "$AUTOPILOT_TICK_SCRIPT" "$IDLE_AUTOSPAWN_TRACE_PATH" "$IDLE_AUTOSPAWN_ENABLED" "$IDLE_AUTOSPAWN_IDLE_SEC" "$IDLE_AUTOSPAWN_COOLDOWN_SEC" "$IDLE_AUTOSPAWN_IMPLEMENTATION_STEP_IDS" "$autospawn_contradiction_abort_active" "$autospawn_contradiction_abort_remaining_sec" "$autospawn_contradiction_latch_repaired" "$autospawn_contradiction_latch_repair_reason" <<'PY'
import json
import os
import pathlib
import subprocess
import sys
import time

state_path = pathlib.Path(sys.argv[1])
tick_path = pathlib.Path(sys.argv[2])
trace_path = pathlib.Path(sys.argv[3])
enabled_raw = str(sys.argv[4]).strip().lower()
try:
    idle_threshold_sec = max(0, int(sys.argv[5]))
except Exception:
    idle_threshold_sec = 1800
try:
    cooldown_sec = max(0, int(sys.argv[6]))
except Exception:
    cooldown_sec = 900
impl_ids_csv = str(sys.argv[7] or "").strip()
abort_active_raw = str(sys.argv[8] or "0").strip().lower()
try:
    abort_remaining_sec = max(0, int(sys.argv[9]))
except Exception:
    abort_remaining_sec = 0
repair_marker_raw = str(sys.argv[10] or "0").strip().lower()
repair_reason_raw = str(sys.argv[11] or "").strip()

impl_step_ids = [x.strip() for x in impl_ids_csv.split(",") if x.strip()]
if not impl_step_ids:
    impl_step_ids = ["apply_fixes"]

enabled = enabled_raw not in {"", "0", "false", "off", "no", "disabled"}
contradiction_abort_active = abort_active_raw in {"1", "true", "yes", "on"} and abort_remaining_sec > 0
contradiction_latch_repaired = repair_marker_raw in {"1", "true", "yes", "on"}
contradiction_latch_repair_reason = repair_reason_raw if contradiction_latch_repaired else ""


def now_ts() -> int:
    raw = os.environ.get("OPENCLAW_AUTOPILOT_FIXED_NOW_TS")
    if raw is not None and str(raw).strip() != "":
        try:
            parsed = int(str(raw).strip())
            if parsed > 0:
                return parsed
        except Exception:
            pass
    return int(time.time())


def iso_from_ts(ts: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(ts)))


def read_json(path: pathlib.Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def first_non_empty(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def as_int(value):
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except Exception:
        return None


def step_ready(step: dict, now_value: int) -> bool:
    if str(step.get("status") or "queued") != "queued":
        return False
    next_after = step.get("next_after_ts")
    if next_after is None or str(next_after).strip() == "":
        return True
    try:
        return int(next_after) <= int(now_value)
    except Exception:
        return True


def is_impl_step(step: dict) -> bool:
    sid = str(step.get("id") or "").strip()
    if not sid:
        return False
    if sid in impl_step_ids:
        return True
    role = str(step.get("role_required") or step.get("role") or "").strip().lower()
    lane = str(step.get("lane") or "").strip().lower()
    if role == "executor" or lane == "implementation":
        return True
    sid_lower = sid.lower()
    return sid_lower.startswith("apply_") or "implement" in sid_lower


def pid_alive(raw_pid):
    if raw_pid is None:
        return None
    try:
        pid = int(raw_pid)
    except Exception:
        return None
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


now_value = now_ts()
now_iso = iso_from_ts(now_value)
previous = read_json(trace_path)
if not isinstance(previous, dict):
    previous = {}
prev_counters = previous.get("counters") if isinstance(previous.get("counters"), dict) else {}

result = {
    "schema": "claw.no_nudge_idle_lane_autospawn.v1",
    "updated_at": now_iso,
    "updated_ts": now_value,
    "trace_path": str(trace_path),
    "status": "skipped",
    "enabled": enabled,
    "idle_threshold_sec": idle_threshold_sec,
    "cooldown_sec": cooldown_sec,
    "implementation_step_ids": impl_step_ids,
    "ready_work_exists": False,
    "idle_threshold_exceeded": False,
    "target_step_id": None,
    "idle_sec": None,
    "skip_reason": None,
    "launched": False,
    "launched_step_id": None,
    "tick_returncode": None,
    "tick_first_line": "",
    "error": None,
    "contradiction_abort_active": contradiction_abort_active,
    "contradiction_abort_remaining_sec": int(abort_remaining_sec),
    "contradiction_latch_repaired": contradiction_latch_repaired,
    "contradiction_latch_repair_reason": contradiction_latch_repair_reason or None,
    "evaluation": {
        "state_exists": state_path.exists(),
        "tick_exists": tick_path.exists(),
        "tick_executable": os.access(tick_path, os.X_OK),
        "active_impl_running": False,
        "active_step_id": None,
        "active_pid_alive": None,
        "ready_step_ids": [],
        "cooldown_remaining_sec": 0,
        "contradiction_abort_active": contradiction_abort_active,
        "contradiction_abort_remaining_sec": int(abort_remaining_sec),
        "contradiction_latch_repaired": contradiction_latch_repaired,
        "contradiction_latch_repair_reason": contradiction_latch_repair_reason or None,
    },
    "last_attempt_ts": previous.get("last_attempt_ts"),
    "last_attempt_at": previous.get("last_attempt_at"),
    "last_launch_ts": previous.get("last_launch_ts"),
    "last_launch_at": previous.get("last_launch_at"),
    "counters": {
        "attempts_total": int(prev_counters.get("attempts_total") or 0),
        "launches_total": int(prev_counters.get("launches_total") or 0),
        "failures_total": int(prev_counters.get("failures_total") or 0),
    },
}

attempted = False

if not enabled:
    result["skip_reason"] = "autospawn_disabled"
elif not state_path.exists():
    result["skip_reason"] = "autopilot_state_missing"
else:
    state = read_json(state_path)
    if not isinstance(state, dict):
        result["status"] = "error"
        result["skip_reason"] = "autopilot_state_invalid_json"
        result["error"] = f"failed_to_parse:{state_path}"
    else:
        steps = [row for row in (state.get("steps") or []) if isinstance(row, dict)]
        impl_steps = [row for row in steps if is_impl_step(row)]
        step_by_id = {str(row.get("id") or ""): row for row in steps}

        active = state.get("active") if isinstance(state.get("active"), dict) else {}
        active_step_id = str(active.get("step_id") or "").strip()
        active_step = step_by_id.get(active_step_id)
        active_is_impl = bool(active_step and is_impl_step(active_step))
        active_pid_alive = pid_alive(active.get("pid")) if active_is_impl else None
        active_impl_running = bool(active_is_impl and (active_pid_alive is True or (active and active_pid_alive is None)))

        ready_impl_steps = [row for row in impl_steps if step_ready(row, now_value)]
        ready_ids = [str(row.get("id") or "") for row in ready_impl_steps if str(row.get("id") or "").strip()]
        target_step_id = ready_ids[0] if ready_ids else ""

        impl_activity_ts = []
        for row in impl_steps:
            for key in ("last_finished_ts", "last_started_ts", "updated_ts", "started_ts", "finished_ts"):
                ts = as_int(row.get(key))
                if ts is not None:
                    impl_activity_ts.append(ts)
        active_start_ts = as_int(active.get("start_ts"))
        if active_is_impl and active_start_ts is not None:
            impl_activity_ts.append(active_start_ts)

        last_impl_ts = max(impl_activity_ts) if impl_activity_ts else None
        idle_sec = None if last_impl_ts is None else max(0, int(now_value - last_impl_ts))

        ready_work_exists = bool(ready_impl_steps)
        idle_threshold_exceeded = bool(ready_work_exists and not active_impl_running and (idle_sec is None or idle_sec >= idle_threshold_sec))

        result["ready_work_exists"] = ready_work_exists
        result["idle_threshold_exceeded"] = idle_threshold_exceeded
        result["target_step_id"] = target_step_id or None
        result["idle_sec"] = idle_sec
        result["evaluation"] = {
            "state_exists": True,
            "tick_exists": tick_path.exists(),
            "tick_executable": os.access(tick_path, os.X_OK),
            "active_impl_running": active_impl_running,
            "active_step_id": active_step_id or None,
            "active_pid_alive": active_pid_alive,
            "ready_step_ids": ready_ids,
            "cooldown_remaining_sec": 0,
        }

        if active_impl_running:
            result["skip_reason"] = "implementation_lane_active"
        elif not ready_work_exists:
            result["skip_reason"] = "no_ready_implementation_work"
        elif not idle_threshold_exceeded:
            result["skip_reason"] = "idle_below_threshold"
        else:
            if contradiction_abort_active:
                result["skip_reason"] = "contradiction_latched_auto_abort"
                result["evaluation"]["contradiction_abort_active"] = True
                result["evaluation"]["contradiction_abort_remaining_sec"] = int(abort_remaining_sec)
            else:
                # Cooldown should represent post-launch quiet time, not post-failure backoff.
                # If an autospawn attempt failed to launch work, keep trying on subsequent
                # watchdog runs so persistent launch failures cannot hide behind cooldown.
                last_launch_ts = as_int(previous.get("last_launch_ts"))
                if cooldown_sec > 0 and last_launch_ts is not None:
                    elapsed = max(0, now_value - last_launch_ts)
                    if elapsed < cooldown_sec:
                        result["skip_reason"] = "cooldown_active"
                        result["evaluation"]["cooldown_remaining_sec"] = int(cooldown_sec - elapsed)
                    else:
                        result["skip_reason"] = ""
                else:
                    result["skip_reason"] = ""

            if not result["skip_reason"]:
                if not (tick_path.exists() and os.access(tick_path, os.X_OK)):
                    result["status"] = "tick_failed"
                    result["skip_reason"] = "autopilot_tick_missing_or_not_executable"
                    result["error"] = str(tick_path)
                    attempted = True
                else:
                    attempted = True
                    try:
                        cp = subprocess.run(
                            [str(tick_path)],
                            text=True,
                            capture_output=True,
                            check=False,
                            timeout=300,
                            env={**os.environ},
                        )
                        tick_first_line = first_non_empty(cp.stdout) or first_non_empty(cp.stderr)
                        result["tick_returncode"] = int(cp.returncode)
                        result["tick_first_line"] = tick_first_line

                        after_state = read_json(state_path)
                        if isinstance(after_state, dict):
                            after_steps = [row for row in (after_state.get("steps") or []) if isinstance(row, dict)]
                            after_map = {str(row.get("id") or ""): row for row in after_steps}
                            after_active = after_state.get("active") if isinstance(after_state.get("active"), dict) else {}
                            after_active_id = str(after_active.get("step_id") or "").strip()
                            after_step = after_map.get(after_active_id)
                            after_is_impl = bool(after_step and is_impl_step(after_step))
                            after_pid_alive = pid_alive(after_active.get("pid")) if after_is_impl else None
                            launched = bool(after_is_impl and (after_pid_alive is True or (after_active and after_pid_alive is None)))
                            result["launched"] = launched
                            result["launched_step_id"] = after_active_id or None
                            if launched:
                                result["status"] = "launched"
                                result["last_launch_ts"] = now_value
                                result["last_launch_at"] = now_iso
                            elif cp.returncode != 0:
                                result["status"] = "tick_failed"
                            else:
                                result["status"] = "attempted_no_launch"
                        elif cp.returncode != 0:
                            result["status"] = "tick_failed"
                        else:
                            result["status"] = "attempted_no_launch"
                    except Exception as exc:
                        result["status"] = "tick_failed"
                        result["error"] = str(exc)

if attempted:
    result["last_attempt_ts"] = now_value
    result["last_attempt_at"] = now_iso
    result["counters"]["attempts_total"] = int(result["counters"].get("attempts_total") or 0) + 1
    if bool(result.get("launched")):
        result["counters"]["launches_total"] = int(result["counters"].get("launches_total") or 0) + 1
    else:
        result["counters"]["failures_total"] = int(result["counters"].get("failures_total") or 0) + 1

if not result.get("status"):
    result["status"] = "skipped"

if not result.get("skip_reason") and result["status"] == "skipped":
    result["skip_reason"] = "no_autospawn_action"

write_json(trace_path, result)
print(json.dumps(result, ensure_ascii=False))
PY
)"
autospawn_rc=$?
set -e
if [[ "$autospawn_rc" -ne 0 || -z "$autospawn_summary_json" ]]; then
  emit_blocker "idle_lane_autospawn_eval_failed" "idle_lane_autospawn_eval_failed; rc=${autospawn_rc}" "$AUTOPILOT_STATE_FILE"
  exit 0
fi

set +e
autospawn_parsed_output="$(python3 - "$autospawn_summary_json" 2>/tmp/run_no_nudge_continuity_watchdog_autospawn_parse.err <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
print(str(obj.get("status") or ""))
print("1" if bool(obj.get("ready_work_exists")) else "0")
print("1" if bool(obj.get("idle_threshold_exceeded")) else "0")
print(str(obj.get("target_step_id") or ""))
idle_sec = obj.get("idle_sec")
print("" if idle_sec is None else str(idle_sec))
print(str(obj.get("skip_reason") or ""))
print(str(obj.get("trace_path") or ""))
tick_returncode = obj.get("tick_returncode")
print("" if tick_returncode is None else str(tick_returncode))
print(str(obj.get("tick_first_line") or ""))
print(str(obj.get("error") or ""))
print(str(obj.get("launched_step_id") or ""))
print("1" if bool(obj.get("contradiction_abort_active")) else "0")
print(str(int(obj.get("contradiction_abort_remaining_sec") or 0)))
print("1" if bool(obj.get("contradiction_latch_repaired")) else "0")
print(str(obj.get("contradiction_latch_repair_reason") or ""))
PY
)"
autospawn_parse_rc=$?
set -e
if [[ "$autospawn_parse_rc" -ne 0 ]]; then
  err="$(cat /tmp/run_no_nudge_continuity_watchdog_autospawn_parse.err 2>/dev/null || true)"
  emit_blocker "idle_lane_autospawn_summary_invalid_json" "idle_lane_autospawn_summary_invalid_json; err=${err:0:180}" "$AUTOPILOT_STATE_FILE"
  exit 0
fi

readarray -t autospawn_parsed <<<"$autospawn_parsed_output"

autospawn_status="${autospawn_parsed[0]:-skipped}"
autospawn_ready_work="${autospawn_parsed[1]:-0}"
autospawn_idle_exceeded="${autospawn_parsed[2]:-0}"
autospawn_target_step="${autospawn_parsed[3]:-}"
autospawn_idle_sec="${autospawn_parsed[4]:-}"
autospawn_skip_reason="${autospawn_parsed[5]:-}"
autospawn_trace_path="${autospawn_parsed[6]:-$IDLE_AUTOSPAWN_TRACE_PATH}"
autospawn_tick_returncode="${autospawn_parsed[7]:-}"
autospawn_tick_first_line="${autospawn_parsed[8]:-}"
autospawn_error="${autospawn_parsed[9]:-}"
autospawn_launched_step="${autospawn_parsed[10]:-}"
autospawn_contradiction_abort_active="${autospawn_parsed[11]:-${autospawn_contradiction_abort_active:-0}}"
autospawn_contradiction_abort_remaining_sec="${autospawn_parsed[12]:-${autospawn_contradiction_abort_remaining_sec:-0}}"
autospawn_contradiction_latch_repaired="${autospawn_parsed[13]:-${autospawn_contradiction_latch_repaired:-0}}"
autospawn_contradiction_latch_repair_reason="${autospawn_parsed[14]:-${autospawn_contradiction_latch_repair_reason:-}}"

autospawn_latch_repair_suffix=""
if [[ "$autospawn_contradiction_latch_repaired" == "1" ]]; then
  autospawn_latch_repair_suffix="; contradiction_latch_repaired=1; contradiction_latch_repair_reason=${autospawn_contradiction_latch_repair_reason:-unknown}"
fi
status_suffix="${autospawn_latch_repair_suffix}${restore_drill_refresh_suffix}"
status_suffix+="${CRON_SESSION_CARD_LIFECYCLE_RECONCILE_STATUS_SUFFIX}"

emit_autospawn_contradiction_blocker() {
  local contradiction_kind="${1:-unknown_contradiction}"
  local contradiction_detail="${2:-status_contradiction}"
  local contradiction_fingerprint="${3:-task=run_no_nudge_continuity_watchdog;key=idle_lane_autospawn_status_contradiction;contradiction=${contradiction_kind:-unknown_contradiction}}"

  readarray -t latch_update < <(record_autospawn_contradiction_latch "$contradiction_kind" "$contradiction_detail" "$autospawn_trace_path")
  local contradiction_latched_now="${latch_update[0]:-0}"
  local contradiction_consecutive="${latch_update[1]:-0}"
  local contradiction_total="${latch_update[2]:-0}"
  local contradiction_abort_until_ts="${latch_update[3]:-0}"
  local contradiction_abort_until_at="${latch_update[4]:-}"

  if [[ "$contradiction_latched_now" == "1" ]]; then
    emit_blocker \
      "idle_lane_autospawn_contradiction_latched" \
      "idle_lane_autospawn_contradiction_latched; contradiction=${contradiction_kind}; consecutive=${contradiction_consecutive}; total=${contradiction_total}; threshold=${IDLE_AUTOSPAWN_CONTRADICTION_LATCH_AFTER}; abort_sec=${IDLE_AUTOSPAWN_CONTRADICTION_ABORT_SEC}; abort_until_ts=${contradiction_abort_until_ts}; abort_until_at=${contradiction_abort_until_at:-none}; detail=${contradiction_detail}; trace=${autospawn_trace_path}" \
      "$autospawn_trace_path" \
      "task=run_no_nudge_continuity_watchdog;key=idle_lane_autospawn_contradiction_latched;contradiction=${contradiction_kind};threshold=${IDLE_AUTOSPAWN_CONTRADICTION_LATCH_AFTER};consecutive=${contradiction_consecutive}"
  else
    emit_blocker \
      "idle_lane_autospawn_status_contradiction" \
      "idle_lane_autospawn_status_contradiction; ${contradiction_detail}; contradiction=${contradiction_kind}; consecutive=${contradiction_consecutive}; total=${contradiction_total}; threshold=${IDLE_AUTOSPAWN_CONTRADICTION_LATCH_AFTER}; trace=${autospawn_trace_path}" \
      "$autospawn_trace_path" \
      "$contradiction_fingerprint"
  fi
}

case "$autospawn_status" in
  skipped|launched|tick_failed|attempted_no_launch|error)
    ;;
  *)
    clear_autospawn_contradiction_latch
    emit_blocker \
      "idle_lane_autospawn_invalid_status" \
      "idle_lane_autospawn_invalid_status; status=${autospawn_status}; ready_work=${autospawn_ready_work}; idle_exceeded=${autospawn_idle_exceeded}; step=${autospawn_target_step:-unknown}; trace=${autospawn_trace_path}" \
      "$autospawn_trace_path" \
      "task=run_no_nudge_continuity_watchdog;key=idle_lane_autospawn_invalid_status;status=${autospawn_status}"
    exit 0
    ;;
esac

if [[ "$autospawn_status" == "error" ]]; then
  clear_autospawn_contradiction_latch
  emit_blocker \
    "idle_lane_autospawn_error" \
    "idle_lane_autospawn_error; skip_reason=${autospawn_skip_reason:-none}; error=${autospawn_error:-none}; step=${autospawn_target_step:-unknown}; trace=${autospawn_trace_path}" \
    "$autospawn_trace_path" \
    "task=run_no_nudge_continuity_watchdog;key=idle_lane_autospawn_error;skip_reason=${autospawn_skip_reason:-none}"
  exit 0
fi

if [[ "$autospawn_status" == "launched" ]]; then
  if [[ -z "$autospawn_launched_step" || "$autospawn_ready_work" != "1" || "$autospawn_idle_exceeded" != "1" || "$autospawn_tick_returncode" != "0" ]]; then
    emit_autospawn_contradiction_blocker \
      "launched_tuple_mismatch" \
      "status=launched; launched_step=${autospawn_launched_step:-none}; ready_work=${autospawn_ready_work}; idle_exceeded=${autospawn_idle_exceeded}; tick_returncode=${autospawn_tick_returncode:-none}; target_step=${autospawn_target_step:-unknown}" \
      "task=run_no_nudge_continuity_watchdog;key=idle_lane_autospawn_status_contradiction;status=launched;tick_returncode=${autospawn_tick_returncode:-none}"
    exit 0
  fi
  clear_autospawn_contradiction_latch
  echo "PROGRESS: idle_lane_autospawn_launched; step=${autospawn_launched_step:-$autospawn_target_step}; idle_sec=${autospawn_idle_sec:-unknown}; trace=${autospawn_trace_path}${status_suffix}"
  exit 0
fi

if [[ "$autospawn_status" == "skipped" ]]; then
  if [[ "$autospawn_contradiction_abort_active" == "1" && "$autospawn_ready_work" == "1" && "$autospawn_idle_exceeded" == "1" && "$autospawn_skip_reason" == "contradiction_latched_auto_abort" ]]; then
    emit_blocker \
      "idle_lane_autospawn_contradiction_latched" \
      "idle_lane_autospawn_contradiction_latched; contradiction=${autospawn_contradiction_last_kind:-unknown}; consecutive=${autospawn_contradiction_consecutive_seen:-0}; total=${autospawn_contradiction_total_seen:-0}; threshold=${IDLE_AUTOSPAWN_CONTRADICTION_LATCH_AFTER}; abort_remaining_sec=${autospawn_contradiction_abort_remaining_sec:-0}; last_reason=${autospawn_contradiction_last_reason:-none}; skip_reason=${autospawn_skip_reason}; step=${autospawn_target_step:-unknown}; trace=${autospawn_trace_path}" \
      "$autospawn_trace_path" \
      "task=run_no_nudge_continuity_watchdog;key=idle_lane_autospawn_contradiction_latched;status=skipped;skip_reason=contradiction_latched_auto_abort"
    exit 0
  fi

  if [[ -n "$autospawn_launched_step" || -n "$autospawn_tick_returncode" ]]; then
    emit_autospawn_contradiction_blocker \
      "skipped_with_attempt_markers" \
      "status=skipped; contradiction=skipped_with_attempt_markers; launched_step=${autospawn_launched_step:-none}; tick_returncode=${autospawn_tick_returncode:-none}; skip_reason=${autospawn_skip_reason:-none}" \
      "task=run_no_nudge_continuity_watchdog;key=idle_lane_autospawn_status_contradiction;status=skipped;contradiction=skipped_with_attempt_markers"
    exit 0
  fi

  if [[ "$autospawn_ready_work" == "1" && "$autospawn_idle_exceeded" == "1" && "$autospawn_skip_reason" != "cooldown_active" && "$autospawn_skip_reason" != "contradiction_latched_auto_abort" ]]; then
    emit_autospawn_contradiction_blocker \
      "actionable_without_attempt" \
      "status=skipped; contradiction=actionable_without_attempt; ready_work=${autospawn_ready_work}; idle_exceeded=${autospawn_idle_exceeded}; skip_reason=${autospawn_skip_reason:-none}; step=${autospawn_target_step:-unknown}" \
      "task=run_no_nudge_continuity_watchdog;key=idle_lane_autospawn_status_contradiction;status=skipped;contradiction=actionable_without_attempt;skip_reason=${autospawn_skip_reason:-none}"
    exit 0
  fi

  clear_autospawn_contradiction_latch
fi

if [[ "$autospawn_status" == "tick_failed" || "$autospawn_status" == "attempted_no_launch" ]]; then
  clear_autospawn_contradiction_latch
  if [[ "$autospawn_ready_work" == "1" && "$autospawn_idle_exceeded" == "1" ]]; then
    emit_blocker \
      "idle_lane_autospawn_failed" \
      "idle_lane_autospawn_failed; status=${autospawn_status}; step=${autospawn_target_step:-unknown}; idle_sec=${autospawn_idle_sec:-unknown}; skip_reason=${autospawn_skip_reason:-none}; tick=${autospawn_tick_first_line:-none}; error=${autospawn_error:-none}; trace=${autospawn_trace_path}" \
      "$autospawn_trace_path" \
      "task=run_no_nudge_continuity_watchdog;key=idle_lane_autospawn_failed;status=${autospawn_status};step=${autospawn_target_step:-unknown};skip_reason=${autospawn_skip_reason:-none}"
    exit 0
  fi
fi

clear_autospawn_contradiction_latch

if [[ "$warning_json" != "[]" ]]; then
  echo "PROGRESS: continuity healthy_with_warnings; age_sec=${age_sec}; warning_reasons=${warning_json}${status_suffix}"
  exit 0
fi

if [[ "$refreshed" -eq 1 ]]; then
  echo "PROGRESS: continuity surfaces refreshed internally; age_sec=${age_sec}${status_suffix}"
else
  echo "READY: continuity reminder watchdog healthy; age_sec=${age_sec}${status_suffix}"
fi

exit 0
