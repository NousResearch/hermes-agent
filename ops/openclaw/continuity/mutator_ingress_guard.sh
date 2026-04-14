#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
SCRIPT_NAME=""
ACTION_TOKEN=""
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"
AUDIT_PATH="${OPENCLAW_CONTINUITY_MUTATOR_INGRESS_AUDIT_PATH:-$ROOT/state/continuity/latest/mutator_ingress_audit.jsonl}"
RISK_TIER="${OPENCLAW_MUTATION_RISK_TIER:-low}"
MUTATION_OPERATION="${OPENCLAW_MUTATION_OPERATION:-}"
MUTATION_TICKET="${OPENCLAW_MUTATION_TICKET:-}"
AUTHORITY_CONTRACT_PATH="${OPENCLAW_LANE_AUTHORITY_CONTRACT_PATH:-$ROOT/docs/ops/templates/lane_topology_authority_contract.template.json}"
AUTHORITY_GATE_SCRIPT="$ROOT/ops/openclaw/continuity/lane_authority_gate.py"
AUTHORITY_FORCE_ALL="${OPENCLAW_AUTHORITY_GATE_FORCE_ALL:-0}"
INTERNAL_BYPASS_ALLOWLIST_MODE="${OPENCLAW_INTERNAL_BYPASS_ALLOWLIST_MODE:-soft}"
INTERNAL_BYPASS_ALLOWLIST="${OPENCLAW_INTERNAL_BYPASS_ALLOWLIST:-}"
INTERNAL_BYPASS_ALLOWLIST_PATH="${OPENCLAW_INTERNAL_BYPASS_ALLOWLIST_PATH:-}"
INTERNAL_BYPASS_ENFORCE_EVIDENCE_REF="${OPENCLAW_INTERNAL_BYPASS_ENFORCE_EVIDENCE_REF:-}"
DUMP_INTERNAL_BYPASS_ALLOWLIST=0

ATTESTATIONS=()
ATTESTATION_OBJECTS=()
INTERNAL_BYPASS_ALLOWLIST_READY=0
INTERNAL_BYPASS_ALLOWLIST_ENTRIES=()

usage() {
  cat <<'EOF'
Usage: mutator_ingress_guard.sh --script <name> [options]

Fail-closed ingress guard for direct mutator script entrypoints.

Allowed ingress modes:
  1) Token-validated: --action-token/--truth-anchor accepted by truth_anchor_guard.sh
  2) Internal bypass: OPENCLAW_INTERNAL_MUTATION=1 and OPENCLAW_INTERNAL_MUTATION_CALLSITE set

Options:
  --script <name>              Script identifier for audit records
  --action-token <value>       Canonical mutation token (preferred)
  --truth-anchor <value>       Legacy alias of --action-token
  --allow-legacy-anchor        Allow legacy anchor-only token mode

  --risk-tier <tier>           Mutation risk tier (low|medium|high|critical; default: low)
  --mutation-operation <id>    Expected operation_id for authority ticket checks
  --mutation-ticket <value>    Ticket JSON string, @path, or path (high-risk token path)
  --attestation <name>         Satisfied attestation (repeatable)
  --attestation-object <value> Structured attestation JSON string, @path, or path (repeatable)
  --dump-internal-bypass-allowlist
                               Print the built-in/configured internal bypass allowlist and exit

Environment knobs (internal bypass Stage B/Stage C):
  OPENCLAW_INTERNAL_BYPASS_ALLOWLIST_MODE=soft|off|enforce
  OPENCLAW_INTERNAL_BYPASS_ALLOWLIST="callsite_a,callsite_b,..."     (optional additions)
  OPENCLAW_INTERNAL_BYPASS_ALLOWLIST_PATH=/abs/path/to/allowlist.txt   (optional additions)
  OPENCLAW_INTERNAL_BYPASS_ENFORCE_EVIDENCE_REF=<evidence_ref>          (explicit break-glass evidence)
  # Unknown internal callsites are fail-closed by default.
  # Explicit break-glass evidence is required to allow unknown callsites.
  # High-risk break-glass bypass also requires lane authority gate ticket/attestation checks.

  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --script)
      SCRIPT_NAME="${2:-}"; shift 2 ;;
    --action-token|--truth-anchor)
      ACTION_TOKEN="${2:-}"; shift 2 ;;
    --allow-legacy-anchor)
      ALLOW_LEGACY_ANCHOR=1; shift ;;
    --risk-tier)
      RISK_TIER="${2:-}"; shift 2 ;;
    --mutation-operation)
      MUTATION_OPERATION="${2:-}"; shift 2 ;;
    --mutation-ticket)
      MUTATION_TICKET="${2:-}"; shift 2 ;;
    --attestation)
      ATTESTATIONS+=("${2:-}"); shift 2 ;;
    --attestation-object)
      ATTESTATION_OBJECTS+=("${2:-}"); shift 2 ;;
    --dump-internal-bypass-allowlist)
      DUMP_INTERNAL_BYPASS_ALLOWLIST=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ "$DUMP_INTERNAL_BYPASS_ALLOWLIST" != "1" && -z "$SCRIPT_NAME" ]]; then
  echo "missing --script" >&2
  exit 2
fi

case "${RISK_TIER,,}" in
  low|medium|high|critical)
    RISK_TIER="${RISK_TIER,,}" ;;
  *)
    echo "invalid --risk-tier: $RISK_TIER (expected low|medium|high|critical)" >&2
    exit 2 ;;
esac

if [[ -z "$MUTATION_OPERATION" ]]; then
  MUTATION_OPERATION="$SCRIPT_NAME"
fi

INTERNAL_BYPASS_ALLOWLIST_MODE="${INTERNAL_BYPASS_ALLOWLIST_MODE,,}"
case "$INTERNAL_BYPASS_ALLOWLIST_MODE" in
  soft|off|enforce)
    ;;
  *)
    echo "invalid OPENCLAW_INTERNAL_BYPASS_ALLOWLIST_MODE: $INTERNAL_BYPASS_ALLOWLIST_MODE (expected soft|off|enforce)" >&2
    exit 2 ;;
esac

trim_ws() {
  local val="${1-}"
  val="${val#"${val%%[![:space:]]*}"}"
  val="${val%"${val##*[![:space:]]}"}"
  printf '%s' "$val"
}

init_internal_bypass_allowlist() {
  if [[ "$INTERNAL_BYPASS_ALLOWLIST_READY" == "1" ]]; then
    return 0
  fi

  INTERNAL_BYPASS_ALLOWLIST_ENTRIES=(
    "continuity.sh:reconcile"
    "continuity.sh:checkpoint"
    "continuity.sh:sync"
    "continuity.sh:queue-sync"
    "continuity.sh:queue-arb"
    "continuity.sh:verify"
    "continuity.sh:normalize-events"
    "continuity.sh:gtc-sync"
    "continuity.sh:librarian"
    "continuity.sh:lock-break"
    "continuity.sh:execution-frontier"
    "reconcile.sh:write_checkpoint"
    "reconcile.sh:sync_latest_artifacts_cooldown"
    "reconcile.sh:sync_latest_artifacts_post_checkpoint"
    "write_checkpoint.sh:sync_latest_artifacts"
    "normalize_event_sources.sh:queue_arbitrator"
    "continuity_current.sh:auto_reconcile"
    "continuity_now.sh:refresh_hook:sync_latest_artifacts"
    "continuity_now.sh:refresh_hook:gtc_v2_sync"
    "continuity_now.sh:auto_orphaned_running_remediation"
    "continuity_now.sh:auto_queue_stale_wave_remediation"
    "swarm_runtime_check.sh:queue_remediate_probe"
    "swarm_runtime_check.sh:gtc_sync_probe"
    "run_hl_autopilot_tick_watchdog.sh:sync_queue_state"
    "run_web_capture_macro.sh:queue_claim_task"
    "run_web_capture_macro.sh:queue_transition_task"
    "run_competitive_parity_harness.sh:queue_claim"
    "run_competitive_parity_harness.sh:queue_transition"
    "verify_gate.sh:reconcile_fallback"
    "verify_then_resume.sh:check_connector_health:gtc_v2_sync_retry"
    "check_queue_cooldown_authority_regressions.py"
    "check_queue_cooldown_authority_regressions.py:fixed_now_authority"
    "harness:check_gtc_publish_transaction_regressions"
    "harness:check_gtc_incident_replay_regressions"
    "harness:check_gtc_queue_artifact_manifest_regressions"
    "check_autopilot_tick_provider_failure_queue_regressions.py:tick"
    "check_delegated_gate_summary_queue_regressions.py:queue_arbitrator"
    "check_delegated_gate_summary_queue_regressions.py:queue_arbitrator_digest_repair"
    "check_delegated_gate_summary_queue_regressions.py:queue_arbitrator_digest_missing"
    "check_delegated_gate_summary_queue_regressions.py:queue_arbitrator_invalid"
    "check_delegated_gate_summary_queue_regressions.py:queue_sync"
    "check_delegated_gate_summary_queue_regressions.py:queue_sync_digest_repair"
    "check_delegated_gate_summary_queue_regressions.py:queue_sync_digest_missing"
    "check_delegated_gate_summary_queue_regressions.py:queue_sync_drop_invalid"
    "check_delegated_gate_summary_queue_regressions.py:queue_sync_failclose_invalid"
    "context_runtime_local_watch.sh:sync_latest_artifacts"
    "context_runtime_local_watch.sh:runtime_unhealthy_checkpoint"
    "context_runtime_local_watch.sh:context_high_checkpoint"
    "context_runtime_local_watch.sh:session_bloat_checkpoint"
    "watchdog_session_bloat.sh:preoverflow_checkpoint"
    "watchdog_session_bloat.sh:hard_reset_checkpoint"
    "execution_frontier_ledger.sh:core_queue_txn_handoff"
  )

  if [[ -n "$INTERNAL_BYPASS_ALLOWLIST" ]]; then
    local raw_item=""
    IFS=',' read -r -a raw_items <<<"$INTERNAL_BYPASS_ALLOWLIST"
    for raw_item in "${raw_items[@]}"; do
      local item=""
      item="$(trim_ws "$raw_item")"
      if [[ -n "$item" ]]; then
        INTERNAL_BYPASS_ALLOWLIST_ENTRIES+=("$item")
      fi
    done
  fi

  if [[ -n "$INTERNAL_BYPASS_ALLOWLIST_PATH" ]]; then
    if [[ -f "$INTERNAL_BYPASS_ALLOWLIST_PATH" ]]; then
      local raw_line=""
      while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
        local stripped=""
        local item=""
        stripped="${raw_line%%#*}"
        item="$(trim_ws "$stripped")"
        if [[ -n "$item" ]]; then
          INTERNAL_BYPASS_ALLOWLIST_ENTRIES+=("$item")
        fi
      done <"$INTERNAL_BYPASS_ALLOWLIST_PATH"
    else
      echo "mutator ingress warning: OPENCLAW_INTERNAL_BYPASS_ALLOWLIST_PATH not found: $INTERNAL_BYPASS_ALLOWLIST_PATH" >&2
    fi
  fi

  INTERNAL_BYPASS_ALLOWLIST_READY=1
}

print_internal_bypass_allowlist() {
  init_internal_bypass_allowlist
  if [[ "${#INTERNAL_BYPASS_ALLOWLIST_ENTRIES[@]}" -eq 0 ]]; then
    return 0
  fi
  printf '%s\n' "${INTERNAL_BYPASS_ALLOWLIST_ENTRIES[@]}" | LC_ALL=C sort -u
}

if [[ "$DUMP_INTERNAL_BYPASS_ALLOWLIST" == "1" ]]; then
  print_internal_bypass_allowlist
  exit 0
fi

is_internal_bypass_callsite_allowlisted() {
  local callsite="${1-}"
  local allowed=""
  init_internal_bypass_allowlist
  for allowed in "${INTERNAL_BYPASS_ALLOWLIST_ENTRIES[@]}"; do
    if [[ -z "$allowed" ]]; then
      continue
    fi
    if [[ "$allowed" == *"*"* || "$allowed" == *"?"* || "$allowed" == *"["* ]]; then
      if [[ "$callsite" == $allowed ]]; then
        return 0
      fi
    elif [[ "$callsite" == "$allowed" ]]; then
      return 0
    fi
  done
  return 1
}

emit_audit() {
  local mode="$1"
  local callsite="$2"
  local detail="$3"
  set +e
  python3 - "$AUDIT_PATH" "$mode" "$SCRIPT_NAME" "$callsite" "$detail" <<'PY' >/dev/null 2>&1
import datetime as dt
import json
import pathlib
import sys

audit_path = pathlib.Path(sys.argv[1])
mode = str(sys.argv[2] or "").strip()
script_name = str(sys.argv[3] or "").strip()
callsite = str(sys.argv[4] or "").strip()
detail = str(sys.argv[5] or "").strip()

audit_path.parent.mkdir(parents=True, exist_ok=True)
row = {
    "ts": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "mode": mode,
    "script": script_name,
    "callsite": callsite or None,
    "detail": detail or None,
}
with audit_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
PY
  set -e
}

authority_gate_required=0
if [[ "$RISK_TIER" == "high" || "$RISK_TIER" == "critical" || "$AUTHORITY_FORCE_ALL" == "1" ]]; then
  authority_gate_required=1
fi

run_authority_gate() {
  if [[ "$authority_gate_required" != "1" ]]; then
    return 0
  fi

  if [[ ! -x "$AUTHORITY_GATE_SCRIPT" ]]; then
    echo "authority gate helper missing: $AUTHORITY_GATE_SCRIPT" >&2
    return 91
  fi

  gate_args=(
    --contract "$AUTHORITY_CONTRACT_PATH"
    --risk-tier "$RISK_TIER"
    --mutation-operation "$MUTATION_OPERATION"
    --json
  )

  if [[ -n "$MUTATION_TICKET" ]]; then
    gate_args+=(--mutation-ticket "$MUTATION_TICKET")
  fi

  local att=""
  for att in "${ATTESTATIONS[@]}"; do
    if [[ -n "${att:-}" ]]; then
      gate_args+=(--attestation "$att")
    fi
  done

  for att_obj in "${ATTESTATION_OBJECTS[@]}"; do
    if [[ -n "${att_obj:-}" ]]; then
      gate_args+=(--attestation-object "$att_obj")
    fi
  done

  local gate_out=""
  if gate_out="$("$AUTHORITY_GATE_SCRIPT" "${gate_args[@]}" 2>&1)"; then
    return 0
  else
    local gate_rc=$?
    if [[ -n "$gate_out" ]]; then
      printf '%s\n' "$gate_out" >&2
    fi
    return "$gate_rc"
  fi
}

if [[ -n "$ACTION_TOKEN" ]]; then
  guard_args=(--action-token "$ACTION_TOKEN")
  if [[ "$ALLOW_LEGACY_ANCHOR" == "1" ]]; then
    guard_args+=(--allow-legacy-anchor)
  fi
  set +e
  "$ROOT/ops/openclaw/continuity/truth_anchor_guard.sh" "${guard_args[@]}" >/dev/null
  guard_rc=$?
  set -e
  if [[ "$guard_rc" -ne 0 ]]; then
    emit_audit "denied" "" "invalid_action_token"
    exit "$guard_rc"
  fi

  set +e
  run_authority_gate
  authority_rc=$?
  set -e
  if [[ "$authority_rc" -ne 0 ]]; then
    emit_audit "denied" "" "authority_contract_block"
    exit "$authority_rc"
  fi

  emit_audit "token_validated" "" "token_path"
  exit 0
fi

INTERNAL_MUTATION="${OPENCLAW_INTERNAL_MUTATION:-0}"
INTERNAL_CALLSITE="${OPENCLAW_INTERNAL_MUTATION_CALLSITE:-}"
if [[ "$INTERNAL_MUTATION" == "1" ]]; then
  if [[ -z "$INTERNAL_CALLSITE" ]]; then
    emit_audit "denied" "" "internal_missing_callsite"
    echo "internal mutation bypass requires OPENCLAW_INTERNAL_MUTATION_CALLSITE" >&2
    exit 2
  fi

  effective_allowlist_mode="$INTERNAL_BYPASS_ALLOWLIST_MODE"
  break_glass_evidence_ref="$(trim_ws "$INTERNAL_BYPASS_ENFORCE_EVIDENCE_REF")"

  allowlist_callsite_known=0
  allowlist_detail=""
  if is_internal_bypass_callsite_allowlisted "$INTERNAL_CALLSITE"; then
    allowlist_callsite_known=1
    allowlist_detail="internal_env_allowlisted"
  fi

  if [[ "$allowlist_callsite_known" == "1" ]]; then
    if [[ "$effective_allowlist_mode" == "enforce" && "$authority_gate_required" == "1" ]]; then
      set +e
      run_authority_gate
      internal_authority_rc=$?
      set -e
      if [[ "$internal_authority_rc" -ne 0 ]]; then
        emit_audit "denied" "$INTERNAL_CALLSITE" "internal_authority_contract_block"
        exit "$internal_authority_rc"
      fi
      allowlist_detail="internal_env_allowlisted_authority_enforced"
    elif [[ "$effective_allowlist_mode" == "off" ]]; then
      allowlist_detail="internal_env_allowlist_disabled"
    fi

    emit_audit "internal_bypass" "$INTERNAL_CALLSITE" "$allowlist_detail"
    exit 0
  fi

  if [[ -z "$break_glass_evidence_ref" ]]; then
    emit_audit "denied" "$INTERNAL_CALLSITE" "internal_callsite_not_allowlisted"
    echo "internal mutation bypass denied: callsite not allowlisted and OPENCLAW_INTERNAL_BYPASS_ENFORCE_EVIDENCE_REF is missing (break-glass evidence required) ($INTERNAL_CALLSITE)" >&2
    exit 2
  fi

  break_glass_detail="internal_env_unknown_callsite_break_glass_allow"
  if [[ "$authority_gate_required" == "1" ]]; then
    set +e
    run_authority_gate
    internal_authority_rc=$?
    set -e
    if [[ "$internal_authority_rc" -ne 0 ]]; then
      emit_audit "denied" "$INTERNAL_CALLSITE" "internal_unknown_callsite_break_glass_authority_block"
      exit "$internal_authority_rc"
    fi
    break_glass_detail="internal_env_unknown_callsite_break_glass_authority_enforced"
  fi

  emit_audit "internal_bypass" "$INTERNAL_CALLSITE" "$break_glass_detail"
  exit 0
fi

emit_audit "denied" "$INTERNAL_CALLSITE" "missing_token_or_internal_bypass"
echo "mutator ingress denied for ${SCRIPT_NAME}: requires --action-token (legacy: --truth-anchor) or OPENCLAW_INTERNAL_MUTATION=1 with OPENCLAW_INTERNAL_MUTATION_CALLSITE" >&2
exit 2
