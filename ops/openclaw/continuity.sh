#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
CONT_DIR="$ROOT/ops/openclaw/continuity"
CONTINUITY_CMD="bash \"$ROOT/ops/openclaw/continuity.sh\""

usage() {
  cat <<EOF
Usage: continuity.sh [global-options] <command> [args]

Global options:
  --action-token <value>      Canonical mutation precondition token
  --truth-anchor <value>      Legacy alias of --action-token
  --allow-unanchored-mutate   Bypass token precondition for this invocation
                              (also set OPENCLAW_ALLOW_UNANCHORED_MUTATION=1)
  --allow-legacy-anchor       Accept anchor-only tokens (break-glass only)
                              (also set OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY=1)

Commands:
  now         Render compact continuity status (passes args to continuity_now.sh)
  current     Compute successor-safe continuity/current surface (passes args to continuity_current.sh)
  worker-health-canary
              Build execution-supervisor worker-health canary evidence artifact
              (passes args to continuity/execution_supervisor_worker_health_canary.py)
  execution-status
              Read durable execution-program status artifact (passes args to continuity/execution_program_status.sh)
  execution-frontier
              Read/advance deterministic execution frontier ledger, including guarded
              supervisor transition and bounded autonomous-dispatch transition for wave-close advancement
              (passes args to continuity/execution_frontier_ledger.sh)
  execution-frontier-antistall-check
              Run fail-closed sustained-soak regression checks for execution-frontier anti-stall truthfulness
              (passes args to continuity/check_execution_frontier_antistall_regressions.py)
  routing-preflight-route-check
              Validate route/provider/audit operator-surface parity for routing-preflight session-route artifacts
              (passes args to continuity/check_routing_preflight_route_decision_operator_surface.py)
  handover    Render/check handover/latest surfaces + staleness (passes args to handover_latest.sh)
  reset-ready-refresh
              Refresh continuity/current + aligned successor proof + handover/latest
              and fail closed if published reset-ready surfaces disagree
              (passes args to continuity/reset_ready_refresh.sh)
  blocker-registry
              Generate canonical blocker registry from continuity/current truth
              (passes args to continuity/blocker_registry.sh)
  mission-control
              Render operator mission control headline/actions (passes args to operator_mission_control.sh)
  triage-console
              Render high-signal operator triage HUD from mission-control truth
              (passes args to operator_triage_console.sh)
  federated-evidence-gate
              Run standalone federated-evidence packet support gate (LT-08)
              (passes args to scripts/federated_evidence_packet_gate.py)
  gate-os     Emit unified GateOS snapshot (passes args to gate_os_snapshot.sh)
  history     Render continuity audit rollup/history surface (passes args to history.sh)
  reconcile   Run low-risk drift reconcile flow (passes args to reconcile.sh)
  verify      Run verify-before-mutate gate (passes args to verify_then_resume.sh)
  verify-gate-status
              Show verify-gate preflight strict-autonomy effective mode/source
              (passes args to verify_gate_status.sh)
  restore-drill-refresh
              Refresh restore-drill evidence when stale (passes args to restore_drill_refresh.sh)
  sync        Sync latest bridge/handover from latest checkpoint (passes args to sync_latest_artifacts.sh)
  checkpoint  Write a continuity checkpoint (passes args to write_checkpoint.sh)
  queue-sync  Mirror autopilot JSON into continuity queue (passes args to queue_sync_from_autopilot_json.sh)
  queue-arb   Deterministic queue arbitration helper (passes args to queue_arbitrator.sh)
  queue-replay
              Deterministic queue journal replay/projection verification (passes args to queue_replay_verify.sh)
  repo-review-closeout
              Verify repo-review fold-in/closeout claims against queue truth (passes args to continuity/check_repo_review_queue_closeout.py)
  failover-replay-evidence
              Run deterministic Wave-3 failover replay evidence emitter
              (passes args to continuity/failover_replay_evidence.py)
  failover-stress-soak
              Run deterministic A3 failover/succession stress-soak harness
              (passes args to continuity/failover_stress_soak.py)
  failover-stress-runtime-evidence
              Run guarded A3 failover-stress runtime evidence command
              (passes args to continuity/failover_stress_runtime_evidence.py)
  failover-stress-runtime-check
              Run bounded A3 failover runtime-evidence summary wrapper
              (passes args to continuity/check_a3_failover_runtime_evidence.py)
  a6-multi-host-jitter-check
              Run deterministic A6 multi-host jitter/chaos false-positive resistance harness
              (passes args to continuity/a6_multi_host_jitter_harness.py)
  lock-break  Operator-audited lock-break workflow (passes args to lock_break.sh)
  librarian   Librarian/Curator runtime scaffold (passes args to librarian_curator.sh)
  promotion-gate
              Run deterministic promotion contract gate runner (passes args to scripts/promotion_gate_runner.py)
  model-rollout-gate
              Run deterministic model qualification/rollout gate runner (passes args to scripts/model_rollout_gate_runner.py)
  model-rollout-controller
              Consume model rollout gate decisions into ledger/events
              (passes args to scripts/model_rollout_ledger_controller.py)
  model-rollout-health
              Build deterministic rollout health snapshot for ring gates
              (passes args to scripts/model_rollout_health_snapshot.py)
  model-rollout-cost
              Build deterministic cost-governance telemetry snapshot from rollout/routing surfaces
              (passes args to scripts/model_rollout_cost_governance_snapshot.py)
  model-route-policy-lint
              Run deterministic long-window route-policy soak/lint snapshot
              (passes args to scripts/model_route_policy_soak_lint.py)
  model-rollout-soak
              Build deterministic ring-soak automation snapshot from rollout ledger + health
              (passes args to scripts/model_rollout_ring_soak_snapshot.py)
  model-rollout-dashboard
              Build consolidated rollout operator dashboard snapshot from health/cost/soak surfaces
              (passes args to scripts/model_rollout_dashboard_snapshot.py)
  knowledge-queue
              Legacy compatibility helper for bounded queue flow (passes args to scripts/knowledge_review_queue.py)
  lane-crossover-guard
              Evaluate lane crossover packet ingress guard (passes args to continuity/lane_crossover_ingress_guard.py)
  bridge-ingest
              Run cross-lane bridge ingest validator/runtime gate (passes args to continuity/cross_lane_bridge_ingest.py)
  source-material-guard
              Evaluate source-material classification/routing firewall policy
              (passes args to continuity/source_material_classification_guard.py)
  session-route
              Resolve deterministic route-policy routing decision (session/task/risk -> route class/model)
              Strict transport-decision conformance is fail-closed by default.
              Worker-allocation dispatch contract is enforced by default for worker_slice requests
              (scope_shape, verification_class, worker_topology, fold_in_target).
              Legacy bypass is bounded and explicit via --legacy-allow-missing-transport-decision.
              Legacy worker-allocation bypass is bounded and explicit via --legacy-allow-missing-worker-allocation-contract.
              Telegram direct-lane offload is enforced by default for heavy/non-trivial worker execution (medium/high/critical risk or validator-required), plus coding worker slices (unless explicit tiny exception contract is satisfied). Declared worker offload (`worker_lane=subagent_default`) must include worker-target topology evidence (`lane_name`, `agent_id`, `session_key`) and pass worker-target conformance checks (worker pattern allowlist + no cockpit lane tokens + session_key agent parity). --legacy-allow-telegram-direct-heavy-on-dm is a bounded heavy-route compatibility bypass only; non-trivial/coding/session-kind checks remain fail-closed.
              Deprecated --allow-missing-transport-decision/--no-require-transport-decision are ignored.
              (passes args to scripts/session_topology_router.py)
  session-transport-route
              Resolve deterministic transport/topic routing decision (chat/thread -> lane/agent/session key)
              (passes args to scripts/session_topology_transport_router.py)
  promotion-queue
              Canonical knowledge review/approval/promotion queue runtime (passes args to scripts/knowledge_promotion_queue.py)
  shared-memory
              Shared memory fabric lifecycle runtime (passes args to scripts/shared_memory_fabric.py)
  markdown-gate
              Run markdown conversion quality gate (passes args to scripts/markdown_conversion_quality_gate.py)
  material-classify
              Run deterministic source material classifier (passes args to scripts/source_material_classifier.py)
  knowledge-ingest
              Run production knowledge ingestion evaluator/apply runtime (passes args to scripts/production_knowledge_ingestion.py)
  doc-intake-closeout
              Run document/PDF subagent-intake closeout integration gate (passes args to scripts/document_intake_batch_integration_gate.py)
  release-evidence-gate
              Run release evidence ladder governance gate (passes args to scripts/release_evidence_ladder_gate.py)
  db-check    Run continuity DB integrity + invariants checker (passes args to db_integrity_check.sh)
  swarm-check Run swarm runtime doctor (passes args to swarm_runtime_check.sh)
  slot-fill-check
              Validate subagent slot-fill protocol/runbook wiring
              (passes args to continuity/check_slot_fill_protocol.sh)
  autonomy-regressions
              Run unified autonomy/continuity regression cluster harness
              (passes args to continuity/check_autonomy_continuity_regressions.py)
  autonomy-smoke
              Run minimal high-signal autonomy/continuity smoke checklist
              (passes args to continuity/check_autonomy_continuity_regressions.py --profile smoke)
  parity-run  Run competitive parity harness runner (passes args to architecture/run_competitive_parity_harness.sh)
  web-capture Run low-noise web capture macro wrapper (passes args to run_web_capture_macro.sh)
  web-capture-scheduler
              Run deterministic multi-domain web capture fairness scheduler (passes args to run_web_capture_scheduler.sh)
  gtc-sync    Sync Ground-Truth Connectors v2 evidence/index/latest surfaces (passes args to continuity/gtc_v2_sync.sh)
  gtc-schema-check
              Validate generated GTC latest surfaces against schema contracts (passes args to continuity/gtc_latest_schema_check.sh)
  gtc-replay  Build deterministic incident replay bundle from GTC surfaces/index (passes args to continuity/gtc_incident_replay.sh)
  normalize-events  Normalize continuity event source namespaces (passes args to normalize_event_sources.sh)

Examples:
  ${CONTINUITY_CMD} now --json
  ${CONTINUITY_CMD} current --refresh --json
  ${CONTINUITY_CMD} worker-health-canary --json
  ${CONTINUITY_CMD} execution-status --refresh --json
  ${CONTINUITY_CMD} execution-frontier --refresh --json
  ${CONTINUITY_CMD} execution-frontier-antistall-check --json
  ${CONTINUITY_CMD} routing-preflight-route-check --json
  ${CONTINUITY_CMD} --action-token <current.action_token> execution-frontier supervisor-advance-wave-close --reason "wave done" --json
  ${CONTINUITY_CMD} --action-token <current.action_token> execution-frontier supervisor-autonomous-dispatch --reason "watchdog_tick" --json
  ${CONTINUITY_CMD} --action-token <current.action_token> execution-frontier supervisor-reset-txn-handoff-guard --reason "manual soak guard reset" --json
  ${CONTINUITY_CMD} --action-token <current.action_token> execution-frontier advance-wave-close --reason "wave done" --json
  ${CONTINUITY_CMD} handover --refresh --json
  ${CONTINUITY_CMD} reset-ready-refresh --json
  ${CONTINUITY_CMD} blocker-registry --json
  ${CONTINUITY_CMD} mission-control --refresh --json
  ${CONTINUITY_CMD} triage-console --json
  ${CONTINUITY_CMD} federated-evidence-gate --packet tests/fixtures/lt08/federated_evidence_packet_valid_v1.json --json
  ${CONTINUITY_CMD} gate-os --refresh --json
  ${CONTINUITY_CMD} history --json
  ${CONTINUITY_CMD} --action-token <current.action_token> reconcile
  ${CONTINUITY_CMD} queue-replay --strict --json
  ${CONTINUITY_CMD} repo-review-closeout --target-row A8 --target-row C6 --report-path reports/repo_wave7_requested_batch_foldin_synthesis_openclaw_2026-03-26.md --json
  ${CONTINUITY_CMD} failover-replay-evidence --json
  ${CONTINUITY_CMD} failover-stress-soak --cycles 2 --json
  ${CONTINUITY_CMD} failover-stress-runtime-evidence --cycles 2 --json
  ${CONTINUITY_CMD} failover-stress-runtime-check --require-live-assertions
  ${CONTINUITY_CMD} a6-multi-host-jitter-check --json
  ${CONTINUITY_CMD} verify-gate-status --json
  ${CONTINUITY_CMD} restore-drill-refresh --json
  ${CONTINUITY_CMD} autonomy-regressions --json
  ${CONTINUITY_CMD} autonomy-smoke --json
  ${CONTINUITY_CMD} autonomy-regressions --profile critical-path --benchmark-baseline tests/fixtures/a6/autonomy_regression_critical_path_baseline_v1.json --benchmark-output state/continuity/latest/autonomy_regression_critical_path_benchmark_scorecard_latest.json --json
  ${CONTINUITY_CMD} slot-fill-check --json
  ${CONTINUITY_CMD} gtc-replay --incident-index 1 --json
  ${CONTINUITY_CMD} lock-break --task-id autopilot:quality_gate --reason "manual recovery" --operator yq --action-token <...> --json
  ${CONTINUITY_CMD} librarian ingest --json
  ${CONTINUITY_CMD} promotion-gate --candidate memory/research_cases/<id>/CANDIDATE/candidates/<candidate>/promotion_candidate.json --json
  ${CONTINUITY_CMD} model-rollout-gate --packet docs/ops/templates/model_qualification_packet.template.json --no-decision-log --json
  ${CONTINUITY_CMD} model-rollout-health --json
  ${CONTINUITY_CMD} model-rollout-cost --json
  ${CONTINUITY_CMD} model-route-policy-lint --window-hours 168 --json
  ${CONTINUITY_CMD} model-rollout-soak --json
  ${CONTINUITY_CMD} model-rollout-dashboard --json
  ${CONTINUITY_CMD} model-rollout-controller --json
  ${CONTINUITY_CMD} session-transport-route --topology docs/ops/templates/session_topology_transport_contract.template.json --request docs/ops/templates/session_topology_transport_route_request.template.json --json > /tmp/transport_decision.json
  ${CONTINUITY_CMD} session-route --topology docs/ops/templates/session_topology_contract.template.json --request docs/ops/templates/session_route_request.template.json --qualification-decision state/continuity/latest/example_model_gate_decision.json --transport-decision /tmp/transport_decision.json --json
  ${CONTINUITY_CMD} session-route --legacy-allow-missing-worker-allocation-contract --topology docs/ops/templates/session_topology_contract.template.json --request docs/ops/templates/session_route_request.template.json --qualification-decision state/continuity/latest/example_model_gate_decision.json --transport-decision /tmp/transport_decision.json --json
  ${CONTINUITY_CMD} session-route --legacy-allow-missing-transport-decision --topology docs/ops/templates/session_topology_contract.template.json --request docs/ops/templates/session_route_request.template.json --qualification-decision state/continuity/latest/example_model_gate_decision.json --json
  ${CONTINUITY_CMD} session-route --legacy-allow-telegram-direct-heavy-on-dm --topology docs/ops/templates/session_topology_contract.template.json --request docs/ops/templates/session_route_request.template.json --qualification-decision state/continuity/latest/example_model_gate_decision.json --transport-decision /tmp/transport_decision.json --json
  ${CONTINUITY_CMD} promotion-queue enqueue --candidate memory/research_cases/<id>/CANDIDATE/candidates/<candidate>/promotion_candidate.json --json
  ${CONTINUITY_CMD} shared-memory status --json
  ${CONTINUITY_CMD} markdown-gate --packet docs/ops/templates/markdown_conversion_gate_packet.template.json --no-decision-log --json
  ${CONTINUITY_CMD} material-classify --packet docs/ops/templates/source_material_classification_packet.template.json --no-decision-log --json
  ${CONTINUITY_CMD} knowledge-ingest evaluate --packet docs/ops/templates/production_knowledge_ingestion_packet.template.json --json
  ${CONTINUITY_CMD} doc-intake-closeout --packet docs/ops/templates/document_intake_batch_integration.template.json --json
  ${CONTINUITY_CMD} release-evidence-gate --bundle docs/ops/templates/release_evidence_bundle.template.json --json
  ${CONTINUITY_CMD} lane-crossover-guard --packet docs/ops/templates/lane_crossover_signal.template.json --to-lane-id lane.column_a.no_nudge_autonomy --to-lane-epoch epoch_20260320_a1 --allow-from lane.column_b.swarm_orchestration=epoch_20260320_b1
  ${CONTINUITY_CMD} bridge-ingest --bridge docs/ops/templates/cross_lane_bridge_object.template.json --to-lane-id lane.column_a.no_nudge_autonomy --to-lane-epoch epoch_20260320_a1 --allow-from lane.column_b.swarm_orchestration=epoch_20260320_b1
  ${CONTINUITY_CMD} source-material-guard --classification docs/ops/templates/source_material_classification.template.json --json
EOF
}

TRUTH_ANCHOR=""
ALLOW_UNANCHORED_MUTATE="${OPENCLAW_ALLOW_UNANCHORED_MUTATION:-0}"
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --action-token|--truth-anchor)
      TRUTH_ANCHOR="${2:-}"
      shift 2 ;;
    --allow-unanchored-mutate)
      ALLOW_UNANCHORED_MUTATE=1
      shift ;;
    --allow-legacy-anchor)
      ALLOW_LEGACY_ANCHOR=1
      shift ;;
    --)
      shift
      break ;;
    -h|--help|help)
      usage
      exit 0 ;;
    *)
      break ;;
  esac
done

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  usage >&2
  exit 2
fi
shift || true

is_mutating=0
case "$cmd" in
  reconcile|sync|checkpoint|queue-sync|normalize-events|gtc-sync|lock-break)
    is_mutating=1 ;;
  verify)
    is_mutating=0
    for arg in "$@"; do
      if [[ "$arg" == "--execute" ]]; then
        is_mutating=1
        break
      fi
    done ;;
  execution-frontier)
    subcmd="${1:-}"
    case "$subcmd" in
      advance-wave-close|supervisor-advance-wave-close|supervisor-autonomous-dispatch|supervisor-reset-txn-handoff-guard)
        is_mutating=1 ;;
      *)
        is_mutating=0 ;;
    esac ;;
  queue-arb)
    subcmd="${1:-}"
    case "$subcmd" in
      claim|transition|remediate)
        is_mutating=1 ;;
      *)
        is_mutating=0 ;;
    esac ;;
  librarian)
    subcmd="${1:-}"
    case "$subcmd" in
      ingest|promote|build-index)
        is_mutating=1 ;;
      *)
        is_mutating=0 ;;
    esac ;;
  knowledge-queue)
    subcmd="${1:-}"
    case "$subcmd" in
      enqueue|transition)
        is_mutating=1 ;;
      *)
        is_mutating=0 ;;
    esac ;;
  promotion-queue)
    subcmd="${1:-}"
    case "$subcmd" in
      enqueue|review|promote)
        is_mutating=1 ;;
      *)
        is_mutating=0 ;;
    esac ;;
  shared-memory)
    subcmd="${1:-}"
    case "$subcmd" in
      promote|conflict|demote)
        is_mutating=1 ;;
      *)
        is_mutating=0 ;;
    esac ;;
  model-rollout-controller)
    is_mutating=1 ;;
  knowledge-ingest)
    subcmd="${1:-}"
    case "$subcmd" in
      ingest|apply|ingest-multi-host)
        is_mutating=1 ;;
      *)
        is_mutating=0 ;;
    esac ;;
  *)
    is_mutating=0 ;;
esac

if [[ "$is_mutating" -eq 1 && "$ALLOW_UNANCHORED_MUTATE" != "1" ]]; then
  if [[ -z "$TRUTH_ANCHOR" ]]; then
    for ((i=1; i <= $#; i++)); do
      arg_i="${!i}"
      if [[ "$arg_i" == "--action-token" || "$arg_i" == "--truth-anchor" ]]; then
        next_i=$((i + 1))
        TRUTH_ANCHOR="${!next_i:-}"
        break
      fi
    done
  fi

  if [[ -z "$TRUTH_ANCHOR" ]]; then
    echo "mutating command requires --action-token (legacy alias: --truth-anchor; override: --allow-unanchored-mutate)" >&2
    echo "hint: ${CONTINUITY_CMD} current --refresh --json  # use .action_token" >&2
    exit 2
  fi
  guard_args=(--script "continuity.sh" --action-token "$TRUTH_ANCHOR")
  if [[ "$ALLOW_LEGACY_ANCHOR" == "1" ]]; then
    guard_args+=(--allow-legacy-anchor)
  fi
  "$CONT_DIR/mutator_ingress_guard.sh" "${guard_args[@]}" >/dev/null
fi

if [[ "$ALLOW_LEGACY_ANCHOR" == "1" ]]; then
  export OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY=1
fi

if [[ "$is_mutating" -eq 1 ]]; then
  export OPENCLAW_INTERNAL_MUTATION=1
  export OPENCLAW_INTERNAL_MUTATION_CALLSITE="continuity.sh:${cmd}"
fi

continuity_truthy() {
  local raw="${1:-}"
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$raw" in
    1|true|yes|y|on)
      return 0 ;;
    *)
      return 1 ;;
  esac
}

prepare_verify_strict_autonomy_context() {
  local verify_gate_lib="$ROOT/ops/openclaw/lib/verify_gate.sh"
  local strict_autonomy_override=""
  local arg=""

  for arg in "$@"; do
    case "$arg" in
      --strict-autonomy-regressions)
        strict_autonomy_override="1" ;;
      --no-strict-autonomy-regressions)
        strict_autonomy_override="0" ;;
    esac
  done

  local resolved_enabled="0"
  local resolved_required="0"
  local resolved_source="disabled"

  if [[ -f "$verify_gate_lib" ]]; then
    # shellcheck source=/dev/null
    source "$verify_gate_lib"
    if declare -F openclaw_verify_gate_resolve_strict_autonomy >/dev/null 2>&1; then
      openclaw_verify_gate_resolve_strict_autonomy "$strict_autonomy_override"
      resolved_enabled="${OPENCLAW_VERIFY_GATE_RESOLVED_ENABLED:-0}"
      resolved_required="${OPENCLAW_VERIFY_GATE_RESOLVED_REQUIRED:-0}"
      resolved_source="${OPENCLAW_VERIFY_GATE_RESOLVED_SOURCE:-disabled}"
    fi
  fi

  if [[ "$resolved_source" == "disabled" && "$resolved_enabled" == "0" && "$resolved_required" == "0" ]]; then
    local strict_autonomy_policy_raw="${OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REGRESSIONS-}"
    local strict_autonomy_legacy_raw="${OPENCLAW_STRICT_AUTONOMY_REGRESSIONS-}"
    local strict_autonomy_required_raw="${OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REQUIRED:-0}"
    local strict_autonomy_policy_set="0"
    local strict_autonomy_legacy_set="0"
    local strict_autonomy="1"

    if [[ -n "${OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REGRESSIONS+x}" ]]; then
      strict_autonomy_policy_set="1"
    fi
    if [[ -n "${OPENCLAW_STRICT_AUTONOMY_REGRESSIONS+x}" ]]; then
      strict_autonomy_legacy_set="1"
    fi

    resolved_source="default_on"
    if [[ "$strict_autonomy_policy_set" == "1" ]]; then
      strict_autonomy="$strict_autonomy_policy_raw"
      resolved_source="verify_gate_policy_env"
    elif [[ "$strict_autonomy_legacy_set" == "1" ]]; then
      strict_autonomy="$strict_autonomy_legacy_raw"
      resolved_source="legacy_env"
    fi

    if [[ "$strict_autonomy_override" == "1" ]]; then
      strict_autonomy="1"
      resolved_source="wrapper_flag_enable"
    elif [[ "$strict_autonomy_override" == "0" ]]; then
      strict_autonomy="0"
      resolved_source="wrapper_flag_disable"
    fi

    if continuity_truthy "$strict_autonomy_required_raw"; then
      resolved_required="1"
      strict_autonomy="1"
      resolved_source="verify_gate_required_env"
    fi

    if continuity_truthy "$strict_autonomy"; then
      resolved_enabled="1"
    fi
  fi

  export OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_EFFECTIVE_ENABLED="$resolved_enabled"
  export OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_EFFECTIVE_REQUIRED="$resolved_required"
  export OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_EFFECTIVE_SOURCE="$resolved_source"
}

prepare_session_route_transport_conformance_args() {
  SESSION_ROUTE_ARGS=()

  local legacy_override="0"
  local saw_deprecated_legacy_flag="0"
  local saw_deprecated_require_flag="0"
  local worker_legacy_override="0"
  local worker_require_override="0"
  local arg=""

  for arg in "$@"; do
    case "$arg" in
      --legacy-allow-missing-transport-decision)
        legacy_override="1"
        ;;
      --legacy-allow-missing-worker-allocation-contract)
        worker_legacy_override="1"
        ;;
      --require-worker-allocation-contract)
        worker_require_override="1"
        ;;
      --allow-missing-transport-decision|--no-require-transport-decision)
        saw_deprecated_legacy_flag="1"
        ;;
      --require-transport-decision)
        saw_deprecated_require_flag="1"
        ;;
      *)
        SESSION_ROUTE_ARGS+=("$arg")
        ;;
    esac
  done

  local require_raw="${OPENCLAW_SESSION_ROUTE_REQUIRE_TRANSPORT_DECISION:-1}"
  local require_required_raw="${OPENCLAW_SESSION_ROUTE_REQUIRE_TRANSPORT_DECISION_REQUIRED:-0}"
  local worker_require_raw="${OPENCLAW_SESSION_ROUTE_REQUIRE_WORKER_ALLOCATION_CONTRACT:-1}"
  local worker_require_required_raw="${OPENCLAW_SESSION_ROUTE_REQUIRE_WORKER_ALLOCATION_CONTRACT_REQUIRED:-0}"

  local effective_required="1"
  local source="default_strict"
  local worker_effective_required="1"
  local worker_source="default_strict"

  if [[ -n "${OPENCLAW_SESSION_ROUTE_REQUIRE_TRANSPORT_DECISION+x}" ]]; then
    if continuity_truthy "$require_raw"; then
      source="policy_env_strict"
    else
      source="legacy_policy_env_disable_ignored"
    fi
  fi

  if [[ "$saw_deprecated_require_flag" == "1" ]]; then
    source="deprecated_require_flag_noop"
  fi

  if [[ "$saw_deprecated_legacy_flag" == "1" && "$legacy_override" != "1" ]]; then
    source="deprecated_legacy_flag_ignored"
  fi

  if [[ "$legacy_override" == "1" ]]; then
    effective_required="0"
    source="wrapper_flag_legacy_bypass"
  fi

  if continuity_truthy "$require_required_raw"; then
    effective_required="1"
    source="policy_required_env"
  fi

  if [[ -n "${OPENCLAW_SESSION_ROUTE_REQUIRE_WORKER_ALLOCATION_CONTRACT+x}" ]]; then
    if continuity_truthy "$worker_require_raw"; then
      worker_source="policy_env_strict"
    else
      worker_effective_required="0"
      worker_source="policy_env_disable"
    fi
  fi

  if [[ "$worker_require_override" == "1" ]]; then
    worker_effective_required="1"
    worker_source="wrapper_flag_require_worker_contract"
  fi

  if [[ "$worker_legacy_override" == "1" ]]; then
    worker_effective_required="0"
    worker_source="wrapper_flag_legacy_worker_contract_bypass"
  fi

  if continuity_truthy "$worker_require_required_raw"; then
    worker_effective_required="1"
    worker_source="policy_required_env"
  fi

  if [[ "$effective_required" != "1" ]]; then
    SESSION_ROUTE_ARGS+=("--legacy-allow-missing-transport-decision")
  fi

  if [[ "$worker_effective_required" == "1" ]]; then
    SESSION_ROUTE_ARGS+=("--require-worker-allocation-contract")
  else
    SESSION_ROUTE_ARGS+=("--legacy-allow-missing-worker-allocation-contract")
  fi

  export OPENCLAW_SESSION_ROUTE_TRANSPORT_CONFORMANCE_EFFECTIVE_REQUIRED="$effective_required"
  export OPENCLAW_SESSION_ROUTE_TRANSPORT_CONFORMANCE_EFFECTIVE_SOURCE="$source"
  export OPENCLAW_SESSION_ROUTE_WORKER_ALLOCATION_CONTRACT_EFFECTIVE_REQUIRED="$worker_effective_required"
  export OPENCLAW_SESSION_ROUTE_WORKER_ALLOCATION_CONTRACT_EFFECTIVE_SOURCE="$worker_source"
}

case "$cmd" in
  now)
    exec "$CONT_DIR/continuity_now.sh" "$@" ;;
  current)
    exec "$CONT_DIR/continuity_current.sh" "$@" ;;
  worker-health-canary)
    exec python3 "$CONT_DIR/execution_supervisor_worker_health_canary.py" "$@" ;;
  execution-status)
    exec "$CONT_DIR/execution_program_status.sh" "$@" ;;
  execution-frontier)
    exec "$CONT_DIR/execution_frontier_ledger.sh" "$@" ;;
  execution-frontier-antistall-check)
    exec python3 "$CONT_DIR/check_execution_frontier_antistall_regressions.py" "$@" ;;
  routing-preflight-route-check)
    exec python3 "$CONT_DIR/check_routing_preflight_route_decision_operator_surface.py" "$@" ;;
  handover)
    exec "$CONT_DIR/handover_latest.sh" "$@" ;;
  reset-ready-refresh)
    exec "$CONT_DIR/reset_ready_refresh.sh" "$@" ;;
  blocker-registry)
    exec "$CONT_DIR/blocker_registry.sh" "$@" ;;
  mission-control)
    exec "$CONT_DIR/operator_mission_control.sh" "$@" ;;
  triage-console)
    exec "$CONT_DIR/operator_triage_console.sh" "$@" ;;
  federated-evidence-gate)
    exec python3 "$ROOT/scripts/federated_evidence_packet_gate.py" "$@" ;;
  gate-os)
    exec "$CONT_DIR/gate_os_snapshot.sh" "$@" ;;
  history)
    exec "$CONT_DIR/history.sh" "$@" ;;
  reconcile)
    exec "$CONT_DIR/reconcile.sh" "$@" ;;
  verify)
    verify_args=()
    for arg in "$@"; do
      if [[ "$arg" == "--json" ]]; then
        continue
      fi
      verify_args+=("$arg")
    done
    prepare_verify_strict_autonomy_context "${verify_args[@]}"
    exec "$CONT_DIR/verify_then_resume.sh" "${verify_args[@]}" ;;
  verify-gate-status)
    exec "$CONT_DIR/verify_gate_status.sh" "$@" ;;
  restore-drill-refresh)
    exec "$CONT_DIR/restore_drill_refresh.sh" "$@" ;;
  sync)
    exec "$CONT_DIR/sync_latest_artifacts.sh" "$@" ;;
  checkpoint)
    exec "$CONT_DIR/write_checkpoint.sh" "$@" ;;
  queue-sync)
    exec "$CONT_DIR/queue_sync_from_autopilot_json.sh" "$@" ;;
  queue-arb)
    exec "$CONT_DIR/queue_arbitrator.sh" "$@" ;;
  queue-replay)
    exec "$CONT_DIR/queue_replay_verify.sh" "$@" ;;
  repo-review-closeout)
    exec python3 "$CONT_DIR/check_repo_review_queue_closeout.py" "$@" ;;
  failover-replay-evidence)
    exec python3 "$CONT_DIR/failover_replay_evidence.py" "$@" ;;
  failover-stress-soak)
    exec python3 "$CONT_DIR/failover_stress_soak.py" "$@" ;;
  failover-stress-runtime-evidence)
    exec python3 "$CONT_DIR/failover_stress_runtime_evidence.py" "$@" ;;
  failover-stress-runtime-check)
    exec python3 "$CONT_DIR/check_a3_failover_runtime_evidence.py" "$@" ;;
  a6-multi-host-jitter-check)
    exec python3 "$CONT_DIR/a6_multi_host_jitter_harness.py" "$@" ;;
  lock-break)
    exec "$CONT_DIR/lock_break.sh" "$@" ;;
  librarian)
    exec "$CONT_DIR/librarian_curator.sh" "$@" ;;
  promotion-gate)
    exec python3 "$ROOT/scripts/promotion_gate_runner.py" "$@" ;;
  model-rollout-gate)
    exec python3 "$ROOT/scripts/model_rollout_gate_runner.py" "$@" ;;
  model-rollout-controller)
    exec python3 "$ROOT/scripts/model_rollout_ledger_controller.py" "$@" ;;
  model-rollout-health)
    exec python3 "$ROOT/scripts/model_rollout_health_snapshot.py" "$@" ;;
  model-rollout-cost)
    exec python3 "$ROOT/scripts/model_rollout_cost_governance_snapshot.py" "$@" ;;
  model-route-policy-lint)
    exec python3 "$ROOT/scripts/model_route_policy_soak_lint.py" "$@" ;;
  model-rollout-soak)
    exec python3 "$ROOT/scripts/model_rollout_ring_soak_snapshot.py" "$@" ;;
  model-rollout-dashboard)
    exec python3 "$ROOT/scripts/model_rollout_dashboard_snapshot.py" "$@" ;;
  session-route)
    prepare_session_route_transport_conformance_args "$@"
    exec python3 "$ROOT/scripts/session_topology_router.py" "${SESSION_ROUTE_ARGS[@]}" ;;
  session-transport-route)
    exec python3 "$ROOT/scripts/session_topology_transport_router.py" "$@" ;;
  knowledge-queue)
    exec python3 "$ROOT/scripts/knowledge_review_queue.py" "$@" ;;
  promotion-queue)
    exec python3 "$ROOT/scripts/knowledge_promotion_queue.py" "$@" ;;
  shared-memory)
    exec python3 "$ROOT/scripts/shared_memory_fabric.py" "$@" ;;
  markdown-gate)
    exec python3 "$ROOT/scripts/markdown_conversion_quality_gate.py" "$@" ;;
  material-classify)
    exec python3 "$ROOT/scripts/source_material_classifier.py" "$@" ;;
  knowledge-ingest)
    exec python3 "$ROOT/scripts/production_knowledge_ingestion.py" "$@" ;;
  doc-intake-closeout)
    exec python3 "$ROOT/scripts/document_intake_batch_integration_gate.py" "$@" ;;
  release-evidence-gate)
    exec python3 "$ROOT/scripts/release_evidence_ladder_gate.py" "$@" ;;
  lane-crossover-guard)
    exec python3 "$CONT_DIR/lane_crossover_ingress_guard.py" "$@" ;;
  bridge-ingest)
    exec python3 "$CONT_DIR/cross_lane_bridge_ingest.py" "$@" ;;
  source-material-guard)
    exec python3 "$CONT_DIR/source_material_classification_guard.py" "$@" ;;
  db-check)
    exec "$CONT_DIR/db_integrity_check.sh" "$@" ;;
  swarm-check)
    exec "$CONT_DIR/swarm_runtime_check.sh" "$@" ;;
  slot-fill-check)
    exec "$CONT_DIR/check_slot_fill_protocol.sh" "$@" ;;
  autonomy-regressions)
    exec "$CONT_DIR/check_autonomy_continuity_regressions.py" "$@" ;;
  autonomy-smoke)
    exec "$CONT_DIR/check_autonomy_continuity_regressions.py" --profile smoke "$@" ;;
  parity-run)
    exec "$ROOT/ops/openclaw/architecture/run_competitive_parity_harness.sh" "$@" ;;
  web-capture)
    exec "$ROOT/ops/openclaw/run_web_capture_macro.sh" "$@" ;;
  web-capture-scheduler)
    exec "$ROOT/ops/openclaw/run_web_capture_scheduler.sh" "$@" ;;
  gtc-sync)
    exec "$CONT_DIR/gtc_v2_sync.sh" "$@" ;;
  gtc-schema-check)
    exec "$CONT_DIR/gtc_latest_schema_check.sh" "$@" ;;
  gtc-replay)
    exec "$CONT_DIR/gtc_incident_replay.sh" "$@" ;;
  normalize-events)
    exec "$CONT_DIR/normalize_event_sources.sh" "$@" ;;
  -h|--help|help)
    usage
    exit 0 ;;
  *)
    echo "unknown continuity command: $cmd" >&2
    usage >&2
    exit 2 ;;
esac
