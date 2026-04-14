#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
TARGET_SESSION_KEY="${OPENCLAW_TARGET_SESSION_KEY:-agent:codex-orchestrator-pro:telegram:direct:5936691533}"
TARGET_AGENT_ID="${OPENCLAW_TARGET_AGENT_ID:-$(printf '%s' "$TARGET_SESSION_KEY" | cut -d: -f2)}"
if [[ -z "$TARGET_AGENT_ID" || "$TARGET_AGENT_ID" == "$TARGET_SESSION_KEY" ]]; then
  TARGET_AGENT_ID="codex-orchestrator-pro"
fi
SESSION_STORE_PATH="${OPENCLAW_SESSION_STORE_PATH:-/home/yeqiuqiu/.openclaw/agents/${TARGET_AGENT_ID}/sessions/sessions.json}"
THRESHOLD_PCT="${OPENCLAW_CONTEXT_THRESHOLD_PCT:-0.85}"
PREVENTIVE_THRESHOLD_PCT="${OPENCLAW_CONTEXT_PREVENTIVE_THRESHOLD_PCT:-0.80}"
CHECKPOINT_TRIGGER="${OPENCLAW_CONTEXT_CHECKPOINT_TRIGGER:-context_preoverflow}"
BLOAT_THRESHOLD_MB="${OPENCLAW_SESSION_BLOAT_THRESHOLD_MB:-50}"
MARKER_FILE_DEFAULT="${OPENCLAW_CONTEXT_MARKER_FILE:-/tmp/context_watch_threshold_fired}"
EVENT_COOLDOWN_SEC="${OPENCLAW_CONTEXT_WATCH_EVENT_COOLDOWN_SEC:-1800}"
# Default to explicit READY/PROGRESS first-line protocol so wrappers can always
# extract a deterministic status line. Set OPENCLAW_CONTEXT_WATCH_EMIT_PROTOCOL_LINES=0
# to emit INTERNAL_STATUS lines instead.
EMIT_PROTOCOL_LINES="${OPENCLAW_CONTEXT_WATCH_EMIT_PROTOCOL_LINES:-1}"

MARKER_FILE="$MARKER_FILE_DEFAULT"
DRY_RUN=0
FORCE_CONTEXT_HIGH=0
FORCE_SESSION_BLOAT=0
FORCE_RUNTIME_UNHEALTHY=0

usage() {
  cat <<'EOF'
Usage: context_runtime_local_watch.sh [options]

Local-first context/session/runtime watcher.
- Uses ground-truth snapshot + deterministic host data.
- Routes events through shared event router fingerprint state.
- Generates structured continuity checkpoint + compatibility handover when context threshold is hit.

Options:
  --dry-run                 Evaluate and route events, but do not create checkpoint/handover/marker/memory writes.
  --marker-file <path>      Marker file path (default: /tmp/context_watch_threshold_fired)
  --force-context-high      Test mode: force context-high condition true.
  --force-session-bloat     Test mode: force session-bloat condition true.
  --force-runtime-unhealthy Test mode: force runtime-unhealthy condition true.
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1; shift ;;
    --marker-file)
      MARKER_FILE="${2:-}"; shift 2 ;;
    --force-context-high)
      FORCE_CONTEXT_HIGH=1; shift ;;
    --force-session-bloat)
      FORCE_SESSION_BLOAT=1; shift ;;
    --force-runtime-unhealthy)
      FORCE_RUNTIME_UNHEALTHY=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

EVENT_ROUTER="$ROOT/ops/openclaw/event_router.sh"
BLOCKER_ROUTING_LIB="$ROOT/ops/openclaw/lib/blocker_routing.sh"
SNAPSHOT_SCRIPT="$ROOT/ops/openclaw/snapshot_ground_truth.sh"
CHECKPOINT_SCRIPT="$ROOT/ops/openclaw/continuity/write_checkpoint.sh"
HANDOVER_COMPAT_SCRIPT="$ROOT/ops/openclaw/continuity/render_context_handover_compat.sh"
SYNC_LATEST_SCRIPT="$ROOT/ops/openclaw/continuity/sync_latest_artifacts.sh"
NO_NUDGE_CRON_GUARD_SCRIPT="$ROOT/ops/openclaw/no_nudge_continuity_cron_guard.sh"
SCHEDULER_GOVERNANCE_GUARD_SCRIPT="$ROOT/ops/openclaw/web_capture_scheduler_governance_guard.sh"
NO_NUDGE_GUARD_PROTOCOL_LIB="$ROOT/ops/openclaw/lib/no_nudge_guard_protocol.sh"

if [[ ! -f "$BLOCKER_ROUTING_LIB" ]]; then
  echo "BLOCKER: missing blocker routing lib: $BLOCKER_ROUTING_LIB"
  exit 0
fi

# shellcheck source=ops/openclaw/lib/blocker_routing.sh
source "$BLOCKER_ROUTING_LIB"

if [[ ! -f "$NO_NUDGE_GUARD_PROTOCOL_LIB" ]]; then
  echo "BLOCKER: missing no-nudge guard protocol lib: $NO_NUDGE_GUARD_PROTOCOL_LIB"
  exit 0
fi

# shellcheck source=ops/openclaw/lib/no_nudge_guard_protocol.sh
source "$NO_NUDGE_GUARD_PROTOCOL_LIB"

OPENCLAW_BLOCKER_EVENT_ROUTER="$EVENT_ROUTER"
OPENCLAW_BLOCKER_SOURCE="watchdog.context_runtime_local"
OPENCLAW_BLOCKER_COOLDOWN_SEC="$EVENT_COOLDOWN_SEC"
OPENCLAW_BLOCKER_DEFAULT_EVIDENCE_REF="$ROOT/state/ground_truth/latest.json"

emit_blocker() {
  local key="${1:-context_runtime_local_blocker}"
  local summary="${2:-task=context_runtime_local_watch; reason=unknown_blocker}"
  local evidence_ref="${3:-$OPENCLAW_BLOCKER_DEFAULT_EVIDENCE_REF}"
  if declare -F openclaw_watchdog_route_blocker >/dev/null 2>&1; then
    local routed_line=""
    set +e
    routed_line="$(openclaw_watchdog_route_blocker "$key" "$summary" "$evidence_ref")"
    set -e
    if [[ -n "$routed_line" ]]; then
      printf '%s\n' "$routed_line"
    else
      printf 'BLOCKER: %s\n' "$summary"
    fi
  else
    printf 'BLOCKER: %s\n' "$summary"
  fi
}

emit_status() {
  local level="${1:-INFO}"
  local summary="${2:-task=context_runtime_local_watch; status=ok}"
  if [[ "$EMIT_PROTOCOL_LINES" == "1" ]]; then
    printf '%s: %s\n' "$level" "$summary"
  else
    printf 'INTERNAL_STATUS: level=%s; %s\n' "$level" "$summary"
  fi
}

if ! declare -F openclaw_no_nudge_guard_first_line >/dev/null 2>&1; then
  emit_blocker "no_nudge_cron_guard_protocol_helper_missing" "task=context_runtime_local_watch; reason=missing_no_nudge_guard_protocol_helper" "$NO_NUDGE_GUARD_PROTOCOL_LIB"
  exit 0
fi
if ! declare -F openclaw_protocol_first_non_empty_line >/dev/null 2>&1 \
  || ! declare -F openclaw_protocol_sanitize_inline >/dev/null 2>&1 \
  || ! declare -F openclaw_protocol_line_is_guard_ready_or_blocker >/dev/null 2>&1; then
  emit_blocker "scheduler_governance_guard_protocol_helper_missing" "task=context_runtime_local_watch; reason=missing_protocol_accept_helper" "$NO_NUDGE_GUARD_PROTOCOL_LIB"
  exit 0
fi

if [[ ! -x "$EVENT_ROUTER" ]]; then
  emit_blocker "event_router_missing" "task=context_runtime_local_watch; reason=missing_event_router" "$EVENT_ROUTER"
  exit 0
fi
if [[ ! -x "$SNAPSHOT_SCRIPT" ]]; then
  emit_blocker "snapshot_script_missing" "task=context_runtime_local_watch; reason=missing_snapshot_script" "$SNAPSHOT_SCRIPT"
  exit 0
fi
if [[ ! -x "$CHECKPOINT_SCRIPT" ]]; then
  emit_blocker "checkpoint_script_missing" "task=context_runtime_local_watch; reason=missing_checkpoint_writer" "$CHECKPOINT_SCRIPT"
  exit 0
fi
if [[ ! -x "$HANDOVER_COMPAT_SCRIPT" ]]; then
  emit_blocker "handover_script_missing" "task=context_runtime_local_watch; reason=missing_handover_renderer" "$HANDOVER_COMPAT_SCRIPT"
  exit 0
fi
if [[ ! -x "$SYNC_LATEST_SCRIPT" ]]; then
  emit_blocker "sync_script_missing" "task=context_runtime_local_watch; reason=missing_sync_latest" "$SYNC_LATEST_SCRIPT"
  exit 0
fi
if [[ ! -x "$NO_NUDGE_CRON_GUARD_SCRIPT" ]]; then
  emit_blocker "no_nudge_cron_guard_missing" "task=context_runtime_local_watch; reason=missing_no_nudge_cron_guard" "$NO_NUDGE_CRON_GUARD_SCRIPT"
  exit 0
fi
if [[ ! -x "$SCHEDULER_GOVERNANCE_GUARD_SCRIPT" ]]; then
  emit_blocker "scheduler_governance_guard_missing" "task=context_runtime_local_watch; reason=missing_scheduler_governance_guard" "$SCHEDULER_GOVERNANCE_GUARD_SCRIPT"
  exit 0
fi

guard_line="$(openclaw_no_nudge_guard_first_line "$NO_NUDGE_CRON_GUARD_SCRIPT" "/tmp/context_runtime_local_watch_guard.err")"
if [[ "$guard_line" == BLOCKER:* ]]; then
  emit_blocker "no_nudge_cron_guard_blocker" "task=context_runtime_local_watch; reason=no_nudge_cron_guard_blocker; detail=${guard_line#BLOCKER: }" "$NO_NUDGE_CRON_GUARD_SCRIPT"
  exit 0
fi

set +e
scheduler_guard_out="$($SCHEDULER_GOVERNANCE_GUARD_SCRIPT 2>/tmp/context_runtime_local_watch_scheduler_guard.err)"
scheduler_guard_rc=$?
set -e
scheduler_guard_first_line="$(openclaw_protocol_first_non_empty_line "$scheduler_guard_out")"
if [[ "$scheduler_guard_rc" -ne 0 ]]; then
  err="$(cat /tmp/context_runtime_local_watch_scheduler_guard.err 2>/dev/null || true)"
  emit_blocker "scheduler_governance_guard_exec_failed" "task=context_runtime_local_watch; reason=scheduler_governance_guard_exec_failed; rc=${scheduler_guard_rc}; err=${err:0:180}" "$SCHEDULER_GOVERNANCE_GUARD_SCRIPT"
  exit 0
fi
if [[ -z "$scheduler_guard_first_line" ]]; then
  emit_blocker "scheduler_governance_guard_invalid_protocol" "task=context_runtime_local_watch; reason=scheduler_governance_guard_invalid_protocol; detail=empty_first_line" "$SCHEDULER_GOVERNANCE_GUARD_SCRIPT"
  exit 0
fi
if ! openclaw_protocol_line_is_guard_ready_or_blocker "$scheduler_guard_first_line"; then
  sanitized_scheduler_guard_line="$(openclaw_protocol_sanitize_inline "$scheduler_guard_first_line")"
  emit_blocker "scheduler_governance_guard_invalid_protocol" "task=context_runtime_local_watch; reason=scheduler_governance_guard_invalid_protocol; detail=unexpected_first_line; first_line=${sanitized_scheduler_guard_line:0:180}" "$SCHEDULER_GOVERNANCE_GUARD_SCRIPT"
  exit 0
fi
if [[ "$scheduler_guard_first_line" == BLOCKER:* ]]; then
  emit_blocker "scheduler_governance_guard_blocker" "task=context_runtime_local_watch; reason=scheduler_governance_guard_blocker; detail=${scheduler_guard_first_line#BLOCKER: }" "$SCHEDULER_GOVERNANCE_GUARD_SCRIPT"
  exit 0
fi

snapshot_out="$($SNAPSHOT_SCRIPT 2>/tmp/context_runtime_local_watch_snapshot.err)" || {
  err="$(cat /tmp/context_runtime_local_watch_snapshot.err 2>/dev/null || true)"
  emit_blocker "snapshot_failed" "task=context_runtime_local_watch; reason=snapshot_ground_truth_failed; err=${err:0:180}" "$SNAPSHOT_SCRIPT"
  exit 0
}

# Keep machine-readable continuity latest bridge current with runtime truth even when no new checkpoint is written.
set +e
OPENCLAW_INTERNAL_MUTATION=1 OPENCLAW_INTERNAL_MUTATION_CALLSITE="context_runtime_local_watch.sh:sync_latest_artifacts" "$SYNC_LATEST_SCRIPT" --skip-render >/tmp/context_runtime_local_watch_sync.err 2>&1
sync_rc=$?
set -e
if [[ "$sync_rc" -ne 0 ]]; then
  # Best-effort only; keep watcher low-noise when latest pointer does not exist yet.
  :
fi

state_json="$(python3 - "$ROOT" "$TARGET_SESSION_KEY" "$TARGET_AGENT_ID" "$SESSION_STORE_PATH" "$THRESHOLD_PCT" "$PREVENTIVE_THRESHOLD_PCT" "$BLOAT_THRESHOLD_MB" "$FORCE_CONTEXT_HIGH" "$FORCE_SESSION_BLOAT" "$FORCE_RUNTIME_UNHEALTHY" <<'PY'
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
target_key = sys.argv[2]
target_agent_id = str(sys.argv[3] or "").strip() or "codex-orchestrator-pro"
session_store_path = pathlib.Path(sys.argv[4]).expanduser()
threshold = float(sys.argv[5])
preventive_threshold = float(sys.argv[6])
bloat_threshold_mb = int(sys.argv[7])
force_context_high = bool(int(sys.argv[8]))
force_session_bloat = bool(int(sys.argv[9]))
force_runtime_unhealthy = bool(int(sys.argv[10]))

if preventive_threshold > threshold:
    preventive_threshold = threshold

latest_path = root / "state" / "ground_truth" / "latest.json"
latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.exists() else {}
snapshot_rel = str(latest.get("snapshot_path") or "")
snapshot_path = (root / snapshot_rel).resolve() if snapshot_rel else pathlib.Path("")
snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_rel and snapshot_path.exists() else {}

sessions = snapshot.get("sessions") or {}
target_session = sessions.get("target_session") or {}
pct = sessions.get("target_session_pct")
if pct is None:
    total = float(target_session.get("totalTokens") or 0)
    context = float(target_session.get("contextTokens") or 0)
    pct = (total / context) if context > 0 else 0.0

session_file_size_bytes = None
session_store = session_store_path
if session_store.exists():
    try:
        sessions_map = json.loads(session_store.read_text(encoding="utf-8"))
        meta = sessions_map.get(target_key) or {}
        sf = pathlib.Path(str(meta.get("sessionFile") or ""))
        if sf.exists() and sf.is_file():
            session_file_size_bytes = sf.stat().st_size
    except Exception:
        session_file_size_bytes = None

if session_file_size_bytes is None:
    session_file_size_bytes = int(target_session.get("sessionFileSizeBytes") or 0)

session_bloat = (session_file_size_bytes / (1024 * 1024)) >= bloat_threshold_mb if session_file_size_bytes else False

anomalies = snapshot.get("anomalies") or []
runtime_unhealthy = any((a.get("key") == "gateway_unhealthy") for a in anomalies if isinstance(a, dict))

if force_context_high:
    pct = max(pct or 0.0, threshold)
if force_session_bloat:
    session_bloat = True
if force_runtime_unhealthy:
    runtime_unhealthy = True

context_high = (pct or 0.0) >= threshold
context_preventive = (pct or 0.0) >= preventive_threshold and not context_high

if context_high:
    context_state = "high"
elif context_preventive:
    context_state = "preventive"
else:
    context_state = "normal"

fingerprint_input = "|".join(
    [
        f"context_state={context_state}",
        f"session_bloat_state={'high' if session_bloat else 'normal'}",
        f"runtime_state={'unhealthy' if runtime_unhealthy else 'healthy'}",
        f"preventive_threshold={preventive_threshold:.3f}",
        f"threshold={threshold:.3f}",
        f"bloat_mb_threshold={bloat_threshold_mb}",
    ]
)

result = {
    "snapshot_id": snapshot.get("snapshot_id"),
    "snapshot_path": snapshot_rel,
    "target_session_key": target_key,
    "target_agent_id": target_agent_id,
    "pct": float(pct or 0.0),
    "threshold_pct": threshold,
    "preventive_threshold_pct": preventive_threshold,
    "context_high": context_high,
    "context_preventive": context_preventive,
    "session_file_size_bytes": int(session_file_size_bytes or 0),
    "session_bloat": bool(session_bloat),
    "bloat_threshold_mb": int(bloat_threshold_mb),
    "runtime_unhealthy": bool(runtime_unhealthy),
    "fingerprint_input": fingerprint_input,
}
print(json.dumps(result, ensure_ascii=False))
PY
)"

context_high="$(python3 -c 'import json,sys; print("1" if json.loads(sys.argv[1]).get("context_high") else "0")' "$state_json")"
context_preventive="$(python3 -c 'import json,sys; print("1" if json.loads(sys.argv[1]).get("context_preventive") else "0")' "$state_json")"
session_bloat="$(python3 -c 'import json,sys; print("1" if json.loads(sys.argv[1]).get("session_bloat") else "0")' "$state_json")"
runtime_unhealthy="$(python3 -c 'import json,sys; print("1" if json.loads(sys.argv[1]).get("runtime_unhealthy") else "0")' "$state_json")"
pct_str="$(python3 -c 'import json,sys; obj=json.loads(sys.argv[1]); print("{:.6f}".format(float(obj.get("pct",0) or 0.0)))' "$state_json")"
snapshot_path_rel="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("snapshot_path") or "")' "$state_json")"
fingerprint_input="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("fingerprint_input") or "")' "$state_json")"

if [[ "$context_high" -eq 0 && -f "$MARKER_FILE" ]]; then
  rm -f "$MARKER_FILE"
fi

event_severity="info"
if [[ "$runtime_unhealthy" -eq 1 ]]; then
  event_severity="critical"
elif [[ "$context_high" -eq 1 || "$context_preventive" -eq 1 || "$session_bloat" -eq 1 ]]; then
  event_severity="warn"
fi

set +e
openclaw_route_event \
  --event-router "$EVENT_ROUTER" \
  --source "local.context_runtime_watch" \
  --key "session_context_runtime" \
  --severity "$event_severity" \
  --summary "context_high=${context_high};context_preventive=${context_preventive};session_bloat=${session_bloat};runtime_unhealthy=${runtime_unhealthy};pct=${pct_str}" \
  --evidence-ref "$snapshot_path_rel" \
  --fingerprint-input "$fingerprint_input" \
  --cooldown-sec "$EVENT_COOLDOWN_SEC" \
  2>/tmp/context_runtime_local_watch_router.err
router_rc=$?
set -e

if [[ "$router_rc" -ne 0 && "$router_rc" -ne 20 ]]; then
  err="$(cat /tmp/context_runtime_local_watch_router.err 2>/dev/null || true)"
  emit_blocker "event_router_failed" "task=context_runtime_local_watch; reason=event_router_failed_${router_rc}; err=${err:0:160}" "$EVENT_ROUTER"
  exit 0
fi

emit_router=0
if [[ "$router_rc" -eq 0 ]]; then
  emit_router=1
fi

if [[ "$runtime_unhealthy" -eq 1 && "$emit_router" -eq 1 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    emit_blocker "runtime_unhealthy_dry_run" "task=context_runtime_local_watch; reason=runtime_unhealthy_dry_run; snapshot=${snapshot_path_rel}" "$snapshot_path_rel"
    exit 0
  fi

  set +e
  runtime_chk_out="$(OPENCLAW_INTERNAL_MUTATION=1 OPENCLAW_INTERNAL_MUTATION_CALLSITE="context_runtime_local_watch.sh:runtime_unhealthy_checkpoint" "$CHECKPOINT_SCRIPT" \
    --trigger runtime_truth_critical \
    --status BLOCKER \
    --objective "Local runtime truth indicates gateway/rpc unhealthy" \
    --blocker-reason "ground_truth anomaly gateway_unhealthy" \
    --next-action "openclaw gateway status --json" \
    --next-action "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/watchdog_gateway_health.sh" \
    --verify-cmd "openclaw gateway status --json >/dev/null" 2>/tmp/context_runtime_local_watch_runtime_checkpoint.err)"
  rc_runtime_chk=$?
  set -e

  if [[ "$rc_runtime_chk" -ne 0 ]]; then
    err="$(cat /tmp/context_runtime_local_watch_runtime_checkpoint.err 2>/dev/null || true)"
    emit_blocker "runtime_unhealthy_checkpoint_failed" "task=context_runtime_local_watch; reason=runtime_unhealthy_checkpoint_write_failed; err=${err:0:160}" "$CHECKPOINT_SCRIPT"
    exit 0
  fi

  checkpoint_id="$(python3 -c 'import json,sys; print((json.loads(sys.argv[1]).get("checkpoint_id") or ""))' "$runtime_chk_out")"
  "$HANDOVER_COMPAT_SCRIPT" --checkpoint "$checkpoint_id" >/dev/null 2>&1 || true
  emit_blocker "runtime_unhealthy" "task=context_runtime_local_watch; reason=runtime_unhealthy; checkpoint=${checkpoint_id}; snapshot=${snapshot_path_rel}" "$snapshot_path_rel"
  exit 0
fi

need_context_action=0
if [[ "$context_high" -eq 1 ]]; then
  if [[ ! -f "$MARKER_FILE" ]]; then
    need_context_action=1
  elif [[ ! -e "$ROOT/state/continuity/latest/handover_latest.json" ]]; then
    need_context_action=1
  fi
fi

if [[ "$need_context_action" -eq 1 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    emit_status "READY" "context_high detected (dry-run); pct=${pct_str}; snapshot=${snapshot_path_rel}"
    exit 0
  fi

  set +e
  checkpoint_out="$(OPENCLAW_INTERNAL_MUTATION=1 OPENCLAW_INTERNAL_MUTATION_CALLSITE="context_runtime_local_watch.sh:context_high_checkpoint" "$CHECKPOINT_SCRIPT" \
    --trigger "$CHECKPOINT_TRIGGER" \
    --status PROGRESS \
    --objective "Telegram direct lane near overflow; checkpoint for thin-lane successor continuity (context >= ${THRESHOLD_PCT})." \
    --next-action "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/verify_then_resume.sh" \
    --next-action "openclaw sessions --agent ${TARGET_AGENT_ID} --active 1440 --json" \
    --next-action "echo 'Telegram DM thin lane only: keep cockpit summaries/approvals here; run heavy orchestration in worker sessions.'" \
    --verify-cmd "openclaw sessions --agent ${TARGET_AGENT_ID} --active 1440 --json >/dev/null" \
    --verify-cmd "openclaw gateway status --json >/dev/null" \
    --verify-cmd "openclaw cron list --json >/dev/null" 2>/tmp/context_runtime_local_watch_checkpoint.err)"
  checkpoint_rc=$?
  set -e

  if [[ "$checkpoint_rc" -ne 0 ]]; then
    err="$(cat /tmp/context_runtime_local_watch_checkpoint.err 2>/dev/null || true)"
    emit_blocker "context_high_checkpoint_failed" "task=context_runtime_local_watch; reason=context_high_checkpoint_write_failed; err=${err:0:160}" "$CHECKPOINT_SCRIPT"
    exit 0
  fi

  checkpoint_id="$(python3 -c 'import json,sys; print((json.loads(sys.argv[1]).get("checkpoint_id") or ""))' "$checkpoint_out")"

  set +e
  handover_out="$($HANDOVER_COMPAT_SCRIPT --checkpoint "$checkpoint_id" 2>/tmp/context_runtime_local_watch_handover.err)"
  handover_rc=$?
  set -e

  if [[ "$handover_rc" -ne 0 ]]; then
    err="$(cat /tmp/context_runtime_local_watch_handover.err 2>/dev/null || true)"
    emit_blocker "context_high_handover_failed" "task=context_runtime_local_watch; reason=context_high_handover_render_failed; checkpoint=${checkpoint_id}; err=${err:0:160}" "$HANDOVER_COMPAT_SCRIPT"
    exit 0
  fi

  mkdir -p "$ROOT/memory"
  mem_file="$ROOT/memory/$(date +%F).md"
  printf '%s\n' "- [context-watch] local-first watcher fired at $(date -Iseconds): pct=${pct_str}; checkpoint=${checkpoint_id}" >> "$mem_file"

  touch "$MARKER_FILE"

  handover_rel="$(python3 -c 'import json,sys; print((json.loads(sys.argv[1]).get("handover_path") or "reports/handover_context_latest.md"))' "$handover_out")"
  emit_status "READY" "context>=${THRESHOLD_PCT}; pct=${pct_str}; checkpoint=${checkpoint_id}; handover=${handover_rel}; thin_lane=telegram_dm; recommend /reset"
  exit 0
fi

if [[ "$session_bloat" -eq 1 && "$emit_router" -eq 1 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    emit_status "PROGRESS" "session_bloat detected (dry-run); snapshot=${snapshot_path_rel}"
    exit 0
  fi

  set +e
  bloat_chk_out="$(OPENCLAW_INTERNAL_MUTATION=1 OPENCLAW_INTERNAL_MUTATION_CALLSITE="context_runtime_local_watch.sh:session_bloat_checkpoint" "$CHECKPOINT_SCRIPT" \
    --trigger session_bloat \
    --status PROGRESS \
    --objective "Session transcript size exceeded configured bloat threshold" \
    --next-action "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/watchdog_session_bloat.sh" \
    --verify-cmd "openclaw sessions --agent ${TARGET_AGENT_ID} --active 1440 --json >/dev/null" 2>/tmp/context_runtime_local_watch_bloat_checkpoint.err)"
  bloat_chk_rc=$?
  set -e

  if [[ "$bloat_chk_rc" -ne 0 ]]; then
    err="$(cat /tmp/context_runtime_local_watch_bloat_checkpoint.err 2>/dev/null || true)"
    emit_blocker "session_bloat_checkpoint_failed" "task=context_runtime_local_watch; reason=session_bloat_checkpoint_write_failed; err=${err:0:160}" "$CHECKPOINT_SCRIPT"
    exit 0
  fi

  checkpoint_id="$(python3 -c 'import json,sys; print((json.loads(sys.argv[1]).get("checkpoint_id") or ""))' "$bloat_chk_out")"
  "$HANDOVER_COMPAT_SCRIPT" --checkpoint "$checkpoint_id" >/dev/null 2>&1 || true
  emit_status "PROGRESS" "session_bloat threshold reached; checkpoint=${checkpoint_id}; snapshot=${snapshot_path_rel}"
  exit 0
fi


if [[ "$context_high" -eq 1 || "$context_preventive" -eq 1 || "$session_bloat" -eq 1 || "$runtime_unhealthy" -eq 1 ]]; then
  emit_status "PROGRESS" "context_state=context_high:${context_high};context_preventive:${context_preventive};session_bloat:${session_bloat};runtime_unhealthy:${runtime_unhealthy};pct=${pct_str};router_rc=${router_rc};snapshot=${snapshot_path_rel}"
else
  emit_status "READY" "context_state=normal;session_bloat=0;runtime_unhealthy=0;pct=${pct_str};snapshot=${snapshot_path_rel}"
fi

exit 0
