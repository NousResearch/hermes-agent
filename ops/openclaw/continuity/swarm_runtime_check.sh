#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
JSON_OUT=0
STRICT=0

usage() {
  cat <<'EOF'
Usage: swarm_runtime_check.sh [options]

Operator-facing swarm runtime doctor:
- contract operability check
- subagent slot-fill protocol/runbook check
- continuity DB integrity check
- continuity now runtime surface
- Ground-Truth Connectors v2 gateboard sync/readiness
- GTC latest surface schema gate
- queue ready/lock snapshot
- queue replay projection truth

Options:
  --json       JSON output
  --strict     Exit non-zero when swarm runtime is not healthy
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_OUT=1; shift ;;
    --strict)
      STRICT=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

python3 - "$ROOT" "$JSON_OUT" "$STRICT" <<'PY'
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set

root = pathlib.Path(sys.argv[1]).resolve()
json_out = bool(int(sys.argv[2]))
strict = bool(int(sys.argv[3]))

sys.path.insert(0, str((root / "ops" / "openclaw" / "continuity").resolve()))
try:
    from fixed_now import now_iso_utc as _helper_now_iso_utc
except Exception:  # pragma: no cover
    _helper_now_iso_utc = None


def now_iso() -> str:
    if _helper_now_iso_utc is not None:
        try:
            return str(_helper_now_iso_utc())
        except Exception:
            pass
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_json(cmd: List[str], timeout: int = 60) -> Dict[str, Any]:
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except Exception as exc:
        return {
            "ok": False,
            "returncode": 127,
            "error": f"exec_failed:{exc}",
            "stdout": "",
            "stderr": str(exc),
            "payload": {},
            "command": cmd,
        }

    payload: Dict[str, Any] = {}
    if (cp.stdout or "").strip():
        try:
            obj = json.loads(cp.stdout)
            if isinstance(obj, dict):
                payload = obj
        except Exception:
            payload = {}

    return {
        "ok": cp.returncode == 0,
        "returncode": cp.returncode,
        "stdout": cp.stdout or "",
        "stderr": cp.stderr or "",
        "payload": payload,
        "command": cmd,
    }


cont_dir = root / "ops" / "openclaw" / "continuity"
arch_dir = root / "ops" / "openclaw" / "architecture"

operability = run_json([str(arch_dir / "check_swarm_operability.sh"), "--json"], timeout=60)
slot_fill = run_json([str(cont_dir / "check_slot_fill_protocol.sh"), "--json"], timeout=60)
db_check = run_json([str(cont_dir / "db_integrity_check.sh"), "--strict", "--json"], timeout=60)
now = run_json([str(cont_dir / "continuity_now.sh"), "--json"], timeout=60)
ready = run_json([str(cont_dir / "queue_arbitrator.sh"), "ready-list", "--limit", "20", "--json"], timeout=30)
locks = run_json([str(cont_dir / "queue_arbitrator.sh"), "locks", "--active-only", "--limit", "20", "--json"], timeout=30)
handoffs = run_json([str(cont_dir / "queue_arbitrator.sh"), "handoffs", "--limit", "20", "--json"], timeout=30)
remediate = run_json(
    [
        "env",
        "OPENCLAW_INTERNAL_MUTATION=1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE=swarm_runtime_check.sh:queue_remediate_probe",
        str(cont_dir / "queue_arbitrator.sh"),
        "remediate",
        "--limit",
        "20",
        "--json",
    ],
    timeout=30,
)
queue_replay = run_json([str(cont_dir / "queue_replay_verify.sh"), "--json"], timeout=45)
gtc_sync = run_json(
    [
        "env",
        "OPENCLAW_INTERNAL_MUTATION=1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE=swarm_runtime_check.sh:gtc_sync_probe",
        str(cont_dir / "gtc_v2_sync.sh"),
        "--json",
    ],
    timeout=60,
)
gtc_schema = run_json([str(cont_dir / "gtc_latest_schema_check.sh"), "--json"], timeout=60)

op_payload = operability.get("payload") or {}
slot_fill_payload = slot_fill.get("payload") or {}
db_payload = db_check.get("payload") or {}
now_payload = now.get("payload") or {}
ready_payload = ready.get("payload") or {}
locks_payload = locks.get("payload") or {}
handoffs_payload = handoffs.get("payload") or {}
remediate_payload = remediate.get("payload") or {}
queue_replay_payload = queue_replay.get("payload") or {}
gtc_payload = gtc_sync.get("payload") or {}
gtc_schema_payload = gtc_schema.get("payload") or {}


def parse_non_negative_int(payload: Dict[str, Any], key: str, errors: List[str], error_key: Optional[str] = None) -> Any:
    issue = error_key or f"{key}_missing_or_invalid"
    raw = payload.get(key) if isinstance(payload, dict) else None
    if isinstance(raw, bool):
        errors.append(issue)
        return None
    if isinstance(raw, int):
        if raw >= 0:
            return raw
        errors.append(issue)
        return None
    if isinstance(raw, str):
        token = raw.strip()
        if token.isdigit():
            return int(token)
    errors.append(issue)
    return None


def parse_required_bool(payload: Dict[str, Any], key: str, errors: List[str], error_key: Optional[str] = None) -> Any:
    issue = error_key or f"{key}_missing_or_invalid"
    raw = payload.get(key) if isinstance(payload, dict) else None
    if isinstance(raw, bool):
        return raw
    errors.append(issue)
    return None


def parse_required_dict(payload: Dict[str, Any], key: str, errors: List[str], error_key: Optional[str] = None) -> Dict[str, Any]:
    issue = error_key or f"{key}_missing_or_invalid"
    raw = payload.get(key) if isinstance(payload, dict) else None
    if isinstance(raw, dict):
        return raw
    errors.append(issue)
    return {}


def parse_required_list(payload: Dict[str, Any], key: str, errors: List[str], error_key: Optional[str] = None) -> List[Any]:
    issue = error_key or f"{key}_missing_or_invalid"
    raw = payload.get(key) if isinstance(payload, dict) else None
    if isinstance(raw, list):
        return raw
    errors.append(issue)
    return []


def parse_required_command(payload: Dict[str, Any], expected: str, errors: List[str], error_key: Optional[str] = None) -> Optional[str]:
    issue = error_key or f"command_missing_or_invalid_expected_{expected}"
    raw = payload.get("command") if isinstance(payload, dict) else None
    if isinstance(raw, str):
        token = raw.strip()
        if token == expected:
            return token
    errors.append(issue)
    return None


def parse_required_token(
    payload: Dict[str, Any],
    key: str,
    allowed_tokens: Set[str],
    errors: List[str],
    error_key: Optional[str] = None,
) -> Optional[str]:
    issue = error_key or f"{key}_missing_or_invalid"
    raw = payload.get(key) if isinstance(payload, dict) else None
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in allowed_tokens:
            return token
    errors.append(issue)
    return None


_db_contract_errors: List[str] = []
critical_failures = parse_non_negative_int(db_payload, "critical_failures", _db_contract_errors)
warn_failures = parse_non_negative_int(db_payload, "warn_failures", _db_contract_errors)
db_check_count = parse_non_negative_int(db_payload, "check_count", _db_contract_errors)
db_contract_errors = sorted(set(_db_contract_errors))
db_payload_contract_ok = len(db_contract_errors) == 0

db_command_ok = bool(db_check.get("ok"))
db_integrity_ok = db_payload_contract_ok and critical_failures == 0

ready_flag = bool(now_payload.get("ready") is True)
operability_ok = bool(op_payload.get("ok") is True)
slot_fill_ok = bool(slot_fill_payload.get("ok") is True)
gtc_gateboard_raw = gtc_payload.get("gateboard") if isinstance(gtc_payload, dict) else {}
gtc_gateboard = gtc_gateboard_raw if isinstance(gtc_gateboard_raw, dict) else {}
gtc_warning_reasons_raw = gtc_gateboard.get("warning_reasons") if isinstance(gtc_gateboard.get("warning_reasons"), list) else []
gtc_warning_reasons = [str(reason).strip() for reason in gtc_warning_reasons_raw if str(reason).strip()]
gtc_blocking_reasons_raw = gtc_gateboard.get("blocking_reasons") if isinstance(gtc_gateboard.get("blocking_reasons"), list) else []
gtc_blocking_reasons = [str(reason).strip() for reason in gtc_blocking_reasons_raw if str(reason).strip()]
gtc_queue_handoff_gate_binding_degraded = "queue_task_handoff_gate_binding_degraded" in gtc_warning_reasons
gtc_mutate_allowed = bool(gtc_gateboard.get("mutate_allowed") is True)
gtc_schema_ok = bool(gtc_schema_payload.get("ok") is True)
queue_command_ok = bool(ready.get("ok") and locks.get("ok") and handoffs.get("ok") and remediate.get("ok"))

_queue_contract_errors: List[str] = []
queue_now = parse_required_dict(now_payload, "queue", _queue_contract_errors, "now_queue_missing_or_invalid")
role_unset_raw = parse_non_negative_int(
    queue_now,
    "role_required_unset_count",
    _queue_contract_errors,
    "now_queue_role_required_unset_count_missing_or_invalid",
)
review_mismatch_raw = parse_non_negative_int(
    queue_now,
    "review_role_mismatch_count",
    _queue_contract_errors,
    "now_queue_review_role_mismatch_count_missing_or_invalid",
)
stale_locks_raw = parse_non_negative_int(
    queue_now,
    "stale_active_file_lock_count",
    _queue_contract_errors,
    "now_queue_stale_active_file_lock_count_missing_or_invalid",
)
dependency_blocked_count_raw = parse_non_negative_int(
    queue_now,
    "dependency_blocked_count",
    _queue_contract_errors,
    "now_queue_dependency_blocked_count_missing_or_invalid",
)
ready_count = parse_non_negative_int(
    ready_payload,
    "ready_count",
    _queue_contract_errors,
    "ready_ready_count_missing_or_invalid",
)
ready_items = parse_required_list(
    ready_payload,
    "items",
    _queue_contract_errors,
    "ready_items_missing_or_invalid",
)
ready_payload_ok = parse_required_bool(
    ready_payload,
    "ok",
    _queue_contract_errors,
    "ready_ok_missing_or_invalid",
)
parse_required_command(
    ready_payload,
    "ready-list",
    _queue_contract_errors,
    "ready_command_missing_or_invalid",
)
lock_items = parse_required_list(locks_payload, "items", _queue_contract_errors, "locks_items_missing_or_invalid")
locks_payload_ok = parse_required_bool(
    locks_payload,
    "ok",
    _queue_contract_errors,
    "locks_ok_missing_or_invalid",
)
parse_required_command(
    locks_payload,
    "locks",
    _queue_contract_errors,
    "locks_command_missing_or_invalid",
)
handoffs_count = parse_non_negative_int(
    handoffs_payload,
    "count",
    _queue_contract_errors,
    "handoffs_count_missing_or_invalid",
)
handoffs_items = parse_required_list(
    handoffs_payload,
    "items",
    _queue_contract_errors,
    "handoffs_items_missing_or_invalid",
)
handoffs_payload_ok = parse_required_bool(
    handoffs_payload,
    "ok",
    _queue_contract_errors,
    "handoffs_ok_missing_or_invalid",
)
parse_required_command(
    handoffs_payload,
    "handoffs",
    _queue_contract_errors,
    "handoffs_command_missing_or_invalid",
)
remediate_dry_run = parse_required_bool(
    remediate_payload,
    "dry_run",
    _queue_contract_errors,
    "remediate_dry_run_missing_or_invalid",
)
remediate_payload_ok = parse_required_bool(
    remediate_payload,
    "ok",
    _queue_contract_errors,
    "remediate_ok_missing_or_invalid",
)
parse_required_command(
    remediate_payload,
    "remediate",
    _queue_contract_errors,
    "remediate_command_missing_or_invalid",
)
remediate_preview = parse_required_dict(
    remediate_payload,
    "preview",
    _queue_contract_errors,
    "remediate_preview_missing_or_invalid",
)
remediate_overdue_items = parse_required_list(
    remediate_preview,
    "overdue_active_locks",
    _queue_contract_errors,
    "remediate_preview_overdue_active_locks_missing_or_invalid",
)
remediate_terminal_items = parse_required_list(
    remediate_preview,
    "terminal_task_active_locks",
    _queue_contract_errors,
    "remediate_preview_terminal_task_active_locks_missing_or_invalid",
)
remediate_blocked_items = parse_required_list(
    remediate_preview,
    "blocked_tasks_with_resolved_dependencies",
    _queue_contract_errors,
    "remediate_preview_blocked_tasks_with_resolved_dependencies_missing_or_invalid",
)
remediate_orphaned_running_items = parse_required_list(
    remediate_preview,
    "orphaned_running_without_locks",
    _queue_contract_errors,
    "remediate_preview_orphaned_running_without_locks_missing_or_invalid",
)
if isinstance(ready_count, int) and len(ready_items) != ready_count:
    _queue_contract_errors.append("ready_count_items_mismatch")
if isinstance(handoffs_count, int) and len(handoffs_items) != handoffs_count:
    _queue_contract_errors.append("handoffs_count_items_mismatch")
if ready_payload_ok is False:
    _queue_contract_errors.append("ready_ok_false")
if locks_payload_ok is False:
    _queue_contract_errors.append("locks_ok_false")
if handoffs_payload_ok is False:
    _queue_contract_errors.append("handoffs_ok_false")
if remediate_payload_ok is False:
    _queue_contract_errors.append("remediate_ok_false")
if remediate_dry_run is False:
    _queue_contract_errors.append("remediate_dry_run_false")
queue_contract_errors = sorted(set(_queue_contract_errors))
queue_payload_contract_ok = len(queue_contract_errors) == 0
queue_snapshot_ok = queue_command_ok and queue_payload_contract_ok

_queue_replay_contract_errors: List[str] = []
queue_replay_summary = parse_required_dict(
    queue_replay_payload,
    "summary",
    _queue_replay_contract_errors,
    "queue_replay_summary_missing_or_invalid",
)
queue_replay_status = parse_required_token(
    queue_replay_summary,
    "status",
    {"pass", "warn", "fail"},
    _queue_replay_contract_errors,
    "queue_replay_status_missing_or_invalid",
)
queue_replay_task_count = parse_non_negative_int(
    queue_replay_summary,
    "task_count",
    _queue_replay_contract_errors,
    "queue_replay_task_count_missing_or_invalid",
)
queue_replay_status_mismatch_count = parse_non_negative_int(
    queue_replay_summary,
    "status_mismatch_count",
    _queue_replay_contract_errors,
    "queue_replay_status_mismatch_count_missing_or_invalid",
)
queue_replay_active_status_mismatch_count = parse_non_negative_int(
    queue_replay_summary,
    "active_status_mismatch_count",
    _queue_replay_contract_errors,
    "queue_replay_active_status_mismatch_count_missing_or_invalid",
)
queue_replay_legacy_status_mismatch_count = parse_non_negative_int(
    queue_replay_summary,
    "legacy_status_mismatch_count",
    _queue_replay_contract_errors,
    "queue_replay_legacy_status_mismatch_count_missing_or_invalid",
)
queue_replay_role_mismatch_count = parse_non_negative_int(
    queue_replay_summary,
    "role_mismatch_count",
    _queue_replay_contract_errors,
    "queue_replay_role_mismatch_count_missing_or_invalid",
)
queue_replay_discontinuity_task_count = parse_non_negative_int(
    queue_replay_summary,
    "discontinuity_task_count",
    _queue_replay_contract_errors,
    "queue_replay_discontinuity_task_count_missing_or_invalid",
)
queue_replay_soft_discontinuity_task_count = parse_non_negative_int(
    queue_replay_summary,
    "soft_discontinuity_task_count",
    _queue_replay_contract_errors,
    "queue_replay_soft_discontinuity_task_count_missing_or_invalid",
)
queue_replay_historical_discontinuity_task_count = parse_non_negative_int(
    queue_replay_summary,
    "historical_discontinuity_task_count",
    _queue_replay_contract_errors,
    "queue_replay_historical_discontinuity_task_count_missing_or_invalid",
)
queue_replay_status_mismatches = parse_required_list(
    queue_replay_payload,
    "status_mismatches",
    _queue_replay_contract_errors,
    "queue_replay_status_mismatches_missing_or_invalid",
)
queue_replay_active_status_mismatches = parse_required_list(
    queue_replay_payload,
    "active_status_mismatches",
    _queue_replay_contract_errors,
    "queue_replay_active_status_mismatches_missing_or_invalid",
)
queue_replay_legacy_status_mismatches = parse_required_list(
    queue_replay_payload,
    "legacy_status_mismatches",
    _queue_replay_contract_errors,
    "queue_replay_legacy_status_mismatches_missing_or_invalid",
)
queue_replay_role_mismatches = parse_required_list(
    queue_replay_payload,
    "role_mismatches",
    _queue_replay_contract_errors,
    "queue_replay_role_mismatches_missing_or_invalid",
)
queue_replay_discontinuities = parse_required_list(
    queue_replay_payload,
    "discontinuities",
    _queue_replay_contract_errors,
    "queue_replay_discontinuities_missing_or_invalid",
)
queue_replay_soft_discontinuities = parse_required_list(
    queue_replay_payload,
    "soft_discontinuities",
    _queue_replay_contract_errors,
    "queue_replay_soft_discontinuities_missing_or_invalid",
)
queue_replay_historical_discontinuities = parse_required_list(
    queue_replay_payload,
    "historical_discontinuities",
    _queue_replay_contract_errors,
    "queue_replay_historical_discontinuities_missing_or_invalid",
)
if isinstance(queue_replay_status_mismatch_count, int) and len(queue_replay_status_mismatches) != queue_replay_status_mismatch_count:
    _queue_replay_contract_errors.append("queue_replay_status_mismatch_count_items_mismatch")
if isinstance(queue_replay_active_status_mismatch_count, int) and len(queue_replay_active_status_mismatches) != queue_replay_active_status_mismatch_count:
    _queue_replay_contract_errors.append("queue_replay_active_status_mismatch_count_items_mismatch")
if isinstance(queue_replay_legacy_status_mismatch_count, int) and len(queue_replay_legacy_status_mismatches) != queue_replay_legacy_status_mismatch_count:
    _queue_replay_contract_errors.append("queue_replay_legacy_status_mismatch_count_items_mismatch")
if isinstance(queue_replay_role_mismatch_count, int) and len(queue_replay_role_mismatches) != queue_replay_role_mismatch_count:
    _queue_replay_contract_errors.append("queue_replay_role_mismatch_count_items_mismatch")
if isinstance(queue_replay_discontinuity_task_count, int) and len(queue_replay_discontinuities) != queue_replay_discontinuity_task_count:
    _queue_replay_contract_errors.append("queue_replay_discontinuity_task_count_items_mismatch")
if isinstance(queue_replay_soft_discontinuity_task_count, int) and len(queue_replay_soft_discontinuities) != queue_replay_soft_discontinuity_task_count:
    _queue_replay_contract_errors.append("queue_replay_soft_discontinuity_task_count_items_mismatch")
if isinstance(queue_replay_historical_discontinuity_task_count, int) and len(queue_replay_historical_discontinuities) != queue_replay_historical_discontinuity_task_count:
    _queue_replay_contract_errors.append("queue_replay_historical_discontinuity_task_count_items_mismatch")
if all(
    isinstance(x, int)
    for x in [
        queue_replay_status_mismatch_count,
        queue_replay_active_status_mismatch_count,
        queue_replay_legacy_status_mismatch_count,
    ]
):
    if queue_replay_status_mismatch_count != queue_replay_active_status_mismatch_count + queue_replay_legacy_status_mismatch_count:
        _queue_replay_contract_errors.append("queue_replay_status_mismatch_partition_mismatch")
if queue_replay_status == "pass":
    if int(queue_replay_active_status_mismatch_count or 0) > 0 or int(queue_replay_role_mismatch_count or 0) > 0:
        _queue_replay_contract_errors.append("queue_replay_pass_with_active_residue")
    if int(queue_replay_legacy_status_mismatch_count or 0) > 0 or int(queue_replay_discontinuity_task_count or 0) > 0:
        _queue_replay_contract_errors.append("queue_replay_pass_with_warn_residue")
if queue_replay_status == "warn" and (int(queue_replay_active_status_mismatch_count or 0) > 0 or int(queue_replay_role_mismatch_count or 0) > 0):
    _queue_replay_contract_errors.append("queue_replay_warn_with_fail_residue")
if queue_replay_status == "fail" and int(queue_replay_active_status_mismatch_count or 0) == 0 and int(queue_replay_role_mismatch_count or 0) == 0:
    _queue_replay_contract_errors.append("queue_replay_fail_without_fail_residue")
queue_replay_contract_errors = sorted(set(_queue_replay_contract_errors))
queue_replay_payload_contract_ok = len(queue_replay_contract_errors) == 0
queue_replay_command_ok = bool(queue_replay.get("ok"))
queue_replay_ok = queue_replay_command_ok and queue_replay_payload_contract_ok and queue_replay_status in {"pass", "warn"}

role_unset = int(role_unset_raw or 0)
review_mismatch = int(review_mismatch_raw or 0)
stale_locks = int(stale_locks_raw or 0)
dependency_blocked_count = int(dependency_blocked_count_raw or 0)
remediate_overdue = len(remediate_overdue_items)
remediate_terminal = len(remediate_terminal_items)
remediate_blocked = len(remediate_blocked_items)
remediate_orphaned_running = len(remediate_orphaned_running_items)

not_ready_raw = now_payload.get("not_ready_reasons") if isinstance(now_payload, dict) else []
not_ready_reasons = not_ready_raw if isinstance(not_ready_raw, list) else []
not_ready_reason_tokens = [str(reason) for reason in not_ready_reasons if isinstance(reason, str) and str(reason).strip()]
verify_gate_preflight = ((now_payload.get("verify") or {}).get("gate_preflight") or {}) if isinstance(now_payload, dict) else {}
predicted_gate = (verify_gate_preflight.get("predicted_gate") or {}) if isinstance(verify_gate_preflight, dict) else {}
verify_preflight_ready_to_run = bool(predicted_gate.get("ready_to_run") is True)
verify_then_resume_active = str(os.environ.get("OPENCLAW_VERIFY_THEN_RESUME_ACTIVE", "0")).strip() == "1"
self_referential_now_blockers = {
    "verify_blocker",
    "gtc_gateboard_blocked",
    "ground_truth_capture_drift",
    # During an active verify cycle, READY-status evidence residue from the
    # previous verify report is expected and should not deadlock the next
    # verify pass.
    "verify_status_evidence_missing",
    "verify_status_evidence_invalid",
    "verify_status_evidence_stale",
    "verify_status_evidence_untrusted",
}
self_referential_gtc_blockers = {"verify_status_not_ready:BLOCKER"}
continuity_now_effective_ready = bool(
    ready_flag
    or (
        (verify_preflight_ready_to_run or verify_then_resume_active)
        and bool(not_ready_reason_tokens)
        and set(not_ready_reason_tokens).issubset(self_referential_now_blockers)
    )
)
gtc_mutate_allowed_effective = bool(
    gtc_mutate_allowed
    or (
        (verify_preflight_ready_to_run or verify_then_resume_active)
        and bool(gtc_blocking_reasons)
        and set(gtc_blocking_reasons).issubset(self_referential_gtc_blockers)
    )
)

overall_ok = all(
    [
        operability.get("ok") and operability_ok,
        slot_fill.get("ok") and slot_fill_ok,
        db_command_ok and db_integrity_ok,
        now.get("ok") and continuity_now_effective_ready,
        queue_snapshot_ok,
        queue_replay_ok,
        gtc_sync.get("ok") and gtc_mutate_allowed_effective,
        gtc_schema.get("ok") and gtc_schema_ok,
        not gtc_queue_handoff_gate_binding_degraded,
    ]
)

recovery_cmds: List[str] = []
if not db_payload_contract_ok:
    recovery_cmds.append(f"bash {cont_dir}/db_integrity_check.sh --strict --json")
if not queue_payload_contract_ok:
    recovery_cmds.extend(
        [
            f"bash {cont_dir}/continuity_now.sh --json",
            f"bash {cont_dir}/queue_arbitrator.sh ready-list --json",
            f"bash {cont_dir}/queue_arbitrator.sh locks --active-only --json",
            f"bash {cont_dir}/queue_arbitrator.sh handoffs --json",
            f"bash {cont_dir}/queue_arbitrator.sh remediate --json",
        ]
    )
if not queue_replay_payload_contract_ok:
    recovery_cmds.append(f"bash {root}/ops/openclaw/continuity.sh queue-replay --json")
if queue_replay_status == "fail":
    recovery_cmds.append(f"bash {root}/ops/openclaw/continuity.sh queue-replay --strict --json")
elif queue_replay_status == "warn":
    recovery_cmds.append(f"bash {root}/ops/openclaw/continuity.sh queue-replay --json")
if role_unset > 0 or review_mismatch > 0:
    recovery_cmds.append(f"bash {cont_dir}/db_integrity_check.sh --strict --json")
if not slot_fill_ok:
    recovery_cmds.append(f"bash {cont_dir}/check_slot_fill_protocol.sh --json")
if stale_locks > 0:
    recovery_cmds.append(f"bash {cont_dir}/queue_arbitrator.sh locks --active-only --json")
if remediate_overdue > 0 or remediate_terminal > 0 or remediate_blocked > 0:
    recovery_cmds.append(
        f"bash {cont_dir}/queue_arbitrator.sh remediate --expire-overdue-locks --release-terminal-locks --requeue-resolved-blocked --json"
    )
if dependency_blocked_count > 0:
    recovery_cmds.append(f"bash {cont_dir}/history.sh --source-preset control-plane --hours 24 --json")
if (now_payload.get("parity") or {}).get("due") is True:
    recovery_cmds.append(f"bash {root}/ops/openclaw/run_competitive_parity_harness.sh --force")
if not_ready_reason_tokens and set(not_ready_reason_tokens).issubset({"pointer_drift", "ground_truth_capture_drift"}):
    recovery_cmds.append(
        f"bash {root}/ops/openclaw/continuity.sh --action-token <current.action_token> reconcile --json"
    )
if not gtc_mutate_allowed:
    recovery_cmds.append(
        f"bash {root}/ops/openclaw/continuity.sh --action-token <current.action_token> gtc-sync --strict --json"
    )
    recovery_cmds.append(f"bash {root}/ops/openclaw/snapshot_ground_truth.sh")
if not gtc_schema_ok:
    recovery_cmds.append(f"bash {cont_dir}/gtc_latest_schema_check.sh --strict --json")
if gtc_queue_handoff_gate_binding_degraded:
    recovery_cmds.append(f"bash {cont_dir}/queue_arbitrator.sh handoffs --limit 20 --json")

deduped_cmds: List[str] = []
seen_cmds = set()
for cmd_line in recovery_cmds:
    if cmd_line in seen_cmds:
        continue
    seen_cmds.add(cmd_line)
    deduped_cmds.append(cmd_line)

summary = {
    "ok": overall_ok,
    "generated_at": now_iso(),
    "schema_version": "swarm.runtime.check.v1",
    "checks": {
        "swarm_operability": {
            "command_ok": bool(operability.get("ok")),
            "ok": operability_ok,
            "ready": op_payload.get("ready"),
            "expected_roles": op_payload.get("expected_roles"),
            "missing": op_payload.get("missing"),
        },
        "slot_fill_protocol": {
            "command_ok": bool(slot_fill.get("ok")),
            "ok": slot_fill_ok,
            "critical_failures": slot_fill_payload.get("critical_failures"),
            "warn_failures": slot_fill_payload.get("warn_failures"),
            "schema_version": slot_fill_payload.get("schema_version"),
        },
        "db_integrity": {
            "command_ok": db_command_ok,
            "ok": db_integrity_ok,
            "payload_contract_ok": db_payload_contract_ok,
            "contract_errors": db_contract_errors,
            "critical_failures": critical_failures,
            "warn_failures": warn_failures,
            "check_count": db_check_count,
        },
        "continuity_now": {
            "command_ok": bool(now.get("ok")),
            "ready": ready_flag,
            "effective_ready": continuity_now_effective_ready,
            "verify_preflight_ready_to_run": verify_preflight_ready_to_run,
            "not_ready_reasons": now_payload.get("not_ready_reasons"),
            "warning_reasons": now_payload.get("warning_reasons"),
        },
        "queue_replay": {
            "command_ok": queue_replay_command_ok,
            "ok": queue_replay_ok,
            "payload_contract_ok": queue_replay_payload_contract_ok,
            "contract_errors": queue_replay_contract_errors,
            "status": queue_replay_status,
            "task_count": queue_replay_task_count,
            "status_mismatch_count": queue_replay_status_mismatch_count,
            "active_status_mismatch_count": queue_replay_active_status_mismatch_count,
            "legacy_status_mismatch_count": queue_replay_legacy_status_mismatch_count,
            "role_mismatch_count": queue_replay_role_mismatch_count,
            "discontinuity_task_count": queue_replay_discontinuity_task_count,
            "soft_discontinuity_task_count": queue_replay_soft_discontinuity_task_count,
            "historical_discontinuity_task_count": queue_replay_historical_discontinuity_task_count,
            "normalized_replay_categories": {
                "active": int(queue_replay_active_status_mismatch_count or 0) + int(queue_replay_role_mismatch_count or 0),
                "legacy": int(queue_replay_legacy_status_mismatch_count or 0),
                "historical": int(queue_replay_historical_discontinuity_task_count or 0),
                "soft": int(queue_replay_soft_discontinuity_task_count or 0),
            },
        },
        "gtc_v2": {
            "command_ok": bool(gtc_sync.get("ok")),
            "mutate_allowed": gtc_mutate_allowed,
            "effective_mutate_allowed": gtc_mutate_allowed_effective,
            "status": gtc_gateboard.get("status"),
            "blocking_reasons": gtc_gateboard.get("blocking_reasons"),
            "warning_reasons": gtc_warning_reasons,
            "queue_handoff_gate_binding_degraded": gtc_queue_handoff_gate_binding_degraded,
            "inserted_evidence": gtc_payload.get("inserted_evidence"),
        },
        "gtc_latest_schema": {
            "command_ok": bool(gtc_schema.get("ok")),
            "ok": gtc_schema_ok,
            "error_count": gtc_schema_payload.get("error_count"),
            "surface_count": gtc_schema_payload.get("surface_count"),
            "connector_count": gtc_schema_payload.get("connector_count"),
            "generation_consistent": (gtc_schema_payload.get("generation_consistency") or {}).get("ok"),
        },
        "queue_snapshot": {
            "command_ok": queue_command_ok,
            "ok": queue_snapshot_ok,
            "payload_contract_ok": queue_payload_contract_ok,
            "contract_errors": queue_contract_errors,
            "ready_count": ready_count,
            "ready_items_count": len(ready_items),
            "active_lock_rows": len(lock_items),
            "stale_active_lock_count": stale_locks_raw,
            "role_required_unset_count": role_unset_raw,
            "review_role_mismatch_count": review_mismatch_raw,
            "dependency_blocked_count": dependency_blocked_count_raw,
            "handoff_packets_recent": handoffs_count,
            "handoff_items_count": len(handoffs_items),
            "remediation_preview": {
                "overdue_active_locks": remediate_overdue,
                "terminal_task_active_locks": remediate_terminal,
                "resolved_blocked_tasks": remediate_blocked,
                "orphaned_running_without_locks": remediate_orphaned_running,
            },
        },
        "handoffs": {
            "command_ok": bool(handoffs.get("ok")),
            "count": handoffs_count,
        },
        "remediation": {
            "command_ok": bool(remediate.get("ok")),
            "dry_run": remediate_dry_run,
            "preview": remediate_preview,
        },
    },
    "recommended_recovery_commands": deduped_cmds,
}

if json_out:
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("SWARM RUNTIME CHECK")
    print(f"- ok: {summary['ok']}")
    print(
        "- operability: "
        f"ok={summary['checks']['swarm_operability']['ok']} "
        f"ready={summary['checks']['swarm_operability']['ready']}"
    )
    print(
        "- slot_fill_protocol: "
        f"ok={summary['checks']['slot_fill_protocol']['ok']} "
        f"critical_failures={summary['checks']['slot_fill_protocol']['critical_failures']} "
        f"warn_failures={summary['checks']['slot_fill_protocol']['warn_failures']}"
    )
    print(
        "- db_integrity: "
        f"ok={summary['checks']['db_integrity']['ok']} "
        f"payload_contract_ok={summary['checks']['db_integrity']['payload_contract_ok']} "
        f"critical_failures={critical_failures} warn_failures={warn_failures}"
    )
    print(
        "- continuity_now: "
        f"ready={summary['checks']['continuity_now']['ready']} "
        f"warnings={summary['checks']['continuity_now']['warning_reasons'] or []}"
    )
    qr = summary["checks"]["queue_replay"]
    print(
        "- queue_replay: "
        f"ok={qr.get('ok')} "
        f"status={qr.get('status') or 'n/a'} "
        f"active={qr.get('active_status_mismatch_count')} "
        f"role_mismatch={qr.get('role_mismatch_count')} "
        f"legacy={qr.get('legacy_status_mismatch_count')} "
        f"historical={qr.get('historical_discontinuity_task_count')} "
        f"soft={qr.get('soft_discontinuity_task_count')}"
    )
    print(
        "- gtc_v2: "
        f"mutate_allowed={summary['checks']['gtc_v2']['mutate_allowed']} "
        f"status={summary['checks']['gtc_v2']['status'] or 'n/a'} "
        f"blocking={summary['checks']['gtc_v2']['blocking_reasons'] or []} "
        f"warnings={summary['checks']['gtc_v2']['warning_reasons'] or []} "
        f"handoff_binding_degraded={summary['checks']['gtc_v2']['queue_handoff_gate_binding_degraded']}"
    )
    print(
        "- gtc_latest_schema: "
        f"ok={summary['checks']['gtc_latest_schema']['ok']} "
        f"errors={summary['checks']['gtc_latest_schema']['error_count']} "
        f"connectors={summary['checks']['gtc_latest_schema']['connector_count']} "
        f"generation_consistent={summary['checks']['gtc_latest_schema']['generation_consistent']}"
    )
    qs = summary["checks"]["queue_snapshot"]
    print(
        "- queue_snapshot: "
        f"ok={qs.get('ok')} "
        f"payload_contract_ok={qs.get('payload_contract_ok')} "
        f"ready={qs.get('ready_count')} "
        f"dependency_blocked={qs.get('dependency_blocked_count')} "
        f"role_unset={qs.get('role_required_unset_count')} "
        f"review_role_mismatch={qs.get('review_role_mismatch_count')} "
        f"stale_locks={qs.get('stale_active_lock_count')}"
    )
    if summary.get("recommended_recovery_commands"):
        print("- recommended_recovery_commands:")
        for cmd in summary.get("recommended_recovery_commands") or []:
            print(f"  - {cmd}")

if strict and not summary["ok"]:
    raise SystemExit(1)
PY
