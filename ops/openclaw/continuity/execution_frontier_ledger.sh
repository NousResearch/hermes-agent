#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
JSON_OUT=0
REFRESH=0
ACTION="show"
NEXT_CANDIDATE_OVERRIDE=""
ADVANCE_REASON=""
ACTION_TOKEN=""
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"
MUTATION_TICKET="${OPENCLAW_MUTATION_TICKET:-}"
ATTESTATIONS=()
ATTESTATION_OBJECTS=()

usage() {
  cat <<'EOF'
Usage: execution_frontier_ledger.sh [show|advance-wave-close|supervisor-advance-wave-close|supervisor-autonomous-dispatch|supervisor-reset-txn-handoff-guard] [options]

Read or advance the canonical execution frontier ledger.

Commands:
  show                         Read ledger (default)
  advance-wave-close           Mark current wave as closed and advance to selected next wave candidate
  supervisor-advance-wave-close
                               Guarded control-plane transition: re-read ledger and only advance when
                               selector_state=ready_for_dispatch, close_condition_met=true, and next candidate exists
  supervisor-autonomous-dispatch
                               Fail-closed autonomous supervisor transition: only advances when
                               canonical supervisor_state marks autonomous_dispatch_eligible=true
  supervisor-reset-txn-handoff-guard
                               Operator-audited reset lane for transactional-handoff soak guard.
                               Requires --reason and only applies when soak guard is active.

Options:
  --refresh               Recompute continuity/current before reading/advancing
  --json                  Print raw JSON payload
  --next-candidate <id>   Override next candidate when advancing wave close
  --reason <text>         Optional operator reason persisted in transition metadata
  --action-token <value>  Canonical mutation token for direct mutating entrypoint calls
  --truth-anchor <value>  Legacy alias of --action-token
  --allow-legacy-anchor   Allow legacy anchor-only token mode
  --mutation-ticket <v>   Ticket JSON string, @path, or path (high-risk authority checks)
  --attestation <name>    Satisfied attestation name (repeatable)
  --attestation-object <v>
                          Structured attestation object JSON string, @path, or path (repeatable)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    show|advance-wave-close|supervisor-advance-wave-close|supervisor-autonomous-dispatch|supervisor-reset-txn-handoff-guard)
      ACTION="$1"
      shift ;;
    --refresh)
      REFRESH=1
      shift ;;
    --json)
      JSON_OUT=1
      shift ;;
    --next-candidate)
      NEXT_CANDIDATE_OVERRIDE="${2:-}"
      shift 2 ;;
    --reason)
      ADVANCE_REASON="${2:-}"
      shift 2 ;;
    --action-token|--truth-anchor)
      ACTION_TOKEN="${2:-}"
      shift 2 ;;
    --allow-legacy-anchor)
      ALLOW_LEGACY_ANCHOR=1
      shift ;;
    --mutation-ticket)
      MUTATION_TICKET="${2:-}"
      shift 2 ;;
    --attestation)
      ATTESTATIONS+=("${2:-}")
      shift 2 ;;
    --attestation-object)
      ATTESTATION_OBJECTS+=("${2:-}")
      shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ "$ACTION" != "show" ]]; then
  guard_args=(--script "execution_frontier_ledger.sh")
  case "$ACTION" in
    advance-wave-close)
      guard_args+=(--risk-tier "medium" --mutation-operation "execution_frontier:advance_wave_close") ;;
    supervisor-advance-wave-close)
      guard_args+=(--risk-tier "high" --mutation-operation "execution_frontier:supervisor_advance_wave_close") ;;
    supervisor-autonomous-dispatch)
      guard_args+=(--risk-tier "high" --mutation-operation "execution_frontier:supervisor_autonomous_dispatch") ;;
    supervisor-reset-txn-handoff-guard)
      guard_args+=(--risk-tier "high" --mutation-operation "execution_frontier:supervisor_reset_txn_handoff_guard") ;;
  esac

  if [[ -n "$ACTION_TOKEN" ]]; then
    guard_args+=(--action-token "$ACTION_TOKEN")
  fi
  if [[ "$ALLOW_LEGACY_ANCHOR" == "1" ]]; then
    guard_args+=(--allow-legacy-anchor)
  fi
  if [[ -n "$MUTATION_TICKET" ]]; then
    guard_args+=(--mutation-ticket "$MUTATION_TICKET")
  fi

  for att in "${ATTESTATIONS[@]}"; do
    if [[ -n "${att:-}" ]]; then
      guard_args+=(--attestation "$att")
    fi
  done

  for att_obj in "${ATTESTATION_OBJECTS[@]}"; do
    if [[ -n "${att_obj:-}" ]]; then
      guard_args+=(--attestation-object "$att_obj")
    fi
  done

  "$ROOT/ops/openclaw/continuity/mutator_ingress_guard.sh" "${guard_args[@]}"
fi

if [[ "$REFRESH" == "1" ]]; then
  bash "$ROOT/ops/openclaw/continuity/continuity_current.sh" --refresh >/dev/null
fi

python3 - "$ROOT" "$JSON_OUT" "$ACTION" "$NEXT_CANDIDATE_OVERRIDE" "$ADVANCE_REASON" <<'PY'
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

root = pathlib.Path(sys.argv[1]).resolve()
json_out = bool(int(sys.argv[2]))
action = str(sys.argv[3] or "show").strip() or "show"
next_candidate_override = str(sys.argv[4] or "").strip()
advance_reason = str(sys.argv[5] or "").strip()

ledger_path = root / "state" / "continuity" / "latest" / "execution_frontier_ledger.json"
transition_lock_path = pathlib.Path(
    os.environ.get(
        "OPENCLAW_EXECUTION_FRONTIER_SUPERVISOR_LOCK_PATH",
        str(root / "state" / "continuity" / "locks" / "execution_frontier_supervisor_transition.lock"),
    )
).resolve()
transition_attempt_latest_path = (
    root / "state" / "continuity" / "latest" / "execution_frontier_transition_attempt_latest.json"
)
transition_attempt_history_path = (
    root / "state" / "continuity" / "history" / "execution_frontier_transition_attempts.jsonl"
)
core_queue_txn_script_path = pathlib.Path(
    os.environ.get(
        "OPENCLAW_CORE_ROADMAP_QUEUE_TXN_SCRIPT_PATH",
        str(root / "ops" / "openclaw" / "continuity" / "core_roadmap_queue_layer_txn.sh"),
    )
).resolve()
core_queue_txn_handoff_enabled = str(
    os.environ.get("OPENCLAW_EXECUTION_FRONTIER_CORE_QUEUE_TXN_HANDOFF_ENABLED", "0")
).strip().lower() in {"1", "true", "yes", "y", "on"}
core_queue_txn_worker_id = (
    str(os.environ.get("OPENCLAW_EXECUTION_FRONTIER_CORE_QUEUE_TXN_WORKER_ID", "execution_frontier_supervisor")).strip()
    or "execution_frontier_supervisor"
)
core_queue_txn_terminal_state = str(
    os.environ.get("OPENCLAW_EXECUTION_FRONTIER_CORE_QUEUE_TXN_TERMINAL_STATE", "done")
).strip().lower() or "done"
if core_queue_txn_terminal_state not in {"done", "blocked", "retry"}:
    core_queue_txn_terminal_state = "done"
try:
    core_queue_txn_timeout_sec = max(
        1,
        int(str(os.environ.get("OPENCLAW_EXECUTION_FRONTIER_CORE_QUEUE_TXN_TIMEOUT_SEC", "10")).strip() or "10"),
    )
except Exception:
    core_queue_txn_timeout_sec = 10

try:
    core_queue_txn_claim_lease_sec = max(
        1,
        int(str(os.environ.get("OPENCLAW_EXECUTION_FRONTIER_CORE_QUEUE_TXN_CLAIM_LEASE_SEC", "900")).strip() or "900"),
    )
except Exception:
    core_queue_txn_claim_lease_sec = 900

try:
    core_queue_txn_retry_cooldown_sec = max(
        0,
        int(str(os.environ.get("OPENCLAW_EXECUTION_FRONTIER_CORE_QUEUE_TXN_RETRY_COOLDOWN_SEC", "300")).strip() or "300"),
    )
except Exception:
    core_queue_txn_retry_cooldown_sec = 300

core_queue_txn_handoff_soak_path = pathlib.Path(
    os.environ.get(
        "OPENCLAW_EXECUTION_FRONTIER_CORE_QUEUE_TXN_HANDOFF_SOAK_PATH",
        str(root / "state" / "continuity" / "latest" / "core_roadmap_queue_transaction_handoff_soak.json"),
    )
).resolve()
core_queue_txn_handoff_soak_schema = "clawd.core_roadmap_queue_txn_handoff_soak.v1"
core_queue_txn_handoff_guard_enabled = str(
    os.environ.get("OPENCLAW_EXECUTION_FRONTIER_CORE_QUEUE_TXN_HANDOFF_GUARD_ENABLED", "1")
).strip().lower() in {"1", "true", "yes", "y", "on"}
try:
    core_queue_txn_handoff_guard_consecutive_block_threshold = max(
        1,
        int(
            str(
                os.environ.get(
                    "OPENCLAW_EXECUTION_FRONTIER_CORE_QUEUE_TXN_HANDOFF_GUARD_CONSECUTIVE_BLOCK_THRESHOLD",
                    "3",
                )
            ).strip()
            or "3"
        ),
    )
except Exception:
    core_queue_txn_handoff_guard_consecutive_block_threshold = 3

core_queue_txn_handoff_guard_reset_actor = (
    str(os.environ.get("OPENCLAW_EXECUTION_FRONTIER_CORE_QUEUE_TXN_HANDOFF_GUARD_RESET_ACTOR", "operator")).strip()
    or "operator"
)


WAVE_CLOSE_ACTIONS = {"advance-wave-close", "supervisor-advance-wave-close", "supervisor-autonomous-dispatch"}
RESET_GUARD_ACTIONS = {"supervisor-reset-txn-handoff-guard"}
LOCKED_ACTIONS = WAVE_CLOSE_ACTIONS | RESET_GUARD_ACTIONS


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except Exception:
        return str(path)


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def parse_wave(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None

    text = str(value or "").strip()
    if not text:
        return None

    if text.isdigit():
        try:
            parsed = int(text)
            return parsed if parsed >= 0 else None
        except Exception:
            return None

    m = re.search(r"wave[_:-]?(\d+)", text, flags=re.IGNORECASE)
    if m:
        try:
            parsed = int(m.group(1))
            return parsed if parsed >= 0 else None
        except Exception:
            return None

    m = re.search(r"(?:^|[;\s])cycle=(\d+)(?:/\d+)?(?:$|[;\s])", text)
    if m:
        try:
            parsed = int(m.group(1))
            return parsed if parsed >= 0 else None
        except Exception:
            return None

    return None


def dedupe_strings(values: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in (values or []):
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def extract_frontier_queue(payload: Dict[str, Any]) -> Dict[str, Any]:
    queue_obj = payload.get("frontier_queue") if isinstance(payload.get("frontier_queue"), dict) else {}
    present = bool(queue_obj)

    ready_rows = queue_obj.get("ready_candidates") if isinstance(queue_obj.get("ready_candidates"), list) else []
    blocked_rows = (
        queue_obj.get("dependency_blocked_candidates")
        if isinstance(queue_obj.get("dependency_blocked_candidates"), list)
        else []
    )

    ready_candidate_ids: List[str] = []
    for row in ready_rows:
        task_id = str(row.get("task_id") if isinstance(row, dict) else row or "").strip()
        if task_id:
            ready_candidate_ids.append(task_id)
    ready_candidate_ids = dedupe_strings(ready_candidate_ids)

    blocked_candidate_ids: List[str] = []
    for row in blocked_rows:
        task_id = str(row.get("task_id") if isinstance(row, dict) else row or "").strip()
        if task_id:
            blocked_candidate_ids.append(task_id)
    blocked_candidate_ids = dedupe_strings(blocked_candidate_ids)

    next_candidates = dedupe_strings(
        (queue_obj.get("next_candidates") if isinstance(queue_obj.get("next_candidates"), list) else [])
        + ready_candidate_ids
    )

    queue_source = str(
        queue_obj.get("queue_source")
        or payload.get("queue_source")
        or "continuity_os_queue_db"
    ).strip() or "continuity_os_queue_db"

    ready_count = max(_to_int(queue_obj.get("ready_count")) or 0, len(ready_candidate_ids))
    dependency_blocked_count = max(
        _to_int(queue_obj.get("dependency_blocked_count")) or 0,
        len(blocked_candidate_ids),
    )

    return {
        "present": present,
        "queue_source": queue_source,
        "ready_count": ready_count,
        "dependency_blocked_count": dependency_blocked_count,
        "ready_candidate_ids": ready_candidate_ids,
        "blocked_candidate_ids": blocked_candidate_ids,
        "next_candidates": next_candidates,
        "dependency_model_available": bool(queue_obj.get("dependency_model_available") is True),
    }


def atomic_write(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[pathlib.Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            tmp_path = pathlib.Path(fh.name)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def append_jsonl(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def record_transition_attempt(
    *,
    decision: str,
    advance_applied: bool,
    lock_acquired: bool,
    block_reason: Optional[str] = None,
    block_reasons: Optional[List[str]] = None,
    error: Optional[str] = None,
    selector_state: Optional[str] = None,
    close_condition_met: Optional[bool] = None,
    next_candidate: Optional[str] = None,
    previous_current_wave: Optional[int] = None,
    selected_next_wave: Optional[int] = None,
    autonomous_dispatch_eligible: Optional[bool] = None,
    supervisor_state: Optional[str] = None,
    frontier_queue_ready_count: Optional[int] = None,
    frontier_queue_dependency_blocked_count: Optional[int] = None,
    next_candidate_in_frontier_ready: Optional[bool] = None,
    next_candidate_dependency_blocked: Optional[bool] = None,
    next_candidate_resolution: Optional[str] = None,
    txn_handoff_enabled: Optional[bool] = None,
    txn_handoff_status: Optional[str] = None,
    txn_handoff_phase: Optional[str] = None,
    txn_handoff_error: Optional[str] = None,
    txn_handoff_soak_contract_status: Optional[str] = None,
    txn_handoff_soak_guard_active: Optional[bool] = None,
    txn_handoff_soak_consecutive_blocked: Optional[int] = None,
    txn_handoff_soak_attempts_total: Optional[int] = None,
) -> Dict[str, Any]:
    row = {
        "schema": "clawd.execution_frontier_transition_attempt.v1",
        "recorded_at": now_iso(),
        "pid": os.getpid(),
        "action": action,
        "operator_reason": advance_reason or None,
        "decision": str(decision or "unknown").strip() or "unknown",
        "advance_applied": bool(advance_applied),
        "block_reason": str(block_reason or "").strip() or None,
        "block_reasons": dedupe_strings(block_reasons or []),
        "error": str(error or "").strip() or None,
        "selector_state": str(selector_state or "").strip() or None,
        "close_condition_met": close_condition_met if isinstance(close_condition_met, bool) else None,
        "next_candidate": str(next_candidate or "").strip() or None,
        "previous_current_wave": previous_current_wave if isinstance(previous_current_wave, int) else None,
        "selected_next_wave": selected_next_wave if isinstance(selected_next_wave, int) else None,
        "autonomous_dispatch_eligible": (
            autonomous_dispatch_eligible if isinstance(autonomous_dispatch_eligible, bool) else None
        ),
        "supervisor_state": str(supervisor_state or "").strip() or None,
        "frontier_queue_ready_count": (
            frontier_queue_ready_count if isinstance(frontier_queue_ready_count, int) and frontier_queue_ready_count >= 0 else None
        ),
        "frontier_queue_dependency_blocked_count": (
            frontier_queue_dependency_blocked_count
            if isinstance(frontier_queue_dependency_blocked_count, int) and frontier_queue_dependency_blocked_count >= 0
            else None
        ),
        "next_candidate_in_frontier_ready": (
            next_candidate_in_frontier_ready if isinstance(next_candidate_in_frontier_ready, bool) else None
        ),
        "next_candidate_dependency_blocked": (
            next_candidate_dependency_blocked if isinstance(next_candidate_dependency_blocked, bool) else None
        ),
        "next_candidate_resolution": str(next_candidate_resolution or "").strip() or None,
        "txn_handoff_enabled": txn_handoff_enabled if isinstance(txn_handoff_enabled, bool) else None,
        "txn_handoff_status": str(txn_handoff_status or "").strip() or None,
        "txn_handoff_phase": str(txn_handoff_phase or "").strip() or None,
        "txn_handoff_error": str(txn_handoff_error or "").strip() or None,
        "txn_handoff_soak_contract_status": str(txn_handoff_soak_contract_status or "").strip() or None,
        "txn_handoff_soak_guard_active": (
            txn_handoff_soak_guard_active if isinstance(txn_handoff_soak_guard_active, bool) else None
        ),
        "txn_handoff_soak_consecutive_blocked": (
            txn_handoff_soak_consecutive_blocked
            if isinstance(txn_handoff_soak_consecutive_blocked, int) and txn_handoff_soak_consecutive_blocked >= 0
            else None
        ),
        "txn_handoff_soak_attempts_total": (
            txn_handoff_soak_attempts_total
            if isinstance(txn_handoff_soak_attempts_total, int) and txn_handoff_soak_attempts_total >= 0
            else None
        ),
        "ledger_path": rel(ledger_path),
        "lock": {
            "path": rel(transition_lock_path),
            "acquired": bool(lock_acquired),
            "mode": "exclusive_nonblocking",
        },
    }
    atomic_write(transition_attempt_latest_path, row)
    append_jsonl(transition_attempt_history_path, row)
    return {
        "latest": rel(transition_attempt_latest_path),
        "history": rel(transition_attempt_history_path),
    }


def acquire_transition_lock() -> Optional[int]:
    transition_lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(transition_lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return None
        raise

    lock_record = {
        "locked_at": now_iso(),
        "pid": os.getpid(),
        "action": action,
    }
    try:
        os.ftruncate(fd, 0)
        os.write(fd, (json.dumps(lock_record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    except Exception:
        pass
    return fd


def release_transition_lock(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        os.close(fd)
    except Exception:
        pass


def collect_autonomous_dispatch_block_reasons(
    payload: Dict[str, Any],
    *,
    selected_next_candidate: Optional[str] = None,
    selected_next_candidate_in_frontier_ready: Optional[bool] = None,
    selected_next_candidate_dependency_blocked: Optional[bool] = None,
) -> Dict[str, Any]:
    supervisor_obj = payload.get("supervisor_state") if isinstance(payload.get("supervisor_state"), dict) else {}
    block_reasons = dedupe_strings(supervisor_obj.get("autonomous_dispatch_block_reasons") or [])

    transition_obj = payload.get("transition") if isinstance(payload.get("transition"), dict) else {}
    stalled_obj = payload.get("stalled_detection") if isinstance(payload.get("stalled_detection"), dict) else {}

    selector_state = str(transition_obj.get("selector_state") or "").strip() or "unknown"
    close_condition_met = _to_bool(transition_obj.get("close_condition_met"))
    payload_next_candidate = str(payload.get("next_candidate") or "").strip() or None
    next_candidate = str(selected_next_candidate or payload_next_candidate or "").strip() or None
    active_worker_count = _to_int(payload.get("active_worker_count")) or 0
    stalled = bool(stalled_obj.get("stalled") is True)
    dispatch_status = str(transition_obj.get("dispatch_status") or "").strip() or "missing"

    frontier_queue = extract_frontier_queue(payload)
    ready_candidate_ids = frontier_queue.get("ready_candidate_ids") or []
    blocked_candidate_ids = frontier_queue.get("blocked_candidate_ids") or []

    if isinstance(selected_next_candidate_in_frontier_ready, bool):
        next_candidate_in_frontier_ready = selected_next_candidate_in_frontier_ready
    elif ready_candidate_ids:
        next_candidate_in_frontier_ready = bool(next_candidate and next_candidate in ready_candidate_ids)
    else:
        next_candidate_in_frontier_ready = None

    if isinstance(selected_next_candidate_dependency_blocked, bool):
        next_candidate_dependency_blocked = selected_next_candidate_dependency_blocked
    else:
        next_candidate_dependency_blocked = bool(next_candidate and next_candidate in blocked_candidate_ids)

    if selector_state != "ready_for_dispatch":
        block_reasons.append("selector_state_not_ready_for_dispatch")
    if close_condition_met is not True:
        block_reasons.append("close_condition_not_met")
    if not next_candidate:
        block_reasons.append("next_candidate_missing")
    if active_worker_count > 0:
        block_reasons.append("active_workers_present")
    if stalled:
        block_reasons.append("stalled_detection_active")
    if dispatch_status == "launched":
        block_reasons.append("dispatch_already_launched")

    if frontier_queue.get("present"):
        if ready_candidate_ids:
            if not next_candidate:
                block_reasons.append("frontier_queue_ready_candidate_missing")
            elif next_candidate_in_frontier_ready is False:
                block_reasons.append("next_candidate_not_ready_in_frontier_queue")
        elif int(frontier_queue.get("dependency_blocked_count") or 0) > 0:
            block_reasons.append("frontier_queue_only_dependency_blocked_candidates")

        if next_candidate_dependency_blocked:
            block_reasons.append("next_candidate_dependency_blocked")

    eligible_flag = _to_bool(supervisor_obj.get("autonomous_dispatch_eligible"))
    if eligible_flag is not True:
        block_reasons.append("supervisor_state_not_autonomous_dispatch_eligible")

    return {
        "supervisor_state": str(supervisor_obj.get("state") or "").strip() or None,
        "autonomous_dispatch_eligible": eligible_flag,
        "block_reasons": dedupe_strings(block_reasons),
        "frontier_queue_ready_count": int(frontier_queue.get("ready_count") or 0),
        "frontier_queue_dependency_blocked_count": int(frontier_queue.get("dependency_blocked_count") or 0),
    }


def _default_txn_handoff_soak_payload() -> Dict[str, Any]:
    return {
        "schema": core_queue_txn_handoff_soak_schema,
        "generated_at": now_iso(),
        "contract_status": "missing",
        "issues": [],
        "guard": {
            "enabled": bool(core_queue_txn_handoff_guard_enabled),
            "consecutive_block_threshold": int(core_queue_txn_handoff_guard_consecutive_block_threshold),
            "active": False,
            "reason": None,
        },
        "counters": {
            "attempts_total": 0,
            "applied_total": 0,
            "blocked_total": 0,
            "skipped_total": 0,
            "consecutive_blocked": 0,
        },
        "reset": {
            "schema": "clawd.core_roadmap_queue_txn_handoff_guard_reset.v1",
            "total": 0,
            "last_reset_at": None,
            "last_reset_reason": None,
            "last_reset_actor": None,
            "last_reset_action": None,
            "last_reset_id": None,
            "history": [],
        },
        "last_attempt": None,
        "last_applied_at": None,
        "last_blocked_at": None,
        "last_skipped_at": None,
        "source_refs": {
            "execution_frontier_ledger": rel(ledger_path),
            "core_roadmap_queue_transaction_runtime": rel(root / "state" / "continuity" / "latest" / "core_roadmap_queue_transaction_runtime.json"),
        },
    }


def _normalize_txn_handoff_soak_reset_event(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None

    reset_at = str(raw.get("reset_at") or "").strip() or None
    reason = str(raw.get("reason") or "").strip() or None
    if not reset_at or not reason:
        return None

    prev_consecutive_blocked = _to_int(raw.get("previous_consecutive_blocked"))
    if prev_consecutive_blocked is None or prev_consecutive_blocked < 0:
        prev_consecutive_blocked = 0

    prev_blocked_total = _to_int(raw.get("previous_blocked_total"))
    if prev_blocked_total is None or prev_blocked_total < 0:
        prev_blocked_total = 0

    return {
        "reset_at": reset_at,
        "reason": reason,
        "actor": str(raw.get("actor") or "").strip() or None,
        "action": str(raw.get("action") or "").strip() or None,
        "reset_id": str(raw.get("reset_id") or "").strip() or None,
        "previous_consecutive_blocked": prev_consecutive_blocked,
        "previous_guard_active": bool(raw.get("previous_guard_active") is True),
        "previous_blocked_total": prev_blocked_total,
    }


def _normalize_txn_handoff_soak(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = _default_txn_handoff_soak_payload()
    issues: List[str] = []

    schema = str(payload.get("schema") or "").strip()
    if schema and schema != core_queue_txn_handoff_soak_schema:
        issues.append("soak_schema_unexpected")

    guard_obj = payload.get("guard") if isinstance(payload.get("guard"), dict) else {}
    if not isinstance(payload.get("guard"), dict):
        issues.append("soak_guard_missing")
    guard_enabled = core_queue_txn_handoff_guard_enabled
    if isinstance(guard_obj.get("enabled"), bool):
        guard_enabled = bool(guard_obj.get("enabled"))
    else:
        issues.append("soak_guard_enabled_missing")

    threshold = _to_int(guard_obj.get("consecutive_block_threshold"))
    if threshold is None or threshold < 1:
        threshold = core_queue_txn_handoff_guard_consecutive_block_threshold

    counters_obj = payload.get("counters") if isinstance(payload.get("counters"), dict) else {}
    if not isinstance(payload.get("counters"), dict):
        issues.append("soak_counters_missing")
    attempts_total = max(0, _to_int(counters_obj.get("attempts_total")) or 0)
    applied_total = max(0, _to_int(counters_obj.get("applied_total")) or 0)
    blocked_total = max(0, _to_int(counters_obj.get("blocked_total")) or 0)
    skipped_total = max(0, _to_int(counters_obj.get("skipped_total")) or 0)
    consecutive_blocked = max(0, _to_int(counters_obj.get("consecutive_blocked")) or 0)

    guard_active = bool(guard_enabled and consecutive_blocked >= threshold and blocked_total > 0)

    last_attempt = payload.get("last_attempt") if isinstance(payload.get("last_attempt"), dict) else None
    if isinstance(last_attempt, dict):
        last_attempt = {
            "recorded_at": str(last_attempt.get("recorded_at") or "").strip() or None,
            "action": str(last_attempt.get("action") or "").strip() or None,
            "decision": str(last_attempt.get("decision") or "").strip() or None,
            "status": str(last_attempt.get("status") or "").strip() or None,
            "phase": str(last_attempt.get("phase") or "").strip() or None,
            "error": str(last_attempt.get("error") or "").strip() or None,
            "task_id": str(last_attempt.get("task_id") or "").strip() or None,
            "selected_next_wave": _to_int(last_attempt.get("selected_next_wave")),
            "next_candidate": str(last_attempt.get("next_candidate") or "").strip() or None,
            "block_reasons": dedupe_strings(last_attempt.get("block_reasons") or []),
        }

    reset_obj = payload.get("reset") if isinstance(payload.get("reset"), dict) else {}
    reset_total = max(0, _to_int(reset_obj.get("total")) or 0)
    reset_history: List[Dict[str, Any]] = []
    for raw in (reset_obj.get("history") if isinstance(reset_obj.get("history"), list) else []):
        row = _normalize_txn_handoff_soak_reset_event(raw)
        if row:
            reset_history.append(row)
    if len(reset_history) > 32:
        reset_history = reset_history[-32:]
    if reset_total < len(reset_history):
        reset_total = len(reset_history)

    last_reset = _normalize_txn_handoff_soak_reset_event(reset_obj.get("last_reset"))
    if last_reset is None and reset_history:
        last_reset = reset_history[-1]

    out["generated_at"] = str(payload.get("generated_at") or out["generated_at"])
    out["contract_status"] = "ok" if not issues else "invalid"
    out["issues"] = dedupe_strings(issues)
    out["guard"] = {
        "enabled": bool(guard_enabled),
        "consecutive_block_threshold": int(threshold),
        "active": bool(guard_active),
        "reason": "consecutive_handoff_blocks_threshold_reached" if guard_active else None,
    }
    out["counters"] = {
        "attempts_total": attempts_total,
        "applied_total": applied_total,
        "blocked_total": blocked_total,
        "skipped_total": skipped_total,
        "consecutive_blocked": consecutive_blocked,
    }
    out["reset"] = {
        "schema": "clawd.core_roadmap_queue_txn_handoff_guard_reset.v1",
        "total": int(reset_total),
        "last_reset_at": (
            str(reset_obj.get("last_reset_at") or "").strip()
            or (str(last_reset.get("reset_at") or "").strip() if isinstance(last_reset, dict) else "")
            or None
        ),
        "last_reset_reason": (
            str(reset_obj.get("last_reset_reason") or "").strip()
            or (str(last_reset.get("reason") or "").strip() if isinstance(last_reset, dict) else "")
            or None
        ),
        "last_reset_actor": (
            str(reset_obj.get("last_reset_actor") or "").strip()
            or (str(last_reset.get("actor") or "").strip() if isinstance(last_reset, dict) else "")
            or None
        ),
        "last_reset_action": (
            str(reset_obj.get("last_reset_action") or "").strip()
            or (str(last_reset.get("action") or "").strip() if isinstance(last_reset, dict) else "")
            or None
        ),
        "last_reset_id": (
            str(reset_obj.get("last_reset_id") or "").strip()
            or (str(last_reset.get("reset_id") or "").strip() if isinstance(last_reset, dict) else "")
            or None
        ),
        "history": reset_history,
        "last_reset": last_reset,
    }
    out["last_attempt"] = last_attempt
    out["last_applied_at"] = str(payload.get("last_applied_at") or "").strip() or None
    out["last_blocked_at"] = str(payload.get("last_blocked_at") or "").strip() or None
    out["last_skipped_at"] = str(payload.get("last_skipped_at") or "").strip() or None
    return out


def _read_txn_handoff_soak() -> Dict[str, Any]:
    if not core_queue_txn_handoff_soak_path.exists():
        return _default_txn_handoff_soak_payload()

    try:
        raw = json.loads(core_queue_txn_handoff_soak_path.read_text(encoding="utf-8"))
    except Exception as exc:
        out = _default_txn_handoff_soak_payload()
        out["contract_status"] = "invalid"
        out["issues"] = [f"soak_parse_failed:{exc}"]
        return out

    if not isinstance(raw, dict):
        out = _default_txn_handoff_soak_payload()
        out["contract_status"] = "invalid"
        out["issues"] = ["soak_not_object"]
        return out

    return _normalize_txn_handoff_soak(raw)


def _write_txn_handoff_soak(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = _normalize_txn_handoff_soak(payload)
    out["generated_at"] = now_iso()
    atomic_write(core_queue_txn_handoff_soak_path, out)
    return out


def _record_txn_handoff_soak_attempt(
    *,
    soak_before: Optional[Dict[str, Any]],
    decision: str,
    next_candidate: Optional[str],
    selected_next_wave: Optional[int],
    block_reasons: Optional[List[str]],
    txn_handoff_status: Optional[str],
    txn_handoff_phase: Optional[str],
    txn_handoff_error: Optional[str],
) -> Dict[str, Any]:
    seed = soak_before if isinstance(soak_before, dict) and soak_before else _read_txn_handoff_soak()
    if str(seed.get("contract_status") or "").strip() == "missing":
        seed["contract_status"] = "ok"

    counters = seed.get("counters") if isinstance(seed.get("counters"), dict) else {}
    attempts_total = max(0, _to_int(counters.get("attempts_total")) or 0) + 1
    applied_total = max(0, _to_int(counters.get("applied_total")) or 0)
    blocked_total = max(0, _to_int(counters.get("blocked_total")) or 0)
    skipped_total = max(0, _to_int(counters.get("skipped_total")) or 0)
    consecutive_blocked = max(0, _to_int(counters.get("consecutive_blocked")) or 0)

    status_text = str(txn_handoff_status or "").strip().lower()
    if status_text == "applied":
        applied_total += 1
        consecutive_blocked = 0
    elif status_text == "blocked":
        blocked_total += 1
        consecutive_blocked += 1
    else:
        skipped_total += 1

    guard_obj = seed.get("guard") if isinstance(seed.get("guard"), dict) else {}
    guard_enabled = (
        bool(guard_obj.get("enabled"))
        if isinstance(guard_obj.get("enabled"), bool)
        else bool(core_queue_txn_handoff_guard_enabled)
    )
    threshold = _to_int(guard_obj.get("consecutive_block_threshold"))
    if threshold is None or threshold < 1:
        threshold = core_queue_txn_handoff_guard_consecutive_block_threshold
    guard_active = bool(guard_enabled and consecutive_blocked >= threshold and blocked_total > 0)

    ts = now_iso()
    updated = dict(seed)
    updated["generated_at"] = ts
    updated["contract_status"] = "ok"
    updated["issues"] = dedupe_strings(updated.get("issues") or [])
    updated["guard"] = {
        "enabled": guard_enabled,
        "consecutive_block_threshold": int(threshold),
        "active": guard_active,
        "reason": "consecutive_handoff_blocks_threshold_reached" if guard_active else None,
    }
    updated["counters"] = {
        "attempts_total": attempts_total,
        "applied_total": applied_total,
        "blocked_total": blocked_total,
        "skipped_total": skipped_total,
        "consecutive_blocked": consecutive_blocked,
    }
    updated["last_attempt"] = {
        "recorded_at": ts,
        "action": action,
        "decision": str(decision or "unknown").strip() or "unknown",
        "status": status_text or "skipped",
        "phase": str(txn_handoff_phase or "").strip() or None,
        "error": str(txn_handoff_error or "").strip() or None,
        "task_id": str(next_candidate or "").strip() or None,
        "selected_next_wave": selected_next_wave if isinstance(selected_next_wave, int) else None,
        "next_candidate": str(next_candidate or "").strip() or None,
        "block_reasons": dedupe_strings(block_reasons or []),
    }
    if status_text == "applied":
        updated["last_applied_at"] = ts
    elif status_text == "blocked":
        updated["last_blocked_at"] = ts
    else:
        updated["last_skipped_at"] = ts

    return _write_txn_handoff_soak(updated)


def _reset_txn_handoff_guard(*, reason: str) -> Dict[str, Any]:
    reason_text = str(reason or "").strip()
    if len(reason_text) < 8:
        return {
            "ok": False,
            "phase": "preflight",
            "error": "reset_reason_too_short",
            "detail": "provide --reason with at least 8 characters",
        }

    soak_before = _read_txn_handoff_soak()
    contract_status = str(soak_before.get("contract_status") or "missing").strip() or "missing"
    if contract_status == "invalid":
        return {
            "ok": False,
            "phase": "preflight",
            "error": "soak_contract_invalid",
            "soak": _txn_handoff_soak_summary(soak_before),
        }

    guard_obj = soak_before.get("guard") if isinstance(soak_before.get("guard"), dict) else {}
    guard_active = bool(guard_obj.get("active") is True)
    if guard_active is not True:
        return {
            "ok": False,
            "phase": "preflight",
            "error": "soak_guard_not_active",
            "soak": _txn_handoff_soak_summary(soak_before),
        }

    counters = soak_before.get("counters") if isinstance(soak_before.get("counters"), dict) else {}
    previous_consecutive = max(0, _to_int(counters.get("consecutive_blocked")) or 0)
    previous_blocked_total = max(0, _to_int(counters.get("blocked_total")) or 0)

    ts = now_iso()
    reset_id = hashlib.sha256(
        f"{ts}|{reason_text}|{core_queue_txn_handoff_guard_reset_actor}|{os.getpid()}".encode("utf-8")
    ).hexdigest()[:24]
    reset_event = {
        "reset_at": ts,
        "reason": reason_text,
        "actor": core_queue_txn_handoff_guard_reset_actor,
        "action": action,
        "reset_id": reset_id,
        "previous_consecutive_blocked": previous_consecutive,
        "previous_guard_active": True,
        "previous_blocked_total": previous_blocked_total,
    }

    reset_obj = soak_before.get("reset") if isinstance(soak_before.get("reset"), dict) else {}
    existing_history = reset_obj.get("history") if isinstance(reset_obj.get("history"), list) else []
    reset_history: List[Dict[str, Any]] = []
    for raw in existing_history:
        row = _normalize_txn_handoff_soak_reset_event(raw)
        if row:
            reset_history.append(row)
    reset_history.append(reset_event)
    if len(reset_history) > 32:
        reset_history = reset_history[-32:]

    reset_total = max(
        len(reset_history),
        max(0, _to_int(reset_obj.get("total")) or 0) + 1,
    )

    updated = dict(soak_before)
    updated["contract_status"] = "ok"
    updated["issues"] = dedupe_strings(updated.get("issues") or [])
    updated["generated_at"] = ts
    updated["guard"] = {
        "enabled": bool(guard_obj.get("enabled") is True),
        "consecutive_block_threshold": max(
            1,
            _to_int(guard_obj.get("consecutive_block_threshold"))
            or core_queue_txn_handoff_guard_consecutive_block_threshold,
        ),
        "active": False,
        "reason": None,
    }
    updated["counters"] = {
        "attempts_total": max(0, _to_int(counters.get("attempts_total")) or 0),
        "applied_total": max(0, _to_int(counters.get("applied_total")) or 0),
        "blocked_total": previous_blocked_total,
        "skipped_total": max(0, _to_int(counters.get("skipped_total")) or 0),
        "consecutive_blocked": 0,
    }
    updated["reset"] = {
        "schema": "clawd.core_roadmap_queue_txn_handoff_guard_reset.v1",
        "total": int(reset_total),
        "last_reset_at": ts,
        "last_reset_reason": reason_text,
        "last_reset_actor": core_queue_txn_handoff_guard_reset_actor,
        "last_reset_action": action,
        "last_reset_id": reset_id,
        "history": reset_history,
        "last_reset": reset_event,
    }

    soak_after = _write_txn_handoff_soak(updated)
    return {
        "ok": True,
        "status": "reset_applied",
        "soak_before": _txn_handoff_soak_summary(soak_before),
        "soak_after": _txn_handoff_soak_summary(soak_after),
        "reset_event": reset_event,
    }


def _txn_handoff_soak_summary(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    counters = payload.get("counters") if isinstance(payload.get("counters"), dict) else {}
    guard_obj = payload.get("guard") if isinstance(payload.get("guard"), dict) else {}
    reset_obj = payload.get("reset") if isinstance(payload.get("reset"), dict) else {}
    last_reset = reset_obj.get("last_reset") if isinstance(reset_obj.get("last_reset"), dict) else {}

    return {
        "path": rel(core_queue_txn_handoff_soak_path),
        "contract_status": str(payload.get("contract_status") or "missing").strip() or "missing",
        "issues": dedupe_strings(payload.get("issues") or []),
        "attempts_total": max(0, _to_int(counters.get("attempts_total")) or 0),
        "applied_total": max(0, _to_int(counters.get("applied_total")) or 0),
        "blocked_total": max(0, _to_int(counters.get("blocked_total")) or 0),
        "skipped_total": max(0, _to_int(counters.get("skipped_total")) or 0),
        "consecutive_blocked": max(0, _to_int(counters.get("consecutive_blocked")) or 0),
        "guard_active": bool(guard_obj.get("active") is True),
        "guard_enabled": bool(guard_obj.get("enabled") is True),
        "guard_consecutive_block_threshold": max(1, _to_int(guard_obj.get("consecutive_block_threshold")) or 1),
        "guard_reset_total": max(0, _to_int(reset_obj.get("total")) or 0),
        "last_guard_reset_at": (
            str(reset_obj.get("last_reset_at") or "").strip()
            or str(last_reset.get("reset_at") or "").strip()
            or None
        ),
        "last_guard_reset_reason": (
            str(reset_obj.get("last_reset_reason") or "").strip()
            or str(last_reset.get("reason") or "").strip()
            or None
        ),
        "last_guard_reset_actor": (
            str(reset_obj.get("last_reset_actor") or "").strip()
            or str(last_reset.get("actor") or "").strip()
            or None
        ),
        "last_guard_reset_action": (
            str(reset_obj.get("last_reset_action") or "").strip()
            or str(last_reset.get("action") or "").strip()
            or None
        ),
        "last_guard_reset_id": (
            str(reset_obj.get("last_reset_id") or "").strip()
            or str(last_reset.get("reset_id") or "").strip()
            or None
        ),
        "last_attempt_at": (
            str((payload.get("last_attempt") or {}).get("recorded_at") or "").strip()
            if isinstance(payload.get("last_attempt"), dict)
            else None
        )
        or None,
    }


def _run_core_queue_txn(args: List[str]) -> Dict[str, Any]:
    cmd = ["bash", str(core_queue_txn_script_path), *args, "--json"]
    env = os.environ.copy()
    env["OPENCLAW_ROOT"] = str(root)
    env["OPENCLAW_INTERNAL_MUTATION"] = "1"
    env["OPENCLAW_INTERNAL_MUTATION_CALLSITE"] = "execution_frontier_ledger.sh:core_queue_txn_handoff"

    try:
        cp = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
            env=env,
            timeout=core_queue_txn_timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "command": cmd,
            "returncode": None,
            "error": "txn_timeout",
            "detail": str(exc),
            "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else None,
            "stderr": (exc.stderr or "") if isinstance(exc.stderr, str) else None,
            "payload": {},
        }

    payload: Dict[str, Any] = {}
    out_text = str(cp.stdout or "").strip()
    if out_text:
        try:
            parsed = json.loads(out_text)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = {}

    return {
        "ok": cp.returncode == 0,
        "command": cmd,
        "returncode": cp.returncode,
        "error": str((payload.get("error") if isinstance(payload, dict) else "") or "").strip() or None,
        "detail": str(cp.stderr or "").strip() or None,
        "stdout": str(cp.stdout or "").strip() or None,
        "stderr": str(cp.stderr or "").strip() or None,
        "payload": payload,
    }


def _perform_core_queue_txn_handoff(*, next_candidate: str) -> Dict[str, Any]:
    if not core_queue_txn_script_path.exists():
        return {
            "ok": False,
            "phase": "preflight",
            "error": "txn_script_missing",
            "detail": str(core_queue_txn_script_path),
        }

    claim_step = _run_core_queue_txn(
        [
            "claim",
            "--worker",
            core_queue_txn_worker_id,
            "--task-id",
            next_candidate,
            "--lease-sec",
            str(core_queue_txn_claim_lease_sec),
            "--reason",
            "execution_frontier_supervisor_handoff",
        ]
    )
    claim_payload = claim_step.get("payload") if isinstance(claim_step.get("payload"), dict) else {}
    if claim_step.get("ok") is not True:
        return {
            "ok": False,
            "phase": "claim",
            "error": str(claim_step.get("error") or "claim_failed").strip() or "claim_failed",
            "step": claim_step,
            "claim_payload": claim_payload,
        }

    claim_obj = claim_payload.get("claim") if isinstance(claim_payload.get("claim"), dict) else {}
    claim_token = str(claim_obj.get("claim_token") or "").strip()
    claim_epoch_raw = claim_obj.get("claim_epoch")
    try:
        claim_epoch = int(claim_epoch_raw)
    except Exception:
        claim_epoch = None

    if not claim_token or claim_epoch is None:
        return {
            "ok": False,
            "phase": "claim",
            "error": "claim_response_missing_token_or_epoch",
            "step": claim_step,
            "claim_payload": claim_payload,
        }

    running_step = _run_core_queue_txn(
        [
            "commit",
            "--task-id",
            next_candidate,
            "--claim-epoch",
            str(claim_epoch),
            "--claim-token",
            claim_token,
            "--to-state",
            "running",
            "--reason",
            "execution_frontier_supervisor_handoff_running",
        ]
    )
    if running_step.get("ok") is not True:
        return {
            "ok": False,
            "phase": "commit_running",
            "error": str(running_step.get("error") or "commit_running_failed").strip() or "commit_running_failed",
            "step": running_step,
            "claim_payload": claim_payload,
        }

    terminal_cmd = [
        "commit",
        "--task-id",
        next_candidate,
        "--claim-epoch",
        str(claim_epoch),
        "--claim-token",
        claim_token,
        "--to-state",
        core_queue_txn_terminal_state,
        "--reason",
        "execution_frontier_supervisor_handoff_terminal",
    ]
    if core_queue_txn_terminal_state == "retry":
        terminal_cmd.extend(["--cooldown-sec", str(core_queue_txn_retry_cooldown_sec)])

    terminal_step = _run_core_queue_txn(terminal_cmd)
    if terminal_step.get("ok") is not True:
        return {
            "ok": False,
            "phase": "commit_terminal",
            "error": str(terminal_step.get("error") or "commit_terminal_failed").strip() or "commit_terminal_failed",
            "step": terminal_step,
            "claim_payload": claim_payload,
        }

    terminal_payload = terminal_step.get("payload") if isinstance(terminal_step.get("payload"), dict) else {}
    return {
        "ok": True,
        "status": "applied",
        "task_id": next_candidate,
        "worker_id": core_queue_txn_worker_id,
        "terminal_state": core_queue_txn_terminal_state,
        "claim_epoch": claim_epoch,
        "runtime_path": str(terminal_payload.get("runtime_path") or "").strip() or None,
        "steps": {
            "claim": claim_payload,
            "commit_running": running_step.get("payload") if isinstance(running_step.get("payload"), dict) else {},
            "commit_terminal": terminal_payload,
        },
    }


def load_payload() -> Dict[str, Any]:
    if not ledger_path.exists():
        err = {
            "ok": False,
            "error": "execution_frontier_ledger_missing",
            "path": str(ledger_path),
        }
        if json_out:
            print(json.dumps(err, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(
                "EXECUTION FRONTIER LEDGER: missing "
                f"(generate via: bash {root}/ops/openclaw/continuity/continuity_current.sh --refresh)"
            )
        raise SystemExit(1)

    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except Exception as exc:
        err = {
            "ok": False,
            "error": "execution_frontier_ledger_invalid_json",
            "path": str(ledger_path),
            "detail": str(exc),
        }
        if json_out:
            print(json.dumps(err, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"EXECUTION FRONTIER LEDGER: invalid_json path={ledger_path}")
        raise SystemExit(1)

    if not isinstance(payload, dict):
        err = {
            "ok": False,
            "error": "execution_frontier_ledger_not_object",
            "path": str(ledger_path),
        }
        if json_out:
            print(json.dumps(err, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"EXECUTION FRONTIER LEDGER: invalid_object path={ledger_path}")
        raise SystemExit(1)

    return payload


lock_fd: Optional[int] = None
if action in LOCKED_ACTIONS:
    lock_fd = acquire_transition_lock()
    if lock_fd is None:
        result = {
            "ok": False,
            "action": action,
            "decision": "BLOCK",
            "error": "execution_frontier_transition_blocked",
            "block_reason": "supervisor_transition_lock_unavailable",
            "block_reasons": ["supervisor_transition_lock_unavailable"],
            "advance_applied": False,
            "path": str(ledger_path),
            "lock_path": str(transition_lock_path),
        }
        result["attempt_evidence"] = record_transition_attempt(
            decision="BLOCK",
            advance_applied=False,
            lock_acquired=False,
            block_reason="supervisor_transition_lock_unavailable",
            block_reasons=["supervisor_transition_lock_unavailable"],
            error="execution_frontier_transition_blocked",
        )

        if json_out:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(
                "EXECUTION FRONTIER LEDGER ADVANCE: "
                "decision=BLOCK "
                "reason=supervisor_transition_lock_unavailable "
                f"lock={transition_lock_path}"
            )
        raise SystemExit(3)

try:
    if action in RESET_GUARD_ACTIONS:
        payload = load_payload()
        reset_result = _reset_txn_handoff_guard(reason=advance_reason or "")
        if reset_result.get("ok") is True:
            soak_after = reset_result.get("soak_after") if isinstance(reset_result.get("soak_after"), dict) else {}
            result = {
                "ok": True,
                "action": action,
                "decision": "APPLY",
                "advance_applied": False,
                "guard_reset_applied": True,
                "path": str(ledger_path),
                "transaction_runtime_handoff_soak": soak_after,
                "reset_event": reset_result.get("reset_event"),
                "lock_path": str(transition_lock_path),
            }
            result["attempt_evidence"] = record_transition_attempt(
                decision="APPLY",
                advance_applied=False,
                lock_acquired=True,
                selector_state=str(((payload.get("transition") or {}) if isinstance(payload.get("transition"), dict) else {}).get("selector_state") or "").strip() or None,
                close_condition_met=(
                    ((payload.get("transition") or {}) if isinstance(payload.get("transition"), dict) else {}).get("close_condition_met")
                    if isinstance(((payload.get("transition") or {}) if isinstance(payload.get("transition"), dict) else {}).get("close_condition_met"), bool)
                    else None
                ),
                next_candidate=str(payload.get("next_candidate") or "").strip() or None,
                previous_current_wave=parse_wave(payload.get("current_wave")),
                selected_next_wave=parse_wave(payload.get("next_candidate_wave")),
                txn_handoff_enabled=True,
                txn_handoff_status="guard_reset_applied",
                txn_handoff_phase="guard_reset",
                txn_handoff_soak_contract_status=str(soak_after.get("contract_status") or "").strip() or None,
                txn_handoff_soak_guard_active=(
                    bool(soak_after.get("guard_active")) if isinstance(soak_after.get("guard_active"), bool) else None
                ),
                txn_handoff_soak_consecutive_blocked=_to_int(soak_after.get("consecutive_blocked")),
                txn_handoff_soak_attempts_total=_to_int(soak_after.get("attempts_total")),
            )
            if json_out:
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(
                    "EXECUTION FRONTIER LEDGER TXN HANDOFF GUARD RESET: "
                    f"applied=true actor={core_queue_txn_handoff_guard_reset_actor}"
                )
            raise SystemExit(0)

        soak = reset_result.get("soak") if isinstance(reset_result.get("soak"), dict) else _txn_handoff_soak_summary(_read_txn_handoff_soak())
        error_code = str(reset_result.get("error") or "guard_reset_failed").strip() or "guard_reset_failed"
        block_reason_map = {
            "reset_reason_too_short": "transaction_runtime_handoff_soak_guard_reset_reason_invalid",
            "soak_contract_invalid": "transaction_runtime_handoff_soak_contract_invalid",
            "soak_guard_not_active": "transaction_runtime_handoff_soak_guard_not_active",
        }
        block_reason = block_reason_map.get(error_code, "transaction_runtime_handoff_soak_guard_reset_blocked")
        result = {
            "ok": False,
            "action": action,
            "decision": "BLOCK",
            "error": "execution_frontier_transition_blocked",
            "block_reason": block_reason,
            "block_reasons": [block_reason],
            "advance_applied": False,
            "guard_reset_applied": False,
            "path": str(ledger_path),
            "transaction_runtime_handoff": {
                "ok": False,
                "phase": "guard_reset",
                "error": error_code,
                "detail": str(reset_result.get("detail") or "").strip() or None,
            },
            "transaction_runtime_handoff_soak": soak,
            "lock_path": str(transition_lock_path),
        }
        result["attempt_evidence"] = record_transition_attempt(
            decision="BLOCK",
            advance_applied=False,
            lock_acquired=True,
            block_reason=block_reason,
            block_reasons=[block_reason],
            error="execution_frontier_transition_blocked",
            selector_state=str(((payload.get("transition") or {}) if isinstance(payload.get("transition"), dict) else {}).get("selector_state") or "").strip() or None,
            close_condition_met=(
                ((payload.get("transition") or {}) if isinstance(payload.get("transition"), dict) else {}).get("close_condition_met")
                if isinstance(((payload.get("transition") or {}) if isinstance(payload.get("transition"), dict) else {}).get("close_condition_met"), bool)
                else None
            ),
            next_candidate=str(payload.get("next_candidate") or "").strip() or None,
            previous_current_wave=parse_wave(payload.get("current_wave")),
            selected_next_wave=parse_wave(payload.get("next_candidate_wave")),
            txn_handoff_enabled=True,
            txn_handoff_status="guard_reset_blocked",
            txn_handoff_phase="guard_reset",
            txn_handoff_error=error_code,
            txn_handoff_soak_contract_status=str(soak.get("contract_status") or "").strip() or None,
            txn_handoff_soak_guard_active=(bool(soak.get("guard_active")) if isinstance(soak.get("guard_active"), bool) else None),
            txn_handoff_soak_consecutive_blocked=_to_int(soak.get("consecutive_blocked")),
            txn_handoff_soak_attempts_total=_to_int(soak.get("attempts_total")),
        )
        if json_out:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(
                "EXECUTION FRONTIER LEDGER TXN HANDOFF GUARD RESET: "
                f"applied=false reason={block_reason}"
            )
        raise SystemExit(3)

    payload = load_payload()

    if action in WAVE_CLOSE_ACTIONS:
        previous_current_wave = parse_wave(payload.get("current_wave"))
        previous_last_completed_wave = parse_wave(payload.get("last_completed_wave"))

        transition_snapshot = payload.get("transition") if isinstance(payload.get("transition"), dict) else {}
        selector_state = str(transition_snapshot.get("selector_state") or "").strip() or "unknown"
        close_condition_met = (
            transition_snapshot.get("close_condition_met")
            if isinstance(transition_snapshot.get("close_condition_met"), bool)
            else None
        )

        frontier_queue = extract_frontier_queue(payload)
        frontier_ready_ids = frontier_queue.get("ready_candidate_ids") or []
        frontier_blocked_ids = frontier_queue.get("blocked_candidate_ids") or []
        frontier_next_candidates = frontier_queue.get("next_candidates") or []

        next_candidate_resolution = "payload"
        payload_next_candidate = str(payload.get("next_candidate") or "").strip()
        next_candidate = ""
        if next_candidate_override:
            next_candidate = next_candidate_override
            next_candidate_resolution = "manual_override"
        elif frontier_next_candidates:
            if payload_next_candidate and payload_next_candidate in frontier_ready_ids:
                next_candidate = payload_next_candidate
                next_candidate_resolution = "payload_frontier_ready"
            else:
                next_candidate = str(frontier_next_candidates[0] or "").strip()
                next_candidate_resolution = "frontier_queue_next_candidate"
        else:
            next_candidate = payload_next_candidate
            next_candidate_resolution = "payload"

        next_candidate = str(next_candidate or "").strip()
        next_candidate_in_frontier_ready: Optional[bool] = None
        if frontier_ready_ids:
            next_candidate_in_frontier_ready = bool(next_candidate and next_candidate in frontier_ready_ids)
        next_candidate_dependency_blocked = bool(next_candidate and next_candidate in frontier_blocked_ids)

        selected_next_wave = parse_wave(payload.get("next_candidate_wave"))
        if selected_next_wave is None:
            selected_next_wave = parse_wave(next_candidate)
        if selected_next_wave is None and isinstance(previous_current_wave, int):
            selected_next_wave = max(0, previous_current_wave + 1)

        block_reasons: List[str] = []
        if selector_state != "ready_for_dispatch":
            block_reasons.append("selector_state_not_ready_for_dispatch")
        if close_condition_met is not True:
            block_reasons.append("close_condition_not_met")
        if not next_candidate:
            block_reasons.append("next_candidate_missing")
        if not isinstance(previous_current_wave, int):
            block_reasons.append("current_wave_missing")

        if action in {"supervisor-advance-wave-close", "supervisor-autonomous-dispatch"} and frontier_queue.get("present"):
            frontier_ready_count = int(frontier_queue.get("ready_count") or 0)
            frontier_blocked_count = int(frontier_queue.get("dependency_blocked_count") or 0)
            if frontier_ready_count <= 0:
                if frontier_blocked_count > 0:
                    block_reasons.append("frontier_queue_only_dependency_blocked_candidates")
                else:
                    block_reasons.append("frontier_queue_ready_candidate_missing")
            if next_candidate and next_candidate_in_frontier_ready is False:
                block_reasons.append("next_candidate_not_ready_in_frontier_queue")
            if next_candidate_dependency_blocked:
                block_reasons.append("next_candidate_dependency_blocked")

        autonomous_dispatch_eligible: Optional[bool] = None
        supervisor_state_name: Optional[str] = None
        if action == "supervisor-autonomous-dispatch":
            autonomous_guard = collect_autonomous_dispatch_block_reasons(
                payload,
                selected_next_candidate=next_candidate or None,
                selected_next_candidate_in_frontier_ready=next_candidate_in_frontier_ready,
                selected_next_candidate_dependency_blocked=next_candidate_dependency_blocked,
            )
            autonomous_dispatch_eligible = autonomous_guard.get("autonomous_dispatch_eligible")
            supervisor_state_name = autonomous_guard.get("supervisor_state")
            block_reasons.extend(autonomous_guard.get("block_reasons") or [])

        txn_handoff_enabled_for_action = bool(
            core_queue_txn_handoff_enabled
            and action == "supervisor-autonomous-dispatch"
            and str(frontier_queue.get("queue_source") or "").strip().startswith("core_roadmap_queue_layer")
        )
        txn_handoff_result: Optional[Dict[str, Any]] = None
        txn_handoff_soak_before: Optional[Dict[str, Any]] = None
        txn_handoff_soak_after: Optional[Dict[str, Any]] = None
        if txn_handoff_enabled_for_action:
            txn_handoff_soak_before = _read_txn_handoff_soak()

        if txn_handoff_enabled_for_action and not block_reasons:
            soak_contract_status = str((txn_handoff_soak_before or {}).get("contract_status") or "missing").strip() or "missing"
            soak_guard = (
                (txn_handoff_soak_before or {}).get("guard")
                if isinstance((txn_handoff_soak_before or {}).get("guard"), dict)
                else {}
            )
            soak_guard_active = bool(soak_guard.get("active") is True)
            if soak_contract_status == "invalid":
                txn_handoff_result = {
                    "ok": False,
                    "phase": "preflight",
                    "error": "soak_contract_invalid",
                    "soak": _txn_handoff_soak_summary(txn_handoff_soak_before),
                }
                block_reasons.append("transaction_runtime_handoff_soak_contract_invalid")
            elif soak_guard_active:
                txn_handoff_result = {
                    "ok": False,
                    "phase": "preflight",
                    "error": "soak_guard_active",
                    "soak": _txn_handoff_soak_summary(txn_handoff_soak_before),
                }
                block_reasons.append("transaction_runtime_handoff_soak_guard_active")
            elif not next_candidate:
                block_reasons.append("transaction_runtime_handoff_next_candidate_missing")
            elif not next_candidate.startswith("core_roadmap:"):
                block_reasons.append("transaction_runtime_handoff_non_core_task")
            else:
                txn_handoff_result = _perform_core_queue_txn_handoff(next_candidate=next_candidate)
                if txn_handoff_result.get("ok") is not True:
                    phase = str(txn_handoff_result.get("phase") or "unknown").strip() or "unknown"
                    error = str(txn_handoff_result.get("error") or "handoff_failed").strip() or "handoff_failed"
                    block_reasons.append(f"transaction_runtime_handoff_{phase}_failed:{error}")
                    block_reasons.append(f"transaction_runtime_handoff_{phase}_failed")
                else:
                    payload["transaction_runtime_handoff"] = txn_handoff_result

        block_reasons = dedupe_strings(block_reasons)
        txn_handoff_status = None
        txn_handoff_phase = None
        txn_handoff_error = None
        if isinstance(txn_handoff_result, dict):
            txn_handoff_phase = str(txn_handoff_result.get("phase") or "").strip() or None
            txn_handoff_error = str(txn_handoff_result.get("error") or "").strip() or None
            txn_handoff_status = "applied" if txn_handoff_result.get("ok") is True else "blocked"

        if txn_handoff_enabled_for_action and not txn_handoff_status:
            handoff_blocked = any(
                str(reason or "").startswith("transaction_runtime_handoff_") for reason in (block_reasons or [])
            )
            txn_handoff_status = "blocked" if handoff_blocked else "skipped"

        if txn_handoff_enabled_for_action:
            txn_handoff_soak_after = _record_txn_handoff_soak_attempt(
                soak_before=txn_handoff_soak_before,
                decision="BLOCK" if block_reasons else "APPLY",
                next_candidate=next_candidate or None,
                selected_next_wave=selected_next_wave,
                block_reasons=block_reasons,
                txn_handoff_status=txn_handoff_status,
                txn_handoff_phase=txn_handoff_phase,
                txn_handoff_error=txn_handoff_error,
            )

        if block_reasons:
            result = {
                "ok": False,
                "action": action,
                "decision": "BLOCK",
                "error": "execution_frontier_transition_blocked",
                "block_reason": block_reasons[0],
                "block_reasons": block_reasons,
                "advance_applied": False,
                "previous_current_wave": previous_current_wave,
                "selected_next_wave": selected_next_wave,
                "next_candidate": next_candidate or None,
                "next_candidate_resolution": next_candidate_resolution,
                "next_candidate_in_frontier_ready": (
                    next_candidate_in_frontier_ready if isinstance(next_candidate_in_frontier_ready, bool) else None
                ),
                "next_candidate_dependency_blocked": bool(next_candidate_dependency_blocked),
                "frontier_queue_ready_count": int(frontier_queue.get("ready_count") or 0),
                "frontier_queue_dependency_blocked_count": int(frontier_queue.get("dependency_blocked_count") or 0),
                "selector_state": selector_state,
                "close_condition_met": close_condition_met,
                "path": str(ledger_path),
                "payload": payload,
                "lock_path": str(transition_lock_path),
            }
            if txn_handoff_enabled_for_action:
                result["transaction_runtime_handoff"] = txn_handoff_result or {
                    "ok": False,
                    "phase": txn_handoff_phase,
                    "error": txn_handoff_error or "handoff_blocked_before_execution",
                }
                result["transaction_runtime_handoff_soak"] = _txn_handoff_soak_summary(txn_handoff_soak_after)

            result["attempt_evidence"] = record_transition_attempt(
                decision="BLOCK",
                advance_applied=False,
                lock_acquired=True,
                block_reason=block_reasons[0],
                block_reasons=block_reasons,
                error="execution_frontier_transition_blocked",
                selector_state=selector_state,
                close_condition_met=close_condition_met,
                next_candidate=next_candidate or None,
                previous_current_wave=previous_current_wave,
                selected_next_wave=selected_next_wave,
                autonomous_dispatch_eligible=autonomous_dispatch_eligible,
                supervisor_state=supervisor_state_name,
                frontier_queue_ready_count=int(frontier_queue.get("ready_count") or 0),
                frontier_queue_dependency_blocked_count=int(frontier_queue.get("dependency_blocked_count") or 0),
                next_candidate_in_frontier_ready=next_candidate_in_frontier_ready,
                next_candidate_dependency_blocked=next_candidate_dependency_blocked,
                next_candidate_resolution=next_candidate_resolution,
                txn_handoff_enabled=txn_handoff_enabled_for_action,
                txn_handoff_status=txn_handoff_status,
                txn_handoff_phase=txn_handoff_phase,
                txn_handoff_error=txn_handoff_error,
                txn_handoff_soak_contract_status=(
                    str((txn_handoff_soak_after or {}).get("contract_status") or "").strip() or None
                ),
                txn_handoff_soak_guard_active=(
                    bool((((txn_handoff_soak_after or {}).get("guard") or {}).get("active") is True))
                    if isinstance((txn_handoff_soak_after or {}).get("guard"), dict)
                    else None
                ),
                txn_handoff_soak_consecutive_blocked=(
                    _to_int(((txn_handoff_soak_after or {}).get("counters") or {}).get("consecutive_blocked"))
                    if isinstance((txn_handoff_soak_after or {}).get("counters"), dict)
                    else None
                ),
                txn_handoff_soak_attempts_total=(
                    _to_int(((txn_handoff_soak_after or {}).get("counters") or {}).get("attempts_total"))
                    if isinstance((txn_handoff_soak_after or {}).get("counters"), dict)
                    else None
                ),
            )

            if json_out:
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(
                    "EXECUTION FRONTIER LEDGER ADVANCE: "
                    f"decision=BLOCK "
                    f"reason={result.get('block_reason')} "
                    f"selector={selector_state} "
                    f"close={close_condition_met} "
                    f"next={next_candidate or '-'}"
                )
            raise SystemExit(3)

        if next_candidate:
            payload["next_candidate"] = next_candidate
            if next_candidate_override:
                payload["next_candidate_source"] = "manual_override"
            elif next_candidate_resolution in {"frontier_queue_next_candidate", "payload_frontier_ready"}:
                payload["next_candidate_source"] = "frontier_queue_ready_candidate"

        frontier_queue_obj = payload.get("frontier_queue") if isinstance(payload.get("frontier_queue"), dict) else {}
        frontier_queue_obj["schema"] = str(frontier_queue_obj.get("schema") or "clawd.execution_frontier_queue.v1")
        frontier_queue_obj["ready_count"] = int(frontier_queue.get("ready_count") or 0)
        frontier_queue_obj["dependency_blocked_count"] = int(frontier_queue.get("dependency_blocked_count") or 0)
        frontier_queue_obj["dependency_model_available"] = bool(frontier_queue.get("dependency_model_available") is True)
        frontier_queue_obj["next_candidates"] = dedupe_strings(
            (frontier_queue.get("next_candidates") or []) + ([next_candidate] if next_candidate else [])
        )[:5]
        payload["frontier_queue"] = frontier_queue_obj
        payload["next_candidates"] = frontier_queue_obj.get("next_candidates")

        payload["last_completed_wave"] = max(
            previous_current_wave,
            previous_last_completed_wave if isinstance(previous_last_completed_wave, int) else 0,
        )
        if isinstance(selected_next_wave, int):
            payload["current_wave"] = selected_next_wave
            payload["next_candidate_wave"] = selected_next_wave

        payload["active_worker_count"] = 0
        payload["program_state"] = "waiting" if next_candidate else "idle"
        payload["blocked_reason"] = None
        payload["last_progress_at"] = now_iso()

        labels = dedupe_strings(payload.get("active_labels") or [])
        labels = [label for label in labels if not label.startswith("dispatch:")]
        labels.append("transition:wave_closed")
        if next_candidate:
            labels.append(f"focus:{next_candidate}")
        payload["active_labels"] = dedupe_strings(labels)[:16]

        transition = payload.get("transition") if isinstance(payload.get("transition"), dict) else {}
        transition["selector_state"] = "advanced_wave_closed"
        transition["close_condition_met"] = True
        transition["proposed_next_wave"] = selected_next_wave

        default_reason = "manual_advance_wave_close"
        if action == "supervisor-advance-wave-close":
            default_reason = "supervisor_guarded_wave_close"
        elif action == "supervisor-autonomous-dispatch":
            default_reason = "supervisor_autonomous_dispatch"

        transition["reason"] = advance_reason or default_reason
        transition["advanced_at"] = now_iso()
        transition["advance_applied"] = True
        transition["previous_current_wave"] = previous_current_wave
        transition["advance_mode"] = "supervisor_guarded"
        if action == "supervisor-autonomous-dispatch":
            transition["advance_mode"] = "supervisor_autonomous_dispatch"
        if txn_handoff_enabled_for_action:
            transition["transaction_runtime_handoff"] = {
                "enabled": True,
                "status": txn_handoff_status or ("applied" if isinstance(txn_handoff_result, dict) and txn_handoff_result.get("ok") is True else "unknown"),
                "worker_id": core_queue_txn_worker_id,
                "terminal_state": core_queue_txn_terminal_state,
                "phase": txn_handoff_phase,
                "error": txn_handoff_error,
                "claim_epoch": (
                    int(txn_handoff_result.get("claim_epoch"))
                    if isinstance(txn_handoff_result, dict) and isinstance(txn_handoff_result.get("claim_epoch"), int)
                    else None
                ),
                "soak": _txn_handoff_soak_summary(txn_handoff_soak_after),
            }
        payload["transition"] = transition

        stalled_detection = payload.get("stalled_detection") if isinstance(payload.get("stalled_detection"), dict) else {}
        stalled_detection["stalled"] = False
        stalled_detection["idle_for_sec"] = 0
        stalled_detection["reason"] = None
        stalled_detection["stalled_after_sec"] = _to_int(stalled_detection.get("stalled_after_sec")) or 1800
        payload["stalled_detection"] = stalled_detection

        supervisor_state_obj = payload.get("supervisor_state") if isinstance(payload.get("supervisor_state"), dict) else {}
        supervisor_state_obj["state_version"] = "execution_frontier_supervisor_state.v1"
        supervisor_state_obj["state"] = "advanced_wave_closed"
        supervisor_state_obj["reason"] = transition.get("reason")
        supervisor_state_obj["idle_phase"] = "grace"
        supervisor_state_obj["autonomous_dispatch_eligible"] = False
        supervisor_state_obj["autonomous_dispatch_block_reasons"] = ["selector_state_not_ready_for_dispatch"]
        supervisor_state_obj["selector_state"] = "advanced_wave_closed"
        supervisor_state_obj["dispatch_status"] = str(transition.get("dispatch_status") or "missing") or "missing"
        supervisor_state_obj["close_condition_met"] = True
        payload["supervisor_state"] = supervisor_state_obj

        if txn_handoff_enabled_for_action:
            payload["transaction_runtime_handoff_soak"] = _txn_handoff_soak_summary(txn_handoff_soak_after)
            source_refs_obj = payload.get("source_refs") if isinstance(payload.get("source_refs"), dict) else {}
            source_refs_obj["core_roadmap_queue_transaction_handoff_soak"] = rel(core_queue_txn_handoff_soak_path)
            payload["source_refs"] = source_refs_obj

        payload["generated_at"] = now_iso()
        atomic_write(ledger_path, payload)

        result = {
            "ok": True,
            "action": action,
            "decision": "APPLY",
            "advance_applied": True,
            "previous_current_wave": previous_current_wave,
            "selected_next_wave": selected_next_wave,
            "next_candidate": next_candidate or None,
            "next_candidate_resolution": next_candidate_resolution,
            "path": str(ledger_path),
            "payload": payload,
            "lock_path": str(transition_lock_path),
        }
        if txn_handoff_enabled_for_action:
            result["transaction_runtime_handoff"] = txn_handoff_result or {
                "ok": True,
                "status": "applied",
                "task_id": next_candidate or None,
            }
            result["transaction_runtime_handoff_soak"] = _txn_handoff_soak_summary(txn_handoff_soak_after)
        result["attempt_evidence"] = record_transition_attempt(
            decision="APPLY",
            advance_applied=True,
            lock_acquired=True,
            selector_state=selector_state,
            close_condition_met=close_condition_met,
            next_candidate=next_candidate or None,
            previous_current_wave=previous_current_wave,
            selected_next_wave=selected_next_wave,
            autonomous_dispatch_eligible=autonomous_dispatch_eligible,
            supervisor_state=supervisor_state_name,
            frontier_queue_ready_count=int(frontier_queue.get("ready_count") or 0),
            frontier_queue_dependency_blocked_count=int(frontier_queue.get("dependency_blocked_count") or 0),
            next_candidate_in_frontier_ready=next_candidate_in_frontier_ready,
            next_candidate_dependency_blocked=next_candidate_dependency_blocked,
            next_candidate_resolution=next_candidate_resolution,
            txn_handoff_enabled=txn_handoff_enabled_for_action,
            txn_handoff_status=txn_handoff_status,
            txn_handoff_phase=txn_handoff_phase,
            txn_handoff_error=txn_handoff_error,
            txn_handoff_soak_contract_status=(
                str((txn_handoff_soak_after or {}).get("contract_status") or "").strip() or None
            ),
            txn_handoff_soak_guard_active=(
                bool((((txn_handoff_soak_after or {}).get("guard") or {}).get("active") is True))
                if isinstance((txn_handoff_soak_after or {}).get("guard"), dict)
                else None
            ),
            txn_handoff_soak_consecutive_blocked=(
                _to_int(((txn_handoff_soak_after or {}).get("counters") or {}).get("consecutive_blocked"))
                if isinstance((txn_handoff_soak_after or {}).get("counters"), dict)
                else None
            ),
            txn_handoff_soak_attempts_total=(
                _to_int(((txn_handoff_soak_after or {}).get("counters") or {}).get("attempts_total"))
                if isinstance((txn_handoff_soak_after or {}).get("counters"), dict)
                else None
            ),
        )

        if json_out:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(
                "EXECUTION FRONTIER LEDGER ADVANCE: "
                f"applied={result.get('advance_applied')} "
                f"from_wave={previous_current_wave} "
                f"to_wave={selected_next_wave} "
                f"next={next_candidate or '-'}"
            )
        raise SystemExit(0)

    if action != "show":
        err = {
            "ok": False,
            "error": "unknown_action",
            "action": action,
        }
        if json_out:
            print(json.dumps(err, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"EXECUTION FRONTIER LEDGER: unknown action '{action}'")
        raise SystemExit(2)

    if json_out:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            "EXECUTION FRONTIER LEDGER: "
            f"state={payload.get('program_state') or 'unknown'} "
            f"wave={payload.get('current_wave')} "
            f"last_completed={payload.get('last_completed_wave')} "
            f"frontier={payload.get('frontier_lane') or '-'} "
            f"next={payload.get('next_candidate') or '-'} "
            f"workers={payload.get('active_worker_count')}"
        )
finally:
    release_transition_lock(lock_fd)
PY
