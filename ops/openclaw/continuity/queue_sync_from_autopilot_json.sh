#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
STATE_JSON="$ROOT/ops/autopilot/state/hl_terminal_v1.json"
DB_PATH="$ROOT/state/continuity/continuity_os.sqlite"
SOURCE="autopilot"
ACTION_TOKEN=""
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"
MUTATION_TICKET=""
declare -a MUTATION_ATTESTATIONS=()
declare -a MUTATION_ATTESTATION_OBJECTS=()
INVALID_PROVIDER_SUMMARY_MODE="${OPENCLAW_QUEUE_SYNC_INVALID_PROVIDER_SUMMARY_MODE:-drop}"

usage() {
  cat <<'EOF'
Usage: queue_sync_from_autopilot_json.sh [options]

Mirror autopilot JSON state into continuity work_queue + task_transitions.

Options:
  --json <path>     Autopilot state JSON path (default: ops/autopilot/state/hl_terminal_v1.json)
  --db <path>       Continuity sqlite path (default: state/continuity/continuity_os.sqlite)
  --source <name>   Source label for queue rows (default: autopilot)
  --action-token <value>
                    Canonical mutation token for direct entrypoint use.
  --truth-anchor <value>
                    Legacy alias of --action-token.
  --allow-legacy-anchor
                    Allow legacy anchor-only token mode for direct token validation.
  --mutation-ticket <value>
                    Authority ticket JSON string, @path, or path (high-risk token path).
  --attestation <name>
                    Satisfied authority attestation name (repeatable).
  --attestation-object <value>
                    Structured authority attestation JSON string, @path, or path (repeatable).
  --invalid-provider-summary-mode <drop|fail_close>
                    Handling mode for invalid autopilot.provider_failure_summary.v1 payloads
                    (default: OPENCLAW_QUEUE_SYNC_INVALID_PROVIDER_SUMMARY_MODE or drop).
  --fail-close-invalid-provider-summary
                    Alias for --invalid-provider-summary-mode fail_close.
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      STATE_JSON="${2:-}"; shift 2 ;;
    --db)
      DB_PATH="${2:-}"; shift 2 ;;
    --source)
      SOURCE="${2:-}"; shift 2 ;;
    --action-token|--truth-anchor)
      ACTION_TOKEN="${2:-}"; shift 2 ;;
    --allow-legacy-anchor)
      ALLOW_LEGACY_ANCHOR=1; shift ;;
    --mutation-ticket)
      MUTATION_TICKET="${2:-}"; shift 2 ;;
    --attestation)
      MUTATION_ATTESTATIONS+=("${2:-}"); shift 2 ;;
    --attestation-object)
      MUTATION_ATTESTATION_OBJECTS+=("${2:-}"); shift 2 ;;
    --invalid-provider-summary-mode)
      INVALID_PROVIDER_SUMMARY_MODE="${2:-}"; shift 2 ;;
    --fail-close-invalid-provider-summary)
      INVALID_PROVIDER_SUMMARY_MODE="fail_close"; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

case "$INVALID_PROVIDER_SUMMARY_MODE" in
  drop|fail_close)
    ;;
  *)
    echo "invalid --invalid-provider-summary-mode: $INVALID_PROVIDER_SUMMARY_MODE (expected: drop|fail_close)" >&2
    exit 2
    ;;
esac

guard_args=(
  --script "queue_sync_from_autopilot_json.sh"
  --risk-tier "high"
  --mutation-operation "queue_sync_from_autopilot_json:sync"
)
if [[ -n "$ACTION_TOKEN" ]]; then
  guard_args+=(--action-token "$ACTION_TOKEN")
fi
if [[ "$ALLOW_LEGACY_ANCHOR" == "1" ]]; then
  guard_args+=(--allow-legacy-anchor)
fi
if [[ -n "$MUTATION_TICKET" ]]; then
  guard_args+=(--mutation-ticket "$MUTATION_TICKET")
fi
for att in "${MUTATION_ATTESTATIONS[@]}"; do
  if [[ -n "${att:-}" ]]; then
    guard_args+=(--attestation "$att")
  fi
done
for att_obj in "${MUTATION_ATTESTATION_OBJECTS[@]}"; do
  if [[ -n "${att_obj:-}" ]]; then
    guard_args+=(--attestation-object "$att_obj")
  fi
done
"$ROOT/ops/openclaw/continuity/mutator_ingress_guard.sh" "${guard_args[@]}"

if [[ ! -f "$STATE_JSON" ]]; then
  echo "{\"ok\":false,\"error\":\"missing_state_json\",\"path\":\"$STATE_JSON\"}"
  exit 1
fi

OPENCLAW_CONTINUITY_DB_PATH="$DB_PATH" "$ROOT/ops/openclaw/continuity/init_db.sh" >/dev/null

python3 - "$STATE_JSON" "$DB_PATH" "$SOURCE" "$ROOT" "$INVALID_PROVIDER_SUMMARY_MODE" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import sqlite3
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

state_path = pathlib.Path(sys.argv[1]).resolve()
db_path = pathlib.Path(sys.argv[2]).resolve()
source = sys.argv[3]
root = pathlib.Path(sys.argv[4]).resolve()
invalid_provider_summary_mode = str(sys.argv[5] or "drop").strip().lower()
if invalid_provider_summary_mode not in {"drop", "fail_close"}:
    raise SystemExit(f"invalid invalid_provider_summary_mode: {invalid_provider_summary_mode}")

if str(root / "src") not in sys.path:
    sys.path.insert(0, str(root / "src"))
continuity_path = root / "ops" / "openclaw" / "continuity"
if str(continuity_path) not in sys.path:
    sys.path.insert(0, str(continuity_path))

try:
    from walletdb.provider_failure import (  # type: ignore
        PROVIDER_FAILURE_SUMMARY_SCHEMA_VERSION,
        validate_provider_failure_summary,
    )
except Exception as exc:  # pragma: no cover - defensive fallback
    PROVIDER_FAILURE_SUMMARY_SCHEMA_VERSION = "autopilot.provider_failure_summary.v1"
    _validator_import_error = str(exc)

    def validate_provider_failure_summary(summary: Any, *, strict: bool = True) -> Dict[str, Any]:
        return {
            "ok": False,
            "issues": [f"provider_failure_summary_validator_unavailable:{_validator_import_error}"],
            "schema_version": str((summary or {}).get("schema_version") if isinstance(summary, dict) else ""),
        }

try:
    from fixed_now import now_iso_utc as _helper_now_iso_utc, now_ts as _helper_now_ts, ts_to_iso_utc as _helper_ts_to_iso_utc  # type: ignore
except Exception:  # pragma: no cover - optional helper in minimal test roots
    _helper_now_iso_utc = None
    _helper_now_ts = None
    _helper_ts_to_iso_utc = None

state = json.loads(state_path.read_text(encoding="utf-8"))


def current_ts() -> int:
    if callable(_helper_now_ts):
        try:
            return int(_helper_now_ts())
        except Exception:
            pass
    return int(dt.datetime.now(dt.timezone.utc).timestamp())


def now_iso() -> str:
    if callable(_helper_now_iso_utc):
        try:
            return str(_helper_now_iso_utc())
        except Exception:
            pass
    return dt.datetime.fromtimestamp(current_ts(), tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ts_to_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        n = int(value)
        if n <= 0:
            return None
        if callable(_helper_ts_to_iso_utc):
            try:
                return str(_helper_ts_to_iso_utc(n))
            except Exception:
                pass
        return dt.datetime.fromtimestamp(n, tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def future_cooldown_until(next_after_ts: Any, *, queue_status: str, now_ts: Optional[int] = None) -> Optional[str]:
    """Project autopilot next_after_ts into queue cooldown_until only when still in the future."""
    if str(queue_status or "").upper() != "QUEUED":
        return None
    try:
        ts = int(next_after_ts)
    except Exception:
        return None
    if ts <= 0:
        return None
    ref_now = int(now_ts) if now_ts is not None else current_ts()
    if ts <= ref_now:
        return None
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


ALLOWED_ROLES = {"planner", "executor", "validator", "sre_watchdog", "librarian", "outer_gate"}


def role_for_step(step_id: str) -> str:
    planner = {
        "sync_spec_context",
        "audit_alignment",
        "audit_runtime_probes",
        "audit_breaktests",
        "synth_fix_plan",
    }
    if step_id in planner:
        return "planner"
    if step_id == "apply_fixes":
        return "executor"
    if step_id == "quality_gate":
        return "validator"
    return "outer_gate"


def role_required_for_status(base_role: str, queue_status: str) -> str:
    role = str(base_role or "").strip()
    if role not in ALLOWED_ROLES:
        role = "planner"

    if queue_status == "REVIEW":
        return "validator"
    if queue_status == "DONE":
        return "librarian"
    if queue_status in {"BLOCKED", "FAILED", "ROLLED_BACK"}:
        return "sre_watchdog"
    if queue_status == "RUNNING":
        return role
    # QUEUED default keeps declared execution boundary for deterministic claims.
    return role


def map_status(step_id: str, raw: str) -> str:
    v = (raw or "").strip().lower()
    if v == "queued":
        return "QUEUED"
    if v == "processing":
        return "RUNNING"
    if v == "running":
        return "RUNNING"
    if v == "blocked":
        return "BLOCKED"
    if v in {"failed", "error"}:
        return "FAILED"
    if v == "done":
        if step_id == "quality_gate":
            return "DONE"
        return "REVIEW"
    return "QUEUED"


def first_line(text: Any, limit: int = 240) -> str:
    if text is None:
        return ""
    raw = str(text).strip().splitlines()
    if not raw:
        return ""
    line = raw[0].strip()
    return line[:limit]


def event_id(task_id: str, from_status: Optional[str], to_status: str, created_at: str, reason: str) -> str:
    seed = f"{task_id}|{from_status or ''}|{to_status}|{created_at}|{reason}"
    return "tevt_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def artifact_id(task_id: str, artifact_path: str, artifact_type: str) -> str:
    seed = f"{task_id}|{artifact_type}|{artifact_path}"
    return "tart_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def classify_artifact_type(path: str) -> str:
    low = str(path or "").lower()
    if low.endswith(".log"):
        return "run_log"
    if low.endswith(".exit"):
        return "run_exit"
    if low.endswith(".md"):
        return "markdown"
    if low.endswith(".json"):
        return "json"
    if low.endswith(".sqlite"):
        return "sqlite"
    if "autopilot_artifacts" in low:
        return "autopilot_artifact"
    if "ops/autopilot/state" in low:
        return "autopilot_state"
    return "evidence"


def to_rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except Exception:
        return str(path)


def dedupe_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        v = str(raw or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def normalize_sha256(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if len(raw) == 64 and all(ch in "0123456789abcdef" for ch in raw):
        return raw
    return None


def parse_evidence_refs(raw: str) -> List[str]:
    return dedupe_keep_order([p.strip() for p in str(raw or "").split("|") if str(p).strip()])


gate_summary_validation_issues: List[Dict[str, Any]] = []
gate_summary_fail_close_issues: List[Dict[str, Any]] = []
DELEGATED_GATE_SUMMARY_SCHEMA_VERSION = "autopilot.delegated_gate_summary.v1"


def _record_gate_summary_issue(
    *,
    context: str,
    summary: Optional[Dict[str, Any]],
    issues: List[str],
    code: str,
    fail_close: bool,
) -> None:
    issue_entry = {
        "context": context,
        "code": code,
        "schema_version": str((summary or {}).get("schema_version") or ""),
        "summary_signature": str((summary or {}).get("summary_signature") or "")[:80] or None,
        "issues": [str(item) for item in issues if str(item).strip()][:8],
    }
    gate_summary_validation_issues.append(issue_entry)
    if fail_close and invalid_provider_summary_mode == "fail_close":
        gate_summary_fail_close_issues.append(issue_entry)


def _resolve_gate_summary_path(raw: Any) -> Optional[pathlib.Path]:
    text = str(raw or "").strip()
    if not text:
        return None
    path = pathlib.Path(text)
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve()
    except Exception:
        return path


def _sha256_file_with_status(path: Optional[pathlib.Path]) -> Tuple[Optional[str], str]:
    if path is None:
        return None, "path_missing"
    try:
        if not path.exists():
            return None, "path_missing"
        if not path.is_file():
            return None, "path_not_file"
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), "ok"
    except Exception:
        return None, "path_unreadable"


def _bind_delegated_gate_summary(summary: Dict[str, Any], *, context: str) -> Dict[str, Any]:
    schema_version = str(summary.get("schema_version") or "")
    if schema_version != DELEGATED_GATE_SUMMARY_SCHEMA_VERSION:
        return dict(summary)

    bound = dict(summary)

    fields = [
        ("completion_packet_path", "completion_packet_sha256", "completion_packet"),
        ("decision_path", "decision_sha256", "decision"),
    ]
    for path_field, sha_field, binding_kind in fields:
        raw_path = str(summary.get(path_field) or "").strip()
        declared_sha = normalize_sha256(summary.get(sha_field))
        if not declared_sha:
            bound.pop(sha_field, None)

        if not raw_path:
            if declared_sha:
                _record_gate_summary_issue(
                    context=context,
                    summary=summary,
                    code=f"delegated_{binding_kind}_sha256_unverifiable_dropped",
                    issues=[f"{path_field}_missing", "digest_dropped"],
                    fail_close=False,
                )
                bound.pop(sha_field, None)
            continue

        actual_sha, status = _sha256_file_with_status(_resolve_gate_summary_path(raw_path))
        if actual_sha:
            if declared_sha and declared_sha != actual_sha:
                _record_gate_summary_issue(
                    context=context,
                    summary=summary,
                    code=f"delegated_{binding_kind}_sha256_mismatch_repaired",
                    issues=[
                        f"{sha_field}_mismatch",
                        f"declared={declared_sha[:16]}...",
                        f"actual={actual_sha[:16]}...",
                    ],
                    fail_close=False,
                )
            bound[sha_field] = actual_sha
            continue

        issue_bits = [f"{path_field}_{status}"]
        if declared_sha:
            issue_bits.append(f"declared={declared_sha[:16]}...")
            issue_bits.append("digest_dropped")
        issue_bits.append("path_dropped")
        _record_gate_summary_issue(
            context=context,
            summary=summary,
            code=f"delegated_{binding_kind}_path_unverifiable_dropped",
            issues=issue_bits,
            fail_close=False,
        )
        bound.pop(sha_field, None)
        bound.pop(path_field, None)

    return bound


def _validate_gate_summary(summary: Dict[str, Any], *, context: str) -> Optional[Dict[str, Any]]:
    schema_version = str(summary.get("schema_version") or "")
    if schema_version != PROVIDER_FAILURE_SUMMARY_SCHEMA_VERSION:
        return summary

    verdict = validate_provider_failure_summary(summary, strict=True)
    if bool(verdict.get("ok") is True):
        return summary

    issues = verdict.get("issues") if isinstance(verdict.get("issues"), list) else []
    _record_gate_summary_issue(
        context=context,
        summary=summary,
        code="invalid_provider_failure_summary_schema",
        issues=[str(item) for item in issues],
        fail_close=True,
    )
    return None


def parse_gate_summary(value: Any, *, context: str = "") -> Optional[Dict[str, Any]]:
    parsed: Optional[Dict[str, Any]] = None
    if isinstance(value, dict):
        parsed = dict(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            maybe = json.loads(text)
        except Exception:
            return None
        if isinstance(maybe, dict):
            parsed = dict(maybe)
    if not isinstance(parsed, dict):
        return None

    bound = _bind_delegated_gate_summary(parsed, context=context)
    return _validate_gate_summary(bound, context=context)


def step_transition_reason(raw_status: str, gate_summary: Optional[Dict[str, Any]]) -> str:
    if isinstance(gate_summary, dict):
        v = str(gate_summary.get("queue_reason") or "").strip()
        if v:
            return v
    return f"autopilot_state:{raw_status}"


def gate_summary_refs(gate_summary: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(gate_summary, dict):
        return []
    refs: List[str] = []
    for field in ("decision_path", "completion_packet_path"):
        raw = str(gate_summary.get(field) or "").strip()
        if not raw:
            continue
        p = pathlib.Path(raw)
        refs.append(to_rel(p) if p.is_absolute() else raw)
    return dedupe_keep_order(refs)


def gate_summary_artifact_bindings(gate_summary: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    if not isinstance(gate_summary, dict):
        return {}

    bindings: Dict[str, Dict[str, str]] = {}
    fields = [
        ("completion_packet_path", "completion_packet_sha256", "completion_packet"),
        ("decision_path", "decision_sha256", "gate_decision"),
    ]

    for path_field, sha_field, binding_kind in fields:
        raw_path = str(gate_summary.get(path_field) or "").strip()
        digest = normalize_sha256(gate_summary.get(sha_field))
        if not raw_path or not digest:
            continue

        aliases = set()
        aliases.add(raw_path)

        path_obj = pathlib.Path(raw_path)
        if path_obj.is_absolute():
            aliases.add(to_rel(path_obj))
            try:
                aliases.add(str(path_obj.resolve()))
            except Exception:
                aliases.add(str(path_obj))
        else:
            resolved = _resolve_gate_summary_path(raw_path)
            if resolved is not None:
                aliases.add(str(resolved))
                aliases.add(to_rel(resolved))

        for alias in aliases:
            key = str(alias or "").strip()
            if not key:
                continue
            bindings[key] = {"sha256": digest, "binding": binding_kind}

    return bindings


DEGRADED_RECONCILE_SCHEMA_VERSION = "autopilot.degraded_local_reconcile.v1"
DEGRADED_RECONCILE_REASON = "autopilot_degraded_local_execution_reconciled"
DEGRADED_STALE_TASK_DRAIN_REASON = "autopilot_degraded_local_execution_stale_task_drained"
DEGRADED_STALE_TASK_RECOVERY_COUNTERS_SCHEMA_VERSION = "autopilot.degraded_stale_task_recovery_counters.v1"
DEGRADED_LOCAL_RUN_PENDING_LIMIT = 24
DEGRADED_LOCAL_RUN_RECONCILED_LIMIT = 8


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


DEGRADED_LOCAL_RUN_PENDING_MAX_AGE_SEC = max(0, env_int("OPENCLAW_AUTOPILOT_DEGRADED_RUN_PENDING_MAX_AGE_SEC", 1209600))
DEGRADED_LOCAL_RUN_RECONCILED_MAX_AGE_SEC = max(0, env_int("OPENCLAW_AUTOPILOT_DEGRADED_RUN_RECONCILED_MAX_AGE_SEC", 604800))
DEGRADED_LOCAL_RUN_PENDING_STALE_AFTER_SEC = max(60, env_int("OPENCLAW_AUTOPILOT_DEGRADED_RUN_PENDING_STALE_AFTER_SEC", 3600))
DEGRADED_PENDING_STALE_SIGNAL_SCHEMA_VERSION = "autopilot.degraded_pending_stale_signal.v1"
DEGRADED_PENDING_STALE_SIGNAL_AFTER_TICKS = max(
    1,
    env_int("OPENCLAW_AUTOPILOT_DEGRADED_PENDING_STALE_SIGNAL_AFTER_TICKS", 3),
)
DEGRADED_PENDING_STALE_SIGNAL_COOLDOWN_SEC = max(
    300,
    env_int("OPENCLAW_AUTOPILOT_DEGRADED_PENDING_STALE_SIGNAL_COOLDOWN_SEC", 21600),
)
EVENT_ROUTER_PATH = pathlib.Path(
    os.environ.get("OPENCLAW_EVENT_ROUTER_SCRIPT")
    or str(root / "ops" / "openclaw" / "event_router.sh")
).resolve()


def degraded_reconcile_event_id(task_id: str, run_id: str, *, transition_kind: str = "degraded_local_reconcile") -> str:
    kind = str(transition_kind or "degraded_local_reconcile").strip() or "degraded_local_reconcile"
    seed = f"{task_id}|{kind}|{run_id}"
    return "tevt_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def to_int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def resolve_degraded_reconcile_target_status(
    *,
    step_id: str,
    current_status: str,
    run: Dict[str, Any],
) -> Tuple[str, str]:
    """Resolve deterministic stale-task drain target for completed degraded local runs.

    Returns: (target_status, resolution_reason)
    """
    completion_status_raw = str(run.get("completion_queue_status") or "").strip()
    completion_status = map_status(step_id, completion_status_raw) if completion_status_raw else ""
    completion_exit_code = to_int_or_none(run.get("completion_exit_code"))

    if completion_status in {"REVIEW", "DONE", "FAILED", "BLOCKED", "ROLLED_BACK"}:
        return completion_status, "completion_queue_status_terminal"

    if completion_exit_code is not None:
        if completion_exit_code == 0:
            return ("DONE" if step_id == "quality_gate" else "REVIEW"), "completion_exit_code_zero"
        return "FAILED", "completion_exit_code_nonzero"

    if current_status in {"RUNNING", "QUEUED"}:
        return "BLOCKED", "completion_terminal_status_missing"

    return current_status, "no_terminal_signal"


def degraded_run_completed_iso(run: Dict[str, Any]) -> Optional[str]:
    raw_iso = str(run.get("completed_iso") or "").strip()
    if raw_iso:
        return raw_iso
    return ts_to_iso(run.get("completed_ts"))


def degraded_run_age_seconds(run: Dict[str, Any], *, reconciled: bool, ref_ts: Optional[int] = None) -> Optional[int]:
    # Pending rows can carry both started_ts and completed_ts when subprocess completion
    # has already been observed but reconciliation is still in-flight. Prefer
    # completed_ts for pending-age projection so long-running jobs that just
    # completed do not immediately look like stale backlog noise.
    anchor_fields = ["reconciled_ts", "completed_ts", "started_ts"] if reconciled else ["completed_ts", "started_ts"]
    now_ref = int(ref_ts) if ref_ts is not None else current_ts()
    for field in anchor_fields:
        ts_val = to_int_or_none(run.get(field))
        if ts_val is None or ts_val <= 0:
            continue
        age_sec = now_ref - ts_val
        return age_sec if age_sec >= 0 else 0
    return None


def persist_continuity_event(cur: sqlite3.Cursor, event_obj: Dict[str, Any]) -> None:
    event_id = str(event_obj.get("event_id") or "").strip()
    created_at = str(event_obj.get("created_at") or event_obj.get("timestamp") or "").strip()
    source_name = str(event_obj.get("source") or "").strip()
    event_key = str(event_obj.get("key") or event_obj.get("event_key") or "").strip()
    severity = str(event_obj.get("severity") or "info").strip().lower()
    route_key = str(event_obj.get("route_key") or f"{source_name}|{event_key}").strip()
    fingerprint = str(event_obj.get("fingerprint") or "").strip()
    if not (event_id and created_at and source_name and event_key and route_key and fingerprint):
        return
    if severity not in {"info", "warn", "critical"}:
        severity = "info"

    cur.execute(
        """
INSERT OR REPLACE INTO continuity_events (
  event_id, created_at, source, event_key, severity, fingerprint,
  emitted, changed, cooldown_elapsed, suppress_reason, summary,
  evidence_ref, route_key, state_file, metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            event_id,
            created_at,
            source_name,
            event_key,
            severity,
            fingerprint,
            int(bool(event_obj.get("emit"))),
            int(bool(event_obj.get("changed"))),
            int(bool(event_obj.get("cooldown_elapsed"))),
            event_obj.get("suppress_reason"),
            event_obj.get("summary"),
            event_obj.get("evidence_ref"),
            route_key,
            event_obj.get("state_file"),
            json.dumps(event_obj, ensure_ascii=False, sort_keys=True),
        ),
    )


def emit_runtime_event(
    cur: sqlite3.Cursor,
    *,
    event_key: str,
    severity: str,
    summary: str,
    evidence_ref: str,
    fingerprint_input: str,
    cooldown_sec: int,
) -> Dict[str, Any]:
    if not EVENT_ROUTER_PATH.exists() or not os.access(EVENT_ROUTER_PATH, os.X_OK):
        return {
            "ok": False,
            "error": "event_router_missing",
            "event_router": str(EVENT_ROUTER_PATH),
            "source": "runtime.queue_infra_degraded",
            "event_key": event_key,
            "severity": severity,
        }

    cmd = [
        str(EVENT_ROUTER_PATH),
        "--source",
        "runtime.queue_infra_degraded",
        "--key",
        str(event_key),
        "--severity",
        str(severity),
        "--summary",
        str(summary),
        "--evidence-ref",
        str(evidence_ref),
        "--fingerprint-input",
        str(fingerprint_input),
        "--cooldown-sec",
        str(max(0, int(cooldown_sec))),
        "--no-persist",
    ]
    cp = subprocess.run(cmd, text=True, capture_output=True)

    payload: Dict[str, Any] = {}
    stdout = str(cp.stdout or "").strip()
    if stdout:
        try:
            maybe_payload = json.loads(stdout)
            if isinstance(maybe_payload, dict):
                payload = maybe_payload
        except Exception:
            payload = {}

    if cp.returncode == 0 and payload:
        try:
            persist_continuity_event(cur, payload)
        except Exception:
            pass

    return {
        "ok": cp.returncode in (0, 20),
        "returncode": int(cp.returncode),
        "event_key": str(event_key),
        "severity": str(severity),
        "emitted": cp.returncode == 0,
        "suppressed": cp.returncode == 20,
        "stderr": str(cp.stderr or "")[:240],
        "router": payload,
    }


def next_gate_for_status(to_status: str) -> str:
    if to_status == "RUNNING":
        return "execution"
    if to_status == "REVIEW":
        return "validator_gate"
    if to_status == "DONE":
        return "librarian_archive"
    if to_status in {"BLOCKED", "FAILED", "ROLLED_BACK"}:
        return "sre_recovery"
    return "claim"


def parent_task_id(cur: sqlite3.Cursor, task_id: str) -> Optional[str]:
    row = cur.execute(
        """
SELECT depends_on_task_id
FROM task_dependencies
WHERE task_id = ? AND relation = 'blocks'
ORDER BY created_at DESC, depends_on_task_id ASC
LIMIT 1
""",
        (task_id,),
    ).fetchone()
    if not row:
        return None
    val = str(row[0] or "").strip()
    return val or None


def handoff_packet_id(task_id: str, transition_id: str, created_at: str, from_role: str, to_role: str) -> str:
    seed = f"{task_id}|{transition_id}|{created_at}|{from_role}|{to_role}"
    return "thp_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def persist_handoff_packet(
    cur: sqlite3.Cursor,
    *,
    task_id: str,
    transition_event_id: str,
    from_status: Optional[str],
    to_status: str,
    from_role: str,
    to_role: str,
    created_at: str,
    reason: str,
    evidence_ref: str,
    retry_count: int,
    gate_summary: Optional[Dict[str, Any]] = None,
) -> bool:
    src = str(from_role or "").strip().lower()
    dst = str(to_role or "").strip().lower()
    if src not in ALLOWED_ROLES or dst not in ALLOWED_ROLES or src == dst:
        return False

    parent = parent_task_id(cur, task_id)
    packet_id = handoff_packet_id(task_id, transition_event_id, created_at, src, dst)
    linkage = {
        "task_id": task_id,
        "parent_task_id": parent,
        "transition_event_id": transition_event_id,
    }
    gate_meta = {
        "from_status": from_status,
        "to_status": to_status,
        "reason": reason,
        "policy_override": False,
        "gate_summary": gate_summary if isinstance(gate_summary, dict) and gate_summary else None,
    }

    cur.execute(
        """
INSERT OR REPLACE INTO task_handoff_packets (
  packet_id, task_id, parent_task_id, transition_event_id,
  from_role, to_role, from_status, to_status, created_at,
  evidence_refs_json, gate_metadata_json, task_linkage_json, lock_refs_json,
  next_gate, budget_tokens_used, model_tier, retry_count, failure_signature
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            packet_id,
            task_id,
            parent,
            transition_event_id,
            src,
            dst,
            from_status,
            to_status,
            created_at,
            json.dumps(parse_evidence_refs(evidence_ref), ensure_ascii=False),
            json.dumps(gate_meta, ensure_ascii=False, sort_keys=True),
            json.dumps(linkage, ensure_ascii=False, sort_keys=True),
            json.dumps([], ensure_ascii=False),
            next_gate_for_status(to_status),
            0,
            "unknown",
            int(retry_count or 0),
            (
                str((gate_summary or {}).get("summary_signature") or reason)[:240]
                if to_status in {"BLOCKED", "FAILED", "ROLLED_BACK"}
                else None
            ),
        ),
    )
    return True


def run_tag_from_ts(value: Any) -> Optional[str]:
    try:
        n = int(value)
        if n <= 0:
            return None
        return dt.datetime.fromtimestamp(n, tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    except Exception:
        return None


def latest_run_refs(step_id: str, last_started_ts: Any) -> List[str]:
    runs_dir = root / "ops" / "autopilot" / "runs"
    refs: List[str] = []
    if not runs_dir.exists():
        return refs

    tag = run_tag_from_ts(last_started_ts)
    if tag:
        base = runs_dir / f"{tag}_{step_id}"
        for ext in (".log", ".exit", ".completion_packet.json"):
            p = pathlib.Path(str(base) + ext)
            if p.exists():
                refs.append(to_rel(p))
        if refs:
            return refs

    logs = sorted(runs_dir.glob(f"*_{step_id}.log"))
    if logs:
        log = logs[-1]
        refs.append(to_rel(log))
        exit_path = log.with_suffix(".exit")
        if exit_path.exists():
            refs.append(to_rel(exit_path))
        packet_path = log.with_suffix(".completion_packet.json")
        if packet_path.exists():
            refs.append(to_rel(packet_path))
    return refs


def step_artifact_refs(step_id: str, repo_path: Optional[pathlib.Path]) -> List[str]:
    if repo_path is None:
        return []
    artifact_map = {
        "sync_spec_context": [
            "autopilot_artifacts/spec/spec_backlog_summary.md",
        ],
        "audit_alignment": [
            "autopilot_artifacts/audit_alignment.md",
            "autopilot_artifacts/audit_alignment.json",
        ],
        "audit_runtime_probes": [
            "autopilot_artifacts/audit_runtime_probes.md",
            "autopilot_artifacts/audit_runtime_probes.json",
        ],
        "audit_breaktests": [
            "autopilot_artifacts/audit_breaktests.md",
            "autopilot_artifacts/audit_breaktests.json",
        ],
        "synth_fix_plan": [
            "autopilot_artifacts/fix_plan.md",
            "autopilot_artifacts/fix_plan.json",
        ],
        "apply_fixes": [
            "autopilot_artifacts/apply_fixes.md",
            "autopilot_artifacts/p0_progress.md",
            "autopilot_artifacts/p0_progress.json",
        ],
        "quality_gate": [
            "autopilot_artifacts/quality_gate.md",
            "autopilot_artifacts/quality_gate.json",
        ],
    }
    refs: List[str] = []
    for rel in artifact_map.get(step_id, []):
        p = (repo_path / rel).resolve()
        if p.exists():
            refs.append(str(p))
    return refs


def step_evidence_ref(step: Dict[str, Any], repo_path: Optional[pathlib.Path], gate_summary: Optional[Dict[str, Any]] = None) -> str:
    step_id = str(step.get("id") or "").strip()
    refs = [to_rel(state_path)]
    refs.extend(latest_run_refs(step_id, step.get("last_started_ts")))
    refs.extend(step_artifact_refs(step_id, repo_path))
    refs.extend(gate_summary_refs(gate_summary))
    refs = dedupe_keep_order(refs)
    return " | ".join(refs)


def step_lock_targets(step: Dict[str, Any], repo_path: Optional[pathlib.Path]) -> List[str]:
    step_id = str(step.get("id") or "").strip()
    refs: List[str] = []

    refs.append(to_rel(state_path))
    refs.extend(latest_run_refs(step_id, step.get("last_started_ts")))

    if repo_path is not None:
        for rel in step_artifact_refs(step_id, repo_path):
            refs.append(str(pathlib.Path(rel).resolve()))

    # Keep only write-heavy operational paths for lock arbitration.
    filtered: List[str] = []
    for ref in refs:
        raw = str(ref or "").strip()
        if not raw:
            continue
        low = raw.lower()
        if "autopilot_artifacts" in low or "ops/autopilot/state" in low or "/ops/autopilot/runs/" in low:
            filtered.append(raw)
    return dedupe_keep_order(filtered)


steps = state.get("steps") or []
active = state.get("active") if isinstance(state, dict) else None
paused = bool(state.get("paused")) if isinstance(state, dict) else False
cycle = int(state.get("cycle") or 0)
max_cycles = int(state.get("max_cycles") or 0)
repo_path_raw = str(((state.get("repo") or {}).get("path") or "")).strip() if isinstance(state, dict) else ""
repo_path = pathlib.Path(repo_path_raw).resolve() if repo_path_raw else None

db_path.parent.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(db_path)
cur = con.cursor()

rows_upserted = 0
transitions_inserted = 0
transitions_backfilled = 0
dependencies_upserted = 0
task_file_targets_upserted = 0
task_artifacts_upserted = 0
handoff_packets_upserted = 0
degraded_local_runs_seen = 0
degraded_local_runs_backfilled = 0
degraded_local_runs_marked_reconciled = 0
degraded_local_runs_pending = 0
degraded_local_runs_reconciled = 0
degraded_local_runs_pruned = 0
degraded_local_runs_pruned_by_age = 0
degraded_local_runs_pruned_by_count = 0
degraded_local_runs_pending_stale = 0
degraded_local_runs_pending_oldest_age_sec = 0
degraded_local_runs_reconciled_oldest_age_sec = 0
degraded_local_handoffs_upserted = 0
degraded_local_runs_stale_drain_attempted = 0
degraded_local_runs_stale_drain_applied = 0
degraded_local_runs_stale_drain_failed = 0
degraded_local_runs_stale_processing_recovered = 0
degraded_local_runs_stale_running_recovered = 0
degraded_pending_stale_signal_active = False
degraded_pending_stale_signal_streak = 0
degraded_pending_stale_signal_emitted = False
degraded_pending_stale_signal_event: Optional[Dict[str, Any]] = None
state_runs_writeback = False
evidence_refs_used: List[str] = []

step_evidence_cache: Dict[str, str] = {}
step_raw_status_by_id: Dict[str, str] = {}
ordered_task_ids: List[str] = []

for step in steps:
    if not isinstance(step, dict):
        continue

    step_id = str(step.get("id") or "").strip()
    if not step_id:
        continue

    task_id = f"autopilot:{step_id}"
    raw_status = str(step.get("status") or "queued")
    step_raw_status_by_id[step_id] = str(raw_status or "").strip().lower()
    queue_status = map_status(step_id, raw_status)
    step_role_required = role_required_for_status(role_for_step(step_id), queue_status)
    title = str(step.get("title") or step_id)
    acceptance = first_line(step.get("prompt") or step.get("cmd") or "")
    attempts = int(step.get("attempts") or 0)
    max_attempts = int(step.get("max_attempts") or 3)
    last_error = first_line(step.get("last_error") or "", limit=400)
    cooldown_until = future_cooldown_until(step.get("next_after_ts"), queue_status=queue_status)

    created_at = ts_to_iso(step.get("last_started_ts")) or now_iso()
    updated_at = ts_to_iso(step.get("last_finished_ts")) or ts_to_iso(step.get("last_started_ts")) or now_iso()

    prev_row = cur.execute("SELECT status FROM work_queue WHERE task_id = ?", (task_id,)).fetchone()
    prev_status = prev_row[0] if prev_row else None

    cur.execute(
        """
INSERT INTO work_queue (
  task_id, source, title, acceptance_criteria, status, role_required, assigned_agent,
  retry_count, max_retries, last_error_log, cooldown_until, created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(task_id) DO UPDATE SET
  source=excluded.source,
  title=excluded.title,
  acceptance_criteria=excluded.acceptance_criteria,
  status=excluded.status,
  role_required=excluded.role_required,
  assigned_agent=excluded.assigned_agent,
  retry_count=excluded.retry_count,
  max_retries=excluded.max_retries,
  last_error_log=excluded.last_error_log,
  cooldown_until=excluded.cooldown_until,
  updated_at=excluded.updated_at
""",
        (
            task_id,
            source,
            title,
            acceptance,
            queue_status,
            step_role_required,
            "codex-orchestrator-pro",
            attempts,
            max_attempts,
            last_error or None,
            cooldown_until,
            created_at,
            updated_at,
        ),
    )
    rows_upserted += 1

    gate_summary = parse_gate_summary(step.get("delegated_gate_summary"), context=f"{step_id}:transition")
    evid = step_evidence_ref(step, repo_path, gate_summary)
    step_evidence_cache[task_id] = evid
    ordered_task_ids.append(task_id)

    # Refresh task_artifacts for this task from current evidence refs.
    cur.execute("DELETE FROM task_artifacts WHERE task_id = ?", (task_id,))
    evid_refs = dedupe_keep_order([p.strip() for p in str(evid or "").split("|") if str(p).strip()])

    gate_artifact_bindings = gate_summary_artifact_bindings(gate_summary)

    for ref in evid_refs:
        a_type = classify_artifact_type(ref)
        a_id = artifact_id(task_id, ref, a_type)
        artifact_binding = gate_artifact_bindings.get(ref) if isinstance(gate_artifact_bindings, dict) else None
        artifact_sha = normalize_sha256((artifact_binding or {}).get("sha256") if isinstance(artifact_binding, dict) else None)
        artifact_meta = None
        if artifact_sha:
            binding_kind = str((artifact_binding or {}).get("binding") or "delegated_gate_summary").strip() or "delegated_gate_summary"
            artifact_meta = {
                "source": "delegated_gate_summary",
                "binding": binding_kind,
            }
            if binding_kind == "completion_packet":
                artifact_meta["completion_packet"] = True
        cur.execute(
            """
INSERT OR REPLACE INTO task_artifacts (
  artifact_id, task_id, artifact_type, artifact_path, sha256, metadata_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?)
""",
            (
                a_id,
                task_id,
                a_type,
                ref,
                artifact_sha,
                json.dumps(artifact_meta, ensure_ascii=False, sort_keys=True) if artifact_meta else None,
                updated_at,
            ),
        )
        task_artifacts_upserted += 1

    # Refresh lock-target declarations for this task.
    cur.execute("DELETE FROM task_file_targets WHERE task_id = ?", (task_id,))
    for target in step_lock_targets(step, repo_path):
        cur.execute(
            """
INSERT OR REPLACE INTO task_file_targets (
  task_id, file_path, lock_mode, created_at
) VALUES (?, ?, 'exclusive', ?)
""",
            (task_id, target, updated_at),
        )
        task_file_targets_upserted += 1

    if prev_status != queue_status:
        role = role_for_step(step_id)
        reason = step_transition_reason(raw_status, gate_summary)
        ev_id = event_id(task_id, prev_status, queue_status, updated_at, reason)
        cur.execute(
            """
INSERT OR IGNORE INTO task_transitions (
  event_id, task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""",
            (ev_id, task_id, prev_status, queue_status, role, reason, evid, updated_at),
        )
        if cur.rowcount:
            transitions_inserted += 1
            evidence_refs_used.extend([evid])
            from_role = role_required_for_status(role_for_step(step_id), str(prev_status or "QUEUED")) if prev_status else ""
            if persist_handoff_packet(
                cur,
                task_id=task_id,
                transition_event_id=ev_id,
                from_status=prev_status,
                to_status=queue_status,
                from_role=from_role,
                to_role=step_role_required,
                created_at=updated_at,
                reason=reason,
                evidence_ref=evid,
                retry_count=attempts,
                gate_summary=gate_summary,
            ):
                handoff_packets_upserted += 1

has_degraded_state = isinstance(state.get("queue_infra_degraded"), dict)
degraded_state = state.get("queue_infra_degraded") if has_degraded_state else {}
degraded_state_before = json.loads(json.dumps(degraded_state, ensure_ascii=False)) if has_degraded_state else {}
degraded_mode_active = bool(degraded_state.get("degraded_mode") is True)
raw_degraded_runs = degraded_state.get("degraded_local_runs") if isinstance(degraded_state.get("degraded_local_runs"), list) else []
degraded_local_runs_seen = len(raw_degraded_runs)
last_reconciled_meta: Optional[Dict[str, Any]] = None

if not degraded_mode_active and raw_degraded_runs:
    for run in raw_degraded_runs:
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("run_id") or "").strip()
        step_id = str(run.get("step_id") or "").strip()
        if not run_id or not step_id:
            continue

        completed_at = degraded_run_completed_iso(run)
        if not completed_at:
            continue

        task_id = str(run.get("task_id") or "").strip() or f"autopilot:{step_id}"
        task_row = cur.execute(
            "SELECT status, role_required, retry_count FROM work_queue WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if not task_row:
            continue

        current_status = str(task_row[0] or "QUEUED")
        current_role = str(task_row[1] or "").strip().lower()
        if current_role not in ALLOWED_ROLES:
            current_role = role_required_for_status(role_for_step(step_id), current_status)
        retry_count = int(task_row[2] or 0)

        target_status, reconcile_resolution = resolve_degraded_reconcile_target_status(
            step_id=step_id,
            current_status=current_status,
            run=run,
        )
        status_drain_applied = False
        status_transition_kind = "degraded_local_reconcile"
        transition_reason = DEGRADED_RECONCILE_REASON
        if target_status != current_status:
            degraded_local_runs_stale_drain_attempted += 1
            status_transition_kind = "degraded_local_stale_task_drain"
            transition_reason = DEGRADED_STALE_TASK_DRAIN_REASON
            target_role_required = role_required_for_status(role_for_step(step_id), target_status)
            cur.execute(
                """
UPDATE work_queue
SET status = ?, role_required = ?, updated_at = ?
WHERE task_id = ?
""",
                (target_status, target_role_required, completed_at, task_id),
            )
            if cur.rowcount:
                status_drain_applied = True
                degraded_local_runs_stale_drain_applied += 1
                completion_status_raw = str(run.get("completion_queue_status") or "").strip().lower()
                step_status_raw = str(step_raw_status_by_id.get(step_id) or "").strip().lower()
                if completion_status_raw == "processing" or step_status_raw == "processing":
                    degraded_local_runs_stale_processing_recovered += 1
                if (
                    current_status == "RUNNING"
                    or completion_status_raw in {"running", "processing"}
                    or step_status_raw in {"running", "processing"}
                ):
                    degraded_local_runs_stale_running_recovered += 1
                current_role = target_role_required
            else:
                degraded_local_runs_stale_drain_failed += 1
                continue

        refs: List[str] = []
        for field in ("completion_evidence_refs", "launch_evidence_refs"):
            raw_refs = run.get(field)
            if isinstance(raw_refs, list):
                refs.extend([str(item).strip() for item in raw_refs if str(item or "").strip()])
            elif isinstance(raw_refs, str):
                refs.extend(parse_evidence_refs(raw_refs))
        if task_id in step_evidence_cache and step_evidence_cache.get(task_id):
            refs.extend(parse_evidence_refs(step_evidence_cache.get(task_id) or ""))
        refs.append(to_rel(state_path))
        evid = " | ".join(dedupe_keep_order(refs))

        reconcile_summary: Dict[str, Any] = {
            "schema_version": DEGRADED_RECONCILE_SCHEMA_VERSION,
            "queue_reason": transition_reason,
            "degraded_run_id": run_id,
            "degraded_reason": str(run.get("reason") or ""),
            "mode": str(run.get("mode") or "degraded_local_pickup"),
            "completed_at": completed_at,
            "queue_from_status": current_status,
            "queue_to_status": target_status,
            "reconcile_resolution": reconcile_resolution,
            "status_drain_applied": bool(status_drain_applied),
            "completion_queue_status": str(run.get("completion_queue_status") or ""),
            "completion_queue_reason": str(run.get("completion_queue_reason") or ""),
            "completion_exit_code": to_int_or_none(run.get("completion_exit_code")),
        }
        completion_gate_summary = run.get("gate_summary") if isinstance(run.get("gate_summary"), dict) else None
        if completion_gate_summary:
            reconcile_summary["completion_gate_summary"] = completion_gate_summary

        ev_id = degraded_reconcile_event_id(task_id, run_id, transition_kind=status_transition_kind)
        cur.execute(
            """
INSERT OR IGNORE INTO task_transitions (
  event_id, task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""",
            (
                ev_id,
                task_id,
                current_status,
                target_status,
                "sre_watchdog",
                transition_reason,
                evid,
                completed_at,
            ),
        )
        transition_exists = bool(cur.rowcount)
        if cur.rowcount:
            transitions_inserted += 1
            degraded_local_runs_backfilled += 1
            evidence_refs_used.extend([evid])
            if persist_handoff_packet(
                cur,
                task_id=task_id,
                transition_event_id=ev_id,
                from_status=current_status,
                to_status=target_status,
                from_role="sre_watchdog",
                to_role=current_role,
                created_at=completed_at,
                reason=transition_reason,
                evidence_ref=evid,
                retry_count=retry_count,
                gate_summary=reconcile_summary,
            ):
                handoff_packets_upserted += 1
                degraded_local_handoffs_upserted += 1
        else:
            row = cur.execute("SELECT 1 FROM task_transitions WHERE event_id = ? LIMIT 1", (ev_id,)).fetchone()
            transition_exists = row is not None

        if transition_exists:
            reconciled_ts = current_ts()
            reconciled_iso = now_iso()
            if run.get("reconciled") is not True:
                degraded_local_runs_marked_reconciled += 1
                run["reconciled"] = True
                run["reconciled_ts"] = reconciled_ts
                run["reconciled_iso"] = reconciled_iso
                run["reconcile_reason"] = transition_reason
                run["reconcile_transition_event_id"] = ev_id
                run["reconciled_queue_from_status"] = current_status
                run["reconciled_queue_to_status"] = target_status
                run["reconcile_resolution"] = reconcile_resolution
                run["stale_task_drained"] = bool(status_drain_applied)
            else:
                run.setdefault("reconcile_reason", transition_reason)
                run.setdefault("reconcile_transition_event_id", ev_id)
                run.setdefault("reconciled_ts", to_int_or_none(run.get("completed_ts")) or reconciled_ts)
                run.setdefault("reconciled_iso", str(run.get("completed_iso") or reconciled_iso))
                run.setdefault("reconciled_queue_from_status", current_status)
                run.setdefault("reconciled_queue_to_status", target_status)
                run.setdefault("reconcile_resolution", reconcile_resolution)
                run.setdefault("stale_task_drained", bool(status_drain_applied))
            last_reconciled_meta = {
                "run_id": run_id,
                "task_id": task_id,
                "event_id": ev_id,
                "reconciled_ts": int(run.get("reconciled_ts") or reconciled_ts),
                "reconciled_iso": str(run.get("reconciled_iso") or reconciled_iso),
                "from_status": current_status,
                "to_status": target_status,
                "reason": transition_reason,
            }

if has_degraded_state:
    cleaned_pending: List[Dict[str, Any]] = []
    cleaned_reconciled: List[Dict[str, Any]] = []
    for item in raw_degraded_runs:
        if not isinstance(item, dict):
            continue
        run_id = str(item.get("run_id") or "").strip()
        if not run_id:
            continue

        rec = dict(item)
        rec["run_id"] = run_id
        rec.setdefault("schema_version", str(degraded_state.get("degraded_local_runs_schema_version") or "autopilot.queue_degraded_run.v1"))

        is_reconciled = bool(rec.get("reconciled") is True)
        rec["reconciled"] = is_reconciled
        if is_reconciled:
            rec.setdefault("reconcile_reason", DEGRADED_RECONCILE_REASON)
            rec.setdefault("reconcile_transition_event_id", "")
            rec["reconciled_ts"] = to_int_or_none(rec.get("reconciled_ts")) or to_int_or_none(rec.get("completed_ts")) or current_ts()
            rec["reconciled_iso"] = str(rec.get("reconciled_iso") or ts_to_iso(rec.get("reconciled_ts")) or "")
            cleaned_reconciled.append(rec)
        else:
            cleaned_pending.append(rec)

    ref_now_ts = current_ts()
    pending_pruned_by_age = 0
    reconciled_pruned_by_age = 0

    age_kept_pending: List[Dict[str, Any]] = []
    for rec in cleaned_pending:
        age_sec = degraded_run_age_seconds(rec, reconciled=False, ref_ts=ref_now_ts)
        if DEGRADED_LOCAL_RUN_PENDING_MAX_AGE_SEC > 0 and age_sec is not None and age_sec > DEGRADED_LOCAL_RUN_PENDING_MAX_AGE_SEC:
            pending_pruned_by_age += 1
            continue
        age_kept_pending.append(rec)

    age_kept_reconciled: List[Dict[str, Any]] = []
    for rec in cleaned_reconciled:
        age_sec = degraded_run_age_seconds(rec, reconciled=True, ref_ts=ref_now_ts)
        if DEGRADED_LOCAL_RUN_RECONCILED_MAX_AGE_SEC > 0 and age_sec is not None and age_sec > DEGRADED_LOCAL_RUN_RECONCILED_MAX_AGE_SEC:
            reconciled_pruned_by_age += 1
            continue
        age_kept_reconciled.append(rec)

    kept_pending = age_kept_pending[-DEGRADED_LOCAL_RUN_PENDING_LIMIT:]
    kept_reconciled = age_kept_reconciled[-DEGRADED_LOCAL_RUN_RECONCILED_LIMIT:]

    degraded_local_runs_pruned_by_age = pending_pruned_by_age + reconciled_pruned_by_age
    degraded_local_runs_pruned_by_count = (len(age_kept_pending) - len(kept_pending)) + (len(age_kept_reconciled) - len(kept_reconciled))
    degraded_local_runs_pruned = degraded_local_runs_pruned_by_age + degraded_local_runs_pruned_by_count

    pending_oldest_age = 0
    pending_oldest_run_id = ""
    pending_oldest_task_id = ""
    pending_stale_count = 0
    for rec in kept_pending:
        age_sec = degraded_run_age_seconds(rec, reconciled=False, ref_ts=ref_now_ts)
        if age_sec is None:
            continue
        if age_sec >= DEGRADED_LOCAL_RUN_PENDING_STALE_AFTER_SEC:
            pending_stale_count += 1
        if age_sec > pending_oldest_age:
            pending_oldest_age = int(age_sec)
            pending_oldest_run_id = str(rec.get("run_id") or "")
            pending_oldest_task_id = str(rec.get("task_id") or "").strip()
            if not pending_oldest_task_id:
                step_hint = str(rec.get("step_id") or "").strip()
                if step_hint:
                    pending_oldest_task_id = f"autopilot:{step_hint}"

    reconciled_oldest_age = 0
    reconciled_oldest_run_id = ""
    for rec in kept_reconciled:
        age_sec = degraded_run_age_seconds(rec, reconciled=True, ref_ts=ref_now_ts)
        if age_sec is None:
            continue
        if age_sec > reconciled_oldest_age:
            reconciled_oldest_age = int(age_sec)
            reconciled_oldest_run_id = str(rec.get("run_id") or "")

    degraded_local_runs_pending = len(kept_pending)
    degraded_local_runs_reconciled = len(kept_reconciled)
    degraded_local_runs_pending_stale = int(pending_stale_count)
    degraded_local_runs_pending_oldest_age_sec = int(pending_oldest_age)
    degraded_local_runs_reconciled_oldest_age_sec = int(reconciled_oldest_age)

    degraded_state["degraded_local_runs"] = kept_pending + kept_reconciled
    degraded_state["degraded_local_runs_pending"] = degraded_local_runs_pending
    degraded_state["degraded_local_runs_reconciled"] = degraded_local_runs_reconciled
    degraded_state["degraded_local_runs_pending_limit"] = DEGRADED_LOCAL_RUN_PENDING_LIMIT
    degraded_state["degraded_local_runs_reconciled_limit"] = DEGRADED_LOCAL_RUN_RECONCILED_LIMIT
    degraded_state["degraded_local_runs_pending_max_age_sec"] = DEGRADED_LOCAL_RUN_PENDING_MAX_AGE_SEC
    degraded_state["degraded_local_runs_reconciled_max_age_sec"] = DEGRADED_LOCAL_RUN_RECONCILED_MAX_AGE_SEC
    degraded_state["degraded_local_runs_pending_stale_after_sec"] = DEGRADED_LOCAL_RUN_PENDING_STALE_AFTER_SEC
    degraded_state["degraded_local_runs_pending_stale_count"] = degraded_local_runs_pending_stale
    degraded_state["degraded_local_runs_pending_oldest_age_sec"] = degraded_local_runs_pending_oldest_age_sec
    degraded_state["degraded_local_runs_pending_oldest_run_id"] = pending_oldest_run_id
    degraded_state["degraded_local_runs_reconciled_oldest_age_sec"] = degraded_local_runs_reconciled_oldest_age_sec
    degraded_state["degraded_local_runs_reconciled_oldest_run_id"] = reconciled_oldest_run_id

    if degraded_local_runs_marked_reconciled > 0:
        degraded_state["degraded_local_runs_reconciled_total"] = int(degraded_state.get("degraded_local_runs_reconciled_total") or 0) + degraded_local_runs_marked_reconciled
    else:
        degraded_state.setdefault("degraded_local_runs_reconciled_total", int(degraded_state.get("degraded_local_runs_reconciled_total") or 0))

    degraded_state.setdefault("degraded_local_runs_pruned_by_age_total", int(degraded_state.get("degraded_local_runs_pruned_by_age_total") or 0))
    degraded_state.setdefault("degraded_local_runs_pruned_by_count_total", int(degraded_state.get("degraded_local_runs_pruned_by_count_total") or 0))

    if degraded_local_runs_pruned > 0:
        degraded_state["degraded_local_runs_pruned_total"] = int(degraded_state.get("degraded_local_runs_pruned_total") or 0) + degraded_local_runs_pruned
        degraded_state["degraded_local_runs_pruned_by_age_total"] = int(degraded_state.get("degraded_local_runs_pruned_by_age_total") or 0) + degraded_local_runs_pruned_by_age
        degraded_state["degraded_local_runs_pruned_by_count_total"] = int(degraded_state.get("degraded_local_runs_pruned_by_count_total") or 0) + degraded_local_runs_pruned_by_count
        degraded_state["degraded_local_runs_last_pruned_by_age"] = degraded_local_runs_pruned_by_age
        degraded_state["degraded_local_runs_last_pruned_by_count"] = degraded_local_runs_pruned_by_count
        degraded_state["degraded_local_runs_last_pruned_pending_by_age"] = pending_pruned_by_age
        degraded_state["degraded_local_runs_last_pruned_reconciled_by_age"] = reconciled_pruned_by_age
        degraded_state["degraded_local_runs_last_pruned_iso"] = now_iso()
    else:
        degraded_state.setdefault("degraded_local_runs_pruned_total", int(degraded_state.get("degraded_local_runs_pruned_total") or 0))

    stale_recovery_raw = degraded_state.get("stale_task_recovery_counters")
    stale_recovery = dict(stale_recovery_raw) if isinstance(stale_recovery_raw, dict) else {}
    stale_recovery["schema_version"] = DEGRADED_STALE_TASK_RECOVERY_COUNTERS_SCHEMA_VERSION
    stale_recovery["attempts_total"] = int(stale_recovery.get("attempts_total") or 0) + degraded_local_runs_stale_drain_attempted
    stale_recovery["recovered_total"] = int(stale_recovery.get("recovered_total") or 0) + degraded_local_runs_stale_drain_applied
    stale_recovery["failed_total"] = int(stale_recovery.get("failed_total") or 0) + degraded_local_runs_stale_drain_failed
    stale_recovery["stale_processing_recovered_total"] = int(stale_recovery.get("stale_processing_recovered_total") or 0) + degraded_local_runs_stale_processing_recovered
    stale_recovery["stale_running_recovered_total"] = int(stale_recovery.get("stale_running_recovered_total") or 0) + degraded_local_runs_stale_running_recovered
    stale_recovery["attempts_last_tick"] = int(degraded_local_runs_stale_drain_attempted)
    stale_recovery["recovered_last_tick"] = int(degraded_local_runs_stale_drain_applied)
    stale_recovery["failed_last_tick"] = int(degraded_local_runs_stale_drain_failed)
    stale_recovery["stale_processing_recovered_last_tick"] = int(degraded_local_runs_stale_processing_recovered)
    stale_recovery["stale_running_recovered_last_tick"] = int(degraded_local_runs_stale_running_recovered)
    if degraded_local_runs_stale_drain_attempted > 0 or not str(stale_recovery.get("updated_at") or "").strip():
        stale_recovery["updated_at"] = now_iso()
    if degraded_local_runs_stale_drain_applied > 0:
        stale_recovery["last_recovery_at"] = now_iso()
        stale_recovery["last_recovery"] = {
            "at": now_iso(),
            "reason": DEGRADED_STALE_TASK_DRAIN_REASON,
            "attempted": int(degraded_local_runs_stale_drain_attempted),
            "recovered": int(degraded_local_runs_stale_drain_applied),
            "stale_processing_recovered": int(degraded_local_runs_stale_processing_recovered),
            "stale_running_recovered": int(degraded_local_runs_stale_running_recovered),
        }
    elif degraded_local_runs_stale_drain_failed > 0:
        stale_recovery["last_failure_at"] = now_iso()
        stale_recovery["last_failure_reason"] = "stale_task_drain_update_failed"
    degraded_state["stale_task_recovery_counters"] = stale_recovery

    signal_raw = degraded_state.get("degraded_pending_stale_signal")
    signal = dict(signal_raw) if isinstance(signal_raw, dict) else {}
    prev_signal_active = bool(signal.get("active") is True)
    prev_streak = max(0, int(signal.get("stale_ticks_consecutive") or 0))
    prev_active_since_ts = to_int_or_none(signal.get("active_since_ts"))

    stale_present = degraded_local_runs_pending_stale > 0
    signal_streak = (prev_streak + 1) if stale_present else 0
    signal_active = prev_signal_active
    if stale_present and signal_streak >= DEGRADED_PENDING_STALE_SIGNAL_AFTER_TICKS:
        signal_active = True
    if not stale_present:
        signal_active = False

    signal["schema_version"] = DEGRADED_PENDING_STALE_SIGNAL_SCHEMA_VERSION
    signal["active"] = bool(signal_active)
    signal["activate_after_ticks"] = int(DEGRADED_PENDING_STALE_SIGNAL_AFTER_TICKS)
    signal["cooldown_sec"] = int(DEGRADED_PENDING_STALE_SIGNAL_COOLDOWN_SEC)
    signal["stale_ticks_consecutive"] = int(signal_streak)
    signal["pending_stale_count"] = int(degraded_local_runs_pending_stale)
    signal["pending_total"] = int(degraded_local_runs_pending)
    signal["pending_oldest_age_sec"] = int(degraded_local_runs_pending_oldest_age_sec)
    signal["pending_oldest_run_id"] = pending_oldest_run_id
    signal["pending_oldest_task_id"] = pending_oldest_task_id
    signal["degraded_mode"] = bool(degraded_mode_active)
    signal["last_eval_ts"] = int(ref_now_ts)
    signal["last_eval_iso"] = ts_to_iso(ref_now_ts) or now_iso()
    signal["recovery_command"] = "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh queue-sync --json"
    signal["inspect_command"] = (
        f"bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/queue_arbitrator.sh trace --task-id {pending_oldest_task_id} --json"
        if pending_oldest_task_id
        else "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/queue_arbitrator.sh ready-list --json"
    )

    signal.setdefault("activation_count", int(signal.get("activation_count") or 0))
    signal.setdefault("recovery_count", int(signal.get("recovery_count") or 0))

    signal_event: Optional[Dict[str, Any]] = None
    next_emit_after_ts = max(0, int(signal.get("next_emit_after_ts") or 0))
    should_try_emit_active = bool(signal_active) and ((not prev_signal_active) or ref_now_ts >= next_emit_after_ts)

    evidence_ref = to_rel(state_path)
    if signal_active:
        if not prev_signal_active:
            signal["activation_count"] = int(signal.get("activation_count") or 0) + 1
            signal["active_since_ts"] = int(ref_now_ts)
            signal["active_since_iso"] = ts_to_iso(ref_now_ts) or now_iso()
            signal["last_transition"] = "activated"
            signal["last_transition_ts"] = int(ref_now_ts)
            signal["last_transition_iso"] = ts_to_iso(ref_now_ts) or now_iso()
        else:
            signal["active_since_ts"] = int(prev_active_since_ts or ref_now_ts)
            signal["active_since_iso"] = str(signal.get("active_since_iso") or ts_to_iso(signal.get("active_since_ts")) or "")

        if should_try_emit_active:
            signal_summary = (
                "sustained stale degraded backlog "
                f"stale_pending={degraded_local_runs_pending_stale} "
                f"pending_total={degraded_local_runs_pending} "
                f"oldest_age_sec={degraded_local_runs_pending_oldest_age_sec} "
                f"oldest_run={pending_oldest_run_id or 'none'} "
                f"streak={signal_streak}/{DEGRADED_PENDING_STALE_SIGNAL_AFTER_TICKS}"
            )
            signal_fp = (
                "state=active|"
                f"pending_stale={degraded_local_runs_pending_stale}|"
                f"pending_total={degraded_local_runs_pending}|"
                f"oldest_run={pending_oldest_run_id or 'none'}|"
                f"degraded_mode={int(degraded_mode_active)}"
            )
            signal_event = emit_runtime_event(
                cur,
                event_key="degraded_pending_backlog_stale_sustained",
                severity="warn",
                summary=signal_summary,
                evidence_ref=evidence_ref,
                fingerprint_input=signal_fp,
                cooldown_sec=DEGRADED_PENDING_STALE_SIGNAL_COOLDOWN_SEC,
            )
            if bool(signal_event.get("emitted")):
                signal["last_emit_ts"] = int(ref_now_ts)
                signal["last_emit_iso"] = ts_to_iso(ref_now_ts) or now_iso()
                signal["next_emit_after_ts"] = int(ref_now_ts) + int(DEGRADED_PENDING_STALE_SIGNAL_COOLDOWN_SEC)
                signal["next_emit_after_iso"] = ts_to_iso(signal.get("next_emit_after_ts")) or ""
                degraded_pending_stale_signal_emitted = True
    elif prev_signal_active:
        signal["active"] = False
        signal["active_since_ts"] = None
        signal["active_since_iso"] = None
        signal["next_emit_after_ts"] = 0
        signal["next_emit_after_iso"] = ""
        signal["recovery_count"] = int(signal.get("recovery_count") or 0) + 1
        signal["last_transition"] = "recovered"
        signal["last_transition_ts"] = int(ref_now_ts)
        signal["last_transition_iso"] = ts_to_iso(ref_now_ts) or now_iso()
        signal_summary = (
            "stale degraded backlog recovered "
            f"stale_pending={degraded_local_runs_pending_stale} pending_total={degraded_local_runs_pending}"
        )
        signal_event = emit_runtime_event(
            cur,
            event_key="degraded_pending_backlog_recovered",
            severity="info",
            summary=signal_summary,
            evidence_ref=evidence_ref,
            fingerprint_input="state=recovered|pending_stale=0",
            cooldown_sec=300,
        )
        if bool(signal_event.get("emitted")):
            signal["last_emit_ts"] = int(ref_now_ts)
            signal["last_emit_iso"] = ts_to_iso(ref_now_ts) or now_iso()
            degraded_pending_stale_signal_emitted = True

    if signal_event is not None:
        signal["last_event"] = signal_event

    degraded_pending_stale_signal_active = bool(signal.get("active") is True)
    degraded_pending_stale_signal_streak = int(signal.get("stale_ticks_consecutive") or 0)
    degraded_pending_stale_signal_event = signal_event

    degraded_state["degraded_pending_stale_signal"] = signal
    degraded_state["degraded_local_runs_pending_stale_signal_active"] = degraded_pending_stale_signal_active
    degraded_state["degraded_local_runs_pending_stale_signal_streak"] = degraded_pending_stale_signal_streak

    if last_reconciled_meta is not None:
        degraded_state["last_reconciled_degraded_run"] = last_reconciled_meta

    state["queue_infra_degraded"] = degraded_state
    state_runs_writeback = degraded_state != degraded_state_before
# Cleanup stale autopilot metadata rows no longer present in current state payload.
active_task_ids = set(ordered_task_ids + ["autopilot:cycle"])
if active_task_ids:
    placeholders = ",".join(["?"] * len(active_task_ids))
    cur.execute(
        f"DELETE FROM task_file_targets WHERE task_id LIKE 'autopilot:%' AND task_id NOT IN ({placeholders})",
        tuple(active_task_ids),
    )
    cur.execute(
        f"DELETE FROM task_artifacts WHERE task_id LIKE 'autopilot:%' AND task_id NOT IN ({placeholders})",
        tuple(active_task_ids),
    )

# Backfill latest transition evidence refs when status did not change this run.
state_ref_abs = str(state_path)
state_ref_rel = to_rel(state_path)
for task_id, evid in step_evidence_cache.items():
    if not evid:
        continue
    cur.execute(
        """
UPDATE task_transitions
SET evidence_ref = ?
WHERE event_id = (
  SELECT event_id FROM task_transitions
  WHERE task_id = ?
  ORDER BY created_at DESC
  LIMIT 1
)
AND (
  evidence_ref IS NULL OR evidence_ref = '' OR evidence_ref = ? OR evidence_ref = ?
)
""",
        (evid, task_id, state_ref_abs, state_ref_rel),
    )
    if cur.rowcount:
        transitions_backfilled += int(cur.rowcount)
        evidence_refs_used.extend([evid])

# Rebuild deterministic task dependencies (linearized autopilot DAG).
cur.execute(
    """
DELETE FROM task_dependencies
WHERE task_id LIKE 'autopilot:%' OR depends_on_task_id LIKE 'autopilot:%'
"""
)

for idx, task_id in enumerate(ordered_task_ids):
    if idx == 0:
        continue
    prev_task_id = ordered_task_ids[idx - 1]
    cur.execute(
        """
INSERT OR REPLACE INTO task_dependencies (task_id, depends_on_task_id, relation, created_at)
VALUES (?, ?, 'blocks', ?)
""",
        (task_id, prev_task_id, now_iso()),
    )
    dependencies_upserted += 1

# Aggregate cycle status
step_statuses = [str((s or {}).get("status") or "").lower() for s in steps if isinstance(s, dict)]
blocked_present = any(s == "blocked" for s in step_statuses)
all_done = bool(step_statuses) and all(s == "done" for s in step_statuses)
any_running = any(s == "running" for s in step_statuses)

if blocked_present:
    cycle_status = "BLOCKED"
elif all_done and paused:
    cycle_status = "DONE"
elif any_running or active:
    cycle_status = "RUNNING"
elif paused and not all_done:
    cycle_status = "BLOCKED"
else:
    cycle_status = "QUEUED"

cycle_task_id = "autopilot:cycle"
cycle_title = "HL autopilot cycle orchestration"
cycle_acceptance = "All autopilot steps reach done with quality_gate validated"
cycle_role_required = role_required_for_status("outer_gate", cycle_status)
cycle_updated_at = now_iso()
cycle_created_at = ts_to_iso(state.get("last_tick_ts")) or cycle_updated_at
cycle_error = None
if blocked_present:
    cycle_error = "one_or_more_steps_blocked"
elif paused and not all_done:
    cycle_error = "paused_with_incomplete_steps"

prev_cycle = cur.execute("SELECT status FROM work_queue WHERE task_id = ?", (cycle_task_id,)).fetchone()
prev_cycle_status = prev_cycle[0] if prev_cycle else None

cur.execute(
    """
INSERT INTO work_queue (
  task_id, source, title, acceptance_criteria, status, role_required, assigned_agent,
  retry_count, max_retries, last_error_log, cooldown_until, created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(task_id) DO UPDATE SET
  source=excluded.source,
  title=excluded.title,
  acceptance_criteria=excluded.acceptance_criteria,
  status=excluded.status,
  role_required=excluded.role_required,
  assigned_agent=excluded.assigned_agent,
  retry_count=excluded.retry_count,
  max_retries=excluded.max_retries,
  last_error_log=excluded.last_error_log,
  cooldown_until=excluded.cooldown_until,
  updated_at=excluded.updated_at
""",
    (
        cycle_task_id,
        source,
        cycle_title,
        cycle_acceptance,
        cycle_status,
        cycle_role_required,
        "autopilot",
        max(0, cycle - 1),
        max_cycles if max_cycles > 0 else 50,
        cycle_error,
        None,
        cycle_created_at,
        cycle_updated_at,
    ),
)
rows_upserted += 1

# autopilot:cycle depends on every step task.
for task_id in ordered_task_ids:
    cur.execute(
        """
INSERT OR REPLACE INTO task_dependencies (task_id, depends_on_task_id, relation, created_at)
VALUES (?, ?, 'blocks', ?)
""",
        (cycle_task_id, task_id, now_iso()),
    )
    dependencies_upserted += 1

# Refresh cycle task lock targets + artifacts.
cur.execute("DELETE FROM task_file_targets WHERE task_id = ?", (cycle_task_id,))
cur.execute("DELETE FROM task_artifacts WHERE task_id = ?", (cycle_task_id,))
for target in dedupe_keep_order([to_rel(state_path)]):
    cur.execute(
        """
INSERT OR REPLACE INTO task_file_targets (task_id, file_path, lock_mode, created_at)
VALUES (?, ?, 'exclusive', ?)
""",
        (cycle_task_id, target, cycle_updated_at),
    )
    task_file_targets_upserted += 1

cyc_reason = f"paused={int(paused)};active={int(bool(active))};cycle={cycle}/{max_cycles}"
cycle_evidence = [to_rel(state_path)]
if isinstance(active, dict):
    for field in ("log_path", "exit_code_path"):
        p_raw = str(active.get(field) or "").strip()
        if not p_raw:
            continue
        p = pathlib.Path(p_raw)
        if p.exists():
            cycle_evidence.append(to_rel(p))
cyc_evid = " | ".join(dedupe_keep_order(cycle_evidence))
for ref in dedupe_keep_order(cycle_evidence):
    a_type = classify_artifact_type(ref)
    a_id = artifact_id(cycle_task_id, ref, a_type)
    cur.execute(
        """
INSERT OR REPLACE INTO task_artifacts (
  artifact_id, task_id, artifact_type, artifact_path, sha256, metadata_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?)
""",
        (a_id, cycle_task_id, a_type, ref, None, None, cycle_updated_at),
    )
    task_artifacts_upserted += 1

if prev_cycle_status != cycle_status:
    cyc_ev = event_id(cycle_task_id, prev_cycle_status, cycle_status, cycle_updated_at, cyc_reason)
    cur.execute(
        """
INSERT OR IGNORE INTO task_transitions (
  event_id, task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""",
        (cyc_ev, cycle_task_id, prev_cycle_status, cycle_status, "outer_gate", cyc_reason, cyc_evid, cycle_updated_at),
    )
    if cur.rowcount:
        transitions_inserted += 1
        evidence_refs_used.extend([cyc_evid])
        from_role = role_required_for_status("outer_gate", str(prev_cycle_status or "QUEUED")) if prev_cycle_status else ""
        if persist_handoff_packet(
            cur,
            task_id=cycle_task_id,
            transition_event_id=cyc_ev,
            from_status=prev_cycle_status,
            to_status=cycle_status,
            from_role=from_role,
            to_role=cycle_role_required,
            created_at=cycle_updated_at,
            reason=cyc_reason,
            evidence_ref=cyc_evid,
            retry_count=max(0, cycle - 1),
        ):
            handoff_packets_upserted += 1

cur.execute(
    """
UPDATE task_transitions
SET evidence_ref = ?
WHERE event_id = (
  SELECT event_id FROM task_transitions
  WHERE task_id = ?
  ORDER BY created_at DESC
  LIMIT 1
)
AND (
  evidence_ref IS NULL OR evidence_ref = '' OR evidence_ref = ? OR evidence_ref = ?
)
""",
    (cyc_evid, cycle_task_id, state_ref_abs, state_ref_rel),
)
if cur.rowcount:
    transitions_backfilled += int(cur.rowcount)
    evidence_refs_used.extend([cyc_evid])

role_required_counts: Dict[str, int] = {}
for row in cur.execute(
    """
SELECT COALESCE(NULLIF(TRIM(role_required), ''), 'UNSET') AS role_required_key, COUNT(*)
FROM work_queue
WHERE task_id LIKE 'autopilot:%'
GROUP BY role_required_key
ORDER BY role_required_key
"""
).fetchall():
    role_required_counts[str(row[0])] = int(row[1] or 0)

role_required_unset = int(role_required_counts.get("UNSET") or 0)

if invalid_provider_summary_mode == "fail_close" and gate_summary_fail_close_issues:
    con.rollback()
    con.close()
    print(
        json.dumps(
            {
                "ok": False,
                "error": "invalid_provider_failure_summary_schema",
                "invalid_provider_summary_mode": invalid_provider_summary_mode,
                "gate_summaries_invalid": len(gate_summary_fail_close_issues),
                "gate_summary_issue_sample": gate_summary_fail_close_issues[:5],
                "hint": "repair malformed legacy summaries or rerun with --invalid-provider-summary-mode drop",
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(2)

con.commit()
con.close()

if state_runs_writeback:
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, state_path)

evidence_refs_sample = dedupe_keep_order(evidence_refs_used)[:8]

print(
    json.dumps(
        {
            "ok": True,
            "state_json": str(state_path),
            "db_path": str(db_path),
            "rows_upserted": rows_upserted,
            "transitions_inserted": transitions_inserted,
            "transitions_backfilled": transitions_backfilled,
            "dependencies_upserted": dependencies_upserted,
            "task_file_targets_upserted": task_file_targets_upserted,
            "task_artifacts_upserted": task_artifacts_upserted,
            "handoff_packets_upserted": handoff_packets_upserted,
            "degraded_local_runs_seen": degraded_local_runs_seen,
            "degraded_local_runs_backfilled": degraded_local_runs_backfilled,
            "degraded_local_runs_marked_reconciled": degraded_local_runs_marked_reconciled,
            "degraded_local_runs_pending": degraded_local_runs_pending,
            "degraded_local_runs_reconciled": degraded_local_runs_reconciled,
            "degraded_local_runs_pruned": degraded_local_runs_pruned,
            "degraded_local_runs_pruned_by_age": degraded_local_runs_pruned_by_age,
            "degraded_local_runs_pruned_by_count": degraded_local_runs_pruned_by_count,
            "degraded_local_runs_pending_stale": degraded_local_runs_pending_stale,
            "degraded_local_runs_pending_oldest_age_sec": degraded_local_runs_pending_oldest_age_sec,
            "degraded_local_runs_reconciled_oldest_age_sec": degraded_local_runs_reconciled_oldest_age_sec,
            "degraded_local_handoffs_upserted": degraded_local_handoffs_upserted,
            "degraded_local_runs_stale_drain_attempted": degraded_local_runs_stale_drain_attempted,
            "degraded_local_runs_stale_drain_applied": degraded_local_runs_stale_drain_applied,
            "degraded_local_runs_stale_drain_failed": degraded_local_runs_stale_drain_failed,
            "degraded_local_runs_stale_processing_recovered": degraded_local_runs_stale_processing_recovered,
            "degraded_local_runs_stale_running_recovered": degraded_local_runs_stale_running_recovered,
            "degraded_local_runs_recovery_counters": (
                degraded_state.get("stale_task_recovery_counters")
                if isinstance(degraded_state.get("stale_task_recovery_counters"), dict)
                else None
            ),
            "degraded_local_runs_state_writeback": state_runs_writeback,
            "degraded_pending_stale_signal_active": degraded_pending_stale_signal_active,
            "degraded_pending_stale_signal_streak": degraded_pending_stale_signal_streak,
            "degraded_pending_stale_signal_emitted": degraded_pending_stale_signal_emitted,
            "degraded_pending_stale_signal_event": degraded_pending_stale_signal_event,
            "queue_infra_degraded_mode": degraded_mode_active,
            "gate_summaries_invalid": len(gate_summary_validation_issues),
            "gate_summary_issue_sample": gate_summary_validation_issues[:5],
            "invalid_provider_summary_mode": invalid_provider_summary_mode,
            "role_required_counts": role_required_counts,
            "role_required_unset": role_required_unset,
            "evidence_refs_linked": len(dedupe_keep_order(evidence_refs_used)),
            "evidence_refs_sample": evidence_refs_sample,
            "cycle_status": cycle_status,
            "paused": paused,
            "cycle": cycle,
            "max_cycles": max_cycles,
            "active_step": (active or {}).get("step_id") if isinstance(active, dict) else None,
        },
        ensure_ascii=False,
    )
)
PY
