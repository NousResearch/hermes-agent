#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
DB_PATH="${OPENCLAW_CONTINUITY_DB_PATH:-$ROOT/state/continuity/continuity_os.sqlite}"
ACTION_TOKEN=""
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"
MUTATION_TICKET=""
declare -a MUTATION_ATTESTATIONS=()
declare -a MUTATION_ATTESTATION_OBJECTS=()

usage() {
  cat <<'EOF'
Usage: queue_arbitrator.sh <command> [options]

Deterministic queue arbitration helper for continuity work_queue + task_dependencies + file_locks.

Commands:
  ready-list     Show QUEUED tasks with satisfied blocking dependencies.
  claim          Atomically claim one ready task for an agent and acquire declared file locks.
  transition     Transition one task status and optionally release active file locks.
  trace          Replay-oriented task trace (dependencies, transitions, artifacts, locks, handoffs).
  handoffs       Show persisted role-handoff packets.
  locks          Show current file lock table summary.
  remediate      Guided lock/dependency remediation helper (dry-run by default).

Options (ready-list):
  --limit <n>                Max rows (default: 20)
  --json                     JSON output

Options (claim):
  --agent <name>             Required agent identity
  --actor-role <role>        Claim actor role (default: planner)
  --task-id <id>             Optional explicit task_id to claim
  --lock-ttl-sec <n>         Optional lock TTL seconds (default: 3600)
  --json                     JSON output

Options (transition):
  --task-id <id>             Required task id
  --to-status <status>       Required status (QUEUED|RUNNING|REVIEW|DONE|BLOCKED|FAILED|ROLLED_BACK)
  --actor-role <role>        Role for transition event (default: validator)
  --reason <text>            Transition reason
  --evidence-ref <path|...>  Optional evidence ref
  --gate-summary-json <json> Optional structured gate summary (stored in handoff gate_metadata)
  --release-locks            Release ACTIVE file locks held by this task
  --allow-any-transition     Bypass transition policy matrix (manual recovery only)
  --json                     JSON output

Options (trace):
  --task-id <id>             Required task id
  --transitions-limit <n>    Transition rows to include (default: 20)
  --json                     JSON output

Options (handoffs):
  --task-id <id>             Optional filter by task id
  --limit <n>                Max rows (default: 20)
  --json                     JSON output

Options (locks):
  --active-only              Show only ACTIVE locks
  --json                     JSON output

Options (remediate):
  --task-id <id>             Optional task id focus for requeue preview/apply
  --expire-overdue-locks     Mark overdue ACTIVE locks as EXPIRED
  --release-terminal-locks   Release ACTIVE locks held by terminal tasks
  --requeue-resolved-blocked Requeue BLOCKED tasks whose blocking deps are DONE
  --requeue-orphaned-running Requeue RUNNING tasks with no active lock and stale heartbeat
  --orphaned-running-min-sec <n>
                             Staleness threshold for orphaned RUNNING requeue (default: 1800)
  --apply                    Apply requested remediations (default: dry-run)
  --limit <n>                Max preview rows (default: 20)
  --json                     JSON output

Global:
  --db <path>                Continuity DB path override
  --action-token <value>     Canonical mutation token for direct mutating commands
  --truth-anchor <value>     Legacy alias of --action-token
  --allow-legacy-anchor      Allow legacy anchor-only token mode for direct token validation
  --mutation-ticket <value>  Authority ticket JSON string, @path, or path (high-risk token path)
  --attestation <name>       Satisfied authority attestation (repeatable)
  --attestation-object <value> Structured attestation JSON string, @path, or path (repeatable)
  -h, --help
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

cmd="${1:-}"
shift || true

LIMIT="20"
JSON_OUT=0
AGENT=""
TASK_ID=""
LOCK_TTL_SEC="3600"
TO_STATUS=""
ACTOR_ROLE=""
REASON=""
EVIDENCE_REF=""
GATE_SUMMARY_JSON=""
RELEASE_LOCKS=0
ALLOW_ANY_TRANSITION=0
TRANSITIONS_LIMIT="20"
ACTIVE_ONLY=0
REMEDIATE_APPLY=0
REMEDIATE_EXPIRE_OVERDUE=0
REMEDIATE_RELEASE_TERMINAL=0
REMEDIATE_REQUEUE_RESOLVED=0
REMEDIATE_REQUEUE_ORPHANED_RUNNING=0
ORPHANED_RUNNING_MIN_SEC="1800"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)
      DB_PATH="${2:-}"; shift 2 ;;
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
    --limit)
      LIMIT="${2:-}"; shift 2 ;;
    --json)
      JSON_OUT=1; shift ;;
    --agent)
      AGENT="${2:-}"; shift 2 ;;
    --task-id)
      TASK_ID="${2:-}"; shift 2 ;;
    --lock-ttl-sec)
      LOCK_TTL_SEC="${2:-}"; shift 2 ;;
    --to-status)
      TO_STATUS="${2:-}"; shift 2 ;;
    --actor-role)
      ACTOR_ROLE="${2:-}"; shift 2 ;;
    --reason)
      REASON="${2:-}"; shift 2 ;;
    --evidence-ref)
      EVIDENCE_REF="${2:-}"; shift 2 ;;
    --gate-summary-json)
      GATE_SUMMARY_JSON="${2:-}"; shift 2 ;;
    --release-locks)
      RELEASE_LOCKS=1; shift ;;
    --allow-any-transition)
      ALLOW_ANY_TRANSITION=1; shift ;;
    --transitions-limit)
      TRANSITIONS_LIMIT="${2:-}"; shift 2 ;;
    --active-only)
      ACTIVE_ONLY=1; shift ;;
    --apply)
      REMEDIATE_APPLY=1; shift ;;
    --expire-overdue-locks)
      REMEDIATE_EXPIRE_OVERDUE=1; shift ;;
    --release-terminal-locks)
      REMEDIATE_RELEASE_TERMINAL=1; shift ;;
    --requeue-resolved-blocked)
      REMEDIATE_REQUEUE_RESOLVED=1; shift ;;
    --requeue-orphaned-running)
      REMEDIATE_REQUEUE_ORPHANED_RUNNING=1; shift ;;
    --orphaned-running-min-sec)
      ORPHANED_RUNNING_MIN_SEC="${2:-}"; shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if ! [[ "$ORPHANED_RUNNING_MIN_SEC" =~ ^[0-9]+$ ]]; then
  echo "invalid --orphaned-running-min-sec: $ORPHANED_RUNNING_MIN_SEC (expected integer >= 0)" >&2
  exit 2
fi

requires_mutator_guard=0
case "$cmd" in
  claim|transition|remediate)
    requires_mutator_guard=1 ;;
esac

if [[ "$requires_mutator_guard" == "1" ]]; then
  mutation_risk_tier="medium"
  case "$cmd" in
    transition|remediate)
      mutation_risk_tier="high" ;;
  esac

  guard_args=(
    --script "queue_arbitrator.sh"
    --risk-tier "$mutation_risk_tier"
    --mutation-operation "queue_arbitrator:${cmd}"
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
fi

# Outer file lock to serialize mutating arbitrator paths across processes.
if [[ "$cmd" == "claim" || "$cmd" == "transition" || "$cmd" == "remediate" ]]; then
  LOCK_DIR="$ROOT/state/continuity/locks"
  LOCK_FILE="$LOCK_DIR/queue_arbitrator.mutex.lock"
  LOCK_WAIT_SEC="${OPENCLAW_QUEUE_ARB_LOCK_WAIT_SEC:-30}"
  mkdir -p "$LOCK_DIR"
  exec {QUEUE_ARB_LOCK_FD}>"$LOCK_FILE"
  if ! flock -w "$LOCK_WAIT_SEC" "$QUEUE_ARB_LOCK_FD"; then
    echo "queue arbitrator busy: failed to acquire mutex within ${LOCK_WAIT_SEC}s ($LOCK_FILE)" >&2
    exit 1
  fi
fi

OPENCLAW_CONTINUITY_DB_PATH="$DB_PATH" "$ROOT/ops/openclaw/continuity/init_db.sh" >/dev/null

OPENCLAW_ROOT="$ROOT" python3 - "$cmd" "$DB_PATH" "$LIMIT" "$JSON_OUT" "$AGENT" "$TASK_ID" "$LOCK_TTL_SEC" "$TO_STATUS" "$ACTOR_ROLE" "$REASON" "$EVIDENCE_REF" "$GATE_SUMMARY_JSON" "$RELEASE_LOCKS" "$ALLOW_ANY_TRANSITION" "$TRANSITIONS_LIMIT" "$ACTIVE_ONLY" "$REMEDIATE_APPLY" "$REMEDIATE_EXPIRE_OVERDUE" "$REMEDIATE_RELEASE_TERMINAL" "$REMEDIATE_REQUEUE_RESOLVED" "$REMEDIATE_REQUEUE_ORPHANED_RUNNING" "$ORPHANED_RUNNING_MIN_SEC" <<'PY'
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

cmd = str(sys.argv[1] or "").strip()
db_path = str(sys.argv[2] or "").strip()
limit = max(1, min(200, int(sys.argv[3] or 20)))
json_out = bool(int(sys.argv[4]))
agent = str(sys.argv[5] or "").strip()
task_id_arg = str(sys.argv[6] or "").strip()
lock_ttl_sec = max(1, int(sys.argv[7] or 3600))
to_status = str(sys.argv[8] or "").strip().upper()
actor_role_input = str(sys.argv[9] or "").strip().lower()
reason = str(sys.argv[10] or "").strip()
evidence_ref = str(sys.argv[11] or "").strip()
gate_summary_input = str(sys.argv[12] or "").strip()
release_locks = bool(int(sys.argv[13]))
allow_any_transition = bool(int(sys.argv[14]))
transitions_limit = max(1, min(200, int(sys.argv[15] or 20)))
active_only = bool(int(sys.argv[16]))
remediate_apply = bool(int(sys.argv[17]))
remediate_expire_overdue = bool(int(sys.argv[18]))
remediate_release_terminal = bool(int(sys.argv[19]))
remediate_requeue_resolved = bool(int(sys.argv[20]))
remediate_requeue_orphaned_running = bool(int(sys.argv[21]))
orphaned_running_min_sec = max(0, int(sys.argv[22] or 1800))

ALLOWED_TRANSITION_STATUS = {"REVIEW", "DONE", "BLOCKED", "FAILED", "ROLLED_BACK", "RUNNING", "QUEUED"}
ALLOWED_TRANSITIONS = {
    "QUEUED": {"RUNNING", "BLOCKED", "FAILED", "ROLLED_BACK", "QUEUED"},
    "RUNNING": {"REVIEW", "DONE", "BLOCKED", "FAILED", "QUEUED", "RUNNING"},
    "REVIEW": {"DONE", "BLOCKED", "FAILED", "QUEUED", "RUNNING", "REVIEW"},
    "DONE": {"ROLLED_BACK", "DONE"},
    "BLOCKED": {"QUEUED", "RUNNING", "BLOCKED"},
    "FAILED": {"QUEUED", "RUNNING", "FAILED", "ROLLED_BACK"},
    "ROLLED_BACK": {"QUEUED", "RUNNING", "ROLLED_BACK"},
}
ALLOWED_ACTOR_ROLES = {"planner", "executor", "validator", "sre_watchdog", "librarian", "outer_gate"}

if actor_role_input and actor_role_input not in ALLOWED_ACTOR_ROLES:
    print(json.dumps({"ok": False, "error": "invalid_actor_role", "actor_role": actor_role_input, "allowed": sorted(ALLOWED_ACTOR_ROLES)}, ensure_ascii=False, indent=2))
    raise SystemExit(2)

if cmd == "claim":
    actor_role = actor_role_input or "planner"
elif cmd == "transition":
    actor_role = actor_role_input or "validator"
else:
    actor_role = actor_role_input or "planner"

root_path = pathlib.Path(str(os.environ.get("OPENCLAW_ROOT") or "/home/yeqiuqiu/clawd-architect")).resolve()
if str(root_path / "src") not in sys.path:
    sys.path.insert(0, str(root_path / "src"))
continuity_path = root_path / "ops" / "openclaw" / "continuity"
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
    from fixed_now import now_iso_utc as _helper_now_iso_utc, now_ts as _helper_now_ts  # type: ignore
except Exception:  # pragma: no cover - optional helper in minimal test roots
    _helper_now_iso_utc = None
    _helper_now_ts = None


def clock_now_ts() -> int:
    if callable(_helper_now_ts):
        try:
            return int(_helper_now_ts())
        except Exception:
            pass
    return int(dt.datetime.now(dt.timezone.utc).timestamp())


DELEGATED_GATE_SUMMARY_SCHEMA_VERSION = "autopilot.delegated_gate_summary.v1"
REPO_REVIEW_CLOSEOUT_SCHEMA = "clawd.repo_review_queue_closeout_verifier.v1"
REPO_REVIEW_CLAIM_REPORT_RE = re.compile(r"(^|/)repo_[^\n]*?(foldin|closeout)[^\n]*\.md$", re.IGNORECASE)
REPO_REVIEW_CLOSEOUT_VERIFIER_SCRIPT = continuity_path / "check_repo_review_queue_closeout.py"


def normalize_sha256(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if len(raw) == 64 and all(ch in "0123456789abcdef" for ch in raw):
        return raw
    return None


def resolve_gate_summary_path(raw: Any) -> Optional[pathlib.Path]:
    text = str(raw or "").strip()
    if not text:
        return None
    path = pathlib.Path(text)
    if not path.is_absolute():
        path = root_path / path
    try:
        return path.resolve()
    except Exception:
        return path


def sha256_file_with_status(path: Optional[pathlib.Path]) -> Tuple[Optional[str], str]:
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


def bind_delegated_gate_summary(summary: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    if str(summary.get("schema_version") or "") != DELEGATED_GATE_SUMMARY_SCHEMA_VERSION:
        return dict(summary), []

    bound = dict(summary)
    repairs: List[Dict[str, str]] = []

    fields = [
        ("completion_packet_path", "completion_packet_sha256"),
        ("decision_path", "decision_sha256"),
    ]
    for path_field, sha_field in fields:
        raw_path = str(summary.get(path_field) or "").strip()
        declared_sha = normalize_sha256(summary.get(sha_field))
        if not declared_sha:
            bound.pop(sha_field, None)

        if not raw_path:
            if declared_sha:
                repairs.append(
                    {
                        "field": sha_field,
                        "declared": declared_sha,
                        "actual": "",
                        "status": "path_missing",
                        "action": "dropped_unverifiable_digest",
                    }
                )
                bound.pop(sha_field, None)
            continue

        actual_sha, status = sha256_file_with_status(resolve_gate_summary_path(raw_path))
        if actual_sha:
            if declared_sha and declared_sha != actual_sha:
                repairs.append(
                    {
                        "field": sha_field,
                        "declared": declared_sha,
                        "actual": actual_sha,
                    }
                )
            bound[sha_field] = actual_sha
            continue

        repairs.append(
            {
                "field": sha_field,
                "declared": declared_sha or "",
                "actual": "",
                "status": status,
                "action": "dropped_unverifiable_binding",
                "path_field": path_field,
            }
        )
        bound.pop(sha_field, None)
        bound.pop(path_field, None)

    return bound, repairs


def validate_gate_summary_payload(summary: Dict[str, Any]) -> Dict[str, Any]:
    schema_version = str(summary.get("schema_version") or "")
    if schema_version != PROVIDER_FAILURE_SUMMARY_SCHEMA_VERSION:
        return {"ok": True, "issues": []}
    verdict = validate_provider_failure_summary(summary, strict=True)
    issues = verdict.get("issues") if isinstance(verdict.get("issues"), list) else []
    return {
        "ok": bool(verdict.get("ok") is True),
        "issues": [str(item) for item in issues][:12],
    }


gate_summary: Optional[Dict[str, Any]] = None
gate_summary_binding_repairs: List[Dict[str, str]] = []
if gate_summary_input:
    try:
        maybe_summary = json.loads(gate_summary_input)
    except Exception:
        print(json.dumps({"ok": False, "error": "invalid_gate_summary_json"}, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    if not isinstance(maybe_summary, dict):
        print(json.dumps({"ok": False, "error": "gate_summary_must_be_object"}, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    maybe_summary_bound, gate_summary_binding_repairs = bind_delegated_gate_summary(maybe_summary)
    summary_verdict = validate_gate_summary_payload(maybe_summary_bound)
    if not summary_verdict.get("ok"):
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "invalid_gate_summary_schema",
                    "issues": summary_verdict.get("issues") or [],
                    "schema_version": maybe_summary_bound.get("schema_version"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(2)
    gate_summary = maybe_summary_bound


def normalize_role(value: Optional[str]) -> str:
    role = str(value or "").strip().lower()
    return role if role in ALLOWED_ACTOR_ROLES else ""


def next_role_required(prev_status: str, to_status: str, prev_role_required: str, actor_role_value: str) -> str:
    prev_role = normalize_role(prev_role_required)
    actor = normalize_role(actor_role_value)

    if to_status == "REVIEW":
        return "validator"
    if to_status == "DONE":
        return "librarian"
    if to_status in {"BLOCKED", "FAILED", "ROLLED_BACK"}:
        return "sre_watchdog"
    if to_status == "RUNNING":
        return actor or prev_role or "executor"
    # QUEUED
    return prev_role or actor or "planner"


def now_iso() -> str:
    if callable(_helper_now_iso_utc):
        try:
            return str(_helper_now_iso_utc())
        except Exception:
            pass
    return dt.datetime.fromtimestamp(clock_now_ts(), tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def transition_event_id(task_id: str, from_status: Optional[str], to_status: str, created_at: str, actor_role: str, reason: str) -> str:
    seed = f"{task_id}|{from_status or ''}|{to_status}|{created_at}|{actor_role}|{reason}"
    return "tevt_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def lock_id(task_id: str, file_path: str, acquired_at: str) -> str:
    seed = f"{task_id}|{file_path}|{acquired_at}"
    return "flk_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


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
        return path.resolve().relative_to(root_path).as_posix()
    except Exception:
        return str(path)


def delegated_gate_summary_artifact_rows(task_id: str, gate_summary_payload: Optional[Dict[str, Any]], created_at: str) -> List[Dict[str, Any]]:
    if not isinstance(gate_summary_payload, dict):
        return []
    if str(gate_summary_payload.get("schema_version") or "") != DELEGATED_GATE_SUMMARY_SCHEMA_VERSION:
        return []

    rows: List[Dict[str, Any]] = []
    seen = set()

    fields = [
        ("completion_packet_path", "completion_packet_sha256", "completion_packet"),
        ("decision_path", "decision_sha256", "gate_decision"),
    ]
    for path_field, sha_field, binding_kind in fields:
        raw_path = str(gate_summary_payload.get(path_field) or "").strip()
        digest = normalize_sha256(gate_summary_payload.get(sha_field))
        if not raw_path or not digest:
            continue

        path_obj = pathlib.Path(raw_path)
        artifact_path = to_rel(path_obj) if path_obj.is_absolute() else raw_path
        artifact_type = classify_artifact_type(artifact_path)
        row_id = artifact_id(task_id, artifact_path, artifact_type)
        if row_id in seen:
            continue
        seen.add(row_id)

        metadata: Dict[str, Any] = {
            "source": "delegated_gate_summary",
            "binding": binding_kind,
        }
        if binding_kind == "completion_packet":
            metadata["completion_packet"] = True

        rows.append(
            {
                "artifact_id": row_id,
                "artifact_type": artifact_type,
                "artifact_path": artifact_path,
                "sha256": digest,
                "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                "created_at": created_at,
            }
        )

    return rows


def upsert_delegated_gate_summary_artifacts(
    cur: sqlite3.Cursor,
    *,
    task_id: str,
    gate_summary_payload: Optional[Dict[str, Any]],
    created_at: str,
) -> int:
    rows = delegated_gate_summary_artifact_rows(task_id, gate_summary_payload, created_at)
    if not rows:
        return 0

    for row in rows:
        cur.execute(
            """
INSERT OR REPLACE INTO task_artifacts (
  artifact_id, task_id, artifact_type, artifact_path, sha256, metadata_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?)
""",
            (
                row["artifact_id"],
                task_id,
                row["artifact_type"],
                row["artifact_path"],
                row["sha256"],
                row["metadata_json"],
                row["created_at"],
            ),
        )

    return len(rows)


def normalize_paths(rows) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        path = str(row[0] or "").strip()
        mode = str(row[1] or "exclusive").strip().lower() or "exclusive"
        if not path:
            continue
        if mode not in {"exclusive", "shared"}:
            mode = "exclusive"
        out.append({"file_path": path, "lock_mode": mode})
    return out


def gc_expired(cur: sqlite3.Cursor, now: str) -> int:
    cur.execute(
        """
UPDATE file_locks
SET lock_state = 'EXPIRED', released_at = ?
WHERE lock_state = 'ACTIVE'
  AND lock_expires_at IS NOT NULL
  AND lock_expires_at <= ?
""",
        (now, now),
    )
    return int(cur.rowcount or 0)


def ready_rows(cur: sqlite3.Cursor, lim: int, *, now: str, specific_task_id: str = ""):
    base = """
SELECT
  w.task_id,
  w.title,
  w.status,
  w.role_required,
  w.created_at,
  w.updated_at,
  w.retry_count,
  w.max_retries
FROM work_queue w
WHERE w.status = 'QUEUED'
  AND (w.cooldown_until IS NULL OR w.cooldown_until = '' OR w.cooldown_until <= ?)
  AND NOT EXISTS (
    SELECT 1
    FROM task_dependencies d
    LEFT JOIN work_queue dep ON dep.task_id = d.depends_on_task_id
    WHERE d.task_id = w.task_id
      AND d.relation = 'blocks'
      AND COALESCE(dep.status, 'MISSING') <> 'DONE'
  )
"""
    params: List[Any] = [now]
    if specific_task_id:
        base += " AND w.task_id = ?"
        params.append(specific_task_id)
    base += " ORDER BY w.created_at ASC LIMIT ?"
    params.append(lim)
    return cur.execute(base, tuple(params)).fetchall()


def inspect_claim_blocker(cur: sqlite3.Cursor, task_id: str, *, now: str) -> Optional[Dict[str, Any]]:
    row = cur.execute(
        "SELECT status, role_required, cooldown_until FROM work_queue WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if not row:
        return {"task_id": task_id, "reason": "task_not_found"}

    status = str(row[0] or "").strip().upper()
    role_required = normalize_role(row[1])
    cooldown_until = str(row[2] or "").strip()
    if status != "QUEUED":
        payload: Dict[str, Any] = {
            "task_id": task_id,
            "reason": "status_not_queued",
            "status": status or "UNKNOWN",
        }
        if role_required:
            payload["role_required"] = role_required
        return payload

    if cooldown_until and cooldown_until > now:
        retry_after_sec = None
        try:
            retry_after_sec = max(
                1,
                int((dt.datetime.fromisoformat(cooldown_until.replace("Z", "+00:00")) - dt.datetime.fromisoformat(now.replace("Z", "+00:00"))).total_seconds()),
            )
        except Exception:
            retry_after_sec = None
        payload = {
            "task_id": task_id,
            "reason": "cooldown_active",
            "cooldown_until": cooldown_until,
            "role_required": role_required or None,
        }
        if retry_after_sec is not None:
            payload["retry_after_sec"] = int(retry_after_sec)
        return payload

    blockers = cur.execute(
        """
SELECT d.depends_on_task_id, COALESCE(dep.status, 'MISSING')
FROM task_dependencies d
LEFT JOIN work_queue dep ON dep.task_id = d.depends_on_task_id
WHERE d.task_id = ?
  AND d.relation = 'blocks'
  AND COALESCE(dep.status, 'MISSING') <> 'DONE'
ORDER BY d.depends_on_task_id ASC
LIMIT 8
""",
        (task_id,),
    ).fetchall()
    if blockers:
        return {
            "task_id": task_id,
            "reason": "dependency_blocked",
            "blockers": [
                {
                    "depends_on_task_id": str(item[0] or ""),
                    "status": str(item[1] or "MISSING"),
                }
                for item in blockers
            ],
            "role_required": role_required or None,
        }

    return {"task_id": task_id, "reason": "no_claimable_task"}


def conflict_rows(cur: sqlite3.Cursor, task_id: str, file_paths: List[str]):
    if not file_paths:
        return []
    placeholders = ",".join(["?"] * len(file_paths))
    query = f"""
SELECT file_path, locked_by_task_id, lock_mode, acquired_at, lock_expires_at
FROM file_locks
WHERE lock_state = 'ACTIVE'
  AND locked_by_task_id <> ?
  AND file_path IN ({placeholders})
ORDER BY acquired_at ASC
"""
    params: List[Any] = [task_id]
    params.extend(file_paths)
    return cur.execute(query, tuple(params)).fetchall()


def split_refs(raw: str) -> List[str]:
    refs: List[str] = []
    seen = set()
    for piece in [p.strip() for p in str(raw or "").split("|") if str(p).strip()]:
        if piece in seen:
            continue
        seen.add(piece)
        refs.append(piece)
    return refs


def normalize_ref_token(raw: Any) -> str:
    return str(raw or "").strip().replace("\\", "/")


def ref_aliases(raw_ref: Any) -> set[str]:
    ref = normalize_ref_token(raw_ref)
    if not ref:
        return set()

    aliases: set[str] = {ref}
    root = root_path.resolve()
    ref_path = pathlib.Path(ref)
    try:
        resolved = ref_path.resolve() if ref_path.is_absolute() else (root / ref_path).resolve()
        aliases.add(normalize_ref_token(resolved))
        try:
            aliases.add(normalize_ref_token(resolved.relative_to(root)))
        except ValueError:
            pass
    except Exception:
        pass

    return {val for val in aliases if val}


def resolve_inside_root(relative_path: str) -> Optional[pathlib.Path]:
    candidate = (root_path / str(relative_path or "")).resolve()
    root = root_path.resolve()
    try:
        if not candidate.is_relative_to(root):
            return None
    except AttributeError:
        if str(candidate).startswith(str(root)) is False:
            return None
    return candidate


def repo_review_claim_report_refs(evidence_refs: List[str]) -> List[str]:
    refs: List[str] = []
    seen: set[str] = set()
    for raw in evidence_refs:
        ref = normalize_ref_token(raw)
        if not ref:
            continue
        lowered = ref.lower()
        if not REPO_REVIEW_CLAIM_REPORT_RE.search(lowered):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        refs.append(ref)
    return refs


def load_repo_review_closeout_verifier() -> Tuple[Any, Optional[str]]:
    script_path = REPO_REVIEW_CLOSEOUT_VERIFIER_SCRIPT
    try:
        spec = importlib.util.spec_from_file_location("check_repo_review_queue_closeout", script_path)
        if spec is None or spec.loader is None:
            return None, "repo_review_closeout_spec_unavailable"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        return None, f"repo_review_closeout_load_failed:{exc}"

    evaluate_fn = getattr(module, "evaluate", None)
    resolve_path_fn = getattr(module, "resolve_repo_path", None)
    if not callable(evaluate_fn) or not callable(resolve_path_fn):
        return None, "repo_review_closeout_contract_unavailable"
    return module, None


def resolve_repo_review_verifier_payloads(evidence_refs: List[str]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for raw in evidence_refs:
        path = resolve_inside_root(raw)
        if path is None or not path.exists() or not path.is_file():
            continue
        try:
            maybe = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(maybe, dict):
            continue
        if str(maybe.get("schema") or "").strip() != REPO_REVIEW_CLOSEOUT_SCHEMA:
            continue
        payloads.append(maybe)
    return payloads


def replay_repo_review_closeout_payload(verifier_payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    module, load_err = load_repo_review_closeout_verifier()
    if module is None:
        return None, load_err or "repo_review_closeout_gate_unavailable"

    try:
        resolve_repo_path = getattr(module, "resolve_repo_path")
        evaluate = getattr(module, "evaluate")
        default_primary = str(getattr(module, "DEFAULT_PRIMARY_PATH"))
        default_normalized = str(getattr(module, "DEFAULT_NORMALIZED_PATH"))
        default_terminal = list(getattr(module, "DEFAULT_TERMINAL_STATUSES"))

        primary_raw = str(verifier_payload.get("primary_path") or default_primary)
        normalized_raw = str(verifier_payload.get("normalized_path") or default_normalized)
        report_raw = str(verifier_payload.get("report_path") or "").strip()
        if not report_raw:
            return None, "repo_review_closeout_report_path_missing"

        target_rows_raw = verifier_payload.get("target_rows")
        target_rows = [str(item).strip() for item in (target_rows_raw or []) if str(item).strip()]
        if not target_rows:
            return None, "repo_review_closeout_target_rows_missing"

        terminal_raw = verifier_payload.get("terminal_statuses")
        terminal_statuses = [str(item).strip() for item in (terminal_raw or []) if str(item).strip()]
        if not terminal_statuses:
            terminal_statuses = [str(item).strip() for item in default_terminal if str(item).strip()]

        repo_root = root_path.resolve()
        primary_path = resolve_repo_path(repo_root, primary_raw)
        normalized_path = resolve_repo_path(repo_root, normalized_raw)
        report_path = resolve_repo_path(repo_root, report_raw)

        replay = evaluate(
            repo_root=repo_root,
            primary_path=primary_path,
            normalized_path=normalized_path,
            target_rows=target_rows,
            terminal_statuses=terminal_statuses,
            report_path=report_path,
        )
    except Exception as exc:
        return None, f"repo_review_closeout_replay_failed:{exc}"

    if not isinstance(replay, dict):
        return None, "repo_review_closeout_replay_invalid_payload"
    return replay, None


def enforce_repo_review_closeout_transition(
    *,
    task_id: str,
    to_status_value: str,
    evidence_ref_text: str,
    allow_transition_override: bool,
) -> Optional[Dict[str, Any]]:
    if to_status_value != "DONE":
        return None

    # Keep explicit manual-recovery override semantics unchanged.
    if allow_transition_override:
        return None

    evidence_refs = split_refs(evidence_ref_text)
    report_refs = repo_review_claim_report_refs(evidence_refs)
    if not report_refs:
        return None

    verifier_payloads = resolve_repo_review_verifier_payloads(evidence_refs)
    if not verifier_payloads:
        return {
            "ok": False,
            "error": "repo_review_closeout_verifier_evidence_required",
            "task_id": task_id,
            "to_status": to_status_value,
            "report_refs": report_refs,
            "detail": (
                "Repo-review fold-in/closeout transition claim requires verifier evidence ref "
                "(schema clawd.repo_review_queue_closeout_verifier.v1) in --evidence-ref"
            ),
        }

    for report_ref in report_refs:
        report_alias = ref_aliases(report_ref)
        matched_payload: Optional[Dict[str, Any]] = None
        for payload in verifier_payloads:
            payload_report = str(payload.get("report_path") or "").strip()
            if not payload_report:
                continue
            if ref_aliases(payload_report) & report_alias:
                matched_payload = payload
                break

        if matched_payload is None:
            return {
                "ok": False,
                "error": "repo_review_closeout_verifier_report_binding_missing",
                "task_id": task_id,
                "to_status": to_status_value,
                "report_ref": report_ref,
                "detail": "No verifier payload in --evidence-ref binds to this repo-review closeout report",
            }

        replay, replay_err = replay_repo_review_closeout_payload(matched_payload)
        if replay_err is not None:
            return {
                "ok": False,
                "error": "repo_review_closeout_verifier_replay_unavailable",
                "task_id": task_id,
                "to_status": to_status_value,
                "report_ref": report_ref,
                "detail": replay_err,
            }

        replay_decision = str((replay or {}).get("decision") or "").strip().upper()
        if replay_decision != "PASS":
            return {
                "ok": False,
                "error": "repo_review_closeout_verifier_replay_blocked",
                "task_id": task_id,
                "to_status": to_status_value,
                "report_ref": report_ref,
                "block_reason": str((replay or {}).get("block_reason") or "unknown"),
                "verifier": replay,
            }

    return None


def parse_json_field(raw: Any, fallback: Any):
    txt = str(raw or "").strip()
    if not txt:
        return fallback
    try:
        return json.loads(txt)
    except Exception:
        return fallback


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


def active_lock_refs(cur: sqlite3.Cursor, task_id: str) -> List[str]:
    rows = cur.execute(
        """
SELECT file_path
FROM file_locks
WHERE locked_by_task_id = ? AND lock_state = 'ACTIVE'
ORDER BY file_path ASC
""",
        (task_id,),
    ).fetchall()
    return [str(r[0]) for r in rows if str(r[0] or "").strip()]


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


def handoff_packet_id(task_id: str, transition_id: str, created_at: str, from_role: str, to_role: str) -> str:
    seed = f"{task_id}|{transition_id}|{created_at}|{from_role}|{to_role}"
    return "thp_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def persist_handoff_packet(
    cur: sqlite3.Cursor,
    *,
    task_id: str,
    transition_id: str,
    from_status: Optional[str],
    to_status: str,
    from_role: str,
    to_role: str,
    reason_text: str,
    evidence_ref_text: str,
    created_at: str,
    policy_override: bool,
    retry_count: int,
    lock_refs: Optional[List[str]] = None,
    gate_summary: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    src_role = normalize_role(from_role)
    dst_role = normalize_role(to_role)
    if not src_role or not dst_role or src_role == dst_role:
        return None

    parent = parent_task_id(cur, task_id)
    refs = split_refs(evidence_ref_text)
    linkage = {
        "task_id": task_id,
        "parent_task_id": parent,
        "transition_event_id": transition_id,
    }
    gate_meta = {
        "from_status": from_status,
        "to_status": to_status,
        "reason": reason_text or None,
        "policy_override": bool(policy_override),
        "gate_summary": gate_summary if isinstance(gate_summary, dict) and gate_summary else None,
    }
    packet_id = handoff_packet_id(task_id, transition_id, created_at, src_role, dst_role)
    next_gate = next_gate_for_status(to_status)
    model_tier = str(os.environ.get("OPENCLAW_SWARM_MODEL_TIER", "unknown") or "unknown").strip() or "unknown"

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
            transition_id,
            src_role,
            dst_role,
            from_status,
            to_status,
            created_at,
            json.dumps(refs, ensure_ascii=False),
            json.dumps(gate_meta, ensure_ascii=False, sort_keys=True),
            json.dumps(linkage, ensure_ascii=False, sort_keys=True),
            json.dumps(lock_refs or [], ensure_ascii=False),
            next_gate,
            0,
            model_tier,
            int(retry_count or 0),
            (
                str((gate_summary or {}).get("summary_signature") or reason_text)[:240]
                if to_status in {"BLOCKED", "FAILED", "ROLLED_BACK"}
                else None
            ),
        ),
    )
    return packet_id


def default_role_for_task(task_id: str) -> str:
    tid = str(task_id or "").strip()
    if tid == "autopilot:cycle":
        return "outer_gate"
    if tid == "autopilot:apply_fixes":
        return "executor"
    if tid == "autopilot:quality_gate":
        return "validator"
    if tid.startswith("autopilot:"):
        return "planner"
    if tid == "parity:weekly_harness":
        return "validator"
    if tid == "continuity:normalize_event_sources":
        return "sre_watchdog"
    return "planner"


con = sqlite3.connect(db_path)
con.row_factory = sqlite3.Row
cur = con.cursor()

if cmd == "ready-list":
    rows = ready_rows(cur, limit, now=now_iso())
    payload = []
    for r in rows:
        payload.append(
            {
                "task_id": r["task_id"],
                "title": r["title"],
                "status": r["status"],
                "role_required": r["role_required"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "retry_count": int(r["retry_count"] or 0),
                "max_retries": int(r["max_retries"] or 0),
            }
        )
    out = {"ok": True, "command": cmd, "ready_count": len(payload), "items": payload}
    if json_out:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("READY TASKS")
        print(f"- count: {len(payload)}")
        for item in payload:
            print(
                f"- {item['task_id']} status={item['status']} role_required={item.get('role_required') or 'n/a'} "
                f"retry={item['retry_count']}/{item['max_retries']} title={item['title']}"
            )
    con.close()
    raise SystemExit(0)

if cmd == "claim":
    if not agent:
        out = {"ok": False, "error": "agent_required", "hint": "use --agent <name>"}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        con.close()
        raise SystemExit(2)

    claim_candidates = ready_rows(cur, limit if not task_id_arg else 1, now=now_iso(), specific_task_id=task_id_arg)
    skipped: List[Dict[str, Any]] = []
    claimed: Optional[Dict[str, Any]] = None

    for row in claim_candidates:
        task_id = str(row["task_id"])
        started_at = now_iso()

        cur.execute("BEGIN IMMEDIATE")
        gc_expired(cur, started_at)

        targets = normalize_paths(
            cur.execute(
                "SELECT file_path, lock_mode FROM task_file_targets WHERE task_id = ? ORDER BY file_path",
                (task_id,),
            ).fetchall()
        )
        file_paths = [t["file_path"] for t in targets]
        conflicts = conflict_rows(cur, task_id, file_paths)
        if conflicts:
            con.rollback()
            skipped.append(
                {
                    "task_id": task_id,
                    "reason": "file_lock_conflict",
                    "conflicts": [
                        {
                            "file_path": c[0],
                            "locked_by_task_id": c[1],
                            "lock_mode": c[2],
                            "acquired_at": c[3],
                            "lock_expires_at": c[4],
                        }
                        for c in conflicts
                    ],
                }
            )
            continue

        prev_row = cur.execute("SELECT status, role_required, retry_count FROM work_queue WHERE task_id = ?", (task_id,)).fetchone()
        prev_status = str(prev_row[0]) if prev_row else None
        prev_role_required = normalize_role(prev_row[1] if prev_row else "")
        retry_count = int((prev_row[2] if prev_row else 0) or 0)

        if prev_role_required and actor_role != prev_role_required:
            con.rollback()
            skipped.append(
                {
                    "task_id": task_id,
                    "reason": "role_required_mismatch",
                    "expected_role": prev_role_required,
                    "actor_role": actor_role,
                }
            )
            continue

        next_role = next_role_required(prev_status or "QUEUED", "RUNNING", prev_role_required, actor_role)

        cur.execute(
            """
UPDATE work_queue
SET status = 'RUNNING', role_required = ?, assigned_agent = ?, updated_at = ?
WHERE task_id = ? AND status = 'QUEUED'
""",
            (next_role, agent, started_at, task_id),
        )
        if int(cur.rowcount or 0) != 1:
            con.rollback()
            skipped.append({"task_id": task_id, "reason": "lost_claim_race"})
            continue

        tr_reason = reason or "arbitrator_claim"
        tr_event_id = transition_event_id(task_id, prev_status, "RUNNING", started_at, actor_role, tr_reason)
        cur.execute(
            """
INSERT OR IGNORE INTO task_transitions (
  event_id, task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""",
            (tr_event_id, task_id, prev_status, "RUNNING", actor_role, tr_reason, None, started_at),
        )

        lock_expires_at = (
            dt.datetime.fromtimestamp(clock_now_ts() + lock_ttl_sec, tz=dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

        for target in targets:
            lid = lock_id(task_id, target["file_path"], started_at)
            try:
                cur.execute(
                    """
INSERT INTO file_locks (
  lock_id, file_path, lock_mode, lock_state, locked_by_task_id,
  lock_reason, acquired_at, lock_expires_at, released_at
) VALUES (?, ?, ?, 'ACTIVE', ?, ?, ?, ?, NULL)
""",
                    (
                        lid,
                        target["file_path"],
                        target["lock_mode"],
                        task_id,
                        "queue_claim",
                        started_at,
                        lock_expires_at,
                    ),
                )
            except sqlite3.IntegrityError:
                con.rollback()
                skipped.append(
                    {
                        "task_id": task_id,
                        "reason": "file_lock_conflict_race",
                        "conflicts": [{"file_path": target["file_path"]}],
                    }
                )
                break
        else:
            packet_id = persist_handoff_packet(
                cur,
                task_id=task_id,
                transition_id=tr_event_id,
                from_status=prev_status,
                to_status="RUNNING",
                from_role=prev_role_required,
                to_role=next_role,
                reason_text=tr_reason,
                evidence_ref_text="",
                created_at=started_at,
                policy_override=False,
                retry_count=retry_count,
                lock_refs=active_lock_refs(cur, task_id),
            )
            con.commit()
            claimed = {
                "task_id": task_id,
                "agent": agent,
                "actor_role": actor_role,
                "claimed_at": started_at,
                "lock_ttl_sec": lock_ttl_sec,
                "role_required": next_role,
                "lock_targets": targets,
                "handoff_packet_id": packet_id,
            }
            break

    if claimed is None and task_id_arg and not skipped:
        blocker = inspect_claim_blocker(cur, task_id_arg, now=now_iso())
        if isinstance(blocker, dict) and blocker:
            skipped.append(blocker)

    out = {
        "ok": claimed is not None,
        "command": cmd,
        "claimed": claimed,
        "skipped": skipped,
    }
    if claimed is None:
        out["error"] = "no_claimable_task"
    if json_out or claimed is None:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(
            f"CLAIMED: {claimed['task_id']} agent={claimed['agent']} actor_role={claimed.get('actor_role')} "
            f"role_required={claimed.get('role_required')} lock_targets={len(claimed['lock_targets'])} "
            f"handoff_packet={claimed.get('handoff_packet_id') or 'none'}"
        )
    con.close()
    raise SystemExit(0 if claimed is not None else 1)

if cmd == "transition":
    if not task_id_arg:
        print(json.dumps({"ok": False, "error": "task_id_required"}, ensure_ascii=False, indent=2))
        con.close()
        raise SystemExit(2)
    if to_status not in ALLOWED_TRANSITION_STATUS:
        print(json.dumps({"ok": False, "error": "invalid_to_status", "allowed": sorted(ALLOWED_TRANSITION_STATUS)}, ensure_ascii=False, indent=2))
        con.close()
        raise SystemExit(2)

    ts = now_iso()
    cur.execute("BEGIN IMMEDIATE")

    prev_row = cur.execute("SELECT status, role_required, retry_count FROM work_queue WHERE task_id = ?", (task_id_arg,)).fetchone()
    if not prev_row:
        con.rollback()
        print(json.dumps({"ok": False, "error": "task_not_found", "task_id": task_id_arg}, ensure_ascii=False, indent=2))
        con.close()
        raise SystemExit(1)
    prev_status = str(prev_row[0])
    prev_role_required = normalize_role(prev_row[1])
    retry_count = int(prev_row[2] or 0)

    allowed_targets = ALLOWED_TRANSITIONS.get(prev_status, set())
    if not allow_any_transition and to_status not in allowed_targets:
        con.rollback()
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "invalid_transition",
                    "task_id": task_id_arg,
                    "from_status": prev_status,
                    "to_status": to_status,
                    "allowed_to_status": sorted(allowed_targets),
                    "hint": "use --allow-any-transition for explicit manual recovery overrides",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        con.close()
        raise SystemExit(2)

    if not allow_any_transition and prev_role_required and actor_role != prev_role_required:
        con.rollback()
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "role_required_mismatch",
                    "task_id": task_id_arg,
                    "from_status": prev_status,
                    "to_status": to_status,
                    "actor_role": actor_role,
                    "expected_role_required": prev_role_required,
                    "hint": "use --allow-any-transition for explicit manual recovery overrides",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        con.close()
        raise SystemExit(2)

    repo_review_closeout_violation = enforce_repo_review_closeout_transition(
        task_id=task_id_arg,
        to_status_value=to_status,
        evidence_ref_text=evidence_ref,
        allow_transition_override=allow_any_transition,
    )
    if isinstance(repo_review_closeout_violation, dict):
        con.rollback()
        print(json.dumps(repo_review_closeout_violation, ensure_ascii=False, indent=2))
        con.close()
        raise SystemExit(2)

    next_role = next_role_required(prev_status, to_status, prev_role_required, actor_role)

    cur.execute(
        "UPDATE work_queue SET status = ?, role_required = ?, updated_at = ? WHERE task_id = ?",
        (to_status, next_role, ts, task_id_arg),
    )

    tr_event_id = transition_event_id(task_id_arg, prev_status, to_status, ts, actor_role, reason or "manual_transition")
    cur.execute(
        """
INSERT OR IGNORE INTO task_transitions (
  event_id, task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""",
        (tr_event_id, task_id_arg, prev_status, to_status, actor_role, reason or "manual_transition", evidence_ref or None, ts),
    )

    lock_refs_before_release = active_lock_refs(cur, task_id_arg)
    packet_id = persist_handoff_packet(
        cur,
        task_id=task_id_arg,
        transition_id=tr_event_id,
        from_status=prev_status,
        to_status=to_status,
        from_role=prev_role_required,
        to_role=next_role,
        reason_text=reason or "manual_transition",
        evidence_ref_text=evidence_ref,
        created_at=ts,
        policy_override=allow_any_transition,
        retry_count=retry_count,
        lock_refs=lock_refs_before_release,
        gate_summary=gate_summary,
    )

    task_artifacts_upserted = upsert_delegated_gate_summary_artifacts(
        cur,
        task_id=task_id_arg,
        gate_summary_payload=gate_summary,
        created_at=ts,
    )

    released = 0
    if release_locks:
        cur.execute(
            """
UPDATE file_locks
SET lock_state = 'RELEASED', released_at = ?
WHERE locked_by_task_id = ? AND lock_state = 'ACTIVE'
""",
            (ts, task_id_arg),
        )
        released = int(cur.rowcount or 0)

    con.commit()
    out = {
        "ok": True,
        "command": cmd,
        "task_id": task_id_arg,
        "from_status": prev_status,
        "to_status": to_status,
        "actor_role": actor_role,
        "from_role_required": prev_role_required or None,
        "role_required": next_role,
        "released_locks": released,
        "event_id": tr_event_id,
        "handoff_packet_id": packet_id,
        "task_artifacts_upserted": task_artifacts_upserted,
        "updated_at": ts,
        "policy_override": allow_any_transition,
        "gate_summary": gate_summary if isinstance(gate_summary, dict) and gate_summary else None,
        "gate_summary_binding_repairs": gate_summary_binding_repairs,
    }
    if json_out:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(
            f"TRANSITIONED: {task_id_arg} {prev_status}->{to_status} actor_role={actor_role} "
            f"role_required={next_role}; released_locks={released}; handoff_packet={packet_id or 'none'}"
        )
    con.close()
    raise SystemExit(0)

if cmd == "trace":
    if not task_id_arg:
        print(json.dumps({"ok": False, "error": "task_id_required"}, ensure_ascii=False, indent=2))
        con.close()
        raise SystemExit(2)

    task_row = cur.execute(
        """
SELECT task_id, source, title, acceptance_criteria, status, role_required, assigned_agent,
       retry_count, max_retries, last_error_log, cooldown_until, created_at, updated_at
FROM work_queue
WHERE task_id = ?
""",
        (task_id_arg,),
    ).fetchone()

    if not task_row:
        print(json.dumps({"ok": False, "error": "task_not_found", "task_id": task_id_arg}, ensure_ascii=False, indent=2))
        con.close()
        raise SystemExit(1)

    dependencies = [
        {
            "depends_on_task_id": r[0],
            "relation": r[1],
            "depends_on_status": r[2],
        }
        for r in cur.execute(
            """
SELECT d.depends_on_task_id, d.relation, w.status
FROM task_dependencies d
LEFT JOIN work_queue w ON w.task_id = d.depends_on_task_id
WHERE d.task_id = ?
ORDER BY d.depends_on_task_id
""",
            (task_id_arg,),
        ).fetchall()
    ]

    dependents = [
        {
            "task_id": r[0],
            "relation": r[1],
            "task_status": r[2],
        }
        for r in cur.execute(
            """
SELECT d.task_id, d.relation, w.status
FROM task_dependencies d
LEFT JOIN work_queue w ON w.task_id = d.task_id
WHERE d.depends_on_task_id = ?
ORDER BY d.task_id
""",
            (task_id_arg,),
        ).fetchall()
    ]

    targets = [
        {
            "file_path": r[0],
            "lock_mode": r[1],
            "declared_at": r[2],
        }
        for r in cur.execute(
            """
SELECT file_path, lock_mode, created_at
FROM task_file_targets
WHERE task_id = ?
ORDER BY file_path
""",
            (task_id_arg,),
        ).fetchall()
    ]

    artifacts = [
        {
            "artifact_id": r[0],
            "artifact_type": r[1],
            "artifact_path": r[2],
            "sha256": r[3],
            "created_at": r[4],
        }
        for r in cur.execute(
            """
SELECT artifact_id, artifact_type, artifact_path, sha256, created_at
FROM task_artifacts
WHERE task_id = ?
ORDER BY created_at DESC
""",
            (task_id_arg,),
        ).fetchall()
    ]

    transitions = [
        {
            "event_id": r[0],
            "from_status": r[1],
            "to_status": r[2],
            "actor_role": r[3],
            "reason": r[4],
            "evidence_ref": r[5],
            "created_at": r[6],
        }
        for r in cur.execute(
            """
SELECT event_id, from_status, to_status, actor_role, reason, evidence_ref, created_at
FROM task_transitions
WHERE task_id = ?
ORDER BY created_at DESC, rowid DESC
LIMIT ?
""",
            (task_id_arg, transitions_limit),
        ).fetchall()
    ]

    locks = [
        {
            "lock_id": r[0],
            "file_path": r[1],
            "lock_mode": r[2],
            "lock_state": r[3],
            "lock_reason": r[4],
            "acquired_at": r[5],
            "lock_expires_at": r[6],
            "released_at": r[7],
        }
        for r in cur.execute(
            """
SELECT lock_id, file_path, lock_mode, lock_state, lock_reason, acquired_at, lock_expires_at, released_at
FROM file_locks
WHERE locked_by_task_id = ?
ORDER BY acquired_at DESC
""",
            (task_id_arg,),
        ).fetchall()
    ]

    handoffs = []
    for r in cur.execute(
        """
SELECT packet_id, transition_event_id, parent_task_id, from_role, to_role,
       from_status, to_status, created_at, evidence_refs_json, gate_metadata_json,
       task_linkage_json, lock_refs_json, next_gate, budget_tokens_used, model_tier,
       retry_count, failure_signature
FROM task_handoff_packets
WHERE task_id = ?
ORDER BY created_at DESC
LIMIT ?
""",
        (task_id_arg, transitions_limit),
    ).fetchall():
        evidence_refs = parse_json_field(r[8], [])
        gate_metadata = parse_json_field(r[9], {})
        task_linkage = parse_json_field(r[10], {})
        lock_refs = parse_json_field(r[11], [])

        handoffs.append(
            {
                "packet_id": r[0],
                "transition_event_id": r[1],
                "parent_task_id": r[2],
                "from_role": r[3],
                "to_role": r[4],
                "from_status": r[5],
                "to_status": r[6],
                "created_at": r[7],
                "evidence_refs": evidence_refs if isinstance(evidence_refs, list) else [],
                "gate_metadata": gate_metadata if isinstance(gate_metadata, dict) else {},
                "task_linkage": task_linkage if isinstance(task_linkage, dict) else {},
                "lock_refs": lock_refs if isinstance(lock_refs, list) else [],
                "next_gate": r[12],
                "budget_tokens_used": int(r[13] or 0),
                "model_tier": r[14],
                "retry_count": int(r[15] or 0) if r[15] is not None else None,
                "failure_signature": r[16],
            }
        )

    out = {
        "ok": True,
        "command": cmd,
        "task": {
            "task_id": task_row[0],
            "source": task_row[1],
            "title": task_row[2],
            "acceptance_criteria": task_row[3],
            "status": task_row[4],
            "role_required": task_row[5],
            "assigned_agent": task_row[6],
            "retry_count": int(task_row[7] or 0),
            "max_retries": int(task_row[8] or 0),
            "last_error_log": task_row[9],
            "cooldown_until": task_row[10],
            "created_at": task_row[11],
            "updated_at": task_row[12],
        },
        "dependencies": dependencies,
        "dependents": dependents,
        "file_targets": targets,
        "artifacts": artifacts,
        "transitions": transitions,
        "file_locks": locks,
        "handoff_packets": handoffs,
    }

    if json_out:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(
            f"TRACE: {task_id_arg} status={out['task']['status']} role_required={out['task'].get('role_required') or 'n/a'} "
            f"agent={out['task']['assigned_agent'] or 'n/a'}"
        )
        print(f"- dependencies: {len(dependencies)}")
        print(f"- dependents: {len(dependents)}")
        print(f"- file_targets: {len(targets)}")
        print(f"- artifacts: {len(artifacts)}")
        print(f"- transitions: {len(transitions)}")
        print(f"- file_locks: {len(locks)}")
        print(f"- handoff_packets: {len(handoffs)}")

    con.close()
    raise SystemExit(0)

if cmd == "handoffs":
    where = ""
    params: List[Any] = []
    if task_id_arg:
        where = "WHERE task_id = ?"
        params.append(task_id_arg)

    rows = cur.execute(
        f"""
SELECT packet_id, task_id, parent_task_id, transition_event_id,
       from_role, to_role, from_status, to_status, created_at,
       evidence_refs_json, gate_metadata_json, task_linkage_json, lock_refs_json,
       next_gate, budget_tokens_used, model_tier, retry_count, failure_signature
FROM task_handoff_packets
{where}
ORDER BY created_at DESC
LIMIT ?
""",
        tuple(params + [limit]),
    ).fetchall()

    items: List[Dict[str, Any]] = []
    for r in rows:
        items.append(
            {
                "packet_id": r[0],
                "task_id": r[1],
                "parent_task_id": r[2],
                "transition_event_id": r[3],
                "from_role": r[4],
                "to_role": r[5],
                "from_status": r[6],
                "to_status": r[7],
                "created_at": r[8],
                "evidence_refs": parse_json_field(r[9], []),
                "gate_metadata": parse_json_field(r[10], {}),
                "task_linkage": parse_json_field(r[11], {}),
                "lock_refs": parse_json_field(r[12], []),
                "next_gate": r[13],
                "budget_tokens_used": int(r[14] or 0),
                "model_tier": r[15],
                "retry_count": int(r[16] or 0) if r[16] is not None else None,
                "failure_signature": r[17],
            }
        )

    out = {
        "ok": True,
        "command": cmd,
        "task_id": task_id_arg or None,
        "count": len(items),
        "items": items,
    }
    if json_out:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"HANDOFF PACKETS: {len(items)}")
        for item in items:
            print(
                f"- {item['packet_id']} task={item['task_id']} "
                f"{item.get('from_role')}->{item.get('to_role')} "
                f"status={item.get('from_status') or 'n/a'}->{item.get('to_status')} "
                f"at={item.get('created_at')}"
            )
    con.close()
    raise SystemExit(0)

if cmd == "remediate":
    now = now_iso()

    where_overdue = ""
    where_terminal = ""
    where_blocked = ""
    where_orphaned = ""
    params_overdue: List[Any] = [now]
    params_terminal: List[Any] = []
    params_blocked: List[Any] = []
    params_orphaned: List[Any] = []

    if task_id_arg:
        where_overdue = " AND locked_by_task_id = ?"
        params_overdue.append(task_id_arg)
        where_terminal = " AND fl.locked_by_task_id = ?"
        params_terminal.append(task_id_arg)
        where_blocked = " AND w.task_id = ?"
        params_blocked.append(task_id_arg)
        where_orphaned = " AND w.task_id = ?"
        params_orphaned.append(task_id_arg)

    overdue_rows = cur.execute(
        f"""
SELECT lock_id, file_path, locked_by_task_id, lock_expires_at
FROM file_locks
WHERE lock_state = 'ACTIVE'
  AND lock_expires_at IS NOT NULL
  AND lock_expires_at <= ?
  {where_overdue}
ORDER BY lock_expires_at ASC
LIMIT ?
""",
        tuple(params_overdue + [limit]),
    ).fetchall()

    terminal_rows = cur.execute(
        f"""
SELECT fl.lock_id, fl.file_path, fl.locked_by_task_id, w.status
FROM file_locks fl
JOIN work_queue w ON w.task_id = fl.locked_by_task_id
WHERE fl.lock_state = 'ACTIVE'
  AND w.status IN ('DONE','FAILED','BLOCKED','ROLLED_BACK')
  {where_terminal}
ORDER BY fl.acquired_at ASC
LIMIT ?
""",
        tuple(params_terminal + [limit]),
    ).fetchall()

    blocked_rows = cur.execute(
        f"""
SELECT w.task_id, w.role_required, w.retry_count,
       GROUP_CONCAT(d.depends_on_task_id || ':' || COALESCE(dep.status, 'MISSING'), ' | ') AS blockers
FROM work_queue w
LEFT JOIN task_dependencies d ON d.task_id = w.task_id AND d.relation = 'blocks'
LEFT JOIN work_queue dep ON dep.task_id = d.depends_on_task_id
WHERE w.status = 'BLOCKED'
  {where_blocked}
GROUP BY w.task_id, w.role_required, w.retry_count
HAVING SUM(CASE WHEN COALESCE(dep.status, 'MISSING') <> 'DONE' THEN 1 ELSE 0 END) = 0
ORDER BY w.updated_at ASC
LIMIT ?
""",
        tuple(params_blocked + [limit]),
    ).fetchall()

    cutoff = (
        dt.datetime.fromtimestamp(clock_now_ts() - orphaned_running_min_sec, tz=dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    def load_orphaned_running_rows() -> List[Tuple[Any, ...]]:
        return cur.execute(
            f"""
SELECT w.task_id, w.assigned_agent, w.role_required, w.updated_at
FROM work_queue w
WHERE w.status = 'RUNNING'
  AND w.updated_at <= ?
  {where_orphaned}
  AND NOT EXISTS (
    SELECT 1 FROM file_locks fl
    WHERE fl.locked_by_task_id = w.task_id
      AND fl.lock_state = 'ACTIVE'
  )
ORDER BY w.updated_at ASC
LIMIT ?
""",
            tuple([cutoff] + params_orphaned + [limit]),
        ).fetchall()

    orphaned_rows = load_orphaned_running_rows()

    preview = {
        "overdue_active_locks": [
            {
                "lock_id": r[0],
                "file_path": r[1],
                "locked_by_task_id": r[2],
                "lock_expires_at": r[3],
            }
            for r in overdue_rows
        ],
        "terminal_task_active_locks": [
            {
                "lock_id": r[0],
                "file_path": r[1],
                "locked_by_task_id": r[2],
                "task_status": r[3],
            }
            for r in terminal_rows
        ],
        "blocked_tasks_with_resolved_dependencies": [
            {
                "task_id": r[0],
                "role_required": r[1],
                "retry_count": int(r[2] or 0),
                "blockers": [p.strip() for p in str(r[3] or "").split("|") if str(p).strip()],
            }
            for r in blocked_rows
        ],
        "orphaned_running_without_locks": [
            {
                "task_id": r[0],
                "assigned_agent": r[1],
                "role_required": r[2],
                "updated_at": r[3],
                "stale_cutoff": cutoff,
            }
            for r in orphaned_rows
        ],
    }

    requested = {
        "expire_overdue_locks": remediate_expire_overdue,
        "release_terminal_locks": remediate_release_terminal,
        "requeue_resolved_blocked": remediate_requeue_resolved,
        "requeue_orphaned_running": remediate_requeue_orphaned_running,
        "orphaned_running_min_sec": orphaned_running_min_sec,
        "apply": remediate_apply,
    }

    if not any([remediate_expire_overdue, remediate_release_terminal, remediate_requeue_resolved, remediate_requeue_orphaned_running]):
        requested["expire_overdue_locks"] = True
        requested["release_terminal_locks"] = True
        requested["requeue_resolved_blocked"] = True
        requested["requeue_orphaned_running"] = True

    applied = {
        "expired_locks": 0,
        "released_terminal_locks": 0,
        "requeued_tasks": 0,
        "requeued_orphaned_running": 0,
        "requeued_orphaned_running_post_lock_cleanup": 0,
        "orphaned_running_candidates_rechecked": 0,
        "transition_events": [],
    }

    if remediate_apply:
        cur.execute("BEGIN IMMEDIATE")
        if requested["expire_overdue_locks"]:
            cur.execute(
                f"""
UPDATE file_locks
SET lock_state = 'EXPIRED', released_at = ?
WHERE lock_state = 'ACTIVE'
  AND lock_expires_at IS NOT NULL
  AND lock_expires_at <= ?
  {where_overdue}
""",
                tuple([now, now] + (params_overdue[1:] if task_id_arg else [])),
            )
            applied["expired_locks"] = int(cur.rowcount or 0)

        if requested["release_terminal_locks"]:
            cur.execute(
                f"""
UPDATE file_locks
SET lock_state = 'RELEASED', released_at = ?
WHERE lock_state = 'ACTIVE'
  AND locked_by_task_id IN (
    SELECT task_id FROM work_queue
    WHERE status IN ('DONE','FAILED','BLOCKED','ROLLED_BACK')
    {"AND task_id = ?" if task_id_arg else ""}
  )
""",
                tuple([now] + ([task_id_arg] if task_id_arg else [])),
            )
            applied["released_terminal_locks"] = int(cur.rowcount or 0)

        if requested["requeue_resolved_blocked"]:
            for row in blocked_rows:
                task_id = str(row[0] or "").strip()
                if not task_id:
                    continue
                role_required = default_role_for_task(task_id)
                cur.execute(
                    "UPDATE work_queue SET status = 'QUEUED', role_required = ?, updated_at = ? WHERE task_id = ?",
                    (role_required, now, task_id),
                )
                if int(cur.rowcount or 0) != 1:
                    continue
                ev_id = transition_event_id(task_id, "BLOCKED", "QUEUED", now, "sre_watchdog", "remediate_requeue_resolved_blocked")
                blocker_ref = " | ".join([p.strip() for p in str(row[3] or "").split("|") if str(p).strip()])
                cur.execute(
                    """
INSERT OR IGNORE INTO task_transitions (
  event_id, task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at
) VALUES (?, ?, 'BLOCKED', 'QUEUED', 'sre_watchdog', 'remediate_requeue_resolved_blocked', ?, ?)
""",
                    (ev_id, task_id, blocker_ref or None, now),
                )
                applied["requeued_tasks"] += 1
                applied["transition_events"].append(ev_id)

        if requested["requeue_orphaned_running"]:
            preview_orphaned_task_ids = {
                str(row[0] or "").strip()
                for row in orphaned_rows
                if str(row[0] or "").strip()
            }
            orphaned_rows_apply = load_orphaned_running_rows()
            applied["orphaned_running_candidates_rechecked"] = int(len(orphaned_rows_apply))

            for row in orphaned_rows_apply:
                task_id = str(row[0] or "").strip()
                if not task_id:
                    continue
                role_required = default_role_for_task(task_id)
                cur.execute(
                    "UPDATE work_queue SET status = 'QUEUED', role_required = ?, assigned_agent = NULL, updated_at = ?, last_error_log = ? WHERE task_id = ? AND status = 'RUNNING'",
                    (
                        role_required,
                        now,
                        f"orphaned_running_auto_requeue:{now}",
                        task_id,
                    ),
                )
                if int(cur.rowcount or 0) != 1:
                    continue
                ev_id = transition_event_id(task_id, "RUNNING", "QUEUED", now, "sre_watchdog", "orphaned_running_auto_requeue")
                cur.execute(
                    """
INSERT OR IGNORE INTO task_transitions (
  event_id, task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at
) VALUES (?, ?, 'RUNNING', 'QUEUED', 'sre_watchdog', 'orphaned_running_auto_requeue', ?, ?)
""",
                    (ev_id, task_id, f"stale_cutoff={cutoff}", now),
                )
                applied["requeued_orphaned_running"] += 1
                if task_id not in preview_orphaned_task_ids:
                    applied["requeued_orphaned_running_post_lock_cleanup"] += 1
                applied["transition_events"].append(ev_id)

        con.commit()

    recommended = []
    if preview["overdue_active_locks"] or preview["terminal_task_active_locks"] or preview["blocked_tasks_with_resolved_dependencies"] or preview["orphaned_running_without_locks"]:
        cmd_parts = [
            "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/queue_arbitrator.sh remediate",
            "--expire-overdue-locks",
            "--release-terminal-locks",
            "--requeue-resolved-blocked",
            "--requeue-orphaned-running",
            f"--orphaned-running-min-sec {orphaned_running_min_sec}",
            "--apply",
            "--json",
        ]
        if task_id_arg:
            cmd_parts.insert(1, f"--task-id {task_id_arg}")
        recommended.append(" ".join(cmd_parts))

    out = {
        "ok": True,
        "command": cmd,
        "task_id": task_id_arg or None,
        "dry_run": not remediate_apply,
        "requested": requested,
        "preview": preview,
        "applied": applied,
        "recommended_commands": recommended,
    }
    if json_out:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(
            f"REMEDIATE dry_run={out['dry_run']} overdue={len(preview['overdue_active_locks'])} "
            f"terminal_locks={len(preview['terminal_task_active_locks'])} "
            f"resolved_blocked={len(preview['blocked_tasks_with_resolved_dependencies'])} "
            f"orphaned_running={len(preview['orphaned_running_without_locks'])}"
        )
        if recommended:
            print("- recommended:")
            for cmd_line in recommended:
                print(f"  - {cmd_line}")
    con.close()
    raise SystemExit(0)

if cmd == "locks":
    now = now_iso()
    cur.execute("BEGIN IMMEDIATE")
    expired = gc_expired(cur, now)
    con.commit()

    lock_where = "WHERE lock_state = 'ACTIVE'" if active_only else ""
    rows = cur.execute(
        f"""
SELECT lock_id, file_path, lock_mode, lock_state, locked_by_task_id, acquired_at, lock_expires_at, released_at
FROM file_locks
{lock_where}
ORDER BY acquired_at DESC
LIMIT ?
""",
        (limit,),
    ).fetchall()

    payload = [
        {
            "lock_id": r[0],
            "file_path": r[1],
            "lock_mode": r[2],
            "lock_state": r[3],
            "locked_by_task_id": r[4],
            "acquired_at": r[5],
            "lock_expires_at": r[6],
            "released_at": r[7],
        }
        for r in rows
    ]
    out = {"ok": True, "command": cmd, "active_only": active_only, "expired_released": expired, "items": payload}
    if json_out:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"LOCKS: {len(payload)} active_only={active_only} (expired_marked={expired})")
        for r in payload:
            print(f"- {r['file_path']} state={r['lock_state']} task={r['locked_by_task_id']} expires={r['lock_expires_at'] or 'none'}")
    con.close()
    raise SystemExit(0)

con.close()
print(json.dumps({"ok": False, "error": "unknown_command", "command": cmd}, ensure_ascii=False, indent=2))
raise SystemExit(2)
PY
