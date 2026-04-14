#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
DB_PATH="${OPENCLAW_CONTINUITY_DB_PATH:-$ROOT/state/continuity/continuity_os.sqlite}"
TASK_ID=""
JSON_OUT=0
STRICT=0
WRITE_ARTIFACTS=1

usage() {
  cat <<'EOF'
Usage: queue_replay_verify.sh [options]

Deterministically replay task_transitions journal and verify projected queue state.

Options:
  --task-id <id>       Verify only one task_id
  --db <path>          Continuity sqlite path override
  --strict             Exit non-zero on active replay/role mismatches (legacy mismatches remain warn-only)
  --no-write           Do not persist projection/report artifacts
  --json               JSON output
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-id)
      TASK_ID="${2:-}"; shift 2 ;;
    --db)
      DB_PATH="${2:-}"; shift 2 ;;
    --strict)
      STRICT=1; shift ;;
    --no-write)
      WRITE_ARTIFACTS=0; shift ;;
    --json)
      JSON_OUT=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

OPENCLAW_CONTINUITY_DB_PATH="$DB_PATH" "$ROOT/ops/openclaw/continuity/init_db.sh" >/dev/null

python3 - "$ROOT" "$DB_PATH" "$TASK_ID" "$JSON_OUT" "$STRICT" "$WRITE_ARTIFACTS" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import sqlite3
import sys
from typing import Any, Dict, List, Optional

root = pathlib.Path(sys.argv[1]).resolve()
db_path = str(sys.argv[2] or "").strip()
task_id_filter = str(sys.argv[3] or "").strip()
json_out = bool(int(sys.argv[4]))
strict = bool(int(sys.argv[5]))
write_artifacts = bool(int(sys.argv[6]))

sys.path.insert(0, str((root / "ops" / "openclaw" / "continuity").resolve()))
try:
    from fixed_now import now_iso_utc as _helper_now_iso_utc
except Exception:  # pragma: no cover
    _helper_now_iso_utc = None

latest_dir = root / "state" / "continuity" / "latest"
projection_path = latest_dir / "queue_replay_projection.json"
report_path = latest_dir / "queue_replay_verify.json"


def now_iso() -> str:
    if _helper_now_iso_utc is not None:
        try:
            return str(_helper_now_iso_utc())
        except Exception:
            pass
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def normalize_status(raw: Optional[str]) -> Optional[str]:
    val = str(raw or "").strip().upper()
    return val or None


ACTIVE_STATUSES = {"QUEUED", "RUNNING", "REVIEW", "BLOCKED"}
TERMINAL_STATUSES = {"DONE", "FAILED", "ROLLED_BACK"}

HISTORICAL_STALE_REQUEUE_REASON_ALLOWLIST = {
    "autopilot_state:queued",
    "replay_projection_reconcile_s3a",
    "s4a_tokenized_reconcile_status_to_projection",
}
HISTORICAL_STALE_REQUEUE_CUTOFF_UTC = dt.datetime(2026, 3, 15, tzinfo=dt.timezone.utc)


def parse_utc_iso(value: Optional[str]) -> Optional[dt.datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def classify_discontinuity(
    expected_from: Optional[str],
    journal_from: Optional[str],
    to_status: str,
    *,
    reason: Optional[str],
    created_at: Optional[str],
) -> Dict[str, str]:
    expected = normalize_status(expected_from)
    journal = normalize_status(journal_from)
    to_norm = normalize_status(to_status) or "QUEUED"

    if not expected or not journal:
        return {"bucket": "hard", "reason": "from_status_projection_divergence"}

    # Idempotent replay rows where the transition lands back on the projected state.
    if to_norm == expected:
        return {"bucket": "soft", "reason": "idempotent_transition_to_projected_status"}

    # Historical reopen pattern from wrappers that re-claimed from QUEUED while
    # the journal omitted the preceding terminal->QUEUED reset row.
    if journal == "QUEUED" and to_norm == "RUNNING" and expected in TERMINAL_STATUSES:
        return {"bucket": "historical", "reason": "terminal_reopen_without_reset_transition"}

    # Historical residue from pre-S5-A autopilot multi-writer rows where a stale
    # REVIEW->QUEUED emit raced after RUNNING projection.
    created_dt = parse_utc_iso(created_at)
    reason_norm = str(reason or "").strip().lower()
    if (
        journal == "REVIEW"
        and expected == "RUNNING"
        and to_norm == "QUEUED"
        and reason_norm in HISTORICAL_STALE_REQUEUE_REASON_ALLOWLIST
        and created_dt is not None
        and created_dt < HISTORICAL_STALE_REQUEUE_CUTOFF_UTC
    ):
        return {"bucket": "historical", "reason": "stale_review_requeue_pre_s5a"}

    return {"bucket": "hard", "reason": "from_status_projection_divergence"}


def classify_status_mismatch(queue_status: str, projected_status: str) -> Dict[str, str]:
    queue_norm = normalize_status(queue_status) or "QUEUED"
    projected_norm = normalize_status(projected_status) or "QUEUED"

    # Legacy/historical pattern: queue is terminal but replay projection lands in
    # an active state due to stale historical transitions.
    if queue_norm in TERMINAL_STATUSES and projected_norm in ACTIVE_STATUSES:
        return {
            "bucket": "legacy",
            "reason": "terminal_queue_status_with_active_projection",
        }

    # Everything else is treated as an active integrity fault and remains fail-close.
    return {
        "bucket": "active",
        "reason": "active_queue_replay_integrity_fault",
    }


def expected_role(task_id: str, source: str, status: str) -> str:
    t = task_id
    s = normalize_status(status) or "QUEUED"
    src = str(source or "").strip().lower()

    if t == "autopilot:cycle":
        return "outer_gate"

    if t == "autopilot:quality_gate":
        if s in {"RUNNING", "REVIEW"}:
            return "validator"
        if s == "DONE":
            return "librarian"
        if s in {"BLOCKED", "FAILED", "ROLLED_BACK"}:
            return "sre_watchdog"
        return "validator"

    if t == "autopilot:apply_fixes":
        if s == "RUNNING":
            return "executor"
        if s == "REVIEW":
            return "validator"
        if s == "DONE":
            return "librarian"
        if s in {"BLOCKED", "FAILED", "ROLLED_BACK"}:
            return "sre_watchdog"
        return "executor"

    if t.startswith("autopilot:") or src == "autopilot":
        if s == "RUNNING":
            return "planner"
        if s == "REVIEW":
            return "validator"
        if s == "DONE":
            return "librarian"
        if s in {"BLOCKED", "FAILED", "ROLLED_BACK"}:
            return "sre_watchdog"
        return "planner"

    if t == "parity:weekly_harness":
        if s == "DONE":
            return "librarian"
        if s in {"BLOCKED", "FAILED", "ROLLED_BACK"}:
            return "sre_watchdog"
        return "validator"

    if t == "continuity:normalize_event_sources":
        if s == "DONE":
            return "librarian"
        if s in {"BLOCKED", "FAILED", "ROLLED_BACK"}:
            return "sre_watchdog"
        return "sre_watchdog"

    if t.startswith("continuity:web_capture:"):
        if s == "DONE":
            return "librarian"
        if s in {"BLOCKED", "FAILED", "ROLLED_BACK"}:
            return "sre_watchdog"
        return "validator"

    if s == "REVIEW":
        return "validator"
    if s == "DONE":
        return "librarian"
    if s in {"BLOCKED", "FAILED", "ROLLED_BACK"}:
        return "sre_watchdog"
    if s == "RUNNING":
        return "executor"
    return "planner"


con = sqlite3.connect(db_path)
cur = con.cursor()

queue_where = ""
params: List[Any] = []
if task_id_filter:
    queue_where = "WHERE task_id = ?"
    params.append(task_id_filter)

queue_rows = cur.execute(
    f"""
SELECT task_id, source, status, role_required, retry_count, updated_at
FROM work_queue
{queue_where}
ORDER BY task_id ASC
""",
    tuple(params),
).fetchall()

projection_rows: List[Dict[str, Any]] = []
status_mismatch: List[Dict[str, Any]] = []
active_status_mismatch: List[Dict[str, Any]] = []
legacy_status_mismatch: List[Dict[str, Any]] = []
role_mismatch: List[Dict[str, Any]] = []
discontinuities: List[Dict[str, Any]] = []
soft_discontinuities: List[Dict[str, Any]] = []
historical_discontinuities: List[Dict[str, Any]] = []

for row in queue_rows:
    task_id, source, queue_status_raw, role_required_raw, retry_count, updated_at = row
    queue_status = normalize_status(queue_status_raw) or "QUEUED"
    role_required = str(role_required_raw or "").strip().lower() or None

    transitions = cur.execute(
        """
SELECT event_id, from_status, to_status, actor_role, reason, created_at
FROM task_transitions
WHERE task_id = ?
ORDER BY created_at ASC, rowid ASC
""",
        (task_id,),
    ).fetchall()

    projected_status: Optional[str] = None
    transition_discontinuities: List[Dict[str, Any]] = []
    transition_soft_discontinuities: List[Dict[str, Any]] = []
    transition_historical_discontinuities: List[Dict[str, Any]] = []

    for ev in transitions:
        event_id, from_status_raw, to_status_raw, actor_role, reason, created_at = ev
        from_status = normalize_status(from_status_raw)
        to_status = normalize_status(to_status_raw) or "QUEUED"

        if projected_status is None and from_status:
            projected_status = from_status

        if projected_status is not None and from_status and from_status != projected_status:
            row = {
                "event_id": event_id,
                "expected_from_status": projected_status,
                "journal_from_status": from_status,
                "to_status": to_status,
                "actor_role": actor_role,
                "reason": reason,
                "created_at": created_at,
            }
            classification = classify_discontinuity(
                projected_status,
                from_status,
                to_status,
                reason=reason,
                created_at=created_at,
            )
            row["classification_bucket"] = classification["bucket"]
            row["classification_reason"] = classification["reason"]
            if classification["bucket"] == "soft":
                transition_soft_discontinuities.append(row)
            elif classification["bucket"] == "historical":
                transition_historical_discontinuities.append(row)
            else:
                transition_discontinuities.append(row)

        projected_status = to_status

    if projected_status is None:
        projected_status = queue_status

    expected = expected_role(task_id, str(source or ""), projected_status)
    role_ok = role_required == expected
    status_ok = projected_status == queue_status

    row_obj = {
        "task_id": task_id,
        "source": source,
        "queue_status": queue_status,
        "projected_status": projected_status,
        "status_match": status_ok,
        "role_required": role_required,
        "expected_role_required": expected,
        "role_match": role_ok,
        "retry_count": int(retry_count or 0),
        "updated_at": updated_at,
        "transition_count": len(transitions),
        "discontinuity_count": len(transition_discontinuities),
        "soft_discontinuity_count": len(transition_soft_discontinuities),
        "historical_discontinuity_count": len(transition_historical_discontinuities),
    }
    projection_rows.append(row_obj)

    if not status_ok:
        mismatch_row = {
            "task_id": task_id,
            "queue_status": queue_status,
            "projected_status": projected_status,
            "transition_count": len(transitions),
        }
        mismatch_class = classify_status_mismatch(queue_status, projected_status)
        mismatch_row.update(
            {
                "mismatch_bucket": mismatch_class["bucket"],
                "mismatch_reason": mismatch_class["reason"],
            }
        )
        status_mismatch.append(mismatch_row)
        if mismatch_class["bucket"] == "legacy":
            legacy_status_mismatch.append(dict(mismatch_row))
        else:
            active_status_mismatch.append(dict(mismatch_row))
    if not role_ok:
        role_mismatch.append(
            {
                "task_id": task_id,
                "queue_status": queue_status,
                "role_required": role_required,
                "expected_role_required": expected,
            }
        )
    if transition_discontinuities:
        discontinuities.append(
            {
                "task_id": task_id,
                "rows": transition_discontinuities,
            }
        )
    if transition_soft_discontinuities:
        soft_discontinuities.append(
            {
                "task_id": task_id,
                "rows": transition_soft_discontinuities,
            }
        )
    if transition_historical_discontinuities:
        historical_discontinuities.append(
            {
                "task_id": task_id,
                "rows": transition_historical_discontinuities,
            }
        )

projection_rows.sort(key=lambda r: r["task_id"])

projection_digest_seed = json.dumps(
    [{"task_id": r["task_id"], "projected_status": r["projected_status"], "expected_role_required": r["expected_role_required"]} for r in projection_rows],
    ensure_ascii=False,
    sort_keys=True,
)
projection_digest = hashlib.sha256(projection_digest_seed.encode("utf-8")).hexdigest()

status = "pass"
if active_status_mismatch or role_mismatch:
    status = "fail"
elif legacy_status_mismatch or discontinuities:
    status = "warn"

summary = {
    "task_count": len(projection_rows),
    "status_mismatch_count": len(status_mismatch),
    "active_status_mismatch_count": len(active_status_mismatch),
    "legacy_status_mismatch_count": len(legacy_status_mismatch),
    "role_mismatch_count": len(role_mismatch),
    "discontinuity_task_count": len(discontinuities),
    "soft_discontinuity_task_count": len(soft_discontinuities),
    "historical_discontinuity_task_count": len(historical_discontinuities),
    "status": status,
    "projection_digest": projection_digest,
}

projection_payload = {
    "schema": "clawd.queue_replay_projection.v1",
    "generated_at": now_iso(),
    "workspace_id": "clawd-architect",
    "summary": summary,
    "tasks": projection_rows,
}

report_payload = {
    "schema": "clawd.queue_replay_verify.v1",
    "generated_at": projection_payload["generated_at"],
    "workspace_id": "clawd-architect",
    "task_id_filter": task_id_filter or None,
    "summary": summary,
    "status_mismatches": status_mismatch,
    "active_status_mismatches": active_status_mismatch,
    "legacy_status_mismatches": legacy_status_mismatch,
    "role_mismatches": role_mismatch,
    "discontinuities": discontinuities,
    "soft_discontinuities": soft_discontinuities,
    "historical_discontinuities": historical_discontinuities,
    "projection_ref": str(projection_path.relative_to(root)),
}

if write_artifacts:
    atomic_write(projection_path, projection_payload)
    atomic_write(report_path, report_payload)

if json_out:
    print(json.dumps(report_payload, ensure_ascii=False, indent=2))
else:
    print(
        "QUEUE REPLAY VERIFY "
        f"status={summary['status']} tasks={summary['task_count']} "
        f"status_mismatch={summary['status_mismatch_count']} "
        f"active_status_mismatch={summary['active_status_mismatch_count']} "
        f"legacy_status_mismatch={summary['legacy_status_mismatch_count']} "
        f"role_mismatch={summary['role_mismatch_count']} "
        f"discontinuity_tasks={summary['discontinuity_task_count']} "
        f"soft_discontinuity_tasks={summary['soft_discontinuity_task_count']} "
        f"historical_discontinuity_tasks={summary['historical_discontinuity_task_count']}"
    )

con.close()

if strict and summary["status"] == "fail":
    raise SystemExit(1)
PY
