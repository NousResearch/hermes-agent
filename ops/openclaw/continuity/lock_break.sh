#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
DB_PATH="${OPENCLAW_CONTINUITY_DB_PATH:-$ROOT/state/continuity/continuity_os.sqlite}"
TASK_ID=""
FILE_PATH=""
LOCK_ID=""
REASON=""
OPERATOR=""
TRUTH_ANCHOR=""
JSON_OUT=0
APPLY=0
REQUEUE_RUNNING=1
MAX_LOCKS=25
FORCE=0
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"
MUTATION_TICKET=""
declare -a MUTATION_ATTESTATIONS=()
declare -a MUTATION_ATTESTATION_OBJECTS=()

usage() {
  cat <<'EOF'
Usage: lock_break.sh [options]

Operator-only audited lock break workflow.

Default mode is dry-run preview. --apply performs lock release + audit write.

Selectors (at least one required):
  --task-id <id>        Filter ACTIVE locks by owning task
  --file-path <path>    Filter ACTIVE locks by file path
  --lock-id <id>        Filter ACTIVE lock by lock_id

Required:
  --reason <text>       Why lock-break is required
  --operator <name>     Operator identity for audit envelope
  --action-token <tok>  Canonical mutation token from continuity_current .action_token
  --truth-anchor <tok>  Legacy alias of --action-token (anchor-only requires explicit override)
  --mutation-ticket <value>
                        Authority ticket JSON string, @path, or path (required for apply token path)
  --attestation <name>  Satisfied authority attestation name (repeatable)
  --attestation-object <value>
                        Structured authority attestation JSON string, @path, or path (repeatable)

Execution options:
  --apply               Apply lock break (default: dry-run)
  --no-requeue-running  Do not requeue RUNNING tasks after lock release
  --max-locks <n>       Safety cap for apply path (default: 25)
  --force               Allow apply when candidate lock count > max-locks
  --allow-legacy-anchor Allow anchor-only token mode (break-glass)
                        (also set OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY=1)
  --db <path>           Continuity DB path override
  --json                JSON output
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-id)
      TASK_ID="${2:-}"; shift 2 ;;
    --file-path)
      FILE_PATH="${2:-}"; shift 2 ;;
    --lock-id)
      LOCK_ID="${2:-}"; shift 2 ;;
    --reason)
      REASON="${2:-}"; shift 2 ;;
    --operator)
      OPERATOR="${2:-}"; shift 2 ;;
    --action-token|--truth-anchor)
      TRUTH_ANCHOR="${2:-}"; shift 2 ;;
    --mutation-ticket)
      MUTATION_TICKET="${2:-}"; shift 2 ;;
    --attestation)
      MUTATION_ATTESTATIONS+=("${2:-}"); shift 2 ;;
    --attestation-object)
      MUTATION_ATTESTATION_OBJECTS+=("${2:-}"); shift 2 ;;
    --apply)
      APPLY=1; shift ;;
    --no-requeue-running)
      REQUEUE_RUNNING=0; shift ;;
    --max-locks)
      MAX_LOCKS="${2:-}"; shift 2 ;;
    --force)
      FORCE=1; shift ;;
    --allow-legacy-anchor)
      ALLOW_LEGACY_ANCHOR=1; shift ;;
    --db)
      DB_PATH="${2:-}"; shift 2 ;;
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

if [[ -z "$TASK_ID" && -z "$FILE_PATH" && -z "$LOCK_ID" ]]; then
  echo "lock-break requires at least one selector (--task-id|--file-path|--lock-id)" >&2
  exit 2
fi
if [[ -z "$REASON" ]]; then
  echo "missing --reason" >&2
  exit 2
fi
if [[ -z "$OPERATOR" ]]; then
  echo "missing --operator" >&2
  exit 2
fi
if ! [[ "$MAX_LOCKS" =~ ^[0-9]+$ ]]; then
  echo "invalid --max-locks: $MAX_LOCKS" >&2
  exit 2
fi

mutation_risk_tier="medium"
mutation_operation="lock_break:preview"
if [[ "$APPLY" == "1" ]]; then
  mutation_risk_tier="high"
  mutation_operation="lock_break:apply"
fi

guard_args=(
  --script "lock_break.sh"
  --risk-tier "$mutation_risk_tier"
  --mutation-operation "$mutation_operation"
)
if [[ -n "$TRUTH_ANCHOR" ]]; then
  guard_args+=(--action-token "$TRUTH_ANCHOR")
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

OPENCLAW_CONTINUITY_DB_PATH="$DB_PATH" "$ROOT/ops/openclaw/continuity/init_db.sh" >/dev/null

python3 - "$ROOT" "$DB_PATH" "$TASK_ID" "$FILE_PATH" "$LOCK_ID" "$REASON" "$OPERATOR" "$TRUTH_ANCHOR" "$APPLY" "$JSON_OUT" "$REQUEUE_RUNNING" "$MAX_LOCKS" "$FORCE" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import sqlite3
import sys
import tempfile
from typing import Any, Dict, List, Optional

root = pathlib.Path(sys.argv[1]).resolve()
db_path = str(sys.argv[2] or "").strip()
task_id = str(sys.argv[3] or "").strip()
file_path = str(sys.argv[4] or "").strip()
lock_id = str(sys.argv[5] or "").strip()
reason = str(sys.argv[6] or "").strip()
operator = str(sys.argv[7] or "").strip()
action_token = str(sys.argv[8] or "").strip()
apply = bool(int(sys.argv[9]))
json_out = bool(int(sys.argv[10]))
requeue_running = bool(int(sys.argv[11]))
max_locks = max(1, int(sys.argv[12] or 25))
force = bool(int(sys.argv[13]))

latest_dir = root / "state" / "continuity" / "latest"
audit_dir = root / "state" / "continuity" / "lock_break"
current_path = root / "state" / "continuity" / "current.json"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def transition_event_id(task: str, from_status: Optional[str], to_status: str, created_at: str, why: str) -> str:
    seed = f"{task}|{from_status or ''}|{to_status}|{created_at}|{why}"
    return "tr_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


truth_anchor = {}
if current_path.exists():
    try:
        truth_anchor = (json.loads(current_path.read_text(encoding="utf-8")) or {}).get("truth_anchor") or {}
    except Exception:
        truth_anchor = {}

con = sqlite3.connect(db_path)
con.row_factory = sqlite3.Row
cur = con.cursor()

where_parts = ["lock_state = 'ACTIVE'"]
params: List[Any] = []
if task_id:
    where_parts.append("locked_by_task_id = ?")
    params.append(task_id)
if file_path:
    where_parts.append("file_path = ?")
    params.append(file_path)
if lock_id:
    where_parts.append("lock_id = ?")
    params.append(lock_id)

rows = cur.execute(
    f"""
SELECT lock_id, file_path, lock_mode, locked_by_task_id, acquired_at, lock_expires_at
FROM file_locks
WHERE {' AND '.join(where_parts)}
ORDER BY acquired_at ASC, lock_id ASC
""",
    tuple(params),
).fetchall()

candidates = [
    {
        "lock_id": str(r["lock_id"]),
        "file_path": str(r["file_path"]),
        "lock_mode": str(r["lock_mode"]),
        "locked_by_task_id": str(r["locked_by_task_id"]),
        "acquired_at": str(r["acquired_at"]),
        "lock_expires_at": str(r["lock_expires_at"] or ""),
    }
    for r in rows
]

preview = {
    "candidate_lock_count": len(candidates),
    "candidates": candidates,
}

if not candidates:
    payload = {
        "ok": False,
        "error": "no_active_locks_matched",
        "dry_run": not apply,
        "selector": {"task_id": task_id or None, "file_path": file_path or None, "lock_id": lock_id or None},
        "preview": preview,
    }
    if json_out:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("LOCK BREAK: no active locks matched")
    raise SystemExit(1)

if apply and len(candidates) > max_locks and not force:
    payload = {
        "ok": False,
        "error": "safety_cap_exceeded",
        "candidate_lock_count": len(candidates),
        "max_locks": max_locks,
        "hint": "rerun with --force if this lock-break is intentional",
    }
    if json_out:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"LOCK BREAK BLOCKED: candidates={len(candidates)} exceeds max_locks={max_locks}")
    raise SystemExit(1)

if not apply:
    payload = {
        "ok": True,
        "dry_run": True,
        "selector": {"task_id": task_id or None, "file_path": file_path or None, "lock_id": lock_id or None},
        "reason": reason,
        "operator": operator,
        "action_token": action_token,
        "truth_anchor_token": action_token,
        "preview": preview,
        "next_command": "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/lock_break.sh --apply ...",
    }
    if json_out:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"LOCK BREAK PREVIEW: candidates={len(candidates)} operator={operator}")
    raise SystemExit(0)

now = now_iso()
released_lock_ids: List[str] = []
requeued_task_ids: List[str] = []
transition_events: List[str] = []

audit_seed = {
    "now": now,
    "operator": operator,
    "reason": reason,
    "truth_anchor": truth_anchor,
    "action_token": action_token,
    "truth_anchor_token": action_token,
    "selector": {"task_id": task_id, "file_path": file_path, "lock_id": lock_id},
    "lock_ids": [c["lock_id"] for c in candidates],
}
audit_id = "lba_" + hashlib.sha256(json.dumps(audit_seed, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:24]

try:
    cur.execute("BEGIN")

    for c in candidates:
        cur.execute(
            """
UPDATE file_locks
SET lock_state = 'RELEASED', released_at = ?
WHERE lock_id = ? AND lock_state = 'ACTIVE'
""",
            (now, c["lock_id"]),
        )
        if int(cur.rowcount or 0) > 0:
            released_lock_ids.append(c["lock_id"])

    impacted_tasks = sorted({c["locked_by_task_id"] for c in candidates})

    if requeue_running and impacted_tasks:
        for task in impacted_tasks:
            row = cur.execute(
                "SELECT status FROM work_queue WHERE task_id = ?",
                (task,),
            ).fetchone()
            prev_status = str(row[0] or "") if row else ""
            if prev_status != "RUNNING":
                continue
            cur.execute(
                """
UPDATE work_queue
SET status = 'QUEUED', assigned_agent = NULL, updated_at = ?
WHERE task_id = ?
""",
                (now, task),
            )
            ev_reason = f"lock_break:{audit_id}"
            ev_id = transition_event_id(task, "RUNNING", "QUEUED", now, ev_reason)
            cur.execute(
                """
INSERT OR IGNORE INTO task_transitions (
  event_id, task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at
) VALUES (?, ?, 'RUNNING', 'QUEUED', 'sre_watchdog', ?, ?, ?)
""",
                (ev_id, task, ev_reason, f"state/continuity/lock_break/{audit_id}.json", now),
            )
            requeued_task_ids.append(task)
            transition_events.append(ev_id)

    cur.execute(
        """
INSERT INTO lock_break_audit (
  audit_id, created_at, operator_id, reason,
  selector_json, truth_anchor_json, truth_anchor_token,
  released_lock_ids_json, released_lock_count,
  requeued_task_ids_json, transition_event_ids_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            audit_id,
            now,
            operator,
            reason,
            json.dumps({"task_id": task_id or None, "file_path": file_path or None, "lock_id": lock_id or None}, ensure_ascii=False, sort_keys=True),
            json.dumps(truth_anchor, ensure_ascii=False, sort_keys=True),
            action_token,
            json.dumps(released_lock_ids, ensure_ascii=False),
            len(released_lock_ids),
            json.dumps(requeued_task_ids, ensure_ascii=False),
            json.dumps(transition_events, ensure_ascii=False),
        ),
    )

    con.commit()
except Exception:
    con.rollback()
    raise
finally:
    con.close()

payload = {
    "schema": "clawd.queue_lock_break_audit.v1",
    "generated_at": now,
    "workspace_id": "clawd-architect",
    "ok": True,
    "dry_run": False,
    "audit_id": audit_id,
    "operator": operator,
    "reason": reason,
    "selector": {"task_id": task_id or None, "file_path": file_path or None, "lock_id": lock_id or None},
    "truth_anchor": truth_anchor,
    "action_token": action_token,
    "truth_anchor_token": action_token,
    "released_lock_count": len(released_lock_ids),
    "released_lock_ids": released_lock_ids,
    "requeued_task_count": len(requeued_task_ids),
    "requeued_task_ids": requeued_task_ids,
    "transition_events": transition_events,
}

atomic_write(audit_dir / f"{audit_id}.json", payload)
atomic_write(latest_dir / "lock_break_last.json", payload)

if json_out:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
else:
    print(
        f"LOCK BREAK APPLIED audit_id={audit_id} released={len(released_lock_ids)} "
        f"requeued={len(requeued_task_ids)}"
    )
PY
