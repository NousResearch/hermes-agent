#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
DB_PATH="${OPENCLAW_CONTINUITY_DB_PATH:-$ROOT/state/continuity/continuity_os.sqlite}"
JSON_OUT=0
STRICT=0

usage() {
  cat <<'EOF'
Usage: db_integrity_check.sh [options]

Low-cost continuity DB integrity + queue invariants checker.

Options:
  --db <path>        Continuity sqlite path (default: state/continuity/continuity_os.sqlite)
  --json             JSON output
  --strict           Exit non-zero when critical checks fail
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)
      DB_PATH="${2:-}"; shift 2 ;;
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

OPENCLAW_CONTINUITY_DB_PATH="$DB_PATH" "$ROOT/ops/openclaw/continuity/init_db.sh" >/dev/null

python3 - "$DB_PATH" "$JSON_OUT" "$STRICT" "$ROOT" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sqlite3
import sys
from typing import Any, Dict, List

db_path = str(sys.argv[1] or "").strip()
json_out = bool(int(sys.argv[2]))
strict = bool(int(sys.argv[3]))
root = pathlib.Path(str(sys.argv[4] or "/home/yeqiuqiu/clawd-architect")).resolve()

if str(root / "src") not in sys.path:
    sys.path.insert(0, str(root / "src"))
if str(root / "ops" / "openclaw" / "continuity") not in sys.path:
    sys.path.insert(0, str(root / "ops" / "openclaw" / "continuity"))

try:
    from fixed_now import now_iso_utc as _helper_now_iso_utc, now_ts as _helper_now_ts
except Exception:  # pragma: no cover
    _helper_now_iso_utc = None
    _helper_now_ts = None


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


def clock_now_ts() -> int:
    if _helper_now_ts is not None:
        try:
            return int(_helper_now_ts())
        except Exception:
            pass
    return int(dt.datetime.now(dt.timezone.utc).timestamp())


def clock_now_dt() -> dt.datetime:
    return dt.datetime.fromtimestamp(clock_now_ts(), tz=dt.timezone.utc)


def clock_now_iso() -> str:
    if _helper_now_iso_utc is not None:
        try:
            return str(_helper_now_iso_utc())
        except Exception:
            pass
    return dt.datetime.fromtimestamp(clock_now_ts(), tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def now_iso() -> str:
    return clock_now_iso()


def q1(cur: sqlite3.Cursor, query: str, params=()):
    row = cur.execute(query, params).fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


def parse_iso(value: str):
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(raw)
    except Exception:
        return None


def column_exists(cur: sqlite3.Cursor, table: str, column: str) -> bool:
    rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(r[1] or "") == column for r in rows)


def table_exists(cur: sqlite3.Cursor, table: str) -> bool:
    row = cur.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ).fetchone()
    return bool(int(row[0] or 0)) if row else False


ALLOWED_ROLES = {"planner", "executor", "validator", "sre_watchdog", "librarian", "outer_gate"}
recent_handoff_cutoff_iso = (
    clock_now_dt() - dt.timedelta(days=7)
).replace(microsecond=0).isoformat().replace("+00:00", "Z")


con = sqlite3.connect(db_path)
cur = con.cursor()

checks: List[Dict[str, Any]] = []

# 1) SQLite internal integrity
integrity_rows = [str(r[0]) for r in cur.execute("PRAGMA integrity_check").fetchall()]
integrity_ok = len(integrity_rows) == 1 and integrity_rows[0].lower() == "ok"
checks.append(
    {
        "name": "sqlite_integrity_check",
        "severity": "critical" if not integrity_ok else "info",
        "ok": integrity_ok,
        "details": integrity_rows,
    }
)

# 2) Foreign key consistency
fk_rows = cur.execute("PRAGMA foreign_key_check").fetchall()
fk_ok = len(fk_rows) == 0
checks.append(
    {
        "name": "foreign_key_check",
        "severity": "critical" if not fk_ok else "info",
        "ok": fk_ok,
        "rows": len(fk_rows),
    }
)

# 3) Active lock uniqueness (defense-in-depth despite partial unique index)
active_lock_conflicts = q1(
    cur,
    """
SELECT COUNT(*)
FROM (
  SELECT file_path
  FROM file_locks
  WHERE lock_state = 'ACTIVE'
  GROUP BY file_path
  HAVING COUNT(*) > 1
)
""",
)
checks.append(
    {
        "name": "active_lock_uniqueness",
        "severity": "critical" if active_lock_conflicts > 0 else "info",
        "ok": active_lock_conflicts == 0,
        "conflict_paths": active_lock_conflicts,
    }
)

# 4) Active locks on terminal tasks (should normally be 0)
terminal_lock_count = q1(
    cur,
    """
SELECT COUNT(*)
FROM file_locks fl
LEFT JOIN work_queue w ON w.task_id = fl.locked_by_task_id
WHERE fl.lock_state = 'ACTIVE'
  AND COALESCE(w.status, 'MISSING') IN ('DONE','FAILED','BLOCKED','ROLLED_BACK')
""",
)
checks.append(
    {
        "name": "terminal_task_active_locks",
        "severity": "warn" if terminal_lock_count > 0 else "info",
        "ok": terminal_lock_count == 0,
        "rows": terminal_lock_count,
    }
)

# 5) Dependency graph sanity: no self loops
self_loops = q1(
    cur,
    "SELECT COUNT(*) FROM task_dependencies WHERE task_id = depends_on_task_id",
)
checks.append(
    {
        "name": "task_dependency_self_loop",
        "severity": "critical" if self_loops > 0 else "info",
        "ok": self_loops == 0,
        "rows": self_loops,
    }
)

# 6) Dependency graph sanity: direct two-node cycle count
mutual_cycles = q1(
    cur,
    """
SELECT COUNT(*)
FROM task_dependencies d1
JOIN task_dependencies d2
  ON d1.task_id = d2.depends_on_task_id
 AND d1.depends_on_task_id = d2.task_id
WHERE d1.task_id < d1.depends_on_task_id
""",
)
checks.append(
    {
        "name": "task_dependency_mutual_cycle",
        "severity": "critical" if mutual_cycles > 0 else "info",
        "ok": mutual_cycles == 0,
        "rows": mutual_cycles,
    }
)

# 7) Queue flow sanity: RUNNING tasks should have at least one RUNNING transition in history.
running_without_transition = q1(
    cur,
    """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status = 'RUNNING'
  AND NOT EXISTS (
    SELECT 1
    FROM task_transitions t
    WHERE t.task_id = w.task_id
      AND t.to_status = 'RUNNING'
  )
""",
)
checks.append(
    {
        "name": "running_task_transition_trace",
        "severity": "warn" if running_without_transition > 0 else "info",
        "ok": running_without_transition == 0,
        "rows": running_without_transition,
    }
)

# 8) Event source normalization hygiene.
invalid_event_sources = q1(
    cur,
    """
SELECT COUNT(*)
FROM continuity_events
WHERE source NOT GLOB 'continuity.*'
  AND source NOT GLOB 'watchdog.*'
  AND source NOT GLOB 'runtime.*'
  AND source NOT GLOB 'local.*'
""",
)
checks.append(
    {
        "name": "continuity_event_source_namespace",
        "severity": "warn" if invalid_event_sources > 0 else "info",
        "ok": invalid_event_sources == 0,
        "rows": invalid_event_sources,
        "allowed_prefixes": ["continuity.", "watchdog.", "runtime.", "local."],
    }
)

# 9) Provenance contract: source/task_id namespace alignment.
source_prefix_mismatch = q1(
    cur,
    """
SELECT COUNT(*)
FROM work_queue
WHERE (source = 'autopilot' AND task_id NOT LIKE 'autopilot:%')
   OR (source = 'competitive_parity' AND task_id NOT LIKE 'parity:%')
   OR (source = 'continuity_ops' AND task_id NOT LIKE 'continuity:%')
""",
)
checks.append(
    {
        "name": "work_queue_source_task_namespace_contract",
        "severity": "critical" if source_prefix_mismatch > 0 else "info",
        "ok": source_prefix_mismatch == 0,
        "rows": source_prefix_mismatch,
        "rules": {
            "autopilot": "task_id LIKE autopilot:%",
            "competitive_parity": "task_id LIKE parity:%",
            "continuity_ops": "task_id LIKE continuity:%",
        },
    }
)

# 10) Role contract metadata exists and is normalized.
role_required_col_present = column_exists(cur, "work_queue", "role_required")
checks.append(
    {
        "name": "work_queue_role_required_column_present",
        "severity": "critical" if not role_required_col_present else "info",
        "ok": role_required_col_present,
    }
)

if role_required_col_present:
    invalid_role_required = q1(
        cur,
        """
SELECT COUNT(*)
FROM work_queue
WHERE role_required IS NOT NULL
  AND TRIM(role_required) <> ''
  AND LOWER(TRIM(role_required)) NOT IN ('planner','executor','validator','sre_watchdog','librarian','outer_gate')
""",
    )
else:
    invalid_role_required = 0

checks.append(
    {
        "name": "work_queue_role_required_enum",
        "severity": "critical" if invalid_role_required > 0 else "info",
        "ok": invalid_role_required == 0,
        "rows": invalid_role_required,
        "allowed_roles": sorted(ALLOWED_ROLES),
    }
)

if role_required_col_present:
    active_role_missing = q1(
        cur,
        """
SELECT COUNT(*)
FROM work_queue
WHERE status IN ('QUEUED','RUNNING','REVIEW')
  AND (role_required IS NULL OR TRIM(role_required) = '')
""",
    )
else:
    active_role_missing = 0

checks.append(
    {
        "name": "active_tasks_role_required_present",
        "severity": "critical" if active_role_missing > 0 else "info",
        "ok": active_role_missing == 0,
        "rows": active_role_missing,
    }
)

if role_required_col_present:
    review_role_mismatch = q1(
        cur,
        """
SELECT COUNT(*)
FROM work_queue
WHERE status = 'REVIEW'
  AND LOWER(TRIM(COALESCE(role_required, ''))) <> 'validator'
""",
    )
else:
    review_role_mismatch = 0

checks.append(
    {
        "name": "review_status_requires_validator_role",
        "severity": "warn" if review_role_mismatch > 0 else "info",
        "ok": review_role_mismatch == 0,
        "rows": review_role_mismatch,
    }
)

if role_required_col_present:
    running_role_mismatch = q1(
        cur,
        """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status = 'RUNNING'
  AND EXISTS (
    SELECT 1
    FROM task_transitions t
    WHERE t.task_id = w.task_id
      AND t.to_status = 'RUNNING'
    ORDER BY t.created_at DESC
    LIMIT 1
  )
  AND LOWER(TRIM(COALESCE(w.role_required, ''))) <> LOWER(TRIM(COALESCE((
    SELECT t.actor_role
    FROM task_transitions t
    WHERE t.task_id = w.task_id
      AND t.to_status = 'RUNNING'
    ORDER BY t.created_at DESC
    LIMIT 1
  ), '')))
""",
    )
else:
    running_role_mismatch = 0

checks.append(
    {
        "name": "running_role_matches_latest_running_transition_actor",
        "severity": "warn" if running_role_mismatch > 0 else "info",
        "ok": running_role_mismatch == 0,
        "rows": running_role_mismatch,
    }
)

invalid_transition_roles = q1(
    cur,
    """
SELECT COUNT(*)
FROM task_transitions
WHERE LOWER(TRIM(COALESCE(actor_role, ''))) NOT IN ('planner','executor','validator','sre_watchdog','librarian','outer_gate')
""",
)
checks.append(
    {
        "name": "task_transition_actor_role_enum",
        "severity": "warn" if invalid_transition_roles > 0 else "info",
        "ok": invalid_transition_roles == 0,
        "rows": invalid_transition_roles,
        "allowed_roles": sorted(ALLOWED_ROLES),
    }
)

handoff_table_present = table_exists(cur, "task_handoff_packets")
checks.append(
    {
        "name": "task_handoff_packets_table_present",
        "severity": "critical" if not handoff_table_present else "info",
        "ok": handoff_table_present,
    }
)

if handoff_table_present:
    required_handoff_cols = [
        "packet_id",
        "task_id",
        "parent_task_id",
        "transition_event_id",
        "from_role",
        "to_role",
        "created_at",
        "evidence_refs_json",
        "gate_metadata_json",
        "next_gate",
    ]
    missing_handoff_cols = [c for c in required_handoff_cols if not column_exists(cur, "task_handoff_packets", c)]
else:
    missing_handoff_cols = required_handoff_cols = []

checks.append(
    {
        "name": "task_handoff_packets_required_columns",
        "severity": "critical" if missing_handoff_cols else "info",
        "ok": len(missing_handoff_cols) == 0,
        "missing": missing_handoff_cols,
        "required": required_handoff_cols,
    }
)

if handoff_table_present and not missing_handoff_cols:
    invalid_handoff_roles = q1(
        cur,
        """
SELECT COUNT(*)
FROM task_handoff_packets
WHERE LOWER(TRIM(COALESCE(from_role, ''))) NOT IN ('planner','executor','validator','sre_watchdog','librarian','outer_gate')
   OR LOWER(TRIM(COALESCE(to_role, ''))) NOT IN ('planner','executor','validator','sre_watchdog','librarian','outer_gate')
""",
    )

    handoff_missing_required = q1(
        cur,
        """
SELECT COUNT(*)
FROM task_handoff_packets
WHERE TRIM(COALESCE(task_id, '')) = ''
   OR TRIM(COALESCE(from_role, '')) = ''
   OR TRIM(COALESCE(to_role, '')) = ''
   OR TRIM(COALESCE(created_at, '')) = ''
   OR TRIM(COALESCE(next_gate, '')) = ''
   OR TRIM(COALESCE(evidence_refs_json, '')) = ''
   OR TRIM(COALESCE(gate_metadata_json, '')) = ''
""",
    )

    handoff_tracking_row = cur.execute(
        "SELECT MIN(created_at) FROM task_handoff_packets"
    ).fetchone()
    handoff_tracking_start = str((handoff_tracking_row[0] if handoff_tracking_row else "") or "").strip()
    if not handoff_tracking_start:
        handoff_tracking_start = recent_handoff_cutoff_iso

    recent_missing_handoff = q1(
        cur,
        """
SELECT COUNT(*)
FROM task_transitions t
WHERE t.created_at >= ?
  AND t.to_status IN ('REVIEW','DONE','BLOCKED','FAILED','ROLLED_BACK')
  AND EXISTS (
    SELECT 1
    FROM work_queue w
    WHERE w.task_id = t.task_id
      AND w.source IN ('autopilot','competitive_parity','continuity_ops')
  )
  AND NOT EXISTS (
    SELECT 1
    FROM task_handoff_packets hp
    WHERE hp.transition_event_id = t.event_id
  )
""",
        (handoff_tracking_start,),
    )
else:
    invalid_handoff_roles = 0
    handoff_missing_required = 0
    recent_missing_handoff = 0
    handoff_tracking_start = recent_handoff_cutoff_iso

checks.append(
    {
        "name": "task_handoff_role_enum",
        "severity": "warn" if invalid_handoff_roles > 0 else "info",
        "ok": invalid_handoff_roles == 0,
        "rows": invalid_handoff_roles,
        "allowed_roles": sorted(ALLOWED_ROLES),
    }
)

checks.append(
    {
        "name": "task_handoff_required_fields_present",
        "severity": "warn" if handoff_missing_required > 0 else "info",
        "ok": handoff_missing_required == 0,
        "rows": handoff_missing_required,
    }
)

checks.append(
    {
        "name": "recent_role_transition_handoff_linked",
        "severity": "warn" if recent_missing_handoff > 0 else "info",
        "ok": recent_missing_handoff == 0,
        "rows": recent_missing_handoff,
        "window_days": 7,
        "tracking_start": handoff_tracking_start,
    }
)

provider_summary_rows_checked = 0
provider_summary_rows_invalid = 0
provider_summary_issue_sample: List[Dict[str, Any]] = []
if handoff_table_present and not missing_handoff_cols:
    for row in cur.execute(
        """
SELECT packet_id, gate_metadata_json
FROM task_handoff_packets
WHERE gate_metadata_json IS NOT NULL
  AND TRIM(gate_metadata_json) <> ''
ORDER BY created_at DESC
LIMIT 2000
"""
    ).fetchall():
        packet_id = str(row[0] or "")
        raw_meta = str(row[1] or "").strip()
        if not raw_meta:
            continue
        try:
            gate_meta = json.loads(raw_meta)
        except Exception:
            continue
        if not isinstance(gate_meta, dict):
            continue
        summary = gate_meta.get("gate_summary")
        if not isinstance(summary, dict):
            continue
        if str(summary.get("schema_version") or "") != PROVIDER_FAILURE_SUMMARY_SCHEMA_VERSION:
            continue

        provider_summary_rows_checked += 1
        verdict = validate_provider_failure_summary(summary, strict=True)
        if bool(verdict.get("ok") is True):
            continue

        provider_summary_rows_invalid += 1
        if len(provider_summary_issue_sample) < 5:
            issues = verdict.get("issues") if isinstance(verdict.get("issues"), list) else []
            provider_summary_issue_sample.append(
                {
                    "packet_id": packet_id,
                    "issues": [str(item) for item in issues][:4],
                }
            )

checks.append(
    {
        "name": "handoff_provider_failure_summary_schema_valid",
        "severity": "critical" if provider_summary_rows_invalid > 0 else "info",
        "ok": provider_summary_rows_invalid == 0,
        "rows": provider_summary_rows_invalid,
        "checked_rows": provider_summary_rows_checked,
        "schema_version": PROVIDER_FAILURE_SUMMARY_SCHEMA_VERSION,
        "samples": provider_summary_issue_sample,
        "recommended_commands": (
            [
                "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/repair_provider_failure_summaries.sh --db /home/yeqiuqiu/clawd-architect/state/continuity/continuity_os.sqlite --json",
                "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/repair_provider_failure_summaries.sh --db /home/yeqiuqiu/clawd-architect/state/continuity/continuity_os.sqlite --apply --json",
            ]
            if provider_summary_rows_invalid > 0
            else []
        ),
    }
)

# 11) DONE tasks should include validator-owned DONE transition.
done_without_validator = q1(
    cur,
    """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status = 'DONE'
  AND w.source IN ('autopilot', 'competitive_parity')
  AND NOT EXISTS (
    SELECT 1
    FROM task_transitions t
    WHERE t.task_id = w.task_id
      AND t.to_status = 'DONE'
      AND t.actor_role = 'validator'
  )
""",
)
checks.append(
    {
        "name": "done_transition_validator_owned",
        "severity": "warn" if done_without_validator > 0 else "info",
        "ok": done_without_validator == 0,
        "rows": done_without_validator,
    }
)

# 11) Terminal tasks should retain evidence refs for terminal transition replay.
terminal_without_evidence = q1(
    cur,
    """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status IN ('DONE','BLOCKED','FAILED','ROLLED_BACK')
  AND NOT EXISTS (
    SELECT 1
    FROM task_transitions t
    WHERE t.task_id = w.task_id
      AND t.to_status = w.status
      AND t.evidence_ref IS NOT NULL
      AND TRIM(t.evidence_ref) <> ''
  )
""",
)
checks.append(
    {
        "name": "terminal_transition_evidence_present",
        "severity": "warn" if terminal_without_evidence > 0 else "info",
        "ok": terminal_without_evidence == 0,
        "rows": terminal_without_evidence,
    }
)

# 12) Non-queued tasks should have at least one artifact trace row.
nonqueued_without_artifacts = q1(
    cur,
    """
SELECT COUNT(*)
FROM work_queue w
WHERE w.status <> 'QUEUED'
  AND NOT EXISTS (
    SELECT 1 FROM task_artifacts a WHERE a.task_id = w.task_id
  )
""",
)
checks.append(
    {
        "name": "nonqueued_task_artifact_trace",
        "severity": "warn" if nonqueued_without_artifacts > 0 else "info",
        "ok": nonqueued_without_artifacts == 0,
        "rows": nonqueued_without_artifacts,
    }
)

# 13) Competitive parity weekly loop presence/freshness (warn-only).
parity_task_present = q1(
    cur,
    "SELECT COUNT(*) FROM work_queue WHERE task_id = 'parity:weekly_harness'",
)
parity_row = cur.execute(
    """
SELECT t.created_at
FROM task_transitions t
WHERE t.task_id = 'parity:weekly_harness'
  AND t.to_status = 'DONE'
ORDER BY t.created_at DESC
LIMIT 1
""",
).fetchone()
parity_last_done_at = str(parity_row[0] or "").strip() if parity_row else ""
parity_last_done_age_sec = None
if parity_last_done_at:
    parsed = parse_iso(parity_last_done_at)
    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        parity_last_done_age_sec = int((clock_now_dt() - parsed).total_seconds())

parity_freshness_limit_sec = 9 * 24 * 3600
parity_freshness_ok = parity_task_present > 0 and parity_last_done_age_sec is not None and parity_last_done_age_sec <= parity_freshness_limit_sec
checks.append(
    {
        "name": "parity_weekly_harness_freshness",
        "severity": "warn" if not parity_freshness_ok else "info",
        "ok": parity_freshness_ok,
        "task_present": parity_task_present > 0,
        "last_done_at": parity_last_done_at or None,
        "last_done_age_sec": parity_last_done_age_sec,
        "freshness_limit_sec": parity_freshness_limit_sec,
    }
)

if role_required_col_present:
    parity_role_mismatch = q1(
        cur,
        """
SELECT COUNT(*)
FROM work_queue
WHERE task_id = 'parity:weekly_harness'
  AND LOWER(TRIM(COALESCE(role_required, ''))) NOT IN ('validator','librarian','sre_watchdog')
""",
    )
else:
    parity_role_mismatch = 0

checks.append(
    {
        "name": "parity_role_required_contract",
        "severity": "warn" if parity_role_mismatch > 0 else "info",
        "ok": parity_role_mismatch == 0,
        "rows": parity_role_mismatch,
    }
)

# 14) GTC v2 connector substrate integrity (warn/critical mix).
gtc_tables = [
    "gtc_connector",
    "gtc_evidence_index",
    "gtc_latest_pointer",
    "gtc_artifact",
    "gtc_evidence_artifact",
    "gtc_task_evidence",
    "gtc_checkpoint_evidence",
]
missing_gtc_tables = [t for t in gtc_tables if not table_exists(cur, t)]
checks.append(
    {
        "name": "gtc_tables_present",
        "severity": "critical" if missing_gtc_tables else "info",
        "ok": len(missing_gtc_tables) == 0,
        "missing": missing_gtc_tables,
        "required": gtc_tables,
    }
)

if not missing_gtc_tables:
    gtc_seq_conflicts = q1(
        cur,
        """
SELECT COUNT(*)
FROM (
  SELECT connector_id, monotonic_seq
  FROM gtc_evidence_index
  GROUP BY connector_id, monotonic_seq
  HAVING COUNT(*) > 1
)
""",
    )

    gtc_pointer_orphans = q1(
        cur,
        """
SELECT COUNT(*)
FROM gtc_latest_pointer p
LEFT JOIN gtc_evidence_index e ON e.evidence_id = p.evidence_id
WHERE e.evidence_id IS NULL
""",
    )

    gtc_pointer_connector_orphans = q1(
        cur,
        """
SELECT COUNT(*)
FROM gtc_latest_pointer p
LEFT JOIN gtc_connector c ON c.connector_id = p.connector_id
WHERE c.connector_id IS NULL
""",
    )

    gtc_evidence_without_connector = q1(
        cur,
        """
SELECT COUNT(*)
FROM gtc_evidence_index e
LEFT JOIN gtc_connector c ON c.connector_id = e.connector_id
WHERE c.connector_id IS NULL
""",
    )

    gtc_latest_count = q1(cur, "SELECT COUNT(*) FROM gtc_latest_pointer")
    gtc_connector_count = q1(cur, "SELECT COUNT(*) FROM gtc_connector")

    checks.append(
        {
            "name": "gtc_monotonic_seq_uniqueness",
            "severity": "critical" if gtc_seq_conflicts > 0 else "info",
            "ok": gtc_seq_conflicts == 0,
            "rows": gtc_seq_conflicts,
        }
    )

    checks.append(
        {
            "name": "gtc_latest_pointer_evidence_fk",
            "severity": "warn" if gtc_pointer_orphans > 0 else "info",
            "ok": gtc_pointer_orphans == 0,
            "rows": gtc_pointer_orphans,
        }
    )

    checks.append(
        {
            "name": "gtc_latest_pointer_connector_fk",
            "severity": "warn" if gtc_pointer_connector_orphans > 0 else "info",
            "ok": gtc_pointer_connector_orphans == 0,
            "rows": gtc_pointer_connector_orphans,
        }
    )

    checks.append(
        {
            "name": "gtc_evidence_connector_fk",
            "severity": "critical" if gtc_evidence_without_connector > 0 else "info",
            "ok": gtc_evidence_without_connector == 0,
            "rows": gtc_evidence_without_connector,
        }
    )

    checks.append(
        {
            "name": "gtc_latest_pointer_coverage",
            "severity": "warn" if (gtc_connector_count > 0 and gtc_latest_count == 0) else "info",
            "ok": not (gtc_connector_count > 0 and gtc_latest_count == 0),
            "connector_count": gtc_connector_count,
            "latest_pointer_count": gtc_latest_count,
        }
    )

# 15) Lock-break audit envelope integrity.
lock_break_table_present = table_exists(cur, "lock_break_audit")
checks.append(
    {
        "name": "lock_break_audit_table_present",
        "severity": "warn" if not lock_break_table_present else "info",
        "ok": lock_break_table_present,
    }
)

if lock_break_table_present:
    malformed_audit_rows = 0
    released_count_mismatch = 0
    missing_transition_refs = 0

    rows = cur.execute(
        """
SELECT audit_id, released_lock_ids_json, released_lock_count, transition_event_ids_json
FROM lock_break_audit
"""
    ).fetchall()

    for r in rows:
        try:
            released_ids = json.loads(str(r[1] or "[]"))
            transition_ids = json.loads(str(r[3] or "[]"))
            if not isinstance(released_ids, list):
                released_ids = []
            if not isinstance(transition_ids, list):
                transition_ids = []
        except Exception:
            malformed_audit_rows += 1
            continue

        if int(r[2] or 0) != len(released_ids):
            released_count_mismatch += 1

        for ev_id in transition_ids:
            ev = cur.execute("SELECT 1 FROM task_transitions WHERE event_id = ?", (str(ev_id),)).fetchone()
            if ev is None:
                missing_transition_refs += 1

    checks.append(
        {
            "name": "lock_break_audit_json_parseable",
            "severity": "warn" if malformed_audit_rows > 0 else "info",
            "ok": malformed_audit_rows == 0,
            "rows": malformed_audit_rows,
        }
    )

    checks.append(
        {
            "name": "lock_break_released_count_matches_ids",
            "severity": "warn" if released_count_mismatch > 0 else "info",
            "ok": released_count_mismatch == 0,
            "rows": released_count_mismatch,
        }
    )

    checks.append(
        {
            "name": "lock_break_transition_refs_exist",
            "severity": "warn" if missing_transition_refs > 0 else "info",
            "ok": missing_transition_refs == 0,
            "rows": missing_transition_refs,
        }
    )

# 16) Web-capture scheduler governed state artifact integrity.
db_path_obj = pathlib.Path(db_path).resolve()
if len(db_path_obj.parents) >= 3:
    inferred_root = db_path_obj.parents[2]
else:
    inferred_root = db_path_obj.parent

scheduler_state_override = str(os.environ.get("OPENCLAW_WEB_CAPTURE_SCHEDULER_STATE_PATH") or "").strip()
if scheduler_state_override:
    scheduler_state_path = pathlib.Path(scheduler_state_override).resolve()
else:
    scheduler_state_path = inferred_root / "state" / "continuity" / "latest" / "web_capture_scheduler_state.json"

try:
    scheduler_freshness_limit_sec = max(0, int(os.environ.get("OPENCLAW_WEB_CAPTURE_SCHEDULER_MAX_AGE_SEC", "21600")))
except Exception:
    scheduler_freshness_limit_sec = 21600

scheduler_state_exists = scheduler_state_path.exists()
checks.append(
    {
        "name": "web_capture_scheduler_state_present",
        "severity": "critical" if not scheduler_state_exists else "info",
        "ok": scheduler_state_exists,
        "path": str(scheduler_state_path),
    }
)

scheduler_obj: Dict[str, Any] = {}
if scheduler_state_exists:
    try:
        scheduler_obj = json.loads(scheduler_state_path.read_text(encoding="utf-8"))
        scheduler_parse_ok = isinstance(scheduler_obj, dict)
    except Exception:
        scheduler_obj = {}
        scheduler_parse_ok = False

    checks.append(
        {
            "name": "web_capture_scheduler_state_parseable",
            "severity": "critical" if not scheduler_parse_ok else "info",
            "ok": scheduler_parse_ok,
        }
    )

    if scheduler_parse_ok:
        scheduler_schema_ok = str(scheduler_obj.get("schema_version") or "") == "openclaw.web_capture.scheduler_state.v1"
        checks.append(
            {
                "name": "web_capture_scheduler_schema_version",
                "severity": "critical" if not scheduler_schema_ok else "info",
                "ok": scheduler_schema_ok,
                "value": scheduler_obj.get("schema_version"),
            }
        )

        scheduler_contract = scheduler_obj.get("contract") if isinstance(scheduler_obj.get("contract"), dict) else {}
        scheduler_contract_ok = bool(scheduler_contract.get("state_valid"))
        scheduler_contract_errors = scheduler_contract.get("validation_errors") if isinstance(scheduler_contract.get("validation_errors"), list) else []
        checks.append(
            {
                "name": "web_capture_scheduler_contract_state_valid",
                "severity": "critical" if not scheduler_contract_ok else "info",
                "ok": scheduler_contract_ok,
                "validation_errors": scheduler_contract_errors,
            }
        )

        scheduler_updated_at = str(scheduler_obj.get("updated_at") or "").strip()
        scheduler_updated_dt = parse_iso(scheduler_updated_at)
        scheduler_updated_valid = scheduler_updated_dt is not None
        scheduler_age_sec = None
        scheduler_fresh = None
        if scheduler_updated_dt is not None:
            if scheduler_updated_dt.tzinfo is None:
                scheduler_updated_dt = scheduler_updated_dt.replace(tzinfo=dt.timezone.utc)
            scheduler_age_sec = max(0, int((clock_now_dt() - scheduler_updated_dt).total_seconds()))
            scheduler_fresh = scheduler_age_sec <= scheduler_freshness_limit_sec

        checks.append(
            {
                "name": "web_capture_scheduler_updated_at_parseable",
                "severity": "warn" if not scheduler_updated_valid else "info",
                "ok": scheduler_updated_valid,
                "value": scheduler_updated_at or None,
            }
        )

        checks.append(
            {
                "name": "web_capture_scheduler_freshness",
                "severity": "warn" if scheduler_fresh is False else "info",
                "ok": scheduler_fresh is not False,
                "age_sec": scheduler_age_sec,
                "freshness_limit_sec": scheduler_freshness_limit_sec,
            }
        )

        scheduler_summary = scheduler_obj.get("summary") if isinstance(scheduler_obj.get("summary"), dict) else {}
        summary_total = scheduler_summary.get("total_macros") if isinstance(scheduler_summary.get("total_macros"), int) else None
        summary_eligible = scheduler_summary.get("eligible_macros") if isinstance(scheduler_summary.get("eligible_macros"), int) else None
        macros_len = len(scheduler_obj.get("macros")) if isinstance(scheduler_obj.get("macros"), list) else None
        summary_consistent = (
            summary_total is not None
            and macros_len is not None
            and summary_total == macros_len
            and summary_eligible is not None
            and summary_eligible <= summary_total
        )
        checks.append(
            {
                "name": "web_capture_scheduler_summary_consistency",
                "severity": "warn" if not summary_consistent else "info",
                "ok": summary_consistent,
                "summary_total_macros": summary_total,
                "summary_eligible_macros": summary_eligible,
                "macros_len": macros_len,
            }
        )

critical_failures = [c for c in checks if c.get("severity") == "critical" and not c.get("ok")]
warn_failures = [c for c in checks if c.get("severity") == "warn" and not c.get("ok")]

summary = {
    "ok": len(critical_failures) == 0,
    "generated_at": now_iso(),
    "db_path": db_path,
    "check_count": len(checks),
    "critical_failures": len(critical_failures),
    "warn_failures": len(warn_failures),
    "checks": checks,
}

if json_out:
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print("CONTINUITY DB CHECK")
    print(f"- ok: {summary['ok']}")
    print(f"- critical_failures: {summary['critical_failures']}")
    print(f"- warn_failures: {summary['warn_failures']}")
    for item in checks:
        status = "PASS" if item.get("ok") else "FAIL"
        print(f"- [{status}] {item.get('name')} severity={item.get('severity')}")

con.close()

if strict and not summary["ok"]:
    raise SystemExit(1)
PY
