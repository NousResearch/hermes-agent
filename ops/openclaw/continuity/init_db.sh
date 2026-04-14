#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
DB_PATH="${OPENCLAW_CONTINUITY_DB_PATH:-$ROOT/state/continuity/continuity_os.sqlite}"

mkdir -p "$(dirname "$DB_PATH")"
mkdir -p "$ROOT/state/continuity/checkpoints" "$ROOT/state/continuity/latest" "$ROOT/state/continuity/snapshots"

python3 - "$DB_PATH" <<'PY'
import pathlib
import sqlite3
import sys

db_path = pathlib.Path(sys.argv[1])
db_path.parent.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(db_path)
cur = con.cursor()

cur.executescript(
    """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS checkpoints (
  checkpoint_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  trigger TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('READY','PROGRESS','BLOCKER')),
  objective TEXT NOT NULL,
  parent_checkpoint_id TEXT,
  json_path TEXT NOT NULL,
  md_path TEXT NOT NULL,
  snapshot_path TEXT NOT NULL,
  repo_branch TEXT,
  repo_head TEXT,
  integrity_sha256 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS work_queue (
  task_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  title TEXT NOT NULL,
  acceptance_criteria TEXT,
  status TEXT NOT NULL CHECK(status IN ('QUEUED','RUNNING','REVIEW','BLOCKED','DONE','FAILED','ROLLED_BACK')),
  role_required TEXT,
  assigned_agent TEXT,
  retry_count INTEGER NOT NULL DEFAULT 0,
  max_retries INTEGER NOT NULL DEFAULT 3,
  last_error_log TEXT,
  cooldown_until TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_work_queue_status ON work_queue(status);
CREATE INDEX IF NOT EXISTS idx_work_queue_agent ON work_queue(assigned_agent);

CREATE TABLE IF NOT EXISTS task_transitions (
  event_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  from_status TEXT,
  to_status TEXT NOT NULL,
  actor_role TEXT NOT NULL,
  reason TEXT,
  evidence_ref TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES work_queue(task_id)
);

CREATE TABLE IF NOT EXISTS task_handoff_packets (
  packet_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  parent_task_id TEXT,
  transition_event_id TEXT,
  from_role TEXT NOT NULL,
  to_role TEXT NOT NULL,
  from_status TEXT,
  to_status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  evidence_refs_json TEXT,
  gate_metadata_json TEXT,
  task_linkage_json TEXT,
  lock_refs_json TEXT,
  next_gate TEXT NOT NULL,
  budget_tokens_used INTEGER NOT NULL DEFAULT 0,
  model_tier TEXT NOT NULL DEFAULT 'unknown',
  retry_count INTEGER,
  failure_signature TEXT,
  FOREIGN KEY(task_id) REFERENCES work_queue(task_id) ON DELETE CASCADE,
  FOREIGN KEY(parent_task_id) REFERENCES work_queue(task_id) ON DELETE SET NULL,
  FOREIGN KEY(transition_event_id) REFERENCES task_transitions(event_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_handoff_task_created ON task_handoff_packets(task_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_handoff_transition ON task_handoff_packets(transition_event_id);

CREATE TABLE IF NOT EXISTS task_dependencies (
  task_id TEXT NOT NULL,
  depends_on_task_id TEXT NOT NULL,
  relation TEXT NOT NULL DEFAULT 'blocks' CHECK(relation IN ('blocks','soft')),
  created_at TEXT NOT NULL,
  PRIMARY KEY(task_id, depends_on_task_id),
  FOREIGN KEY(task_id) REFERENCES work_queue(task_id) ON DELETE CASCADE,
  FOREIGN KEY(depends_on_task_id) REFERENCES work_queue(task_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_task_dependencies_depends ON task_dependencies(depends_on_task_id);

CREATE TABLE IF NOT EXISTS task_file_targets (
  task_id TEXT NOT NULL,
  file_path TEXT NOT NULL,
  lock_mode TEXT NOT NULL DEFAULT 'exclusive' CHECK(lock_mode IN ('exclusive','shared')),
  created_at TEXT NOT NULL,
  PRIMARY KEY(task_id, file_path),
  FOREIGN KEY(task_id) REFERENCES work_queue(task_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_task_file_targets_path ON task_file_targets(file_path);

CREATE TABLE IF NOT EXISTS file_locks (
  lock_id TEXT PRIMARY KEY,
  file_path TEXT NOT NULL,
  lock_mode TEXT NOT NULL CHECK(lock_mode IN ('exclusive','shared')),
  lock_state TEXT NOT NULL CHECK(lock_state IN ('ACTIVE','RELEASED','EXPIRED')),
  locked_by_task_id TEXT NOT NULL,
  lock_reason TEXT,
  acquired_at TEXT NOT NULL,
  lock_expires_at TEXT,
  released_at TEXT,
  FOREIGN KEY(locked_by_task_id) REFERENCES work_queue(task_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_file_locks_path_state ON file_locks(file_path, lock_state);
CREATE INDEX IF NOT EXISTS idx_file_locks_task_state ON file_locks(locked_by_task_id, lock_state);
CREATE UNIQUE INDEX IF NOT EXISTS idx_file_locks_active_path_unique
ON file_locks(file_path)
WHERE lock_state = 'ACTIVE';

CREATE TABLE IF NOT EXISTS task_artifacts (
  artifact_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  artifact_type TEXT NOT NULL,
  artifact_path TEXT NOT NULL,
  sha256 TEXT,
  metadata_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES work_queue(task_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_task_artifacts_task_type ON task_artifacts(task_id, artifact_type);

CREATE TABLE IF NOT EXISTS lock_break_audit (
  audit_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  operator_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  selector_json TEXT NOT NULL,
  truth_anchor_json TEXT NOT NULL,
  truth_anchor_token TEXT NOT NULL,
  released_lock_ids_json TEXT NOT NULL,
  released_lock_count INTEGER NOT NULL DEFAULT 0,
  requeued_task_ids_json TEXT NOT NULL,
  transition_event_ids_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lock_break_audit_created ON lock_break_audit(created_at DESC);

CREATE TABLE IF NOT EXISTS cooldown_locks (
  lock_id TEXT PRIMARY KEY,
  scope_type TEXT NOT NULL,
  scope_key TEXT NOT NULL,
  reason TEXT NOT NULL,
  lock_until TEXT NOT NULL,
  source_checkpoint_id TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cooldown_scope ON cooldown_locks(scope_type, scope_key);

CREATE TABLE IF NOT EXISTS rollback_events (
  rollback_id TEXT PRIMARY KEY,
  boundary_level TEXT NOT NULL,
  trigger_reason TEXT NOT NULL,
  command_executed TEXT NOT NULL,
  result_status TEXT NOT NULL,
  checkpoint_id TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS continuity_events (
  event_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  source TEXT NOT NULL,
  event_key TEXT NOT NULL,
  severity TEXT NOT NULL CHECK(severity IN ('info','warn','critical')),
  fingerprint TEXT NOT NULL,
  emitted INTEGER NOT NULL,
  changed INTEGER NOT NULL,
  cooldown_elapsed INTEGER NOT NULL,
  suppress_reason TEXT,
  summary TEXT,
  evidence_ref TEXT,
  route_key TEXT NOT NULL,
  state_file TEXT,
  metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_continuity_events_route_created ON continuity_events(route_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_continuity_events_key_created ON continuity_events(event_key, created_at DESC);

CREATE TABLE IF NOT EXISTS gtc_connector (
  connector_id TEXT PRIMARY KEY,
  connector_type TEXT NOT NULL,
  display_name TEXT NOT NULL,
  freshness_ttl_ms INTEGER NOT NULL DEFAULT 60000,
  stale_severity TEXT NOT NULL DEFAULT 'warning',
  config_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_monotonic_seq INTEGER NOT NULL DEFAULT 0,
  UNIQUE(connector_type, connector_id)
);
CREATE INDEX IF NOT EXISTS idx_gtc_connector_type ON gtc_connector(connector_type);

CREATE TABLE IF NOT EXISTS gtc_evidence_index (
  evidence_id TEXT PRIMARY KEY,
  connector_id TEXT NOT NULL REFERENCES gtc_connector(connector_id),
  connector_type TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  monotonic_seq INTEGER NOT NULL,
  subject_kind TEXT NOT NULL,
  subject_id TEXT NOT NULL,
  severity_max TEXT,
  jsonl_path TEXT NOT NULL,
  jsonl_line_no INTEGER NOT NULL,
  payload_sha256 TEXT NOT NULL,
  facts_json TEXT NOT NULL,
  refs_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gtc_evidence_subject ON gtc_evidence_index(subject_kind, subject_id, monotonic_seq);
CREATE INDEX IF NOT EXISTS idx_gtc_evidence_time ON gtc_evidence_index(observed_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_gtc_evidence_connector_seq ON gtc_evidence_index(connector_id, monotonic_seq);

CREATE TABLE IF NOT EXISTS gtc_artifact (
  sha256 TEXT PRIMARY KEY,
  media_type TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  path TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gtc_evidence_artifact (
  evidence_id TEXT NOT NULL REFERENCES gtc_evidence_index(evidence_id),
  sha256 TEXT NOT NULL REFERENCES gtc_artifact(sha256),
  role TEXT NOT NULL,
  PRIMARY KEY (evidence_id, sha256, role)
);

CREATE TABLE IF NOT EXISTS gtc_task_evidence (
  task_id TEXT NOT NULL,
  evidence_id TEXT NOT NULL REFERENCES gtc_evidence_index(evidence_id),
  PRIMARY KEY (task_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS gtc_checkpoint_evidence (
  checkpoint_id TEXT NOT NULL,
  evidence_id TEXT NOT NULL REFERENCES gtc_evidence_index(evidence_id),
  PRIMARY KEY (checkpoint_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS gtc_latest_pointer (
  pointer_key TEXT PRIMARY KEY,
  connector_id TEXT NOT NULL REFERENCES gtc_connector(connector_id),
  evidence_id TEXT NOT NULL REFERENCES gtc_evidence_index(evidence_id),
  observed_at TEXT NOT NULL,
  freshness_ttl_ms INTEGER NOT NULL,
  updated_at TEXT NOT NULL
);
"""
)

cols = {str(r[1]) for r in cur.execute("PRAGMA table_info(work_queue)").fetchall()}
if "role_required" not in cols:
    cur.execute("ALTER TABLE work_queue ADD COLUMN role_required TEXT")

cur.execute("CREATE INDEX IF NOT EXISTS idx_work_queue_role_required ON work_queue(role_required)")

cur.execute(
    """
UPDATE work_queue
SET role_required = CASE
  WHEN task_id = 'autopilot:cycle' THEN 'outer_gate'
  WHEN task_id = 'autopilot:quality_gate' THEN CASE
    WHEN status = 'RUNNING' THEN 'validator'
    WHEN status = 'REVIEW' THEN 'validator'
    WHEN status = 'DONE' THEN 'librarian'
    WHEN status IN ('BLOCKED','FAILED','ROLLED_BACK') THEN 'sre_watchdog'
    ELSE 'validator'
  END
  WHEN task_id = 'autopilot:apply_fixes' THEN CASE
    WHEN status = 'RUNNING' THEN 'executor'
    WHEN status = 'REVIEW' THEN 'validator'
    WHEN status = 'DONE' THEN 'librarian'
    WHEN status IN ('BLOCKED','FAILED','ROLLED_BACK') THEN 'sre_watchdog'
    ELSE 'executor'
  END
  WHEN task_id LIKE 'autopilot:%' THEN CASE
    WHEN status = 'RUNNING' THEN 'planner'
    WHEN status = 'REVIEW' THEN 'validator'
    WHEN status = 'DONE' THEN 'librarian'
    WHEN status IN ('BLOCKED','FAILED','ROLLED_BACK') THEN 'sre_watchdog'
    ELSE 'planner'
  END
  WHEN task_id = 'parity:weekly_harness' THEN CASE
    WHEN status = 'DONE' THEN 'librarian'
    WHEN status IN ('BLOCKED','FAILED','ROLLED_BACK') THEN 'sre_watchdog'
    ELSE 'validator'
  END
  WHEN task_id = 'continuity:normalize_event_sources' THEN CASE
    WHEN status = 'DONE' THEN 'librarian'
    WHEN status IN ('BLOCKED','FAILED','ROLLED_BACK') THEN 'sre_watchdog'
    ELSE 'sre_watchdog'
  END
  ELSE CASE
    WHEN status = 'REVIEW' THEN 'validator'
    WHEN status = 'DONE' THEN 'librarian'
    WHEN status IN ('BLOCKED','FAILED','ROLLED_BACK') THEN 'sre_watchdog'
    WHEN status = 'RUNNING' THEN 'executor'
    ELSE 'planner'
  END
END
WHERE role_required IS NULL OR TRIM(role_required) = ''
"""
)

con.commit()
con.close()

print(f"OK: initialized continuity DB at {db_path}")
PY
