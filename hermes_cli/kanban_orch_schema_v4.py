"""
ORCH V4 Schema — DDL + triggers from §4 of the implementation contract.

Source: 拆卡機制四大缺陷-詳細實作計畫.md §4.3 (normative schema) + §4.4 (triggers).
Provides apply_schema(conn) which executes all DDL via executescript().
"""

import sqlite3
import os

_SCHEMA_DIR = os.path.dirname(os.path.abspath(__file__))

EXPECTED_TABLES: frozenset[str] = frozenset({
    "kanban_board_identity", "kanban_schema_migrations", "kanban_write_fence",
    "kanban_commit_clock", "kanban_migration_operations", "orch_rollback_operations",
    "orch_replay_selectors", "orch_origins", "orch_requests", "orch_request_requirements",
    "orch_plans", "orch_plan_nodes", "orch_plan_edges", "orch_plan_coverage",
    "orch_plan_materializations", "orch_nodes", "orch_external_edges",
    "orch_node_acceptances", "orch_stage_leases", "orch_stage_attempts",
    "orch_results", "orch_delivery_manifests", "orch_delivery_obligations",
    "orch_delivery_manifest_entries", "orch_delivery_attempts",
    "orch_delivery_attempt_events", "orch_delivery_receipts",
    "orch_events", "orch_reconcile_queue", "orch_effect_ledger",
    "orch_delivery_resend_authorizations", "orch_mutation_log",
})

# Minimal stubs for existing Kanban tables that V4 triggers reference.
# These match the live DB shape (from recon) but are minimal for temp DB testing.
_STUB_DDL = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT,
    assignee TEXT,
    status TEXT NOT NULL,
    priority INTEGER DEFAULT 0,
    created_by TEXT,
    created_at INTEGER NOT NULL,
    started_at INTEGER,
    completed_at INTEGER,
    workspace_kind TEXT NOT NULL DEFAULT 'scratch',
    workspace_path TEXT,
    claim_lock TEXT,
    claim_expires INTEGER,
    tenant TEXT,
    result TEXT,
    idempotency_key TEXT,
    spawn_failures INTEGER NOT NULL DEFAULT 0,
    worker_pid INTEGER,
    last_spawn_error TEXT,
    max_runtime_seconds INTEGER,
    last_heartbeat_at INTEGER,
    current_run_id INTEGER,
    workflow_template_id TEXT,
    current_step_key TEXT,
    skills TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_failure_error TEXT,
    max_retries INTEGER,
    branch_name TEXT,
    model_override TEXT,
    session_id TEXT,
    goal_mode INTEGER NOT NULL DEFAULT 0,
    goal_max_turns INTEGER,
    project_id TEXT,
    block_kind TEXT,
    block_recurrences INTEGER NOT NULL DEFAULT 0,
    source_chat TEXT,
    source_fingerprint TEXT,
    done_when TEXT,
    evidence TEXT,
    human_needed INTEGER NOT NULL DEFAULT 0,
    next_action TEXT,
    block_revision INTEGER NOT NULL DEFAULT 0,
    provider_override TEXT,
    reasoning_effort TEXT,
    orch_board_instance_id TEXT,
    orch_tenant_scope TEXT,
    orch_id TEXT,
    orch_plan_version INTEGER,
    orch_node_key TEXT,
    orch_binding_revision INTEGER,
    orch_cancel_epoch INTEGER,
    created_by_run_id INTEGER,
    cancellation_requested_at INTEGER
);

CREATE TABLE IF NOT EXISTS task_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at INTEGER,
    ended_at INTEGER,
    outcome TEXT,
    outcome_digest TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS task_links (
    parent_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'depends',
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE IF NOT EXISTS task_comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    author TEXT,
    body TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS task_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS task_intakes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    intake_data TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS task_input_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    request_kind TEXT,
    status TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS task_review_requirements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    requirement TEXT,
    met INTEGER DEFAULT 0,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS kanban_notify_subs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    platform TEXT,
    chat_id TEXT,
    thread_id TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS kanban_delivery_outbox (
    input_request_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    block_revision INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'sending',
    created_at INTEGER NOT NULL,
    UNIQUE(task_id, block_revision),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS lost_and_found (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    found_key TEXT,
    found_data TEXT,
    found_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS independent_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    reviewer TEXT,
    status TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS independent_review_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER NOT NULL,
    evidence TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (review_id) REFERENCES independent_reviews(id)
);

CREATE TABLE IF NOT EXISTS independent_review_cancellations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER NOT NULL,
    reason TEXT,
    cancelled_at INTEGER NOT NULL,
    FOREIGN KEY (review_id) REFERENCES independent_reviews(id)
);

CREATE TABLE IF NOT EXISTS kanban_chief_escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    escalated_by TEXT,
    reason TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS task_attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    file_path TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);
"""


def create_existing_table_stubs(conn: sqlite3.Connection) -> None:
    """Create the legacy Kanban tables referenced by V4 foreign keys/triggers.

    V4 is compiled in isolation in tests and migrations, while the live
    Kanban database owns these tables.  Keep these definitions deliberately
    compatible with the live columns used by the V4 binding guards; they are
    not a second source of truth for the legacy schema.
    """
    conn.executescript(_STUB_DDL)


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply all V4 schema DDL + triggers to a fresh SQLite connection.

    The caller normally enables foreign keys, but enabling it here as well
    makes the isolated compiler fail closed instead of silently accepting an
    invalid test connection.  Legacy table stubs must exist before SQLite
    compiles V4 foreign keys and triggers that reference them.
    """
    conn.execute("PRAGMA foreign_keys=ON")
    create_existing_table_stubs(conn)

    sql_path = os.path.join(_SCHEMA_DIR, "orch_v4_schema.sql")
    with open(sql_path, "r", encoding="utf-8") as f:
        ddl = f.read()
    conn.executescript(ddl)
    conn.commit()

    # Register orch_capability_ok stub UDF (temp DB only).
    # In production, this is a runtime-private SQLite authorizer.
    # For tests, it always returns 1 (allow) so triggers pass.
    conn.create_function("orch_capability_ok", 7, lambda *a: 1)


def get_table_names(conn: sqlite3.Connection) -> list[str]:
    """Return sorted list of table names in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def get_trigger_names(conn: sqlite3.Connection) -> list[str]:
    """Return sorted list of trigger names."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]
