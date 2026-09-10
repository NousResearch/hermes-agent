"""ORCH V4 Sidecar Schema — DDL loader for sidecar orch_v4.db.

Source: 拆卡機制四大缺陷-詳細實作計畫.md §15.0 Sidecar Architecture Decision.

This module applies the sidecar schema to a fresh SQLite connection.
Unlike the in-place schema (kanban_orch_schema_v4.py), this does NOT create
stub native tables (tasks, task_links, etc.) — those live in the native
kanban.db and are accessed read-only via the bridge.

Hard rules:
- No REFERENCES to native tables (tasks.id, etc.) — soft FK handled by bridge
- No stub table creation — sidecar is pure orch_* tables
- foreign_keys=ON enforced
- orch_capability_ok UDF registered as allow-all (bridge enforces real capability)
"""

import sqlite3
import os

_SCHEMA_DIR = os.path.dirname(os.path.abspath(__file__))

EXPECTED_SIDECAR_TABLES: frozenset[str] = frozenset({
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

# Tables that exist in native kanban.db (NOT created in sidecar)
NATIVE_TABLES = frozenset({
    "tasks", "task_links", "task_comments", "task_events", "task_intakes",
    "task_input_requests", "task_review_requirements", "kanban_notify_subs",
    "kanban_delivery_outbox", "lost_and_found", "independent_reviews",
    "independent_review_evidence", "independent_review_cancellations",
    "kanban_chief_escalations", "task_attachments", "task_runs",
})


def apply_sidecar_schema(conn: sqlite3.Connection, *, test_open_capability: bool = False) -> None:
    """Apply sidecar V4 schema DDL to a fresh SQLite connection.

    This does NOT create native table stubs. The sidecar DB contains only
    orch_* tables + board identity + write fence + commit clock + migration ops.
    Cross-DB FKs to native tables (tasks.id) are removed; the bridge enforces
    soft FK at the application layer.

    Capability UDF is fail-closed by default.
    """
    from hermes_cli.kanban_orch_capability import install_fail_closed_udf, install_test_open_udf

    conn.execute("PRAGMA foreign_keys=ON")

    sql_path = os.path.join(_SCHEMA_DIR, "orch_v4_schema_sidecar.sql")
    with open(sql_path, "r", encoding="utf-8") as f:
        ddl = f.read()
    conn.executescript(ddl)
    conn.commit()

    if test_open_capability:
        install_test_open_udf(conn)
    else:
        install_fail_closed_udf(conn)

    from hermes_cli.kanban_orch_digest_udf import apply_digest_guards

    apply_digest_guards(conn)


def get_sidecar_table_names(conn: sqlite3.Connection) -> list[str]:
    """Return sorted list of table names in the sidecar DB."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def get_sidecar_trigger_names(conn: sqlite3.Connection) -> list[str]:
    """Return sorted list of trigger names in the sidecar DB."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def verify_no_native_tables(conn: sqlite3.Connection) -> bool:
    """Verify that no native Kanban tables exist in the sidecar DB.

    Returns True if clean (no native tables found).
    """
    tables = set(get_sidecar_table_names(conn))
    leaked = tables & NATIVE_TABLES
    if leaked:
        raise RuntimeError(f"Native tables leaked into sidecar: {leaked}")
    return True


__all__ = [
    "EXPECTED_SIDECAR_TABLES", "NATIVE_TABLES",
    "apply_sidecar_schema", "get_sidecar_table_names",
    "get_sidecar_trigger_names", "verify_no_native_tables",
]
