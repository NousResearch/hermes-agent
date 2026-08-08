"""Seam tests for the kanban_db models extraction (R1-S1).

Proves that ``hermes_cli.kanban_db`` re-exports the exact same objects as
``hermes_cli.kanban_db_models`` (identity-preserving shim) and that the moved
block is self-contained: dataclass construction works standalone and
``SCHEMA_SQL`` is byte-identical to what the rest of the module consumes.
"""

from __future__ import annotations

import sqlite3

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_models as m


# The 6 names the shim must re-export, identity-preserving.
SHIM_NAMES = ("Task", "Run", "Comment", "Attachment", "Event", "SCHEMA_SQL")


def test_shim_identity_all_six_names():
    """Every re-exported name is the *same object* in both modules."""
    for name in SHIM_NAMES:
        assert getattr(kb, name) is getattr(m, name), name


def test_shim_no_star_import():
    """The shim is explicit-name only — no accidental namespace pollution."""
    # kanban_db itself imports json at module top, so probe something that is
    # *only* in the models module: the from_row helper must not leak as a
    # module-level name on kanban_db.
    assert "from_row" not in dir(kb), "from_row should only exist on classes"
    # And the models module must be standalone: it must not import kanban_db.
    assert "kanban_db" not in dir(m), "models module must not import kanban_db"


def test_task_from_row_roundtrip_aggressive():
    """Task.from_row round-trips a full sqlite3.Row with JSON skills parsing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT 't-1' AS id, 'Ship it' AS title, 'body' AS body, 'alice' AS assignee,
               'in_progress' AS status, 3 AS priority, 'bob' AS created_by,
               1000 AS created_at, NULL AS started_at, NULL AS completed_at,
               'default' AS workspace_kind, '/tmp/w' AS workspace_path,
               NULL AS claim_lock, NULL AS claim_expires, NULL AS tenant,
               NULL AS branch_name, NULL AS project_id, NULL AS result,
               NULL AS idempotency_key, 0 AS consecutive_failures,
               NULL AS worker_pid, NULL AS last_failure_error,
               NULL AS max_runtime_seconds, NULL AS last_heartbeat_at,
               NULL AS current_run_id, NULL AS workflow_template_id,
               NULL AS current_step_key, NULL AS last_outcome,
               NULL AS last_error, NULL AS resume_from_step, NULL AS deadline,
               NULL AS block_kind, 0 AS block_recurrences, '["a","b"]' AS skills
        """
    ).fetchone()

    task = m.Task.from_row(row)
    assert task.id == "t-1"
    assert task.title == "Ship it"
    assert task.assignee == "alice"
    assert task.skills == ["a", "b"], "JSON skills must be parsed to a list of str"
    assert task.consecutive_failures == 0
    assert task.block_recurrences == 0
    # Optional fields with NULLs stay None (not dropped, not mangled).
    assert task.branch_name is None
    assert task.worker_pid is None


def test_run_from_row_roundtrip_aggressive():
    """Run.from_row parses metadata JSON and coerces ints."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT 42 AS id, 't-1' AS task_id, 'default' AS profile, NULL AS step_key,
               'completed' AS status, NULL AS claim_lock, NULL AS claim_expires,
               NULL AS worker_pid, NULL AS max_runtime_seconds,
               NULL AS last_heartbeat_at, 1000 AS started_at, 2000 AS ended_at,
               'success' AS outcome, 'ok' AS summary, '{"n":1}' AS metadata,
               NULL AS error
        """
    ).fetchone()

    run = m.Run.from_row(row)
    assert run.id == 42
    assert run.task_id == "t-1"
    assert run.metadata == {"n": 1}
    assert run.ended_at == 2000


def test_dataclass_construction_and_defaults():
    """Dataclasses construct standalone with documented defaults (no kanban_db deps)."""
    t = m.Task(
        id="t-1", title="x", body=None, assignee=None, status="todo",
        priority=0, created_by=None, created_at=0, started_at=None,
        completed_at=None, workspace_kind="default", workspace_path=None,
        claim_lock=None, claim_expires=None, tenant=None,
    )
    assert t.branch_name is None
    assert t.consecutive_failures == 0
    assert t.block_recurrences == 0
    assert t.current_run_id is None

    c = m.Comment(id=1, task_id="t-1", author="a", body="b", created_at=5)
    assert c.created_at == 5

    e = m.Event(id=1, task_id="t-1", kind="created", payload=None, created_at=5)
    assert e.kind == "created"


def test_schema_integrity_seven_tables():
    """SCHEMA_SQL declares all 7 tables and the load-bearing indexes."""
    tables = [
        "tasks", "task_links", "task_comments", "task_events",
        "task_runs", "task_attachments", "kanban_notify_subs",
    ]
    for t in tables:
        assert f"CREATE TABLE IF NOT EXISTS {t}" in m.SCHEMA_SQL, t
    for idx in (
        "idx_tasks_assignee_status", "idx_attachments_task", "idx_notify_task",
        "idx_runs_task", "idx_runs_status", "idx_comments_task",
    ):
        assert f"CREATE INDEX IF NOT EXISTS {idx}" in m.SCHEMA_SQL, idx


def test_schema_roundtrip_executes():
    """SCHEMA_SQL is executable SQL — create the full schema in-memory."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(m.SCHEMA_SQL)
    created = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {
        "tasks", "task_links", "task_comments", "task_events",
        "task_runs", "task_attachments", "kanban_notify_subs",
    } <= created
