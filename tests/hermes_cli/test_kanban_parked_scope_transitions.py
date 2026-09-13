"""Operator resumes consume proof for the observed worker generation only."""
from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as connect
from hermes_cli import kanban_transitions as transitions
from hermes_cli import kanban_worker_scope as scopes


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    db_path = tmp_path / "board.db"
    with connect.connect_closing(db_path) as conn:
        task_id = kb.create_task(conn, title="Resume after worker exit", assignee="default")
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (task_id,))
        yield conn, task_id, db_path


def resume(conn, task_id, operation):
    if operation == "promote":
        return transitions.promote_task(conn, task_id, actor="operator")[0]
    return transitions.unblock_task(conn, task_id)


@pytest.mark.parametrize("operation", ["promote", "unblock"])
@pytest.mark.parametrize("state", [None, "dead", "active", "unknown", "unsupported"])
def test_operator_resume_requires_empty_scope_and_preserves_unscoped_recovery(
    board, monkeypatch, operation, state,
):
    conn, task_id, _ = board
    scope = "hermes-kanban-task-r1.scope" if state is not None else None
    conn.execute("UPDATE tasks SET worker_scope = ? WHERE id = ?", (scope, task_id))
    probes = []

    def probe(unit):
        probes.append(unit)
        return state

    monkeypatch.setattr(scopes, "_kanban_scope_state", probe)
    allowed = state in (None, "dead")
    assert resume(conn, task_id, operation) is allowed
    row = conn.execute("SELECT status, worker_scope FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert tuple(row) == (("ready", None) if allowed else ("blocked", scope))
    assert probes == ([] if state is None else [scope])


@pytest.mark.parametrize("operation", ["promote", "unblock"])
@pytest.mark.parametrize("changed_field", ["worker_scope", "current_run_id", "worker_pid_started_at"])
def test_operator_resume_rejects_replaced_attempt_during_scope_probe(
    board, monkeypatch, operation, changed_field,
):
    conn, task_id, db_path = board
    old_scope = "hermes-kanban-task-r1.scope"
    new_value = "hermes-kanban-task-r2.scope" if changed_field == "worker_scope" else 2
    conn.execute(
        "UPDATE tasks SET worker_scope = ?, current_run_id = 1, "
        "worker_pid_started_at = 1 WHERE id = ?", (old_scope, task_id),
    )
    before_events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]

    def replace_attempt(unit):
        assert unit == old_scope
        assert not conn.in_transaction  # scope I/O never holds the DB write lock
        with sqlite3.connect(db_path) as concurrent:
            concurrent.execute(
                f"UPDATE tasks SET {changed_field} = ? WHERE id = ?", (new_value, task_id),
            )
        return "dead"

    monkeypatch.setattr(scopes, "_kanban_scope_state", replace_attempt)
    assert resume(conn, task_id, operation) is False
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row["status"] == "blocked"
    assert row[changed_field] == new_value
    assert row["worker_scope"] == (new_value if changed_field == "worker_scope" else old_scope)
    assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == before_events


@pytest.mark.parametrize("operation", ["request_review", "request_changes"])
def test_invalid_review_provenance_is_rejected_before_worker_termination(
    board, monkeypatch, operation,
):
    from hermes_cli import kanban_claims as claims
    from hermes_cli import kanban_worker_handoff as handoff
    from hermes_cli import kanban_worker_identity as identity

    conn, task_id, _ = board
    status = "review" if operation == "request_changes" else "ready"
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    if operation == "request_review":
        # A prior rework run exists, but its reviewer provenance was lost.
        conn.execute(
            "INSERT INTO task_runs(task_id, status, started_at, ended_at, outcome) "
            "VALUES (?, 'done', 1, 2, 'changes_requested')", (task_id,),
        )
        claimed = claims.claim_task(conn, task_id)
    else:
        # A reviewer was dispatched without any implementer handoff history.
        claimed = claims.claim_review_task(conn, task_id)
    assert claimed is not None
    monkeypatch.setattr(handoff, "_handoff_caller_is_worker", lambda *a, **k: False)
    terminations = []

    def terminate(*args, **kwargs):
        terminations.append((args, kwargs))
        return {"terminated": True}

    monkeypatch.setattr(identity, "_terminate_reclaimed_worker", terminate)
    before = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if operation == "request_review":
        ok, reason = transitions.request_review(
            conn, task_id, expected_run_id=claimed.current_run_id, with_reason=True,
        )
    else:
        ok, reason = transitions.request_changes(
            conn, task_id, expected_run_id=claimed.current_run_id, reason="More work needed",
        )
    assert ok is False
    assert "provenance" in reason or "review_requested" in reason
    assert terminations == []
    after = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert tuple(after) == tuple(before)
