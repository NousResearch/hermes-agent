from __future__ import annotations

import hashlib

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def running_execution(tmp_path):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path)
    conn = kb.connect(db_path)
    now = 1_700_000_000
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at, started_at, workspace_kind) "
        "VALUES ('t_receipt', 'receipt', 'running', ?, ?, 'scratch')",
        (now, now),
    )
    run_id = conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at) "
        "VALUES ('t_receipt', 'running', ?) RETURNING id",
        (now,),
    ).fetchone()[0]
    conn.execute(
        "UPDATE tasks SET current_run_id = ? WHERE id = 't_receipt'", (run_id,)
    )
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES ('t_receipt', ?, 'claimed', NULL, ?)",
        (run_id, now),
    )
    conn.commit()
    yield conn
    conn.close()


def _record(conn, execution, *, operation_id="op-1", receipt=b"exact\x00bytes"):
    return kb.record_verified_external_receipt(
        conn,
        operation_id=operation_id,
        task_id=execution.task_id,
        run_id=execution.run_id,
        revision=execution.revision,
        receipt=receipt,
        receipt_sha256=hashlib.sha256(receipt).hexdigest(),
        verified_at=1_700_000_001,
    )


def test_active_execution_revision_tracks_run_event_cursor(running_execution):
    conn = running_execution
    first = kb.get_active_execution(conn, "t_receipt")
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES ('t_receipt', ?, 'heartbeat', NULL, 1700000001)",
        (first.run_id,),
    )
    conn.commit()
    second = kb.get_active_execution(conn, "t_receipt")
    assert second.run_id == first.run_id
    assert second.revision > first.revision


def test_receipt_replay_is_exact_and_does_not_change_lifecycle(running_execution):
    conn = running_execution
    execution = kb.get_active_execution(conn, "t_receipt")
    admitted = _record(conn, execution)
    task = kb.get_task(conn, "t_receipt")
    assert task is not None and task.status == "running" and task.completed_at is None
    assert kb.get_active_execution(conn, "t_receipt") == execution

    conn.execute(
        "UPDATE tasks SET status = 'done', completed_at = 1700000002, "
        "current_run_id = NULL WHERE id = 't_receipt'"
    )
    conn.execute(
        "UPDATE task_runs SET status = 'done', ended_at = 1700000002 WHERE id = ?",
        (execution.run_id,),
    )
    conn.commit()
    replay = _record(conn, execution)
    assert replay == admitted
    assert replay.receipt == b"exact\x00bytes"
    assert replay.verified_at == 1_700_000_001


def test_receipt_conflicts_and_stale_or_inactive_admission_fail_closed(running_execution):
    conn = running_execution
    execution = kb.get_active_execution(conn, "t_receipt")
    _record(conn, execution)
    with pytest.raises(ValueError, match="conflicting-external-receipt"):
        _record(conn, execution, receipt=b"different")
    with pytest.raises(ValueError, match="stale-kanban-run"):
        _record(conn, execution.__class__(execution.task_id, execution.run_id + 1, execution.revision), operation_id="op-run")
    with pytest.raises(ValueError, match="stale-kanban-revision"):
        _record(conn, execution.__class__(execution.task_id, execution.run_id, execution.revision + 1), operation_id="op-revision")

    conn.execute("UPDATE tasks SET status = 'done', completed_at = 1700000002 WHERE id = 't_receipt'")
    conn.commit()
    with pytest.raises(ValueError, match="inactive-kanban-execution"):
        _record(conn, execution, operation_id="op-completed")
    with pytest.raises(ValueError, match="inactive-kanban-execution"):
        kb.get_active_execution(conn, "missing")
