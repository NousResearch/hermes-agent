"""Regression coverage for create-time worktree and dispatcher failure typing."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def test_create_refuses_unanchored_worktree(conn) -> None:
    with pytest.raises(ValueError, match="worktree workspace requires a path"):
        kb.create_task(conn, title="cannot dispatch", workspace_kind="worktree")

    assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_spawn_failure_breaker_sets_infrastructure_block_kind(conn) -> None:
    task_id = kb.create_task(conn, title="worker cannot spawn", assignee="builder")
    assert kb.claim_task(conn, task_id, claimer="builder:test") is not None

    assert kbd._record_task_failure(
        conn,
        task_id,
        "workspace could not be materialized",
        outcome="spawn_failed",
        failure_limit=1,
        release_claim=True,
        end_run=True,
    )

    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "blocked"
    assert task.block_kind == "infrastructure"
    assert conn.execute(
        "SELECT count(*) FROM tasks WHERE status='blocked' AND block_kind IS NULL"
    ).fetchone()[0] == 0


def test_gave_up_breaker_sets_infrastructure_block_kind(conn) -> None:
    task_id = kb.create_task(conn, title="dispatcher gave up", assignee="builder")

    assert kbd._record_task_failure(
        conn,
        task_id,
        "dispatcher retry budget exhausted",
        outcome="gave_up",
        failure_limit=1,
    )

    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "blocked"
    assert task.block_kind == "infrastructure"


@pytest.mark.parametrize("outcome", ["crashed", "timed_out"])
def test_worker_failure_breaker_does_not_set_infrastructure_block_kind(
    conn, outcome: str
) -> None:
    task_id = kb.create_task(conn, title=f"worker {outcome}", assignee="builder")

    assert kbd._record_task_failure(
        conn,
        task_id,
        f"worker {outcome}",
        outcome=outcome,
        failure_limit=1,
    )

    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "blocked"
    assert task.block_kind is None
