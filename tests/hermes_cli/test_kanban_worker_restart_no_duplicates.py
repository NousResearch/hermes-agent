"""Proof: recovery after a worker/dispatcher restart never creates a
duplicate task.

Two restart scenarios named in the P0 card's acceptance bar:

1. A worker process dies mid-run (crash, host reboot, OOM kill) before its
   claim's TTL naturally expires. The dispatcher's restart-safe reclaim
   (``release_stale_claims``) must return the SAME task to the pool, not
   spawn a sibling row for the same unit of work.
2. An orchestrator/dispatcher itself restarts mid-fan-out and, not knowing
   whether its earlier ``kanban_create`` call landed, retries with the same
   ``idempotency_key``. It must get back the SAME task id, not a duplicate.

Both mechanisms already exist in the kernel (``release_stale_claims``,
``create_task``'s idempotency check); this file is the missing proof that
they hold under a restart specifically, not just under normal operation.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _all_task_ids(conn) -> set:
    return {row["id"] for row in conn.execute("SELECT id FROM tasks")}


def test_crashed_worker_reclaim_creates_no_duplicate_task(conn):
    """A worker dies mid-run (no clean shutdown, no kanban_complete/block
    call): its expired claim is reclaimed to the pool, not duplicated."""
    task_id = kb.create_task(conn, title="do the thing", assignee="anika")
    before_ids = _all_task_ids(conn)
    assert task_id in before_ids
    assert len(before_ids) == 1

    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.status == "running"

    # Simulate the worker process having vanished (crash/OOM/host reboot):
    # its claim TTL has expired and no live PID is recorded for it.
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ?, worker_pid = NULL, "
            "worker_started_at = NULL WHERE id = ?",
            (int(time.time()) - 3600, task_id),
        )

    reclaimed_count = kb.release_stale_claims(conn)
    assert reclaimed_count == 1

    after_ids = _all_task_ids(conn)
    assert after_ids == before_ids, "reclaim must never create a sibling task row"
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status in ("ready", "todo")
    assert task.claim_lock is None

    # The restarted dispatcher can claim and complete the SAME task again --
    # proving it is genuinely workable again, not a zombie.
    reclaimed_task = kb.claim_task(conn, task_id)
    assert reclaimed_task is not None and reclaimed_task.status == "running"
    assert kb.complete_task(conn, task_id, summary="done after restart")
    assert _all_task_ids(conn) == before_ids


def test_orchestrator_restart_retries_create_with_idempotency_key(conn):
    """An orchestrator that doesn't know whether its first kanban_create call
    landed (it restarted mid-fan-out) retries with the same idempotency_key
    and must get the SAME task id back, never a duplicate."""
    key = "fanout-child-3-of-5"
    first_id = kb.create_task(
        conn, title="delivery slice 3", assignee="builder", idempotency_key=key,
    )
    assert len(_all_task_ids(conn)) == 1

    # "Restart": a fresh call, unaware the first one succeeded, retries.
    second_id = kb.create_task(
        conn, title="delivery slice 3", assignee="builder", idempotency_key=key,
    )
    assert second_id == first_id
    assert len(_all_task_ids(conn)) == 1, "idempotency_key retry must not duplicate the task"

    # A third retry after the task has since completed still returns the
    # same id (archived is the only state idempotency_key does NOT reuse).
    assert kb.complete_task(conn, first_id, summary="shipped")
    third_id = kb.create_task(
        conn, title="delivery slice 3", assignee="builder", idempotency_key=key,
    )
    assert third_id == first_id
    assert len(_all_task_ids(conn)) == 1


def test_dispatcher_reclaim_tick_is_itself_idempotent(conn):
    """Calling the reclaim pass twice in a row (e.g. two dispatcher ticks
    racing after a restart) must not double-count or double-create."""
    task_id = kb.create_task(conn, title="do the thing", assignee="anika")
    kb.claim_task(conn, task_id)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ?, worker_pid = NULL, "
            "worker_started_at = NULL WHERE id = ?",
            (int(time.time()) - 3600, task_id),
        )

    first = kb.release_stale_claims(conn)
    second = kb.release_stale_claims(conn)
    assert first == 1
    assert second == 0, "a second reclaim pass on an already-reclaimed task is a no-op"
    assert len(_all_task_ids(conn)) == 1
