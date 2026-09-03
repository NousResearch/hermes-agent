"""Commit-conditional scope-stop intents (Gate B pass 4, finding O).

A worker scope-stop queued mid-write-transaction is process state, not a
DB row: without the intent buffer, a dashboard outer-transaction ROLLBACK
undid the demotion writes but left the queued stop alive, killing the
worker of a row that was running again. ``write_txn`` therefore collects
stop intents per savepoint level and flushes them only at the outermost
COMMIT; every rollback path discards them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path: Path):
    db = kb.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def quiet_stop_service(monkeypatch: pytest.MonkeyPatch):
    """Keep the background service out of the picture: reset state, stop
    thread spawn, and make every scope probe report 'active' so requests
    take the queueing path instead of the confirmed-dead fast path."""
    kb.reset_scope_stop_service_for_tests()
    monkeypatch.setattr(kb, "_ensure_scope_stop_thread", lambda: None)
    monkeypatch.setattr(kb, "_kanban_scope_state", lambda unit: "active")
    yield
    kb.reset_scope_stop_service_for_tests()


def _queued() -> set[str]:
    with kb._scope_stop_lock:
        return set(kb._scope_stop_pending)


def test_stop_requested_inside_txn_flushes_after_commit(conn):
    with kb.write_txn(conn):
        assert kb.request_worker_scope_stop(
            "u1.scope", task_id="t1", conn=conn,
        ) is False
        # Deferred: not on the queue while the transaction is open.
        assert _queued() == set()
    assert _queued() == {"u1.scope"}


def test_stop_requested_without_conn_queues_immediately(conn):
    """Post-AA contract: the intent stack is keyed by CONNECTION, so a
    request that does not name its connection cannot be folded into any
    transaction — it queues immediately. The legacy no-conn call sites
    run outside any transaction by construction."""
    with kb.write_txn(conn):
        assert kb.request_worker_scope_stop("noconn.scope") is False
        assert _queued() == {"noconn.scope"}


def test_stop_requested_inside_txn_discarded_on_rollback(conn):
    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with kb.write_txn(conn):
            assert kb.request_worker_scope_stop(
                "u2.scope", task_id="t2", conn=conn,
            ) is False
            raise Boom
    assert _queued() == set()


def test_nested_savepoint_rollback_discards_only_its_own_intents(conn):
    with kb.write_txn(conn):
        kb.request_worker_scope_stop("outer.scope", conn=conn)
        with pytest.raises(RuntimeError):
            with kb.write_txn(conn, allow_nested=True):
                kb.request_worker_scope_stop("inner.scope", conn=conn)
                raise RuntimeError("inner level aborts")
        # The inner savepoint's intent died with its rollback; the outer
        # one is still pending its commit.
        assert _queued() == set()
    assert _queued() == {"outer.scope"}


def test_nested_savepoint_release_folds_intents_into_outer_commit(conn):
    with kb.write_txn(conn):
        with kb.write_txn(conn, allow_nested=True):
            kb.request_worker_scope_stop("folded.scope", conn=conn)
        # RELEASE promoted the intent to the outer level — still deferred
        # until the OUTERMOST commit, not just the savepoint's.
        assert _queued() == set()
    assert _queued() == {"folded.scope"}


def test_invalidate_phase0_under_outer_rollback_queues_no_stop(
    conn, monkeypatch: pytest.MonkeyPatch,
):
    """The dashboard composition that motivated the fix: the ancestor
    reopen's Phase 0 defers a scoped descendant's stop inside the
    dashboard's outer transaction. When that transaction rolls back, the
    deferral writes vanish AND no stop may be queued — otherwise the
    still-running row's worker is killed by a demotion that never
    happened."""
    parent_id = kb.create_task(conn, title="ancestor", assignee="planner")
    assert kb.complete_task(conn, parent_id)
    child_id = kb.create_task(
        conn, title="scoped child", assignee="builder", parents=[parent_id],
    )
    assert kb.claim_task(conn, child_id) is not None
    host = kb._claimer_id().split(":", 1)[0]
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_lock = ?, claim_expires = 9999999999, "
            "worker_pid = 424242, worker_pid_started_at = 55, "
            "worker_registered_at = 1, worker_scope = 'old-run.scope' "
            "WHERE id = ?",
            (f"{host}:424242", child_id),
        )

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with kb.write_txn(conn):
            kb.invalidate_descendants_for_parent_reopen(
                conn, parent_id, author="dashboard",
            )
            raise Boom

    assert _queued() == set()
    row = conn.execute(
        "SELECT status, worker_scope, claim_lock FROM tasks WHERE id = ?",
        (child_id,),
    ).fetchone()
    # Rollback restored the pre-invalidation state entirely.
    assert row["status"] == "running"
    assert row["worker_scope"] == "old-run.scope"
    assert row["claim_lock"] is not None


def test_reset_scope_stop_service_stops_a_lingering_thread():
    """reset_scope_stop_service_for_tests must stop a running service
    thread, not just clear its state: the daemon parks on the module
    wake event, so leaving it alive lets it drain a queue that a LATER
    test file flushes (found by the pass 8b AE single-process gate
    run — the txn tests passed alone and failed in-process)."""
    import threading
    import time as _time

    thread = threading.Thread(
        target=kb._scope_stop_service_loop, name="lingering-for-reset",
        daemon=True,
    )
    # Register it the way _ensure_scope_stop_thread does (the autouse
    # fixture no-ops that spawner, so the test drives it directly).
    kb._scope_stop_thread = thread
    thread.start()
    _time.sleep(0.05)
    assert thread.is_alive(), "service loop should park on the wake event"

    kb.reset_scope_stop_service_for_tests()

    thread.join(timeout=2.0)
    assert not thread.is_alive(), "reset must stop a lingering service thread"
    assert kb._scope_stop_thread is None


def test_lingering_service_thread_does_not_drain_a_later_queue(conn):
    """The isolation failure the reset fix closes, end to end: a service
    thread spawned by an earlier file (parked on the wake event) must
    not consume a queue that a later file flushes after commit."""
    import threading
    import time as _time

    thread = threading.Thread(
        target=kb._scope_stop_service_loop, name="lingering-drain",
        daemon=True,
    )
    kb._scope_stop_thread = thread
    thread.start()
    _time.sleep(0.05)
    # What every test file's reset-using fixture does between files.
    kb.reset_scope_stop_service_for_tests()

    with kb.write_txn(conn):
        assert kb.request_worker_scope_stop("late.scope", conn=conn) is False
        assert _queued() == set()
    # The flushed intent stays on the queue: nothing drains it out from
    # under the assertion.
    assert _queued() == {"late.scope"}
    kb.reset_scope_stop_service_for_tests()
