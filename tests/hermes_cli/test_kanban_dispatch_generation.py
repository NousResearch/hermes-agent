"""Public recovery and dispatch must preserve attempt ownership across races."""

from __future__ import annotations
import os
import sqlite3
import time
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_claims as claims
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_worker_identity as identity
from hermes_cli import kanban_worker_recovery as recovery
from tests.hermes_cli.kanban_scope_support import (
    conn,
    kanban_home,
    _stable_module_identity,
)


def running(conn, *, review=False):
    tid = kb.create_task(conn, title="owned recovery", assignee="worker")
    if review:
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        conn.commit()
    task = (claims.claim_review_task if review else claims.claim_task)(conn, tid)
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET worker_pid=?,worker_pid_started_at=42,max_runtime_seconds=1,"
        "last_heartbeat_at=?,claim_expires=? WHERE id=?",
        (12345, now - 10000, now - 10, tid),
    )
    conn.execute(
        "UPDATE task_runs SET started_at=? WHERE id=?",
        (now - 10000, task.current_run_id),
    )
    conn.commit()
    return task


def test_spawn_callback_exception_never_launches_twice():
    launches = []

    def launch(task, workspace, board=None):
        launches.append(board)
        raise TypeError("failure after starting child")

    with pytest.raises(TypeError, match="after starting"):
        dispatch._call_spawn_fn(launch, None, "/tmp", "board")
    assert launches == ["board"]


def test_delayed_pid_cannot_replace_successor(conn, monkeypatch):
    first = running(conn)
    kb._end_run(conn, first.id, outcome="reclaimed", status="reclaimed")
    conn.execute(
        "UPDATE tasks SET status='ready',claim_lock=NULL,claim_expires=NULL WHERE id=?",
        (first.id,),
    )
    conn.commit()
    second = claims.claim_task(conn, first.id, claimer=first.claim_lock)
    monkeypatch.setattr(identity, "_worker_pid_start_time", lambda pid: 99)
    before = dict(
        conn.execute("SELECT * FROM tasks WHERE id=?", (first.id,)).fetchone()
    )
    assert not dispatch._set_worker_pid(
        conn,
        first.id,
        777,
        expected_run_id=first.current_run_id,
        expected_claim_lock=first.claim_lock,
    )
    assert (
        dict(conn.execute("SELECT * FROM tasks WHERE id=?", (first.id,)).fetchone())
        == before
    )
    assert second.current_run_id != first.current_run_id


@pytest.mark.parametrize("mode", ["timeout", "stale"])
def test_failed_stop_keeps_claim_and_attempt(conn, monkeypatch, mode):
    task = running(conn)
    monkeypatch.setattr(identity, "_worker_pid_identity_state", lambda *a: "alive")

    def denied(pid, sig):
        raise PermissionError("test denied")

    result = (
        recovery.enforce_max_runtime(conn, signal_fn=denied)
        if mode == "timeout"
        else recovery.detect_stale_running(
            conn, stale_timeout_seconds=1, signal_fn=denied
        )
    )
    assert result == []
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task.id,)).fetchone()
    assert (
        row["status"],
        row["current_run_id"],
        row["claim_lock"],
        row["consecutive_failures"],
    ) == ("running", task.current_run_id, task.claim_lock, 0)
    assert (
        conn.execute(
            "SELECT ended_at FROM task_runs WHERE id=?", (task.current_run_id,)
        ).fetchone()[0]
        is None
    )


@pytest.mark.parametrize("review", [False, True])
def test_successful_timeout_settles_original_phase_once(conn, monkeypatch, review):
    task = running(conn, review=review)
    state = {"alive": True}
    monkeypatch.setattr(
        identity,
        "_worker_pid_identity_state",
        lambda *a: "alive" if state["alive"] else "dead",
    )

    def stop(pid, sig):
        state["alive"] = False

    assert recovery.enforce_max_runtime(conn, signal_fn=stop) == [task.id]
    assert recovery.enforce_max_runtime(conn, signal_fn=stop) == []
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task.id,)).fetchone()
    assert row["status"] == ("review" if review else "ready")
    assert row["consecutive_failures"] == 1
    assert row["current_run_id"] is None
    assert (
        conn.execute(
            "SELECT outcome FROM task_runs WHERE id=?", (task.current_run_id,)
        ).fetchone()[0]
        == "timed_out"
    )


def test_fresh_heartbeat_invalidates_stale_scan_before_signal(conn, monkeypatch):
    task = running(conn)
    reserve = claims.reserve_reclaim

    def refresh(c, tid, snapshot, **kwargs):
        c.execute(
            "UPDATE tasks SET last_heartbeat_at=? WHERE id=?", (int(time.time()), tid)
        )
        c.commit()
        return reserve(c, tid, snapshot, **kwargs)

    monkeypatch.setattr(claims, "reserve_reclaim", refresh)

    def unexpected(*args, **kwargs):
        raise AssertionError("fresh worker signalled")

    assert (
        recovery.detect_stale_running(
            conn, stale_timeout_seconds=1, signal_fn=unexpected
        )
        == []
    )
    assert (
        conn.execute(
            "SELECT current_run_id FROM tasks WHERE id=?", (task.id,)
        ).fetchone()[0]
        == task.current_run_id
    )


def test_timeout_budget_is_settled_before_retry_becomes_visible(conn, monkeypatch):
    task = running(conn)
    state = {"alive": True}
    monkeypatch.setattr(
        identity,
        "_worker_pid_identity_state",
        lambda *a: "alive" if state["alive"] else "dead",
    )
    record = dispatch._record_task_failure
    observations = []
    path = next(r[2] for r in conn.execute("PRAGMA database_list") if r[1] == "main")

    def inspect(c, tid, *args, **kwargs):
        with sqlite3.connect(path) as observer:
            observations.append(
                observer.execute(
                    "SELECT status,current_run_id FROM tasks WHERE id=?", (tid,)
                ).fetchone()
            )
        return record(c, tid, *args, **kwargs)

    monkeypatch.setattr(dispatch, "_record_task_failure", inspect)

    def stop(pid, sig):
        state["alive"] = False

    assert recovery.enforce_max_runtime(conn, signal_fn=stop) == [task.id]
    assert observations == [("running", task.current_run_id)]


def test_scope_names_are_unique_for_identical_runs_in_distinct_boards(tmp_path):
    from hermes_cli import kanban_worker_scope as scope

    first = tmp_path / "first" / "kanban.db"
    second = tmp_path / "second" / "kanban.db"
    a = scope._kanban_worker_scope_unit("t_imported", 7, db_path=str(first))
    b = scope._kanban_worker_scope_unit("t_imported", 7, db_path=str(second))
    assert a != b
    assert (
        scope._task_id_from_kanban_scope_unit(a)
        == scope._task_id_from_kanban_scope_unit(b)
        == "t_imported"
    )
    assert (
        scope._kanban_worker_scope_unit(
            "t_imported", 7, db_path=str(first.parent / ".." / "first" / "kanban.db")
        )
        == a
    )


@pytest.mark.parametrize("fingerprints,expected", [([42, 42], 42), ([42, 84], None)])
def test_legacy_identity_does_not_bless_recycled_pid(
    monkeypatch, tmp_path, fingerprints, expected
):
    import psutil

    path = tmp_path / "kanban.db"

    class Process:
        def __init__(self, pid):
            pass

        def create_time(self):
            return 100.0

        def is_running(self):
            return True

        def environ(self):
            return {
                "HERMES_KANBAN_TASK": "t_a",
                "HERMES_KANBAN_RUN_ID": "7",
                "HERMES_KANBAN_DB": str(path),
            }

    monkeypatch.setattr(psutil, "Process", Process)
    values = iter(fingerprints)
    monkeypatch.setattr(identity, "_worker_pid_start_time", lambda pid: next(values))
    assert identity._legacy_worker_fingerprint(123, "t_a", 7, str(path)) == expected


def test_queued_old_scope_cannot_mark_successor_stop_pending(conn):
    task = running(conn)
    conn.execute(
        "UPDATE tasks SET worker_scope='new.scope',worker_registered_at=NULL WHERE id=?",
        (task.id,),
    )
    conn.commit()
    assert recovery._mark_run_stop_pending(task.id, expected_scope="old.scope") is False
    assert conn.execute(
        "SELECT stop_pending FROM task_runs WHERE id=?", (task.current_run_id,)
    ).fetchone()[0] in (None, 0)
    assert recovery._mark_run_stop_pending(task.id, expected_scope="new.scope") is True
    assert (
        conn.execute(
            "SELECT stop_pending FROM task_runs WHERE id=?", (task.current_run_id,)
        ).fetchone()[0]
        == 1
    )


@pytest.mark.parametrize("outcome", ["normal", "superseded", "unconfirmed"])
def test_public_dispatch_accounts_for_delayed_or_unconfirmed_launch(
    conn,
    monkeypatch,
    kanban_home,
    outcome,
):
    from hermes_cli import kanban_worker_spawn as spawn
    from tests.hermes_cli.kanban_scope_support import _spawnable_profile

    _spawnable_profile(kanban_home)
    tid = kb.create_task(conn, title="launch ownership", assignee="elias")
    monkeypatch.setattr(identity, "_worker_pid_start_time", lambda pid: 42)
    stops = []

    def stop(pid, lock, **kwargs):
        stops.append((pid, kwargs.get("run_id"), kwargs.get("scope_unit")))
        return {
            "host_local": True,
            "termination_attempted": bool(pid),
            "terminated": False,
        }

    monkeypatch.setattr(identity, "_terminate_reclaimed_worker", stop)
    launched = []
    successor = []

    def launch(task, workspace, board=None):
        launched.append(task.current_run_id)
        if outcome == "superseded":
            # A second connection moves ownership while the launch callback
            # is outside a database transaction, just as another controller can.
            path = claims._reclaim_db_path(conn)
            from hermes_cli import kanban_db_connect as connect

            other = connect.connect(__import__("pathlib").Path(path))
            try:
                assert claims.reclaim_task(other, tid, reason="move during launch")
                successor.append(claims.claim_task(other, tid, claimer=task.claim_lock))
            finally:
                other.close()
        if outcome == "unconfirmed":
            raise spawn.WorkerLaunchUnconfirmed(
                "scope remains alive", 123, "owned.scope"
            )
        return spawn._SpawnedWorkerPid(123, "owned.scope")

    result = dispatch.dispatch_once(conn, spawn_fn=launch)
    assert len(launched) == 1
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] == "running"
    assert row["consecutive_failures"] == 0
    if outcome == "superseded":
        assert row["current_run_id"] == successor[0].current_run_id
        assert row["worker_pid"] is None
        assert not result.spawned
        assert (123, launched[0], "owned.scope") in stops
        old = conn.execute(
            "SELECT worker_pid,worker_scope FROM task_runs WHERE id=?", (launched[0],)
        ).fetchone()
        assert tuple(old) == (123, "owned.scope")
    else:
        assert row["current_run_id"] == launched[0]
        assert (row["worker_pid"], row["worker_scope"]) == (123, "owned.scope")
        assert bool(result.spawned) == (outcome == "normal")
        assert row["claim_expires"] > int(time.time())


@pytest.mark.parametrize("stop_confirmed", [False, True])
def test_handoff_stop_reservation_preserves_intent_until_verified(
    conn, monkeypatch, stop_confirmed
):
    from hermes_cli import kanban_worker_handoff as handoff
    from hermes_cli import kanban_worker_stop as stop

    task = running(conn)
    conn.execute("UPDATE tasks SET worker_scope='handoff.scope' WHERE id=?", (task.id,))
    conn.commit()
    assert handoff._defer_own_worker_handoff(
        conn,
        task.id,
        "handoff.scope",
        {
            "handoff": "review_requested",
            "reviewer": "reviewer",
            "implementer": "worker",
            "summary": "done",
        },
        claim_lock=task.claim_lock,
        expected_run_id=task.current_run_id,
    )
    conn.execute(
        "UPDATE tasks SET claim_expires=? WHERE id=?", (int(time.time()) - 5, task.id)
    )
    conn.commit()
    snapshots = []

    def verify(scope, **kwargs):
        snapshots.append(
            dict(conn.execute("SELECT * FROM tasks WHERE id=?", (task.id,)).fetchone())
        )
        assert not claims.heartbeat_claim(
            conn, task.id, expected_run_id=task.current_run_id
        )
        return stop_confirmed

    monkeypatch.setattr(stop, "request_worker_scope_stop", verify)
    monkeypatch.setattr(identity, "_run_worker_alive", lambda row: (False, "test"))
    assert claims.release_stale_claims(conn) == 0
    assert len(snapshots) == 1
    assert snapshots[0]["reclaim_reserved_at"] is not None
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task.id,)).fetchone()
    assert row["consecutive_failures"] == 0
    if stop_confirmed:
        assert row["status"] == "review"
        assert row["assignee"] == "reviewer"
        assert row["worker_scope"] is None
        assert row["reclaim_reserved_at"] is None
        assert claims.release_stale_claims(conn) == 0
        assert len(snapshots) == 1
    else:
        assert row["status"] == "running"
        assert row["current_run_id"] == task.current_run_id
        assert row["worker_scope"] == "handoff.scope"
        assert row["claim_expires"] > int(time.time())
