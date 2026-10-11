"""Kanban worker admission is retained only while its exact run is live."""
from __future__ import annotations

import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import admission_runtime as ar
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.admission_contract import AdmissionCaller


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def _worktree(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "README").write_text("base\n")
    _git(repo, "add", "README")
    _git(repo, "commit", "-m", "base")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    worker = tmp_path / "worker"
    _git(repo, "worktree", "add", "-b", "worker/lifecycle", str(worker), "HEAD")
    return worker


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    worker = _worktree(tmp_path)
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="admission lifecycle", assignee="default")
        task = kb.claim_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        admission = ar.admit_writer(
            request_id=ar.kanban_request_id(task_id, task.current_run_id),
            caller=AdmissionCaller.KANBAN,
            workspace=str(worker),
            priority=0,
            writer_id=f"kanban:{task_id}",
            kanban_cap=1,
        )
        assert admission.state == "running"
        yield conn, task_id, task.current_run_id


def _admission_row() -> tuple[str, int | None]:
    with sqlite3.connect(ar.ledger_path()) as conn:
        row = conn.execute(
            "SELECT state, lease_expires_at FROM admission_requests ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    return row[0], row[1]


def _running_occupancy() -> int:
    with sqlite3.connect(ar.ledger_path()) as conn:
        return conn.execute("SELECT COUNT(*) FROM admission_requests WHERE state = 'running'").fetchone()[0]


def test_normal_completion_releases_once(lifecycle):
    conn, task_id, run_id = lifecycle

    assert kb.complete_task(conn, task_id, summary="done", expected_run_id=run_id)
    assert _admission_row()[0] == "released"
    assert not ar.release_kanban_admission(task_id, run_id)


def test_heartbeat_renews_without_premature_release(lifecycle, monkeypatch):
    conn, task_id, run_id = lifecycle
    with sqlite3.connect(ar.ledger_path(), isolation_level=None) as ledger:
        created_at, old_expiry = ledger.execute(
            "SELECT created_at, lease_expires_at FROM admission_requests"
        ).fetchone()
    monkeypatch.setattr(ar.time, "time", lambda: int(created_at) + 10)

    assert kbd.heartbeat_worker(conn, task_id, expected_run_id=run_id)
    state, renewed_expiry = _admission_row()
    assert state == "running"
    assert renewed_expiry == int(old_expiry) + 10


def test_repeated_heartbeat_renewals_preserve_the_immutable_lease_span(lifecycle, monkeypatch):
    conn, task_id, run_id = lifecycle
    with sqlite3.connect(ar.ledger_path(), isolation_level=None) as ledger:
        created_at = ledger.execute("SELECT created_at FROM admission_requests").fetchone()[0]

    monkeypatch.setattr(ar.time, "time", lambda: int(created_at) + 10)
    assert kbd.heartbeat_worker(conn, task_id, expected_run_id=run_id)
    assert _admission_row()[1] == int(created_at) + 310
    monkeypatch.setattr(ar.time, "time", lambda: int(created_at) + 20)
    assert kbd.heartbeat_worker(conn, task_id, expected_run_id=run_id)
    assert _admission_row()[1] == int(created_at) + 320


@pytest.mark.parametrize("path", ["quota", "provider", "crash", "unknown", "timeout", "expired", "orphan", "stale"])
def test_zero_retry_failure_paths_require_operator_reconciliation(lifecycle, monkeypatch, path):
    conn, task_id, run_id = lifecycle
    now = int(time.time())
    conn.execute("UPDATE tasks SET max_retries=0, worker_pid=987654, worker_started_at='retained-fingerprint', started_at=?, last_heartbeat_at=NULL WHERE id=?", (now - 120, task_id))
    conn.execute("UPDATE task_runs SET started_at=?, worker_pid=987654, worker_started_at='retained-fingerprint' WHERE id=?", (now - 120, run_id))
    monkeypatch.setattr(kbd, "_worker_alive", lambda *_: False)
    monkeypatch.setattr(kb, "_worker_alive", lambda *_: False)
    exit_sweeps = {
        "quota": ("rate_limited", kb.KANBAN_RATE_LIMIT_EXIT_CODE),
        "provider": ("terminal_provider", kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE),
        "crash": ("nonzero_exit", 1),
        "unknown": ("unknown", None),
    }
    reclaimers = {
        "quota": lambda: _exit_reclaim(conn, kbd, exit_sweeps["quota"]),
        "provider": lambda: _exit_reclaim(conn, kbd, exit_sweeps["provider"]),
        "crash": lambda: _exit_reclaim(conn, kbd, exit_sweeps["crash"]),
        "unknown": lambda: _exit_reclaim(conn, kbd, exit_sweeps["unknown"]),
        "timeout": lambda: _timeout_reclaim(conn, task_id, kbd),
        "expired": lambda: _expired_reclaim(conn, task_id, now, kb),
        "orphan": lambda: _orphan_reclaim(conn, task_id, kbd),
        "stale": lambda: kbd.detect_stale_running(conn, stale_timeout_seconds=1, signal_fn=lambda *_: None),
    }
    reclaimed = reclaimers[path]()
    if path in exit_sweeps:
        assert (reclaimed.rate_limited if path == "quota" else reclaimed.crashed) == [task_id]
    else:
        assert task_id in reclaimed
    task = kb.get_task(conn, task_id)
    assert task.status == "blocked"
    assert task.block_kind == "capability"
    events = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='blocked' ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    assert events is not None and "reconcil" in events[0].lower()
    kb.recompute_ready(conn)
    assert kb.get_task(conn, task_id).status == "blocked"
    assert kb.claim_task(conn, task_id) is None
    runs = kb.list_runs(conn, task_id)
    assert len(runs) == 1
    retained = conn.execute("SELECT worker_pid, worker_started_at FROM task_runs WHERE id=?", (run_id,)).fetchone()
    assert tuple(retained) == (987654, 'retained-fingerprint')


def _exit_reclaim(conn, dispatch, classification):
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda *_: classification)
    monkeypatch.setattr(dispatch, "_worker_log_exit_code", lambda *_, **__: None)
    try:
        return dispatch._reclaim_dead_workers(conn)
    finally:
        monkeypatch.undo()


def _timeout_reclaim(conn, task_id, dispatch):
    conn.execute("UPDATE tasks SET max_runtime_seconds=1 WHERE id=?", (task_id,))
    return dispatch.enforce_max_runtime(conn, signal_fn=lambda *_: None)


def _expired_reclaim(conn, task_id, now, db):
    conn.execute("UPDATE tasks SET claim_expires=?, worker_pid=NULL WHERE id=?", (now - 1, task_id))
    return [task_id] if db.release_stale_claims(conn, signal_fn=lambda *_: None) == 1 else []


def _orphan_reclaim(conn, task_id, dispatch):
    conn.execute("UPDATE tasks SET claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?", (task_id,))
    return dispatch.reconcile_orphaned_running(conn)


def test_zero_retry_fresh_task_gets_one_initial_claim(lifecycle):
    conn, _, _ = lifecycle
    fresh = kb.create_task(conn, title="fresh one shot", assignee="default", max_retries=0)
    kb.recompute_ready(conn)
    claimed = kb.claim_task(conn, fresh)
    assert claimed is not None
    assert len(kb.list_runs(conn, fresh)) == 1
    assert kb.claim_task(conn, fresh) is None


def test_zero_retry_dead_worker_is_blocked_without_replay(lifecycle, monkeypatch):
    conn, task_id, run_id = lifecycle
    conn.execute("UPDATE tasks SET max_retries = 0 WHERE id = ?", (task_id,))
    conn.execute(
        "UPDATE tasks SET worker_pid = ?, started_at = ? WHERE id = ?",
        (987654, int(time.time()) - 120, task_id),
    )
    monkeypatch.setattr(kbd, "_worker_alive", lambda *_args: False)

    assert kbd._reclaim_dead_workers(conn).crashed == [task_id]
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "blocked"
    events = conn.execute("SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    assert events is not None and events[0] == "crashed"
    assert len(kb.list_runs(conn, task_id)) == 1
    assert kb.claim_task(conn, task_id) is None
    assert _admission_row()[0] == "released"


def test_dead_worker_reclaim_releases_reservation(lifecycle, monkeypatch):
    conn, task_id, _run_id = lifecycle
    task = kb.get_task(conn, task_id)
    assert task is not None
    conn.execute(
        "UPDATE tasks SET worker_pid = ?, started_at = ? WHERE id = ?",
        (987654, int(time.time()) - 120, task_id),
    )
    monkeypatch.setattr(kbd, "_worker_alive", lambda *_args: False)

    assert kbd._reclaim_dead_workers(conn).crashed == [task_id]
    assert _admission_row()[0] == "released"


@pytest.mark.parametrize("terminal", ["manual_reclaim", "archive", "schedule", "delete"])
def test_explicit_terminal_transitions_release_reservation(lifecycle, terminal):
    conn, task_id, _run_id = lifecycle

    if terminal == "manual_reclaim":
        assert kb.reclaim_task(conn, task_id, signal_fn=lambda *_args: None)
    elif terminal == "archive":
        assert kb.archive_task(conn, task_id, signal_fn=lambda *_args: None)
    elif terminal == "schedule":
        assert kb.schedule_task(conn, task_id, reason="operator pause")
    else:
        assert kb.delete_task(conn, task_id)

    assert _admission_row()[0] == "released"


def test_timeout_releases_attempt_scoped_reservation(lifecycle, monkeypatch):
    conn, task_id, run_id = lifecycle
    old = int(time.time()) - 30
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = ?, started_at = ? WHERE id = ?", (987654, old, task_id))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = ?", (old, run_id))
        conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = ?", (task_id,))
    monkeypatch.setattr(kbd, "_worker_alive", lambda *_args: False)

    assert task_id in kbd.enforce_max_runtime(conn, signal_fn=lambda *_args: None)
    assert _admission_row()[0] == "released"


def test_stale_and_orphan_reclaims_release_reservation(lifecycle, monkeypatch):
    conn, task_id, run_id = lifecycle
    old = int(time.time()) - 1_000
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ?, last_heartbeat_at = NULL WHERE id = ?", (old, task_id))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = ?", (old, run_id))
    assert kbd.detect_stale_running(conn, stale_timeout_seconds=1) == [task_id]
    assert _admission_row()[0] == "released"

    retried = kb.claim_task(conn, task_id)
    assert retried is not None and retried.current_run_id is not None
    admission = ar.admit_writer(
        request_id=ar.kanban_request_id(task_id, retried.current_run_id), caller=AdmissionCaller.KANBAN,
        workspace=str(_worktree(Path(conn.execute("PRAGMA database_list").fetchone()[2]).parent)), priority=0,
        writer_id=f"kanban:{task_id}", kanban_cap=1,
    )
    assert admission.state == "running"
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_lock = NULL, claim_expires = NULL WHERE id = ?", (task_id,))
    assert kbd.reconcile_orphaned_running(conn) == [task_id]
    assert _admission_row()[0] == "released"


def test_heartbeat_noops_for_uninitialized_inherited_ledger(lifecycle, monkeypatch):
    conn, task_id, run_id = lifecycle
    empty_ledger = ar.ledger_path().with_name("empty-admission.db")
    empty_ledger.touch()
    monkeypatch.setenv("HERMES_ADMISSION_LEDGER", str(empty_ledger))

    assert kbd.heartbeat_worker(conn, task_id, expected_run_id=run_id)
    assert ar.renew_kanban_admission(task_id, run_id) is False
    assert ar.release_kanban_admission(task_id, run_id) is False


def test_expired_claim_reclaim_releases_and_allows_a_new_admission(lifecycle):
    conn, task_id, _run_id = lifecycle
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 1, task_id))

    assert kb.release_stale_claims(conn) == 1
    assert _admission_row()[0] == "released"
    assert _running_occupancy() == 0
    retried = kb.claim_task(conn, task_id)
    assert retried is not None and retried.current_run_id is not None
    admission = ar.admit_writer(
        request_id=ar.kanban_request_id(task_id, retried.current_run_id), caller=AdmissionCaller.KANBAN,
        workspace=str(_worktree(Path(conn.execute("PRAGMA database_list").fetchone()[2]).parent)), priority=0,
        writer_id=f"kanban:{task_id}", kanban_cap=1,
    )
    assert admission.state == "running"


@pytest.mark.parametrize("reclaim_path", ["manual", "expired_claim", "stale_heartbeat", "orphan"])
def test_every_reclaim_path_frees_capacity_for_the_next_attempt(lifecycle, reclaim_path):
    conn, task_id, run_id = lifecycle
    if reclaim_path == "manual":
        assert kb.reclaim_task(conn, task_id, signal_fn=lambda *_args: None)
    elif reclaim_path == "expired_claim":
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 1, task_id))
        assert kb.release_stale_claims(conn) == 1
    elif reclaim_path == "stale_heartbeat":
        old = int(time.time()) - 1_000
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET started_at = ?, last_heartbeat_at = NULL WHERE id = ?", (old, task_id))
            conn.execute("UPDATE task_runs SET started_at = ? WHERE id = ?", (old, run_id))
        assert kbd.detect_stale_running(conn, stale_timeout_seconds=1) == [task_id]
    else:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET claim_lock = NULL, claim_expires = NULL WHERE id = ?", (task_id,))
        assert kbd.reconcile_orphaned_running(conn) == [task_id]

    assert _running_occupancy() == 0
    retried = kb.claim_task(conn, task_id)
    assert retried is not None and retried.current_run_id is not None
    next_admission = ar.admit_writer(
        request_id=ar.kanban_request_id(task_id, retried.current_run_id), caller=AdmissionCaller.KANBAN,
        workspace=str(_worktree(Path(conn.execute("PRAGMA database_list").fetchone()[2]).parent)), priority=0,
        writer_id=f"kanban:{task_id}", kanban_cap=1,
    )
    assert next_admission.state == "running"
