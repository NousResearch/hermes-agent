"""Regression tests for #57596 — terminate host-local workers on block/archive.

When a kanban task transitions to ``blocked`` or ``archived``, the dispatcher
must SIGTERM the associated host-local worker process instead of only clearing
``worker_pid`` in the database and leaving the process running.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def _claim_host_worker(conn, task_id: str, *, pid: int = 4242) -> str:
    host = kb._claimer_id().split(":", 1)[0]
    claimer = f"{host}:worker"
    claimed = kb.claim_task(conn, task_id, claimer=claimer)
    assert claimed is not None
    kb._set_worker_pid(conn, task_id, pid)
    return claimer


def test_block_task_terminates_host_local_worker(conn, monkeypatch):
    calls: list[tuple] = []

    def _fake_terminate(pid, claim_lock, *, signal_fn=None):
        calls.append((pid, claim_lock, signal_fn))
        return {"termination_attempted": True, "host_local": True, "terminated": True}

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _fake_terminate)

    tid = kb.create_task(conn, title="running", assignee="w")
    claimer = _claim_host_worker(conn, tid)

    assert kb.block_task(conn, tid, reason="needs human") is True
    assert calls == [(4242, claimer, None)]
    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "blocked"
    assert task.worker_pid is None


def test_archive_task_terminates_host_local_worker(conn, monkeypatch):
    calls: list[tuple] = []

    def _fake_terminate(pid, claim_lock, *, signal_fn=None):
        calls.append((pid, claim_lock, signal_fn))
        return {"termination_attempted": True, "host_local": True, "terminated": True}

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _fake_terminate)

    tid = kb.create_task(conn, title="running", assignee="w")
    claimer = _claim_host_worker(conn, tid)

    assert kb.archive_task(conn, tid) is True
    assert calls == [(4242, claimer, None)]
    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "archived"
    assert task.worker_pid is None


def test_block_task_noops_termination_without_worker_handles(conn, monkeypatch):
    calls: list[tuple] = []

    def _track(pid, claim_lock, *, signal_fn=None):
        calls.append((pid, claim_lock))
        return {}

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _track)

    tid = kb.create_task(conn, title="ready", assignee="w")
    conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (tid,))
    assert kb.block_task(conn, tid, reason="stuck") is True
    assert calls == [(None, None)]
