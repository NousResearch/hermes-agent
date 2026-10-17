"""Real-process lane recovery and transactional memory admission contracts."""
import multiprocessing
import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli.kanban_lane_liveness import identity_alive, process_identity, reservation_alive
from hermes_cli.kanban_provider_lanes import Candidate, LaneLedger


@pytest.mark.platforms("linux", "macos")
def test_real_worker_identity_and_recycled_pid(monkeypatch):
    from hermes_cli import kanban_db_dispatch as dispatch

    fingerprint = dispatch._process_fingerprint(os.getpid())
    assert fingerprint is not None
    assert identity_alive(process_identity(os.getpid(), fingerprint)) is True
    assert identity_alive(process_identity(os.getpid(), "different-boot|0")) is False
    monkeypatch.setattr(dispatch, "_process_fingerprint", lambda pid: None)
    assert identity_alive(process_identity(os.getpid(), fingerprint)) is None


@pytest.mark.parametrize("identity", [None, "garbage", "null", "[]", '{}',
                                        '{"pid":true,"fingerprint":"x"}'])
def test_unknown_process_identity_is_not_dead(identity):
    assert identity_alive(identity) is None


@pytest.mark.platforms("linux", "macos")
def test_pending_reservation_recovers_retained_run_worker(tmp_path):
    from hermes_cli.kanban_db_dispatch import _process_fingerprint

    board = tmp_path / "board.db"
    fingerprint = _process_fingerprint(os.getpid())
    with sqlite3.connect(board) as conn:
        conn.execute("CREATE TABLE task_runs (id INTEGER, task_id TEXT, worker_pid INTEGER, worker_started_at TEXT)")
        conn.execute("INSERT INTO task_runs VALUES (1, 'task', ?, ?)", (os.getpid(), fingerprint))
    row = dict(board=str(board), task="task", run=1, worker=None, owner="dead-dispatcher")
    assert reservation_alive(row) is True
    assert reservation_alive({**row, "task": "another-task"}) is None
    assert reservation_alive({**row, "run": 2}) is None
    assert reservation_alive({**row, "board": "slug-not-path"}) is None
    assert reservation_alive({**row, "board": str(tmp_path / "missing.db")}) is None
    assert not (tmp_path / "missing.db").exists()


@pytest.mark.platforms("linux", "macos")
def test_worker_exit_releases_slot_on_next_admission(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(target=os.getpid)
    child.start()
    child.join(timeout=30)
    assert child.exitcode == 0
    ledger = LaneLedger(tmp_path / "ledger.db")
    route = (Candidate("anthropic", "fixture"),)
    def reserve(task):
        return ledger.reserve(board=str(tmp_path / "board.db"), run=1,
                              candidates=route, owner="owner", alive=reservation_alive,
                              task=task)
    try:
        admitted = reserve("dead")
        assert admitted is not None
        assert child.pid is not None
        token, _ = admitted
        ledger.bind_worker(token, "owner", process_identity(child.pid, "old-boot|0"))
        assert reserve("new")
        assert [r["task"] for r in ledger.snapshot()] == ["new"]
    finally:
        ledger.close()


def _memory_race(path, barrier, results, name):
    ledger = LaneLedger(Path(path))
    try:
        barrier.wait(timeout=30)
        admitted = ledger.reserve(board=name, task=name, run=1, owner=name,
            candidates=(Candidate("anthropic", "fixture"),), alive=lambda row: True,
            memory_sample=lambda: 150, minimum_free_bytes=100, worker_headroom_bytes=50)
        results.put(bool(admitted))
    finally:
        ledger.close()


def test_two_boards_cannot_spend_same_memory_headroom(tmp_path):
    path = tmp_path / "lanes.db"
    LaneLedger(path).close()
    ctx = multiprocessing.get_context("spawn")
    barrier, results = ctx.Barrier(2), ctx.Queue()
    children = [ctx.Process(target=_memory_race, args=(str(path), barrier, results, name))
                for name in ("one", "two")]
    try:
        for child in children:
            child.start()
        observations = [results.get(timeout=40) for _ in children]
        for child in children:
            child.join(timeout=30)
            assert child.exitcode == 0
        assert sorted(observations) == [False, True]
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=30)
        results.close()
        results.join_thread()


@pytest.mark.parametrize("sample", [None, True, -1, 149])
def test_atomic_memory_guard_fails_closed(tmp_path, sample):
    ledger = LaneLedger(tmp_path / "ledger.db")
    try:
        assert ledger.reserve(board="board", task="task", run=1, owner="owner",
            candidates=(Candidate("anthropic", "fixture"),), alive=lambda row: True,
            memory_sample=lambda: sample, minimum_free_bytes=100, worker_headroom_bytes=50) is None
        assert ledger.snapshot() == []
    finally:
        ledger.close()
