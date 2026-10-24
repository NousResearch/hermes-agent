"""A session heartbeat tick has one durable owner across managers and processes."""

import multiprocessing
import sqlite3

import pytest

from hermes_cli import heartbeat
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    with SessionDB(db_path=tmp_path / "state.db") as store:
        monkeypatch.setattr(heartbeat, "_get_session_db", lambda: store)
        heartbeat.save_heartbeat("shared", heartbeat.HeartbeatState("tick", 60, created_at=1))
        yield store


def test_independent_managers_claim_each_due_tick_once(db):
    first = heartbeat.HeartbeatManager("shared")
    second = heartbeat.HeartbeatManager("shared")
    assert first.due_prompt(now=1000) is not None
    assert second.due_prompt(now=1001) is None
    assert heartbeat.load_heartbeat("shared").fire_count == 1
    # The loser can claim the next interval, and the former winner must see it.
    assert second.due_prompt(now=1060) is not None
    assert first.due_prompt(now=1061) is None
    state = heartbeat.load_heartbeat("shared")
    assert (state.last_fired_at, state.fire_count) == (1060, 2)
    assert first.abandon_fire() is False  # a later claim superseded the first fire
    assert second.abandon_fire() is True
    assert second.abandon_fire() is False
    assert first.due_prompt(now=1060) is not None
    assert second.due_prompt(now=1061) is None


@pytest.mark.parametrize("operation", ["pause", "resume", "clear", "set"])
@pytest.mark.parametrize("phase", ["claim", "abandon"])
def test_concurrent_edit_wins_at_the_write_boundary(db, monkeypatch, operation, phase):
    manager = heartbeat.HeartbeatManager("shared")
    if phase == "abandon":
        assert manager.due_prompt(now=1000) is not None
    write = db._write_rowcount
    edited = []

    def edit_before_write(sql, params):
        other = heartbeat.HeartbeatManager("shared")
        if operation == "set":
            other.set("replacement", 120)
        else:
            getattr(other, operation)()
        edited.append(db.get_meta("heartbeat:shared"))
        return write(sql, params)

    monkeypatch.setattr(db, "_write_rowcount", edit_before_write)
    if phase == "claim":
        assert manager.due_prompt(now=1000) is None
    else:
        assert manager.abandon_fire() is False
    assert db.get_meta("heartbeat:shared") == edited[0]


def test_failed_claim_does_not_inject_or_consume_the_tick(db, monkeypatch):
    manager = heartbeat.HeartbeatManager("shared")
    original = db.get_meta("heartbeat:shared")

    def fail_write(*args):
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(db, "_write_rowcount", fail_write)
        assert manager.due_prompt(now=1000) is None
    assert db.get_meta("heartbeat:shared") == original
    assert manager.state.fire_count == 0
    assert manager.abandon_fire() is False
    assert manager.due_prompt(now=1000) is not None
    claimed = db.get_meta("heartbeat:shared")
    with monkeypatch.context() as patch:
        patch.setattr(db, "_write_rowcount", fail_write)
        assert manager.abandon_fire() is False
    assert db.get_meta("heartbeat:shared") == claimed
    assert manager.abandon_fire() is True
    assert db.get_meta("heartbeat:shared") == original


def _claim_in_process(path, ready, reads, results):
    with SessionDB(db_path=path) as store:
        heartbeat._get_session_db = lambda: store
        manager = heartbeat.HeartbeatManager("shared")
        ready.wait(timeout=30)
        get_meta = store.get_meta

        def synchronized_read(key):
            raw = get_meta(key)
            if key == "heartbeat:shared":
                # Force both claimers to read the same due row before either writes.
                reads.wait(timeout=30)
            return raw

        store.get_meta = synchronized_read
        results.put(manager.due_prompt(now=1000))


def test_processes_claim_the_same_due_row_once(db):
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    ready, reads = ctx.Barrier(2), ctx.Barrier(2)
    workers = [ctx.Process(target=_claim_in_process, args=(db.db_path, ready, reads, results))
               for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=45)
            assert worker.exitcode == 0
        prompts = [results.get(timeout=5) for _ in workers]
        assert sum(prompt is not None for prompt in prompts) == 1
        state = heartbeat.load_heartbeat("shared")
        assert (state.last_fired_at, state.fire_count) == (1000, 1)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
        results.close()
        results.join_thread()
