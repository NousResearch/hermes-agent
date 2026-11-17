"""A kanban admission whose state.db write is slow delays the admission, never the event loop.

``run_task`` (the gateway's ``kanban.run`` handler) resolved and created the attempt's session
inline on the event loop through the synchronous ``create_local_session``. Its creation commit reaches
``SessionDB._execute_write``, which takes the store's process-wide writer lock and holds it
across ``conn.commit()``, so a slow holder elsewhere (a WAL checkpoint, an fsync on a busy disk)
stopped the loop for as long as the holder took: every session, timer and socket behind one card
admission, long enough to trip a liveness watchdog. The create RPC already commits through
``tracked_write`` (``gateway/session_controls.py``); this entry point did not.

These tests hold a real SQLite write lock on a real store and require the loop to keep
answering while the admission waits.
"""

import asyncio
import json
from itertools import pairwise
import sqlite3
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB

HOLD_SECONDS = 40.0          # what the other writer is willing to hold
PATIENCE_S = 5.0             # the admission's own write-lock patience, shortened from the real 20 s


class _Holder:
    """A second writer holding the store's write lock (``BEGIN IMMEDIATE`` until released)."""

    def __init__(self, path):
        self.path = str(path)
        self.holding = threading.Event()
        self.release = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._hold, daemon=True, name="lock-holder")

    def _hold(self):
        conn = None
        try:
            conn = sqlite3.connect(self.path, timeout=5)  # sqlite3 objects are thread-bound
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO state_meta(key,value) VALUES('holder','1') "
                "ON CONFLICT(key) DO UPDATE SET value='1'")
            self.holding.set()
            self.release.wait(HOLD_SECONDS)
            conn.commit()
        except sqlite3.Error as exc:  # surfaced to the test, never swallowed
            self.error = exc
        finally:
            if conn is not None:
                conn.close()

    def __enter__(self):
        self.thread.start()
        assert self.holding.wait(10), f"lock holder never took the write lock: {self.error!r}"
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.thread.join(timeout=10)


@dataclass
class _Receipt:
    id: str = "receipt-1"
    status: str = "queued"


def _store(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db._WRITE_PATIENCE_S = db._TRANSCRIPT_WRITE_PATIENCE_S = db._ACTIVITY_WRITE_PATIENCE_S = PATIENCE_S
    return db


def _fakes(monkeypatch, db, tmp_path, write_threads=None):
    """The minimum a ``run_task`` admission touches, with the creation commit's store write REAL."""
    from gateway import session_kanban

    def fake_stage(authority, actor, params, **_kwargs):
        return {"session_id": "local-session-1", "request_id": params["request_id"]}

    def fake_commit(store, *, epoch, receipt):
        # A real write on the real store, on whatever thread run_task commits from.
        if write_threads is not None:
            write_threads.append(threading.get_ident())
        store.create_session(receipt["session_id"], "kanban", model="m", cwd=str(tmp_path))
        return receipt

    def fake_build_policy(connection, params, config):
        policy = SimpleNamespace(request_json="{}", kanban_json=json.dumps({
            "db": str(tmp_path / "kanban.db"), "board": params["board"],
            "task_id": params["task_id"], "run_id": params["run_id"]}))
        return policy, {}

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=()):
            # no owner_admitted marker yet: the claim is admitted by this attempt
            return SimpleNamespace(fetchone=lambda: None)

    class _Task:
        id = "t_00000001"
        status = "running"
        current_run_id = 1
        claim_lock = "host:1"

    monkeypatch.setattr(session_kanban, "build_kanban_policy", fake_build_policy)
    monkeypatch.setattr("gateway.session_local.stage_local_session", fake_stage)
    monkeypatch.setattr("hermes_state_local.commit_local_session", fake_commit)
    monkeypatch.setattr("gateway.session_local_recovery.restore_local_session",
                        lambda authority, sid: SimpleNamespace(session_id=sid))
    monkeypatch.setattr("gateway.run._load_gateway_config", dict)
    monkeypatch.setattr("hermes_cli.kanban_db_connect.connect_closing", lambda _path: _Conn())
    monkeypatch.setattr("hermes_cli.kanban_db.write_txn", lambda conn: _Conn())
    monkeypatch.setattr("hermes_cli.kanban_db.get_task", lambda conn, task_id: _Task())
    monkeypatch.setattr("hermes_cli.kanban_db._append_event", lambda *a, **k: None)

    async def submit(actor, submission, _kanban_card=None):
        return _Receipt()

    async def resume(ref, params):
        return None

    authority = SimpleNamespace(db=db, epoch=1, profile_id=str(tmp_path / "profile"), submit=submit)
    connection = SimpleNamespace(
        native_owner=True,
        actor=SimpleNamespace(subject="owner", profile_id=authority.profile_id, capabilities=("session:create",)),
        authority=authority, resume=resume, submit=submit)
    params = {"board": "default", "task_id": "t_00000001", "run_id": 1, "claim_lock": "host:1"}
    return connection, params


async def _with_probe(body, interval=0.02):
    """Run *body* while a probe coroutine records every time the loop schedules it."""
    seen = []

    async def probe():
        while True:
            seen.append(time.monotonic())
            await asyncio.sleep(interval)  # health: allow HX013 -- the loop heartbeat probe; its gaps are the measurement

    task = asyncio.ensure_future(probe())
    try:
        result = await body()
    finally:
        task.cancel()
    return seen, result


def test_a_held_write_lock_does_not_stall_the_event_loop(tmp_path, monkeypatch):
    from gateway.session_kanban import run_task

    db = _store(tmp_path)
    connection, params = _fakes(monkeypatch, db, tmp_path)

    async def body():
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError) as refused:
            await asyncio.wait_for(run_task(connection, params), timeout=30)
        return time.monotonic() - started, str(refused.value)

    with _Holder(db.db_path) as holder:
        seen, (blocked_for, message) = asyncio.run(_with_probe(body))
        assert holder.error is None
    assert "locked" in message.lower(), message
    assert blocked_for >= 1.0, f"the admission never actually waited: {blocked_for:.2f}s"
    # The loop answered its probe throughout the blocked admission: these ticks are what a
    # liveness watchdog measures.
    gaps = [b - a for a, b in pairwise(seen)]
    worst = max(gaps) if gaps else blocked_for
    per_second = len(seen) / max(blocked_for, 0.001)
    assert per_second >= 10, (
        f"the loop only ran {len(seen)} probes in {blocked_for:.2f}s ({per_second:.1f}/s) "
        f"while the write was blocked; worst gap {worst:.2f}s")
    assert worst < 1.0, f"the loop went silent for {worst:.2f}s inside a blocked admission"


def test_the_creation_commit_runs_off_the_loop_thread(tmp_path, monkeypatch):
    from gateway.session_kanban import run_task

    db = _store(tmp_path)
    wrote_on: list = []
    connection, params = _fakes(monkeypatch, db, tmp_path, write_threads=wrote_on)
    loop_thread = {}

    async def body():
        loop_thread["id"] = threading.get_ident()
        return await asyncio.wait_for(run_task(connection, params), timeout=15)

    result = asyncio.run(body())
    assert result["session_id"] == "local-session-1"
    assert wrote_on, "the creation was never committed"
    assert loop_thread["id"] not in wrote_on, "the creation commit ran on the event loop's own thread"


def test_an_admission_with_the_lock_free_completes(tmp_path, monkeypatch):
    """Off the loop must not mean slower or lossy: the same admission still commits and returns."""
    from gateway.session_kanban import run_task

    db = _store(tmp_path)
    connection, params = _fakes(monkeypatch, db, tmp_path)
    result = asyncio.run(asyncio.wait_for(run_task(connection, params), timeout=15))
    assert result["session_id"] == "local-session-1"
    assert db.get_session(result["session_id"]) is not None
