"""A full routing save must not write the sessions.json mirror on the event loop.

``_persist_routing_data`` rewrites the multi-MB legacy ``sessions.json`` mirror (atomic write + fsync +
``os.replace``) after committing state.db. Full saves are still reached synchronously from coroutines
(adapter observe paths call ``SessionStore.get_or_create_session`` inline), so on a loaded disk the
rename stalls the whole gateway. Once state.db has committed the mirror may lag: it goes to a single
latest-wins worker. Without a state.db commit the mirror is the only copy and stays synchronous.
"""

import asyncio
import json
import threading
import time

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionStore


class _DB:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def replace_gateway_routing_entries(self, entries, scope=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("state.db down")


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = SessionStore(tmp_path, GatewayConfig(sessions_dir=tmp_path))
    s._test_db = _DB()
    monkeypatch.setattr(SessionStore, "_routing_scope", lambda self: "test")
    monkeypatch.setattr(
        SessionStore, "_routing_db_method",
        lambda self, name: getattr(getattr(self, "_test_db", None), name, None))
    yield s
    s.fence_sessions_json_mirror()


class _HeldWrite:
    def __init__(self, store, monkeypatch):
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.writes = []
        real = SessionStore._save_sessions_json

        def held(this, data):
            self.entered.set()
            self.gate.wait(10.0)
            self.writes.append((threading.current_thread().name, dict(data)))
            return real(this, data)

        monkeypatch.setattr(SessionStore, "_save_sessions_json", held)


def _mirror(tmp_path):
    return json.loads((tmp_path / "sessions.json").read_text())


def test_loop_caller_does_not_wait_on_the_mirror_write(store, tmp_path, monkeypatch):
    held = _HeldWrite(store, monkeypatch)

    async def scenario():
        ticked = asyncio.Event()

        async def sibling():
            await asyncio.sleep(0)
            ticked.set()

        asyncio.create_task(sibling())
        store._persist_routing_data({"k": {"session_id": "s1"}}, 1)  # returns with the write held
        await asyncio.wait_for(ticked.wait(), 5.0)
        return threading.current_thread().name

    loop_thread = asyncio.run(scenario())
    assert held.entered.wait(5.0)
    held.gate.set()
    store.fence_sessions_json_mirror()
    assert held.writes and held.writes[0][0] != loop_thread
    assert _mirror(tmp_path)["k"] == {"session_id": "s1"}
    assert store._persisted_routing_generation == 1


def test_gate_proof_off_loop_caller_writes_synchronously(store, tmp_path, monkeypatch):
    held = _HeldWrite(store, monkeypatch)
    threading.Timer(0.3, held.gate.set).start()
    t0 = time.monotonic()
    store._persist_routing_data({"k": {"session_id": "s1"}}, 1)
    assert time.monotonic() - t0 >= 0.25  # the barrier really holds the write
    assert held.writes[0][0] == threading.current_thread().name
    assert _mirror(tmp_path)["k"] == {"session_id": "s1"}


def test_latest_snapshot_wins_and_an_older_one_never_overwrites_it(store, tmp_path, monkeypatch):
    held = _HeldWrite(store, monkeypatch)

    async def scenario():
        for gen in range(1, 5):
            store._persist_routing_data({"k": {"session_id": f"s{gen}"}}, gen)

    asyncio.run(scenario())
    held.gate.set()
    store.fence_sessions_json_mirror()
    # A newer synchronous write, then a stale lane snapshot, must leave the newer one on disk.
    store._persist_routing_data({"k": {"session_id": "s9"}}, 9)
    store._write_sessions_json_mirror({"k": {"session_id": "stale"}}, 3)
    assert _mirror(tmp_path)["k"] == {"session_id": "s9"}
    assert len(held.writes) <= 3  # superseded intermediate snapshots were skipped


def test_without_a_state_db_commit_the_mirror_stays_synchronous(store, tmp_path, monkeypatch):
    store._test_db = _DB(fail=True)

    async def scenario():
        store._persist_routing_data({"k": {"session_id": "only-copy"}}, 1)
        # Written before returning: it is the only durable copy.
        return _mirror(tmp_path)["k"]

    assert asyncio.run(scenario()) == {"session_id": "only-copy"}
