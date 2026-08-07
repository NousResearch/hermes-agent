"""Lazy backend re-init tests for the mem0 memory provider.

Covers the resilience fix: a mem0 backend that failed to initialize (vector
store down at session start) or was invalidated by a transport failure
mid-session is rebuilt from config on the next operation — instead of
requiring a session restart (/new) to recover.
"""

import json
import threading

import pytest

from plugins.memory.mem0 import (
    Mem0MemoryProvider,
    _REINIT_COOLDOWN_SECS,
)


class FakeBackend:
    """Minimal stand-in for a mem0 backend (search/add/close)."""

    def __init__(self, search_result=None, search_error=None, add_error=None):
        self.search_result = search_result if search_result is not None else []
        self.search_error = search_error
        self.add_error = add_error
        self.closed = False

    def search(self, *args, **kwargs):
        if self.search_error is not None:
            raise self.search_error
        return self.search_result

    def add(self, *args, **kwargs):
        if self.add_error is not None:
            raise self.add_error
        return {"results": []}

    def close(self):
        self.closed = True


def _make_provider() -> Mem0MemoryProvider:
    p = Mem0MemoryProvider()
    p._mode = "oss"
    p._config = {"oss": {"vector_store": {"provider": "qdrant"}}}
    p._user_id = "test-user"
    p._agent_id = "hermes"
    p._channel = "cli"
    return p


class TestEnsureBackend:
    def test_reinitializes_after_failed_init(self, monkeypatch):
        # Session started while the store was down: initialize() left
        # _backend = None and _init_error set. Once the store returns, the
        # next operation must rebuild the backend in place.
        p = _make_provider()
        p._backend = None
        p._init_error = "connection refused"
        p._consecutive_failures = 3

        fresh = FakeBackend(search_result=[{"id": "1", "memory": "fato", "score": 0.9}])
        monkeypatch.setattr(p, "_create_backend", lambda: fresh)

        backend = p._ensure_backend()

        assert backend is fresh
        assert p._backend is fresh
        assert p._init_error == ""
        assert p._consecutive_failures == 0

    def test_second_call_returns_cached_backend(self, monkeypatch):
        p = _make_provider()
        fresh = FakeBackend()
        monkeypatch.setattr(p, "_create_backend", lambda: fresh)

        assert p._ensure_backend() is fresh
        # Backend is alive: no further _create_backend calls.
        monkeypatch.setattr(p, "_create_backend", lambda: pytest.fail("rebuilt live backend"))
        assert p._ensure_backend() is fresh

    def test_throttle_prevents_hammering_dead_store(self, monkeypatch):
        import time as _time

        p = _make_provider()
        p._backend = None
        p._reinit_backend_at = _time.monotonic() + 60
        monkeypatch.setattr(p, "_create_backend", lambda: pytest.fail("create called while throttled"))

        assert p._ensure_backend() is None

    def test_failed_reinit_rearms_throttle(self, monkeypatch):
        import time as _time

        p = _make_provider()
        p._backend = None
        p._reinit_backend_at = 0.0
        monkeypatch.setattr(p, "_create_backend", lambda: None)

        assert p._ensure_backend() is None
        assert p._reinit_backend_at > _time.monotonic() + _REINIT_COOLDOWN_SECS - 5


class TestInvalidation:
    def test_connection_error_invalidates_backend(self):
        p = _make_provider()
        p._backend = FakeBackend(search_error=ConnectionError("Connection refused"))

        out = json.loads(p.handle_tool_call("mem0_search", {"query": "teste"}))

        assert "error" in out
        assert p._backend is None  # invalidated for lazy rebuild
        assert p._reinit_backend_at > 0

    def test_client_error_does_not_invalidate(self):
        p = _make_provider()
        p._backend = FakeBackend(search_error=ValueError("Memory not found: 123"))

        json.loads(p.handle_tool_call("mem0_search", {"query": "teste"}))

        assert p._backend is not None  # user-caused error: backend stays

    def test_tool_call_heals_after_store_returns(self, monkeypatch):
        # First call hits a dead store and invalidates the backend; the
        # store comes back; the next call rebuilds and succeeds.
        p = _make_provider()
        p._backend = FakeBackend(search_error=ConnectionError("Connection refused"))

        out1 = json.loads(p.handle_tool_call("mem0_search", {"query": "teste"}))
        assert "error" in out1
        assert p._backend is None

        # Store is back — clear the cooldown and let _create_backend succeed.
        p._reinit_backend_at = 0.0
        healthy = FakeBackend(search_result=[{"id": "1", "memory": "fato", "score": 0.9}])
        monkeypatch.setattr(p, "_create_backend", lambda: healthy)

        out2 = json.loads(p.handle_tool_call("mem0_search", {"query": "teste"}))

        assert out2["count"] == 1
        assert out2["results"][0]["memory"] == "fato"
        assert p._backend is healthy

    def test_invalidate_closes_old_backend(self):
        p = _make_provider()
        dead = FakeBackend(search_error=ConnectionError("timeout"))
        p._backend = dead

        p._invalidate_backend()

        assert dead.closed is True
        assert p._backend is None


class TestSyncTurn:
    def test_sync_reinitializes_and_records(self, monkeypatch):
        p = _make_provider()
        p._backend = None
        p._reinit_backend_at = 0.0
        added = []
        added_event = threading.Event()
        healthy = FakeBackend()

        def fake_create():
            return healthy

        def fake_add(self_, messages, **kwargs):
            added.append(messages)
            added_event.set()

        monkeypatch.setattr(p, "_create_backend", fake_create)
        monkeypatch.setattr(FakeBackend, "add", fake_add)

        p.sync_turn("user msg", "assistant msg")

        # Deterministic sync: the fake backend's add() signals completion,
        # then we join the worker thread — no polling.
        assert added_event.wait(2), "sync worker never ran"
        worker = p._sync_thread
        if worker is not None:
            worker.join(timeout=2)
        assert added[0][0]["role"] == "user"
        assert added[0][1]["role"] == "assistant"
        assert p._backend is healthy


class _GateBackend(FakeBackend):
    """Backend whose add() blocks until released — for event-synced tests."""

    def __init__(self):
        super().__init__()
        self.added = []
        self.in_add = threading.Event()
        self.release_add = threading.Event()
        self.drained = threading.Event()

    def add(self, messages, **kwargs):
        self.added.append(messages)
        self.in_add.set()
        self.release_add.wait(timeout=5)
        if len(self.added) >= 2:
            self.drained.set()
        return {"results": []}


class TestSyncBacklog:
    def test_turns_are_buffered_not_dropped_while_worker_busy(self, monkeypatch):
        # Worker is stuck in a slow extraction; two more turns arrive. The
        # newest must be coalesced into the backlog and drained after the
        # first completes — never silently dropped (the old 5s-join skip).
        p = _make_provider()
        p._backend = None
        p._reinit_backend_at = 0.0
        gate = _GateBackend()
        monkeypatch.setattr(p, "_create_backend", lambda: gate)

        p.sync_turn("t1 user", "t1 asst")
        assert gate.in_add.wait(2), "worker never started first add"

        p.sync_turn("t2 user", "t2 asst")  # buffered
        p.sync_turn("t3 user", "t3 asst")  # coalesces over t2
        gate.release_add.set()

        assert gate.drained.wait(3), "backlog never drained"
        worker = p._sync_thread
        if worker is not None:
            worker.join(timeout=2)

        contents = [[m["content"] for m in msgs] for msgs in gate.added]
        assert contents[0] == ["t1 user", "t1 asst"]
        # t2 was coalesced away; t3 (the newest) survived — nothing dropped.
        assert contents[1] == ["t3 user", "t3 asst"]
        assert len(contents) == 2

    def test_sync_retries_after_transport_failure(self, monkeypatch):
        # First add hits a dead store (backend invalidated, worker exits).
        # The next sync_turn spawns a fresh worker that re-initializes the
        # backend and delivers the turn.
        p = _make_provider()
        p._backend = None
        p._reinit_backend_at = 0.0
        calls = []
        call_event = threading.Event()

        def flaky_add(self_, messages, **kwargs):
            calls.append(messages)
            call_event.set()
            if len(calls) == 1:
                raise ConnectionError("Connection refused")
            return {"results": []}

        healthy = FakeBackend()
        monkeypatch.setattr(FakeBackend, "add", flaky_add)
        monkeypatch.setattr(p, "_create_backend", lambda: healthy)

        p.sync_turn("u1", "a1")
        assert call_event.wait(2), "first add never ran"
        worker = p._sync_thread
        if worker is not None:
            worker.join(timeout=2)
        assert p._backend is None  # invalidated for lazy rebuild

        # Cooldown elapses; the next turn triggers a fresh worker + rebuild.
        call_event.clear()
        p._reinit_backend_at = 0.0
        p.sync_turn("u2", "a2")
        assert call_event.wait(2), "retry add never ran"
        worker = p._sync_thread
        if worker is not None:
            worker.join(timeout=2)
        assert p._backend is healthy  # re-initialized
        assert [[m["content"] for m in calls[1]]] == [["u2", "a2"]]

    def test_worker_exits_when_queue_empty(self, monkeypatch):
        p = _make_provider()
        p._backend = None
        p._reinit_backend_at = 0.0
        added_event = threading.Event()
        healthy = FakeBackend()

        def fake_add(self_, messages, **kwargs):
            added_event.set()
            return {"results": []}

        monkeypatch.setattr(p, "_create_backend", lambda: healthy)
        monkeypatch.setattr(FakeBackend, "add", fake_add)

        p.sync_turn("u", "a")
        assert added_event.wait(2), "sync worker never ran"
        worker = p._sync_thread
        if worker is not None:
            worker.join(timeout=2)
        assert not worker.is_alive(), "worker leaked after draining the queue"


class TestEOFDependencyError:
    def test_eof_reports_missing_dependency(self, monkeypatch):
        # mem0 prompts interactively when a provider dependency is missing;
        # in a non-interactive session that raises EOFError from inside the
        # backend constructor. _create_backend must translate it into a
        # clear "missing dependency" error instead of a transport-looking
        # "EOF when reading a line".
        class BoomBackend:
            def __init__(self, *args, **kwargs):
                raise EOFError("EOF when reading a line")

        monkeypatch.setattr("plugins.memory.mem0._backend.OSSBackend", BoomBackend)
        p = _make_provider()

        assert p._create_backend() is None
        assert "missing provider dependency" in p._init_error
        assert "EOF when reading a line" not in p._init_error
