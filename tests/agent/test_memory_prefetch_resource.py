"""Durable resource tracking for an external MemoryManager prefetch thread."""
import sqlite3
import threading
from contextvars import ContextVar
from agent import memory_manager as memory_module

import pytest

from agent.chat_execution_context import bind_chat_execution
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from gateway.chat_execution_recovery import ChatExecutionStore


class Provider(MemoryProvider):
    @property
    def name(self):
        return "external"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def __init__(self, body):
        self.body = body

    def prefetch(self, query, *, session_id=""):
        return self.body(query, session_id)


def started(home):
    store = ChatExecutionStore(home)
    assert store.start("key", 1, "session")
    assert store.worker_started("key", 1)
    return store


@pytest.mark.parametrize("later", ["fence", "successor"])
def test_prefetch_worker_reserves_before_thread_offer_and_finishes_after_physical_return(tmp_path, monkeypatch, later):
    store = started(tmp_path)
    entered, release = threading.Event(), threading.Event()
    ids = []
    def leaf(query, session_id):
        row = ChatExecutionStore(tmp_path).inspect_resources("key")[0]
        assert row["state"] == "started"
        assert row["session_id"] == "session"
        entered.set()
        release.wait()
        return "result"
    manager = MemoryManager()
    provider = Provider(leaf)
    monkeypatch.setattr(memory_module, "_MAX_INFLIGHT_CALLS", 1)
    manager._external_prefetch_timeout = 0.01
    original = memory_module.spawn_context_thread
    threads = []
    def factory(fn, **kwargs):
        rows = ChatExecutionStore(tmp_path).inspect_resources("key")
        assert len(rows) == 1 and rows[0]["state"] == "reserved"
        ids.append(rows[0]["resource_id"])
        thread = original(fn, **kwargs)
        threads.append(thread)
        return thread
    monkeypatch.setattr("agent.memory_manager.spawn_context_thread", factory)
    try:
        with bind_chat_execution(tmp_path, "key", 1):
            assert manager._prefetch_provider(provider, "q", session_id="session") == ""
            assert manager._prefetch_provider(provider, "second", session_id="session") == ""
        assert len(ChatExecutionStore(tmp_path).inspect_resources("key")) == 1
        assert entered.wait(5)
        assert ChatExecutionStore(tmp_path).inspect_resources("key")[0]["state"] == "started"
        assert store.worker_finished("key", 1)
        if later == "fence":
            assert store.begin_recovery("key", 1, "fence")
        else:
            assert store.reserve("key", 1, 1) == 2
        assert ChatExecutionStore(tmp_path).inspect_resources("key")[0]["state"] == "started"
        with bind_chat_execution(tmp_path, "key", 1 if later == "fence" else 2):
            assert manager._prefetch_provider(provider, "later-turn", session_id="session") == ""
        assert len(ids) == 1
    finally:
        release.set()
        for thread in threads:
            thread.join(10)
            assert not thread.is_alive()
    assert len(ids) == 1 and ids[0]
    assert ChatExecutionStore(tmp_path).inspect_resources("key")[0]["state"] == "finished"


@pytest.mark.parametrize("later", ["fence", "successor", "parent_finished"])
def test_stale_resource_start_refuses_provider_body(tmp_path, monkeypatch, later):
    store = started(tmp_path)
    called = []
    manager = MemoryManager()
    provider = Provider(lambda *args: called.append(args))
    from agent.memory_manager import spawn_context_thread as original
    threads = []
    def factory(fn, **kwargs):
        changes = {"fence": lambda: store.begin_recovery("key", 1, "stop"),
                   "successor": lambda: store.reserve("key", 1, 1),
                   "parent_finished": lambda: store.worker_finished("key", 1)}
        assert changes[later]()
        thread = original(fn, **kwargs)
        threads.append(thread)
        return thread
    monkeypatch.setattr("agent.memory_manager.spawn_context_thread", factory)
    try:
        with bind_chat_execution(tmp_path, "key", 1):
            with pytest.raises(RuntimeError, match="admission"):
                manager._prefetch_provider(provider, "q", session_id="session")
    finally:
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
    assert called == []
    assert store.inspect_resources("key")[0]["state"] == "reserved"


def test_thread_start_failure_leaves_reserved_resource(tmp_path, monkeypatch):
    store = started(tmp_path)
    manager = MemoryManager()
    called = []
    class FailedThread:
        def start(self):
            raise RuntimeError("thread start failed")
    monkeypatch.setattr("agent.memory_manager.spawn_context_thread", lambda *args, **kwargs: FailedThread())
    with bind_chat_execution(tmp_path, "key", 1):
        with pytest.raises(RuntimeError, match="thread start failed"):
            manager._prefetch_provider(Provider(lambda *args: called.append(args)), "q")
    assert called == []
    rows = store.inspect_resources("key")
    assert len(rows) == 1 and rows[0]["state"] == "reserved"


def test_provider_exception_remains_nonfatal_through_prefetch_all():
    manager = MemoryManager()
    provider = Provider(lambda *args: (_ for _ in ()).throw(ValueError("provider failure")))
    manager._providers = [provider]
    assert manager.prefetch_all("q", session_id="session") == ""


def test_unbound_legacy_prefetch_runs_normally():
    manager = MemoryManager()
    assert manager._prefetch_provider(Provider(lambda q, s: "legacy"), "q", session_id="s") == "legacy"


def test_reservation_failure_starts_no_provider_body(tmp_path, monkeypatch):
    started(tmp_path)
    called = []
    def broken(self):
        raise sqlite3.OperationalError("offline")
    monkeypatch.setattr(ChatExecutionStore, "_db", broken)
    manager = MemoryManager()
    with bind_chat_execution(tmp_path, "key", 1):
        with pytest.raises(sqlite3.OperationalError):
            manager._prefetch_provider(Provider(lambda *args: called.append(args)), "q")
    assert called == []


@pytest.mark.parametrize("phase", ["reserve", "start", "finish", "construct", "provider", "normal"])
def test_actual_prefetch_persistence_and_construction_boundaries(tmp_path, monkeypatch, phase):
    store = started(tmp_path)
    manager = MemoryManager()
    bodies, offers, threads = [], [], []
    def leaf(query, session_id):
        row = ChatExecutionStore(tmp_path).inspect_resources("key")[0]
        assert (row["kind"], row["state"]) == ("memory_prefetch_thread", "started")
        bodies.append("body")
        if phase == "provider":
            raise ValueError("provider failure")
        return "ok"
    original = memory_module.spawn_context_thread
    def factory(fn, **kwargs):
        row = ChatExecutionStore(tmp_path).inspect_resources("key")[0]
        assert row["state"] == "reserved"
        assert kwargs["args"][0].resource_id == row["resource_id"]
        offers.append("offered")
        if phase == "construct":
            raise RuntimeError("construction failure")
        thread = original(fn, **kwargs)
        threads.append(thread)
        return thread
    monkeypatch.setattr(memory_module, "spawn_context_thread", factory)
    methods = {"reserve": "reserve_concurrent_tool_resource", "start": "start_concurrent_tool_resource",
               "finish": "finish_concurrent_tool_resource"}
    def broken(*a, **kw):
        raise sqlite3.OperationalError("persistence failure")
    if phase in methods:
        monkeypatch.setattr(ChatExecutionStore, methods[phase], broken)
    errors = {"reserve": sqlite3.OperationalError, "start": sqlite3.OperationalError,
              "construct": RuntimeError, "provider": ValueError}
    try:
        with bind_chat_execution(tmp_path, "key", 1):
            if phase in errors:
                with pytest.raises(errors[phase]):
                    manager._prefetch_provider(Provider(leaf), "q")
            else:
                assert manager._prefetch_provider(Provider(leaf), "q") == "ok"
    finally:
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
    assert offers == ([] if phase == "reserve" else ["offered"])
    assert bodies == ([] if phase in {"reserve", "start", "construct"} else ["body"])
    expected = {"reserve": [], "start": ["reserved"], "finish": ["started"],
                "construct": ["reserved"], "provider": ["finished"], "normal": ["finished"]}
    assert [row["state"] for row in store.inspect_resources("key")] == expected[phase]
    assert store.inspect("key", 1)["effects"] == "unknown"


def test_prefetch_captures_profile_identity_before_actual_thread_start(tmp_path, monkeypatch):
    from agent.chat_execution_context import observe_tool_before_execution
    home, other = tmp_path / "home", tmp_path / "other"
    store, other_store = started(home), started(other)
    manager = MemoryManager()
    marker = ContextVar("prefetch_profile_probe", default="absent")
    threads, seen = [], []
    original = memory_module.spawn_context_thread
    def factory(fn, **kwargs):
        handle = kwargs["args"][0]
        assert handle.home == str(home.resolve())
        thread = original(fn, **kwargs)
        real_start = thread.start
        def start():
            token = marker.set("foreign-profile")
            try:
                with bind_chat_execution(other, "key", 1):
                    real_start()
            finally:
                marker.reset(token)
        thread.start = start
        threads.append(thread)
        return thread
    monkeypatch.setattr(memory_module, "spawn_context_thread", factory)
    def leaf(*args):
        assert observe_tool_before_execution()
        seen.append(marker.get())
        return "ok"
    token = marker.set("original-profile")
    try:
        with bind_chat_execution(home, "key", 1):
            assert manager._prefetch_provider(Provider(leaf), "q") == "ok"
    finally:
        marker.reset(token)
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
    assert seen == ["original-profile"]
    assert store.inspect_resources("key")[0]["state"] == "finished"
    assert other_store.inspect_resources("key") == []
    with store._db() as db:
        assert db.execute("SELECT tool_observed FROM executions WHERE session_key='key' AND generation=1").fetchone()[0] == 1
    with other_store._db() as db:
        assert db.execute("SELECT tool_observed FROM executions WHERE session_key='key' AND generation=1").fetchone()[0] == 0


def test_distinct_prefetch_workers_have_unique_resources(tmp_path):
    store = started(tmp_path)
    manager = MemoryManager()
    provider = Provider(lambda *a: "ok")
    with bind_chat_execution(tmp_path, "key", 1):
        assert manager._prefetch_provider(provider, "q", session_id="session") == "ok"
        assert manager._prefetch_provider(provider, "q", session_id="session") == "ok"
    rows = store.inspect_resources("key")
    assert len(rows) == 2
    assert rows[0]["resource_id"] != rows[1]["resource_id"]
    assert all(row["state"] == "finished" for row in rows)
    assert store.inspect("key", 1)["effects"] == "unknown"
