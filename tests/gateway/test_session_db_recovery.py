"""Regression coverage for recoverable gateway SessionDB opens (#93088)."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

from gateway.session_db_recovery import RecoverableHandleCache


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_failed_open_obeys_backoff_then_recovers() -> None:
    clock = _Clock()
    cache = RecoverableHandleCache(clock=clock, initial_retry_delay=2, max_retry_delay=8)
    path = Path("profile/state.db")
    handle = object()
    calls = 0

    def opener():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("private/path/state.db is unavailable")
        return handle

    assert cache.get(path, opener) is None
    assert cache.status_for(path) == "unavailable"
    clock.now = 1.99
    assert cache.get(path, opener) is None
    assert calls == 1

    clock.now = 2.0
    assert cache.get(path, opener) is handle
    assert cache.get(path, opener) is handle
    assert calls == 2
    assert cache.status_for(path) == "ok"


def test_retry_is_single_flight_for_concurrent_callers() -> None:
    clock = _Clock()
    cache = RecoverableHandleCache(clock=clock, initial_retry_delay=1, max_retry_delay=8)
    path = Path("profile/state.db")
    entered = threading.Event()
    release = threading.Event()
    handle = object()
    calls = 0

    def opener():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("first open fails")
        entered.set()
        assert release.wait(timeout=5)
        return handle

    assert cache.get(path, opener) is None
    clock.now = 1.0
    result: list[object] = []
    thread = threading.Thread(target=lambda: result.append(cache.get(path, opener)))
    thread.start()
    assert entered.wait(timeout=5)

    # The opener runs outside the state lock. Other callers observe in-flight
    # and keep using the fallback rather than opening or blocking behind it.
    assert cache.get(path, opener) is None
    assert calls == 2
    assert cache.status_for(path) == "retrying"

    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result == [handle]
    assert cache.get(path, opener) is handle
    assert calls == 2


def test_runtime_health_is_sanitized_and_recovers() -> None:
    clock = _Clock()
    cache = RecoverableHandleCache(clock=clock, initial_retry_delay=1)
    path = Path("secret/profile/state.db")
    writes: list[dict] = []
    calls = 0

    def opener():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("database disk image is malformed at secret/profile/state.db")
        return object()

    with patch("gateway.status.write_runtime_status", side_effect=lambda **kw: writes.append(kw)):
        assert cache.get(path, opener) is None
        clock.now = 1.0
        assert cache.get(path, opener) is not None

    assert writes[-1] == {"session_store": {"status": "ok"}}
    serialized = repr(writes)
    assert "secret/profile" not in serialized
    assert "malformed" not in serialized


def test_session_store_and_runner_reopen_after_failed_construction(monkeypatch, tmp_path) -> None:
    import hermes_state
    from gateway.run import GatewayRunner, _SESSION_DB_UNPINNED
    from gateway.session import SessionStore, _DB_UNPINNED

    db_path = tmp_path / "state.db"
    clock = _Clock()
    opened: list[object] = []

    def fail_once_session_db():
        if not opened:
            opened.append(None)
            raise OSError("temporary open failure")
        handle = object()
        opened.append(handle)
        return handle

    monkeypatch.setattr(hermes_state, "SessionDB", fail_once_session_db)
    monkeypatch.setattr(hermes_state, "_default_db_path", lambda: db_path)

    store = object.__new__(SessionStore)
    store._db_pinned = _DB_UNPINNED
    store._db_handles = {}
    store._db_handles_lock = threading.Lock()
    store._db_handle_cache = RecoverableHandleCache(
        handles=store._db_handles,
        lock=store._db_handles_lock,
        clock=clock,
        initial_retry_delay=1,
    )
    assert store._db is None
    assert store._db is None
    assert len(opened) == 1
    clock.now = 1.0
    assert store._db is opened[-1]

    runner_opened: list[object] = []

    def runner_fail_once():
        if not runner_opened:
            runner_opened.append(None)
            raise OSError("temporary open failure")
        handle = object()
        runner_opened.append(handle)
        return handle

    monkeypatch.setattr(hermes_state, "SessionDB", runner_fail_once)
    monkeypatch.setattr(hermes_state, "AsyncSessionDB", lambda db: ("async", db))
    runner = object.__new__(GatewayRunner)
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    runner._session_db_init_error = "temporary open failure"
    runner._session_db_handles = {}
    runner._session_db_handles_lock = threading.Lock()
    runner._session_db_handle_cache = RecoverableHandleCache(
        handles=runner._session_db_handles,
        lock=runner._session_db_handles_lock,
        clock=clock,
        initial_retry_delay=1,
    )
    assert runner._session_db is None
    assert runner._session_db is None
    assert len(runner_opened) == 1
    clock.now = 2.0
    assert runner._session_db == ("async", runner_opened[-1])
    assert runner._session_db_init_error is None


def test_non_cacheable_guard_is_retried_immediately() -> None:
    cache = RecoverableHandleCache()
    path = Path("state.db")
    calls = 0

    def opener():
        nonlocal calls
        calls += 1
        raise RuntimeError("live-system guard")

    for _ in range(2):
        try:
            cache.get(
                path,
                opener,
                non_cacheable=lambda exc: "live-system guard" in str(exc),
            )
        except RuntimeError:
            pass
    assert calls == 2
    assert cache.status_for(path) == "unknown"
