"""Regression coverage for recoverable gateway SessionDB opens (#93088)."""

from __future__ import annotations

import json
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


def test_close_all_rejects_and_closes_inflight_success() -> None:
    cache = RecoverableHandleCache()
    path = Path("state.db")
    entered = threading.Event()
    release = threading.Event()
    handle = object()
    closed: list[object] = []
    result: list[object | None] = []

    def opener():
        entered.set()
        assert release.wait(timeout=5)
        return handle

    thread = threading.Thread(target=lambda: result.append(cache.get(path, opener)))
    thread.start()
    assert entered.wait(timeout=5)
    cache.close_all(closed.append)
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result == [None]
    assert closed == [handle]
    assert cache.status_for(path) == "unknown"


def test_close_all_preserves_inflight_failure() -> None:
    cache = RecoverableHandleCache()
    path = Path("state.db")
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    failure = OSError("original open failure")

    def opener():
        entered.set()
        assert release.wait(timeout=5)
        raise failure

    def run() -> None:
        try:
            cache.get(path, opener, raise_on_error=True)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(timeout=5)
    cache.close_all(lambda handle: None)
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == [failure]
    assert cache.status_for(path) == "unknown"


def test_recovered_db_rows_survive_fallback_structural_save(monkeypatch, tmp_path) -> None:
    import hermes_state
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionEntry, SessionSource, SessionStore, _now

    db_path = tmp_path / "state.db"
    sessions_dir = tmp_path / "sessions"
    scope = str(sessions_dir.resolve())
    now = _now()
    durable = SessionEntry(
        session_key="agent:main:telegram:dm:durable",
        session_id="durable-session",
        platform=Platform.TELEGRAM,
        chat_type="dm",
        created_at=now,
        updated_at=now,
        origin=SessionSource(platform=Platform.TELEGRAM, chat_id="durable"),
    )
    deleted = SessionEntry(
        session_key="agent:main:telegram:dm:deleted",
        session_id="deleted-session",
        platform=Platform.TELEGRAM,
        chat_type="dm",
        created_at=now,
        updated_at=now,
        origin=SessionSource(platform=Platform.TELEGRAM, chat_id="deleted"),
    )
    changed = SessionEntry(
        session_key="agent:main:telegram:dm:changed",
        session_id="changed-before-recovery",
        platform=Platform.TELEGRAM,
        chat_type="dm",
        created_at=now,
        updated_at=now,
        origin=SessionSource(platform=Platform.TELEGRAM, chat_id="changed"),
    )
    database = hermes_state.SessionDB(db_path=db_path)
    for entry in (durable, deleted, changed):
        database.save_gateway_routing_entry(
            entry.session_key,
            json.dumps(entry.to_dict()),
            scope=scope,
        )
    database.close()
    sessions_dir.mkdir()
    (sessions_dir / "sessions.json").write_text(
        json.dumps(
            {
                deleted.session_key: deleted.to_dict(),
                changed.session_key: changed.to_dict(),
            }
        ),
        encoding="utf-8",
    )

    real_session_db = hermes_state.SessionDB
    calls = 0

    def fail_once_session_db(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary open failure")
        return real_session_db(db_path=db_path)

    monkeypatch.setattr(hermes_state, "SessionDB", fail_once_session_db)
    monkeypatch.setattr(hermes_state, "_default_db_path", lambda: db_path)
    store = SessionStore(
        sessions_dir,
        GatewayConfig(sessions_dir=sessions_dir, write_sessions_json=False),
    )
    store._ensure_loaded()
    store._db_handle_cache._unavailable[db_path].next_retry_at = 0

    current = SessionEntry(
        session_key="agent:main:telegram:dm:current",
        session_id="current-session",
        platform=Platform.TELEGRAM,
        chat_type="dm",
        created_at=now,
        updated_at=now,
        origin=SessionSource(platform=Platform.TELEGRAM, chat_id="current"),
    )
    with store._lock:
        store._entries.pop(deleted.session_key)
        store._entries[changed.session_key].session_id = "changed-during-fallback"
        store._entries[current.session_key] = current
        store._save()

    rows = store._db.load_gateway_routing_entries(scope=scope)
    assert set(rows) == {durable.session_key, changed.session_key, current.session_key}
    assert store._entries[durable.session_key].session_id == durable.session_id
    assert (
        json.loads(rows[changed.session_key])["session_id"]
        == "changed-during-fallback"
    )
    assert store._entries[current.session_key].session_id == current.session_id
    store.close_all_db_handles()
