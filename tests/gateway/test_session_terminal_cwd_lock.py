"""A turn's terminal cwd persists without blocking routing lookups (#158204)."""

import threading
from types import SimpleNamespace

import pytest

import hermes_constants
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_terminal_cwd import persist_turn_cwd
from tools.terminal_tool import clear_session_cwd, record_session_cwd


@pytest.fixture
def cwd_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    config = GatewayConfig(write_sessions_json=False)
    sessions_dir = tmp_path / "sessions"
    store = SessionStore(sessions_dir, config)
    entry = store.get_or_create_session(
        SessionSource(platform=Platform.LOCAL, chat_id="s0", chat_type="dm")
    )
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    record_session_cwd(entry.session_key, str(cwd))
    try:
        yield store, entry, cwd
    finally:
        clear_session_cwd(entry.session_key)
        store.close_all_db_handles()


def test_store_lock_is_available_during_turn_cwd_write(cwd_turn, monkeypatch):
    store, entry, _ = cwd_turn
    routing = store.persistence.routing
    entered = threading.Event()
    release = threading.Event()

    def hold_write(write):
        def wrapped(*args, **kwargs):
            if not entered.is_set():
                entered.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("test did not release the routing write")
            return write(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        routing, "upsert_routing_entry", hold_write(routing.upsert_routing_entry)
    )
    monkeypatch.setattr(
        routing, "replace_routing_entries", hold_write(routing.replace_routing_entries)
    )
    runner = SimpleNamespace(session_store=store)
    worker = threading.Thread(
        target=persist_turn_cwd,
        args=(runner, entry.session_key, entry.created_at),
        daemon=True,
    )
    worker.start()
    try:
        assert entered.wait(timeout=5), "cwd save did not reach the routing write"
        assert store._lock.acquire(timeout=2), (
            "cwd routing write held the session-store lock"
        )
        store._lock.release()
    finally:
        release.set()
        worker.join(timeout=5)
        assert not worker.is_alive(), "cwd save did not finish"


def test_turn_cwd_survives_store_restart(cwd_turn):
    store, entry, cwd = cwd_turn
    persist_turn_cwd(
        SimpleNamespace(session_store=store), entry.session_key, entry.created_at
    )
    store.close_all_db_handles()
    assert not (store.sessions_dir / "sessions.json").exists()

    restarted = SessionStore(store.sessions_dir, store.config)
    try:
        assert restarted.get_session_metadata(entry.session_key, "terminal_cwd") == str(
            cwd
        )
    finally:
        restarted.close_all_db_handles()
