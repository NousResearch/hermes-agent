"""The runner's storage interface handle and the authority built on it open no store until a verb runs."""
import sqlite3
import threading

import pytest

from gateway.run import _SESSION_DB_UNPINNED
from gateway.run_runtime_init import GatewayRuntimeInitMixin
from hermes_constants import get_hermes_home


class _Runner(GatewayRuntimeInitMixin):
    def __init__(self):
        self._session_db_pinned = _SESSION_DB_UNPINNED
        self._session_db_handles_lock = threading.Lock()


@pytest.fixture
def connects(monkeypatch):
    calls = []
    real = sqlite3.connect

    def counting(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("database"))
        return real(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", counting)
    return calls


def test_runner_without_a_session_store_has_no_handle(connects):
    runner = _Runner()
    assert runner.persistence is None
    assert connects == []


def test_session_store_owns_a_handle_per_path_that_the_runner_borrows(tmp_path, monkeypatch, connects):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    before = len(connects)
    launch = store.persistence
    assert store.persistence is launch and connects[before:] == []
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(other))  # the scope a multiplexed turn binds
    secondary = store.persistence
    assert secondary is not launch and secondary.scope.profile_dir == other.resolve()
    assert (other / "state.db").exists()  # the scope's handle opens eagerly

    runner = _Runner()
    runner.session_store = store
    assert runner.persistence is secondary
    secondary.ready()  # the runner closes none it borrowed
    assert (other / "state.db").exists()  # and the count above would see an open
    store.close_all_db_handles()
    with pytest.raises(RuntimeError):
        secondary.ready()
    store._db = None
    assert store.persistence is None


def test_api_adapter_without_a_runtime_keeps_a_handle_beside_each_cached_store(tmp_path, monkeypatch, connects):
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.platforms.api_server_store import adapter_persistence

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "seam-test-secret"}))
    try:
        assert adapter_persistence(adapter) is None  # no cached store yet
        db = adapter._ensure_store()
        before = len(connects)
        handle = adapter_persistence(adapter)
        assert connects[before:] == [] and adapter_persistence(adapter) is handle
        assert handle.scope.profile_dir == get_hermes_home().resolve()
        assert db is not None
    finally:
        adapter._close_cached_session_dbs()
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
    with pytest.raises(RuntimeError):
        handle.ready()
    assert adapter_persistence(adapter) is None


def test_pinned_session_db_has_no_handle():
    runner = _Runner()
    runner._session_db_pinned = None
    assert runner.persistence is None
