"""The runner's storage interface handle and the authority built on it open no store until a verb runs."""
import sqlite3
import threading

import pytest

from gateway.run import _SESSION_DB_UNPINNED
from gateway.run_runtime_init import GatewayRuntimeInitMixin
from gateway.session_authority import SessionAuthority
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


def test_runner_handle_is_per_profile_and_opens_nothing(tmp_path, monkeypatch, connects):
    runner = _Runner()
    launch = runner.persistence
    assert runner.persistence is launch
    assert launch.scope.profile_dir == get_hermes_home().resolve()

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(other))
    secondary = runner.persistence
    assert secondary is not launch
    assert secondary.scope.profile_dir == other.resolve()

    authority = SessionAuthority(runner, profile_id=str(other), instance_id="owner", epoch=1,
                                 persistence=secondary)
    assert authority.persistence is secondary
    assert connects == []
    assert not (other / "state.db").exists()
    secondary.ready()
    assert (other / "state.db").exists()

    runner.close_all_persistence_handles()
    assert runner.persistence is not secondary


def test_runner_handle_is_absent_while_it_cannot_open_and_opens_on_a_later_read(monkeypatch):
    import persistence_assembly

    runner = _Runner()
    real = persistence_assembly.open_persistence

    def fail(*args, **kwargs):
        raise ValueError("unknown persistence provider 'nope'")

    monkeypatch.setattr(persistence_assembly, "open_persistence", fail)
    assert runner.persistence is None
    monkeypatch.setattr(persistence_assembly, "open_persistence", real)
    handle = runner.persistence
    assert handle is not None and runner.persistence is handle
    runner.close_all_persistence_handles()


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
    runner.close_all_persistence_handles()
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
