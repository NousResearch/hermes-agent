"""One gateway process must not hold two SessionDB handles on one state.db.

Regression coverage for #98573.  ``SessionStore`` and ``GatewayRunner`` each
cached a handle per resolved path, and both resolve the SAME
``_default_db_path()``.  The process therefore held two writer connections and
two independent read pools against one file, so the descriptor budget doubled
for nothing -- and doubled again per profile on a multiplexed gateway, until a
long-lived process passed the 256 soft ``RLIMIT_NOFILE`` a service manager
hands it and unrelated code paths started failing with EMFILE while the
process stayed alive.

The runner now borrows the store's handle and caches only the async wrapper.
Ownership follows: the store closes the connection, the runner does not.
"""

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner, _SESSION_DB_UNPINNED
from gateway.session import SessionStore
from gateway.session_db_recovery import RecoverableHandleCache


def _live_count(path) -> int:
    """Live-connection count the tracking registry holds for *path*."""
    import hermes_cli.sqlite_safe_read as mod

    with mod._live_lock:
        return mod._live_connections.get(mod._key(path), 0)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A gateway home under tmp_path, with path resolution going through it."""
    import hermes_state

    root = tmp_path / "hermes"
    root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    # The suite-wide fixture re-points DEFAULT_DB_PATH, which trips the
    # deliberate escape hatch in _default_db_path() and would pin every lookup
    # to one fixed path. Restore the import-time snapshot so resolution runs
    # through get_hermes_home() the way production does; HERMES_HOME above
    # keeps it inside tmp_path. Same reasoning as
    # test_multiplex_session_db_profile_scope.py.
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    return root


@pytest.fixture
def store(home):
    with patch("gateway.session.SessionStore._ensure_loaded"):
        s = SessionStore(sessions_dir=home / "sessions", config=GatewayConfig())
    s._loaded = True
    yield s
    s.close_all_db_handles()


def _runner_with(store) -> GatewayRunner:
    """A GatewayRunner with only what the handle path touches.

    Built with ``object.__new__`` deliberately: ``GatewayRunner.__init__``
    starts platforms, executors and schedulers, none of which this contract
    involves, and the method under test already supports this shape (it
    rebuilds the cache when ``__init__`` did not).
    """
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    runner._session_db_handles = {}
    runner._session_db_handles_lock = threading.Lock()
    runner._session_db_handle_cache = RecoverableHandleCache(
        handles=runner._session_db_handles,
        lock=runner._session_db_handles_lock,
    )
    runner._session_db_init_error = None
    return runner


def test_runner_borrows_the_stores_handle_instead_of_opening_a_second(store):
    """The runner's SessionDB must BE the store's, not a twin of it."""
    store_db = store._db
    assert store_db is not None, "fixture must have a usable SQLite handle"
    path = Path(store_db.db_path)
    before = _live_count(path)
    assert before >= 1, "the store's writer connection should be live"

    runner = _runner_with(store)
    wrapper = runner._open_session_db_for_active_scope()

    assert wrapper is not None
    assert wrapper._db is store_db, (
        "the runner opened its own SessionDB; one process now holds two writer "
        "connections and two read pools against one state.db"
    )
    assert _live_count(path) == before, (
        f"live connections went {before} -> {_live_count(path)}; resolving the "
        f"runner's handle must not cost a descriptor"
    )


def test_runner_shutdown_sweep_leaves_the_borrowed_handle_open(store):
    """Ownership: the store closes its connection, the runner must not.

    The shutdown sequence sweeps the store first and the runner second, so a
    runner that closed the borrowed handle would be closing an already-closed
    connection -- harmless today, and a use-after-close the moment the two
    sweeps are reordered or one of them is made conditional.
    """
    store_db = store._db
    path = Path(store_db.db_path)
    runner = _runner_with(store)
    assert runner._open_session_db_for_active_scope() is not None

    runner.close_all_session_db_handles()

    assert _live_count(path) >= 1, "the runner closed the handle the store owns"
    assert store_db._conn is not None, "borrowed writer connection was closed"
    # The wrapper cache is still drained -- not closing is not the same as not
    # forgetting.
    assert runner._session_db_handles == {}


def test_runner_without_a_session_store_still_opens_its_own(home):
    """Lightweight runners (no store wired) keep the standalone behaviour."""
    runner = object.__new__(GatewayRunner)
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    runner._session_db_handles = {}
    runner._session_db_handles_lock = threading.Lock()
    runner._session_db_handle_cache = RecoverableHandleCache(
        handles=runner._session_db_handles,
        lock=runner._session_db_handles_lock,
    )
    runner._session_db_init_error = None

    wrapper = runner._open_session_db_for_active_scope()
    try:
        assert wrapper is not None
        assert wrapper._db is not None
    finally:
        runner.close_all_session_db_handles()


def test_unavailable_store_handle_does_not_resurrect_a_second_open(store):
    """When the store has no handle, the runner reports it -- it does not open one."""
    store._db = None  # pins the store's handle to "unavailable"
    runner = _runner_with(store)

    assert runner._open_session_db_for_active_scope() is None
    assert runner._session_db_handles == {}, (
        "a duplicate handle was cached on the store's failure path"
    )
