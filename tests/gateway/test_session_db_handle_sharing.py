"""One gateway process must not hold two storage handles on one state.db.

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
from persistence_assembly import close_stores_under, open_persistence, profile_scope, store_location
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session


def _live_count(path) -> int:
    """Live-connection count the tracking registry holds for *path*."""
    import hermes_cli.sqlite_safe_read as mod

    with mod._live_lock:
        return mod._live_connections.get(mod._key(path), 0)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A gateway home under tmp_path, with path resolution going through it."""
    root = tmp_path / "hermes"
    root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
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
    """The runner's handle must BE the store's, not a twin of it."""
    handle = store.persistence
    assert handle is not None, "fixture must have a usable SQLite handle"
    path = Path(store_location(handle.scope))
    before = _live_count(path)
    assert before >= 1, "the store's writer connection should be live"

    runner = _runner_with(store)
    wrapper = runner._open_session_db_for_active_scope()

    assert wrapper is not None
    assert wrapper is store.persistence, (
        "the runner opened its own store; one process now holds two writer "
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
    holder = raw_holder(store.persistence)
    path = Path(store_location(store.persistence.scope))
    runner = _runner_with(store)
    assert runner._open_session_db_for_active_scope() is not None

    runner.close_all_session_db_handles()

    assert _live_count(path) >= 1, "the runner closed the handle the store owns"
    assert holder._conn is not None, "borrowed writer connection was closed"
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
        wrapper.ready()
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


def test_a_handle_the_registry_tore_down_is_reopened_through_the_registry(store, home):
    """Profile unserve and delete force-close the profile's generation (``close_all_under``).

    Both gateway caches kept serving that dead object: the store's handle and the runner's
    async wrapper around it. The store's self-heal then reopened a writer the registry does not
    know about, so the agent's ``acquire`` got a second writer on the same file, and after a
    delete + recreate every call raised ``StateDbReplacedError`` until restart.
    """
    runner = _runner_with(store)
    first = raw_holder(store.persistence)
    seed_session(store.persistence, "before-unserve", "telegram")
    first_wrapper = runner._open_session_db_for_active_scope()
    assert first_wrapper is store.persistence
    assert close_stores_under(home) == 1

    second = raw_holder(store.persistence)

    assert second is not first
    seed_session(store.persistence, "after-unserve", "telegram")
    assert first._conn is None, "the torn-down handle was revived outside the registry"
    with open_persistence(profile_scope(home=home)) as acquired:
        assert raw_holder(acquired) is second, "one file, one writer: the agent must share the store's handle"
    second_wrapper = runner._open_session_db_for_active_scope()
    assert second_wrapper is store.persistence and second_wrapper.sessions.get("after-unserve") is not None


def test_api_server_profile_cache_reopens_a_handle_the_registry_tore_down(home):
    """The API adapter's per-home cache serves routed profiles under their runtime scope; after the
    same ``close_all_under`` and a recreate of the home, the cached handle writes to the new file."""
    import shutil

    from gateway.platforms.api_server import APIServerAdapter
    from persistence.conversation.sessions import NewSession
    from persistence_assembly import open_persistence_reader

    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._home_stores = {}
    adapter._session_db_cache_lock = threading.Lock()
    adapter._session_db_cache_closed = False
    profile = home / "profiles" / "work"
    profile.mkdir(parents=True)
    adapter._open_and_cache_session_db(profile)
    assert close_stores_under(profile) == 1
    shutil.rmtree(profile)
    profile.mkdir(parents=True)

    second = adapter._open_and_cache_session_db(profile)
    try:
        second.sessions.create(NewSession(session_id="after-unserve", source="api_server"))
        with open_persistence_reader(profile) as reader:
            assert reader.sessions.get("after-unserve") is not None
    finally:
        adapter._close_cached_session_dbs()
        close_stores_under(profile)
