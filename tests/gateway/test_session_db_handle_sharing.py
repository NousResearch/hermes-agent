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
        registry_backed=True,
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
        registry_backed=True,
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


def test_a_handle_the_registry_tore_down_is_reopened_through_the_registry(store, home):
    """Profile unserve and delete force-close the profile's generation (``close_all_under``).

    Both gateway caches kept serving that dead object: the store's SessionDB and the runner's
    async wrapper around it. The store's self-heal then reopened a writer the registry does not
    know about, so the agent's ``acquire`` got a second writer on the same file, and after a
    delete + recreate every call raised ``StateDbReplacedError`` until restart.
    """
    import hermes_state_registry as registry

    runner = _runner_with(store)
    first = store._db
    first.create_session("before-unserve", source="telegram")
    first_wrapper = runner._open_session_db_for_active_scope()
    assert first_wrapper._db is first
    path = Path(first.db_path)
    assert registry.close_all_under(home) == 1

    second = store._db

    assert second is not first
    second.create_session("after-unserve", source="telegram")
    assert first._conn is None, "the torn-down handle was revived outside the registry"
    acquired = registry.acquire(path)
    try:
        assert acquired is second, "one file, one writer: the agent must share the store's handle"
    finally:
        registry.release(acquired)
    second_wrapper = runner._open_session_db_for_active_scope()
    assert second_wrapper is not first_wrapper and second_wrapper._db is second


def test_api_server_profile_cache_reopens_a_handle_the_registry_tore_down(home):
    """The API adapter's per-home cache serves routed profiles under their runtime scope; the
    same ``close_all_under`` must evict its entry too."""
    import hermes_state_registry as registry
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._session_dbs = {}
    adapter._session_db_cache_lock = threading.Lock()
    adapter._session_db_cache_closed = False
    profile = home / "profiles" / "work"
    profile.mkdir(parents=True)
    first = adapter._open_and_cache_session_db(profile)
    assert registry.close_all_under(profile) == 1

    second = adapter._open_and_cache_session_db(profile)
    try:
        assert second is not first
        second.create_session("after-unserve", source="api_server")
    finally:
        registry.close_all_under(profile)


def test_a_handle_the_registry_tears_down_mid_open_is_never_published(store, home, monkeypatch):
    """``close_all_under`` can land between the store's ``acquire`` and the cache publishing the
    handle (a profile unserve racing that profile's first routed turn). The publication then saw a
    handle the registry no longer owned, filed it as a plain never-shared one, and every later read
    served the closed connection without reopening.
    """
    import hermes_state_registry as registry

    real_acquire = registry.acquire
    opened: list = []

    def acquire_then_unserve(db_path=None):
        db = real_acquire(db_path)
        opened.append(db)
        if len(opened) == 1:
            registry.close_all_under(home)  # the unserve lands before the cache publishes
        return db

    store.close_all_db_handles()  # the first ``_db`` read below must go through the opener
    monkeypatch.setattr(registry, "acquire", acquire_then_unserve)

    assert store._db is None, "the torn-down handle was published"
    second = store._db

    assert len(opened) == 2 and second is opened[1]
    assert opened[0]._conn is None and second._conn is not None
    second.create_session("after-unserve", source="telegram")
    acquired = real_acquire(Path(second.db_path))
    try:
        assert acquired is second, "one file, one writer: the agent must share the store's handle"
    finally:
        registry.release(acquired)


def test_a_wrapper_around_a_handle_torn_down_mid_open_is_never_published(store, home, monkeypatch):
    """The same window one layer up: the runner borrows the store's handle, and the registry can
    tear it down after the store published it but before the runner's cache publishes the wrapper."""
    import hermes_state
    import hermes_state_registry as registry

    real_async = hermes_state.AsyncSessionDB
    wrapped: list = []

    def wrap_then_unserve(db):
        wrapper = real_async(db)
        wrapped.append(wrapper)
        if len(wrapped) == 1:
            registry.close_all_under(home)
        return wrapper

    monkeypatch.setattr(hermes_state, "AsyncSessionDB", wrap_then_unserve)
    runner = _runner_with(store)
    first_db = store._db

    assert runner._open_session_db_for_active_scope() is None, "the torn-down wrapper was published"
    second = runner._open_session_db_for_active_scope()

    assert len(wrapped) == 2 and second is wrapped[1] and second._db is store._db
    assert first_db._conn is None and second._db is not first_db and second._db._conn is not None


def test_a_retired_retry_keeps_the_pending_recovery_and_a_later_live_open_completes_it(store, home, monkeypatch):
    """Startup failed (``database is locked``), the retry's handle was retired before publication, the
    third open is live. The recovery owed since the startup failure must still complete: the real
    callback clears the startup error, so the pre-broadcast re-check sends no false outage warning."""
    import asyncio

    import hermes_state_registry as registry

    runner = _runner_with(store)
    runner.session_store = None
    runner._session_db_init_error = "database is locked"
    real_acquire = registry.acquire
    calls = 0

    def acquire(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("database is locked")
        db = real_acquire(*args, **kwargs)
        if calls == 2:
            registry.close_all_under(home)  # the unserve lands before the cache publishes
        return db

    monkeypatch.setattr(registry, "acquire", acquire)
    assert runner._open_session_db_for_active_scope() is None  # the startup failure
    for state in runner._session_db_handle_cache._unavailable.values():
        state.next_retry_at = 0  # past the backoff
    assert runner._open_session_db_for_active_scope() is None  # retired before publication: not cached
    healthy = runner._open_session_db_for_active_scope()  # no backoff owed after a retired retry
    assert healthy is not None and calls == 3
    healthy._db.create_session("healthy", source="telegram")
    sent = []
    runner._home_channel_transports = lambda: [("telegram", {}, "home", object())]

    async def send(*args):
        sent.append(args[3])

    runner._send_home_channel_message = send
    try:
        asyncio.run(runner._send_session_db_warning_notifications())
        assert runner._session_db_init_error is None, "the recovery owed to the startup failure was dropped"
        assert sent == [], "a false outage warning went out after the database recovered"
    finally:
        runner.close_all_session_db_handles()
