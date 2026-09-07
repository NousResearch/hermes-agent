"""Fresh-home dashboard boot must not race two first openers on state.db.

On a HERMES_HOME with no state.db yet (first boot of a new profile or
install) the lifespan starts ``statedb-eager-reconcile`` (a ``SessionDB``
open, which quarantines a zero-byte state.db) and ``hosted-room-startup``
(``prune_disbanded_rooms`` -> ``hosted_rooms_common.connect`` -> a plain
``sqlite3.connect`` on the same file). When the hosted-room connect wins it
creates the 0-byte file; the reconcile thread cannot see that connection in
the in-process live-connection registry, treats the file as zeroed, renames
it away and creates a fresh one. Both inodes then share the ``-wal``/``-shm``
sidecars (SQLite names them from the path string), the second WAL opener
truncates the ``-shm`` the first has mapped, and the next wal-index access
is a Bus error.

Three properties close that window; each is pinned here without
reproducing the crash:

* the hosted-room store opens state.db through the tracked helper, so the
  zeroed check's ``has_live_connection`` guard sees it (the contract
  ``tests/test_zeroed_state_db.py`` pins for tracked connections);
* the lifespan lets hosted-room startup run only after the eager reconcile
  has finished initialising state.db;
* lifespan shutdown joins its startup threads, so none of them runs on past
  the home it was started for.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server as web_server_mod

# Daemon threads the dashboard lifespan starts; each may open state.db.
_STARTUP_THREADS = (
    "statedb-eager-reconcile",
    "hosted-room-startup",
    "local-runtime-boot",
)


@pytest.fixture(autouse=True)
def _join_lifespan_startup_threads():
    """Never let a startup thread outlive the test's HERMES_HOME.

    Only matters when an assertion fails before the lifespan joined them.
    """
    before = set(threading.enumerate())
    yield
    for thread in set(threading.enumerate()) - before:
        if thread.name.startswith(_STARTUP_THREADS):
            thread.join(timeout=30.0)


def test_hosted_room_store_connection_is_tracked_while_open(tmp_path, monkeypatch):
    """The startup-path connect is visible to ``has_live_connection``.

    That registry is what ``is_zeroed_state_db`` consults before it
    quarantines a 0-byte file, so an untracked first opener is exactly the
    connection the reconcile thread would rename out from under.
    """
    from gateway import hosted_rooms
    from hermes_cli.sqlite_safe_read import has_live_connection

    db = tmp_path / "state.db"
    live_while_pruning: list[bool] = []
    real_prune = hosted_rooms._prune_disbanded_rooms_locked

    def prune_and_observe(conn, **kwargs):
        live_while_pruning.append(has_live_connection(db))
        return real_prune(conn, **kwargs)

    monkeypatch.setattr(hosted_rooms, "_prune_disbanded_rooms_locked", prune_and_observe)

    assert not db.exists()
    hosted_rooms.prune_disbanded_rooms(db)

    assert live_while_pruning == [True]
    assert not has_live_connection(db)
    assert db.stat().st_size > 0


def test_hosted_room_read_connection_is_tracked_while_open(tmp_path, monkeypatch):
    """The read path reuses the same opener contract as the write path."""
    from gateway import hosted_rooms
    from hermes_cli.sqlite_safe_read import has_live_connection

    db = tmp_path / "state.db"
    hosted_rooms.prune_disbanded_rooms(db)  # initialise the store

    seen: list[bool] = []
    real_read_connection = hosted_rooms._read_connection

    def read_connection_and_observe(db_path):
        conn = real_read_connection(db_path)
        seen.append(has_live_connection(db))
        return conn

    monkeypatch.setattr(hosted_rooms, "_read_connection", read_connection_and_observe)
    hosted_rooms.list_rooms(db)

    assert seen == [True]
    assert not has_live_connection(db)


def test_fresh_home_hosted_room_startup_waits_for_state_db_init(monkeypatch):
    """On an empty home, hosted-room startup runs only after state.db exists.

    The reconcile is held back so a hosted-room start that does not wait
    for it is observed deterministically rather than by scheduling luck.
    """
    from hermes_constants import get_hermes_home
    from tui_gateway import methods_groups

    state_db = get_hermes_home() / "state.db"
    assert not state_db.exists()

    release_reconcile = threading.Event()
    reconcile_entered = threading.Event()
    hosted_started = threading.Event()
    observed: dict[str, object] = {}
    real_reconcile = web_server_mod._eager_reconcile_own_session_db

    def held_reconcile(*args, **kwargs):
        release_reconcile.wait(timeout=2.0)
        reconcile_entered.set()
        return real_reconcile(*args, **kwargs)

    def record_hosted_start():
        observed["reconcile_entered"] = reconcile_entered.is_set()
        observed["state_db_size"] = (
            state_db.stat().st_size if state_db.exists() else None
        )
        hosted_started.set()
        release_reconcile.set()
        return None

    monkeypatch.setattr(web_server_mod, "_warm_gateway_module", lambda: None)
    monkeypatch.setattr(
        web_server_mod, "_eager_reconcile_own_session_db", held_reconcile
    )
    monkeypatch.setattr(
        methods_groups, "start_hosted_room_service", record_hosted_start
    )

    with TestClient(web_server_mod.app, raise_server_exceptions=False):
        assert hosted_started.wait(timeout=30.0)
        assert reconcile_entered.wait(timeout=30.0)

    assert observed["reconcile_entered"] is True, (
        "hosted-room startup ran before the eager state.db reconcile"
    )
    assert observed["state_db_size"], (
        f"hosted-room startup saw no initialised state.db: {observed}"
    )
    assert state_db.stat().st_size > 0


def test_lifespan_shutdown_joins_startup_threads(monkeypatch):
    """No startup thread survives the lifespan's exit.

    Each held thread finishes only after the test starts leaving the
    lifespan, a little later than the shutdown work takes, so a lifespan
    that does not join it exits while it is still alive.
    """
    import hermes_cli.local_runtime.bootstrap as bootstrap
    from tui_gateway import methods_groups

    leaving = threading.Event()
    real_reconcile = web_server_mod._eager_reconcile_own_session_db

    def reconcile_after_leaving(*args, **kwargs):
        leaving.wait(timeout=30.0)
        time.sleep(0.5)
        return real_reconcile(*args, **kwargs)

    def local_runtime_after_leaving(_config):
        leaving.wait(timeout=30.0)
        time.sleep(0.5)

    monkeypatch.setattr(web_server_mod, "_warm_gateway_module", lambda: None)
    monkeypatch.setattr(
        web_server_mod, "_eager_reconcile_own_session_db", reconcile_after_leaving
    )
    monkeypatch.setattr(methods_groups, "start_hosted_room_service", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_local_runtime", local_runtime_after_leaving)

    with TestClient(web_server_mod.app, raise_server_exceptions=False):
        alive_inside = {thread.name for thread in threading.enumerate()}
        assert {"statedb-eager-reconcile", "local-runtime-boot"} <= alive_inside
        leaving.set()

    alive_after = sorted(
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(_STARTUP_THREADS)
    )
    assert alive_after == [], (
        f"lifespan exited with startup threads still running: {alive_after}"
    )
