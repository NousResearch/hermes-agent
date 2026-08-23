"""Regression coverage for #88532.

A multiplexed gateway serves every profile from one process.  ``SessionStore``
used to bind a single ``SessionDB`` during ``__init__``, freezing it to the
process's own root home, so a named profile's sessions were physically written
to the root ``state.db`` even though ``_profile_runtime_scope`` had already
redirected ``get_hermes_home()`` for that turn.  The rows carried the correct
``profile_name``, which is why the only visible symptom was the desktop listing
a profile's session under the default bot: the desktop reads
``profiles/<name>/state.db``, which never received the write.

These tests pin the handle to the *active* scope rather than to construction
time.  ``test_write_under_profile_scope_lands_in_profile_store`` is the one
that reproduces the report; it fails against the pre-fix code with the session
row sitting in the root store.
"""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def multiplex_homes(tmp_path, monkeypatch):
    """A root home plus a named profile home, with HERMES_HOME on the root.

    Mirrors the reported layout: one gateway process launched under the root
    home, serving a ``fitness`` profile whose store lives under
    ``profiles/fitness``.
    """
    import hermes_state

    root = tmp_path / "hermes"
    profile = root / "profiles" / "fitness"
    root.mkdir(parents=True)
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))

    # The suite-wide fixture in conftest re-points ``hermes_state.DEFAULT_DB_PATH``
    # at a fake home, which trips the deliberate escape hatch in
    # ``_default_db_path()``: a re-pointed constant wins over everything,
    # including the context-local override.  That is correct for tests that
    # want one fixed DB, but it would pin every lookup here to a single path
    # and make these assertions vacuous.  Restore the import-time snapshot so
    # the hatch is closed and resolution goes through ``get_hermes_home()``,
    # which is what production does.  ``HERMES_HOME`` above still keeps that
    # resolution inside ``tmp_path``, so no real store is ever opened.
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    return root, profile


def _make_store(root: Path) -> SessionStore:
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=root / "sessions", config=GatewayConfig())
    store._loaded = True
    return store


def _session_ids(db_path: Path) -> set:
    """Read session ids straight out of a state.db, or empty if absent."""
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT id FROM sessions").fetchall()
    except sqlite3.OperationalError:
        # No sessions table: nothing was ever written here.
        return set()
    finally:
        conn.close()
    return {r[0] for r in rows}


def test_store_uses_root_db_when_no_profile_scope_is_active(multiplex_homes):
    """Single-profile gateways are unaffected: no scope, same path as before."""
    root, _profile = multiplex_homes
    store = _make_store(root)

    assert Path(store._db.db_path) == root / "state.db"


def test_db_handle_follows_the_active_profile_scope(multiplex_homes):
    """The handle is resolved per access, not frozen at construction."""
    root, profile = multiplex_homes
    store = _make_store(root)

    # Constructed outside any scope, exactly as the gateway constructs it.
    assert Path(store._db.db_path) == root / "state.db"

    token = set_hermes_home_override(str(profile))
    try:
        assert Path(store._db.db_path) == profile / "state.db"
    finally:
        reset_hermes_home_override(token)

    # And the scope is restored once the turn's scope exits.
    assert Path(store._db.db_path) == root / "state.db"


def test_write_under_profile_scope_lands_in_profile_store(multiplex_homes):
    """The reported bug: the row must be in the profile's own file.

    This is the assertion the issue makes by hand with ``sqlite3``: the
    session for profile ``fitness`` belongs in ``profiles/fitness/state.db``
    and must NOT be in the root store.
    """
    root, profile = multiplex_homes
    store = _make_store(root)

    token = set_hermes_home_override(str(profile))
    try:
        store._db.create_session("20260817_233028_542fda58", "feishu")
    finally:
        reset_hermes_home_override(token)

    assert _session_ids(profile / "state.db") == {"20260817_233028_542fda58"}
    assert _session_ids(root / "state.db") == set()


def test_handles_are_cached_per_path(multiplex_homes):
    """One handle per profile: no reopen per message, no sharing across profiles."""
    root, profile = multiplex_homes
    store = _make_store(root)

    root_first = store._db
    root_second = store._db
    assert root_first is root_second

    token = set_hermes_home_override(str(profile))
    try:
        profile_first = store._db
        profile_second = store._db
    finally:
        reset_hermes_home_override(token)

    assert profile_first is profile_second
    assert profile_first is not root_first


def test_explicitly_pinned_handle_still_wins(multiplex_homes):
    """``store._db = ...`` remains authoritative for every subsequent read.

    Guardrail rather than a bug reproduction: a large number of existing
    tests install a fake handle or disable the DB this way, and the property
    must not quietly resolve past a deliberate assignment.
    """
    root, profile = multiplex_homes
    store = _make_store(root)

    sentinel = object()
    store._db = sentinel
    token = set_hermes_home_override(str(profile))
    try:
        assert store._db is sentinel
    finally:
        reset_hermes_home_override(token)

    # Disabling the DB (the JSONL-fallback path) must survive scope changes.
    store._db = None
    token = set_hermes_home_override(str(profile))
    try:
        assert store._db is None
    finally:
        reset_hermes_home_override(token)


def test_close_all_db_handles_sweeps_every_profile_handle(multiplex_homes):
    """Teardown must release every cached per-profile handle, not just the
    one the tearing-down task's own scope resolves.

    Follow-up hardening for the per-path cache: ``gateway/run.py``'s
    teardown path closes ``store._db`` (root scope only); the sweep closes
    the rest so secondary profiles' WAL locks are released before a
    ``--replace`` restart reopens their stores.
    """
    root, profile = multiplex_homes
    store = _make_store(root)

    root_db = store._db
    token = set_hermes_home_override(str(profile))
    try:
        profile_db = store._db
    finally:
        reset_hermes_home_override(token)
    assert root_db is not profile_db

    store.close_all_db_handles()

    # Both handles are closed (connection released) and the cache is empty,
    # so the next access opens a fresh handle rather than a dead one.
    assert root_db._conn is None
    assert profile_db._conn is None
    assert store._db_handles == {}
    fresh = store._db
    assert fresh is not root_db
    assert fresh._conn is not None
    fresh.close()


def test_runner_session_db_follows_the_active_profile_scope(multiplex_homes):
    """GatewayRunner._session_db is the same frozen-handle class of bug.

    /resume, /title, /history and session search run inside
    ``_profile_runtime_scope`` on a multiplexed gateway and must read the
    serving profile's state.db.  Exercise the property on a bare runner shell
    (full construction wires adapters and is irrelevant to the seam under
    test).
    """
    import threading

    from gateway.run import GatewayRunner, _SESSION_DB_UNPINNED

    root, profile = multiplex_homes
    runner = object.__new__(GatewayRunner)
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    runner._session_db_handles = {}
    runner._session_db_handles_lock = threading.Lock()

    root_db = runner._session_db
    assert Path(root_db._db.db_path) == root / "state.db"

    token = set_hermes_home_override(str(profile))
    try:
        profile_db = runner._session_db
        assert Path(profile_db._db.db_path) == profile / "state.db"
        # Cached per path: same wrapper identity on re-access.
        assert runner._session_db is profile_db
    finally:
        reset_hermes_home_override(token)

    assert runner._session_db is root_db

    # Pinning (how suites install fakes / disable the DB) wins across scopes.
    runner._session_db = None
    token = set_hermes_home_override(str(profile))
    try:
        assert runner._session_db is None
    finally:
        reset_hermes_home_override(token)
    runner._session_db_pinned = _SESSION_DB_UNPINNED

    runner.close_all_session_db_handles()
    assert runner._session_db_handles == {}
    assert root_db._db._conn is None
    assert profile_db._db._conn is None
