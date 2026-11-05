"""The SQLite store stamps its own profile onto new session rows (#99222).

Every profile-tree ``state.db`` belongs to exactly one profile, so when a
creation path passes no ``profile_name`` the store derives its own owner
instead of persisting NULL. NULL rows minted after the one-shot #94724
legacy-owner backfill stayed NULL forever and vanished from profile-keyed
consumers (desktop sidebar scope matching, ``@session:<profile>/<id>`` deep
links). Stores outside the profile tree must NOT guess: they keep NULL.
"""

import contextlib
import sqlite3

import pytest

from persistence import ProfileScope
from persistence.sqlite import SQLiteBackend
from persistence.sqlite.conversation._store.compression import publish_compression_child
from persistence.sqlite.conversation._store.sessions import backfill_null_session_profiles, create_session


@pytest.fixture
def hermes_root(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    (root / "profiles" / "workprof").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


@contextlib.contextmanager
def _store(db_path):
    """A writable SQLite store on *db_path*, closed on exit."""
    scope = ProfileScope(profile_dir=db_path.parent, name=db_path.parent.name)
    backend = SQLiteBackend({scope.hermes_home_key: db_path})
    try:
        yield backend.open(scope)
    finally:
        backend.close()


def _profile_of(db_path, session_id):
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT profile_name FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def test_default_store_stamps_default(hermes_root):
    with _store(hermes_root / "state.db") as store:
        create_session(store.holder(), "s_default", source="cli")
    assert _profile_of(hermes_root / "state.db", "s_default") == "default"


def test_named_profile_store_stamps_own_name(hermes_root):
    db_path = hermes_root / "profiles" / "workprof" / "state.db"
    with _store(db_path) as store:
        create_session(store.holder(), "s_prof", source="desktop")
    assert _profile_of(db_path, "s_prof") == "workprof"


def test_explicit_profile_name_wins(hermes_root):
    with _store(hermes_root / "state.db") as store:
        create_session(store.holder(), "s_explicit", source="cli", profile_name="llm-wiki")
    assert _profile_of(hermes_root / "state.db", "s_explicit") == "llm-wiki"


def test_store_outside_profile_tree_never_guesses(hermes_root, tmp_path):
    db_path = tmp_path / "elsewhere" / "state.db"
    db_path.parent.mkdir()
    with _store(db_path) as store:
        create_session(store.holder(), "s_outside", source="cli")
    assert _profile_of(db_path, "s_outside") is None


def test_compression_child_of_null_parent_is_stamped(hermes_root):
    db_path = hermes_root / "state.db"
    with _store(db_path) as store:
        holder = store.holder()
        create_session(holder, "s_parent", source="cli")
        # Simulate a legacy pre-ownership parent row.
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE sessions SET profile_name = NULL WHERE id = ?", ("s_parent",)
        )
        conn.commit()
        conn.close()
        publish_compression_child(
            holder,
            parent_session_id="s_parent",
            child_session_id="s_child",
            source="cli",
            messages=[{"role": "user", "content": "hi"}],
            require_compression_lease=False,
        )
    assert _profile_of(db_path, "s_child") == "default"


def test_peer_self_heal_insert_is_stamped(hermes_root):
    db_path = hermes_root / "state.db"
    with _store(db_path) as store:
        # No prior row: the #82616 self-heal INSERT creates it.
        store.routing.record_session_peer(
            "s_selfheal",
            source="telegram",
            user_id="u1",
            session_key="k1",
            chat_id="c1",
        )
    assert _profile_of(db_path, "s_selfheal") == "default"


def test_legacy_backfill_still_targets_only_null(hermes_root):
    """The one-shot #94724 backfill contract is unchanged: explicit owners are
    never overwritten, and new rows no longer regenerate its input."""
    db_path = hermes_root / "state.db"
    with _store(db_path) as store:
        holder = store.holder()
        create_session(holder, "s_new", source="cli")
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE sessions SET profile_name = NULL WHERE id = 's_new'")
        conn.commit()
        conn.close()
        assert backfill_null_session_profiles(holder, "workprof") == 1
        assert backfill_null_session_profiles(holder, "workprof") == 0
    assert _profile_of(db_path, "s_new") == "workprof"
