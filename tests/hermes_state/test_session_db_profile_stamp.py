"""SessionDB stamps its own store's profile onto new session rows (#99222).

Every profile-tree ``state.db`` belongs to exactly one profile, so when a
creation path passes no ``profile_name`` the store derives its own owner
instead of persisting NULL. NULL rows minted after the one-shot #94724
legacy-owner backfill stayed NULL forever and vanished from profile-keyed
consumers (desktop sidebar scope matching, ``@session:<profile>/<id>`` deep
links). Stores outside the profile tree must NOT guess — they keep NULL.
"""

import sqlite3

import pytest

import hermes_state
from hermes_state import SessionDB


@pytest.fixture
def hermes_root(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    (root / "profiles" / "workprof").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    # get_default_hermes_root memoizes on (native_home, env) — the env change
    # invalidates the memo by itself, but re-point DEFAULT_DB_PATH so any
    # default-constructed SessionDB in the module under test stays sandboxed.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", root / "state.db")
    return root


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
    db = SessionDB(db_path=hermes_root / "state.db")
    try:
        db.create_session("s_default", source="cli")
    finally:
        db.close()
    assert _profile_of(hermes_root / "state.db", "s_default") == "default"


def test_named_profile_store_stamps_own_name(hermes_root):
    db_path = hermes_root / "profiles" / "workprof" / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("s_prof", source="desktop")
    finally:
        db.close()
    assert _profile_of(db_path, "s_prof") == "workprof"


def test_explicit_profile_name_wins(hermes_root):
    db = SessionDB(db_path=hermes_root / "state.db")
    try:
        db.create_session("s_explicit", source="cli", profile_name="llm-wiki")
    finally:
        db.close()
    assert _profile_of(hermes_root / "state.db", "s_explicit") == "llm-wiki"


def test_store_outside_profile_tree_never_guesses(hermes_root, tmp_path):
    db_path = tmp_path / "elsewhere" / "state.db"
    db_path.parent.mkdir()
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("s_outside", source="cli")
    finally:
        db.close()
    assert _profile_of(db_path, "s_outside") is None


def test_compression_child_of_null_parent_is_stamped(hermes_root):
    db_path = hermes_root / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("s_parent", source="cli")
        # Simulate a legacy pre-ownership parent row.
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE sessions SET profile_name = NULL WHERE id = ?", ("s_parent",)
        )
        conn.commit()
        conn.close()
        db.publish_compression_child(
            parent_session_id="s_parent",
            child_session_id="s_child",
            source="cli",
            messages=[{"role": "user", "content": "hi"}],
            require_compression_lease=False,
        )
    finally:
        db.close()
    assert _profile_of(db_path, "s_child") == "default"


def test_peer_self_heal_insert_is_stamped(hermes_root):
    db_path = hermes_root / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        # No prior row: the #82616 self-heal INSERT creates it.
        db.record_gateway_session_peer(
            "s_selfheal",
            source="telegram",
            user_id="u1",
            session_key="k1",
            chat_id="c1",
        )
    finally:
        db.close()
    assert _profile_of(db_path, "s_selfheal") == "default"


def test_legacy_backfill_still_targets_only_null(hermes_root):
    """The one-shot #94724 backfill contract is unchanged: explicit owners are
    never overwritten, and new rows no longer regenerate its input."""
    db_path = hermes_root / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("s_new", source="cli")
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE sessions SET profile_name = NULL WHERE id = 's_new'")
        conn.commit()
        conn.close()
        assert db.backfill_null_session_profiles("workprof") == 1
        assert db.backfill_null_session_profiles("workprof") == 0
    finally:
        db.close()
    assert _profile_of(db_path, "s_new") == "workprof"
