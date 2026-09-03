"""Focused contracts for source-restricted automatic title refreshes."""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _auto_title(db, session_id, title, source=SessionDB.TITLE_SOURCE_LLM):
    db.create_session(session_id, source="cli")
    assert db.set_auto_title(session_id, title, source=source)


def test_refreshes_only_when_the_existing_automatic_source_matches(db):
    _auto_title(db, "llm", "Initial LLM Title")
    _auto_title(db, "derived", "Initial Derived Title", SessionDB.TITLE_SOURCE_DERIVED)

    assert db.refresh_auto_title("llm", "  Refreshed\nLLM\tTitle  ", source="llm")
    assert db.get_session_title("llm") == "Refreshed LLM Title"
    assert db.get_session_title_source("llm") == "llm"

    assert not db.refresh_auto_title("derived", "Must Not Promote", source="llm")
    assert db.get_session_title("derived") == "Initial Derived Title"
    assert db.get_session_title_source("derived") == "derived"


def test_refresh_never_overwrites_user_or_legacy_provenance(db):
    _auto_title(db, "user", "Automatic Before Rename")
    assert db.set_session_title("user", "User Rename")

    _auto_title(db, "legacy", "Legacy Title")
    db._conn.execute("UPDATE sessions SET title_source = NULL WHERE id = 'legacy'")
    db._conn.commit()

    for session_id, expected in (("user", "User Rename"), ("legacy", "Legacy Title")):
        assert not db.refresh_auto_title(session_id, "Late Automatic Title", source="llm")
        assert db.get_session_title(session_id) == expected


def test_refresh_rejects_non_automatic_sources_and_missing_sessions(db):
    _auto_title(db, "known", "Known Title")

    with pytest.raises(ValueError, match="invalid automatic title source"):
        db.refresh_auto_title("known", "Nope", source="user")
    assert not db.refresh_auto_title("missing", "No Session", source="llm")


def test_refresh_rejects_non_lineage_collisions_without_losing_current_title(db):
    _auto_title(db, "target", "Current Title")
    _auto_title(db, "other", "Taken Title")

    with pytest.raises(ValueError, match="already in use by session other"):
        db.refresh_auto_title("target", "Taken Title", source="llm")
    assert db.get_session_title("target") == "Current Title"
    assert db.get_session_title("other") == "Taken Title"


def test_refresh_transfers_a_collision_from_a_compression_ancestor(db):
    _auto_title(db, "parent", "Conversation Title")
    _auto_title(db, "child", "Child Provisional Title")
    db._conn.execute(
        "UPDATE sessions SET ended_at = 10, end_reason = 'compression' WHERE id = 'parent'"
    )
    db._conn.execute(
        "UPDATE sessions SET parent_session_id = 'parent', started_at = 11 WHERE id = 'child'"
    )
    db._conn.commit()

    assert db.refresh_auto_title("child", "Conversation Title", source="llm")
    assert db.get_session_title("parent") is None
    assert db.get_session_title("child") == "Conversation Title"
    assert db.get_session_title_source("child") == "llm"


def test_refresh_cas_does_not_clobber_a_user_rename(db):
    """A late LLM result must lose once a user rename has changed provenance."""
    _auto_title(db, "session", "Initial Automatic Title")
    assert db.set_session_title("session", "Concurrent User Rename")

    assert not db.refresh_auto_title("session", "Late LLM Result", source="llm")
    assert db.get_session_title("session") == "Concurrent User Rename"
    assert db.get_session_title_source("session") == "user"
