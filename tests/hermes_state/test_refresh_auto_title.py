"""Focused contracts for provenance-safe automatic title refreshes."""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _auto_title(db, session_id, title, source=SessionDB.TITLE_SOURCE_LLM):
    db.create_session(session_id, source="cli")
    assert db.set_auto_title(session_id, title, source=source)


def _title_state(db, session_id):
    row = db.get_session(session_id)
    return row["title"], row["title_source"]


def _link_compression(db, parent_id, child_id):
    db._conn.execute(
        "UPDATE sessions SET ended_at = 10, end_reason = 'compression' WHERE id = ?",
        (parent_id,),
    )
    db._conn.execute(
        "UPDATE sessions SET parent_session_id = ?, started_at = 11 WHERE id = ?",
        (parent_id, child_id),
    )
    db._conn.commit()


def test_refresh_allows_equal_source_and_upgrade_but_refuses_downgrade(db):
    _auto_title(db, "session", "Initial Derived", SessionDB.TITLE_SOURCE_DERIVED)

    assert db.refresh_auto_title(
        "session", "  Refreshed\nDerived\tTitle  ", source=SessionDB.TITLE_SOURCE_DERIVED
    )
    assert _title_state(db, "session") == ("Refreshed Derived Title", "derived")

    assert db.refresh_auto_title("session", "Upgraded LLM Title", source=SessionDB.TITLE_SOURCE_LLM)
    assert _title_state(db, "session") == ("Upgraded LLM Title", "llm")

    assert not db.refresh_auto_title(
        "session", "Must Not Downgrade", source=SessionDB.TITLE_SOURCE_DERIVED
    )
    assert _title_state(db, "session") == ("Upgraded LLM Title", "llm")


def test_refresh_never_overwrites_user_or_legacy_provenance(db):
    _auto_title(db, "user", "Automatic Before Rename")
    assert db.set_session_title("user", "User Rename")

    _auto_title(db, "legacy", "Legacy Title")
    db._conn.execute("UPDATE sessions SET title_source = NULL WHERE id = 'legacy'")
    db._conn.commit()

    for session_id, expected in (
        ("user", ("User Rename", "user")),
        ("legacy", ("Legacy Title", None)),
    ):
        assert not db.refresh_auto_title(session_id, "Late Automatic Title", source="llm")
        assert _title_state(db, session_id) == expected


def test_refresh_preserves_hidden_canonical_bot_chat_identity(db):
    _auto_title(
        db,
        "canonical",
        SessionDB.CANONICAL_BOT_CHAT_TITLE,
        SessionDB.TITLE_SOURCE_DERIVED,
    )
    assert db.set_session_hidden("canonical", True)

    assert not db.refresh_auto_title("canonical", "New Topic", source=SessionDB.TITLE_SOURCE_LLM)
    assert _title_state(db, "canonical") == (
        SessionDB.CANONICAL_BOT_CHAT_TITLE,
        SessionDB.TITLE_SOURCE_DERIVED,
    )

    assert db.set_session_hidden("canonical", False)
    assert db.refresh_auto_title("canonical", "New Topic", source=SessionDB.TITLE_SOURCE_LLM)
    assert _title_state(db, "canonical") == ("New Topic", SessionDB.TITLE_SOURCE_LLM)


def test_refresh_does_not_steal_title_from_hidden_canonical_ancestor(db):
    _auto_title(
        db,
        "parent",
        SessionDB.CANONICAL_BOT_CHAT_TITLE,
        SessionDB.TITLE_SOURCE_DERIVED,
    )
    assert db.set_session_hidden("parent", True)
    _auto_title(db, "child", "Child Provisional Title", SessionDB.TITLE_SOURCE_DERIVED)
    _link_compression(db, "parent", "child")

    assert not db.refresh_auto_title(
        "child",
        SessionDB.CANONICAL_BOT_CHAT_TITLE,
        source=SessionDB.TITLE_SOURCE_LLM,
    )
    assert _title_state(db, "parent") == (
        SessionDB.CANONICAL_BOT_CHAT_TITLE,
        SessionDB.TITLE_SOURCE_DERIVED,
    )
    assert _title_state(db, "child") == ("Child Provisional Title", SessionDB.TITLE_SOURCE_DERIVED)


def test_empty_refresh_preserves_existing_automatic_title_and_source(db):
    _auto_title(db, "session", "Existing Derived", SessionDB.TITLE_SOURCE_DERIVED)

    for empty_title in (None, "", " \n\t "):
        assert not db.refresh_auto_title("session", empty_title, source=SessionDB.TITLE_SOURCE_LLM)
        assert _title_state(db, "session") == ("Existing Derived", "derived")


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
    assert _title_state(db, "target") == ("Current Title", "llm")
    assert _title_state(db, "other") == ("Taken Title", "llm")


@pytest.mark.parametrize(
    ("ancestor_source", "refresh_source"),
    ((SessionDB.TITLE_SOURCE_USER, SessionDB.TITLE_SOURCE_LLM),
     (SessionDB.TITLE_SOURCE_LLM, SessionDB.TITLE_SOURCE_DERIVED)),
)
def test_refresh_protects_manual_or_higher_authority_compression_ancestor(
    db, ancestor_source, refresh_source
):
    _auto_title(db, "parent", "Conversation Title", SessionDB.TITLE_SOURCE_DERIVED)
    if ancestor_source == SessionDB.TITLE_SOURCE_USER:
        assert db.set_session_title("parent", "Conversation Title")
    else:
        assert db.refresh_auto_title("parent", "Conversation Title", source=ancestor_source)
    _auto_title(db, "child", "Child Provisional Title", SessionDB.TITLE_SOURCE_DERIVED)
    _link_compression(db, "parent", "child")

    assert not db.refresh_auto_title("child", "Conversation Title", source=refresh_source)
    assert _title_state(db, "parent") == ("Conversation Title", ancestor_source)
    assert _title_state(db, "child") == ("Child Provisional Title", "derived")


def test_refresh_transfers_allowed_automatic_ancestor_and_clears_provenance(db):
    _auto_title(db, "parent", "Conversation Title", SessionDB.TITLE_SOURCE_DERIVED)
    _auto_title(db, "child", "Child Provisional Title", SessionDB.TITLE_SOURCE_DERIVED)
    _link_compression(db, "parent", "child")

    assert db.refresh_auto_title("child", "Conversation Title", source=SessionDB.TITLE_SOURCE_LLM)
    assert _title_state(db, "parent") == (None, None)
    assert _title_state(db, "child") == ("Conversation Title", "llm")


def test_refresh_cas_does_not_clobber_a_user_rename(db):
    """A late LLM result must lose once a user rename has changed provenance."""
    _auto_title(db, "session", "Initial Automatic Title")
    assert db.set_session_title("session", "Concurrent User Rename")

    assert not db.refresh_auto_title("session", "Late LLM Result", source="llm")
    assert _title_state(db, "session") == ("Concurrent User Rename", "user")
