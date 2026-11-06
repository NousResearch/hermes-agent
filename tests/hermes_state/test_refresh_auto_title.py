"""Focused contracts for provenance-safe automatic title refreshes."""

import threading

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

    for empty_title in (None, "", " \n\t ", "\u200b"):
        assert not db.refresh_auto_title("session", empty_title, source=SessionDB.TITLE_SOURCE_LLM)
        assert _title_state(db, "session") == ("Existing Derived", "derived")


def test_refresh_rejects_non_automatic_sources_and_missing_sessions(db):
    _auto_title(db, "known", "Known Title")

    with pytest.raises(ValueError, match="invalid automatic title source"):
        db.refresh_auto_title("known", "Nope", source="user")
    assert not db.refresh_auto_title("missing", "No Session", source="llm")


def test_refresh_skips_non_lineage_collisions_without_losing_current_title(db):
    """A non-lineage conflict is transient for automatic refreshes: skip (False),
    keep both titles, never raise — the titler thread must survive collisions."""
    _auto_title(db, "target", "Current Title")
    _auto_title(db, "other", "Taken Title")

    assert not db.refresh_auto_title("target", "Taken Title", source="llm")
    assert _title_state(db, "target") == ("Current Title", "llm")
    assert _title_state(db, "other") == ("Taken Title", "llm")

    # The collision can clear later (the holder is re-titled): a retry then lands.
    assert db.refresh_auto_title("other", "Renamed By Titler", source="llm")
    assert db.refresh_auto_title("target", "Taken Title", source="llm")
    assert _title_state(db, "target") == ("Taken Title", "llm")


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


def test_refresh_protects_legacy_untitled_source_compression_ancestor(db):
    """A pre-provenance ancestor (title_source NULL ranks as user) keeps its
    title: only derived/llm ancestors may be cleared by a descendant refresh."""
    _auto_title(db, "parent", "Legacy Conversation", SessionDB.TITLE_SOURCE_DERIVED)
    db._conn.execute("UPDATE sessions SET title_source = NULL WHERE id = 'parent'")
    db._conn.commit()
    _auto_title(db, "child", "Child Provisional Title", SessionDB.TITLE_SOURCE_DERIVED)
    _link_compression(db, "parent", "child")

    assert not db.refresh_auto_title("child", "Legacy Conversation", source="llm")
    assert _title_state(db, "parent") == ("Legacy Conversation", None)
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


def test_concurrent_refreshes_never_double_write_one_title(tmp_path):
    """Two handles racing the same target title: the duplicate check lives inside
    the write transaction, so exactly one refresh lands and the title stays unique."""
    db_path = tmp_path / "state.db"
    handle_a = SessionDB(db_path=db_path)
    handle_b = SessionDB(db_path=db_path)
    try:
        _auto_title(handle_a, "a", "Old Title A")
        _auto_title(handle_a, "b", "Old Title B")
        barrier = threading.Barrier(2)
        results = {}

        def _refresh(handle, session_id):
            barrier.wait(timeout=30)
            results[session_id] = handle.refresh_auto_title(
                session_id, "Contested Title", source="llm"
            )

        threads = [
            threading.Thread(target=_refresh, args=(handle_a, "a")),
            threading.Thread(target=_refresh, args=(handle_b, "b")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive()

        # Whichever ordering SQLite serializes, one refresh wins and one loses.
        assert sorted(results.values()) == [False, True]
        holders = handle_a._conn.execute(
            "SELECT id FROM sessions WHERE title = 'Contested Title'"
        ).fetchall()
        assert len(holders) == 1
        winner = holders[0]["id"]
        assert results[winner] is True
        assert _title_state(handle_a, winner) == ("Contested Title", "llm")
        loser = "b" if winner == "a" else "a"
        assert _title_state(handle_a, loser) == (f"Old Title {loser.upper()}", "llm")
    finally:
        handle_a.close()
        handle_b.close()
