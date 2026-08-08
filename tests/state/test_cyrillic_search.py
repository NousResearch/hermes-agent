"""Regression coverage for Cyrillic substring search routing."""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def test_cyrillic_detection_covers_basic_block():
    contains = SessionDB._contains_cyrillic
    assert contains("суббота") is True
    assert contains("Українська") is True
    assert contains("Български") is True
    assert contains("hello world") is False
    assert contains("") is False


def test_cyrillic_substring_query_uses_trigram_index(db):
    db.create_session(session_id="s1", source="cli")
    db.append_message(
        "s1",
        role="user",
        content="Мы едем с друзьями в субботу на озеро.",
    )
    statements = []
    db._conn.set_trace_callback(statements.append)

    results = db.search_messages("суббот")

    assert len(results) == 1
    assert results[0]["session_id"] == "s1"
    assert ">>>суббот<<<" in results[0]["snippet"]
    assert any("FROM messages_fts_trigram" in statement for statement in statements)
    assert db._describe_search_path("суббот") == "trigram"


def test_short_cyrillic_query_falls_back_to_like(db):
    db.create_session(session_id="s1", source="cli")
    db.append_message("s1", role="user", content="Журавль стоит у воды.")
    statements = []
    db._conn.set_trace_callback(statements.append)

    results = db.search_messages("Жу")

    assert len(results) == 1
    assert results[0]["session_id"] == "s1"
    assert any(
        "FROM messages m" in statement and " LIKE " in statement
        for statement in statements
    )
    assert db._describe_search_path("Жу") == "like_scan"


def test_cyrillic_trigram_preserves_source_filter(db):
    db.create_session(session_id="s1", source="cli")
    db.create_session(session_id="s2", source="telegram")
    db.append_message("s1", role="user", content="Встреча состоится в субботу.")
    db.append_message("s2", role="user", content="Поездка запланирована на субботу.")

    results = db.search_messages("суббот", source_filter=["telegram"])

    assert len(results) == 1
    assert results[0]["source"] == "telegram"


def test_cyrillic_falls_back_when_trigram_is_unavailable(db):
    db.create_session(session_id="s1", source="cli")
    db.append_message("s1", role="user", content="Встреча состоится в субботу.")
    db._trigram_available = False

    results = db.search_messages("суббот")

    assert len(results) == 1
    assert results[0]["session_id"] == "s1"
