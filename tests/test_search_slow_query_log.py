"""Tests for the session-search slow-query log (salvaged from PR #65544)."""

import logging

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session(session_id="s1", source="cli", model="m")
    d.append_message("s1", role="user", content="hello graphiti 일본 MCP 정리")
    yield d
    d.close()


def test_slow_log_emitted_at_zero_threshold(db, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "0")
    with caplog.at_level(logging.INFO, logger="hermes_state"):
        rows = db.search_messages("graphiti", limit=5)
    assert rows
    slow = [r for r in caplog.records if "slow session search" in r.getMessage()]
    assert slow, "threshold 0 must log every search"
    msg = slow[0].getMessage()
    assert "path=" in msg and "rows=1" in msg


def test_no_log_under_threshold(db, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "60000")
    with caplog.at_level(logging.INFO, logger="hermes_state"):
        db.search_messages("graphiti", limit=5)
    assert not [r for r in caplog.records if "slow session search" in r.getMessage()]


def test_path_attribution(db):
    # Without the cjk tokenizer loaded, routing matches the pre-cjk shape.
    assert db._describe_search_path("graphiti OR neo4j") == "fts5"
    assert db._describe_search_path("우선순위 캘린더") == "trigram"
    assert db._describe_search_path("일본 MCP") == "like_scan"


def test_path_attribution_cjk_available(db):
    # With the bigram index available, CJK queries (including 2-char terms)
    # route to fts_cjk; lone 1-char CJK runs keep the LIKE route.
    db._fts_cjk_available = True
    try:
        assert db._describe_search_path("일본 MCP") == "fts_cjk"
        assert db._describe_search_path("우선순위 캘린더") == "fts_cjk"
        assert db._describe_search_path("가 alone") == "like_scan"
        assert db._describe_search_path("graphiti OR neo4j") == "fts5"
    finally:
        db._fts_cjk_available = False


def test_results_unchanged_by_wrapper(db, monkeypatch):
    monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "0")
    rows = db.search_messages("graphiti", limit=5)
    assert rows and rows[0]["session_id"] == "s1"
