"""Regression tests for SQLite builds without FTS5 trigram tokenizer."""

import hermes_state
from hermes_state import SessionDB


def test_session_db_falls_back_when_trigram_tokenizer_is_unavailable(tmp_path, monkeypatch):
    broken_trigram_sql = hermes_state.FTS_TRIGRAM_SQL.replace(
        "tokenize='trigram'",
        "tokenize='definitely_missing_tokenizer'",
    )
    fallback_sql = broken_trigram_sql.replace(
        "    content,\n    tokenize='definitely_missing_tokenizer'",
        "    content",
    )
    monkeypatch.setattr(hermes_state, "FTS_TRIGRAM_SQL", broken_trigram_sql)
    monkeypatch.setattr(hermes_state, "FTS_TRIGRAM_FALLBACK_SQL", fallback_sql)

    db = SessionDB(tmp_path / "state.db")
    db.create_session("fallback-session", source="cli", model="test-model")
    db.append_message("fallback-session", "user", "ordinary fallback search needle")

    results = db.search_messages("needle", limit=5)

    assert results
    assert results[0]["session_id"] == "fallback-session"
