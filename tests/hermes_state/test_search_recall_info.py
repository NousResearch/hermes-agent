"""``search_messages(recall_info=...)`` says which route produced the rows and whether a bound stopped it.

FTS answers are ranked; the canonical LIKE fallback is an unranked, row- and time-bounded scan. A caller
must be able to tell "no more matches" from "the bounded scan stopped here" without inferring it.
"""

import sqlite3
from contextlib import closing

import pytest

import hermes_state_search
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    with closing(SessionDB(tmp_path / "state.db")) as session_db:
        session_db.create_session("s1", source="cli")
        for n in range(3):
            session_db.append_message("s1", "user", f"cobalt lantern note {n}")
        session_db.append_message("s1", "user", "unrelated words")
        yield session_db


def _search(db, query="cobalt lantern", **kw):
    info = {"stale": "cleared on entry"}
    rows = db.search_messages(query, recall_info=info, **kw)
    return rows, info


def test_fts_route_reports_fts_and_row_bound(db):
    rows, info = _search(db, limit=10)
    assert len(rows) == 3
    assert info == {"source": "fts", "fallback_reason": None, "truncated": False, "deadline_hit": False,
                    "canonical_gap_rows": 0}
    rows, info = _search(db, limit=2)
    assert len(rows) == 2 and info["source"] == "fts" and info["truncated"] is True


def test_disabled_fts_reports_canonical_fallback_and_its_bound(db):
    db._fts_enabled = False
    rows, info = _search(db, limit=10)
    assert len(rows) == 3
    assert info["source"] == "canonical_fallback" and info["fallback_reason"] == "fts_unavailable"
    assert info["truncated"] is False and info["deadline_hit"] is False
    rows, info = _search(db, limit=3)
    assert len(rows) == 3 and info["truncated"] is True  # the row bound, not proof of completeness


def test_missing_fts_table_reports_query_error_fallback(db, monkeypatch):
    read_all = db._read_all

    def missing_fts(sql, params=()):
        if "messages_fts" in sql:
            raise sqlite3.OperationalError("no such table: messages_fts")
        return read_all(sql, params)

    monkeypatch.setattr(db, "_read_all", missing_fts)
    rows, info = _search(db)
    assert len(rows) == 3
    assert (info["source"], info["fallback_reason"]) == ("canonical_fallback", "fts_query_error")


def test_stale_fts_and_tool_role_report_their_reason(db):
    _, info = _search(db, role_filter=["tool"])
    assert (info["source"], info["fallback_reason"]) == ("canonical_fallback", "tool_role")
    db._fts_stale = True
    _, info = _search(db)
    assert (info["source"], info["fallback_reason"]) == ("canonical_fallback", "fts_stale")


def test_canonical_deadline_raises_and_reports_deadline_hit(db, monkeypatch):
    db._conn.executemany(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES ('s1', 'user', ?, ?)",
        [(f"filler row {n}", 1.0 + n) for n in range(5000)])
    db._conn.commit()
    db._fts_enabled = False
    monkeypatch.setattr(hermes_state_search, "_CANONICAL_SEARCH_TIMEOUT_SECONDS", 0.0)
    info: dict = {}
    with pytest.raises(TimeoutError):
        db.search_messages("cobalt lantern", recall_info=info)
    assert info["source"] == "canonical_fallback" and info["deadline_hit"] is True


def test_recall_info_is_optional(db):
    assert len(db.search_messages("cobalt lantern")) == 3
