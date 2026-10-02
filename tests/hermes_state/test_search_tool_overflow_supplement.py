"""An explicit ``role_filter=['tool']`` search answers from FTS first and only then
supplements the bounded tail, instead of always full-scanning ``messages`` via LIKE.

Tool rows are indexed up to ``FTS_TOOL_CONTENT_PREFIX_CHARS`` of ``content``
(``_FTS_NEW_INDEXED_CONTENT_SQL``, shared by the v23 and legacy DDL), with
``tool_name``/``tool_calls`` indexed whole. A term past that boundary is therefore
invisible to MATCH, which is why the route used to skip FTS entirely. These tests pin
the replacement contract: same recall, but the LIKE scan runs only when MATCH
underfills the page and is bounded to rows that can actually hide a match.
"""

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_TOOL_CONTENT_PREFIX_CHARS


def _long_message(prefix: str, tail: str) -> str:
    padding = "padding " * (FTS_TOOL_CONTENT_PREFIX_CHARS // len("padding ") + 8)
    return f"{prefix} {padding} {tail}"


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    if not session_db._fts_enabled:
        session_db.close()
        pytest.skip("SQLite FTS5 unavailable")
    session_db.create_session("session", source="cli")
    try:
        yield session_db
    finally:
        session_db.close()


@pytest.fixture
def spy(monkeypatch):
    """Count supplement invocations without changing its behaviour."""
    calls = []
    original = SessionDB._search_tool_overflow

    def counting(self, query, limit, **filters):
        calls.append(query)
        return original(self, query, limit, **filters)

    monkeypatch.setattr(SessionDB, "_search_tool_overflow", counting)
    return calls


def test_term_inside_the_prefix_is_served_without_the_like_supplement(db, spy):
    """The common case: MATCH fills the page, so the bounded scan never runs."""
    tool_id = db.append_message(
        "session", role="tool", content=_long_message("prefix-token", "tail-token"),
        tool_name="terminal")

    assert [row["id"] for row in db.search_messages("prefix-token", role_filter=["tool"], limit=1)] == [tool_id]
    assert spy == []


def test_term_past_the_prefix_is_still_found(db, spy):
    """Recall is preserved: the supplement recovers what MATCH cannot see."""
    tool_id = db.append_message(
        "session", role="tool", content=_long_message("prefix-token", "tail-token"),
        tool_name="terminal")

    assert db.search_messages("tail-token") == []
    assert [row["id"] for row in db.search_messages("tail-token", role_filter=["tool"])] == [tool_id]
    assert len(spy) == 1  # the query arrives FTS-sanitized, so pin the call, not its spelling


def test_supplement_does_not_duplicate_a_row_match_already_returned(db):
    """A row whose term appears both inside and past the prefix is returned once."""
    tool_id = db.append_message(
        "session", role="tool", content=_long_message("echo-token", "echo-token"),
        tool_name="terminal")

    assert [row["id"] for row in db.search_messages("echo-token", role_filter=["tool"])] == [tool_id]


def test_supplement_honours_the_role_filter(db):
    """Only tool rows are truncated, so the scan must not widen the caller's filter."""
    db.append_message("session", role="assistant", content="an assistant row mentioning shared-tail")
    tool_id = db.append_message(
        "session", role="tool", content=_long_message("t", "shared-tail"), tool_name="terminal")

    assert [row["id"] for row in db.search_messages("shared-tail", role_filter=["tool"])] == [tool_id]


def test_supplement_honours_a_source_filter(db):
    """``_search_filter_clauses`` applies to the supplement like every other route."""
    db.create_session("other", source="telegram")
    db.append_message(
        "other", role="tool", content=_long_message("x", "scoped-tail"), tool_name="terminal")
    kept = db.append_message(
        "session", role="tool", content=_long_message("y", "scoped-tail"), tool_name="terminal")

    rows = db.search_messages("scoped-tail", role_filter=["tool"], source_filter=["cli"])
    assert [row["id"] for row in rows] == [kept]


def test_tool_name_and_tool_calls_need_no_supplement(db, spy):
    """Both are indexed whole, so a match there is a plain FTS hit."""
    tool_id = db.append_message(
        "session", role="tool", content=_long_message("p", "t"),
        tool_name="distinctive_tool_name")

    assert [row["id"] for row in db.search_messages(
        "distinctive_tool_name", role_filter=["tool"], limit=1)] == [tool_id]
    assert spy == []
