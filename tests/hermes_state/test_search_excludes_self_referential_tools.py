"""Search results hide tool output that merely restates already-indexed messages.

A ``session_search`` result row is a JSON dump of snippets from other messages. Indexing it
means a later search competes with a replay of an earlier one, and the quoted messages are
themselves indexed, so the hit is never the thing the person wanted. These rows are filtered
at query time in ``_search_filter_clauses``, next to the ``display_kind='hidden'`` predicate
that exists for the same reason.
"""

import pytest

from hermes_state import SessionDB
from hermes_state_common import SEARCH_EXCLUDED_TOOL_NAMES


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


def test_session_search_output_is_not_a_search_hit(db):
    original = db.append_message(
        "session", role="assistant", content="the estuary model uses distinctive-term for routing")
    db.append_message(
        "session", role="tool", tool_name="session_search",
        content='{"success": true, "results": [{"snippet": "distinctive-term"}]}')

    assert [row["id"] for row in db.search_messages("distinctive-term")] == [original]


def test_excluded_output_is_hidden_even_for_an_explicit_tool_search(db):
    db.append_message(
        "session", role="tool", tool_name="session_search",
        content='{"results": [{"snippet": "tool-scoped-term"}]}')

    assert db.search_messages("tool-scoped-term", role_filter=["tool"]) == []


def test_other_tool_output_is_untouched(db):
    kept = db.append_message(
        "session", role="tool", tool_name="terminal", content="terminal-only-term in the output")

    assert [row["id"] for row in db.search_messages("terminal-only-term")] == [kept]
    assert [row["id"] for row in db.search_messages(
        "terminal-only-term", role_filter=["tool"])] == [kept]


def test_the_excluded_set_is_narrow(db):
    """A guard: widening this set silently removes recall, so it should stay deliberate."""
    assert SEARCH_EXCLUDED_TOOL_NAMES == ("session_search",)


def test_filter_does_not_disturb_source_or_role_filters(db):
    """The new predicate carries binds; a misordered param list would corrupt every route."""
    db.create_session("other", source="telegram")
    telegram_row = db.append_message(
        "other", role="assistant", content="combined-filter-term over telegram")
    db.append_message("session", role="assistant", content="combined-filter-term over cli")

    rows = db.search_messages(
        "combined-filter-term", source_filter=["telegram"], role_filter=["assistant"])
    assert [row["id"] for row in rows] == [telegram_row]
