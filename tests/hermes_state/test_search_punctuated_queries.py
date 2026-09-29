"""Punctuated queries find their row instead of silently returning nothing.

Regression: the FTS5 sanitizer let '-', '.' and '`' reach MATCH unquoted (they are not bareword
characters): '-rf' parses as a column filter, a stray dot is a syntax error. The resulting
OperationalError was answered with "no results" before any fallback ran — so session_search for
'.env', 'git push --force' or a sentence ending in a period came back empty.
"""

import pytest

from hermes_state import SessionDB

_STORED = ("add OPENAI_API_KEY to the .env file, then run git push --force and rm -rf build; "
           "the `deploy` step fixed it.")


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session(session_id="s1", source="cli", model="m")
    d.append_message("s1", role="user", content=_STORED)
    yield d
    d.close()


@pytest.mark.parametrize("query", [
    ".env", "~/.env", "git push --force", "rm -rf", "-rf build", "fixed it.", "fixed it...",
    "`deploy`", "push AND NOT pull", "push OR OR force",
])
def test_punctuated_query_finds_its_row(db, query):
    rows = db.search_messages(query)
    assert rows and rows[0]["session_id"] == "s1", query
