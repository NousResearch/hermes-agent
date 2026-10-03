"""Punctuated queries find their row instead of silently returning nothing.

Regression: the FTS5 sanitizer let '-', '.' and '`' reach MATCH unquoted (they are not bareword
characters): '-rf' parses as a column filter, a stray dot is a syntax error. The resulting
OperationalError was answered with "no results" before any fallback ran — so session_search for
'.env', 'git push --force' or a sentence ending in a period came back empty.
"""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_search import SessionSearchMixin

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


def test_sanitize_punctuation_contract():
    """Pin the sanitizer output: the quoted retry is the safety net, not the primary path.

    The end-to-end row tests above pass with step 5b removed (the retry absorbs the syntax
    error and still finds the row), so the sanitisation itself needs a direct contract:
    '-', '.' and '`' outside step-5 phrases become whitespace (not FTS5 syntax), and
    operator runs collapse. Step 5's quoted phrases keep their dots/hyphens untouched.
    """
    s = SessionSearchMixin._sanitize_fts5_query
    assert s(".env") == "env"
    assert s("rm -rf").split() == ["rm", "rf"]
    assert s("fixed it.") == "fixed it"
    assert s("`deploy`") == "deploy"
    assert s("push AND NOT pull") == "push NOT pull"
    assert s("push OR OR force") == "push OR force"
    assert s("my-app.config") == '"my-app.config"'


def test_corruption_on_the_quoted_retry_still_fails_open(db, monkeypatch):
    """The quoted retry runs inside the syntax-error handler; corruption it hits must still reach
    the fail-open path (answer from canonical rows), not escape as an exception."""
    real_read_all = db._read_all
    errors = [sqlite3.OperationalError("fts5: syntax error near \".\""),
              sqlite3.DatabaseError('fts5: corrupt structure record for table "messages_fts"')]

    def _read_all(sql, params=()):
        if errors and "MATCH" in sql:
            raise errors.pop(0)
        return real_read_all(sql, params)

    monkeypatch.setattr(db, "_read_all", _read_all)
    rows = db.search_messages("deploy")
    assert not errors
    assert rows and rows[0]["session_id"] == "s1"
