"""Session search must find terms that carry ordinary punctuation.

Every one of these characters makes FTS5's query parser raise
``fts5: syntax error near "<c>"`` when it reaches ``MATCH`` outside a quoted
phrase, and ``search_messages`` catches ``OperationalError`` and returns ``[]``.
So the failure was invisible: no error, just "no results" for terms that are
plainly in the transcript.

``_sanitize_fts5_query`` quoted ``.``/``-``/``_`` runs but nothing else, which
left the whole rest of the class silently empty — issue numbers, file paths,
apostrophes, emails, percentages, prices, commas, question marks.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_state import SessionDB

# ASCII punctuation FTS5's query parser rejects outside a quoted phrase
# (``alpha<c>beta`` -> ``fts5: syntax error near "<c>"``). Spelled out here so
# this file exercises the contract directly rather than importing whatever the
# implementation currently believes.
FTS5_REJECTED_PUNCT = "!\"#$%&'(),-./:;<=>?@[\]^`{|}~+"

SESSION_ID = "20260802_120000_fff666"
CONTENT = (
    "see issue #123 and src/main.py — don't ship at 50%, "
    "ping alice@example.com, ok? cost $5 a,b chat-send my-app.config.ts sp_new"
)


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session(SESSION_ID, source="cli")
    session_db.append_message(SESSION_ID, "user", CONTENT)
    session_db.append_message(SESSION_ID, "assistant", "noted")
    return session_db


@pytest.mark.parametrize(
    "query",
    [
        "#123",
        "src/main.py",
        "don't",
        "50%",
        "alice@example.com",
        "ok?",
        "$5",
        "a,b",
    ],
)
def test_punctuated_terms_are_findable(db, query):
    """The regression: these are in the transcript and must be found."""
    assert db.search_messages(query, limit=5)


@pytest.mark.parametrize("char", list(FTS5_REJECTED_PUNCT))
def test_every_punctuation_char_produces_a_valid_match_expression(db, char):
    """Whole-class guard: no character may reach MATCH as a syntax error."""
    expression = SessionDB._sanitize_fts5_query(f"alpha{char}beta")
    if not expression:
        return  # sanitized away entirely — nothing reaches MATCH
    with db._lock:
        db._conn.execute(
            "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH ?",
            (expression,),
        ).fetchone()


def test_existing_hyphen_and_dot_quoting_still_applies(db):
    assert db.search_messages("chat-send", limit=5)
    assert db.search_messages("my-app.config.ts", limit=5)
    assert db.search_messages("sp_new", limit=5)


def test_prefix_search_keeps_its_star_outside_the_phrase(db):
    """``*`` is FTS5's prefix operator — quoting must not swallow it."""
    assert SessionDB._sanitize_fts5_query("chat-send*") == '"chat-send"*'
    assert db.search_messages("chat-send*", limit=5)


def test_user_quoted_phrase_and_booleans_survive(db):
    assert db.search_messages('"don\'t ship"', limit=5)
    assert db.search_messages("issue AND main", limit=5)
    assert SessionDB._sanitize_fts5_query("issue AND main") == "issue AND main"


def test_punctuation_only_token_is_dropped_not_quoted(db):
    """An empty phrase is itself an FTS5 syntax error."""
    expression = SessionDB._sanitize_fts5_query("issue --- main")

    assert '""' not in expression
    with db._lock:
        db._conn.execute(
            "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH ?",
            (expression,),
        ).fetchone()


def test_plain_word_query_is_left_alone(db):
    assert SessionDB._sanitize_fts5_query("issue") == "issue"
    assert db.search_messages("issue", limit=5)
