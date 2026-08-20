"""session_search must find file paths, emails and percentages.

``_sanitize_fts5_query`` quotes dotted/hyphenated terms so FTS5's tokenizer
does not split ``chat-send`` into ``chat AND send``. It matched the dotted run
with ``\\b(\\w+(?:[._-]\\w+)+)\\b`` and quoted **that substring**, which is not
the whole token whenever the term also contains a separator ``\\w`` does not
cover (``/``, ``@``, ``%``, ``\\``):

    scripts/deploy.sh  ->  scripts/"deploy.sh"
    user@example.com   ->  user@"example.com"
    61.2%              ->  "61.2"%

Each of those leaves the separator outside the quotes, where FTS5 rejects it:
``fts5: syntax error near "/"``. The search layer swallows
``sqlite3.OperationalError`` into an empty result, so the tool reported "no
matches" for content that was sitting in the transcript — file paths, emails
and percentages, i.e. exactly the terms this step exists to make matchable.

Quoting the whole token keeps the separator inside the phrase, where the
tokenizer splits it like any other non-word character.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_state import SessionDB

_sanitize = SessionDB._sanitize_fts5_query


@pytest.fixture
def fts():
    """A real FTS5 table — the parse error only exists inside SQLite."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(content)")
    for doc in (
        "the deploy script lives at scripts/deploy.sh and needs sudo",
        "user@example.com asked about the API key rotation",
        "coverage rose from 61.2% to 78.9% this sprint",
        "we fixed the bug in hermes-agent yesterday",
        "check the file my-app.config.ts for the setting",
    ):
        conn.execute("INSERT INTO t(content) VALUES (?)", (doc,))
    yield conn
    conn.close()


def _hits(conn, query: str) -> int:
    return len(
        conn.execute(
            "SELECT rowid FROM t WHERE t MATCH ?", (_sanitize(query),)
        ).fetchall()
    )


class TestSeparatorTokensAreMatchable:
    @pytest.mark.parametrize(
        "query",
        ["scripts/deploy.sh", "user@example.com", "61.2%"],
    )
    def test_query_parses_and_finds_its_document(self, fts, query):
        # Before the fix this raised OperationalError, which the search layer
        # turns into zero results.
        assert _hits(fts, query) == 1

    @pytest.mark.parametrize(
        "query",
        ["scripts/deploy.sh", "user@example.com", "61.2%"],
    )
    def test_the_whole_token_is_quoted_not_a_slice(self, query):
        assert _sanitize(query) == f'"{query}"'


class TestPreviouslyWorkingQueriesStillWork:
    @pytest.mark.parametrize(
        "query",
        ["deploy.sh", "hermes-agent", "my-app.config.ts"],
    )
    def test_plain_dotted_and_hyphenated_terms(self, fts, query):
        assert _sanitize(query) == f'"{query}"'
        assert _hits(fts, query) == 1

    def test_prefix_operator_stays_outside_the_phrase(self):
        """A trailing * is FTS5's prefix operator, not a literal asterisk."""
        assert _sanitize("deploy*") == "deploy*"
        assert _sanitize("my-app.config*") == '"my-app.config"*'

    def test_plain_words_are_untouched(self):
        assert _sanitize("hello world") == "hello world"
        assert _sanitize("retry loop") == "retry loop"

    def test_whitespace_runs_survive(self):
        """Colon stripping leaves a double space; the rejoin must not eat it."""
        assert _sanitize("TODO: fix") == "TODO  fix"
        assert _sanitize("a\tb") == "a\tb"

    def test_dangling_operators_and_bare_stars_still_handled(self):
        assert _sanitize("hello AND") == "hello"
        assert _sanitize("OR world") == "world"
        assert _sanitize("***") == ""

    def test_balanced_quoted_phrase_is_preserved_verbatim(self, fts):
        assert _sanitize('"deploy script"') == '"deploy script"'
        assert _hits(fts, '"deploy script"') == 1


class TestEverySanitizedQueryIsParsable:
    """The sanitizer's stated job: never hand FTS5 something it rejects."""

    @pytest.mark.parametrize(
        "query",
        [
            "scripts/deploy.sh", "user@example.com", "61.2%", "deploy.sh",
            "hermes-agent", "my-app.config.ts", "TODO: fix", "C++",
            "foo(bar)", "^start", "3-way", "deploy*", "hello AND",
            "a/b/c.d", "x@y.z/w", "1.2.3-rc.1", "path\\to\\file.txt",
        ],
    )
    def test_no_operational_error(self, fts, query):
        fts.execute("SELECT rowid FROM t WHERE t MATCH ?", (_sanitize(query),))
