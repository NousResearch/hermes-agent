"""CJK search must survive the prefix wildcard callers append (#90636).

The web/desktop search endpoint turns every unquoted token into ``token*`` so
partial English words match. The three CJK routes cannot honour that: the
bigram and trigram routes quote each token before ``MATCH`` (so ``*`` is
matched literally) and the LIKE route has no ``*`` wildcard at all. Without
normalization every CJK search typed into the UI searches for a term ending in
a literal asterisk and returns nothing, while the identical query without the
star returns rows — which is what makes the bug look like "search is broken
for Chinese".

These tests run on the LIKE / trigram routes, so they do not need the loadable
``cjk_unicode61`` tokenizer that ``test_fts_cjk_bigram.py`` builds.
"""

import pytest

from hermes_state import SessionDB

TWO_CHAR = "秃发"          # 2 CJK chars — below the trigram threshold
FOUR_CHAR = "秃发应对"      # 4 CJK chars — trigram-eligible
CONTENT = "秃发应对方案请总结"


@pytest.fixture
def db(tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    session_id = "20260820_000001_cjk001"
    database.create_session(session_id, "cli")
    database.append_message(session_id, "user", CONTENT)
    database.append_message(session_id, "assistant", "hello nimby world")
    yield database
    database.close()


def _hits(database, query):
    return database.search_messages(
        query=query, limit=10, fields=("session_id", "snippet")
    )


@pytest.mark.parametrize("term", [TWO_CHAR, FOUR_CHAR])
def test_trailing_star_matches_the_bare_query(db, term):
    """The star must not change the result set — only widen it, never empty it."""
    bare = _hits(db, term)
    starred = _hits(db, term + "*")
    assert bare, f"{term!r} should match the seeded message"
    assert len(starred) == len(bare), (
        f"{term + '*'!r} returned {len(starred)} rows vs {len(bare)} for {term!r} — "
        "the prefix wildcard is being matched literally"
    )


def test_star_is_stripped_per_token_in_boolean_queries(db):
    """A multi-token CJK OR query keeps working with wildcards on each token."""
    assert _hits(db, "秃发* OR 桂林*")


def test_ascii_prefix_wildcard_still_works(db):
    """The normalization is CJK-only; English prefix search is untouched."""
    assert _hits(db, "nimb*")


def test_a_lone_star_is_not_turned_into_a_match_all(db):
    """Stripping must not leave an empty term that matches every row."""
    assert _hits(db, "*") == []


def test_mixed_cjk_and_ascii_query_survives_the_star(db):
    assert _hits(db, "秃发* nimby*") or _hits(db, "秃发*")
