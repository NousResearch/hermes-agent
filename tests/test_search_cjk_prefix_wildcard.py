"""CJK search must survive the prefix wildcard callers append (#90636).

The web/desktop search endpoint turns every unquoted token into ``token*`` so
partial English words match. None of the CJK routes can honour that star: the
bigram and trigram routes quote each token before ``MATCH`` (so ``*`` is matched
literally) and the LIKE route has no ``*`` wildcard at all (only ``%``/``_``).
Left in place, every CJK search typed into the desktop/web search box becomes
a search for a term ending in a literal asterisk and returns nothing, while
the identical query without the star returns rows.

Runs against a real SessionDB in a temp HERMES_HOME, on the trigram and LIKE
routes only — no ``cjk_unicode61`` tokenizer toolchain is required.
"""

from hermes_state import SessionDB

TWO_CHAR = "秃发"          # 2 CJK chars — below the trigram threshold, LIKE route
FOUR_CHAR = "秃发应对"      # 4 CJK chars — trigram-eligible


def make_db(tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    session_id = "20260820_000001_cjk001"
    database.create_session(session_id, "cli")
    database.append_message(session_id, "user", "关于秃发应对的讨论内容，请总结")
    database.append_message(session_id, "assistant", "hello nimby world")
    return database


def hits(database, query):
    return database.search_messages(query=query, limit=10, fields=("session_id", "snippet"))


def test_trailing_star_matches_the_bare_query(tmp_path):
    """The star must widen or keep the result set, never empty it."""
    database = make_db(tmp_path)
    try:
        for term in (TWO_CHAR, FOUR_CHAR):
            bare = hits(database, term)
            starred = hits(database, term + "*")
            assert bare, f"{term!r} should match the seeded message"
            assert len(starred) == len(bare), (
                f"{term + '*'!r} returned {len(starred)} rows vs {len(bare)} for {term!r} — "
                "the caller-appended prefix wildcard is being matched literally"
            )
    finally:
        database.close()


def test_star_is_stripped_per_token_in_boolean_queries(tmp_path):
    """A multi-token CJK OR query keeps working with a wildcard per token."""
    database = make_db(tmp_path)
    try:
        assert hits(database, "秃发* OR 桂林*")
    finally:
        database.close()


def test_ascii_prefix_wildcard_still_works(tmp_path):
    """The normalization is CJK-only; the ASCII prefix search is untouched."""
    database = make_db(tmp_path)
    try:
        assert hits(database, "nimb*")
    finally:
        database.close()


def test_a_lone_star_is_not_turned_into_a_match_all(tmp_path):
    """Stripping must not leave an empty term that matches every row."""
    database = make_db(tmp_path)
    try:
        assert hits(database, "*") == []
    finally:
        database.close()


def test_quoted_cjk_phrase_survives_the_star(tmp_path):
    """The quoted workaround from the report still matches once starred."""
    database = make_db(tmp_path)
    try:
        assert hits(database, '"秃发"' + "*")
        assert hits(database, '"秃发"')
    finally:
        database.close()


def test_mixed_cjk_and_ascii_query_survives_the_star(tmp_path):
    database = make_db(tmp_path)
    try:
        found = hits(database, "秃发* nimby*")
        assert found
    finally:
        database.close()
