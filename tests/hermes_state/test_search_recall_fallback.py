"""Tests for the search_messages zero-result recall fallback.

FTS5 MATCH ANDs every term, so a naturally-phrased multi-word query returns
nothing unless all of its words co-occur in a single message. search_messages
now retries such queries OR-joined — only when the strict AND pass found
nothing, and never when the query uses explicit FTS5 syntax (quotes, ``*``,
uppercase booleans). These tests pin both halves: the recall win and the
preserved precision-first semantics.
"""
import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _text(row):
    """Matched text of a result row (default projection carries a snippet,
    not full content), with the match markers stripped."""
    return (row.get("content") or row["snippet"]).replace(">>>", "").replace("<<<", "")


def _seed(db, sid="s1", source="cli"):
    db.create_session(sid, source=source)
    ids = {}
    ids["vet"] = db.append_message(
        sid, role="assistant",
        content="go vet reported: composite literal uses unkeyed fields in lang/lexer/lexer.go",
    )
    ids["docker"] = db.append_message(
        sid, role="user", content="the docker deployment failed on the staging host"
    )
    ids["filler"] = db.append_message(
        sid, role="assistant", content="sounds good, proceeding with the plan"
    )
    return ids


class TestRecallFallback:
    def test_natural_language_query_recalls(self, db):
        """The motivating case: a question-shaped query whose terms never
        co-occur in one message must still find the relevant message."""
        _seed(db)
        rows = db.search_messages("what error did go vet report for lang/lexer/lexer.go")
        assert rows, "fallback should recall when the AND pass finds nothing"
        assert any("go vet" in _text(r) for r in rows)

    def test_specific_terms_rank_first(self, db):
        """BM25 puts the message matching the rare, specific terms on top even
        though the broadened query ORs in common words."""
        _seed(db)
        rows = db.search_messages("what did go vet say about lexer.go exactly")
        assert rows and "go vet" in _text(rows[0])

    def test_and_pass_still_wins_when_it_matches(self, db):
        """Queries whose terms do co-occur are answered by the strict pass —
        the fallback never runs, so precision semantics are unchanged."""
        _seed(db)
        rows = db.search_messages("docker deployment")
        assert len(rows) == 1
        assert "docker deployment" in _text(rows[0])


class TestFallbackDoesNotApply:
    def test_explicit_boolean_query_is_not_rewritten(self, db):
        """A NOT query that matches nothing must stay empty: the caller used
        explicit syntax and opted into exact semantics."""
        _seed(db)
        assert db.search_messages("docker NOT deployment") == []

    def test_quoted_phrase_is_not_rewritten(self, db):
        _seed(db)
        assert db.search_messages('"docker lexer"') == []

    def test_prefix_wildcard_is_not_rewritten(self, db):
        _seed(db)
        assert db.search_messages("dockerz* lexerz*") == []

    def test_single_term_miss_stays_empty(self, db):
        """One term has nothing to broaden into."""
        _seed(db)
        assert db.search_messages("kubernetes") == []

    def test_no_term_matches_anywhere_stays_empty(self, db):
        _seed(db)
        assert db.search_messages("zeppelin quasar nimbus") == []


class TestBroadenedRelevance:
    def test_single_common_word_match_is_not_a_hit(self, db):
        """A broadened retry must not surface rows sharing only one common
        word with the query — the regression that broke
        tools/test_session_search.py's rewind-exclusion test."""
        _seed(db)
        # 'sounds good, proceeding with the plan' shares only 'plan'
        assert db.search_messages("deleted plan gamma") == []

    def test_two_distinct_words_qualify(self, db):
        _seed(db)
        rows = db.search_messages("staging docker gamma")
        assert rows and "docker deployment" in _text(rows[0])

    def test_identifier_term_matches_on_token_boundaries_only(self, db):
        """'lexer.go' must not count 'mylexer.gone' as a real-signal hit."""
        db.create_session("s2", source="cli")
        db.append_message("s2", role="user",
                          content="renamed the tool to mylexer.gonewild for fun")
        assert db.search_messages("zzqx lexer.go") == []

    def test_word_term_matches_on_token_boundaries_only(self, db):
        """'plan' inside 'planet' is not a match toward the 2-term bar."""
        db.create_session("s3", source="cli")
        db.append_message("s3", role="user",
                          content="the planet spins and the star burns brightly")
        assert db.search_messages("deploy plan star gamma") == []

    def test_stopword_only_query_returns_nothing(self, db):
        _seed(db)
        assert db.search_messages("what was that about") == []


class TestFallbackRespectsFilters:
    def test_source_exclusion_applies_to_broadened_pass(self, db):
        """The retry re-issues the same filters: a message in an excluded
        source must not surface just because the query was broadened."""
        _seed(db, sid="s-tool", source="tool")
        rows = db.search_messages(
            "what error did go vet report for lang/lexer/lexer.go",
            exclude_sources=["tool"],
        )
        assert rows == []

    def test_role_filter_applies_to_broadened_pass(self, db):
        _seed(db)
        rows = db.search_messages(
            "what error did go vet report for lang/lexer/lexer.go",
            role_filter=["user"],
        )
        assert all(r["role"] == "user" for r in rows)


class TestBroadenHelper:
    def test_identifier_terms_survive_whole(self):
        q = SessionDB._broaden_fts5_query("error in lang/lexer/lexer.go line 237")
        # Slashed/dotted identifiers are phrase-quoted so FTS5 neither splits
        # them nor chokes on the ``/``.
        assert '"lang/lexer/lexer.go"' in q.split(" OR ")

    def test_dedupes_and_caps_terms(self):
        q = SessionDB._broaden_fts5_query("alpha alpha " + " ".join(f"t{i}" for i in range(20)))
        terms = q.split(" OR ")
        assert terms.count("alpha") == 1
        assert len(terms) == SessionDB._BROADEN_MAX_TERMS

    def test_stopwords_dropped_when_content_terms_remain(self):
        q = SessionDB._broaden_fts5_query("what did the compiler report for main.rs")
        terms = q.split(" OR ")
        assert "what" not in terms and "the" not in terms
        assert "compiler" in terms and '"main.rs"' in terms

    def test_all_stopword_query_does_not_broaden(self):
        # An OR over stopwords could only dredge up noise; stopword-only
        # queries keep their strict-pass emptiness.
        assert SessionDB._broaden_fts5_query("what was that about") is None

    def test_lowercase_or_is_a_plain_word(self):
        # FTS5 booleans are uppercase-only; "cats or dogs" is a plain query
        # and should broaden.
        assert SessionDB._broaden_fts5_query("cats or dogs") is not None
        assert SessionDB._broaden_fts5_query("cats OR dogs") is None
