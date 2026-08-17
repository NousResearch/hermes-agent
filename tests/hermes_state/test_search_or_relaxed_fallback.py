"""OR-relaxed zero-result retry for paraphrased session search recall.

Ported from nearai/ironclaw#7553 (``Filter::FtsRanked``): FTS5's implicit
AND between query terms means a multi-word query worded even slightly
differently from the stored sentence returns nothing — a fact saved as
"Sarah prefers the standup meeting scheduled early on Thursday mornings" is
invisible to "when does Sarah like her standup scheduled" purely because the
stored text has no "like". When the exact-match search (and the substring
fallbacks) return zero rows, ``search_messages`` retries the same FTS index
with the terms OR-joined, ranked by bm25 so rows covering more of the terms
surface first.

The retry is strictly additive: it only fires on a zero-result miss, never
reorders existing hits, and respects explicit boolean operators.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session(session_id="s1", source="cli", model="m")
    d.append_message(
        "s1",
        role="user",
        content=(
            "Sarah prefers the standup meeting scheduled early on "
            "Thursday mornings"
        ),
    )
    d.append_message(
        "s1", role="assistant", content="Noted, standup moved to Thursday."
    )
    d.append_message(
        "s1", role="user", content="graphiti daemon looks healthy today"
    )
    yield d
    try:
        d.close()
    except Exception:
        pass


class TestOrRelaxedQueryHelper:
    def test_multi_term_query_relaxes_to_or(self):
        assert (
            SessionDB._or_relaxed_query("sarah standup scheduled")
            == "sarah OR standup OR scheduled"
        )

    def test_single_term_returns_none(self):
        assert SessionDB._or_relaxed_query("standup") is None

    def test_explicit_or_is_respected(self):
        assert SessionDB._or_relaxed_query("alpha OR beta") is None

    def test_explicit_not_is_respected(self):
        assert SessionDB._or_relaxed_query("python NOT java") is None

    def test_explicit_and_tokens_are_dropped(self):
        assert (
            SessionDB._or_relaxed_query("alpha AND beta")
            == "alpha OR beta"
        )

    def test_quoted_phrase_is_one_unit(self):
        assert (
            SessionDB._or_relaxed_query('"docker networking" tls')
            == '"docker networking" OR tls'
        )

    def test_lone_quoted_phrase_returns_none(self):
        assert SessionDB._or_relaxed_query('"docker networking"') is None


class TestParaphrasedRecall:
    def test_exact_query_still_matches_exactly(self, db):
        rows = db.search_messages("standup Thursday")
        assert rows, "exact-term query must match without relaxation"
        assert "standup" in rows[0]["snippet"].lower()

    def test_paraphrased_query_recovers_via_or_retry(self, db):
        # "when does Sarah like her standup scheduled": implicit AND requires
        # "like", which the stored sentence lacks — exact match returns
        # nothing; the OR-relaxed retry must recover the row.
        rows = db.search_messages("when does Sarah like her standup scheduled")
        assert rows, "paraphrased query must recover via the OR-relaxed retry"
        joined = " ".join(r["snippet"].lower() for r in rows)
        assert "standup" in joined

    def test_or_retry_recovers_all_partially_matching_rows(self, db):
        rows = db.search_messages("sarah standup thursday daemon")
        # No stored row contains every term, so only the OR retry can answer;
        # both partial matches must come back (bm25 ordering between them is
        # backend-weighted, not coverage-guaranteed).
        joined = " ".join(r["snippet"].lower() for r in rows)
        assert "standup" in joined
        assert "daemon" in joined

    def test_genuinely_absent_terms_still_return_empty(self, db):
        assert db.search_messages("zebra xylophone quantum") == []

    def test_explicit_not_query_is_not_relaxed(self, db):
        # "standup NOT Thursday" must exclude the Thursday rows — relaxation
        # would wrongly resurrect them.
        rows = db.search_messages("standup NOT Thursday")
        for r in rows:
            assert "thursday" not in r["snippet"].lower()

    def test_role_filter_applies_to_relaxed_retry(self, db):
        rows = db.search_messages(
            "when does Sarah like her standup scheduled",
            role_filter=["assistant"],
        )
        for r in rows:
            assert r["role"] == "assistant"
