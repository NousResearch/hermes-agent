"""One endpoint must stay one bucket in ``session_model_usage``.

``billing_base_url`` is part of the usage table's primary key, but the two writers
spell the same endpoint differently: the main loop passes the configured string, while
the auxiliary path reads the route back off the OpenAI client, whose ``base_url`` is an
``httpx.URL`` that always renders with a trailing slash. Without canonicalisation a
single endpoint is split across two rows and every per-endpoint total reads low.
"""
import pytest

from hermes_state import SessionDB
from hermes_state_usage import canonical_billing_base_url

ENDPOINT = "https://llm.example.com/v1"


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _calls_by_base_url(db, session_id):
    """billing_base_url -> total api_call_count, the way an operator asks."""
    with db._lock:
        rows = db._conn.execute(
            "SELECT billing_base_url, SUM(api_call_count) AS calls"
            " FROM session_model_usage WHERE session_id = ? GROUP BY 1",
            (session_id,),
        ).fetchall()
    return {r["billing_base_url"]: r["calls"] for r in rows}


class TestCanonicalBillingBaseUrl:
    @pytest.mark.parametrize("raw", [ENDPOINT, ENDPOINT + "/", ENDPOINT + "///"])
    def test_trailing_slashes_collapse(self, raw):
        assert canonical_billing_base_url(raw) == ENDPOINT

    def test_route_without_trailing_slash_is_untouched(self):
        assert canonical_billing_base_url("acp://copilot") == "acp://copilot"

    def test_missing_route_stays_empty(self):
        assert canonical_billing_base_url(None) == ""
        assert canonical_billing_base_url("") == ""

    def test_bare_slash_is_not_emptied_into_the_unknown_bucket(self):
        # Stripping to "" would merge a degenerate route with the rows that mean
        # "we don't know the route", which is a different statement.
        assert canonical_billing_base_url("/") == "/"


class TestUsageBuckets:
    def test_main_loop_and_auxiliary_share_one_bucket(self, db):
        db.create_session("s-split", source="cli")
        db.update_token_counts(
            "s-split",
            input_tokens=1000,
            output_tokens=100,
            model="some-model",
            billing_provider="custom",
            billing_base_url=ENDPOINT,          # configured spelling
            api_call_count=3,
        )
        db.record_auxiliary_usage(
            "s-split",
            "title_generation",
            model="some-model",
            billing_provider="custom",
            billing_base_url=ENDPOINT + "/",    # httpx.URL spelling
            input_tokens=200,
            output_tokens=10,
            api_call_count=1,
        )

        buckets = _calls_by_base_url(db, "s-split")
        assert buckets == {ENDPOINT: 4}, f"endpoint split across spellings: {buckets!r}"

    def test_sessions_row_matches_the_usage_row_spelling(self, db):
        db.create_session("s-sum", source="cli")
        db.update_token_counts(
            "s-sum",
            input_tokens=10,
            model="some-model",
            billing_provider="custom",
            billing_base_url=ENDPOINT + "/",
            api_call_count=1,
        )
        with db._lock:
            row = db._conn.execute(
                "SELECT billing_base_url FROM sessions WHERE id = ?", ("s-sum",)
            ).fetchone()
        assert row["billing_base_url"] == ENDPOINT
        assert list(_calls_by_base_url(db, "s-sum")) == [ENDPOINT]

    def test_update_session_billing_route_canonicalises(self, db):
        db.create_session("s-route", source="cli")
        db.update_session_billing_route(
            "s-route", provider="custom", base_url=ENDPOINT + "/"
        )
        with db._lock:
            row = db._conn.execute(
                "SELECT billing_base_url FROM sessions WHERE id = ?", ("s-route",)
            ).fetchone()
        assert row["billing_base_url"] == ENDPOINT

    def test_distinct_endpoints_stay_separate(self, db):
        """Canonicalisation must not over-merge genuinely different routes."""
        other = "https://llm.example.com/openai"
        db.create_session("s-two", source="cli")
        for url in (ENDPOINT, other):
            db.update_token_counts(
                "s-two",
                input_tokens=10,
                model="some-model",
                billing_provider="custom",
                billing_base_url=url,
                api_call_count=1,
            )
        assert set(_calls_by_base_url(db, "s-two")) == {ENDPOINT, other}
