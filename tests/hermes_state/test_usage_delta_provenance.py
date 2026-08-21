"""Regression tests for session_model_usage provenance consistency (issue #75805).

``_record_model_usage`` resolves the (model, billing_provider) pair with
consistent provenance: both halves from the call when present, the whole
session pair when neither is present, and the missing half tagged
unknown/empty when exactly one is present. Mixing a call half with the
session's last-write-wins half mints composite keys that never served a
request (e.g. ``gpt-5.5 @ api.deepseek.com`` after a provider-outage retry
storm), misattributing spend across providers.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _usage_rows(db, session_id):
    with db._lock:
        rows = db._conn.execute(
            "SELECT model, billing_provider, billing_base_url, billing_mode,"
            " api_call_count, input_tokens FROM session_model_usage"
            " WHERE session_id = ? ORDER BY model, billing_provider",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


class TestUsageDeltaProvenance:
    def test_both_halves_from_call_are_authoritative(self, db):
        """Both halves supplied: the call's pair wins, even if the session
        row holds a different pair."""
        db.create_session("s1", source="cli", model="session-model")
        db.update_session_billing_route(
            "s1", provider="session-provider",
            base_url="https://session.example/v1", billing_mode="api_key",
        )
        db.update_token_counts(
            "s1", input_tokens=100, model="call-model",
            billing_provider="call-provider",
            billing_base_url="https://call.example/v1",
            billing_mode="api_key", api_call_count=1,
        )
        rows = _usage_rows(db, "s1")
        assert len(rows) == 1
        assert rows[0]["model"] == "call-model"
        assert rows[0]["billing_provider"] == "call-provider"
        assert rows[0]["billing_base_url"] == "https://call.example/v1"
        assert rows[0]["billing_mode"] == "api_key"

    def test_token_only_delta_inherits_whole_session_pair(self, db):
        """Neither half supplied: inherit the session pair wholesale (same
        provenance), so token-only accounting still lands on the right row."""
        db.create_session("s2", source="cli", model="session-model")
        db.update_session_billing_route(
            "s2", provider="session-provider",
            base_url="https://session.example/v1", billing_mode="api_key",
        )
        db.update_token_counts(
            "s2", input_tokens=50, api_call_count=1,
        )
        rows = _usage_rows(db, "s2")
        assert len(rows) == 1
        assert rows[0]["model"] == "session-model"
        assert rows[0]["billing_provider"] == "session-provider"
        assert rows[0]["billing_base_url"] == "https://session.example/v1"
        assert rows[0]["billing_mode"] == "api_key"

    def test_model_only_delta_does_not_borrow_session_provider(self, db):
        """Regression for #75805: a delta carrying model but no provider must
        NOT pair it with the session's last-write-wins provider — that mints
        fabricated keys (e.g. ``gpt-5.5 @ api.deepseek.com``)."""
        db.create_session("s3", source="cli", model="gpt-5.5")
        db.update_session_billing_route(
            "s3", provider="api.deepseek.com",
            base_url="https://api.deepseek.com/v1", billing_mode="api_key",
        )
        # Fallback churn: the call knows the model but not the billing route.
        db.update_token_counts(
            "s3", input_tokens=100, model="gpt-5.5",
            billing_provider=None, billing_base_url=None,
            billing_mode=None, api_call_count=1,
        )
        rows = _usage_rows(db, "s3")
        assert len(rows) == 1
        assert rows[0]["model"] == "gpt-5.5"
        # The provider half is tagged empty, never borrowed from the session.
        assert rows[0]["billing_provider"] == ""
        assert rows[0]["billing_base_url"] == ""
        assert rows[0]["billing_mode"] == ""
        # No fabricated ``gpt-5.5 @ api.deepseek.com`` composite.
        assert not any(
            r["model"] == "gpt-5.5" and r["billing_provider"] == "api.deepseek.com"
            for r in rows
        )

    def test_provider_only_delta_tags_model_unknown(self, db):
        """Symmetric case: a delta carrying provider but no model must not
        pair it with the session's model."""
        db.create_session("s4", source="cli", model="session-model")
        db.update_session_billing_route(
            "s4", provider="session-provider",
            base_url="https://session.example/v1", billing_mode="api_key",
        )
        db.update_token_counts(
            "s4", input_tokens=100, model=None,
            billing_provider="call-provider",
            billing_base_url="https://call.example/v1",
            billing_mode="api_key", api_call_count=1,
        )
        rows = _usage_rows(db, "s4")
        assert len(rows) == 1
        assert rows[0]["model"] == "unknown"
        assert rows[0]["billing_provider"] == "call-provider"
        assert rows[0]["billing_base_url"] == "https://call.example/v1"
        assert rows[0]["billing_mode"] == "api_key"
        assert not any(
            r["model"] == "session-model" and r["billing_provider"] == "call-provider"
            for r in rows
        )

    def test_fallback_churn_keeps_real_pairs_and_unknowns_separate(self, db):
        """The outage scenario from the issue: session lands on provider A,
        then deltas arrive half-specified. Real (model, provider) pairs must
        survive and half-specified deltas must not merge into composites."""
        db.create_session("s5", source="cli", model="gpt-5.5")
        db.update_session_billing_route(
            "s5", provider="api.deepseek.com",
            base_url="https://api.deepseek.com/v1", billing_mode="api_key",
        )
        # Real call on the deepseek route.
        db.update_token_counts(
            "s5", input_tokens=10, model="gpt-5.5",
            billing_provider="api.deepseek.com",
            billing_base_url="https://api.deepseek.com/v1",
            billing_mode="api_key", api_call_count=1,
        )
        # Half-specified deltas during the retry storm.
        db.update_token_counts(
            "s5", input_tokens=5, model="gpt-5.5", api_call_count=1,
        )
        db.update_token_counts(
            "s5", input_tokens=3, billing_provider="api.deepseek.com",
            api_call_count=1,
        )
        rows = _usage_rows(db, "s5")
        # Exactly the real pair plus the two half-tagged rows — no composites.
        pairs = {(r["model"], r["billing_provider"]) for r in rows}
        assert pairs == {
            ("gpt-5.5", "api.deepseek.com"),  # real call
            ("gpt-5.5", ""),                  # model-only delta
            ("unknown", "api.deepseek.com"),  # provider-only delta
        }
        # Tokens land on their respective rows, nothing double-counted.
        by_pair = {(r["model"], r["billing_provider"]): r for r in rows}
        assert by_pair[("gpt-5.5", "api.deepseek.com")]["input_tokens"] == 10
        assert by_pair[("gpt-5.5", "")]["input_tokens"] == 5
        assert by_pair[("unknown", "api.deepseek.com")]["input_tokens"] == 3
