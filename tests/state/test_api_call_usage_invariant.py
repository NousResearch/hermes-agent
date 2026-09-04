"""Invariant test: api_call_usage detail rows must reconcile with session_model_usage aggregate.

Both tables are written from the same chokepoint (_record_model_usage) inside
the same transaction.  This test asserts the fundamental consistency
invariant:

    sum(detail.tokens)  ==  aggregate.tokens

for every (session, model, provider, base_url, billing_mode, task) bucket
fully inside the recorded window.  Any future drift (e.g. a third write path
that touches only one table) would break this invariant and fail the test.
"""

import pytest

from hermes_state import SessionDB


class TestApiCallUsageInvariant:
    """Aggregate and detail tables must stay consistent by construction."""

    def test_detail_sums_match_aggregate(self, tmp_path):
        """sum(detail) == aggregate for each bucket after multiple calls."""
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session("s1", "cli", model="m1")
            # Simulate three incremental API calls on the same route.
            db.update_token_counts(
                "s1", input_tokens=100, output_tokens=50,
                model="m1", billing_provider="prov",
                api_call_count=1,
            )
            db.update_token_counts(
                "s1", input_tokens=200, output_tokens=80,
                model="m1", billing_provider="prov",
                api_call_count=1,
            )
            db.update_token_counts(
                "s1", input_tokens=50, output_tokens=20,
                model="m1", billing_provider="prov",
                api_call_count=1,
            )

            agg = db._conn.execute(
                "SELECT input_tokens, output_tokens, api_call_count "
                "FROM session_model_usage "
                "WHERE session_id=? AND model=?",
                ("s1", "m1"),
            ).fetchone()
            assert agg is not None

            detail = db._conn.execute(
                "SELECT "
                "  COALESCE(SUM(input_tokens), 0), "
                "  COALESCE(SUM(output_tokens), 0), "
                "  COALESCE(SUM(api_call_count), 0) "
                "FROM api_call_usage "
                "WHERE session_id=? AND model=?",
                ("s1", "m1"),
            ).fetchone()
            assert detail is not None

            assert detail[0] == agg["input_tokens"]
            assert detail[1] == agg["output_tokens"]
            assert detail[2] == agg["api_call_count"]
        finally:
            db.close()

    def test_detail_rows_count_matches_calls(self, tmp_path):
        """Each update_token_counts call produces exactly one detail row."""
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session("s2", "cli", model="m2")
            for _ in range(5):
                db.update_token_counts(
                    "s2", input_tokens=10,
                    model="m2", billing_provider="prov",
                    api_call_count=1,
                )
            count = db._conn.execute(
                "SELECT COUNT(*) FROM api_call_usage WHERE session_id=?",
                ("s2",),
            ).fetchone()[0]
            assert count == 5
        finally:
            db.close()

    def test_different_models_tracked_separately(self, tmp_path):
        """Calls on different models produce separate detail and aggregate rows."""
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session("s3", "cli", model="m-a")
            db.update_token_counts(
                "s3", input_tokens=100,
                model="m-a", billing_provider="prov",
                api_call_count=1,
            )
            db.update_token_counts(
                "s3", input_tokens=200,
                model="m-b", billing_provider="prov",
                api_call_count=1,
            )

            for model, expected_input in (("m-a", 100), ("m-b", 200)):
                agg = db._conn.execute(
                    "SELECT input_tokens FROM session_model_usage "
                    "WHERE session_id=? AND model=?",
                    ("s3", model),
                ).fetchone()
                detail_sum = db._conn.execute(
                    "SELECT SUM(input_tokens) FROM api_call_usage "
                    "WHERE session_id=? AND model=?",
                    ("s3", model),
                ).fetchone()[0]
                assert agg["input_tokens"] == expected_input
                assert detail_sum == expected_input
        finally:
            db.close()

    def test_cost_reconciliation(self, tmp_path):
        """Estimated and actual costs also reconcile between detail and aggregate."""
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session("s4", "cli", model="m4")
            db.update_token_counts(
                "s4", input_tokens=100,
                model="m4", billing_provider="prov",
                api_call_count=1,
                estimated_cost_usd=0.05,
                actual_cost_usd=0.04,
            )
            db.update_token_counts(
                "s4", input_tokens=200,
                model="m4", billing_provider="prov",
                api_call_count=1,
                estimated_cost_usd=0.10,
                actual_cost_usd=0.08,
            )
            agg = db._conn.execute(
                "SELECT estimated_cost_usd, actual_cost_usd "
                "FROM session_model_usage "
                "WHERE session_id=? AND model=?",
                ("s4", "m4"),
            ).fetchone()
            detail = db._conn.execute(
                "SELECT "
                "  ROUND(SUM(estimated_cost_usd), 6), "
                "  ROUND(SUM(actual_cost_usd), 6) "
                "FROM api_call_usage "
                "WHERE session_id=? AND model=?",
                ("s4", "m4"),
            ).fetchone()
            assert detail[0] == pytest.approx(agg["estimated_cost_usd"])
            assert detail[1] == pytest.approx(agg["actual_cost_usd"])
        finally:
            db.close()
