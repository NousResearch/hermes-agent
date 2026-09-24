"""Cost-field honesty gate at the SessionDB write boundary (issue #120057).

A field report observed one ``sessions`` row where ``estimated_cost_usd`` held
``1.33e179`` and four metadata columns held prose fragments, silently poisoning
every downstream cost aggregate (``SUM(estimated_cost_usd)`` over the store).
The producing write could not be reconstructed, so the contract here is
containment at the store boundary: whatever a caller hands these write APIs,
the persisted row must stay inside the cost enums and numeric domain — and the
clean fields of the same write (token counts) must still land.
"""

import logging
import math

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _row(db, session_id):
    with db._lock:
        row = db._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row)


class TestCostAmountGate:
    def test_implausible_cost_is_discarded_and_marked_unknown(self, db):
        db.create_session("s1", source="cli")
        db.update_token_counts(
            "s1", input_tokens=100, estimated_cost_usd=1.3329519031027556e179,
            cost_status="estimated", cost_source="provider_models_api",
            model="test-model", billing_provider="openrouter",
        )
        row = _row(db, "s1")
        assert row["input_tokens"] == 100  # clean fields of the same write land
        assert row["estimated_cost_usd"] == 0  # the poisoned amount never lands
        assert row["cost_status"] == "unknown"
        assert row["cost_source"] == "none"

    @pytest.mark.parametrize("bad", [-5.0, float("inf"), float("nan"), "not-a-number"])
    def test_non_finite_or_negative_costs_never_reach_the_row(self, db, bad):
        db.create_session("s1", source="cli")
        db.update_token_counts("s1", output_tokens=7, estimated_cost_usd=bad,
                               cost_status="estimated")
        row = _row(db, "s1")
        assert row["estimated_cost_usd"] == 0
        assert row["cost_status"] == "unknown"

    def test_discarded_cost_warns_once_per_field(self, db, caplog):
        db.create_session("s1", source="cli")
        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            db.update_token_counts("s1", estimated_cost_usd=1e30)
        assert any("estimated_cost_usd" in r.message for r in caplog.records)

    def test_legitimate_costs_accumulate_unchanged(self, db):
        db.create_session("s1", source="cli")
        db.update_token_counts("s1", estimated_cost_usd=0.25, cost_status="estimated",
                               cost_source="provider_models_api")
        db.update_token_counts("s1", estimated_cost_usd=0.25, cost_status="estimated",
                               cost_source="provider_models_api")
        row = _row(db, "s1")
        assert row["estimated_cost_usd"] == pytest.approx(0.5)
        assert row["cost_status"] == "estimated"
        assert row["cost_source"] == "provider_models_api"


class TestCostEnumGate:
    def test_out_of_enum_status_and_source_do_not_reach_the_row(self, db):
        db.create_session("s1", source="cli")
        db.update_token_counts("s1", input_tokens=10, cost_status="of extrac",
                               cost_source="tion with the other 16",
                               estimated_cost_usd=0.01)
        row = _row(db, "s1")
        assert row["input_tokens"] == 10
        assert row["cost_status"] is None  # garbage enum never lands
        assert row["cost_source"] is None
        assert row["estimated_cost_usd"] == pytest.approx(0.01)  # trusted amount kept

    def test_all_enum_values_pass(self, db):
        db.create_session("s1", source="cli")
        for status in ("actual", "estimated", "included", "unknown"):
            db.update_token_counts("s1", cost_status=status)
        for source in ("provider_cost_api", "provider_generation_api",
                       "provider_models_api", "official_docs_snapshot",
                       "user_override", "custom_contract", "none"):
            db.update_token_counts("s1", cost_source=source)
        row = _row(db, "s1")
        assert row["cost_status"] == "unknown"
        assert row["cost_source"] == "none"


class TestBillingRouteGate:
    def test_prose_route_update_is_rejected_wholesale(self, db):
        db.create_session("s1", source="cli")
        db.update_session_billing_route("s1", provider="openrouter",
                                        base_url="https://openrouter.ai/api/v1")
        db.update_session_billing_route(
            "s1", provider=" it's no", base_url="t zero.\n\nLet me try one mor")
        row = _row(db, "s1")
        assert row["billing_provider"] == "openrouter"  # last good route survives
        assert row["billing_base_url"] == "https://openrouter.ai/api/v1"

    def test_prose_route_via_first_accounted_usage_never_lands(self, db):
        db.create_session("s1", source="cli")
        db.update_token_counts(
            "s1", input_tokens=10, model="test-model",
            billing_provider=" it's no", billing_base_url="t zero.\n\nLet me try one mor",
            estimated_cost_usd=0.01, cost_status="estimated")
        row = _row(db, "s1")
        assert row["input_tokens"] == 10
        assert row["billing_base_url"] is None  # whitespace is never a URL
        assert row["estimated_cost_usd"] == pytest.approx(0.01)

    def test_empty_base_url_still_allowed(self, db):
        db.create_session("s1", source="cli")
        db.update_token_counts("s1", input_tokens=1, model="m",
                               billing_provider="anthropic", billing_base_url="")
        row = _row(db, "s1")
        assert row["billing_provider"] == "anthropic"


class TestBackfillStampGate:
    def test_prose_stamp_is_rejected(self, db):
        db.create_session("s1", source="cli")
        stamped = db.backfill_null_session_profiles(" URLs.")
        assert stamped == 0
        assert _row(db, "s1")["profile_name"] is None

    def test_valid_stamp_still_backfills(self, db):
        db.create_session("s1", source="cli")
        stamped = db.backfill_null_session_profiles("master")
        assert stamped == 1
        assert _row(db, "s1")["profile_name"] == "master"


class TestAuxUsageCostGate:
    def test_implausible_aux_cost_is_discarded(self, db):
        db.create_session("s1", source="cli")
        db.record_auxiliary_usage("s1", "vision", model="gemini-3-flash",
                                  input_tokens=500, estimated_cost_usd=1e300)
        with db._lock:
            rows = db._conn.execute(
                "SELECT estimated_cost_usd, input_tokens FROM session_model_usage "
                "WHERE session_id = ?", ("s1",)).fetchall()
        assert len(rows) == 1
        assert rows[0]["estimated_cost_usd"] == 0.0
        assert rows[0]["input_tokens"] == 500

    def test_legitimate_aux_cost_flows(self, db):
        db.create_session("s1", source="cli")
        db.record_auxiliary_usage("s1", "vision", model="gemini-3-flash",
                                  estimated_cost_usd=0.5)
        with db._lock:
            cost = db._conn.execute(
                "SELECT estimated_cost_usd FROM session_model_usage "
                "WHERE session_id = ?", ("s1",)).fetchone()["estimated_cost_usd"]
        assert cost == 0.5
