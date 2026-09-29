"""Billed-cost accounting (estimator-usage-cost-fix): the provider's per-response
invoice (``usage.cost`` on OpenRouter) becomes the primary cost figure
(``actual_cost_usd``, ``cost_status='actual'``) with the token-rate estimate kept
as the fallback, and provider generation ids persist for exact
GET /api/v1/generation back-sampling."""

import logging
from decimal import Decimal
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "test_state.db")
    yield session_db
    session_db.close()


# ---- extract_billed_cost ----


class TestExtractBilledCost:
    def test_openrouter_dict_shape(self):
        from agent.usage_pricing import extract_billed_cost
        assert extract_billed_cost({"prompt_tokens": 19, "cost": 0.000361}) == Decimal("0.000361")

    def test_sdk_object_with_cost_attribute(self):
        from agent.usage_pricing import extract_billed_cost
        usage = SimpleNamespace(prompt_tokens=19, cost=0.000361)
        assert extract_billed_cost(usage) == Decimal("0.000361")

    def test_sdk_object_with_model_extra(self):
        # OpenAI SDK versions that park unknown response fields in model_extra.
        from agent.usage_pricing import extract_billed_cost
        usage = SimpleNamespace(prompt_tokens=19, cost=None, model_extra={"cost": 0.0002})
        assert extract_billed_cost(usage) == Decimal("0.0002")

    def test_missing_none_and_negative_read_as_none(self):
        from agent.usage_pricing import extract_billed_cost
        assert extract_billed_cost(None) is None
        assert extract_billed_cost({"prompt_tokens": 3}) is None
        assert extract_billed_cost({"cost": None}) is None
        assert extract_billed_cost({"cost": "not-a-number"}) is None
        assert extract_billed_cost({"cost": -0.5}) is None

    def test_zero_cost_is_actual(self):
        # A free response is an invoiced $0, not "unknown".
        from agent.usage_pricing import extract_billed_cost
        assert extract_billed_cost({"cost": 0}) == Decimal("0")


# ---- ledger: billed cost through update_token_counts / record_auxiliary_usage ----


class TestLedgerBilledCost:
    def test_actual_cost_accumulates_on_sessions_and_model_usage(self, db):
        db.update_token_counts(
            "s1", input_tokens=100, output_tokens=5, model="z-ai/glm-5.3",
            billing_provider="openrouter", api_call_count=1,
            estimated_cost_usd=0.01, actual_cost_usd=0.0091,
            cost_status="actual", cost_source="provider_cost_api",
        )
        db.update_token_counts(
            "s1", input_tokens=50, output_tokens=2, model="z-ai/glm-5.3",
            billing_provider="openrouter", api_call_count=1,
            estimated_cost_usd=0.005, actual_cost_usd=0.004,
        )
        with db._lock:
            row = db._conn.execute(
                "SELECT actual_cost_usd, estimated_cost_usd, cost_status FROM sessions WHERE id='s1'"
            ).fetchone()
            mrow = db._conn.execute(
                "SELECT actual_cost_usd, cost_status FROM session_model_usage WHERE session_id='s1'"
            ).fetchone()
        assert row["actual_cost_usd"] == pytest.approx(0.0131)
        assert row["cost_status"] == "actual"
        assert mrow["actual_cost_usd"] == pytest.approx(0.0131)
        assert mrow["cost_status"] == "actual"

    def test_cost_only_delta_still_attributes_to_its_route(self, db):
        # usage.cost present but token fields stripped by a proxy: the per-route row
        # must still exist (has_accounted_usage gate, not bare has_usage).
        db.update_token_counts(
            "s2", model="z-ai/glm-5.3", billing_provider="openrouter",
            actual_cost_usd=0.001, cost_status="actual", cost_source="provider_cost_api",
        )
        with db._lock:
            rows = db._conn.execute(
                "SELECT model, actual_cost_usd FROM session_model_usage WHERE session_id='s2'"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["model"] == "z-ai/glm-5.3"
        assert rows[0]["actual_cost_usd"] == pytest.approx(0.001)

    def test_none_actual_cost_keeps_stored_value(self, db):
        # Providers without usage.cost pass None: the delta path must not zero the row.
        db.update_token_counts("s3", input_tokens=10, api_call_count=1,
                               actual_cost_usd=0.002, model="m", billing_provider="p")
        db.update_token_counts("s3", input_tokens=10, api_call_count=1,
                               actual_cost_usd=None, model="m", billing_provider="p")
        with db._lock:
            row = db._conn.execute(
                "SELECT actual_cost_usd FROM sessions WHERE id='s3'").fetchone()
        assert row["actual_cost_usd"] == pytest.approx(0.002)

    def test_aux_usage_carries_actual_cost_and_status(self, db):
        db.record_auxiliary_usage(
            "s6", "vision", model="google/gemini-3-flash-preview", billing_provider="openrouter",
            input_tokens=100, output_tokens=10, estimated_cost_usd=0.0001,
            actual_cost_usd=0.00009, cost_status="actual", cost_source="provider_cost_api",
        )
        with db._lock:
            row = db._conn.execute(
                "SELECT actual_cost_usd, estimated_cost_usd, cost_status, cost_source "
                "FROM session_model_usage WHERE session_id='s6' AND task='vision'").fetchone()
        assert row["actual_cost_usd"] == pytest.approx(0.00009)
        assert row["estimated_cost_usd"] == pytest.approx(0.0001)
        assert row["cost_status"] == "actual"
        assert row["cost_source"] == "provider_cost_api"


# ---- generation ids ----


class TestGenerationIds:
    def test_record_and_idempotency(self, db):
        db.record_generation_id("s4", "gen-1", model="z-ai/glm-5.3", provider="openrouter")
        db.record_generation_id("s4", "gen-1", model="z-ai/glm-5.3", provider="openrouter")
        db.record_generation_id("s4", "gen-2", model="z-ai/glm-5.3", provider="openrouter", task="vision")
        with db._lock:
            rows = db._conn.execute(
                "SELECT generation_id, task FROM generation_ids "
                "WHERE session_id='s4' ORDER BY generation_id").fetchall()
        assert [r["generation_id"] for r in rows] == ["gen-1", "gen-2"]
        assert [r["task"] for r in rows] == ["", "vision"]

    def test_empty_args_are_a_noop(self, db):
        db.record_generation_id("", "gen-x")
        db.record_generation_id("s5", "")
        with db._lock:
            count = db._conn.execute("SELECT COUNT(*) FROM generation_ids").fetchone()[0]
        assert count == 0


# ---- end to end: record_response_usage ----


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    a = AIAgent(api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
                api_mode="chat_completions", model="anthropic/claude-fable-5.1", session_id="t", platform="cli",
                quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
                enabled_toolsets=["file"])
    # CLI-platform agents don't open a store by default; own one for the ledger assertions
    # (same pattern as tests/tui_gateway/test_gateway_owned_session_reap.py).
    a._session_db = SessionDB(db_path=tmp_path / "state.db")
    a._owns_session_db = True
    return a


def _response(cost, gid="gen-1789811821-abc"):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=62_889, completion_tokens=7, total_tokens=62_896, cost=cost,
            prompt_tokens_details=SimpleNamespace(cached_tokens=34_283, cache_write_tokens=28_604),
            completion_tokens_details=None,
        ),
        id=gid, model="anthropic/claude-fable-5.1",
    )


class TestRecordResponseUsageBilledCost:
    def test_billed_cost_and_generation_id_reach_the_ledger(self, tmp_path, monkeypatch):
        a = _agent(tmp_path, monkeypatch)
        try:
            from agent import turn_usage
            turn_usage.record_response_usage(
                a, _response(0.000361), messages=[{"role": "user", "content": "hi"}],
                api_call_count=1, api_duration=0.2, compression_attempts=0, max_compression_attempts=3,
            )
            assert a._session_db.flush_token_counts()
            with a._session_db._lock:
                srow = a._session_db._conn.execute(
                    "SELECT actual_cost_usd, cost_status, cost_source FROM sessions WHERE id='t'"
                ).fetchone()
                grow = a._session_db._conn.execute(
                    "SELECT session_id, model, provider, task FROM generation_ids "
                    "WHERE generation_id='gen-1789811821-abc'").fetchone()
        finally:
            a.close()
        assert srow["actual_cost_usd"] == pytest.approx(0.000361)
        assert srow["cost_status"] == "actual"
        assert srow["cost_source"] == "provider_cost_api"
        assert grow["session_id"] == "t"
        assert grow["provider"] == "nous"
        assert grow["task"] == ""
        assert a.session_billed_cost_usd == pytest.approx(0.000361)

    def test_estimate_still_recorded_alongside_billed_cost(self, tmp_path, monkeypatch):
        a = _agent(tmp_path, monkeypatch)
        try:
            from agent import turn_usage
            turn_usage.record_response_usage(
                a, _response(0.000361), messages=[{"role": "user", "content": "hi"}],
                api_call_count=1, api_duration=0.2, compression_attempts=0, max_compression_attempts=3,
            )
            assert a._session_db.flush_token_counts()
            with a._session_db._lock:
                row = a._session_db._conn.execute(
                    "SELECT estimated_cost_usd, actual_cost_usd FROM sessions WHERE id='t'").fetchone()
        finally:
            a.close()
        # the estimate stays populated (fallback + delta-vs-billed drift signal)
        assert row["estimated_cost_usd"] > 0
        assert row["actual_cost_usd"] == pytest.approx(0.000361)

    def test_log_line_carries_billed(self, tmp_path, monkeypatch, caplog):
        a = _agent(tmp_path, monkeypatch)
        try:
            from agent import turn_usage
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="agent.turn_usage"):
                turn_usage.record_response_usage(
                    a, _response(0.000123), messages=[{"role": "user", "content": "hi"}],
                    api_call_count=1, api_duration=0.2, compression_attempts=0, max_compression_attempts=3,
                )
            line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("API call #"))
        finally:
            a.close()
        assert " billed=0.000123" in line
        # the pre-existing prefix is unchanged, so older parsers keep matching
        assert line.startswith(
            "API call #1: model=anthropic/claude-fable-5.1 provider=nous "
            "in=62889 out=7 total=62896 latency=0.2s"
        )

    def test_no_billed_cost_keeps_estimated_status(self, tmp_path, monkeypatch):
        a = _agent(tmp_path, monkeypatch)
        try:
            from agent import turn_usage
            turn_usage.record_response_usage(
                a, _response(None, gid=None), messages=[{"role": "user", "content": "hi"}],
                api_call_count=1, api_duration=0.2, compression_attempts=0, max_compression_attempts=3,
            )
            assert a._session_db.flush_token_counts()
            with a._session_db._lock:
                row = a._session_db._conn.execute(
                    "SELECT actual_cost_usd, cost_status FROM sessions WHERE id='t'").fetchone()
                gids = a._session_db._conn.execute(
                    "SELECT COUNT(*) FROM generation_ids").fetchone()[0]
        finally:
            a.close()
        assert row["actual_cost_usd"] is None  # NULL = never billed, not billed-$0
        assert row["cost_status"] in ("estimated", "unknown")
        assert gids == 0
