"""Subscription-included cost accounting (Claude Max via DirectSDK).

Two figures, never summed together: out-of-pocket spend
(``estimated_cost_usd``/``actual_cost_usd`` = 0, ``cost_status='included'``,
``cost_source='subscription_included'``) and the native list-price equivalent
(``list_price_equiv_usd``), kept as the subscription-allowance gauge.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB

DIRECTSDK = "claude-subscription-route"


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "test_state.db")
    yield session_db
    session_db.close()


def _stub_agent(db):
    """Minimal agent stand-in for record_response_usage (full AIAgent construction
    is not needed for the ledger assertions)."""
    compressor = SimpleNamespace(
        awaiting_real_usage_after_compression=False,
        update_from_response=lambda *_a, **_k: None,
        note_usage_less_response=lambda: None,
        _verify_compaction_cleared_threshold=False,
        threshold_tokens=0,
    )
    agent = SimpleNamespace(
        context_compressor=compressor, session_api_calls=0,
        session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0,
        session_input_tokens=0, session_output_tokens=0, session_cache_read_tokens=0,
        session_cache_write_tokens=0, session_reasoning_tokens=0,
        session_estimated_cost_usd=0.0, session_list_price_usd=0.0,
        session_cost_status="unknown", session_cost_source="none",
        _api_latency_history=[], _api_output_history=[],
        verbose_logging=False, quiet_mode=True,
        model="claude-max-route", provider=DIRECTSDK, base_url=f"process://{DIRECTSDK}",
        api_mode="chat_completions", api_key="", client=None,
        _session_db=db, _session_db_created=True, session_id="t",
    )
    return agent


def _usage_response():
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=62_889, completion_tokens=7, total_tokens=62_896, cost=None,
            prompt_tokens_details=SimpleNamespace(cached_tokens=34_283),
            completion_tokens_details=None,
        ),
        id="msg_probe01", model="claude-max-route",
    )


class TestLedgerSubscriptionIncluded:
    def test_included_row_zero_spend_with_list_price(self, db, tmp_path, monkeypatch):
        # Isolated HERMES_HOME: the turn_usage import chain stats <home>/manifest.json.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent import turn_usage
        from agent.usage_pricing import CostResult

        a = _stub_agent(db)

        def fake_estimate(model, usage, *, provider=None, base_url=None, api_key=None):
            return CostResult(amount_usd=Decimal(0), status="included", source="subscription_included",
                              label="included", list_price_usd=Decimal("1.25"),
                              notes=("subscription-included (Claude Max)",))

        monkeypatch.setattr(turn_usage, "estimate_usage_cost", fake_estimate)
        monkeypatch.setattr(turn_usage, "_loop_mod", lambda: SimpleNamespace(
            _should_rearm_compression_budget=lambda *a, **k: False))
        monkeypatch.setattr(turn_usage, "_agent_session_source", lambda agent: "cli")
        turn_usage.record_response_usage(
            a, _usage_response(), messages=[{"role": "user", "content": "hi"}],
            api_call_count=1, api_duration=0.2, compression_attempts=0, max_compression_attempts=3,
        )
        assert db.flush_token_counts()
        with db._lock:
            srow = db._conn.execute(
                "SELECT estimated_cost_usd, actual_cost_usd, cost_status, cost_source, "
                "list_price_equiv_usd, billing_mode FROM sessions WHERE id='t'").fetchone()
            mrow = db._conn.execute(
                "SELECT estimated_cost_usd, actual_cost_usd, cost_status, cost_source, list_price_equiv_usd "
                "FROM session_model_usage WHERE session_id='t' AND task=''").fetchone()
        assert srow["estimated_cost_usd"] == 0
        assert srow["actual_cost_usd"] is None
        assert srow["cost_status"] == "included"
        assert srow["cost_source"] == "subscription_included"
        assert srow["list_price_equiv_usd"] == pytest.approx(1.25)
        assert srow["billing_mode"] == "subscription_included"
        assert mrow["estimated_cost_usd"] == 0
        assert mrow["cost_status"] == "included"
        assert mrow["cost_source"] == "subscription_included"
        assert mrow["list_price_equiv_usd"] == pytest.approx(1.25)
        # Spend stays truthful ($0) while the gauge accumulates apart from it.
        assert a.session_estimated_cost_usd == 0
        assert a.session_list_price_usd == pytest.approx(1.25)


class TestCodexIncludedRoute:
    def test_codex_route_is_included_at_zero_without_a_gauge(self):
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

        result = estimate_usage_cost(
            "gpt-5.4-mini",
            CanonicalUsage(input_tokens=1000, output_tokens=500),
            provider="openai-codex",
        )
        assert result.status == "included"
        assert result.amount_usd == Decimal("0")
        assert result.source == "none"
        assert result.list_price_usd is None
        assert result.notes


class TestOpenRouterBilledRowsUnchanged:
    def test_billed_cost_row_keeps_actual_status_and_no_list_price(self, db):
        db.update_token_counts(
            "s4", input_tokens=100, output_tokens=5, model="z-ai/glm-5.3",
            billing_provider="openrouter", api_call_count=1,
            estimated_cost_usd=0.01, actual_cost_usd=0.0091,
            cost_status="actual", cost_source="provider_cost_api",
        )
        db.update_token_counts(
            "s4", input_tokens=50, output_tokens=2, model="z-ai/glm-5.3",
            billing_provider="openrouter", api_call_count=1,
            estimated_cost_usd=0.005, actual_cost_usd=0.004,
        )
        with db._lock:
            row = db._conn.execute(
                "SELECT estimated_cost_usd, actual_cost_usd, cost_status, cost_source, "
                "list_price_equiv_usd FROM sessions WHERE id='s4'").fetchone()
            mrow = db._conn.execute(
                "SELECT estimated_cost_usd, actual_cost_usd, cost_status, cost_source "
                "FROM session_model_usage WHERE session_id='s4'").fetchone()
        assert row["estimated_cost_usd"] == pytest.approx(0.015)
        assert row["actual_cost_usd"] == pytest.approx(0.0131)
        assert row["cost_status"] == "actual"
        assert row["cost_source"] == "provider_cost_api"
        assert row["list_price_equiv_usd"] == 0  # no list-price gauge on billed routes
        assert mrow["cost_status"] == "actual"
        assert mrow["cost_source"] == "provider_cost_api"


class TestInsightsIncludedBucket:
    def test_stored_cost_status_wins_over_token_only_reestimate(self):
        from agent.insights import InsightsEngine, _estimate_cost

        engine = object.__new__(InsightsEngine)
        ds_session = {
            "model": "claude-max-route", "billing_provider": DIRECTSDK,
            "billing_base_url": f"process://{DIRECTSDK}", "input_tokens": 100, "output_tokens": 50,
            "cache_read_tokens": 0, "cache_write_tokens": 0, "reasoning_tokens": 0,
            "estimated_cost_usd": 0.0, "actual_cost_usd": None,
            "cost_status": "included",  # the ledger row is the recorded truth
            "tool_call_count": 0, "message_count": 2, "started_at": 1.0, "ended_at": 2.0,
        }
        billed_session = {
            "model": "z-ai/glm-5.3", "billing_provider": "openrouter",
            "billing_base_url": "https://openrouter.ai/api/v1", "input_tokens": 100, "output_tokens": 50,
            "cache_read_tokens": 0, "cache_write_tokens": 0, "reasoning_tokens": 0,
            "estimated_cost_usd": 0.01, "actual_cost_usd": 0.0091,
            "cost_status": "actual",
            "tool_call_count": 0, "message_count": 2, "started_at": 1.0, "ended_at": 2.0,
        }
        no_status_session = {
            "model": "z-ai/glm-5.3", "billing_provider": "openrouter",
            "billing_base_url": "https://openrouter.ai/api/v1", "input_tokens": 100, "output_tokens": 50,
            "cache_read_tokens": 0, "cache_write_tokens": 0, "reasoning_tokens": 0,
            "cost_status": None,  # falls back to the re-estimate
            "tool_call_count": 0, "message_count": 2, "started_at": 1.0, "ended_at": 2.0,
        }
        stats = {"total_messages": 6, "user_messages": 3, "assistant_messages": 3, "tool_messages": 0}
        o = engine._compute_overview([ds_session, billed_session, no_status_session], stats)
        assert o["included_cost_sessions"] == 1
        assert o["unknown_cost_sessions"] == 0
        # The token-only re-estimate cannot say "included" (no native cost in the
        # session row) — the stored status is what puts the session in the bucket;
        # the DS session itself contributes $0 to the estimate.
        ds_est, ds_status = _estimate_cost(ds_session)
        assert ds_est == 0 and ds_status == "unknown"
        assert 0 < o["estimated_cost"] < 0.01  # only the re-estimated billed routes
        lines = engine._cost_lines(
            o, ("  Estimated: {}", "  Included: {} session(s) (subscription)", "  Unknown: {} session(s)"))
        assert any("Included: 1 session(s)" in ln for ln in lines)