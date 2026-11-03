"""The provider-reported cost is recorded at the turn seam.

When the route reports its own billed cost for the call (OpenRouter's
``usage.cost``), that figure is the call's ACTUAL cost: it must land in the
actual accumulator with ``cost_status='actual'`` / ``cost_source='provider_response'``
and must NOT also be estimated into ``session_estimated_cost_usd`` (insights
reconciles estimated vs actual, so a double count would inflate the session).
Routes that report no cost keep the existing estimation path unchanged.

Synthetic usage objects only — no provider I/O, no real agent (the turn seam is
exercised on a minimal stand-in, like the other turn_usage tests).
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest


class _Compressor:
    """Just the surface record_response_usage touches."""

    threshold_tokens = 1_000_000
    awaiting_real_usage_after_compression = False
    _verify_compaction_cleared_threshold = False

    def __init__(self):
        self.updates = []

    def update_from_response(self, usage_dict):
        self.updates.append(usage_dict)

    def note_usage_less_response(self):
        pass


class _DB:
    """Records queued token/cost deltas instead of writing them."""

    def __init__(self):
        self.calls = []

    def queue_token_counts(self, session_id, **kwargs):
        self.calls.append((session_id, kwargs))

    def patch_session_model_config(self, *args, **kwargs):
        return None


def _agent(**overrides):
    agent = SimpleNamespace(
        model="@preset/test-preset", provider="openrouter", api_mode="chat_completions",
        base_url="https://openrouter.ai/api/v1", api_key="k", session_id="t",
        quiet_mode=True, verbose_logging=False, log_prefix="", client=None,
        context_compressor=_Compressor(), _session_db=_DB(), _session_db_created=True,
        _safe_print=lambda *a, **k: None, _vprint=lambda *a, **k: None,
        session_api_calls=0, session_prompt_tokens=0, session_completion_tokens=0,
        session_total_tokens=0, session_input_tokens=0, session_output_tokens=0,
        session_cache_read_tokens=0, session_cache_write_tokens=0,
        session_reasoning_tokens=0, session_estimated_cost_usd=0.0,
        session_actual_cost_usd=0.0, session_cost_status="unknown", session_cost_source="none",
        _usage_anchor=None, _turn_base_usage_anchor=None, _persist_disabled=True,
        _api_latency_history=None, _api_output_history=None,
    )
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def _usage(cost=None, prompt=1000, completion=100):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion,
             "total_tokens": prompt + completion}
    if cost is not None:
        usage["cost"] = cost
    return SimpleNamespace(usage=usage, model="@preset/test-preset", provider="SomeUpstream")


def _run(agent, response, monkeypatch, estimate=None):
    from agent import turn_usage

    calls = []

    def _fake_estimate(*args, **kwargs):
        calls.append((args, kwargs))
        return estimate

    monkeypatch.setattr(turn_usage, "estimate_usage_cost", _fake_estimate)
    # The compression-budget helper lives in conversation_loop, whose import runs the
    # entry-point bootstrap; a fixed verdict keeps this a unit test of the accounting seam.
    monkeypatch.setattr(
        turn_usage, "_loop_mod",
        lambda: SimpleNamespace(_should_rearm_compression_budget=lambda *a, **k: False),
    )
    turn_usage.record_response_usage(
        agent, response, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
        api_duration=0.2, compression_attempts=0, max_compression_attempts=3,
    )
    return calls


def _estimate(amount="0.25", status="estimated", source="provider_models_api"):
    return SimpleNamespace(amount_usd=None if amount is None else Decimal(amount),
                           status=status, source=source)


def test_provider_cost_is_recorded_as_actual_and_not_estimated(monkeypatch):
    agent = _agent()
    billed = 0.000635248
    calls = _run(agent, _usage(cost=billed), monkeypatch, estimate=_estimate())

    # The provider's figure is the actual cost...
    assert agent.session_actual_cost_usd == pytest.approx(billed)
    assert agent.session_cost_status == "actual"
    assert agent.session_cost_source == "provider_response"
    # ...and the estimate was neither consulted nor accumulated (no double count).
    assert calls == []
    assert agent.session_estimated_cost_usd == 0.0
    (_sid, kwargs), = agent._session_db.calls
    assert kwargs["actual_cost_usd"] == pytest.approx(billed)
    assert kwargs["estimated_cost_usd"] is None
    assert kwargs["cost_status"] == "actual"
    assert kwargs["cost_source"] == "provider_response"


def test_absent_provider_cost_keeps_the_estimation_path(monkeypatch):
    agent = _agent()
    calls = _run(agent, _usage(), monkeypatch, estimate=_estimate("0.0042"))

    assert len(calls) == 1
    assert agent.session_estimated_cost_usd == pytest.approx(0.0042)
    assert agent.session_actual_cost_usd == 0.0
    assert agent.session_cost_status == "estimated"
    assert agent.session_cost_source == "provider_models_api"
    (_sid, kwargs), = agent._session_db.calls
    assert kwargs["actual_cost_usd"] is None
    assert kwargs["estimated_cost_usd"] == pytest.approx(0.0042)
    assert kwargs["cost_status"] == "estimated"


@pytest.mark.parametrize("cost", [-0.5, float("nan"), float("inf"), "not-a-number"])
def test_unusable_provider_cost_falls_through_to_estimation(monkeypatch, cost):
    """A negative or non-finite figure is not a price; estimation still runs (fail-open)."""
    agent = _agent()
    calls = _run(agent, _usage(cost=cost), monkeypatch, estimate=_estimate("0.01"))

    assert len(calls) == 1
    assert agent.session_cost_status == "estimated"
    assert agent.session_estimated_cost_usd == pytest.approx(0.01)
    assert agent.session_actual_cost_usd == 0.0


def test_zero_provider_cost_is_a_real_zero(monkeypatch):
    """A free route bills exactly $0 — an actual figure, not a missing one."""
    agent = _agent()
    calls = _run(agent, _usage(cost=0), monkeypatch, estimate=_estimate("0.5"))

    assert calls == []
    assert agent.session_actual_cost_usd == 0.0
    assert agent.session_cost_status == "actual"
    assert agent.session_cost_source == "provider_response"


def test_provider_cost_accumulates_across_calls(monkeypatch):
    agent = _agent()
    _run(agent, _usage(cost=0.001), monkeypatch, estimate=_estimate())
    _run(agent, _usage(cost=0.002), monkeypatch, estimate=_estimate())

    assert agent.session_actual_cost_usd == pytest.approx(0.003)
    assert agent.session_estimated_cost_usd == 0.0


def test_moa_fold_with_provider_cost_does_not_raise(monkeypatch):
    """``CanonicalUsage.__add__`` drops ``raw_usage``; a provider cost carried there must not
    break the fold (no ``cost`` field was added to the dataclass for exactly this reason)."""
    from agent.usage_pricing import CanonicalUsage

    ref_usage = CanonicalUsage(input_tokens=400, output_tokens=40, raw_usage={"cost": 0.5})
    folded = CanonicalUsage(input_tokens=1000, output_tokens=100, raw_usage={"cost": 0.25}) + ref_usage
    assert (folded.prompt_tokens, folded.output_tokens) == (1400, 140)

    class _MoAClient:
        last_aggregator_slot = None

        def consume_reference_usage(self):
            return ref_usage, 0.02

        def consume_and_save_trace(self, *args, **kwargs):
            return None

    agent = _agent(client=_MoAClient())
    billed = 0.0009
    _run(agent, _usage(cost=billed), monkeypatch, estimate=_estimate("0.1"))

    # The aggregator is billed exactly; advisor fan-out stays priced, not billed.
    assert agent.session_actual_cost_usd == pytest.approx(billed)
    assert agent.session_estimated_cost_usd == pytest.approx(0.02)
    assert agent.session_cost_status == "actual"
    assert agent.session_cost_source == "provider_response"
    (_sid, kwargs), = agent._session_db.calls
    assert kwargs["actual_cost_usd"] == pytest.approx(billed)
    assert kwargs["estimated_cost_usd"] == pytest.approx(0.02)


def test_provider_cost_read_from_dict_usage_without_raw_usage():
    """``normalize_usage`` may produce no ``raw_usage``; a plain ``usage['cost']`` is still read."""
    from agent.turn_usage import _provider_actual_cost

    assert _provider_actual_cost(None, SimpleNamespace(usage={"cost": 0.5})) == 0.5
    assert _provider_actual_cost(SimpleNamespace(raw_usage=None), SimpleNamespace(usage={"cost": 0.5})) == 0.5
    assert _provider_actual_cost(SimpleNamespace(raw_usage={"cost": 0.75}), None) == 0.75
    assert _provider_actual_cost(SimpleNamespace(raw_usage={}), None) is None
    assert _provider_actual_cost(None, SimpleNamespace(usage=None)) is None


def test_cost_source_literal_admits_provider_response():
    """The new source must be part of the alias, not merely a string used at runtime."""
    from typing import get_args

    from agent.usage_pricing import CostSource

    assert "provider_response" in get_args(CostSource)
