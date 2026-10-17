"""Regression guard: MoA reference slots must record real token usage (#123157).

``call_llm`` rebuilds every auxiliary adapter's usage as an OpenAI-Chat-shaped
object (``prompt_tokens``/``completion_tokens``) no matter which provider the
slot targets, but ``_price_reference_response`` used to normalize that object
with the slot's OWN provider/api_mode. For Codex-Responses and Anthropic slots
that selected a shape whose field names never exist on the adapted object, so
every reference recorded ``usage: {all zeros}`` and ``cost_usd: 0`` — advisor
spend was invisible in traces and session totals.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest


@pytest.fixture
def priced_providers(monkeypatch):
    """Stub pricing so the test observes which provider each slot is priced at."""
    seen: list[str | None] = []

    def fake_estimate(model_name, usage, *, provider=None, **kwargs):
        seen.append(provider)
        return SimpleNamespace(amount_usd=Decimal("1.25"), status="estimated", source="stub")

    monkeypatch.setattr("agent.usage_pricing.estimate_usage_cost", fake_estimate)
    return seen


def test_codex_reference_slot_records_chat_shaped_usage(priced_providers):
    from agent.moa_loop import _price_reference_response

    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=20000, completion_tokens=1500, total_tokens=21500)
    )
    slot = {"model": "gpt-6-sol"}
    runtime = {"provider": "openai-codex", "api_mode": "codex_responses"}

    usage, amount, status, _source = _price_reference_response(response, slot, runtime)

    assert usage.input_tokens == 20000
    assert usage.output_tokens == 1500
    assert amount == Decimal("1.25")
    assert status == "estimated"
    # Pricing still runs at the slot's OWN provider rate, not the Chat shape.
    assert priced_providers == ["openai-codex"]


def test_anthropic_reference_slot_records_chat_shaped_usage(priced_providers):
    from agent.moa_loop import _price_reference_response

    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=562, completion_tokens=400, total_tokens=962)
    )
    slot = {"model": "claude-opus-5-5"}
    runtime = {"provider": "anthropic", "api_mode": "anthropic_messages"}

    usage, _amount, _status, _source = _price_reference_response(response, slot, runtime)

    assert usage.input_tokens == 562
    assert usage.output_tokens == 400
    assert priced_providers == ["anthropic"]


def test_chat_shape_cache_details_are_honored_when_adapters_expose_them(priced_providers):
    """If an adapter carries Chat-style detail buckets (cf. #105273/#71287), the
    cached prefix is split out of the prompt total instead of double-billing."""
    from agent.moa_loop import _price_reference_response

    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=20000,
            completion_tokens=1500,
            total_tokens=21500,
            prompt_tokens_details={"cached_tokens": 18000},
        )
    )
    slot = {"model": "gpt-6-sol"}
    runtime = {"provider": "openai-codex", "api_mode": "codex_responses"}

    usage, _amount, _status, _source = _price_reference_response(response, slot, runtime)

    assert usage.input_tokens == 2000
    assert usage.cache_read_tokens == 18000
    assert usage.output_tokens == 1500
