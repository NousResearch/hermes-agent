"""Negative catalog rates are "pricing varies" sentinels, never prices."""

from decimal import Decimal

import pytest

from agent import usage_pricing
from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

_USAGE = CanonicalUsage(input_tokens=1_000_000, output_tokens=1_000_000)


def _cost(monkeypatch, pricing):
    monkeypatch.setattr(usage_pricing, "fetch_model_metadata", lambda *a, **k: {"openrouter/auto": {"pricing": pricing}})
    return estimate_usage_cost("openrouter/auto", _USAGE, provider="openrouter", base_url="https://openrouter.ai/api/v1")


def test_variable_price_sentinel_is_unknown_not_negative(monkeypatch):
    # OpenRouter's live catalog entry for its auto router.
    result = _cost(monkeypatch, {"prompt": "-1", "completion": "-1"})
    assert result.amount_usd is None or result.amount_usd >= 0
    assert result.status == "unknown"


@pytest.mark.parametrize("pricing,expected", [
    ({"prompt": "-1", "completion": "0.000002"}, None),  # one side unknown -> the total is unknown, never under-counted
    ({"prompt": "0", "completion": "0"}, Decimal(0)),  # free models keep their zero price
    ({"prompt": "-0", "completion": "0.000002"}, Decimal(2)),
])
def test_mixed_and_zero_rates(monkeypatch, pricing, expected):
    result = _cost(monkeypatch, pricing)
    assert result.amount_usd == expected


@pytest.mark.parametrize("pricing", [
    {"prompt": "0.000001", "completion": "0.000002", "request": "-1"},  # unknown fee, not "no fee"
    {"prompt": "NaN", "completion": "0.000002"},
    {"prompt": "0.000001", "completion": "Infinity"},
])
def test_sentinel_request_fee_and_non_finite_rates_are_unknown(monkeypatch, pricing):
    result = _cost(monkeypatch, pricing)
    assert (result.status, result.amount_usd) == ("unknown", None)
