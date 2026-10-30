"""Zen regression contracts complementing the external-catalog PR (#71911).

Controlled rates, real catalog lookup and cost calculation; no public-price snapshots.
"""

from copy import deepcopy
from decimal import Decimal

import pytest

from agent import models_dev, usage_pricing


BASE = {"input": 2, "output": 7, "cache_read": 0.25, "cache_write": 3}
ABOVE = {"input": 5, "output": 13, "cache_read": 0.5, "cache_write": 8}
THRESHOLD = 272_000
MODEL = "relay-context-fixture"
ENDPOINT = "https://opencode.ai/zen/v1"


@pytest.fixture
def seed_cost(monkeypatch):
    monkeypatch.setattr(usage_pricing, "fetch_endpoint_model_metadata", lambda *a, **kw: {})

    def seed(cost):
        monkeypatch.setattr(models_dev, "_models_dev_cache", {
            "opencode": {"models": {MODEL: {"cost": deepcopy(cost)}}},
        })
        monkeypatch.setattr(models_dev, "_models_dev_cache_time", float("inf"))

    return seed


@pytest.mark.parametrize("prompt_tokens", [THRESHOLD - 1, THRESHOLD, THRESHOLD + 1, 300_000])
def test_context_tier_prices_the_whole_request_including_both_caches(seed_cost, prompt_tokens):
    seed_cost({**BASE, "tiers": [{"tier": {"type": "context", "size": THRESHOLD}, **ABOVE}]})
    usage = usage_pricing.normalize_usage({
        "input_tokens": prompt_tokens,
        "output_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 100_000, "cache_write_tokens": 100_000},
        "output_tokens_details": {"reasoning_tokens": 900},
    }, provider="opencode-zen", api_mode="codex_responses")
    rates = ABOVE if prompt_tokens > THRESHOLD else BASE
    expected = sum(Decimal(tokens) * Decimal(str(rates[key])) for key, tokens in (
        ("input", prompt_tokens - 200_000), ("output", 1000),
        ("cache_read", 100_000), ("cache_write", 100_000),
    )) / Decimal(1_000_000)

    result = usage_pricing.estimate_usage_cost(MODEL, usage, provider="opencode-zen", base_url=ENDPOINT)

    assert (result.status, result.source) == ("estimated", "provider_models_api")
    assert result.amount_usd == expected
    assert usage.prompt_tokens == prompt_tokens
    assert usage.reasoning_tokens == 900  # Already inside output_tokens, not charged again.


@pytest.mark.parametrize("cost,known", [
    (BASE, True),
    (dict.fromkeys(BASE, 0), True),
    ({}, False),
    ({"input": 2, "output": 7, "cache_write": 3}, False),
    ({**BASE, "tiers": [{"tier": {"type": "context", "size": THRESHOLD},
                         "input": 5, "output": 13, "cache_write": 8}]}, False),
    ({**BASE, "tiers": [{"tier": {"type": "context", "size": -1}, **ABOVE}]}, False),
    ({**BASE, "tiers": [{"tier": {"type": "unknown", "size": THRESHOLD}, **ABOVE}]}, False),
    ({**BASE, "tiers": [
        {"tier": {"type": "context", "size": THRESHOLD}, **ABOVE},
        {"tier": {"type": "context", "size": 400_000}, **ABOVE},
    ]}, False),
])
def test_incomplete_or_unrepresentable_catalog_is_unknown_not_a_flat_price(seed_cost, cost, known):
    """PricingEntry supports one context tier; never silently discard other pricing rules."""
    seed_cost(cost)
    usage = usage_pricing.CanonicalUsage(
        input_tokens=100_000, output_tokens=1000, cache_read_tokens=100_000, cache_write_tokens=100_000,
    )

    result = usage_pricing.estimate_usage_cost(MODEL, usage, provider="opencode-zen", base_url=ENDPOINT)

    if known:
        expected = sum(Decimal(tokens) * Decimal(str(cost[key])) for key, tokens in (
            ("input", usage.input_tokens), ("output", usage.output_tokens),
            ("cache_read", usage.cache_read_tokens), ("cache_write", usage.cache_write_tokens),
        )) / Decimal(1_000_000)
        assert (result.status, result.amount_usd) == ("estimated", expected)
    else:
        assert (result.status, result.amount_usd) == ("unknown", None)
