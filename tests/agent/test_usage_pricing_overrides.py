"""Behaviour contracts: user ``pricing_overrides`` and Anthropic 1-hour cache-write pricing."""

from dataclasses import replace
from decimal import Decimal

import json

import pytest

from agent.usage_pricing import estimate_usage_cost, normalize_usage
from hermes_constants import get_hermes_home


def _write_config(data: dict) -> None:
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(json.dumps(data), encoding="utf-8")  # JSON is valid YAML


@pytest.mark.parametrize("entry, model, provider, base_url", [
    # A custom gateway model: no snapshot, no /models pricing.
    ({"provider": "custom", "model": "relay-model-1"}, "relay-model-1", "custom", "http://127.0.0.1:9/v1"),
    # ``custom:<name>`` spelling against the canonical route; full id vs basename.
    ({"provider": "custom:fireworks", "model": "kimi-k2p6"},
     "accounts/fireworks/models/kimi-k2p6", "custom:fireworks", "https://api.fireworks.ai/inference/v1"),
    # No provider: matches every route, and wins over the bundled snapshot.
    ({"model": "claude-opus-5"}, "claude-opus-5", "anthropic", None),
])
def test_pricing_overrides_in_config_price_the_route(entry, model, provider, base_url):
    _write_config({"pricing_overrides": [
        {**entry, "output_per_million": 9},  # missing a required rate: skipped, never masks the next
        {**entry, "input_per_million": 1, "output_per_million": 3, "cache_write_1h_per_million": 4},
    ]})
    usage = normalize_usage({
        "input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_creation_input_tokens": 1_000_000,
        "cache_creation": {"ephemeral_1h_input_tokens": 1_000_000},
    }, provider="anthropic", api_mode="anthropic_messages")

    result = estimate_usage_cost(model, usage, provider=provider, base_url=base_url)

    assert (result.source, result.status, result.amount_usd) == ("user_override", "estimated", Decimal(8))


@pytest.mark.parametrize("model, provider, served", [
    ("claude-opus-5", "anthropic", {"speed": "fast"}),
    ("gpt-6.1-sol", "openai", {"service_tier": "ultrafast"}),
    ("gpt-6-luna", "openai", {"service_tier": "ultrafast"}),  # no bundled Ultrafast row
])
def test_pricing_override_stays_final_at_a_served_speed(model, provider, served):
    _write_config({"pricing_overrides": [{"model": model, "input_per_million": 1, "output_per_million": 3}]})
    usage = replace(normalize_usage({"input_tokens": 1_000_000, "output_tokens": 1_000_000}, provider="anthropic"),
                    raw_usage=served)

    result = estimate_usage_cost(model, usage, provider=provider)

    assert (result.source, result.amount_usd) == ("user_override", Decimal(4))


@pytest.mark.parametrize("bad", [
    {"input_per_million": "NaN"},
    {"output_per_million": "Infinity"},
    {"input_per_million": -1},
    {"cache_read_per_million": "sNaN"},
    {"request_cost": -0.5},
])
def test_malformed_override_rate_is_skipped_for_the_next_valid_entry(bad):
    entry = {"provider": "custom", "model": "relay-model-1", "input_per_million": 1, "output_per_million": 3}
    _write_config({"pricing_overrides": [{**entry, **bad}, entry]})
    usage = normalize_usage({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})

    result = estimate_usage_cost("relay-model-1", usage, provider="custom", base_url="http://127.0.0.1:9/v1")

    assert (result.source, result.amount_usd, result.label) == ("user_override", Decimal(4), "~$4.00")


@pytest.mark.parametrize("model, five_min_rate, one_hour_rate", [
    ("claude-opus-5", "6.25", "10"),
    ("claude-opus-4.8", "6.25", "10"),
    ("claude-sonnet-5", "2.50", "4"),
    ("claude-sonnet-5-5", "2.50", "4"),
    ("claude-fable-5.1", "12.50", "20"),
    ("claude-mythos-5", "12.50", "20"),
])
def test_anthropic_1h_cache_writes_bill_at_the_1h_rate(model, five_min_rate, one_hour_rate):
    # https://platform.claude.com/docs/en/about-claude/pricing: 5m writes 1.25x input, 1h writes 2x.
    usage = normalize_usage({
        "input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 3_000_000,
        "cache_creation": {"ephemeral_5m_input_tokens": 1_000_000, "ephemeral_1h_input_tokens": 2_000_000},
    }, provider="anthropic")

    result = estimate_usage_cost(model, usage, provider="anthropic")

    assert result.amount_usd == Decimal(five_min_rate) + 2 * Decimal(one_hour_rate)
