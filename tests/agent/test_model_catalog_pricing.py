"""The remote model catalog prices a model the install ships no rates for.

A model published after a release cost "unknown" until the next release carried its rates. The
cached manifest now publishes the bundled rate snapshots. A bundled rate is never replaced, and a
malformed manifest rate is unknown, never a price.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from agent import usage_pricing
from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
from hermes_cli import model_catalog

NEW_CLAUDE = "claude-test-9"
ONE_MILLION_EACH = CanonicalUsage(input_tokens=1_000_000, output_tokens=1_000_000)
RATES = {"input_cost_per_million": "1.00", "output_cost_per_million": "2.00", "pricing_version": "test-2026-11"}


@pytest.fixture
def cached_pricing():
    """Write a manifest with this ``pricing`` section into the test's ``HERMES_HOME`` cache."""
    model_catalog.reset_cache()

    def write(pricing: dict) -> None:
        path = Path(os.environ["HERMES_HOME"]) / "cache" / "model_catalog.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "providers": {}, "pricing": pricing}), encoding="utf-8")

    yield write
    model_catalog.reset_cache()


def _cost(model: str, usage: CanonicalUsage = ONE_MILLION_EACH):
    return estimate_usage_cost(model, usage, provider="anthropic")


def test_a_model_without_bundled_rates_is_priced_from_the_manifest(cached_pricing):
    model = f"{NEW_CLAUDE}-1"
    cached_pricing({"official_docs": {"anthropic": {model: RATES}}})

    result = _cost(model)

    assert result.amount_usd == Decimal("3.00")
    assert (result.status, result.source, result.pricing_version) == ("estimated", "official_docs_snapshot", "test-2026-11")
    assert _cost(f"anthropic/{NEW_CLAUDE}.1").amount_usd == Decimal("3.00")  # prefix and dotted version, as a bundled row


@pytest.mark.parametrize("bad", [
    {"input_cost_per_million": "-1", "output_cost_per_million": "2.00"},
    {"input_cost_per_million": "NaN", "output_cost_per_million": "2.00"},
    {"input_cost_per_million": "1.00", "output_cost_per_million": "Infinity"},
    {"input_cost_per_million": "1.00", "output_cost_per_million": "2.00", "request_cost": "-1"},
    {"input_cost_per_million": "1.00"},
    {"input_cost_per_million": "cheap", "output_cost_per_million": "2.00"},
])
def test_a_malformed_manifest_rate_is_unknown_never_a_price(cached_pricing, bad):
    cached_pricing({"official_docs": {"anthropic": {NEW_CLAUDE: bad}}})

    assert _cost(NEW_CLAUDE).amount_usd is None


def test_a_fast_mode_response_is_priced_from_the_manifest_fast_rates(cached_pricing):
    fast_rates = {"input_cost_per_million": "10.00", "output_cost_per_million": "20.00"}
    cached_pricing({"official_docs": {"anthropic": {NEW_CLAUDE: RATES}}, "anthropic_fast_mode": {NEW_CLAUDE: fast_rates}})
    served_fast = CanonicalUsage(input_tokens=1_000_000, output_tokens=1_000_000, raw_usage={"speed": "fast"})

    assert _cost(NEW_CLAUDE, served_fast).amount_usd == Decimal("30.00")


def test_an_ultrafast_response_is_priced_from_the_manifest_ultrafast_rates(cached_pricing):
    cached_pricing({
        "official_docs": {"openai": {"gpt-9-test": RATES}},
        "openai_ultrafast": {"gpt-9-test": {"input_cost_per_million": "6.00", "output_cost_per_million": "12.00"}},
    })
    served = CanonicalUsage(input_tokens=1_000_000, output_tokens=1_000_000, raw_usage={"service_tier": "ultrafast"})

    result = estimate_usage_cost("gpt-9-test", served, provider="openai", base_url="https://api.openai.com/v1")

    assert result.amount_usd == Decimal("18.00")


@pytest.mark.parametrize(("tier", "hour", "expected"), [
    ("priority", 7, "6"), ("fast", 7, "6"),
    ("ultrafast", 7, "9"), ("ultrafast", 8, "18"),  # the launch discount ends 2027-03-01 08:00 UTC
])
def test_a_model_the_catalog_added_to_the_gpt62_contract_is_priced_at_the_tier_it_was_served(
        cached_pricing, monkeypatch, tier, hour, expected):
    model = "gpt-6.2-test"
    cached_pricing({"official_docs": {"openai": {model: RATES}}})
    cache = Path(os.environ["HERMES_HOME"], "cache", "model_catalog.json")
    manifest = json.loads(cache.read_text(encoding="utf-8"))
    manifest["model_sets"] = {"openai_gpt62_contract": [model]}
    cache.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(usage_pricing, "_UTC_NOW", lambda: datetime(2027, 3, 1, hour, tzinfo=UTC))
    served = CanonicalUsage(input_tokens=1_000_000, output_tokens=1_000_000, raw_usage={"service_tier": tier})

    result = estimate_usage_cost(model, served, provider="openai", base_url="https://api.openai.com/v1")

    assert result.amount_usd == Decimal(expected)


@pytest.mark.parametrize(("url", "advertised", "expected"), [
    ("https://relay.example/v1", {"prompt": "0.000004", "completion": "0.000008"}, Decimal(12)),  # the relay's own rate
    ("https://eu.api.openai.com/v1", None, Decimal(3)),  # no rate advertised: the catalog's
])
def test_a_rate_the_endpoint_advertises_wins_over_the_catalogs(cached_pricing, monkeypatch, url, advertised, expected):
    model = "catalog-test-model"
    cached_pricing({"official_docs": {"openai": {model: RATES}}})
    monkeypatch.setattr(usage_pricing, "fetch_endpoint_model_metadata",
                        lambda *a, **kw: {model: {"pricing": advertised}} if advertised else {})

    assert estimate_usage_cost(model, ONE_MILLION_EACH, provider="openai", base_url=url).amount_usd == expected


@pytest.mark.parametrize("tier", ["priority", "fast", "ultrafast"])
@pytest.mark.parametrize("catalog_has_rate", [False, True])
def test_a_contract_tier_never_reprices_an_endpoints_own_tariff(cached_pricing, monkeypatch, tier, catalog_has_rate):
    model = "gpt-6.2-test"
    cached_pricing({"official_docs": {"openai": {model: RATES} if catalog_has_rate else {}}})
    cache = Path(os.environ["HERMES_HOME"], "cache", "model_catalog.json")
    manifest = json.loads(cache.read_text(encoding="utf-8"))
    manifest["model_sets"] = {"openai_gpt62_contract": [model]}
    cache.write_text(json.dumps(manifest), encoding="utf-8")
    rates = {"prompt": "0.000004", "completion": "0.000008"}
    monkeypatch.setattr(usage_pricing, "fetch_endpoint_model_metadata", lambda *a, **kw: {model: {"pricing": rates}})
    route = {"provider": "openai", "base_url": "https://relay.example/v1"}
    served = CanonicalUsage(input_tokens=1_000_000, output_tokens=1_000_000, raw_usage={"service_tier": tier})

    assert estimate_usage_cost(model, ONE_MILLION_EACH, **route).amount_usd == Decimal(12)
    assert estimate_usage_cost(model, served, **route).amount_usd is None


@pytest.mark.parametrize("threshold", ["200000", 0, -1, True, None])
def test_a_malformed_or_missing_tier_threshold_discards_the_entry(cached_pricing, threshold):
    row = dict(RATES, input_cost_per_million_above="4.00", output_cost_per_million_above="8.00")
    if threshold is not None:
        row["tier_threshold_tokens"] = threshold
    cached_pricing({"official_docs": {"anthropic": {NEW_CLAUDE: row}}})
    large_prompt = CanonicalUsage(input_tokens=300_000, output_tokens=1_000_000)

    assert _cost(NEW_CLAUDE, large_prompt).amount_usd is None
