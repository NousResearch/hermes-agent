"""Pricing for gateway and subscription routes.

Three defects this pins:
1. A route whose models a gateway serves had no price at all — every billable call through the
   fleet's Bifrost gateway came back ``cost_status='unknown'``.
2. A price list that quotes one prompt rate and no cache-hit rate marked every cached call
   unknown instead of billing the cache read at the prompt rate (strata's 540M cached tokens).
3. The gateway's published price for a vendor-prefixed id can be the vendor's peak card while the
   gateway bills the off-peak one; the vendor's own official snapshot has to win.
"""
from decimal import Decimal

from agent.usage_pricing import (
    CanonicalUsage,
    PricingEntry,
    estimate_usage_cost,
    get_pricing_entry,
    resolve_billing_route,
)


def _usage(input_tokens=0, output_tokens=0, cache_read_tokens=0, request_count=1):
    return CanonicalUsage(
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens, request_count=request_count,
    )


# --- a gateway route with no dispatch-time catalog mapping -----------------------

class TestGatewayRoutePricing:
    def test_gateway_route_without_catalog_alias_has_no_route(self):
        """Regression guard for the shape that produced the gap: a gateway provider the config
        does not map to a catalogued vendor must not silently resolve to a priced route."""
        route = resolve_billing_route("some-model", provider="some-gateway")
        assert route.billing_mode == "unknown"

    def test_endpoint_models_api_prices_a_gateway_model(self, monkeypatch):
        """A gateway that publishes per-model prices on its own ``/models`` is a price route:
        the model resolves through ``provider_models_api`` with its rates."""
        import agent.usage_pricing as pricing

        metadata = {
            "gateway-model": {"pricing": {"prompt": "0.000001", "completion": "0.000002"}},
        }
        monkeypatch.setattr(pricing, "fetch_endpoint_model_metadata", lambda *_a, **_k: metadata)
        result = estimate_usage_cost(
            "gateway-model", _usage(input_tokens=1_000_000, output_tokens=1_000_000),
            provider="a-gateway", base_url="https://gateway.example.com/v1",
        )
        assert result.status == "estimated"
        assert result.source == "provider_models_api"
        # $1/M input + $2/M output over one million each.
        assert result.amount_usd == Decimal("3.000000")


# --- a price list with no cache-hit rate ---------------------------------------

class TestCacheReadPricedAsInput:
    """A list quoting one prompt rate bills cache hits at that rate; the call is priced, and the
    entry says so instead of leaving its cache bucket unpriced (which returned ``unknown``)."""

    def test_entry_records_the_fallback(self, monkeypatch):
        import agent.usage_pricing as pricing

        metadata = {"m": {"pricing": {"prompt": "0.0000001", "completion": "0.0000003"}}}
        monkeypatch.setattr(pricing, "fetch_endpoint_model_metadata", lambda *_a, **_k: metadata)
        entry = get_pricing_entry("m", provider="g", base_url="https://g.example.com/v1")
        assert entry is not None
        assert entry.cache_read_priced_as_input is True
        assert entry.cache_read_cost_per_million is None

    def test_cached_call_is_priced_not_unknown(self, monkeypatch):
        import agent.usage_pricing as pricing

        metadata = {"m": {"pricing": {"prompt": "0.0000001", "completion": "0.0000003"}}}
        monkeypatch.setattr(pricing, "fetch_endpoint_model_metadata", lambda *_a, **_k: metadata)
        result = estimate_usage_cost(
            "m", _usage(input_tokens=1_000_000, cache_read_tokens=9_000_000, output_tokens=1_000_000),
            provider="g", base_url="https://g.example.com/v1",
        )
        assert result.status == "estimated"
        # 10M prompt tokens at $0.10/M + 1M output at $0.30/M.
        assert result.amount_usd == Decimal("1.3000000")

    def test_explicit_cache_rate_is_not_overridden(self, monkeypatch):
        import agent.usage_pricing as pricing

        metadata = {"m": {"pricing": {"prompt": "0.0000001", "completion": "0.0000003",
                                      "input_cache_read": "0.00000001"}}}
        monkeypatch.setattr(pricing, "fetch_endpoint_model_metadata", lambda *_a, **_k: metadata)
        entry = get_pricing_entry("m", provider="g", base_url="https://g.example.com/v1")
        assert entry is not None and entry.cache_read_priced_as_input is False
        assert entry.cache_read_cost_per_million == Decimal("0.01")


# --- vendor-prefixed gateway model ids -----------------------------------------

class TestUpstreamPrefixedModels:
    """``<vendor>/<model>`` ids the gateway serves: the vendor's official snapshot is the price
    route, ahead of the gateway's own (peak-vs-off-peak) quote for that id."""

    def test_vendor_snapshot_wins_over_gateway_metadata(self, monkeypatch):
        import agent.usage_pricing as pricing

        # The gateway publishes a rate for the same id; the vendor's snapshot must win.
        monkeypatch.setattr(
            pricing, "fetch_endpoint_model_metadata",
            lambda *_a, **_k: {"deepseek/deepseek-flash": {"pricing": {"prompt": "0.0000014"}}},
        )
        entry = get_pricing_entry(
            "deepseek/deepseek-flash", provider="bifrost", base_url="https://bifrost.example.com/v1",
        )
        assert entry is not None
        assert entry.source == "official_docs_snapshot"
        # Off-peak DeepSeek Flash card, not the gateway's peak quote.
        assert entry.input_cost_per_million == Decimal("0.15")

    def test_vendor_catalog_rate_is_the_fallback_when_no_snapshot(self, monkeypatch):
        import agent.usage_pricing as pricing

        monkeypatch.setattr(pricing, "fetch_endpoint_model_metadata", lambda *_a, **_k: {})
        monkeypatch.setattr(
            pricing, "_models_dev_pricing_entry",
            lambda route: (PricingEntry(input_cost_per_million=Decimal("0.5"),
                                        output_cost_per_million=Decimal("1.5"), source="provider_models_api")
                           if route.provider == "mistral" else None),
        )
        entry = get_pricing_entry(
            "mistral/some-model", provider="bifrost", base_url="https://bifrost.example.com/v1",
        )
        assert entry is not None and entry.input_cost_per_million == Decimal("0.5")


# --- subscription routes -------------------------------------------------------

class TestSubscriptionRoutes:
    def test_antigravity_oauth_is_subscription_included(self):
        route = resolve_billing_route("gemini-3.8-flash-low", provider="antigravity-oauth")
        assert route.billing_mode == "subscription_included"
        result = estimate_usage_cost("gemini-3.8-flash-low", _usage(input_tokens=1_000),
                                     provider="antigravity-oauth", base_url="agy://local")
        assert result.status == "included" and result.amount_usd == Decimal(0)

    def test_vibe_console_model_is_subscription_included_on_either_route(self):
        for provider, base_url in (("mistral-vibe", "https://api.eu.mistral.ai/v1"),
                                   ("bifrost", "https://bifrost.example.com/v1")):
            route = resolve_billing_route("mistral-vibe/mistral-vibe-cli-fast", provider=provider,
                                          base_url=base_url)
            assert route.billing_mode == "subscription_included", provider
            result = estimate_usage_cost("mistral-vibe/mistral-vibe-cli-fast",
                                         _usage(input_tokens=10_000),
                                         provider=provider, base_url=base_url)
            assert result.status == "included" and result.amount_usd == Decimal(0), provider

    def test_catalogued_model_on_the_subscription_route_still_prices(self, monkeypatch):
        """A Vibe-subscription route serves catalogued models too (deepseek-v4-1-flash-1017);
        those price from the vendor card named by the route's ``catalog_provider``, not as
        included."""
        import agent.usage_pricing as pricing

        monkeypatch.setattr(pricing, "_configured_catalog_provider",
                            lambda name: "mistral" if name == "mistral-vibe" else None)
        monkeypatch.setattr(
            pricing, "_models_dev_pricing_entry",
            lambda route: (PricingEntry(input_cost_per_million=Decimal("0.3"),
                                        output_cost_per_million=Decimal("1.2"),
                                        source="provider_models_api")
                           if route.provider == "mistral" else None),
        )
        route = resolve_billing_route("deepseek-v4-1-flash-1017", provider="mistral-vibe",
                                      base_url="https://api.eu.mistral.ai/v1")
        assert route.provider == "mistral"
        result = estimate_usage_cost("deepseek-v4-1-flash-1017", _usage(input_tokens=1_000_000),
                                     provider="mistral-vibe",
                                     base_url="https://api.eu.mistral.ai/v1")
        assert result.status == "estimated"
        assert result.amount_usd == Decimal("0.300")
