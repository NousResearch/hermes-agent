"""Behavior contracts for the Xiaomi MiMo v2.6 registration.

Invariant tests only — no list snapshots (per the no-change-detector-tests policy).
These pin the two behaviors that would silently regress:

1. Pricing reachability: the ``("xiaomi", <model>)`` official-docs keys must be
   reachable through ``resolve_billing_route`` from the provider the runtime
   actually bills as, so cost reporting estimates a real number instead of
   leaving ``cost_status="unknown"`` on every MiMo turn.
2. Cache economics: MiMo's implicit prompt cache is what makes long agent loops
   affordable, so the snapshot must carry a cache-read rate far below the
   cache-miss input rate. A row with only input/output bills cached turns at the
   full input rate — a ~50x overstatement on a tool-heavy session.
"""

from decimal import Decimal

from agent.usage_pricing import (
    _OFFICIAL_DOCS_PRICING,
    _lookup_official_docs_pricing,
    resolve_billing_route,
)

_MIMO_26 = ("mimo-v2.6-flash", "mimo-v2.6-pro")


class TestXiaomiPricingReachability:
    def test_flash_is_priced_from_the_runtime_provider(self):
        route = resolve_billing_route("mimo-v2.6-flash", provider="xiaomi")
        entry = _lookup_official_docs_pricing(route)
        assert entry is not None
        assert entry.input_cost_per_million == Decimal("0.14")
        assert entry.output_cost_per_million == Decimal("0.28")

    def test_every_26_model_has_a_row(self):
        for slug in _MIMO_26:
            assert ("xiaomi", slug) in _OFFICIAL_DOCS_PRICING, slug


class TestXiaomiCacheEconomics:
    def test_cache_read_is_a_fraction_of_a_cache_miss(self):
        for slug in _MIMO_26:
            entry = _OFFICIAL_DOCS_PRICING[("xiaomi", slug)]
            assert entry.cache_read_cost_per_million is not None, slug
            assert (
                entry.cache_read_cost_per_million
                < entry.input_cost_per_million / 10
            ), slug

    def test_output_is_the_priciest_direction(self):
        # Reasoning tokens bill as output on MiMo (thinking is on by default), so a
        # snapshot pricing output at or below input would hide the dominant cost.
        for slug in _MIMO_26:
            entry = _OFFICIAL_DOCS_PRICING[("xiaomi", slug)]
            assert entry.output_cost_per_million > entry.input_cost_per_million, slug
