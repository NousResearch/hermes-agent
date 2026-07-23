"""Behavior contracts for the GPT-5.6 (Sol/Terra/Luna) registration.

Invariant tests only — no list snapshots (per the no-change-detector-tests
policy). These pin the two behaviors that would silently regress:

1. Version/tier sorting: the flagship Sol must outrank Terra/Luna and the
   whole 5.6 series must outrank 5.5, so `/model gpt` resolves to the
   flagship rather than an alphabetical-first cheap tier.
2. Pricing reachability: the ("openai", <model>) official-docs pricing keys
   must be reachable from BOTH the bare "openai" provider and the
   "openai-api" picker slug (resolve_billing_route normalizes the latter).
"""

from decimal import Decimal

from agent.usage_pricing import (
    _OFFICIAL_DOCS_PRICING,
    _lookup_official_docs_pricing,
    resolve_billing_route,
)
from hermes_cli.model_switch import _model_sort_key


class TestGpt56SortInvariants:
    def test_sol_outranks_terra_and_luna(self):
        models = ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
        models.sort(key=lambda m: _model_sort_key(m, "gpt"))
        assert models[0] == "gpt-5.6-sol"

    def test_56_series_outranks_55(self):
        models = ["gpt-5.5", "gpt-5.5-pro", "gpt-5.6-sol"]
        models.sort(key=lambda m: _model_sort_key(m, "gpt"))
        assert models[0] == "gpt-5.6-sol"

    def test_aggregator_prefix_form(self):
        models = ["openai/gpt-5.5-pro", "openai/gpt-5.6-sol"]
        models.sort(key=lambda m: _model_sort_key(m, "openai/gpt"))
        assert models[0] == "openai/gpt-5.6-sol"


    def test_base_sol_outranks_sol_pro_for_alias_default(self):
        # "-pro" high-effort variants parse as suffix "sol-pro" (rank 1), so
        # `/model gpt` defaults to base Sol rather than the high-effort mode.
        models = ["gpt-5.6-sol-pro", "gpt-5.6-sol"]
        models.sort(key=lambda m: _model_sort_key(m, "gpt"))
        assert models[0] == "gpt-5.6-sol"


class TestGpt56PricingRoute:
    def test_official_pricing_reachable_from_openai(self):
        route = resolve_billing_route("gpt-5.6-sol", provider="openai")
        entry = _lookup_official_docs_pricing(route)
        assert entry is not None
        assert entry.input_cost_per_million == Decimal("5.00")

    def test_official_pricing_reachable_from_openai_api_slug(self):
        # "openai-api" is the picker slug for direct api.openai.com and must
        # normalize to the "openai" pricing key space.
        route = resolve_billing_route("gpt-5.6-sol", provider="openai-api")
        assert route.provider == "openai"
        entry = _lookup_official_docs_pricing(route)
        assert entry is not None
        assert entry.input_cost_per_million == Decimal("5.00")

    def test_cache_write_is_1_25x_input_for_56_series(self):
        for slug in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            entry = _OFFICIAL_DOCS_PRICING[("openai", slug)]
            assert entry.input_cost_per_million is not None, slug
            assert entry.cache_write_cost_per_million == (
                entry.input_cost_per_million * Decimal("1.25")
            ), slug
            assert entry.cache_read_cost_per_million == (
                entry.input_cost_per_million * Decimal("0.10")
            ), slug

    def test_pro_variants_alias_to_base_tier_pricing(self):
        # -pro high-effort modes bill at the same per-token rates as their
        # base tiers; the snapshot aliases them rather than duplicating rows.
        for base in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            assert (
                _OFFICIAL_DOCS_PRICING[("openai", f"{base}-pro")]
                is _OFFICIAL_DOCS_PRICING[("openai", base)]
            ), base


class TestGpt56CodexCompaction:
    """Codex OAuth caps the whole gpt-5.6 family at 272K, same as 5.4/5.5, so
    the compaction auto-raise (0.85) must fire for every 5.6 variant on the
    openai-codex route and NOT on the direct-API/OpenRouter routes."""

    def test_autoraise_applies_to_all_56_on_codex(self):
        from agent.auxiliary_client import _compression_threshold_for_model

        for slug in (
            "gpt-5.6-sol",
            "gpt-5.6-sol-pro",
            "gpt-5.6-terra",
            "gpt-5.6-terra-pro",
            "gpt-5.6-luna",
            "gpt-5.6-luna-pro",
        ):
            assert (
                _compression_threshold_for_model(slug, provider="openai-codex")
                == 0.85
            ), slug

    def test_no_autoraise_on_direct_api_route(self):
        from agent.auxiliary_client import _compression_threshold_for_model

        # Direct OpenAI API / OpenRouter expose the full 1.05M window, so the
        # 272K-cap override must NOT apply there.
        assert (
            _compression_threshold_for_model("gpt-5.6-sol", provider="openai")
            is None
        )
        assert (
            _compression_threshold_for_model(
                "openai/gpt-5.6-sol", provider="openrouter"
            )
            is None
        )

    def test_autoraise_respects_opt_out(self):
        from agent.auxiliary_client import _compression_threshold_for_model

        assert (
            _compression_threshold_for_model(
                "gpt-5.6-sol",
                provider="openai-codex",
                allow_codex_gpt55_autoraise=False,
            )
            is None
        )
