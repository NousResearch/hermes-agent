"""Tests for Z.AI and Ollama Cloud billing-route resolution.

Fleet reality: `zai` and `ollama-cloud` provider entries in config.yaml
point at subscription endpoints (Z.AI Coding Plan /api/coding/paas/v4,
Ollama Cloud ollama.com/v1). These bill by plan quota — there is no
per-token charge, so spend tracking must treat them as $0
(subscription_included), exactly like openai-codex.

Pay-as-you-go z.ai (api.z.ai/api/paas/v4) DOES bill per token — it gets
real _OFFICIAL_DOCS_PRICING entries.
"""

from decimal import Decimal

from agent.usage_pricing import (
    _OFFICIAL_DOCS_PRICING,
    get_pricing_entry,
    has_known_pricing,
    resolve_billing_route,
)


class TestZaiCodingPlanRoute:
    def test_coding_plan_provider_subscription_included(self):
        route = resolve_billing_route(
            "glm-5.2", provider="zai", base_url="https://api.z.ai/api/coding/paas/v4"
        )
        assert route.billing_mode == "subscription_included"
        assert route.provider == "zai"

    def test_coding_plan_base_url_subscription_included(self):
        # provider renamed in config.yaml — the /coding/ path segment is
        # the durable signal
        route = resolve_billing_route(
            "glm-5.2",
            provider="my-zai-rename",
            base_url="https://api.z.ai/api/coding/paas/v4",
        )
        assert route.billing_mode == "subscription_included"

    def test_coding_plan_responses_endpoint_subscription_included(self):
        # https://api.z.ai/api/v1 (OpenAI Responses protocol) is a Coding
        # Plan endpoint with no "/coding/" in the path
        route = resolve_billing_route(
            "glm-5.2",
            provider="zai",
            base_url="https://api.z.ai/api/v1",
        )
        assert route.billing_mode == "subscription_included"

    def test_coding_plan_anthropic_endpoint_subscription_included(self):
        # https://api.z.ai/api/anthropic (Anthropic Messages protocol)
        route = resolve_billing_route(
            "glm-5.2",
            provider="zai",
            base_url="https://api.z.ai/api/anthropic",
        )
        assert route.billing_mode == "subscription_included"

    def test_coding_plan_has_known_pricing(self):
        assert has_known_pricing(
            "glm-5.2", "zai", "https://api.z.ai/api/coding/paas/v4"
        )

    def test_coding_plan_entry_costs_zero(self):
        entry = get_pricing_entry(
            "glm-5.2", provider="zai", base_url="https://api.z.ai/api/coding/paas/v4"
        )
        assert entry is not None
        assert entry.input_cost_per_million == Decimal("0")
        assert entry.output_cost_per_million == Decimal("0")


class TestZaiPayAsYouGo:
    def test_paas_endpoint_resolves_real_pricing(self):
        # api.z.ai/api/paas/v4 (no /coding/) bills per token
        entry = get_pricing_entry(
            "glm-5.2", provider="zai", base_url="https://api.z.ai/api/paas/v4"
        )
        assert entry is not None
        assert entry.input_cost_per_million == Decimal("1.40")
        assert entry.output_cost_per_million == Decimal("4.40")
        assert entry.cache_read_cost_per_million == Decimal("0.26")

    def test_paas_has_known_pricing(self):
        assert has_known_pricing("glm-5.2", "zai", "https://api.z.ai/api/paas/v4")

    def test_paas_fleet_models_all_known(self):
        for model in ("glm-5.2", "glm-5.1", "glm-4.7", "glm-4.5-air"):
            assert has_known_pricing(
                model, "zai", "https://api.z.ai/api/paas/v4"
            ), model

    def test_official_docs_table_has_zai_keys(self):
        assert ("zai", "glm-5.2") in _OFFICIAL_DOCS_PRICING
        assert ("zai", "glm-4.7") in _OFFICIAL_DOCS_PRICING
        entry = _OFFICIAL_DOCS_PRICING[("zai", "glm-5.2")]
        assert entry.source == "official_docs_snapshot"
        assert entry.pricing_version == "zai-pricing-2026-08"


class TestOllamaCloudRoute:
    def test_ollama_cloud_provider_subscription_included(self):
        route = resolve_billing_route(
            "deepseek-v4-flash:0731",
            provider="ollama-cloud",
            base_url="https://ollama.com/v1",
        )
        assert route.billing_mode == "subscription_included"
        assert route.provider == "ollama-cloud"

    def test_ollama_cloud_host_match_subscription_included(self):
        # provider renamed — ollama.com host is the durable signal
        route = resolve_billing_route(
            "deepseek-v4-flash:0731",
            provider="renamed-provider",
            base_url="https://ollama.com/v1",
        )
        assert route.billing_mode == "subscription_included"

    def test_ollama_cloud_fleet_models_all_known(self):
        for model in (
            "deepseek-v4-flash:0731",
            "deepseek-v4-pro:0813",
            "kimi-k2.6:cloud",
        ):
            assert has_known_pricing(
                model, "ollama-cloud", "https://ollama.com/v1"
            ), model

    def test_ollama_cloud_entry_costs_zero(self):
        entry = get_pricing_entry(
            "deepseek-v4-flash:0731",
            provider="ollama-cloud",
            base_url="https://ollama.com/v1",
        )
        assert entry is not None
        assert entry.input_cost_per_million == Decimal("0")
        assert entry.output_cost_per_million == Decimal("0")

    def test_local_ollama_still_unknown(self):
        # Local ollama (11434) must NOT get subscription treatment — it's
        # self-hosted compute with no per-token price; it stays unknown
        # exactly as before this change.
        route = resolve_billing_route(
            "llama4:8b", provider="ollama", base_url="http://127.0.0.1:11434/v1"
        )
        assert route.billing_mode == "unknown"
        assert not has_known_pricing(
            "llama4:8b", "ollama", "http://127.0.0.1:11434/v1"
        )


class TestExistingRoutesUnchanged:
    def test_openrouter_still_official_models_api(self):
        route = resolve_billing_route(
            "z-ai/glm-5.2",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
        )
        assert route.billing_mode == "official_models_api"

    def test_nous_still_official_models_api(self):
        route = resolve_billing_route(
            "Hermes-4",
            provider="nous",
            base_url="https://inference-api.nousresearch.com/v1",
        )
        assert route.billing_mode == "official_models_api"

    def test_unknown_provider_still_unknown(self):
        route = resolve_billing_route("mystery-model", provider="acme", base_url="")
        assert route.billing_mode == "unknown"

    def test_fireworks_glm_unaffected(self):
        # fireworks glm entries existed before; zai branches must not shadow them
        entry = get_pricing_entry(
            "glm-5p2", provider="fireworks", base_url="https://api.fireworks.ai/v1"
        )
        assert entry is not None
        assert entry.input_cost_per_million == Decimal("1.40")


class TestFleetAuditScenario:
    """The fleet-audit symptom: sessions on zai/ollama-cloud showed up as
    'unknown pricing' in Hermes Insights. All fleet routes must resolve."""

    def test_all_fleet_routes_resolve(self):
        fleet = [
            ("deepseek-v4-flash:0731", "ollama-cloud", "https://ollama.com/v1"),
            ("deepseek-v4-pro:0813", "ollama-cloud", "https://ollama.com/v1"),
            ("glm-5.2", "zai", "https://api.z.ai/api/coding/paas/v4"),
            ("glm-5.3", "zai", "https://api.z.ai/api/coding/paas/v4"),
        ]
        for model, provider, base_url in fleet:
            assert has_known_pricing(model, provider, base_url), (model, provider)
