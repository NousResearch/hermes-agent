from types import SimpleNamespace

from agent.usage_pricing import (
    CanonicalUsage,
    NOTIONAL_ANTHROPIC_PROVIDERS,
    estimate_usage_cost,
    get_pricing_entry,
    has_known_pricing,
    normalize_usage,
    resolve_billing_route,
)


def test_normalize_usage_anthropic_keeps_cache_buckets_separate():
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=2000,
        cache_creation_input_tokens=400,
    )

    normalized = normalize_usage(usage, provider="anthropic", api_mode="anthropic_messages")

    assert normalized.input_tokens == 1000
    assert normalized.output_tokens == 500
    assert normalized.cache_read_tokens == 2000
    assert normalized.cache_write_tokens == 400
    assert normalized.prompt_tokens == 3400


def test_normalize_usage_openai_subtracts_cached_prompt_tokens():
    usage = SimpleNamespace(
        prompt_tokens=3000,
        completion_tokens=700,
        prompt_tokens_details=SimpleNamespace(cached_tokens=1800),
    )

    normalized = normalize_usage(usage, provider="openai", api_mode="chat_completions")

    assert normalized.input_tokens == 1200
    assert normalized.cache_read_tokens == 1800
    assert normalized.output_tokens == 700


def test_normalize_usage_openai_reads_top_level_anthropic_cache_fields():
    """Some OpenAI-compatible proxies (OpenRouter, Cline) expose
    Anthropic-style cache token counts at the top level of the usage object when
    routing Claude models, instead of nesting them in prompt_tokens_details.

    Regression guard for the bug fixed in cline/cline#10266 — before this fix,
    the chat-completions branch of normalize_usage() only read
    prompt_tokens_details.cache_write_tokens and completely missed the
    cache_creation_input_tokens case, so cache writes showed as 0 and reflected
    inputTokens were overstated by the cache-write amount.
    """
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        prompt_tokens_details=SimpleNamespace(cached_tokens=500),
        cache_creation_input_tokens=300,
    )

    normalized = normalize_usage(usage, provider="openrouter", api_mode="chat_completions")

    # Expected: cache read from prompt_tokens_details.cached_tokens (preferred),
    # cache write from top-level cache_creation_input_tokens (fallback).
    assert normalized.cache_read_tokens == 500
    assert normalized.cache_write_tokens == 300
    # input_tokens = prompt_total - cache_read - cache_write = 1000 - 500 - 300 = 200
    assert normalized.input_tokens == 200
    assert normalized.output_tokens == 200


def test_normalize_usage_openai_reads_top_level_cache_read_when_details_missing():
    """Some proxies expose only top-level Anthropic-style fields with no
    prompt_tokens_details object. Regression guard for cline/cline#10266.
    """
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        cache_read_input_tokens=500,
        cache_creation_input_tokens=300,
    )

    normalized = normalize_usage(usage, provider="openrouter", api_mode="chat_completions")

    assert normalized.cache_read_tokens == 500
    assert normalized.cache_write_tokens == 300
    assert normalized.input_tokens == 200


def test_normalize_usage_openai_prefers_prompt_tokens_details_over_top_level():
    """When both prompt_tokens_details and top-level Anthropic fields are
    present, we prefer the OpenAI-standard nested fields. Top-level Anthropic
    fields are only a fallback when the nested ones are absent/zero.
    """
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        prompt_tokens_details=SimpleNamespace(cached_tokens=600, cache_write_tokens=150),
        # Intentionally different values — proving we ignore these when details exist.
        cache_read_input_tokens=999,
        cache_creation_input_tokens=999,
    )

    normalized = normalize_usage(usage, provider="openrouter", api_mode="chat_completions")

    assert normalized.cache_read_tokens == 600
    assert normalized.cache_write_tokens == 150


def test_openrouter_models_api_pricing_is_converted_from_per_token_to_per_million(monkeypatch):
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata",
        lambda: {
            "anthropic/claude-opus-4.6": {
                "pricing": {
                    "prompt": "0.000005",
                    "completion": "0.000025",
                    "input_cache_read": "0.0000005",
                    "input_cache_write": "0.00000625",
                }
            }
        },
    )

    entry = get_pricing_entry(
        "anthropic/claude-opus-4.6",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
    )

    assert float(entry.input_cost_per_million) == 5.0
    assert float(entry.output_cost_per_million) == 25.0
    assert float(entry.cache_read_cost_per_million) == 0.5
    assert float(entry.cache_write_cost_per_million) == 6.25


def test_estimate_usage_cost_marks_true_subscription_routes_included(monkeypatch):
    """The included short-circuit must still fire for any route whose
    billing_mode is 'subscription_included'. (Codex is no longer such a route —
    it's now notional-OpenRouter — so we force the mode via the resolver to keep
    covering the included code path.)"""
    from agent import usage_pricing as up

    monkeypatch.setattr(
        "agent.usage_pricing.resolve_billing_route",
        lambda *a, **k: up.BillingRoute(
            provider="flat-sub", model="x", billing_mode="subscription_included"
        ),
    )
    result = estimate_usage_cost(
        "x", CanonicalUsage(input_tokens=1000, output_tokens=500), provider="flat-sub"
    )
    assert result.status == "included"
    assert result.amount_usd is not None and float(result.amount_usd) == 0.0  # type: ignore[arg-type]


def test_notional_anthropic_includes_f2_failover_aliases():
    """Regression: the -f2 failover lane (Sub#3 / claude-usw-f2 VPS) must be a
    notional-Anthropic provider, same as bare and -f1. Missing aliases caused
    every f2-routed turn to price as 'unknown'/$0, blinding /cost to ~25% of
    real Opus spend (audit 2026-06-13)."""
    for provider in ("claude-api-proxy-f2", "claude-bridge-f2"):
        assert provider in NOTIONAL_ANTHROPIC_PROVIDERS, provider
        result = estimate_usage_cost(
            "claude-opus-4-8",
            CanonicalUsage(input_tokens=1000, output_tokens=100,
                           cache_read_tokens=5000, cache_write_tokens=200),
            provider=provider,
        )
        assert result.status == "estimated", f"{provider}: {result.status}"
        assert result.amount_usd is not None and float(result.amount_usd) > 0


def test_notional_anthropic_providers_price_at_official_rates():
    """The 4 Claude subscription proxies/bridges must price claude-opus-4-8 at
    official Anthropic rates ($5/$25 per M, $0.50 cache-read, $6.25 cache-write)
    and label the result 'estimated' — NOT 'unknown' (which suppresses /cost
    cards) and NOT 'included'/$0 (which hides spend in rollups)."""
    usage = CanonicalUsage(
        input_tokens=500_000, output_tokens=559,
        cache_read_tokens=499_000, cache_write_tokens=1000,
    )
    # 500000*5/1e6 + 559*25/1e6 + 499000*0.5/1e6 + 1000*6.25/1e6 = 2.769725
    expected = 2.769725
    for provider in sorted(NOTIONAL_ANTHROPIC_PROVIDERS):
        result = estimate_usage_cost("claude-opus-4-8", usage, provider=provider)
        assert result.status == "estimated", f"{provider}: {result.status}"
        assert result.amount_usd is not None, f"{provider} priced None"
        assert float(result.amount_usd) == round(expected, 6), (
            f"{provider}: {result.amount_usd}"
        )


def test_notional_anthropic_route_resolves_to_anthropic_billing():
    """resolve_billing_route must rewrite the proxy provider to 'anthropic' with
    the docs-snapshot billing mode so all downstream pricing lookups work."""
    for provider in NOTIONAL_ANTHROPIC_PROVIDERS:
        route = resolve_billing_route("claude-opus-4-8", provider=provider)
        assert route.provider == "anthropic", provider
        assert route.billing_mode == "official_docs_snapshot", provider
        assert route.model == "claude-opus-4-8", provider


def test_notional_anthropic_accepts_dot_notation_and_prefixed_model():
    """Dot-notation (claude-opus-4.8) and anthropic/ prefix must still resolve
    through the normal Anthropic normalization once the provider is remapped."""
    usage = CanonicalUsage(input_tokens=1000, output_tokens=1000)
    for model in ("claude-opus-4.8", "anthropic/claude-opus-4-8", "claude-opus-4-7"):
        result = estimate_usage_cost(model, usage, provider="claude-bridge")
        assert result.status == "estimated", f"{model}: {result.status}"
        assert result.amount_usd is not None and float(result.amount_usd) > 0  # type: ignore[arg-type]


def test_notional_anthropic_has_known_pricing():
    """has_known_pricing must return True for the proxy providers (used by
    callers to decide whether to attempt cost display)."""
    for provider in NOTIONAL_ANTHROPIC_PROVIDERS:
        assert has_known_pricing("claude-opus-4-8", provider=provider), provider


_CODEX_STUB_METADATA = {
    "gpt-5.5": {"pricing": {"prompt": "0.000005", "completion": "0.00003"}},
    "gpt-5.4": {"pricing": {"prompt": "0.0000025", "completion": "0.000015"}},
    "gpt-5.3-codex": {"pricing": {"prompt": "0.00000175", "completion": "0.000014"}},
}


def test_openai_codex_route_resolves_to_notional_openrouter():
    """openai-codex must resolve to an OpenRouter-priced route (notional cost
    visibility), NOT subscription_included/$0 — so /cost cards can fire."""
    from agent.usage_pricing import NOTIONAL_OPENROUTER_PROVIDERS

    assert "openai-codex" in NOTIONAL_OPENROUTER_PROVIDERS
    route = resolve_billing_route("gpt-5.5", provider="openai-codex")
    assert route.provider == "openrouter"
    assert route.billing_mode == "official_models_api"
    assert route.model == "gpt-5.5"


def test_openai_codex_priced_from_openrouter_catalog(monkeypatch):
    """A codex turn is priced at the underlying OpenAI model's live OpenRouter
    rate and labelled 'estimated'."""
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata", lambda: _CODEX_STUB_METADATA
    )
    usage = CanonicalUsage(input_tokens=120_000, output_tokens=8_000)
    # 120000*5/1e6 + 8000*30/1e6 = 0.6 + 0.24 = 0.84
    result = estimate_usage_cost("gpt-5.5", usage, provider="openai-codex")
    assert result.status == "estimated"
    assert result.amount_usd is not None and float(result.amount_usd) == 0.84  # type: ignore[arg-type]
    assert result.source == "provider_models_api"


def test_openai_codex_minus_codex_suffix_falls_back_to_base_model(monkeypatch):
    """gpt-5.5-codex is absent from the OpenRouter catalog while gpt-5.5 is
    present; the '-codex' fallback must price it as the base model."""
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata", lambda: _CODEX_STUB_METADATA
    )
    usage = CanonicalUsage(input_tokens=120_000, output_tokens=8_000)
    result = estimate_usage_cost("gpt-5.5-codex", usage, provider="openai-codex")
    assert result.status == "estimated"
    assert result.amount_usd is not None and float(result.amount_usd) == 0.84  # type: ignore[arg-type]


def test_openai_codex_exact_codex_id_preferred_over_fallback(monkeypatch):
    """When the exact '-codex' id IS in the catalog, use it directly (no strip)."""
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata", lambda: _CODEX_STUB_METADATA
    )
    entry = get_pricing_entry("gpt-5.3-codex", provider="openai-codex")
    assert entry is not None
    assert float(entry.input_cost_per_million) == 1.75  # type: ignore[arg-type]
    assert float(entry.output_cost_per_million) == 14.0  # type: ignore[arg-type]


def test_openai_codex_unknown_model_stays_unknown(monkeypatch):
    """A codex model absent from the catalog with no base fallback must degrade
    to 'unknown' (NOT fabricate a price)."""
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata", lambda: _CODEX_STUB_METADATA
    )
    usage = CanonicalUsage(input_tokens=1000, output_tokens=500)
    result = estimate_usage_cost("gpt-9-imaginary", usage, provider="openai-codex")
    assert result.status == "unknown"
    assert result.amount_usd is None


def test_estimate_usage_cost_refuses_cache_pricing_without_official_cache_rate(monkeypatch):
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata",
        lambda: {
            "google/gemini-2.5-pro": {
                "pricing": {
                    "prompt": "0.00000125",
                    "completion": "0.00001",
                }
            }
        },
    )

    result = estimate_usage_cost(
        "google/gemini-2.5-pro",
        CanonicalUsage(input_tokens=1000, output_tokens=500, cache_read_tokens=100),
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
    )

    assert result.status == "unknown"


def test_custom_endpoint_models_api_pricing_is_supported(monkeypatch):
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_endpoint_model_metadata",
        lambda base_url, api_key=None: {
            "zai-org/GLM-5-TEE": {
                "pricing": {
                    "prompt": "0.0000005",
                    "completion": "0.000002",
                }
            }
        },
    )

    entry = get_pricing_entry(
        "zai-org/GLM-5-TEE",
        provider="custom",
        base_url="https://llm.chutes.ai/v1",
        api_key="test-key",
    )

    assert float(entry.input_cost_per_million) == 0.5
    assert float(entry.output_cost_per_million) == 2.0


def test_deepseek_v4_pro_pricing_entry_exists():
    """Regression test: deepseek-v4-pro must have a pricing entry.

    Before this fix, deepseek-v4-pro sessions showed as unknown cost
    in hermes insights because the _OFFICIAL_DOCS_PRICING table had no
    entry for that model.  See #24218.
    """
    entry = get_pricing_entry(
        "deepseek-v4-pro",
        provider="deepseek",
    )

    assert entry is not None
    assert entry.input_cost_per_million is not None
    assert entry.output_cost_per_million is not None
    assert float(entry.input_cost_per_million) == 1.74
    assert float(entry.output_cost_per_million) == 3.48
    assert float(entry.cache_read_cost_per_million) == 0.0145


def test_deepseek_v4_pro_estimate_usage_cost():
    """Ensure deepseek-v4-pro sessions get a dollar estimate, not unknown."""
    result = estimate_usage_cost(
        "deepseek-v4-pro",
        CanonicalUsage(input_tokens=1000000, output_tokens=500000),
        provider="deepseek",
    )

    assert result.status == "estimated"
    assert result.amount_usd is not None
    # 1M input × $1.74/M + 500K output × $3.48/M = $1.74 + $1.74 = $3.48
    assert float(result.amount_usd) == 3.48
