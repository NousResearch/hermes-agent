from types import SimpleNamespace

from agent.usage_pricing import (
    CanonicalUsage,
    estimate_usage_cost,
    get_pricing_entry,
    normalize_usage,
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
    """Some OpenAI-compatible proxies (OpenRouter, Vercel AI Gateway, Cline) expose
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


def test_estimate_usage_cost_marks_subscription_routes_included():
    result = estimate_usage_cost(
        "gpt-5.3-codex",
        CanonicalUsage(input_tokens=1000, output_tokens=500),
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert result.status == "included"
    assert float(result.amount_usd) == 0.0


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


# ─────────────────────────────────────────────────────────────────────────
# pricing_overrides — user-supplied prices win over auto-discovery sources.
# ─────────────────────────────────────────────────────────────────────────


def test_pricing_overrides_resolve_for_custom_provider(monkeypatch):
    """A user override for a custom_providers-routed model (e.g. Fireworks)
    should resolve via get_pricing_entry, even though Fireworks has no
    pricing API and isn't in the bundled docs snapshot."""
    monkeypatch.setattr(
        "agent.usage_pricing._load_user_pricing_overrides",
        lambda: [
            {
                "provider": "custom:fireworks",
                "model": "kimi-k2p6",
                "input_per_million": 0.95,
                "output_per_million": 4.00,
                "cache_read_per_million": 0.16,
                "source_url": "https://docs.fireworks.ai/serverless/pricing",
            }
        ],
    )

    entry = get_pricing_entry(
        "accounts/fireworks/models/kimi-k2p6",
        provider="custom:fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
    )

    assert entry is not None
    assert entry.source == "user_override"
    assert float(entry.input_cost_per_million) == 0.95
    assert float(entry.output_cost_per_million) == 4.00
    assert float(entry.cache_read_cost_per_million) == 0.16


def test_pricing_overrides_match_full_model_id_or_basename(monkeypatch):
    """The matcher should accept either the full slash-prefixed id or the
    basename, so users don't have to know which form Hermes routes with
    for their particular custom_providers configuration."""
    monkeypatch.setattr(
        "agent.usage_pricing._load_user_pricing_overrides",
        lambda: [
            {
                # Full id form
                "provider": "custom:fireworks",
                "model": "accounts/fireworks/models/kimi-k2p6",
                "input_per_million": 1.0,
                "output_per_million": 2.0,
            }
        ],
    )

    # Basename form on the route should still hit the full-id override.
    entry = get_pricing_entry(
        "accounts/fireworks/models/kimi-k2p6",
        provider="custom:fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
    )
    assert entry is not None
    assert float(entry.input_cost_per_million) == 1.0


def test_pricing_overrides_take_precedence_over_docs_snapshot(monkeypatch):
    """When both a user override and a bundled snapshot entry exist for the
    same (provider, model), the override wins. This is the supported way
    to correct stale rates without waiting for a Hermes release."""
    monkeypatch.setattr(
        "agent.usage_pricing._load_user_pricing_overrides",
        lambda: [
            {
                "provider": "anthropic",
                "model": "claude-opus-4-7",
                "input_per_million": 99.0,  # absurd value to prove override wins
                "output_per_million": 99.0,
            }
        ],
    )

    entry = get_pricing_entry("claude-opus-4-7", provider="anthropic")

    assert entry is not None
    assert entry.source == "user_override"
    assert float(entry.input_cost_per_million) == 99.0


def test_pricing_overrides_skip_entries_missing_required_fields(monkeypatch):
    """Entries without input_per_million OR output_per_million are skipped
    silently — we don't want a single bad entry to mask later valid ones
    or crash cost computation."""
    monkeypatch.setattr(
        "agent.usage_pricing._load_user_pricing_overrides",
        lambda: [
            {"provider": "custom:fireworks", "model": "kimi-k2p6"},  # no rates
            {
                "provider": "custom:fireworks",
                "model": "kimi-k2p6",
                "input_per_million": 0.95,
                "output_per_million": 4.00,
            },
        ],
    )

    entry = get_pricing_entry(
        "accounts/fireworks/models/kimi-k2p6",
        provider="custom:fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
    )

    # The first (incomplete) entry should be skipped; the second wins.
    assert entry is not None
    assert float(entry.input_cost_per_million) == 0.95
