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


def test_normalize_usage_reads_deepseek_native_cache_hit_tokens():
    """DeepSeek's native API (api.deepseek.com) reports context-cache hits as
    top-level prompt_cache_hit_tokens / prompt_cache_miss_tokens (with
    prompt_tokens = hit + miss), not OpenAI's nested
    prompt_tokens_details.cached_tokens. Before this fix, direct DeepSeek
    sessions always normalized to cache_read_tokens=0 — cache hits were
    invisible in accounting and billed at the full input rate (#61871)."""
    usage = SimpleNamespace(
        prompt_tokens=2000,
        completion_tokens=400,
        prompt_cache_hit_tokens=1500,
        prompt_cache_miss_tokens=500,
    )

    normalized = normalize_usage(usage, provider="deepseek", api_mode="chat_completions")

    assert normalized.cache_read_tokens == 1500
    # prompt_tokens includes cached; input = 2000 - 1500 = the miss bucket
    assert normalized.input_tokens == 500
    assert normalized.output_tokens == 400


def test_normalize_usage_nested_details_win_over_deepseek_top_level():
    """When a proxy forwards both shapes, the OpenAI nested value wins and
    the DeepSeek top-level field is not double-read."""
    usage = SimpleNamespace(
        prompt_tokens=2000,
        completion_tokens=100,
        prompt_tokens_details=SimpleNamespace(cached_tokens=900),
        prompt_cache_hit_tokens=1500,
    )

    normalized = normalize_usage(usage, provider="deepseek", api_mode="chat_completions")

    assert normalized.cache_read_tokens == 900
    assert normalized.input_tokens == 1100


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


def test_nous_portal_pricing_preserves_vendor_prefixed_model_ids(monkeypatch):
    seen = {}

    def _fake_fetch_endpoint_model_metadata(base_url, api_key=None):
        seen["base_url"] = base_url
        return {
            "openai/gpt-5.5-pro": {
                "pricing": {
                    "prompt": "0.000025",
                    "completion": "0.000125",
                }
            }
        }

    monkeypatch.setattr(
        "agent.usage_pricing.fetch_endpoint_model_metadata",
        _fake_fetch_endpoint_model_metadata,
    )

    entry = get_pricing_entry("openai/gpt-5.5-pro", provider="nous")

    assert seen["base_url"] == "https://inference-api.nousresearch.com/v1"
    assert float(entry.input_cost_per_million) == 25.0
    assert float(entry.output_cost_per_million) == 125.0


def test_deepseek_v4_pro_pricing_entry_exists():
    """Regression test: deepseek-v4-pro must have a pricing entry.

    Before this fix, deepseek-v4-pro sessions showed as unknown cost
    in hermes insights because the _OFFICIAL_DOCS_PRICING table had no
    entry for that model.  See #24218.  Rates track the 2026-07 price cut
    ($1.74/$3.48 → $0.435/$0.87).
    """
    entry = get_pricing_entry(
        "deepseek-v4-pro",
        provider="deepseek",
    )

    assert entry is not None
    assert entry.input_cost_per_million is not None
    assert entry.output_cost_per_million is not None
    assert float(entry.input_cost_per_million) == 0.435
    assert float(entry.output_cost_per_million) == 0.87
    assert float(entry.cache_read_cost_per_million) == 0.003625


def test_deepseek_v4_pro_estimate_usage_cost():
    """Ensure deepseek-v4-pro sessions get a dollar estimate, not unknown."""
    result = estimate_usage_cost(
        "deepseek-v4-pro",
        CanonicalUsage(input_tokens=1000000, output_tokens=500000),
        provider="deepseek",
    )

    assert result.status == "estimated"
    assert result.amount_usd is not None
    # 1M input × $0.435/M + 500K output × $0.87/M = $0.435 + $0.435 = $0.87
    assert float(result.amount_usd) == 0.87


def test_deepseek_deprecated_aliases_price_as_v4_flash():
    """Invariant: deepseek-chat / deepseek-reasoner are deprecated aliases for
    deepseek-v4-flash's non-thinking / thinking modes (deprecation 2026-07-24)
    — they must bill at identical rates to the flash entry, or sessions on the
    legacy names over/under-report cost."""
    flash = get_pricing_entry("deepseek-v4-flash", provider="deepseek")
    assert flash is not None
    for alias in ("deepseek-chat", "deepseek-reasoner"):
        entry = get_pricing_entry(alias, provider="deepseek")
        assert entry is not None, alias
        assert entry.input_cost_per_million == flash.input_cost_per_million, alias
        assert entry.output_cost_per_million == flash.output_cost_per_million, alias
        assert (
            entry.cache_read_cost_per_million == flash.cache_read_cost_per_million
        ), alias


def test_deepseek_rows_all_carry_cache_read_pricing():
    """Invariant: DeepSeek publishes a cache-hit rate for every current model;
    every deepseek snapshot row must carry cache_read < input so cached
    sessions estimate correctly instead of billing reads at full price."""
    from agent.usage_pricing import _OFFICIAL_DOCS_PRICING

    ds_rows = [k for k in _OFFICIAL_DOCS_PRICING if k[0] == "deepseek"]
    assert ds_rows, "expected at least one deepseek pricing row"
    for key in ds_rows:
        entry = _OFFICIAL_DOCS_PRICING[key]
        assert entry.cache_read_cost_per_million is not None, key
        assert entry.cache_read_cost_per_million < entry.input_cost_per_million, key


def test_bedrock_claude_rows_all_carry_cache_pricing():
    """Invariant: every Bedrock Claude pricing row must carry cache-read AND
    cache-write rates, otherwise a cached session prices as ``unknown``.

    Bedrock Claude routes through the AnthropicBedrock SDK and injects
    cache_control, so cached tokens are always reported — the pricing layer
    must be able to value them.  See #50295.
    """
    from agent.usage_pricing import _OFFICIAL_DOCS_PRICING

    claude_rows = [
        (prov, model)
        for (prov, model) in _OFFICIAL_DOCS_PRICING
        if prov == "bedrock" and "claude" in model
    ]
    assert claude_rows, "expected at least one bedrock Claude pricing row"
    for key in claude_rows:
        entry = _OFFICIAL_DOCS_PRICING[key]
        assert entry.input_cost_per_million is not None, key
        assert entry.cache_read_cost_per_million is not None, key
        assert entry.cache_write_cost_per_million is not None, key
        # Cache reads are cheaper than fresh input; cache writes cost more.
        assert entry.cache_read_cost_per_million < entry.input_cost_per_million, key
        assert entry.cache_write_cost_per_million > entry.input_cost_per_million, key


def test_bedrock_cross_region_profile_prefix_resolves_to_pricing():
    """Cross-region inference profiles (us./global./eu. prefixes) must resolve
    to the same pricing entry as the bare foundation-model id.  Without prefix
    normalization, ``us.anthropic.claude-*`` sessions price as unknown.
    """
    bedrock_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
    bare = get_pricing_entry(
        "anthropic.claude-sonnet-4-5", provider="bedrock", base_url=bedrock_url
    )
    assert bare is not None
    for prefix in ("us.", "global.", "eu."):
        scoped = get_pricing_entry(
            f"{prefix}anthropic.claude-sonnet-4-5",
            provider="bedrock",
            base_url=bedrock_url,
        )
        assert scoped is not None, prefix
        assert scoped.input_cost_per_million == bare.input_cost_per_million
        assert scoped.cache_read_cost_per_million == bare.cache_read_cost_per_million


def test_bedrock_claude_cached_session_estimates_cost_not_unknown():
    """A Bedrock Claude session with cache hits must produce a dollar estimate,
    not ``unknown`` — the user-visible symptom in #50295.
    """
    bedrock_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
    usage = SimpleNamespace(
        input_tokens=55,
        output_tokens=7113,
        cache_read_input_tokens=1369379,
        cache_creation_input_tokens=42135,
    )
    canonical = normalize_usage(usage, provider="bedrock", api_mode="anthropic_messages")
    assert canonical.cache_read_tokens == 1369379
    assert canonical.cache_write_tokens == 42135

    result = estimate_usage_cost(
        "us.anthropic.claude-opus-4-6",
        canonical,
        provider="bedrock",
        base_url=bedrock_url,
    )
    assert result.status == "estimated"
    assert result.amount_usd is not None

def test_fireworks_kimi_k2p6_resolves_with_full_model_path():
    """Fireworks model ids look like accounts/fireworks/models/<name>;
    the routing layer must strip the prefix so the dict lookup succeeds."""
    entry = get_pricing_entry(
        "accounts/fireworks/models/kimi-k2p6",
        provider="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
    )

    assert entry is not None
    assert float(entry.input_cost_per_million) == 0.95
    assert float(entry.output_cost_per_million) == 4.00
    assert float(entry.cache_read_cost_per_million) == 0.16
    assert entry.source == "official_docs_snapshot"


def test_fireworks_base_url_host_match_alone_routes_to_pricing():
    """Provider not explicitly passed; routing infers fireworks from the host."""
    entry = get_pricing_entry(
        "accounts/fireworks/models/deepseek-v4-pro",
        base_url="https://api.fireworks.ai/inference/v1",
    )

    assert entry is not None
    assert float(entry.input_cost_per_million) == 1.74
    assert float(entry.output_cost_per_million) == 3.48


def test_fireworks_qwen3p7_plus_estimate_usage_cost():
    """End-to-end: Fireworks Qwen3.7-Plus sessions report a dollar estimate."""
    result = estimate_usage_cost(
        "accounts/fireworks/models/qwen3p7-plus",
        CanonicalUsage(input_tokens=1_000_000, output_tokens=500_000),
        provider="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
    )

    assert result.status == "estimated"
    assert result.amount_usd is not None
    # 1M input × $0.40/M + 500K output × $1.60/M = $0.40 + $0.80 = $1.20
    assert float(result.amount_usd) == 1.20


def test_fireworks_router_fast_tier_prices_distinctly():
    """Fast serving tiers live under accounts/fireworks/routers/<name>-fast and
    bill at higher rates than the standard model — the routing layer's
    rsplit("/", 1) must land on the distinct fast-tier entry."""
    standard = get_pricing_entry(
        "accounts/fireworks/models/kimi-k2p6",
        provider="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
    )
    fast = get_pricing_entry(
        "accounts/fireworks/routers/kimi-k2p6-fast",
        provider="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
    )
    assert standard is not None and fast is not None
    assert fast.input_cost_per_million > standard.input_cost_per_million
    assert fast.output_cost_per_million > standard.output_cost_per_million


def test_fireworks_plugin_fallback_models_all_have_pricing():
    """Invariant: every model in the Fireworks provider plugin's
    fallback_models (the picker's curated safety net) must resolve to a
    pricing entry — otherwise the default picker choices bill as unknown."""
    from providers import get_provider_profile

    profile = get_provider_profile("fireworks")
    assert profile is not None
    for mid in profile.fallback_models:
        entry = get_pricing_entry(
            mid,
            provider="fireworks",
            base_url="https://api.fireworks.ai/inference/v1",
        )
        assert entry is not None, f"no pricing entry for fallback model {mid}"
        assert entry.input_cost_per_million is not None, mid


def test_fireworks_rows_all_carry_cache_read_pricing():
    """Invariant: Fireworks publishes cached-input rates for every serverless
    model, and Hermes prompt caching is active on Fireworks sessions — every
    snapshot row must carry a cache_read rate cheaper than fresh input."""
    from agent.usage_pricing import _OFFICIAL_DOCS_PRICING

    fw_rows = [k for k in _OFFICIAL_DOCS_PRICING if k[0] == "fireworks"]
    assert fw_rows, "expected at least one fireworks pricing row"
    for key in fw_rows:
        entry = _OFFICIAL_DOCS_PRICING[key]
        assert entry.cache_read_cost_per_million is not None, key
        assert entry.cache_read_cost_per_million < entry.input_cost_per_million, key


def test_deepseek_v4_flash_pricing_entry_exists():
    """Regression test: deepseek-v4-flash must have a pricing entry.

    Before this fix, deepseek-v4-flash sessions showed $0.00 / cost_source
    "none" because the _OFFICIAL_DOCS_PRICING table had an entry for
    deepseek-v4-pro but not the (newer) flash model.  DeepSeek's /models
    endpoint returns no pricing, so the official-docs snapshot is the only
    source for direct-provider routes.
    """
    entry = get_pricing_entry(
        "deepseek-v4-flash",
        provider="deepseek",
    )

    assert entry is not None
    assert float(entry.input_cost_per_million) == 0.14
    assert float(entry.output_cost_per_million) == 0.28
    assert float(entry.cache_read_cost_per_million) == 0.0028


def test_deepseek_v4_flash_estimate_usage_cost():
    """Ensure deepseek-v4-flash sessions get a dollar estimate, not $0/none."""
    result = estimate_usage_cost(
        "deepseek-v4-flash",
        CanonicalUsage(input_tokens=1000000, output_tokens=500000),
        provider="deepseek",
    )

    assert result.status == "estimated"
    assert result.amount_usd is not None
    # 1M input × $0.14/M + 500K output × $0.28/M = $0.14 + $0.14 = $0.28
    assert float(result.amount_usd) == 0.28
