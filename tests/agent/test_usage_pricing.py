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


def test_normalize_usage_bedrock_converse_cache_point_round_trips():
    """End-to-end contract for the Converse cachePoint feature: the usage
    shape bedrock_adapter.normalize_converse_response() produces (prompt_tokens
    folded from inputTokens + cacheRead + cacheWrite, cache fields under their
    Anthropic names) must normalize back to the original cache_read/write
    split and the original Converse inputTokens value."""
    usage = SimpleNamespace(
        prompt_tokens=50 + 900 + 300,
        completion_tokens=20,
        cache_read_input_tokens=900,
        cache_creation_input_tokens=300,
    )

    normalized = normalize_usage(usage, provider="bedrock", api_mode="bedrock_converse")

    assert normalized.cache_read_tokens == 900
    assert normalized.cache_write_tokens == 300
    assert normalized.input_tokens == 50
    assert normalized.output_tokens == 20


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


def test_bedrock_current_gen_claude_rows_resolve():
    """Current-gen Claude models (Opus 4.8/4.7, Sonnet 5) must have Bedrock
    pricing rows so cached sessions report a dollar cost, not ``unknown``.
    Assert each resolves via the bare id and a cross-region inference profile
    (us./global. prefix), that every id for a given model resolves to the same
    entry, and that the row carries the cache fields a Bedrock Claude session
    needs.

    (Version-suffixed IDs like ``...-v1:0`` are covered separately by the
    normalizer test in the suffix-strip change; this test intentionally sticks
    to id shapes that resolve on ``main`` so it is independent of that PR.)
    """
    url = "https://bedrock-runtime.us-east-1.amazonaws.com"
    for bare in (
        "anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-7",
        "anthropic.claude-sonnet-5",
    ):
        ref = get_pricing_entry(bare, provider="bedrock", base_url=url)
        assert ref is not None, bare
        assert ref.input_cost_per_million is not None, bare
        assert ref.output_cost_per_million is not None, bare
        # Output costs more than input across the Claude line; sanity-check the
        # row isn't malformed (input < output).
        assert ref.output_cost_per_million > ref.input_cost_per_million, bare
        # Cache fields present so cached sessions price correctly (the #50295
        # symptom was unknown cost on cached Bedrock Claude sessions).
        assert ref.cache_read_cost_per_million is not None, bare
        assert ref.cache_write_cost_per_million is not None, bare
        # Cross-region inference profiles resolve to the same entry.
        for mid in (f"us.{bare}", f"global.{bare}"):
            entry = get_pricing_entry(mid, provider="bedrock", base_url=url)
            assert entry is not None, mid
            assert entry.input_cost_per_million == ref.input_cost_per_million, mid
            assert entry.output_cost_per_million == ref.output_cost_per_million, mid


def test_bedrock_cross_region_profile_prefix_resolves_to_pricing():
    """Cross-region inference profiles must resolve to the same pricing entry
    as the bare foundation-model id.  Without prefix normalization a scoped
    ``<region>.anthropic.claude-*`` session prices as unknown.

    Asia-Pacific (``apac.``) and Australia (``au.``) are included because AWS
    uses the full ``apac.`` prefix, not ``ap.`` — a bare ``ap.`` never matches
    an ``apac.*`` id, so those geographies previously priced as unknown.
    """
    bedrock_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
    bare = get_pricing_entry(
        "anthropic.claude-sonnet-4-5", provider="bedrock", base_url=bedrock_url
    )
    assert bare is not None
    for prefix in ("us.", "global.", "eu.", "apac.", "au."):
        scoped = get_pricing_entry(
            f"{prefix}anthropic.claude-sonnet-4-5",
            provider="bedrock",
            base_url=bedrock_url,
        )
        assert scoped is not None, prefix
        assert scoped.input_cost_per_million == bare.input_cost_per_million
        assert scoped.cache_read_cost_per_million == bare.cache_read_cost_per_million


def test_bedrock_versioned_inference_profile_resolves_to_bare_pricing():
    """Bedrock profile IDs may include the provider's dated version suffix.

    The pricing table intentionally uses shorter model-family IDs, so the
    lookup needs a longest-prefix fallback after stripping the region scope.
    """
    bare = get_pricing_entry("anthropic.claude-sonnet-4-6", provider="bedrock")
    assert bare is not None

    for model in (
        "us.anthropic.claude-sonnet-4-6-20250514-v1:0",
        "global.anthropic.claude-sonnet-4-6-20250514-v1:0",
    ):
        scoped = get_pricing_entry(model, provider="bedrock")
        assert scoped is not None, model
        assert scoped.input_cost_per_million == bare.input_cost_per_million
        assert scoped.output_cost_per_million == bare.output_cost_per_million
        assert scoped.cache_read_cost_per_million == bare.cache_read_cost_per_million
        assert scoped.cache_write_cost_per_million == bare.cache_write_cost_per_million


def test_bedrock_pricing_supports_less_common_inference_profile_prefixes():
    """AWS also exposes profile scopes beyond us./global./eu.; those should
    not silently fall through to unknown pricing.
    """
    bare = get_pricing_entry("anthropic.claude-haiku-4-5", provider="bedrock")
    entry = get_pricing_entry(
        "apac.anthropic.claude-haiku-4-5-20251001-v1:0",
        provider="bedrock",
    )

    assert bare is not None
    assert entry is not None
    for field in (
        "input_cost_per_million",
        "output_cost_per_million",
        "cache_read_cost_per_million",
        "cache_write_cost_per_million",
    ):
        assert getattr(entry, field) == getattr(bare, field)


def test_bedrock_unknown_model_continuation_does_not_use_base_pricing():
    """Unrecognized Bedrock SKUs must remain unknown rather than inheriting a
    similarly named model family's price.
    """
    assert (
        get_pricing_entry(
            "anthropic.claude-sonnet-4-6-experimental",
            provider="bedrock",
        )
        is None
    )


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


def test_cache_write_falls_back_to_input_rate_when_no_premium_published(monkeypatch):
    """A provider that publishes no separate cache-write rate must still price
    the turn. Cache-write tokens are prompt tokens billed as ordinary input
    when there is no write premium, so a missing rate means "no premium", not
    "unpriceable". Contract: the turn costs exactly the same as one where those
    tokens had arrived as plain input tokens."""
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata",
        lambda: {
            "no-write-premium-model": {
                "pricing": {
                    "prompt": "0.000005",
                    "completion": "0.00003",
                    "input_cache_read": "0.0000005",
                    # no cache_write / input_cache_write key — the real-world gap
                }
            }
        },
    )
    kwargs = dict(provider="openrouter", base_url="https://openrouter.ai/api/v1")

    with_cache_write = estimate_usage_cost(
        "no-write-premium-model",
        CanonicalUsage(
            input_tokens=6,
            output_tokens=3104,
            cache_read_tokens=206366,
            cache_write_tokens=114435,
        ),
        **kwargs,
    )
    # Same prompt-token budget, but the cache-write tokens arrive as plain input.
    as_plain_input = estimate_usage_cost(
        "no-write-premium-model",
        CanonicalUsage(
            input_tokens=6 + 114435,
            output_tokens=3104,
            cache_read_tokens=206366,
            cache_write_tokens=0,
        ),
        **kwargs,
    )

    assert as_plain_input.status == "estimated"  # control: this always worked
    assert with_cache_write.status == "estimated"
    assert with_cache_write.amount_usd == as_plain_input.amount_usd
    assert any("input rate" in note for note in with_cache_write.notes)


def test_published_cache_write_premium_is_never_replaced_by_the_input_rate(monkeypatch):
    """The fallback may only fill a gap. When a cache-write rate IS published
    it must be used verbatim — including a $0 rate, which is a real published
    value and must not be mistaken for "missing" and inflated to input."""
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata",
        lambda: {
            "premium-model": {
                "pricing": {
                    "prompt": "0.000005",
                    "completion": "0.00003",
                    "input_cache_write": "0.00000625",  # 1.25x input, Anthropic-style
                }
            },
            "free-write-model": {
                "pricing": {
                    "prompt": "0.000005",
                    "completion": "0.00003",
                    "input_cache_write": "0",  # published as free
                }
            },
        },
    )
    kwargs = dict(provider="openrouter", base_url="https://openrouter.ai/api/v1")
    usage = CanonicalUsage(input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000)

    premium = estimate_usage_cost("premium-model", usage, **kwargs)
    free = estimate_usage_cost("free-write-model", usage, **kwargs)
    premium_entry = get_pricing_entry("premium-model", **kwargs)
    input_rate = premium_entry.input_cost_per_million

    # Published premium honoured, and it is strictly above the input rate —
    # so it demonstrably was not overwritten by the fallback.
    assert premium.amount_usd == premium_entry.cache_write_cost_per_million
    assert premium.amount_usd > input_rate
    # A published zero stays zero rather than being read as "missing".
    assert free.amount_usd == 0
    # Neither annotates the fallback.
    assert not any("input rate" in n for n in premium.notes)
    assert not any("input rate" in n for n in free.notes)


def test_cache_write_still_unknown_when_input_rate_is_also_missing(monkeypatch):
    """Guard on the bail-out: with neither a cache-write nor an input rate the
    route is genuinely unpriceable and must stay unknown. The fallback must not
    paper over a truly absent price."""
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata",
        lambda: {"output-only-model": {"pricing": {"completion": "0.00001"}}},
    )

    result = estimate_usage_cost(
        "output-only-model",
        CanonicalUsage(input_tokens=0, output_tokens=10, cache_write_tokens=5000),
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
    )

    assert result.status == "unknown"
    assert result.amount_usd is None


def test_cache_read_does_not_fall_back_to_the_input_rate(monkeypatch):
    """The cache-read/cache-write asymmetry is deliberate and must hold: a
    cache READ is discounted, so substituting the input rate would over-bill
    rather than fill a gap. A missing cache-read rate stays unknown even
    though the input rate is known."""
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata",
        lambda: {
            "no-read-rate-model": {
                "pricing": {"prompt": "0.000005", "completion": "0.00003"}
            }
        },
    )

    result = estimate_usage_cost(
        "no-read-rate-model",
        CanonicalUsage(input_tokens=10, output_tokens=10, cache_read_tokens=5000),
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
    )

    assert result.status == "unknown"


def test_no_shipped_route_is_unpriceable_solely_because_of_cache_write():
    """Invariant over the shipped pricing snapshot: for every entry that can
    price a plain input+output turn, adding cache-write tokens must not turn
    the turn unpriceable. This is the bug class — on the pre-fix code every
    snapshot entry lacking a cache-write premium (OpenAI, DeepSeek, Google,
    Bedrock Nova, ...) dropped the whole turn to status=unknown."""
    from agent.usage_pricing import _OFFICIAL_DOCS_PRICING

    regressed = []
    for (provider, model), entry in _OFFICIAL_DOCS_PRICING.items():
        if entry.input_cost_per_million is None or entry.output_cost_per_million is None:
            continue
        baseline = estimate_usage_cost(
            model,
            CanonicalUsage(input_tokens=1000, output_tokens=500),
            provider=provider,
        )
        if baseline.status != "estimated":
            continue  # not priceable for unrelated reasons; not our contract
        with_write = estimate_usage_cost(
            model,
            CanonicalUsage(input_tokens=1000, output_tokens=500, cache_write_tokens=50_000),
            provider=provider,
        )
        if with_write.status != "estimated" or with_write.amount_usd is None:
            regressed.append(f"{provider}/{model}")
        elif with_write.amount_usd <= baseline.amount_usd:
            regressed.append(f"{provider}/{model} (cache-write billed as free)")

    assert not regressed, (
        "shipped routes made unpriceable by cache-write tokens alone: "
        f"{sorted(regressed)}"
    )
