from types import SimpleNamespace

from agent.usage_pricing import (
    CanonicalUsage,
    estimate_usage_cost,
    get_pricing_entry,
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
    """Cross-region inference profiles must resolve to the bare model pricing entry."""
    bedrock_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
    bare = get_pricing_entry(
        "anthropic.claude-sonnet-4-5", provider="bedrock", base_url=bedrock_url
    )
    assert bare is not None
    for prefix in ("us.", "global.", "eu.", "ap.", "jp."):
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


def test_bedrock_claude_opus_48_prices_via_region_prefix_strip():
    """A Bedrock cross-Region inference id must price after the prefix is stripped."""
    usage = CanonicalUsage(input_tokens=1_000_000, output_tokens=200_000)
    for model in (
        "us.anthropic.claude-opus-4-8",
        "eu.anthropic.claude-opus-4-8",
        "apac.anthropic.claude-opus-4-8",
        "global.anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-8",  # bare id (no prefix) also resolves
    ):
        result = estimate_usage_cost(model, usage, provider="bedrock")
        assert result.status == "estimated", model
        # 1M input × $5/M + 200K output × $25/M = $5.00 + $5.00 = $10.00
        assert float(result.amount_usd) == 10.0, model


def test_bedrock_claude_opus_48_includes_cache_rates_no_double_count():
    """Cache tokens price at the cache rate only."""
    usage = CanonicalUsage(
        input_tokens=100_000,
        output_tokens=20_000,
        cache_read_tokens=500_000,
        cache_write_tokens=40_000,
    )
    result = estimate_usage_cost(
        "us.anthropic.claude-opus-4-8", usage, provider="bedrock"
    )
    # input 100K×$5 + output 20K×$25 + cacheR 500K×$0.50 + cacheW 40K×$6.25
    # = 0.50 + 0.50 + 0.25 + 0.25 = $1.50
    assert result.status == "estimated"
    assert float(result.amount_usd) == 1.50


def test_bedrock_route_clears_base_url_to_avoid_network(monkeypatch):
    """The Bedrock route must use the pure official-docs dict lookup."""
    def _boom(*args, **kwargs):
        raise AssertionError("endpoint metadata fetch must not be called")

    monkeypatch.setattr("agent.usage_pricing.fetch_endpoint_model_metadata", _boom)
    route = resolve_billing_route(
        "us.anthropic.claude-opus-4-8",
        provider="bedrock",
        base_url="https://bedrock-runtime.ca-central-1.amazonaws.com",
    )
    assert route.base_url == ""
    assert route.billing_mode == "official_docs_snapshot"
    entry = get_pricing_entry(
        "us.anthropic.claude-opus-4-8",
        provider="bedrock",
        base_url="https://bedrock-runtime.ca-central-1.amazonaws.com",
    )
    assert entry is not None
    assert float(entry.input_cost_per_million) == 5.0


def test_bedrock_base_url_host_detection_without_explicit_provider():
    """A caller that passes the bedrock-runtime base_url but no provider name
    must still route to bedrock (host match)."""
    route = resolve_billing_route(
        "us.anthropic.claude-opus-4-8",
        provider=None,
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )
    assert route.provider == "bedrock"
    assert route.model == "anthropic.claude-opus-4-8"


def test_bedrock_unknown_model_stays_unknown():
    """A Bedrock id with no exact pricing entry must fail safe to unknown."""
    result = estimate_usage_cost(
        "us.meta.llama4-maverick",
        CanonicalUsage(input_tokens=1000, output_tokens=500),
        provider="bedrock",
    )
    assert result.status == "unknown"
    assert result.amount_usd is None


def test_bedrock_us_gov_prefix_strips_and_prices():
    """The us-gov. region prefix must also normalize to the priced id."""
    result = estimate_usage_cost(
        "us-gov.anthropic.claude-opus-4-8",
        CanonicalUsage(input_tokens=1_000_000, output_tokens=200_000),
        provider="bedrock",
    )
    assert result.status == "estimated"
    assert float(result.amount_usd) == 10.0


def test_bedrock_host_detection_does_not_false_positive_on_substring():
    """A non-AWS URL that merely CONTAINS 'bedrock' must NOT be routed to the
    bedrock docs-snapshot pricing (the substring-match false-positive class)."""
    route = resolve_billing_route(
        "claude-opus-4-8",
        provider=None,
        base_url="https://my-bedrock-proxy.example.com/v1",
    )
    assert route.provider != "bedrock"


def test_bedrock_branch_does_not_override_explicit_provider():
    """An explicit non-bedrock provider must win even if its base_url contains
    the substring 'bedrock' (the branch must key on host, not substring)."""
    route = resolve_billing_route(
        "claude-opus-4-8",
        provider="anthropic",
        base_url="https://anthropic-via-bedrock-proxy.example.com/v1",
    )
    assert route.provider == "anthropic"
    assert route.billing_mode == "official_docs_snapshot"


# ──────────────────────────────────────────────────────────────────────────
# Phase 1 — codex tables + bedrock-decoration helper lifted into core (inert)
# These mirror the verbatim spend_core tables/arithmetic so the unified core
# path can reprice company-OAuth codex without importing scripts/spend_core.py.
# ──────────────────────────────────────────────────────────────────────────
def test_codex_pricing_tables_present_and_cover_live_models():
    """The codex rate tables must cover every live company-OAuth model."""
    from agent.usage_pricing import CODEX_PRICING_STANDARD, CODEX_PRICING_PRIORITY

    models = (
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5",
    )
    for table in (CODEX_PRICING_STANDARD, CODEX_PRICING_PRIORITY):
        for model in models:
            assert model in table, model
            assert len(table[model]) == 4  # input, cached input, cache write, output


def test_codex_pricing_rates_match_openai_official_docs():
    from agent.usage_pricing import CODEX_PRICING_STANDARD, CODEX_PRICING_PRIORITY

    assert CODEX_PRICING_STANDARD["gpt-5.6-sol"] == (5.00, 0.50, 6.25, 30.00)
    assert CODEX_PRICING_STANDARD["gpt-5.6-terra"] == (2.50, 0.25, 3.125, 15.00)
    assert CODEX_PRICING_STANDARD["gpt-5.6-luna"] == (1.00, 0.10, 1.25, 6.00)
    assert CODEX_PRICING_PRIORITY["gpt-5.6-sol"] == (10.00, 1.00, 12.50, 60.00)
    assert CODEX_PRICING_PRIORITY["gpt-5.6-terra"] == (5.00, 0.50, 6.25, 30.00)
    assert CODEX_PRICING_PRIORITY["gpt-5.6-luna"] == (2.00, 0.20, 2.50, 12.00)
    assert CODEX_PRICING_PRIORITY["gpt-5.5"] == (12.50, 1.25, None, 75.00)
    assert CODEX_PRICING_PRIORITY["gpt-5.4"] == (5.00, 0.50, None, 30.00)
    assert CODEX_PRICING_PRIORITY["gpt-5.4-mini"] == (1.50, 0.15, None, 9.00)


def test_codex_cost_prices_gpt_5_6_cache_writes():
    from agent.usage_pricing import codex_cost

    usage = CanonicalUsage(
        input_tokens=1_000_000,
        output_tokens=200_000,
        cache_read_tokens=500_000,
        cache_write_tokens=100_000,
    )

    usd, key = codex_cost("gpt-5.6-terra", usage, tier="priority")

    expected = (
        1_000_000 * 5.00 / 1e6
        + 500_000 * 0.50 / 1e6
        + 100_000 * 6.25 / 1e6
        + 200_000 * 30.00 / 1e6
    )
    assert key == "gpt-5.6-terra"
    assert usd == expected


def test_codex_cost_uses_verbatim_float_arithmetic():
    """codex_cost must use the exact float arithmetic spend_core uses
    (inp*c_in/1e6 + cr*c_cached/1e6 + out*c_out/1e6) so the frozen golden,
    captured from that float math, reproduces bit-for-bit."""
    from agent.usage_pricing import codex_cost

    usage = CanonicalUsage(
        input_tokens=1_000_000,
        output_tokens=200_000,
        cache_read_tokens=500_000,
    )
    usd, key = codex_cost("gpt-5.5", usage, tier="priority")
    # priority gpt-5.5 = 12.50/1.25/75 per 1M
    expected = 1_000_000 * 12.50 / 1e6 + 500_000 * 1.25 / 1e6 + 200_000 * 75.00 / 1e6
    assert key == "gpt-5.5"
    assert usd == expected  # exact float equality — verbatim arithmetic


def test_codex_cost_standard_tier_differs_from_priority():
    from agent.usage_pricing import codex_cost

    usage = CanonicalUsage(input_tokens=1_000_000, output_tokens=0)
    std, _ = codex_cost("gpt-5.5", usage, tier="standard")
    pri, _ = codex_cost("gpt-5.5", usage, tier="priority")
    assert std == 5.00  # standard gpt-5.5 input
    assert pri == 12.50  # priority gpt-5.5 input


def test_codex_cost_unknown_model_returns_none_none():
    """A gpt-* model NOT in the table returns (None, None) so callers can
    FALL THROUGH to the family fallback instead of silently pricing $0."""
    from agent.usage_pricing import codex_cost

    usage = CanonicalUsage(input_tokens=100_000, output_tokens=10_000)
    usd, key = codex_cost("gpt-9.9-imaginary", usage, tier="priority")
    assert usd is None
    assert key is None


def test_strip_bedrock_decorations_matches_spend_core():
    """_strip_bedrock_decorations must strip region/provider/version/date
    decorations exactly as spend_core.normalize_bedrock does."""
    from agent.usage_pricing import _strip_bedrock_decorations

    assert (
        _strip_bedrock_decorations("us.anthropic.claude-opus-4-8")
        == "claude-opus-4-8"
    )
    assert (
        _strip_bedrock_decorations("eu.anthropic.claude-sonnet-4-5-v1:0")
        == "claude-sonnet-4-5"
    )
    assert (
        _strip_bedrock_decorations("apac.anthropic.claude-opus-4-8-20250101")
        == "claude-opus-4-8"
    )
    assert _strip_bedrock_decorations("claude-opus-4-8") == "claude-opus-4-8"


def test_phase1_helpers_are_inert_existing_estimate_unchanged():
    """Lifting the helpers must NOT change estimate_usage_cost: a codex row is
    still $0 included (upstream contract) until the gate is wired in Phase 2."""
    usage = CanonicalUsage(input_tokens=1_000_000, output_tokens=200_000)
    result = estimate_usage_cost("gpt-5.5", usage, provider="openai-codex")
    assert result.status == "included"
    assert result.amount_usd == 0


# ──────────────────────────────────────────────────────────────────────────
# Phase 2 — CorrectionsConfig + loader + apply_corrections + gated wiring
# ──────────────────────────────────────────────────────────────────────────
from decimal import Decimal  # noqa: E402


def _codex_usage():
    # 1M input, 200K output, 500K cache-read
    return CanonicalUsage(
        input_tokens=1_000_000, output_tokens=200_000, cache_read_tokens=500_000
    )


def test_corrections_config_defaults_all_off():
    """CorrectionsConfig() defaults: disabled, priority tier, factor 1 (no
    bedrock uplift) — the upstream-identical baseline."""
    from agent.usage_pricing import CorrectionsConfig

    cfg = CorrectionsConfig()
    assert cfg.enabled is False
    assert cfg.codex_tier == "priority"
    assert cfg.bedrock_cross_region_factor == Decimal("1")


def test_no_corrections_sentinel_is_disabled():
    from agent.usage_pricing import _NO_CORRECTIONS

    assert _NO_CORRECTIONS.enabled is False


def test_gate_off_codex_is_still_zero_included():
    """Gate ABSENT/OFF: openai-codex prices $0 included, byte-identical to
    upstream (criterion 5). corrections=None falls back to ambient load."""
    from agent.usage_pricing import CorrectionsConfig

    res = estimate_usage_cost(
        "gpt-5.5", _codex_usage(), provider="openai-codex",
        corrections=CorrectionsConfig(enabled=False),
    )
    assert res.status == "included"
    assert res.amount_usd == 0


def test_gate_on_codex_prices_real_api_rate():
    """Gate ON: openai-codex reprices at real OpenAI API priority rates (C1)."""
    from agent.usage_pricing import CorrectionsConfig

    res = estimate_usage_cost(
        "gpt-5.5", _codex_usage(), provider="openai-codex",
        corrections=CorrectionsConfig(enabled=True, codex_tier="priority"),
    )
    # priority gpt-5.5: 1M*12.50 + 500K*1.25 + 200K*75 per 1M = 12.50+0.625+15.0
    assert res.status == "estimated"
    assert abs(float(res.amount_usd) - 28.125) < 0.01


def test_gate_on_codex_tier_standard():
    from agent.usage_pricing import CorrectionsConfig

    res = estimate_usage_cost(
        "gpt-5.5", _codex_usage(), provider="openai-codex",
        corrections=CorrectionsConfig(enabled=True, codex_tier="standard"),
    )
    # standard gpt-5.5: 1M*5 + 500K*0.50 + 200K*30 per 1M = 5.0+0.25+6.0
    assert abs(float(res.amount_usd) - 11.25) < 0.01


def test_gate_on_codex_unknown_model_falls_through_not_zero():
    """A codex gpt-* model NOT in the table must FALL THROUGH (codex_cost None)
    — NOT return $0. With no family match it stays $0 included exactly as
    spend_core does (the syn_codex_unknown_gpt golden case)."""
    from agent.usage_pricing import CorrectionsConfig

    res = estimate_usage_cost(
        "gpt-9.9-imaginary", _codex_usage(), provider="openai-codex",
        corrections=CorrectionsConfig(enabled=True),
    )
    # gpt-9.9 not in codex table; no anthropic family match -> $0 included
    assert res.amount_usd == 0
    assert res.status == "included"


def test_gate_on_c2_mislabeled_claude_stamped_codex():
    """C2: a us.anthropic.* row mislabeled openai-codex must price by MODEL
    family (anthropic), not $0. (the syn_mislabel_claude_codex golden case)."""
    from agent.usage_pricing import CorrectionsConfig

    usage = CanonicalUsage(input_tokens=120_000, output_tokens=30_000)
    res = estimate_usage_cost(
        "us.anthropic.claude-opus-4-8", usage, provider="openai-codex",
        corrections=CorrectionsConfig(enabled=True),
    )
    # opus-4-8 $5/$25: 120K*5 + 30K*25 per 1M = 0.60 + 0.75 = 1.35
    assert res.status == "estimated"
    assert abs(float(res.amount_usd) - 1.35) < 0.01


def test_gate_on_c2_mislabeled_gpt_stamped_bedrock():
    """C2: a gpt-5.5 row mislabeled bedrock must price by MODEL family (codex
    real rate), not via the bedrock anthropic table (the
    syn_mislabel_gpt_bedrock golden case)."""
    from agent.usage_pricing import CorrectionsConfig

    usage = CanonicalUsage(input_tokens=120_000, output_tokens=30_000)
    res = estimate_usage_cost(
        "gpt-5.5", usage, provider="bedrock",
        corrections=CorrectionsConfig(enabled=True, codex_tier="priority"),
    )
    # priority gpt-5.5: 120K*12.50 + 30K*75 per 1M = 1.50 + 2.25 = 3.75
    assert res.status == "estimated"
    assert abs(float(res.amount_usd) - 3.75) < 0.01


def test_gate_on_bedrock_base_unchanged_with_factor_one():
    """Decision-1 option A: with factor=1 a bedrock row prices at base rate,
    identical to gate-off (no uplift)."""
    from agent.usage_pricing import CorrectionsConfig

    usage = CanonicalUsage(input_tokens=1_000_000, output_tokens=200_000)
    base = estimate_usage_cost("us.anthropic.claude-opus-4-8", usage, provider="bedrock")
    on = estimate_usage_cost(
        "us.anthropic.claude-opus-4-8", usage, provider="bedrock",
        corrections=CorrectionsConfig(enabled=True, bedrock_cross_region_factor=Decimal("1")),
    )
    assert float(on.amount_usd) == float(base.amount_usd) == 10.0


def test_gate_on_bedrock_factor_uplift_applies_when_set():
    """The bedrock uplift knob multiplies a bedrock-routed row when factor != 1
    (the mechanism shipped gated; default stays 1)."""
    from agent.usage_pricing import CorrectionsConfig

    usage = CanonicalUsage(input_tokens=1_000_000, output_tokens=200_000)
    on = estimate_usage_cost(
        "us.anthropic.claude-opus-4-8", usage, provider="bedrock",
        corrections=CorrectionsConfig(enabled=True, bedrock_cross_region_factor=Decimal("1.10")),
    )
    assert abs(float(on.amount_usd) - 11.0) < 0.001


def test_gate_on_anthropic_row_not_doubled():
    """A normal anthropic (non-bedrock) priced row is unchanged by corrections
    (no codex repricing, no bedrock factor)."""
    from agent.usage_pricing import CorrectionsConfig

    usage = CanonicalUsage(input_tokens=120_000, output_tokens=30_000)
    base = estimate_usage_cost("claude-opus-4-8", usage, provider="anthropic")
    on = estimate_usage_cost(
        "claude-opus-4-8", usage, provider="anthropic",
        corrections=CorrectionsConfig(enabled=True, bedrock_cross_region_factor=Decimal("1.10")),
    )
    assert float(on.amount_usd) == float(base.amount_usd) == 1.35


def test_load_corrections_config_default_off(monkeypatch, tmp_path):
    """No cost_corrections block -> all-off defaults, never raises."""
    import agent.usage_pricing as up

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("model:\n  default: x\n")
    monkeypatch.setattr(up, "_corrections_config_path", lambda: str(cfg_file))
    up.load_corrections_config.cache_clear() if hasattr(up.load_corrections_config, "cache_clear") else None
    cfg = up.load_corrections_config(force_reload=True)
    assert cfg.enabled is False
    assert cfg.bedrock_cross_region_factor == Decimal("1")


def test_load_corrections_config_reads_block(monkeypatch, tmp_path):
    import agent.usage_pricing as up

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "cost_corrections:\n  enabled: true\n  codex_tier: standard\n"
        "  bedrock_cross_region_factor: 1.0\n"
    )
    monkeypatch.setattr(up, "_corrections_config_path", lambda: str(cfg_file))
    monkeypatch.delenv("SPEND_CODEX_TIER", raising=False)
    cfg = up.load_corrections_config(force_reload=True)
    assert cfg.enabled is True
    assert cfg.codex_tier == "standard"
    assert cfg.bedrock_cross_region_factor == Decimal("1.0")


def test_load_corrections_config_env_tier_wins(monkeypatch, tmp_path):
    """SPEND_CODEX_TIER env alias overrides the config block tier."""
    import agent.usage_pricing as up

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("cost_corrections:\n  enabled: true\n  codex_tier: standard\n")
    monkeypatch.setattr(up, "_corrections_config_path", lambda: str(cfg_file))
    monkeypatch.setenv("SPEND_CODEX_TIER", "priority")
    cfg = up.load_corrections_config(force_reload=True)
    assert cfg.codex_tier == "priority"


def test_load_corrections_config_malformed_is_safe(monkeypatch, tmp_path):
    """A malformed cost_corrections block must NOT raise — safe defaults."""
    import agent.usage_pricing as up

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("cost_corrections: [not, a, mapping]\n")
    monkeypatch.setattr(up, "_corrections_config_path", lambda: str(cfg_file))
    cfg = up.load_corrections_config(force_reload=True)
    assert cfg.enabled is False


def test_cost_result_has_breakdown_fields():
    """CostResult gains breakdown / base_amount_usd / adjustments (Phase 5
    consumes these; they must exist from Phase 2)."""
    from agent.usage_pricing import CostResult

    usage = CanonicalUsage(input_tokens=1_000_000, output_tokens=200_000)
    res = estimate_usage_cost("claude-opus-4-8", usage, provider="anthropic")
    assert hasattr(res, "breakdown")
    assert hasattr(res, "base_amount_usd")
    assert hasattr(res, "adjustments")
