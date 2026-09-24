from decimal import Decimal
from types import SimpleNamespace

import pytest

from agent.usage_pricing import (
    _OFFICIAL_DOCS_PRICING,
    CanonicalUsage,
    format_cost_label,
    estimate_usage_cost,
    get_pricing_entry,
    normalize_usage,
    resolve_billing_route,
)


def test_astra_whole_request_price_tier_includes_cache_writes():
    below = estimate_usage_cost(
        "gpt-6-astra",
        CanonicalUsage(input_tokens=100_000, output_tokens=10_000, cache_read_tokens=10_000, cache_write_tokens=10_000),
        provider="openai",
    )
    above = estimate_usage_cost(
        "gpt-6-astra",
        CanonicalUsage(input_tokens=100_000, output_tokens=10_000, cache_read_tokens=100_000, cache_write_tokens=100_001),
        provider="openai",
    )

    # Whole-request tier: crossing 272K prompt tokens re-prices every component of the request,
    # including cache writes, at the *_above rates — so the cost ratio exceeds the token ratio.
    entry = _OFFICIAL_DOCS_PRICING[("openai", "gpt-6-astra")]
    assert above.amount_usd == (
        Decimal(100_000) * entry.input_cost_per_million_above
        + Decimal(10_000) * entry.output_cost_per_million_above
        + Decimal(100_000) * entry.cache_read_cost_per_million_above
        + Decimal(100_001) * entry.cache_write_cost_per_million_above
    ) / Decimal(1_000_000)
    assert below.amount_usd < above.amount_usd


_MODELS_DEV_REGISTRY = {
    "openai": {"models": {"gpt-5-nano": {"cost": {"input": 0.05, "output": 0.4, "cache_read": 0.005}}}},
    "xai": {"models": {"grok-4.3": {"cost": {"input": 1.25, "output": 2.5, "cache_read": 0.2}}}},
}
_USAGE = CanonicalUsage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=1_000_000)


@pytest.fixture
def models_dev_registry(monkeypatch):
    """A models.dev cache holding the vendors' rate cards; the providers' own /models carry no prices."""
    import agent.models_dev as models_dev

    monkeypatch.setattr(models_dev, "_models_dev_cache", _MODELS_DEV_REGISTRY)
    monkeypatch.setattr("agent.usage_pricing.fetch_endpoint_model_metadata", lambda *_a, **_k: {})


@pytest.mark.parametrize(("provider", "base_url", "model", "expected"), [
    ("openai-api", "https://api.openai.com/v1", "gpt-5-nano", ("estimated", Decimal("0.455"))),
    ("openai", "", "gpt-5-nano", ("estimated", Decimal("0.455"))),
    ("xai", "https://api.x.ai/v1", "grok-4.3", ("estimated", Decimal("3.95"))),
    # The vendor's list price needs the vendor's own API: same provider name on
    # someone else's host, a downgraded origin, a subscription route or a custom
    # endpoint keep ``unknown`` rather than inheriting it.
    ("xai", "https://grok-relay.example.com/v1", "grok-4.3", ("unknown", None)),
    ("xai", "http://api.x.ai/v1", "grok-4.3", ("unknown", None)),
    ("xai-oauth", "https://api.x.ai/v1", "grok-4.3", ("unknown", None)),
    ("custom", "https://api.x.ai/v1", "grok-4.3", ("unknown", None)),
])
def test_direct_first_party_route_prices_models_missing_from_snapshot(models_dev_registry, provider, base_url, model, expected):
    cost = estimate_usage_cost(model, _USAGE, provider=provider, base_url=base_url)

    assert (cost.status, cost.amount_usd) == expected


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


















def test_bundled_pricing_skips_endpoint_metadata(monkeypatch):
    """An exact bundled price must not block on the provider's /models API."""
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_endpoint_model_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("endpoint metadata should not be fetched")
        ),
    )

    entry = get_pricing_entry(
        "deepseek-chat",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
    )

    assert entry is not None
    assert entry.source == "official_docs_snapshot"


def test_unknown_model_falls_back_to_endpoint_metadata(monkeypatch):
    """Models absent from the bundled table still use endpoint pricing."""
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_endpoint_model_metadata",
        lambda *_args, **_kwargs: {
            "deepseek-future": {
                "pricing": {"prompt": "0.000001", "completion": "0.000002"}
            }
        },
    )

    entry = get_pricing_entry(
        "deepseek-future",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
    )

    assert entry is not None
    assert entry.source == "provider_models_api"
    assert entry.input_cost_per_million == Decimal("1")
    assert entry.output_cost_per_million == Decimal("2")


def test_openrouter_metadata_uses_highest_applicable_prompt_tier(monkeypatch):
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata",
        lambda: {
            "openai/tiered": {
                "pricing": {
                    "prompt": "0.000002",
                    "completion": "0.000010",
                    "input_cache_read": "0.0000002",
                    "input_cache_write": "0.0000025",
                    "overrides": [
                        {
                            "min_prompt_tokens": 128_000,
                            "prompt": "0.00000195",
                            "input_cache_read": "0.0000003",
                        },
                        {
                            "min_prompt_tokens": 32_000,
                            "prompt": "0.00000156",
                            "completion": "0.000012",
                            "input_cache_read": "0.00000025",
                            "input_cache_write": "0.000003",
                        },
                    ],
                }
            }
        },
    )

    def estimate(input_tokens, cache_read_tokens, cache_write_tokens):
        return estimate_usage_cost(
            "openai/tiered",
            CanonicalUsage(
                input_tokens=input_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                output_tokens=1_000,
            ),
            provider="openrouter",
        ).amount_usd

    assert estimate(4_000, 3_000, 3_000) == Decimal("0.0261")
    assert estimate(20_000, 10_000, 10_000) == Decimal("0.0757")
    # The 128k row omits completion and cache-write, so those components
    # inherit the 32k rates while input and cache-read use the highest tier.
    assert estimate(100_000, 50_000, 50_000) == Decimal("0.3720")




def test_deepseek_deprecated_aliases_price_as_flash():
    """Invariant: deepseek-v4-flash / deepseek-chat / deepseek-reasoner are retired aliases
    served by the current Flash model — they must bill at identical rates to the
    ``deepseek-flash`` entry, or sessions on the legacy names over/under-report cost."""
    flash = get_pricing_entry("deepseek-flash", provider="deepseek")
    assert flash is not None
    for alias in ("deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner"):
        entry = get_pricing_entry(alias, provider="deepseek")
        assert entry is not None, alias
        assert entry.input_cost_per_million == flash.input_cost_per_million, alias
        assert entry.output_cost_per_million == flash.output_cost_per_million, alias
        assert (
            entry.cache_read_cost_per_million == flash.cache_read_cost_per_million
        ), alias




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












def test_google_and_vertex_routes_share_official_pricing_snapshot():
    """Direct Gemini, Vertex, and Vertex's OpenAI-compatible hostname must
    all normalize to the Google official-pricing route.
    """
    routes = (
        resolve_billing_route("model", provider="gemini"),
        resolve_billing_route("google/model", provider="vertex"),
        resolve_billing_route(
            "google/model",
            provider="custom",
            base_url="https://aiplatform.googleapis.com/v1/projects/example",
        ),
    )

    assert all(route.provider == "google" for route in routes)
    assert all(route.billing_mode == "official_docs_snapshot" for route in routes)


def test_vertex_default_model_estimates_cached_usage(monkeypatch):
    """The bundled Vertex profile's default auxiliary model must fall back to
    Google snapshot pricing when the OpenAI-compatible endpoint has no model
    metadata, including for cache-read accounting.
    """
    from providers import get_provider_profile

    monkeypatch.setattr(
        "agent.usage_pricing.fetch_endpoint_model_metadata",
        lambda *_args, **_kwargs: {},
    )
    vertex = get_provider_profile("vertex")
    result = estimate_usage_cost(
        vertex.default_aux_model,
        CanonicalUsage(input_tokens=100, output_tokens=100, cache_read_tokens=100),
        provider=vertex.name,
        base_url=vertex.base_url,
    )

    assert result.status == "estimated"
    assert result.amount_usd is not None and result.amount_usd > 0


def test_curated_google_flash_models_resolve_official_snapshot_pricing(monkeypatch):
    """Every ``google/gemini-*-flash`` model curated for the OpenRouter and Nous
    pickers must also bill through the Google official-docs snapshot on the
    direct Gemini and Vertex routes — a model pickable via the aggregators but
    ``unknown`` to Google-route accounting is a catalog/pricing drift.
    """
    from hermes_cli.models_catalog_static import OPENROUTER_MODELS, _PROVIDER_MODELS

    monkeypatch.setattr(
        "agent.usage_pricing.fetch_endpoint_model_metadata",
        lambda *_args, **_kwargs: {},
    )
    curated = {m for m, _desc in OPENROUTER_MODELS} | set(_PROVIDER_MODELS["nous"])
    flash = sorted(m for m in curated if m.startswith("google/gemini-") and m.endswith("-flash"))
    assert flash, "expected curated google/gemini-*-flash picker entries"
    usage = CanonicalUsage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=1_000_000)
    for model in flash:
        bare = model.split("/", 1)[1]
        gemini = estimate_usage_cost(bare, usage, provider="gemini")
        vertex = estimate_usage_cost(model, usage, provider="vertex")
        assert gemini.status == "estimated", (model, gemini.status)
        assert gemini.source == "official_docs_snapshot", model
        assert vertex.amount_usd == gemini.amount_usd, model
        # Direct-route models the picker offers must also be pickable directly.
        assert bare in _PROVIDER_MODELS["gemini"], model
        assert model in _PROVIDER_MODELS["vertex"], model






# ---------------------------------------------------------------------------
# Cost label formatting (#79220: sub-cent costs render as $0.00)
# ---------------------------------------------------------------------------


class TestFormatCostLabel:
    """Tests for magnitude-scaled cost label formatting."""

    def test_zero_renders_as_dollar_zero(self):
        assert format_cost_label(Decimal("0")) == "$0.00"

    def test_sub_cent_renders_4dp(self):
        """Costs below $0.01 render at 4 decimal places (#79220)."""
        label = format_cost_label(Decimal("0.004640"))
        assert label == "~$0.0046"
        # Must NOT be $0.00
        assert "$0.00" != label

    def test_exactly_one_cent_renders_2dp(self):
        """$0.01 renders at 2dp."""
        assert format_cost_label(Decimal("0.01")) == "~$0.01"

    def test_normal_cost_renders_2dp(self):
        assert format_cost_label(Decimal("1.23")) == "~$1.23"



    def test_below_4dp_floor_never_reads_zero(self):
        """Amounts below $0.00005 must not render as '~$0.0000' (#79220).

        4dp truncation of a positive amount would produce a zero-looking
        label — the exact dishonesty the formatter exists to fix.
        """
        label = format_cost_label(Decimal("0.00004"))
        assert label == "~$<0.0001"
        # Exact boundary: $0.00005 rounds to 0.0000 under ROUND_HALF_EVEN
        # and must also take the fallback.
        assert format_cost_label(Decimal("0.00005")) == "~$<0.0001"



# ---------------------------------------------------------------------------
# Subscription-included cost notes
# ---------------------------------------------------------------------------


class TestSubscriptionIncludedNotes:
    """Subscription-included costs should carry a note clarifying no invoice."""

    def test_included_cost_has_note(self):
        """estimate_usage_cost for subscription-included route includes a note."""
        # openai-codex is subscription_included
        usage = CanonicalUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
        )
        result = estimate_usage_cost(
            "gpt-5.4-mini",
            usage,
            provider="openai-codex",
        )
        assert result.status == "included"
        assert result.amount_usd == Decimal("0")
        assert len(result.notes) > 0


def test_normalize_usage_reads_kimi_top_level_cached_tokens():
    """Kimi/Moonshot's native API reports context-cache hits as a top-level
    usage.cached_tokens, not OpenAI's nested
    prompt_tokens_details.cached_tokens and not DeepSeek's
    prompt_cache_hit_tokens. Neither existing fallback matches that name, so
    direct Kimi sessions normalized to cache_read_tokens=0 — the hits were
    invisible in accounting and billed at the full input rate (#65722)."""
    usage = SimpleNamespace(
        prompt_tokens=3000,
        completion_tokens=250,
        cached_tokens=1800,
    )

    normalized = normalize_usage(usage, provider="kimi", api_mode="chat_completions")

    assert normalized.cache_read_tokens == 1800
    # prompt_tokens includes the cached prefix: 3000 - 1800 = fresh input
    assert normalized.input_tokens == 1200
    assert normalized.output_tokens == 250


def test_kimi_fallback_does_not_override_the_nested_openai_shape():
    """A provider that reports BOTH shapes must keep the nested value.

    The new branch is last in the chain, so it only fills a genuine zero.
    """
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=100,
        prompt_tokens_details=SimpleNamespace(cached_tokens=400),
        cached_tokens=999,  # must be ignored
    )

    normalized = normalize_usage(usage, provider="kimi", api_mode="chat_completions")

    assert normalized.cache_read_tokens == 400


def test_kimi_fallback_does_not_override_deepseek_hit_tokens():
    usage = SimpleNamespace(
        prompt_tokens=2000,
        completion_tokens=100,
        prompt_cache_hit_tokens=1500,
        cached_tokens=999,  # must be ignored
    )

    normalized = normalize_usage(usage, provider="deepseek", api_mode="chat_completions")

    assert normalized.cache_read_tokens == 1500


def test_usage_without_any_cache_fields_still_normalizes():
    usage = SimpleNamespace(prompt_tokens=500, completion_tokens=50)

    normalized = normalize_usage(usage, provider="kimi", api_mode="chat_completions")

    assert normalized.cache_read_tokens == 0
    assert normalized.input_tokens == 500


def test_normalize_usage_handles_dict_shaped_usage():
    """Regression test for #74314: when the Responses API returns usage as a
    plain dict (e.g. from a middleware/proxy that deserialises JSON to dict
    instead of a typed SDK object), normalize_usage() must read the same
    token counts as it would from an attribute-style object.

    Before this fix, getattr() on a dict silently returned 0 for every field,
    so token counts and cost appeared as zero for dict-shaped usage.
    """
    # Same payload as both a dict and a SimpleNamespace
    payload = {
        "input_tokens": 100,
        "output_tokens": 20,
        "input_tokens_details": {"cached_tokens": 60, "cache_creation_tokens": 10},
    }
    ns = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        input_tokens_details=SimpleNamespace(cached_tokens=60, cache_creation_tokens=10),
    )

    dict_result = normalize_usage(payload, api_mode="codex_responses")
    ns_result = normalize_usage(ns, api_mode="codex_responses")

    assert dict_result.input_tokens == ns_result.input_tokens, f"input_tokens: dict={dict_result.input_tokens} vs ns={ns_result.input_tokens}"
    assert dict_result.output_tokens == ns_result.output_tokens, f"output_tokens: dict={dict_result.output_tokens} vs ns={ns_result.output_tokens}"
    assert dict_result.cache_read_tokens == ns_result.cache_read_tokens, f"cache_read: dict={dict_result.cache_read_tokens} vs ns={ns_result.cache_read_tokens}"
    assert dict_result.cache_write_tokens == ns_result.cache_write_tokens, f"cache_write: dict={dict_result.cache_write_tokens} vs ns={ns_result.cache_write_tokens}"
    # Sanity: values must be non-zero (the whole point of the bug)
    assert dict_result.input_tokens > 0
    assert dict_result.cache_read_tokens > 0


def test_normalize_usage_handles_dict_openai_chat_completions():
    """Dict-shaped usage must also work in the default (OpenAI chat-completions)
    branch, not just the codex_responses branch.
    """
    payload = {
        "prompt_tokens": 500,
        "completion_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 200},
        "completion_tokens_details": {"reasoning_tokens": 30},
    }

    result = normalize_usage(payload, api_mode="chat_completions")

    assert result.output_tokens == 100
    assert result.cache_read_tokens == 200
    assert result.input_tokens == 500 - 200  # prompt_total - cache_read
    assert result.reasoning_tokens == 30


def test_normalize_usage_openai_reads_nested_cache_creation_tokens():
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=100,
            cache_creation_input_tokens=300,
        ),
    )

    normalized = normalize_usage(usage, provider="openrouter", api_mode="chat_completions")

    assert normalized.cache_read_tokens == 100
    assert normalized.cache_write_tokens == 300
    assert normalized.input_tokens == 600


def test_normalize_usage_openai_reads_mapping_cache_creation_tokens():
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "prompt_tokens_details": {"cache_creation_input_tokens": 300},
    }

    normalized = normalize_usage(usage, provider="openrouter", api_mode="chat_completions")

    assert normalized.cache_write_tokens == 300
    assert normalized.input_tokens == 700


def test_normalize_usage_openai_prefers_nested_cache_write_tokens():
    usage = SimpleNamespace(
        prompt_tokens=1000,
        prompt_tokens_details=SimpleNamespace(
            cache_write_tokens=200,
            cache_creation_input_tokens=300,
        ),
        cache_creation_input_tokens=400,
        cache_write_tokens=500,
    )

    normalized = normalize_usage(usage, provider="openrouter", api_mode="chat_completions")

    assert normalized.cache_write_tokens == 200


def test_normalize_usage_mapping_preserves_reasoning_tokens():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 40},
        "completion_tokens_details": {"reasoning_tokens": 12},
    }

    normalized = normalize_usage(usage, provider="openrouter", api_mode="chat_completions")

    assert normalized.reasoning_tokens == 12


def test_normalize_usage_mapping_anthropic_fields():
    usage = {
        "input_tokens": 80,
        "output_tokens": 20,
        "cache_read_input_tokens": 50,
        "cache_creation_input_tokens": 10,
        "output_tokens_details": {"reasoning_tokens": 7},
    }

    normalized = normalize_usage(usage, provider="anthropic", api_mode="anthropic_messages")

    assert normalized.cache_read_tokens == 50
    assert normalized.cache_write_tokens == 10
    assert normalized.reasoning_tokens == 7


def test_normalize_usage_mapping_codex_fields():
    usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "input_tokens_details": {
            "cached_tokens": 60,
            "cache_creation_tokens": 10,
        },
        "output_tokens_details": {"reasoning_tokens": 5},
    }

    normalized = normalize_usage(usage, provider="openai-codex", api_mode="codex_responses")

    assert normalized.input_tokens == 30
    assert normalized.cache_read_tokens == 60
    assert normalized.cache_write_tokens == 10
    assert normalized.reasoning_tokens == 5


def test_normalize_usage_clamps_negative_counters():
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=-5,
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=-10,
            cache_write_tokens=-20,
        ),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=-3),
    )

    normalized = normalize_usage(usage, provider="openrouter", api_mode="chat_completions")

    assert normalized.input_tokens == 100
    assert normalized.output_tokens == 0
    assert normalized.cache_read_tokens == 0
    assert normalized.cache_write_tokens == 0
    assert normalized.reasoning_tokens == 0


def test_normalize_usage_clamps_inconsistent_cache_total():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "prompt_tokens_details": {
            "cached_tokens": 80,
            "cache_creation_input_tokens": 50,
        },
    }

    normalized = normalize_usage(usage, provider="openrouter", api_mode="chat_completions")

    assert normalized.input_tokens == 0
    assert normalized.prompt_tokens == 130


def test_normalize_usage_codex_responses_reads_cache_write_tokens():
    """GPT-5.6+ explicit prompt caching reports cache writes as
    input_tokens_details.cache_write_tokens (billed at 1.25x), per OpenAI's
    documented Responses API schema. Before this fix, the codex_responses
    branch only read the undocumented `cache_creation_tokens` name and always
    normalized cache writes to 0."""
    usage = SimpleNamespace(
        input_tokens=2006,
        output_tokens=400,
        input_tokens_details=SimpleNamespace(cached_tokens=1920, cache_write_tokens=50),
    )

    normalized = normalize_usage(usage, provider="openai", api_mode="codex_responses")

    assert normalized.cache_read_tokens == 1920
    assert normalized.cache_write_tokens == 50
    assert normalized.input_tokens == 2006 - 1920 - 50


def test_normalize_usage_codex_responses_falls_back_to_cache_creation_tokens():
    """If cache_write_tokens is absent, fall back to the legacy
    cache_creation_tokens name rather than reporting 0."""
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=100,
        input_tokens_details=SimpleNamespace(cached_tokens=200, cache_creation_tokens=80),
    )

    normalized = normalize_usage(usage, provider="openai", api_mode="codex_responses")

    assert normalized.cache_write_tokens == 80


def test_normalize_usage_reads_qwen_flat_cached_tokens():
    """Some Alibaba/Qwen regional endpoints report cache reads as a flat
    `usage.cached_tokens` field with no `prompt_tokens_details` wrapper at
    all. Before this fix, those responses fell through every branch and
    normalized to cache_read_tokens=0, undercounting cost."""
    usage = SimpleNamespace(
        prompt_tokens=2000,
        completion_tokens=300,
        cached_tokens=1200,
    )

    normalized = normalize_usage(usage, provider="qwen", api_mode="chat_completions")

    assert normalized.cache_read_tokens == 1200
    assert normalized.input_tokens == 800


def test_normalize_usage_nested_details_win_over_qwen_flat_top_level():
    """When both shapes are present, the nested OpenAI-style value wins and
    the flat Qwen field is not double-read."""
    usage = SimpleNamespace(
        prompt_tokens=2000,
        completion_tokens=100,
        prompt_tokens_details=SimpleNamespace(cached_tokens=900),
        cached_tokens=1200,
    )

    normalized = normalize_usage(usage, provider="qwen", api_mode="chat_completions")

    assert normalized.cache_read_tokens == 900
    assert normalized.input_tokens == 1100


# ── Context-tiered pricing (Gemini Pro >200k prompts, #93469) ─────────────


def test_gemini_31_pro_below_tier_threshold_uses_base_rates():
    """Prompts at or below 200k tokens bill at the base rates — the tier
    fields must not change any below-threshold estimate."""
    result = estimate_usage_cost(
        "gemini-3.1-pro",
        CanonicalUsage(input_tokens=100_000, output_tokens=10_000),
        provider="google",
    )
    # 100k * $2/M + 10k * $12/M
    assert result.amount_usd == Decimal("0.32")

    at_threshold = estimate_usage_cost(
        "gemini-3.1-pro",
        CanonicalUsage(input_tokens=200_000, output_tokens=10_000),
        provider="google",
    )
    # Exactly 200k is still the lower tier (Google bills "> 200k" higher).
    # 200k * $2/M + 10k * $12/M
    assert at_threshold.amount_usd == Decimal("0.52")


def test_gemini_31_pro_above_tier_threshold_uses_tiered_rates_whole_request():
    """Once the prompt exceeds 200k tokens the >200k rates ($4 input /
    $18 output per million) apply to the ENTIRE request, not just the
    marginal tokens — matching Google's billing semantics (#93469).

    Before the fix this request priced at 250k*$2/M + 10k*$12/M = $0.62,
    under-counting input 2x and output 1.5x."""
    result = estimate_usage_cost(
        "gemini-3.1-pro",
        CanonicalUsage(input_tokens=250_000, output_tokens=10_000),
        provider="google",
    )
    # 250k * $4/M + 10k * $18/M
    assert result.amount_usd == Decimal("1.18")
    assert result.status == "estimated"


def test_gemini_31_pro_cache_read_tokens_count_toward_tier_and_tier_rate():
    """prompt_tokens (input + cache read + cache write) drives tier selection,
    and cache reads above the threshold bill at the $0.40/M tier rate."""
    result = estimate_usage_cost(
        "gemini-3.1-pro",
        CanonicalUsage(input_tokens=150_000, cache_read_tokens=100_000),
        provider="google",
    )
    # prompt = 250k > 200k → 150k * $4/M + 100k * $0.40/M
    assert result.amount_usd == Decimal("0.64")


def test_gemini_31_pro_preview_alias_shares_tiered_pricing():
    """The provider-emitted preview id aliases the canonical row, so it must
    pick up the tier fields too."""
    result = estimate_usage_cost(
        "gemini-3.1-pro-preview",
        CanonicalUsage(input_tokens=250_000, output_tokens=10_000),
        provider="google",
    )
    assert result.amount_usd == Decimal("1.18")


def test_gemini_25_pro_tiered_rates_with_cache_read_fallback():
    """gemini-2.5-pro tiers at the same 200k threshold ($2.50 input / $15
    output above). Its snapshot has no tiered cache-read rate, so cache reads
    fall back to the base $0.125/M even above the threshold."""
    result = estimate_usage_cost(
        "gemini-2.5-pro",
        CanonicalUsage(input_tokens=250_000, output_tokens=10_000),
        provider="google",
    )
    # 250k * $2.50/M + 10k * $15/M
    assert result.amount_usd == Decimal("0.775")

    with_cache = estimate_usage_cost(
        "gemini-2.5-pro",
        CanonicalUsage(input_tokens=150_000, cache_read_tokens=100_000),
        provider="google",
    )
    # prompt = 250k > 200k → 150k * $2.50/M + 100k * $0.125/M (base fallback)
    assert with_cache.amount_usd == Decimal("0.3875")


def test_flat_entries_unaffected_by_tier_machinery():
    """Entries without tier fields keep pricing every token at the flat rate
    no matter how large the prompt is."""
    entry = get_pricing_entry("gemini-3.1-flash-lite", provider="google")
    assert entry is not None
    assert entry.tier_threshold_tokens is None

    result = estimate_usage_cost(
        "gemini-3.1-flash-lite",
        CanonicalUsage(input_tokens=250_000, output_tokens=10_000),
        provider="google",
    )
    # 250k * $0.25/M + 10k * $1.50/M
    assert result.amount_usd == Decimal("0.0775")


def _anthropic_usage(speed=None):
    """Canonical usage from an Anthropic-shaped usage payload; ``speed`` rides ``raw_usage``."""
    payload = {"input_tokens": 100_000, "output_tokens": 10_000, "cache_read_input_tokens": 200_000, "cache_creation_input_tokens": 50_000}
    if speed:
        payload["speed"] = speed
    return normalize_usage(payload, provider="anthropic", api_mode="anthropic_messages")


def test_anthropic_fast_mode_responses_price_from_the_fast_rate_row():
    from agent.model_metadata import _ANTHROPIC_FAST_MODE_MODELS
    from agent.usage_pricing import _ANTHROPIC_FAST_MODE_PRICING

    # Every model the fast-mode gate sends ``speed`` to has a fast rate and a standard rate.
    assert set(_ANTHROPIC_FAST_MODE_PRICING) == set(_ANTHROPIC_FAST_MODE_MODELS)
    for model in _ANTHROPIC_FAST_MODE_MODELS:
        standard_entry = get_pricing_entry(model, provider="anthropic")
        assert standard_entry is not None
        fast = estimate_usage_cost(model, _anthropic_usage("fast"), provider="anthropic")
        standard = estimate_usage_cost(model, _anthropic_usage(), provider="anthropic")
        assert fast.pricing_version == _ANTHROPIC_FAST_MODE_PRICING[model].pricing_version
        assert standard.pricing_version == standard_entry.pricing_version
        assert fast.amount_usd > standard.amount_usd
        # A response the API reports as standard speed is billed like one without the field.
        assert estimate_usage_cost(model, _anthropic_usage("standard"), provider="anthropic").amount_usd == standard.amount_usd
    # Vendor-prefixed, dotted ids price from the same fast row.
    assert estimate_usage_cost("anthropic/claude-opus-5.5", _anthropic_usage("fast"), provider="anthropic").amount_usd == (
        estimate_usage_cost("claude-opus-5-5", _anthropic_usage("fast"), provider="anthropic").amount_usd
    )


def test_anthropic_fast_response_without_a_fast_rate_is_unknown():
    result = estimate_usage_cost("claude-sonnet-4-6", _anthropic_usage("fast"), provider="anthropic")
    assert result.amount_usd is None
    assert result.status == "unknown"
