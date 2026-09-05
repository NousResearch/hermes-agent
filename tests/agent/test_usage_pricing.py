from types import SimpleNamespace

from agent.usage_pricing import (
    CanonicalUsage,
    format_cost_label,
    estimate_usage_cost,
    get_pricing_entry,
    normalize_usage,
    resolve_billing_route,
)
from decimal import Decimal








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










def test_claude_agent_sdk_routes_as_subscription_included():
    """claude-agent-sdk is the subscription twin of openai-codex: the SDK
    bills the Claude subscription (OAuth), so the billing route must be
    subscription_included — never the metered anthropic docs snapshot."""
    from agent.usage_pricing import resolve_billing_route

    route = resolve_billing_route("claude-opus-4-8", provider="claude-agent-sdk")
    assert route.provider == "claude-agent-sdk"
    assert route.billing_mode == "subscription_included"

    result = estimate_usage_cost(
        "claude-opus-4-8",
        CanonicalUsage(input_tokens=1000, output_tokens=500),
        provider="claude-agent-sdk",
        base_url="",
    )
    assert result.status == "included"
    assert float(result.amount_usd) == 0.0








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


def test_normalize_usage_minimax_logs_cache_observability(caplog):
    """MiniMax providers on the Anthropic wire emit a debug-level
    cache-observability line recording the observable fields
    (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens),
    so an operator can see real cache behavior without trusting the
    misleading cache_read number (constant +128 floor on MiniMax-M3).
    Standard logging level gating applies — no separate opt-in flag.
    """
    usage = SimpleNamespace(
        input_tokens=1,
        output_tokens=11,
        cache_read_input_tokens=8594,
        cache_creation_input_tokens=0,
    )

    with caplog.at_level("DEBUG", logger="agent.usage_pricing"):
        normalize_usage(
            usage,
            provider="minimax-cn",
            api_mode="anthropic_messages",
        )

    cache_obs_records = [r for r in caplog.records if "cache_observability" in r.message]
    assert len(cache_obs_records) == 1
    record = cache_obs_records[0]
    assert "input_tokens=1" in record.message
    assert "output_tokens=11" in record.message
    assert "cache_read_tokens=8594" in record.message
    assert "cache_write_tokens=0" in record.message


def test_normalize_usage_native_anthropic_no_cache_observability(caplog):
    """The MiniMax cache-observability line must NOT fire for native
    Anthropic: there cache_read_input_tokens is exact and billable, so
    the MiniMax-specific "+128 floor / unreliable hit signal" note would
    be false and misleading in the logs.
    """
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=50,
        cache_creation_input_tokens=10,
    )

    with caplog.at_level("DEBUG", logger="agent.usage_pricing"):
        result = normalize_usage(
            usage,
            provider="anthropic",
            api_mode="anthropic_messages",
        )

    assert all("cache_observability" not in rec.message for rec in caplog.records)
    # Token normalization itself is unaffected.
    assert result.input_tokens == 100
    assert result.cache_read_tokens == 50
    assert result.cache_write_tokens == 10


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

    def test_large_cost_renders_2dp(self):
        assert format_cost_label(Decimal("42.50")) == "~$42.50"

    def test_very_small_sub_cent(self):
        """Even very small costs render non-zero."""
        label = format_cost_label(Decimal("0.0001"))
        assert label == "~$0.0001"
        assert label != "$0.00"

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

    def test_sub_cent_deepseek_scenario(self):
        """Reproduce the #79220 reproduction: DeepSeek at $0.004640."""
        # DeepSeek V4 Pro: 8K input + 1.2K output + 32K cache read
        # = $0.004640 per turn
        amount = Decimal("0.004640")
        label = format_cost_label(amount)
        assert "0.0046" in label
        assert label != "$0.00"
        assert label != "~$0.00"


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
        assert any("subscription" in note.lower() for note in result.notes)


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
