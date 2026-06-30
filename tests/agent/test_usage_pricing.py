from types import SimpleNamespace

from agent.usage_pricing import (
    CanonicalUsage,
    NOTIONAL_ANTHROPIC_PROVIDERS,
    estimate_usage_cost,
    get_pricing_entry,
    has_known_pricing,
    is_notional_anthropic_provider,
    normalize_usage,
    resolve_billing_route,
)


# Representative notional-Anthropic provider keys: the exact base members PLUS a
# spread of the -fN failover family (matched by pattern, not membership). Tests
# that assert "every notional provider prices correctly" iterate THIS, not the
# frozenset — because the frozenset deliberately holds only the base names now.
_REPRESENTATIVE_NOTIONAL = [
    "claude-api-proxy",
    "claude-bridge",
    "claude-pool",
    "claude-app",
    "claude-api-proxy-f1",
    "claude-api-proxy-f2",
    "claude-api-proxy-f5",
    "claude-bridge-f1",
    "claude-bridge-f3",
    "claude-bridge-f5",
]


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


def test_notional_anthropic_includes_fn_failover_family_by_pattern():
    """Regression (the recurring f-lane alias gap): every -fN failover lane —
    claude-api-proxy-fN / claude-bridge-fN for ANY integer N — must price as
    notional-Anthropic, matched by PATTERN (not frozenset membership) so a new
    lane never needs a code edit. Missing aliases caused -f2 (audit 2026-06-13,
    ~25% of Opus spend) and then claude-pool/-f3/-f4/-f5 (audit 2026-06-17,
    ~$872 over 2 days) to price 'unknown'/$0, blinding /cost."""
    for provider in (
        "claude-api-proxy-f1", "claude-api-proxy-f2", "claude-api-proxy-f3",
        "claude-api-proxy-f4", "claude-api-proxy-f5",
        "claude-bridge-f1", "claude-bridge-f2", "claude-bridge-f3",
        "claude-bridge-f4", "claude-bridge-f5", "claude-pool",
    ):
        assert is_notional_anthropic_provider(provider), provider
        result = estimate_usage_cost(
            "claude-opus-4-8",
            CanonicalUsage(input_tokens=1000, output_tokens=100,
                           cache_read_tokens=5000, cache_write_tokens=200),
            provider=provider,
        )
        assert result.status == "estimated", f"{provider}: {result.status}"
        assert result.amount_usd is not None and float(result.amount_usd) > 0


def test_notional_anthropic_pattern_is_open_ended_on_n():
    """The -fN pattern must accept ANY integer N (today -f1..-f5, tomorrow -f6+),
    proving a future failover lane is covered with zero code change."""
    for provider in (
        "claude-bridge-f6", "claude-bridge-f9", "claude-bridge-f12",
        "claude-api-proxy-f7", "claude-api-proxy-f10", "claude-api-proxy-f99",
    ):
        assert is_notional_anthropic_provider(provider), provider
        route = resolve_billing_route("claude-opus-4-8", provider=provider)
        assert route.billing_mode == "official_docs_snapshot", provider


def test_notional_anthropic_pattern_rejects_non_failover():
    """The pattern must be ANCHORED + integer-only — it must NOT match a lookalike
    that isn't a real failover lane (else we'd misprice an unrelated provider as
    Anthropic). Guards against a greedy/unanchored regex."""
    for provider in (
        "claude-bridge-frobnicate", "claude-api-proxy-foo", "claude-bridge-f",
        "claude-bridgef3", "xclaude-bridge-f3", "claude-bridge-f3x",
        "claude-bridge-f3-extra", "claude-pool-f3", "claude-poolx",
        "openai-codex", "anthropic", "openrouter", "", "   ",
    ):
        assert not is_notional_anthropic_provider(provider), repr(provider)
    # None must be safe too (Optional param)
    assert not is_notional_anthropic_provider(None)


def test_notional_anthropic_providers_price_at_official_rates():
    """The notional Claude subscription proxies/bridges/pool must price
    claude-opus-4-8 at official Anthropic rates ($5/$25 per M, $0.50 cache-read,
    $6.25 cache-write) and label the result 'estimated' — NOT 'unknown' (which
    suppresses /cost cards) and NOT 'included'/$0 (which hides spend in
    rollups). Iterates the representative set incl. -fN pattern members."""
    usage = CanonicalUsage(
        input_tokens=500_000, output_tokens=559,
        cache_read_tokens=499_000, cache_write_tokens=1000,
    )
    # 500000*5/1e6 + 559*25/1e6 + 499000*0.5/1e6 + 1000*6.25/1e6 = 2.769725
    expected = 2.769725
    for provider in _REPRESENTATIVE_NOTIONAL:
        result = estimate_usage_cost("claude-opus-4-8", usage, provider=provider)
        assert result.status == "estimated", f"{provider}: {result.status}"
        assert result.amount_usd is not None, f"{provider} priced None"
        assert float(result.amount_usd) == round(expected, 6), (
            f"{provider}: {result.amount_usd}"
        )


def test_notional_anthropic_route_resolves_to_anthropic_billing():
    """resolve_billing_route must rewrite the proxy provider to 'anthropic' with
    the docs-snapshot billing mode so all downstream pricing lookups work."""
    for provider in _REPRESENTATIVE_NOTIONAL:
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
    """has_known_pricing must return True for the notional providers (used by
    callers to decide whether to attempt cost display) — incl. -fN pattern
    members and claude-pool, since it routes through resolve_billing_route."""
    for provider in _REPRESENTATIVE_NOTIONAL:
        assert has_known_pricing("claude-opus-4-8", provider=provider), provider


_CODEX_STUB_METADATA = {
    "gpt-5.5": {"pricing": {"prompt": "0.000005", "completion": "0.00003"}},
    "gpt-5.4": {"pricing": {"prompt": "0.0000025", "completion": "0.000015"}},
    "gpt-5.3-codex": {"pricing": {"prompt": "0.00000175", "completion": "0.000014"}},
}


def test_anthropic_dated_release_suffix_falls_back_to_base_snapshot():
    """A new-scheme Anthropic id with a trailing -YYYYMMDD release date that has
    no explicit snapshot entry must fall back to its base model's pricing —
    NOT price 'unknown'. Fixes the dated-Haiku gap (claude-haiku-4-5-20251001
    priced $0 while bare claude-haiku-4-5 priced fine; audit 2026-06-17)."""
    usage = CanonicalUsage(input_tokens=1000, output_tokens=100,
                           cache_read_tokens=5000, cache_write_tokens=200)
    base = estimate_usage_cost("claude-haiku-4-5", usage, provider="claude-pool")
    dated = estimate_usage_cost("claude-haiku-4-5-20251001", usage, provider="claude-pool")
    assert base.status == "estimated"
    assert dated.status == "estimated", f"dated priced {dated.status}"
    assert dated.amount_usd is not None and dated.amount_usd == base.amount_usd, (
        f"dated {dated.amount_usd} != base {base.amount_usd}"
    )
    # Also covers a hypothetical future dated Opus id.
    opus = estimate_usage_cost("claude-opus-4-8-20260115", usage, provider="claude-bridge")
    assert opus.status == "estimated" and opus.amount_usd is not None


def test_anthropic_old_scheme_dated_model_is_not_date_stripped():
    """No-regression guard: an OLD-scheme id whose date is PART of the canonical
    name (claude-3-5-haiku-20241022) must hit its OWN snapshot entry directly,
    never get date-stripped to a non-existent base (claude-3-5-haiku). It has
    its own distinct rate, so stripping would mis-price it."""
    from agent.usage_pricing import _strip_anthropic_release_date
    # The strip helper must DECLINE the old-scheme name (no -N-N version tail).
    assert _strip_anthropic_release_date("claude-3-5-haiku-20241022") is None
    assert _strip_anthropic_release_date("claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert _strip_anthropic_release_date("claude-opus-4-8-20260115") == "claude-opus-4-8"
    # Non-dated / partial-date names are left alone.
    for n in ("claude-haiku-4-5", "claude-opus-4-8", "claude-haiku-4-5-2025", "gpt-5.5"):
        assert _strip_anthropic_release_date(n) is None, n
    # End-to-end: the old-scheme dated Haiku still prices via its own entry,
    # and its rate is DISTINCT from the new 4-5 Haiku (proves no cross-contamination).
    usage = CanonicalUsage(input_tokens=1_000_000, output_tokens=0)
    old = estimate_usage_cost("claude-3-5-haiku-20241022", usage, provider="anthropic")
    new = estimate_usage_cost("claude-haiku-4-5", usage, provider="anthropic")
    assert old.status == "estimated" and new.status == "estimated"
    assert old.amount_usd != new.amount_usd, "old-scheme must keep its own rate"


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


def test_estimate_usage_cost_prices_cache_write_at_input_rate_when_no_cache_write_rate(monkeypatch):
    """OpenAI-family routes (e.g. gpt-5.x via codex/openrouter) publish no
    separate cache-write rate — cache-write tokens are billed at the input
    rate. The estimator must price the turn (status 'estimated'), NOT drop it
    as unpriced and silently lose real spend. Regression for the lone
    remaining 'unpriced' turn on the tokens.ace by-profile chart (2026-06)."""
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata",
        lambda: {
            "gpt-5.5": {
                "pricing": {
                    "prompt": "0.000005",       # $5/M input
                    "completion": "0.00003",    # $30/M output
                    "input_cache_read": "0.0000005",  # $0.50/M cache-read
                    # NO cache-write / input_cache_write key — the real-world gap
                }
            }
        },
    )

    result = estimate_usage_cost(
        "gpt-5.5",
        CanonicalUsage(
            input_tokens=6,
            output_tokens=3104,
            cache_read_tokens=206366,
            cache_write_tokens=114435,
        ),
        provider="openai-codex",
    )

    assert result.status == "estimated"
    assert result.amount_usd is not None
    # cache-write billed at the $5/M input rate, not dropped.
    expected = (6 * 5 + 3104 * 30 + 206366 * 0.5 + 114435 * 5) / 1_000_000
    assert abs(float(result.amount_usd) - expected) < 1e-4
    assert any("input rate" in n for n in result.notes)


def test_estimate_usage_cost_still_unknown_when_input_and_cache_write_both_missing(monkeypatch):
    """If BOTH the input rate AND the cache-write rate are missing, a turn with
    cache-write tokens is genuinely unpriceable — still bail to unknown (the
    input-rate fallback must not paper over a truly missing price)."""
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata",
        lambda: {
            "mystery-model": {
                "pricing": {
                    "completion": "0.00001",  # only output priced
                }
            }
        },
    )

    result = estimate_usage_cost(
        "mystery-model",
        CanonicalUsage(input_tokens=0, output_tokens=10, cache_write_tokens=5000),
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


# ---------------------------------------------------------------------------
# SPEC-C — per-class cost breakdown on CostResult (tokens.ace priciest hover).
# The engine already computes each class term then sums; these assert it now
# ALSO exposes the four parts, that they reconcile to amount_usd, and that the
# OpenAI-family cache-write-at-input-rate path attributes to cache_write.
# ---------------------------------------------------------------------------
def test_cost_result_exposes_per_class_breakdown_summing_to_total():
    usage = CanonicalUsage(
        input_tokens=0,
        output_tokens=130_000,
        cache_read_tokens=109_700_000,
        cache_write_tokens=646_000,
    )
    r = estimate_usage_cost("claude-opus-4-8", usage, provider="claude-api-proxy")
    assert r.status == "estimated"
    # the four new per-class fields exist and are non-None on a priced turn
    assert r.cost_input_usd is not None
    assert r.cost_output_usd is not None
    assert r.cost_cache_read_usd is not None
    assert r.cost_cache_write_usd is not None
    # they reconcile EXACTLY to the total (Decimal arithmetic, same terms)
    parts = (r.cost_input_usd + r.cost_output_usd
             + r.cost_cache_read_usd + r.cost_cache_write_usd)
    assert parts == r.amount_usd
    # and each equals tokens × the published rate
    from agent.usage_pricing import get_pricing_entry
    e = get_pricing_entry("claude-opus-4-8", provider="claude-api-proxy")
    assert r.cost_cache_read_usd == (
        __import__("decimal").Decimal(109_700_000) * e.cache_read_cost_per_million
        / __import__("decimal").Decimal(1_000_000))


def test_per_class_unknown_route_leaves_all_four_none():
    usage = CanonicalUsage(input_tokens=1000, output_tokens=500)
    r = estimate_usage_cost("totally-unknown-model-xyz", usage, provider="mystery")
    assert r.status == "unknown"
    assert r.amount_usd is None
    assert r.cost_input_usd is None
    assert r.cost_output_usd is None
    assert r.cost_cache_read_usd is None
    assert r.cost_cache_write_usd is None


def test_per_class_pricing_is_linear_under_aggregation():
    # INV-7: price(A) + price(B) == price(A+B). The backfill (one aggregate
    # call) reproduces the live per-call sum ONLY if this holds.
    A = CanonicalUsage(input_tokens=1000, output_tokens=500,
                       cache_read_tokens=2_000_000, cache_write_tokens=10_000)
    B = CanonicalUsage(input_tokens=3000, output_tokens=20_000,
                       cache_read_tokens=500_000, cache_write_tokens=0)
    AB = CanonicalUsage(input_tokens=4000, output_tokens=20_500,
                        cache_read_tokens=2_500_000, cache_write_tokens=10_000)
    ra = estimate_usage_cost("claude-opus-4-8", A, provider="claude-api-proxy")
    rb = estimate_usage_cost("claude-opus-4-8", B, provider="claude-api-proxy")
    rab = estimate_usage_cost("claude-opus-4-8", AB, provider="claude-api-proxy")
    assert ra.amount_usd + rb.amount_usd == rab.amount_usd
    # per-class is linear too
    assert ra.cost_cache_read_usd + rb.cost_cache_read_usd == rab.cost_cache_read_usd
    assert ra.cost_output_usd + rb.cost_output_usd == rab.cost_output_usd


def test_cache_write_at_input_rate_attributes_to_cache_write_not_uncached():
    # INV-6: an OpenAI-family route with no separate cache-write rate bills
    # cache-write at the INPUT rate — that $ belongs to cost_cache_write_usd,
    # and cost_input_usd reflects only the real input_tokens.
    from agent.usage_pricing import get_pricing_entry
    e = get_pricing_entry("gpt-5.5", provider="openrouter")
    if e is None or e.input_cost_per_million is None:
        import pytest
        pytest.skip("gpt-5.5 openrouter pricing unavailable in this env")
    # only proceed if this route truly has no separate cache-write rate
    if e.cache_write_cost_per_million is not None:
        import pytest
        pytest.skip("route publishes a separate cache-write rate; INV-6 path N/A")
    usage = CanonicalUsage(input_tokens=1000, output_tokens=0,
                           cache_read_tokens=0, cache_write_tokens=2000)
    r = estimate_usage_cost("gpt-5.5", usage, provider="openrouter")
    from decimal import Decimal
    expect_cw = Decimal(2000) * e.input_cost_per_million / Decimal(1_000_000)
    expect_in = Decimal(1000) * e.input_cost_per_million / Decimal(1_000_000)
    assert r.cost_cache_write_usd == expect_cw
    assert r.cost_input_usd == expect_in
    # the cache-write $ must NOT be folded into uncached/input
    assert r.cost_input_usd != r.cost_cache_write_usd or 1000 == 2000


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
