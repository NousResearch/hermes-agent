"""Tests for the MiniMax M3.1 Flash list-price pin in agent/usage_pricing.py.

MiniMax published M3.1 Flash on 2026-10-19 at a flat rate ($0.15 / $0.60 / $0.015 per
million tokens) with no long-prompt step-up and no priority tier. The ``minimax``
provider row in the bundled snapshot table only carried M2.7, so a session on M3.1
Flash priced at ``unknown`` and reported no cost at all — the model was invisible to
the cost display even once it was selectable.

These tests pin the published rates and guard the deliberate scope choices:
  - the entry is a standalone one (its own pricing version and source URL), so the
    2026-04 snapshot that still carries M2.7 keeps its own version;
  - the rate is flat — no whole-request tier above the prompt threshold;
  - the ``minimax-cn`` row is untouched (the CN price sheet is published in CNY, and
    Hermes has no documented CNY->USD convention for it).
"""

from decimal import Decimal

from agent.usage_pricing import (
    _OFFICIAL_DOCS_PRICING,
    CanonicalUsage,
    estimate_usage_cost,
    get_pricing_entry,
)

MODEL = "MiniMax-M3.1-Flash"
BASE_URL = "https://api.minimax.io/anthropic"


def test_m31_flash_has_a_bundled_pricing_entry():
    entry = get_pricing_entry(MODEL, provider="minimax", base_url=BASE_URL)

    assert entry is not None, "M3.1 Flash must resolve to a bundled price, not unknown"
    assert entry.input_cost_per_million == Decimal("0.15")
    assert entry.output_cost_per_million == Decimal("0.60")
    assert entry.cache_read_cost_per_million == Decimal("0.015")
    assert entry.source == "official_docs_snapshot"
    assert entry.pricing_version == "minimax-pricing-2026-10"
    assert entry.source_url == "https://platform.minimax.io/docs/pricing/overview"


def test_m31_flash_cost_is_estimated_not_unknown():
    usage = CanonicalUsage(input_tokens=19_070, output_tokens=10, cache_read_tokens=128)

    result = estimate_usage_cost(MODEL, usage, provider="minimax", base_url=BASE_URL)

    assert result.status == "estimated"
    assert result.source == "official_docs_snapshot"
    assert result.amount_usd == (
        Decimal(19_070) * Decimal("0.15")
        + Decimal(10) * Decimal("0.60")
        + Decimal(128) * Decimal("0.015")
    ) / Decimal(1_000_000)


def test_m31_flash_rate_is_flat_with_no_long_prompt_tier():
    """M3 charges more above 512k input; M3.1 Flash is a single flat rate."""
    entry = get_pricing_entry(MODEL, provider="minimax", base_url=BASE_URL)
    assert entry is not None

    assert entry.tier_threshold_tokens is None
    assert entry.input_cost_per_million_above is None
    assert entry.output_cost_per_million_above is None

    below = estimate_usage_cost(
        MODEL, CanonicalUsage(input_tokens=100_000, output_tokens=1_000), provider="minimax"
    )
    above = estimate_usage_cost(
        MODEL, CanonicalUsage(input_tokens=900_000, output_tokens=1_000), provider="minimax"
    )
    assert below.amount_usd is not None and above.amount_usd is not None
    # Linear in tokens: the 9x input does not re-price the request.
    assert above.amount_usd > below.amount_usd
    assert (above.amount_usd - below.amount_usd) == (
        Decimal(800_000) * Decimal("0.15") / Decimal(1_000_000)
    )


def test_m2x_row_keeps_its_own_snapshot_version():
    """The new entry must not restamp the M2.7 row's April price sheet."""
    entry = get_pricing_entry("MiniMax-M2.7", provider="minimax", base_url=BASE_URL)

    assert entry is not None
    assert entry.pricing_version == "minimax-pricing-2026-04"
    assert entry.input_cost_per_million == Decimal("0.30")
    assert entry.output_cost_per_million == Decimal("1.20")


def test_minimax_cn_row_is_untouched():
    """The CN price sheet is published in CNY; this pin covers the global route only."""
    assert ("minimax-cn", "minimax-m3.1-flash") not in _OFFICIAL_DOCS_PRICING

    entry = get_pricing_entry("MiniMax-M2.7", provider="minimax-cn", base_url="https://api.minimaxi.com/anthropic")
    assert entry is not None
    assert entry.pricing_version == "minimax-pricing-2026-04"


def test_entry_is_keyed_by_the_lowercased_model_id():
    assert ("minimax", "minimax-m3.1-flash") in _OFFICIAL_DOCS_PRICING
