"""Tests for GPT-5.4 / GPT-5.2 family pricing entries.

Fleet audit finding (2026-08): sessions on openai-api gpt-5.4 /
gpt-5.4-mini showed as unknown pricing in Hermes Insights because
_OFFICIAL_DOCS_PRICING had no entries for the 5.4 generation — the
table jumped from gpt-4.1 straight to gpt-5.6.
"""

from decimal import Decimal

from agent.usage_pricing import get_pricing_entry, has_known_pricing


def test_gpt54_family_entries_exist():
    """Regression: the 5.4 generation must have pricing entries.

    Before this fix, gpt-5.4 / gpt-5.4-mini sessions showed unknown cost
    in Hermes Insights. Rates from the per-model docs pages (snapshot
    2026-08).
    """
    expected = {
        "gpt-5.4": ("2.50", "0.25", "15.00"),
        "gpt-5.4-mini": ("0.75", "0.075", "4.50"),
        "gpt-5.4-nano": ("0.20", "0.02", "1.25"),
        "gpt-5.2": ("1.75", "0.175", "14.00"),
    }
    for model, (inp, cached, out) in expected.items():
        entry = get_pricing_entry(model, provider="openai-api")
        assert entry is not None, model
        assert entry.input_cost_per_million == Decimal(inp), model
        assert entry.output_cost_per_million == Decimal(out), model
        assert entry.cache_read_cost_per_million == Decimal(cached), model


def test_gpt54_family_has_known_pricing():
    for model in ("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.2"):
        assert has_known_pricing(model, "openai-api", "https://api.openai.com/v1"), model
