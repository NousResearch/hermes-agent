"""Tests for snapshot-id date-suffix stripping in pricing lookups.

Fleet audit finding (2026-08): vendor snapshot ids — gpt-5.6-sol-2026-07-09,
gpt-5.4-2026-03-05, claude-opus-4-8-20260801 — resolved to no pricing entry
because _OFFICIAL_DOCS_PRICING is keyed on base names only. Sessions pinned
to a snapshot showed unknown cost in Hermes Insights even though snapshots
bill at the base model's rates.
"""

from agent.usage_pricing import get_pricing_entry


# (provider, base, snapshot) — snapshot must resolve to the base's rates.
SNAPSHOT_MATRIX = [
    ("openai", "gpt-5.6-sol", "gpt-5.6-sol-2026-07-09"),       # ISO dashed
    ("openai", "gpt-5.6-luna", "gpt-5.6-luna-2026-07-09"),     # ISO dashed
    ("openai", "o3", "o3-2025-04-16"),                          # ISO dashed
    ("openai", "gpt-4.1", "gpt-4.1-2025-04-14"),               # ISO dashed
    ("openai", "gpt-4o", "gpt-4o-2024-11-20"),                 # ISO dashed
    ("anthropic", "claude-opus-4-8", "claude-opus-4-8-20260801"),  # compact
]


def test_snapshot_ids_bill_at_base_rates():
    """Invariant: a dated snapshot of a base model prices identically to the
    base — only the weights are frozen, never the price."""
    for provider, base, snapshot in SNAPSHOT_MATRIX:
        base_entry = get_pricing_entry(base, provider=provider)
        snap_entry = get_pricing_entry(snapshot, provider=provider)
        assert snap_entry is not None, snapshot
        assert base_entry is not None, base
        assert snap_entry.input_cost_per_million == base_entry.input_cost_per_million, snapshot
        assert snap_entry.output_cost_per_million == base_entry.output_cost_per_million, snapshot
        assert snap_entry.cache_read_cost_per_million == base_entry.cache_read_cost_per_million, snapshot


def test_dated_table_keys_still_direct_hit():
    """The table carries some dated keys natively (claude-opus-4-7-20250507).
    Direct lookup must keep winning — stripping is a fallback, not a
    replacement."""
    entry = get_pricing_entry("claude-opus-4-7-20250507", provider="anthropic")
    assert entry is not None
    assert entry.input_cost_per_million is not None


def test_unknown_model_stays_unknown():
    """Negative control: an 8-digit suffix must not make an unknown model
    resolve. Stripping only retries; it never invents pricing."""
    assert get_pricing_entry("no-such-model-12345678", provider="openai") is None
    assert get_pricing_entry("no-such-model-2026-01-01", provider="openai") is None
