"""A cache miss re-sends the cached prefix at no less than the base input rate.

Some catalogs publish a positive ``cache_write_cost_per_million`` that is below the read rate (for
example a storage-style write price). Pricing the miss at that write rate collapses ``miss_cost`` to
zero, so ``evaluate_economics`` answers ``stop`` and the warmer never keeps the cache alive, even
though the miss it would prevent is billed as a full uncached resend at the input rate (#158687).
"""
from __future__ import annotations

from types import SimpleNamespace

from agent.prompt_cache_lifecycle.economics import evaluate_economics


def _pricing(*, inp: float, out: float, read: float, write: float) -> SimpleNamespace:
    return SimpleNamespace(input_cost_per_million=inp, output_cost_per_million=out,
                           cache_read_cost_per_million=read, cache_write_cost_per_million=write, request_cost=0.0)


def test_miss_is_priced_at_least_at_the_input_rate_when_the_write_rate_is_below_the_read_rate():
    decision = evaluate_economics(_pricing(inp=3.00, out=15.00, read=0.15, write=0.0833),
                                  cached_tokens=600_000, uncached_tokens=2_000)
    assert decision.action == "warm"
    assert decision.within_miss is True
    # The miss re-sends 600k tokens at $3.00/M ($1.80) instead of reading them at $0.15/M ($0.09).
    assert abs(decision.miss_cost - 1.71) < 1e-6
    assert decision.expected_savings > 1.00


def test_a_write_rate_above_the_input_rate_still_prices_the_miss():
    """Anthropic-style 1h writes at 2x input: the write rate governs, unchanged."""
    decision = evaluate_economics(_pricing(inp=5.00, out=25.00, read=0.50, write=10.00), cached_tokens=100_000)
    assert decision.action == "warm"
    assert abs(decision.miss_cost - 0.95) < 1e-6  # 100k * ($10.00 - $0.50) / 1M


def test_no_write_rate_falls_back_to_the_input_rate_unchanged():
    decision = evaluate_economics(_pricing(inp=2.00, out=8.00, read=0.20, write=0.0), cached_tokens=100_000)
    assert abs(decision.miss_cost - 0.18) < 1e-6  # 100k * ($2.00 - $0.20) / 1M
