"""Hermetic gap tests — freshness calculator (1 test).

Implements the freshness sub-section of the hermetic_test_gap_analysis.md:

* test_fr_calc_11_source_updated_at_in_future_returns_zero_staleness
"""

from __future__ import annotations


def test_fr_calc_11_source_updated_at_in_future_returns_zero_staleness(
    b1_engine_with_clock_skew,
):
    """Clock skew (source_updated_at > observed_at) → staleness_days == 0.

    The max(0, delta_days) clamp in _make_freshness prevents negative
    staleness when source clocks are ahead of observation time.
    """
    engine = b1_engine_with_clock_skew
    pack = engine.dry_run(
        objective_id="b1-fr-calc-11",
        objective_text="discovery future-dated",
    )
    assert pack.hits, "expected at least one hit"
    for h in pack.hits:
        assert h.freshness.staleness_days == 0, (
            f"staleness_days={h.freshness.staleness_days} for future-dated source"
        )
        # freshness should be 'current' with score 1.0
        assert h.freshness.freshness == "current"
        assert h.freshness.freshness_score == 1.0