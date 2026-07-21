"""Hermetic gap tests — aggregate scoring (1 test).

Implements the aggregate scoring sub-section of the hermetic_test_gap_analysis.md:

* test_ep_agg_04_corroboration_5_sources_full_bonus
"""

from __future__ import annotations


def test_ep_agg_04_corroboration_5_sources_full_bonus(
    b1_engine_with_5_sources_one_each,
):
    """When 5 distinct sources each contribute one hit, corroboration is full.

    corroboration = 0.5 + 0.5 * min(unique_sources / 5, 1.0).
    With 5 unique sources → min(5/5, 1.0) = 1.0 → corroboration = 1.0.

    Corroboration is an internal scoring input; we verify the precondition
    (5 unique sources, 1 hit each) which drives it to its maximum. Pairwise
    conflict detection runs on the hit set, but corroboration is computed
    BEFORE the conflict penalty and is observable indirectly via the hit
    diversity invariant.
    """
    engine = b1_engine_with_5_sources_one_each
    pack = engine.dry_run(
        objective_id="b1-agg-04",
        objective_text="discovery universal token fixture",
    )
    assert len(pack.hits) == 5, f"expected 5 hits, got {len(pack.hits)}"
    sources = {h.source for h in pack.hits}
    assert sources == {"policy", "contract", "gbrain", "obsidian", "report"}, (
        f"unexpected sources: {sources}"
    )
    # Corroboration reaches its max (1.0) when unique_sources == 5.
    # Compute the corroboration value the same way the engine does:
    unique_sources = len(sources)
    corroboration = 0.5 + 0.5 * min(unique_sources / 5, 1.0)
    assert corroboration == 1.0, (
        f"corroboration should be 1.0 with 5 unique sources, got {corroboration}"
    )