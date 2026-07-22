"""Hermetic gap tests — no-op / rollback / audit (2 tests).

Implements:

* test_no_op_05_rollback_after_partial_provider_failure
* test_audit_nomut_05_multi_conflict_appends_one_event_per_high
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────
# 2.11.1 — rollback chain: idempotent
# ─────────────────────────────────────────────────────────────────────


def test_no_op_05_rollback_after_partial_provider_failure(
    b1_engine_with_partial_provider_failure,
):
    """discover() with one failing provider + one OK → rollback() returns
    True (deleted 1 key), then False on a second call (idempotent).
    """
    engine = b1_engine_with_partial_provider_failure
    # discover() — one provider fails, one succeeds
    pack = engine.discover(
        objective_id="b1-noop-05",
        objective_text="discovery partial content",
    )
    # obsidian failed → in sources_failed
    assert "obsidian" in pack.sources_failed, pack.sources_failed
    # gbrain succeeded → in sources_queried
    assert "gbrain" in pack.sources_queried, pack.sources_queried
    # Hits from gbrain present
    assert len(pack.hits) >= 1

    # First rollback → True (deleted the state_meta key)
    assert engine.rollback("b1-noop-05") is True

    # Second rollback → False (already gone)
    assert engine.rollback("b1-noop-05") is False


# ─────────────────────────────────────────────────────────────────────
# 2.12.1 — audit event count: N high conflicts → N audit events
# ─────────────────────────────────────────────────────────────────────


def test_audit_nomut_05_multi_conflict_appends_one_event_per_high(
    b1_engine_with_three_policy_vs_obsidian_conflicts,
):
    """Pairwise detection with N policy + N obsidian hits → N×N high conflicts
    → exactly N×N audit events with gate_type='knowledge_conflict'
    severity='high'. The contract is "1 audit event per high conflict" —
    independent of whether the conflict came from pairwise enumeration.
    """
    engine = b1_engine_with_three_policy_vs_obsidian_conflicts
    pack = engine.dry_run(
        objective_id="b1-audit-nomut-05",
        objective_text="discovery multi conflict alpha beta",
    )
    # Verify all conflicts are high-severity policy_vs_goal
    high = [c for c in pack.conflicts if c.severity == "high"]
    assert all(c.conflict_type == "policy_vs_goal" for c in high), (
        f"unexpected conflict_type in high set: {[c.conflict_type for c in high]}"
    )
    # Pairwise detection: 3 policy × 3 obsidian = 9 conflicts
    assert len(high) == 9, (
        f"expected 9 high-severity conflicts (3×3 pairwise), got {len(high)}"
    )

    # Verify 9 audit events (1 per high conflict)
    events = engine._audit_sink.get_events()
    pvg_events = [
        e for e in events
        if e.get("gate_type") == "knowledge_conflict" and e.get("severity") == "high"
    ]
    assert len(pvg_events) == 9, (
        f"expected 9 audit events (1 per high conflict), got {len(pvg_events)}: {pvg_events}"
    )