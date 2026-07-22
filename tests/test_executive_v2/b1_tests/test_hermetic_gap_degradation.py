"""Hermetic gap tests — degradation flags (2 tests).

Implements the degradation sub-section of the hermetic_test_gap_analysis.md:

* test_fr_deg_05_multi_flag_precedence_human_beats_degraded
* test_fr_deg_06_ready_with_caveats_when_only_medium_conflicts
"""

from __future__ import annotations

from tests.test_executive_v2.canary_b1.evidence_pack import (
    EvidencePackEngine,
    _make_hit_v2,
    SOURCE_TTL_DAYS,
)
from tests.test_executive_v2.canary_b1.conftest import _InMemoryStorage


OBS = "2026-07-08T20:00:00+00:00"
UPD = "2026-07-08T20:00:00+00:00"


# ─────────────────────────────────────────────────────────────────────
# 2.10.1 — precedence: high-severity conflict beats degraded freshness
# ─────────────────────────────────────────────────────────────────────


def test_fr_deg_05_multi_flag_precedence_human_beats_degraded():
    """When both high conflict AND degraded freshness are present,
    summary_text starts with [REQUIRES_HUMAN] (precedence: human > degraded).

    Source: evidence_pack.py _build_summary — high_conflicts branch fires
    before overall_freshness check.

    Snippets must differ token-wise to avoid near-dup dedup collapsing them.
    """
    def _policy(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source="policy",
            hit_id="b1-deg-05-policy",
            title="deg policy",
            relevance_score=0.9,
            snippet="deg alpha policy content unique",
            source_uri="state_meta[objective_policy_decision:b1-deg-05-policy]",
            source_updated_at=UPD,
            retrieval_mode="metadata_only",
            observed_at=OBS,
            ttl_days=SOURCE_TTL_DAYS["policy"],
        )]

    def _obsidian(query, *, max_hits: int = 5, observed_at: str):
        # Old source_updated_at to force degraded freshness
        return [_make_hit_v2(
            source="obsidian",
            hit_id="b1-deg-05-obsidian",
            title="deg obsidian",
            relevance_score=0.9,
            snippet="deg beta obsidian content unique",
            source_uri="file://obsidian/b1-deg-05-obsidian",
            source_updated_at="2026-01-01T00:00:00+00:00",  # very stale
            retrieval_mode="snippet",
            observed_at=OBS,
            ttl_days=SOURCE_TTL_DAYS["obsidian"],
        )]

    bundle = {"policy": _policy, "obsidian": _obsidian}
    engine = EvidencePackEngine(
        sources=bundle, storage=_InMemoryStorage(), audit_sink=None,
    )
    pack = engine.dry_run(
        objective_id="b1-deg-05",
        objective_text="deg content",
    )
    # policy_vs_obsidian → high severity → [REQUIRES_HUMAN] wins precedence
    high = [c for c in pack.conflicts if c.severity == "high"]
    assert high, f"expected high-severity conflict, got {pack.conflicts}"
    assert pack.summary_text.startswith("[REQUIRES_HUMAN]"), (
        f"expected [REQUIRES_HUMAN] prefix, got: {pack.summary_text[:60]}"
    )


# ─────────────────────────────────────────────────────────────────────
# 2.10.2 — edge case of happy path: only medium conflicts → READY_WITH_CAVEATS
# ─────────────────────────────────────────────────────────────────────


def test_fr_deg_06_ready_with_caveats_when_only_medium_conflicts():
    """medium-only conflicts + freshness ≥ 0.5 + confidence ≥ 0.4 →
    summary starts with [READY_WITH_CAVEATS].
    """
    def _gbrain(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source="gbrain",
            hit_id="b1-deg-06-gbrain",
            title="deg gbrain",
            relevance_score=0.9,
            snippet="deg alpha gbrain content unique",
            source_uri="gbrain://b1-deg-06-gbrain",
            source_updated_at=UPD,
            retrieval_mode="semantic_search",
            observed_at=OBS,
            ttl_days=SOURCE_TTL_DAYS["gbrain"],
        )]

    def _obsidian(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source="obsidian",
            hit_id="b1-deg-06-obsidian",
            title="deg obsidian",
            relevance_score=0.9,
            snippet="deg beta obsidian content unique",
            source_uri="file://obsidian/b1-deg-06-obsidian",
            source_updated_at=UPD,
            retrieval_mode="snippet",
            observed_at=OBS,
            ttl_days=SOURCE_TTL_DAYS["obsidian"],
        )]

    # {gbrain, obsidian} → memory_vs_evidence (medium)
    bundle = {"gbrain": _gbrain, "obsidian": _obsidian}
    engine = EvidencePackEngine(
        sources=bundle, storage=_InMemoryStorage(), audit_sink=None,
    )
    pack = engine.dry_run(
        objective_id="b1-deg-06",
        objective_text="deg content",
    )
    # Confirm we got medium-only conflicts (not high)
    assert pack.conflicts
    assert all(c.severity in {"low", "medium"} for c in pack.conflicts), (
        f"expected low/medium only, got severities: {[c.severity for c in pack.conflicts]}"
    )
    assert pack.overall_freshness_score >= 0.5
    assert pack.overall_confidence >= 0.4
    assert pack.summary_text.startswith("[READY_WITH_CAVEATS]"), (
        f"expected [READY_WITH_CAVEATS] prefix, got: {pack.summary_text[:60]}"
    )