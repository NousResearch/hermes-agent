"""Hermetic gap tests — citations (3 tests).

Implements the citations sub-section of the hermetic_test_gap_analysis.md:

* test_ep_cit_07_no_citations_when_no_hits
* test_ep_cit_08_citation_id_unique_per_citation
* test_ep_cit_09_citation_fingerprint_changes_with_statement
"""

from __future__ import annotations

from tests.test_executive_v2.canary_b1.evidence_pack import _build_citations
from tests.test_executive_v2.canary_b1.fake_providers import (
    empty_spec,
    make_provider_bundle,
)


# ─────────────────────────────────────────────────────────────────────
# 2.3.1 — edge: no hits → no citations
# ─────────────────────────────────────────────────────────────────────


def test_ep_cit_07_no_citations_when_no_hits(hermetic_evidence_pack_engine):
    """When no hits match, citations list is empty."""
    # Build an engine whose ALL sources are empty
    from tests.test_executive_v2.canary_b1.evidence_pack import EvidencePackEngine
    from tests.test_executive_v2.canary_b1.conftest import _InMemoryStorage
    from tests.test_executive_v2.canary_b1 import evidence_pack as _ep

    storage = _InMemoryStorage()

    bundle = make_provider_bundle(
        gbrain_spec=empty_spec("gbrain"),
        obsidian_spec=empty_spec("obsidian"),
        report_spec=empty_spec("report"),
        policy_spec=empty_spec("policy"),
        contract_spec=empty_spec("contract"),
    )
    engine = EvidencePackEngine(sources=bundle, storage=storage, audit_sink=None)
    pack = engine.dry_run(
        objective_id="b1-cit-07",
        objective_text="anything",
    )
    assert pack.hits == []
    assert pack.citations == []


# ─────────────────────────────────────────────────────────────────────
# 2.3.2 — invariant: citation_id is unique per citation
# ─────────────────────────────────────────────────────────────────────


def test_ep_cit_08_citation_id_unique_per_citation(hermetic_evidence_pack_engine):
    """Every citation in a pack has a unique citation_id."""
    engine, _ = hermetic_evidence_pack_engine
    pack = engine.dry_run(
        objective_id="b1-cit-08",
        objective_text="discovery policy contract report notes",
    )
    assert pack.citations, "expected citations"
    ids = [c.citation_id for c in pack.citations]
    assert len(set(ids)) == len(ids), f"duplicate citation_id: {ids}"


# ─────────────────────────────────────────────────────────────────────
# 2.3.3 — regression: citation fingerprint changes with statement
# ─────────────────────────────────────────────────────────────────────


def test_ep_cit_09_citation_fingerprint_changes_with_statement():
    """Two citations with different statements → different fingerprints."""
    obs = "2026-07-08T20:00:00+00:00"

    def _hit_for_statement(stmt: str):
        from tests.test_executive_v2.canary_b1.evidence_pack import _make_hit_v2, SOURCE_TTL_DAYS
        return _make_hit_v2(
            source="policy",
            hit_id="b1-cit-09",
            title="b1 cit-09",
            relevance_score=0.9,
            snippet=stmt,
            source_uri="state_meta[objective_policy_decision:b1-cit-09]",
            source_updated_at="2026-07-08T20:00:00+00:00",
            retrieval_mode="metadata_only",
            observed_at=obs,
            ttl_days=SOURCE_TTL_DAYS["policy"],
        )

    h1 = _hit_for_statement("statement version alpha")
    h2 = _hit_for_statement("statement version beta")

    # Citations built from a single hit (top_n=1)
    cits_v1 = _build_citations([h1], top_n=1, observed_at=obs)
    cits_v2 = _build_citations([h2], top_n=1, observed_at=obs)
    assert len(cits_v1) == 1
    assert len(cits_v2) == 1
    assert cits_v1[0].fingerprint != cits_v2[0].fingerprint, (
        "fingerprint must change when statement changes"
    )