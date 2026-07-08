"""Hermetic gap tests — EvidencePack structure (4 tests).

Implements the structure sub-section of the hermetic_test_gap_analysis.md:

* test_ep_str_11_extra_keys_rejected_in_to_dict
* test_ep_str_12_objective_text_clamped_at_10000
* test_ep_str_13_query_with_zero_sources_requested
* test_ep_str_14_objective_id_with_unicode
"""

from __future__ import annotations

from tests.test_executive_v2.canary_b1.evidence_pack import FINGERPRINT_RE


# ─────────────────────────────────────────────────────────────────────
# 2.1.1 — schema contract: extra keys on to_dict()
# ─────────────────────────────────────────────────────────────────────


def test_ep_str_11_extra_keys_rejected_in_to_dict(hermetic_evidence_pack_engine):
    """to_dict() exposes ONLY the documented schema fields.

    Even if a caller attaches a new attribute to the EvidencePack instance
    after construction, to_dict() must not surface it.
    """
    engine, _ = hermetic_evidence_pack_engine
    pack = engine.dry_run(
        objective_id="b1-str-11",
        objective_text=f"something discovery",  # ensure token overlap
    )
    # Attach a new attribute after construction (caller has no guard, but
    # to_dict() is a contract that must NOT surface extra keys).
    pack.injected_garbage = "should not appear"  # type: ignore[attr-defined]

    d = pack.to_dict()
    assert "injected_garbage" not in d, (
        f"to_dict() surfaced extra key: {set(d.keys()) - _DOCUMENTED_KEYS}"
    )
    # Pre-defined schema keys (per evidence_pack_schema.md)
    assert set(d.keys()) == _DOCUMENTED_KEYS, (
        f"unexpected keys: {set(d.keys()) ^ _DOCUMENTED_KEYS}"
    )


# Documented schema keys for EvidencePack.to_dict()
_DOCUMENTED_KEYS = {
    "objective_id", "query_fingerprint", "sources_queried", "sources_failed",
    "hits", "citations", "conflicts", "missing_information",
    "overall_freshness_score", "overall_confidence",
    "summary_text", "summary_fingerprint", "duration_ms",
    "created_at", "schema_version",
}


# ─────────────────────────────────────────────────────────────────────
# 2.1.2 — boundary: objective_text clamped at 10000 chars
# ─────────────────────────────────────────────────────────────────────


def test_ep_str_12_objective_text_clamped_at_10000(
    hermetic_evidence_pack_engine,
):
    """objective_text > 10000 chars is clamped; missing_information records it."""
    engine, _ = hermetic_evidence_pack_engine
    huge = ("x" * 10_001) + " discovery token"  # > 10_000 chars total
    pack = engine.dry_run(
        objective_id="b1-str-12",
        objective_text=huge,
    )
    assert any(
        "objective_text clamped to 10000 chars" in m
        for m in pack.missing_information
    ), f"missing_information: {pack.missing_information!r}"
    assert pack.sources_queried, "expected at least one source queried"


# ─────────────────────────────────────────────────────────────────────
# 2.1.3 — edge: zero sources_requested
# ─────────────────────────────────────────────────────────────────────


def test_ep_str_13_query_with_zero_sources_requested(
    hermetic_evidence_pack_engine,
):
    """Zero usable sources yields an empty pack.

    The engine ignores unknown source names (filtered by ALLOWED_SOURCES),
    so passing a non-allowed source has the same observable effect as zero
    sources — an empty pack with no hits/citations/freshness.
    """
    engine, _ = hermetic_evidence_pack_engine
    pack = engine.dry_run(
        objective_id="b1-str-13",
        objective_text="anything",
        # Not in ALLOWED_SOURCES → filtered out → zero effective sources
        sources_requested=("nonexistent_source",),
    )
    assert pack.sources_queried == []
    assert pack.hits == []
    assert pack.citations == []
    assert pack.overall_freshness_score == 0.0
    assert pack.overall_confidence == 0.0


# ─────────────────────────────────────────────────────────────────────
# 2.1.4 — unicode objective_id round-trip
# ─────────────────────────────────────────────────────────────────────


def test_ep_str_14_objective_id_with_unicode(hermetic_evidence_pack_engine):
    """objective_id with multi-byte unicode is preserved; fingerprints are 64 hex."""
    engine, _ = hermetic_evidence_pack_engine
    obj_id = "obj-ñoño-🚀"
    pack = engine.dry_run(
        objective_id=obj_id,
        objective_text="anything",
    )
    assert pack.objective_id == obj_id
    assert FINGERPRINT_RE.fullmatch(pack.query_fingerprint) is not None
    assert FINGERPRINT_RE.fullmatch(pack.summary_fingerprint) is not None