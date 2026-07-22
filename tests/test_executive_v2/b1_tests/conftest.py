"""Hermetic fixtures for the B1 Knowledge Discovery additional test suite.

Reuses fixtures from ``canary_b1.conftest`` (frozen_time, in_memory_storage,
audit_capture, fake_*_spec, provider_bundle, hermetic_evidence_pack_engine).

Adds b1-specific fixtures called for in the design:

* ``b1_engine_with_5_sources_one_each`` — bundle with exactly one hit per
  allowed source (policy, contract, report, gbrain, obsidian). Used for the
  corroboration full-bonus test.
* ``b1_engine_with_clock_skew`` — frozen time + source_updated_at in the
  future. Used for the staleness clamp test.
* ``b1_engine_with_policy_vs_obsidian_high`` — policy hit vs obsidian hit
  designed to produce a ``policy_vs_goal`` conflict with severity ``high``.
* ``b1_engine_with_unicode_objective`` — engine pre-configured for unicode
  objective_id / objective_text tests.

All fixtures are deterministic and in-memory.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from tests.test_executive_v2.canary_b1.conftest import (  # noqa: F401  (re-export)
    CANARY_FROZEN_TIME_UTC,
    audit_capture,
    default_off_flags,
    fake_contract_spec,
    fake_gbrain_spec,
    fake_obsidian_spec,
    fake_policy_spec,
    fake_reports_spec,
    frozen_time,
    hermetic_evidence_pack_engine,
    in_memory_storage,
    provider_bundle,
    self_improvement_disabled,
)
from tests.test_executive_v2.canary_b1.evidence_pack import (
    EvidencePackEngine,
    KnowledgeHitV2,
    _make_hit_v2,
    SOURCE_TTL_DAYS,
)
from tests.test_executive_v2.canary_b1.fake_providers import (
    FakeProviderSpec,
    default_gbrain_spec,
    default_obsidian_spec,
    default_reports_spec,
    empty_spec,
    failing_spec,
    make_provider_bundle,
)


# ─────────────────────────────────────────────────────────────────────
# Reusable token for query overlap (engines need tokens ≥ 3 chars to match)
# ─────────────────────────────────────────────────────────────────────

_UNIVERSAL_TOKEN = "discovery"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _hit(
    source: str,
    hit_id: str,
    title: str,
    snippet: str,
    relevance: float,
    *,
    observed_at: str,
    source_updated_at: Optional[str] = None,
    retrieval_mode: str = "metadata_only",
    source_uri: Optional[str] = None,
    quote: Optional[str] = None,
    line_range: Optional[str] = None,
    hash_sha256: Optional[str] = None,
) -> KnowledgeHitV2:
    """Build a KnowledgeHitV2 inline for fixtures."""
    return _make_hit_v2(
        source=source,
        hit_id=hit_id,
        title=title,
        relevance_score=relevance,
        snippet=snippet,
        source_uri=source_uri or f"{source}://{hit_id}",
        source_updated_at=source_updated_at,
        retrieval_mode=retrieval_mode,
        quote=quote,
        line_range=line_range,
        hash_sha256=hash_sha256,
        observed_at=observed_at,
        ttl_days=SOURCE_TTL_DAYS[source],
    )


# ─────────────────────────────────────────────────────────────────────
# 5-sources-one-each engine (corroboration full-bonus)
# ─────────────────────────────────────────────────────────────────────


def _one_hit_per_source_provider(
    source: str, hit_id: str, title: str, snippet: str, relevance: float,
    observed_at: str, source_updated_at: str,
):
    """Closure that returns a provider callable yielding exactly one hit."""
    from tests.test_executive_v2.canary_b1.evidence_pack import _make_hit_v2

    def _provider(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source=source,
            hit_id=hit_id,
            title=title,
            relevance_score=relevance,
            snippet=snippet,
            source_uri=f"{source}://{hit_id}",
            source_updated_at=source_updated_at,
            retrieval_mode="metadata_only",
            observed_at=observed_at,
            ttl_days=SOURCE_TTL_DAYS[source],
        )]
    return _provider


@pytest.fixture
def b1_engine_with_5_sources_one_each(
    frozen_time,
    in_memory_storage,
    audit_capture,
) -> EvidencePackEngine:
    """Engine with exactly 1 hit per allowed source (5 sources × 1 hit).

    Used by ``test_ep_agg_04_corroboration_5_sources_full_bonus``.
    """
    observed = frozen_time
    updated = "2026-07-08T20:00:00+00:00"
    sources = {
        "policy": _one_hit_per_source_provider(
            "policy", "b1-policy-001", "policy decision fixture",
            f"{_UNIVERSAL_TOKEN} policy notes", 0.9,
            observed, updated,
        ),
        "contract": _one_hit_per_source_provider(
            "contract", "b1-contract-001", "contract fixture",
            f"{_UNIVERSAL_TOKEN} contract constraints", 0.9,
            observed, updated,
        ),
        "gbrain": _one_hit_per_source_provider(
            "gbrain", "b1-gbrain-001", "gbrain fixture",
            f"{_UNIVERSAL_TOKEN} gbrain entity", 0.9,
            observed, updated,
        ),
        "obsidian": _one_hit_per_source_provider(
            "obsidian", "b1-obsidian-001", "obsidian fixture",
            f"{_UNIVERSAL_TOKEN} obsidian note", 0.9,
            observed, updated,
        ),
        "report": _one_hit_per_source_provider(
            "report", "b1-report-001", "report fixture",
            f"{_UNIVERSAL_TOKEN} report content", 0.9,
            observed, updated,
        ),
    }
    return EvidencePackEngine(
        sources=sources,
        storage=in_memory_storage,
        audit_sink=audit_capture,
    )


# ─────────────────────────────────────────────────────────────────────
# Clock skew engine (source_updated_at in the future)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def b1_engine_with_clock_skew(
    frozen_time,
    in_memory_storage,
    audit_capture,
) -> EvidencePackEngine:
    """Engine with a gbrain hit whose source_updated_at is in the future."""
    observed = frozen_time  # 2026-07-08T20:00:00+00:00
    future = "2099-01-01T00:00:00+00:00"

    def _provider(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source="gbrain",
            hit_id="b1-skew-001",
            title=f"{_UNIVERSAL_TOKEN} future-dated gbrain hit",
            relevance_score=0.9,
            snippet=f"{_UNIVERSAL_TOKEN} future-dated snippet",
            source_uri=f"gbrain://b1-skew-001",
            source_updated_at=future,
            retrieval_mode="metadata_only",
            observed_at=observed,
            ttl_days=SOURCE_TTL_DAYS["gbrain"],
        )]

    sources = {"gbrain": _provider}
    return EvidencePackEngine(
        sources=sources,
        storage=in_memory_storage,
        audit_sink=audit_capture,
    )


# ─────────────────────────────────────────────────────────────────────
# policy_vs_obsidian conflict engine
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def b1_engine_with_policy_vs_obsidian_high(
    frozen_time,
    in_memory_storage,
    audit_capture,
) -> EvidencePackEngine:
    """Engine designed to produce a policy_vs_goal conflict (severity=high).

    The pair {policy, obsidian} triggers policy_vs_goal → high per the
    classifier. We use distinct hit_ids + distinct snippets with low token
    overlap to avoid being captured by an earlier rule (scope).

    Tokens ≥ 3 chars: "policy" and "obsidian" share no common tokens with the
    objective (different content domains), so the conflict detector lands on
    policy_vs_goal.
    """
    observed = frozen_time
    updated = "2026-07-08T20:00:00+00:00"

    def _policy(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source="policy",
            hit_id="b1-conflict-policy-001",
            title=f"{_UNIVERSAL_TOKEN} policy decision alpha",
            relevance_score=0.9,
            snippet=f"{_UNIVERSAL_TOKEN} decision alpha approved",
            source_uri="state_meta[objective_policy_decision:b1-conflict-policy-001]",
            source_updated_at=updated,
            retrieval_mode="metadata_only",
            observed_at=observed,
            ttl_days=SOURCE_TTL_DAYS["policy"],
        )]

    def _obsidian(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source="obsidian",
            hit_id="b1-conflict-obsidian-001",
            title=f"{_UNIVERSAL_TOKEN} obsidian diary beta",
            relevance_score=0.9,
            snippet=f"{_UNIVERSAL_TOKEN} diary beta rejected",
            source_uri="file://obsidian/b1-conflict-obsidian-001",
            source_updated_at=updated,
            retrieval_mode="snippet",
            observed_at=observed,
            ttl_days=SOURCE_TTL_DAYS["obsidian"],
        )]

    sources = {"policy": _policy, "obsidian": _obsidian}
    return EvidencePackEngine(
        sources=sources,
        storage=in_memory_storage,
        audit_sink=audit_capture,
    )


# ─────────────────────────────────────────────────────────────────────
# Multi-conflict policy_vs_obsidian engine (for audit count test)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def b1_engine_with_three_policy_vs_obsidian_conflicts(
    frozen_time,
    in_memory_storage,
    audit_capture,
) -> EvidencePackEngine:
    """Engine that produces 3 distinct policy_vs_goal conflicts (3 high audit events)."""
    observed = frozen_time
    updated = "2026-07-08T20:00:00+00:00"

    def _policy(query, *, max_hits: int = 5, observed_at: str):
        out = []
        for i in range(3):
            out.append(_make_hit_v2(
                source="policy",
                hit_id=f"b1-multi-policy-{i:03d}",
                title=f"{_UNIVERSAL_TOKEN} policy decision alpha-{i}",
                relevance_score=0.9,
                snippet=f"{_UNIVERSAL_TOKEN} decision alpha-{i} approved",
                source_uri=f"state_meta[objective_policy_decision:b1-multi-policy-{i:03d}]",
                source_updated_at=updated,
                retrieval_mode="metadata_only",
                observed_at=observed,
                ttl_days=SOURCE_TTL_DAYS["policy"],
            ))
        return out

    def _obsidian(query, *, max_hits: int = 5, observed_at: str):
        out = []
        for i in range(3):
            out.append(_make_hit_v2(
                source="obsidian",
                hit_id=f"b1-multi-obsidian-{i:03d}",
                title=f"{_UNIVERSAL_TOKEN} obsidian diary beta-{i}",
                relevance_score=0.9,
                snippet=f"{_UNIVERSAL_TOKEN} diary beta-{i} rejected",
                source_uri=f"file://obsidian/b1-multi-obsidian-{i:03d}",
                source_updated_at=updated,
                retrieval_mode="snippet",
                observed_at=observed,
                ttl_days=SOURCE_TTL_DAYS["obsidian"],
            ))
        return out

    sources = {"policy": _policy, "obsidian": _obsidian}
    return EvidencePackEngine(
        sources=sources,
        storage=in_memory_storage,
        audit_sink=audit_capture,
    )


# ─────────────────────────────────────────────────────────────────────
# Multi-source partial-failure engine (rollback test)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def b1_engine_with_partial_provider_failure(
    frozen_time,
    in_memory_storage,
    audit_capture,
) -> EvidencePackEngine:
    """Engine with one provider raising and others OK (rollback test)."""
    observed = frozen_time
    updated = "2026-07-08T20:00:00+00:00"

    def _ok_gbrain(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source="gbrain",
            hit_id="b1-partial-gbrain-001",
            title=f"{_UNIVERSAL_TOKEN} gbrain partial ok",
            relevance_score=0.9,
            snippet=f"{_UNIVERSAL_TOKEN} partial ok snippet",
            source_uri="gbrain://b1-partial-gbrain-001",
            source_updated_at=updated,
            retrieval_mode="metadata_only",
            observed_at=observed,
            ttl_days=SOURCE_TTL_DAYS["gbrain"],
        )]

    def _failing_obsidian(query, *, max_hits: int = 5, observed_at: str):
        raise RuntimeError("simulated partial provider failure")

    sources = {"gbrain": _ok_gbrain, "obsidian": _failing_obsidian}
    return EvidencePackEngine(
        sources=sources,
        storage=in_memory_storage,
        audit_sink=audit_capture,
    )


# ─────────────────────────────────────────────────────────────────────
# Provider with 10 hits (cap tests)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def b1_provider_with_10_hits():
    """Spec with 10 canned gbrain hits, used for cap/clamp tests."""
    hits = tuple({
        "hit_id": f"b1-cap-gbrain-{i:03d}",
        "title": f"{_UNIVERSAL_TOKEN} cap gbrain hit {i}",
        "relevance_score": 0.5 + 0.05 * i,
        "snippet": f"{_UNIVERSAL_TOKEN} cap snippet number {i} content",
        "source_updated_at": "2026-07-08T20:00:00+00:00",
    } for i in range(10))
    return FakeProviderSpec(name="gbrain", hits=hits, is_available=True)


# ─────────────────────────────────────────────────────────────────────
# Provider with empty hits
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def b1_provider_with_empty_hits():
    return empty_spec("gbrain")


# ─────────────────────────────────────────────────────────────────────
# Provider that raises (failure mode test)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def b1_provider_with_query_exception():
    return failing_spec("gbrain", RuntimeError("simulated provider failure"))


# ─────────────────────────────────────────────────────────────────────
# Hit builders for tie / dedup ranking tests
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def b1_hit_factory(frozen_time):
    """Factory producing KnowledgeHitV2 with custom snippet for tie/dedup tests."""
    observed = frozen_time
    updated = "2026-07-08T20:00:00+00:00"

    def _make(
        source: str,
        hit_id: str,
        snippet: str,
        relevance: float = 0.9,
        freshness_score: float = 1.0,
    ) -> KnowledgeHitV2:
        from tests.test_executive_v2.canary_b1.evidence_pack import (
            _make_hit_v2,
            FreshnessPolicy,
            ProvenanceEnvelope,
        )
        h = _make_hit_v2(
            source=source,
            hit_id=hit_id,
            title=f"{_UNIVERSAL_TOKEN} {hit_id}",
            relevance_score=relevance,
            snippet=snippet,
            source_uri=f"{source}://{hit_id}",
            source_updated_at=updated,
            retrieval_mode="metadata_only",
            observed_at=observed,
            ttl_days=SOURCE_TTL_DAYS[source],
        )
        # Override freshness_score for tie tests (so relevance × freshness × sp is identical)
        return KnowledgeHitV2(
            source=h.source,
            hit_id=h.hit_id,
            title=h.title,
            relevance_score=h.relevance_score,
            snippet=h.snippet,
            location=h.location,
            fingerprint=h.fingerprint,
            created_at=h.created_at,
            provenance=h.provenance,
            freshness=FreshnessPolicy(
                observed_at=h.freshness.observed_at,
                source_updated_at=h.freshness.source_updated_at,
                staleness_days=h.freshness.staleness_days,
                freshness=h.freshness.freshness,
                freshness_score=freshness_score,
            ),
            effective_score=h.effective_score,
        )
    return _make