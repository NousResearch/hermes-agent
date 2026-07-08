"""Adapter contract tests (8 tests, parametrized → 28 cases).

Implements the contract tests from adapter_contract_test_plan.md.

These tests use fake providers directly (not the engine, except for failure
modes) to verify each provider satisfies the EvidencePack v2 contract:

* test_contract_01_provider_returns_knowledge_hit_v2_list (5 cases)
* test_contract_02_every_hit_has_provenance_with_read_only_true (5 cases)
* test_contract_03_producer_name_in_canonical_registry (5 cases)
* test_contract_04_provider_respects_max_hits_per_source (5 cases)
* test_contract_05_provider_clamp_max_hits_to_one_when_zero (1 case)
* test_contract_06_provider_does_not_return_more_than_top_n (1 case)
* test_contract_07_provider_query_exception_marked_as_source_failed (4 cases)
* test_contract_08_provider_with_no_hits_returns_empty_list (1 case)

Total: 27 test cases (parametrized expansions of 8 contract tests).
"""

from __future__ import annotations

import pytest

from tests.test_executive_v2.canary_b1.evidence_pack import (
    EvidencePackEngine,
    KnowledgeHitV2,
    KnowledgeQuery,
)
from tests.test_executive_v2.canary_b1.conftest import _InMemoryStorage
from tests.test_executive_v2.canary_b1.fake_providers import (
    PRODUCER_NAME,
    contract_provider,
    empty_spec,
    failing_spec,
    gbrain_provider,
    obsidian_provider,
    policy_provider,
    report_provider,
    FakeProviderSpec,
)


OBS = "2026-07-08T20:00:00+00:00"


# ─────────────────────────────────────────────────────────────────────
# Spec factory (one row per source)
# ─────────────────────────────────────────────────────────────────────


def _spec_for(source: str, n_hits: int = 1) -> FakeProviderSpec:
    """Build a FakeProviderSpec with `n_hits` canned hits for `source`."""
    base_hit = {
        "title": f"contract {source} title",
        "relevance_score": 0.9,
        "snippet": f"contract {source} content alpha",
        "source_updated_at": OBS,
    }
    if source == "gbrain":
        hits = tuple({**base_hit,
                     "hit_id": f"contract-{source}-{i:03d}",
                     "snippet": f"contract {source} content alpha variant-{i}"}
                    for i in range(n_hits))
    elif source == "obsidian":
        hits = tuple({**base_hit,
                     "hit_id": f"contract-{source}-{i:03d}",
                     "snippet": f"contract {source} content alpha variant-{i}"}
                    for i in range(n_hits))
    elif source == "report":
        hits = tuple({**base_hit,
                     "hit_id": f"contract-{source}-{i:03d}",
                     "source_uri": f"report://contract-{source}-{i:03d}",
                     "snippet": f"contract {source} content alpha variant-{i}"}
                    for i in range(n_hits))
    elif source == "policy":
        hits = tuple({**base_hit,
                     "hit_id": f"contract-{source}-{i:03d}",
                     "warnings": (f"alpha warning {i}",),
                     "snippet": f"contract {source} content alpha variant-{i}"}
                    for i in range(n_hits))
    elif source == "contract":
        hits = tuple({**base_hit,
                     "hit_id": f"contract-{source}-{i:03d}",
                     "success_criteria": (f"alpha criterion {i}",),
                     "snippet": f"contract {source} content alpha variant-{i}"}
                    for i in range(n_hits))
    else:
        raise ValueError(f"unknown source: {source}")
    return FakeProviderSpec(name=source, hits=hits, is_available=True)


def _provider_for(source: str, spec: FakeProviderSpec):
    """Return the matching provider factory for `source`."""
    return {
        "gbrain": gbrain_provider,
        "obsidian": obsidian_provider,
        "report": report_provider,
        "policy": policy_provider,
        "contract": contract_provider,
    }[source](spec)


def _query(text: str = "alpha content variant", idx: int = 0) -> KnowledgeQuery:
    return KnowledgeQuery(
        objective_id=f"contract-{idx}",
        objective_text=text,
    )


# ─────────────────────────────────────────────────────────────────────
# 2.1.1 — contract shape: returns list[KnowledgeHitV2]
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("source_name", ["gbrain", "obsidian", "report", "policy", "contract"])
def test_contract_01_provider_returns_knowledge_hit_v2_list(source_name):
    """Each fake provider returns list[KnowledgeHitV2] for any valid query."""
    spec = _spec_for(source_name, n_hits=1)
    provider = _provider_for(source_name, spec)
    q = _query(idx=1)
    result = provider(q, max_hits=5, observed_at=OBS)
    assert isinstance(result, list), f"{source_name}: not a list"
    assert all(isinstance(h, KnowledgeHitV2) for h in result), (
        f"{source_name}: non-KnowledgeHitV2 elements: {result}"
    )


# ─────────────────────────────────────────────────────────────────────
# 2.1.2 — safety invariant: 100% of hits have provenance.read_only=True
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("source_name", ["gbrain", "obsidian", "report", "policy", "contract"])
def test_contract_02_every_hit_has_provenance_with_read_only_true(source_name):
    """Every hit returned by every provider has provenance.read_only == True."""
    spec = _spec_for(source_name, n_hits=3)
    provider = _provider_for(source_name, spec)
    for i in range(5):
        q = _query(text=f"alpha content variant-{i}", idx=i)
        result = provider(q, max_hits=5, observed_at=OBS)
        assert result, f"{source_name}: empty result for query {i}"
        for h in result:
            assert h.provenance.read_only is True, (
                f"{source_name} hit {h.hit_id} has read_only={h.provenance.read_only}"
            )


# ─────────────────────────────────────────────────────────────────────
# 2.1.3 — contract identifier: producer in canonical registry
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("source_name", ["gbrain", "obsidian", "report", "policy", "contract"])
def test_contract_03_producer_name_in_canonical_registry(source_name):
    """producer ∈ PRODUCER_NAME.values() for every fake provider."""
    spec = _spec_for(source_name, n_hits=1)
    provider = _provider_for(source_name, spec)
    q = _query(idx=3)
    result = provider(q, max_hits=5, observed_at=OBS)
    assert result, f"{source_name}: empty result"
    for h in result:
        assert h.provenance.producer in PRODUCER_NAME.values(), (
            f"{source_name}: producer {h.provenance.producer} not in registry"
        )
        assert h.provenance.producer == PRODUCER_NAME[source_name], (
            f"{source_name}: expected producer {PRODUCER_NAME[source_name]!r}, "
            f"got {h.provenance.producer!r}"
        )


# ─────────────────────────────────────────────────────────────────────
# 2.2.1 — cap: provider respects max_hits
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("source_name", ["gbrain", "obsidian", "report", "policy", "contract"])
def test_contract_04_provider_respects_max_hits_per_source(source_name):
    """Provider returns ≤ max_hits hits."""
    spec = _spec_for(source_name, n_hits=10)
    provider = _provider_for(source_name, spec)
    q = _query(idx=4)
    result = provider(q, max_hits=3, observed_at=OBS)
    assert len(result) <= 3, (
        f"{source_name}: returned {len(result)} > 3 cap"
    )


# ─────────────────────────────────────────────────────────────────────
# 2.2.2 — clamp behavior: max_hits=0
# ─────────────────────────────────────────────────────────────────────


def test_contract_05_provider_clamp_max_hits_to_one_when_zero():
    """max_hits=0 at the provider level returns [] (slice [:0]).

    The engine clamps max_hits_per_source to 1 BEFORE calling the provider,
    so providers never actually receive max_hits=0 in production. The
    provider-level behavior is to honor the slice (return []).
    """
    spec = _spec_for("gbrain", n_hits=10)
    provider = gbrain_provider(spec)
    q = _query(idx=5)
    result = provider(q, max_hits=0, observed_at=OBS)
    assert isinstance(result, list)
    assert len(result) == 0, f"expected [] for max_hits=0, got {len(result)}"


# ─────────────────────────────────────────────────────────────────────
# 2.2.3 — hard cap: ≤ top_n
# ─────────────────────────────────────────────────────────────────────


def test_contract_06_provider_does_not_return_more_than_top_n():
    """10 hits available, max_hits=5 → exactly 5 returned (not 6, not 4)."""
    spec = _spec_for("gbrain", n_hits=10)
    provider = gbrain_provider(spec)
    q = _query(idx=6)
    result = provider(q, max_hits=5, observed_at=OBS)
    assert len(result) == 5, f"expected 5 hits, got {len(result)}"


# ─────────────────────────────────────────────────────────────────────
# 2.3.1 — failure modes: provider exception → sources_failed
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "exc_type",
    [RuntimeError, OSError, ValueError, TimeoutError],
)
def test_contract_07_provider_query_exception_marked_as_source_failed(exc_type):
    """Provider that raises → engine marks source as failed, doesn't propagate."""
    bundle = {"gbrain": gbrain_provider(failing_spec("gbrain", exc_type("simulated")))}
    engine = EvidencePackEngine(
        sources=bundle, storage=_InMemoryStorage(), audit_sink=None,
    )
    pack = engine.dry_run(
        objective_id=f"contract-07-{exc_type.__name__}",
        objective_text="alpha content",
    )
    assert "gbrain" in pack.sources_failed, (
        f"expected gbrain in sources_failed, got {pack.sources_failed}"
    )
    assert "gbrain" not in pack.sources_queried, (
        f"gbrain should not be in sources_queried when it failed, got {pack.sources_queried}"
    )


# ─────────────────────────────────────────────────────────────────────
# 2.3.2 — empty result: provider returns []
# ─────────────────────────────────────────────────────────────────────


def test_contract_08_provider_with_no_hits_returns_empty_list():
    """Empty spec → engine produces pack with 0 hits but the source IS queried."""
    bundle = {
        "gbrain": gbrain_provider(empty_spec("gbrain")),
        "obsidian": obsidian_provider(empty_spec("obsidian")),
    }
    engine = EvidencePackEngine(
        sources=bundle, storage=_InMemoryStorage(), audit_sink=None,
    )
    pack = engine.dry_run(
        objective_id="contract-08",
        objective_text="alpha content",
    )
    assert len(pack.hits) == 0
    # Sources WERE queried (queried ≠ found)
    assert "gbrain" in pack.sources_queried
    assert "obsidian" in pack.sources_queried