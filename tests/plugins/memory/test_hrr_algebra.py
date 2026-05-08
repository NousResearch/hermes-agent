"""Regression tests for the HRR encode/probe algebra.

These would have caught the Strategy-B probe bug fixed in
walker/hrr-probe-algebra-fix:
- Strategy A round-trip: insert fact → probe entity → raw_sim > 0.30
- Crosstalk: probe an unrelated entity → raw_sim near zero
- Bundle invertibility: ``unbind(bundle, role)`` recovers bound atoms
- Determinism: encode_atom is byte-stable across calls
- Storage round-trip: phases → bytes → phases is bit-exact
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from plugins.memory.holographic import holographic as hrr
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


DIM = 2048


# ---------------------------------------------------------------------------
# Primitive-level tests (would catch encoding regressions)
# ---------------------------------------------------------------------------


def test_encode_atom_deterministic_within_process():
    v1 = hrr.encode_atom("walker anderson", DIM)
    v2 = hrr.encode_atom("walker anderson", DIM)
    assert np.array_equal(v1, v2)


def test_encode_atom_differs_across_inputs():
    a = hrr.encode_atom("apple", DIM)
    b = hrr.encode_atom("banana", DIM)
    assert not np.array_equal(a, b)
    # Random phase atoms should be near-orthogonal
    assert abs(hrr.similarity(a, b)) < 0.1


def test_bind_unbind_roundtrip():
    a = hrr.encode_atom("apple", DIM)
    k = hrr.encode_atom("__hrr_role_entity__", DIM)
    recovered = hrr.unbind(hrr.bind(a, k), k)
    assert hrr.similarity(recovered, a) > 0.99


def test_storage_roundtrip_is_bit_exact():
    v = hrr.encode_atom("apollo energy group", DIM)
    restored = hrr.bytes_to_phases(hrr.phases_to_bytes(v))
    assert np.array_equal(v, restored)


def test_bundle_recovery_at_n2():
    """Strategy A: unbind by role atom recovers value bound to that role."""
    a = hrr.encode_atom("apple", DIM)
    b = hrr.encode_atom("banana", DIM)
    k_a = hrr.encode_atom("k_a", DIM)
    k_b = hrr.encode_atom("k_b", DIM)
    bundle = hrr.bundle(hrr.bind(a, k_a), hrr.bind(b, k_b))
    residual_a = hrr.unbind(bundle, k_a)
    assert hrr.similarity(residual_a, a) > 0.55
    assert hrr.similarity(residual_a, b) < 0.10


def test_bundle_recovery_capacity_n8():
    """Strategy A still produces signal at N=8 components."""
    target = hrr.encode_atom("target", DIM)
    k_target = hrr.encode_atom("k_target", DIM)
    components = [hrr.bind(target, k_target)]
    for i in range(7):
        x = hrr.encode_atom(f"distractor_{i}", DIM)
        kx = hrr.encode_atom(f"k_dist_{i}", DIM)
        components.append(hrr.bind(x, kx))
    bundle = hrr.bundle(*components)
    residual = hrr.unbind(bundle, k_target)
    sim_target = hrr.similarity(residual, target)
    sim_random = hrr.similarity(residual, hrr.encode_atom("zebra", DIM))
    assert sim_target > 0.20
    assert sim_target - sim_random > 0.15


# ---------------------------------------------------------------------------
# End-to-end probe tests through MemoryStore + FactRetriever
# ---------------------------------------------------------------------------


@pytest.fixture
def store_and_retriever():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = MemoryStore(db_path=path, hrr_dim=DIM, default_trust=0.85)
    retriever = FactRetriever(
        store=store,
        temporal_decay_half_life=0,
        reinforce_on_retrieval=False,  # don't pollute test state
        hrr_dim=DIM,
        hrr_weight=0.3,
    )
    yield store, retriever
    try:
        os.unlink(path)
    except OSError:
        pass


def _raw_sim_from_score(score: float, fact: dict) -> float:
    """Invert ``max(sim, 0) * reinforced_trust`` to recover raw cosine sim."""
    base = float(fact.get("trust_score", 0) or 0)
    hc = int(fact.get("helpful_count", 0) or 0)
    mult = base * (1.0 + 0.1 * min(hc, 20))
    if mult <= 0:
        return 0.0
    return score / mult


def test_probe_recovers_target_entity(store_and_retriever):
    store, retriever = store_and_retriever
    target_id = store.add_fact("Walker Anderson is a person.", category="identity")
    distractor_id = store.add_fact("The Eiffel Tower is in Paris, France.", category="general")

    results = retriever.probe(entity="Walker Anderson", limit=10)
    by_id = {r["fact_id"]: r for r in results}
    assert target_id in by_id
    assert distractor_id in by_id  # both surface; target should rank first

    target_sim = _raw_sim_from_score(by_id[target_id]["score"], by_id[target_id])
    distractor_sim = _raw_sim_from_score(by_id[distractor_id]["score"], by_id[distractor_id])

    assert target_sim > 0.30, f"target raw_sim={target_sim:.4f}; expected > 0.30"
    assert distractor_sim < 0.15, f"distractor raw_sim={distractor_sim:.4f}; expected < 0.15"
    assert results[0]["fact_id"] == target_id


def test_probe_unrelated_entity_returns_low_signal(store_and_retriever):
    store, retriever = store_and_retriever
    store.add_fact("Walker Anderson is a person.", category="identity")
    store.add_fact("The Eiffel Tower is in Paris, France.", category="general")

    results = retriever.probe(entity="Tokyo", limit=10)
    for r in results:
        sim = _raw_sim_from_score(r["score"], r)
        assert sim < 0.20, f"crosstalk sim={sim:.4f} on fact {r['fact_id']}; expected < 0.20"


def test_reason_and_semantics(store_and_retriever):
    store, retriever = store_and_retriever
    both = store.add_fact("Walker Anderson met James Paddock.", category="general")
    only_walker = store.add_fact("Walker Anderson is a person.", category="identity")
    only_paddock = store.add_fact("James Paddock drives a truck.", category="general")

    results = retriever.reason(entities=["Walker Anderson", "James Paddock"], limit=10)
    by_id = {r["fact_id"]: r for r in results}
    sim_both = _raw_sim_from_score(by_id[both]["score"], by_id[both])
    sim_walker = _raw_sim_from_score(by_id[only_walker]["score"], by_id[only_walker])
    sim_paddock = _raw_sim_from_score(by_id[only_paddock]["score"], by_id[only_paddock])

    assert sim_both > sim_walker
    assert sim_both > sim_paddock


def test_related_finds_entity_in_content_slot(store_and_retriever):
    """Entities mentioned in content but not extracted as structural entities
    should still surface via related()."""
    store, retriever = store_and_retriever
    # _RE_CAPITALIZED requires two consecutive capitalized words; "tokyo"
    # alone isn't extracted as a structural entity, so it lives only in
    # the content slot.
    fact_id = store.add_fact("The trip routes through tokyo briefly.", category="general")
    distractor = store.add_fact("Walker Anderson is a person.", category="identity")

    results = retriever.related(entity="tokyo", limit=10)
    by_id = {r["fact_id"]: r for r in results}
    assert fact_id in by_id
    sim_target = _raw_sim_from_score(by_id[fact_id]["score"], by_id[fact_id])
    sim_distractor = _raw_sim_from_score(by_id[distractor]["score"], by_id[distractor])
    assert sim_target > sim_distractor
