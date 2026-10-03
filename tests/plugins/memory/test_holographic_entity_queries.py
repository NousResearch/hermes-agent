"""probe/related/reason must answer from the entity index, not from HRR unbinding.

Regression: all three ranked candidates with ``hrr.unbind(fact_vector, role_bound_key)``.
Unbinding a role-bound key cannot recover that fact's content signal, so every candidate
scored as noise and the ordering collapsed to trust order -- facts *linked* to the entity
under test ranked below unrelated facts. ``contradict()`` in the same file already answers
the same question exactly from ``entities``/``fact_entities``, and needs no numpy.

Contracts asserted here (relationships between data, not snapshots):
  * ``probe(entity)`` is exactly the set of facts linked to that entity -- including a
    fact whose ``content`` never names the entity, which keyword search cannot find.
  * ``related(entity)`` is the facts of co-occurring entities, minus the entity's own.
  * ``reason([a, b])`` is exactly ``probe(a) & probe(b)``.
  * All three hold with numpy unavailable.
"""
from __future__ import annotations

import pytest

from plugins.memory.holographic import holographic as hrr
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


def _link(store: MemoryStore, fact_id: int, *names: str) -> None:
    """Link entities to a fact directly.

    ``MemoryStore._extract_entities`` only matches capitalized multi-word phrases, so
    single-word names never auto-link. The *query* path is what is under test, so links
    are seeded explicitly instead of depending on extraction.
    """
    for name in names:
        store._write(
            "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
            (fact_id, store._resolve_entity(name)),
        )


@pytest.fixture
def entity_facts(tmp_path):
    """Five facts: two entities, a shared fact, an index-only link, and an unlinked fact.

    ``index_only`` is linked to Robosmart but never names it, so it is reachable only
    through the entity index -- the case a keyword fallback silently drops.
    """
    store = MemoryStore(str(tmp_path / "entity_queries.db"))
    ids = {
        "shared": store.add_fact("Robosmart asked Miranda to own the renewal.", category="project"),
        "robosmart_named": store.add_fact("The Robosmart pilot renewed for another quarter.", category="project"),
        "index_only": store.add_fact("The client renewed the pilot for another quarter.", category="project"),
        "miranda_named": store.add_fact("Miranda is shadowing the discovery calls this month.", category="project"),
        "unlinked": store.add_fact("Compaction settings tuned to 0.85 threshold.", category="tool"),
    }
    # add_fact also extracts entities by regex, which links extras (e.g. "The Robosmart" out of
    # "The Robosmart pilot..."); the query path is what is under test, so start from the links
    # this test declares instead of whatever extraction inferred.
    store._write("DELETE FROM fact_entities")
    _link(store, ids["shared"], "Robosmart", "Miranda")
    _link(store, ids["robosmart_named"], "Robosmart")
    _link(store, ids["index_only"], "Robosmart")
    _link(store, ids["miranda_named"], "Miranda")
    yield FactRetriever(store=store), ids
    store.close()


def _ids(facts) -> set:
    return {fact["fact_id"] for fact in facts}


def test_probe_and_related_answer_from_the_entity_index(entity_facts):
    retriever, ids = entity_facts

    assert _ids(retriever.probe("Robosmart")) == {
        ids["shared"], ids["robosmart_named"], ids["index_only"],
    }
    # related(): the co-occurring entity's facts, minus the subject's own facts.
    assert _ids(retriever.related("Robosmart")) == {ids["miranda_named"]}


def test_reason_is_the_probe_intersection_and_needs_no_numpy(entity_facts, monkeypatch):
    retriever, ids = entity_facts

    probe_a = _ids(retriever.probe("Robosmart"))
    probe_b = _ids(retriever.probe("Miranda"))
    assert _ids(retriever.reason(["Robosmart", "Miranda"])) == probe_a & probe_b == {ids["shared"]}

    # The index lookup needs no vector math: results must not change without numpy.
    monkeypatch.setattr(hrr, "_HAS_NUMPY", False)
    assert _ids(retriever.probe("Robosmart")) == {
        ids["shared"], ids["robosmart_named"], ids["index_only"],
    }
    assert _ids(retriever.reason(["Robosmart", "Miranda"])) == {ids["shared"]}
