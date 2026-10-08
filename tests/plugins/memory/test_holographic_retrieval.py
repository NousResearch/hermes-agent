"""Tests for FactRetriever FTS5 query sanitization.

These tests cover the fix where raw natural-language queries passed to
FTS5 MATCH were AND-joined by default, dropping recall to zero on any
multi-word prose query. The sanitizer drops stopwords and OR-joins the
remaining content tokens as phrase literals.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore

# ---------------------------------------------------------------------------
# _sanitize_fts_query — unit tests (no DB required)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected_tokens",
    [
        # stopwords dropped
        ("what happened with the deployment rollback", {"happened", "deployment", "rollback"}),
        # single content word passes through
        ("compaction", {"compaction"}),
        # all stopwords → falls back to raw
        ("the and of", None),  # None = sentinel for fallback-to-raw
        # empty string → empty output
        ("", ""),
        # FTS5 operator characters stripped
        ("context: length-probe", {"context", "lengthprobe"}),
        # trailing punctuation stripped by tokenizer
        ("hello, world!", {"hello", "world"}),
    ],
)
def test_sanitize_fts_query_extracts_content_tokens(query, expected_tokens):
    result = FactRetriever._sanitize_fts_query(query)

    if expected_tokens == "":
        assert result == ""
        return

    if expected_tokens is None:
        # Pathological case: all stopwords — should fall back to raw query
        assert result == query
        return

    # OR-joined phrase literals: `"tok1" OR "tok2" OR ...`
    # Extract the tokens between quotes, order-independent.
    import re
    matches = re.findall(r'"([^"]+)"', result)
    assert set(matches) == expected_tokens, f"got {result!r}"

# ---------------------------------------------------------------------------
# Integration test — actually run _fts_candidates against an in-memory DB
# ---------------------------------------------------------------------------

@pytest.fixture
def retriever_with_facts(tmp_path):
    """MemoryStore seeded with a few facts for retrieval tests."""
    db_path = tmp_path / "test_facts.db"
    store = MemoryStore(str(db_path))
    store.add_fact(
        content="The Thursday deployment rollback failed because of stale migration state.",
        category="project",
    )
    store.add_fact(
        content="Compaction settings tuned to 0.85 threshold.",
        category="tool",
    )
    store.add_fact(
        content="Venice.ai advertises availableContextTokens inside model_spec.",
        category="tool",
    )
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()

def test_prefetch_recovers_prose_query(retriever_with_facts):
    """A natural-language query should now match the relevant fact.

    Before the sanitizer fix, 'what happened with the deployment rollback'
    returned zero hits because FTS5 required every token to co-occur.
    """
    results = retriever_with_facts.search(
        "what happened with the deployment rollback"
    )
    assert len(results) >= 1
    # The top hit should be the deployment rollback fact
    assert "deployment rollback" in results[0]["content"].lower()

# ---------------------------------------------------------------------------
# Loop-invariant encode hoists (perf) — search/probe/related must encode
# constant vectors ONCE per call, not once per candidate/row.
# encode_text/encode_atom are deterministic (SHA-256 counter blocks), so the
# hoisted vectors are bit-identical to the per-iteration values they replace.
# ---------------------------------------------------------------------------

from plugins.memory.holographic import holographic as hrr

def test_encode_functions_are_deterministic():
    """Soundness premise of the hoists: same input -> identical vector."""
    import numpy as np

    assert np.array_equal(hrr.encode_text("deploy target", 1024),
                          hrr.encode_text("deploy target", 1024))
    assert np.array_equal(hrr.encode_atom("__hrr_role_content__", 1024),
                          hrr.encode_atom("__hrr_role_content__", 1024))

# ---------------------------------------------------------------------------
# Relevance floor + zero-phase presence test on the pure-vector paths (#132347)
# — probe/related/reason ranked EVERY fact with a ~0-similarity noise score and
# sliced the top `limit`, so any query (including the empty string and strings
# that match nothing) returned the whole store. Rows must clear a similarity
# floor to be returned; when no row does, the query falls back to the FTS5
# path, which only ever returns rows FTS5 actually matched.
#
# probe()/reason() additionally scored against the WRONG reference vector
# (a role_content-bound target that is quasi-orthogonal to the unbind residual
# whether or not the entity is present): unbind(bind(a, b), bind(a, b)) cancels
# to the ZERO-PHASE vector, so that is the correct "term was present" reference.
# With it, probe/reason carry real signal (match ~0.5-0.6, noise <=0.04) instead
# of leaning on the FTS fallback.
# ---------------------------------------------------------------------------

@pytest.fixture
def retriever_with_entities(tmp_path):
    """MemoryStore seeded with facts about distinct multi-word entities (the
    entity-extraction regex only links Capitalized Multi-Word names)."""
    store = MemoryStore(str(tmp_path / "facts.db"))
    for content in (
        "Alice Johnson lives in Paris and works at BNP Paribas",
        "Bob Smith prefers dark mode in every editor",
        "Charlie Brown plays piano on weekends",
        "Diana Reyes speaks four languages fluently",
        "Erik Larsson runs marathons before sunrise",
    ):
        store.add_fact(content=content, category="general")
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()

@pytest.mark.parametrize("query", ["", "Grimsdhal", "ZZQQ_TOTALLY_UNRELATED_STRING"])
def test_vector_queries_do_not_leak_the_whole_store(retriever_with_entities, query):
    """A query that matches nothing must return nothing, not the top slice of the store."""
    r = retriever_with_entities
    assert r.probe(query) == []
    assert r.related(query) == []
    assert r.reason([query, "another_nonexistent_entity"]) == []

def test_related_keeps_true_structural_match(retriever_with_entities):
    """related() has real signal (sim ~0.66 for a stored entity vs <=0.04 noise):
    the matching fact survives the floor and ranks first."""
    results = retriever_with_entities.related("Alice Johnson")
    assert [f["content"] for f in results] == ["Alice Johnson lives in Paris and works at BNP Paribas"]

def test_probe_returns_the_stored_entity_fact(retriever_with_entities):
    """With the zero-phase presence test, probe('Alice Johnson') scores ~0.56 on the one
    fact encoding the entity (noise <=0.04 elsewhere), so the vector path returns it
    directly; the FTS fallback only ever sees queries no vector row cleared."""
    results = retriever_with_entities.probe("Alice Johnson")
    assert [f["content"] for f in results] == ["Alice Johnson lives in Paris and works at BNP Paribas"]

def test_probe_category_bank_does_not_leak(retriever_with_entities):
    """The category-bank branch of probe() ranks the same rows against the bank
    residual; an unrelated query must not leak the category through it either."""
    assert retriever_with_entities.probe("ZZQQ_TOTALLY_UNRELATED_STRING", category="general") == []
    assert retriever_with_entities.probe("", category="general") == []

@pytest.fixture
def retriever_alice_variants(tmp_path):
    """Three Alice-related facts: two encode the entity "Alice Johnson" structurally
    (Capitalized Multi-Word names), one mentions it only in lowercase — FTS-visible
    tokens but never an entity term, so the vector path cannot and must not return it."""
    store = MemoryStore(str(tmp_path / "facts.db"))
    for content in (
        "Alice Johnson works on the vision team at Acme Robotics",
        "Alice Johnson also reviews books on weekends",
        "the passphrase alice johnson is lowercase",
    ):
        store.add_fact(content=content, category="general")
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()

def test_probe_is_structural_not_keyword(retriever_alice_variants):
    """probe('Alice Johnson') must return exactly the facts encoding the entity as a
    structural term. The lowercase mention has no entity term (its zero-phase residual
    is noise) yet its tokens ARE matched by the FTS fallback query — so excluding it
    proves the zero-phase vector ranking carried the query, not keyword search."""
    results = retriever_alice_variants.probe("Alice Johnson")
    assert sorted(f["content"] for f in results) == [
        "Alice Johnson also reviews books on weekends",
        "Alice Johnson works on the vision team at Acme Robotics",
    ]

def test_reason_and_semantics_requires_every_entity(retriever_alice_variants):
    """reason() is a vector-space JOIN: only the fact encoding BOTH entities survives
    the floor (the Alice-only fact falls below it on the Acme Robotics residual). The
    FTS fallback OR-joins tokens and would return the Alice-only fact too, so the
    exact single-fact result proves the AND semantics ran on the vector path."""
    results = retriever_alice_variants.reason(["Alice Johnson", "Acme Robotics"])
    assert [f["content"] for f in results] == ["Alice Johnson works on the vision team at Acme Robotics"]

def test_probe_category_bank_degrades_to_per_fact_vectors(retriever_alice_variants):
    """The category-bank residual is diluted by the bank's superposition (even a
    matching fact scores far below the floor), so the bank branch must fall through
    to the per-fact zero-phase path and still return exactly the structural matches."""
    results = retriever_alice_variants.probe("Alice Johnson", category="general")
    assert sorted(f["content"] for f in results) == [
        "Alice Johnson also reviews books on weekends",
        "Alice Johnson works on the vision team at Acme Robotics",
    ]
