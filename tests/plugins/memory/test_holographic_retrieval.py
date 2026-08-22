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


@pytest.fixture
def hoisted_retriever(tmp_path):
    """30 facts with HRR vectors, default dim (smaller dims trip an
    inhomogeneous-shape edge in the fact encoder).

    NOTE: a real tmp_path db, NOT ":memory:" — MemoryStore resolves the
    path and shares one process-wide connection per file, so ":memory:"
    becomes a literal ./:memory: file that leaks state across runs (and
    the NULL-vector test below would permanently corrupt it)."""
    store = MemoryStore(str(tmp_path / "hoist_store.db"))
    for i in range(30):
        store.add_fact(
            content=f"deploy target {i} setting alpha beta gamma option {i % 7}",
            category="fact" if i % 2 else "preference",
            tags=f"entity_{i % 5} deploy",
        )
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()


def _counting_spy(monkeypatch, attr):
    calls = []
    real = getattr(hrr, attr)

    def wrapper(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(hrr, attr, wrapper)
    return calls


def test_encode_functions_are_deterministic():
    """Soundness premise of the hoists: same input -> identical vector."""
    import numpy as np

    assert np.array_equal(hrr.encode_text("deploy target", 1024),
                          hrr.encode_text("deploy target", 1024))
    assert np.array_equal(hrr.encode_atom("__hrr_role_content__", 1024),
                          hrr.encode_atom("__hrr_role_content__", 1024))


def test_search_encodes_query_vector_once(hoisted_retriever, monkeypatch):
    calls = _counting_spy(monkeypatch, "encode_text")
    results = hoisted_retriever.search("deploy target setting")
    assert results  # the HRR path actually engaged
    assert len(calls) == 1, (
        f"query vector encoded {len(calls)}x in one search() — "
        "loop-invariant hoist regressed"
    )


def test_search_results_bit_identical_to_unhoisted(hoisted_retriever):
    """Parity: hoisted search() must produce the exact same results as the
    equivalent unhoisted loop.

    Replicates the unhoisted loop (query vector encoded per candidate) as
    the reference and compares full scored output for exact equality. The
    reference uses the same corrected query encoding (encode_query) as
    search(), so this test pins the hoist, not the encoding.
    """
    r = hoisted_retriever
    query = "deploy target setting"
    new_results = r.search(query)

    # --- pre-fix reference ---
    candidates = r._fts_candidates(query, None, 0.3, 10 * 3)
    query_tokens = r._tokenize(query)
    scored = []
    for fact in candidates:
        content_tokens = r._tokenize(fact["content"])
        tag_tokens = r._tokenize(fact.get("tags", ""))
        all_tokens = content_tokens | tag_tokens
        jaccard = r._jaccard_similarity(query_tokens, all_tokens)
        fts_score = fact.get("fts_rank", 0.0)
        if r.hrr_weight > 0 and fact.get("hrr_vector"):
            fact_vec = hrr.bytes_to_phases(fact["hrr_vector"])
            query_vec = hrr.encode_query(query, r.hrr_dim)  # per-candidate
            hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0
        else:
            hrr_sim = 0.5
        relevance = (r.fts_weight * fts_score
                     + r.jaccard_weight * jaccard
                     + r.hrr_weight * hrr_sim)
        fact["score"] = relevance * fact["trust_score"]
        scored.append(fact)
    scored.sort(key=lambda x: x["score"], reverse=True)
    old_results = scored[:10]
    for fact in old_results:
        fact.pop("hrr_vector", None)

    assert new_results == old_results


def test_related_encodes_role_atoms_once(hoisted_retriever, monkeypatch):
    calls = _counting_spy(monkeypatch, "encode_atom")
    results = hoisted_retriever.related("entity_1")
    assert results
    role_calls = [a for a in calls
                  if a and str(a[0]).startswith("__hrr_role_")]
    assert len(role_calls) == 2, (
        f"role atoms encoded {len(role_calls)}x in one related() — "
        "expected exactly 2 (role_entity + role_content, hoisted)"
    )


def test_probe_encodes_role_atom_once(hoisted_retriever, monkeypatch):
    calls = _counting_spy(monkeypatch, "encode_atom")
    results = hoisted_retriever.probe("entity_1")
    assert results
    role_content_calls = [a for a in calls
                          if a and a[0] == "__hrr_role_content__"]
    assert len(role_content_calls) == 1, (
        f"role_content atom encoded {len(role_content_calls)}x in one "
        "probe() — loop-invariant hoist regressed"
    )


def test_search_without_vectors_never_encodes(hoisted_retriever, monkeypatch):
    """Migrated DBs can have FTS candidates with NULL hrr_vector
    (MemoryStore._init_db adds the column without backfilling existing
    facts). The lazy hoist must not encode a query vector nothing will
    use — pre-fix main encoded only beneath fact.get('hrr_vector')."""
    store = hoisted_retriever.store
    store._conn.execute("UPDATE facts SET hrr_vector = NULL")
    store._conn.commit()
    calls = _counting_spy(monkeypatch, "encode_text")
    results = hoisted_retriever.search("deploy target setting")
    assert results  # candidates exist; neutral hrr_sim=0.5 path
    assert calls == [], (
        f"encode_text called {len(calls)}x with zero vector candidates — "
        "lazy hoist regressed to eager"
    )


# ---------------------------------------------------------------------------
# HRR query-space fix: search() must compare the query in the fact space.
#
# Facts store content bound to ROLE_CONTENT (encode_fact). The old
# search() compared a raw, unbound query bundle against that, which is
# unrelated to every stored fact — the HRR score was pure noise.
# encode_query binds the query to ROLE_CONTENT and restores the signal.
# ---------------------------------------------------------------------------

DIM = 1024


def test_encode_query_lives_in_the_fact_space():
    """encode_query must land in the same space as encode_fact.

    The old search() compared a raw, unbound encode_text bundle, which
    is unrelated to stored fact vectors.
    """
    fact_vec = hrr.encode_fact("the deployment rollback failed", [], DIM)
    matching = hrr.encode_query("deployment rollback", DIM)
    unrelated = hrr.encode_query("compaction threshold tuning", DIM)
    raw = hrr.encode_text("deployment rollback", DIM)  # the old encoding: unbound

    bound_sim = hrr.similarity(matching, fact_vec)
    raw_sim = hrr.similarity(raw, fact_vec)
    unrelated_sim = hrr.similarity(unrelated, fact_vec)

    # Same-space query is clearly similar to a fact sharing its content...
    assert bound_sim > 0.5, f"expected a real signal, got {bound_sim:.3f}"
    # ...and clearly beats the old raw (unbound) comparison...
    assert bound_sim > raw_sim + 0.1, (
        f"bound query ({bound_sim:.3f}) must beat the old raw comparison "
        f"({raw_sim:.3f})"
    )
    # ...and an unrelated query.
    assert bound_sim > unrelated_sim + 0.1, (
        f"matching query ({bound_sim:.3f}) must beat unrelated "
        f"({unrelated_sim:.3f})"
    )
    # Binding scrambles the vector: the bound query is unrelated to its
    # own unbound bundle.
    assert hrr.similarity(matching, raw) < 0.5


def test_search_pure_hrr_separates_matching_fact(tmp_path):
    """With HRR as the only scoring signal, the matching fact must win.

    Regression test for the fix: before it, every candidate scored ~0.5
    (noise), so the HRR signal was dead weight and a matching fact had
    no advantage.
    """
    store = MemoryStore(str(tmp_path / "query_space.db"))
    try:
        store.add_fact(
            content="deployment rollback migration stale state",
            category="project",
        )
        store.add_fact(
            content="deployment config tuned for high traffic",
            category="project",
        )
        store.add_fact(
            content="rollback procedure documented in the runbook",
            category="project",
        )
        retriever = FactRetriever(
            store=store,
            fts_weight=0.0,
            jaccard_weight=0.0,
            hrr_weight=1.0,
        )
        results = retriever.search("deployment rollback migration state", limit=3)
    finally:
        store.close()

    assert results, "expected FTS candidates"
    target = next(r for r in results if "migration stale" in r["content"])
    distractors = [r for r in results if "migration stale" not in r["content"]]
    assert distractors, "distractors must be FTS candidates too"
    margin = target["score"] - max(d["score"] for d in distractors)
    assert margin > 0.1, (
        "HRR must clearly separate the matching fact from candidates that "
        f"share only one query token (margin={margin:.3f}, "
        f"target={target['score']:.3f})"
    )
