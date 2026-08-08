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
    """Parity: hoisted search() must produce the exact pre-fix results.

    Replicates the pre-fix loop (query vector encoded per candidate) as the
    reference and compares full scored output for exact equality.
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
            query_vec = hrr.encode_text(query, r.hrr_dim)  # per-candidate
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

    # ``retrieval_count`` is bookkeeping, not scoring: the reference rows are
    # re-read from the DB *after* search() has already recorded its retrievals,
    # so they carry the incremented value while the returned dicts hold the
    # pre-increment snapshot. Compare everything else — the hoist parity this
    # test exists to prove is about the scored output.
    for fact in (*new_results, *old_results):
        fact.pop("retrieval_count", None)

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
# retrieval_count bookkeeping — the retriever is the live read path, so it
# must record retrievals too. Previously only MemoryStore.search_facts (which
# has no callers) incremented, leaving the counter permanently zero.
# ---------------------------------------------------------------------------

def _counts(store):
    return dict(
        store._conn.execute("SELECT fact_id, retrieval_count FROM facts").fetchall()
    )


def test_search_increments_retrieval_count(retriever_with_facts):
    """search() bumps the counter for facts it actually returns."""
    store = retriever_with_facts.store
    assert set(_counts(store).values()) == {0}

    results = retriever_with_facts.search("compaction")
    assert len(results) >= 1

    after = _counts(store)
    for fact in results:
        assert after[fact["fact_id"]] == 1
    # Facts that were not returned stay untouched
    returned = {f["fact_id"] for f in results}
    assert all(c == 0 for fid, c in after.items() if fid not in returned)


def test_probe_increments_retrieval_count(retriever_with_facts):
    """probe() shares the same bookkeeping path as search()."""
    store = retriever_with_facts.store
    results = retriever_with_facts.probe("compaction")

    after = _counts(store)
    for fact in results:
        assert after[fact["fact_id"]] >= 1


def test_zero_hit_search_records_nothing(retriever_with_facts):
    """An empty result set must not touch any counter."""
    store = retriever_with_facts.store
    before = _counts(store)
    retriever_with_facts.search("zzzznomatchzzzz")
    assert _counts(store) == before


def test_record_retrievals_is_idempotent_per_call(retriever_with_facts):
    """Two searches produce two increments — the counter accumulates."""
    store = retriever_with_facts.store
    first = retriever_with_facts.search("compaction")
    fact_id = first[0]["fact_id"]
    assert _counts(store)[fact_id] == 1
    retriever_with_facts.search("compaction")
    assert _counts(store)[fact_id] == 2


def test_record_retrievals_empty_list_is_noop(retriever_with_facts):
    """Guard clause: empty id list must not issue a malformed UPDATE."""
    store = retriever_with_facts.store
    before = _counts(store)
    store.record_retrievals([])
    assert _counts(store) == before


def test_related_increments_retrieval_count(retriever_with_facts):
    """related() returns facts to the caller, so it counts them too."""
    store = retriever_with_facts.store
    results = retriever_with_facts.related("deployment")
    assert len(results) >= 1

    after = _counts(store)
    for fact in results:
        assert after[fact["fact_id"]] >= 1
    returned = {f["fact_id"] for f in results}
    assert all(c == 0 for fid, c in after.items() if fid not in returned)


def test_reason_increments_retrieval_count(retriever_with_facts):
    """reason() is a live fact_store read path — its results count."""
    store = retriever_with_facts.store
    results = retriever_with_facts.reason(["deployment", "migration"])
    assert len(results) >= 1

    after = _counts(store)
    for fact in results:
        assert after[fact["fact_id"]] >= 1


def test_probe_category_bank_branch_increments(retriever_with_facts):
    """The category-bank branch of probe() returns via _score_facts_by_vector.

    That helper has its own return path, which was previously uncounted even
    though probe()'s direct-scoring branch was fixed.
    """
    store = retriever_with_facts.store
    results = retriever_with_facts.probe("compaction", category="tool")
    assert len(results) >= 1

    after = _counts(store)
    for fact in results:
        assert after[fact["fact_id"]] >= 1


def test_score_facts_by_vector_increments_directly(retriever_with_facts):
    """_score_facts_by_vector counts whatever it hands back."""
    import numpy as np

    from plugins.memory.holographic import holographic as hrr

    store = retriever_with_facts.store
    target = hrr.encode_text("compaction", retriever_with_facts.hrr_dim)
    results = retriever_with_facts._score_facts_by_vector(target, limit=2)
    assert len(results) >= 1

    after = _counts(store)
    for fact in results:
        assert after[fact["fact_id"]] >= 1
    assert isinstance(target, np.ndarray)


def test_list_facts_increments_retrieval_count(retriever_with_facts):
    """fact_store(action='list') surfaces facts, so list_facts() counts them."""
    store = retriever_with_facts.store
    assert set(_counts(store).values()) == {0}

    facts = store.list_facts()
    assert len(facts) == 3

    after = _counts(store)
    for fact in facts:
        assert after[fact["fact_id"]] == 1


def test_list_facts_respects_limit_when_counting(retriever_with_facts):
    """Only the truncated result set is counted, not every scanned row."""
    store = retriever_with_facts.store
    facts = store.list_facts(limit=1)
    assert len(facts) == 1

    after = _counts(store)
    assert after[facts[0]["fact_id"]] == 1
    assert sum(after.values()) == 1


def test_list_facts_empty_result_records_nothing(retriever_with_facts):
    """A filter that matches nothing must leave every counter alone."""
    store = retriever_with_facts.store
    before = _counts(store)
    assert store.list_facts(min_trust=99.0) == []
    assert _counts(store) == before
