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
# contradict() — attribute-value conflict detection (2026-08-31)
# Two facts about the same subject can be semantically near-identical prose
# yet state conflicting values for the same structured attribute (e.g. an
# expiry date). The HRR-vector similarity treats them as duplicates, so the
# original score never surfaced them. We now extract common attributes
# (dates, money, percents, capacities, versions) and flag value mismatches.
# ---------------------------------------------------------------------------

def _make_store(db_path, facts):
    store = MemoryStore(str(db_path))
    for content, category in facts:
        store.add_fact(content=content, category=category)
    return store


def test_contradict_detects_attribute_value_mismatch(tmp_path):
    """Same subject, same attribute (date), different value → surfaced."""
    store = _make_store(
        tmp_path / "attr_conflict.db",
        [
            ('"GITHUB_TOKEN" expires 2026-11-14', "tool"),
            ('"GITHUB_TOKEN" expires 2026-11-24', "tool"),
        ],
    )
    retriever = FactRetriever(store=store)
    results = retriever.contradict()
    assert results, "attribute mismatch should be reported as a contradiction"
    top = results[0]
    assert top["attribute_conflicts"] == ["date"], (
        f"expected date conflict, got {top['attribute_conflicts']}"
    )
    assert top["contradiction_score"] >= 0.3


def test_contradict_ignores_same_attribute_same_value(tmp_path):
    """Same subject, same attribute, same value → not a conflict."""
    store = _make_store(
        tmp_path / "attr_same.db",
        [
            ('"GITHUB_TOKEN" expires 2026-11-24', "tool"),
            ('"GITHUB_TOKEN" expires 2026-11-24; rotate early', "tool"),
        ],
    )
    retriever = FactRetriever(store=store)
    results = retriever.contradict()
    assert not results, "identical attribute values must not be a conflict"


def test_contradict_no_attribute_fields_preserved(tmp_path):
    """Existing behaviour: no attributes → no attribute_conflicts field noise."""
    store = _make_store(
        tmp_path / "attr_none.db",
        [
            ('"Server" 1.9GB RAM 1 core', "tool"),
            ('"Server" runs gateway and webui', "tool"),
        ],
    )
    retriever = FactRetriever(store=store)
    results = retriever.contradict()
    # Both share the "Server" entity; no shared attribute differs.
    for r in results:
        assert r["attribute_conflicts"] == []


def test_extract_attribute_values_unit():
    """Attribute extraction recognises structured values and ignores prose."""
    assert FactRetriever._extract_attribute_values(
        '"GITHUB_TOKEN" expires 2026-11-14, rotate before then'
    ) == {"date": "2026-11-14"}
    assert FactRetriever._extract_attribute_values(
        "Budget is $20/month"
    )["money"] == "$20"
    assert FactRetriever._extract_attribute_values(
        "Server 1.9GB RAM, 30G disk"
    )["capacity"] == "1.9GB"
    assert FactRetriever._extract_attribute_values(
        "no structured values here"
    ) == {}
