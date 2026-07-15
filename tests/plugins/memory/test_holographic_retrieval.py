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


def test_sanitize_fts_query_never_crashes_on_fts5_specials():
    """Queries with FTS5 operator characters must not produce malformed SQL."""
    problematic = [
        'test " query',
        "test * query",
        "test (a OR b) query",
        "test^2 query",
        "test:colon query",
        "test-hyphen query",
        "a" * 1000,  # long query
    ]
    for q in problematic:
        result = FactRetriever._sanitize_fts_query(q)
        # We just need it to return a string without raising
        assert isinstance(result, str)


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


def test_prefetch_single_keyword_still_works(retriever_with_facts):
    """Single-term queries (pre-fix working case) remain working."""
    results = retriever_with_facts.search("compaction")
    assert len(results) >= 1
    assert "Compaction" in results[0]["content"] or "compaction" in results[0]["content"].lower()


def test_prefetch_stopword_only_query_empty(retriever_with_facts):
    """Pure stopword queries return zero results but don't crash."""
    # Pass to _sanitize_fts_query directly first so we know what happens
    assert FactRetriever._sanitize_fts_query("the and of") == "the and of"
    # search() handles the likely-zero-hit case gracefully
    results = retriever_with_facts.search("the and of")
    # Either zero results or it errored-gracefully to [] — both are fine
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# LIKE fallback — CJK sub-string matching and LIKE metacharacter escaping
# ---------------------------------------------------------------------------

@pytest.fixture
def retriever_with_cjk_facts(tmp_path):
    """MemoryStore seeded with CJK facts for LIKE-fallback tests."""
    db_path = tmp_path / "test_cjk.db"
    store = MemoryStore(str(db_path))
    store.add_fact(
        content="香港腾讯云服务器配置记录",
        category="infra",
        tags="香港,腾讯,cloud",
    )
    store.add_fact(
        content="sing-box 代理节点部署在东京",
        category="infra",
        tags="sing-box,proxy",
    )
    store.add_fact(
        content="数据库连接池最大100，最小10",
        category="db",
        tags="mysql,池",
    )
    store.add_fact(
        content="IP地址 43.132.88.99 属于香港机房",
        category="infra",
        tags="ip,香港",
    )
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()


def test_cjk_substring_in_content(retriever_with_cjk_facts):
    """Searching a CJK sub-string should match via LIKE fallback.

    FTS5's unicode61 tokenizer indexes ``"香港腾讯云服务器配置记录"``
    as a single token, so searching ``"香港"`` returns zero FTS5 hits.
    The LIKE fallback catches it.
    """
    results = retriever_with_cjk_facts.search("香港")
    assert len(results) >= 1
    # Should find both facts containing "香港"
    contents = [r["content"] for r in results]
    assert any("香港腾讯云" in c for c in contents)
    assert any("43.132" in c for c in contents)


def test_cjk_substring_in_tags(retriever_with_cjk_facts):
    """LIKE fallback scans tags column too, not just content."""
    results = retriever_with_cjk_facts.search("腾讯")
    assert len(results) >= 1
    contents = [r["content"] for r in results]
    assert any("香港腾讯云" in c for c in contents)


def test_like_fallback_merges_with_fts5_dedup(retriever_with_cjk_facts):
    """LIKE results that FTS5 already returned are de-duplicated."""
    # "sing-box" contains a hyphen — FTS5 sanitizer strips it and
    # matches via "sing" OR "box". LIKE also matches the sub-string.
    # The merge should not produce duplicates.
    results = retriever_with_cjk_facts.search("sing-box")
    fact_ids = [r["fact_id"] for r in results]
    assert len(fact_ids) == len(set(fact_ids)), "duplicate fact_ids in results"


def test_literal_percent_is_escaped(retriever_with_cjk_facts):
    """User input containing % must be treated as a literal, not wildcard."""
    # Insert a fact containing a literal % sign
    retriever_with_cjk_facts.store.add_fact(
        content="磁盘使用率 95%，需要扩容",
        category="ops",
    )
    # Searching "95%" should match the fact with literal "95%"
    results = retriever_with_cjk_facts.search("95%")
    assert len(results) >= 1
    assert any("95%" in r["content"] for r in results)
    # Should NOT match every fact (which a wildcard % would do)
    assert len(results) <= 2  # at most the fact(s) with "95%"


def test_literal_underscore_is_escaped(retriever_with_cjk_facts):
    """User input containing _ must be treated as a literal, not wildcard."""
    # Insert a fact with literal underscores
    retriever_with_cjk_facts.store.add_fact(
        content="环境变量 DATABASE_URL 需要更新",
        category="config",
    )
    # Searching "DATABASE_URL" should match, _ should not act as wildcard
    results = retriever_with_cjk_facts.search("DATABASE_URL")
    assert len(results) >= 1
    assert any("DATABASE_URL" in r["content"] for r in results)


def test_numeric_fragment_matches(retriever_with_cjk_facts):
    """Numeric IP fragments should match via LIKE fallback."""
    results = retriever_with_cjk_facts.search("43.132")
    assert len(results) >= 1
    assert any("43.132" in r["content"] for r in results)


def test_existing_english_search_unaffected(retriever_with_cjk_facts):
    """Well-formed English queries still work (FTS5 path unaffected)."""
    results = retriever_with_cjk_facts.search("sing-box")
    assert len(results) >= 1
    assert any("sing-box" in r["content"] for r in results)
