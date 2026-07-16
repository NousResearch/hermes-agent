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
        # all stopwords → safely quoted implicit-AND fallback
        ("the and of", {"the", "and", "of"}),
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

    # OR-joined phrase literals: `"tok1" OR "tok2" OR ...`
    # Extract the tokens between quotes, order-independent.
    import re
    matches = re.findall(r'"([^"]+)"', result)
    assert set(matches) == expected_tokens, f"got {result!r}"


def test_sanitize_fts_query_splits_contiguous_cjk_question_scaffolding():
    assert FactRetriever._sanitize_fts_query("课程平台的支付链路是什么？") == (
        '"课程平台"* OR "支付链路"*'
    )


def test_sanitize_fts_query_splits_mixed_language_boundaries():
    assert FactRetriever._sanitize_fts_query("Hermes的部署状态是什么？") == (
        '"hermes" OR "部署状态"*'
    )


def test_sanitize_fts_query_does_not_split_cjk_words_containing_particle_chars():
    assert FactRetriever._sanitize_fts_query("目的地是什么？") == '"目的地"*'
    assert FactRetriever._sanitize_fts_query("有目的地前往北京吗？") == (
        '"有目的地前往北京"*'
    )
    assert FactRetriever._sanitize_fts_query("的确如此") == '"的确如此"*'


def test_sanitize_fts_query_preserves_english_token_behavior():
    assert FactRetriever._sanitize_fts_query("deployment rollback") == (
        '"deployment" OR "rollback"'
    )


def test_sanitize_fts_query_never_crashes_on_fts5_specials():
    """Queries with FTS5 operator characters must not produce malformed SQL."""
    problematic = [
        'test " query',
        "test * query",
        "test (a OR b) query",
        "test^2 query",
        "test:colon query",
        "test-hyphen query",
        "\x00OR",
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
    store.add_fact(
        content="课程平台支付链路在运营手册里有唯一记录。检索关键词：课程平台 支付链路",
        category="project",
    )
    store.add_fact(
        content="Hermes的部署状态正常。检索关键词：Hermes 部署状态",
        category="project",
    )
    store.add_fact(
        content="C language and R language are both supported.",
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


def test_retriever_normalizes_cjk_question_against_indexable_keywords(
    retriever_with_facts,
):
    results = retriever_with_facts.search("课程平台的支付链路是什么？")

    assert any("支付链路" in result["content"] for result in results)


def test_memory_store_search_facts_recalls_mixed_language_question(
    retriever_with_facts,
):
    results = retriever_with_facts.store.search_facts("Hermes的部署状态是什么？")

    assert any("Hermes 部署状态" in result["content"] for result in results)


def test_memory_store_search_facts_quotes_operator_only_fallback(
    retriever_with_facts,
):
    assert retriever_with_facts.store.search_facts("OR") == []


def test_memory_store_search_facts_handles_nul_fallback(retriever_with_facts):
    assert retriever_with_facts.store.search_facts("\x00OR") == []


def test_short_token_fallback_preserves_implicit_and_semantics(retriever_with_facts):
    results = retriever_with_facts.store.search_facts("c r")

    assert any("C language and R language" in result["content"] for result in results)


def test_prefetch_stopword_only_query_empty(retriever_with_facts):
    """Pure stopword queries return zero results but don't crash."""
    # Pass to _sanitize_fts_query directly first so we know what happens
    assert FactRetriever._sanitize_fts_query("the and of") == '"the" "and" "of"'
    # search() handles the likely-zero-hit case gracefully
    results = retriever_with_facts.search("the and of")
    # Either zero results or it errored-gracefully to [] — both are fine
    assert isinstance(results, list)
