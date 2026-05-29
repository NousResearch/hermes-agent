"""Regression tests for the FTS5 OR-fallback in the holographic memory plugin.

The bug: SQLite FTS5 with the default ``simple`` tokenizer treats whitespace
between tokens as logical AND, and does no CJK word segmentation. Multi-token
Chinese queries from agent natural language therefore almost always missed,
even when the data contained every key token, because compound CJK tokens
in the stored content did not match the agent's individual word boundaries.

These tests pin the OR-fallback behavior at both layers the tool can reach:
``MemoryStore.search_facts`` (direct caller) and ``FactRetriever._fts_candidates``
(the path used by the agent-facing ``fact_store(action='search')`` tool).
"""

from __future__ import annotations

import pytest

from plugins.memory.holographic.store import (
    MemoryStore,
    _build_or_fallback,
    _is_explicit_fts5_query,
)
from plugins.memory.holographic.retrieval import FactRetriever


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


def test_or_fallback_quotes_each_token() -> None:
    out = _build_or_fallback("Kerpen 仓库 DPD")
    assert out == '"Kerpen" OR "仓库" OR "DPD"'


def test_or_fallback_skips_single_token() -> None:
    assert _build_or_fallback("DPD") == ""
    assert _build_or_fallback("  仓库  ") == ""


def test_or_fallback_strips_embedded_quotes() -> None:
    # Embedded double-quotes would break the OR construction; strip them.
    assert _build_or_fallback('foo "bar" baz') == '"foo" OR "bar" OR "baz"'


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Kerpen DPD", False),
        ('"Kerpen"', True),
        ("Kerpen OR DPD", True),
        ("Kerpen AND DPD", True),
        ("Kerpen*", True),
        ("(Kerpen)", True),
        ("Kerpen NOT DPD", True),
        ("Kerpen NEAR DPD", True),
        ("中文 普通 查询", False),
    ],
)
def test_explicit_fts5_detection(query: str, expected: bool) -> None:
    assert _is_explicit_fts5_query(query) is expected


# ---------------------------------------------------------------------------
# End-to-end through the store
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    s = MemoryStore(db_path=str(tmp_path / "test.db"))
    # Content modeled on a real-world contact note. The compound CJK token
    # "仓库的发货对接和提货事情" is what the simple tokenizer keeps as a
    # single token — a bare query of "仓库" cannot match it under AND.
    s.add_fact(
        "联系人 > +49 172 2363399：意大利裔德国人，DPD 快递公司的客户经理，"
        "足球教练；正在帮安排 Kerpen 仓库的发货对接和提货事情。",
        category="contact",
    )
    s.add_fact(
        "Davide Venturiello 是 DPD 客户经理/足球教练，email Davide.Venturiello@dpd.de。",
        category="contact",
    )
    s.add_fact(
        "Aachen 的奔驰 GLE 车灯由 Abdullah (AutoFabrik Aachen) 修理。",
        category="contact",
    )
    return s


def test_store_search_multi_token_cjk_query_hits(store: MemoryStore) -> None:
    # Pre-patch: this returned [] because '仓库' and '客户经理' don't exist
    # as standalone tokens (FTS5 simple tokenizer doesn't segment CJK), so
    # the AND-conjunctive MATCH failed even though 'DPD', 'Kerpen', and
    # '2363399' are individually present.
    results = store.search_facts(
        "Kerpen 仓库 DPD 客户经理 足球教练 2363399",
        min_trust=0.0,
        limit=5,
    )
    assert results, "OR-fallback should rescue this multi-token CJK query"
    contents = " ".join(r["content"] for r in results)
    assert "DPD" in contents


def test_store_search_explicit_fts5_query_is_left_alone(store: MemoryStore) -> None:
    # Explicit OR — should pass through without re-rewriting.
    results = store.search_facts('"DPD" OR "GLE"', min_trust=0.0, limit=5)
    assert results
    contents = " ".join(r["content"] for r in results)
    assert "DPD" in contents and "GLE" in contents


def test_store_search_unmatched_or_returns_empty(store: MemoryStore) -> None:
    # When none of the tokens exist, OR-fallback should still return nothing.
    results = store.search_facts(
        "完全不存在的内容 xyzzy nonexistenttoken",
        min_trust=0.0,
        limit=5,
    )
    assert results == []


# ---------------------------------------------------------------------------
# End-to-end through the retriever (the path the agent tool uses)
# ---------------------------------------------------------------------------


def test_retriever_search_multi_token_cjk_query_hits(store: MemoryStore) -> None:
    # Pre-patch: FactRetriever.search delegated stage-1 to _fts_candidates,
    # which used raw MATCH ?. With no candidates from stage 1, stage 2/3
    # never ran and the agent-facing tool returned `{"count": 0}` even when
    # the answer was clearly in the store.
    retriever = FactRetriever(store)
    results = retriever.search(
        "Kerpen 仓库 DPD 联系人 Davide Venturiello",
        min_trust=0.0,
        limit=5,
    )
    assert results, "Retriever should surface DPD/Davide rows for this query"
    contents = " ".join(r["content"] for r in results)
    assert "DPD" in contents or "Davide" in contents


def test_retriever_single_token_still_works(store: MemoryStore) -> None:
    # Single token never triggers the OR-fallback (build returns ""), so
    # this also pins that the patch does not regress simple queries.
    retriever = FactRetriever(store)
    results = retriever.search("DPD", min_trust=0.0, limit=5)
    assert results
