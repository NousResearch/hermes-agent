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
# Trigram CJK search — mixed CJK/English recall + DDL caching
# (2026-08-31, addressing AI review on #97050)
# ---------------------------------------------------------------------------

def test_mixed_cjk_english_query_recalls_english_term(tmp_path):
    """A mixed query must not silently drop the English term.

    The trigram index tokenizes English content into overlapping 3-grams
    ("deployment" -> "dep"/"epl"/"loy"...), so a MATCH token that keeps the
    whole word ("deployment") matches nothing. The fix windows Latin runs the
    same way as CJK, restoring recall for the English part.
    """
    store = MemoryStore(str(tmp_path / "mixed.db"))
    store.add_fact(content="The deployment rollback failed due to stale state.", category="project")
    store.add_fact(content="内存多大决定上下文窗口上限。", category="tool")
    retriever = FactRetriever(store=store)

    results = retriever.search("deployment内存", limit=5)
    assert results, "mixed CJK/English query should return candidates"
    # The English-bearing fact must be in the top results
    assert any("deployment" in r["content"].lower() for r in results), (
        "English term silently dropped in mixed query"
    )


def test_trigram_available_cached_no_ddl_on_repeat(tmp_path):
    """_trigram_available() must probe the build only once.

    The probe executes CREATE/DROP DDL on the live store; repeating it on
    every search adds write-lock contention. After the first call the result
    is cached and no further DDL runs.
    """
    store = MemoryStore(str(tmp_path / "ddl_cache.db"))
    retriever = FactRetriever(store=store)

    # Wrap the live connection with a counting proxy so we can observe DDL
    # without touching the read-only sqlite3.Connection.execute attribute.
    real_conn = store._conn

    class CountingConn:
        def __init__(self, real):
            self._real = real
            self.probe_ddl = 0

        def execute(self, sql, *args, **kwargs):
            if isinstance(sql, str) and "_tgram_probe" in sql:
                self.probe_ddl += 1
            return self._real.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    counting = CountingConn(real_conn)
    retriever.store._conn = counting
    try:
        assert retriever._trigram_available() in (True, False)
        first_probe_count = counting.probe_ddl
        for _ in range(5):
            retriever._trigram_available()
        assert counting.probe_ddl == first_probe_count, (
            f"DDL probe repeated: {counting.probe_ddl} after first call"
        )
    finally:
        retriever.store._conn = real_conn


def test_or_tokens_trigram_windows_latin_runs():
    """Latin runs are windowed into 3-grams, not kept whole.

    "deployment" must yield dep/epl/loy/... so the MATCH token exists in the
    trigram index; a whole-word token would match nothing.
    """
    tokens = FactRetriever._or_tokens_trigram("deployment内存")
    assert '"deployment"' not in tokens, "whole English word must not be a token"
    assert '"dep"' in tokens and '"loy"' in tokens, "English 3-grams missing"


def test_window_tokens_matches_or_tokens_latin():
    """Ranking windows and MATCH tokens must share the same splitter."""
    match = FactRetriever._or_tokens_trigram("deployment内存")
    windows = FactRetriever._window_tokens("deployment内存")
    # Every window appears (quoted) in the MATCH expression
    for w in windows:
        assert f'"{w}"' in match, f"window {w!r} missing from MATCH tokens"


def test_legacy_store_backfills_trigram_index(tmp_path):
    """A store that predates the trigram table must index its facts when reopened.

    ``facts_fts_trigram`` is an external-content table (``content=facts``), so
    creating it on a store that already holds facts leaves the index empty while
    ``SELECT count(*)`` on the table still reports every fact. Without the
    backfill, every CJK query on an upgraded store silently returns nothing.
    """
    db_path = tmp_path / "legacy.db"
    store = MemoryStore(str(db_path))
    store.add_fact(content="VPS 内存只有 1.9GB，后台常驻进程要控制在两个以内", category="project")
    # Simulate a pre-patch store: remove the trigram table and its sync triggers.
    for stmt in (
        "DROP TRIGGER IF EXISTS facts_trigram_ai",
        "DROP TRIGGER IF EXISTS facts_trigram_ad",
        "DROP TRIGGER IF EXISTS facts_trigram_au",
        "DROP TABLE IF EXISTS facts_fts_trigram",
    ):
        store._conn.execute(stmt)
    store._conn.commit()
    store.close()

    reopened = MemoryStore(str(db_path))
    try:
        indexed = reopened._conn.execute("SELECT count(*) FROM facts_fts_trigram_docsize").fetchone()[0]
        assert indexed == 1, f"legacy store reopened with an empty trigram index ({indexed} docs indexed)"
        results = FactRetriever(store=reopened).search("VPS内存多大", limit=5)
        assert results, "CJK query returned nothing on a store whose index was never built"
        assert "VPS" in results[0]["content"]
    finally:
        reopened.close()


def test_backfill_skips_already_indexed_store(tmp_path):
    """Reopening an indexed store must not rebuild it again."""
    db_path = tmp_path / "indexed.db"
    store = MemoryStore(str(db_path))
    store.add_fact(content="A股数据源以 akshare 为主，iFinD 作兜底", category="project")
    store.close()

    reopened = MemoryStore(str(db_path))
    try:
        indexed = reopened._conn.execute("SELECT count(*) FROM facts_fts_trigram_docsize").fetchone()[0]
        assert indexed == 1, "backfill must leave an already-built index alone"
        assert FactRetriever(store=reopened).search("A股数据源", limit=5)
    finally:
        reopened.close()
