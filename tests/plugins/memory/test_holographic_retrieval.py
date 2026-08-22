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
        ("context: length-probe", {"context", "length-probe"}),
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
        # Pathological case: all stopwords — fall back to quoted raw query
        assert result == f'"{query}"'
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
# CJK bigram FTS5 tests — zero-recall fix for Chinese / Japanese / Korean
# ---------------------------------------------------------------------------

# CJK bigram FTS5 tests — zero-recall fix for Chinese / Japanese / Korean
# ---------------------------------------------------------------------------


class TestCjkToFts5Bigrams:
    """Unit tests for _cjk_to_fts5_bigrams — index-side CJK tokenization."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            # Pure CJK: single chars + 2-gram splits
            ("你好世界", "你 好 世 界 你好 好世 世界"),
            ("배포는", "배 포 는 배포 포는"),          # Korean 3-char Hangul
            ("こんにちは", "こ ん に ち は こん んに にち ちは"),  # Japanese Hiragana
            # Mixed script: boundary spaces between ASCII and CJK
            ("Hermes는", "Hermes 는"),        # ASCII + single Hangul
            ("Hello世界", "Hello 世 界 世界"),       # ASCII + 2-char CJK
            ("世界Hello", "世 界 世界 Hello"),       # CJK + ASCII
            ("A는B", "A 는 B"),               # CJK between ASCII
            # Pure ASCII: unchanged
            ("hello", "hello"),
            ("hello world", "hello world"),
            # Empty / edge
            ("", ""),
            # Single CJK chars
            ("는", "는"),
            ("中", "中"),
        ],
    )
    def test_cjk_bigrams(self, text, expected):
        assert FactRetriever._cjk_to_fts5_bigrams(text) == expected


class TestSanitizeFtsQueryCjk:
    """Unit tests for _sanitize_fts_query — CJK query tokenization."""

    @pytest.mark.parametrize(
        "query,expected_bigrams",
        [
            # Chinese: 2-gram phrase (joined within a run to preserve precision)
            ("你好", {"你好"}),
            ("你好世界", {"你好 好世 世界"}),
            # Korean: Hangul bigrams
            ("배포", {"배포"}),
            ("배포는", {"배포 포는"}),
            # Japanese
            ("こんにちは", {"こん んに にち ちは"}),
            # Mixed script: CJK + ASCII both present (OR across runs)
            ("Hermes 배포", {"hermes", "배포"}),
            # Single CJK char
            ("는", {"는"}),
        ],
    )
    def test_cjk_query_tokens(self, query, expected_bigrams):
        import re as _re
        result = FactRetriever._sanitize_fts_query(query)
        matches = set(_re.findall(r'"([^"]+)"', result))
        assert matches == expected_bigrams, f"got {result!r}"


# ---------------------------------------------------------------------------
# Integration — actual SQLite FTS5 CJK recall
# ---------------------------------------------------------------------------


@pytest.fixture
def retriever_with_cjk_facts(tmp_path):
    """MemoryStore seeded with CJK facts for bigram retrieval tests."""
    db_path = tmp_path / "test_cjk_facts.db"
    store = MemoryStore(str(db_path))
    store.add_fact(
        content="小明每天下午三点去图书馆看书。",
        category="general",
    )
    store.add_fact(
        content="배포 파이프라인이 금요일 오후에 실패했습니다.",
        category="project",
    )
    store.add_fact(
        content="こんにちは、今日の天気はとてもいいです。",
        category="general",
    )
    store.add_fact(
        content="Hermes는 새로운 버전을 배포했습니다.",
        category="tool",
    )
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()


@pytest.mark.parametrize(
    "query,expected_substring",
    [
        # Chinese substring recall
        ("图书馆", "图书馆"),
        ("看书", "看书"),
        ("下午三点", "下午三点"),
        # Korean substring recall (the core zero-recall bug)
        ("배포", "배포"),
        ("파이프라인", "파이프라인"),
        ("실패", "실패"),
        # Japanese substring recall
        ("こんにちは", "こんにちは"),
        ("天気", "天気"),
        # Mixed script: search Korean term inside mixed fact
        ("배포", "배포"),  # should match both Korean and mixed facts
    ],
)
def test_cjk_substring_recall(retriever_with_cjk_facts, query, expected_substring):
    """Substring queries against CJK content must return results.

    Before bigram FTS5, FTS5's unicode61 tokenizer treated the entire
    CJK run as one token — any substring query returned zero hits.
    """
    results = retriever_with_cjk_facts.search(query)
    assert len(results) >= 1, f"zero results for CJK query {query!r}"
    contents = [r["content"] for r in results]
    assert any(expected_substring in c for c in contents), (
        f"expected {expected_substring!r} in results for query {query!r}, "
        f"got {contents}"
    )


def test_cjk_fts5_migration_creates_fts_content_column(retriever_with_cjk_facts):
    """After CJK migration, facts table must have an fts_content column."""
    store = retriever_with_cjk_facts.store
    columns = {
        row[1]
        for row in store._conn.execute("PRAGMA table_info(facts)").fetchall()
    }
    assert "fts_content" in columns, "fts_content column missing after migration"


def test_cjk_fts5_migration_uses_bigram_index(retriever_with_cjk_facts):
    """FTS5 index must use fts_content column (not old content column)."""
    store = retriever_with_cjk_facts.store
    fts_cols = {
        row[1]
        for row in store._conn.execute("PRAGMA table_info(facts_fts)").fetchall()
    }
    assert "fts_content" in fts_cols, "FTS5 not migrated to fts_content"
    assert "content" not in fts_cols, "FTS5 still using old content column"


# ---------------------------------------------------------------------------
# CJK bigram FTS5 tests — zero-recall fix for Chinese / Japanese / Korean
# ---------------------------------------------------------------------------

# CJK bigram FTS5 tests — zero-recall fix for Chinese / Japanese / Korean
# ---------------------------------------------------------------------------


class TestCjkToFts5Bigrams:
    """Unit tests for _cjk_to_fts5_bigrams — index-side CJK tokenization."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            # Pure CJK: single chars + 2-gram splits
            ("你好世界", "你 好 世 界 你好 好世 世界"),
            ("배포는", "배 포 는 배포 포는"),          # Korean 3-char Hangul
            ("こんにちは", "こ ん に ち は こん んに にち ちは"),  # Japanese Hiragana
            # Mixed script: boundary spaces between ASCII and CJK
            ("Hermes는", "Hermes 는"),        # ASCII + single Hangul
            ("Hello世界", "Hello 世 界 世界"),       # ASCII + 2-char CJK
            ("世界Hello", "世 界 世界 Hello"),       # CJK + ASCII
            ("A는B", "A 는 B"),               # CJK between ASCII
            # Pure ASCII: unchanged
            ("hello", "hello"),
            ("hello world", "hello world"),
            # Empty / edge
            ("", ""),
            # Single CJK chars
            ("는", "는"),
            ("中", "中"),
        ],
    )
    def test_cjk_bigrams(self, text, expected):
        assert FactRetriever._cjk_to_fts5_bigrams(text) == expected


class TestSanitizeFtsQueryCjk:
    """Unit tests for _sanitize_fts_query — CJK query tokenization."""

    @pytest.mark.parametrize(
        "query,expected_bigrams",
        [
            # Chinese: 2-gram phrase (joined within a run to preserve precision)
            ("你好", {"你好"}),
            ("你好世界", {"你好 好世 世界"}),
            # Korean: Hangul bigrams
            ("배포", {"배포"}),
            ("배포는", {"배포 포는"}),
            # Japanese
            ("こんにちは", {"こん んに にち ちは"}),
            # Mixed script: CJK + ASCII both present (OR across runs)
            ("Hermes 배포", {"hermes", "배포"}),
            # Single CJK char
            ("는", {"는"}),
        ],
    )
    def test_cjk_query_tokens(self, query, expected_bigrams):
        import re as _re
        result = FactRetriever._sanitize_fts_query(query)
        matches = set(_re.findall(r'"([^"]+)"', result))
        assert matches == expected_bigrams, f"got {result!r}"


# ---------------------------------------------------------------------------
# Integration — actual SQLite FTS5 CJK recall
# ---------------------------------------------------------------------------


@pytest.fixture
def retriever_with_cjk_facts(tmp_path):
    """MemoryStore seeded with CJK facts for bigram retrieval tests."""
    db_path = tmp_path / "test_cjk_facts.db"
    store = MemoryStore(str(db_path))
    store.add_fact(
        content="小明每天下午三点去图书馆看书。",
        category="general",
    )
    store.add_fact(
        content="배포 파이프라인이 금요일 오후에 실패했습니다.",
        category="project",
    )
    store.add_fact(
        content="こんにちは、今日の天気はとてもいいです。",
        category="general",
    )
    store.add_fact(
        content="Hermes는 새로운 버전을 배포했습니다.",
        category="tool",
    )
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()


@pytest.mark.parametrize(
    "query,expected_substring",
    [
        # Chinese substring recall
        ("图书馆", "图书馆"),
        ("看书", "看书"),
        ("下午三点", "下午三点"),
        # Korean substring recall (the core zero-recall bug)
        ("배포", "배포"),
        ("파이프라인", "파이프라인"),
        ("실패", "실패"),
        # Japanese substring recall
        ("こんにちは", "こんにちは"),
        ("天気", "天気"),
        # Mixed script: search Korean term inside mixed fact
        ("배포", "배포"),  # should match both Korean and mixed facts
    ],
)
def test_cjk_substring_recall(retriever_with_cjk_facts, query, expected_substring):
    """Substring queries against CJK content must return results.

    Before bigram FTS5, FTS5's unicode61 tokenizer treated the entire
    CJK run as one token — any substring query returned zero hits.
    """
    results = retriever_with_cjk_facts.search(query)
    assert len(results) >= 1, f"zero results for CJK query {query!r}"
    contents = [r["content"] for r in results]
    assert any(expected_substring in c for c in contents), (
        f"expected {expected_substring!r} in results for query {query!r}, "
        f"got {contents}"
    )


def test_cjk_fts5_migration_creates_fts_content_column(retriever_with_cjk_facts):
    """After CJK migration, facts table must have an fts_content column."""
    store = retriever_with_cjk_facts.store
    columns = {
        row[1]
        for row in store._conn.execute("PRAGMA table_info(facts)").fetchall()
    }
    assert "fts_content" in columns, "fts_content column missing after migration"


def test_cjk_fts5_migration_uses_bigram_index(retriever_with_cjk_facts):
    """FTS5 index must use fts_content column (not old content column)."""
    store = retriever_with_cjk_facts.store
    fts_cols = {
        row[1]
        for row in store._conn.execute("PRAGMA table_info(facts_fts)").fetchall()
    }
    assert "fts_content" in fts_cols, "FTS5 not migrated to fts_content"
    assert "content" not in fts_cols, "FTS5 still using old content column"
