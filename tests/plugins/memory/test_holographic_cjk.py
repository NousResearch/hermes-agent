"""CJK segmentation tests for the holographic memory store.

Covers the fix for zero recall on Chinese content: FTS5's unicode61
tokenizer treats a contiguous CJK run as ONE token, so Chinese facts
never matched Chinese prose queries (and even embedded ASCII words like
"mihomo" couldn't match when unspaced). The fix pre-segments text with
jieba on both the write side (standalone facts_fts stores segmented
copies) and the query side (_sanitize_fts_query / _tokenize /
hrr.encode_text), with transparent migration of legacy external-content
databases and self-healing on index drift.
"""
from __future__ import annotations

import re
import sqlite3

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic import textseg
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore

requires_jieba = pytest.mark.skipif(
    not textseg._HAS_JIEBA, reason="jieba not installed"
)

_ZH_FACT_RUSTDESK = (
    "向日葵已卸载，远程桌面已替换为RustDesk。ID: 317897904，"
    "systemd服务rustdesk.service开机自启。"
)
_ZH_FACT_MIHOMO = (
    "mihomo代理订阅已更换新机场，节点实测可用。"
    "外网不通时先测节点连通性再更新订阅。"
)
_EN_FACT = "The Thursday deployment rollback failed because of stale migration state."


def _quoted_tokens(fts_query: str) -> set[str]:
    return set(re.findall(r'"([^"]+)"', fts_query))


# ---------------------------------------------------------------------------
# textseg unit tests
# ---------------------------------------------------------------------------

def test_tokenize_ascii_is_plain_whitespace_split():
    # ASCII text must keep byte-identical historical behavior, jieba or not.
    assert textseg.tokenize("restart the RustDesk service") == [
        "restart", "the", "RustDesk", "service",
    ]
    assert textseg.segment_for_index(_EN_FACT) == _EN_FACT


def test_tokenize_without_jieba_falls_back_to_whitespace(monkeypatch):
    monkeypatch.setattr(textseg, "_HAS_JIEBA", False)
    assert textseg.tokenize("帮我更新mihomo的订阅") == ["帮我更新mihomo的订阅"]
    assert textseg.segment_for_index("帮我更新mihomo的订阅") == "帮我更新mihomo的订阅"
    assert textseg.current_fts_mode() == textseg.FTS_MODE_PLAIN


@requires_jieba
def test_tokenize_segments_cjk_and_extracts_ascii_islands():
    tokens = textseg.tokenize("帮我更新mihomo的订阅")
    assert "mihomo" in tokens          # unspaced ASCII island becomes a token
    assert "订阅" in tokens             # real word, not the whole clause
    assert "帮我更新mihomo的订阅" not in tokens


@requires_jieba
def test_segment_for_index_space_joins_cjk():
    segmented = textseg.segment_for_index(_ZH_FACT_MIHOMO)
    assert " " in segmented
    assert "mihomo" in segmented.split()


# ---------------------------------------------------------------------------
# Query sanitizer
# ---------------------------------------------------------------------------

@requires_jieba
def test_sanitize_fts_query_segments_chinese_prose():
    result = FactRetriever._sanitize_fts_query("帮我更新mihomo的订阅")
    tokens = _quoted_tokens(result)
    assert "mihomo" in tokens
    assert "订阅" in tokens
    # The unsegmented clause must not survive as a single phrase literal.
    assert not any("帮我更新" in t for t in tokens)


@requires_jieba
def test_sanitize_fts_query_drops_chinese_stopwords():
    result = FactRetriever._sanitize_fts_query("帮我看一下这个订阅现在怎么样")
    tokens = _quoted_tokens(result)
    assert "订阅" in tokens
    assert "帮我" not in tokens
    assert "这个" not in tokens
    assert "现在" not in tokens


def test_sanitize_fts_query_english_behavior_unchanged():
    result = FactRetriever._sanitize_fts_query(
        "what happened with the deployment rollback"
    )
    assert _quoted_tokens(result) == {"happened", "deployment", "rollback"}


# ---------------------------------------------------------------------------
# End-to-end: Chinese prose query → Chinese fact
# ---------------------------------------------------------------------------

@pytest.fixture
def zh_retriever(tmp_path):
    store = MemoryStore(str(tmp_path / "zh_facts.db"))
    store.add_fact(content=_ZH_FACT_RUSTDESK, category="general")
    store.add_fact(content=_ZH_FACT_MIHOMO, category="general")
    store.add_fact(content=_EN_FACT, category="project")
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()


@requires_jieba
def test_chinese_prose_query_hits_chinese_fact(zh_retriever):
    results = zh_retriever.search("启动远程桌面")
    assert results, "Chinese prose query must match the RustDesk fact"
    assert "远程桌面" in results[0]["content"]


@requires_jieba
def test_unspaced_ascii_island_query_hits(zh_retriever):
    results = zh_retriever.search("帮我更新mihomo的订阅")
    assert results, "embedded ASCII word must be matchable without spaces"
    assert "mihomo" in results[0]["content"]


@requires_jieba
def test_english_query_still_hits_english_fact(zh_retriever):
    results = zh_retriever.search("deployment rollback")
    assert results
    assert "deployment rollback" in results[0]["content"]


def test_store_search_facts_uses_same_pipeline(tmp_path):
    store = MemoryStore(str(tmp_path / "sf.db"))
    try:
        store.add_fact(content=_EN_FACT, category="project")
        assert store.search_facts("deployment rollback")
    finally:
        store.close()


@requires_jieba
def test_add_fact_indexes_segmented_copy(tmp_path):
    store = MemoryStore(str(tmp_path / "seg.db"))
    try:
        fact_id = store.add_fact(content=_ZH_FACT_MIHOMO, category="general")
        fts_content = store._conn.execute(
            "SELECT content FROM facts_fts WHERE rowid = ?", (fact_id,)
        ).fetchone()[0]
        original = store._conn.execute(
            "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()[0]
        assert original == _ZH_FACT_MIHOMO          # source text untouched
        assert fts_content != original               # index copy is segmented
        assert "mihomo" in fts_content.split()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Migration + self-healing
# ---------------------------------------------------------------------------

_LEGACY_SCHEMA = """
CREATE TABLE facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE VIRTUAL TABLE facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;
"""


def _make_legacy_db(path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(_LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO facts (content, category, tags, trust_score) VALUES (?,?,?,?)",
        (_ZH_FACT_RUSTDESK, "general", "", 0.5),
    )
    conn.execute(
        "INSERT INTO facts (content, category, tags, trust_score) VALUES (?,?,?,?)",
        (_EN_FACT, "project", "", 0.5),
    )
    conn.commit()
    conn.close()


def test_legacy_external_content_db_migrates(tmp_path):
    db = tmp_path / "legacy.db"
    _make_legacy_db(db)

    store = MemoryStore(str(db))
    try:
        ddl = store._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='facts_fts'"
        ).fetchone()[0]
        assert "content=" not in ddl, "external-content FTS must be replaced"
        triggers = store._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger'"
            " AND name IN ('facts_ai','facts_ad','facts_au')"
        ).fetchone()[0]
        assert triggers == 0, "legacy triggers must be dropped"
        version = store._conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == textseg.current_fts_mode()
        # All rows reindexed
        n_facts = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        n_fts = store._conn.execute("SELECT COUNT(*) FROM facts_fts").fetchone()[0]
        assert n_facts == n_fts == 2
        # English recall works in both modes
        retriever = FactRetriever(store=store)
        assert retriever.search("deployment rollback")
    finally:
        store.close()


@requires_jieba
def test_migrated_legacy_db_gains_chinese_recall(tmp_path):
    db = tmp_path / "legacy_zh.db"
    _make_legacy_db(db)

    store = MemoryStore(str(db))
    try:
        retriever = FactRetriever(store=store)
        results = retriever.search("启动远程桌面")
        assert results, "post-migration Chinese prose query must hit"
        assert "远程桌面" in results[0]["content"]
    finally:
        store.close()


def test_fts_self_heals_after_unindexed_write(tmp_path):
    """A fact written by a stale process (no FTS maintenance) is recovered
    by the row-count drift check on the next open."""
    db = tmp_path / "drift.db"
    store = MemoryStore(str(db))
    store.add_fact(content=_EN_FACT, category="project")
    # Simulate a pre-migration process appending directly to facts,
    # bypassing the standalone FTS index entirely.
    store._conn.execute(
        "INSERT INTO facts (content, category, tags, trust_score) VALUES (?,?,?,?)",
        ("orphan fact about the venice context probe", "tool", "", 0.5),
    )
    store._conn.commit()
    store.close()

    store2 = MemoryStore(str(db))
    try:
        n_facts = store2._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        n_fts = store2._conn.execute("SELECT COUNT(*) FROM facts_fts").fetchone()[0]
        assert n_facts == n_fts == 2, "drift check must rebuild the index"
        assert store2.search_facts("venice context probe")
    finally:
        store2.close()


@requires_jieba
def test_mode_flip_reindexes_on_reopen(tmp_path, monkeypatch):
    """DB built without jieba upgrades to segmented mode when jieba appears."""
    db = tmp_path / "modeflip.db"
    monkeypatch.setattr(textseg, "_HAS_JIEBA", False)
    store = MemoryStore(str(db))
    store.add_fact(content=_ZH_FACT_MIHOMO, category="general")
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == textseg.FTS_MODE_PLAIN
    store.close()
    monkeypatch.undo()

    store2 = MemoryStore(str(db))
    try:
        assert (
            store2._conn.execute("PRAGMA user_version").fetchone()[0]
            == textseg.FTS_MODE_SEGMENTED
        )
        retriever = FactRetriever(store=store2)
        assert retriever.search("帮我更新mihomo的订阅")
    finally:
        store2.close()
