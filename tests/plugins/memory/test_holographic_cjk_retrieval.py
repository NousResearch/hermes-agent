"""CJK retrieval regression tests for the holographic fact store (#85524, #83593, #73868).

The default unicode61 FTS5 tokenizer stores an unsegmented CJK run as ONE token, so both
fact content and a natural-language Chinese query collapse to single long tokens that never
match — per-turn prefetch and manual search silently return zero hits for entire language
families. The fix indexes a shared bigram token soup in ``facts.search_text`` and expands
queries with the same tokenizer, so a 2+ char CJK substring matches regardless of spacing.

These are behavior contracts: query-form-independence (spaced == packed == natural sentence)
and cross-language recall, not snapshots of any tokenizer output.
"""
from __future__ import annotations

import pytest

# numpy is OPTIONAL for these paths: the bigram tokenizer is pure regex and FactRetriever
# redistributes weights when numpy is absent (#17350 tracks the silent degradation, but
# CJK keyword retrieval must be tested in exactly that numpy-less environment too).
from plugins.memory.holographic.cjk_tokenize import build_search_text, tokenize_for_index
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore

CJK_FACTS = [
    ("PCB供应商账期口径：月结30天=本月底对账起算30天+次月10日电汇现金", "账期 供应商 商务"),
    ("本地OCR首选GLM-OCR：识图优先用本地模型而非云端", "OCR 识图 本地"),
    ("星河电路黄永强常驻深圳沙井", "星河 黄永强"),
]

# (query in each form, the fact content it must surface)
QUERY_FORMS = {
    "账期": CJK_FACTS[0][0],
    "供应商账期": CJK_FACTS[0][0],
    "满坤的账期是多少": CJK_FACTS[0][0],          # natural sentence, unseen entity word
    "月结": CJK_FACTS[0][0],
    "怎么识图": CJK_FACTS[1][0],
    "本地OCR": CJK_FACTS[1][0],
    "识图用什么模型": CJK_FACTS[1][0],
    "黄永强背景": CJK_FACTS[2][0],
    "星河黄永强常驻哪里": CJK_FACTS[2][0],
}


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=tmp_path / "facts.db")
    for content, tags in CJK_FACTS:
        s.add_fact(content, tags=tags)
    yield s
    s.close()


@pytest.fixture
def retriever(store):
    return FactRetriever(store=store)


# ---------------------------------------------------------------------------
# tokenizer unit contracts
# ---------------------------------------------------------------------------

def test_tokenize_cjk_run_becomes_bigrams():
    # A 4-char CJK run yields 3 overlapping bigrams — the contract that makes substring
    # matching work for any 2+ char slice of the query or the content.
    assert tokenize_for_index("供应商账期") == ["供应", "应商", "商账", "账期"]


def test_tokenize_latin_unchanged_and_cjk_mixed():
    assert tokenize_for_index("OCR 识图") == ["ocr", "识图"]
    assert tokenize_for_index("P160金顺") == ["p160", "金顺"]


def test_single_cjk_char_passes_through():
    assert tokenize_for_index("图") == ["图"]


def test_index_and_query_share_the_same_bigrams():
    # THE core invariant (#85524): query-side and index-side tokenization must agree,
    # or bigrams never line up and recall stays zero.
    content = "供应商账期口径月结"
    query_bigrams = set(tokenize_for_index("供应商账期"))
    index_bigrams = set(tokenize_for_index(content))
    assert query_bigrams & index_bigrams, "query bigrams must overlap the indexed bigrams"


def test_build_search_text_joins_with_spaces():
    # Space-joined so unicode61 (which splits on whitespace) sees each bigram as a token.
    assert build_search_text("供应商", "账期") == "供应 应商 账期"


# ---------------------------------------------------------------------------
# retrieval integration contracts (real DB, real FTS5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected_content", sorted(QUERY_FORMS.items()))
def test_cjk_query_forms_all_retrieve(retriever, query, expected_content):
    """Packed (no-space) natural-language CJK queries must surface the fact — the exact
    form per-turn prefetch receives. Spaced and packed forms are equivalent."""
    results = retriever.search(query, limit=5)
    assert any(expected_content.startswith(r["content"][:20]) for r in results), \
        f"query {query!r} returned {[r['content'][:30] for r in results]}"


def test_spaced_and_packed_forms_are_equivalent(retriever):
    """Same words with/without spaces must return overlapping result sets."""
    spaced = {r["content"] for r in retriever.search("供应商 账期", limit=5)}
    packed = {r["content"] for r in retriever.search("供应商账期", limit=5)}
    assert spaced & packed


def test_english_recall_unchanged(retriever, store):
    """English/latin retrieval must stay byte-identical behavior — no regression from the
    bigram path (latin runs pass through untouched)."""
    store.add_fact("The deployment rollback failed on staging", tags="deploy rollback")
    results = retriever.search("deployment rollback", limit=5)
    assert results and "rollback" in results[0]["content"]


def test_retrieval_count_increments_on_search(store, retriever):
    """#78801 contract: every surfaced fact's retrieval_count must advance — the column
    was declared and documented but never written by any read path."""
    before = {r["fact_id"]: r["retrieval_count"] for r in store.list_facts(limit=10)}
    retriever.search("账期", limit=5)
    after = {r["fact_id"]: r["retrieval_count"] for r in store.list_facts(limit=10)}
    assert any(after[fid] > before[fid] for fid in after), \
        f"retrieval_count did not move: before={before} after={after}"


def test_pre_existing_db_gets_migrated(store, retriever, tmp_path):
    """A DB written by the OLD schema (content+tags FTS only, no search_text) must be
    upgraded transparently on first MemoryStore open: bigrams backfilled, FTS rebuilt,
    and CJK queries immediately retrievable."""
    db_path = tmp_path / "old_schema.db"
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL UNIQUE, "
                 "category TEXT DEFAULT 'general', tags TEXT DEFAULT '', trust_score REAL DEFAULT 0.5, "
                 "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0, "
                 "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                 "hrr_vector BLOB)")
    conn.execute("CREATE VIRTUAL TABLE facts_fts USING fts5(content, tags, content=facts, content_rowid=fact_id)")
    conn.execute("INSERT INTO facts (content, tags) VALUES (?, ?)", CJK_FACTS[0])
    conn.execute("INSERT INTO facts_fts(rowid, content, tags) VALUES (1, ?, ?)", CJK_FACTS[0])
    conn.commit()
    conn.close()

    migrated = MemoryStore(db_path=db_path)  # opening triggers the migration
    try:
        r = FactRetriever(store=migrated)
        results = r.search("满坤的账期是多少", limit=5)
        assert any(CJK_FACTS[0][0].startswith(x["content"][:20]) for x in results)
    finally:
        migrated.close()
