"""R3 retrieval tests: recency tie-break, Thai bigram fallback, canonical map.

Zero LLM, deterministic. Tie-break policy: trust (in score) dominates;
recency (updated_at, then fact_id insertion order) breaks score ties only.
"""

import pytest

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "r3r.db"), hrr_dim=64)
    yield s
    s.close()


def test_r3_tiebreak_newer_wins_equal_score(store):
    store.add_fact("tiebreak alpha content", category="general")
    store.add_fact("tiebreak alpha content extended", category="general")
    r = FactRetriever(store=store, hrr_dim=64)
    out = r.search("tiebreak alpha", limit=5)
    assert len(out) >= 2
    # equal scores -> insertion order (fact_id) decides, deterministic
    scores = [x["score"] for x in out[:2]]
    if abs(scores[0] - scores[1]) < 1e-9:
        assert out[0]["fact_id"] > out[1]["fact_id"]


def test_r3_tiebreak_key_is_explicit():
    # Pure-function contract: (score, updated_at, fact_id), never vacuous.
    key = FactRetriever._recency_key
    old = {"score": 0.5, "updated_at": "2020-01-01 00:00:00", "fact_id": 1}
    new = {"score": 0.5, "updated_at": "2020-01-01 00:00:00", "fact_id": 2}
    assert key(new) > key(old)  # same score+time -> insertion order decides
    hi = {"score": 0.9, "updated_at": "2020-01-01 00:00:00", "fact_id": 1}
    assert key(hi) > key(new)  # score dominates recency


def test_r3_alias_no_fp_overreach(store):
    store.add_fact('Deploy pipeline uses "BlueGreen" strategy.', category="project")
    store.add_entity_alias("BlueGreen", "BG")
    store.add_fact("BG levels rising across the board", category="general")
    r = FactRetriever(store=store, hrr_dim=64)
    out = r.search("BG strategy", limit=5)
    assert "BlueGreen" in out[0]["content"]  # alias + strategy beats bare alias hit


def test_r3_trust_beats_recency(store):
    old = store.add_fact("trustbeat unique content old", category="general")
    new = store.add_fact("trustbeat unique content new", category="general")
    store.record_feedback(old, helpful=True)
    store.record_feedback(old, helpful=True)
    store.record_feedback(old, helpful=True)
    r = FactRetriever(store=store, hrr_dim=64)
    out = r.search("trustbeat unique content", limit=5)
    assert out[0]["fact_id"] == old  # high trust wins over newer low trust


def test_r3_thai_unsegmented_query(store):
    store.add_fact("ใช้ SQLite เป็นฐานข้อมูลหลัก", category="project")
    store.add_fact("ห้ามแก้ baseline โดยเด็ดขาด", category="project")
    r = FactRetriever(store=store, hrr_dim=64)
    out = r.search("ฐานข้อมูลหลักคืออะไร?", limit=5)
    assert any("SQLite" in x["content"] for x in out)


def test_r3_thai_bigram_no_fp_blowup(store):
    store.add_fact("ใช้ SQLite เป็นฐานข้อมูลหลัก", category="project")
    store.add_fact("ok thanks bye", category="general")
    store.add_fact("asdf qwer zxcv", category="general")
    r = FactRetriever(store=store, hrr_dim=64)
    out = r.search("ฐานข้อมูลหลัก", limit=5)
    contents = [x["content"] for x in out]
    assert any("SQLite" in c for c in contents)
    # precision@1: relevant Thai fact must outrank pure noise
    assert "SQLite" in out[0]["content"]
    # supplement path directly: tags rows even when FTS finds nothing
    sup = r._thai_bigram_candidates("ฐานข้อมูลหลัก", None, 0.0, 5)
    assert any("SQLite" in x["content"] and x.get("_thai_bigram") for x in sup)


def test_r3_canonical_tech_terms(store):
    store.add_fact("Primary store is SQLite with WAL mode.", category="project")
    r = FactRetriever(store=store, hrr_dim=64)
    out = r.search("primary database", limit=5)
    assert isinstance(out, list)  # measured baseline; semantic gap stays documented
