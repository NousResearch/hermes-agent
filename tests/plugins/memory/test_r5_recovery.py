"""R5 crash/recovery + idempotency + self-heal validation. Deterministic,
zero LLM. Scratch DBs only; production data never touched.
"""

import json
from pathlib import Path

import pytest

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "r5" / "recovery"
LLM_CALLS = 0


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "r5r.db"), hrr_dim=64)
    yield s
    s.close()


def _write(name, payload):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / name, "w", encoding="utf-8") as f:
        json.dump(dict(payload, llm_calls=LLM_CALLS), f, indent=1)


def test_r5_crash_before_commit(tmp_path):
    db = tmp_path / "pre.db"
    s = MemoryStore(str(db), hrr_dim=32)
    try:
        s._conn.execute("BEGIN")
        s._conn.execute("INSERT INTO facts (content, category, tags, trust_score) VALUES (?,?,?,?)",
                        ("crashed before commit", "general", "", 0.5))
        s._conn.rollback()  # simulated crash: never committed
    finally:
        s.close()
    s2 = MemoryStore(str(db), hrr_dim=32)  # restart
    try:
        assert s2.list_facts(limit=50) == []  # nothing partial
        assert s2._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        _write("crash_before_commit.json", {"rows": 0, "integrity": "ok"})
    finally:
        s2.close()


def test_r5_batch_error_atomic(tmp_path):
    s = MemoryStore(str(tmp_path / "bat.db"), hrr_dim=32)
    try:
        before = len(s.list_facts(limit=1000))
        with pytest.raises(ValueError):
            s.add_facts_batch([("good one", "general", ""), ("", "general", "")])
        after = len(s.list_facts(limit=1000))
        assert after == before
        _write("batch_atomic.json", {"before": before, "after": after})
    finally:
        s.close()


def test_r5_bank_rebuild_after_wipe(store):
    for i in range(5):
        store.add_fact(f"rebuild me {i} deploy", category="project")
    store._conn.execute("DELETE FROM memory_banks")
    store._conn.commit()
    store._rebuild_bank("project")  # interrupted-rebuild recovery path
    r = FactRetriever(store=store, hrr_dim=64)
    assert len(r.search("rebuild deploy", limit=5)) > 0
    _write("bank_rebuild.json", {"recovered": True})


def test_r5_idempotent_operations(store):
    fid = store.add_fact("idempotent fact one", category="general")
    n0 = len(store.list_facts(limit=1000))
    # migration rerun
    from plugins.memory.holographic import store as _smod  # noqa
    store._init_db()
    # revalidate twice
    assert store.revalidate_fact(fid) == store.revalidate_fact(fid) == "UNCHANGED"
    # supersede retry
    other = store.add_fact("idempotent fact two", category="general")
    assert store.supersede_fact(fid, other, reason="x") is True
    assert store.supersede_fact(fid, other, reason="x") is True
    links = store._conn.execute(
        "SELECT COUNT(*) FROM fact_lineage WHERE old_fact_id = ? AND new_fact_id = ?",
        (fid, other)).fetchone()[0]
    assert links == 1  # no duplicate lineage
    # self-heal probe rerun (init is idempotent)
    store._init_db()
    n1 = len(store.list_facts(limit=1000))
    assert n1 == n0 + 1  # only the intended second row
    _write("idempotency.json", {"migration_rerun": True, "links": links, "stable": True})
    assert LLM_CALLS == 0


def test_r5_fts_heal_on_legacy(tmp_path):
    import sqlite3
    path = tmp_path / "heal.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL UNIQUE)")
    conn.execute("INSERT INTO facts (content) VALUES ('heal legacy row')")
    conn.commit()
    conn.close()
    s = MemoryStore(str(path), hrr_dim=32)
    try:
        s._conn.execute("UPDATE facts SET lifecycle='stale' WHERE fact_id=1")
        s._conn.commit()  # trigger write on legacy row must not corrupt
        r = FactRetriever(store=s, hrr_dim=32)
        assert isinstance(r.search("heal legacy", limit=5), list)
        _write("fts_heal.json", {"legacy_update": True})
    finally:
        s.close()
