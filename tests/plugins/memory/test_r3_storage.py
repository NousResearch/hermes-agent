"""R3 storage tests: batch API, transaction atomicity, crash recovery,
incremental bank exactness, atom-cache correctness. Zero LLM, deterministic.
"""

import sqlite3
import threading

import pytest

from plugins.memory.holographic import holographic as hrr
from plugins.memory.holographic.store import MemoryStore

needs_numpy = pytest.mark.skipif(
    __import__("importlib").util.find_spec("numpy") is None,
    reason="HRR paths need numpy (repo convention)",
)


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "r3.db"), hrr_dim=64)
    yield s
    s.close()


def test_r3_batch_returns_ids_and_matches_singles(tmp_path):
    a = MemoryStore(str(tmp_path / "a.db"), hrr_dim=64)
    b = MemoryStore(str(tmp_path / "b.db"), hrr_dim=64)
    try:
        items = [(f"batch fact {i} deploy cache", "project", "") for i in range(10)]
        ids_batch = b.add_facts_batch(items)
        ids_single = [a.add_fact(c, category=cat) for c, cat, _t in items]
        assert len(ids_batch) == 10 and len(set(ids_batch)) == 10
        ca = sorted(f["content"] for f in a.list_facts(limit=50))
        cb = sorted(f["content"] for f in b.list_facts(limit=50))
        assert ca == cb
    finally:
        a.close()
        b.close()


def test_r3_batch_duplicate_returns_existing(store):
    first = store.add_fact("dup batch fact", category="general")
    ids = store.add_facts_batch([("dup batch fact", "general", ""),
                                 ("fresh batch fact", "general", "")])
    assert ids[0] == first
    assert ids[1] != first
    assert len(store.list_facts(limit=50)) == 2  # no row growth on dupe


def test_r3_batch_atomic_on_error(store):
    before = len(store.list_facts(limit=1000))
    with pytest.raises(Exception):
        store.add_facts_batch([("good fact one", "general", ""),
                               ("", "general", ""),  # empty -> ValueError
                               ("good fact two", "general", "")])
    after = len(store.list_facts(limit=1000))
    assert after == before  # all-or-nothing: partial batch rolled back


def test_r3_batch_read_after_write(store):
    ids = store.add_facts_batch([("readback fact alpha", "project", "")])
    rows = [f for f in store.list_facts(limit=50) if f["fact_id"] == ids[0]]
    assert rows and rows[0]["content"] == "readback fact alpha"


@needs_numpy
def test_r3_incremental_bank_matches_full_rebuild(tmp_path):
    from plugins.memory.holographic.store import MemoryStore as MS
    s = MS(str(tmp_path / "inc.db"), hrr_dim=64)
    try:
        for i in range(8):
            s.add_fact(f"incremental fact {i} deploy cache", category="project")
        inc_vec = s._conn.execute(
            "SELECT vector FROM memory_banks WHERE bank_name = 'cat:project'").fetchone()["vector"]
        # force full rebuild and compare
        s._rebuild_bank("project")
        full_vec = s._conn.execute(
            "SELECT vector FROM memory_banks WHERE bank_name = 'cat:project'").fetchone()["vector"]
        assert inc_vec == full_vec  # bitwise exact, not approximate
    finally:
        s.close()


@needs_numpy
def test_r3_bank_rebuild_recovers(store):
    for i in range(5):
        store.add_fact(f"recover fact {i} deploy", category="project")
    store._conn.execute("DELETE FROM memory_banks WHERE bank_name = 'cat:project'")
    store._conn.commit()
    store._rebuild_bank("project")
    row = store._conn.execute(
        "SELECT fact_count FROM memory_banks WHERE bank_name = 'cat:project'").fetchone()
    assert row["fact_count"] == 5  # derived state rebuilds from facts


@needs_numpy
def test_r3_atom_cache_correctness():
    hrr.clear_atom_cache()
    v1 = hrr.encode_atom("deploy", 64)
    v2 = hrr.encode_atom("deploy", 64)
    import numpy as np
    assert bool((v1 == v2).all())
    assert hrr.atom_cache_info()["hits"] >= 1
    assert hrr.atom_cache_info()["size"] <= hrr.ATOM_CACHE_MAX


@needs_numpy
def test_r3_atom_cache_bounded_eviction():
    hrr.clear_atom_cache(max_entries=8)
    try:
        for i in range(30):
            hrr.encode_atom(f"evict-word-{i}", 32)
        assert hrr.atom_cache_info()["size"] <= 8
    finally:
        hrr.clear_atom_cache(max_entries=4096)  # restore default bound


def test_r3_no_duplicate_lineage_on_retry(store):
    ids1 = store.add_facts_batch([("retry fact one", "general", "")])
    ids2 = store.add_facts_batch([("retry fact one", "general", "")])
    assert ids1 == ids2
    assert len(store.list_facts(limit=50)) == 1
