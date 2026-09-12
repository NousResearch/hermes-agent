"""R8 release tests (NEW, current contract only).

Canonical release smoke (R8-15) plus gap fills from the test migration matrix:
unhelpful-feedback delta, dedupe via the current read API, constraint
prefetch. Deterministic, stdlib only, zero LLM.
"""

import pytest

from plugins.memory.holographic import operations as ops
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "r8.db"), hrr_dim=64)
    yield s
    s.close()


def _trust(store, fid):
    rows = {r["fact_id"]: r for r in store.list_facts(limit=1000)}
    return rows[fid]["trust_score"]


def test_r8_release_smoke(tmp_path):
    from plugins.memory.holographic import HolographicMemoryProvider
    prov = HolographicMemoryProvider(config={"db_path": str(tmp_path / "rel.db"), "hrr_dim": 64})
    prov.initialize(session_id="release")
    try:
        assert ops.health_check(prov._store)["status"] in ("HEALTHY", "DEGRADED")
        fid = prov._store.add_fact("release fact deploy pipeline", category="project")
        assert FactRetriever(store=prov._store, hrr_dim=64).search("release deploy", limit=3)
        prov._store.record_feedback(fid, helpful=True)
        # contradiction visible via current API (slot pass, deterministic)
        a = prov._store.add_fact("cache mode = safe", category="project")
        b = prov._store.add_fact("cache mode = aggressive", category="project")
        pairs = {(c["fact_a"]["content"], c["fact_b"]["content"])
                 for c in FactRetriever(store=prov._store, hrr_dim=64).contradict(limit=50)}
        assert ("cache mode = safe", "cache mode = aggressive") in pairs or \
               ("cache mode = aggressive", "cache mode = safe") in pairs
        assert prov._store.supersede_fact(a, b, reason="release") is True
        assert prov._store.mark_stale(a, reason="x") is False  # superseded, not active
        assert prov.prefetch("release deploy") != ""
        assert ops.run_maintenance(prov._store, mode="LIGHT")["ok"]
        meta = ops.create_backup(prov._store, tmp_path / "rb")
        assert ops.verify_backup(meta["path"])["ok"]
        assert ops.restore_to_scratch(meta["path"], str(tmp_path / "rs.db"))["ok"]
        assert ops.health_check(prov._store)["status"] in ("HEALTHY", "DEGRADED")
    finally:
        prov.shutdown()


def test_r8_feedback_deltas(store):
    fid = store.add_fact("r8 feedback delta fact", category="general")
    before = _trust(store, fid)
    store.record_feedback(fid, helpful=True)
    assert _trust(store, fid) == pytest.approx(before + 0.05)
    store.record_feedback(fid, helpful=False)
    assert _trust(store, fid) == pytest.approx(before + 0.05 - 0.10)


def test_r8_dedupe_current_api(store):
    from plugins.memory.holographic import cognition as cog
    a = store.add_fact("deploy process = blue green deploy", category="project")
    b = store.add_fact("deploy process = blue-green deploy!", category="project")
    rows = {r["fact_id"]: r for r in store.list_facts(limit=100)}
    assert cog.dedupe_keys(rows[a]["content"])["slot_key"] == \
        cog.dedupe_keys(rows[b]["content"])["slot_key"]


def test_r8_constraint_prefetch(tmp_path):
    from plugins.memory.holographic import HolographicMemoryProvider
    prov = HolographicMemoryProvider(config={"db_path": str(tmp_path / "c.db"), "hrr_dim": 64})
    prov.initialize(session_id="r8c")
    try:
        prov._store.add_fact("ห้ามแก้ baseline โดยเด็ดขาด", category="project")
        out = prov.prefetch("baseline ห้ามแก้")
        assert isinstance(out, str) and "ห้ามแก้ baseline" in out
    finally:
        prov.shutdown()


def test_r8_bounded_retrieval(store):
    for i in range(60):
        store.add_fact(f"r8 bulk fact {i} pagination deploy", category="project")
    r = FactRetriever(store=store, hrr_dim=64)
    assert len(r.search("r8 bulk pagination", limit=50)) <= 50
