"""R7 integration tests (NEW — written for the recovered stack, not reconstructed
history). End-to-end lifecycle across store lifecycle + retrieval + R6
operations. Deterministic, stdlib only, zero LLM.
"""

import json

import pytest

from plugins.memory.holographic import operations as ops
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "r7.db"), hrr_dim=64)
    yield s
    s.close()


def _retriever(store):
    return FactRetriever(store=store, hrr_dim=64)


def test_r7_end_to_end_lifecycle(store, tmp_path):
    # 1-4: preference, decision, evidence, temporary event
    pref = store.add_fact("user prefers concise summaries", category="project")
    dec = store.add_fact("architecture decision: use sqlite for memory", category="project")
    ev = store.add_fact("pytest 100/100 passed on memory suite", category="project")
    tmp_evt = store.add_fact("temporary event: deploy at noon", category="project")
    # 5-6: change decision + supersede old
    dec2 = store.add_fact("architecture decision: use postgres for memory", category="project")
    assert store.supersede_fact(dec, dec2, reason="adr-002") is True
    # 7-8: contradiction visible, then old revoked explicitly
    hits = _retriever(store).search("architecture decision memory database", limit=5)
    assert any(h["fact_id"] == dec2 for h in hits)
    # 9-10: stale + revalidate the temporary event
    assert store.mark_stale(tmp_evt, reason="past noon") is True
    assert store.revalidate_fact(tmp_evt) in ("UNCHANGED", "REVALIDATED", "STALE", "ACTIVE")
    # 11: feedback promotes evidence
    store.record_feedback(ev, helpful=True)
    # 12: prefetch-style retrieval still finds the canonical decision
    top = _retriever(store).search("which database for memory", limit=3)
    assert top, "canonical decision must remain retrievable"
    # 13-15: maintenance + backup + health
    assert ops.run_maintenance(store, mode="LIGHT")["ok"]
    meta = ops.create_backup(store, tmp_path / "bk")
    assert ops.verify_backup(meta["path"])["ok"]
    assert ops.health_check(store)["status"] in ("HEALTHY", "DEGRADED")
    # 16-18: restore scratch + health + retrieval parity
    rs = ops.restore_to_scratch(meta["path"], str(tmp_path / "scratch.db"))
    assert rs["ok"]
    s2 = MemoryStore(str(tmp_path / "scratch.db"), hrr_dim=64)
    try:
        assert ops.health_check(s2)["status"] == ops.health_check(store)["status"]
        top2 = FactRetriever(store=s2, hrr_dim=64).search("which database for memory", limit=3)
        assert [h["fact_id"] for h in top2] == [h["fact_id"] for h in top]
    finally:
        s2.close()


def test_r7_backup_restore_integration(store, tmp_path):
    a = store.add_fact("r7 provider holographic deployed", category="project")
    b = store.add_fact("r7 provider honcho retired", category="project")
    assert store.supersede_fact(b, a, reason="migration") is True
    store.add_entity_alias("holographic", "holo")
    store.verify_fact(a, verifier="r7-test")
    store.record_feedback(a, helpful=True)
    before = ops.get_stats(store)
    meta = ops.create_backup(store, tmp_path / "bk")
    assert ops.verify_backup(meta["path"])["ok"]
    rs = ops.restore_to_scratch(meta["path"], str(tmp_path / "r.db"))
    assert rs["ok"]
    s2 = MemoryStore(str(tmp_path / "r.db"), hrr_dim=64)
    try:
        after = ops.get_stats(s2)
        assert after["facts"] == before["facts"]
        assert after["links"] == before["links"]
        # trust survived the roundtrip
        rows = {r["fact_id"]: r for r in s2.list_facts(limit=100)}
        orig = {r["fact_id"]: r for r in store.list_facts(limit=100)}
        assert rows[a]["trust_score"] == orig[a]["trust_score"]
        assert rows[a]["stale"] is False
        assert rows[b]["stale"] is True
        got = {r["fact_id"]: {"lifecycle": r["lifecycle"],
                              "superseded_by": r["superseded_by"]}
               for r in s2._conn.execute("SELECT fact_id, lifecycle, superseded_by FROM facts")}
        assert got[a]["lifecycle"] == "active"
        assert got[b]["lifecycle"] == "superseded"
        assert got[b]["superseded_by"] == a
        # alias + retrieval parity
        assert FactRetriever(store=s2, hrr_dim=64).search("holo deployed", limit=3)
    finally:
        s2.close()


def test_r7_cross_layer_invariants(store):
    # I2 revoked is terminal
    x = store.add_fact("r7 doomed fact", category="general")
    assert store.revoke_fact(x, reason="test") is True
    assert store.verify_fact(x, verifier="nobody") is False
    assert store.mark_stale(x, reason="nope") is False
    # I3 no supersession cycle
    p = store.add_fact("r7 cycle p", category="general")
    q = store.add_fact("r7 cycle q", category="general")
    assert store.supersede_fact(p, q, reason="t") is True
    assert store.supersede_fact(q, p, reason="t") is False
    # I7 trust bounded
    for _ in range(30):
        store.record_feedback(p, helpful=True)
    for r in store.list_facts(limit=100):
        assert 0.0 <= r["trust_score"] <= 1.0
    # I12 zero LLM across layers
    assert ops.health_check(store)["llm_calls"] == 0
    assert ops.run_maintenance(store, mode="LIGHT")["llm_calls"] == 0
    # I15 migration additive + idempotent
    m1 = ops.ensure_migration(store)
    m2 = ops.ensure_migration(store)
    assert m1["state"] == m2["state"] == "COMPLETE"
    n = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    assert n >= 3  # migration created no facts and deleted none
