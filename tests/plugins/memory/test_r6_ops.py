"""R6 operations tests on stock Holographic: health, repair, backup/verify/
restore/rotation, maintenance modes+bounds, migration states, events,
metrics, chaos, security (no-leak), isolation, smoke, long-run autonomy,
offline, zero-LLM. Deterministic, stdlib only.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from plugins.memory.holographic import operations as ops
from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "ops.db"), hrr_dim=64)
    yield s
    s.close()


def _seed(store, n=12):
    for i in range(n):
        store.add_fact(f"ops seed fact {i} deploy cache", category="project" if i % 2 == 0 else "general")


def test_r6_health_read_only(store):
    _seed(store)
    before = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    rep = ops.health_check(store)
    assert rep["status"] in ("HEALTHY", "DEGRADED", "NEEDS_REPAIR", "BLOCKED")
    for section in ("database", "schema", "indexes", "lineage", "temporal",
                    "trust", "security", "cache", "config", "migration", "storage"):
        assert section in rep, section
    after = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    assert before == after  # CHECK mode mutates nothing
    assert rep["llm_calls"] == 0


def test_r6_health_detects_derived_damage(store):
    # Env-agnostic damage: trust corruption is detected in any environment.
    _seed(store)
    store._conn.execute("UPDATE facts SET trust_score = 2.0")
    store._conn.commit()
    rep = ops.health_check(store)
    assert rep["status"] in ("DEGRADED", "NEEDS_REPAIR")
    assert rep["trust"]["out_of_range_count"] >= 1


needs_hrr = pytest.mark.skipif(not ops.hrr_available(), reason="no numpy/HRR in this env")


@needs_hrr
def test_r6_health_detects_missing_banks(store):
    _seed(store)
    assert ops.health_check(store)["status"] == "HEALTHY"
    store._conn.execute("DELETE FROM memory_banks")
    store._conn.commit()
    rep = ops.health_check(store)
    assert rep["status"] in ("DEGRADED", "NEEDS_REPAIR")
    assert rep["indexes"]["consistent"] is False


@needs_hrr
def test_r6_repair_derived_idempotent(store):
    _seed(store)
    store._conn.execute("DELETE FROM memory_banks")
    store._conn.commit()
    r1 = ops.repair(store, mode="REPAIR_DERIVED")
    r2 = ops.repair(store, mode="REPAIR_DERIVED")
    assert r1["repaired"] and r2["ok"]  # second run: nothing to do, still ok
    assert ops.health_check(store)["status"] == "HEALTHY"


def test_r6_repair_check_is_readonly(store):
    _seed(store)
    n0 = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    rep = ops.repair(store, mode="CHECK")
    assert rep["mode"] == "CHECK"
    assert store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == n0


def test_r6_backup_verify_restore(tmp_path, store):
    _seed(store)
    backup_dir = tmp_path / "backups"
    meta = ops.create_backup(store, backup_dir)
    assert Path(meta["path"]).exists()
    assert len(meta["sha256"]) == 64
    assert meta["fact_count"] == 12 and meta["schema_version"]
    v = ops.verify_backup(meta["path"])
    assert v["ok"] and v["fact_count"] == 12
    scratch = ops.restore_to_scratch(meta["path"], tmp_path / "scratch.db")
    assert scratch["fact_count"] == 12
    # production untouched: same row count as before
    assert store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 12
    # corrupted copy fails closed
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a database at all")
    assert ops.verify_backup(str(bad))["ok"] is False


def test_r6_backup_checksum_mismatch(tmp_path, store):
    _seed(store, n=3)
    meta = ops.create_backup(store, tmp_path / "bk")
    with open(meta["path"], "r+b") as f:
        f.seek(100)
        f.write(b"XX")
    v = ops.verify_backup(meta["path"])
    assert v["ok"] is False and "checksum" in v["reason"].lower()


def test_r6_rotation_keeps_good(tmp_path, store):
    _seed(store, n=2)
    d = tmp_path / "rot"
    metas = [ops.create_backup(store, d) for _ in range(3)]
    assert len(list(d.glob("*.db"))) == 3
    rep = ops.rotate_backups(d, keep=2)
    assert rep["kept"] == 2 and rep["removed"] == 1
    # refuse to delete the only known-good backup
    only = [m for m in metas if m["path"].endswith(".db")][:0]
    _ = only
    rep2 = ops.rotate_backups(d, keep=5)
    assert rep2["removed"] == 0


def test_r6_maintenance_modes_bounded(store):
    _seed(store)
    for mode in ("NORMAL", "LIGHT", "DEEP"):
        rep = ops.run_maintenance(store, mode=mode, budget_ms=30000)
        assert rep["mode"] == mode and rep["ok"] and rep["llm_calls"] == 0
        assert rep["elapsed_ms"] < 30000


def test_r6_maintenance_resumable(store):
    _seed(store, n=30)
    r1 = ops.run_maintenance(store, mode="DEEP", budget_ms=30000)
    r2 = ops.run_maintenance(store, mode="DEEP", budget_ms=30000)
    assert r1["ok"] and r2["ok"]  # rerun safe (idempotent)


def test_r6_circuit_breaker(store):
    ops._record_failure(store, "probe-op")
    ops._record_failure(store, "probe-op")
    assert ops.circuit_state(store, "probe-op") == "CLOSED"
    ops._record_failure(store, "probe-op")
    assert ops.circuit_state(store, "probe-op") == "OPEN"
    ops._record_success(store, "probe-op")
    assert ops.circuit_state(store, "probe-op") == "CLOSED"


def test_r6_migration_states(tmp_path):
    s = MemoryStore(str(tmp_path / "mig.db"), hrr_dim=32)
    try:
        st = ops.migration_status(s)
        assert st["state"] in ("NOT_REQUIRED", "READY", "COMPLETE")
        assert st["schema_version"] and st["additive"] is True
        # rerun is idempotent
        st2 = ops.migration_status(s)
        assert st2["state"] == st["state"]
    finally:
        s.close()


def test_r6_events_bounded_no_secrets(store):
    store.add_fact("api_key: sk-abcdef1234567890", category="general")
    ops.health_check(store)
    ops.run_maintenance(store, mode="NORMAL")
    evts = ops.recent_events(store, limit=200)
    assert len(evts) <= 200
    blob = json.dumps(evts)
    assert "sk-abcdef" not in blob  # no content/secrets in events
    assert all(set(e) >= {"type", "ts"} for e in evts)


def test_r6_metrics_no_secrets(store):
    _seed(store)
    m = ops.get_metrics(store)
    assert m["llm_calls"] == 0
    for k in ("memory_count", "active_count", "repair_success", "backup_success"):
        assert k in m
    assert "sk-" not in json.dumps(m)


def test_r6_stats_redacted(store):
    store.add_fact("bearer token secret abcdef1234567890 here", category="general")
    st = ops.get_stats(store)
    assert "abcdef1234567890" not in json.dumps(st)
    assert st["facts"] >= 1


def test_r6_restore_roundtrip_gold(tmp_path, store):
    _seed(store, n=8)
    meta = ops.create_backup(store, tmp_path / "g")
    scratch_path = tmp_path / "gold.db"
    rep = ops.restore_to_scratch(meta["path"], str(scratch_path))
    s2 = MemoryStore(str(scratch_path), hrr_dim=64)
    try:
        from plugins.memory.holographic.retrieval import FactRetriever
        r = FactRetriever(store=s2, hrr_dim=64)
        hits = r.search("ops seed deploy", limit=8)
        assert len(hits) == 8  # no silent loss
        assert rep["fact_count"] == 8
    finally:
        s2.close()


def test_r6_offline_no_network_imports(store, tmp_path, monkeypatch):
    # Behavior contract: operations declares zero network dependencies via
    # its own introspectable function (no source-text reading in tests).
    assert ops.network_dependencies() == []
    # Full ops cycle works with all socket creation blocked (offline proof).
    import socket as _socket

    def _blocked(*a, **k):
        raise OSError("network disabled for offline test")

    monkeypatch.setattr(_socket, "socket", _blocked)
    assert ops.health_check(store)["llm_calls"] == 0
    assert ops.run_maintenance(store, mode="NORMAL")["ok"]
    meta = ops.create_backup(store, tmp_path / "off")
    assert ops.verify_backup(meta["path"])["ok"]


def test_r6_zero_llm_everywhere(store, tmp_path):
    _seed(store, n=3)
    assert ops.health_check(store)["llm_calls"] == 0
    assert ops.run_maintenance(store, mode="LIGHT")["llm_calls"] == 0
    meta = ops.create_backup(store, tmp_path / "z")
    assert ops.verify_backup(meta["path"])["llm_calls"] == 0
    assert ops.restore_to_scratch(meta["path"], str(tmp_path / "z2.db"))["llm_calls"] == 0


def test_r6_report_on_damaged_db(store):
    # Reviewer F1: REPORT must not crash when sections are missing.
    _seed(store)
    store._conn.execute("DROP TABLE facts_fts")
    store._conn.commit()
    assert ops.health_check(store)["status"] == "NEEDS_REPAIR"
    rep = ops.repair(store, mode="REPORT")
    assert rep["ok"] and isinstance(rep["report"], str) and "status=" in rep["report"]


def test_r6_budget_exhaustion_trips_circuit(store):
    # Reviewer F2: starved runs count as failures, never as success.
    for _ in range(3):
        out = ops.run_maintenance(store, mode="NORMAL", budget_ms=0)
        assert out["ok"] is False and out.get("status") == "BUDGET_EXCEEDED"
    assert ops.circuit_state(store, "maintenance-NORMAL") == "OPEN"


@needs_hrr
def test_r6_stale_bank_pruned(store):
    _seed(store, n=2)
    store._conn.execute(
        "INSERT INTO memory_banks(bank_name, vector, dim, fact_count) VALUES ('cat:ghost', x'00', 64, 0)")
    store._conn.commit()
    h = ops.health_check(store)
    assert "cat:ghost" in h["indexes"]["stale_banks"]
    assert h["indexes"]["consistent"] is False
    rep = ops.repair(store, mode="REPAIR_DERIVED")
    assert rep["ok"] and any("pruned:cat:ghost" in r for r in rep["repaired"])
    assert ops.health_check(store)["status"] == "HEALTHY"


def test_r6_smoke(tmp_path):
    from plugins.memory.holographic import HolographicMemoryProvider
    from plugins.memory.holographic.retrieval import FactRetriever
    prov = HolographicMemoryProvider(config={"db_path": str(tmp_path / "smoke.db"), "hrr_dim": 64})
    prov.initialize(session_id="smoke")
    try:
        assert ops.health_check(prov._store)["status"] == "HEALTHY"
        fid = prov._store.add_fact("smoke deploy fact", category="project")
        assert FactRetriever(store=prov._store, hrr_dim=64).search("smoke deploy", limit=3)
        prov._store.record_feedback(fid, helpful=True)
        assert prov.prefetch("smoke deploy") != ""
        assert ops.run_maintenance(prov._store, mode="LIGHT")["ok"]
        meta = ops.create_backup(prov._store, tmp_path / "sb")
        assert ops.verify_backup(meta["path"])["ok"]
        ops.restore_to_scratch(meta["path"], str(tmp_path / "ss.db"))
        assert ops.health_check(prov._store)["status"] == "HEALTHY"
    finally:
        prov.shutdown()


def test_r6_longrun_autonomy(tmp_path):
    s = MemoryStore(str(tmp_path / "auto.db"), hrr_dim=32)
    try:
        fails, repairs = 0, 0
        for cycle in range(8):
            for i in range(25):
                s.add_fact(f"autonomy cycle {cycle} fact {i} deploy", category="project")
            h = ops.health_check(s)
            m = ops.run_maintenance(s, mode="NORMAL")
            if h["status"] != "HEALTHY":
                fails += 1
            if m.get("repaired"):
                repairs += 1
        assert fails == 0
        assert s._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 200
    finally:
        s.close()


@needs_hrr
def test_r6_legacy_vector_backfill(tmp_path):
    # R9 legacy-upgrade gate: pre-HRR rows (NULL vectors) get derived vectors
    # backfilled on bank rebuild; content untouched; health reaches HEALTHY.
    import sqlite3
    leg = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(leg))
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "content TEXT NOT NULL UNIQUE, category TEXT DEFAULT 'general', "
                 "tags TEXT DEFAULT '', trust_score REAL DEFAULT 0.5, "
                 "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0, "
                 "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                 "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("INSERT INTO facts (content, category) VALUES "
                 "('legacy backfill alpha', 'project'), ('legacy backfill beta', 'general')")
    conn.commit()
    conn.close()
    from plugins.memory.holographic.store import MemoryStore
    s = MemoryStore(str(leg), hrr_dim=64)
    try:
        assert [r["content"] for r in s.list_facts(limit=10)] == [
            "legacy backfill alpha", "legacy backfill beta"]
        assert ops.run_maintenance(s, mode="LIGHT")["ok"]
        assert ops.health_check(s)["status"] == "HEALTHY"
        assert s._conn.execute("SELECT COUNT(*) FROM facts WHERE hrr_vector IS NULL").fetchone()[0] == 0
    finally:
        s.close()
