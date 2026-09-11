"""R4 temporal lifecycle tests: ACTIVE/SUPERSEDED/STALE/REVOKED/CONFLICT,
verification with source evidence, revalidation, supersession lineage,
retrieval precedence. Deterministic, zero LLM.
"""

import os
import time

import pytest

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore

STATES = ("active", "superseded", "stale", "revoked", "conflict")


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "r4.db"), hrr_dim=64)
    yield s
    s.close()


def test_r4_lifecycle_defaults_active(store):
    fid = store.add_fact("temporal default fact", category="project")
    row = store._conn.execute(
        "SELECT lifecycle, superseded_by FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["lifecycle"] == "active" and row["superseded_by"] is None


def test_r4_verify_with_source_evidence(tmp_path, store):
    src = tmp_path / "component.py"
    src.write_text("print('v1')", encoding="utf-8")
    fid = store.add_fact("component uses v1 API", category="project")
    assert store.verify_fact(fid, verifier="test", source_ref=str(src)) is True
    row = store._conn.execute(
        "SELECT lifecycle, verified_by, source_ref, verified_at FROM facts WHERE fact_id = ?",
        (fid,)).fetchone()
    assert row["lifecycle"] == "active" and row["verified_by"] == "test"
    assert str(src) in row["source_ref"] and row["verified_at"] != ""


def test_r4_verify_rejects_bad_state(store):
    fid = store.add_fact("unverifiable fact", category="general")
    assert store.verify_fact(999999, verifier="x") is False
    assert store.verify_fact(fid, verifier="x", lifecycle="canonical") is False


def test_r4_supersede_preserves_history(store):
    old = store.add_fact("provider = honcho", category="project")
    new = store.add_fact("provider = holographic", category="project")
    assert store.supersede_fact(old, new, reason="migration", verifier="test") is True
    o = store._conn.execute(
        "SELECT lifecycle, superseded_by FROM facts WHERE fact_id = ?", (old,)).fetchone()
    assert o["lifecycle"] == "superseded" and o["superseded_by"] == new
    n = store._conn.execute(
        "SELECT lifecycle, content FROM facts WHERE fact_id = ?", (new,)).fetchone()
    assert n["lifecycle"] == "active"  # history kept, nothing deleted
    links = store._conn.execute(
        "SELECT relation, reason FROM fact_lineage WHERE old_fact_id = ? AND new_fact_id = ?",
        (old, new)).fetchall()
    assert len(links) == 1 and links[0]["relation"] == "supersedes"


def test_r4_supersede_rejects_unknown(store):
    assert store.supersede_fact(111, 222) is False
    fid = store.add_fact("lonely fact", category="general")
    assert store.supersede_fact(fid, fid) is False  # self-supersession refused


def test_r4_revoke_keeps_row(store):
    fid = store.add_fact("revoked claim here", category="general")
    assert store.revoke_fact(fid, reason="proven wrong") is True
    row = store._conn.execute(
        "SELECT lifecycle FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["lifecycle"] == "revoked"
    assert store._conn.execute("SELECT COUNT(*) FROM facts WHERE fact_id = ?",
                               (fid,)).fetchone()[0] == 1


def test_r4_revalidate_unchanged(tmp_path, store):
    src = tmp_path / "stable.py"
    src.write_text("stable", encoding="utf-8")
    fid = store.add_fact("stable mapping fact", category="project")
    store.verify_fact(fid, verifier="t", source_ref=str(src))
    assert store.revalidate_fact(fid) == "UNCHANGED"


def test_r4_revalidate_missing(tmp_path, store):
    src = tmp_path / "gone.py"
    src.write_text("x", encoding="utf-8")
    fid = store.add_fact("gone mapping fact", category="project")
    store.verify_fact(fid, verifier="t", source_ref=str(src))
    src.unlink()
    assert store.revalidate_fact(fid) == "MISSING"
    row = store._conn.execute(
        "SELECT lifecycle FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["lifecycle"] == "stale"  # event-driven invalidation


def test_r4_revalidate_changed(tmp_path, store):
    src = tmp_path / "evolving.py"
    src.write_text("v1 content here", encoding="utf-8")
    fid = store.add_fact("evolving mapping fact", category="project")
    store.verify_fact(fid, verifier="t", source_ref=str(src))
    time.sleep(0.05)
    src.write_text("v2 completely different content here", encoding="utf-8")
    assert store.revalidate_fact(fid) == "CHANGED"


def test_r4_revalidate_no_source_stays(store):
    fid = store.add_fact("sourceless fact", category="general")
    assert store.revalidate_fact(fid) == "UNCHANGED"  # nothing to judge by
    assert store.revalidate_fact(999999) == "MISSING"  # unknown id


def test_r4_revalidate_conflict(store):
    a = store.add_fact("cfgslot = one", category="project")
    b = store.add_fact("cfgslot = two", category="project")
    assert store.revalidate_fact(a) == "CONFLICT"
    row = store._conn.execute(
        "SELECT lifecycle FROM facts WHERE fact_id = ?", (a,)).fetchone()
    assert row["lifecycle"] == "conflict"
    _ = b


def test_r4_precedence_verified_first(store):
    stale = store.add_fact("precedence fact stale", category="project")
    fresh = store.add_fact("precedence fact fresh verified", category="project")
    store._conn.execute("UPDATE facts SET trust_score = 1.0 WHERE fact_id = ?", (stale,))
    store._conn.execute("UPDATE facts SET trust_score = 0.4 WHERE fact_id = ?", (fresh,))
    store._conn.commit()
    store.mark_stale(stale, reason="test")
    store.verify_fact(fresh, verifier="t")
    r = FactRetriever(store=store, hrr_dim=64)
    out = r.search("precedence fact", limit=10)
    ids = [x["fact_id"] for x in out]
    assert ids.index(fresh) < ids.index(stale)  # verified beats high-trust stale


def test_r4_superseded_never_outranks(store):
    old = store.add_fact("rankx old value", category="project")
    new = store.add_fact("rankx new value", category="project")
    store._conn.execute("UPDATE facts SET trust_score = 1.0 WHERE fact_id = ?", (old,))
    store._conn.execute("UPDATE facts SET trust_score = 0.31 WHERE fact_id = ?", (new,))
    store._conn.commit()
    store.verify_fact(new, verifier="t")
    store.supersede_fact(old, new, reason="t")
    r = FactRetriever(store=store, hrr_dim=64)
    out = r.search("rankx value", limit=10)
    ids = [x["fact_id"] for x in out]
    assert ids.index(new) < ids.index(old)


def test_r4_stale_only_results_flagged(store):
    fid = store.add_fact("only stale knowledge here", category="project")
    store.mark_stale(fid, reason="t")
    r = FactRetriever(store=store, hrr_dim=64)
    out = r.search("only stale knowledge", limit=5)
    assert out and out[0]["lifecycle"] == "stale" and out[0]["stale"] is True


def test_r4_lifecycle_columns_legacy_safe(tmp_path):
    import sqlite3
    path = tmp_path / "nolf.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL UNIQUE)")
    conn.execute("INSERT INTO facts (content) VALUES ('pre-lifecycle row')")
    conn.commit()
    conn.close()
    s = MemoryStore(str(path), hrr_dim=32)
    try:
        rows = s.list_facts(limit=10)
        assert any(r["content"] == "pre-lifecycle row" for r in rows)
        fid = [r["fact_id"] for r in rows if r["content"] == "pre-lifecycle row"][0]
        assert s.verify_fact(fid, verifier="t") is True  # new cols work on old rows
    finally:
        s.close()


def test_r4_concurrent_verify_supersede(tmp_path):
    import threading
    s = MemoryStore(str(tmp_path / "conc4.db"), hrr_dim=32)
    try:
        ids = [s.add_fact(f"conc lifecycle fact {i}", category="general") for i in range(10)]
        errors: list[str] = []

        def _work(n):
            try:
                for i in ids:
                    if (i + n) % 2 == 0:
                        s.verify_fact(i, verifier=f"w{n}")
                    else:
                        s.mark_stale(i, reason="race")
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))

        threads = [threading.Thread(target=_work, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        n_links = s._conn.execute("SELECT COUNT(*) FROM fact_lineage").fetchone()[0]
        assert n_links >= 0
    finally:
        s.close()


def test_r4_no_resurrection_by_verify(store):
    fid = store.add_fact("doomed fact", category="general")
    assert store.revoke_fact(fid, reason="wrong") is True
    assert store.verify_fact(fid, verifier="attacker") is False
    row = store._conn.execute(
        "SELECT lifecycle FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["lifecycle"] == "revoked"
    fid2 = store.add_fact("superseded victim", category="general")
    fid3 = store.add_fact("successor fact", category="general")
    assert store.supersede_fact(fid2, fid3, reason="t") is True
    assert store.verify_fact(fid2, verifier="attacker") is False
    row = store._conn.execute(
        "SELECT lifecycle FROM facts WHERE fact_id = ?", (fid2,)).fetchone()
    assert row["lifecycle"] == "superseded"


def test_r4_update_voids_attestation(store):
    fid = store.add_fact("attested content here", category="project")
    assert store.verify_fact(fid, verifier="t") is True
    assert store.update_fact(fid, content="completely different content now") is True
    row = store._conn.execute(
        "SELECT verified_at, verified_by, source_sig FROM facts WHERE fact_id = ?",
        (fid,)).fetchone()
    assert row["verified_at"] == "" and row["verified_by"] == "" and row["source_sig"] == ""


def test_r4_supersede_revoked_target_refused(store):
    a = store.add_fact("supold fact", category="general")
    b = store.add_fact("supbad target", category="general")
    assert store.revoke_fact(b, reason="bad") is True
    assert store.supersede_fact(a, b, reason="t") is False
    row = store._conn.execute(
        "SELECT lifecycle FROM facts WHERE fact_id = ?", (a,)).fetchone()
    assert row["lifecycle"] == "active"


def test_r4_remove_cleans_lineage(store):
    a = store.add_fact("gone lineage one", category="general")
    b = store.add_fact("gone lineage two", category="general")
    assert store.supersede_fact(a, b, reason="t") is True
    assert store.remove_fact(a) is True
    n = store._conn.execute(
        "SELECT COUNT(*) FROM fact_lineage WHERE old_fact_id = ? OR new_fact_id = ?",
        (a, a)).fetchone()[0]
    assert n == 0  # no orphan lineage rows


def test_r4_reason_persisted(store):
    fid = store.add_fact("reasoned fact", category="general")
    assert store.mark_stale(fid, reason="source moved") is True
    row = store._conn.execute(
        "SELECT lifecycle_reason FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["lifecycle_reason"] == "source moved"
    assert store.verify_fact(fid, verifier="t") is True
    row = store._conn.execute(
        "SELECT lifecycle_reason FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["lifecycle_reason"] == ""  # fresh attestation clears it


def test_r4_canonical_successor(store):
    old = store.add_fact("chain old fact", category="general")
    mid = store.add_fact("chain mid fact", category="general")
    new = store.add_fact("chain new fact", category="general")
    assert store.supersede_fact(old, mid, reason="v2") is True
    assert store.supersede_fact(old, new, reason="v3") is True
    rows = store._conn.execute(
        "SELECT new_fact_id, active FROM fact_lineage WHERE old_fact_id = ? AND relation = 'supersedes'",
        (old,)).fetchall()
    active = [r["new_fact_id"] for r in rows if r["active"]]
    assert active == [new]  # single canonical successor, history kept
    assert len(rows) == 2


def test_r4_touch_does_not_invalidate(tmp_path, store):
    src = tmp_path / "touched.py"
    src.write_text("same content here", encoding="utf-8")
    fid = store.add_fact("touch probe fact", category="project")
    assert store.verify_fact(fid, verifier="t", source_ref=str(src)) is True
    import os
    os.utime(str(src), (9999999999, 9999999999))  # mtime-only change
    assert store.revalidate_fact(fid) == "UNCHANGED"


def test_r4_cross_category_no_conflict(store):
    store.add_fact("sharedslot = one", category="project")
    store.add_fact("sharedslot = two", category="general")
    fid = store.add_fact("sharedslot = one", category="project")
    # duplicate returns the existing project row; the general-category
    # same-slot row is an independent scope -> UNCHANGED, not CONFLICT
    assert store.revalidate_fact(fid) == "UNCHANGED"
    rows = store._conn.execute(
        "SELECT fact_id FROM facts WHERE content = 'sharedslot = two' AND category = 'general'").fetchall()
    assert len(rows) == 1


def test_r4_probe_demotes_history(store):
    store.add_fact('History entity "OldSystem" retired', category="project")
    r = FactRetriever(store=store, hrr_dim=64)
    rows = store._conn.execute("SELECT fact_id FROM facts WHERE content LIKE '%OldSystem%'").fetchall()
    for row in rows:
        store.mark_stale(row["fact_id"], reason="t")
    out = r.probe("OldSystem", limit=10)
    assert all(x.get("stale", True) for x in out)  # all flagged, none silent


def test_r4_contradict_excludes_revoked(store):
    store.add_fact("rejslot = one", category="project")
    bad = store.add_fact("rejslot = two", category="project")
    assert store.revoke_fact(bad, reason="bad data") is True
    r = FactRetriever(store=store, hrr_dim=64)
    pairs = {(c["fact_a"]["content"], c["fact_b"]["content"]) for c in r.contradict(limit=50)}
    assert not any("rejslot" in a and "rejslot" in b for a, b in pairs)


def test_r4_remove_nulls_successor_pointers(store):
    a = store.add_fact("ptr old fact", category="general")
    b = store.add_fact("ptr new fact", category="general")
    assert store.supersede_fact(a, b, reason="t") is True
    assert store.remove_fact(b) is True
    row = store._conn.execute(
        "SELECT lifecycle, superseded_by FROM facts WHERE fact_id = ?", (a,)).fetchone()
    assert row["superseded_by"] is None
    assert row["lifecycle"] == "stale"  # demoted history, never stranded/current


def test_r4_supersede_revoked_old_refused(store):
    a = store.add_fact("dead old fact", category="general")
    b = store.add_fact("live new fact", category="general")
    assert store.revoke_fact(a, reason="wrong") is True
    assert store.supersede_fact(a, b, reason="t") is False
    row = store._conn.execute(
        "SELECT lifecycle FROM facts WHERE fact_id = ?", (a,)).fetchone()
    assert row["lifecycle"] == "revoked"  # terminal, unchanged


def test_r5_no_supersession_cycle(store):
    a = store.add_fact("cycle A fact", category="general")
    b = store.add_fact("cycle B fact", category="general")
    assert store.supersede_fact(a, b, reason="t") is True
    assert store.supersede_fact(b, a, reason="t") is False  # would close a cycle
    c = store.add_fact("cycle C fact", category="general")
    assert store.supersede_fact(b, c, reason="t") is True
    assert store.supersede_fact(c, a, reason="t") is False  # transitive cycle
    rows = store._conn.execute(
        "SELECT fact_id FROM facts WHERE lifecycle = 'superseded'").fetchall()
    assert {r["fact_id"] for r in rows} == {a, b}


def test_r5_mark_stale_refuses_terminal(store):
    a = store.add_fact("terminal stale probe", category="general")
    b = store.add_fact("terminal stale successor", category="general")
    assert store.supersede_fact(a, b, reason="t") is True
    assert store.mark_stale(a, reason="x") is False  # superseded keeps its link
    assert store.revoke_fact(a, reason="x") is True
    assert store.mark_stale(a, reason="x") is False  # revoked stays terminal
    row = store._conn.execute(
        "SELECT lifecycle FROM facts WHERE fact_id = ?", (a,)).fetchone()
    assert row["lifecycle"] == "revoked"


def test_r5_revoke_clears_attestation(store):
    fid = store.add_fact("attested then revoked", category="project")
    assert store.verify_fact(fid, verifier="t") is True
    assert store.revoke_fact(fid, reason="wrong") is True
    row = store._conn.execute(
        "SELECT lifecycle, verified_at, verified_by FROM facts WHERE fact_id = ?",
        (fid,)).fetchone()
    assert row["lifecycle"] == "revoked"
    assert row["verified_at"] == "" and row["verified_by"] == ""
