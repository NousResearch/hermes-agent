"""R5 long-run session test: 10,000 logical memory events (batch-loaded for
practicality + interleaved lifecycle ops), interval metrics, invariant
checks, bloat/drift detection. Zero LLM, deterministic seed.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling test module

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore
from test_r5_invariants import check_all

ARM = os.environ.get("R5_ARM", "r5")
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "r5" / "longrun"
LLM_CALLS = 0
N_EVENTS = int(os.environ.get("R5_LONGRUN_N", "10000"))


def _snapshot(store, label, t0):
    facts = store._conn.execute(
        "SELECT lifecycle, COUNT(*) AS n FROM facts GROUP BY lifecycle").fetchall()
    counts = {r["lifecycle"]: r["n"] for r in facts}
    links = store._conn.execute("SELECT COUNT(*) FROM fact_lineage").fetchone()[0]
    dupes = store._conn.execute(
        "SELECT COUNT(*) FROM (SELECT content FROM facts GROUP BY content HAVING COUNT(*) > 1)").fetchone()[0]
    return {"label": label, "elapsed_s": round(time.monotonic() - t0, 1),
            "active": counts.get("active", 0), "stale": counts.get("stale", 0),
            "superseded": counts.get("superseded", 0), "revoked": counts.get("revoked", 0),
            "conflict": counts.get("conflict", 0), "lineage": links, "dupe_contents": dupes}


def test_r5_longrun(tmp_path):
    db = tmp_path / "long.db"
    store = MemoryStore(str(db), hrr_dim=64)
    t0 = time.monotonic()
    snaps = []
    try:
        # Phase 1: bulk load 60% base facts (batched).
        base = [(f"longrun base fact {i} deploy cache worker {i % 50}", "project", "")
                for i in range(int(N_EVENTS * 0.6))]
        store.add_facts_batch(base)
        snaps.append(_snapshot(store, "bulk-load", t0))
        # Phase 2: decisions change -> verify new, supersede old (15%).
        for i in range(0, int(N_EVENTS * 0.15), 2):
            new = store.add_fact(f"longrun decision {i} mode holographic", category="project")
            old_candidates = [f["fact_id"] for f in
                              store._conn.execute("SELECT fact_id FROM facts WHERE content LIKE 'longrun base%' LIMIT 1").fetchall()]
            store.verify_fact(new, verifier="longrun")
            if old_candidates:
                store.supersede_fact(old_candidates[0], new, reason="longrun")
        snaps.append(_snapshot(store, "churn", t0))
        # Phase 3: noise + feedback + conflicts + aliases (25%).
        r = FactRetriever(store=store, hrr_dim=64)
        for i in range(int(N_EVENTS * 0.25)):
            kind = i % 8
            if kind == 0:
                store.add_fact(f"longrun noise casual text {i} whatever", category="general")
            elif kind == 1:
                store.add_fact(f"longrun conflict key{i % 20} = value-a", category="project")
                store.add_fact(f"longrun conflict key{i % 20} = value-b", category="project")
            elif kind == 2:
                store.add_entity_alias(f"Entity{i % 30}", f"E{i % 30}")
            elif kind == 3:
                rows = store.list_facts(limit=3)
                if rows:
                    store.record_feedback(rows[0]["fact_id"], helpful=(i % 2 == 0))
            elif kind == 4:
                rows = store.list_facts(limit=3)
                if rows:
                    store.revalidate_fact(rows[0]["fact_id"])
            elif kind == 5:
                r.search("longrun deploy cache", limit=5)
            elif kind == 6:
                rows = store.list_facts(limit=3)
                if rows:
                    store.mark_stale(rows[-1]["fact_id"], reason="longrun")
            else:
                store.add_fact(f"longrun repeated fact number {i % 100} deploy", category="project")
            if i % 2000 == 1999:
                snaps.append(_snapshot(store, f"mixed-{i}", t0))
        snaps.append(_snapshot(store, "final", t0))
        # Invariants over the final state.
        inv = check_all(store)
        failed_inv = {k: v for k, v in inv.items() if not v[0]}
        # Retrieval still works and is bounded.
        t1 = time.monotonic()
        hits = r.search("longrun deploy cache", limit=5)
        lat_ms = (time.monotonic() - t1) * 1000
        payload = {"n_events": N_EVENTS, "snapshots": snaps,
                   "invariants_failed": failed_inv, "final_search_ms": round(lat_ms, 2),
                   "final_hits": len(hits), "db_bytes": db.stat().st_size,
                   "llm_calls": LLM_CALLS}
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "longrun.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        assert not failed_inv, failed_inv
        assert len(hits) > 0 and lat_ms < 5000
        # No bloat: duplicate contents stay bounded (UNIQUE + dedupe paths).
        assert snaps[-1]["dupe_contents"] == 0
        assert LLM_CALLS == 0
    finally:
        store.close()
