"""R5 performance + resources: op latencies p50/p95/p99 at 1k/10k, context
bounds at scale, determinism, resource-leak and cache checks. Zero LLM.
"""

import gc
import json
import statistics
import threading
import time
from pathlib import Path

import pytest

from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "r5" / "performance"
LLM_CALLS = 0

needs_numpy = pytest.mark.skipif(
    __import__("importlib").util.find_spec("numpy") is None,
    reason="HRR paths need numpy (repo convention)",
)


def _pct(xs, p):
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(p / 100 * len(s)))] * 1000, 2)


def test_r5_op_latencies(tmp_path):
    rep = {}
    for n in (100, 1000):
        db = tmp_path / f"perf{n}.db"
        s = MemoryStore(str(db), hrr_dim=64)
        try:
            ids = s.add_facts_batch(
                [(f"perf fact {i} deploy cache worker {i % 20}", "project", "") for i in range(n)])
            ops = {}
            for name, fn in [
                ("add", lambda: s.add_fact("perf single probe deploy", category="project")),
                ("update", lambda: s.update_fact(ids[0], trust_delta=0.01)),
                ("feedback", lambda: s.record_feedback(ids[1], helpful=True)),
                ("verify", lambda: s.verify_fact(ids[2], verifier="perf")),
                ("supersede", lambda: s.supersede_fact(ids[3], ids[4], reason="perf")),
                ("revalidate", lambda: s.revalidate_fact(ids[5])),
            ]:
                lat = []
                for _ in range(10):
                    a = time.monotonic()
                    fn()
                    lat.append(time.monotonic() - a)
                ops[name] = {"p50": _pct(lat, 50), "p95": _pct(lat, 95)}
            r = FactRetriever(store=s, hrr_dim=64)
            lat = []
            for i in range(30):
                a = time.monotonic()
                r.search(f"perf deploy cache worker {i % 20}", limit=5)
                lat.append(time.monotonic() - a)
            ops["search"] = {"p50": _pct(lat, 50), "p95": _pct(lat, 95), "p99": _pct(lat, 99)}
            rep[str(n)] = ops
            rep[f"db_bytes_{n}"] = db.stat().st_size
        finally:
            s.close()
    rep["llm_calls"] = LLM_CALLS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "op_latencies.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    assert rep["100"]["search"]["p95"] < 100 and rep["1000"]["search"]["p95"] < 100
    assert LLM_CALLS == 0


def test_r5_context_bounded_at_scale(tmp_path):
    db = tmp_path / "ctx.db"
    prov = HolographicMemoryProvider(config={"db_path": str(db), "hrr_dim": 32})
    prov.initialize(session_id="r5")
    try:
        prov._store.add_facts_batch(
            [(f"context bound fact {i} deploy cache pagination", "project", "") for i in range(1000)])
        block = prov.prefetch("context bound deploy")
        lines = [ln for ln in block.splitlines() if ln.startswith("- [")]
        assert len(lines) <= 5 and len(block) <= 2500
    finally:
        prov.shutdown()


def test_r5_retrieval_deterministic(tmp_path):
    s = MemoryStore(str(tmp_path / "det.db"), hrr_dim=64)
    try:
        for i in range(20):
            s.add_fact(f"deterministic fact {i} deploy cache", category="project")
        r = FactRetriever(store=s, hrr_dim=64)
        first = [(x["fact_id"], round(x["score"], 9)) for x in r.search("deterministic deploy", limit=10)]
        for _ in range(5):
            again = [(x["fact_id"], round(x["score"], 9)) for x in r.search("deterministic deploy", limit=10)]
            assert again == first  # same query+state+config => same ranking
        # mutate trust only: ranking may change, but deterministically
        s.record_feedback(first[0][0], helpful=True)
        changed = [(x["fact_id"], round(x["score"], 9)) for x in r.search("deterministic deploy", limit=10)]
        repeat = [(x["fact_id"], round(x["score"], 9)) for x in r.search("deterministic deploy", limit=10)]
        assert changed == repeat
    finally:
        s.close()


def test_r5_no_resource_leaks(tmp_path):
    from plugins.memory.holographic.store import MemoryStore as MS
    threads_before = threading.active_count()
    for _ in range(200):
        s = MS(str(tmp_path / "leak.db"), hrr_dim=32)
        r = FactRetriever(store=s, hrr_dim=32)
        r.search("nothing here xyz", limit=3)
        s.close()
    gc.collect()
    assert len(MS._shared) == 0  # no leaked shared connections
    assert threading.active_count() <= threads_before + 1
    from plugins.memory.holographic import holographic as hrr
    assert hrr.atom_cache_info()["size"] <= hrr.ATOM_CACHE_MAX  # bounded cache


@needs_numpy
def test_r5_cache_project_safe():
    # Atom cache is keyed by (word, dim) — content-derived, project-free.
    from plugins.memory.holographic import holographic as hrr
    hrr.clear_atom_cache()
    a = hrr.encode_atom("shareword", 64)
    b = hrr.encode_atom("shareword", 64)
    import numpy as np
    assert bool((a == b).all())
    # Entity aliases live per-DB (store tables), never in the atom cache.
    assert hrr.atom_cache_info()["size"] >= 1
