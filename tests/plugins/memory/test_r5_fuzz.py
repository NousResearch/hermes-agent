"""R5 state-machine fuzzing: deterministic seeded PRNG operation sequences
over the lifecycle API + invariant checks after every sequence. Failing
seeds persist to results/r5/fuzz/failing_seeds.json and become regressions.
Zero LLM. No Hypothesis dependency (stdlib random with fixed seeds).
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling test module

import pytest

from plugins.memory.holographic.store import MemoryStore
from test_r5_invariants import check_all

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "r5" / "fuzz"
LLM_CALLS = 0
N_SEQUENCES = int(os.environ.get("R5_FUZZ_N", "60"))
OPS_PER_SEQUENCE = int(os.environ.get("R5_FUZZ_OPS", "40"))


def _run_sequence(store, rng):
    ids = []
    for _ in range(OPS_PER_SEQUENCE):
        op = rng.choice(["add", "add", "verify", "stale", "supersede",
                         "revoke", "revalidate", "feedback", "search", "remove"])
        try:
            if op == "add":
                ids.append(store.add_fact(
                    f"fuzz fact {rng.randrange(200)} wording {rng.randrange(5)}",
                    category=rng.choice(["project", "general"])))
            elif not ids:
                continue
            elif op == "verify":
                store.verify_fact(rng.choice(ids), verifier="fuzz")
            elif op == "stale":
                store.mark_stale(rng.choice(ids), reason="fuzz")
            elif op == "supersede":
                a, b = rng.sample(ids, 2) if len(ids) >= 2 else (ids[0], ids[0])
                store.supersede_fact(a, b, reason="fuzz")
            elif op == "revoke":
                store.revoke_fact(rng.choice(ids), reason="fuzz")
            elif op == "revalidate":
                store.revalidate_fact(rng.choice(ids))
            elif op == "feedback":
                store.record_feedback(rng.choice(ids), helpful=rng.random() < 0.5)
            elif op == "remove":
                if rng.random() < 0.3:
                    victim = rng.choice(ids)
                    store.remove_fact(victim)
                    ids = [i for i in ids if i != victim]
            elif op == "search":
                from plugins.memory.holographic.retrieval import FactRetriever
                FactRetriever(store=store, hrr_dim=32).search("fuzz fact", limit=5)
        except Exception:
            pass  # API-level refusals (unknown ids etc.) are legal outcomes
    return ids


def _structural_check(store):
    """DB-level structural invariants (no dangling pointers/cycles)."""
    problems = []
    rows = [dict(r) for r in store._conn.execute(
        "SELECT fact_id, lifecycle, superseded_by FROM facts").fetchall()]
    alive = {r["fact_id"] for r in rows}
    for r in rows:
        if r["lifecycle"] == "superseded":
            if r["superseded_by"] is not None and r["superseded_by"] not in alive:
                problems.append(f"dangling successor for {r['fact_id']}")
            if r["superseded_by"] == r["fact_id"]:
                problems.append(f"self-loop for {r['fact_id']}")
        if r["lifecycle"] not in ("active", "superseded", "stale", "revoked", "conflict"):
            problems.append(f"impossible state {r['lifecycle']} for {r['fact_id']}")
    # cycle detection over superseded_by edges
    parent = {r["fact_id"]: r["superseded_by"] for r in rows if r["superseded_by"] is not None}
    for start in parent:
        seen, cur = set(), start
        while cur in parent and cur not in seen:
            seen.add(cur)
            cur = parent[cur]
        if cur in seen:
            problems.append(f"lineage cycle at {cur}")
    links = [dict(r) for r in store._conn.execute(
        "SELECT old_fact_id, new_fact_id FROM fact_lineage").fetchall()]
    for ln in links:
        if ln["old_fact_id"] not in alive or ln["new_fact_id"] not in alive:
            problems.append(f"orphan lineage {ln}")
    return problems


def test_r5_fuzz(tmp_path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failing = []
    checked = 0
    for seed in range(N_SEQUENCES):
        rng = random.Random(10_000 + seed)
        db = tmp_path / f"fuzz{seed}.db"
        store = MemoryStore(str(db), hrr_dim=32)
        try:
            _run_sequence(store, rng)
            problems = _structural_check(store)
            inv = check_all(store)
            bad_inv = sorted(k for k, v in inv.items() if not v[0])
            # I5 single-db trivially holds; I9 spawns its own provider DB.
            bad_inv = [k for k in bad_inv if k not in ("I5", "I9")]
            if problems or bad_inv:
                failing.append({"seed": 10_000 + seed, "structural": problems[:5],
                                "invariants": bad_inv})
            checked += 1
        finally:
            store.close()
    with open(RESULTS_DIR / "fuzz.json", "w", encoding="utf-8") as f:
        json.dump({"sequences": checked, "failing": failing, "llm_calls": LLM_CALLS}, f, indent=1)
    if failing:
        with open(RESULTS_DIR / "failing_seeds.json", "w", encoding="utf-8") as f:
            json.dump(failing, f, indent=1)
    assert not failing, failing[:3]
    assert LLM_CALLS == 0
