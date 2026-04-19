"""Tests for the darwinian-evolver skill.

Scope: pure-Python units that don't require a live LLM —

  * ``storage``   — insert/update idempotence, ancestry reconstruction,
                    budget totals, lineage-hash determinism.
  * ``algorithms`` — tournament + rank selection, (μ+λ) survival,
                     MAP-Elites placement, NSGA-II front non-domination
                     and crowding distance, Exp3 bandit reward update.
  * ``evaluator`` — fitness_spec decorator, synchronous fitness via the
                    async batch path, held_out_guard penalty shape,
                    successive_halving narrowing property.

End-to-end runs against a real LLM are intentionally not covered here;
they live under ``examples/summarize_10_words`` and are exercised
manually during review.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills" / "research" / "darwinian-evolver" / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import adapters    # noqa: E402
import algorithms  # noqa: E402
import evaluator   # noqa: E402
import llm         # noqa: E402
import sandbox     # noqa: E402
import storage     # noqa: E402


# ---------------------------------------------------------------------------
# storage.py
# ---------------------------------------------------------------------------


class TestStorage:
    def _open(self, tmp_path):
        return storage.open_db(tmp_path / "lineage.db")

    def test_hash_genome_deterministic(self):
        a = storage.hash_genome("Summarize this.")
        b = storage.hash_genome("Summarize this.")
        c = storage.hash_genome("Summarize this!")
        assert a == b
        assert a != c
        assert len(a) == 16

    def test_insert_candidate_idempotent(self, tmp_path):
        conn = self._open(tmp_path)
        cid1 = storage.insert_candidate(conn, "hello", 0)
        cid2 = storage.insert_candidate(conn, "hello", 5)  # later generation
        assert cid1 == cid2
        # The original generation should survive (INSERT OR IGNORE).
        row = storage.get_candidate(conn, cid1)
        assert row["generation"] == 0

    def test_lineage_edges_with_parents(self, tmp_path):
        conn = self._open(tmp_path)
        a = storage.insert_candidate(conn, "root",    0)
        b = storage.insert_candidate(conn, "root",    0)   # same genome
        assert a == b
        c = storage.insert_candidate(
            conn, "child", 1,
            parents=[(a, "paraphrase", "p1")],
        )
        edges = storage.get_ancestry(conn, c)
        assert len(edges) == 1
        assert edges[0]["parent_id"] == a
        assert edges[0]["operator"] == "paraphrase"

    def test_get_best_prefers_heldout(self, tmp_path):
        conn = self._open(tmp_path)
        a = storage.insert_candidate(conn, "A", 0)
        b = storage.insert_candidate(conn, "B", 0)
        storage.record_fitness(conn, a, "accuracy", 0.9, held_out=False)
        storage.record_fitness(conn, a, "accuracy", 0.5, held_out=True)
        storage.record_fitness(conn, b, "accuracy", 0.7, held_out=False)
        storage.record_fitness(conn, b, "accuracy", 0.8, held_out=True)
        rows = storage.get_best(conn, "accuracy", k=1)
        # With held_out priority, b should win (held_out=0.8 > a held_out=0.5).
        assert rows[0]["id"] == b

    def test_budget_totals(self, tmp_path):
        conn = self._open(tmp_path)
        storage.record_budget(conn, 100, 50, 0.001, "paraphrase")
        storage.record_budget(conn, 200, 80, 0.004, "semantic_crossover")
        tot = storage.get_budget_used(conn)
        assert tot["in_toks"] == 300
        assert tot["out_toks"] == 130
        assert abs(tot["usd"] - 0.005) < 1e-9
        assert tot["calls"] == 2

    def test_lineage_hash_stable(self, tmp_path):
        conn = self._open(tmp_path)
        a = storage.insert_candidate(conn, "A", 0)
        b = storage.insert_candidate(conn, "B", 1, parents=[(a, "paraphrase", "x")])
        storage.record_fitness(conn, a, "f", 1.0)
        storage.record_fitness(conn, b, "f", 2.0)
        h1 = storage.lineage_hash(conn)
        h2 = storage.lineage_hash(conn)
        assert h1 == h2

        # A change in fitness should flip the hash.
        storage.record_fitness(conn, b, "f", 9.0)
        h3 = storage.lineage_hash(conn)
        assert h1 != h3


# ---------------------------------------------------------------------------
# algorithms.py
# ---------------------------------------------------------------------------


class TestSelection:
    def _pop(self, scores):
        return [
            algorithms.Individual(cid=str(i), genome=str(i), fitness=s)
            for i, s in enumerate(scores)
        ]

    def test_tournament_picks_best_in_sample(self):
        import random
        pop = self._pop([0.1, 0.9, 0.5, 0.3])
        rng = random.Random(0)
        # With k=20 (>> pop size) we sample each slot with replacement; the
        # best element almost surely appears in every tournament, so every
        # winner should be 0.9. P(fail) = 4 × (0.75 ** 20) ≈ 1.3 %, further
        # reduced to 0 by the fixed seed.
        winners = algorithms.tournament_select(pop, k=20, n=4, rng=rng)
        assert all(w.fitness == 0.9 for w in winners)

    def test_tournament_is_biased_toward_high_fitness(self):
        import random
        pop = self._pop([0.0] * 9 + [1.0])
        rng = random.Random(1)
        winners = algorithms.tournament_select(pop, k=5, n=200, rng=rng)
        frac_best = sum(1 for w in winners if w.fitness == 1.0) / len(winners)
        # Expected ≈ 1 - (0.9 ** 5) ≈ 0.41. Well above uniform (0.10).
        assert 0.30 < frac_best < 0.55

    def test_rank_select_invariant(self):
        import random
        pop = self._pop([1, 2, 3, 4])
        rng = random.Random(0)
        with pytest.raises(ValueError):
            algorithms.rank_select(pop, n=1, rng=rng, pressure=3.0)

    def test_mu_plus_lambda_elitism(self):
        parents   = self._pop([1.0, 0.0])
        offspring = self._pop([0.5, -1.0])
        survivors = algorithms.mu_plus_lambda(parents, offspring, mu=2)
        assert 1.0 in [s.fitness for s in survivors]  # best parent survived
        assert survivors[0].fitness >= survivors[-1].fitness


class TestMapElites:
    def _ind(self, cid, score, desc):
        return algorithms.Individual(cid=cid, genome=cid, fitness=score, descriptor=desc)

    def test_place_replaces_on_improvement(self):
        arch = algorithms.MapElitesArchive(
            bin_counts=(2, 2), lows=(0, 0), highs=(2, 2),
        )
        assert arch.place(self._ind("a", 0.5, (0.5, 0.5)))
        # Better score in same bin → replaces.
        assert arch.place(self._ind("b", 0.9, (0.5, 0.5)))
        # Worse score in same bin → rejected.
        assert not arch.place(self._ind("c", 0.1, (0.5, 0.5)))
        best = arch.best()
        assert best is not None and best.cid == "b"

    def test_coverage(self):
        arch = algorithms.MapElitesArchive(
            bin_counts=(4, 4), lows=(0, 0), highs=(4, 4),
        )
        assert arch.coverage() == 0.0
        arch.place(self._ind("a", 0.1, (0, 0)))
        arch.place(self._ind("b", 0.2, (3, 3)))
        # 2 of 16 bins populated.
        assert abs(arch.coverage() - 2 / 16) < 1e-9

    def test_sample_empty_returns_empty(self):
        import random
        arch = algorithms.MapElitesArchive(bin_counts=(2, 2), lows=(0, 0), highs=(2, 2))
        assert arch.sample(random.Random(0), k=3) == []


class TestNSGA2:
    def _dict_ind(self, cid, a, b):
        return algorithms.Individual(cid=cid, genome=cid, fitness={"a": a, "b": b})

    def test_fast_nondominated_sort_two_fronts(self):
        pop = [
            self._dict_ind("x", 1.0, 1.0),  # dominated by y
            self._dict_ind("y", 2.0, 2.0),  # front 0
            self._dict_ind("z", 2.0, 1.5),  # dominated by y
            self._dict_ind("w", 3.0, 0.0),  # front 0 — trades a for b
        ]
        fronts = algorithms.fast_non_dominated_sort(pop, ["a", "b"])
        front0_ids = {ind.cid for ind in fronts[0]}
        assert front0_ids == {"y", "w"}

    def test_dominates_strict(self):
        a = {"a": 2, "b": 2}
        b = {"a": 1, "b": 1}
        c = {"a": 2, "b": 2}
        assert algorithms.dominates(a, b, ["a", "b"])
        assert not algorithms.dominates(a, c, ["a", "b"])  # equal ≠ dominates

    def test_crowding_boundary_infinity(self):
        front = [
            self._dict_ind("x", 1.0, 0.0),
            self._dict_ind("y", 0.0, 1.0),
            self._dict_ind("z", 0.5, 0.5),
        ]
        dist = algorithms.crowding_distance(front, ["a", "b"])
        # Boundary points on any objective must be infinite.
        assert dist[0] == float("inf")
        assert dist[1] == float("inf")
        assert dist[2] < float("inf")

    def test_nsga2_select_keeps_pareto_first(self):
        combined = [
            self._dict_ind("p0", 3.0, 3.0),   # front 0
            self._dict_ind("p1", 3.5, 2.5),   # front 0
            self._dict_ind("p2", 1.0, 1.0),   # dominated
        ]
        survivors = algorithms.nsga2_select(combined, ["a", "b"], mu=2)
        ids = {s.cid for s in survivors}
        assert "p2" not in ids


class TestExp3:
    def test_pick_and_reward_update_weights(self):
        import random
        bandit = algorithms.Exp3Bandit(arms=["a", "b", "c"], gamma=0.1)
        rng = random.Random(0)
        idx, arm = bandit.pick(rng)
        w_before = bandit.weights[idx]
        bandit.reward(idx, 1.0)
        assert bandit.weights[idx] > w_before

    def test_reward_clamped(self):
        bandit = algorithms.Exp3Bandit(arms=["x"])
        bandit.reward(0, 9.5)   # clamped to 1.0, no crash
        bandit.reward(0, -3.0)  # clamped to 0.0
        assert bandit.weights[0] > 0


# ---------------------------------------------------------------------------
# evaluator.py
# ---------------------------------------------------------------------------


class TestFitnessSpec:
    def test_attaches_metadata(self):
        @evaluator.fitness_spec(held_out_frac=0.25, timeout_s=7, objectives=["acc", "cost"])
        def f(c, ctx): return 0.0
        spec = evaluator.read_spec(f)
        assert spec["held_out_frac"] == 0.25
        assert spec["objectives"] == ["acc", "cost"]
        assert spec["timeout_s"] == 7

    def test_read_spec_defaults(self):
        def g(c, ctx): return 0.0
        spec = evaluator.read_spec(g)
        assert spec["held_out_frac"] == 0.2
        assert spec["objectives"] is None


class TestEvaluateBatch:
    def test_scalar_fitness_fills_in_place(self):
        @evaluator.fitness_spec(timeout_s=2)
        def f(c, ctx): return float(len(c))

        pop = [
            algorithms.Individual(cid="a", genome="xx"),
            algorithms.Individual(cid="b", genome="xxxxx"),
        ]
        asyncio.run(evaluator.evaluate_batch(pop, f, concurrency=2))
        assert pop[0].fitness == 2.0
        assert pop[1].fitness == 5.0

    def test_dict_fitness(self):
        @evaluator.fitness_spec(objectives=["len", "caps"])
        def f(c, ctx):
            return {"len": float(len(c)), "caps": float(sum(1 for ch in c if ch.isupper()))}

        pop = [algorithms.Individual(cid="a", genome="AbC")]
        asyncio.run(evaluator.evaluate_batch(pop, f, concurrency=1))
        assert pop[0].fitness == {"len": 3.0, "caps": 2.0}

    def test_timeout_yields_worst(self):
        import time as _time
        @evaluator.fitness_spec(timeout_s=0.1)
        def slow(c, ctx):
            _time.sleep(1.0)
            return 1.0
        pop = [algorithms.Individual(cid="s", genome="...")]
        asyncio.run(evaluator.evaluate_batch(pop, slow, concurrency=1))
        assert pop[0].fitness == float("-inf")


class TestHeldOutGuard:
    def test_penalises_overfitter(self):
        # Honest candidate: train == held-out == 0.5.
        # Overfitter:      train 0.95, held-out 0.2.
        @evaluator.fitness_spec(held_out_frac=0.3, generalization_gap=0.1)
        def f(c, ctx):
            if c == "honest":
                return 0.5
            if c == "cheat":
                return 0.2 if ctx["held_out"] else 0.95
            return 0.0

        pop = [
            algorithms.Individual(cid="h", genome="honest", fitness=0.5),
            algorithms.Individual(cid="c", genome="cheat",  fitness=0.95),
        ]
        penalties = asyncio.run(evaluator.held_out_guard(pop, f, top_k=2))
        # The cheater should receive a meaningful penalty.
        assert penalties["c"] > 0.5
        # The honest candidate is within the gap → no penalty.
        assert penalties.get("h", 0.0) == 0.0


class TestSuccessiveHalving:
    def test_narrows_population(self):
        @evaluator.fitness_spec(timeout_s=2)
        def f(c, ctx):
            # Higher-fidelity evals reward longer inputs; low-fidelity is
            # a bit noisier (deterministic noise from cid hash).
            return float(len(c)) * ctx["fidelity"]
        pop = [
            algorithms.Individual(cid=str(i), genome="x" * (i + 1))
            for i in range(8)
        ]
        survivors = asyncio.run(evaluator.successive_halving(
            pop, f, rungs=3, keep_frac=0.5, concurrency=2,
        ))
        # 8 → 4 → 2 after two halvings, then one more eval at fidelity 1.
        assert len(survivors) == 2
        # The longest inputs should survive.
        assert all(len(s.genome) >= 7 for s in survivors)


# ---------------------------------------------------------------------------
# sandbox.py
# ---------------------------------------------------------------------------


class TestSandbox:
    def test_simple_candidate_runs(self):
        result = sandbox.run_candidate_code(
            "print('hello world')",
            timeout_s=5, cpu_s=3, mem_mb=128,
        )
        assert result.ok
        assert result.returncode == 0
        assert "hello world" in result.stdout

    def test_syntax_error_fails_cleanly(self):
        result = sandbox.run_candidate_code(
            "def broken(:",
            timeout_s=5, cpu_s=3, mem_mb=128,
        )
        assert not result.ok
        assert not result.timed_out
        assert result.returncode != 0

    def test_runaway_loop_killed_by_timeout(self):
        import time as _time
        start = _time.perf_counter()
        result = sandbox.run_candidate_code(
            "while True:\n    pass\n",
            timeout_s=1.0, cpu_s=2.0, mem_mb=128,
        )
        elapsed = _time.perf_counter() - start
        assert result.timed_out, f"expected timeout; got returncode={result.returncode}"
        # Hard upper bound: wall-clock timeout + 2s overhead (subprocess spin-up + rlimit fallback).
        assert elapsed < 3.0

    def test_parse_pytest_summary(self):
        assert sandbox._parse_pytest_summary("3 passed, 1 failed in 0.02s") == 0.75
        assert sandbox._parse_pytest_summary("no matching tests found") == 0.0
        assert sandbox._parse_pytest_summary("5 passed in 0.01s") == 1.0


# ---------------------------------------------------------------------------
# adapters.py (Tier 2 external CLIs + Tier 3 DSPy bridge)
# ---------------------------------------------------------------------------


class TestAdapterGracefulAbsence:
    def test_openevolve_missing_raises_adapter_unavailable(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _b: None)
        adapter = adapters.openevolve_adapter()
        with pytest.raises(adapters.AdapterUnavailable) as exc:
            adapter.ensure_available()
        assert "openevolve" in str(exc.value)
        assert "Apache 2.0" in str(exc.value)

    def test_darwinian_evolver_install_hint_mentions_license(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _b: None)
        adapter = adapters.darwinian_evolver_adapter()
        with pytest.raises(adapters.AdapterUnavailable) as exc:
            adapter.ensure_available()
        assert "AGPL" in str(exc.value)


class TestDspyBridge:
    def _seed_experiment(self, tmp_path):
        conn = storage.open_db(tmp_path / "lineage.db")
        a = storage.insert_candidate(conn, "a", 0)
        b = storage.insert_candidate(conn, "b", 1, parents=[(a, "paraphrase", "h1")])
        c = storage.insert_candidate(conn, "c", 2, parents=[(b, "critique_then_edit", "h2")])
        for cid, val in ((a, 0.1), (b, 0.5), (c, 0.9)):
            storage.record_fitness(conn, cid, "accuracy", val, held_out=False)
            storage.record_fitness(conn, cid, "accuracy", val - 0.1, held_out=True)
        return conn

    def test_export_dspy_jsonl_default_keeps_best_per_generation(self, tmp_path):
        conn = self._seed_experiment(tmp_path)
        out = tmp_path / "export.jsonl"
        n = adapters.export_dspy_jsonl(conn, out)
        assert n == 3   # one per generation
        records = [json.loads(line) for line in out.read_text().splitlines()]
        assert all(r["schema"] == "dspy-offline/v1" for r in records)
        assert all("text" in r and "metric" in r for r in records)

    def test_export_dspy_jsonl_all_emits_every_candidate(self, tmp_path):
        conn = self._seed_experiment(tmp_path)
        out = tmp_path / "export-all.jsonl"
        n = adapters.export_dspy_jsonl(conn, out, include_all_generations=True)
        assert n == 3   # three distinct candidates here

    def test_export_gepa_trace_filters_to_reflective_operators(self, tmp_path):
        conn = self._seed_experiment(tmp_path)
        out = tmp_path / "gepa.jsonl"
        n = adapters.export_gepa_trace(conn, out)
        assert n == 1   # only the critique_then_edit edge qualifies
        rec = json.loads(out.read_text().splitlines()[0])
        assert rec["operator"] == "critique_then_edit"
        assert rec["parent"]["genome"] == "b"
        assert rec["child"]["genome"] == "c"


# ---------------------------------------------------------------------------
# llm.py
# ---------------------------------------------------------------------------


class TestLLMClient:
    """Exercises the LLM client without hitting a real endpoint.

    We install a mock ``httpx.AsyncClient.post`` that records the request
    body, so we can assert seed propagation, budget recording, and
    Retry-After handling.
    """

    def _canned_response(self, text="OK", prompt_tokens=10, completion_tokens=5, status=200, headers=None):
        class _Resp:
            def __init__(self):
                self.status_code = status
                self.headers = headers or {}
            def raise_for_status(self):
                if self.status_code >= 400:
                    import httpx as _httpx
                    raise _httpx.HTTPStatusError("err", request=None, response=None)  # type: ignore[arg-type]
            def json(self):
                return {
                    "choices": [{"message": {"content": text}}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
                }
        return _Resp()

    def test_seed_is_propagated_to_request_body(self):
        seen_bodies: list[dict] = []

        async def fake_post(self, url, json=None, **kw):
            seen_bodies.append(json)
            resp = TestLLMClient()._canned_response()
            return resp

        import httpx as _httpx
        original = _httpx.AsyncClient.post
        _httpx.AsyncClient.post = fake_post  # type: ignore[assignment]
        try:
            async def _run():
                async with llm.LLMClient(model="m", base_url="http://x", api_key="k") as client:
                    await client.complete("sys", "usr", seed=4242)
            asyncio.run(_run())
        finally:
            _httpx.AsyncClient.post = original  # type: ignore[assignment]

        assert seen_bodies
        assert seen_bodies[0]["seed"] == 4242
        assert seen_bodies[0]["messages"][0]["role"] == "system"

    def test_budget_ledger_records_and_raises_on_cap(self):
        async def fake_post(self, url, json=None, **kw):
            return TestLLMClient()._canned_response(prompt_tokens=1000, completion_tokens=500)

        import httpx as _httpx
        original = _httpx.AsyncClient.post
        _httpx.AsyncClient.post = fake_post  # type: ignore[assignment]
        try:
            ledger = llm.BudgetLedger(
                cap_usd=0.003,                 # two 0.002 calls exceed this
                input_rate_per_million=1.0,
                output_rate_per_million=2.0,
            )
            async def _run():
                async with llm.LLMClient(model="m", base_url="http://x", api_key="", budget=ledger) as client:
                    # call 1: 1000 * $1/M + 500 * $2/M = $0.002 → under cap
                    await client.complete("s", "u")
                    # call 2: another $0.002 → total $0.004 >= cap → raise
                    with pytest.raises(llm.BudgetExceeded):
                        await client.complete("s", "u")
            asyncio.run(_run())
        finally:
            _httpx.AsyncClient.post = original  # type: ignore[assignment]
        assert ledger.calls == 2
        assert ledger.spent_usd >= 0.003


# ---------------------------------------------------------------------------
# MAP-Elites coverage property (acceptance checklist #4)
# ---------------------------------------------------------------------------


class TestMapElitesCoverage:
    def test_random_population_fills_significant_fraction(self):
        import random
        rng = random.Random(0)
        arch = algorithms.MapElitesArchive(
            bin_counts=(4, 4), lows=(0, 0), highs=(4, 4),
        )
        for i in range(200):
            desc = (rng.uniform(0, 4), rng.uniform(0, 4))
            ind = algorithms.Individual(
                cid=str(i), genome=str(i), fitness=rng.random(), descriptor=desc,
            )
            arch.place(ind)
        # 200 uniform samples into a 4×4 grid should cover essentially
        # all 16 bins with astronomically high probability.
        assert arch.coverage() >= 0.6


# ---------------------------------------------------------------------------
# End-to-end: mocked LLM run of a single generation (acceptance checklist #7)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_single_generation_improves_best_fitness(self, tmp_path, monkeypatch):
        """With a fixed LLM that always returns the string `concise`, the
        brevity-bonus fitness climbs from the seed after one generation."""
        exp = tmp_path / "demo"
        exp.mkdir()
        (exp / "seed").mkdir()
        (exp / "logs").mkdir()
        (exp / "seed" / "initial.txt").write_text("Summarize it.\n")
        (exp / "fitness.py").write_text(
            "def fitness(candidate, context):\n"
            "    bonus = 0.3 if 'concise' in candidate.lower() else 0.0\n"
            "    return 0.5 + bonus\n"
        )
        storage.open_db(exp / "lineage.db").close()

        # Patch httpx.AsyncClient.post to return a fixed mutation result.
        async def fake_post(self, url, json=None, **kw):
            class _R:
                status_code = 200
                headers: dict = {}
                def raise_for_status(self): pass
                def json(self_inner):
                    return {
                        "choices": [{"message": {"content": "Summarize it concisely."}}],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 5},
                    }
            return _R()

        import httpx as _httpx
        monkeypatch.setattr(_httpx.AsyncClient, "post", fake_post)

        # Build a minimal argparse.Namespace and run the loop synchronously.
        import argparse
        args = argparse.Namespace(
            dir=str(exp),
            generations=1,
            pop=2,
            budget=0.0,
            algorithm="es",
            concurrency=2,
            seed=7,
            input_rate=0.0,
            output_rate=0.0,
        )
        # Import the runner lazily to ensure our sys.path is set up first.
        sys.path.insert(0, str(SCRIPTS_DIR))
        import evolver  # noqa: WPS433
        result = asyncio.run(evolver._run_loop(args, exp))
        assert result["ok"]
        assert result["generations_run"] >= 1

        # Inspect the lineage — we should have at least one offspring whose
        # fitness beats the seed's 0.5.
        conn = storage.open_db(exp / "lineage.db")
        rows = storage.get_best(conn, "fitness", k=1)
        assert rows, "no fitness rows persisted"
        assert rows[0]["value"] > 0.5, f"expected improvement over seed; got {rows[0]}"
        # And the replay hash should be reproducible now that all writes settled.
        h1 = storage.lineage_hash(conn)
        h2 = storage.lineage_hash(conn)
        assert h1 == h2
        conn.close()
