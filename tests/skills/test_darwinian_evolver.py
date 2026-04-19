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
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills" / "research" / "darwinian-evolver" / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import algorithms  # noqa: E402
import evaluator   # noqa: E402
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
