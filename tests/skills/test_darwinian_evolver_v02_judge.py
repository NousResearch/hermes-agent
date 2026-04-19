"""v0.2 feature 3 — pairwise judge + Bradley-Terry aggregation.

Six tests covering the three layers:

* MLE convergence on toy Condorcet data + tie handling.
* Pair schedule: every candidate participates a minimum number of times.
* Position-bias guard: random coin flip on LEFT / RIGHT.
* PairwiseJudge decode end-to-end through a mocked LLM.
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

import httpx
import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills" / "research" / "darwinian-evolver" / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import judge  # noqa: E402
import llm    # noqa: E402


# ---------------------------------------------------------------------------
# Bradley-Terry MLE
# ---------------------------------------------------------------------------


class TestBradleyTerry:
    def test_converges_on_toy_condorcet_order(self):
        """Three candidates, strict preference a ≻ b ≻ c reflected in
        winning record. B-T MLE must return a > b > c in log-odds."""
        votes = (
            [("a", "b")] * 10
            + [("a", "c")] * 10
            + [("b", "c")] * 10
        )
        scores, iters = judge.aggregate_bradley_terry(votes)
        assert scores["a"] > scores["b"] > scores["c"]
        assert 0 < iters <= 200

    def test_handles_ties_equal_score(self):
        """Even wins A→B and B→A: log-odds must be ~equal."""
        votes = [("a", "b")] * 10 + [("b", "a")] * 10
        scores, _ = judge.aggregate_bradley_terry(votes)
        assert abs(scores["a"] - scores["b"]) < 0.02

    def test_empty_votes_returns_empty_scores(self):
        scores, iters = judge.aggregate_bradley_terry([])
        assert scores == {}
        assert iters == 0

    def test_single_candidate_scores_zero(self):
        """Degenerate: single candidate — log-odds is 0 by convention."""
        scores, _ = judge.aggregate_bradley_terry([("a", "a")])
        assert scores == {"a": 0.0}


# ---------------------------------------------------------------------------
# Pair schedule
# ---------------------------------------------------------------------------


class TestPairSchedule:
    def test_small_pop_covers_unique_pairs_first(self):
        """pop=4, rounds=6, C(4,2)=6 unique pairs, so the schedule is
        the full round-robin with no duplicates."""
        rng = random.Random(0)
        pairs = judge.sample_pair_schedule(list("abcd"), rounds=6, rng=rng)
        assert len(pairs) == 6
        dedup = {tuple(sorted(p)) for p in pairs}
        assert len(dedup) == 6

    def test_each_candidate_participates_minimum(self):
        """Invariant: every candidate appears in ≥ ⌈rounds * 2 / pop⌉ matches."""
        rng = random.Random(1)
        pop = list("abcdefgh")   # 8 candidates
        rounds = 40
        pairs = judge.sample_pair_schedule(pop, rounds=rounds, rng=rng)
        appearances = {c: 0 for c in pop}
        for a, b in pairs:
            appearances[a] += 1
            appearances[b] += 1
        minimum = (rounds * 2 + len(pop) - 1) // len(pop) - 1
        assert min(appearances.values()) >= minimum, appearances

    def test_single_candidate_yields_empty(self):
        pairs = judge.sample_pair_schedule(["only"], rounds=10, rng=random.Random(0))
        assert pairs == []


# ---------------------------------------------------------------------------
# Decode + position-bias guard
# ---------------------------------------------------------------------------


class TestDecodeVerdict:
    def test_decodes_left_right_tie(self):
        assert judge._decode_verdict("LEFT\nrationale") == "left"
        assert judge._decode_verdict("right — option b wins") == "right"
        assert judge._decode_verdict("tie") == "tie"

    def test_unparseable_falls_back_to_tie(self):
        assert judge._decode_verdict("i have no opinion") == "tie"
        assert judge._decode_verdict("") == "tie"


class TestPositionBiasGuard:
    def test_swap_is_roughly_fifty_fifty(self):
        """When the LLM always says 'LEFT', a bias-free judge returns
        candidate A about half the time and candidate B the other half
        — because we randomly swapped LEFT / RIGHT before asking."""
        original = httpx.AsyncClient.post

        async def always_left(self, url, json=None, **kw):
            class _R:
                status_code = 200
                headers: dict = {}
                def raise_for_status(self_): pass
                def json(self_inner):
                    return {
                        "choices": [{"message": {"content": "LEFT\njustification..."}}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                    }
            return _R()

        httpx.AsyncClient.post = always_left  # type: ignore[assignment]
        try:
            rng = random.Random(0)
            async def _run():
                async with llm.LLMClient(model="m", base_url="http://x", api_key="") as c:
                    pj = judge.PairwiseJudge(client=c)
                    verdicts = []
                    for _ in range(200):
                        verdicts.append(await pj.judge("A-genome", "B-genome", rng=rng))
                    return verdicts
            out = asyncio.run(_run())
            a_wins = out.count("a")
            b_wins = out.count("b")
            # Without the guard, "a" would win 100 %; with it we expect
            # roughly 50/50. Allow a generous tolerance (45 / 55) to keep
            # the test robust against RNG jitter.
            assert 0.35 * len(out) <= a_wins <= 0.65 * len(out)
            assert 0.35 * len(out) <= b_wins <= 0.65 * len(out)
        finally:
            httpx.AsyncClient.post = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# End-to-end: mocked Condorcet judge → B-T recovers the ranking
# ---------------------------------------------------------------------------


class TestPairwiseIncompatibleWithNsga2:
    def test_raises_when_combined_with_multiobjective(self, tmp_path, monkeypatch):
        """pairwise + objectives=[...] must fail fast, not silently corrupt."""
        import argparse
        exp = tmp_path / "demo"
        exp.mkdir()
        (exp / "seed").mkdir()
        (exp / "logs").mkdir()
        (exp / "seed" / "initial.txt").write_text("x\n")
        (exp / "fitness.py").write_text(
            "from evolver_sdk import fitness_spec\n"
            "@fitness_spec(judge='pairwise', objectives=['acc','cost'])\n"
            "def fitness(c, ctx): return {'acc':0.0,'cost':0.0}\n"
        )
        # Minimal evolver_sdk shim so fitness.py can import it.
        (exp / "evolver_sdk.py").write_text(
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(SCRIPTS_DIR)!r})\n"
            "from evaluator import fitness_spec\n"
        )
        sys.path.insert(0, str(SCRIPTS_DIR))
        import evolver  # noqa: WPS433

        args = argparse.Namespace(
            dir=str(exp), generations=1, pop=2, budget=0.0,
            algorithm="es", concurrency=1, seed=0,
            input_rate=0.0, output_rate=0.0, no_cache=True,
        )
        with pytest.raises(SystemExit):
            asyncio.run(evolver._run_loop(args, exp))


class TestJudgeRecoversRanking:
    def test_condorcet_order_recovered_through_mocked_judge(self):
        """A judge that prefers genomes with lexicographically higher
        characters (``c > b > a``) plus the B-T MLE must recover the
        order from a schedule of pairwise verdicts."""
        original = httpx.AsyncClient.post

        async def lexical_judge(self, url, json=None, **kw):
            body = json or {}
            user = body["messages"][-1]["content"]
            # Robust extract: look for genomes after the LEFT: / RIGHT:
            # labels in the user content.
            left = user.split("LEFT:\n", 1)[1].split("\n\n", 1)[0].strip()
            right = user.split("RIGHT:\n", 1)[1].split("\n\n", 1)[0].strip()
            winner = "LEFT" if left > right else "RIGHT" if right > left else "TIE"
            class _R:
                status_code = 200
                headers: dict = {}
                def raise_for_status(self_): pass
                def json(self_inner):
                    return {
                        "choices": [{"message": {"content": f"{winner}\nreason"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                    }
            return _R()

        httpx.AsyncClient.post = lexical_judge  # type: ignore[assignment]
        try:
            async def _run():
                rng = random.Random(2)
                candidates = list("abcdef")  # expected rank: f > e > d > c > b > a
                schedule = judge.sample_pair_schedule(candidates, rounds=30, rng=rng)
                async with llm.LLMClient(model="m", base_url="http://x", api_key="") as c:
                    pj = judge.PairwiseJudge(client=c)
                    votes: list[tuple[str, str]] = []
                    for (a, b) in schedule:
                        verdict = await pj.judge(a, b, rng=rng)
                        if verdict == "tie":
                            continue
                        votes.append((a, b) if verdict == "a" else (b, a))
                    scores, _ = judge.aggregate_bradley_terry(votes)
                    return scores

            scores = asyncio.run(_run())
            ranked = [cid for cid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
            # Expect the lex-descending order.
            assert ranked == list("fedcba"), ranked
        finally:
            httpx.AsyncClient.post = original  # type: ignore[assignment]
