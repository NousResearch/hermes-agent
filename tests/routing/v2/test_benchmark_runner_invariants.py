"""Invariant / property-based TDD for benchmark_runner (suite B of double TDD).

These tests don't assert exact values. They assert LAWS that must hold for
every input drawn from broad distributions. If a future refactor breaks any
law, one of these catches it — complementary to the spec suite.

No external deps (stdlib only): uses random with fixed seeds as a cheap
property generator.
"""
from __future__ import annotations

import random

import pytest

from agent import benchmark_runner as br


def _suite(rng: random.Random, n_cats=3, n_models=4, n_prompts=5):
    cats = [f"cat_{i}" for i in range(n_cats)]
    models = [f"model_{j}" for j in range(n_models)]
    return {c: {m: [f"p_{k}" for k in range(n_prompts)] for m in models} for c in cats}


def _evaluator(scores):
    def _ev(category, model, prompt):
        return {
            "score": scores[(category, model, prompt)],
            "latency_ms": 100.0,
            "cost_usd": 0.0005,
        }
    return _ev


@pytest.mark.parametrize("seed", list(range(5)))
def test_scores_always_in_unit_interval(seed: int):
    rng = random.Random(seed)
    suite = _suite(rng)
    scores = {
        (c, m, p): rng.uniform(0, 1)
        for c, models in suite.items() for m, prompts in models.items() for p in prompts
    }
    rep = br.run_benchmarks(suite, _evaluator(scores), seed=seed)
    for cat, models in rep.categories.items():
        for model, r in models.items():
            for s in r.raw_scores:
                assert 0.0 <= s <= 1.0, f"score out of range: {s}"
            assert 0.0 <= r.score_mean <= 1.0
            assert r.score_ci_low <= r.score_mean <= r.score_ci_high


@pytest.mark.parametrize("seed", list(range(5)))
def test_ranking_is_total_and_stable(seed: int):
    rng = random.Random(seed)
    suite = _suite(rng)
    scores = {
        (c, m, p): rng.uniform(0, 1)
        for c, models in suite.items() for m, prompts in models.items() for p in prompts
    }
    rep = br.run_benchmarks(suite, _evaluator(scores), seed=seed)
    for cat in rep.categories:
        ranked = br.rank_models(rep, cat)
        # all unique, sorted descending
        assert len(ranked) == len(rep.categories[cat])
        for a, b in zip(ranked, ranked[1:]):
            assert a[1] >= b[1]


@pytest.mark.parametrize("seed", list(range(5)))
def test_determinism_under_fixed_seed(seed: int):
    rng = random.Random(seed)
    suite = _suite(rng)
    scores = {
        (c, m, p): rng.uniform(0, 1)
        for c, models in suite.items() for m, prompts in models.items() for p in prompts
    }
    ev = _evaluator(scores)
    r1 = br.run_benchmarks(suite, ev, seed=123)
    r2 = br.run_benchmarks(suite, ev, seed=123)
    # Same evaluator + same seed => identical aggregates
    for cat in r1.categories:
        for model in r1.categories[cat]:
            a = r1.categories[cat][model]
            b = r2.categories[cat][model]
            assert a.score_mean == b.score_mean
            assert a.score_ci_low == b.score_ci_low
            assert a.score_ci_high == b.score_ci_high


def test_regression_is_transitive_with_threshold():
    # If model drops by >threshold between runs, detection must flag it.
    suite = {"code": {"m1": ["p1", "p2", "p3"]}}
    prev = br.run_benchmarks(
        suite, lambda c, m, p: {"score": 0.9, "latency_ms": 10, "cost_usd": 0}, seed=0
    )
    curr = br.run_benchmarks(
        suite, lambda c, m, p: {"score": 0.5, "latency_ms": 10, "cost_usd": 0}, seed=0
    )
    regs = br.detect_regressions(prev, curr, threshold=0.05)
    assert len(regs) == 1
    # Below threshold -> no regression
    curr_mild = br.run_benchmarks(
        suite, lambda c, m, p: {"score": 0.88, "latency_ms": 10, "cost_usd": 0}, seed=0
    )
    assert br.detect_regressions(prev, curr_mild, threshold=0.05) == []


def test_latency_and_cost_penalize_weighted_score():
    fast_cheap = br.weighted_score(0.8, latency_ms=50, cost_usd=0.0001)
    slow_expensive = br.weighted_score(0.8, latency_ms=5000, cost_usd=0.05)
    assert fast_cheap > slow_expensive


def test_percentile_monotonic():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert br._percentile(vals, 50) <= br._percentile(vals, 95)
    assert br._percentile(vals, 95) <= br._percentile(vals, 100)
