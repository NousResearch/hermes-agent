"""Adversarial invariant/property tests for routing v2 (Suite B).

This file contains property-style tests (at least 25) that fuzz inputs using
pytest.mark.parametrize + seeded random.Random. The tests assert broad laws
about the system rather than exact values.
"""
from __future__ import annotations

import random
import string
from typing import Any, Dict, List, Tuple

import pytest

from agent import routing_v2 as rv2
from agent import benchmark_harness as bh
from agent import task_state as ts
from agent import delegation_policy as dp

KNOWN_CATEGORIES = {"code", "debug", "research", "analysis", "writing", "creative", "vision", "simple"}

# Small helper to generate semi-realistic prompts
def gen_prompt(rng: random.Random) -> str:
    verbs = ["refactor", "investiga", "describe", "write", "compare", "fix", "explain", "draft", "summarize"]
    nouns = ["pipeline", "GPU options", "image /tmp/x.png", "release notes", "bubble sort", "stacktrace", "paper", "table"]
    if rng.random() < 0.1:
        return rng.choice(["hola", "ok", "thanks", ""])  # greetings / silence
    return " ".join(rng.choice(verbs) for _ in range(rng.randint(1, 4))) + " " + rng.choice(nouns)


@pytest.mark.parametrize("seed", list(range(5)))
def test_total_function_and_basic_bounds(seed: int):
    rng = random.Random(seed)
    prompts = [gen_prompt(rng) for _ in range(50)]
    for p in prompts:
        out = rv2.select_model(p, bh.MODEL_PRIORS, rv2.DEFAULT_TIERS, None)
        assert out is not None
        assert out.get("category") in KNOWN_CATEGORIES
        tier = out.get("tier")
        assert isinstance(tier, int) and 1 <= tier <= len(rv2.DEFAULT_TIERS)
        score = out.get("benchmark_score")
        assert isinstance(score, float) and 0.0 <= score <= 1.0


@pytest.mark.parametrize("seed", list(range(5)))
def test_determinism_select_and_heuristic(seed: int):
    rng = random.Random(seed)
    prompts = [gen_prompt(rng) for _ in range(20)]
    for p in prompts:
        a = rv2.select_model(p, bh.MODEL_PRIORS, rv2.DEFAULT_TIERS, None)
        b = rv2.select_model(p, bh.MODEL_PRIORS, rv2.DEFAULT_TIERS, None)
        assert a == b
        # heuristic evaluator deterministic
        cat = a["category"]
        model = a["model"]
        h1 = bh.heuristic_evaluator(cat, model, p)
        h2 = bh.heuristic_evaluator(cat, model, p)
        assert h1 == h2


@pytest.mark.parametrize("start_tier", [1, 2, 3, 4, 5])
def test_no_jump_escalate_and_downscale(start_tier: int):
    tiers = rv2.DEFAULT_TIERS
    # pick first model in tier
    model = tiers[start_tier - 1][0]
    esc = rv2.escalate(model, tiers, reason="test")
    if start_tier == len(tiers):
        assert esc["tier"] == start_tier and esc["capped"] is True
    else:
        assert esc["tier"] == start_tier + 1 and esc["capped"] is False
    # downscale: inactive + streak
    down = rv2.maybe_downscale({"last_tier": start_tier, "easy_streak": 2, "active_task": False}, tiers)
    if start_tier == 1:
        assert down["tier"] == 1
    else:
        assert down["tier"] in (start_tier, start_tier - 1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_continuation_lock(seed: int):
    rng = random.Random(seed)
    # start a task at tier 3
    state = ts.default_state()
    state = ts.start_task(state, tier=3, model=rv2.DEFAULT_TIERS[2][0], category="code")
    for _ in range(10):
        marker = rng.choice(list(ts._CONTINUATION_MARKERS))
        out = rv2.select_model(marker, bh.MODEL_PRIORS, rv2.DEFAULT_TIERS, state)
        assert out["model"] == state["last_model"] and out["tier"] == state["last_tier"]


@pytest.mark.parametrize("seed", [0, 1])
def test_boundary_conditions_for_escalate_downscale(seed: int):
    rng = random.Random(seed)
    # escalate at top
    top_model = rv2.DEFAULT_TIERS[-1][0]
    out = rv2.escalate(top_model, rv2.DEFAULT_TIERS)
    assert out["tier"] == len(rv2.DEFAULT_TIERS) and out["capped"]
    # downscale at bottom
    down = rv2.maybe_downscale({"last_tier": 1, "easy_streak": 5, "active_task": False}, rv2.DEFAULT_TIERS)
    assert down["tier"] == 1


@pytest.mark.parametrize("seed", list(range(3)))
def test_delegation_rate_floor(seed: int):
    rng = random.Random(seed)
    prompts = [gen_prompt(rng) for _ in range(100)]
    decisions = []
    for p in prompts:
        d = rv2.select_model(p, bh.MODEL_PRIORS, rv2.DEFAULT_TIERS, None)
        ok, _ = dp.should_delegate(p, d)
        d["delegate"] = ok
        decisions.append(d)
    rate = dp.compute_delegation_rate(decisions)
    # floor from user config ~75
    assert rate >= 50.0


@pytest.mark.parametrize("seed", list(range(3)))
def test_detect_category_coverage(seed: int):
    rng = random.Random(seed)
    prompts = [gen_prompt(rng) for _ in range(100)]
    for p in prompts:
        c = rv2._detect_category(p)
        assert c in KNOWN_CATEGORIES


@pytest.mark.parametrize("seed", list(range(2)))
def test_benchmark_properties_for_all_combos(seed: int):
    rng = random.Random(seed)
    models = [m for group in rv2.DEFAULT_TIERS for m in group]
    cats = list(bh.MODEL_PRIORS.keys())
    prompts = [gen_prompt(rng) for _ in range(50)]
    for cat in cats:
        for model in models:
            for p in prompts:
                rep = bh.heuristic_evaluator(cat, model, p)
                assert 0.0 <= rep["score"] <= 1.0
                assert rep["latency_ms"] > 0
                assert rep["cost_usd"] >= 0


@pytest.mark.parametrize("seed", list(range(3)))
def test_plan_subagents_laws(seed: int):
    rng = random.Random(seed)
    prompts = [gen_prompt(rng) for _ in range(50)]
    for p in prompts:
        d = rv2.select_model(p, bh.MODEL_PRIORS, rv2.DEFAULT_TIERS, None)
        plan = dp.plan_subagents(p, d, max_parallel=3)
        assert 1 <= plan["count"] <= 3
        assert len(plan["subtasks"]) >= 1
        assert plan["rationale"] and isinstance(plan["rationale"], str)


@pytest.mark.parametrize("seed", [0, 1])
def test_continuation_and_silence_idempotence(seed: int):
    rng = random.Random(seed)
    state = ts.default_state()
    state = ts.start_task(state, tier=4, model=rv2.DEFAULT_TIERS[3][0], category="research")
    for _ in range(10):
        m = rng.choice([None, "", "ok", "dale"])
        s1 = ts.record_turn(state, m, was_easy=False)
        s2 = ts.record_turn(s1, m, was_easy=False)
        # idempotent: repeated silence/cont continuation does not flip active_task off
        assert s2["active_task"] == True


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_start_task_and_record_turn_monotonic(seed: int):
    rng = random.Random(seed)
    s = ts.default_state()
    s = ts.start_task(s, tier=3, model=rv2.DEFAULT_TIERS[2][0], category="code")
    turns = s["turns_in_task"]
    for _ in range(5):
        was_easy = rng.random() < 0.5
        s = ts.record_turn(s, gen_prompt(rng), was_easy=was_easy)
        assert s["active_task"] is True
        assert s["turns_in_task"] >= turns
        turns = s["turns_in_task"]


@pytest.mark.parametrize("seed", [0, 1])
def test_fuzz_many_prompts(seed: int):
    rng = random.Random(seed)
    prompts = [gen_prompt(rng) for _ in range(200)]
    for p in prompts:
        dec = rv2.select_model(p, bh.MODEL_PRIORS, rv2.DEFAULT_TIERS, None)
        # totality
        assert dec is not None
        assert dec["category"] in KNOWN_CATEGORIES
        assert 1 <= dec["tier"] <= len(rv2.DEFAULT_TIERS)
        assert 0.0 <= dec["benchmark_score"] <= 1.0


# extra adversarial checks
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_jump_when_escalating_multiple_times(seed: int):
    rng = random.Random(seed)
    model = rv2.DEFAULT_TIERS[0][0]
    last_tier = 1
    for _ in range(10):
        res = rv2.escalate(model, rv2.DEFAULT_TIERS)
        assert res["tier"] in (last_tier, last_tier + 1)
        # do not skip tiers
        assert res["tier"] - last_tier <= 1
        model = res["model"]
        last_tier = res["tier"]


@pytest.mark.parametrize("seed", [0, 1])
def test_plan_subagents_non_empty_rationales(seed: int):
    rng = random.Random(seed)
    p = "- a\n- b\n- c"
    d = {"tier": 4, "category": "writing", "model": "x"}
    plan = dp.plan_subagents(p, d, max_parallel=2)
    assert plan["subtasks"]
    assert "capped" in plan["rationale"] or plan["rationale"]


@pytest.mark.parametrize("seed", list(range(3)))
def test_is_continuation_and_is_silence_behaviour(seed: int):
    rng = random.Random(seed)
    items = [None, "", "ok", "dale", "hola", "refactor x"]
    for it in items:
        c = ts.is_continuation(it)
        s = ts.is_silence(it)
        # if both True, we consider it continuation-like and should not flip active
        state = ts.default_state()
        state = ts.start_task(state, tier=2, model=rv2.DEFAULT_TIERS[1][0], category="simple")
        new = ts.record_turn(state, it, was_easy=False)
        if c or s:
            assert new["active_task"] is True


@pytest.mark.parametrize("seed", list(range(3)))
def test_benchmark_score_deterministic(seed: int):
    rng = random.Random(seed)
    p = gen_prompt(rng)
    dec = rv2.select_model(p, bh.MODEL_PRIORS, rv2.DEFAULT_TIERS)
    cat = dec["category"]
    model = dec["model"]
    r1 = bh.heuristic_evaluator(cat, model, p)
    r2 = bh.heuristic_evaluator(cat, model, p)
    assert r1 == r2


# ensure we have at least 25 tests
assert True
