"""Invariant TDD for delegation_policy + planner (suite B of double TDD)."""
from __future__ import annotations

import random

import pytest

from agent import delegation_policy as dp


GREETINGS = ["hola", "hi", "hello", "ok", "gracias", "thanks"]
TIME_QUERIES = ["what time is it", "que hora es", "what's the date", "cual es la fecha"]


@pytest.mark.parametrize("g", GREETINGS)
def test_greeting_is_always_local(g):
    d = {"tier": 1, "model": "glm-5.1", "category": "simple"}
    ok, _ = dp.should_delegate(g, d)
    assert ok is False


@pytest.mark.parametrize("q", TIME_QUERIES)
def test_time_queries_are_local(q):
    d = {"tier": 1, "model": "glm-5.1", "category": "simple"}
    ok, _ = dp.should_delegate(q, d)
    assert ok is False


@pytest.mark.parametrize("tier", [3, 4, 5])
def test_high_tier_always_delegates_when_not_exception(tier):
    d = {"tier": tier, "model": "qwen3-coder-next", "category": "code"}
    ok, _ = dp.should_delegate("refactor pipeline please", d)
    assert ok is True


@pytest.mark.parametrize("seed", range(5))
def test_rate_floor_holds_on_mixed_batch(seed: int):
    rng = random.Random(seed)
    prompts = ["hola", "refactor foo.py", "analyze latency", "what time",
               "investiga GPUs", "draft release notes",
               "describe image /t/x.png", "fix stacktrace"]
    decisions = []
    for _ in range(20):
        p = rng.choice(prompts)
        # simulate a reasonable decision based on length
        if p in ("hola", "what time"):
            dec = {"tier": 1, "model": "glm-5.1", "category": "simple"}
        else:
            dec = {"tier": 4, "model": "qwen3-coder-next", "category": "code"}
        ok, _ = dp.should_delegate(p, dec)
        dec["delegate"] = ok
        decisions.append(dec)
    rate = dp.compute_delegation_rate(decisions)
    # With max 2/8 exceptions in our prompt pool, the bulk must delegate
    assert rate >= 50.0  # loose law; strict floor asserted elsewhere


def test_planner_cap_is_always_respected():
    prompt = "\n".join([f"- item {i}" for i in range(20)])
    d = {"tier": 4, "model": "qwen3-coder-next", "category": "code"}
    for cap in [1, 2, 3, 5]:
        plan = dp.plan_subagents(prompt, d, max_parallel=cap)
        assert plan["count"] <= cap
        assert len(plan["subtasks"]) == plan["count"]


@pytest.mark.parametrize("tier", [4, 5])
def test_high_tier_preserves_quality_tag(tier):
    plan = dp.plan_subagents("- a\n- b", {"tier": tier, "category": "code", "model": "x"}, max_parallel=3)
    assert f"quality_tier{tier}_inherit" in plan["rationale"]


def test_planner_is_deterministic():
    prompt = "compare postgres vs mysql vs sqlite replication"
    d = {"tier": 4, "model": "deepseek-v3.2", "category": "research"}
    p1 = dp.plan_subagents(prompt, d, max_parallel=3)
    p2 = dp.plan_subagents(prompt, d, max_parallel=3)
    assert p1 == p2
