"""Spec TDD for delegation_policy + planner (suite A of double TDD)."""
from __future__ import annotations

import pytest

from agent import delegation_policy as dp
from agent import routing_v2 as rv2
from agent import benchmark_harness as bh
from agent import benchmark_runner as br


@pytest.fixture(scope="module")
def benches():
    rep = br.run_benchmarks(bh.DEFAULT_SUITE, bh.heuristic_evaluator, seed=0)
    return br.report_to_benchmarks(rep)


# ---------- should_delegate ----------

def test_should_delegate_on_tier4_code():
    d = {"tier": 4, "model": "qwen3-coder-next", "category": "code"}
    ok, why = dp.should_delegate("refactor X", d)
    assert ok is True and why == "tier_or_non_cheap"


def test_should_not_delegate_on_exception_greeting():
    d = {"tier": 1, "model": "glm-5.1", "category": "simple"}
    ok, why = dp.should_delegate("hola", d)
    assert ok is False and why.startswith("exception:greeting")


def test_should_not_delegate_on_trivial_time():
    d = {"tier": 1, "model": "glm-5.1", "category": "simple"}
    ok, why = dp.should_delegate("what time is it", d)
    assert ok is False and why.startswith("exception:time_query")


def test_should_delegate_non_cheap_tier1():
    d = {"tier": 1, "model": "kimi-k2.5", "category": "simple"}
    ok, why = dp.should_delegate("proxy", d)
    assert ok is True


def test_should_respect_user_exceptions():
    d = {"tier": 4, "model": "qwen3-coder-next", "category": "code"}
    ok, why = dp.should_delegate("do local quick thing", d, exceptions=[r"local quick"])
    assert ok is False and why.startswith("exception:user_pattern")


# ---------- rate helpers ----------

def test_compute_rate_and_floor():
    ds = [{"delegate": True}] * 8 + [{"delegate": False}] * 2
    rate = dp.compute_delegation_rate(ds)
    assert rate == 80.0
    assert dp.assert_rate_ok(rate, floor=75.0) is True
    assert dp.assert_rate_ok(50.0, floor=75.0) is False


# ---------- plan_subagents ----------

def test_plan_bullets_split():
    prompt = "please do:\n- migrate DB\n- rewrite client\n- update docs"
    d = {"tier": 4, "model": "qwen3-coder-next", "category": "code"}
    plan = dp.plan_subagents(prompt, d, max_parallel=3)
    assert plan["count"] == 3
    assert "bullet_split" in plan["rationale"]


def test_plan_code_multiple_files():
    prompt = "fix bug in tools/foo.py and utils/bar.py"
    d = {"tier": 4, "model": "qwen3-coder-next", "category": "code"}
    plan = dp.plan_subagents(prompt, d, max_parallel=3)
    assert plan["count"] == 2
    assert any("foo.py" in s for s in plan["subtasks"])


def test_plan_research_vs_split():
    prompt = "compare postgres vs mysql vs sqlite replication"
    d = {"tier": 4, "model": "deepseek-v3.2", "category": "research"}
    plan = dp.plan_subagents(prompt, d, max_parallel=3)
    assert plan["count"] == 3
    assert "vs_split" in plan["rationale"]


def test_plan_writing_n_drafts():
    prompt = "give me 3 drafts of a release note"
    d = {"tier": 4, "model": "mistral-large-3:675b", "category": "writing"}
    plan = dp.plan_subagents(prompt, d, max_parallel=5)
    assert plan["count"] == 3
    assert "drafts_count" in plan["rationale"]


def test_plan_default_single_task():
    prompt = "brief analysis of this diagram"
    d = {"tier": 4, "model": "qwen3.5:397b", "category": "analysis"}
    plan = dp.plan_subagents(prompt, d, max_parallel=3)
    assert plan["count"] == 1
    assert plan["subtasks"][0]


def test_plan_capped_at_max_parallel():
    prompt = "\n".join([f"- task {i}" for i in range(10)])
    d = {"tier": 3, "model": "kimi-k2.5", "category": "code"}
    plan = dp.plan_subagents(prompt, d, max_parallel=3)
    assert plan["count"] == 3
    assert "capped_at_3" in plan["rationale"]


def test_select_model_enrichment(benches):
    dec = rv2.select_model("refactor tools/foo.py and fix utils/bar.py",
                           benches, rv2.DEFAULT_TIERS, task_state=None)
    assert "delegate" in dec and dec["delegate"] is True
    assert "subagents" in dec and dec["subagents"]["count"] >= 2
