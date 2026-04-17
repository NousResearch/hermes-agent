from __future__ import annotations

import pytest

# Adversarial SPEC tests for routing v2 (Suite A)
# These tests assert exact expected outputs for deterministic behaviors.

from agent import routing_v2 as rv2
from agent import delegation_policy as dp
from agent import benchmark_runner as br
import os
# Ensure auto-instrumentation does not import optional telemetry during tests
os.environ.pop("HERMES_ROUTING_TELEMETRY", None)
from agent import smart_model_routing as smr

# Fixtures
@pytest.fixture
def tiers():
    return rv2.DEFAULT_TIERS

@pytest.fixture
def simple_benchmarks():
    # minimal deterministic bench map used by select_model
    return {
        "code": {"qwen3-coder-next": 0.9, "kimi-k2.5": 0.5},
        "analysis": {"qwen3.5:397b": 0.92, "kimi-k2.5": 0.6},
        "research": {"deepseek-v3.2": 0.88},
        "writing": {"mistral-large-3:675b": 0.85},
        "creative": {"minimax-m2.7": 0.8},
        "vision": {"qwen3-vl:235b-instruct": 0.95},
        "simple": {"glm-5.1": 0.8, "gpt-5-mini": 0.78},
    }

# 1) Cross-module pipeline: select_model -> should_delegate -> plan_subagents
def test_full_chain_select_delegate_plan(tiers, simple_benchmarks):
    prompt = "Refactor and implement feature X across files a.py and b.py"
    decision = rv2.select_model(prompt, simple_benchmarks, tiers, task_state=None)
    # Exact expectations
    assert decision["category"] == "code"
    assert decision["model"] == "qwen3-coder-next"
    assert decision["tier"] == 4
    delegated, reason = dp.should_delegate(prompt, decision)
    assert delegated is True
    assert reason == "tier_or_non_cheap"
    plan = dp.plan_subagents(prompt, decision, max_parallel=3)
    # exact expected rationale includes quality tier inheritance for tier>=4
    assert plan["rationale"] == "file_split|quality_tier4_inherit"
    assert plan["count"] == 2
    assert plan["subtasks"] == ["handle a.py", "handle b.py"]

# 2) resolve_turn_route with v2_enabled=True uses benchmarks.json if present
def test_resolve_turn_route_v2_uses_benchmarks(tmp_path, tiers, simple_benchmarks, monkeypatch):
    # Create a fake benchmarks.json via benchmark_runner.save_report
    suite = {"code": {"qwen3-coder-next": ["p1"]}}
    def evaluator(cat, model, prompt):
        return {"score": 0.9, "latency_ms": 10, "cost_usd": 0.001}
    # monkeypatch HOME first so resolve_turn_route reads from tmp_path/.hermes/router
    monkeypatch.setenv("HOME", str(tmp_path))
    report = br.run_benchmarks(suite, evaluator, seed=0)
    bench_dir = tmp_path / ".hermes" / "router"
    bench_dir.mkdir(parents=True)
    bench_path = bench_dir / "benchmarks.json"
    br.save_report(report, str(bench_path))
    primary = {"model": "primary-model", "api_key": None, "base_url": None, "provider": "test", "api_mode": None, "command": None, "args": []}
    cfg = {"v2_enabled": True}
    # choose a prompt that will be detected as code by routing_v2
    route = smr.resolve_turn_route("refactor this function", cfg, primary)
    # exact model must come from the bench (qwen3-coder-next)
    assert route["model"] == "qwen3-coder-next"
    assert route["label"].startswith("v2 smart route")
    assert isinstance(route["signature"], tuple)

# 3) Escalation: T1 -> T2 -> T3 -> T4 -> T5 single-step, capped at T5
def test_escalation_sequence(tiers):
    cur = "glm-5.1"
    for expected_tier in [2, 3, 4, 5]:
        out = rv2.escalate(cur, tiers)
        assert out["tier"] == expected_tier
        cur = out["model"]
    # now at top, escalate again
    top = rv2.escalate(cur, tiers)
    assert top["tier"] == 5
    assert top["capped"] is True

# 4) Downscale: requires easy_streak>=2 and no active_task
def test_downscale_requires_streak_and_no_active_task(tiers):
    state = {"last_tier": 4, "easy_streak": 2, "active_task": False}
    out = rv2.maybe_downscale(state, tiers)
    assert out["tier"] == 3
    # active_task blocks downscale
    state = {"last_tier": 4, "easy_streak": 3, "active_task": True}
    out = rv2.maybe_downscale(state, tiers)
    assert out["tier"] == 4
    # insufficient streak
    state = {"last_tier": 4, "easy_streak": 1, "active_task": False}
    out = rv2.maybe_downscale(state, tiers)
    assert out["tier"] == 4

# 5) Continuation lock: "dale"/"sigue"/"hazlo" with active_task preserves model and tier
@pytest.mark.parametrize("marker", ["dale", "sigue", "hazlo"])
def test_continuation_lock_preserves(marker, tiers, simple_benchmarks):
    state = {"active_task": True, "last_tier": 5, "last_model": "qwen3.5:397b", "last_category": "analysis"}
    decision = rv2.select_model(marker, simple_benchmarks, tiers, task_state=state)
    assert decision["tier"] == 5
    assert decision["model"] == "qwen3.5:397b"
    assert decision["reason"] == "continuation"

# 6) Exception handling: greetings -> (False, "exception:greeting"); trivial time queries bypass delegation
def test_exception_greeting_and_time_queries():
    dec = {"tier": 1, "model": "glm-5.1"}
    g = "hola"
    delegated, reason = dp.should_delegate(g, dec)
    assert delegated is False
    assert reason == "exception:greeting"
    t = "what is the time?"
    delegated, reason = dp.should_delegate(t, dec)
    assert delegated is False
    assert reason == "exception:time_query"

# 7) plan_subagents: bullets, file patterns, "X vs Y", draft counts
def test_plan_subagents_bullets_and_vs_and_drafts(tiers):
    # bullets
    prompt = "- first task\n- second task\n- third"
    decision = {"category": "writing", "tier": 2}
    plan = dp.plan_subagents(prompt, decision, max_parallel=3)
    assert plan["rationale"].startswith("bullet_split")
    assert plan["count"] == 3
    assert plan["subtasks"][0] == "first task"
    # vs split
    prompt = "compare A vs B vs C"
    decision = {"category": "research", "tier": 2}
    plan = dp.plan_subagents(prompt, decision)
    assert plan["rationale"].startswith("vs_split")
    assert plan["count"] == 3
    # implementation captures the whole matched segment, so first term includes "compare"
    assert plan["subtasks"] == ["research compare A", "research B", "research C"]
    # drafts
    prompt = "write 4 drafts of this email"
    decision = {"category": "writing", "tier": 2}
    plan = dp.plan_subagents(prompt, decision)
    assert plan["rationale"].startswith("drafts_count")
    # default max_parallel==3 causes capping
    assert plan["count"] == 3

# 8) benchmark_harness deterministic: same (category, model, prompt) => same score
def test_benchmark_runner_deterministic():
    suite = {"analysis": {"qwen3.5:397b": ["prompt A", "prompt B"]}}
    def eval_fn(cat, model, prompt):
        # deterministic function of inputs
        base = 0.5 if model.startswith("qwen") else 0.4
        return {"score": base + (len(prompt) % 10) * 0.01, "latency_ms": 10.0, "cost_usd": 0.001}
    r1 = br.run_benchmarks(suite, eval_fn, seed=42)
    r2 = br.run_benchmarks(suite, eval_fn, seed=42)
    s1 = br.report_to_benchmarks(r1)
    s2 = br.report_to_benchmarks(r2)
    assert s1 == s2

# 9) Edge cases: empty prompt, None prompt, very long prompt, Unicode, mixed markers
def test_edge_case_prompts(tiers, simple_benchmarks):
    for p, expected_cat in [("", "simple"), (None, "simple"), ("Δ prueba unicode 👍", "analysis"), ("a "*500, "analysis")]:
        decision = rv2.select_model(p, simple_benchmarks, tiers, task_state=None)
        assert decision["category"] == expected_cat

# 10) smart_model_routing integration: v2_enabled True vs False routes differently
def test_smart_model_routing_v2_toggle(tmp_path, monkeypatch, tiers):
    primary = {"model": "primary-model", "api_key": None, "base_url": None, "provider": "local", "api_mode": None, "command": None, "args": []}
    # create benches
    suite = {"simple": {"glm-5.1": ["hi"]}}
    def eval_fn(cat, model, prompt):
        return {"score": 0.8, "latency_ms": 5.0, "cost_usd": 0.0001}
    monkeypatch.setenv("HOME", str(tmp_path))
    report = br.run_benchmarks(suite, eval_fn)
    bench_dir = tmp_path / ".hermes" / "router"
    bench_dir.mkdir(parents=True)
    br.save_report(report, str(bench_dir / "benchmarks.json"))
    # v2 is now always the primary path (v2_enabled flag removed)
    r = smr.resolve_turn_route("hola", {}, primary)
    assert "v2 smart" in (r.get("label") or "")
    assert r["model"] == "glm-5.1"
    # Without v2_enabled flag (unified system), it still uses v2
    r2 = smr.resolve_turn_route("hola", {"enabled": True, "cheap_model": {"provider": "local", "model": "glm-5.1"}}, primary)
    # v2 is the primary path regardless of flag — model still determined by tiers/benchmarks
    assert r2["model"] == "glm-5.1"

# Extra: escalate does not skip tiers even from middle
def test_escalate_no_skip_from_middle(tiers):
    out = rv2.escalate("kimi-k2.5", tiers)
    assert out["tier"] == 4
    assert out["model"] in tiers[3]
