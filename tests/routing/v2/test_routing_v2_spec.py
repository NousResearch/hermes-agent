"""
TDD adversarial suite for Routing v2 (RED phase).

These tests SPECIFY the target behavior for the upgraded router:

1) Category-aware model selection (code/analysis/research/vision/...).
2) Multi-factor scoring (complexity + category + affinity + benchmark).
3) Task continuity / stickiness: "sigue"/"continua" NEVER downgrades model.
4) Graceful escalation: exactly one tier step up on failure (no T1->T5 jumps).
5) Graceful downscaling: only after N consecutive easy turns.
6) Benchmark-backed choice: best model per category comes from benchmarks.json.
7) No-validation loop: if user stays silent, router holds current tier
   (does NOT silently drop to cheap_model).

The current system (agent.smart_model_routing) only implements a binary
simple/strong switch, so these tests are expected to FAIL until v2 lands.
Runner: pytest tests/routing/v2 -x
"""

from __future__ import annotations

import pytest

# v2 module does not exist yet — import is inside tests to allow collection
# even when the module is missing, so pytest shows a clean red.


# ---------------------------------------------------------------------------
# Fixtures — synthetic benchmark + affinity stores
# ---------------------------------------------------------------------------

@pytest.fixture
def benchmarks():
    return {
        "code":     {"qwen3-coder-next": 0.92, "kimi-k2.5": 0.78, "glm-5.1": 0.55, "gpt-5-mini": 0.60},
        "analysis": {"qwen3.5:397b": 0.90, "kimi-k2.5": 0.80, "glm-5.1": 0.58, "gpt-5-mini": 0.62},
        "research": {"deepseek-v3.2": 0.88, "kimi-k2.5": 0.74, "glm-5.1": 0.55},
        "writing":  {"mistral-large-3:675b": 0.89, "kimi-k2.5": 0.76},
        "creative": {"minimax-m2.7": 0.85, "kimi-k2.5": 0.72},
        "vision":   {"qwen3-vl:235b-instruct": 0.91},
        "simple":   {"glm-5.1": 0.80, "gpt-5-mini": 0.78},
    }

@pytest.fixture
def tiers():
    # Ordered low -> high capability. Router must walk exactly ONE step.
    return [
        ["glm-5.1"],                               # T1
        ["gpt-5-mini"],                            # T2
        ["kimi-k2.5"],                             # T3
        ["deepseek-v3.2", "qwen3-coder-next"],     # T4
        ["qwen3.5:397b", "mistral-large-3:675b",
         "minimax-m2.7", "qwen3-vl:235b-instruct"],# T5
    ]


# ---------------------------------------------------------------------------
# 1) Category-aware selection
# ---------------------------------------------------------------------------

def test_category_code_picks_coder_model(benchmarks, tiers):
    from agent.routing_v2 import select_model
    decision = select_model(
        prompt="refactor this function and fix the failing pytest in tools/foo.py",
        benchmarks=benchmarks, tiers=tiers, task_state=None,
    )
    assert decision["category"] == "code"
    assert decision["model"] == "qwen3-coder-next"

def test_category_research_picks_deepseek(benchmarks, tiers):
    from agent.routing_v2 import select_model
    decision = select_model(
        prompt="investiga las últimas noticias sobre fusión nuclear y resume fuentes",
        benchmarks=benchmarks, tiers=tiers, task_state=None,
    )
    assert decision["category"] == "research"
    assert decision["model"] == "deepseek-v3.2"

def test_category_simple_picks_cheap(benchmarks, tiers):
    from agent.routing_v2 import select_model
    decision = select_model(
        prompt="hola",
        benchmarks=benchmarks, tiers=tiers, task_state=None,
    )
    assert decision["category"] == "simple"
    assert decision["model"] in {"glm-5.1", "gpt-5-mini"}
    assert decision["tier"] <= 2


# ---------------------------------------------------------------------------
# 2) Task continuity — "sigue"/"continua" must NOT downgrade
# ---------------------------------------------------------------------------

def test_continuation_preserves_tier(benchmarks, tiers):
    from agent.routing_v2 import select_model
    state = {"active_task": True, "last_tier": 4, "last_model": "qwen3-coder-next",
             "last_category": "code", "turns_in_task": 2}
    decision = select_model(
        prompt="sigue",  # short, would normally trigger cheap route
        benchmarks=benchmarks, tiers=tiers, task_state=state,
    )
    assert decision["tier"] >= 4, "continuation must keep heavy tier"
    assert decision["model"] == "qwen3-coder-next"
    assert decision["reason"].startswith("continuation")

def test_continuation_even_if_prompt_is_tiny(benchmarks, tiers):
    from agent.routing_v2 import select_model
    state = {"active_task": True, "last_tier": 5, "last_model": "qwen3.5:397b",
             "last_category": "analysis", "turns_in_task": 4}
    for word in ["ok", "dale", "continua", "sigue con eso"]:
        decision = select_model(prompt=word, benchmarks=benchmarks,
                                tiers=tiers, task_state=state)
        assert decision["tier"] >= 5, f"downgrade on {word!r}"


# ---------------------------------------------------------------------------
# 3) Graceful escalation — exactly one tier step per failure
# ---------------------------------------------------------------------------

def test_escalation_steps_one_tier(benchmarks, tiers):
    from agent.routing_v2 import escalate
    nxt = escalate(current_model="gpt-5-mini", tiers=tiers, reason="low_quality")
    assert nxt["tier"] == 3
    assert nxt["model"] == "kimi-k2.5"

def test_escalation_never_jumps_multiple_tiers(benchmarks, tiers):
    from agent.routing_v2 import escalate
    nxt = escalate(current_model="glm-5.1", tiers=tiers, reason="user_correction")
    assert nxt["tier"] == 2, "must not jump T1->T5"
    assert "glm" not in nxt["model"]

def test_escalation_caps_at_top_tier(benchmarks, tiers):
    from agent.routing_v2 import escalate
    nxt = escalate(current_model="qwen3.5:397b", tiers=tiers, reason="low_quality")
    assert nxt["tier"] == 5  # already at top
    assert nxt["capped"] is True


# ---------------------------------------------------------------------------
# 4) Graceful downscaling — only after 2 easy turns, and never skipping tiers
# ---------------------------------------------------------------------------

def test_no_downscale_on_single_easy_turn(benchmarks, tiers):
    from agent.routing_v2 import maybe_downscale
    state = {"last_tier": 4, "easy_streak": 1, "active_task": True}
    out = maybe_downscale(state=state, tiers=tiers)
    assert out["tier"] == 4

def test_downscale_one_step_after_streak(benchmarks, tiers):
    from agent.routing_v2 import maybe_downscale
    state = {"last_tier": 4, "easy_streak": 2, "active_task": False}
    out = maybe_downscale(state=state, tiers=tiers)
    assert out["tier"] == 3, "downscale must be exactly one tier"


# ---------------------------------------------------------------------------
# 5) Benchmark-backed selection — best bench wins within category/tier
# ---------------------------------------------------------------------------

def test_benchmark_drives_selection(benchmarks, tiers):
    from agent.routing_v2 import select_model
    bumped = dict(benchmarks)
    bumped["code"] = {"qwen3-coder-next": 0.55, "kimi-k2.5": 0.95, "glm-5.1": 0.40}
    decision = select_model(
        prompt="debug this python stacktrace please",
        benchmarks=bumped, tiers=tiers, task_state=None,
    )
    assert decision["model"] == "kimi-k2.5"
    assert decision["benchmark_score"] >= 0.9


# ---------------------------------------------------------------------------
# 6) No-validation hold — silence does not trigger downgrade
# ---------------------------------------------------------------------------

def test_silence_holds_current_tier(benchmarks, tiers):
    from agent.routing_v2 import select_model
    state = {"active_task": True, "last_tier": 4, "last_model": "qwen3-coder-next",
             "last_category": "code", "turns_in_task": 3, "user_validated": None}
    decision = select_model(
        prompt="",  # empty / silence
        benchmarks=benchmarks, tiers=tiers, task_state=state,
    )
    assert decision["tier"] >= 4
    assert decision["reason"] in {"silence_hold", "continuation"}


# ---------------------------------------------------------------------------
# 7) Vision forces multimodal tier regardless of length
# ---------------------------------------------------------------------------

def test_vision_forces_multimodal_model(benchmarks, tiers):
    from agent.routing_v2 import select_model
    decision = select_model(
        prompt="describe this image: /tmp/foo.png",
        benchmarks=benchmarks, tiers=tiers, task_state=None,
    )
    assert decision["category"] == "vision"
    assert decision["model"] == "qwen3-vl:235b-instruct"
