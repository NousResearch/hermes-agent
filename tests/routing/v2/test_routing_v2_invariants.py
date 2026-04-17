"""Invariant TDD for routing_v2 (parallel double TDD suite).

Complements test_routing_v2_spec.py with property-style laws:
- Continuation preserves tier across a fuzz of markers.
- Escalation is +1 tier (or capped) for all starting tiers.
- Downscale is -1 tier only when streak>=2 and task inactive.
- Category detection is total (no None returned).

No external deps — stdlib random only.
"""
from __future__ import annotations

import random

import pytest

from agent import routing_v2 as rv2


BENCH = {
    "code":     {"qwen3-coder-next": 0.92, "kimi-k2.5": 0.78, "glm-5.1": 0.55, "gpt-5-mini": 0.60},
    "analysis": {"qwen3.5:397b": 0.90, "kimi-k2.5": 0.80, "glm-5.1": 0.58, "gpt-5-mini": 0.62},
    "research": {"deepseek-v3.2": 0.88, "kimi-k2.5": 0.74, "glm-5.1": 0.55},
    "writing":  {"mistral-large-3:675b": 0.89, "kimi-k2.5": 0.76},
    "creative": {"minimax-m2.7": 0.85, "kimi-k2.5": 0.72},
    "vision":   {"qwen3-vl:235b-instruct": 0.91},
    "simple":   {"glm-5.1": 0.80, "gpt-5-mini": 0.78},
}

TIERS = [
    ["glm-5.1"],
    ["gpt-5-mini"],
    ["kimi-k2.5"],
    ["deepseek-v3.2", "qwen3-coder-next"],
    ["qwen3.5:397b", "mistral-large-3:675b", "minimax-m2.7", "qwen3-vl:235b-instruct"],
]


CONT_MARKERS = ["sigue", "continua", "continúa", "dale", "ok", "resume", "mismo tema"]


@pytest.mark.parametrize("marker", CONT_MARKERS)
@pytest.mark.parametrize("start_tier", [2, 3, 4, 5])
def test_continuation_never_downgrades(marker: str, start_tier: int):
    state = {
        "active_task": True,
        "last_tier": start_tier,
        "last_model": TIERS[start_tier - 1][0],
        "last_category": "code",
        "turns_in_task": 3,
    }
    out = rv2.select_model(marker, BENCH, TIERS, state)
    assert out["tier"] >= start_tier, f"downgrade on marker={marker!r}"


@pytest.mark.parametrize("seed", list(range(8)))
def test_select_model_always_returns_valid_fields(seed: int):
    rng = random.Random(seed)
    prompts = [
        "hola", "refactor X", "investiga Y", "describe image /t/a.png",
        "", "pytest falla en foo", "analiza los resultados", "dale",
    ]
    prompt = rng.choice(prompts)
    out = rv2.select_model(prompt, BENCH, TIERS, None)
    assert "category" in out and isinstance(out["category"], str)
    assert "model" in out and out["model"]
    assert "tier" in out and 1 <= out["tier"] <= len(TIERS)
    assert 0.0 <= out["benchmark_score"] <= 1.0


@pytest.mark.parametrize("start_model", [g[0] for g in TIERS])
def test_escalation_is_exactly_one_step_or_capped(start_model: str):
    out = rv2.escalate(start_model, TIERS, reason="test")
    # find original tier
    orig = None
    for idx, grp in enumerate(TIERS, start=1):
        if start_model in grp:
            orig = idx
            break
    assert orig is not None
    if orig == len(TIERS):
        assert out["capped"] is True
        assert out["tier"] == orig
    else:
        assert out["capped"] is False
        assert out["tier"] == orig + 1


@pytest.mark.parametrize("last_tier", [2, 3, 4, 5])
def test_downscale_requires_streak_and_inactive(last_tier: int):
    # active task: never downscale
    out = rv2.maybe_downscale({"last_tier": last_tier, "easy_streak": 5, "active_task": True}, TIERS)
    assert out["tier"] == last_tier
    # streak < 2: no downscale
    out = rv2.maybe_downscale({"last_tier": last_tier, "easy_streak": 1, "active_task": False}, TIERS)
    assert out["tier"] == last_tier
    # streak >= 2 and inactive: exactly one step
    out = rv2.maybe_downscale({"last_tier": last_tier, "easy_streak": 2, "active_task": False}, TIERS)
    assert out["tier"] == max(1, last_tier - 1)
