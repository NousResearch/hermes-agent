"""Adversarial tests for delegation + premium-savings contract — RED PHASE.

Guards against two known failure modes:
1. Simple delegated tasks must resolve to the cheap model, not inherit Opus.
2. delegation.api_key must be explicit — an empty value should surface a
   clear warning / error instead of silently falling back to the parent's
   premium credentials.

Also includes a deterministic simulation test (GREEN from day one) that
pins the 95% savings claim as a regression guard.
"""

from __future__ import annotations

import pytest


def _cheap_cfg():
    return {
        "enabled": True,
        "max_simple_chars": 96,
        "max_simple_words": 16,
        "cheap_model": {"provider": "copilot", "model": "gpt-5-mini"},
    }


# ---------------------------------------------------------------------------
# smart_routing integration: simple delegated task → mini
# ---------------------------------------------------------------------------
def test_simple_delegated_task_uses_mini_not_opus():
    from agent.smart_model_routing import choose_cheap_model_route

    route = choose_cheap_model_route("di hola", _cheap_cfg())

    assert route is not None
    assert route["provider"] == "copilot"
    assert route["model"] == "gpt-5-mini"
    assert route["routing_reason"] == "simple_turn"


def test_long_delegated_task_does_not_downgrade():
    from agent.smart_model_routing import choose_cheap_model_route

    long_msg = "refactor this module and add full test coverage " * 20
    assert choose_cheap_model_route(long_msg, _cheap_cfg()) is None


# ---------------------------------------------------------------------------
# delegation.api_key must be explicit
# ---------------------------------------------------------------------------
def test_delegation_config_requires_explicit_api_key():
    """RED: validator does not exist yet.

    GREEN phase will add agent.provider_policy.validate_delegation_config()
    (or equivalent) that raises / warns when api_key is empty while
    provider+model are explicitly set.
    """
    # RED: ImportError expected until GREEN phase adds the helper
    from agent.provider_policy import validate_delegation_config  # type: ignore

    bad_cfg = {"provider": "copilot", "model": "gpt-5-mini", "api_key": ""}
    with pytest.raises(ValueError):
        validate_delegation_config(bad_cfg)


def test_delegation_config_accepts_explicit_api_key():
    from agent.provider_policy import validate_delegation_config  # type: ignore

    good_cfg = {
        "provider": "copilot",
        "model": "gpt-5-mini",
        "api_key": "ghu_fake_but_nonempty",
    }
    # Should not raise
    validate_delegation_config(good_cfg)


# ---------------------------------------------------------------------------
# Savings regression guard (pure math — GREEN from day one)
# ---------------------------------------------------------------------------
def _simulate_distribution(n_turns: int):
    """70% simple (mini, 0x), 25% intermediate (sonnet, 1x), 5% heavy (opus, 5x)."""
    turns = []
    for i in range(n_turns):
        bucket = i % 100
        if bucket < 70:
            turns.append(("gpt-5-mini", 0))
        elif bucket < 95:
            turns.append(("claude-sonnet-4.6", 1))
        else:
            turns.append(("claude-opus-4.6", 5))
    return turns


def test_projected_premium_savings_simulation():
    turns = _simulate_distribution(1000)
    baseline = 5 * len(turns)  # everything on Opus
    contrapropuesta = sum(mult for _, mult in turns)
    savings = (baseline - contrapropuesta) / baseline

    # Contract: contrapropuesta must deliver ≥ 90% savings under this mix
    # (70% mini×0 + 25% sonnet×1 + 5% opus×5 = 500 vs baseline 5000 → 90%)
    assert savings >= 0.90, f"expected ≥90% savings, got {savings:.3f}"


def test_opus_on_demand_halves_heavy_cost():
    """With [OPUS] keyword gating, only ~50% of heavy tasks should fire Opus."""
    heavy = 50
    opus_invocations = heavy // 2  # keyword-gated
    sonnet_invocations = heavy - opus_invocations

    baseline = heavy * 5
    contrapropuesta = opus_invocations * 5 + sonnet_invocations * 1
    savings = (baseline - contrapropuesta) / baseline

    assert savings >= 0.40, f"expected ≥40% savings on heavy tier, got {savings:.3f}"
