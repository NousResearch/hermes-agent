from types import SimpleNamespace

import pytest

from agent.fast_mode import (
    DEFAULT_FAST_AUTO_ON_SECONDS,
    begin_fast_mode_turn,
    effective_request_overrides,
    normalize_fast_auto_on_seconds,
)


def _agent(model="gpt-5.4", **overrides):
    values = {
        "model": model,
        "service_tier": "auto",
        "fast_auto_on_seconds": 60,
        "request_overrides": {"unrelated": "preserved"},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("value", [None, "", 0, -1, float("inf"), True])
def test_invalid_auto_cutoff_uses_default(value):
    assert normalize_fast_auto_on_seconds(value) == DEFAULT_FAST_AUTO_ON_SECONDS


def test_auto_fast_is_active_through_cutoff_then_removed():
    agent = _agent(request_overrides={"service_tier": "priority", "unrelated": 1})
    begin_fast_mode_turn(agent, now=100.0)

    assert effective_request_overrides(agent, now=160.0) == {
        "service_tier": "priority",
        "unrelated": 1,
    }
    assert effective_request_overrides(agent, now=160.001) == {"unrelated": 1}


def test_auto_fast_resets_for_each_user_turn():
    agent = _agent()
    begin_fast_mode_turn(agent, now=100.0)
    assert "service_tier" not in effective_request_overrides(agent, now=161.0)

    begin_fast_mode_turn(agent, now=200.0)
    assert effective_request_overrides(agent, now=200.0)["service_tier"] == "priority"


def test_auto_fast_uses_anthropic_speed_override():
    agent = _agent(model="anthropic/claude-opus-4.6")
    begin_fast_mode_turn(agent, now=10.0)

    assert effective_request_overrides(agent, now=20.0) == {
        "speed": "fast",
        "unrelated": "preserved",
    }
    assert effective_request_overrides(agent, now=71.0) == {"unrelated": "preserved"}


def test_explicit_fast_mode_is_unchanged():
    agent = _agent(
        service_tier="priority",
        request_overrides={"service_tier": "priority", "unrelated": 1},
    )
    begin_fast_mode_turn(agent, now=100.0)

    assert agent._fast_mode_turn_started_at is None
    assert effective_request_overrides(agent, now=1000.0) == {
        "service_tier": "priority",
        "unrelated": 1,
    }


def test_unsupported_model_never_adds_a_fast_override():
    agent = _agent(model="openrouter/some-unsupported-model")
    begin_fast_mode_turn(agent, now=10.0)

    assert effective_request_overrides(agent, now=10.0) == {"unrelated": "preserved"}
