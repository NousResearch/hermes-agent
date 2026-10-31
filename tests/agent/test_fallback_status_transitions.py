"""Fallback route status is published at real route transitions.

The snapshot helper is only half the contract: the route health the desktop
shows is produced because a REAL transition (a fallback switch, a primary
restore) publishes it. These tests drive the real transition functions so a
call site that stops notifying fails here instead of silently leaving the
menu stale.
"""

import threading
import time
from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import _fallback_status_snapshot, _notify_fallback_status


def _agent(**overrides):
    values = {
        "provider": "primary-provider",
        "model": "primary-model",
        "base_url": "https://primary.example/v1",
        "_primary_runtime": {"provider": "primary-provider", "model": "primary-model"},
        "_fallback_chain": [
            {"provider": "fallback-a", "model": "model-a"},
            {"provider": "fallback-b", "model": "model-b"},
        ],
        "_fallback_activated": False,
        "_rate_limited_until": 0,
        "_fallback_status_reason": None,
    }
    values.update(overrides)
    agent = type("Agent", (), {})()
    for key, value in values.items():
        setattr(agent, key, value)
    return agent


def _wait_for(events, count=1):
    deadline = time.monotonic() + 2
    while len(events) < count and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(events) >= count


def test_primary_to_fallback_publishes_sanitized_route():
    events = []
    agent = _agent(
        provider="fallback-a", model="model-a", _fallback_activated=True,
        _fallback_status_reason="rate limit", fallback_status_callback=events.append,
    )

    _notify_fallback_status(agent)
    _wait_for(events)

    assert events[-1]["active"] == {"provider": "fallback-a", "model": "model-a"}
    assert events[-1]["chain"] == [
        {"provider": "primary-provider", "model": "primary-model"},
        {"provider": "fallback-a", "model": "model-a"},
        {"provider": "fallback-b", "model": "model-b"},
    ]


def test_fallback_switch_and_recovery_publish_each_transition():
    events = []
    agent = _agent(
        provider="fallback-a", model="model-a", _fallback_activated=True,
        fallback_status_callback=events.append,
    )
    _notify_fallback_status(agent)
    _wait_for(events)

    agent.provider, agent.model = "fallback-b", "model-b"
    _notify_fallback_status(agent)
    _wait_for(events, 2)

    agent.provider, agent.model = "primary-provider", "primary-model"
    agent._fallback_activated = False
    agent._rate_limited_until = 0
    _notify_fallback_status(agent)
    _wait_for(events, 3)

    assert [event["active"] for event in events[-3:]] == [
        {"provider": "fallback-a", "model": "model-a"},
        {"provider": "fallback-b", "model": "model-b"},
        None,
    ]


def test_cooldown_is_projected_and_absent_callback_is_safe():
    ready = threading.Event()
    events = []
    def record(event):
        events.append(event)
        ready.set()
    agent = _agent(
        _rate_limited_until=time.monotonic() + 60,
        fallback_status_callback=record,
    )
    _notify_fallback_status(agent)
    assert ready.wait(2)
    assert events[0]["cooldown_until"] is not None
    assert events[0]["active"] is None

    _notify_fallback_status(_agent())


def test_status_snapshot_redacts_private_route_fields():
    agent = _agent(
        provider="fallback-a", model="model-a", _fallback_activated=True,
        _fallback_chain=[{"provider": "fallback-a", "model": "model-a", "credential": "redacted-token"}],
        _fallback_status_reason="https://private.invalid/token",
    )

    snapshot = _fallback_status_snapshot(agent)

    assert "redacted-token" not in repr(snapshot)
    assert "private.invalid" not in repr(snapshot)
    assert "reason" not in snapshot


_PRIMARY_MODEL = "anthropic/claude-opus-4.6"
_FALLBACK = {"provider": "openai", "model": "gpt-4o"}


def _real_agent():
    """An agent with a real fallback chain, built like the other fallback tests."""
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", provider="openrouter",
            model=_PRIMARY_MODEL, quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=[_FALLBACK],
        )
    agent.client = MagicMock()
    agent._rate_limited_until = 0
    return agent


def _fallback_client():
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1"
    client.api_key = "fallback-key"
    return client


def _switch_to_fallback(agent):
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), None)):
        assert agent._try_activate_fallback() is True


def test_a_fallback_switch_publishes_the_new_route_to_the_callback():
    """The switch is the moment the menu must learn the new route."""
    agent = _real_agent()
    events = []
    agent.fallback_status_callback = events.append

    _switch_to_fallback(agent)
    _wait_for(events)

    assert events[-1]["active"] == {"provider": "openai", "model": "gpt-4o"}
    assert events[-1]["chain"][0] == {"provider": "openrouter", "model": _PRIMARY_MODEL}


def test_a_primary_restore_publishes_the_recovered_route_to_the_callback():
    """A restore is the other transition the menu must learn about."""
    agent = _real_agent()
    events = []
    agent.fallback_status_callback = events.append
    _switch_to_fallback(agent)
    _wait_for(events)

    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._restore_primary_runtime() is True
    _wait_for(events, 2)

    assert events[-1]["active"] is None
    assert events[-1]["chain"][0] == {"provider": "openrouter", "model": _PRIMARY_MODEL}
