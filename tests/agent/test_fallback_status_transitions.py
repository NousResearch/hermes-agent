"""Fallback route status is published at real route transitions."""

import threading
import time

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
