"""Unit tests for extra_body_route_scoped (#107836).

Verifies that extra_body_route_scoped keys are merged into extra_body at init,
but automatically stripped when a route switch or fallback occurs.
"""

from types import SimpleNamespace
from agent.agent_init import _merge_custom_provider_extra_body
from agent.chat_completion_helpers import _rescope_fallback_extra_body
from agent.agent_runtime_helpers import _apply_switched_provider_request_overrides


def test_route_scoped_extra_body_merged_at_init():
    agent = SimpleNamespace(
        provider="custom",
        model="m1",
        base_url="http://localhost:8080/v1",
        request_overrides={
            "extra_body": {"global_key": "val1"},
            "extra_body_route_scoped": {"route_key": "val2"},
        },
        _custom_providers=[],
    )
    _merge_custom_provider_extra_body(agent, [])
    assert agent.request_overrides["extra_body"] == {
        "global_key": "val1",
        "route_key": "val2",
    }
    assert agent.request_overrides["extra_body_route_scoped"] == {"route_key"}


def test_fallback_rescope_strips_route_scoped_extra_body():
    agent = SimpleNamespace(
        provider="custom_backup",
        model="m2",
        base_url="http://backup:8080/v1",
        request_overrides={
            "extra_body": {"global_key": "val1", "route_key": "val2"},
            "extra_body_route_scoped": ["route_key"],
        },
        _custom_providers=[],
    )
    _rescope_fallback_extra_body(agent, "m1", "custom", "http://localhost:8080/v1")
    assert agent.request_overrides.get("extra_body") == {"global_key": "val1"}
    assert "extra_body_route_scoped" not in agent.request_overrides


def test_provider_switch_strips_route_scoped_extra_body():
    agent = SimpleNamespace(
        provider="new_provider",
        model="m2",
        base_url="http://new:8080/v1",
        request_overrides={
            "extra_body": {"route_key": "val2"},
            "extra_body_route_scoped": {"route_key": "val2"},
        },
        _custom_providers=[],
    )
    _apply_switched_provider_request_overrides(agent, "new_provider")
    assert "extra_body" not in agent.request_overrides or "route_key" not in agent.request_overrides["extra_body"]
    assert "extra_body_route_scoped" not in agent.request_overrides
