"""Reasoning-effort metadata emitted by the turn finalizer."""

import pytest

from agent.turn_finalizer import _effective_reasoning_effort


@pytest.mark.parametrize(
    "reasoning_config,kwargs,expected",
    [
        (None, {}, None),
        (None, {"api_mode": "codex_responses"}, "medium"),
        ({"enabled": False}, {}, "none"),
        ({"enabled": True, "effort": "low"}, {}, "low"),
        (
            {"enabled": True, "effort": "minimal"},
            {"api_mode": "codex_responses"},
            "low",
        ),
        (
            {"enabled": True, "effort": "ULTRA"},
            {"api_mode": "codex_responses", "model": "gpt-5.6-sol"},
            "max",
        ),
        (
            {"enabled": True, "effort": "xhigh"},
            {
                "api_mode": "codex_responses",
                "provider": "xai-oauth",
                "model": "grok-4.5",
            },
            "high",
        ),
        (
            {"enabled": True, "effort": "high"},
            {
                "api_mode": "codex_responses",
                "provider": "xai-oauth",
                "model": "grok-4",
            },
            None,
        ),
    ],
)
def test_effective_reasoning_effort(reasoning_config, kwargs, expected):
    assert _effective_reasoning_effort(reasoning_config, **kwargs) == expected