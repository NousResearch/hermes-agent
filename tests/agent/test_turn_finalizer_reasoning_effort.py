"""Reasoning-effort metadata emitted by the turn finalizer."""

from types import SimpleNamespace

import pytest

from agent.turn_finalizer import _effective_reasoning_effort


@pytest.mark.parametrize(
    "reasoning_config,kwargs,expected",
    [
        (None, {}, None),
        (None, {"api_mode": "codex_responses"}, None),
        ({"enabled": False}, {}, "none"),
        ({"enabled": True, "effort": "low"}, {}, "low"),
        (
            {"enabled": True, "effort": "ultra"},
            {
                "api_mode": "codex_responses",
                "transport": SimpleNamespace(_last_reasoning_effort="MAX"),
            },
            "max",
        ),
        (
            {"enabled": True, "effort": "high"},
            {
                "api_mode": "codex_responses",
                "transport": SimpleNamespace(_last_reasoning_effort=None),
            },
            None,
        ),
        (
            {"enabled": True, "effort": "minimal"},
            {},
            "minimal",
        ),
    ],
)
def test_effective_reasoning_effort(reasoning_config, kwargs, expected):
    assert _effective_reasoning_effort(reasoning_config, **kwargs) == expected