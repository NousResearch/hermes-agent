"""Solstice sends Gemini's native ``thinking_config``: its client reads only that key, so the generic
``extra_body.reasoning`` fallback left every effort (and thinking-off for titles) off the wire."""

from __future__ import annotations

import pytest

import providers
from agent.auxiliary_client import _build_call_kwargs

_BASE = "https://generativelanguage.googleapis.com/v1alpha"


@pytest.mark.parametrize(
    ("reasoning_config", "expected"),
    [
        ({"enabled": False}, {"thinking_config": {"includeThoughts": False, "thinkingLevel": "low"}}),
        ({"enabled": True, "effort": "medium"}, {"thinking_config": {"includeThoughts": True, "thinkingLevel": "medium"}}),
        ({"enabled": True, "effort": "xhigh"}, {"thinking_config": {"includeThoughts": True, "thinkingLevel": "xhigh"}}),
        (None, {}),  # unset effort keeps the model's own default
    ],
)
def test_solstice_profile_translates_reasoning_to_native_thinking_config(reasoning_config, expected):
    profile = providers.get_provider_profile("solstice")

    assert profile is not None
    assert profile.build_extra_body(model="gemini-4-neon", reasoning_config=reasoning_config, base_url=_BASE) == expected


def test_solstice_title_call_reaches_the_wire_with_thinking_bounded():
    """The title pass's thinking-off must arrive as ``thinking_config`` (not the ignored ``reasoning``)."""
    kwargs = _build_call_kwargs(
        "solstice", "gemini-4-neon", [{"role": "user", "content": "hi"}], max_tokens=512,
        reasoning_config={"enabled": False}, base_url=_BASE, task="title_generation",
    )

    assert kwargs["extra_body"]["thinking_config"] == {"includeThoughts": False, "thinkingLevel": "low"}
    assert "reasoning" not in kwargs["extra_body"]
