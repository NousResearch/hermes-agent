"""Tests for sanitize_anthropic_kwargs (#31673).

Guards the Anthropic Messages dispatch boundary against Responses-API-only
kwargs (``instructions``, ``input``, ``store``, ``parallel_tool_calls``)
leaking in under an api_mode-flip race. The Anthropic SDK raises a
non-retryable ``TypeError`` on any of them, killing the whole turn.
"""

from agent.anthropic_adapter import (
    sanitize_anthropic_kwargs,
)

def _fake_anthropic_call(**kwargs):
    """Mimic the Anthropic SDK's strict kwarg signature."""
    allowed = {
        "model", "messages", "max_tokens", "system", "tools", "tool_choice",
        "extra_body", "extra_headers", "temperature", "top_p", "top_k",
        "thinking", "timeout",
    }
    bad = set(kwargs) - allowed
    if bad:
        raise TypeError(
            "Messages.stream() got an unexpected keyword argument "
            f"{sorted(bad)[0]!r}"
        )
    return "OK"

def test_strips_all_responses_only_keys():
    payload = {
        "model": "claude-sonnet-4-6",
        "instructions": "You are Hermes.",
        "input": [{"role": "user", "content": "hi"}],
        "store": False,
        "parallel_tool_calls": True,
    }
    out = sanitize_anthropic_kwargs(payload)
    assert out is payload  # mutates in place and returns same dict
    assert payload == {"model": "claude-sonnet-4-6"}
    assert _fake_anthropic_call(**payload) == "OK"
