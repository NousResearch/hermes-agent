"""Tests for sanitize_anthropic_kwargs (#31673).

Guards the Anthropic Messages dispatch boundary against Responses-API-only
kwargs (``instructions``, ``input``, ``store``, ``parallel_tool_calls``)
leaking in under an api_mode-flip race. The Anthropic SDK raises a
non-retryable ``TypeError`` on any of them, killing the whole turn.
"""

import logging

import pytest

from agent.anthropic_adapter import (
    _RESPONSES_ONLY_KWARGS,
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


def test_bare_leaked_payload_reproduces_the_typeerror():
    """Without the guard, a Responses-shaped payload raises the issue's error."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _fake_anthropic_call(model="claude-sonnet-4-6", instructions="sys")


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


def test_clean_anthropic_payload_is_untouched():
    payload = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1024,
        "system": "sys",
        "tools": [{"name": "x"}],
    }
    snapshot = dict(payload)
    sanitize_anthropic_kwargs(payload)
    assert payload == snapshot
    assert _fake_anthropic_call(**payload) == "OK"


def test_warns_when_keys_are_stripped(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.anthropic_adapter"):
        sanitize_anthropic_kwargs(
            {"model": "m", "instructions": "sys"}, log_prefix="[pfx] "
        )
    assert any(
        "31673" in r.message and "[pfx] " in r.message
        for r in caplog.records
    ), caplog.records


def test_no_warning_on_clean_payload(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.anthropic_adapter"):
        sanitize_anthropic_kwargs({"model": "m", "messages": []})
    assert not caplog.records


def test_non_dict_input_is_noop():
    assert sanitize_anthropic_kwargs(None) is None
    assert sanitize_anthropic_kwargs("not a dict") == "not a dict"


def test_responses_only_kwargs_membership():
    # Contract: instructions (the reported symptom) plus the sibling
    # Responses-shape keys are all covered.
    assert {"instructions", "input", "store", "parallel_tool_calls"} <= _RESPONSES_ONLY_KWARGS
