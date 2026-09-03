"""Regression test for #6133 — vars() on response objects without __dict__.

The invalid-response debug-logging path in ``agent.conversation_loop``
called ``vars(response)`` bare. SDK response shapes that don't expose
``__dict__`` (slotted classes, some Pydantic models) made that raise
``TypeError: vars() argument must have dict attribute``, crashing the
retry loop instead of degrading the debug log. The fix falls back
through ``__dict__`` -> ``model_dump()`` -> a plain type-name repr.

Scenario (mirrors the reporter's: intermittent, shape-dependent):
  1. First API call returns a slotted response object with no
     choices/error/message/model attributes — invalid, and the provider
     name stays "Unknown", so the loop enters the resp_attrs logging
     branch that used to call ``vars(response)``.
  2. Before this fix, the turn died with TypeError at that point.
     With the fix, the turn retries and the second call returns a valid
     chat-completions response, so the conversation completes.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import agent.conversation_loop as conversation_loop
from run_agent import AIAgent


def _dummy_credential() -> str:
    # Placeholder shaped like a key but built at runtime — no real secret.
    return "t" + "0" * 23


def _make_tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _make_agent():
    """Build a minimal chat-completions AIAgent (mock client, no fallback)."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key=_dummy_credential(),
            base_url="https://api.test.internal/v1",
            provider="test",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


class _SlotsResponse:
    """SDK response shape with no ``__dict__`` and no recognisable fields."""

    __slots__ = ()

    def __repr__(self):  # stable payload for the fallback logging path
        return "<_SlotsResponse invalid>"


def _valid_response(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None, reflections=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return SimpleNamespace(
        choices=[choice], model="test/model", usage=usage, id="resp_ok"
    )


def test_invalid_slots_response_does_not_crash_retry(monkeypatch):
    agent = _make_agent()
    calls = {"n": 0}

    def _fake_api(api_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _SlotsResponse()
        return _valid_response("Recovered")

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api)
    # Collapse the retry backoff so the invalid->valid sequence is instant.
    monkeypatch.setattr(
        conversation_loop, "jittered_backoff", lambda *a, **kw: 0.0
    )

    result = agent.run_conversation("ping")

    assert calls["n"] >= 2
    assert result["completed"] is True
    assert result["final_response"] == "Recovered"
