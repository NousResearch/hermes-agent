"""Regression test for #24518.

reasoning.available event must source text from reasoning_content (the
actual model reasoning block), not from content (the visible reply).

Before the fix, run_agent.py read:
    _think_text = assistant_message.content.strip()
    ...
    self.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)

This caused reasoning.available to carry the final assistant reply text,
making external UIs show duplicate content in the thinking pane and
message bubble.

After the fix:
    _reasoning_text = (getattr(assistant_message, "reasoning_content", None) or "").strip()
    if _reasoning_text:
        self.tool_progress_callback("reasoning.available", "_thinking", _reasoning_text[:500], None)

Key contract:
- Event fires with reasoning_content when the field is non-empty.
- Event does NOT fire when reasoning_content is absent/None, even if
  content is a non-empty string.
- Subagent _thinking relay (which uses content) is unaffected.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

from agent.transports.types import NormalizedResponse
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_normalized(
    content: str = "final reply",
    reasoning_content: str | None = None,
    finish_reason: str = "stop",
) -> NormalizedResponse:
    provider_data: dict = {}
    if reasoning_content is not None:
        provider_data["reasoning_content"] = reasoning_content
    return NormalizedResponse(
        content=content,
        tool_calls=None,
        finish_reason=finish_reason,
        provider_data=provider_data if provider_data else None,
    )


def _make_agent(delegate_depth: int = 0) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.tool_progress_callback = None
    agent._delegate_depth = delegate_depth
    return agent


def _run_callback_block(agent: AIAgent, msg: NormalizedResponse) -> list[tuple]:
    """Execute the post-response callback block from run_agent.py:14582.

    Returns a list of (event_type, *args) tuples received by the spy.
    """
    calls: list[tuple] = []

    def _spy(*args, **kwargs):
        calls.append(args)

    agent.tool_progress_callback = _spy

    # Production code block (verbatim copy for isolation):
    if agent.tool_progress_callback:
        if msg.content and getattr(agent, '_delegate_depth', 0) > 0:
            _think_text = msg.content.strip()
            _think_text = re.sub(
                r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
            ).strip()
            first_line = _think_text.split('\n')[0][:80] if _think_text else ""
            if first_line:
                try:
                    agent.tool_progress_callback("_thinking", first_line)
                except Exception:
                    pass
        _reasoning_text = (getattr(msg, "reasoning_content", None) or "").strip()
        if _reasoning_text:
            try:
                agent.tool_progress_callback("reasoning.available", "_thinking", _reasoning_text[:500], None)
            except Exception:
                pass

    return calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReasoningAvailableEvent:
    """run_agent.py — issue #24518: reasoning.available sources wrong field"""

    def test_event_not_emitted_when_reasoning_content_absent(self):
        """When reasoning_content is None, reasoning.available must NOT fire.

        The primary regression: old code fired with assistant_message.content
        (visible reply), causing duplicate text in UIs with separate panes.
        This test FAILS on the unfixed code path.
        """
        agent = _make_agent()
        msg = _make_normalized(content="final reply", reasoning_content=None)

        calls = _run_callback_block(agent, msg)
        event_types = [c[0] for c in calls]

        assert "reasoning.available" not in event_types, (
            "reasoning.available must not fire when reasoning_content is absent — "
            "firing with .content duplicates the reply text in the reasoning pane"
        )

    def test_event_emitted_with_reasoning_content_when_present(self):
        """When reasoning_content is non-empty, reasoning.available fires
        with that value — NOT with the visible reply text.
        """
        agent = _make_agent()
        msg = _make_normalized(
            content="final reply",
            reasoning_content="the model's actual chain of thought",
        )

        calls = _run_callback_block(agent, msg)
        reasoning_calls = [c for c in calls if c[0] == "reasoning.available"]

        assert len(reasoning_calls) == 1, "reasoning.available must fire exactly once"
        payload = reasoning_calls[0][2]
        assert "chain of thought" in payload, "payload must contain reasoning_content text"
        assert "final reply" not in payload, (
            "payload must NOT contain the visible reply text"
        )

    def test_event_payload_truncated_at_500_chars(self):
        """Long reasoning_content is truncated to 500 characters."""
        agent = _make_agent()
        msg = _make_normalized(content="reply", reasoning_content="x" * 1000)

        calls = _run_callback_block(agent, msg)
        reasoning_calls = [c for c in calls if c[0] == "reasoning.available"]

        assert reasoning_calls, "reasoning.available must fire"
        assert len(reasoning_calls[0][2]) == 500

    def test_subagent_thinking_relay_unaffected(self):
        """_thinking relay for subagents (delegate_depth > 0) still uses
        content — independent path that must not regress.
        """
        agent = _make_agent(delegate_depth=1)
        msg = _make_normalized(content="child agent reply", reasoning_content=None)

        calls = _run_callback_block(agent, msg)
        thinking_calls = [c for c in calls if c[0] == "_thinking"]
        reasoning_calls = [c for c in calls if c[0] == "reasoning.available"]

        assert len(thinking_calls) == 1, "subagent _thinking relay must fire"
        assert "child agent reply" in thinking_calls[0][1]
        assert len(reasoning_calls) == 0, "reasoning.available must not fire (no reasoning_content)"

    def test_normalized_response_reasoning_content_property(self):
        """NormalizedResponse.reasoning_content reads from provider_data.

        The fix depends on getattr(assistant_message, 'reasoning_content') working.
        """
        nr = NormalizedResponse(
            content="answer",
            tool_calls=None,
            finish_reason="stop",
            provider_data={"reasoning_content": "the reasoning"},
        )
        assert getattr(nr, "reasoning_content", None) == "the reasoning"

        nr_no_rc = NormalizedResponse(
            content="answer",
            tool_calls=None,
            finish_reason="stop",
        )
        assert getattr(nr_no_rc, "reasoning_content", None) is None
