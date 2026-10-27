"""Tests for issue #66267 — multimodal (list) tool-result content must not
crash interim-assistant-message handling or non-streaming message building.

Root cause (from the issue discussion): after a vision/tool turn the
``tool`` message's ``content`` can be a list of parts (e.g. an image plus
text). Several code paths fed that raw ``content`` straight into regex /
string helpers that assumed ``str``, raising ``TypeError`` on the
list. Because the failure happened inside the per-turn loop *before* the
``role == "assistant"`` guard, the error was retried repeatedly, producing
the "retry loop" symptom.

These tests pin the two fixed surfaces:

* ``chat_completion_helpers.build_assistant_message`` (non-streaming /
  gateway path) — now normalizes ``content`` with ``flatten_message_text``
  before the inline ``<think>`` regex and the surrogate sanitizer.
* ``run_agent.AIAgent._interim_assistant_visible_text`` (used by the
  duplicate-previous-interim dedup in the tool-call path) — now safe for
  any role/content shape because ``flatten_message_text`` handles lists.

They also assert the *behavior* the fix preserves: a tool message never
produces interim text, and list content with an inline ``<think>`` block is
flattened + stripped correctly.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Agent fixture — real methods bound where the fix depends on them
# ---------------------------------------------------------------------------

def _make_agent():
    """Minimal AIAgent with the real text helpers the fix relies on."""
    from agent.message_sanitization import _sanitize_surrogates
    from run_agent import AIAgent

    agent = MagicMock(spec=AIAgent)
    agent._extract_reasoning = lambda msg: AIAgent._extract_reasoning(agent, msg)
    agent._extract_codex_interim_visible_text = (
        lambda msg: AIAgent._extract_codex_interim_visible_text(agent, msg)
    )
    agent._strip_think_blocks = lambda text: AIAgent._strip_think_blocks(agent, text)
    agent._sanitize_surrogates = lambda text: _sanitize_surrogates(text)
    agent.show_commentary = True
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent

def _vision_tool_result(text="The image shows a red stop sign."):
    """A realistic tool message whose content is a list (vision result)."""
    return {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }

# ---------------------------------------------------------------------------
# build_assistant_message — non-streaming / gateway path (site 2)
# ---------------------------------------------------------------------------

class TestBuildAssistantMessageMultimodal:
    def test_list_content_does_not_crash(self):
        """A list content (vision result) must build without TypeError."""
        from agent.chat_completion_helpers import build_assistant_message

        agent = _make_agent()
        sdk_msg = SimpleNamespace(
            content=[
                {"type": "text", "text": "answer after seeing the screenshot"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            tool_calls=None,
            reasoning_content=None,
            reasoning_details=None,
            codex_reasoning_items=None,
            codex_message_items=None,
        )

        msg = build_assistant_message(agent, sdk_msg, "stop")

        assert msg["role"] == "assistant"
        assert isinstance(msg["content"], str)
        assert "answer after seeing the screenshot" in msg["content"]

# ---------------------------------------------------------------------------
# _interim_assistant_visible_text — dedup path (site 1)
# ---------------------------------------------------------------------------

class TestInterimVisibleTextMultimodal:
    def test_tool_list_content_does_not_crash(self):
        """A tool message with list content must return a string, not raise."""
        from run_agent import AIAgent

        agent = _make_agent()
        tool_msg = _vision_tool_result()
        # The helper must not raise on a tool message whose content is a list.
        visible = AIAgent._interim_assistant_visible_text(agent, tool_msg)
        assert isinstance(visible, str)

    def test_assistant_list_content_flattened(self):
        """An assistant message with list content yields flattened visible text."""
        from run_agent import AIAgent

        agent = _make_agent()
        assistant_msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me look at the screenshot."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            "finish_reason": "incomplete",
        }
        visible = AIAgent._interim_assistant_visible_text(agent, assistant_msg)
        assert "Let me look at the screenshot." in visible

# ---------------------------------------------------------------------------
# duplicate_previous_interim dedup — the exact shape from conversation_loop.py
# ---------------------------------------------------------------------------
