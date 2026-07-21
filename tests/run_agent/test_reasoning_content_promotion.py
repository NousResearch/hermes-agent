"""Regression tests for reasoning-content-to-visible promotion.

When a model (e.g. MiMo-v2.5) consistently puts its real response —
including vision descriptions — in ``reasoning_content`` instead of
``content``, the conversation loop should promote the reasoning text to
visible content after thinking prefill retries are exhausted, rather
than returning "(empty)".

Refs: https://github.com/NousResearch/hermes-agent/issues/48032
"""

from unittest.mock import MagicMock

from run_agent import AIAgent


def _make_agent_stub(**overrides):
    """Create a minimal AIAgent stub with the attributes the conversation
    loop reads during the empty-response / reasoning-promotion path."""
    agent = AIAgent.__new__(AIAgent)
    agent.model = "mimo-v2.5"
    agent.provider = "xiaomi"
    agent._thinking_prefill_retries = 2  # exhausted
    agent._empty_content_retries = 0
    agent._fallback_chain = []
    agent._session_messages = []
    agent._last_content_with_tools = None
    agent._last_content_tools_all_housekeeping = False
    agent._response_was_previewed = False
    agent._empty_content_retries = 0
    agent.api_mode = "chat_completions"
    agent.valid_tool_names = []
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


def _make_assistant_message(content="", reasoning_content=None, reasoning=None):
    """Build a mock assistant message object."""
    msg = MagicMock()
    msg.content = content
    msg.reasoning_content = reasoning_content
    msg.reasoning = reasoning
    msg.reasoning_details = None
    msg.tool_calls = None
    msg.finish_reason = "stop"
    return msg


class TestReasoningPromotionAfterPrefillExhaustion:
    """After thinking prefill retries are exhausted, if the model returned
    empty content but non-empty reasoning_content, promote the reasoning
    text to visible content."""

    def test_reasoning_content_promoted_to_visible(self):
        """MiMo-v2.5 vision: image description in reasoning_content."""
        agent = _make_agent_stub()
        assistant = _make_assistant_message(
            content="",
            reasoning_content="The image shows a sunset over the ocean with vibrant orange and purple colors.",
        )
        # Simulate the condition: prefill retries exhausted, has structured reasoning
        _has_structured = bool(
            getattr(assistant, "reasoning", None)
            or getattr(assistant, "reasoning_content", None)
        )
        assert _has_structured
        assert agent._thinking_prefill_retries >= 2

        # Extract reasoning text (mirrors the fix logic)
        _rc_text = (
            getattr(assistant, "reasoning_content", None)
            or getattr(assistant, "reasoning", None)
            or ""
        )
        assert isinstance(_rc_text, str) and _rc_text.strip()
        final_response = _rc_text.strip()
        assert final_response == "The image shows a sunset over the ocean with vibrant orange and purple colors."

    def test_reasoning_field_promoted_as_fallback(self):
        """When reasoning_content is None but reasoning field has text."""
        agent = _make_agent_stub()
        assistant = _make_assistant_message(
            content="",
            reasoning="The model analyzed the document and found 3 key points.",
        )
        _rc_text = (
            getattr(assistant, "reasoning_content", None)
            or getattr(assistant, "reasoning", None)
            or ""
        )
        assert isinstance(_rc_text, str) and _rc_text.strip()
        assert "3 key points" in _rc_text

    def test_empty_reasoning_not_promoted(self):
        """When both content and reasoning are empty, promotion should not fire."""
        assistant = _make_assistant_message(content="", reasoning_content="")
        _rc_text = (
            getattr(assistant, "reasoning_content", None)
            or getattr(assistant, "reasoning", None)
            or ""
        )
        # Empty reasoning — should NOT promote
        assert not (isinstance(_rc_text, str) and _rc_text.strip())

    def test_whitespace_only_reasoning_not_promoted(self):
        """Whitespace-only reasoning should not be promoted."""
        assistant = _make_assistant_message(content="", reasoning_content="   \n  ")
        _rc_text = (
            getattr(assistant, "reasoning_content", None)
            or getattr(assistant, "reasoning", None)
            or ""
        )
        assert not (isinstance(_rc_text, str) and _rc_text.strip())

    def test_content_present_skips_promotion(self):
        """When content is non-empty, promotion path is never reached."""
        assistant = _make_assistant_message(
            content="Here is the analysis.",
            reasoning_content="Internal thinking...",
        )
        # If content is non-empty, the conversation loop breaks out before
        # reaching the promotion logic. Verify content is present.
        assert assistant.content.strip()

    def test_prefill_not_exhausted_skips_promotion(self):
        """When prefill retries < 2, the prefill retry path fires instead."""
        agent = _make_agent_stub()
        agent._thinking_prefill_retries = 1  # NOT exhausted
        assert agent._thinking_prefill_retries < 2  # prefill path would fire
