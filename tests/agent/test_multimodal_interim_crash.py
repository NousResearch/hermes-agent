"""Tests for multimodal content list handling in interim processing (#66267)."""

import pytest
from unittest.mock import MagicMock


class TestInterimAssistantVisibleText:
    """Verify _interim_assistant_visible_text handles multimodal content lists."""

    def test_multimodal_text_parts_are_extracted(self):
        """A multimodal content list with text parts returns joined text."""
        from run_agent import AIAgent as Agent

        agent = Agent.__new__(Agent)
        agent._extract_codex_interim_visible_text = MagicMock(return_value=None)

        assistant_msg = {
            "content": [
                {"type": "text", "text": "Hello world"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "This is an image."},
            ]
        }

        result = agent._interim_assistant_visible_text(assistant_msg)
        assert result == "Hello world\nThis is an image."

    def test_multimodal_list_with_no_text_returns_empty(self):
        """A multimodal content list with only images returns empty string."""
        from run_agent import AIAgent as Agent

        agent = Agent.__new__(Agent)
        agent._extract_codex_interim_visible_text = MagicMock(return_value=None)

        assistant_msg = {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]
        }

        result = agent._interim_assistant_visible_text(assistant_msg)
        assert result == ""

    def test_multimodal_list_with_think_blocks_stripped(self):
        """Text parts with think blocks have them stripped correctly."""
        from run_agent import AIAgent as Agent

        agent = Agent.__new__(Agent)
        agent._extract_codex_interim_visible_text = MagicMock(return_value=None)

        assistant_msg = {
            "content": [
                {"type": "text", "text": "Here's my reasoning <think>thinking here</think> and conclusion."},
            ]
        }

        result = agent._interim_assistant_visible_text(assistant_msg)
        assert result == "Here's my reasoning  and conclusion."

    def test_plain_string_content_unchanged(self):
        """Plain string content passes through unchanged."""
        from run_agent import AIAgent as Agent

        agent = Agent.__new__(Agent)
        agent._extract_codex_interim_visible_text = MagicMock(return_value=None)

        assistant_msg = {
            "content": "Just a plain string response."
        }

        result = agent._interim_assistant_visible_text(assistant_msg)
        assert result == "Just a plain string response."

    def test_none_content_returns_empty(self):
        """None content returns empty string."""
        from run_agent import AIAgent as Agent

        agent = Agent.__new__(Agent)
        agent._extract_codex_interim_visible_text = MagicMock(return_value=None)

        assistant_msg = {"content": None}

        result = agent._interim_assistant_visible_text(assistant_msg)
        assert result == ""

    def test_missing_content_key_returns_empty(self):
        """Missing content key returns empty string."""
        from run_agent import AIAgent as Agent

        agent = Agent.__new__(Agent)
        agent._extract_codex_interim_visible_text = MagicMock(return_value=None)

        assistant_msg = {}

        result = agent._interim_assistant_visible_text(assistant_msg)
        assert result == ""

    def test_codex_interim_visible_takes_precedence(self):
        """When Codex interim visible text is available, it takes precedence."""
        from run_agent import AIAgent as Agent

        agent = Agent.__new__(Agent)
        agent._extract_codex_interim_visible_text = MagicMock(
            return_value="Codex commentary"
        )

        assistant_msg = {
            "content": [
                {"type": "text", "text": "This should not appear"},
            ]
        }

        result = agent._interim_assistant_visible_text(assistant_msg)
        assert result == "Codex commentary"

    def test_mixed_text_and_non_text_parts(self):
        """Only text parts are extracted; non-text parts are ignored."""
        from run_agent import AIAgent as Agent

        agent = Agent.__new__(Agent)
        agent._extract_codex_interim_visible_text = MagicMock(return_value=None)

        assistant_msg = {
            "content": [
                {"type": "text", "text": "First"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "Second"},
                {"type": "tool_result", "result": "tool output"},  # non-standard type
                {"type": "text", "text": "Third"},
            ]
        }

        result = agent._interim_assistant_visible_text(assistant_msg)
        assert result == "First\nSecond\nThird"