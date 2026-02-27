"""Tests for AIAgent._sanitize_for_api — whitelist-based message sanitization."""

import pytest
from unittest.mock import patch, MagicMock

from run_agent import AIAgent


@pytest.fixture
def agent():
    """Create a minimal AIAgent with mocked dependencies."""
    with patch("run_agent.OpenAI"), \
         patch("run_agent.get_tool_definitions", return_value=[]), \
         patch("run_agent.check_toolset_requirements", return_value={}):
        return AIAgent(
            api_key="test-key",
            base_url="https://api.mistral.ai/v1",
            quiet_mode=True,
        )


class TestSanitizeForApi:
    """Tests for the _sanitize_for_api whitelist sanitizer."""

    def test_finish_reason_stripped(self, agent):
        """finish_reason must not leak to the API (the actual Mistral 422 bug)."""
        msg = {
            "role": "assistant",
            "content": "Hello!",
            "finish_reason": "stop",
        }
        result = agent._sanitize_for_api(msg)
        assert "finish_reason" not in result
        assert result["content"] == "Hello!"
        assert result["role"] == "assistant"

    def test_reasoning_becomes_reasoning_content(self, agent):
        """Internal 'reasoning' field should be converted to 'reasoning_content'."""
        msg = {
            "role": "assistant",
            "content": "Answer.",
            "reasoning": "Thinking step by step...",
            "finish_reason": "stop",
        }
        result = agent._sanitize_for_api(msg)
        assert "reasoning" not in result
        assert "finish_reason" not in result
        assert result["reasoning_content"] == "Thinking step by step..."

    def test_reasoning_content_not_added_when_empty(self, agent):
        """No reasoning_content if reasoning is None/empty."""
        msg = {
            "role": "assistant",
            "content": "Hello!",
            "reasoning": None,
            "finish_reason": "stop",
        }
        result = agent._sanitize_for_api(msg)
        assert "reasoning_content" not in result

    def test_flush_sentinel_stripped(self, agent):
        """Internal _flush_sentinel on user messages must not leak."""
        msg = {
            "role": "user",
            "content": "Please save memories.",
            "_flush_sentinel": "__flush_12345",
        }
        result = agent._sanitize_for_api(msg)
        assert "_flush_sentinel" not in result
        assert result["content"] == "Please save memories."

    def test_standard_assistant_fields_preserved(self, agent):
        """Standard API fields (content, tool_calls, reasoning_details) pass through."""
        tool_calls = [{"id": "tc_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]
        reasoning_details = [{"type": "reasoning.summary", "text": "...", "signature": "abc"}]
        msg = {
            "role": "assistant",
            "content": "Let me search.",
            "tool_calls": tool_calls,
            "reasoning_details": reasoning_details,
            "finish_reason": "tool_calls",
            "reasoning": "I should search.",
        }
        result = agent._sanitize_for_api(msg)
        assert result["tool_calls"] == tool_calls
        assert result["reasoning_details"] == reasoning_details
        assert result["reasoning_content"] == "I should search."
        assert "finish_reason" not in result
        assert "reasoning" not in result

    def test_tool_message_preserves_tool_call_id(self, agent):
        """Tool messages keep tool_call_id and name, drop anything extra."""
        msg = {
            "role": "tool",
            "content": '{"result": "ok"}',
            "tool_call_id": "tc_1",
            "name": "search",
            "some_internal_field": True,
        }
        result = agent._sanitize_for_api(msg)
        assert result["tool_call_id"] == "tc_1"
        assert result["name"] == "search"
        assert "some_internal_field" not in result

    def test_system_message_minimal(self, agent):
        """System messages only keep role and content."""
        msg = {
            "role": "system",
            "content": "You are an assistant.",
            "extra": "should be dropped",
        }
        result = agent._sanitize_for_api(msg)
        assert result == {"role": "system", "content": "You are an assistant."}

    def test_unknown_role_defaults_to_role_content(self, agent):
        """Unknown roles fall back to keeping just role + content."""
        msg = {
            "role": "developer",
            "content": "Some content.",
            "extra": "dropped",
        }
        result = agent._sanitize_for_api(msg)
        assert result == {"role": "developer", "content": "Some content."}
