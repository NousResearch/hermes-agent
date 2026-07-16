"""Regression test: finish_reason=tool_calls with empty tool_calls (#65430).

When a provider returns finish_reason="tool_calls" but the normalized
tool_calls list is None/empty, the conversation loop must NOT treat the
accompanying narration text as the final response.  Instead it should
re-prompt the model to re-emit tool calls (up to 3 retries).

This test validates the fix at the unit level by running the agent with
a mock client that simulates the ghost tool_calls scenario.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from run_agent import AIAgent


def _make_agent():
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


class TestGhostToolCallGuard:
    """finish_reason=tool_calls with empty tool_calls should trigger re-prompts."""

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_ghost_tool_calls_retried_then_normal_stop(self, _mock_close, mock_create):
        """Model returns finish_reason=tool_calls with no tool_calls on first
        call, then returns a proper stop response on second call."""
        call_count = 0

        # We need to mock at the transport level
        agent = _make_agent()

        # Track what normalize_response returns per call
        original_get_transport = agent._get_transport

        def _patched_get_transport():
            transport = original_get_transport()
            original_normalize = transport.normalize_response

            def _mock_normalize(response, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # Ghost: finish_reason=tool_calls but no actual tools
                    return SimpleNamespace(
                        content="Let me check the remaining data...",
                        tool_calls=None,
                        finish_reason="tool_calls",
                        refusal=None,
                        reasoning_content=None,
                        reasoning=None,
                        id="resp-1",
                        provider_data={},
                        model="test/model",
                        usage=SimpleNamespace(
                            prompt_tokens=100, completion_tokens=50,
                            total_tokens=150,
                        ),
                    )
                else:
                    # Proper final response
                    return SimpleNamespace(
                        content="Here is the complete security report with all findings.",
                        tool_calls=None,
                        finish_reason="stop",
                        refusal=None,
                        reasoning_content=None,
                        reasoning=None,
                        id="resp-2",
                        provider_data={},
                        model="test/model",
                        usage=SimpleNamespace(
                            prompt_tokens=200, completion_tokens=100,
                            total_tokens=300,
                        ),
                    )

            transport.normalize_response = _mock_normalize
            return transport

        agent._get_transport = _patched_get_transport

        # Mock the actual API client
        mock_response = MagicMock()
        mock_response.id = "resp-mock"
        mock_response.model = "test/model"
        mock_response.usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )
        mock_response.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content="text", tool_calls=None, role="assistant"
                ),
                finish_reason="stop",
            )
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_create.return_value = mock_client

        result = agent.run_conversation("Run full security audit")

        # Should have called normalize at least twice
        assert call_count >= 2
        # Final response should be the real answer
        assert "complete security report" in result.get("final_response", "")
        # Should NOT contain the ghost narration
        assert "Let me check" not in result.get("final_response", "")

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_ghost_tool_calls_caps_at_three_retries(self, _mock_close, mock_create):
        """After 3 retries of ghost tool_calls, fall through as response."""
        call_count = 0

        agent = _make_agent()
        agent.max_iterations = 20

        original_get_transport = agent._get_transport

        def _patched_get_transport():
            transport = original_get_transport()

            def _mock_normalize(response, **kwargs):
                nonlocal call_count
                call_count += 1
                # Always return ghost tool_calls
                return SimpleNamespace(
                    content="Still working on it...",
                    tool_calls=None,
                    finish_reason="tool_calls",
                    refusal=None,
                    reasoning_content=None,
                    reasoning=None,
                    id=f"resp-{call_count}",
                    provider_data={},
                    model="test/model",
                    usage=SimpleNamespace(
                        prompt_tokens=100, completion_tokens=50,
                        total_tokens=150,
                    ),
                )

            transport.normalize_response = _mock_normalize
            return transport

        agent._get_transport = _patched_get_transport

        mock_response = MagicMock()
        mock_response.id = "resp-mock"
        mock_response.model = "test/model"
        mock_response.usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )
        mock_response.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content="text", tool_calls=None, role="assistant"
                ),
                finish_reason="tool_calls",
            )
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_create.return_value = mock_client

        result = agent.run_conversation("Run full audit")

        # After 3 retries (calls 1-3 trigger re-prompt), call 4 falls through
        assert call_count >= 4
        # Should eventually produce a response (partial)
        assert result.get("final_response") is not None
