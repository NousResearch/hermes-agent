"""Regression test: finish_reason=tool_calls with empty tool_calls (#65430).

When a provider returns finish_reason="tool_calls" but the normalized
tool_calls list is None/empty, the conversation loop must NOT treat the
accompanying narration text as the final response.  Instead it should
re-prompt the model to re-emit tool calls (up to 3 retries).

This test validates the fix at the unit level by running the agent with
a mock client that simulates the ghost tool_calls scenario.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
        # We need to mock at the transport level
        agent = _make_agent()

        # Track API calls to vary responses per-call
        api_call_count = {"n": 0}

        original_get_transport = agent._get_transport

        def _patched_get_transport():
            transport = original_get_transport()

            def _mock_normalize(response, **kwargs):
                # Keyed off the API call counter, NOT per-normalize call.
                # normalize_response may be called multiple times per API
                # response; return consistent results for the same response.
                if api_call_count["n"] == 1:
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

        # Mock the actual API client — vary per call
        def _make_mock_response(*args, **kwargs):
            api_call_count["n"] += 1
            resp = MagicMock()
            resp.id = f"resp-{api_call_count['n']}"
            resp.model = "test/model"
            resp.usage = SimpleNamespace(
                prompt_tokens=100, completion_tokens=50, total_tokens=150
            )
            resp.choices = [
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="text", tool_calls=None, role="assistant"
                    ),
                    finish_reason="stop",
                )
            ]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _make_mock_response
        mock_create.return_value = mock_client

        result = agent.run_conversation("Run full security audit")

        # Exactly 2 API calls: 1 ghost + 1 successful retry
        assert mock_client.chat.completions.create.call_count == 2
        # Final response should be the real answer
        assert "complete security report" in result.get("final_response", "")
        # Should NOT contain the ghost narration
        assert "Let me check" not in result.get("final_response", "")

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_ghost_tool_calls_caps_at_three_retries(self, _mock_close, mock_create):
        """After 3 retries of ghost tool_calls, fall through as response."""
        agent = _make_agent()
        agent.max_iterations = 20

        original_get_transport = agent._get_transport

        def _patched_get_transport():
            transport = original_get_transport()

            def _mock_normalize(response, **kwargs):
                # Always return ghost tool_calls
                return SimpleNamespace(
                    content="Still working on it...",
                    tool_calls=None,
                    finish_reason="tool_calls",
                    refusal=None,
                    reasoning_content=None,
                    reasoning=None,
                    id="resp-ghost",
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

        def _make_mock_response(*args, **kwargs):
            resp = MagicMock()
            resp.id = "resp-mock"
            resp.model = "test/model"
            resp.usage = SimpleNamespace(
                prompt_tokens=100, completion_tokens=50, total_tokens=150
            )
            resp.choices = [
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="text", tool_calls=None, role="assistant"
                    ),
                    finish_reason="tool_calls",
                )
            ]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _make_mock_response
        mock_create.return_value = mock_client

        result = agent.run_conversation("Run full audit")

        # Exactly 4 API calls: 1 initial + 3 retries, then exhaustion
        assert mock_client.chat.completions.create.call_count == 4
        # Should eventually produce a response (partial)
        assert result.get("final_response") is not None
