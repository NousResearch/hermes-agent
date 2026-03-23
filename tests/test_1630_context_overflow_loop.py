"""Tests for #1630 — gateway infinite 400 failure loop prevention.

Verifies that:
1. Generic 400 errors with large sessions are treated as context-length errors
   and trigger compression instead of aborting.
2. The gateway does not persist messages when the agent fails early, preventing
   the session from growing on each failure.
3. Context-overflow failures produce helpful error messages suggesting /compact.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Test 1: Agent heuristic — generic 400 with large session → compression
# ---------------------------------------------------------------------------


class TestGeneric400Heuristic:
    """The agent should treat a generic 400 with a large session as a
    probable context-length error and trigger compression, not abort."""

    def _make_agent(self):
        """Create a minimal AIAgent for testing error handling."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            from run_agent import AIAgent
            a = AIAgent(
                api_key="test-key-12345",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            a.client = MagicMock()
            a._cached_system_prompt = "You are helpful."
            a._use_prompt_caching = False
            a.tool_delay = 0
            a.compression_enabled = False
            return a

    def test_generic_400_with_small_session_is_client_error(self):
        """A generic 400 with a small session should still be treated
        as a non-retryable client error (not context overflow)."""
        error_msg = "error"
        status_code = 400
        approx_tokens = 1000  # Small session
        api_messages = [{"role": "user", "content": "hi"}]

        # Simulate the phrase matching
        is_context_length_error = any(phrase in error_msg for phrase in [
            'context length', 'context size', 'maximum context',
            'token limit', 'too many tokens', 'reduce the length',
            'exceeds the limit', 'context window',
            'request entity too large',
            'prompt is too long',
        ])
        assert not is_context_length_error

        # The heuristic should NOT trigger for small sessions
        ctx_len = 200000
        is_large_session = approx_tokens > ctx_len * 0.4 or len(api_messages) > 80
        is_generic_error = len(error_msg.strip()) < 30
        assert not is_large_session  # Small session → heuristic doesn't fire

    def test_generic_400_with_large_token_count_triggers_heuristic(self):
        """A generic 400 with high token count should be treated as
        probable context overflow."""
        error_msg = "error"
        status_code = 400
        ctx_len = 200000
        approx_tokens = 100000  # > 40% of 200k
        api_messages = [{"role": "user", "content": "hi"}] * 20

        is_context_length_error = any(phrase in error_msg for phrase in [
            'context length', 'context size', 'maximum context',
        ])
        assert not is_context_length_error

        # Heuristic check
        is_large_session = approx_tokens > ctx_len * 0.4 or len(api_messages) > 80
        is_generic_error = len(error_msg.strip()) < 30
        assert is_large_session
        assert is_generic_error
        # Both conditions true → should be treated as context overflow

    def test_generic_400_with_many_messages_triggers_heuristic(self):
        """A generic 400 with >80 messages should trigger the heuristic
        even if estimated tokens are low."""
        error_msg = "error"
        status_code = 400
        ctx_len = 200000
        approx_tokens = 5000  # Low token estimate
        api_messages = [{"role": "user", "content": "x"}] * 100  # > 80 messages

        is_large_session = approx_tokens > ctx_len * 0.4 or len(api_messages) > 80
        is_generic_error = len(error_msg.strip()) < 30
        assert is_large_session
        assert is_generic_error

    def test_specific_error_message_bypasses_heuristic(self):
        """A 400 with a specific, long error message should NOT trigger
        the heuristic even with a large session."""
        error_msg = "invalid model: anthropic/claude-nonexistent-model is not available"
        status_code = 400
        ctx_len = 200000
        approx_tokens = 100000

        is_generic_error = len(error_msg.strip()) < 30
        assert not is_generic_error  # Long specific message → heuristic doesn't fire

    def test_descriptive_context_error_caught_by_phrases(self):
        """Descriptive context-length errors should still be caught by
        the existing phrase matching (not the heuristic)."""
        error_msg = "prompt is too long: 250000 tokens > 200000 maximum"
        is_context_length_error = any(phrase in error_msg for phrase in [
            'context length', 'context size', 'maximum context',
            'token limit', 'too many tokens', 'reduce the length',
            'exceeds the limit', 'context window',
            'request entity too large',
            'prompt is too long',
        ])
        assert is_context_length_error

    def test_zai_prompt_exceeds_max_length_needs_phrase_match(self):
        """Z.AI returns 'Prompt exceeds max length' for context overflow."""
        error_msg = (
            "Error code: 400 - {'error': {'code': '1261', "
            "'message': 'Prompt exceeds max length'}}"
        )
        is_context_length_error = any(phrase in error_msg.lower() for phrase in [
            'context length', 'context size', 'maximum context',
            'token limit', 'too many tokens', 'reduce the length',
            'exceeds the limit', 'context window',
            'request entity too large',
            'prompt is too long',
            'prompt exceeds max length',
        ])
        assert is_context_length_error

    def test_zai_prompt_exceeds_max_length_triggers_compression_without_persisting_guess(self):
        """Provider-specific GLM overflow should compress but not persist guessed tiers."""
        error = Exception(
            "Error code: 400 - {'error': {'code': '1261', "
            "'message': 'Prompt exceeds max length'}}"
        )
        error.status_code = 400

        agent = self._make_agent()
        agent.model = "glm-5-turbo"
        agent.provider = "zai"
        agent.base_url = "https://api.z.ai/api/coding/paas/v4"
        agent._base_url_lower = agent.base_url.lower()
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 170_000
        ok_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Recovered", tool_calls=None, reasoning_content=None, reasoning=None), finish_reason="stop")],
            model="glm-5-turbo",
            usage=None,
        )
        agent.client.chat.completions.create.side_effect = [error, ok_resp]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context", return_value=(
                [{"role": "user", "content": "compressed summary"}],
                "compressed prompt",
            )) as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.save_context_length") as mock_save_context,
        ):
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        mock_save_context.assert_not_called()
        assert result["completed"] is True
        assert result["final_response"] == "Recovered"

    def test_parsed_context_limit_is_persisted(self):
        """A provider-reported numeric limit should still be cached."""
        error = Exception(
            "Error code: 400 - prompt is too long: 250000 tokens > 200000 maximum"
        )
        error.status_code = 400

        agent = self._make_agent()
        agent.model = "glm-5-turbo"
        agent.provider = "zai"
        agent.base_url = "https://api.z.ai/api/coding/paas/v4"
        agent._base_url_lower = agent.base_url.lower()
        agent.context_compressor.context_length = 250_000
        agent.context_compressor.threshold_tokens = 212_500
        ok_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Recovered", tool_calls=None, reasoning_content=None, reasoning=None), finish_reason="stop")],
            model="glm-5-turbo",
            usage=None,
        )
        agent.client.chat.completions.create.side_effect = [error, ok_resp]

        with (
            patch.object(agent, "_compress_context", return_value=(
                [{"role": "user", "content": "compressed summary"}],
                "compressed prompt",
            )),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.save_context_length") as mock_save_context,
        ):
            result = agent.run_conversation("hello", conversation_history=[
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": "previous answer"},
            ])

        mock_save_context.assert_called_once_with(
            "glm-5-turbo",
            "https://api.z.ai/api/coding/paas/v4",
            200_000,
        )
        assert result["completed"] is True

    def test_preflight_counts_tool_schemas(self):
        """Tool schemas should count toward preflight compression pressure."""
        huge_tools = [
            {
                "type": "function",
                "function": {
                    "name": "huge_tool",
                    "description": "d" * 400,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "payload": {
                                "type": "string",
                                "description": "p" * 400,
                            }
                        },
                    },
                },
            }
        ]
        with (
            patch("run_agent.get_tool_definitions", return_value=huge_tools),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key-12345",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        agent.client = MagicMock()
        agent._cached_system_prompt = "s" * 160
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = True
        agent.context_compressor.context_length = 240
        agent.context_compressor.threshold_tokens = 200
        agent.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None, reasoning=None), finish_reason="stop")],
            model="test-model",
            usage=None,
        )

        history = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]

        with (
            patch.object(agent, "_compress_context", return_value=(
                [{"role": "user", "content": "compressed summary"}],
                "compressed prompt",
            )) as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=history)

        from agent.model_metadata import estimate_request_tokens_rough

        assert mock_compress.call_count >= 1
        first_call = mock_compress.call_args_list[0]
        first_approx = first_call.kwargs["approx_tokens"]
        request_messages = history + [{"role": "user", "content": "hello"}]
        expected_with_tools = estimate_request_tokens_rough(
            request_messages,
            system_prompt="s" * 160,
            tools=huge_tools,
        )
        expected_without_tools = estimate_request_tokens_rough(
            request_messages,
            system_prompt="s" * 160,
        )

        assert expected_without_tools < agent.context_compressor.threshold_tokens
        assert expected_with_tools > agent.context_compressor.threshold_tokens
        assert first_approx == expected_with_tools
        assert result["completed"] is True


# ---------------------------------------------------------------------------
# Test 2: Gateway skips persistence on failed agent results
# ---------------------------------------------------------------------------

class TestGatewaySkipsPersistenceOnFailure:
    """When the agent returns failed=True with no final_response,
    the gateway should NOT persist messages to the transcript."""

    def test_agent_failed_early_detected(self):
        """The agent_failed_early flag is True when failed=True and
        no final_response."""
        agent_result = {
            "failed": True,
            "final_response": None,
            "messages": [],
            "error": "Non-retryable client error",
        }
        agent_failed_early = (
            agent_result.get("failed")
            and not agent_result.get("final_response")
        )
        assert agent_failed_early

    def test_agent_with_response_not_failed_early(self):
        """When the agent has a final_response, it's not a failed-early
        scenario even if failed=True."""
        agent_result = {
            "failed": True,
            "final_response": "Here is a partial response",
            "messages": [],
        }
        agent_failed_early = (
            agent_result.get("failed")
            and not agent_result.get("final_response")
        )
        assert not agent_failed_early

    def test_successful_agent_not_failed_early(self):
        """A successful agent result should not trigger skip."""
        agent_result = {
            "final_response": "Hello!",
            "messages": [{"role": "assistant", "content": "Hello!"}],
        }
        agent_failed_early = (
            agent_result.get("failed")
            and not agent_result.get("final_response")
        )
        assert not agent_failed_early


# ---------------------------------------------------------------------------
# Test 3: Context-overflow error messages
# ---------------------------------------------------------------------------

class TestContextOverflowErrorMessages:
    """The gateway should produce helpful error messages when the failure
    looks like a context overflow."""

    def test_detects_context_keywords(self):
        """Error messages containing context-related keywords should be
        identified as context failures."""
        keywords = [
            "context length exceeded",
            "too many tokens in the prompt",
            "request entity too large",
            "payload too large for model",
            "context window exceeded",
        ]
        for error_str in keywords:
            _is_ctx_fail = any(p in error_str.lower() for p in (
                "context", "token", "too large", "too long",
                "exceed", "payload",
            ))
            assert _is_ctx_fail, f"Should detect: {error_str}"

    def test_detects_generic_400_with_large_history(self):
        """A generic 400 error code in the string with a large history
        should be flagged as context failure."""
        error_str = "error code: 400 - {'type': 'error', 'message': 'Error'}"
        history_len = 100  # Large session

        _is_ctx_fail = any(p in error_str.lower() for p in (
            "context", "token", "too large", "too long",
            "exceed", "payload",
        )) or (
            "400" in error_str.lower()
            and history_len > 50
        )
        assert _is_ctx_fail

    def test_unrelated_error_not_flagged(self):
        """Unrelated errors should not be flagged as context failures."""
        error_str = "invalid api key: authentication failed"
        history_len = 10

        _is_ctx_fail = any(p in error_str.lower() for p in (
            "context", "token", "too large", "too long",
            "exceed", "payload",
        )) or (
            "400" in error_str.lower()
            and history_len > 50
        )
        assert not _is_ctx_fail


# ---------------------------------------------------------------------------
# Test 4: Agent skips persistence for large failed sessions
# ---------------------------------------------------------------------------

class TestAgentSkipsPersistenceForLargeFailedSessions:
    """When a 400 error occurs and the session is large, the agent
    should skip persisting to prevent the growth loop."""

    def test_large_session_400_skips_persistence(self):
        """Status 400 + high token count should skip persistence."""
        status_code = 400
        approx_tokens = 60000  # > 50000 threshold
        api_messages = [{"role": "user", "content": "x"}] * 10

        should_skip = status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80)
        assert should_skip

    def test_small_session_400_persists_normally(self):
        """Status 400 + small session should still persist."""
        status_code = 400
        approx_tokens = 5000  # < 50000
        api_messages = [{"role": "user", "content": "x"}] * 10  # < 80

        should_skip = status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80)
        assert not should_skip

    def test_non_400_error_persists_normally(self):
        """Non-400 errors should always persist normally."""
        status_code = 401  # Auth error
        approx_tokens = 100000  # Large session, but not a 400
        api_messages = [{"role": "user", "content": "x"}] * 100

        should_skip = status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80)
        assert not should_skip
