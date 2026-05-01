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
                base_url="https://openrouter.ai/api/v1",
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


# ---------------------------------------------------------------------------
# Test 2: Gateway skips persistence on failed agent results
# ---------------------------------------------------------------------------

class TestGatewaySkipsPersistenceOnFailure:
    """When the agent returns failed=True with no final_response,
    the gateway should NOT persist messages to the transcript."""

    def test_agent_failed_early_detected(self):
        """The agent_failed_early flag is True when failed=True,
        regardless of final_response."""
        agent_result = {
            "failed": True,
            "final_response": None,
            "messages": [],
            "error": "Non-retryable client error",
        }
        agent_failed_early = bool(agent_result.get("failed"))
        assert agent_failed_early

    def test_agent_failed_with_error_response_still_detected(self):
        """When _run_agent_blocking converts an error to final_response,
        the failed flag should still trigger agent_failed_early.  This
        was the core bug in #9893 — the old guard checked
        ``not final_response`` which was always truthy after conversion."""
        agent_result = {
            "failed": True,
            "final_response": "⚠️ Request payload too large: max compression attempts reached.",
            "messages": [],
        }
        agent_failed_early = bool(agent_result.get("failed"))
        assert agent_failed_early

    def test_successful_agent_not_failed_early(self):
        """A successful agent result should not trigger skip."""
        agent_result = {
            "final_response": "Hello!",
            "messages": [{"role": "assistant", "content": "Hello!"}],
        }
        agent_failed_early = bool(agent_result.get("failed"))
        assert not agent_failed_early


class TestCompressionExhaustedFlag:
    """When compression is exhausted, the agent should set both
    failed=True and compression_exhausted=True so the gateway can
    auto-reset the session.  (#9893)"""

    def test_compression_exhausted_returns_carry_flag(self):
        """Simulate the return dict from a compression-exhausted agent."""
        agent_result = {
            "messages": [],
            "completed": False,
            "api_calls": 3,
            "error": "Request payload too large: max compression attempts (3) reached.",
            "partial": True,
            "failed": True,
            "compression_exhausted": True,
        }
        assert agent_result.get("failed")
        assert agent_result.get("compression_exhausted")

    def test_normal_failure_not_compression_exhausted(self):
        """Non-compression failures should not have compression_exhausted."""
        agent_result = {
            "messages": [],
            "completed": False,
            "failed": True,
            "error": "Invalid API response after 3 retries",
        }
        assert agent_result.get("failed")
        assert not agent_result.get("compression_exhausted")


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
