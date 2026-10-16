"""Regression tests for stale-giveup compaction in route_classified_error (#139209).

A wedged session hitting the cross-turn stale-call circuit breaker should
attempt context compaction once when the request exceeds the compressor's
threshold before giving up entirely.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest

from agent.chat_completion_helpers import StaleGiveupError
from agent.error_classifier import FailoverReason


@pytest.fixture
def _stale_error():
    """A StaleGiveupError as raised by _check_stale_giveup."""
    return StaleGiveupError(
        "Provider has been unresponsive (no response received) for "
        "5 consecutive stale attempts — aborting this call to "
        "avoid an indefinite stall. Send another message to try again, "
        "switch models with /model, or start a new session with /new."
    )


@pytest.fixture
def _agent_with_compressor():
    """AIAgent stub with enough attributes for route_classified_error."""
    agent = MagicMock()
    agent.log_prefix = "test: "
    agent.compression_enabled = True
    agent.tools = None
    agent._empty_content_retries = 0
    agent._thinking_prefill_retries = 0
    agent._last_content_with_tools = None
    agent._last_content_tools_all_housekeeping = False
    agent._mute_post_response = False
    agent._fallback_index = 0
    agent._fallback_chain = []  # empty fallback chain: fallback activation yields False
    agent._has_pending_fallback = lambda: False
    agent._try_activate_fallback = MagicMock(return_value=False)
    agent._buffer_diagnostic_status = MagicMock()
    agent._buffer_vprint = MagicMock()
    agent._dump_api_request_debug = MagicMock()
    agent._flush_status_buffer = MagicMock()
    agent._emit_diagnostic_status = MagicMock()
    agent._vprint = MagicMock()
    agent._summarize_api_error = MagicMock(return_value="no response")
    agent._persist_session = MagicMock()
    agent._abort_request_anthropic_client = lambda *a, **k: None

    # Context compressor stub: threshold at 100000 tokens
    compressor = MagicMock()
    compressor.threshold_tokens = 100000
    compressor.context_length = 200000
    compressor.should_compress_info = MagicMock(return_value=(True, None))
    agent.context_compressor = compressor

    # compress_context: return a shorter list (compaction reduced messages)
    agent._compress_context = MagicMock()

    agent._touch_activity = MagicMock()
    agent._invoke_api_request_error_hook = MagicMock()
    agent._extract_api_error_context = MagicMock(return_value={})

    return agent


class TestStaleCompaction:
    """route_classified_error compaction when the stale-call circuit breaker trips."""

    def test_large_context_stale_giveup_compacts_once_and_restarts(self, _stale_error, _agent_with_compressor):
        """Above threshold_tokens: compact once, reset streak, restart."""
        from agent.turn_recovery import route_classified_error
        from agent.turn_retry_state import TurnRetryState
        from agent.model_metadata import estimate_request_tokens_rough

        agent = _agent_with_compressor
        from agent.error_classifier import classify_api_error
        classified = classify_api_error(
            _stale_error, provider="openrouter", model="anthropic/claude-fable-5",
            approx_tokens=126327, context_length=200000, num_messages=274,
        )

        # Build messages large enough that estimate_request_tokens_rough exceeds threshold_tokens (100k)
        big_content = "x" * 500000  # ~125k tokens at 4 chars/token
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": big_content},
        ]
        compressed_messages = [{"role": "system", "content": "sys"}]
        api_messages = list(messages)
        agent._compress_context.return_value = (compressed_messages, "compacted system prompt")

        _retry = TurnRetryState()

        result = route_classified_error(
            agent, _stale_error, classified, _retry,
            error_msg=str(_stale_error), error_context={}, recovered_with_pool=False,
            base_url="https://example.com/v1", model="test-model",
            messages=messages, api_messages=api_messages, system_message="sys",
            active_system_prompt="sys", conversation_history=[],
            retry_count=0, max_retries=5, compression_attempts=0,
            max_compression_attempts=3, api_call_count=1,
            effective_task_id="task-1",
        )

        # Must break to restart with compressed messages
        assert result.action == "break"
        assert result.messages == compressed_messages
        assert _retry.restart_with_compressed_messages is True
        assert _retry.stale_compaction_attempted is True
        agent._compress_context.assert_called_once()

    def test_second_stale_giveup_does_not_compact(self, _stale_error, _agent_with_compressor):
        """stale_compaction_attempted already True: no second compaction."""
        from agent.turn_recovery import route_classified_error
        from agent.turn_retry_state import TurnRetryState

        agent = _agent_with_compressor
        from agent.error_classifier import classify_api_error
        classified = classify_api_error(
            _stale_error, provider="openrouter", model="anthropic/claude-fable-5",
            approx_tokens=126327, context_length=200000, num_messages=274,
        )

        messages = [{"role": "user", "content": "hello"}]
        api_messages = [{"role": "user", "content": "hello"}]

        _retry = TurnRetryState()
        _retry.stale_compaction_attempted = True  # already tried

        result = route_classified_error(
            agent, _stale_error, classified, _retry,
            error_msg=str(_stale_error), error_context={}, recovered_with_pool=False,
            base_url="https://example.com/v1", model="test-model",
            messages=messages, api_messages=api_messages, system_message=None,
            active_system_prompt=None, conversation_history=[],
            retry_count=0, max_retries=5, compression_attempts=0,
            max_compression_attempts=3, api_call_count=1,
            effective_task_id="task-1",
        )

        # Must fall through (no compaction)
        assert result.action == "fallthrough"
        agent._compress_context.assert_not_called()

    def test_small_context_stale_giveup_falls_through(self, _stale_error, _agent_with_compressor):
        """Below threshold_tokens: no compaction, just fall through."""
        from agent.turn_recovery import route_classified_error
        from agent.turn_retry_state import TurnRetryState

        agent = _agent_with_compressor
        from agent.error_classifier import classify_api_error
        classified = classify_api_error(
            _stale_error, provider="openrouter", model="anthropic/claude-fable-5",
            approx_tokens=500, context_length=200000, num_messages=2,
        )

        messages = [{"role": "user", "content": "hi"}]
        api_messages = [{"role": "user", "content": "hi"}]

        _retry = TurnRetryState()

        result = route_classified_error(
            agent, _stale_error, classified, _retry,
            error_msg=str(_stale_error), error_context={}, recovered_with_pool=False,
            base_url="https://example.com/v1", model="test-model",
            messages=messages, api_messages=api_messages, system_message=None,
            active_system_prompt=None, conversation_history=[],
            retry_count=0, max_retries=5, compression_attempts=0,
            max_compression_attempts=3, api_call_count=1,
            effective_task_id="task-1",
        )

        # Small context: falls through without compaction
        assert result.action == "fallthrough"
        agent._compress_context.assert_not_called()

    def test_compression_disabled_stale_giveup_falls_through(self, _stale_error, _agent_with_compressor):
        """compression_enabled=False: no compaction."""
        from agent.turn_recovery import route_classified_error
        from agent.turn_retry_state import TurnRetryState

        agent = _agent_with_compressor
        agent.compression_enabled = False
        from agent.error_classifier import classify_api_error
        classified = classify_api_error(
            _stale_error, provider="openrouter", model="anthropic/claude-fable-5",
            approx_tokens=126327, context_length=200000, num_messages=274,
        )

        messages = [{"role": "user", "content": "hello"}]
        api_messages = [{"role": "user", "content": "hello"}]

        _retry = TurnRetryState()

        result = route_classified_error(
            agent, _stale_error, classified, _retry,
            error_msg=str(_stale_error), error_context={}, recovered_with_pool=False,
            base_url="https://example.com/v1", model="test-model",
            messages=messages, api_messages=api_messages, system_message=None,
            active_system_prompt=None, conversation_history=[],
            retry_count=0, max_retries=5, compression_attempts=0,
            max_compression_attempts=3, api_call_count=1,
            effective_task_id="task-1",
        )

        assert result.action == "fallthrough"
        agent._compress_context.assert_not_called()

    def test_stale_giveup_noop_compaction_falls_through(self, _stale_error, _agent_with_compressor):
        """Compressor returns same message count (cooldown-blocked): fall through, no restart."""
        from agent.turn_recovery import route_classified_error
        from agent.turn_retry_state import TurnRetryState

        agent = _agent_with_compressor
        from agent.error_classifier import classify_api_error
        classified = classify_api_error(
            _stale_error, provider="openrouter", model="anthropic/claude-fable-5",
            approx_tokens=126327, context_length=200000, num_messages=274,
        )

        big_content = "x" * 500000
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": big_content},
        ]
        api_messages = list(messages)
        agent._compress_context.return_value = (list(messages), "compacted system prompt")

        _retry = TurnRetryState()

        result = route_classified_error(
            agent, _stale_error, classified, _retry,
            error_msg=str(_stale_error), error_context={}, recovered_with_pool=False,
            base_url="https://example.com/v1", model="test-model",
            messages=messages, api_messages=api_messages, system_message="sys",
            active_system_prompt="sys", conversation_history=[],
            retry_count=0, max_retries=5, compression_attempts=0,
            max_compression_attempts=3, api_call_count=1,
            effective_task_id="task-1",
        )

        assert result.action == "fallthrough"
        assert _retry.restart_with_compressed_messages is False
        agent._compress_context.assert_called_once()
