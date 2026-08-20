import pytest
from unittest.mock import MagicMock, patch

from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.turn_context import (
    _should_record_compression_timeout_failure,
    build_turn_context,
)


def test_user_cancel_is_not_a_timeout_failure():
    assert _should_record_compression_timeout_failure(InterruptedError("stop")) is False
    assert _should_record_compression_timeout_failure(AuxiliaryExplicitCancellation()) is False
    assert _should_record_compression_timeout_failure(TimeoutError("stalled")) is True
    assert _should_record_compression_timeout_failure(RuntimeError("provider boom")) is True


def test_compression_interrupt_ghosting():
    agent = MagicMock()
    agent.model = "test-model"
    agent.provider = "test"
    agent._memory_nudge_interval = 0
    agent.compression_idle_compact_after_seconds = 0

    # Force compression threshold trigger
    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True):
        with patch("agent.turn_context.estimate_request_tokens_rough", return_value=1000):
            agent.compression_enabled = True
            compressor_mock = MagicMock(threshold_tokens=500)
            compressor_mock.should_defer_preflight_to_real_usage.return_value = False
            compressor_mock.should_compress.return_value = True
            compressor_mock.context_length = 4000
            compressor_mock.last_real_prompt_tokens = 0
            compressor_mock.last_prompt_tokens = 0
            compressor_mock.get_active_compression_failure_cooldown.return_value = None
            agent.context_compressor = compressor_mock
            agent.max_compression_attempts = 1

            # Mock _compress_context to raise InterruptedError
            def mock_compress_context(*args, **kwargs):
                raise InterruptedError("Simulated interrupt")

            agent._compress_context = mock_compress_context

            ctx = build_turn_context(
                agent=agent,
                user_message={"role": "user", "content": "hello"},
                system_message="system prompt",
                conversation_history=[],
                task_id="test-task",
                stream_callback=None,
                persist_user_message=None,
                restore_or_build_system_prompt=MagicMock(return_value="system prompt"),
                install_safe_stdio=MagicMock(),
                sanitize_surrogates=MagicMock(return_value="hello"),
                summarize_user_message_for_log=MagicMock(return_value="hello"),
                set_session_context=MagicMock(),
                set_current_write_origin=MagicMock(),
                ra=MagicMock(),
            )

            # Verification: turn 1 finishes fail-open. A user interrupt is
            # not a summary-model failure — do not arm the 60s+ timeout ladder.
            assert ctx is not None
            assert ctx.preflight_compression_blocked is True
            assert len(ctx.messages) > 0
            compressor_mock.record_timeout_failure.assert_not_called()


def test_compression_explicit_cancellation_ghosting():
    """Verify that AuxiliaryExplicitCancellation (subclass of BaseException) is caught safely."""
    agent = MagicMock()
    agent.model = "test-model"
    agent.provider = "test"
    agent._memory_nudge_interval = 0
    agent.compression_idle_compact_after_seconds = 0

    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True):
        with patch("agent.turn_context.estimate_request_tokens_rough", return_value=1000):
            agent.compression_enabled = True
            compressor_mock = MagicMock(threshold_tokens=500)
            compressor_mock.should_defer_preflight_to_real_usage.return_value = False
            compressor_mock.should_compress.return_value = True
            compressor_mock.context_length = 4000
            compressor_mock.last_real_prompt_tokens = 0
            compressor_mock.last_prompt_tokens = 0
            compressor_mock.get_active_compression_failure_cooldown.return_value = None
            agent.context_compressor = compressor_mock
            agent.max_compression_attempts = 1

            # Mock _compress_context to raise AuxiliaryExplicitCancellation (BaseException subclass)
            def mock_compress_context(*args, **kwargs):
                raise AuxiliaryExplicitCancellation()

            agent._compress_context = mock_compress_context

            ctx = build_turn_context(
                agent=agent,
                user_message={"role": "user", "content": "hello"},
                system_message="system prompt",
                conversation_history=[],
                task_id="test-task-cancellation",
                stream_callback=None,
                persist_user_message=None,
                restore_or_build_system_prompt=MagicMock(return_value="system prompt"),
                install_safe_stdio=MagicMock(),
                sanitize_surrogates=MagicMock(return_value="hello"),
                summarize_user_message_for_log=MagicMock(return_value="hello"),
                set_session_context=MagicMock(),
                set_current_write_origin=MagicMock(),
                ra=MagicMock(),
            )

            # Verification: BaseException subclass is caught, turn completes
            # fail-open, and /stop does not embargo later compression.
            assert ctx is not None
            assert ctx.preflight_compression_blocked is True
            assert len(ctx.messages) > 0
            compressor_mock.record_timeout_failure.assert_not_called()


def test_compression_failure_cooldown_prevents_subsequent_turn_retry():
    agent = MagicMock()
    agent.model = "test-model"
    agent.provider = "test"
    agent._memory_nudge_interval = 0
    agent.compression_idle_compact_after_seconds = 0

    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True):
        with patch("agent.turn_context.estimate_request_tokens_rough", return_value=1000):
            agent.compression_enabled = True
            compressor_mock = MagicMock(threshold_tokens=500)
            compressor_mock.should_defer_preflight_to_real_usage.return_value = False
            compressor_mock.should_compress.return_value = True
            compressor_mock.context_length = 4000
            compressor_mock.last_real_prompt_tokens = 0
            compressor_mock.last_prompt_tokens = 0
            # Active cooldown is now present for turn 2 after previous failure
            compressor_mock.get_active_compression_failure_cooldown.return_value = {
                "remaining_seconds": 58.0,
                "reason": "failure_cooldown",
            }
            agent.context_compressor = compressor_mock
            agent.max_compression_attempts = 1
            agent._compress_context = MagicMock()

            # Turn 2 execution
            ctx2 = build_turn_context(
                agent=agent,
                user_message={"role": "user", "content": "turn 2 message"},
                system_message="system prompt",
                conversation_history=[],
                task_id="test-task-2",
                stream_callback=None,
                persist_user_message=None,
                restore_or_build_system_prompt=MagicMock(return_value="system prompt"),
                install_safe_stdio=MagicMock(),
                sanitize_surrogates=MagicMock(return_value="turn 2 message"),
                summarize_user_message_for_log=MagicMock(return_value="turn 2 message"),
                set_session_context=MagicMock(),
                set_current_write_origin=MagicMock(),
                ra=MagicMock(),
            )

            # Verification: turn 2 skips _compress_context due to active failure cooldown
            assert ctx2 is not None
            assert len(ctx2.messages) > 0
            agent._compress_context.assert_not_called()


def test_compression_timeout_still_arms_cooldown():
    """A real stall/timeout must still persist the failure cooldown ladder."""
    agent = MagicMock()
    agent.model = "test-model"
    agent.provider = "test"
    agent._memory_nudge_interval = 0
    agent.compression_idle_compact_after_seconds = 0

    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True):
        with patch("agent.turn_context.estimate_request_tokens_rough", return_value=1000):
            agent.compression_enabled = True
            compressor_mock = MagicMock(threshold_tokens=500)
            compressor_mock.should_defer_preflight_to_real_usage.return_value = False
            compressor_mock.should_compress.return_value = True
            compressor_mock.context_length = 4000
            compressor_mock.last_real_prompt_tokens = 0
            compressor_mock.last_prompt_tokens = 0
            compressor_mock.get_active_compression_failure_cooldown.return_value = None
            agent.context_compressor = compressor_mock
            agent.max_compression_attempts = 1

            def mock_compress_context(*args, **kwargs):
                raise TimeoutError("Auxiliary streamed call timed out")

            agent._compress_context = mock_compress_context

            ctx = build_turn_context(
                agent=agent,
                user_message={"role": "user", "content": "hello"},
                system_message="system prompt",
                conversation_history=[],
                task_id="test-task-timeout",
                stream_callback=None,
                persist_user_message=None,
                restore_or_build_system_prompt=MagicMock(return_value="system prompt"),
                install_safe_stdio=MagicMock(),
                sanitize_surrogates=MagicMock(return_value="hello"),
                summarize_user_message_for_log=MagicMock(return_value="hello"),
                set_session_context=MagicMock(),
                set_current_write_origin=MagicMock(),
                ra=MagicMock(),
            )

            assert ctx is not None
            assert ctx.preflight_compression_blocked is True
            compressor_mock.record_timeout_failure.assert_called_once()
