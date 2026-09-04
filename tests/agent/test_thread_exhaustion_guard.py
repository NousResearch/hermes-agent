"""Tests for thread exhaustion guard in chat_completion_helpers and tool_executor.

Verifies that RuntimeError("can't start new thread") from OS thread exhaustion
is:
1. Propagated immediately (not caught/ignored)
2. Classified as non-retryable by error_classifier
3. Guarded in all threading.Thread().start() and executor.submit() sites
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from tools.thread_exhaustion import is_thread_exhaustion_error


class TestThreadExhaustionDetection:
    """Test the is_thread_exhaustion_error helper."""

    def test_canonical_cpython_message(self):
        """Canonical CPython error text is detected."""
        exc = RuntimeError("can't start new thread")
        assert is_thread_exhaustion_error(exc)

    def test_legacy_cpython_message(self):
        """Pre-3.8 CPython error text is detected."""
        exc = RuntimeError("can't create new thread")
        assert is_thread_exhaustion_error(exc)

    def test_cannot_variant(self):
        """'cannot' variant is detected."""
        exc = RuntimeError("cannot start new thread")
        assert is_thread_exhaustion_error(exc)

    def test_unable_variant(self):
        """'unable to create' variant is detected."""
        exc = RuntimeError("unable to create new thread")
        assert is_thread_exhaustion_error(exc)

    def test_case_insensitive(self):
        """Detection is case-insensitive."""
        exc = RuntimeError("Can't Start New Thread")
        assert is_thread_exhaustion_error(exc)

    def test_other_runtime_error_not_detected(self):
        """Other RuntimeError messages are not detected."""
        exc = RuntimeError("some other error")
        assert not is_thread_exhaustion_error(exc)

    def test_interpreter_shutdown_not_detected(self):
        """Interpreter shutdown error is distinct from thread exhaustion."""
        exc = RuntimeError("cannot schedule new futures after interpreter shutdown")
        assert not is_thread_exhaustion_error(exc)

    def test_non_runtime_error_not_detected(self):
        """Non-RuntimeError exceptions are not detected."""
        assert not is_thread_exhaustion_error(ValueError("can't start new thread"))
        assert not is_thread_exhaustion_error(OSError("can't start new thread"))


class TestErrorClassification:
    """Test error_classifier integration."""

    def test_thread_exhaustion_classified_correctly(self):
        """Thread exhaustion is classified as non-retryable."""
        exc = RuntimeError("can't start new thread")
        result = classify_api_error(exc, provider="test", model="test-model")
        
        assert result.reason == FailoverReason.thread_exhaustion
        assert result.retryable is False
        assert result.should_fallback is False
        assert result.should_compress is False

    def test_thread_exhaustion_takes_precedence(self):
        """Thread exhaustion is classified before generic RuntimeError."""
        exc = RuntimeError("can't start new thread")
        result = classify_api_error(
            exc,
            provider="test",
            model="test-model",
            approx_tokens=100000,
            context_length=200000,
        )
        
        # Should be thread_exhaustion, not unknown
        assert result.reason == FailoverReason.thread_exhaustion


class TestChatCompletionHelpersGuard:
    """Test threading.Thread().start() guards in chat_completion_helpers."""

    @patch("threading.Thread")
    def test_non_streaming_thread_exhaustion_propagates(self, mock_thread_class):
        """Non-streaming path propagates thread exhaustion immediately."""
        from agent.chat_completion_helpers import interruptible_api_call
        
        # Simulate thread exhaustion on Thread.start()
        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance
        mock_thread_instance.start.side_effect = RuntimeError("can't start new thread")
        
        # Create minimal mock agent with required attributes
        agent = MagicMock()
        agent.api_mode = "chat_completions"
        agent.provider = "openai"
        agent._interrupt_requested = False
        agent._touch_activity = MagicMock()
        
        api_kwargs = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        
        with pytest.raises(RuntimeError, match="can't start new thread"):
            interruptible_api_call(agent, api_kwargs)

    @patch("threading.Thread")
    def test_non_streaming_other_runtime_error_propagates(self, mock_thread_class):
        """Non-streaming path propagates non-thread-exhaustion RuntimeError."""
        from agent.chat_completion_helpers import interruptible_api_call
        
        # Simulate some other RuntimeError (not thread exhaustion)
        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance
        mock_thread_instance.start.side_effect = RuntimeError("some other error")
        
        # Create minimal mock agent with required attributes
        agent = MagicMock()
        agent.api_mode = "chat_completions"
        agent.provider = "openai"
        agent._interrupt_requested = False
        agent._touch_activity = MagicMock()
        
        api_kwargs = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        
        with pytest.raises(RuntimeError, match="some other error"):
            interruptible_api_call(agent, api_kwargs)

    @patch("threading.Thread")
    def test_streaming_thread_exhaustion_propagates(self, mock_thread_class):
        """Streaming path propagates thread exhaustion immediately."""
        from agent.chat_completion_helpers import interruptible_streaming_api_call
        
        # Simulate thread exhaustion on Thread.start()
        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance
        mock_thread_instance.start.side_effect = RuntimeError("can't start new thread")
        
        # Create minimal mock agent with required attributes
        agent = MagicMock()
        agent.api_mode = "chat_completions"
        agent.provider = "openai"
        agent._interrupt_requested = False
        agent._touch_activity = MagicMock()
        
        api_kwargs = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        
        with pytest.raises(RuntimeError, match="can't start new thread"):
            interruptible_streaming_api_call(agent, api_kwargs)


class TestToolExecutorGuard:
    """Test executor.submit() guard in tool_executor."""

    def test_submit_thread_exhaustion_propagates(self):
        """executor.submit() thread exhaustion is propagated immediately."""
        from agent.tool_executor import _is_thread_exhaustion_submit_error
        
        exc = RuntimeError("can't start new thread")
        assert _is_thread_exhaustion_submit_error(exc)

    def test_submit_interpreter_shutdown_still_works(self):
        """Interpreter shutdown detection still works after adding thread guard."""
        from agent.tool_executor import _is_interpreter_shutdown_submit_error
        
        exc = RuntimeError("cannot schedule new futures after interpreter shutdown")
        assert _is_interpreter_shutdown_submit_error(exc)
        
        # Thread exhaustion should NOT match interpreter shutdown
        from agent.tool_executor import _is_thread_exhaustion_submit_error
        assert not _is_thread_exhaustion_submit_error(exc)

    def test_concurrent_tools_thread_exhaustion_propagates(self):
        """Concurrent tool execution propagates thread exhaustion."""
        from agent.tool_executor import execute_tool_calls_concurrent
        
        # Create minimal mock agent with required attributes
        agent = MagicMock()
        agent.quiet_mode = True
        agent._interrupt_requested = False
        agent._should_emit_quiet_tool_messages.return_value = False
        agent._should_start_quiet_spinner.return_value = False
        agent._current_tool = None
        agent._touch_activity = MagicMock()
        agent._tool_worker_threads = set()
        agent._tool_worker_threads_lock = threading.Lock()
        agent.log_prefix = "[test] "
        
        # Mock tool call
        class FakeToolCall:
            def __init__(self):
                self.id = "call_123"
                self.function = type('obj', (object,), {
                    'name': 'terminal',
                    'arguments': '{"command": "echo test"}'
                })()
        
        # Mock assistant message with tool calls
        assistant_message = type('obj', (object,), {
            'tool_calls': [FakeToolCall()]
        })()
        
        messages = []
        
        # Patch ThreadPoolExecutor.submit to raise thread exhaustion
        def fake_submit(*args, **kwargs):
            raise RuntimeError("can't start new thread")
        
        with patch("tools.daemon_pool.DaemonThreadPoolExecutor.submit", side_effect=fake_submit):
            with pytest.raises(RuntimeError, match="can't start new thread"):
                execute_tool_calls_concurrent(
                    agent,
                    assistant_message=assistant_message,
                    messages=messages,
                    effective_task_id="default",
                )

    def test_concurrent_tools_other_submit_error_propagates(self):
        """Concurrent tool execution propagates non-thread-exhaustion submit errors."""
        from agent.tool_executor import execute_tool_calls_concurrent
        
        # Create minimal mock agent with required attributes
        agent = MagicMock()
        agent.quiet_mode = True
        agent._interrupt_requested = False
        agent._should_emit_quiet_tool_messages.return_value = False
        agent._should_start_quiet_spinner.return_value = False
        agent._current_tool = None
        agent._touch_activity = MagicMock()
        agent._tool_worker_threads = set()
        agent._tool_worker_threads_lock = threading.Lock()
        agent.log_prefix = "[test] "
        
        class FakeToolCall:
            def __init__(self):
                self.id = "call_123"
                self.function = type('obj', (object,), {
                    'name': 'terminal',
                    'arguments': '{"command": "echo test"}'
                })()
        
        # Mock assistant message with tool calls
        assistant_message = type('obj', (object,), {
            'tool_calls': [FakeToolCall()]
        })()
        
        messages = []
        
        # Patch ThreadPoolExecutor.submit to raise some other RuntimeError
        def fake_submit(*args, **kwargs):
            raise RuntimeError("some other submit error")
        
        with patch("tools.daemon_pool.DaemonThreadPoolExecutor.submit", side_effect=fake_submit):
            with pytest.raises(RuntimeError, match="some other submit error"):
                execute_tool_calls_concurrent(
                    agent,
                    assistant_message=assistant_message,
                    messages=messages,
                    effective_task_id="default",
                )
