import pytest
from unittest.mock import MagicMock, patch
import asyncio

from agent.turn_context import build_turn_context

def test_compression_interrupt_ghosting():
    agent = MagicMock()
    agent.model = "test-model"
    agent.provider = "test"
    agent._memory_nudge_interval = 0
    agent.compression_idle_compact_after_seconds = 0
    messages = [{"role": "user", "content": "hello"}]
    
    # Force compression threshold trigger
    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True):
        with patch("agent.turn_context.estimate_request_tokens_rough", return_value=1000) as mock_estimate:
            agent.compression_enabled = True
            agent.context_compressor = MagicMock(threshold_tokens=500)
            agent.context_compressor.should_defer_preflight_to_real_usage = MagicMock(return_value=False)
            agent.context_compressor.should_compress.return_value = True
            agent.context_compressor.context_length = 4000
            agent.context_compressor.last_real_prompt_tokens = 0
            agent.context_compressor.last_prompt_tokens = 0
            agent.context_compressor.get_active_compression_failure_cooldown = MagicMock(return_value=None)
            agent.max_compression_attempts = 1
            
            # Mock _compress_context to raise an exception (simulating user interrupt)
            def mock_compress_context(*args, **kwargs):
                raise InterruptedError("Simulated interrupt")
                
            agent._compress_context = mock_compress_context
                
            # If ghosting bug exists, build_turn_context would crash.
            # With our fix, it should catch the exception, set the flag, and return normally.
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
            
            # Verification
            assert ctx is not None
            assert ctx.preflight_compression_blocked is True
            assert len(ctx.messages) > 0
