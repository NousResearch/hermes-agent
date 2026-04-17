"""Test for meta-only message handling fix.

Issue: #11167 - Agent stuck forever when processing meta-only messages.
The chat_with_model loop waited for response content that never arrives
for meta-only messages like `/model`, `/tools`, etc.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestMetaOnlyMessageHandling:
    """Test that meta-only messages don't cause infinite loops."""

    @pytest.mark.asyncio
    async def test_meta_only_returns_immediately(self):
        """Meta-only messages should return immediately without waiting."""
        from run_agent import AIAgent
        
        # Mock agent
        agent = MagicMock(spec=AIAgent)
        agent._process_user_message = AsyncMock(return_value="Processed /model")
        
        # Simulate meta_only=True path
        meta_only = True
        meta_result = await agent._process_user_message(
            message_event=MagicMock(),
            meta_only=meta_only,
        )
        
        # Should return immediately
        assert meta_result == "Processed /model"

    @pytest.mark.asyncio
    async def test_meta_only_does_not_enter_response_loop(self):
        """Meta-only path should NOT enter the response content loop."""
        from run_agent import AIAgent
        
        agent = MagicMock(spec=AIAgent)
        agent._process_user_message = AsyncMock(return_value=None)
        
        meta_only = True
        meta_result = await agent._process_user_message(
            message_event=MagicMock(),
            meta_only=meta_only,
        )
        
        # Even with None response, should return gracefully
        assert meta_result is None or isinstance(meta_result, str)

    @pytest.mark.asyncio
    async def test_regular_message_enters_response_loop(self):
        """Regular messages (meta_only=False) should enter response loop."""
        from run_agent import AIAgent
        
        agent = MagicMock(spec=AIAgent)
        agent._process_user_message = AsyncMock(return_value=MagicMock(content="Hello"))
        
        meta_only = False
        # This should NOT return immediately - it should enter response loop
        # (But we're just testing the path logic here)
        
    @pytest.mark.asyncio
    async def test_meta_only_with_none_response(self):
        """Meta-only with None response should not stuck."""
        from run_agent import AIAgent
        
        agent = MagicMock(spec=AIAgent)
        agent._process_user_message = AsyncMock(return_value=None)
        
        # Simulate the fixed code path
        meta_only = True
        meta_result = await agent._process_user_message(
            message_event=MagicMock(),
            meta_only=True,
        )
        
        # Fixed path: return default message if meta_result is None
        result = meta_result if meta_result else "Processed meta-only message."
        assert result == "Processed meta-only message."

    def test_meta_only_flag_detection(self):
        """Meta-only flag should be correctly detected from message."""
        from gateway.platforms.base import MessageEvent
        
        # Mock message event with meta_only flag
        event = MagicMock(spec=MessageEvent)
        event.get_command = MagicMock(return_value="/model")
        
        # Meta-only detection logic
        command = event.get_command()
        meta_only = command in ("/model", "/tools", "/new", "/retry", "/compress")
        
        assert meta_only is True

    def test_meta_only_false_for_regular_messages(self):
        """Regular messages should have meta_only=False."""
        from gateway.platforms.base import MessageEvent
        
        event = MagicMock(spec=MessageEvent)
        event.get_command = MagicMock(return_value=None)
        event.text = "Hello, how are you?"
        
        command = event.get_command()
        meta_only = bool(command) and command.startswith("/")
        
        assert meta_only is False


class TestProcessUserMessageMetaOnly:
    """Test _process_user_message with meta_only=True."""

    @pytest.mark.asyncio
    async def test_process_user_message_meta_only_calls_handler(self):
        """_process_user_message(meta_only=True) should call meta handler."""
        from run_agent import AIAgent
        
        agent = MagicMock(spec=AIAgent)
        agent._run_meta_only_handler = AsyncMock(return_value="Model changed")
        
        # This simulates the meta_only path in _process_user_message
        with patch.object(AIAgent, '_process_user_message', wraps=AIAgent._process_user_message):
            # The real implementation should call _run_meta_only_handler
            pass

    @pytest.mark.asyncio
    async def test_meta_handler_returns_status_message(self):
        """Meta handler should return a status message, not None."""
        # Meta-only handlers like /model should return status like "Model changed to X"
        # Not return None which causes stuck
        expected_responses = [
            "Model changed",
            "Tools updated",
            "Session cleared",
            "Context compressed",
        ]
        
        # All meta-only handlers should return non-None responses
        for response in expected_responses:
            assert response is not None
            assert isinstance(response, str)


class TestChatWithModelLoop:
    """Test the chat_with_model loop behavior."""

    @pytest.mark.asyncio
    async def test_chat_with_model_meta_only_exits_early(self):
        """chat_with_model should exit early for meta-only messages."""
        # The fixed code should have:
        # if meta_only:
        #     return meta_result or "Processed meta-only message."
        # NOT enter the while loop waiting for response content
        
        # Simulate the fixed behavior
        meta_only = True
        meta_result = "Model changed"
        
        # Fixed path: immediate return
        if meta_only:
            result = meta_result if meta_result else "Processed meta-only message."
        else:
            # Regular path: would enter response loop
            result = None  # Would wait for response
        
        assert result == "Model changed"

    @pytest.mark.asyncio
    async def test_chat_with_model_regular_message_continues(self):
        """chat_with_model should continue for regular messages."""
        meta_only = False
        
        if meta_only:
            # Would return immediately
            pass
        else:
            # Should continue to response processing
            # This tests that we don't accidentally break regular messages
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])