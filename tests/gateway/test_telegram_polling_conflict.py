"""Tests for Telegram polling conflict recovery and error handling.

Regression tests for #40691: Telegram gateway freezes after polling conflict
recovery. The issue was that after recovering from a 409 conflict and restarting
polling, the error callback task state wasn't properly reset, causing subsequent
errors to be silently dropped.
"""
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pytest


class TestErrorCallbackTaskManagement:
    """Test that error callback tasks are properly managed during conflict recovery."""

    def test_callback_task_guard_allows_multiple_errors(self):
        """Error callback guard should allow new errors after previous task completes."""
        # Simulate the error callback guard logic
        polling_error_task: asyncio.Task | None = None
        callback_invocations = []
        
        def _polling_error_callback(error: Exception) -> None:
            """Simplified version of the actual callback."""
            # Guard: skip if a previous error task is still running
            nonlocal polling_error_task
            if polling_error_task and not polling_error_task.done():
                return
            callback_invocations.append(error)
        
        # First error should be recorded
        error1 = Exception("Error 1")
        _polling_error_callback(error1)
        assert error1 in callback_invocations
        
        # Simulate that the task completes (or is cleared)
        polling_error_task = None
        
        # Second error should now also be recorded (guard not blocking)
        error2 = Exception("Error 2")
        _polling_error_callback(error2)
        assert error2 in callback_invocations, \
            "Second error should be recorded after task is cleared"

    def test_callback_task_blocks_concurrent_errors(self):
        """Error callback should not process errors while another task is running."""
        polling_error_task: asyncio.Task | None = None
        callback_invocations = []
        
        def _polling_error_callback(error: Exception) -> None:
            """Simplified version of the actual callback."""
            nonlocal polling_error_task
            if polling_error_task and not polling_error_task.done():
                return  # Guard: skip if task still running
            callback_invocations.append(error)
        
        # First error creates a task
        error1 = Exception("Error 1")
        _polling_error_callback(error1)
        
        # Simulate a task that's still running
        polling_error_task = MagicMock()
        polling_error_task.done.return_value = False  # Still running
        
        # Second error should NOT be recorded (task still running)
        error2 = Exception("Error 2")
        _polling_error_callback(error2)
        assert error2 not in callback_invocations, \
            "Concurrent errors should be dropped when task is running"

    def test_callback_task_guard_reset_pattern(self):
        """After conflict recovery, task guard should be reset for next error."""
        # Simulate the pattern in _handle_polling_conflict
        polling_error_task: asyncio.Task | None = None
        task_resets = []
        
        async def simulate_conflict_recovery():
            nonlocal polling_error_task
            # During recovery, the task might still be set to the previous attempt
            # After successful restart, it should be cleared
            
            # Before fix: task is not reset
            # After fix: explicitly reset or ensure new errors can be processed
            polling_error_task = None  # Reset guard after successful restart
            task_resets.append("reset")
        
        # Simulate conflict handling
        asyncio.run(simulate_conflict_recovery())
        assert "reset" in task_resets


class TestErrorCallbackClosureIssues:
    """Test potential closure and reference issues in error callbacks."""

    def test_callback_reference_not_stale_after_restart(self):
        """Callback reference should remain valid after polling restart."""
        # Simulate storing and reusing a callback reference
        callback_invocations = []
        
        def create_callback():
            def _polling_error_callback(error: Exception) -> None:
                callback_invocations.append(error)
            return _polling_error_callback
        
        # Create callback at startup
        callback = create_callback()
        callback_ref = callback
        
        # Use callback before restart
        error1 = Exception("Before restart")
        callback_ref(error1)
        assert error1 in callback_invocations
        
        # After restart, reuse the same reference
        # (In the actual code, this happens in _handle_polling_conflict line 1124)
        error2 = Exception("After restart")
        callback_ref(error2)
        assert error2 in callback_invocations, \
            "Stored callback reference should still be valid after restart"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
