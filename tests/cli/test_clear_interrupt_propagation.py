"""Test that clear_interrupt propagates to active children agents.

This test verifies the fix for issue #63463 where slash commands like /new and /clear
failed to interrupt subagent waves because clear_interrupt() did not propagate to children.
"""

import threading
import unittest
from unittest.mock import MagicMock


class TestClearInterruptPropagation(unittest.TestCase):
    """Verify that clear_interrupt propagates to active children."""

    def test_clear_interrupt_propagates_to_children(self):
        """clear_interrupt should call clear_interrupt on all active children."""
        from run_agent import AIAgent

        # Create a parent agent
        parent = AIAgent.__new__(AIAgent)
        parent._interrupt_requested = True
        parent._interrupt_message = "test interrupt"
        parent._interrupt_thread_signal_pending = True
        parent._execution_thread_id = None
        parent._active_children = []
        parent._active_children_lock = threading.Lock()

        # Create mock children
        child1 = MagicMock()
        child2 = MagicMock()

        # Set child interrupt state to True
        child1._interrupt_requested = True
        child1._interrupt_message = "child1 interrupt"
        child2._interrupt_requested = True
        child2._interrupt_message = "child2 interrupt"

        # Add children to parent
        parent._active_children.extend([child1, child2])

        # Call clear_interrupt on parent
        parent.clear_interrupt()

        # Verify parent state is cleared
        assert parent._interrupt_requested is False, "Parent _interrupt_requested should be False"
        assert parent._interrupt_message is None, "Parent _interrupt_message should be None"
        assert parent._interrupt_thread_signal_pending is False, "Parent _interrupt_thread_signal_pending should be False"

        # Verify children clear_interrupt was called
        child1.clear_interrupt.assert_called_once()
        child2.clear_interrupt.assert_called_once()

    def test_clear_interrupt_with_no_children(self):
        """clear_interrupt should work correctly when there are no children."""
        from run_agent import AIAgent

        parent = AIAgent.__new__(AIAgent)
        parent._interrupt_requested = True
        parent._interrupt_message = "test"
        parent._execution_thread_id = None
        parent._active_children = []
        parent._active_children_lock = threading.Lock()

        # Should not raise
        parent.clear_interrupt()

        assert parent._interrupt_requested is False

    def test_clear_interrupt_handles_child_exceptions(self):
        """clear_interrupt should not raise if a child raises an exception."""
        from run_agent import AIAgent

        parent = AIAgent.__new__(AIAgent)
        parent._interrupt_requested = True
        parent._execution_thread_id = None
        parent._active_children = []
        parent._active_children_lock = threading.Lock()

        # Create a child that raises an exception
        child1 = MagicMock()
        child1.clear_interrupt.side_effect = RuntimeError("Child error")

        child2 = MagicMock()
        child2.clear_interrupt.return_value = None

        parent._active_children.extend([child1, child2])

        # Should not raise
        parent.clear_interrupt()

        # Both children should have been attempted
        child1.clear_interrupt.assert_called_once()
        child2.clear_interrupt.assert_called_once()

        # Parent state should still be cleared
        assert parent._interrupt_requested is False


if __name__ == "__main__":
    unittest.main()