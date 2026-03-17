"""End-to-end test simulating CLI interrupt during subagent execution.

Reproduces the exact scenario:
1. Parent agent calls delegate_task
2. Child agent is running (simulated with a slow tool)
3. User "types a message" (simulated by calling parent.interrupt from another thread)
4. Child should detect the interrupt and stop

This tests the COMPLETE path including _run_single_child, _active_children
registration, interrupt propagation, and child detection.
"""

import json
import os
import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from tools.interrupt import set_interrupt, is_interrupted


class TestCLISubagentInterrupt(unittest.TestCase):
    """Simulate exact CLI scenario."""

    def setUp(self):
        set_interrupt(False)

    def tearDown(self):
        set_interrupt(False)

    def test_full_delegate_interrupt_flow(self):
        """Full integration: parent runs delegate_task, main thread interrupts."""
        from run_agent import AIAgent

        interrupt_detected = threading.Event()
        child_started = threading.Event()
        child_api_call_count = 0

        # Create a real-enough parent agent
        parent = AIAgent.__new__(AIAgent)
        parent._interrupt_requested = False
        parent._interrupt_message = None
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        parent.quiet_mode = True
        parent.model = "test/model"
        parent.base_url = "http://localhost:1"
        parent.api_key = "test"
        parent.provider = "test"
        parent.api_mode = "chat_completions"
        parent.platform = "cli"
        parent.enabled_toolsets = ["terminal", "file"]
        parent.providers_allowed = None
        parent.providers_ignored = None
        parent.providers_order = None
        parent.provider_sort = None
        parent.max_tokens = None
        parent.reasoning_config = None
        parent.prefill_messages = None
        parent._session_db = None
        parent._delegate_depth = 0
        parent._delegate_spinner = None
        parent.tool_progress_callback = None

        # We'll track what happens with _active_children
        original_children = parent._active_children

        # Mock the child's run_conversation to simulate a slow operation
        # that checks _interrupt_requested like the real one does
        def mock_child_run_conversation(user_message, **kwargs):
            child_started.set()
            # Find the child in parent._active_children
            child = parent._active_children[-1] if parent._active_children else None
            
            # Simulate the agent loop: poll _interrupt_requested like run_conversation does
            for i in range(100):  # Up to 10 seconds (100 * 0.1s)
                if child and child._interrupt_requested:
                    interrupt_detected.set()
                    return {
                        "final_response": "Interrupted!",
                        "messages": [],
                        "api_calls": 1,
                        "completed": False,
                        "interrupted": True,
                        "interrupt_message": child._interrupt_message,
                    }
                time.sleep(0.1)
            
            return {
                "final_response": "Finished without interrupt",
                "messages": [],
                "api_calls": 5,
                "completed": True,
                "interrupted": False,
            }

        # Patch AIAgent to use our mock
        from tools.delegate_tool import _run_single_child
        from run_agent import IterationBudget

        parent.iteration_budget = IterationBudget(max_total=100)

        # Run delegate in a thread (simulates agent_thread)
        delegate_result = [None]
        delegate_error = [None]

        def run_delegate():
            try:
                with patch('run_agent.AIAgent') as MockAgent:
                    mock_instance = MagicMock()
                    mock_instance._interrupt_requested = False
                    mock_instance._interrupt_message = None
                    mock_instance._active_children = []
                    mock_instance._active_children_lock = threading.Lock()
                    mock_instance.quiet_mode = True
                    mock_instance.run_conversation = mock_child_run_conversation
                    mock_instance.interrupt = lambda msg=None: setattr(mock_instance, '_interrupt_requested', True) or setattr(mock_instance, '_interrupt_message', msg)
                    mock_instance.tools = []
                    MockAgent.return_value = mock_instance

                    # Register child manually (normally done by _build_child_agent)
                    parent._active_children.append(mock_instance)

                    result = _run_single_child(
                        task_index=0,
                        goal="Do something slow",
                        child=mock_instance,
                        parent_agent=parent,
                    )
                    delegate_result[0] = result
            except Exception as e:
                delegate_error[0] = e

        agent_thread = threading.Thread(target=run_delegate, daemon=True)
        agent_thread.start()

        # Wait for child to start
        assert child_started.wait(timeout=5), "Child never started!"

        # Now simulate user interrupt (from main/process thread)
        time.sleep(0.2)  # Give child a moment to be in its loop
        
        print(f"Parent has {len(parent._active_children)} active children")
        assert len(parent._active_children) >= 1, f"Expected child in _active_children, got {len(parent._active_children)}"

        # This is what the CLI does:
        parent.interrupt("Hey stop that")
        
        print(f"Parent._interrupt_requested: {parent._interrupt_requested}")
        for i, child in enumerate(parent._active_children):
            print(f"Child {i}._interrupt_requested: {child._interrupt_requested}")

        # Wait for child to detect interrupt
        detected = interrupt_detected.wait(timeout=3.0)
        
        # Wait for delegate to finish
        agent_thread.join(timeout=5)

        if delegate_error[0]:
            raise delegate_error[0]

        assert detected, "Child never detected the interrupt!"
        result = delegate_result[0]
        assert result is not None, "Delegate returned no result"
        assert result["status"] == "interrupted", f"Expected 'interrupted', got '{result['status']}'"
        print(f"✓ Interrupt detected! Result: {result}")


if __name__ == "__main__":
    unittest.main()
