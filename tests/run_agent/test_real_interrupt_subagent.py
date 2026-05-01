"""Test real interrupt propagation through delegate_task with actual AIAgent.

This uses a real AIAgent with mocked HTTP responses to test the complete
interrupt flow through _run_single_child → child.run_conversation().
"""

import json
import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from tools.interrupt import set_interrupt, is_interrupted


def _make_slow_api_response(delay=5.0):
    """Create a mock that simulates a slow API response (like a real LLM call)."""
    def slow_create(**kwargs):
        # Simulate a slow API call
        time.sleep(delay)
        # Return a simple text response (no tool calls)
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message = MagicMock()
        resp.choices[0].message.content = "Done"
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.refusal = None
        resp.choices[0].finish_reason = "stop"
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 10
        resp.usage.total_tokens = 110
        resp.usage.prompt_tokens_details = None
        return resp
    return slow_create


class TestRealSubagentInterrupt(unittest.TestCase):
    """Test interrupt with real AIAgent child through delegate_tool."""

    def setUp(self):
        set_interrupt(False)
        os.environ.setdefault("OPENAI_API_KEY", "test-key")

    def tearDown(self):
        set_interrupt(False)

    def test_interrupt_child_during_api_call(self):
        """Real AIAgent child interrupted while making API call."""
        from run_agent import AIAgent, IterationBudget

        # Create a real parent agent (just enough to be a parent)
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
        parent.iteration_budget = IterationBudget(max_total=100)
        parent._client_kwargs = {"api_key": "***", "base_url": "http://localhost:1"}
        parent._execution_thread_id = None

        from tools.delegate_tool import _run_single_child

        child_started = threading.Event()
        result_holder = [None]
        error_holder = [None]

        def run_delegate():
            try:
                # Patch the OpenAI client creation inside AIAgent.__init__
                with patch('run_agent.OpenAI') as MockOpenAI:
                    mock_client = MagicMock()
                    # API call takes 5 seconds — should be interrupted before that
                    mock_client.chat.completions.create = _make_slow_api_response(delay=5.0)
                    mock_client.close = MagicMock()
                    MockOpenAI.return_value = mock_client

                    # Patch the instance method so it skips prompt assembly
                    with patch.object(AIAgent, '_build_system_prompt', return_value="You are a test agent"):
                        # Signal when child starts
                        original_run = AIAgent.run_conversation

                        def patched_run(self_agent, *args, **kwargs):
                            child_started.set()
                            return original_run(self_agent, *args, **kwargs)

                        with patch.object(AIAgent, 'run_conversation', patched_run):
                            # Build a real child agent (AIAgent is NOT patched here,
                            # only run_conversation and _build_system_prompt are)
                            child = AIAgent(
                                base_url="http://localhost:1",
                                api_key="test-key",
                                model="test/model",
                                provider="test",
                                api_mode="chat_completions",
                                max_iterations=5,
                                enabled_toolsets=["terminal"],
                                quiet_mode=True,
                                skip_context_files=True,
                                skip_memory=True,
                                platform="cli",
                            )
                            child._delegate_depth = 1
                            parent._active_children.append(child)
                            result = _run_single_child(
                                task_index=0,
                                goal="Test task",
                                child=child,
                                parent_agent=parent,
                            )
                            result_holder[0] = result
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_holder[0] = e

        agent_thread = threading.Thread(target=run_delegate, daemon=True)
        agent_thread.start()

        # Wait for child to start run_conversation
        started = child_started.wait(timeout=10)
        if not started:
            agent_thread.join(timeout=1)
            if error_holder[0]:
                raise error_holder[0]
            self.fail("Child never started run_conversation")

        # Give child time to enter main loop and start API call
        time.sleep(0.5)

        # Verify child is registered
        print(f"Active children: {len(parent._active_children)}")
        self.assertGreaterEqual(len(parent._active_children), 1,
                                "Child not registered in _active_children")

        # Interrupt! (simulating what CLI does)
        start = time.monotonic()
        parent.interrupt("User typed a new message")

        # Check propagation
        child = parent._active_children[0] if parent._active_children else None
        if child:
            print(f"Child._interrupt_requested after parent.interrupt(): {child._interrupt_requested}")
            self.assertTrue(child._interrupt_requested,
                           "Interrupt did not propagate to child!")

        # Wait for delegate to finish (should be fast since interrupted)
        agent_thread.join(timeout=5)
        elapsed = time.monotonic() - start

        if error_holder[0]:
            raise error_holder[0]

        result = result_holder[0]
        self.assertIsNotNone(result, "Delegate returned no result")
        print(f"Result status: {result['status']}, elapsed: {elapsed:.2f}s")
        print(f"Full result: {result}")

        # The child should have been interrupted, not completed the full 5s API call
        self.assertLess(elapsed, 3.0,
                       f"Took {elapsed:.2f}s — interrupt was not detected quickly enough")
        self.assertEqual(result["status"], "interrupted",
                        f"Expected 'interrupted', got '{result['status']}'")


if __name__ == "__main__":
    unittest.main()
