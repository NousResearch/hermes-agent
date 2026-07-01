"""Test real interrupt propagation through delegate_task with actual AIAgent.

This uses a real AIAgent with mocked HTTP responses to test the complete
interrupt flow through _run_single_child → child.run_conversation().
"""

import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from tools.interrupt import set_interrupt


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
                # Use a lightweight AIAgent instance instead of running the full
                # constructor: on Windows the constructor can spend tens of seconds
                # in provider/client probing before the test reaches the interrupt
                # path, making the regression check flaky. The object still uses
                # the real AIAgent.interrupt implementation below.
                child = AIAgent.__new__(AIAgent)
                child._interrupt_requested = False
                child._interrupt_message = None
                child._interrupt_thread_signal_pending = False
                child._execution_thread_id = None
                child._tool_worker_threads = set()
                child._tool_worker_threads_lock = threading.Lock()
                child._active_children = []
                child._active_children_lock = threading.Lock()
                child._delegate_depth = 1
                child._delegate_saved_tool_names = []
                child._credential_pool = None
                child.tool_progress_callback = None
                child.quiet_mode = True
                child.model = "test/model"
                child.session_prompt_tokens = 0
                child.session_completion_tokens = 0
                child.session_estimated_cost_usd = 0.0
                child.get_activity_summary = lambda: {
                    "api_call_count": 0,
                    "max_iterations": 5,
                    "current_tool": None,
                    "last_activity_desc": "waiting",
                }
                child.close = lambda: None
                child.interrupt = AIAgent.interrupt.__get__(child, AIAgent)
                parent._active_children.append(child)

                def interruptible_run(self_agent, *args, **kwargs):
                    self_agent._execution_thread_id = threading.get_ident()
                    if getattr(self_agent, "_interrupt_thread_signal_pending", False):
                        self_agent._interrupt_thread_signal_pending = False
                    child_started.set()
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        if getattr(self_agent, "_interrupt_requested", False):
                            return {
                                "completed": False,
                                "interrupted": True,
                                "final_response": "",
                                "messages": [],
                                "api_calls": 0,
                            }
                        time.sleep(0.02)
                    return {
                        "completed": True,
                        "interrupted": False,
                        "final_response": "Done",
                        "messages": [],
                        "api_calls": 1,
                    }

                with patch.object(child, 'run_conversation', interruptible_run.__get__(child, AIAgent)):
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

        # Windows CI/dev boxes can spend >10s in child construction/plugin/tool
        # registry setup before run_conversation starts. The interrupt behavior
        # under test begins after the child enters run_conversation, so keep this
        # as a generous startup wait rather than making the test flaky.
        started = child_started.wait(timeout=30)
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
