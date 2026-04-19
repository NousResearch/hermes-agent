"""Test interrupt propagation from parent to child agents.

Reproduces the CLI scenario: user sends a message while delegate_task is
running, main thread calls parent.interrupt(), child should stop.
"""

import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from tools.interrupt import set_interrupt, is_interrupted, _interrupt_event


class TestInterruptPropagationToChild(unittest.TestCase):
    """Verify interrupt propagates from parent to child agent."""

    def setUp(self):
        set_interrupt(False)

    def tearDown(self):
        set_interrupt(False)

    def _make_bare_agent(self):
        """Create a bare AIAgent via __new__ with all interrupt-related attrs."""
        from run_agent import AIAgent
        agent = AIAgent.__new__(AIAgent)
        agent._interrupt_requested = False
        agent._interrupt_message = None
        agent._execution_thread_id = None
        agent._interrupt_thread_signal_pending = False
        agent._active_children = []
        agent._active_children_lock = threading.Lock()
        agent.quiet_mode = True
        return agent

    def test_parent_interrupt_sets_child_flag(self):
        """When parent.interrupt() is called, child._interrupt_requested should be set."""
        parent = self._make_bare_agent()
        child = self._make_bare_agent()

        parent._active_children.append(child)

        parent.interrupt("new user message")

        assert parent._interrupt_requested is True
        assert child._interrupt_requested is True
        assert child._interrupt_message == "new user message"
        assert is_interrupted() is False
        assert parent._interrupt_thread_signal_pending is True

    def test_child_clear_interrupt_at_start_clears_thread(self):
        """child.clear_interrupt() at start of run_conversation clears the
        bound execution thread's interrupt flag.
        """
        child = self._make_bare_agent()
        child._interrupt_requested = True
        child._interrupt_message = "msg"
        child._execution_thread_id = threading.current_thread().ident

        # Interrupt for current thread is set
        set_interrupt(True)
        assert is_interrupted() is True

        # child.clear_interrupt() clears both instance flag and thread flag
        child.clear_interrupt()
        assert child._interrupt_requested is False
        assert is_interrupted() is False

    def test_interrupt_during_child_api_call_detected(self):
        """Interrupt set during _interruptible_api_call is detected within 0.5s."""
        child = self._make_bare_agent()
        child.api_mode = "chat_completions"
        child.log_prefix = ""
        child._client_kwargs = {"api_key": "test", "base_url": "http://localhost:1234"}

        # Mock a slow API call
        mock_client = MagicMock()
        def slow_api_call(**kwargs):
            time.sleep(5)  # Would take 5s normally
            return MagicMock()
        mock_client.chat.completions.create = slow_api_call
        mock_client.close = MagicMock()
        child.client = mock_client

        # Set interrupt after 0.2s from another thread
        def set_interrupt_later():
            time.sleep(0.2)
            child.interrupt("stop!")
        t = threading.Thread(target=set_interrupt_later, daemon=True)
        t.start()

        start = time.monotonic()
        try:
            child._interruptible_api_call({"model": "test", "messages": []})
            self.fail("Should have raised InterruptedError")
        except InterruptedError:
            elapsed = time.monotonic() - start
            # Should detect within ~0.5s (0.2s delay + 0.3s poll interval)
            assert elapsed < 1.0, f"Took {elapsed:.2f}s to detect interrupt (expected < 1.0s)"
        finally:
            t.join(timeout=2)
            set_interrupt(False)

    def test_concurrent_interrupt_propagation(self):
        """Simulates exact CLI flow: parent runs delegate in thread, main thread interrupts."""
        parent = self._make_bare_agent()
        child = self._make_bare_agent()

        # Register child (simulating what _run_single_child does)
        parent._active_children.append(child)

        # Simulate child running (checking flag in a loop)
        child_detected = threading.Event()
        def simulate_child_loop():
            while not child._interrupt_requested:
                time.sleep(0.05)
            child_detected.set()

        child_thread = threading.Thread(target=simulate_child_loop, daemon=True)
        child_thread.start()

        # Small delay, then interrupt from "main thread"
        time.sleep(0.1)
        parent.interrupt("user typed something new")

        # Child should detect within 200ms
        detected = child_detected.wait(timeout=1.0)
        assert detected, "Child never detected the interrupt!"
        child_thread.join(timeout=1)
        set_interrupt(False)

    def test_prestart_interrupt_binds_to_execution_thread(self):
        """An interrupt that arrives before startup should bind to the agent thread."""
        agent = self._make_bare_agent()
        barrier = threading.Barrier(2)
        result = {}

        agent.interrupt("stop before start")
        assert agent._interrupt_requested is True
        assert agent._interrupt_thread_signal_pending is True
        assert is_interrupted() is False

        def run_thread():
            from tools.interrupt import set_interrupt as _set_interrupt_for_test

            agent._execution_thread_id = threading.current_thread().ident
            _set_interrupt_for_test(False, agent._execution_thread_id)
            if agent._interrupt_requested:
                _set_interrupt_for_test(True, agent._execution_thread_id)
                agent._interrupt_thread_signal_pending = False
            barrier.wait(timeout=5)
            result["thread_interrupted"] = is_interrupted()

        t = threading.Thread(target=run_thread)
        t.start()
        barrier.wait(timeout=5)
        t.join(timeout=2)

        assert result["thread_interrupted"] is True
        assert agent._interrupt_thread_signal_pending is False


class TestPerThreadInterruptIsolation(unittest.TestCase):
    """Verify that interrupting one agent does NOT affect another agent's thread.

    This is the core fix for the gateway cross-session interrupt leak:
    multiple agents run in separate threads within the same process, and
    interrupting agent A must not kill agent B's running tools.
    """

    def setUp(self):
        set_interrupt(False)

    def tearDown(self):
        set_interrupt(False)

    def test_interrupt_only_affects_target_thread(self):
        """set_interrupt(True, tid) only makes is_interrupted() True on that thread."""
        results = {}
        barrier = threading.Barrier(2)

        def thread_a():
            """Agent A's execution thread — will be interrupted."""
            tid = threading.current_thread().ident
            results["a_tid"] = tid
            barrier.wait(timeout=5)  # sync with thread B
            time.sleep(0.2)  # let the interrupt arrive
            results["a_interrupted"] = is_interrupted()

        def thread_b():
            """Agent B's execution thread — should NOT be affected."""
            tid = threading.current_thread().ident
            results["b_tid"] = tid
            barrier.wait(timeout=5)  # sync with thread A
            time.sleep(0.2)
            results["b_interrupted"] = is_interrupted()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()

        # Wait for both threads to register their TIDs
        time.sleep(0.05)
        while "a_tid" not in results or "b_tid" not in results:
            time.sleep(0.01)

        # Interrupt ONLY thread A (simulates gateway interrupting agent A)
        set_interrupt(True, results["a_tid"])

        ta.join(timeout=3)
        tb.join(timeout=3)

        assert results["a_interrupted"] is True, "Thread A should see the interrupt"
        assert results["b_interrupted"] is False, "Thread B must NOT see thread A's interrupt"

    def test_clear_interrupt_only_clears_target_thread(self):
        """Clearing one thread's interrupt doesn't clear another's."""
        tid_a = 99990001
        tid_b = 99990002
        set_interrupt(True, tid_a)
        set_interrupt(True, tid_b)

        # Clear only A
        set_interrupt(False, tid_a)

        # Simulate checking from thread B's perspective
        from tools.interrupt import _interrupted_threads, _lock
        with _lock:
            assert tid_a not in _interrupted_threads
            assert tid_b in _interrupted_threads

        # Cleanup
        set_interrupt(False, tid_b)


if __name__ == "__main__":
    unittest.main()
