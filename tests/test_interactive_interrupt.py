#!/usr/bin/env python3
"""Interactive interrupt test that mimics the exact CLI flow.

Starts an agent in a thread with a mock delegate_task that takes a while,
then simulates the user typing a message via _interrupt_queue.

Logs every step to stderr (which isn't affected by redirect_stdout)
so we can see exactly where the interrupt gets lost.
"""

import contextlib
import io
import json
import logging
import queue
import sys
import threading
import time
import os

# Force stderr logging so redirect_stdout doesn't swallow it
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr,
                    format="%(asctime)s [%(threadName)s] %(message)s")
log = logging.getLogger("interrupt_test")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
from run_agent import AIAgent, IterationBudget
from tools.interrupt import set_interrupt, is_interrupted

def make_slow_response(delay=2.0):
    """API response that takes a while."""
    def create(**kwargs):
        log.info(f"   🌐 Mock API call starting (will take {delay}s)...")
        time.sleep(delay)
        log.info(f"   🌐 Mock API call completed")
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "Done with the task"
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.refusal = None
        resp.choices[0].finish_reason = "stop"
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 10
        resp.usage.total_tokens = 110
        resp.usage.prompt_tokens_details = None
        return resp
    return create


def main() -> int:
    set_interrupt(False)

    # ─── Create parent agent ───
    parent = AIAgent.__new__(AIAgent)
    parent._interrupt_requested = False
    parent._interrupt_message = None
    parent._active_children = []
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
    parent._client_kwargs = {"api_key": "test", "base_url": "http://localhost:1"}

    # Monkey-patch parent.interrupt to log
    _original_interrupt = AIAgent.interrupt

    def logged_interrupt(self, message=None):
        log.info(f"🔴 parent.interrupt() called with: {message!r}")
        log.info(f"   _active_children count: {len(self._active_children)}")
        _original_interrupt(self, message)
        log.info(f"   After interrupt: _interrupt_requested={self._interrupt_requested}")
        for i, child in enumerate(self._active_children):
            log.info(f"   Child {i}._interrupt_requested={child._interrupt_requested}")

    parent.interrupt = lambda msg=None: logged_interrupt(parent, msg)

    # ─── Simulate the exact CLI flow ───
    interrupt_queue = queue.Queue()
    child_running = threading.Event()
    agent_result = [None]

    def agent_thread_func():
        """Simulates the agent_thread in cli.py's chat() method."""
        log.info("🟢 agent_thread starting")

        with patch("run_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create = make_slow_response(delay=3.0)
            mock_client.close = MagicMock()
            MockOpenAI.return_value = mock_client

            from tools.delegate_tool import _run_single_child

            # Signal that child is about to start
            original_init = AIAgent.__init__

            def patched_init(self_agent, *a, **kw):
                log.info("🟡 Child AIAgent.__init__ called")
                original_init(self_agent, *a, **kw)
                child_running.set()
                log.info(
                    f"🟡 Child started, parent._active_children = {len(parent._active_children)}"
                )

            with patch.object(AIAgent, "__init__", patched_init):
                result = _run_single_child(
                    task_index=0,
                    goal="Do a slow thing",
                    context=None,
                    toolsets=["terminal"],
                    model="test/model",
                    max_iterations=3,
                    parent_agent=parent,
                    task_count=1,
                    override_provider="test",
                    override_base_url="http://localhost:1",
                    override_api_key="test",
                    override_api_mode="chat_completions",
                )
                agent_result[0] = result
                log.info(f"🟢 agent_thread finished. Result status: {result.get('status')}")

    # ─── Start agent thread (like chat() does) ───
    agent_thread = threading.Thread(target=agent_thread_func, name="agent_thread", daemon=True)
    agent_thread.start()

    # ─── Wait for child to start ───
    if not child_running.wait(timeout=10):
        print("FAIL: Child never started", file=sys.stderr)
        set_interrupt(False)
        return 1

    # Give child time to enter its main loop and start API call
    time.sleep(1.0)

    # ─── Simulate user typing a message (like handle_enter does) ───
    log.info("📝 Simulating user typing 'Hey stop that'")
    interrupt_queue.put("Hey stop that")

    # ─── Simulate chat() polling loop (like the real chat() method) ───
    log.info("📡 Starting interrupt queue polling (like chat())")
    interrupt_msg = None
    poll_count = 0
    while agent_thread.is_alive():
        try:
            interrupt_msg = interrupt_queue.get(timeout=0.1)
            if interrupt_msg:
                log.info(f"📨 Got interrupt message from queue: {interrupt_msg!r}")
                log.info("   Calling parent.interrupt()...")
                parent.interrupt(interrupt_msg)
                log.info("   parent.interrupt() returned. Breaking poll loop.")
                break
        except queue.Empty:
            poll_count += 1
            if poll_count % 20 == 0:  # Log every 2s
                log.info(f"   Still polling ({poll_count} iterations)...")

    # ─── Wait for agent to finish ───
    log.info("⏳ Waiting for agent_thread to join...")
    t0 = time.monotonic()
    agent_thread.join(timeout=10)
    elapsed = time.monotonic() - t0
    log.info(f"✅ agent_thread joined after {elapsed:.2f}s")

    # ─── Check results ───
    result = agent_result[0]
    if result:
        log.info(f"Result status: {result['status']}")
        log.info(f"Result duration: {result['duration_seconds']}s")
        if result["status"] == "interrupted" and elapsed < 2.0:
            print("✅ PASS: Interrupt worked correctly!", file=sys.stderr)
            set_interrupt(False)
            return 0
        print(f"❌ FAIL: status={result['status']}, elapsed={elapsed:.2f}s", file=sys.stderr)
        set_interrupt(False)
        return 1

    print("❌ FAIL: No result returned", file=sys.stderr)
    set_interrupt(False)
    return 1


if __name__ == "__main__":
    sys.exit(main())
