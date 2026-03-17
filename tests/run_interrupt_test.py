#!/usr/bin/env python3
"""Run a real interrupt test with actual AIAgent + delegate child.

Not a pytest test — runs directly as a script for live testing.
"""

import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
from run_agent import AIAgent, IterationBudget
from tools.delegate_tool import _run_single_child
from tools.interrupt import set_interrupt, is_interrupted

def main() -> int:
    set_interrupt(False)

    # Create parent agent (minimal)
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
    parent._client_kwargs = {"api_key": "test", "base_url": "http://localhost:1"}

    child_started = threading.Event()
    result_holder = [None]

    def run_delegate():
        with patch("run_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()

            def slow_create(**kwargs):
                time.sleep(3)
                resp = MagicMock()
                resp.choices = [MagicMock()]
                resp.choices[0].message.content = "Done"
                resp.choices[0].message.tool_calls = None
                resp.choices[0].message.refusal = None
                resp.choices[0].finish_reason = "stop"
                resp.usage.prompt_tokens = 100
                resp.usage.completion_tokens = 10
                resp.usage.total_tokens = 110
                resp.usage.prompt_tokens_details = None
                return resp

            mock_client.chat.completions.create = slow_create
            mock_client.close = MagicMock()
            MockOpenAI.return_value = mock_client

            original_init = AIAgent.__init__

            def patched_init(self_agent, *a, **kw):
                original_init(self_agent, *a, **kw)
                child_started.set()

            with patch.object(AIAgent, "__init__", patched_init):
                try:
                    result = _run_single_child(
                        task_index=0,
                        goal="Test slow task",
                        context=None,
                        toolsets=["terminal"],
                        model="test/model",
                        max_iterations=5,
                        parent_agent=parent,
                        task_count=1,
                        override_provider="test",
                        override_base_url="http://localhost:1",
                        override_api_key="test",
                        override_api_mode="chat_completions",
                    )
                    result_holder[0] = result
                except Exception as e:
                    print(f"ERROR in delegate: {e}")
                    import traceback
                    traceback.print_exc()

    print("Starting agent thread...")
    agent_thread = threading.Thread(target=run_delegate, daemon=True)
    agent_thread.start()

    started = child_started.wait(timeout=10)
    if not started:
        print("ERROR: Child never started")
        set_interrupt(False)
        return 1

    time.sleep(0.5)

    print(f"Active children: {len(parent._active_children)}")
    for i, c in enumerate(parent._active_children):
        print(f"  Child {i}: _interrupt_requested={c._interrupt_requested}")

    t0 = time.monotonic()
    parent.interrupt("User typed a new message")
    print("Called parent.interrupt()")

    for i, c in enumerate(parent._active_children):
        print(f"  Child {i} after interrupt: _interrupt_requested={c._interrupt_requested}")
    print(f"Global is_interrupted: {is_interrupted()}")

    agent_thread.join(timeout=10)
    elapsed = time.monotonic() - t0
    print(f"Agent thread finished in {elapsed:.2f}s")

    result = result_holder[0]
    if result:
        print(f"Status: {result['status']}")
        print(f"Duration: {result['duration_seconds']}s")
        if elapsed < 2.0:
            print("✅ PASS: Interrupt detected quickly!")
        else:
            print(f"❌ FAIL: Took {elapsed:.2f}s — interrupt was too slow or not detected")
    else:
        print("❌ FAIL: No result!")

    set_interrupt(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
