"""Test for the concurrent-tool-batch timeout grace window.

On batch timeout, still-running workers get a per-thread interrupt signal
(the same primitive terminal/execute_code check) but were then abandoned
immediately — unlike the sibling user-interrupt branch, which gives workers
a bounded window to notice the signal and exit before moving on. A
cooperative tool that would have stopped within a second or two was instead
unconditionally left running as a detached, unsupervised daemon thread on
every timeout.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)


def _make_agent(monkeypatch):
    """Minimal AIAgent-like stub, mirroring test_start_order_gate.py."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "")
    import run_agent as _ra

    class _Stub:
        _interrupt_requested = False
        _interrupt_message = None
        _execution_thread_id = threading.current_thread().ident
        _interrupt_thread_signal_pending = False
        log_prefix = ""
        quiet_mode = True
        verbose_logging = False
        log_prefix_chars = 200
        _checkpoint_mgr = MagicMock(enabled=False)
        tool_progress_callback = None
        tool_start_callback = None
        tool_complete_callback = None
        tool_progress_mode = "off"
        _todo_store = MagicMock()
        _session_db = None
        valid_tool_names = set()
        _turns_since_memory = 0
        _iters_since_skill = 0
        _current_tool = None
        _last_activity = 0
        _print_fn = print
        session_id = ""
        _current_turn_id = ""
        _current_api_request_id = ""
        _active_children: list = []

        def __init__(self):
            self._tool_worker_threads: set = set()
            self._tool_worker_threads_lock = threading.Lock()
            self._active_children_lock = threading.Lock()

        def _touch_activity(self, desc):
            self._last_activity = time.time()

        def _vprint(self, msg, force=False):
            pass

        def _safe_print(self, msg):
            pass

        def _should_emit_quiet_tool_messages(self):
            return False

        def _should_start_quiet_spinner(self):
            return False

        def _has_stream_consumers(self):
            return False

        def _tool_result_content_for_active_model(self, name, result):
            return result

        def _record_file_mutation_result(self, *a, **kw):
            pass

    stub = _Stub()
    stub._subdirectory_hints = MagicMock()
    stub._subdirectory_hints.check_tool_call = lambda *a, **kw: None
    stub._flush_messages_to_session_db = lambda *a, **kw: None
    stub._append_guardrail_observation = lambda name, result, *a, **kw: result
    stub._execute_tool_calls_concurrent = (
        _ra.AIAgent._execute_tool_calls_concurrent.__get__(stub)
    )
    stub.interrupt = _ra.AIAgent.interrupt.__get__(stub)
    stub.clear_interrupt = _ra.AIAgent.clear_interrupt.__get__(stub)
    stub._apply_pending_steer_to_tool_results = lambda *a, **kw: None
    return stub


class _FakeToolCall:
    def __init__(self, name, call_id):
        self.function = MagicMock(name=name, arguments="{}")
        self.function.name = name
        self.id = call_id


class _FakeAssistantMsg:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


def test_timeout_gives_cooperative_tool_a_chance_to_stop(monkeypatch):
    """A worker still running past the batch deadline must get a bounded
    grace window to notice the per-thread interrupt signal and exit —
    mirroring the sibling user-interrupt branch's `wait(not_done,
    timeout=3.0)` — instead of being abandoned the instant the signal is
    sent."""
    import agent.tool_executor as te
    from tools.interrupt import is_interrupted

    agent = _make_agent(monkeypatch)
    monkeypatch.setattr(te, "_resolve_concurrent_tool_timeout", lambda: 0.2)

    noticed_interrupt = threading.Event()

    def _before_call(name, args):
        # Simulate a cooperative long-running tool (terminal, execute_code):
        # poll the same per-thread interrupt primitive those tools check,
        # and exit promptly once it fires.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if is_interrupted():
                noticed_interrupt.set()
                break
            time.sleep(0.05)
        return MagicMock(allows_execution=True)

    agent._tool_guardrails = MagicMock()
    agent._tool_guardrails.before_call = _before_call
    agent._invoke_tool = MagicMock(side_effect=lambda *a, **kw: '{"ok": true}')

    msg = _FakeAssistantMsg([_FakeToolCall("slow_tool", "tc_1")])
    messages: list = []
    agent._execute_tool_calls_concurrent(msg, messages, "task")

    assert noticed_interrupt.is_set(), (
        "batch was abandoned without giving the timed-out worker a chance "
        "to see the interrupt signal and stop cooperatively"
    )
