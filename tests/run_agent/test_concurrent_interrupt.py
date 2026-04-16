"""Tests for interrupt handling in concurrent tool execution."""

import concurrent.futures
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)


def _make_agent(monkeypatch):
    """Create a minimal AIAgent-like object with just the methods under test."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "")
    # Avoid full AIAgent init — just import the class and build a stub
    import run_agent as _ra

    class _Stub:
        _interrupt_requested = False
        log_prefix = ""
        quiet_mode = True
        verbose_logging = False
        log_prefix_chars = 200
        _checkpoint_mgr = MagicMock(enabled=False)
        _subdirectory_hints = MagicMock()
        tool_progress_callback = None
        tool_start_callback = None
        tool_complete_callback = None
        _todo_store = MagicMock()
        _session_db = None
        valid_tool_names = set()
        _turns_since_memory = 0
        _iters_since_skill = 0
        _current_tool = None
        _last_activity = 0
        _print_fn = print

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

    stub = _Stub()
    # Bind the real methods
    stub._execute_tool_calls_concurrent = _ra.AIAgent._execute_tool_calls_concurrent.__get__(stub)
    stub._invoke_tool = MagicMock(side_effect=lambda *a, **kw: '{"ok": true}')
    return stub


class _FakeToolCall:
    def __init__(self, name, args="{}", call_id="tc_1"):
        self.function = MagicMock(name=name, arguments=args)
        self.function.name = name
        self.id = call_id


class _FakeAssistantMsg:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


def test_concurrent_interrupt_cancels_pending(monkeypatch):
    """When _interrupt_requested is set during concurrent execution,
    the wait loop should exit early and cancelled tools get interrupt messages."""
    agent = _make_agent(monkeypatch)

    # Create a tool that blocks until interrupted
    barrier = threading.Event()

    original_invoke = agent._invoke_tool

    def slow_tool(name, args, task_id, call_id=None):
        if name == "slow_one":
            # Block until the test sets the interrupt
            barrier.wait(timeout=10)
            return '{"slow": true}'
        return '{"fast": true}'

    agent._invoke_tool = MagicMock(side_effect=slow_tool)

    tc1 = _FakeToolCall("fast_one", call_id="tc_fast")
    tc2 = _FakeToolCall("slow_one", call_id="tc_slow")
    msg = _FakeAssistantMsg([tc1, tc2])
    messages = []

    def _set_interrupt_after_delay():
        time.sleep(0.3)
        agent._interrupt_requested = True
        barrier.set()  # unblock the slow tool

    t = threading.Thread(target=_set_interrupt_after_delay)
    t.start()

    agent._execute_tool_calls_concurrent(msg, messages, "test_task")
    t.join()

    # Both tools should have results in messages
    assert len(messages) == 2
    # The interrupt was detected
    assert agent._interrupt_requested is True


def test_concurrent_preflight_interrupt_skips_all(monkeypatch):
    """When _interrupt_requested is already set before concurrent execution,
    all tools are skipped with cancellation messages."""
    agent = _make_agent(monkeypatch)
    agent._interrupt_requested = True

    tc1 = _FakeToolCall("tool_a", call_id="tc_a")
    tc2 = _FakeToolCall("tool_b", call_id="tc_b")
    msg = _FakeAssistantMsg([tc1, tc2])
    messages = []

    agent._execute_tool_calls_concurrent(msg, messages, "test_task")

    assert len(messages) == 2
    assert "skipped due to user interrupt" in messages[0]["content"]
    assert "skipped due to user interrupt" in messages[1]["content"]
    # _invoke_tool should never have been called
    agent._invoke_tool.assert_not_called()
