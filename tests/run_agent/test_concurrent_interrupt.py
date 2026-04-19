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
        _interrupt_message = None
        # Bind to this thread's ident so interrupt() targets a real tid.
        _execution_thread_id = threading.current_thread().ident
        _interrupt_thread_signal_pending = False
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
        # Worker-thread tracking state mirrored from AIAgent.__init__ so the
        # real interrupt() method can fan out to concurrent-tool workers.
        _active_children: list = []

        def __init__(self):
            # Instance-level (not class-level) so each test gets a fresh set.
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

    stub = _Stub()
    # Bind the real methods under test
    stub._execute_tool_calls_concurrent = _ra.AIAgent._execute_tool_calls_concurrent.__get__(stub)
    stub.interrupt = _ra.AIAgent.interrupt.__get__(stub)
    stub.clear_interrupt = _ra.AIAgent.clear_interrupt.__get__(stub)
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


def test_running_concurrent_worker_sees_is_interrupted(monkeypatch):
    """Regression guard for the "interrupt-doesn't-reach-hung-tool" class of
    bug Physikal reported in April 2026.

    Before this fix, `AIAgent.interrupt()` called `_set_interrupt(True,
    _execution_thread_id)` — which only flagged the agent's *main* thread.
    Tools running inside `_execute_tool_calls_concurrent` execute on
    ThreadPoolExecutor worker threads whose tids are NOT the agent's, so
    `is_interrupted()` (which checks the *current* thread's tid) returned
    False inside those tools no matter how many times the gateway called
    `.interrupt()`.  Hung ssh / long curl / big make-build tools would run
    to their own timeout.

    This test runs a fake tool in the concurrent path that polls
    `is_interrupted()` like a real terminal command does, then calls
    `agent.interrupt()` from another thread, and asserts the poll sees True
    within one second.
    """
    from tools.interrupt import is_interrupted

    agent = _make_agent(monkeypatch)

    # Counter plus observation hooks so we can prove the worker saw the flip.
    observed = {"saw_true": False, "poll_count": 0, "worker_tid": None}
    worker_started = threading.Event()

    def polling_tool(name, args, task_id, call_id=None):
        observed["worker_tid"] = threading.current_thread().ident
        worker_started.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            observed["poll_count"] += 1
            if is_interrupted():
                observed["saw_true"] = True
                return '{"interrupted": true}'
            time.sleep(0.05)
        return '{"timed_out": true}'

    agent._invoke_tool = MagicMock(side_effect=polling_tool)

    tc1 = _FakeToolCall("hung_fake_tool_1", call_id="tc1")
    tc2 = _FakeToolCall("hung_fake_tool_2", call_id="tc2")
    msg = _FakeAssistantMsg([tc1, tc2])
    messages = []

    def _interrupt_after_start():
        # Wait until at least one worker is running so its tid is tracked.
        worker_started.wait(timeout=2.0)
        time.sleep(0.2)  # let the other worker enter too
        agent.interrupt("stop requested by test")

    t = threading.Thread(target=_interrupt_after_start)
    t.start()
    start = time.monotonic()
    agent._execute_tool_calls_concurrent(msg, messages, "test_task")
    elapsed = time.monotonic() - start
    t.join(timeout=2.0)

    # The worker must have actually polled is_interrupted — otherwise the
    # test isn't exercising what it claims to.
    assert observed["poll_count"] > 0, (
        "polling_tool never ran — test scaffold issue"
    )
    # The worker must see the interrupt within ~1 s of agent.interrupt()
    # being called.  Before the fix this loop ran until its 5 s own-timeout.
    assert observed["saw_true"], (
        f"is_interrupted() never returned True inside the concurrent worker "
        f"after agent.interrupt() — interrupt-propagation hole regressed. "
        f"worker_tid={observed['worker_tid']!r} poll_count={observed['poll_count']}"
    )
    assert elapsed < 3.0, (
        f"concurrent execution took {elapsed:.2f}s after interrupt — the fan-out "
        f"to worker tids didn't shortcut the tool's poll loop as expected"
    )
    # Also verify cleanup: no stale worker tids should remain after all
    # tools finished.
    assert agent._tool_worker_threads == set(), (
        f"worker tids leaked after run: {agent._tool_worker_threads}"
    )


def test_clear_interrupt_clears_worker_tids(monkeypatch):
    """After clear_interrupt(), stale worker-tid bits must be cleared so the
    next turn's tools — which may be scheduled onto recycled tids — don't
    see a false interrupt."""
    from tools.interrupt import is_interrupted, set_interrupt

    agent = _make_agent(monkeypatch)
    # Simulate a worker having registered but not yet exited cleanly (e.g. a
    # hypothetical bug in the tear-down).  Put a fake tid in the set and
    # flag it interrupted.
    fake_tid = threading.current_thread().ident  # use real tid so is_interrupted can see it
    with agent._tool_worker_threads_lock:
        agent._tool_worker_threads.add(fake_tid)
    set_interrupt(True, fake_tid)
    assert is_interrupted() is True  # sanity

    agent.clear_interrupt()

    assert is_interrupted() is False, (
        "clear_interrupt() did not clear the interrupt bit for a tracked "
        "worker tid — stale interrupt can leak into the next turn"
    )

