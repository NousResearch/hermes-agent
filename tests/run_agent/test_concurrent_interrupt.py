"""Tests for interrupt handling in concurrent tool execution."""

import threading
import time
from unittest.mock import MagicMock

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
    # /steer injection (added in PR #12116) fires after every concurrent
    # tool batch. Stub it as a no-op — this test exercises interrupt
    # fanout, not steer injection.
    stub._apply_pending_steer_to_tool_results = lambda *a, **kw: None
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


def test_concurrent_tool_batch_hard_deadline_abandons_wedged_tool(monkeypatch):
    """A tool with no internal timeout (e.g. read_file shelling out to a wedged
    backend) must not hang the whole batch forever. The wall-clock deadline
    abandons the unfinished worker and the turn proceeds with a timeout result,
    instead of blocking on the executor join (the ~11h production hang)."""
    import agent.tool_executor as te

    agent = _make_agent(monkeypatch)
    # This test runs a tool through the full executor path (the other tests
    # short-circuit at the preflight interrupt), so stub the guardrail gate to
    # permit execution.
    agent._tool_guardrails = MagicMock()
    agent._tool_guardrails.before_call.return_value = MagicMock(allows_execution=True)
    agent._tool_result_content_for_active_model = lambda name, content, *a, **kw: content
    agent._subdirectory_hints.check_tool_call.return_value = ""  # no extra hint appended
    # Short deadline so the test is fast (the 5s poll is the real granularity).
    monkeypatch.setattr(te, "_TOOL_BATCH_TIMEOUT_S", 1.0)

    started = threading.Event()
    release = threading.Event()  # never set during the assert → tool stays wedged

    def _wedged_tool(*args, **kwargs):
        started.set()
        release.wait(timeout=30)  # would block the batch ~30s without the deadline
        return '{"ok": true}'

    agent._invoke_tool = MagicMock(side_effect=_wedged_tool)

    msg = _FakeAssistantMsg([_FakeToolCall("slow_tool", call_id="tc_slow")])
    messages = []

    t0 = time.time()
    try:
        agent._execute_tool_calls_concurrent(msg, messages, "test_task")
        elapsed = time.time() - t0

        # Must return promptly (deadline + ~3s grace + one 5s poll), NOT ~30s,
        # and crucially must not block joining the still-wedged worker.
        assert elapsed < 12, f"batch did not honor the deadline (took {elapsed:.1f}s)"
        assert started.is_set(), "the tool never actually started"
        assert len(messages) == 1
        assert "deadline" in messages[0]["content"].lower(), messages[0]["content"]
    finally:
        release.set()  # let the abandoned worker unwind so the test process is clean

