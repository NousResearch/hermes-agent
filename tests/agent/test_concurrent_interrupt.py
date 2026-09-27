"""Tests for interrupt handling in concurrent tool execution."""

import threading
import time
from types import SimpleNamespace
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

        def _tool_result_content_for_active_model(self, _name, result):
            return result

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


def test_running_concurrent_interrupt_keeps_unknown_effect(monkeypatch):
    """A dispatched worker abandoned by /stop may already have acted."""
    from agent.tool_executor import _ManagedToolResult, execute_tool_calls_concurrent

    agent = _make_agent(monkeypatch)
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    agent._record_file_mutation_result = MagicMock()
    agent._append_guardrail_observation = MagicMock(side_effect=lambda *a, **k: a[2])
    started = threading.Event()

    def _middleware(*_args, begin_execution=None, **_kwargs):
        # Mirror the real dispatch site: the callback is what marks the slot as
        # "may have started acting" (agent/tool_executor.py::_dispatch_authorized_once).
        begin_execution(lambda: None)
        started.set()
        time.sleep(10)
        return _ManagedToolResult(
            result='{"ok": true}', args={}, middleware_trace=[], blocked=False, dispatched=True,
        )

    def _interrupt_running_worker():
        assert started.wait(2)
        agent.interrupt()

    messages = []
    msg = _FakeAssistantMsg([_FakeToolCall("write_file", call_id="tc_running")])
    timer = threading.Thread(target=_interrupt_running_worker, daemon=True)
    timer.start()
    try:
        with patch("agent.tool_executor._run_agent_tool_execution_middleware", side_effect=_middleware):
            execute_tool_calls_concurrent(agent, msg, messages, "test_task", finalize=False)
    finally:
        timer.join(2)

    assert [(m.get("effect_disposition"), m.get("execution_status")) for m in messages] == [
        ("unknown", "cancelled")
    ]


def test_concurrent_keyboard_interrupt_preserves_cancelled_status(monkeypatch):
    """The worker's explicit cancellation marker must survive durable classification."""
    from agent.tool_executor import _append_batch_results, _ConcurrentBatch, _ToolCallRef

    agent = _make_agent(monkeypatch)
    ref = _ToolCallRef("write_file", {}, "task", "tc_ki", [])
    cancelled = ref.emit_cancelled(agent, time.time())
    parsed = SimpleNamespace(
        tool_call=SimpleNamespace(function=SimpleNamespace(name="write_file", arguments="{}")),
        name="write_file", args={}, middleware_trace=[], parse_error=None, scope_block=None,
        ref=lambda _task_id: ref,
    )
    batch = _ConcurrentBatch(agent, [], "task", [parsed], None)
    batch.results[0] = SimpleNamespace(
        ref=ref, result=cancelled, duration=0.0, is_error=True, blocked=False,
        execution_status="cancelled",
    )
    captured = {}

    def _capture(_agent, _messages, _ref, _result, **kwargs):
        captured.update(kwargs)
        return _result, _result, None

    with patch("agent.tool_executor._commit_tool_result", side_effect=_capture):
        assert _append_batch_results(agent, [], "task", batch, MagicMock())
    assert (captured["effect_disposition"], captured["execution_status"]) == (
        "unknown", "cancelled",
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

