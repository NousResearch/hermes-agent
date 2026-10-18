"""Tests for interrupt handling in concurrent tool execution."""

import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace
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


@pytest.mark.parametrize("names", [("slow", "fast"), ("fast", "slow"), ("slow", "slow")])
@pytest.mark.parametrize("failure", ["timeout", "missing_result"])
def test_concurrent_timeout_preserves_each_result(monkeypatch, names, failure):
    """An expired call must not crash collection or discard a successful sibling."""
    from agent import tool_executor
    import tools.daemon_pool

    agent = _make_agent(monkeypatch)
    agent._tool_guardrails = SimpleNamespace(
        before_call=lambda *a: SimpleNamespace(allows_execution=True)
    )
    agent._append_guardrail_observation = lambda name, args, result, **kw: result
    agent._record_file_mutation_result = MagicMock()
    agent._tool_result_content_for_active_model = lambda name, result: result
    agent._subdirectory_hints.check_tool_call.return_value = None
    activity = []
    agent._touch_activity = activity.append
    monkeypatch.setattr(tool_executor, "get_active_env", lambda *a: None)
    # Execute healthy workers immediately and leave the simulated hung worker
    # pending. An already-expired deadline makes this deterministic, with no
    # sleeps, real hung threads, or wall-clock-dependent assertions.
    class Pool:
        def __init__(self, **kw):
            pass

        def submit(self, fn, index, call, name, args, trace):
            future = Future()
            if name != "slow":
                fn(index, call, name, args, trace)
                future.set_result(None)
            elif failure == "missing_result":
                future.set_result(None)
            return future

        def shutdown(self, **kw):
            pass

    monkeypatch.setattr(tools.daemon_pool, "DaemonThreadPoolExecutor", Pool)
    monkeypatch.setattr(tool_executor, "_resolve_concurrent_tool_timeout", lambda: 0.0)
    calls = [_FakeToolCall(name, call_id=f"call_{i}") for i, name in enumerate(names)]
    messages = []
    agent._execute_tool_calls_concurrent(_FakeAssistantMsg(calls), messages, "test_task")

    assert [m["tool_call_id"] for m in messages] == [c.id for c in calls]
    completions = [a for a in activity if a.startswith("tool completed:")]
    assert len(completions) == len(calls)
    for name, message, completion in zip(names, messages, completions):
        if name == "slow":
            assert ("timed out" if failure == "timeout" else "did not return a result") in message["content"]
            assert completion.endswith(" (error)")
        else:
            assert message["content"] == '{"ok": true}'
            assert not completion.endswith(" (error)")

