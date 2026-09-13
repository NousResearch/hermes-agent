"""Tests for the concurrent start-order gate (PR #79571 / issue #79569).

The gate serializes tool dispatch by submit order so approval prompts and
progress output appear in the order the model requested them. A tool that
wedges *during dispatch* must not park every later-ordered worker forever:
before the bound existed, those parked tools never started, the batch deadline
then falsely reported them as "timed out", and the parked threads leaked
permanently (``f.cancel()`` cannot stop a running thread and nothing ever
notified the condition again).
"""

import json
import threading
import time
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)


def _make_agent(monkeypatch):
    """Minimal AIAgent-like stub, mirroring test_concurrent_interrupt.py."""
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
    stub._guardrail_block_result = lambda d: json.dumps({"error": "blocked"})
    return stub


class _FakeToolCall:
    def __init__(self, name, call_id):
        self.function = MagicMock(name=name, arguments="{}")
        self.function.name = name
        self.id = call_id


class _FakeAssistantMsg:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


def _wedge_first_tool(agent, wedged_name, dispatched, stop):
    """Wedge ``wedged_name`` during dispatch; record every real dispatch."""

    def _before_call(name, args):
        if name == wedged_name:
            stop.wait(30)  # released in test teardown, not by the gate
        return MagicMock(allows_execution=True)

    agent._tool_guardrails = MagicMock()
    agent._tool_guardrails.before_call = _before_call

    def _invoke(name, *a, **kw):
        dispatched.append((name, time.monotonic()))
        return json.dumps({"ok": name})

    agent._invoke_tool = MagicMock(side_effect=_invoke)


def test_wedged_dispatch_does_not_starve_later_tools(monkeypatch):
    """A tool wedged during dispatch must not block the rest of the batch.

    Before the gate was bounded, tool_b/tool_c never started and were falsely
    reported as "timed out" despite doing zero work.
    """
    import agent.tool_executor as te

    agent = _make_agent(monkeypatch)
    monkeypatch.setattr(te, "_START_ORDER_GATE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(te, "_resolve_concurrent_tool_timeout", lambda: 6.0)

    dispatched: list = []
    stop = threading.Event()
    _wedge_first_tool(agent, "tool_a", dispatched, stop)

    msg = _FakeAssistantMsg([
        _FakeToolCall("tool_a", "tc_a"),
        _FakeToolCall("tool_b", "tc_b"),
        _FakeToolCall("tool_c", "tc_c"),
    ])
    messages: list = []
    try:
        agent._execute_tool_calls_concurrent(msg, messages, "task")
    finally:
        stop.set()

    names = [n for n, _ in dispatched]
    assert "tool_b" in names and "tool_c" in names, (
        f"later-ordered tools were starved by the wedged dispatch: {names}"
    )

    by_tool = {m.get("name"): m["content"] for m in messages}
    for late in ("tool_b", "tool_c"):
        assert "timed out" not in str(by_tool[late]), (
            f"{late} ran but was still reported as timed out: {by_tool[late]!r}"
        )


def test_gate_timeout_stays_under_the_batch_deadline(monkeypatch):
    """The gate bound must clamp below the batch deadline it sits under.

    With a batch timeout shorter than the stock 120s gate, an unclamped gate
    expires only after the deadline already blamed the parked tools — the exact
    bug the bound exists to fix.
    """
    import agent.tool_executor as te

    agent = _make_agent(monkeypatch)
    monkeypatch.setattr(te, "_resolve_concurrent_tool_timeout", lambda: 2.0)

    dispatched: list = []
    stop = threading.Event()
    _wedge_first_tool(agent, "tool_a", dispatched, stop)

    msg = _FakeAssistantMsg([
        _FakeToolCall("tool_a", "tc_a"),
        _FakeToolCall("tool_b", "tc_b"),
    ])
    messages: list = []
    try:
        agent._execute_tool_calls_concurrent(msg, messages, "task")
    finally:
        stop.set()

    assert "tool_b" in [n for n, _ in dispatched], (
        "gate outlived the batch deadline, so tool_b was blamed without running"
    )


def test_abandoned_batch_does_not_dispatch_late(monkeypatch):
    """A gate-parked worker must abort once the batch is abandoned.

    Otherwise it wakes up after the turn already synthesized its result and
    dispatches the tool anyway — wasted work plus a duplicate post_tool_call
    for a tool_call_id the turn already closed.
    """
    import agent.tool_executor as te

    agent = _make_agent(monkeypatch)
    # Long gate: only the abandonment signal can release the parked workers.
    monkeypatch.setattr(te, "_START_ORDER_GATE_TIMEOUT_S", 30.0)
    monkeypatch.setattr(te, "_resolve_concurrent_tool_timeout", lambda: 60.0)

    dispatched: list = []
    stop = threading.Event()
    _wedge_first_tool(agent, "tool_a", dispatched, stop)

    def _fire_interrupt():
        time.sleep(0.5)
        agent.interrupt("user pressed stop")

    threading.Thread(target=_fire_interrupt, daemon=True).start()

    msg = _FakeAssistantMsg([
        _FakeToolCall("tool_a", "tc_a"),
        _FakeToolCall("tool_b", "tc_b"),
    ])
    messages: list = []
    started = time.monotonic()
    try:
        agent._execute_tool_calls_concurrent(msg, messages, "task")
        returned_at = time.monotonic()
    finally:
        stop.set()
        agent.clear_interrupt()

    assert returned_at - started < 25.0, (
        "batch waited out the full gate timeout instead of releasing parked "
        "workers on abandonment"
    )

    # Give a would-be late worker room to misbehave.
    time.sleep(1.0)
    late = [(n, t) for n, t in dispatched if t > returned_at]
    assert not late, f"tool(s) dispatched after the batch was abandoned: {late}"
    assert agent._current_tool is None, (
        f"_current_tool left pointing at a dead tool: {agent._current_tool!r}"
    )
