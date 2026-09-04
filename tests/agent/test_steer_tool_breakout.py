"""Behavior tests for mid-batch user steering (#28172)."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _tool_call(name: str, index: int):
    return SimpleNamespace(
        id=f"call-{index}",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


class _StubAgent:
    """Narrow executor double with explicit, non-MagicMock state."""

    def __init__(self) -> None:
        self._pending_steer = None
        self._pending_steer_lock = threading.Lock()
        self._interrupt_requested = False
        self._tool_interrupt_reason = None
        self._incremental_persistence_failed = False
        self._last_persistence_error_cause = None
        self._current_tool = None
        self._tool_worker_threads: set[int] = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._subdirectory_hints = MagicMock()
        self._subdirectory_hints.check_tool_call.return_value = None
        self._tool_guardrails = MagicMock()
        self._checkpoint_mgr = MagicMock(enabled=False)
        self._memory_manager = None
        self._context_engine_tool_names = None
        self._print_fn = print
        self._invoke_tool = lambda *_args, **_kwargs: json.dumps({"status": "ok"})
        self.quiet_mode = True
        self.verbose_logging = False
        self.tool_progress_mode = "off"
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self.log_prefix = ""
        self.log_prefix_chars = 200
        self.tool_delay = 0
        self.valid_tool_names = None
        self.session_id = "steer-breakout-test"

    def steer(self, text: str) -> None:
        with self._pending_steer_lock:
            self._pending_steer = text

    def _drain_pending_steer(self):
        with self._pending_steer_lock:
            text = self._pending_steer
            self._pending_steer = None
            return text

    def _apply_pending_steer_to_tool_results(self, messages, _count) -> None:
        text = self._drain_pending_steer()
        if not text:
            return
        for message in reversed(messages):
            if message.get("role") == "tool":
                message["content"] += f"\n\n[OUT-OF-BAND USER MESSAGE]\n{text}"
                return

    def _flush_messages_to_session_db(self, _messages) -> bool:
        return True

    def _touch_activity(self, _description) -> None:
        return None

    def _vprint(self, _message, force=False) -> None:
        return None

    def _safe_print(self, _message) -> None:
        return None

    def _wrap_verbose(self, _prefix, value):
        return value

    def _should_emit_quiet_tool_messages(self) -> bool:
        return False

    def _should_start_quiet_spinner(self) -> bool:
        return False

    def _has_stream_consumers(self) -> bool:
        return False

    def _append_guardrail_observation(
        self, _name, _args, result, failed=False, tool_call_id=None
    ):
        return result

    def _record_file_mutation_result(self, *_args, **_kwargs) -> None:
        return None

    def _tool_result_content_for_active_model(self, _name, result):
        return result


@pytest.fixture
def executor_module(monkeypatch):
    from agent import tool_executor as executor

    executor._test_invoke = lambda *_args, **_kwargs: json.dumps({"status": "ok"})

    def test_runtime():
        return SimpleNamespace(
            handle_function_call=lambda name, args, *_a, **_k: executor._test_invoke(
                name, args
            )
        )

    def passthrough_middleware(
        _agent,
        *,
        function_args,
        execute,
        middleware_trace=None,
        **_kwargs,
    ):
        return executor._ManagedToolResult(
            result=execute(function_args),
            args=function_args,
            middleware_trace=list(middleware_trace or []),
            blocked=False,
            dispatched=True,
        )

    monkeypatch.setattr(
        executor, "_run_agent_tool_execution_middleware", passthrough_middleware
    )
    monkeypatch.setattr(executor, "_ra", test_runtime)
    monkeypatch.setattr(
        executor, "_emit_terminal_post_tool_call", lambda *_a, **_k: None
    )
    monkeypatch.setattr(executor, "enforce_turn_budget", lambda *_a, **_k: None)
    monkeypatch.setattr(executor, "get_active_env", lambda *_a, **_k: None)
    monkeypatch.setattr(
        executor,
        "maybe_persist_tool_result",
        lambda *, content, **_kwargs: content,
    )
    return executor


def _tool_messages(messages):
    return [message for message in messages if message.get("role") == "tool"]


def test_sequential_steer_defers_every_unstarted_call(executor_module):
    agent = _StubAgent()
    invoked: list[str] = []

    def invoke(name, *_args, **_kwargs):
        invoked.append(name)
        if len(invoked) == 1:
            agent.steer("change the implementation approach")
        return json.dumps({"status": "ok", "tool": name})

    agent._invoke_tool = invoke
    executor_module._test_invoke = invoke
    calls = [_tool_call(f"tool-{index}", index) for index in range(3)]
    messages: list[dict] = []

    executor_module.execute_tool_calls_sequential(
        agent,
        SimpleNamespace(tool_calls=calls),
        messages,
        "task",
    )

    tool_messages = _tool_messages(messages)
    assert invoked == ["tool-0"]
    assert len(tool_messages) == len(calls)
    assert "deferred" in tool_messages[1]["content"].lower()
    assert "deferred" in tool_messages[2]["content"].lower()
    assert tool_messages[1]["effect_disposition"] == "none"
    assert tool_messages[2]["effect_disposition"] == "none"
    assert "change the implementation approach" in tool_messages[-1]["content"]


def test_concurrent_steer_cancels_only_queued_calls(executor_module, monkeypatch):
    agent = _StubAgent()
    started = threading.Event()
    release = threading.Event()
    invoked: list[str] = []

    def invoke(name, *_args, **_kwargs):
        invoked.append(name)
        started.set()
        assert release.wait(2.0)
        return json.dumps({"status": "ok", "tool": name})

    agent._invoke_tool = invoke
    monkeypatch.setattr(
        executor_module, "_max_workers_for_tool_batch", lambda _calls: 1
    )
    monkeypatch.setattr(executor_module, "_CONCURRENT_WAIT_POLL_INTERVAL_S", 0.02)

    def send_steer() -> None:
        assert started.wait(1.0)
        agent.steer("do not run the remaining tools")
        time.sleep(0.1)
        release.set()

    sender = threading.Thread(target=send_steer)
    sender.start()
    calls = [_tool_call(f"tool-{index}", index) for index in range(3)]
    messages: list[dict] = []

    executor_module.execute_tool_calls_concurrent(
        agent,
        SimpleNamespace(tool_calls=calls),
        messages,
        "task",
    )
    sender.join(timeout=1.0)

    tool_messages = _tool_messages(messages)
    assert invoked == ["tool-0"]
    assert len(tool_messages) == len(calls)
    assert "ok" in tool_messages[0]["content"]
    assert all(
        "deferred" in message["content"].lower() for message in tool_messages[1:]
    )
    assert all(message["effect_disposition"] == "none" for message in tool_messages[1:])
    assert "do not run the remaining tools" in tool_messages[-1]["content"]


def test_concurrent_pending_steer_submits_no_tools(executor_module):
    agent = _StubAgent()
    invoked: list[str] = []
    agent._invoke_tool = lambda name, *_a, **_k: invoked.append(name)
    agent.steer("replace the whole batch")
    calls = [_tool_call(f"tool-{index}", index) for index in range(3)]
    messages: list[dict] = []

    executor_module.execute_tool_calls_concurrent(
        agent,
        SimpleNamespace(tool_calls=calls),
        messages,
        "task",
    )

    tool_messages = _tool_messages(messages)
    assert invoked == []
    assert len(tool_messages) == len(calls)
    assert all("deferred" in message["content"].lower() for message in tool_messages)
    assert all(message["effect_disposition"] == "none" for message in tool_messages)
    assert "replace the whole batch" in tool_messages[-1]["content"]


def test_segmented_batch_defers_later_segments(executor_module, monkeypatch):
    agent = _StubAgent()
    first = _tool_call("first", 1)
    second = _tool_call("second", 2)
    executed_segments: list[list] = []

    def execute_first_segment(_agent, segment_message, messages, *_args, **_kwargs):
        executed_segments.append(segment_message.tool_calls)
        call = segment_message.tool_calls[0]
        messages.append(
            executor_module.make_tool_result_message(
                call.function.name,
                "ok",
                call.id,
            )
        )
        agent.steer("stop before the next segment")

    monkeypatch.setattr(
        executor_module,
        "execute_tool_calls_sequential",
        execute_first_segment,
    )
    messages: list[dict] = []

    executor_module.execute_tool_calls_segmented(
        agent,
        SimpleNamespace(tool_calls=[first, second]),
        messages,
        "task",
        segments=[("sequential", [first]), ("sequential", [second])],
    )

    tool_messages = _tool_messages(messages)
    assert executed_segments == [[first]]
    assert len(tool_messages) == 2
    assert "deferred" in tool_messages[1]["content"].lower()
    assert tool_messages[1]["effect_disposition"] == "none"
    assert "stop before the next segment" in tool_messages[1]["content"]
