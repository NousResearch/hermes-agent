from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import tool_executor
from agent.tool_guardrails import ToolGuardrailDecision


_ALLOWED_RUNTIME_FIELDS = {
    "event",
    "session_id",
    "tool_call_id",
    "tool_name",
    "mode",
    "task_id",
    "turn_id",
    "api_request_id",
    "elapsed_ms",
    "status",
    "error_type",
    "result_kind",
    "result_chars",
}


class _Guardrails:
    def __init__(self, decision: ToolGuardrailDecision | None = None) -> None:
        self.decision = decision or ToolGuardrailDecision()

    def before_call(self, tool_name, tool_args):
        return self.decision


class _SubdirectoryHints:
    def check_tool_call(self, tool_name, tool_args):
        return ""


class _FakeAgent:
    def __init__(self) -> None:
        self.session_id = "session-123"
        self._current_turn_id = "turn-456"
        self._current_api_request_id = "api-789"
        self._interrupt_requested = False
        self.log_prefix = ""
        self.quiet_mode = True
        self.tool_progress_mode = "off"
        self.verbose_logging = False
        self.log_prefix_chars = 12
        self.valid_tool_names = None
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self.tool_delay = 0
        self._current_tool = None
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        self._context_engine_tool_names = set()
        self._memory_manager = None
        self._checkpoint_mgr = SimpleNamespace(enabled=False)
        self._subdirectory_hints = _SubdirectoryHints()
        self.context_compressor = SimpleNamespace(context_length=200_000)
        self._tool_worker_threads_lock = threading.Lock()
        self._tool_worker_threads = set()
        self.activity = []
        self._tool_guardrails = _Guardrails()

    def _vprint(self, *args, **kwargs):
        return None

    def _safe_print(self, *args, **kwargs):
        return None

    def _wrap_verbose(self, prefix, text):
        return f"{prefix}{text}"

    def _touch_activity(self, message):
        self.activity.append(message)

    def _should_emit_quiet_tool_messages(self):
        return False

    def _should_start_quiet_spinner(self):
        return False

    def _append_guardrail_observation(self, tool_name, tool_args, result, failed=False):
        return result

    def _record_file_mutation_result(self, tool_name, tool_args, result, is_error):
        return None

    def _tool_result_content_for_active_model(self, tool_name, result):
        return result

    def _apply_pending_steer_to_tool_results(self, messages, count):
        return None

    def _guardrail_block_result(self, decision):
        return json.dumps(
            {"error": "SECRET_BLOCK_REASON should not reach runtime logs"},
            ensure_ascii=False,
        )

    def interrupt(self, reason):
        self._interrupt_requested = True


@pytest.fixture
def isolate_tool_executor(monkeypatch):
    monkeypatch.setattr(
        tool_executor,
        "_apply_tool_request_middleware_for_agent",
        lambda agent, *, function_name, function_args, effective_task_id, tool_call_id: (function_args, []),
    )
    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(tool_executor, "maybe_persist_tool_result", lambda *, content, **kwargs: content)
    monkeypatch.setattr(tool_executor, "enforce_turn_budget", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tool_executor,
        "_ra",
        lambda: SimpleNamespace(
            handle_function_call=lambda *args, **kwargs: "SAFE_RESULT should not reach runtime logs",
            _set_interrupt=lambda *args, **kwargs: None,
        ),
    )


def _tool_call(name="terminal", arguments=None, id="call-123"):
    return SimpleNamespace(
        id=id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments if arguments is not None else {"command": "SECRET_COMMAND"}),
        ),
    )


def _assistant_message(*tool_calls):
    return SimpleNamespace(tool_calls=list(tool_calls))


def _runtime_events(caplog):
    events = []
    for record in caplog.records:
        message = record.getMessage()
        if not message.startswith("tool_runtime_event "):
            continue
        payload = json.loads(message.removeprefix("tool_runtime_event "))
        assert set(payload) <= _ALLOWED_RUNTIME_FIELDS
        events.append(payload)
    return events


def _runtime_event_messages(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("tool_runtime_event ")
    ]


def test_source_does_not_restore_forbidden_raw_persistent_log_patterns():
    source = Path(tool_executor.__file__).read_text(encoding="utf-8")
    forbidden_snippets = [
        'logger.error("_invoke_tool raised for %s: %s", function_name, tool_error, exc_info=True)',
        'logger.error("context_engine.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)',
        'logger.error("memory_manager.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)',
        'logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)',
        'logger.info("tool %s failed (%.2fs): %s", function_name, duration, result[:200])',
        'logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)',
        "result_preview = function_result if agent.verbose_logging else",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in source


def test_runtime_event_helper_redacts_to_allowed_schema(caplog):
    agent = _FakeAgent()

    with caplog.at_level(logging.INFO, logger="agent.tool_executor"):
        tool_executor._log_tool_runtime_event(
            agent,
            event="tool.finish",
            tool_name="terminal",
            mode="sequential",
            task_id="task-abc",
            tool_call_id="call-123",
            elapsed_ms=17,
            status="success",
            result="SECRET_RESULT should not be logged",
            function_args={"command": "SECRET_COMMAND should not be logged"},
            error_message="SECRET_EXCEPTION_MESSAGE should not be logged",
            prompt="SECRET_PROMPT should not be logged",
            private_url="https://private.example.invalid/token",
        )

    messages = _runtime_event_messages(caplog)
    assert len(messages) == 1
    assert "SECRET_RESULT" not in messages[0]
    assert "SECRET_COMMAND" not in messages[0]
    assert "SECRET_EXCEPTION_MESSAGE" not in messages[0]
    assert "SECRET_PROMPT" not in messages[0]
    assert "private.example.invalid" not in messages[0]

    payload = _runtime_events(caplog)[0]
    assert payload == {
        "event": "tool.finish",
        "session_id": "session-123",
        "tool_call_id": "call-123",
        "tool_name": "terminal",
        "mode": "sequential",
        "task_id": "task-abc",
        "turn_id": "turn-456",
        "api_request_id": "api-789",
        "elapsed_ms": 17,
        "status": "success",
    }


def test_sequential_execution_emits_start_and_finish_events_without_raw_payloads(
    caplog,
    isolate_tool_executor,
):
    agent = _FakeAgent()
    messages = []

    with caplog.at_level(logging.INFO, logger="agent.tool_executor"):
        tool_executor.execute_tool_calls_sequential(
            agent,
            _assistant_message(_tool_call()),
            messages,
            "task-abc",
        )

    event_messages = _runtime_event_messages(caplog)
    assert "SECRET_COMMAND" not in "\n".join(event_messages)
    assert "SAFE_RESULT" not in "\n".join(event_messages)

    events = _runtime_events(caplog)
    assert [event["event"] for event in events] == ["tool.start", "tool.finish"]
    assert events[0] == {
        "event": "tool.start",
        "session_id": "session-123",
        "tool_call_id": "call-123",
        "tool_name": "terminal",
        "mode": "sequential",
        "task_id": "task-abc",
        "turn_id": "turn-456",
        "api_request_id": "api-789",
        "status": "started",
    }
    assert events[1]["status"] == "success"
    assert events[1]["result_kind"] == "text"
    assert events[1]["result_chars"] == len("SAFE_RESULT should not reach runtime logs")
    assert events[1]["elapsed_ms"] >= 0


def test_sequential_execution_emits_error_event_without_result_text(
    caplog,
    isolate_tool_executor,
    monkeypatch,
):
    agent = _FakeAgent()
    monkeypatch.setattr(tool_executor, "_detect_tool_failure", lambda tool_name, result: (True, ""))

    with caplog.at_level(logging.INFO, logger="agent.tool_executor"):
        tool_executor.execute_tool_calls_sequential(
            agent,
            _assistant_message(_tool_call()),
            [],
            "task-abc",
        )

    event_messages = _runtime_event_messages(caplog)
    assert "SECRET_COMMAND" not in "\n".join(event_messages)
    assert "SAFE_RESULT" not in "\n".join(event_messages)

    events = _runtime_events(caplog)
    assert [event["event"] for event in events] == ["tool.start", "tool.error"]
    assert events[1]["status"] == "error"
    assert events[1]["error_type"] == "tool_result_error"
    assert events[1]["result_kind"] == "text"
    assert events[1]["result_chars"] == len("SAFE_RESULT should not reach runtime logs")


def test_non_verbose_persistent_logs_summarize_errors_without_raw_content(
    caplog,
    isolate_tool_executor,
    monkeypatch,
):
    secret_error = (
        "SECRET_EXCEPTION_MESSAGE command=SECRET_COMMAND "
        "prompt=SECRET_PROMPT https://private.example.invalid/token"
    )

    def _raise_tool_error(*args, **kwargs):
        raise RuntimeError(secret_error)

    monkeypatch.setattr(tool_executor, "_detect_tool_failure", lambda tool_name, result: (True, ""))
    monkeypatch.setattr(
        tool_executor,
        "_ra",
        lambda: SimpleNamespace(
            handle_function_call=_raise_tool_error,
            _set_interrupt=lambda *args, **kwargs: None,
        ),
    )

    agent = _FakeAgent()
    with caplog.at_level(logging.INFO, logger="agent.tool_executor"):
        tool_executor.execute_tool_calls_sequential(
            agent,
            _assistant_message(_tool_call(arguments={"command": "SECRET_COMMAND"})),
            [],
            "task-abc",
        )

    persistent_log_text = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "agent.tool_executor"
    )
    assert "SECRET_EXCEPTION_MESSAGE" not in persistent_log_text
    assert "SECRET_COMMAND" not in persistent_log_text
    assert "SECRET_PROMPT" not in persistent_log_text
    assert "private.example.invalid" not in persistent_log_text
    assert "RuntimeError" in persistent_log_text
    assert "result_kind=text" in persistent_log_text


def test_sequential_blocked_and_cancelled_events_do_not_emit_start(
    caplog,
    isolate_tool_executor,
):
    blocked_agent = _FakeAgent()
    blocked_agent._tool_guardrails = _Guardrails(
        ToolGuardrailDecision(
            action="block",
            code="too_many_repeats",
            message="SECRET_BLOCK_MESSAGE should not reach runtime logs",
            tool_name="terminal",
        )
    )

    with caplog.at_level(logging.INFO, logger="agent.tool_executor"):
        tool_executor.execute_tool_calls_sequential(
            blocked_agent,
            _assistant_message(_tool_call()),
            [],
            "task-abc",
        )

    blocked_messages = _runtime_event_messages(caplog)
    assert "SECRET_BLOCK" not in "\n".join(blocked_messages)
    blocked_events = _runtime_events(caplog)
    assert [event["event"] for event in blocked_events] == ["tool.blocked"]
    assert blocked_events[0]["status"] == "blocked"
    assert blocked_events[0]["error_type"] == "guardrail_block"

    caplog.clear()
    cancelled_agent = _FakeAgent()
    cancelled_agent._interrupt_requested = True

    with caplog.at_level(logging.INFO, logger="agent.tool_executor"):
        tool_executor.execute_tool_calls_sequential(
            cancelled_agent,
            _assistant_message(_tool_call(id="call-cancelled")),
            [],
            "task-abc",
        )

    cancelled_events = _runtime_events(caplog)
    assert [event["event"] for event in cancelled_events] == ["tool.cancelled"]
    assert cancelled_events[0]["status"] == "cancelled"
    assert cancelled_events[0]["error_type"] == "user_interrupt"
    assert cancelled_events[0]["tool_call_id"] == "call-cancelled"


def test_sequential_post_tool_interrupt_emits_cancelled_for_remaining_call_without_start(
    caplog,
    isolate_tool_executor,
    monkeypatch,
):
    agent = _FakeAgent()

    def _interrupt_after_first_call(*args, **kwargs):
        agent._interrupt_requested = True
        return "SAFE_FIRST_RESULT should not reach runtime logs"

    monkeypatch.setattr(
        tool_executor,
        "_ra",
        lambda: SimpleNamespace(
            handle_function_call=_interrupt_after_first_call,
            _set_interrupt=lambda *args, **kwargs: None,
        ),
    )

    with caplog.at_level(logging.INFO, logger="agent.tool_executor"):
        tool_executor.execute_tool_calls_sequential(
            agent,
            _assistant_message(
                _tool_call(id="call-first", arguments={"command": "SECRET_FIRST_COMMAND"}),
                _tool_call(id="call-second", arguments={"command": "SECRET_SECOND_COMMAND"}),
            ),
            [],
            "task-abc",
        )

    event_messages = "\n".join(_runtime_event_messages(caplog))
    assert "SECRET_FIRST_COMMAND" not in event_messages
    assert "SECRET_SECOND_COMMAND" not in event_messages
    assert "SAFE_FIRST_RESULT" not in event_messages

    events = _runtime_events(caplog)
    assert [event["event"] for event in events] == ["tool.start", "tool.finish", "tool.cancelled"]
    assert events[0]["tool_call_id"] == "call-first"
    assert events[1]["tool_call_id"] == "call-first"
    assert events[2]["tool_call_id"] == "call-second"
    assert events[2]["status"] == "cancelled"
    assert events[2]["error_type"] == "user_interrupt"
    assert events[2]["elapsed_ms"] == 0
    assert not any(
        event["event"] == "tool.start" and event["tool_call_id"] == "call-second"
        for event in events
    )


def test_concurrent_execution_emits_start_and_finish_events(
    caplog,
    isolate_tool_executor,
):
    agent = _FakeAgent()

    def _invoke_tool(function_name, function_args, effective_task_id, tool_call_id, **kwargs):
        return f"SAFE_CONCURRENT_RESULT {tool_call_id} should not reach runtime logs"

    agent._invoke_tool = _invoke_tool

    with caplog.at_level(logging.INFO, logger="agent.tool_executor"):
        tool_executor.execute_tool_calls_concurrent(
            agent,
            _assistant_message(
                _tool_call(id="call-a", arguments={"command": "SECRET_A"}),
                _tool_call(id="call-b", arguments={"command": "SECRET_B"}),
            ),
            [],
            "task-abc",
        )

    event_messages = "\n".join(_runtime_event_messages(caplog))
    assert "SECRET_A" not in event_messages
    assert "SECRET_B" not in event_messages
    assert "SAFE_CONCURRENT_RESULT" not in event_messages

    events = _runtime_events(caplog)
    assert [event["event"] for event in events].count("tool.start") == 2
    assert [event["event"] for event in events].count("tool.finish") == 2
    assert {event["mode"] for event in events} == {"concurrent"}
    assert {event["tool_call_id"] for event in events} == {"call-a", "call-b"}
    assert all(set(event) <= _ALLOWED_RUNTIME_FIELDS for event in events)
