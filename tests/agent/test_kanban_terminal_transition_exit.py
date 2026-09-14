"""Worker lifecycle transitions end the current tool round only after durable success."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import agent.turn_tool_round as subject
from agent.turn_tool_validation import ToolValidationVerdict


class DummyAgent:
    def __init__(self, tool_result: str, *, persistence_failed: bool = False):
        self.tool_result = tool_result
        self._incremental_persistence_failed = persistence_failed
        self._tool_guardrail_halt_decision = None
        self._budget_grace_call = False
        self._budget_grace_used = False
        self.verbose_logging = False
        self.quiet_mode = True
        self.log_prefix = ""
        self.session_id = "test-session"
        self.valid_tool_names = {"kanban_block"}
        self.stream_delta_callback = None
        self.iteration_budget = SimpleNamespace(refund=lambda: None)

    @staticmethod
    def _deduplicate_tool_calls(calls):
        return calls

    @staticmethod
    def _cap_delegate_task_calls(calls):
        return calls

    @staticmethod
    def _flush_messages_to_session_db(_messages, _history):
        return True

    @staticmethod
    def _emit_interim_assistant_message(_message):
        return None

    def _execute_tool_calls(self, assistant_message, messages, _task_id, _api_calls):
        call = assistant_message.tool_calls[0]
        messages.append({
            "role": "tool",
            "name": call.function.name,
            "tool_call_id": call.id,
            "content": self.tool_result,
        })

    @staticmethod
    def _flush_status_buffer():
        return None

    @staticmethod
    def _vprint(*_args, **_kwargs):
        return None

    @staticmethod
    def _touch_activity(_note):
        return None


def _tool_call(name: str = "kanban_block"):
    return SimpleNamespace(
        id="call-current",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _run(monkeypatch, *, result: str, persistence_failed: bool = False):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_owned")
    call = _tool_call()
    assistant_message = SimpleNamespace(tool_calls=[call])
    messages = []
    monkeypatch.setattr(
        subject,
        "validate_tool_calls",
        lambda *_args, **_kwargs: ToolValidationVerdict("ok", None, False),
    )
    monkeypatch.setattr(
        subject,
        "stage_tool_call_message",
        lambda *_args, **_kwargs: ({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": "{}"},
            }],
        }, False),
    )
    monkeypatch.setattr(
        subject,
        "compress_after_tool_results",
        lambda *_args, **kwargs: SimpleNamespace(
            messages=kwargs["messages"],
            active_system_prompt=kwargs["active_system_prompt"],
            conversation_history=kwargs["conversation_history"],
            compression_attempts=kwargs["compression_attempts"],
            final_response=kwargs["final_response"],
            turn_exit_reason=kwargs["turn_exit_reason"],
            end_turn=False,
        ),
    )
    verdict = subject.run_tool_round(
        DummyAgent(result, persistence_failed=persistence_failed),
        assistant_message=assistant_message,
        finish_reason="tool_calls",
        messages=messages,
        conversation_history=[],
        api_call_count=1,
        effective_task_id=None,
        user_message="work task",
        system_message="system",
        active_system_prompt="system",
        compression_attempts=0,
        max_compression_attempts=3,
        final_response="",
        failed=False,
        _turn_exit_reason="",
        truncated_tool_call_retries=0,
    )
    return verdict, messages


def test_successful_block_breaks_before_another_provider_turn(monkeypatch):
    verdict, messages = _run(
        monkeypatch,
        result='{"ok": true, "task_id": "t_owned", "status": "blocked"}',
    )
    assert verdict.action == "break"
    assert verdict._turn_exit_reason == "kanban_terminal_transition"
    assert verdict.final_response == "Kanban lifecycle transition completed via `kanban_block`."
    assert messages[-1]["role"] == "tool"


def test_rejected_block_continues_to_model(monkeypatch):
    verdict, _messages = _run(
        monkeypatch,
        result='{"error": "could not block t_owned"}',
    )
    assert verdict.action == "continue"
    assert verdict._turn_exit_reason == ""


def test_persistence_failure_wins_over_successful_block(monkeypatch):
    verdict, _messages = _run(
        monkeypatch,
        result='{"ok": true, "task_id": "t_owned", "status": "blocked"}',
        persistence_failed=True,
    )
    assert verdict.action == "break"
    assert verdict._turn_exit_reason == "session_persistence_failed"
    assert verdict.failed is True
