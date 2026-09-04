"""Tool executor bridge for kanban automatic progress."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agent.tool_executor import _emit_terminal_post_tool_call


@pytest.fixture(autouse=True)
def _isolate_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)


def test_emit_terminal_post_tool_call_records_qualifying_success_once(monkeypatch):
    calls: list[tuple[str, str]] = []

    def _record(evidence_type, detail):
        calls.append((evidence_type, detail))
        return True

    monkeypatch.setattr(
        "tools.kanban_tools.record_automatic_progress_from_env",
        _record,
    )
    monkeypatch.setattr(
        "model_tools._emit_post_tool_call_hook",
        lambda **kwargs: None,
    )

    agent = MagicMock(session_id="sess")
    result = json.dumps({"exit_code": 0, "output": "ok"})

    _emit_terminal_post_tool_call(
        agent,
        function_name="terminal",
        function_args={"command": "pytest"},
        result=result,
        effective_task_id="task",
        tool_call_id="tc1",
        duration_ms=10,
    )

    assert calls == [("tests_passed", "tests passed")]


def test_emit_terminal_post_tool_call_skips_non_qualifying_and_errors(monkeypatch):
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "tools.kanban_tools.record_automatic_progress_from_env",
        lambda evidence_type, detail: calls.append((evidence_type, detail)),
    )
    monkeypatch.setattr(
        "model_tools._emit_post_tool_call_hook",
        lambda **kwargs: None,
    )

    agent = MagicMock(session_id="sess")

    _emit_terminal_post_tool_call(
        agent,
        function_name="terminal",
        function_args={"command": "ls"},
        result=json.dumps({"exit_code": 0, "output": "ok"}),
        effective_task_id="task",
        tool_call_id="tc1",
    )
    _emit_terminal_post_tool_call(
        agent,
        function_name="terminal",
        function_args={"command": "pytest"},
        result=json.dumps({"exit_code": 1, "output": "fail"}),
        effective_task_id="task",
        tool_call_id="tc2",
    )
    _emit_terminal_post_tool_call(
        agent,
        function_name="terminal",
        function_args={"command": "pytest"},
        result=json.dumps({"exit_code": 0}),
        effective_task_id="task",
        tool_call_id="tc3",
        status="timeout",
        error_type="tool_timeout",
        error_message="timed out",
    )

    assert calls == []
