"""Executor consults and records the durable timeout retry circuit."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.tool_executor import _run_agent_tool_execution_middleware


def _agent():
    return SimpleNamespace(
        session_id="session-a",
        quiet_mode=True,
        tool_progress_mode="off",
        _subagent_id=None,
        _current_turn_id="turn",
        _current_api_request_id="req",
        _touch_activity=MagicMock(),
        _append_guardrail_observation=MagicMock(side_effect=lambda _n, _a, r, **_k: r),
        _record_file_mutation_result=MagicMock(),
    )


def test_recent_identical_timeout_blocks_before_dispatch(monkeypatch):
    agent = _agent()
    dispatched = MagicMock(return_value="should not run")
    monkeypatch.setattr(
        "agent.tool_timeout_circuit.is_tool_timeout_blocked",
        lambda name, args, session_id: (name, args, session_id) == (
            "terminal", {"command": "sleep 9"}, "session-a"
        ),
    )
    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, None)):
        result = _run_agent_tool_execution_middleware(
            agent,
            function_name="terminal",
            function_args={"command": "sleep 9"},
            effective_task_id="task",
            tool_call_id="call",
            execute=dispatched,
        )
    assert result.blocked is True
    assert "timed out recently" in result.result
    assert "Do NOT retry unchanged" in result.result
    dispatched.assert_not_called()


def test_terminal_native_timeout_records_final_mutated_args(monkeypatch):
    agent = _agent()
    agent._tool_guardrails = SimpleNamespace(
        before_call=lambda *_args: SimpleNamespace(allows_execution=True)
    )
    recorded = []
    monkeypatch.setattr(
        "agent.tool_timeout_circuit.is_tool_timeout_blocked", lambda *_args: False
    )
    monkeypatch.setattr(
        "agent.tool_timeout_circuit.record_tool_timeout",
        lambda name, args, session_id: recorded.append((name, args, session_id)),
    )
    with patch(
        "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
        return_value=(None, {"command": "final command", "timeout": 1}),
    ):
        result = _run_agent_tool_execution_middleware(
            agent,
            function_name="terminal",
            function_args={"command": "original"},
            effective_task_id="task",
            tool_call_id="call",
            execute=lambda _args: '{"status":"timeout","error_type":"terminal_timeout"}',
            begin_execution=lambda _callback: None,
        )
    assert "terminal_timeout" in result.result
    assert recorded == [
        ("terminal", {"command": "final command", "timeout": 1}, "session-a")
    ]


def test_abandoned_before_dispatch_does_not_publish_timeout_args(monkeypatch):
    agent = _agent()
    agent._tool_guardrails = SimpleNamespace(
        before_call=lambda *_args: SimpleNamespace(allows_execution=True)
    )
    published = []
    monkeypatch.setattr(
        "agent.tool_timeout_circuit.is_tool_timeout_blocked", lambda *_args: False
    )

    def abandon(_callback):
        raise RuntimeError("batch abandoned before dispatch")

    with patch(
        "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
        return_value=(None, {"command": "final command"}),
    ):
        try:
            _run_agent_tool_execution_middleware(
                agent,
                function_name="terminal",
                function_args={"command": "original"},
                effective_task_id="task",
                tool_call_id="call",
                execute=lambda _args: "must not run",
                begin_execution=abandon,
                dispatch_args_sink=published.append,
            )
        except RuntimeError as exc:
            assert str(exc) == "batch abandoned before dispatch"
        else:
            raise AssertionError("abandonment must propagate")

    assert published == []
