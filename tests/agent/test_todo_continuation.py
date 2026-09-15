"""Preserved tasks may prompt bounded reconciliation, never grant new authority."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent


TASK = {"id": "verify", "content": "Verify the local artifact", "status": "in_progress"}


def _response(text="", todos=None):
    calls = None if todos is None else [SimpleNamespace(
        id="todo-write", type="function",
        function=SimpleNamespace(name="todo_list", arguments=json.dumps({"todos": todos})),
    )]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=calls),
                                 finish_reason="tool_calls" if calls else "stop")],
        model="test/model", usage=None,
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        instance = AIAgent(
            session_id="todo-continuation", api_key="test-key",
            base_url="https://example.invalid/v1", provider="openai-compat",
            model="test/model", max_iterations=6, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    instance.valid_tool_names = ["todo_list"]
    instance._intent_ack_continuation = False
    return instance


@pytest.mark.parametrize("budget", [2, 6])
def test_status_with_committed_work_gets_bounded_reconciliation(agent, budget):
    agent.max_iterations = budget
    agent.iteration_budget.max_total = budget
    requests = []

    def call(kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            return _response(todos=[TASK])
        return _response("The artifact is staged; verification remains unfinished.")

    agent._interruptible_api_call = call
    agent._handle_max_iterations = lambda *_a, **_kw: "Budget reached; verification remains unfinished."
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("Prepare and verify the local artifact.")
    if budget == 2:
        assert result["completed"] is False
        assert "Budget reached" in result["final_response"]
    else:
        # One reconciliation opportunity, not an unbounded loop on an unchanged todo.
        assert len(requests) == 3
        assert requests[1]["messages"][:-1] == requests[2]["messages"][:-3]
        assert [m["role"] for m in requests[2]["messages"][-2:]] == ["assistant", "user"]
        assert requests[1]["messages"][0] == requests[2]["messages"][0]
    assert agent._todo_store.read() == [TASK]


@pytest.mark.parametrize("boundary", ["new_topic", "redirect", "late_steer", "interrupt", "approval", "merge", "credentials", "input", "async", "off", "completed"])
def test_todo_reconciliation_preserves_stop_and_yield_boundaries(agent, boundary, monkeypatch):
    requests = []
    text = {
        "approval": "Waiting for your approval before publishing.",
        "merge": "PR is ready; awaiting merge by a maintainer.",
        "credentials": "I need your credentials before continuing.",
        "input": "Which account should I use?",
    }.get(boundary, "The artifact is staged; verification remains unfinished.")
    if boundary == "new_topic":
        agent._todo_store.write([TASK])
    if boundary == "off":
        agent._stall_guards = False
    if boundary == "async":
        monkeypatch.setattr("hermes_cli.goals.count_active_delegations", lambda _sid: 1)

    def call(kwargs):
        requests.append(kwargs)
        if len(requests) == 1 and boundary != "new_topic":
            if boundary == "redirect":
                agent._pending_steer = "Stop that. Just give me the current status."
            task = {**TASK, "status": "completed"} if boundary == "completed" else TASK
            return _response(todos=[task])
        if boundary == "late_steer":
            agent._pending_steer = "Stop that. Just give me the current status."
        if boundary == "interrupt":
            agent._interrupt_requested = True
        return _response(text)

    agent._interruptible_api_call = call
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        agent.run_conversation("What is a checksum?" if boundary == "new_topic" else "Prepare and verify the local artifact.")
    assert len(requests) == (1 if boundary == "new_topic" else 2)


def test_reconciliation_can_resume_work_and_finish(agent):
    answers = iter([
        _response(todos=[TASK]),
        _response("The artifact is staged; verification remains unfinished."),
        _response(todos=[{**TASK, "status": "completed"}]),
        _response("The local artifact is verified."),
    ])
    agent._interruptible_api_call = lambda _kwargs: next(answers)
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("Prepare and verify the local artifact.")
    assert result["final_response"] == "The local artifact is verified."
    assert agent._todo_store.read()[0]["status"] == "completed"


def test_one_nudge_even_when_model_rewrites_same_task_and_across_turns(agent):
    for _ in range(2):
        requests = []

        def call(kwargs):
            requests.append(kwargs)
            if len(requests) in (1, 3):
                return _response(todos=[TASK])
            return _response("The artifact is staged; verification remains unfinished.")

        agent._interruptible_api_call = call
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            agent.run_conversation("Prepare and verify the local artifact.")
        assert len(requests) == 4


@pytest.mark.parametrize("case", ["read", "failed_write", "pending_only", "cancelled", "halt", "process"])
def test_conservative_state_boundaries(agent, case):
    from agent.turn_stop_gates import todo_continuation_nudge
    from tools.todo_tool import todo_tool

    task = {**TASK, "status": {"pending_only": "pending", "cancelled": "cancelled"}.get(case, "in_progress")}
    result = todo_tool(todos=[task], store=agent._todo_store)
    arguments = {} if case == "read" else {"todos": [task]}
    if case == "failed_write":
        result = '{"error": "write failed"}'
    if case == "halt":
        agent._tool_guardrail_halt_decision = object()
    messages = [
        {"role": "user", "content": "Verify it."},
        {"role": "assistant", "tool_calls": [{"id": "write", "function": {
            "name": "todo_list", "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "tool_call_id": "write", "content": result},
    ]
    agent._current_task_id = "owned-turn"
    with (patch("hermes_cli.goals.count_active_delegations", return_value=0),
          patch("hermes_cli.goals.gather_background_processes", return_value=[{
              "notify_on_complete": True,
          }] if case == "process" else []) as processes):
        assert todo_continuation_nudge(agent, messages, "Verification remains unfinished.") is None
        if case == "process":
            processes.assert_called_once_with(owner_task_id="owned-turn")


def test_merge_write_adopts_full_snapshot(agent):
    from agent.turn_stop_gates import todo_continuation_nudge
    from tools.todo_tool import todo_tool

    agent._todo_store.write([TASK, {"id": "later", "content": "Report", "status": "pending"}])
    delta = [{"id": "verify", "content": "Verify changed artifact", "status": "in_progress"}]
    result = todo_tool(todos=delta, merge=True, store=agent._todo_store)
    messages = [
        {"role": "user", "content": "Verify it."},
        {"role": "assistant", "tool_calls": [{"id": "merge", "function": {
            "name": "todo_list", "arguments": json.dumps({"todos": delta, "merge": True})}}]},
        {"role": "tool", "tool_call_id": "merge", "content": result},
    ]
    with (patch("hermes_cli.goals.count_active_delegations", return_value=0),
          patch("hermes_cli.goals.gather_background_processes", return_value=[])):
        assert todo_continuation_nudge(agent, messages, "Verification remains unfinished.")
