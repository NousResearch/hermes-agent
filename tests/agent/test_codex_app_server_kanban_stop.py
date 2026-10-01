"""Codex app-server Kanban workers share Hermes' bounded terminal handoff policy."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.codex_runtime import run_codex_app_server_turn
from agent.kanban_stop import KanbanStopTarget, KANBAN_STOP_MAX_ATTEMPTS


TASK = "t_codex_stop_guard"
TARGET = KanbanStopTarget(task_id=TASK, run_id=41, status="running")


def _turn(text="text only", projected=None):
    return SimpleNamespace(
        interrupted=False,
        error=None,
        thread_id="thread-smoke",
        turn_id="turn-smoke",
        projected_messages=projected or [{"role": "assistant", "content": text}],
        tool_iterations=1,
        final_text=text,
        should_retire=False,
        token_usage_last=None,
        model_context_window=None,
        compacted=False,
    )


def _agent(turns):
    agent = MagicMock()
    agent._codex_session.run_turn.side_effect = list(turns)
    agent._codex_session_prompt = None
    agent._session_db = None
    agent.session_id = "codex-kanban-test"
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent.tool_progress_callback = None
    agent._interrupt_requested = False
    return agent


def _run(agent, messages=None):
    return run_codex_app_server_turn(
        agent,
        user_message="synthetic task",
        original_user_message="synthetic task",
        messages=messages if messages is not None else [{"role": "user", "content": "synthetic task"}],
        effective_task_id=TASK,
    )


@pytest.fixture
def owned_worker(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", TASK)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "41")
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setattr("agent.kanban_stop.kanban_stop_target", lambda **_: TARGET)
    monkeypatch.setattr(
        "agent.transports.hermes_tools_mcp_server.kanban_handoff_tools_available",
        lambda: True,
    )


def _terminal(tool_name):
    return _turn(
        "handoff",
        [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": tool_name, "arguments": "{}"}}
            ]},
            {"role": "tool", "name": tool_name, "tool_call_id": "call-1", "content": "accepted"},
        ],
    )


def test_codex_text_then_complete_gets_shared_nudge_and_passes(owned_worker, monkeypatch):
    # First completed turn leaves the run live; the real shared policy asks for a handoff.
    # The mock board then reports the terminal tool's completed transition.
    targets = iter((TARGET, KanbanStopTarget(task_id=TASK, run_id=41, status="done", terminal_handoff_accepted=True)))
    monkeypatch.setattr("agent.kanban_stop.kanban_stop_target", lambda **_: next(targets))
    agent = _agent([_turn("work finished in text"), _terminal("kanban_complete")])

    result = _run(agent)

    assert agent._codex_session.run_turn.call_count == 2
    second_input = agent._codex_session.run_turn.call_args_list[1].kwargs["user_input"]
    assert "kanban_complete" in second_input
    assert "is still running" in second_input
    assert result["completed"] is True
    assert result["error"] is None
    assert any(
        call.get("function", {}).get("name") == "kanban_complete"
        for message in result["messages"]
        for call in message.get("tool_calls", [])
    )
    roles = [message["role"] for message in result["messages"]]
    assert all(left != right for left, right in zip(roles, roles[1:])), roles
    assert result["messages"][1]["content"] == "work finished in text"
    assert result["messages"][2]["_kanban_stop_synthetic"] is True


def test_codex_text_until_budget_exhaustion_fails_closed(owned_worker):
    agent = _agent([_turn("still only text") for _ in range(KANBAN_STOP_MAX_ATTEMPTS + 1)])

    result = _run(agent)

    assert agent._codex_session.run_turn.call_count == KANBAN_STOP_MAX_ATTEMPTS + 1
    assert result["completed"] is False
    assert result["final_response"] == ""
    assert "budget exhausted" in result["error"]
    assert not any(
        call.get("name") in {"kanban_complete", "kanban_request_review", "kanban_request_changes", "kanban_block"}
        for message in result["messages"]
        for call in message.get("tool_calls", [])
    )


@pytest.mark.parametrize("tool_name,status", [
    ("kanban_complete", "done"),
    ("kanban_request_review", "review"),
    ("kanban_request_changes", "ready"),
    ("kanban_block", "blocked"),
])
def test_existing_terminal_handoff_gets_no_followup_nudge(owned_worker, monkeypatch, tool_name, status):
    # The board status is the acceptance signal for a successful terminal handoff.
    monkeypatch.setattr(
        "agent.kanban_stop.kanban_stop_target",
        lambda **_: KanbanStopTarget(
            task_id=TASK, run_id=41, status=status, terminal_handoff_accepted=True,
        ),
    )
    agent = _agent([_terminal(tool_name)])

    result = _run(agent)

    assert agent._codex_session.run_turn.call_count == 1
    assert result["completed"] is True
    assert result["error"] is None


@pytest.mark.parametrize("context", ["no-owner", "delegated", "cron"])
def test_non_owner_delegate_or_cron_gets_no_nudge(monkeypatch, context):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    if context == "delegated":
        monkeypatch.setenv("HERMES_KANBAN_TASK", TASK)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "41")
    agent = _agent([_turn()])

    if context == "delegated":
        from agent.delegation_context import delegated_child_context
        with delegated_child_context():
            result = _run(agent)
    elif context == "cron":
        from agent.delegation_context import non_dispatcher_owned_context
        with non_dispatcher_owned_context():
            result = _run(agent)
    else:
        result = _run(agent)

    assert agent._codex_session.run_turn.call_count == 1
    assert result["error"] is None


def test_explicit_nudge_disable_gets_no_followup(owned_worker, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    agent = _agent([_turn()])

    result = _run(agent)

    assert agent._codex_session.run_turn.call_count == 1
    assert result["error"] is None


def test_missing_kanban_callbacks_fails_closed(owned_worker, monkeypatch):
    monkeypatch.setattr(
        "agent.transports.hermes_tools_mcp_server.kanban_handoff_tools_available",
        lambda: False,
    )
    agent = _agent([_turn("text only")])

    result = _run(agent)

    assert agent._codex_session.run_turn.call_count == 1
    assert result["completed"] is False
    assert result["final_response"] == ""
    assert "handoff could not be proven" in result["error"]


def test_codex_unreadable_owned_board_state_fails_closed(owned_worker, monkeypatch):
    monkeypatch.setattr("agent.kanban_stop.kanban_stop_target", lambda **_: None)
    agent = _agent([_turn("text only")])

    result = _run(agent)

    assert result["completed"] is False
    assert result["final_response"] == ""
    assert "could not be proven" in result["error"]


def test_effective_handoff_registry_requires_all_four_tools():
    from agent.transports.hermes_tools_mcp_server import (
        EXPOSED_TOOLS,
        kanban_handoff_tools_available,
    )
    from agent.kanban_stop import KANBAN_TERMINAL_HANDOFF_TOOLS

    assert set(KANBAN_TERMINAL_HANDOFF_TOOLS).issubset(EXPOSED_TOOLS)
    assert kanban_handoff_tools_available(set(KANBAN_TERMINAL_HANDOFF_TOOLS)) is True
    assert kanban_handoff_tools_available(set(KANBAN_TERMINAL_HANDOFF_TOOLS[:-1])) is False


def test_dispatcher_mcp_server_refuses_partial_terminal_registry(monkeypatch):
    import model_tools
    from agent.transports.hermes_tools_mcp_server import _build_server

    monkeypatch.setenv("HERMES_KANBAN_TASK", TASK)
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_: [{"type": "function", "function": {"name": "kanban_complete"}}],
    )

    with pytest.raises(RuntimeError, match="missing a terminal handoff tool"):
        _build_server()


def test_board_target_is_read_only_and_bound_to_current_run(monkeypatch, tmp_path):
    from agent.kanban_stop import kanban_stop_target
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    db_path = tmp_path / "isolated" / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", TASK)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "")
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="synthetic target", assignee="coder", workspace_kind="scratch",
            project_id="", initial_status="blocked",
        )
        assert kb.unblock_task(conn, task_id) is True
        task = kb.claim_task(conn, task_id, claimer="synthetic-target-test")
        assert task is not None and task.current_run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    target = kanban_stop_target()
    assert target is not None and target.status == "running"
    with kbc.connect_readonly_closing() as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE tasks SET title = 'forbidden' WHERE id = ?", (task_id,))

    with kbc.connect_closing() as conn:
        assert kb.complete_task(
            conn, task_id, summary="synthetic terminal handoff", expected_run_id=task.current_run_id,
        ) is True
    accepted = kanban_stop_target()
    assert accepted is not None and accepted.status == "done" and accepted.terminal_handoff_accepted is True

    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id + 1))
    assert kanban_stop_target() is None


def test_nonrunning_without_this_runs_accepted_handoff_fails_closed(owned_worker, monkeypatch):
    from agent.kanban_stop import KanbanStopTarget

    monkeypatch.setattr(
        "agent.kanban_stop.kanban_stop_target",
        lambda: KanbanStopTarget(task_id=TASK, run_id=41, status="ready"),
    )
    agent = _agent([_turn("plain text")])

    result = _run(agent)

    assert agent._codex_session.run_turn.call_count == 1
    assert result["completed"] is False
    assert result["final_response"] == ""


def test_missing_owned_board_is_not_created(monkeypatch, tmp_path):
    from agent.kanban_stop import kanban_stop_target

    missing = tmp_path / "missing" / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(missing))
    monkeypatch.setenv("HERMES_KANBAN_TASK", TASK)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "41")

    assert kanban_stop_target() is None
    assert missing.exists() is False


def test_native_hermes_loop_still_uses_existing_kanban_stop_gate(owned_worker):
    from agent.turn_stop_gates import apply_stop_gates

    agent = SimpleNamespace(
        _kanban_stop_nudges=0,
        _interim_content_was_streamed=lambda _text: False,
        _emit_interim_assistant_message=lambda _msg: None,
        _flush_messages_to_session_db=lambda *_args: True,
        _emit_diagnostic_status=lambda _text: None,
        _session_messages=None,
    )
    messages = [{"role": "user", "content": "synthetic"}]
    verdict = apply_stop_gates(
        agent,
        {"role": "assistant", "content": "plain text"},
        final_response="plain text",
        messages=messages,
        conversation_history=None,
        pending_verification_response=None,
        pending_verification_response_previewed=False,
    )

    assert verdict.continue_turn is True
    assert agent._kanban_stop_nudges == 1
    assert messages[-1]["_kanban_stop_synthetic"] is True
