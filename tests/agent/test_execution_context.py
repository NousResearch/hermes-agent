"""Execution-authority context regressions for detached Kanban workers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.execution_context import (
    ExecutionRole,
    bind_agent_execution_context,
    current_execution_role,
    execution_role_from_environment,
    is_kanban_owner_context,
    reset_agent_execution_context,
)
from run_agent import AIAgent


def test_task_scope_alone_does_not_create_card_owner_authority():
    role = execution_role_from_environment({"HERMES_KANBAN_TASK": "task-1"})

    assert role is ExecutionRole.DIRECT


@pytest.mark.parametrize("value", ["", "true", "yes", "2", " 0 ", " 1 "])
def test_card_owner_marker_requires_exact_one(value):
    role = execution_role_from_environment({"HERMES_KANBAN_SESSION": value})

    assert role is ExecutionRole.DIRECT


def test_explicit_card_owner_marker_is_captured():
    role = execution_role_from_environment({"HERMES_KANBAN_SESSION": "1"})

    assert role is ExecutionRole.KANBAN_OWNER


def test_delegated_child_is_not_the_card_owner():
    agent = type(
        "Agent",
        (),
        {"_execution_role": ExecutionRole.KANBAN_OWNER, "_delegate_depth": 1},
    )()

    token = bind_agent_execution_context(agent)
    try:
        assert current_execution_role() is ExecutionRole.DELEGATE
        assert is_kanban_owner_context() is False
    finally:
        reset_agent_execution_context(token)


def test_run_conversation_binds_and_resets_owner_context():
    agent = AIAgent.__new__(AIAgent)
    agent._execution_role = ExecutionRole.KANBAN_OWNER
    agent._delegate_depth = 0
    seen = []

    def _fake_loop(*_args, **_kwargs):
        seen.append(is_kanban_owner_context())
        return {"final_response": "ok"}

    assert is_kanban_owner_context() is False
    with patch("agent.conversation_loop.run_conversation", side_effect=_fake_loop):
        result = agent.run_conversation("hello")

    assert result == {"final_response": "ok"}
    assert seen == [True]
    assert is_kanban_owner_context() is False


def test_run_conversation_resets_context_when_loop_raises():
    agent = AIAgent.__new__(AIAgent)
    agent._execution_role = ExecutionRole.KANBAN_OWNER
    agent._delegate_depth = 0

    def _boom(*_args, **_kwargs):
        assert is_kanban_owner_context() is True
        raise RuntimeError("boom")

    with (
        patch("agent.conversation_loop.run_conversation", side_effect=_boom),
        pytest.raises(RuntimeError, match="boom"),
    ):
        agent.run_conversation("hello")

    assert is_kanban_owner_context() is False
