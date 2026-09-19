"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_STOP_NUDGE",
        "HERMES_DELEGATED_CHILD_CONTEXT",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_enabled_with_kanban_task(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    assert kanban_stop_nudge_enabled() is True


def test_disabled_in_delegated_child_context(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    from agent.delegation_context import delegated_child_context

    with delegated_child_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None

    assert kanban_stop_nudge_enabled() is True


def test_disabled_in_delegated_child_process_context(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")
    clear_kanban_env.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")

    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_disabled_in_non_dispatcher_owned_context(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    from agent.delegation_context import non_dispatcher_owned_context

    with non_dispatcher_owned_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None

    assert kanban_stop_nudge_enabled() is True


def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_disabled_inside_delegated_child(clear_kanban_env):
    from agent.delegation_context import delegated_child_context

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    assert kanban_stop_nudge_enabled() is True
    with delegated_child_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True


def test_nudge_disabled_inside_non_dispatcher_context(clear_kanban_env):
    from agent.delegation_context import non_dispatcher_owned_context

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    assert kanban_stop_nudge_enabled() is True
    with non_dispatcher_owned_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge
    assert "protocol violation" in nudge.lower() or "protocol" in nudge.lower()


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None






# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.






@pytest.mark.parametrize(
    "tool_name,who",
    [
        ("kanban_request_review", "build worker handing off for same-card review"),
        ("kanban_request_changes", "review agent sending the card back"),
    ],
)
def test_no_nudge_after_handoff_tool(clear_kanban_env, tool_name, who):
    """Handoff tools end the worker's turn just like complete/block.

    Both move the card out of ``running``, and the worker is told to call
    them — goals.py's continuation/finalize prompts name
    ``kanban_request_review``; the force-loaded sdlc-review skill names
    ``kanban_request_changes``. Nudging afterwards asks a worker that did
    the right thing to close a card it must not close.
    """
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_handoff")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": tool_name, "tool_call_id": "1", "content": "ok"},
    ]
    assert session_called_kanban_terminal(messages) is True, who
    assert build_kanban_stop_nudge(messages=messages) is None


def test_nudge_still_fires_for_non_terminal_kanban_tool(clear_kanban_env):
    """Widening the set must not swallow the case the guard exists for."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "Let me open the review next.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_comment", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_comment", "tool_call_id": "1", "content": "ok"},
    ]
    assert session_called_kanban_terminal(messages) is False
    nudge = build_kanban_stop_nudge(messages=messages)
    assert nudge is not None
    # The nudge offers every worker exit, not just close-out; a card that must go
    # through review must never be steered to ``kanban_complete`` alone.
    assert "kanban_request_review" in nudge and "kanban_block" in nudge
