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
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch






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


# ── Regression: valid review handoff is terminal for the originating run ──
# Repro of the exit-guard race (task t_170fcd74): an implementation run calls
# kanban_request_review, the tool returns status=review and the run row ends
# with outcome=review_requested; the dispatcher then claims the review as a NEW
# run. The outgoing session must NOT be nudged toward kanban_complete/block —
# doing so risks the stale worker falsely completing/blocking the newer run.


@pytest.mark.parametrize("review_tool", ["kanban_request_review", "kanban_request_changes"])
def test_no_nudge_after_review_handoff(clear_kanban_env, review_tool):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_170fcd74")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "r1",
                    "type": "function",
                    "function": {"name": review_tool, "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": review_tool,
            "tool_call_id": "r1",
            "content": "status=review",
        },
    ]
    # The originating run reached a terminal board state — no further nudge.
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages, attempts=0) is None


def test_review_handoff_terminal_even_from_tool_result_only(clear_kanban_env):
    # Defensive: even if only the tool-result row is present (assistant
    # tool_calls elided from replayed history), the terminal state is honored.
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_170fcd74")
    messages = [
        {
            "role": "tool",
            "name": "kanban_request_review",
            "tool_call_id": "r1",
            "content": "status=review",
        },
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_still_nudges_on_nonterminal_review_tools(clear_kanban_env):
    # Guard against over-broad matching: heartbeat/comment/show are NOT terminal.
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_170fcd74")
    for tool in ("kanban_heartbeat", "kanban_comment", "kanban_show"):
        messages = [
            {
                "role": "assistant",
                "content": "Let me note progress.",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": tool, "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "name": tool, "tool_call_id": "1", "content": "ok"},
        ]
        assert session_called_kanban_terminal(messages) is False, tool
        assert build_kanban_stop_nudge(messages=messages, attempts=0) is not None, tool

# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.




