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
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "{\"ok\": true}"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None



def test_final_kanban_nudge_requires_structured_terminal_call(clear_kanban_env):
    """The last bounded nudge must force a real terminal ToolCall, not narration/status."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_final_guard")

    messages = [
        {"role": "assistant", "content": "I will check the task."},
    ]

    nudge = build_kanban_stop_nudge(
        messages=messages,
        attempts=1,
        max_attempts=2,
    )

    assert nudge is not None
    assert "FINAL Kanban terminal guard" in nudge
    assert "MUST contain exactly one structured terminal Kanban tool call" in nudge
    assert "Do NOT reply with plain text" in nudge
    assert "do NOT" in nudge and "kanban_show" in nudge
    assert "kanban_complete" in nudge
    assert "kanban_request_review" in nudge
    assert "kanban_request_changes" in nudge
    assert "kanban_block" in nudge


def test_kanban_nudge_still_stops_at_bound(clear_kanban_env):
    """The guard remains bounded; it never becomes an unbounded continuation loop."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_final_guard")

    assert build_kanban_stop_nudge(
        messages=[{"role": "assistant", "content": "still no tool"}],
        attempts=2,
        max_attempts=2,
    ) is None

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
        ("kanban_schedule", "worker parking the card on a timed wait"),
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
        {"role": "tool", "name": tool_name, "tool_call_id": "1", "content": "{\"ok\": true}"},
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


@pytest.mark.parametrize(
    "content",
    [
        "done",
        "ok",
        '{"error": "failed"}',
        '{"ok": false}',
        "{broken json",
    ],
)
def test_failed_or_unknown_terminal_tool_result_is_not_success(content, clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_fail")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "kanban_complete",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "1",
            "content": content,
        },
    ]
    assert session_called_kanban_terminal(messages) is False


def test_terminal_result_requires_unique_matching_tool_call_id(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_strict")
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "1",
            "type": "function",
            "function": {"name": "kanban_complete", "arguments": "{}"},
        }],
    }

    assert session_called_kanban_terminal([call]) is False
    assert session_called_kanban_terminal([
        call,
        {"role": "tool", "name": "kanban_complete", "content": '{"ok": true}'},
    ]) is False
    assert session_called_kanban_terminal([
        call,
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "2", "content": '{"ok": true}'},
    ]) is False
    assert session_called_kanban_terminal([
        call,
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": '{"ok": true}'},
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": '{"error": "failed"}'},
    ]) is False
    assert session_called_kanban_terminal([
        call,
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": '{"ok": true}'},
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": '{"ok": true}'},
    ]) is False


def test_terminal_result_requires_matching_tool_name(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_name")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "1",
                "type": "function",
                "function": {"name": "kanban_complete", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "name": "kanban_block",
            "tool_call_id": "1",
            "content": '{"ok": true}',
        },
    ]
    assert session_called_kanban_terminal(messages) is False


def test_terminal_id_must_be_unique_across_all_tools(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_global_id")

    terminal_call = {
        "id": "shared",
        "type": "function",
        "function": {"name": "kanban_complete", "arguments": "{}"},
    }

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [terminal_call],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "shared",
                "type": "function",
                "function": {"name": "other_tool", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "shared",
            "content": '{"ok": true}',
        },
    ]

    assert session_called_kanban_terminal(messages) is False


def test_nonstandard_json_constants_are_not_success(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_strict_json")

    for value in ("NaN", "Infinity", "-Infinity"):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "name": "kanban_complete",
                "tool_call_id": "1",
                "content": f'{{"ok": true, "x": {value}}}',
            },
        ]
        assert session_called_kanban_terminal(messages) is False
