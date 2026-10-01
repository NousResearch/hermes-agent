"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    consecutive_stop_attempts,
    kanban_stop_max_attempts,
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


# ── Budget: only consecutive narrated stops count; the last nudge is restrictive ──


def _tc(name, i="1"):
    return {"id": i, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _work(n):
    """``n`` assistant tool-call rows (real progress) plus their tool results."""
    rows = []
    for i in range(n):
        rows.append({"role": "assistant", "content": "", "tool_calls": [_tc("terminal", str(i))]})
        rows.append({"role": "tool", "name": "terminal", "tool_call_id": str(i), "content": "ok"})
    return rows


def _nudge():
    return {"role": "user", "content": "[System: nudge]", "_kanban_stop_synthetic": True}


def test_budget_resets_when_worker_resumes_work(clear_kanban_env):
    """Fleet pattern: nudge → 20 tool calls → nudge → 20 tool calls → narrated stop.
    Before: attempts=2 ≥ max → None → rc=0 crash. Now the two mid-task nudges were answered
    with work, so the streak is 0 and the guard still fires."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_c13ad517")
    messages = [{"role": "user", "content": "work kanban task"}] + _work(5)
    messages += [_nudge()] + _work(20) + [_nudge()] + _work(20)
    messages.append({"role": "assistant", "content": "Let me find every required prop:"})
    assert consecutive_stop_attempts(messages, attempts=2) == 0
    nudge = build_kanban_stop_nudge(messages=messages, attempts=2)
    assert nudge is not None
    assert "FINAL" not in nudge


def test_consecutive_narrated_stops_exhaust_budget(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_x")
    messages = [{"role": "user", "content": "work"}] + _work(3)
    messages += [_nudge(), {"role": "assistant", "content": "Now I'll finish."}]
    # second nudge about to be issued: streak 1 → this is the last → restrictive text
    assert consecutive_stop_attempts(messages, attempts=1) == 1
    final = build_kanban_stop_nudge(messages=messages, attempts=1)
    assert final is not None and "FINAL" in final and "Do NOT resume" in final
    messages += [_nudge(), {"role": "assistant", "content": "Finishing now."}]
    assert consecutive_stop_attempts(messages, attempts=2) == 2
    assert build_kanban_stop_nudge(messages=messages, attempts=2) is None


def test_one_board_poke_does_not_count_as_progress(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_x")
    messages = [_nudge()] + _work(1) + [{"role": "assistant", "content": "ok, next:"}]
    assert consecutive_stop_attempts(messages, attempts=1) == 1


def test_bare_messages_fall_back_to_counter(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_x")
    assert build_kanban_stop_nudge(messages=[], attempts=2) is None
    assert build_kanban_stop_nudge(messages=[], attempts=0) is not None


def test_max_attempts_env(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_x")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE_MAX", "4")
    assert kanban_stop_max_attempts() == 4
    assert build_kanban_stop_nudge(messages=[], attempts=3) is not None
    assert build_kanban_stop_nudge(messages=[], attempts=4) is None
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE_MAX", "junk")
    assert kanban_stop_max_attempts() == 2
