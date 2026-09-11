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
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_CLAIM_LOCK",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch






def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


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


# ── Regression: #95488 ─────────────────────────────────────────────────
# The old nudge was formatted as a prompt injection ("[System: ...]",
# "do not narrate intent", "immediately", "Never end a turn with only a
# promise"), so safety-trained models refused it and the dispatcher
# crash-looped. The nudge must read as a plain API contract, self-
# authenticate via the harness run id / claim lock, and vary on retry.


def test_nudge_has_no_injection_shape(clear_kanban_env):
    """The nudge must not carry prompt-injection markers (#95488)."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    nudge = build_kanban_stop_nudge(messages=[], attempts=0)
    assert nudge is not None
    assert not nudge.startswith("[System:")
    assert "do not narrate intent" not in nudge.lower()
    assert "immediately" not in nudge.lower()
    assert "never end a turn with only a promise" not in nudge.lower()


def test_nudge_self_authenticates_with_run_id(clear_kanban_env):
    """The nudge echoes the harness run id / claim lock when set (#95488)."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "run_123")
    clear_kanban_env.setenv("HERMES_KANBAN_CLAIM_LOCK", "lock_abc")
    nudge = build_kanban_stop_nudge(messages=[], attempts=0)
    assert nudge is not None
    assert "run_123" in nudge
    assert "lock_abc" in nudge


def test_nudge_self_authenticates_fallback_to_task_id(clear_kanban_env):
    """Without run id / claim lock, the nudge falls back to the task id."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    nudge = build_kanban_stop_nudge(messages=[], attempts=0)
    assert nudge is not None
    assert "t_46be8aa5" in nudge


def test_second_attempt_differs_from_first(clear_kanban_env):
    """The retry nudge must not be an identical replay (#95488)."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    first = build_kanban_stop_nudge(messages=[], attempts=0)
    second = build_kanban_stop_nudge(messages=[], attempts=1)
    assert first is not None
    assert second is not None
    assert first != second
    # Both still carry the terminal-tool contract.
    assert "kanban_complete" in second
    assert "kanban_block" in second


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




