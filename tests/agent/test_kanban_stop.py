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






# ---------------------------------------------------------------------------
# Truncation early-exit sites route through the same nudge
# ---------------------------------------------------------------------------


class TestMaybeNudgeKanbanStop:
    """The output-length / truncated-call-refusal early returns in the
    conversation loop exit BEFORE the finish_reason=stop guard fires, so a
    kanban worker session could end rc=0 without kanban_complete /
    kanban_block (dispatcher records protocol_violation). Those sites must
    consult _maybe_nudge_kanban_stop before returning."""

    @pytest.fixture
    def worker_env(self, clear_kanban_env):
        clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_nudge1")
        return clear_kanban_env

    def _loop_agent(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            _kanban_stop_nudges=0,
            _session_messages=None,
            _emit_status=lambda *_a, **_k: None,
        )

    def test_appends_nudge_and_signals_continue(self, worker_env):
        from agent.conversation_loop import _maybe_nudge_kanban_stop

        agent = self._loop_agent()
        messages = [
            {"role": "user", "content": "work the task"},
            {
                "role": "assistant",
                "content": "I will now write the report.",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "name": "terminal", "tool_call_id": "1", "content": "ok"},
        ]

        proceed = _maybe_nudge_kanban_stop(
            agent, messages, "Response truncated due to output length limit"
        )

        assert proceed is True
        assert agent._kanban_stop_nudges == 1
        # The synthetic user nudge is appended in place...
        assert messages[-1]["role"] == "user"
        assert "kanban_complete" in messages[-1]["content"]
        # ...and the trailing tool result was closed first (strict role
        # alternation: no bare user turn directly after a tool result).
        assert messages[-2]["role"] != "tool"
        assert agent._session_messages is messages

    def test_bounded_at_two_attempts(self, worker_env):
        from agent.conversation_loop import _maybe_nudge_kanban_stop

        agent = self._loop_agent()
        agent._kanban_stop_nudges = 2
        messages = [{"role": "user", "content": "work the task"}]

        assert _maybe_nudge_kanban_stop(agent, messages, "again") is False
        assert agent._kanban_stop_nudges == 2  # unchanged

    def test_inert_outside_kanban_workers(self, clear_kanban_env):
        from agent.conversation_loop import _maybe_nudge_kanban_stop

        agent = self._loop_agent()
        messages = [{"role": "user", "content": "normal chat"}]

        assert _maybe_nudge_kanban_stop(agent, messages, "truncated") is False
        assert messages[-1]["role"] == "user"
        assert len(messages) == 1

    def test_no_nudge_when_already_completed(self, worker_env):
        from agent.conversation_loop import _maybe_nudge_kanban_stop

        agent = self._loop_agent()
        messages = [
            {"role": "user", "content": "work the task"},
            {
                "role": "assistant",
                "content": "done",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "kanban_complete", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "ok"},
        ]

        assert _maybe_nudge_kanban_stop(agent, messages, "truncated") is False

    def test_turn_retry_state_carries_nudge_flag(self):
        from agent.turn_retry_state import TurnRetryState

        state = TurnRetryState()
        assert state.restart_with_kanban_stop_nudge is False
        state.restart_with_kanban_stop_nudge = True
        assert state.restart_with_kanban_stop_nudge is True
