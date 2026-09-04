"""Kanban stop-nudge on truncation early-exits (``agent/turn_truncation.py``).

The truncation early-exits (output-length ceiling, truncated tool-call refusal,
thinking/repetition abort, roll-back, first-response failure) end the turn with a
partial result BEFORE the finish_reason=stop stop-gates run, so a kanban worker
session could end rc=0 without ``kanban_complete`` / ``kanban_block`` and the
dispatcher records ``protocol_violation``. ``_Trunc.end_turn_with_kanaban_nudge``
routes those exits through the same bounded nudge policy the text-stop path uses,
arming ``TurnRetryState.restart_with_kanban_stop_nudge`` so the loop re-enters with
a fresh API call against the nudged history.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
)
from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import _Trunc, recover_from_truncation


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _trunc_state(agent, messages, _retry=None):
    """A minimal ``_Trunc`` with only the fields the nudge path touches."""
    return _Trunc(
        agent=agent, response=None, finish_reason="length",
        conversation_history=None, api_call_count=1,
        effective_task_id=None, current_turn_user_idx=0,
        messages=messages, length_continue_retries=4,
        truncated_response_parts=[], truncated_tool_call_retries=4,
        retry_count=0, compression_attempts=0, _retry=_retry or TurnRetryState(),
    )


def _agent(persisted):
    return SimpleNamespace(
        _kanban_stop_nudges=0,
        _session_messages=None,
        _emit_status=lambda *_a, **_k: None,
        _cleanup_task_resources=lambda *_a, **_k: None,
        _persist_session=lambda *a, **k: persisted.append(a),
        log_prefix="",
    )


def _worker_messages():
    return [
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


class TestEndTurnWithKanbanNudge:
    def test_appends_nudge_and_arms_restart(self, clear_kanban_env):
        clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_nudge1")
        assert kanban_stop_nudge_enabled() is True

        persisted = []
        agent = _agent(persisted)
        messages = _worker_messages()
        retry = TurnRetryState()
        st = _trunc_state(agent, messages, retry)

        verdict = st.end_turn_with_kanban_nudge(
            "Response truncated due to output length limit"
        )

        # Nudge fired: loop must re-enter, not return a partial result.
        assert verdict.action == "break"
        assert retry.restart_with_kanban_stop_nudge is True
        assert agent._kanban_stop_nudges == 1
        # The synthetic user nudge is appended in place...
        assert messages[-1]["role"] == "user"
        assert "kanban_complete" in messages[-1]["content"]
        assert messages[-1].get("_kanban_stop_synthetic") is True
        # ...and the trailing tool result was closed first (strict role
        # alternation: no bare user turn directly after a tool result).
        assert messages[-2]["role"] != "tool"
        assert agent._session_messages is messages
        # The turn did NOT end: no partial-result persistence on this path.
        assert persisted == []

    def test_bounded_at_max_attempts_falls_through_to_end_turn(self, clear_kanban_env):
        clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_nudge2")
        persisted = []
        agent = _agent(persisted)
        agent._kanban_stop_nudges = 2  # build_kanban_stop_nudge default ceiling
        messages = [{"role": "user", "content": "work the task"}]
        retry = TurnRetryState()
        st = _trunc_state(agent, messages, retry)

        verdict = st.end_turn_with_kanban_nudge("truncated again")

        # No nudge left: normal partial-result end (persisted, turn over).
        assert verdict.action == "return"
        assert verdict.result is not None and verdict.result["partial"] is True
        assert retry.restart_with_kanban_stop_nudge is False
        assert agent._kanban_stop_nudges == 2  # unchanged
        assert len(persisted) == 1

    def test_inert_outside_kanban_workers(self, clear_kanban_env):
        persisted = []
        agent = _agent(persisted)
        messages = [{"role": "user", "content": "normal chat"}]
        st = _trunc_state(agent, messages)

        verdict = st.end_turn_with_kanban_nudge("truncated")

        assert verdict.action == "return"
        assert len(messages) == 1  # nothing appended
        assert len(persisted) == 1

    def test_no_nudge_when_already_completed(self, clear_kanban_env):
        clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_nudge3")
        persisted = []
        agent = _agent(persisted)
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
        st = _trunc_state(agent, messages)

        verdict = st.end_turn_with_kanban_nudge("truncated")

        assert verdict.action == "return"
        assert agent._kanban_stop_nudges == 0

    def test_restart_flag_consumed_by_apply_retry_restarts(self, clear_kanban_env):
        from agent.turn_iteration_prep import apply_retry_restarts

        agent = SimpleNamespace(iteration_budget=SimpleNamespace(refund=lambda: None))
        retry = TurnRetryState()
        retry.restart_with_kanban_stop_nudge = True

        verdict = apply_retry_restarts(
            agent, _retry=retry, response=None, interrupted=False,
            messages=[], conversation_history=None, user_message=None,
            api_kwargs=None, current_turn_user_idx=0, final_response=None,
            retry_count=0, api_call_count=1, length_continue_retries=0,
            _preflight_compression_blocked=False, _turn_exit_reason=None,
        )

        assert verdict.action == "continue"
        assert retry.restart_with_kanban_stop_nudge is False  # consumed, no loop

    def test_first_response_failure_also_nudged(self, clear_kanban_env):
        """The first-message-truncated failure exit takes the same nudge path."""
        clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_nudge4")
        persisted = []
        agent = _agent(persisted)
        messages = [{"role": "user", "content": "work the task"}]
        retry = TurnRetryState()

        st = _trunc_state(agent, messages, retry)
        # Simulate the recover_from_truncation tail: single-message history.
        verdict = st.end_turn_with_kanban_nudge(
            "First response truncated due to output length limit", failed=True
        )

        assert verdict.action == "break"
        assert retry.restart_with_kanban_stop_nudge is True
