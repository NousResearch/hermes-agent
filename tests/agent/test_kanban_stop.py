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


# ── Reviewer terminal transitions (t_e90216b9) ───────────────────────
# kanban_request_changes / kanban_request_review close the reviewer's run with
# outcome='changes_requested' / 'review_requested' — a reviewer legitimately
# never calls kanban_complete when rejecting work, so both count as terminal
# for the stop guard.

@pytest.mark.parametrize("tool", ["kanban_request_changes", "kanban_request_review"])
def test_no_nudge_after_reviewer_transitions(clear_kanban_env, tool):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "927")
    messages = [
        {
            "role": "assistant",
            "content": "rejecting: reproduced defect",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": tool, "arguments": '{"reason": "defect"}'},
                }
            ],
        },
        {"role": "tool", "name": tool, "tool_call_id": "1", "content": '{"ok": true}'},
    ]
    assert session_called_kanban_terminal(messages) is True
    # The DB gate is bypassed entirely by the history check here.
    assert build_kanban_stop_nudge(messages=messages, attempts=0) is None
    assert build_kanban_stop_nudge(messages=messages, attempts=1) is None


def test_reviewer_transition_unrecognized_tool_still_nudges(clear_kanban_env, monkeypatch):
    """A tool name outside the terminal set (e.g. kanban_comment) must not
    suppress the guard."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    monkeypatch.setattr(
        "agent.kanban_stop._own_run_is_stale", lambda task_id, run_id: False)
    messages = [
        {
            "role": "assistant",
            "content": "commented",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_comment", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_comment", "tool_call_id": "1", "content": '{"ok": true}'},
    ]
    assert session_called_kanban_terminal(messages) is False
    assert build_kanban_stop_nudge(messages=messages, attempts=0) is not None


# ── Run-ownership gate (t_e90216b9) ──────────────────────────────────

def test_no_nudge_when_own_run_stale(clear_kanban_env, monkeypatch):
    """A worker whose own run already ended (reviewer after request_changes)
    must not be nudged even with no terminal call in message history."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_10d429e2")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "927")
    monkeypatch.setattr(
        "agent.kanban_stop._own_run_is_stale", lambda task_id, run_id: True)
    messages = [
        {"role": "user", "content": "review kanban task t_10d429e2"},
        {"role": "assistant", "content": "review narrative, then stop"},
    ]
    assert build_kanban_stop_nudge(messages=messages, attempts=0) is None


def test_nudge_when_own_run_live(clear_kanban_env, monkeypatch):
    """A live run with no terminal call still gets the nudge (true positive)."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "941")
    monkeypatch.setattr(
        "agent.kanban_stop._own_run_is_stale", lambda task_id, run_id: False)
    messages = [
        {"role": "user", "content": "work kanban task"},
        {"role": "assistant", "content": "narrating instead of finishing"},
    ]
    assert build_kanban_stop_nudge(messages=messages, attempts=0) is not None


def test_no_run_id_env_keeps_legacy_behavior(clear_kanban_env, monkeypatch):
    """Without HERMES_KANBAN_RUN_ID the ownership gate is skipped entirely
    (legacy workers / interactive sessions) and the nudge fires as before."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    # Must never be consulted:
    monkeypatch.setattr(
        "agent.kanban_stop._own_run_is_stale",
        lambda task_id, run_id: (_ for _ in ()).throw(AssertionError("must not be called")))
    messages = [
        {"role": "user", "content": "work kanban task"},
        {"role": "assistant", "content": "narrating instead of finishing"},
    ]
    assert build_kanban_stop_nudge(messages=messages, attempts=0) is not None


def test_db_read_failure_fails_open(clear_kanban_env, monkeypatch):
    """A board read failure inside the ownership helper must fail OPEN
    (return False → nudge may still fire), so a transient DB error never
    silently disables the guard for a genuinely stalled worker."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "941")

    import agent.kanban_stop as ks

    class _Boom:
        def connect(self, board=None):
            raise RuntimeError("db locked")

    monkeypatch.setattr(ks, "_own_run_is_stale", ks._own_run_is_stale)  # real impl
    real_connect = None
    # Patch the connect symbol the helper looks up at call time (lazy import
    # from hermes_cli.kanban_db_connect).
    import hermes_cli.kanban_db_connect as kbc
    real_connect = kbc.connect
    monkeypatch.setattr(kbc, "connect", lambda board=None: (_ for _ in ()).throw(RuntimeError("db locked")))
    try:
        assert ks._own_run_is_stale("t_abc", 941) is False
    finally:
        monkeypatch.setattr(kbc, "connect", real_connect)



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
