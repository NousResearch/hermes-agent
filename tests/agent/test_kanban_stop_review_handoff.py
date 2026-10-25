"""Behavior contract: the turn-end stop guard must treat a successful
``kanban_request_review`` as a terminal handoff for the worker's own run.

Incident (Command Board 2026-09-17): the implementer delivered and called
``kanban_request_review`` (card correctly in ``review``). The stop guard's
terminal set was only {kanban_complete, kanban_block}, so the guard nudged the
still-running implementer session into a ``kanban_complete`` whose
``expected_run_id`` pointed at the already-closed run — a guaranteed rejection
("could not complete ... unknown id, stale run, or already terminal") that
burned turns and fed the crash loop.
"""

from __future__ import annotations

import pytest

from agent.kanban_stop import build_kanban_stop_nudge, session_called_kanban_terminal


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


REVIEW_MESSAGES = [
    {"role": "user", "content": "work kanban task"},
    {
        "role": "assistant",
        "content": "Delivered; handing off to review.",
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "kanban_request_review", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "name": "kanban_request_review", "tool_call_id": "1",
     "content": '{"ok": true, "status": "review"}'},
]


def test_successful_request_review_counts_as_terminal(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    assert session_called_kanban_terminal(REVIEW_MESSAGES) is True
    assert build_kanban_stop_nudge(messages=REVIEW_MESSAGES) is None


def test_failed_request_review_keeps_the_guard_on(clear_kanban_env):
    """A REJECTED review request is not a handoff: the worker still owes the
    board a terminal action, so the nudge must still fire."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    failed = [
        REVIEW_MESSAGES[0],
        REVIEW_MESSAGES[1],
        {"role": "tool", "name": "kanban_request_review", "tool_call_id": "1",
         "content": '{"error": "could not request review: artifact missing"}'},
    ]
    assert session_called_kanban_terminal(failed) is False
    assert build_kanban_stop_nudge(messages=failed) is not None


def test_successful_request_changes_counts_as_terminal(clear_kanban_env):
    """A reviewer's successful ``kanban_request_changes`` equally closes its
    own run (the card returns to the implementer lane)."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        REVIEW_MESSAGES[0],
        REVIEW_MESSAGES[1],
        {"role": "tool", "name": "kanban_request_changes", "tool_call_id": "1",
         "content": '{"ok": true, "status": "ready"}'},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None
