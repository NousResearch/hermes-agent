"""The goal-mode judge must see operator comments, not just the card body.

A human can waive or amend a card's acceptance criteria in a comment; both terminal-handoff
gates (the worker's kanban_complete/request-review tools and the CLI paths) used to judge
only title+body, so a waived criterion still rejected the handoff. The judged goal now
carries the card's recent human comments; the worker's own comments are its output, not
operator intent, and stay out — keyed on the card's assignee, never the caller's profile,
so a human closing a card from the CLI keeps their own waiver in front of the judge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _card(tmp_path, monkeypatch, *, assignee, profile, comments):
    """A goal-mode card authored under ``profile`` with ``comments`` on its thread."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", profile)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="Ship the checklist",
            body="Acceptance: the shipped list has >=40 items.", goal_mode=True,
            assignee=assignee)
        for author, body in comments:
            kb.add_comment(conn, tid, author, body)
    return tid


@pytest.fixture
def waived_goal_card(tmp_path, monkeypatch):
    return _card(tmp_path, monkeypatch, assignee="worker-bot", profile="worker-bot",
                 comments=[("worker-bot", "at 12 items, still adding"),
                           ("operator", "12 items is enough - the >=40 rule is waived."),
                           ("worker-bot", "wrapping up now")])


def _capture_goal(monkeypatch, judge_home, captured):
    """Patch the judge reachable through ``judge_home`` to record the goal text."""
    def _fake_judge(*, goal, last_response, **_kw):
        captured.append(goal)
        return "done", "ok", False, None, False

    monkeypatch.setattr(judge_home, "judge_goal", _fake_judge)
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda purpose: (object(), "judge-model"))


def _assert_operator_notes(goal: str) -> None:
    assert ">=40 items" in goal                       # the card itself stays the base
    assert "Operator notes (from card comments):" in goal
    assert "the >=40 rule is waived" in goal          # the human waiver reaches the judge
    assert "still adding" not in goal                 # the worker's own comments are excluded
    assert "wrapping up now" not in goal


def test_worker_tool_gate_judges_operator_comments(waived_goal_card, monkeypatch):
    from tools import kanban_tools

    captured: list[str] = []
    _capture_goal(monkeypatch, kanban_tools, captured)
    with kbc.connect() as conn:
        kanban_tools._goal_gate(
            "kanban_complete", conn, kb.get_task(conn, waived_goal_card), waived_goal_card,
            "shipped 12 items")
    assert len(captured) == 1
    _assert_operator_notes(captured[0])


def test_cli_gate_judges_operator_comments(waived_goal_card, monkeypatch):
    from hermes_cli import goals
    from hermes_cli import kanban

    captured: list[str] = []
    _capture_goal(monkeypatch, goals, captured)
    with kbc.connect() as conn:
        verdict, rejection = kanban._goal_mode_handoff_rejection(
            conn, kb.get_task(conn, waived_goal_card), "shipped 12 items")
    assert (verdict, rejection) == ("done", None)
    assert len(captured) == 1
    _assert_operator_notes(captured[0])


_SLY_DEMON_COMMENTS = [
    ("sly-demon", "at 12 items, still adding"),
    ("default", "12 items is enough - the >=40 rule is waived."),
    ("sly-demon", "wrapping up now"),
]


def test_worker_caller_sees_human_notes_not_assignee_output(tmp_path, monkeypatch):
    """Caller profile == the worker (assignee "sly-demon"): the human comment
    reaches the judge, the assignee's own output does not."""
    tid = _card(tmp_path, monkeypatch, assignee="sly-demon", profile="sly-demon",
                comments=_SLY_DEMON_COMMENTS)
    with kbc.connect() as conn:
        goal = kb.goal_text_with_operator_notes(conn, kb.get_task(conn, tid))
    assert "the >=40 rule is waived" in goal
    assert "still adding" not in goal
    assert "wrapping up now" not in goal


def test_cli_operator_caller_gets_identical_notes(tmp_path, monkeypatch):
    """A human closing the card from the CLI (caller profile "default") gets the
    same judged goal as the worker: their own waiver is kept, the assignee's
    comments stay out."""
    tid = _card(tmp_path, monkeypatch, assignee="sly-demon", profile="default",
                comments=_SLY_DEMON_COMMENTS)
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        as_operator = kb.goal_text_with_operator_notes(conn, task)
        monkeypatch.setenv("HERMES_PROFILE", "sly-demon")
        as_worker = kb.goal_text_with_operator_notes(conn, task)
    assert as_operator == as_worker
    assert "the >=40 rule is waived" in as_operator
    assert "still adding" not in as_operator
    assert "wrapping up now" not in as_operator


def test_unassigned_card_keeps_every_comment(tmp_path, monkeypatch):
    """No assignee on the card: nothing is excluded as worker output."""
    tid = _card(tmp_path, monkeypatch, assignee=None, profile="default",
                comments=_SLY_DEMON_COMMENTS)
    with kbc.connect() as conn:
        goal = kb.goal_text_with_operator_notes(conn, kb.get_task(conn, tid))
    assert "still adding" in goal
    assert "the >=40 rule is waived" in goal
    assert "wrapping up now" in goal
