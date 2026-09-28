"""The goal-mode judge must see operator comments, not just the card body.

A human can waive or amend a card's acceptance criteria in a comment; both terminal-handoff
gates (the worker's kanban_complete/request-review tools and the CLI paths) used to judge
only title+body, so a waived criterion still rejected the handoff. The judged goal now
carries the card's recent human comments; the worker's own comments are its output, not
operator intent, and stay out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def waived_goal_card(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="Ship the checklist",
            body="Acceptance: the shipped list has >=40 items.", goal_mode=True)
        kb.add_comment(conn, tid, "worker-bot", "at 12 items, still adding")
        kb.add_comment(conn, tid, "operator", "12 items is enough - the >=40 rule is waived.")
        kb.add_comment(conn, tid, "worker-bot", "wrapping up now")
    return tid


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
