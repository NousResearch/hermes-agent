import json
import logging

import pytest

from hermes_cli import kanban_db as kb
from tools import kanban_tools


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    return home


def test_goal_judge_valid_structured_verdict_allows_completion_without_exception_log(
    kanban_home,
    monkeypatch,
    caplog,
):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="goal-gated repair",
            assignee="worker",
            body="Complete only when the judge accepts the evidence.",
            goal_mode=True,
        )

    monkeypatch.setattr(kanban_tools, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(
        kanban_tools,
        "judge_goal",
        lambda **_kwargs: ("done", "evidence satisfies the card", False, None),
    )
    caplog.set_level(logging.WARNING, logger="tools.kanban_tools")

    payload = json.loads(
        kanban_tools._handle_complete(
            {"task_id": task_id, "summary": "Evidence satisfies the card."}
        )
    )

    assert payload["ok"] is True
    assert "goal judge check failed" not in caplog.text
    assert "too many values to unpack" not in caplog.text
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "done"


def test_goal_judge_invalid_result_fails_closed_without_completing(
    kanban_home,
    monkeypatch,
):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="goal-gated repair",
            assignee="worker",
            body="Completion requires a typed judge result.",
            goal_mode=True,
        )

    monkeypatch.setattr(kanban_tools, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(
        kanban_tools,
        "judge_goal",
        lambda **_kwargs: {"verdict": "done", "reason": "dict is not the contract"},
    )

    payload = json.loads(
        kanban_tools._handle_complete(
            {"task_id": task_id, "summary": "Attempted completion."}
        )
    )

    assert "error" in payload
    assert "invalid goal judge result" in payload["error"]
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "ready"
