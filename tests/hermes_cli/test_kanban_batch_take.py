"""Regression tests for selected-card batch planning."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_batch_take as batch
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _response(payload: dict):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(payload)
    return response


def test_batch_take_orders_only_model_selected_conflict(kanban_home):
    with kb.connect() as conn:
        first = kb.create_task(conn, title="Change shared API", body="Edit api.py")
        second = kb.create_task(conn, title="Update API callers", body="Edit clients")
        independent = kb.create_task(conn, title="Write release notes", body="Docs only")

    plan = {"edges": [{"before": first, "after": second, "reason": "shared API"}]}
    with patch("agent.auxiliary_client.call_llm", return_value=_response(plan)):
        outcome = batch.plan_and_take([first, second, independent])

    assert outcome.ok, outcome.reason
    assert outcome.promoted == [first, independent]
    assert outcome.waiting == [second]
    assert outcome.edges == [{"before": first, "after": second, "reason": "shared API"}]
    with kb.connect() as conn:
        assert kb.get_task(conn, first).status == "ready"
        assert kb.get_task(conn, second).status == "todo"
        assert kb.get_task(conn, independent).status == "ready"
        links = conn.execute("SELECT parent_id, child_id FROM task_links").fetchall()
    assert any(row["parent_id"] == first and row["child_id"] == second for row in links)


def test_batch_take_does_not_change_cards_when_planner_output_is_bad(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Keep independent")

    with patch("agent.auxiliary_client.call_llm", return_value=_response({"wrong": []})):
        outcome = batch.plan_and_take([task_id])

    assert not outcome.ok
    assert "malformed" in outcome.reason
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "ready"
        assert conn.execute("SELECT COUNT(*) FROM task_links").fetchone()[0] == 0
