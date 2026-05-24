"""Tests for kanban ↔ Linear issue linkage."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_linear as kl


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_test_key")
    monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_migration_adds_linear_columns(kanban_home):
    with kb.connect() as conn:
        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(tasks)")
        }
    assert "linear_issue_id" in cols
    assert "linear_issue_url" in cols


def test_create_child_inherits_parent_linear_link(kanban_home):
    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="umbrella", assignee="ops")
        kb.set_linear_link(
            conn,
            parent_id,
            "issue-parent-uuid",
            "https://linear.app/acme/issue/ENG-1",
            propagate=False,
        )
        child_id = kb.create_task(
            conn,
            title="child work",
            assignee="ops",
            parents=[parent_id],
        )
        child = kb.get_task(conn, child_id)
    assert child is not None
    assert child.linear_issue_id == "issue-parent-uuid"
    assert child.linear_issue_url == "https://linear.app/acme/issue/ENG-1"


def test_propagate_backfills_existing_children(kanban_home):
    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="umbrella", assignee="ops")
        child_id = kb.create_task(
            conn,
            title="child",
            assignee="ops",
            parents=[parent_id],
        )
        kb.set_linear_link(
            conn,
            parent_id,
            "issue-99",
            "https://linear.app/acme/issue/ENG-99",
            propagate=True,
        )
        child = kb.get_task(conn, child_id)
    assert child is not None
    assert child.linear_issue_id == "issue-99"


def test_child_override_not_overwritten_on_propagate(kanban_home):
    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="umbrella", assignee="ops")
        child_id = kb.create_task(
            conn,
            title="child",
            assignee="ops",
            parents=[parent_id],
        )
        kb.set_linear_link(
            conn,
            child_id,
            "issue-child-only",
            "https://linear.app/acme/issue/ENG-2",
            propagate=False,
        )
        kb.set_linear_link(
            conn,
            parent_id,
            "issue-parent",
            "https://linear.app/acme/issue/ENG-1",
            propagate=True,
        )
        child = kb.get_task(conn, child_id)
    assert child is not None
    assert child.linear_issue_id == "issue-child-only"


def test_ensure_reuses_existing_link(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="already linked", assignee="ops")
        kb.set_linear_link(
            conn,
            task_id,
            "existing-id",
            "https://linear.app/acme/issue/ENG-5",
            propagate=False,
        )
        with patch.object(kl, "_create_linear_issue") as mock_create:
            task = kl.ensure_linear_issue_for_task(
                conn,
                task_id,
                config=kl.LinearKanbanConfig(enabled=True, team="ENG"),
            )
    assert task.linear_issue_id == "existing-id"
    mock_create.assert_not_called()


def test_ensure_inherits_from_parent_instead_of_create(kanban_home):
    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="parent", assignee="ops")
        kb.set_linear_link(
            conn,
            parent_id,
            "parent-issue",
            "https://linear.app/acme/issue/ENG-9",
            propagate=False,
        )
        child_id = kb.create_task(
            conn,
            title="child",
            assignee="ops",
            parents=[parent_id],
        )
        # Simulate a child created before parent was linked.
        conn.execute(
            "UPDATE tasks SET linear_issue_id = NULL, linear_issue_url = NULL "
            "WHERE id = ?",
            (child_id,),
        )
        with patch.object(kl, "_create_linear_issue") as mock_create:
            task = kl.ensure_linear_issue_for_task(
                conn,
                child_id,
                config=kl.LinearKanbanConfig(enabled=True, team="ENG"),
            )
    assert task.linear_issue_id == "parent-issue"
    mock_create.assert_not_called()


def test_ensure_creates_and_propagates(kanban_home):
    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="umbrella", assignee="ops")
        child_id = kb.create_task(
            conn,
            title="child",
            assignee="ops",
            parents=[parent_id],
        )
        conn.execute(
            "UPDATE tasks SET linear_issue_id = NULL, linear_issue_url = NULL "
            "WHERE id = ?",
            (child_id,),
        )
        cfg = kl.LinearKanbanConfig(enabled=True, team="ENG")
        with patch.object(
            kl,
            "_create_linear_issue",
            return_value=("new-uuid", "https://linear.app/acme/issue/ENG-42"),
        ):
            parent = kl.ensure_linear_issue_for_task(conn, parent_id, config=cfg)
        child = kb.get_task(conn, child_id)
    assert parent.linear_issue_id == "new-uuid"
    assert child is not None
    assert child.linear_issue_id == "new-uuid"
