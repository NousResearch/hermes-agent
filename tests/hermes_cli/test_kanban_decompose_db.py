"""Tests for kb.decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None, tenant=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        triage=True,
    )


def test_decompose_creates_children_and_promotes_root(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {
            "title": "research",
            "body": "look at prior art",
            "assignee": "researcher",
            "parents": [],
        },
        {
            "title": "build it",
            "body": "write code",
            "assignee": "engineer",
            "parents": [0],
        },
    ]
    with kb.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])

    # Root flipped to todo with orchestrator assignee, gated by children.
    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    # First child has no internal parents → ready on recompute_ready.
    assert c0.status == "ready"
    assert c0.assignee == "researcher"
    # Second child has parents=[0] → stays in todo until c0 completes.
    assert c1.status == "todo"
    assert c1.assignee == "engineer"


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kb.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert any("Decomposed into" in (c.body or "") for c in comments)
    assert any(ev.kind == "decomposed" for ev in events)


def test_decompose_inherits_project_authority_with_fresh_child_branches(kanban_home):
    with kb.connect() as conn:
        root_id = _create_triage(conn, title="project root", tenant="tenant-a")
        conn.execute(
            "UPDATE tasks SET project_id = ?, project_slug = ?, "
            "workspace_kind = 'worktree', workspace_path = ?, branch_name = ? "
            "WHERE id = ?",
            (
                "p_deadbeef",
                "project-p",
                f"/tmp/project-p/.worktrees/{root_id}",
                f"project-p/{root_id}-project-root",
                root_id,
            ),
        )
        conn.commit()

        child_ids = kb.decompose_triage_task(
            conn,
            root_id,
            root_assignee="orchestrator",
            children=[{"title": "child task", "parents": []}],
            author="decomposer",
        )
        assert child_ids is not None
        child = kb.get_task(conn, child_ids[0])

    assert child is not None
    assert child.tenant == "tenant-a"
    assert child.project_id == "p_deadbeef"
    assert child.project_slug == "project-p"
    assert child.workspace_kind == "worktree"
    assert child.workspace_path is None
    assert child.branch_name == f"project-p/{child.id}-child-task"


@pytest.mark.parametrize(
    "child_override",
    [
        {"workspace_path": "/tmp/project-p/.worktrees/root"},
        {"workspace_kind": "scratch"},
        {"workspace_kind": "bogus"},
    ],
)
def test_project_decompose_rejects_workspace_authority_overrides(
    kanban_home, child_override
):
    with kb.connect() as conn:
        root_id = _create_triage(conn, title="project root", tenant="tenant-a")
        root_path = f"/tmp/project-p/.worktrees/{root_id}"
        conn.execute(
            "UPDATE tasks SET project_id = ?, project_slug = ?, "
            "workspace_kind = 'worktree', workspace_path = ?, branch_name = ? "
            "WHERE id = ?",
            (
                "p_deadbeef",
                "project-p",
                root_path,
                f"project-p/{root_id}-project-root",
                root_id,
            ),
        )
        conn.commit()
        child = {"title": "rejected child", "parents": [], **child_override}

        with pytest.raises(ValueError, match="project-scoped child"):
            kb.decompose_triage_task(
                conn,
                root_id,
                root_assignee="orchestrator",
                children=[child],
                author="decomposer",
            )

        root = kb.get_task(conn, root_id)
        assert root is not None
        assert root.status == "triage"
        assert [task.id for task in kb.list_tasks(conn)] == [root_id]


@pytest.mark.parametrize("malformed", [None, 0, False, "", {}])
def test_decompose_db_rejects_falsy_non_list_parents(kanban_home, malformed):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="parents must be a list"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[{"title": "task A", "parents": malformed}],
            )


def test_decompose_db_rejects_boolean_and_duplicate_parent_indices(kanban_home):
    with kb.connect() as conn:
        bool_tid = _create_triage(conn, title="bool parent")
        with pytest.raises(ValueError, match="not a valid index"):
            kb.decompose_triage_task(
                conn,
                bool_tid,
                root_assignee="orch",
                children=[
                    {"title": "task A", "parents": []},
                    {"title": "task B", "parents": [False]},
                ],
            )

        duplicate_tid = _create_triage(conn, title="duplicate parent")
        with pytest.raises(ValueError, match="duplicate parent index"):
            kb.decompose_triage_task(
                conn,
                duplicate_tid,
                root_assignee="orch",
                children=[
                    {"title": "task A", "parents": []},
                    {"title": "task B", "parents": [0, 0]},
                ],
            )
