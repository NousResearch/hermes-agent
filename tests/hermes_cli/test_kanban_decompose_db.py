"""Tests for kb.decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb


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
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
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


def test_decompose_preserves_project_scope_with_unique_worktrees(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pdb.connect_closing() as pconn:
        project_id = pdb.create_project(
            pconn, name="Widget Project", primary_path=str(repo)
        )

    kb.create_board("scoped", name="Scoped", project_id=project_id)
    with kb.connect(board="scoped") as conn:
        root_id = kb.create_task(
            conn,
            title="Refactor widget project",
            triage=True,
            board="scoped",
        )
        child_ids = kb.decompose_triage_task(
            conn,
            root_id,
            root_assignee="orchestrator",
            children=[
                {"title": "audit widget", "parents": []},
                {"title": "refactor widget", "parents": [0]},
                {"title": "verify widget", "parents": [1]},
            ],
            author="decomposer",
        )

    assert child_ids is not None
    with kb.connect(board="scoped") as conn:
        children = [kb.get_task(conn, child_id) for child_id in child_ids]

    assert {child.workspace_path for child in children if child is not None} == {
        str(repo / ".worktrees" / child_id) for child_id in child_ids
    }
    for child in children:
        assert child is not None
        assert child.project_id == project_id
        assert child.workspace_kind == "worktree"
        assert child.branch_name is not None
        assert child.branch_name.startswith(f"widget-project/{child.id}-")




