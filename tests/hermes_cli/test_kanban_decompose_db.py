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


def _create_triage(
    conn,
    title="rough idea",
    body=None,
    assignee=None,
    tenant=None,
    workspace_kind="scratch",
    workspace_path=None,
):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
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


def test_decompose_returns_none_when_task_missing(kanban_home):
    with kb.connect() as conn:
        result = kb.decompose_triage_task(
            conn,
            "nonexistent",
            root_assignee="orch",
            children=[{"title": "x"}],
            author="me",
        )
    assert result is None


def test_decompose_returns_none_when_task_not_in_triage(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="already a real task")  # not triage
        result = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "x"}],
            author="me",
        )
    assert result is None


def test_decompose_empty_children_returns_none(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        result = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[],
            author="me",
        )
    assert result is None


def test_decompose_rejects_self_parent(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="cannot list itself"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[{"title": "x", "parents": [0]}],
                author="me",
            )


def test_decompose_rejects_out_of_range_parent(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="not a valid index"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[{"title": "x", "parents": [5]}],
                author="me",
            )


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


def test_decompose_derives_workstream_tenant_from_root_title(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(
            conn,
            title="Costing Enhancements — ClickUp 86d30zeaz",
            workspace_kind="worktree",
            workspace_path="/root/worktrees/watertyt-monorepo/feat/costing-enhancements-86d30zeaz",
        )
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "write spec", "assignee": "tech-lead"}],
            author="alice",
        )

    assert child_ids is not None
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        child = kb.get_task(conn, child_ids[0])
        events = kb.list_events(conn, tid)

    assert root is not None
    assert child is not None
    assert root.tenant == "Costing Enhancements — ClickUp 86d30zeaz"
    assert child.tenant == root.tenant
    assert child.workspace_kind == "worktree"
    assert child.workspace_path == root.workspace_path
    decompose_event = next(ev for ev in events if ev.kind == "decomposed")
    assert decompose_event.payload is not None
    assert decompose_event.payload["tenant"] == root.tenant
    assert decompose_event.payload["workspace_path"] == root.workspace_path


def test_create_task_child_inherits_parent_workstream_and_worktree(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(
            conn,
            title="Xero Integrations Framework",
            workspace_kind="worktree",
            workspace_path="/root/worktrees/the-knight-monorepo/feat/xero-integrations-framework",
        )
        child = kb.create_task(conn, title="implement connection UI", parents=[parent])

    with kb.connect() as conn:
        parent_task = kb.get_task(conn, parent)
        child_task = kb.get_task(conn, child)

    assert parent_task is not None
    assert child_task is not None
    assert parent_task.tenant == "Xero Integrations Framework"
    assert child_task.tenant == "Xero Integrations Framework"
    assert child_task.workspace_kind == "worktree"
    assert child_task.workspace_path == parent_task.workspace_path
