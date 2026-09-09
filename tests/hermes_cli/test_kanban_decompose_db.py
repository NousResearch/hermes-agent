"""Tests for decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc


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
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kbc.connect() as conn:
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kbc.connect() as conn:
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
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert any("Decomposed into" in (c.body or "") for c in comments)
    assert any(ev.kind == "decomposed" for ev in events)


def test_decompose_preserves_project_execution_metadata(kanban_home, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        kb, "_resolve_project_link",
        lambda conn, project_id, source_task_id, workspace_kind, workspace_path: (
            project_id or "project-1",
            type("Project", (), {
                "id": project_id or "project-1", "slug": "project-1", "primary_path": str(repo),
            })(),
            str(repo),
            "worktree",
        ),
    )
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="project root",
            triage=True,
            project_id="project-1",
            workspace_kind="worktree",
            workspace_path=str(repo / ".worktrees" / "root"),
            branch_name="continuation/root",
            skills=["verification-gate"],
            max_runtime_seconds=300,
            max_retries=2,
            model_override="claude-sonnet-5",
            provider_override="anthropic",
            reasoning_effort="high",
            goal_mode=True,
            goal_max_turns=12,
            session_id="session-root",
        )
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=[{"title": "child", "assignee": "builder"}],
        )
        child = kb.get_task(conn, child_ids[0])

    assert child is not None
    assert child.project_id == "project-1"
    assert child.workspace_kind == "worktree"
    assert child.workspace_path == str(repo / ".worktrees" / child.id)
    assert child.branch_name.startswith("project-1/")
    assert child.skills == ["verification-gate"]
    assert child.idempotency_key == f"decompose:{tid}:0"
    assert child.max_runtime_seconds == 300
    assert child.max_retries == 2
    assert child.model_override == "claude-sonnet-5"
    assert child.provider_override == "anthropic"
    assert child.reasoning_effort == "high"
    assert child.goal_mode is True
    assert child.goal_max_turns == 12
    assert child.session_id == "session-root"


def test_project_link_recovers_from_canonical_task_when_registry_is_unavailable(
    kanban_home, monkeypatch, tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    workspace = repo / ".worktrees" / "implementation-root"
    with kbc.connect() as conn:
        root_id = kb.create_task(
            conn,
            title="project-linked root",
            workspace_kind="worktree",
            workspace_path=str(workspace),
            branch_name="implementation/root",
        )
        conn.execute("UPDATE tasks SET project_id = ? WHERE id = ?", ("p_demo", root_id))
        conn.commit()

        monkeypatch.setattr(
            "hermes_cli.projects_db.get_project",
            lambda _conn, _ident: None,
        )
        project_id, project, project_repo, workspace_kind = kb._resolve_project_link(
            conn, "p_demo", root_id, "worktree", None,
        )

    assert project_id == "p_demo"
    assert project is not None
    assert project.primary_path == str(repo)
    assert project_repo == str(repo)
    assert workspace_kind == "worktree"




