"""Kanban <-> Projects integration: project-linked tasks get a deterministic
worktree path + branch instead of the random ``wt/<task-id>`` fallback."""

from __future__ import annotations

import os

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb


@pytest.fixture
def kanban_conn(tmp_path):
    c = kb.connect(db_path=tmp_path / "kanban.db")
    try:
        yield c
    finally:
        c.close()


def _make_project(name="Web App", repo="/tmp/webapp"):
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name=name, folders=[repo])
        return pdb.get_project(pc, pid)


def test_project_linked_task_gets_deterministic_worktree_and_branch(kanban_conn):
    proj = _make_project()
    tid = kb.create_task(kanban_conn, title="Add login", project_id=proj.slug)
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id == proj.id
    assert task.workspace_kind == "worktree"
    # Worktree dir anchored under the project's primary repo, keyed on task id.
    assert task.workspace_path == os.path.join(proj.primary_path, ".worktrees", tid)
    # Deterministic branch: <slug>/<task-id>-<title-slug>. NOT a random wt/...
    assert task.branch_name == f"{proj.slug}/{tid}-add-login"
    assert not task.branch_name.startswith("wt/")


def test_explicit_branch_overrides_project_default(kanban_conn):
    proj = _make_project()
    tid = kb.create_task(
        kanban_conn,
        title="x",
        project_id=proj.slug,
        workspace_kind="worktree",
        branch_name="feature/custom",
    )
    task = kb.get_task(kanban_conn, tid)
    assert task.branch_name == "feature/custom"


def test_unlinked_task_unchanged(kanban_conn):
    tid = kb.create_task(kanban_conn, title="plain")
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id is None
    assert task.workspace_kind == "scratch"
    # No branch is persisted — the worker still owns the wt/<id> fallback for
    # genuinely ad-hoc worktree tasks, but unlinked scratch tasks have none.
    assert task.branch_name is None


def test_project_source_fallback_is_tenant_bound(kanban_conn, monkeypatch):
    source_id = kb.create_task(kanban_conn, title="Tenant A source", tenant="tenant-a")
    kanban_conn.execute(
        "UPDATE tasks SET project_id = ?, workspace_kind = 'worktree', "
        "workspace_path = ?, branch_name = ? WHERE id = ?",
        (
            "project-p",
            f"/tmp/tenant-bound/.worktrees/{source_id}",
            f"project-p/{source_id}-tenant-a-source",
            source_id,
        ),
    )
    kanban_conn.commit()
    source = kb.get_task(kanban_conn, source_id)
    assert source is not None
    assert source.workspace_kind == "worktree"
    monkeypatch.setattr(
        pdb,
        "connect_closing",
        lambda: (_ for _ in ()).throw(OSError("project store unavailable")),
    )
    monkeypatch.setattr(
        pdb,
        "get_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("project store unavailable")
        ),
    )

    with pytest.raises(ValueError, match="project not found or unavailable"):
        kb.create_task(
            kanban_conn,
            title="Tenant B child",
            tenant="tenant-b",
            project_id="project-p",
            project_source_task_id=source_id,
            idempotency_key="must-not-be-queried-cross-tenant",
        )

    assert [task.id for task in kb.list_tasks(kanban_conn)] == [source_id]


def _seed_cross_profile_project_source(kanban_conn, *, tenant="tenant-a"):
    source_id = kb.create_task(kanban_conn, title="Canonical source", tenant=tenant)
    kanban_conn.execute(
        "UPDATE tasks SET project_id = ?, project_slug = ?, "
        "workspace_kind = 'worktree', workspace_path = ?, branch_name = ? "
        "WHERE id = ?",
        (
            "p_deadbeef",
            "project-p",
            f"/tmp/project-p/.worktrees/{source_id}",
            f"project-p/{source_id}-canonical-source",
            source_id,
        ),
    )
    kanban_conn.commit()
    return source_id


def _disable_project_store(monkeypatch):
    monkeypatch.setattr(
        pdb,
        "connect_closing",
        lambda: (_ for _ in ()).throw(OSError("project store unavailable")),
    )


def test_project_source_fallback_requires_and_carries_canonical_slug(
    kanban_conn, monkeypatch
):
    source_id = _seed_cross_profile_project_source(kanban_conn)
    _disable_project_store(monkeypatch)

    child_id = kb.create_task(
        kanban_conn,
        title="Child task",
        tenant="tenant-a",
        project_id="p_deadbeef",
        project_source_task_id=source_id,
    )

    child = kb.get_task(kanban_conn, child_id)
    assert child is not None
    assert child.project_id == "p_deadbeef"
    assert child.project_slug == "project-p"
    assert child.workspace_path == f"/tmp/project-p/.worktrees/{child_id}"
    assert child.branch_name == f"project-p/{child_id}-child-task"


@pytest.mark.parametrize(
    ("workspace_path", "branch_name"),
    [
        ("/tmp/project-p/.worktrees/{id}", None),
        ("/tmp/project-p/.worktrees/{id}", "unrelated/{id}-canonical-source"),
        ("/tmp/project-p/.worktrees/{id}", "project-p/wrong-task-leaf"),
        (
            "/tmp/project-p/.worktrees/{id}",
            "project-p/{id}-canonical/source",
        ),
        (
            "/tmp/project-p/.worktrees/{id}",
            "project-p/{id}-canonical/../escape",
        ),
        (
            "/tmp/project-p/.worktrees/../.worktrees/{id}",
            "project-p/{id}-canonical-source",
        ),
    ],
)
def test_project_source_fallback_rejects_malformed_authority_evidence(
    kanban_conn, monkeypatch, workspace_path, branch_name
):
    source_id = _seed_cross_profile_project_source(kanban_conn)
    kanban_conn.execute(
        "UPDATE tasks SET workspace_path = ?, branch_name = ? WHERE id = ?",
        (
            workspace_path.format(id=source_id),
            branch_name.format(id=source_id) if branch_name else None,
            source_id,
        ),
    )
    kanban_conn.commit()
    _disable_project_store(monkeypatch)

    with pytest.raises(ValueError, match="project not found or unavailable"):
        kb.create_task(
            kanban_conn,
            title="Rejected child",
            tenant="tenant-a",
            project_id="p_deadbeef",
            project_source_task_id=source_id,
        )
