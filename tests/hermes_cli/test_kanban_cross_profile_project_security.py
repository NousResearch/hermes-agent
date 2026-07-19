"""Security regressions for cross-profile project-linked Kanban routing."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(path, "config", "user.email", "kanban-security@example.invalid")
    _git(path, "config", "user.name", "Kanban Security Test")
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "fixture")
    return path


def _set_profile(monkeypatch: pytest.MonkeyPatch, home: Path, name: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", name)


def _create_materialized_source(
    conn,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    tenant: str | None,
) -> tuple[kb.Task, Path, str]:
    profile_a = tmp_path / "profiles" / "creator"
    _set_profile(monkeypatch, profile_a, "creator")
    repo = _init_repo(tmp_path / "legit")
    with pdb.connect_closing() as project_conn:
        project_id = pdb.create_project(
            project_conn,
            name="Canonical Project",
            folders=[str(repo)],
        )
    source_id = kb.create_task(
        conn,
        title="source implementation",
        assignee="creator",
        project_id=project_id,
        tenant=tenant,
    )
    source = kb.get_task(conn, source_id)
    assert source is not None
    assert kb.resolve_workspace(source) == repo / ".worktrees" / source_id
    claimed = kb.claim_task(conn, source_id)
    assert claimed is not None
    return claimed, repo, project_id


def test_cross_profile_fallback_rejects_different_tenant(monkeypatch, tmp_path):
    conn = kb.connect(db_path=tmp_path / "kanban.db")
    try:
        source, _repo, project_id = _create_materialized_source(
            conn,
            monkeypatch,
            tmp_path,
            tenant="tenant-a",
        )
        _set_profile(monkeypatch, tmp_path / "profiles" / "worker", "worker")

        with pytest.raises(ValueError, match="tenant"):
            kb.create_task(
                conn,
                title="cross-tenant child",
                assignee="worker",
                project_id=project_id,
                project_source_task_id=source.id,
                tenant="tenant-b",
            )
    finally:
        conn.close()


def test_project_child_dispatch_rejects_post_create_worktrees_symlink_swap(
    monkeypatch,
    tmp_path,
):
    conn = kb.connect(db_path=tmp_path / "kanban.db")
    try:
        source, repo, project_id = _create_materialized_source(
            conn,
            monkeypatch,
            tmp_path,
            tenant="tenant-a",
        )
        _set_profile(monkeypatch, tmp_path / "profiles" / "worker", "worker")
        child_id = kb.create_task(
            conn,
            title="child",
            assignee="worker",
            project_id=project_id,
            project_source_task_id=source.id,
            tenant="tenant-a",
        )
        child = kb.get_task(conn, child_id)
        assert child is not None

        attacker = _init_repo(tmp_path / "attacker")
        attacker_worktrees = attacker / ".worktrees"
        attacker_worktrees.mkdir()
        _git(
            repo, "worktree", "remove", "--force", str(repo / ".worktrees" / source.id)
        )
        shutil.rmtree(repo / ".worktrees")
        (repo / ".worktrees").symlink_to(attacker_worktrees, target_is_directory=True)

        spawned: list[Path] = []

        def _spawn(_task: kb.Task, workspace: Path) -> int:
            spawned.append(workspace)
            return os.getpid()

        result = kb.dispatch_once(conn, spawn_fn=_spawn, failure_limit=3)
        child = kb.get_task(conn, child_id)
        assert child is not None
        assert result.spawned == []
        assert spawned == []
        assert child.status == "ready"
        assert child.last_failure_error is not None
        assert "workspace" in child.last_failure_error
        assert not (attacker_worktrees / child_id).exists()
    finally:
        conn.close()


@pytest.mark.parametrize("dangling", [False, True])
def test_project_child_dispatch_rejects_post_create_workspace_symlink(
    monkeypatch,
    tmp_path,
    dangling,
):
    conn = kb.connect(db_path=tmp_path / "kanban.db")
    try:
        source, repo, project_id = _create_materialized_source(
            conn,
            monkeypatch,
            tmp_path,
            tenant="tenant-a",
        )
        _set_profile(monkeypatch, tmp_path / "profiles" / "worker", "worker")
        child_id = kb.create_task(
            conn,
            title="child",
            assignee="worker",
            project_id=project_id,
            project_source_task_id=source.id,
            tenant="tenant-a",
        )
        expected = repo / ".worktrees" / child_id
        redirected = tmp_path / "redirected-child"
        if not dangling:
            redirected.mkdir()
        expected.symlink_to(redirected, target_is_directory=True)

        spawned: list[Path] = []
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda _task, workspace: spawned.append(workspace) or os.getpid(),
            failure_limit=3,
        )
        child = kb.get_task(conn, child_id)
        assert child is not None
        assert result.spawned == []
        assert spawned == []
        assert child.status == "ready"
        assert child.workspace_kind == "worktree"
        assert child.project_id == project_id
        assert child.last_failure_error is not None
        assert "symlinked" in child.last_failure_error
        assert not redirected.joinpath(".git").exists()
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("missing_source", "does not exist"),
        ("missing_anchor", "trust anchor"),
        ("foreign_project", "different project"),
        ("non_worktree", "not a worktree"),
        ("workspace_traversal", "canonical route"),
        ("foreign_workspace", "canonical route"),
        ("stale_workspace", "stale or missing"),
        ("symlink_workspace", "symlinked"),
        ("ordinary_directory", "registered linked worktree"),
        ("branch_mismatch", "branch"),
        ("anchor_traversal", "path traversal"),
        ("foreign_anchor", "canonical route"),
        ("renamed_repo", "stale or missing"),
    ],
)
def test_cross_profile_fallback_rejects_untrusted_source_matrix(
    monkeypatch,
    tmp_path,
    case,
    error,
):
    conn = kb.connect(db_path=tmp_path / "kanban.db")
    try:
        source, repo, project_id = _create_materialized_source(
            conn,
            monkeypatch,
            tmp_path,
            tenant="tenant-a",
        )
        source_id = source.id
        source_path = repo / ".worktrees" / source.id
        if case == "missing_source":
            source_id = "t_missing00"
        elif case == "missing_anchor":
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET project_repo_root = NULL WHERE id = ?",
                    (source.id,),
                )
        elif case == "foreign_project":
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET project_id = 'p_foreign' WHERE id = ?",
                    (source.id,),
                )
        elif case == "non_worktree":
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET workspace_kind = 'dir' WHERE id = ?",
                    (source.id,),
                )
        elif case == "workspace_traversal":
            traversal = f"{repo}/.worktrees/../.worktrees/{source.id}"
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET workspace_path = ? WHERE id = ?",
                    (traversal, source.id),
                )
        elif case in {"foreign_workspace", "foreign_anchor"}:
            attacker = _init_repo(tmp_path / "attacker")
            if case == "foreign_workspace":
                foreign = attacker / ".worktrees" / source.id
                with kb.write_txn(conn):
                    conn.execute(
                        "UPDATE tasks SET workspace_path = ? WHERE id = ?",
                        (str(foreign), source.id),
                    )
            else:
                with kb.write_txn(conn):
                    conn.execute(
                        "UPDATE tasks SET project_repo_root = ? WHERE id = ?",
                        (str(attacker), source.id),
                    )
        elif case == "stale_workspace":
            _git(repo, "worktree", "remove", "--force", str(source_path))
        elif case == "symlink_workspace":
            relocated = tmp_path / "relocated-source"
            source_path.rename(relocated)
            source_path.symlink_to(relocated, target_is_directory=True)
        elif case == "ordinary_directory":
            _git(repo, "worktree", "remove", "--force", str(source_path))
            source_path.mkdir(parents=True)
        elif case == "branch_mismatch":
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET branch_name = 'feature/custom' WHERE id = ?",
                    (source.id,),
                )
        elif case == "anchor_traversal":
            (repo / "nested").mkdir()
            traversal = f"{repo}/nested/.."
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET project_repo_root = ? WHERE id = ?",
                    (traversal, source.id),
                )
        elif case == "renamed_repo":
            repo.rename(tmp_path / "renamed-legit")
        else:  # pragma: no cover - parameter table is exhaustive
            raise AssertionError(case)

        _set_profile(monkeypatch, tmp_path / "profiles" / "worker", "worker")
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match=error):
            kb.create_task(
                conn,
                title="must not be created",
                assignee="worker",
                project_id=project_id,
                project_source_task_id=source_id,
                tenant="tenant-a",
            )
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert after == before
    finally:
        conn.close()


def test_cross_profile_route_survives_source_project_store_deletion(
    monkeypatch,
    tmp_path,
):
    conn = kb.connect(db_path=tmp_path / "kanban.db")
    try:
        source, repo, project_id = _create_materialized_source(
            conn,
            monkeypatch,
            tmp_path,
            tenant="tenant-a",
        )
        creator_projects_db = tmp_path / "profiles" / "creator" / "projects.db"
        creator_projects_db.unlink()
        _set_profile(monkeypatch, tmp_path / "profiles" / "worker", "worker")
        child_id = kb.create_task(
            conn,
            title="child after project-store deletion",
            assignee="worker",
            project_id=project_id,
            project_source_task_id=source.id,
            tenant="tenant-a",
        )
        child = kb.get_task(conn, child_id)
        assert child is not None
        assert child.project_repo_root == str(repo)
        assert kb.resolve_workspace(child) == repo / ".worktrees" / child_id
        assert (tmp_path / "profiles" / "worker" / "projects.db").exists()
    finally:
        conn.close()


def test_project_route_survives_reclaim_retry_and_goal_mode(
    monkeypatch,
    tmp_path,
):
    conn = kb.connect(db_path=tmp_path / "kanban.db")
    try:
        source, repo, project_id = _create_materialized_source(
            conn,
            monkeypatch,
            tmp_path,
            tenant="tenant-a",
        )
        _set_profile(monkeypatch, tmp_path / "profiles" / "worker", "worker")
        child_id = kb.create_task(
            conn,
            title="goal child",
            assignee="worker",
            project_id=project_id,
            project_source_task_id=source.id,
            tenant="tenant-a",
            goal_mode=True,
            goal_max_turns=7,
        )
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
        first = kb.dispatch_once(
            conn,
            spawn_fn=lambda _task, _workspace: 424242,
        )
        assert {item[0] for item in first.spawned} == {child_id}
        first_task = kb.get_task(conn, child_id)
        assert first_task is not None
        route = (
            first_task.workspace_path,
            first_task.branch_name,
            first_task.project_repo_root,
        )
        assert first_task.goal_mode is True
        assert first_task.goal_max_turns == 7
        assert kb.reclaim_task(
            conn,
            child_id,
            reason="security retry",
            signal_fn=lambda *_args: None,
        )
        second = kb.dispatch_once(
            conn,
            spawn_fn=lambda _task, _workspace: 424243,
        )
        assert {item[0] for item in second.spawned} == {child_id}
        retried = kb.get_task(conn, child_id)
        assert retried is not None
        assert (
            retried.workspace_path,
            retried.branch_name,
            retried.project_repo_root,
        ) == route
        assert retried.workspace_path is not None
        assert Path(retried.workspace_path).is_dir()
        source_after = kb.get_task(conn, source.id)
        assert source_after is not None
        assert source_after.workspace_path == source.workspace_path
        assert (repo / ".worktrees" / source.id).is_dir()
    finally:
        conn.close()


def test_project_anchor_is_visible_in_cli_tool_and_dashboard_shapes(
    monkeypatch,
    tmp_path,
):
    from hermes_cli.kanban import _task_to_dict
    from plugins.kanban.dashboard.plugin_api import _task_dict
    from tools.kanban_tools import _task_summary_dict

    conn = kb.connect(db_path=tmp_path / "kanban.db")
    try:
        source, repo, _project_id = _create_materialized_source(
            conn,
            monkeypatch,
            tmp_path,
            tenant=None,
        )
        assert _task_to_dict(source)["project_repo_root"] == str(repo)
        assert _task_summary_dict(kb, conn, source)["project_repo_root"] == str(repo)
        assert _task_dict(source)["project_repo_root"] == str(repo)
    finally:
        conn.close()
