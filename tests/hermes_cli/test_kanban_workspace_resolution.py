"""Regression tests for the workspace_resolution cluster (s4-w1b extraction).

Covers the pure git helpers moved verbatim from ``hermes_cli.kanban_db``
(cluster c2 / workspace_resolution) into ``hermes_cli.workspace_resolution``.
Both import surfaces are exercised: the new module directly, and the
re-exported names on ``hermes_cli.kanban_db`` (the public API the CLI and
existing tests use).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import hermes_cli.kanban_db as kb
from hermes_cli.workspace_resolution import (
    _ensure_git_worktree,
    _git_branch_exists,
    _git_common_dir,
    _git_current_branch,
    _git_dir,
    _git_toplevel,
    _is_linked_worktree_checkout,
    _nearest_existing_path,
    _repo_root_for_worktree_target,
    _resolve_worktree_workspace,
    resolve_workspace,
    set_branch_name,
    set_workspace_path,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (mirrors test_kanban_db.py)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "kanban@example.com"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Kanban Test"], check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Re-export parity: the moved functions must be THE SAME objects the public
# kanban_db module exposes (tests and CLI reference them by name there).
# ---------------------------------------------------------------------------


def test_moved_names_reexported_on_kanban_db_module():
    for name in (
        "_git_toplevel", "_git_branch_exists", "_git_common_dir", "_git_dir",
        "_git_current_branch", "_is_linked_worktree_checkout",
        "_nearest_existing_path", "_repo_root_for_worktree_target",
        "_ensure_git_worktree", "_resolve_worktree_workspace",
        "resolve_workspace", "set_workspace_path", "set_branch_name",
    ):
        assert getattr(kb, name) is globals()[name], name


def test_direct_module_import_works():
    """The new module must be importable on its own (no import cycle)."""
    import hermes_cli.workspace_resolution as ws
    assert ws.resolve_workspace is resolve_workspace


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def test_git_toplevel_finds_repo_root(tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    assert _git_toplevel(repo) == repo.resolve()
    nested = repo / "sub" / "dir"
    nested.mkdir(parents=True)
    assert _git_toplevel(nested) == repo.resolve()


def test_git_toplevel_none_outside_repo(tmp_path):
    assert _git_toplevel(tmp_path / "not-a-repo") is None


def test_git_branch_exists(tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    assert _git_branch_exists(repo, "main") is True
    assert _git_branch_exists(repo, "no-such-branch") is False


def test_git_common_dir_and_git_dir_agree_on_plain_repo(tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    assert _git_common_dir(repo) == _git_dir(repo)


def test_git_current_branch(tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    assert _git_current_branch(repo) == "main"


def test_is_linked_worktree_checkout_false_for_plain_repo(tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    assert _is_linked_worktree_checkout(repo) is False


def test_nearest_existing_path_walks_up(tmp_path):
    existing = tmp_path / "a"
    existing.mkdir()
    deep = existing / "b" / "c" / "d"
    assert _nearest_existing_path(deep) == existing


def test_repo_root_for_worktree_target(tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    target = repo / ".worktrees" / "t1"
    assert _repo_root_for_worktree_target(target) == repo.resolve()


# ---------------------------------------------------------------------------
# resolve_workspace
# ---------------------------------------------------------------------------


def test_resolve_workspace_scratch_creates_dir(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="scratch task")
        task = kb.get_task(conn, t)
        ws = resolve_workspace(task)
    assert ws.exists()
    assert ws.name == t


def test_resolve_workspace_rejects_relative_dir_path(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="rel", workspace_kind="dir", workspace_path="relative/path",
        )
        task = kb.get_task(conn, t)
        with pytest.raises(ValueError, match="absolute"):
            resolve_workspace(task)


def test_resolve_workspace_dir_creates_absolute_path(kanban_home, tmp_path):
    target = tmp_path / "workdir"
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="dir task", workspace_kind="dir", workspace_path=str(target),
        )
        task = kb.get_task(conn, t)
        ws = resolve_workspace(task)
    assert ws == target
    assert ws.exists()


def test_resolve_workspace_unknown_kind_raises(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="bogus")  # create_task validates the kind,
        # so corrupt the row directly to simulate a kind no resolver knows.
        conn.execute(
            "UPDATE tasks SET workspace_kind = 'teleport', workspace_path = ? "
            "WHERE id = ?",
            (str(kanban_home / "x"), t),
        )
        conn.commit()
        task = kb.get_task(conn, t)
        with pytest.raises(ValueError, match="unknown workspace_kind"):
            resolve_workspace(task)


def test_resolve_workspace_worktree_materializes(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    target = repo / ".worktrees" / "wt-task"
    with kb.connect() as conn:
        t = kb.create_task(
            conn,
            title="wt",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="wt/wt-task",
        )
        task = kb.get_task(conn, t)
        ws = resolve_workspace(task)
    assert ws == target
    assert ws.exists()
    # The worktree must share the repo's common dir (a real linked worktree).
    repo_common = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    ws_common = subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert ws_common == repo_common


def test_set_workspace_path_and_branch_persist(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="persist")
        set_workspace_path(conn, t, "/some/absolute/ws")
        set_branch_name(conn, t, "feature/x")
        row = conn.execute(
            "SELECT workspace_path, branch_name FROM tasks WHERE id = ?", (t,),
        ).fetchone()
        assert row["workspace_path"] == "/some/absolute/ws"
        assert row["branch_name"] == "feature/x"


def test_ensure_git_worktree_creates_branch_and_checkout(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    target = repo / ".worktrees" / "ensured"
    _ensure_git_worktree(repo, target, "wt/ensured")
    assert target.exists()
    assert _git_current_branch(target) == "wt/ensured"


def test_resolve_worktree_workspace_requires_absolute(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="rel wt", workspace_kind="worktree", workspace_path="relative",
        )
        task = kb.get_task(conn, t)
        with pytest.raises(ValueError, match="absolute"):
            _resolve_worktree_workspace(task)
