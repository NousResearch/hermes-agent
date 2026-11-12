"""A ``dir`` workspace must not point at a git repo's main checkout."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    for args in (
        ["init", "-b", "main", str(repo)],
        ["-C", str(repo), "config", "user.email", "kanban@example.com"],
        ["-C", str(repo), "config", "user.name", "Kanban Test"],
        ["-C", str(repo), "commit", "--allow-empty", "-m", "init"],
    ):
        subprocess.run(["git", *args], check=True, capture_output=True, text=True)


def _create_workspace_task(conn, *, kind: str, path: str | None, title: str):
    tid = kb.create_task(
        conn, title=title, workspace_kind=kind, workspace_path=path,
    )
    task = kb.get_task(conn, tid)
    assert task is not None
    return task


def test_dir_workspace_refuses_git_main_checkout(kanban_home, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    with kbc.connect() as conn:
        task = _create_workspace_task(
            conn, kind="dir", path=str(repo), title="dirty main",
        )
        with pytest.raises(ValueError, match="main checkout of git repo") as exc:
            kbw.resolve_workspace(task)
    message = str(exc.value)
    assert str(repo.resolve()) in message
    assert "workspace_kind=worktree" in message


def test_dir_workspace_refuses_subdir_of_git_main_checkout(
    kanban_home, tmp_path, monkeypatch,
):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    sub = repo / "src"
    sub.mkdir()
    missing = repo / "not-created-yet"
    with kbc.connect() as conn:
        existing = _create_workspace_task(
            conn, kind="dir", path=str(sub), title="dirty subdir",
        )
        absent = _create_workspace_task(
            conn, kind="dir", path=str(missing), title="dirty missing child",
        )
        with pytest.raises(ValueError, match="main checkout of git repo"):
            kbw.resolve_workspace(existing)
        with pytest.raises(ValueError, match="main checkout of git repo"):
            kbw.resolve_workspace(absent)
    assert not missing.exists()


def test_dir_workspace_allows_linked_worktree(kanban_home, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    linked = tmp_path / "linked"
    subprocess.run(
        [
            "git", "-C", str(repo), "worktree", "add",
            "-b", "wt-allow", str(linked), "HEAD",
        ],
        check=True, capture_output=True, text=True,
    )
    nested = linked / "pkg"
    nested.mkdir()
    with kbc.connect() as conn:
        root_task = _create_workspace_task(
            conn, kind="dir", path=str(linked), title="linked root",
        )
        nested_task = _create_workspace_task(
            conn, kind="dir", path=str(nested), title="linked nested",
        )
        assert kbw.resolve_workspace(root_task) == linked
        assert kbw.resolve_workspace(nested_task) == nested


def test_dir_workspace_allows_non_git_dir(kanban_home, tmp_path, monkeypatch):
    folder = tmp_path / "plain"
    folder.mkdir()
    with kbc.connect() as conn:
        task = _create_workspace_task(
            conn, kind="dir", path=str(folder), title="plain dir",
        )
        assert kbw.resolve_workspace(task) == folder


def test_scratch_and_worktree_ignore_dir_main_checkout_guard(
    kanban_home, tmp_path, monkeypatch,
):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    with kbc.connect() as conn:
        scratch = _create_workspace_task(
            conn, kind="scratch", path=None, title="scratch",
        )
        scratch_on_repo = _create_workspace_task(
            conn, kind="scratch", path=str(repo), title="scratch on repo",
        )
        worktree = _create_workspace_task(
            conn, kind="worktree", path=str(repo), title="worktree",
        )
        scratch_ws = kbw.resolve_workspace(scratch)
        assert kbw.resolve_workspace(scratch_on_repo) == repo
        worktree_ws = kbw.resolve_workspace(worktree)
    assert scratch_ws.is_dir()
    assert scratch_ws.name == scratch.id
    assert worktree_ws == repo / ".worktrees" / worktree.id
    assert (worktree_ws / ".git").is_file()
