"""Tests for repository-scoped virtualenv links in Git worktrees."""

import subprocess
from pathlib import Path

from hermes_cli.worktree_environment import bootstrap_worktree_environments


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    return path


def _fake_python(environment: Path) -> None:
    python = environment / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)


def test_bootstrap_links_the_repository_environment_into_a_worktree(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "hermes-agent")
    _fake_python(repo / ".venv")
    target = tmp_path / "worktree"
    target.mkdir()

    linked = bootstrap_worktree_environments(
        repo, target, environment_names=(".venv",), require_python=True
    )

    assert linked == (".venv",)
    assert (target / ".venv").is_symlink()
    assert (target / ".venv").resolve() == (repo / ".venv").resolve()


def test_bootstrap_rejects_an_environment_from_another_repository(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path / "hermes-agent")
    foreign = _git_repo(tmp_path / "lunabot")
    _fake_python(foreign / ".venv")
    (repo / ".venv").symlink_to(foreign / ".venv", target_is_directory=True)
    target = tmp_path / "worktree"
    target.mkdir()

    linked = bootstrap_worktree_environments(
        repo, target, environment_names=(".venv",), require_python=True
    )

    assert linked == ()
    assert not (target / ".venv").exists()


def test_bootstrap_reuses_main_checkout_environment_for_existing_worktree(
    tmp_path: Path,
) -> None:
    """Reused linked worktrees inherit the repository env, not the dispatcher env."""
    main = _git_repo(tmp_path / "hermes-agent")
    _fake_python(main / ".venv")
    linked = tmp_path / "existing-worktree"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "--quiet", "-b", "worker", str(linked)],
        check=True,
    )
    target = tmp_path / "nested-worktree"
    target.mkdir()

    linked_names = bootstrap_worktree_environments(
        linked, target, environment_names=(".venv",), require_python=True
    )

    assert linked_names == (".venv",)
    assert (target / ".venv").resolve() == (main / ".venv").resolve()


def test_bootstrap_accepts_the_repository_managed_sibling_environment(
    tmp_path: Path,
) -> None:
    """Hermes' external managed venv remains bound to its Git repository."""
    main = _git_repo(tmp_path / "hermes-agent")
    managed = tmp_path / "venvs" / "hermes-3136"
    _fake_python(managed)
    (main / ".venv").symlink_to(managed, target_is_directory=True)
    target = tmp_path / "worktree"
    target.mkdir()

    linked = bootstrap_worktree_environments(
        main, target, environment_names=(".venv",), require_python=True
    )

    assert linked == (".venv",)
    assert (target / ".venv").resolve() == managed.resolve()


def test_venv_python_path_recognizes_native_windows_layout(
    tmp_path: Path, monkeypatch
) -> None:
    """A native Windows venv exposes Scripts/python.exe, not bin/python (#PR70)."""
    from hermes_cli.worktree_environment import _venv_python_path

    monkeypatch.setattr("hermes_cli.worktree_environment.os.name", "nt")
    assert _venv_python_path(tmp_path) == tmp_path / "Scripts" / "python.exe"

    monkeypatch.setattr("hermes_cli.worktree_environment.os.name", "posix")
    assert _venv_python_path(tmp_path) == tmp_path / "bin" / "python"
