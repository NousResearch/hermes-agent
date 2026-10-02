"""A fresh git worktree must resolve the main checkout's committed dependency
environment, not mint its own never-installed install-state directory.

Regression for the P0: `hermes -m hermes_cli.main` launched from a
`workspace_kind=worktree` kanban task exited before its first tool call with
"no dependency environment is committed for this install; run `hermes pm
repair`" even though `hermes pm status` at the main install reported the sync
already OK. Root cause: ``install_key()`` hashed the worktree's own resolved
path, so ``install_state_dir()`` (and therefore ``committed_venv()``) pointed
at ``installs/<hash-of-worktree-path>/`` -- a directory nothing had ever
installed into -- while the main checkout's install lived under a different
hash entirely.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=30)


def _make_main_checkout_with_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """Real git repo with a real linked worktree (no fixture private env vars)."""
    main = tmp_path / "main-checkout"
    main.mkdir()
    _git(main, "init", "-q")
    _git(main, "config", "user.email", "t@example.com")
    _git(main, "config", "user.name", "t")
    (main / "README.md").write_text("x", encoding="utf-8")
    _git(main, "add", "README.md")
    _git(main, "commit", "-q", "-m", "init")
    worktree = tmp_path / "worktrees" / "task-1"
    worktree.parent.mkdir(parents=True)
    _git(main, "worktree", "add", "-q", "-b", "wt/task-1", str(worktree), "HEAD")
    return main, worktree


def test_install_key_folds_linked_worktree_onto_main_checkout(tmp_path, monkeypatch):
    from pm import environments as runtime_paths

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    main, worktree = _make_main_checkout_with_worktree(tmp_path)

    assert runtime_paths.install_key(worktree) == runtime_paths.install_key(main)
    assert runtime_paths.install_state_dir(worktree) == runtime_paths.install_state_dir(main)


def test_worktree_sees_main_checkouts_committed_environment(tmp_path, monkeypatch):
    """Same reproducer, one level up: an environment committed for the MAIN checkout
    must be what a fresh worktree's ``activate_dependencies`` selects -- never a
    RuntimeError demanding a per-worktree `pm repair`."""
    from pm import environments as runtime_paths

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    main, worktree = _make_main_checkout_with_worktree(tmp_path)

    state = runtime_paths.install_state_dir(main)
    generation = state / "environments" / "committed" / "venv"
    site = runtime_paths.site_packages(generation)
    site.mkdir(parents=True)
    (generation / "pyvenv.cfg").write_text(
        f"home = test\nversion = {sys.version_info.major}.{sys.version_info.minor}.0\n", encoding="utf-8")
    (site / "probe_dependency.py").write_text("value = 'installed'\n", encoding="utf-8")
    state.mkdir(parents=True, exist_ok=True)
    (state / "facts.json").write_text(json.dumps({
        "schema": 1, "packages": {"venv": {"environment": str(generation), "stamp": "verified"}},
    }), encoding="utf-8")

    # Before the fix this raised RuntimeError("no dependency environment is
    # committed for this install"): the worktree's own install_state_dir had
    # no facts.json, so committed_venv(worktree) was None.
    assert runtime_paths.committed_venv(worktree) == generation
    assert runtime_paths.selected_venv(worktree) == generation


def test_worktree_activate_dependencies_selects_committed_venv(tmp_path, monkeypatch):
    """``activate_dependencies`` -- the call every worker launch makes before
    its first import -- must resolve the worktree to the main checkout's
    committed venv instead of raising "no dependency environment is
    committed for this install". Fully isolated: a synthetic repo + a real
    committed venv layout under a tmp_path HERMES_HOME, never the real
    developer home (enforced by tests/home_io_guard.py).
    """
    from pm import environments as runtime_paths

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    main, worktree = _make_main_checkout_with_worktree(tmp_path)

    state = runtime_paths.install_state_dir(main)
    generation = state / "environments" / "committed" / "venv"
    site = runtime_paths.site_packages(generation)
    site.mkdir(parents=True)
    (generation / "pyvenv.cfg").write_text(
        f"home = test\nversion = {sys.version_info.major}.{sys.version_info.minor}.0\n", encoding="utf-8")
    state.mkdir(parents=True, exist_ok=True)
    (state / "facts.json").write_text(json.dumps({
        "schema": 1, "packages": {"venv": {"environment": str(generation), "stamp": "verified"}},
    }), encoding="utf-8")

    # Before the fix, activate_dependencies(worktree) raised RuntimeError
    # ("no dependency environment is committed for this install; run
    # `hermes pm repair`") because install_state_dir(worktree) pointed at a
    # never-installed installs/<hash-of-worktree-path>/ directory.
    runtime_paths.activate_dependencies(worktree)
