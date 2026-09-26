"""Tests for worktree workspace teardown at task completion/archive.

Covers the ownership gap where kanban ``worktree`` workspaces were never
reaped by anything: ``_cleanup_workspace`` preserved them by design, the CLI
startup pruner explicitly skips ``t_*`` worktrees ("dispatcher-driven
lifecycle"), and ``kanban gc`` only swept scratch. A completed or archived
task's linked worktree is now removed when — and only when — it provably
holds no work: clean working tree and every commit reachable from a
remote-tracking ref. Any doubt preserves the worktree.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_db_connect as kbc


def _git(*args: str, cwd: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A project repo with a remote whose history is fully pushed."""
    origin = tmp_path / "origin.git"
    _git("init", "--bare", str(origin))
    project = tmp_path / "project"
    _git("clone", str(origin), str(project))
    _git("-C", str(project), "config", "user.email", "t@example.com")
    _git("-C", str(project), "config", "user.name", "t")
    (project / "README.md").write_text("hello\n", encoding="utf-8")
    _git("-C", str(project), "add", "README.md")
    _git("-C", str(project), "commit", "-m", "init")
    _git("-C", str(project), "push", "origin", "HEAD")
    return project


def _make_worktree(repo: Path, task_id: str, branch: str | None = None) -> Path:
    target = repo / ".worktrees" / task_id
    kbw._ensure_git_worktree(repo, target, branch or f"wt/{task_id}")
    return target


def _branch_exists(repo: Path, branch: str) -> bool:
    out = _git("-C", str(repo), "branch", "--list", branch)
    return bool(out.strip())


# ---------------------------------------------------------------------------
# _cleanup_worktree_workspace unit behavior
# ---------------------------------------------------------------------------


def test_clean_pushed_worktree_removed(repo: Path) -> None:
    wt = _make_worktree(repo, "t_aaaa1111")
    kbw._cleanup_worktree_workspace("t_aaaa1111", str(wt))
    assert not wt.exists()
    # auto-generated task branch goes with it
    assert not _branch_exists(repo, "wt/t_aaaa1111")
    # main checkout untouched
    assert (repo / "README.md").exists()


def test_cleanup_leaves_a_worktree_cwd_before_removal(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows cannot remove a worktree that is the process's current directory."""
    wt = _make_worktree(repo, "t_cwd112425")
    real_git = kbw._git

    def windows_git(repo_root: Path, *args: str, timeout: int) -> subprocess.CompletedProcess:
        if args[:2] == ("worktree", "remove") and Path.cwd().is_relative_to(wt):
            return subprocess.CompletedProcess(
                ["git", *args], 1, stderr="Permission denied: current directory"
            )
        return real_git(repo_root, *args, timeout=timeout)

    monkeypatch.setattr(kbw, "_git", windows_git)
    monkeypatch.chdir(wt)
    kbw._cleanup_worktree_workspace("t_cwd112425", str(wt))

    assert Path.cwd() == repo
    assert not wt.exists()


def test_cleanup_proceeds_when_cwd_was_deleted(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deferred parent cleanup (#33774) runs after the child's scratch cwd was
    rmtree'd; a dead cwd must not preserve a clean, pushed worktree."""
    wt = _make_worktree(repo, "t_deadcwd113073")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(scratch)
    scratch.rmdir()

    kbw._cleanup_worktree_workspace("t_deadcwd113073", str(wt))

    assert not wt.exists()
    assert not _branch_exists(repo, "wt/t_deadcwd113073")


def test_cleanup_retries_worktree_removal_once(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A brief Windows directory-handle delay gets one safe retry."""
    wt = _make_worktree(repo, "t_retry112425")
    real_git = kbw._git
    attempts = 0

    def delayed_remove(repo_root: Path, *args: str, timeout: int) -> subprocess.CompletedProcess:
        nonlocal attempts
        if args[:2] == ("worktree", "remove"):
            attempts += 1
            if attempts == 1:
                return subprocess.CompletedProcess(
                    ["git", *args], 1, stderr="Permission denied: handle pending"
                )
        return real_git(repo_root, *args, timeout=timeout)

    monkeypatch.setattr(kbw, "_git", delayed_remove)
    monkeypatch.setattr(kbw.time, "sleep", lambda _delay: None)
    kbw._cleanup_worktree_workspace("t_retry112425", str(wt))

    assert attempts == 2
    assert not wt.exists()


def test_dirty_worktree_preserved(repo: Path) -> None:
    wt = _make_worktree(repo, "t_bbbb2222")
    (wt / "wip.txt").write_text("uncommitted\n", encoding="utf-8")
    kbw._cleanup_worktree_workspace("t_bbbb2222", str(wt))
    assert wt.is_dir()
    assert (wt / "wip.txt").exists()


def test_unpushed_commits_preserved(repo: Path) -> None:
    wt = _make_worktree(repo, "t_cccc3333")
    (wt / "work.txt").write_text("committed but not pushed\n", encoding="utf-8")
    _git("-C", str(wt), "add", "work.txt")
    _git("-C", str(wt), "commit", "-m", "local work")
    kbw._cleanup_worktree_workspace("t_cccc3333", str(wt))
    assert wt.is_dir()


def test_custom_branch_survives_worktree_removal(repo: Path) -> None:
    wt = _make_worktree(repo, "t_dddd4444", branch="feature/custom")
    kbw._cleanup_worktree_workspace("t_dddd4444", str(wt), "feature/custom")
    assert not wt.exists()
    # only auto-generated wt/* branches are deleted
    assert _branch_exists(repo, "feature/custom")


def test_main_checkout_never_removed(repo: Path) -> None:
    kbw._cleanup_worktree_workspace("t_eeee5555", str(repo))
    assert repo.is_dir()
    assert (repo / "README.md").exists()


def test_non_git_dir_preserved(tmp_path: Path) -> None:
    plain = tmp_path / "not-a-worktree"
    plain.mkdir()
    kbw._cleanup_worktree_workspace("t_ffff6666", str(plain))
    assert plain.is_dir()


def test_tree_dirtied_between_check_and_removal_preserved(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TOCTOU: a tree that becomes dirty after the pre-check is NOT removed.

    Simulates the race by making the pre-check see a clean tree while the
    tree is actually dirty when ``git worktree remove`` runs. Without
    ``--force``, git's own dirty guard re-verifies at removal time and the
    removal fails safe.
    """
    import cli

    wt = _make_worktree(repo, "t_gggg7777")
    (wt / "late-wip.txt").write_text("dirtied after the check\n", encoding="utf-8")
    # Pre-check lies (as if the file appeared just after it ran) — real git
    # must still refuse the removal.
    from hermes_cli import worktree_ops

    monkeypatch.setattr(worktree_ops, "_worktree_is_dirty", lambda _p: False)
    kbw._cleanup_worktree_workspace("t_gggg7777", str(wt))
    assert wt.is_dir()
    assert (wt / "late-wip.txt").exists()


# ---------------------------------------------------------------------------
# Lifecycle integration: complete / archive / deferred parents
# ---------------------------------------------------------------------------


def _worktree_task(conn, repo: Path, title: str = "wt-task") -> tuple[str, Path]:
    tid = kb.create_task(conn, title=title, assignee="worker")
    wt = _make_worktree(repo, tid)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', workspace_path=?, "
            "branch_name=? WHERE id=?",
            (str(wt), f"wt/{tid}", tid),
        )
    return tid, wt


def test_complete_task_reaps_clean_worktree(kanban_home: Path, repo: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, wt = _worktree_task(conn, repo)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        assert kb.complete_task(conn, tid, summary="done")
    assert not wt.exists()
    assert not _branch_exists(repo, f"wt/{tid}")


def test_complete_task_preserves_dirty_worktree(kanban_home: Path, repo: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, wt = _worktree_task(conn, repo)
        (wt / "wip.txt").write_text("unsaved\n", encoding="utf-8")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        assert kb.complete_task(conn, tid, summary="done")
    assert wt.is_dir()
    assert (wt / "wip.txt").exists()


def test_archive_task_reaps_clean_worktree(kanban_home: Path, repo: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, wt = _worktree_task(conn, repo)
        assert kb.archive_task(conn, tid)
    assert not wt.exists()


def test_parent_worktree_deferred_until_children_done(
    kanban_home: Path, repo: Path
) -> None:
    with kbc.connect_closing() as conn:
        parent, parent_wt = _worktree_task(conn, repo, title="parent")
        child = kb.create_task(conn, title="child", assignee="worker")
        kb.link_tasks(conn, parent, child)

        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        assert kb.claim_task(conn, parent, claimer="worker") is not None
        assert kb.complete_task(conn, parent, summary="parent done")
        # child still active -> parent worktree must survive for handoff
        assert parent_wt.is_dir()

        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child,))
        assert kb.claim_task(conn, child, claimer="worker") is not None
        assert kb.complete_task(conn, child, summary="child done")
    # last child terminal -> deferred parent worktree reaped
    assert not parent_wt.exists()


# These two exercise POSIX permission bits directly. Windows has no equivalent
# (``os.chmod`` only toggles the read-only attribute and does not stop a
# directory from being removed), and root bypasses the bits entirely, so the
# "unpatched rmtree leaves the tree behind" guard would not hold in either.
_needs_posix_permissions = pytest.mark.skipif(
    os.name != "posix" or getattr(os, "geteuid", lambda: 1)() == 0,
    reason="needs enforced POSIX permission bits (non-Windows, non-root)",
)


@_needs_posix_permissions
def test_rmtree_force_reaps_read_only_dirs(tmp_path: Path) -> None:
    """Any tool that materialises read-only directories (Pants' execution root,
    Bazel's output base, Go's module cache) defeats ``ignore_errors=True``: the
    walk stops at the first one and the whole workspace survives, unlogged."""

    def build(name: str) -> Path:
        root = tmp_path / name
        deep = root / "repo" / "build_cache" / "inputs" / "deep"
        deep.mkdir(parents=True)
        (deep / "f.txt").write_text("x")
        for d in (deep, deep.parent, deep.parent.parent):
            os.chmod(d, 0o500)
        return root

    stale = build("stale")
    shutil.rmtree(stale, ignore_errors=True)
    assert stale.exists(), "guard: ignore_errors=True must leave the tree behind"
    kbw._rmtree_force(stale)  # leave nothing tmp_path teardown cannot remove

    reaped = build("reaped")
    kbw._rmtree_force(reaped)
    assert not reaped.exists()


@_needs_posix_permissions
def test_rmtree_force_never_chmods_outside_the_tree(tmp_path: Path) -> None:
    """``os.chmod`` follows symlinks, so chmod'ing a failing path would rewrite
    the mode of whatever it points at — a side effect outside the workspace
    being deleted. Only the parent directory and real directories are touched."""
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "secret"
    victim.write_text("x")
    os.chmod(victim, 0o600)

    ws = tmp_path / "ws"
    ro = ws / "repo" / "ro"
    ro.mkdir(parents=True)
    os.symlink(victim, ro / "link")
    os.chmod(ro, 0o500)

    kbw._rmtree_force(ws)

    assert not ws.exists()
    assert victim.exists(), "the symlink target must never be removed"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o600


@_needs_posix_permissions
def test_rmtree_force_never_chmods_the_directory_holding_the_workspace(
    tmp_path: Path,
) -> None:
    """When the workspace itself cannot be unlinked, the write bit is missing
    from the directory that *holds* it — the board's ``workspaces/`` root. That
    is outside the tree being deleted, so removal is left to fail rather than
    widening an operator's permissions behind their back."""
    root = tmp_path / "workspaces"
    root.mkdir()
    ws = root / "t_abc"
    ws.mkdir()
    (ws / "f.txt").write_text("x")
    os.chmod(root, 0o500)

    kbw._rmtree_force(ws)

    assert stat.S_IMODE(root.stat().st_mode) == 0o500
    os.chmod(root, 0o700)  # let tmp_path teardown proceed
