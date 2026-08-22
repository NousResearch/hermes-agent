"""Per-task worktree isolation for decompose siblings.

Decompose children used to inherit the root's literal ``workspace_path``,
so every sibling of a worktree-kind root pointed at the SAME checkout —
and ``_resolve_worktree_workspace``'s existing-checkout shortcut reused it
on whatever branch was there, letting sibling workers run concurrently in
one directory on one branch (cross-task provenance corruption, no lock).

Two-part fix under test:
- ``decompose_triage_task`` leaves worktree children's ``workspace_path``
  unset so each child materializes its own ``<repo>/.worktrees/<child-id>``.
- ``_resolve_worktree_workspace`` falls back to a fresh per-task worktree
  when the requested path is occupied by another task's branch (heals
  pre-existing rows that still carry a shared path).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    )


def _git_output(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _commit_file(repo: Path, name: str, content: str, message: str) -> str:
    (repo / name).write_text(content, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-m", message)
    return _git_output(repo, "rev-parse", "HEAD")


def _make_remote_backed_repo(tmp_path: Path, default_branch: str = "main") -> tuple[Path, Path]:
    remote = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True, capture_output=True, text=True,
    )
    seed.mkdir()
    subprocess.run(
        ["git", "init", "-b", default_branch, str(seed)],
        check=True, capture_output=True, text=True,
    )
    _git(seed, "config", "user.name", "Test User")
    _git(seed, "config", "user.email", "test@example.com")
    _commit_file(seed, "README.md", "base\n", "base")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", default_branch)
    _git(remote, "symbolic-ref", "HEAD", f"refs/heads/{default_branch}")
    subprocess.run(
        ["git", "clone", str(remote), str(checkout)],
        check=True, capture_output=True, text=True,
    )
    _git(checkout, "config", "user.name", "Test User")
    _git(checkout, "config", "user.email", "test@example.com")
    return checkout, seed


def _add_worktree(repo: Path, target: Path, branch: str) -> Path:
    _git(repo, "worktree", "add", str(target), "-b", branch, "HEAD")
    return target


def test_new_worktree_branch_uses_fetched_origin_main(tmp_path):
    repo, seed = _make_remote_backed_repo(tmp_path)
    remote_head = _commit_file(seed, "remote.txt", "remote\n", "advance remote")
    _git(seed, "push", "origin", "main")
    local_head = _commit_file(repo, "local.txt", "local\n", "advance local")
    target = repo / ".worktrees" / "new-task"

    kb._ensure_git_worktree(repo, target, "project/new-task")

    assert _git_output(target, "rev-parse", "HEAD") == remote_head
    assert _git_output(repo, "rev-parse", "refs/remotes/origin/main") == remote_head
    assert remote_head != local_head


def test_new_worktree_branch_uses_non_main_remote_default(tmp_path):
    repo, seed = _make_remote_backed_repo(tmp_path, default_branch="trunk")
    remote_head = _commit_file(seed, "remote.txt", "remote\n", "advance remote")
    _git(seed, "push", "origin", "trunk")
    _git(repo, "symbolic-ref", "--delete", "refs/remotes/origin/HEAD")
    _commit_file(repo, "local.txt", "local\n", "advance local")
    target = repo / ".worktrees" / "new-task"

    kb._ensure_git_worktree(repo, target, "project/new-task")

    assert _git_output(target, "rev-parse", "HEAD") == remote_head


def test_new_worktree_branch_prefers_non_main_default_when_main_exists(tmp_path):
    repo, seed = _make_remote_backed_repo(tmp_path, default_branch="trunk")
    _git(seed, "switch", "-c", "main")
    main_head = _commit_file(seed, "main.txt", "main\n", "create main")
    _git(seed, "push", "origin", "main")
    _git(seed, "switch", "trunk")
    remote_head = _commit_file(seed, "remote.txt", "remote\n", "advance trunk")
    _git(seed, "push", "origin", "trunk")
    target = repo / ".worktrees" / "new-task"

    kb._ensure_git_worktree(repo, target, "project/new-task")

    assert _git_output(target, "rev-parse", "HEAD") == remote_head
    assert remote_head != main_head


def test_new_worktree_branch_refreshes_changed_default_when_old_branch_exists(tmp_path):
    repo, seed = _make_remote_backed_repo(tmp_path, default_branch="main")
    remote = Path(_git_output(repo, "remote", "get-url", "origin"))
    main_head = _git_output(repo, "rev-parse", "refs/remotes/origin/main")
    _git(seed, "switch", "-c", "trunk")
    remote_head = _commit_file(seed, "remote.txt", "remote\n", "create trunk")
    _git(seed, "push", "-u", "origin", "trunk")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/trunk")
    target = repo / ".worktrees" / "new-task"

    kb._ensure_git_worktree(repo, target, "project/new-task")

    assert _git_output(target, "rev-parse", "HEAD") == remote_head
    assert remote_head != main_head


def test_new_worktree_branch_fetches_changed_default_outside_refspec(tmp_path):
    repo, seed = _make_remote_backed_repo(tmp_path, default_branch="main")
    remote = Path(_git_output(repo, "remote", "get-url", "origin"))
    _git(seed, "switch", "-c", "trunk")
    stale_trunk_head = _commit_file(seed, "trunk.txt", "old\n", "create trunk")
    _git(seed, "push", "-u", "origin", "trunk")
    _git(repo, "fetch", "origin", "trunk:refs/remotes/origin/trunk")
    _git(
        repo,
        "config",
        "remote.origin.fetch",
        "+refs/heads/main:refs/remotes/origin/main",
    )
    main_head = _git_output(repo, "rev-parse", "refs/remotes/origin/main")
    remote_head = _commit_file(seed, "remote.txt", "remote\n", "advance trunk")
    _git(seed, "push", "origin", "trunk")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/trunk")
    target = repo / ".worktrees" / "new-task"

    kb._ensure_git_worktree(repo, target, "project/new-task")

    assert _git_output(target, "rev-parse", "HEAD") == remote_head
    assert remote_head != stale_trunk_head
    assert remote_head != main_head


def test_new_worktree_branch_refreshes_dangling_origin_head(tmp_path):
    repo, seed = _make_remote_backed_repo(tmp_path, default_branch="master")
    remote = Path(_git_output(repo, "remote", "get-url", "origin"))
    _git(seed, "switch", "-c", "trunk")
    remote_head = _commit_file(seed, "remote.txt", "remote\n", "create trunk")
    _git(seed, "push", "-u", "origin", "trunk")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/trunk")
    _git(seed, "push", "origin", "--delete", "master")
    local_head = _commit_file(repo, "local.txt", "local\n", "advance local master")
    target = repo / ".worktrees" / "new-task"

    kb._ensure_git_worktree(repo, target, "project/new-task")

    assert _git_output(target, "rev-parse", "HEAD") == remote_head
    assert remote_head != local_head


def test_existing_worktree_branch_keeps_its_tip(tmp_path):
    repo, seed = _make_remote_backed_repo(tmp_path)
    _git(repo, "branch", "project/retry", "HEAD")
    existing_tip = _git_output(repo, "rev-parse", "project/retry")
    _commit_file(seed, "remote.txt", "remote\n", "advance remote")
    _git(seed, "push", "origin", "main")
    target = repo / ".worktrees" / "retry"

    kb._ensure_git_worktree(repo, target, "project/retry")

    assert _git_output(target, "rev-parse", "HEAD") == existing_tip


def test_new_worktree_branch_without_origin_falls_back_to_head(tmp_path):
    repo = _make_repo(tmp_path)
    local_head = _commit_file(repo, "local.txt", "local\n", "advance local")
    target = repo / ".worktrees" / "local-task"

    kb._ensure_git_worktree(repo, target, "project/local-task")

    assert _git_output(target, "rev-parse", "HEAD") == local_head


def test_new_worktree_branch_rejects_unreachable_origin(tmp_path):
    repo = _make_repo(tmp_path)
    _git(repo, "remote", "add", "origin", str(tmp_path / "missing-origin.git"))
    target = repo / ".worktrees" / "new-task"

    with pytest.raises(RuntimeError, match="could not fetch origin"):
        kb._ensure_git_worktree(repo, target, "project/new-task")

    assert not target.exists()


def test_new_worktree_branch_rejects_origin_without_url(tmp_path):
    repo = _make_repo(tmp_path)
    _git(
        repo,
        "config",
        "remote.origin.fetch",
        "+refs/heads/*:refs/remotes/origin/*",
    )
    target = repo / ".worktrees" / "new-task"

    with pytest.raises(RuntimeError, match="could not fetch origin"):
        kb._ensure_git_worktree(repo, target, "project/new-task")

    assert not target.exists()


def test_new_worktree_branch_rejects_unresolved_remote_default(tmp_path):
    repo = _make_repo(tmp_path)
    remote = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True, capture_output=True, text=True,
    )
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "origin", "HEAD:refs/heads/trunk")
    target = repo / ".worktrees" / "new-task"

    with pytest.raises(RuntimeError, match="could not resolve a fetched default branch"):
        kb._ensure_git_worktree(repo, target, "project/new-task")

    assert not target.exists()


def test_new_worktree_branch_rejects_failed_live_default_fetch(tmp_path, monkeypatch):
    repo, _seed = _make_remote_backed_repo(tmp_path)
    target = repo / ".worktrees" / "new-task"
    real_run = subprocess.run

    def fail_targeted_fetch(args, *run_args, **run_kwargs):
        if args[-1] == "+refs/heads/main:refs/remotes/origin/main":
            return subprocess.CompletedProcess(args, 1, "", "targeted fetch failed")
        return real_run(args, *run_args, **run_kwargs)

    monkeypatch.setattr(kb.subprocess, "run", fail_targeted_fetch)

    with pytest.raises(RuntimeError, match="could not fetch the live remote default branch"):
        kb._ensure_git_worktree(repo, target, "project/new-task")

    assert not target.exists()


def test_decompose_worktree_children_get_own_workspace(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="build the feature", triage=True)
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', "
            "workspace_path='/repo/.worktrees/root' WHERE id = ?",
            (root,),
        )
        conn.commit()

        child_ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[
                {"title": "spec it", "assignee": "alice", "parents": []},
                {"title": "implement it", "assignee": "bob", "parents": [0]},
            ],
            author="decomposer",
        )
        assert child_ids is not None and len(child_ids) == 2

        for cid in child_ids:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row["workspace_kind"] == "worktree"
            # Each child resolves its own <repo>/.worktrees/<child-id> at
            # dispatch; the root's literal path must never be shared.
            assert row["workspace_path"] is None




def test_resolve_worktree_falls_back_when_path_occupied(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="second sibling",
            workspace_kind="worktree",
            workspace_path=str(occupied),  # inherited shared/stale path
        )
        task = kb.get_task(conn, tid)

    workspace, branch = kb._resolve_worktree_workspace(task)
    assert workspace == (repo / ".worktrees" / tid).resolve()
    assert branch == f"wt/{tid}"
    # The sibling's checkout is untouched, still on its own branch.
    assert (occupied / "README.md").exists()
    head = subprocess.run(
        ["git", "-C", str(occupied), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == "wt/sibling"




