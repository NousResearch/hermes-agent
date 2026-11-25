"""A ``-w`` session branch holding commits no remote has is never deleted unrecoverably.

Exit cleanup judged only HEAD, so a tree checked out (or detached) onto a pushed ref lost its
branch; the startup orphan-branch pass deleted every worktree-less ``hermes/`` branch outright.
"""

import subprocess

from hermes_cli import worktree_ops
from hermes_cli.worktree_archive import list_archives


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
                          check=True).stdout.strip()


def _repo_with_session(tmp_path, name):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "init")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")  # init is "pushed"
    wt = repo / ".worktrees" / name
    _git(repo, "worktree", "add", "-q", str(wt), "-b", f"hermes/{name}")
    _git(wt, "commit", "-q", "--allow-empty", "-m", "session work")
    return repo, wt, _git(wt, "rev-parse", "HEAD")


def test_exit_keeps_tree_and_branch_when_head_moved_off_unpushed_branch(tmp_path, monkeypatch):
    import cli

    repo, wt, work = _repo_with_session(tmp_path, "hermes-moved")
    _git(wt, "checkout", "-q", "--detach", "origin/main")  # HEAD itself is pushed
    monkeypatch.setattr(cli, "release_lsp_clients", lambda path: None)
    monkeypatch.setattr(cli, "_cprint", lambda *a, **k: None)

    cli._cleanup_worktree({"path": str(wt), "branch": "hermes/hermes-moved", "repo_root": str(repo)})

    assert wt.exists()
    assert _git(repo, "rev-parse", "refs/heads/hermes/hermes-moved") == work


def test_orphan_pass_archives_unpushed_session_branch_before_deleting(tmp_path, monkeypatch):
    from hermes_cli import worktree_archive

    repo, wt, work = _repo_with_session(tmp_path, "hermes-orphan")
    _git(repo, "branch", "hermes/hermes-pushed", "origin/main")  # pushed orphan: no archive needed
    _git(repo, "branch", "hermes/hermes-taken", work)  # unpushed too, checked out mid-pass below
    _git(repo, "worktree", "remove", "--force", str(wt))  # as the exit notice suggests
    real_snapshot = worktree_archive.snapshot_branch

    def snapshot_then_user_checks_out(root, branch, **kw):
        arc = real_snapshot(root, branch, **kw)
        if branch == "hermes/hermes-orphan":
            _git(repo, "worktree", "add", "-q", str(tmp_path / "taken"), "hermes/hermes-taken")
        return arc

    monkeypatch.setattr(worktree_archive, "snapshot_branch", snapshot_then_user_checks_out)

    worktree_ops._prune_orphaned_branches(str(repo))

    assert _git(repo, "branch", "--list", "hermes/*", "--format=%(refname:short)") == "hermes/hermes-taken"
    archived = sorted((a.meta.get("branch"), a.meta.get("head")) for a in list_archives(str(repo)))
    assert archived == [("hermes/hermes-orphan", work), ("hermes/hermes-taken", work)]
