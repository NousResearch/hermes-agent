"""Branch GC keeps a branch whenever a git probe that would vouch for deleting it fails.

Regression for #149593: ``audit_branches`` parsed the empty stdout of a failed
``git rev-list --count`` as "0 commits ahead" and classified the branch ``delete``,
so ``hermes worktree prune`` removed branches that still held unique commits. A
failed ``git worktree list`` likewise emptied the checked-out set, so branches
checked out in a worktree were reported ``delete``. Each failure is also named:
the keep reason (and the prune output) carries git's exit code and error, and a
skipped audit logs a warning.

The real-git row breaks the range query the way the report did (a missing commit
object). The timeout rows go through ``worktree_gc._git``'s own timeout translation
(returncode 124, empty stdout) for one subcommand, against an otherwise intact repo.
"""

import os
import stat
import subprocess

import pytest

from hermes_cli import worktree_gc


def _git(args, cwd):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    result = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",
                            errors="replace", cwd=str(cwd), env=env, check=False)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """origin (bare) + clone, HOME redirected; ``main`` pushed and ``origin/HEAD`` set."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(["init", "--bare", "-b", "main"], origin)
    clone = tmp_path / "repo"
    _git(["clone", str(origin), str(clone)], tmp_path)
    (clone / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["add", "."], clone)
    _git(["commit", "-m", "init"], clone)
    _git(["push", "origin", "main"], clone)
    _git(["remote", "set-head", "origin", "main"], clone)
    (clone / ".worktrees").mkdir()
    return clone


def _branch_with_unique_commit(repo, name):
    _git(["checkout", "-b", name], repo)
    (repo / "wip.py").write_text("unique = 1\n", encoding="utf-8")
    _git(["add", "."], repo)
    _git(["commit", "-m", "unique work"], repo)
    sha = _git(["rev-parse", "HEAD"], repo)
    _git(["checkout", "main"], repo)
    return sha


def _time_out(monkeypatch, subcommand):
    """Make ``worktree_gc._git`` report a timeout for one git subcommand, as it does for a real one."""
    real_git = worktree_gc._git

    def git(args, cwd, timeout=15):
        if args[0] == subcommand:
            return subprocess.CompletedProcess(args=["git", *args], returncode=124, stdout="",
                                               stderr=f"timeout after {timeout}s")
        return real_git(args, cwd, timeout)

    monkeypatch.setattr(worktree_gc, "_git", git)


def _local_branches(repo):
    return _git(["branch", "--format=%(refname:short)"], repo).splitlines()


def test_unique_branch_survives_prune_when_its_range_query_fails(repo):
    sha = _branch_with_unique_commit(repo, "feat/unique")
    loose = repo / ".git" / "objects" / sha[:2] / sha[2:]
    os.chmod(loose, stat.S_IWRITE | stat.S_IREAD)  # git writes loose objects read-only
    loose.unlink()

    records = worktree_gc.audit_branches(str(repo))
    record = next(r for r in records if r.name == "feat/unique")
    assert record.verdict == "keep", record.reason

    actions = worktree_gc.reclaim_branches(str(repo), records=records)
    assert "feat/unique" in _local_branches(repo)
    # The run says why the branch was kept and which git call failed, not "nothing to reclaim".
    assert [a for a in actions if a.startswith("kept branch feat/unique:") and "git rev-list exited" in a]


def test_unique_branch_kept_when_the_range_query_times_out(repo, monkeypatch):
    _branch_with_unique_commit(repo, "feat/slow")
    _time_out(monkeypatch, "rev-list")

    records = worktree_gc.audit_branches(str(repo))
    record = next(r for r in records if r.name == "feat/slow")
    assert record.verdict == "keep", record.reason
    assert "timeout after" in record.reason


def test_no_branch_is_deleted_when_the_worktree_listing_fails(repo, monkeypatch, caplog):
    """A merged branch checked out in a worktree must not read as deletable just
    because the listing that proves it is checked out could not run."""
    _git(["worktree", "add", str(repo / ".worktrees" / "busy"), "-b", "hermes/busy"], repo)
    _time_out(monkeypatch, "worktree")

    with caplog.at_level("WARNING", logger=worktree_gc.__name__):
        records = worktree_gc.audit_branches(str(repo))
    assert [r.name for r in records if r.verdict == "delete"] == []
    assert "git worktree exited 124" in caplog.text
