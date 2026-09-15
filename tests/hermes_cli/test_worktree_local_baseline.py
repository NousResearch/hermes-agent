"""No-remote local-primary baseline for worktree/branch reclaim.

Complements the remote-ref fail-safe: unique commits vs local main/master are
kept, and trees/branches already on that primary remain reclaimable. Uses real
git repos (no mocks).
"""

import os
import subprocess

from hermes_cli import worktree_gc
from hermes_cli import worktree_ops


def _git(args, cwd, env=None):
    e = dict(os.environ)
    e.update({
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    })
    if env:
        e.update(env)
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=str(cwd), env=e,
    )
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout.strip()


def _no_remote_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    repo = tmp_path / "local-repo"
    repo.mkdir()
    _git(["init", "-b", "main", "."], repo)
    (repo / "README.md").write_text("hello\n")
    _git(["add", "."], repo)
    _git(["commit", "-m", "init"], repo)
    (repo / ".worktrees").mkdir()
    return repo


def _add_worktree(repo_path, name, branch=None):
    tree = repo_path / ".worktrees" / name
    branch = branch or f"hermes/{name}"
    _git(["worktree", "add", str(tree), "-b", branch], repo_path)
    return tree, branch


def _verdict(records, name):
    match = [record for record in records if record.name == name]
    assert match, f"no record for {name}"
    return match[0]


def test_unique_commits_without_remote_keep_tree_and_branch(tmp_path, monkeypatch):
    """Issue #111895: unique local work must not be labeled fully merged/pushed."""
    repo = _no_remote_repo(tmp_path, monkeypatch)
    tree, branch = _add_worktree(repo, "hermes-local-work")
    (tree / "new.py").write_text("x = 1\n")
    _git(["add", "."], tree)
    _git(["commit", "-m", "unique local work"], tree)

    records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
    record = _verdict(records, "hermes-local-work")
    assert record.verdict == "keep"
    assert "unique" in record.reason
    assert worktree_gc.reclaim_worktrees(str(repo), records=records) == []
    assert tree.exists()
    assert _git(["rev-parse", "--verify", branch], repo)


def test_untouched_tree_without_remote_reaps(tmp_path, monkeypatch):
    """Local primary still proves a clean unused tree is redundant."""
    repo = _no_remote_repo(tmp_path, monkeypatch)
    tree, branch = _add_worktree(repo, "hermes-clean")
    records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
    assert _verdict(records, "hermes-clean").verdict == "reap"
    actions = worktree_gc.reclaim_worktrees(str(repo), records=records)
    assert any("removed hermes-clean" in a for a in actions)
    assert not tree.exists()
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", branch],
        capture_output=True, text=True, cwd=str(repo),
    )
    assert probe.returncode != 0


def test_locally_merged_unique_sha_without_remote_reaps(tmp_path, monkeypatch):
    """Squash/cherry onto local main is reclaimable even with no origin."""
    repo = _no_remote_repo(tmp_path, monkeypatch)
    tree, _ = _add_worktree(repo, "hermes-merged")
    (tree / "feat.py").write_text("y = 2\n")
    _git(["add", "."], tree)
    _git(["commit", "-m", "feat"], tree)
    sha = _git(["rev-parse", "HEAD"], tree)
    _git(
        ["cherry-pick", sha], repo,
        env={"GIT_COMMITTER_NAME": "other", "GIT_COMMITTER_EMAIL": "o@o"},
    )
    records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
    assert _verdict(records, "hermes-merged").verdict == "reap"


def test_audit_branches_uses_local_primary_without_origin(tmp_path, monkeypatch):
    repo = _no_remote_repo(tmp_path, monkeypatch)
    _git(["branch", "salv-12345", "main"], repo)
    _git(["checkout", "-b", "feat/real-work"], repo)
    (repo / "wip.py").write_text("z = 3\n")
    _git(["add", "."], repo)
    _git(["commit", "-m", "wip"], repo)
    _git(["checkout", "main"], repo)

    records = worktree_gc.audit_branches(str(repo))
    by_name = {record.name: record for record in records}
    assert by_name["salv-12345"].verdict == "delete"
    assert by_name["feat/real-work"].verdict == "keep"
    assert "unique" in by_name["feat/real-work"].reason


def test_merged_upstream_predicate_still_false_without_origin(tmp_path, monkeypatch):
    """Remote-upstream helper stays fail-closed; local proof is a separate function."""
    repo = _no_remote_repo(tmp_path, monkeypatch)
    tree, _ = _add_worktree(repo, "hermes-noremote")
    assert worktree_ops._worktree_commits_all_merged_upstream(str(tree)) is False
    assert worktree_ops._worktree_commits_all_merged_locally(str(tree)) is True
