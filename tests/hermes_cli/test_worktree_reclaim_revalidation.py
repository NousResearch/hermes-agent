"""An old cleanup verdict must not delete work added since the audit."""

import os
import subprocess
import time

import pytest


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "repo"
    _git(tmp_path, "init", "-b", "main", str(root))
    _git(root, "config", "user.name", "Hermes test")
    _git(root, "config", "user.email", "hermes-test@example.invalid")
    (root / "README.md").write_text("initial\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-m", "initial")
    tree = root / ".worktrees" / "scratch"
    _git(root, "worktree", "add", "-b", "scratch", str(tree))
    return root, tree


def _git(cwd, *args):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _add_work(tree, kind):
    (tree / "README.md").write_text("new work after audit\n", encoding="utf-8")
    if kind == "commit":
        _git(tree, "add", "README.md")
        _git(tree, "-c", "commit.gpgsign=false", "commit", "-m", "new unique work")


@pytest.mark.parametrize("kind", ["edit", "commit"])
def test_attended_reclaim_preserves_work_added_after_audit(repository, kind):
    from hermes_cli.worktree_gc import audit_worktrees, reclaim_worktrees

    root, tree = repository
    records = audit_worktrees(str(root), with_sizes=False)
    assert next(record for record in records if record.path == str(tree)).verdict == "reap"
    _add_work(tree, kind)

    reclaim_worktrees(str(root), records=records)

    assert tree.is_dir()
    assert (tree / "README.md").read_text(encoding="utf-8") == "new work after audit\n"
    assert _git(root, "rev-parse", "--verify", "scratch")


@pytest.mark.parametrize("kind", ["edit", "commit", "include-symlink"])
def test_startup_reclaim_preserves_work_added_after_classification(repository, kind):
    from hermes_cli.worktree_ops import _classify_prune_candidates, _reap_prune_verdicts

    root, tree = repository
    if kind == "include-symlink":  # our own scaffolding only: still reclaimed, target untouched
        (root / "node_modules").mkdir()
        (root / ".worktreeinclude").write_text("node_modules\n", encoding="utf-8")
        (root / ".gitignore").write_text("node_modules/\n.worktrees/\n", encoding="utf-8")
        os.symlink(root / "node_modules", tree / "node_modules")
    verdicts = _classify_prune_candidates(str(root), [(tree, time.time() - 86400, False)])
    assert verdicts[0][3] == "reap"
    added_work = kind in {"edit", "commit"}
    if added_work:
        _add_work(tree, kind)

    _reap_prune_verdicts(str(root), verdicts, stale_work_cutoff=0)

    assert tree.is_dir() is added_work
    if added_work:
        assert (tree / "README.md").read_text(encoding="utf-8") == "new work after audit\n"
        assert _git(root, "rev-parse", "--verify", "scratch")
    else:
        assert (root / "node_modules").is_dir()
