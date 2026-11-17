"""``worktrees.shed_artifacts_after_hours``: the sweep drops an idle tree's git-ignored dependency
dirs and nothing else. Real git repos and the real ``_prune_stale_worktrees`` entry point."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import worktree_ops


def _git(args, cwd):
    result = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=str(cwd),
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    clone = tmp_path / "repo"
    _git(["init", "-q", "-b", "main", str(clone)], tmp_path)
    (clone / ".gitignore").write_text(".worktrees/\nnode_modules/\n.venv/\ndist/\n", encoding="utf-8")
    (clone / "app.js").write_text("console.log(1)\n", encoding="utf-8")
    (clone / "pkg").mkdir()
    (clone / "pkg" / "index.js").write_text("module.exports = 1\n", encoding="utf-8")
    _git(["add", "-A"], clone)
    _git(["commit", "-qm", "init"], clone)
    (clone / ".worktrees").mkdir()
    return clone


def _tree_with_artifacts(repo, name, hours):
    path = repo / ".worktrees" / name
    _git(["worktree", "add", "-q", "-b", f"wt/{name}", str(path), "HEAD"], repo)
    (path / "app.js").write_text("console.log('uncommitted')\n", encoding="utf-8")   # dirty tracked
    (path / "notes.md").write_text("untracked, not ignored\n", encoding="utf-8")
    for rel in ("node_modules/left-pad/index.js", "pkg/node_modules/dep/x.js", ".venv/bin/python", "dist/bundle.js"):
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text("artifact\n", encoding="utf-8")
    stamp = time.time() - hours * 3600
    git_dir = Path(_git(["rev-parse", "--absolute-git-dir"], path))
    for item in [*path.rglob("*"), git_dir / "HEAD", git_dir / "logs" / "HEAD", path]:
        os.utime(item, (stamp, stamp), follow_symlinks=False)
    return path


def _config(text):
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


def test_idle_tree_sheds_only_ignored_dependency_dirs_and_keeps_its_work(repo):
    """Opt-in: an idle tree loses node_modules (nested too) and .venv; its tracked edit, untracked
    note, a non-allowlisted ignored dir, its branch and its age clock stay. A young tree and a
    tree a live process works in keep everything."""
    _config("worktrees:\n  shed_artifacts_after_hours: 24\n")
    idle = _tree_with_artifacts(repo, "lane-idle", hours=30)   # named tier is 72h: the tree survives
    young = _tree_with_artifacts(repo, "lane-young", hours=2)
    busy = _tree_with_artifacts(repo, "lane-busy", hours=30)
    idle_mtime = idle.stat().st_mtime
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], cwd=str(busy))
    try:
        worktree_ops._prune_stale_worktrees(str(repo))
    finally:
        sleeper.kill()
        sleeper.wait()

    for gone in ("node_modules", "pkg/node_modules", ".venv"):
        assert not (idle / gone).exists(), f"{gone} should be shed"
    assert (idle / "app.js").read_text(encoding="utf-8-sig") == "console.log('uncommitted')\n"
    assert (idle / "notes.md").exists() and (idle / "pkg" / "index.js").exists()
    assert (idle / "dist" / "bundle.js").exists(), "only allowlisted names may go"
    assert _git(["rev-parse", "--abbrev-ref", "HEAD"], idle) == "wt/lane-idle"
    assert _git(["diff", "--name-only"], idle) == "app.js"
    assert idle.stat().st_mtime == idle_mtime, "shedding must not reset the tree's idle clock"
    for kept in (young, busy):
        assert (kept / "node_modules" / "left-pad" / "index.js").exists()
        assert (kept / ".venv" / "bin" / "python").exists()


def test_shedding_is_off_by_default(repo):
    idle = _tree_with_artifacts(repo, "lane-idle", hours=60)
    worktree_ops._prune_stale_worktrees(str(repo))
    assert (idle / "node_modules" / "left-pad" / "index.js").exists()
    assert (idle / ".venv" / "bin" / "python").exists()
