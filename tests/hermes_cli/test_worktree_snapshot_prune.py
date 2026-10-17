"""Invariants of the snapshot-then-delete worktree sweep (``worktrees.archive_dirty``).

Real git repos, real processes, no mocks: an idle dirty/unpushed tree past its tier is removed
only after a restorable archive exists (I1/I4); live, kanban and young trees are never touched
(I2/I3); ``archive_dirty: false`` keeps the classic never-delete-dirty pruner (I6); the cron sweep
visits the repos the ACTIVE profile registered (I7); a non-interactive exit never prompts (I5).
"""

import hashlib
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
    origin = tmp_path / "origin.git"
    _git(["init", "-q", "--bare", "-b", "main", str(origin)], tmp_path)
    clone = tmp_path / "repo"
    _git(["init", "-q", "-b", "main", str(clone)], tmp_path)
    (clone / ".gitignore").write_text(".worktrees/\nnode_modules/\n", encoding="utf-8")
    (clone / ".gitattributes").write_text("*.txt text eol=crlf\n", encoding="utf-8")
    (clone / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    (clone / "b.py").write_text("x = 1\n", encoding="utf-8")
    _git(["add", "-A"], clone)
    _git(["commit", "-qm", "init"], clone)
    _git(["remote", "add", "origin", str(origin)], clone)
    _git(["push", "-q", "origin", "main"], clone)
    _git(["fetch", "-q", "origin"], clone)
    (clone / ".worktrees").mkdir()
    return clone


def _tree(repo, name, *, dirty=False, unpushed=False):
    path = repo / ".worktrees" / name
    _git(["worktree", "add", "-q", "-b", f"wt/{name}", str(path), "HEAD"], repo)
    if unpushed:
        (path / "feature.py").write_text(f"work for {name}\n", encoding="utf-8")
        _git(["add", "feature.py"], path)
        _git(["commit", "-qm", "wip"], path)
    if dirty:
        (path / "notes.md").write_text("uncommitted\n", encoding="utf-8")
    return path


def _idle(path, hours=100):
    """Age the tree AND its activity (files, HEAD, reflog) so the sweep sees it as idle."""
    stamp = time.time() - hours * 3600
    git_dir = Path(_git(["rev-parse", "--absolute-git-dir"], path))
    for item in [*path.rglob("*"), git_dir / "HEAD", git_dir / "logs" / "HEAD", path]:
        if item.exists():
            os.utime(item, (stamp, stamp), follow_symlinks=False)


def _archives(repo):
    out = _git(["for-each-ref", "--format=%(refname)", "refs/hermes/archive/"], repo)
    return [line for line in out.splitlines() if line]


def _digest(path):
    return {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(path.rglob("*")) if f.is_file() and ".git" not in f.parts and "node_modules" not in f.parts}


def _set_worktrees_config(text):
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


def test_removed_dirty_tree_restores_byte_for_byte_and_archiving_never_touches_the_tree(repo):
    """I1 + I4: HEAD, staged, unstaged (CRLF) and untracked bytes survive delete -> restore;
    the archive step leaves the real index and every working file untouched."""
    from hermes_cli.worktree_archive import restore_archive, snapshot_worktree

    wt = _tree(repo, "lane-roundtrip", unpushed=True)
    (wt / "b.py").write_text("x = 2\n", encoding="utf-8")
    _git(["add", "b.py"], wt)
    (wt / "b.py").write_text("x = 3\n", encoding="utf-8")               # unstaged on top of a staged change
    (wt / "a.txt").write_bytes(b"one\r\nTWO\r\n")     # raw CRLF bytes under eol=crlf
    (wt / "new dir").mkdir()
    (wt / "new dir" / "blob.bin").write_bytes(os.urandom(4096))
    (wt / "node_modules").mkdir()
    (wt / "node_modules" / "dep.js").write_text("ignored, never archived\n", encoding="utf-8")
    (wt / "big.dat").write_bytes(b"\0" * (2 * 1024 * 1024))
    git_dir = Path(_git(["rev-parse", "--absolute-git-dir"], wt))
    index_before, files_before = (git_dir / "index").read_bytes(), _digest(wt)
    head, staged = _git(["rev-parse", "HEAD"], wt), _git(["diff", "--cached"], wt)

    arc = snapshot_worktree(str(repo), wt, reason="test", max_untracked_mb=1)
    assert (git_dir / "index").read_bytes() == index_before
    assert _digest(wt) == files_before
    assert [s["path"] for s in arc.meta["skipped_untracked"]] == ["big.dat"]
    assert "node_modules/dep.js" not in _git(["ls-tree", "-r", "--name-only", arc.sha], repo)

    _git(["worktree", "remove", "--force", str(wt)], repo)
    _git(["branch", "-D", "wt/lane-roundtrip"], repo)
    restored = restore_archive(str(repo), "lane-roundtrip")
    assert restored["path"] == str(wt) and restored["branch"] == "wt/lane-roundtrip"
    assert _digest(wt) == {k: v for k, v in files_before.items() if k != "big.dat"}
    assert _git(["rev-parse", "HEAD"], wt) == head
    assert _git(["diff", "--cached"], wt) == staged


def test_sweep_archives_idle_work_and_never_touches_live_kanban_or_young_trees(repo):
    """I1/I2/I3/I4 on one board through the real pruner entry point."""
    clean = _tree(repo, "lane-clean")
    dirty = _tree(repo, "lane-dirty", dirty=True)
    unpushed = _tree(repo, "hermes-unpushed", unpushed=True)
    live_cwd = _tree(repo, "lane-live-cwd", dirty=True)
    live_lock = _tree(repo, "hermes-live-lock", dirty=True)
    _git(["worktree", "lock", "--reason", f"hermes pid={os.getpid()}", str(live_lock)], repo)
    kanban = _tree(repo, "t_0abc12", dirty=True)
    young = _tree(repo, "lane-young", dirty=True)
    unarchivable = _tree(repo, "lane-embedded", dirty=True)
    _git(["init", "-q", str(unarchivable / "inner")], repo)
    (unarchivable / "inner" / "f").write_text("x", encoding="utf-8")
    for tree in (clean, dirty, unpushed, live_cwd, live_lock, kanban, unarchivable):
        _idle(tree)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], cwd=str(live_cwd))
    try:
        worktree_ops._prune_stale_worktrees(str(repo))
    finally:
        sleeper.kill()
        sleeper.wait()

    assert not clean.exists()
    for gone in (dirty, unpushed):
        assert not gone.exists(), f"idle {gone.name} should be archived then removed"
    archived = {ref.rsplit("/", 1)[-1] for ref in _archives(repo)}
    assert archived == {"lane-dirty", "hermes-unpushed"}
    assert _git(["show", "refs/hermes/archive/lane-dirty:notes.md"], repo) == "uncommitted"
    for kept in (live_cwd, live_lock, kanban, young, unarchivable):
        assert kept.exists() and (kept / "notes.md").read_text(encoding="utf-8-sig") == "uncommitted\n", kept.name


def test_archive_dirty_false_keeps_the_classic_never_delete_dirty_pruner(repo):
    """I6: the opt-out removes exactly what the classic pruner removes and archives nothing."""
    _set_worktrees_config("worktrees:\n  archive_dirty: false\n")
    clean = _tree(repo, "lane-clean")
    dirty = _tree(repo, "lane-dirty", dirty=True)
    unpushed = _tree(repo, "hermes-unpushed", unpushed=True)
    for tree in (clean, dirty, unpushed):
        _idle(tree)
    worktree_ops._prune_stale_worktrees(str(repo))
    assert not clean.exists()
    assert dirty.exists() and unpushed.exists()
    assert _archives(repo) == []


def test_cron_sweep_visits_only_the_active_profiles_registered_repos(repo, tmp_path, monkeypatch):
    """I7: a worktree made under profile A registers its repo for A's sweep only, and a
    registered repo that was deleted drops out."""
    import cron.scheduler as sched
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    monkeypatch.setattr("cron.jobs.load_jobs", list)
    token = set_hermes_home_override(str(home_a))
    try:
        assert worktree_ops._setup_worktree(repo_root=str(repo), sync_base=False, name="reg") is not None
        gone = tmp_path / "gone"
        _git(["init", "-q", str(gone)], tmp_path)
        (gone / ".worktrees").mkdir()
        from hermes_cli.worktree_registry import record_worktree_repo
        record_worktree_repo(gone)
        import shutil
        shutil.rmtree(gone)
        repos_a = sched._worktree_maintenance_repos()
    finally:
        reset_hermes_home_override(token)
    token = set_hermes_home_override(str(home_b))
    try:
        repos_b = sched._worktree_maintenance_repos()
    finally:
        reset_hermes_home_override(token)
    assert os.path.realpath(repo) in repos_a
    assert str(gone) not in repos_a and os.path.realpath(gone) not in repos_a
    assert os.path.realpath(repo) not in repos_b


def test_exit_question_never_prompts_without_a_tty(repo, monkeypatch):
    """I5: a dirty tree at a non-interactive exit is kept without any prompt."""
    from hermes_cli import curses_ui, worktree_exit

    wt = _tree(repo, "hermes-exit", dirty=True)
    monkeypatch.setattr(curses_ui, "curses_radiolist",
                        lambda *a, **k: pytest.fail("non-interactive exit must never prompt"))
    info = {"path": str(wt), "repo_root": str(repo), "branch": "wt/hermes-exit"}
    assert worktree_exit.ask_exit_choice(info) == worktree_exit.KEEP
    assert "exit_choice" not in info and wt.exists()
