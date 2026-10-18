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
    """I1 + I4: HEAD, staged, unstaged (CRLF) and untracked bytes survive delete -> restore,
    even under eol rules from info/attributes and core.attributesFile (which ``--attr-source``
    does not replace); the archive step leaves the real index and every working file untouched."""
    from hermes_cli.worktree_archive import restore_archive, snapshot_worktree

    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "attributes").write_text("*.md text eol=crlf\n", encoding="utf-8")
    (repo / "attrs").write_text("*.cfg text eol=lf\n", encoding="utf-8")
    _git(["config", "core.attributesFile", str(repo / "attrs")], repo)
    wt = _tree(repo, "lane-roundtrip", unpushed=True)
    (wt / "lf.md").write_bytes(b"lf only\n")                 # restore must not add CR
    (wt / "crlf.cfg").write_bytes(b"keep\r\nthe CR\r\n")     # archive must not strip CR
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
    subagent = _tree(repo, "subagent-done", dirty=True)
    deleted = _tree(repo, "lane-deleted")
    (deleted / "sub").mkdir()
    (deleted / "sub" / "x.py").write_text("x\n", encoding="utf-8")
    (deleted / "sub" / "y.py").write_text("y\n", encoding="utf-8")
    _git(["add", "sub"], deleted)
    _git(["commit", "-qm", "sub"], deleted)
    for tree in (clean, dirty, unpushed, live_cwd, live_lock, kanban, unarchivable, deleted):
        _idle(tree)
    _idle(subagent, hours=30)                   # past the 24h hermes-made tier, inside 72h
    (deleted / "sub" / "x.py").unlink()         # the newest work: a deletion in a subdir
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], cwd=str(live_cwd))
    try:
        worktree_ops._prune_stale_worktrees(str(repo))
    finally:
        sleeper.kill()
        sleeper.wait()

    assert not clean.exists()
    for gone in (dirty, unpushed):
        assert not gone.exists(), f"idle {gone.name} should be archived then removed"
    assert not subagent.exists(), "subagent-* trees age on the 24h tier"
    archived = {ref.rsplit("/", 1)[-1] for ref in _archives(repo)}
    assert archived == {"lane-dirty", "hermes-unpushed", "subagent-done"}
    assert (deleted / "sub" / "y.py").exists(), "a fresh deletion is activity: the tree is not idle"
    assert _git(["show", "refs/hermes/archive/lane-dirty:notes.md"], repo) == "uncommitted"
    for kept in (live_cwd, live_lock, kanban, young, unarchivable):
        assert kept.exists() and (kept / "notes.md").read_text(encoding="utf-8-sig") == "uncommitted\n", kept.name


def test_a_tree_resumed_mid_sweep_is_never_removed(repo, monkeypatch, capsys):
    """I2 at the removal boundary: a process entering a later tree while an earlier one is being
    archived, an edit made while a tree's own snapshot ran, or an index-only edit (``git apply
    --cached``: no file, HEAD or reflog mtime moves) made right after capture keeps that tree, and
    the sweep reports it as archived-but-kept, not removed."""
    from hermes_cli import worktree_archive, worktree_cmd, worktree_sweep

    first, second = _tree(repo, "lane-a", dirty=True), _tree(repo, "lane-b", dirty=True)
    staged = _tree(repo, "lane-staged", dirty=True)
    for tree in (first, second, staged):
        _idle(tree)
    real_snapshot, sleepers = worktree_archive.snapshot_worktree, []
    patch = ("--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-x = 1\n+NEW STAGED CONTENT\n")

    def snapshot_then_user_returns(repo_root, wt_path, **kw):
        arc = real_snapshot(repo_root, wt_path, **kw)
        if Path(wt_path).name == "lane-staged":
            subprocess.run(["git", "apply", "--cached", "-"], cwd=str(wt_path), input=patch.encode(), check=True)
            subprocess.run(["git", "status", "--porcelain"], cwd=str(wt_path), capture_output=True, check=True)
            return arc
        (Path(wt_path) / "notes.md").write_text("edited during the snapshot\n", encoding="utf-8")
        sleepers.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=str(second)))
        return arc

    monkeypatch.setattr(worktree_archive, "snapshot_worktree", snapshot_then_user_returns)
    try:
        plan, actions = worktree_cmd._sweep(str(repo), dry_run=False)
    finally:
        for proc in sleepers:
            proc.kill()
            proc.wait()
    assert {d.name for d in plan if d.action == "archive-remove"} == {"lane-a", "lane-b", "lane-staged"}
    assert (first / "notes.md").read_text(encoding="utf-8-sig") == "edited during the snapshot\n"
    assert second.exists() and (second / "notes.md").read_text(encoding="utf-8-sig") == "uncommitted\n"
    assert staged.exists() and _git(["show", ":b.py"], staged) == "NEW STAGED CONTENT"
    assert sorted(ref.rsplit("/", 1)[-1] for ref in _archives(repo)) == ["lane-a", "lane-staged"]
    assert sorted(actions) == ["archived, kept (changed during the sweep) → refs/hermes/archive/lane-a",
                               "archived, kept (changed during the sweep) → refs/hermes/archive/lane-staged"]


def test_dirty_submodule_work_is_never_archived_away(repo, tmp_path, monkeypatch):
    """C1: a snapshot records a submodule only as its gitlink, and removing a tree deletes its
    ``modules/`` repositories. So the real sweep keeps, byte-for-byte: a tree with a dirty
    submodule; a deinitialized one whose retained module repository holds an unpushed private
    commit; and one whose submodule file is edited right after the snapshot step. The shared
    snapshot entry point (exit-time Archive, subagent finalization) refuses them too."""
    from hermes_cli import worktree_archive, worktree_sweep
    from hermes_cli.worktree_archive import ArchiveError, snapshot_worktree

    sub_origin, sub_work = tmp_path / "sub.git", tmp_path / "subwork"
    _git(["init", "-q", "--bare", "-b", "main", str(sub_origin)], tmp_path)
    _git(["init", "-q", "-b", "main", str(sub_work)], tmp_path)
    (sub_work / "file.txt").write_text("base\n", encoding="utf-8")
    _git(["add", "-A"], sub_work)
    _git(["commit", "-qm", "sub"], sub_work)
    _git(["push", "-q", str(sub_origin), "main"], sub_work)
    _git(["-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub_origin), "sub"], repo)
    _git(["commit", "-qm", "add submodule"], repo)
    trees = {}
    for name in ("lane-subdirty", "lane-deinit", "lane-late"):
        trees[name] = _tree(repo, name, dirty=True)
        _git(["-c", "protocol.file.allow=always", "submodule", "update", "-q", "--init"], trees[name])
    dirty_file = trees["lane-subdirty"] / "sub" / "file.txt"
    dirty_file.write_bytes(b"base\nUNCOMMITTED SUBMODULE EDIT\r\n")
    deinit_sub = trees["lane-deinit"] / "sub"
    module_repo = Path(_git(["rev-parse", "--absolute-git-dir"], deinit_sub))
    _git(["switch", "-qc", "private-work"], deinit_sub)
    (deinit_sub / "file.txt").write_text("UNPUSHED\n", encoding="utf-8")
    _git(["commit", "-qam", "private"], deinit_sub)
    private = _git(["rev-parse", "HEAD"], deinit_sub)
    _git(["checkout", "-q", "--detach", "origin/main"], deinit_sub)
    _git(["submodule", "deinit", "-q", "sub"], trees["lane-deinit"])
    for tree in trees.values():
        _idle(tree)
    late_file, late_bytes = trees["lane-late"] / "sub" / "file.txt", [b"base\n"]
    real_snapshot = worktree_archive.snapshot_worktree

    def snapshot_then_submodule_edit(repo_root, wt_path, **kw):
        try:
            return real_snapshot(repo_root, wt_path, **kw)
        finally:
            if Path(wt_path).name == "lane-late":
                late_bytes.append(b"LATE SUBMODULE EDIT\n")
                late_file.write_bytes(late_bytes[-1])

    monkeypatch.setattr(worktree_archive, "snapshot_worktree", snapshot_then_submodule_edit)
    plan = {d.name: d.action for d in worktree_sweep.plan_sweep(str(repo), settings=worktree_sweep.sweep_settings())}
    assert set(plan.values()) == {"keep-submodule-work"}

    worktree_ops._prune_stale_worktrees(str(repo))

    assert dirty_file.read_bytes() == b"base\nUNCOMMITTED SUBMODULE EDIT\r\n"
    assert _git(["--git-dir", str(module_repo), "rev-parse", "private-work"], repo) == private
    assert late_file.read_bytes() == late_bytes[-1]
    assert _archives(repo) == []
    for tree in trees.values():
        with pytest.raises(ArchiveError, match="submodule"):
            snapshot_worktree(str(repo), tree, reason="session exit")


def test_prune_older_than_bounds_what_the_sweep_removes(repo, capsys):
    """`hermes worktree prune --older-than N` never archives or removes a tree idle < N days."""
    import argparse

    from hermes_cli.worktree_cmd import cmd_worktree

    dirty, clean = _tree(repo, "hermes-dirty", dirty=True), _tree(repo, "hermes-clean")
    for tree in (dirty, clean):
        _idle(tree)                                   # ~4 days idle: past every tier
    args = argparse.Namespace(repo=str(repo), worktree_action="prune", dry_run=False, json=True,
                              older_than=30, trees_only=True, branches_only=False)
    assert cmd_worktree(args) == 0
    assert dirty.exists() and clean.exists()
    assert _archives(repo) == []


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
