"""``hermes update`` moves only the local changes the update would collide with, and asks before stashing.

An untracked directory the update never writes (a 55 GB leftover worktree copy) used to be hashed into
the autostash and deleted from disk on every update; declining the restore then parked it in git stash.
"""

import subprocess
import sys

import hermes_cli.main as cli_main
from hermes_cli import update_cmd, update_cmd_stash


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _behind_clone(tmp_path):
    up, work = tmp_path / "up", tmp_path / "work"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    for repo_cfg in (("user.email", "t@t"), ("user.name", "t")):
        _git(up, "config", *repo_cfg)
    (up / "a.txt").write_text("a\n")
    _git(up, "add", ".")
    _git(up, "commit", "-qm", "init")
    _git(tmp_path, "clone", "-q", str(up), str(work))
    (up / "pkg").mkdir()
    (up / "pkg" / "new.py").write_text("upstream\n")
    _git(up, "add", ".")
    _git(up, "commit", "-qm", "adds pkg/new.py")
    _git(work, "fetch", "-q", "origin")
    _git(work, "config", "status.showUntrackedFiles", "no")  # the scan must not inherit display settings
    (work / "a.txt").write_text("a\nmine\n")  # tracked edit
    (work / "pkg").mkdir()
    (work / "pkg" / "new.py").write_text("my copy\n")  # untracked, collides with the update
    (work / "stray").mkdir()
    (work / "stray" / "big.bin").write_text("x" * 4096)  # untracked, the update never writes here
    return work


_TARGETS = ("origin/main", "refs/heads/main")


def test_only_colliding_untracked_files_are_stashed(tmp_path):
    work = _behind_clone(tmp_path)

    ref = update_cmd_stash._stash_local_changes_if_needed(["git"], work, untracked_targets=_TARGETS)

    stashed = _git(work, "stash", "show", "--include-untracked", "--name-only", ref).split()
    assert sorted(stashed) == ["a.txt", "pkg/new.py"]
    assert (work / "stray" / "big.bin").read_text() == "x" * 4096
    _git(work, "merge", "--ff-only", "-q", "origin/main")  # the left-behind file blocks nothing


def test_declining_up_front_discards_without_a_stash_entry(tmp_path, monkeypatch):
    work = _behind_clone(tmp_path)
    monkeypatch.setattr("builtins.input", lambda *a: "n")

    ref = update_cmd_stash._stash_local_changes_if_needed(
        ["git"], work, untracked_targets=_TARGETS, ask_first=True)

    assert ref is None
    assert _git(work, "stash", "list") == ""
    assert (work / "a.txt").read_text() == "a\n"
    assert not (work / "pkg" / "new.py").exists()
    assert (work / "stray" / "big.bin").exists()
    _git(work, "merge", "--ff-only", "-q", "origin/main")


def test_up_to_date_checkout_never_offers_to_delete(tmp_path, monkeypatch):
    work = _behind_clone(tmp_path)
    _git(work, "stash", "push", "-q", "--include-untracked")
    _git(work, "merge", "--ff-only", "-q", "origin/main")
    (work / "a.txt").write_text("a\nmine\n")  # dirty, already on the target commit
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", work)
    monkeypatch.chdir(work)
    for stream in (sys.stdin, sys.stdout):
        monkeypatch.setattr(stream, "isatty", lambda: True, raising=False)

    def refuse(*_a):
        raise AssertionError("asked keep/delete on an update with nothing to pull")
    monkeypatch.setattr("builtins.input", refuse)

    plan = update_cmd._prepare_checkout_for_update(
        ["git"], "main", "main", is_fork=False, assume_yes=False, gateway_mode=False,
        gw_input_fn=None, switch_branch=False, _windows_gateway_resume=None)

    assert plan.commit_count == 0
    assert "a.txt" in _git(work, "stash", "show", "--name-only", plan.auto_stash_ref)  # kept, not deleted
