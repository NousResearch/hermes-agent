"""``hermes update`` moves only the local changes the update would collide with, and asks before stashing.

An untracked directory the update never writes (a 55 GB leftover worktree copy) used to be hashed into
the autostash and deleted from disk on every update; declining the restore then parked it in git stash.
"""

import subprocess

from hermes_cli import update_cmd_stash


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
