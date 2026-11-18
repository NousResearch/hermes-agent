"""#147245: a conflicted autostash restore must leave a genuinely clean worktree.

``git stash apply`` checks the stash's untracked tree out before it merges the tracked diff, so the
``git reset --hard`` that undoes a conflict used to leave those files behind while the update printed
"Working tree reset to clean state." and the recovery command it printed then failed on every one of
them ("already exists, no checkout").
"""
import subprocess

import pytest

from hermes_cli import main as hermes_main


def _git(cwd, *args, check=True):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


@pytest.fixture
def repo(tmp_path):
    """The checkout lives in its own subdir: the test harness's HERMES_HOME sandbox must not sit
    inside it, or it would show up in ``git status``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    return repo


def _repo_with_conflicting_restore(repo):
    """Tracked edit + untracked file autostashed, then the update rewrites the edited file."""
    (repo / "mod.py").write_text("X = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    (repo / "mod.py").write_text("X = 2\n", encoding="utf-8")
    (repo / "notes.md").write_text("my private notes\n", encoding="utf-8")
    stash_ref = hermes_main._stash_local_changes_if_needed(["git"], repo)
    assert stash_ref
    (repo / "mod.py").write_text("X = 3\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "upstream rewrites mod.py")
    return stash_ref


def test_conflicted_restore_leaves_no_untracked_residue(repo, capsys):
    stash_ref = _repo_with_conflicting_restore(repo)
    # Created after the stash was taken: another process's file, not the restore's to delete.
    (repo / "kept.txt").write_text("not stashed\n", encoding="utf-8")

    restored = hermes_main._restore_stashed_changes(["git"], repo, stash_ref, prompt_user=False)

    out = capsys.readouterr().out
    assert restored is False
    status = [ln for ln in _git(repo, "status", "--porcelain").stdout.splitlines() if ln.strip()]
    assert status == ["?? kept.txt"], status
    assert not (repo / "notes.md").exists()
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "not stashed\n"
    # Nothing was lost: the parked stash still carries the untracked file's content.
    assert _git(repo, "show", f"{stash_ref}^3:notes.md").stdout == "my private notes\n"
    # The cleanup is reported, not silent, and the tree is only called clean once it is. Here it is
    # not: `kept.txt` is a foreign untracked file, so the honest verdict is the warning — claiming
    # "reset to clean state" with a file still in the tree is exactly the #147245 lie.
    assert "notes.md" in out
    assert "Working tree reset to clean state." not in out
    assert "Working tree is NOT clean" in out


def test_a_restore_that_leaves_nothing_behind_is_reported_as_clean(repo, capsys):
    """The other side of the same verdict: no untracked file survives, so the clean claim is true."""
    stash_ref = _repo_with_conflicting_restore(repo)

    hermes_main._restore_stashed_changes(["git"], repo, stash_ref, prompt_user=False)

    out = capsys.readouterr().out
    status = [ln for ln in _git(repo, "status", "--porcelain").stdout.splitlines() if ln.strip()]
    assert status == [], status
    assert "Working tree reset to clean state." in out
    assert "NOT clean" not in out


def test_the_printed_recovery_command_works_after_a_conflicted_restore(repo, capsys):
    """The advice printed by the conflict path must run: no path may already exist."""
    stash_ref = _repo_with_conflicting_restore(repo)

    hermes_main._restore_stashed_changes(["git"], repo, stash_ref, prompt_user=False)
    out = capsys.readouterr().out
    assert f"Restore your changes later with: git stash apply {stash_ref}" in out

    apply_result = _git(repo, "stash", "apply", stash_ref, check=False)

    assert "already exists, no checkout" not in apply_result.stderr
    assert "could not restore untracked files from stash" not in apply_result.stderr
    assert (repo / "notes.md").read_text(encoding="utf-8") == "my private notes\n"
    assert "UU mod.py" in _git(repo, "status", "--porcelain").stdout
