"""The stash and churn steps bind the paused gateways' tree gate too (#132338 R3 follow-up).

``git stash push`` / ``stash apply`` and the lockfile-churn ``checkout -- <paths>`` rewrite tracked
files while the Windows pause holds the gateways. Each now records, before git runs, the HEAD it
starts from, the tree's dirty bytes and what it may write; a step interrupted mid-write keeps the
set held instead of being judged against unrelated refs (or, at a moved HEAD, against nothing).

Real git, real update steps, real durable record and gate. The interruption is real too: a git
wrapper copies the durable record as git starts (what was on disk before the write), runs real git,
leaves one file half-written and sends the updater SIGINT, as a terminal's Ctrl-C reaches both.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from hermes_cli import update_cmd, update_cmd_git, update_cmd_stash
from hermes_cli import update_pause_record as pause_record

pytestmark = pytest.mark.platforms("posix")


def git(root, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def commit(root, message, **files) -> str:
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-qm", message)
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.com")
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)
    monkeypatch.setattr(pause_record, "install_root", lambda: root)
    a = commit(root, "A", **{"a.py": "A=1\n", "mine.py": "M=1\n", "package-lock.json": "{}\n"})
    t = commit(root, "T", **{"a.py": "A=2\n"})
    git(root, "reset", "-q", "--hard", a)
    # A fetched ref that never touches the files these steps write: what the gate fell back to.
    git(root, "update-ref", "refs/remotes/origin/main", t)
    return root, a, t


def interrupting_git(tmp_path, match: str, tear) -> list[str]:
    """A git that, on the *match* subcommand, snapshots the durable record, runs real git, leaves
    *tear* half-written and interrupts the updater."""
    wrapper = tmp_path / "git-interrupted"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'case " $* " in *" {match} "*) cp "{pause_record.record_path()}" "{tmp_path / "at-start.json"}"\n'
        f'  "{shutil.which("git")}" "$@"\n'
        f"  printf 'M=' > \"{tear}\"; kill -INT $PPID; sleep 5; exit 130;;\n"
        f'esac\nexec "{shutil.which("git")}" "$@"\n',
        encoding="utf-8")
    wrapper.chmod(0o755)
    return [str(wrapper)]


def pause(root) -> dict:
    token = pause_record.stamp_tree({"resume_needed": True, "profiles": {"default": 999999}}, root)
    pause_record.write(token)
    return token


def gate(root) -> bool:
    return pause_record.tree_is_whole(pause_record.read(pause_record.record_path())["token"], root)[0]


def at_start(tmp_path, head) -> dict:
    """The baseline at *head* in the record as it was on disk the moment git started."""
    token = pause_record.read(tmp_path / "at-start.json")["token"]
    return next(b for b in token["baselines"] if b["pre_sha"] == head)


def test_an_interrupted_stash_push_stays_held(repo, tmp_path):
    root, a, _t = repo
    (root / "mine.py").write_text("M=2  # my edit\n", encoding="utf-8")
    token = pause(root)
    with pytest.raises(KeyboardInterrupt):
        update_cmd_stash._stash_local_changes_if_needed(
            interrupting_git(tmp_path, "stash push", root / "mine.py"), root,
            checkout_move=update_cmd._moves_for(token))
    started = at_start(tmp_path, a)
    assert "mine.py" in started["move_paths"] and started["dirty_at_pause"] == ["mine.py"]
    assert gate(root) is False, "a stash push interrupted mid-write was admitted"


def test_an_interrupted_lockfile_churn_cleanup_stays_held(repo, tmp_path):
    root, a, _t = repo
    (root / "package-lock.json").write_text('{"npm": "rewrote me"}\n', encoding="utf-8")
    token = pause(root)
    with pytest.raises(KeyboardInterrupt):
        update_cmd_git._discard_lockfile_churn(
            interrupting_git(tmp_path, "checkout --", root / "package-lock.json"), root,
            checkout_move=update_cmd._moves_for(token))
    assert at_start(tmp_path, a)["move_paths"] == ["package-lock.json"]
    assert gate(root) is False, "a lockfile cleanup interrupted mid-write was admitted"


@pytest.mark.parametrize("interrupted", [True, False])
def test_a_stash_restore_at_the_moved_head_is_gated_from_that_head(repo, tmp_path, interrupted):
    root, a, t = repo
    (root / "mine.py").write_text("M=2  # my edit\n", encoding="utf-8")
    token = pause(root)
    moves = update_cmd._moves_for(token)
    stash = update_cmd_stash._stash_local_changes_if_needed(["git"], root, checkout_move=moves)
    with update_cmd._checkout_move(token, t):
        git(root, "merge", "-q", "--ff-only", t)  # the pull lands whole
    git_cmd = interrupting_git(tmp_path, "stash apply", root / "mine.py") if interrupted else ["git"]
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            update_cmd_stash._restore_stashed_changes(git_cmd, root, stash, checkout_move=moves)
        started = at_start(tmp_path, t)
        assert started["move_targets"] == [stash] and started["dirty_at_pause"] == []
        assert gate(root) is False, "a stash restore interrupted mid-write at the moved HEAD was admitted"
    else:
        assert update_cmd_stash._restore_stashed_changes(git_cmd, root, stash, checkout_move=moves)
        assert (root / "mine.py").read_text() == "M=2  # my edit\n"
        assert gate(root) is True, "a stash restore that applied cleanly held the paused set"
