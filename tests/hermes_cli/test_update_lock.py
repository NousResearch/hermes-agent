from __future__ import annotations

from hermes_cli.update_lock import (
    UpdateLock,
    UpdateLockBusyError,
    get_update_lock_path,
)


def test_lock_path_uses_worktree_common_git_dir_not_hermes_home(tmp_path, monkeypatch):
    common_git = tmp_path / "source" / ".git"
    worktree_git = common_git / "worktrees" / "feature"
    worktree = tmp_path / "worktree"
    common_git.mkdir(parents=True)
    worktree_git.mkdir(parents=True)
    worktree.mkdir()
    (worktree / ".git").write_text(
        f"gitdir: {worktree_git}\n",
        encoding="utf-8",
    )
    (worktree_git / "commondir").write_text("../..\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-home"))

    assert get_update_lock_path(worktree) == common_git / "hermes-update.lock"


def test_update_lock_is_nonblocking_and_releases_deterministically(tmp_path):
    lock_path = tmp_path / ".git" / "hermes-update.lock"
    first = UpdateLock(lock_path)
    second = UpdateLock(lock_path)

    first.acquire()
    try:
        try:
            second.acquire()
        except UpdateLockBusyError:
            pass
        else:
            raise AssertionError("a second update lock acquisition must fail closed")
    finally:
        first.release()

    second.acquire()
    second.release()
