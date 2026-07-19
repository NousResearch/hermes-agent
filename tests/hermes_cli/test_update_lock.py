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


def test_non_git_install_gets_a_stable_per_install_lock_path(tmp_path):
    install_a = tmp_path / "venv-a" / "site-packages"
    install_b = tmp_path / "venv-b" / "site-packages"
    install_a.mkdir(parents=True)
    install_b.mkdir(parents=True)

    path_a = get_update_lock_path(install_a)
    path_b = get_update_lock_path(install_b)

    assert path_a == get_update_lock_path(install_a)
    assert path_a != path_b
    assert path_a.name.startswith("update-")
    assert path_a.suffix == ".lock"


def test_non_git_install_lock_is_independent_of_user_state_and_profile_paths(
    tmp_path, monkeypatch
):
    install_root = tmp_path / "venv" / "site-packages"
    other_install_root = tmp_path / "other-venv" / "site-packages"
    install_root.mkdir(parents=True)
    other_install_root.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "default"))
    monkeypatch.setenv("HOME", str(tmp_path / "home-a"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-a"))
    first = get_update_lock_path(install_root)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "worker"))
    monkeypatch.setenv("HOME", str(tmp_path / "home-b"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-b"))
    second = get_update_lock_path(install_root)

    assert first == second
    assert first != get_update_lock_path(other_install_root)
    assert str(first) not in {str(tmp_path / "xdg-a"), str(tmp_path / "xdg-b")}


def test_non_git_install_lock_canonicalizes_install_root_symlinks(tmp_path):
    install_root = tmp_path / "venv" / "site-packages"
    install_root.mkdir(parents=True)
    alias = tmp_path / "current-site-packages"
    alias.symlink_to(install_root, target_is_directory=True)

    assert get_update_lock_path(alias) == get_update_lock_path(install_root)


def test_non_git_install_uses_private_deterministic_fallback_when_root_is_read_only(
    tmp_path, monkeypatch
):
    install_root = tmp_path / "system-site-packages"
    install_root.mkdir()
    fallback_root = tmp_path / "tmp"
    fallback_root.mkdir()
    monkeypatch.setattr("hermes_cli.update_lock.os.access", lambda _path, _mode: False)
    monkeypatch.setattr("hermes_cli.update_lock.tempfile.gettempdir", lambda: str(fallback_root))

    path = get_update_lock_path(install_root)
    lock = UpdateLock(path).acquire()
    try:
        assert path.parent == fallback_root / "hermes-update-locks"
        assert path.parent.stat().st_mode & 0o777 == 0o700
    finally:
        lock.release()
