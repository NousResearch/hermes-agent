"""Scratch dir contract: TMPDIR/TMP/TEMP follow HERMES_HOME/cache/scratch unless the user set them."""

import os
import stat
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from hermes_constants import apply_scratch_tmp_env, get_scratch_dir, prune_scratch_dir


def test_scratch_env_follows_home_and_respects_user_tmpdir(tmp_path):
    """Unset temp vars → scratch of env HERMES_HOME; a Hermes-exported value re-derives for a routed
    home; a user/OS-set value (macOS ``/var/folders``, ``%TEMP%``) is never touched."""
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    env = {"HERMES_HOME": str(home_a)}
    assert apply_scratch_tmp_env(env) is True
    scratch_a = str(home_a / "cache" / "scratch")
    assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == env["HERMES_SCRATCH_DIR"] == scratch_a
    assert os.path.isdir(scratch_a)

    env["HERMES_HOME"] = str(home_b)  # child served under another profile's home
    assert apply_scratch_tmp_env(env) is True
    assert env["TMPDIR"] == str(home_b / "cache" / "scratch")

    user_env = {"HERMES_HOME": str(home_a), "TMPDIR": "/var/folders/zz"}
    assert apply_scratch_tmp_env(user_env) is False
    assert user_env["TMPDIR"] == "/var/folders/zz" and "HERMES_SCRATCH_DIR" not in user_env
    assert "TMP" not in user_env  # a partially user-set triple is left exactly as found


def test_bootstrap_import_exports_scratch_to_process_and_children(tmp_path):
    """``import hermes_bootstrap`` alone makes ``tempfile`` (this process AND a child) land in scratch."""
    env = {k: v for k, v in os.environ.items() if k not in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR")}
    env["HERMES_HOME"] = str(tmp_path)
    code = ("import tempfile, os, subprocess, sys; import hermes_bootstrap; "
            "print(tempfile.gettempdir()); "
            "print(subprocess.run([sys.executable, '-c', 'import tempfile;print(tempfile.gettempdir())'],"
            " capture_output=True, text=True).stdout.strip())")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, encoding="utf-8",
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), check=True)
    expected = str(tmp_path / "cache" / "scratch")
    assert out.stdout.split() == [expected, expected]


def test_prune_removes_idle_entries_and_keeps_trees_written_deep_inside(tmp_path):
    """Idle retention: an entry goes when nothing in its subtree was written within the window;
    a tree whose only recent write is three levels down is still in use and stays, even though
    its top-level mtime is ancient (a directory's mtime ignores writes below its children)."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    idle, live, fresh = scratch / "idle", scratch / "live", scratch / "fresh.txt"
    deep = live / "lane" / "wt"
    deep.mkdir(parents=True)
    idle.mkdir()
    (idle / "f").write_text("x", encoding="utf-8")
    (deep / "log").write_text("x", encoding="utf-8")
    fresh.write_text("y", encoding="utf-8")
    ancient = time.time() - 30 * 3600
    for path in (idle, idle / "f", live, live / "lane", deep):
        os.utime(path, (ancient, ancient))
    assert prune_scratch_dir(scratch) == 1
    assert not idle.exists() and live.exists() and fresh.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory modes")
class TestScratchDirPermissionPolicy:
    """get_scratch_dir must honor the home permission policy instead of a blanket 0700:
    an explicit HERMES_HOME_MODE and a managed/shared home win (#117347)."""

    def _isolate_env(self, monkeypatch, tmp_path):
        # HERMES_HOME must point somewhere without a .managed marker, or a marker file in the
        # real home would flip every case into "managed" (the marker is read via get_hermes_home).
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        for var in ("HERMES_HOME_MODE", "HERMES_MANAGED", "HERMES_CONTAINER",
                    "HERMES_SKIP_CHMOD", "HERMES_UID", "HERMES_GID"):
            monkeypatch.delenv(var, raising=False)

    def test_default_is_owner_only(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == 0o700

    def test_explicit_home_mode_retained_with_setgid(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_HOME_MODE", "2770")
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == 0o2770

    def test_managed_env_leaves_preexisting_mode_untouched(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_MANAGED", "nixos")
        pre = tmp_path / "cache" / "scratch"
        pre.mkdir(parents=True)
        os.chmod(pre, 0o2770)
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == 0o2770

    def test_managed_marker_file_leaves_preexisting_mode_untouched(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        self._isolate_env(monkeypatch, tmp_path)
        (home / ".managed").write_text("nixos", encoding="utf-8")
        pre = tmp_path / "cache" / "scratch"
        pre.mkdir(parents=True)
        os.chmod(pre, 0o2770)
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == 0o2770

    def test_empty_managed_marker_counts_as_managed(self, tmp_path, monkeypatch):
        # Legacy NixOS module wrote an empty marker; config.get_managed_system treats it as
        # managed, so the parity twin must too (not re-enable chmod on managed installs).
        home = tmp_path / "home"
        home.mkdir()
        self._isolate_env(monkeypatch, tmp_path)
        (home / ".managed").write_text("", encoding="utf-8")
        pre = tmp_path / "cache" / "scratch"
        pre.mkdir(parents=True)
        os.chmod(pre, 0o2770)
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == 0o2770

    def test_unreadable_managed_marker_counts_as_managed(self, tmp_path, monkeypatch):
        # A marker that exists but cannot be read (OSError -> "") still counts as managed.
        home = tmp_path / "home"
        home.mkdir()
        self._isolate_env(monkeypatch, tmp_path)
        (home / ".managed").mkdir()
        pre = tmp_path / "cache" / "scratch"
        pre.mkdir(parents=True)
        os.chmod(pre, 0o2770)
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == 0o2770

    def test_canonical_container_signal_without_env_override_keeps_operator_mode(self, tmp_path, monkeypatch):
        # Podman/containerd/K8s runtimes often don't export HERMES_CONTAINER; the canonical
        # _detect_container breadth (not the narrower legacy signal set) must skip the chmod.
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setattr("hermes_constants._detect_container", lambda: True)
        pre = tmp_path / "cache" / "scratch"
        pre.mkdir(parents=True)
        os.chmod(pre, 0o750)
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == 0o750

    def test_repeated_calls_do_not_strip_setgid(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_HOME_MODE", "2770")
        first = get_scratch_dir(tmp_path, prune=False)
        second = get_scratch_dir(tmp_path, prune=False)
        assert first == second
        assert stat.S_IMODE(os.stat(second).st_mode) == 0o2770

    def test_container_keeps_operator_mode_but_honors_explicit(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_CONTAINER", "1")
        pre = tmp_path / "cache" / "scratch"
        pre.mkdir(parents=True)
        os.chmod(pre, 0o750)
        assert stat.S_IMODE(os.stat(get_scratch_dir(tmp_path, prune=False)).st_mode) == 0o750
        monkeypatch.setenv("HERMES_HOME_MODE", "2770")
        assert stat.S_IMODE(os.stat(get_scratch_dir(tmp_path, prune=False)).st_mode) == 0o2770

    def test_hermes_uid_gid_applied_to_scratch(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_UID", "1000")
        monkeypatch.setenv("HERMES_GID", "911")
        with patch.object(os, "chown") as mock_chown:
            get_scratch_dir(tmp_path, prune=False)
        mock_chown.assert_called_once_with(tmp_path / "cache" / "scratch", 1000, 911)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_config_and_constants_share_one_policy_implementation(tmp_path, monkeypatch):
    """hermes_constants is the single home of managed / container / HERMES_UID policy: config
    re-exports it (no keep-in-sync twins), so _secure_file skips on the same canonical container
    signal that apply_secure_dir_policy / get_scratch_dir already honor."""
    import hermes_constants
    from hermes_cli import config

    assert config.get_managed_system is hermes_constants.get_managed_system
    assert config._chown_to_hermes_uid is hermes_constants._chown_to_hermes_uid
    assert not hasattr(config, "_is_container")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    for var in ("HERMES_MANAGED", "HERMES_CONTAINER", "HERMES_SKIP_CHMOD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("hermes_constants._detect_container", lambda: True)
    f = tmp_path / "config.yaml"
    f.write_text("", encoding="utf-8")
    os.chmod(f, 0o640)
    config._secure_file(f)
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o640


def test_prune_reaps_process_living_in_idle_entry_and_spares_live_tree(tmp_path):
    """A process whose cwd is inside an idle entry is gone by the time the entry is
    (a lane's headless browsers survived for days with a deleted cwd); one living in a
    tree that is still being written is not touched."""
    import subprocess

    scratch = get_scratch_dir(tmp_path, prune=False)
    idle, live = scratch / "idle-lane" / "wt", scratch / "live-lane" / "wt"
    idle.mkdir(parents=True)
    live.mkdir(parents=True)
    ancient = time.time() - 30 * 3600
    for path in (idle.parent, idle, live.parent, live):
        os.utime(path, (ancient, ancient))
    (live / "log").write_text("still writing", encoding="utf-8")
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    doomed = subprocess.Popen(sleeper, cwd=str(idle), stdin=subprocess.DEVNULL)
    spared = subprocess.Popen(sleeper, cwd=str(live), stdin=subprocess.DEVNULL)
    try:
        assert prune_scratch_dir(scratch) == 1
        assert doomed.wait(timeout=10) is not None
        assert spared.poll() is None
        assert not idle.parent.exists() and live.exists()
    finally:
        for proc in (doomed, spared):
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


def test_prune_releases_git_worktree_registration_of_idle_entry(tmp_path):
    """Deleting a scratch entry that held a linked worktree leaves the repo with no
    dangling registration (10 sat in one repo's ``git worktree list`` after cleanup)."""
    import subprocess

    def git(*args, cwd):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                       stdin=subprocess.DEVNULL, env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
                                                     "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
                                                     "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"})

    repo = tmp_path / "repo"
    repo.mkdir()
    git("init", "-q", cwd=repo)
    (repo / "f").write_text("x", encoding="utf-8")
    git("add", "f", cwd=repo)
    git("commit", "-q", "-m", "init", cwd=repo)
    scratch = get_scratch_dir(tmp_path, prune=False)
    tree = scratch / "lane" / "abwt"
    tree.parent.mkdir()
    git("worktree", "add", "-q", "--detach", str(tree), cwd=repo)
    ancient = time.time() - 30 * 3600
    for dirpath, dirnames, filenames in os.walk(tree.parent):
        for name in dirnames + filenames:
            os.utime(os.path.join(dirpath, name), (ancient, ancient), follow_symlinks=False)
    os.utime(tree.parent, (ancient, ancient))
    assert prune_scratch_dir(scratch) == 1
    listing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=repo, capture_output=True,
                             text=True, stdin=subprocess.DEVNULL, check=True).stdout
    assert str(tree) not in listing and not tree.exists()
