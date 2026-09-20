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
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), check=True)
    expected = str(tmp_path / "cache" / "scratch")
    assert out.stdout.split() == [expected, expected]


def test_prune_removes_only_stale_top_level_entries(tmp_path):
    scratch = get_scratch_dir(tmp_path, prune=False)
    stale, fresh = scratch / "stale", scratch / "fresh.txt"
    stale.mkdir()
    (stale / "f").write_text("x", encoding="utf-8")
    fresh.write_text("y", encoding="utf-8")
    ancient = time.time() - 100 * 3600
    os.utime(stale, (ancient, ancient))
    assert prune_scratch_dir(scratch) == 1
    assert not stale.exists() and fresh.exists()


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
