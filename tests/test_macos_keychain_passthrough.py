"""Tests for macOS Keychain passthrough from sandboxed worker subprocesses.

See: https://github.com/NousResearch/hermes-agent/issues/29015
"""

import os
import platform
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="macOS-only behaviour"
)


class TestKeychainSymlink:
    def _setup_profile(self, tmp_path, monkeypatch, real_keychain=True):
        real_home = tmp_path / "user_home"
        real_home.mkdir()
        if real_keychain:
            (real_home / "Library" / "Keychains").mkdir(parents=True)
            (real_home / "Library" / "Keychains" / "login.keychain-db").write_text("x")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        profile_home = hermes_home / "home"
        profile_home.mkdir()
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        return real_home, profile_home

    def test_keychain_symlink_created_on_make_run_env(self, tmp_path, monkeypatch):
        real_home, profile_home = self._setup_profile(tmp_path, monkeypatch)
        from tools.environments.local import _make_run_env

        run_env = _make_run_env({})
        assert run_env["HOME"] == str(profile_home)
        link = profile_home / "Library" / "Keychains"
        assert link.is_symlink()
        assert os.path.realpath(link) == os.path.realpath(
            real_home / "Library" / "Keychains"
        )
        # Subprocess can read login.keychain-db via the rewritten HOME.
        assert (link / "login.keychain-db").read_text() == "x"

    def test_no_symlink_when_real_home_has_no_keychain(self, tmp_path, monkeypatch):
        real_home, profile_home = self._setup_profile(
            tmp_path, monkeypatch, real_keychain=False
        )
        from tools.environments.local import _make_run_env

        _make_run_env({})
        assert not (profile_home / "Library" / "Keychains").exists()

    def test_idempotent(self, tmp_path, monkeypatch):
        real_home, profile_home = self._setup_profile(tmp_path, monkeypatch)
        from tools.environments.local import _make_run_env

        _make_run_env({})
        _make_run_env({})  # Must not raise (FileExistsError) on second call.
        link = profile_home / "Library" / "Keychains"
        assert link.is_symlink()

    def test_existing_real_dir_not_clobbered(self, tmp_path, monkeypatch):
        real_home, profile_home = self._setup_profile(tmp_path, monkeypatch)
        # User already populated a real directory at the target — leave it.
        target = profile_home / "Library" / "Keychains"
        target.mkdir(parents=True)
        (target / "marker").write_text("user-data")
        from tools.environments.local import _make_run_env

        _make_run_env({})
        assert target.is_dir() and not target.is_symlink()
        assert (target / "marker").read_text() == "user-data"

    def test_wrong_symlink_replaced(self, tmp_path, monkeypatch):
        real_home, profile_home = self._setup_profile(tmp_path, monkeypatch)
        target_parent = profile_home / "Library"
        target_parent.mkdir()
        bogus = tmp_path / "bogus"
        bogus.mkdir()
        (target_parent / "Keychains").symlink_to(bogus)
        from tools.environments.local import _make_run_env

        _make_run_env({})
        link = target_parent / "Keychains"
        assert link.is_symlink()
        assert os.path.realpath(link) == os.path.realpath(
            real_home / "Library" / "Keychains"
        )

    def test_no_op_when_subprocess_home_inactive(self, tmp_path, monkeypatch):
        real_home = tmp_path / "user_home"
        (real_home / "Library" / "Keychains").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.delenv("HERMES_HOME", raising=False)
        from tools.environments.local import _make_run_env

        run_env = _make_run_env({})
        # Real HOME preserved; no rewriting happened.
        assert run_env["HOME"] == str(real_home)
