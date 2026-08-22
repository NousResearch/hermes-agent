"""Tests for the Windows preflight checks in hermes doctor (#91942)."""

import os
import subprocess
import sys
import types

import pytest

import hermes_cli.doctor as doctor_mod


class TestSymlinkPrivilegeCheck:
    def test_ok_when_symlink_creation_succeeds(self, monkeypatch, capsys):
        monkeypatch.setattr(doctor_mod.os, "symlink", lambda *args, **kwargs: None)

        doctor_mod._check_windows_symlink_privilege()

        out = capsys.readouterr().out
        assert "Symlink creation" in out
        assert "elevation" not in out

    def test_warns_with_dev_mode_hint_on_winerror_1314(self, monkeypatch, capsys):
        def _raise_symlink(*args, **kwargs):
            exc = OSError("privilege not held")
            exc.winerror = 1314
            raise exc

        monkeypatch.setattr(doctor_mod.os, "symlink", _raise_symlink)

        doctor_mod._check_windows_symlink_privilege()

        out = capsys.readouterr().out
        assert "requires elevation" in out
        assert "Developer Mode" in out

    def test_warns_generically_on_other_oserror(self, monkeypatch, capsys):
        def _raise_symlink(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(doctor_mod.os, "symlink", _raise_symlink)

        doctor_mod._check_windows_symlink_privilege()

        out = capsys.readouterr().out
        assert "Symlink creation failed" in out
        assert "disk full" in out


class TestGitCredentialPromptCheck:
    def test_skips_when_git_missing(self, monkeypatch, capsys):
        monkeypatch.setattr(doctor_mod, "_safe_which", lambda cmd: None)

        doctor_mod._check_windows_git_credential_prompt()

        assert capsys.readouterr().out == ""

    def test_ok_when_terminal_prompt_disabled(self, monkeypatch, capsys):
        monkeypatch.setattr(doctor_mod, "_safe_which", lambda cmd: "/usr/bin/git")
        monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")

        doctor_mod._check_windows_git_credential_prompt()

        out = capsys.readouterr().out
        assert "GIT_TERMINAL_PROMPT=0" in out

    def test_warns_when_unset_and_no_helper_configured(self, monkeypatch, capsys):
        monkeypatch.setattr(doctor_mod, "_safe_which", lambda cmd: "/usr/bin/git")
        monkeypatch.delenv("GIT_TERMINAL_PROMPT", raising=False)
        monkeypatch.setattr(
            doctor_mod.subprocess,
            "run",
            lambda *a, **kw: types.SimpleNamespace(stdout="", stderr=""),
        )

        doctor_mod._check_windows_git_credential_prompt()

        out = capsys.readouterr().out
        assert "hang" in out
        assert "GIT_TERMINAL_PROMPT=0" in out

    def test_info_when_non_interactive_helper_already_configured(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(doctor_mod, "_safe_which", lambda cmd: "/usr/bin/git")
        monkeypatch.delenv("GIT_TERMINAL_PROMPT", raising=False)
        monkeypatch.setattr(
            doctor_mod.subprocess,
            "run",
            lambda *a, **kw: types.SimpleNamespace(stdout="cache\n", stderr=""),
        )

        doctor_mod._check_windows_git_credential_prompt()

        out = capsys.readouterr().out
        assert "cache" in out
        assert "hang" not in out


class TestBashToolchainCheck:
    def test_ok_when_bash_found(self, monkeypatch, capsys):
        fake_local = types.SimpleNamespace(
            _find_bash=lambda: r"C:\Program Files\Git\bin\bash.exe"
        )
        monkeypatch.setitem(sys.modules, "tools.environments.local", fake_local)

        doctor_mod._check_windows_bash_toolchain()

        out = capsys.readouterr().out
        assert "Git Bash found" in out
        assert "bash.exe" in out

    def test_warns_when_bash_not_found(self, monkeypatch, capsys):
        def _raise():
            raise RuntimeError(
                "Git Bash not found. Hermes Agent requires Git for Windows."
            )

        fake_local = types.SimpleNamespace(_find_bash=_raise)
        monkeypatch.setitem(sys.modules, "tools.environments.local", fake_local)

        doctor_mod._check_windows_bash_toolchain()

        out = capsys.readouterr().out
        assert "Git Bash not found" in out


class TestLongPathsCheck:
    def _install_fake_winreg(self, monkeypatch, value):
        fake = types.ModuleType("winreg")
        fake.HKEY_LOCAL_MACHINE = object()

        class _Key:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        fake.OpenKey = lambda hive, path: _Key()
        fake.QueryValueEx = lambda key, name: (value, 4)
        monkeypatch.setitem(sys.modules, "winreg", fake)

    def test_skips_when_winreg_unavailable(self, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "winreg", None)

        doctor_mod._check_windows_long_paths()

        assert capsys.readouterr().out == ""

    def test_ok_when_long_paths_enabled(self, monkeypatch, capsys):
        self._install_fake_winreg(monkeypatch, 1)

        doctor_mod._check_windows_long_paths()

        out = capsys.readouterr().out
        assert "long path support enabled" in out

    def test_warns_when_long_paths_disabled(self, monkeypatch, capsys):
        self._install_fake_winreg(monkeypatch, 0)

        doctor_mod._check_windows_long_paths()

        out = capsys.readouterr().out
        assert "long path support disabled" in out
        assert "LongPathsEnabled" in out

    def test_warns_when_registry_key_missing(self, monkeypatch, capsys):
        fake = types.ModuleType("winreg")
        fake.HKEY_LOCAL_MACHINE = object()

        def _raise_open_key(hive, path):
            raise OSError("key not found")

        fake.OpenKey = _raise_open_key
        fake.QueryValueEx = lambda key, name: (0, 4)
        monkeypatch.setitem(sys.modules, "winreg", fake)

        doctor_mod._check_windows_long_paths()

        out = capsys.readouterr().out
        assert "long path support disabled" in out


class TestWindowsEnvironmentSection:
    def test_skipped_on_non_windows(self, monkeypatch, capsys):
        monkeypatch.setattr(doctor_mod.sys, "platform", "linux")

        doctor_mod._check_windows_environment()

        assert capsys.readouterr().out == ""

    def test_runs_all_checks_on_windows(self, monkeypatch, capsys):
        monkeypatch.setattr(doctor_mod.sys, "platform", "win32")
        calls = []
        monkeypatch.setattr(
            doctor_mod,
            "_check_windows_symlink_privilege",
            lambda: calls.append("symlink"),
        )
        monkeypatch.setattr(
            doctor_mod,
            "_check_windows_git_credential_prompt",
            lambda: calls.append("git"),
        )
        monkeypatch.setattr(
            doctor_mod, "_check_windows_bash_toolchain", lambda: calls.append("bash")
        )
        monkeypatch.setattr(
            doctor_mod, "_check_windows_long_paths", lambda: calls.append("long_paths")
        )

        doctor_mod._check_windows_environment()

        out = capsys.readouterr().out
        assert "Windows Environment" in out
        assert calls == ["symlink", "git", "bash", "long_paths"]
