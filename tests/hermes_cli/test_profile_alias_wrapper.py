"""Regression tests for profile alias wrapper scripts (#74074)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.profiles import _WRAPPER_MARKER


@pytest.fixture()
def wrapper_dir(tmp_path):
    """Redirect _get_wrapper_dir to a temp directory."""
    with patch("hermes_cli.profiles._get_wrapper_dir", return_value=tmp_path):
        yield tmp_path


@pytest.fixture()
def mock_which():
    """Mock shutil.which to return a predictable hermes.exe path."""
    fake_exe = "/usr/local/bin/hermes" if sys.platform != "win32" else r"C:\Tools\hermes.exe"
    with patch("shutil.which", return_value=fake_exe):
        yield fake_exe


class TestCreateWrapperScript:
    """create_wrapper_script generates correct wrappers (#74074)."""

    def test_windows_bat_is_subcommand_agnostic_passthrough(self, wrapper_dir, mock_which):
        """The .bat wrapper passes all args through without hardcoding a
        subcommand."""
        with patch("hermes_cli.profiles.sys") as mock_sys:
            mock_sys.platform = "win32"
            from hermes_cli.profiles import create_wrapper_script

            result = create_wrapper_script("myprofile")

        assert result is not None
        assert result.suffix == ".bat"
        content = result.read_text(encoding="utf-8")
        assert "-p myprofile %*" in content
        # Must NOT hardcode a subcommand (regression: #74074 review)
        assert "chat" not in content

    def test_windows_bat_contains_marker(self, wrapper_dir, mock_which):
        """The .bat embeds the stable marker for recognition checks."""
        with patch("hermes_cli.profiles.sys") as mock_sys:
            mock_sys.platform = "win32"
            from hermes_cli.profiles import create_wrapper_script

            result = create_wrapper_script("myprofile")

        content = result.read_text(encoding="utf-8")
        assert _WRAPPER_MARKER in content

    def test_windows_bat_bypasses_hermes_cmd_shim(self, wrapper_dir, mock_which):
        """The .bat calls the resolved hermes.exe directly, not bare 'hermes'
        which would resolve to hermes.cmd and inject -p default (#74074)."""
        with patch("hermes_cli.profiles.sys") as mock_sys:
            mock_sys.platform = "win32"
            from hermes_cli.profiles import create_wrapper_script

            result = create_wrapper_script("myprofile")

        content = result.read_text(encoding="utf-8")
        # Must use the resolved exe path (quoted), not bare 'hermes'
        assert f'"{ mock_which}" -p myprofile' in content

    def test_windows_creates_bash_script_alongside_bat(self, wrapper_dir, mock_which):
        """On Windows, a bash script is also created for git-bash (#74074)."""
        with patch("hermes_cli.profiles.sys") as mock_sys:
            mock_sys.platform = "win32"
            from hermes_cli.profiles import create_wrapper_script

            create_wrapper_script("myprofile")

        bash_path = wrapper_dir / "myprofile"
        assert bash_path.exists()
        content = bash_path.read_text(encoding="utf-8")
        assert _WRAPPER_MARKER in content
        assert "#!/bin/sh" in content
        assert '-p myprofile "$@"' in content
        assert "chat" not in content

    def test_posix_script_contains_marker_and_passthrough(self, wrapper_dir, mock_which):
        """The POSIX wrapper has the marker and passes args through."""
        with patch("hermes_cli.profiles.sys") as mock_sys:
            mock_sys.platform = "linux"
            from hermes_cli.profiles import create_wrapper_script

            result = create_wrapper_script("myprofile")

        assert result is not None
        content = result.read_text(encoding="utf-8")
        assert _WRAPPER_MARKER in content
        assert '-p myprofile "$@"' in content
        assert "chat" not in content

    def test_custom_alias_target_profile(self, wrapper_dir, mock_which):
        """A custom alias name targets the correct profile."""
        with patch("hermes_cli.profiles.sys") as mock_sys:
            mock_sys.platform = "linux"
            from hermes_cli.profiles import create_wrapper_script

            result = create_wrapper_script("jarvis", target="work")

        content = result.read_text(encoding="utf-8")
        assert "-p work" in content


class TestResolveHermesExe:
    """_resolve_hermes_exe prefers .exe on Windows (#74074)."""

    def test_windows_prefers_exe_over_cmd(self, wrapper_dir):
        """On Windows, hermes.exe is preferred over hermes.cmd."""
        def fake_which(name):
            if name == "hermes.exe":
                return r"C:\Tools\hermes.exe"
            if name == "hermes":
                return r"C:\Tools\hermes.cmd"  # shim
            return None

        with patch("hermes_cli.profiles.sys") as mock_sys, \
             patch("shutil.which", side_effect=fake_which):
            mock_sys.platform = "win32"
            from hermes_cli.profiles import _resolve_hermes_exe

            result = _resolve_hermes_exe()

        assert result == r"C:\Tools\hermes.exe"
        assert not result.endswith(".cmd")

    def test_posix_uses_which_hermes(self, wrapper_dir):
        """On POSIX, shutil.which('hermes') is used directly."""
        with patch("hermes_cli.profiles.sys") as mock_sys, \
             patch("shutil.which", return_value="/usr/local/bin/hermes"):
            mock_sys.platform = "linux"
            from hermes_cli.profiles import _resolve_hermes_exe

            result = _resolve_hermes_exe()

        assert result == "/usr/local/bin/hermes"


class TestRemoveWrapperScript:
    """remove_wrapper_script removes only our wrappers (#74074)."""

    def test_windows_removes_both_bat_and_bash(self, wrapper_dir):
        """On Windows, both .bat and bash script are removed."""
        bat = wrapper_dir / "myprofile.bat"
        bash = wrapper_dir / "myprofile"
        bat.write_text(f'{_WRAPPER_MARKER}\r\n@echo off\r\n"C:\\Tools\\hermes.exe" -p myprofile %*\r\n')
        bash.write_text(f'{_WRAPPER_MARKER}\n#!/bin/sh\nexec /c/Tools/hermes.exe -p myprofile "$@"\n')

        with patch("hermes_cli.profiles.sys") as mock_sys:
            mock_sys.platform = "win32"
            from hermes_cli.profiles import remove_wrapper_script

            result = remove_wrapper_script("myprofile")

        assert result is True
        assert not bat.exists()
        assert not bash.exists()

    def test_refuses_to_remove_unrelated_script(self, wrapper_dir):
        """A same-named script without our marker is NOT deleted."""
        bat = wrapper_dir / "myprofile.bat"
        bat.write_text('@echo off\r\necho "I am not a hermes wrapper"\r\n')

        with patch("hermes_cli.profiles.sys") as mock_sys:
            mock_sys.platform = "win32"
            from hermes_cli.profiles import remove_wrapper_script

            result = remove_wrapper_script("myprofile")

        assert result is False
        assert bat.exists()  # untouched

    def test_legacy_wrapper_without_marker_still_removed(self, wrapper_dir):
        """Pre-marker wrappers (containing 'hermes -p') are still recognized."""
        script = wrapper_dir / "myprofile"
        script.write_text('#!/bin/sh\nexec hermes -p myprofile "$@"\n')

        with patch("hermes_cli.profiles.sys") as mock_sys:
            mock_sys.platform = "linux"
            from hermes_cli.profiles import remove_wrapper_script

            result = remove_wrapper_script("myprofile")

        assert result is True
        assert not script.exists()
