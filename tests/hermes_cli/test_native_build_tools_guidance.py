"""Tests for _maybe_print_native_build_tools_guidance — native build tool
detection and actionable guidance when npm install fails."""

from unittest.mock import patch
import pytest

from hermes_cli.main import _maybe_print_native_build_tools_guidance


class TestNativeBuildToolsGuidance:
    """Verify that guidance is printed only when build tools are missing."""

    def test_no_guidance_when_all_tools_present(self, capsys):
        """No output when make + gcc are on PATH."""
        with patch("hermes_cli.main.shutil.which", side_effect=lambda cmd: cmd if cmd in ("make", "gcc") else None):
            _maybe_print_native_build_tools_guidance()
        assert capsys.readouterr().out == ""

    def test_no_guidance_when_make_and_gpp_present(self, capsys):
        """No output when make + g++ are on PATH (gcc missing but g++ suffices)."""
        with patch("hermes_cli.main.shutil.which", side_effect=lambda cmd: cmd if cmd in ("make", "g++") else None):
            _maybe_print_native_build_tools_guidance()
        assert capsys.readouterr().out == ""

    def test_guidance_when_make_missing(self, capsys):
        """Prints guidance when make is missing."""
        with patch("hermes_cli.main.shutil.which", side_effect=lambda cmd: cmd if cmd == "gcc" else None):
            _maybe_print_native_build_tools_guidance()
        out = capsys.readouterr().out
        assert "native build tools are missing" in out
        assert "Debian/Ubuntu" in out
        assert "Fedora/RHEL" in out
        assert "openSUSE" in out
        assert "Arch Linux" in out
        assert "Alpine" in out
        assert "hermes update" in out

    def test_guidance_when_compiler_missing(self, capsys):
        """Prints guidance when both gcc and g++ are missing."""
        with patch("hermes_cli.main.shutil.which", side_effect=lambda cmd: cmd if cmd == "make" else None):
            _maybe_print_native_build_tools_guidance()
        out = capsys.readouterr().out
        assert "native build tools are missing" in out

    def test_guidance_when_all_tools_missing(self, capsys):
        """Prints guidance when nothing is available."""
        with patch("hermes_cli.main.shutil.which", return_value=None):
            _maybe_print_native_build_tools_guidance()
        out = capsys.readouterr().out
        assert "native build tools are missing" in out
        assert "node-pty" in out
