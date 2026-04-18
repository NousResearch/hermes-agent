"""Tests for _is_remote_session() headless/environment detection in auth.py.

Covers SSH, CI, headless Linux (no DISPLAY), non-interactive stdin,
and the HERMES_HEADLESS explicit opt-out.
"""

import sys
import pytest


class TestIsRemoteSession:
    """Detects environments where webbrowser.open() is unlikely to work."""

    def _clear_env(self, monkeypatch):
        """Remove all env vars that _is_remote_session checks."""
        for var in (
            "HERMES_HEADLESS",
            "SSH_CLIENT",
            "SSH_TTY",
            "SSH_CONNECTION",
            "CI",
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            "DISPLAY",
            "WAYLAND_DISPLAY",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_hermes_headless_opt_out(self, monkeypatch):
        from hermes_cli.auth import _is_remote_session

        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_HEADLESS", "1")
        assert _is_remote_session() is True

    def test_detects_ssh_client(self, monkeypatch):
        from hermes_cli.auth import _is_remote_session

        self._clear_env(monkeypatch)
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 54321 22")
        assert _is_remote_session() is True

    def test_detects_ssh_tty(self, monkeypatch):
        from hermes_cli.auth import _is_remote_session

        self._clear_env(monkeypatch)
        monkeypatch.setenv("SSH_TTY", "/dev/pts/0")
        assert _is_remote_session() is True

    def test_detects_ssh_connection(self, monkeypatch):
        from hermes_cli.auth import _is_remote_session

        self._clear_env(monkeypatch)
        monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 22 5.6.7.8 9876")
        assert _is_remote_session() is True

    def test_detects_ci(self, monkeypatch):
        from hermes_cli.auth import _is_remote_session

        self._clear_env(monkeypatch)
        monkeypatch.setenv("CI", "true")
        assert _is_remote_session() is True

    def test_detects_github_actions(self, monkeypatch):
        from hermes_cli.auth import _is_remote_session

        self._clear_env(monkeypatch)
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        assert _is_remote_session() is True

    def test_detects_gitlab_ci(self, monkeypatch):
        from hermes_cli.auth import _is_remote_session

        self._clear_env(monkeypatch)
        monkeypatch.setenv("GITLAB_CI", "true")
        assert _is_remote_session() is True

    @pytest.mark.skipif(sys.platform == "win32", reason="DISPLAY check is Linux/macOS only")
    def test_detects_headless_no_display(self, monkeypatch):
        from hermes_cli.auth import _is_remote_session

        self._clear_env(monkeypatch)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert _is_remote_session() is True

    @pytest.mark.skipif(sys.platform == "win32", reason="DISPLAY check is Linux/macOS only")
    def test_has_display_not_headless(self, monkeypatch):
        from hermes_cli.auth import _is_remote_session

        self._clear_env(monkeypatch)
        monkeypatch.setenv("DISPLAY", ":0")
        # Also need stdin to look interactive; we mock it
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        assert _is_remote_session() is False

    @pytest.mark.skipif(sys.platform == "win32", reason="DISPLAY check is Linux/macOS only")
    def test_has_wayland_not_headless(self, monkeypatch):
        from hermes_cli.auth import _is_remote_session

        self._clear_env(monkeypatch)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        assert _is_remote_session() is False

    def test_non_interactive_stdin(self, monkeypatch):
        from hermes_cli.auth import _is_remote_session

        self._clear_env(monkeypatch)
        # On Linux, also set DISPLAY so that's not the blocker
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        assert _is_remote_session() is True

    def test_desktop_linux_not_remote(self, monkeypatch):
        from hermes_cli.auth import _is_remote_session

        self._clear_env(monkeypatch)
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        assert _is_remote_session() is False

    def test_windows_default_not_remote(self, monkeypatch):
        from hermes_cli.auth import _is_remote_session

        self._clear_env(monkeypatch)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        assert _is_remote_session() is False
