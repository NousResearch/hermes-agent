"""Tests for CLI browser CDP auto-launch helpers."""

import os
import queue
import socket
from unittest.mock import patch

from cli import HermesCLI


def _assert_chrome_debug_cmd(cmd, expected_chrome, expected_port):
    """Verify the auto-launch command has all required flags."""
    assert cmd[0] == expected_chrome
    assert f"--remote-debugging-port={expected_port}" in cmd
    assert "--no-first-run" in cmd
    assert "--no-default-browser-check" in cmd
    user_data_args = [a for a in cmd if a.startswith("--user-data-dir=")]
    assert len(user_data_args) == 1, "Expected exactly one --user-data-dir flag"
    assert "chrome-debug" in user_data_args[0]


class TestChromeDebugLaunch:
    def test_windows_launch_uses_browser_found_on_path(self):
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return object()

        with patch("cli.shutil.which", side_effect=lambda name: r"C:\Chrome\chrome.exe" if name == "chrome.exe" else None), \
             patch("cli.os.path.isfile", side_effect=lambda path: path == r"C:\Chrome\chrome.exe"), \
             patch("subprocess.Popen", side_effect=fake_popen):
            assert HermesCLI._try_launch_chrome_debug(9333, "Windows") is True

        _assert_chrome_debug_cmd(captured["cmd"], r"C:\Chrome\chrome.exe", 9333)
        assert captured["kwargs"]["start_new_session"] is True

    def test_windows_launch_falls_back_to_common_install_dirs(self, monkeypatch):
        captured = {}
        program_files = r"C:\Program Files"
        # Use os.path.join so path separators match cross-platform
        installed = os.path.join(program_files, "Google", "Chrome", "Application", "chrome.exe")

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return object()

        monkeypatch.setenv("ProgramFiles", program_files)
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)

        with patch("cli.shutil.which", return_value=None), \
             patch("cli.os.path.isfile", side_effect=lambda path: path == installed), \
             patch("subprocess.Popen", side_effect=fake_popen):
            assert HermesCLI._try_launch_chrome_debug(9222, "Windows") is True

        _assert_chrome_debug_cmd(captured["cmd"], installed, 9222)


class _OpenSocket:
    def settimeout(self, _timeout):
        return None

    def connect(self, _addr):
        return None

    def close(self):
        return None


class TestBrowserConnectValidation:
    def _make_shell(self):
        shell = HermesCLI.__new__(HermesCLI)
        shell._pending_input = queue.Queue()
        return shell

    def test_browser_connect_stores_resolved_websocket_endpoint(self, monkeypatch, capsys):
        import tools.browser_tool as browser_tool

        shell = self._make_shell()
        resolved = "ws://127.0.0.1:9222/devtools/browser/local-abc"

        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
        monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: _OpenSocket())
        monkeypatch.setattr(browser_tool, "_resolve_cdp_override", lambda raw: resolved)
        monkeypatch.setattr(browser_tool, "_is_concrete_cdp_websocket_url", lambda raw: True)

        shell._handle_browser_command("/browser connect")
        output = capsys.readouterr().out

        assert "Browser connected to live Chrome via CDP" in output
        assert f"Endpoint: {resolved}" in output
        assert os.environ["BROWSER_CDP_URL"] == resolved
        assert not shell._pending_input.empty()

    def test_browser_connect_rejects_unresolved_shorthand_endpoint(self, monkeypatch, capsys):
        import tools.browser_tool as browser_tool

        shell = self._make_shell()

        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
        monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: _OpenSocket())
        monkeypatch.setattr(browser_tool, "_resolve_cdp_override", lambda raw: raw)
        monkeypatch.setattr(browser_tool, "_is_concrete_cdp_websocket_url", lambda raw: False)

        shell._handle_browser_command("/browser connect")
        output = capsys.readouterr().out

        assert "Browser connected to live Chrome via CDP" not in output
        assert "Could not resolve a usable CDP WebSocket endpoint" in output
        assert "BROWSER_CDP_URL" not in os.environ
        assert shell._pending_input.empty()
