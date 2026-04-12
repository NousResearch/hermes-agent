"""Tests for CLI browser CDP auto-launch helpers."""

import os
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


def test_cdp_socket_target_uses_configured_host():
    from cli import _get_cdp_socket_target

    assert _get_cdp_socket_target("http://192.0.2.10:9222") == ("192.0.2.10", 9222)
    assert _get_cdp_socket_target("ws://192.0.2.10:9222/devtools/browser/abc") == ("192.0.2.10", 9222)


def test_browser_status_checks_remote_cdp_endpoint(monkeypatch, capsys):
    cli = HermesCLI.__new__(HermesCLI)
    monkeypatch.setenv("BROWSER_CDP_URL", "http://192.0.2.10:9222")

    with patch("cli._is_cdp_endpoint_reachable", return_value=True) as reachable:
        cli._handle_browser_command("/browser status")

    out = capsys.readouterr().out
    assert "Endpoint: http://192.0.2.10:9222" in out
    assert "Status: ✓ reachable" in out
    reachable.assert_called_once_with("http://192.0.2.10:9222")
