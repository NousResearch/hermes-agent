"""Tests for CLI browser CDP auto-launch helpers."""

import os
import subprocess
from unittest.mock import patch

from cli import HermesCLI
from hermes_cli.browser_connect import manual_chrome_debug_command


def _assert_chrome_debug_cmd(cmd, expected_chrome, expected_port):
    """Verify the auto-launch command has all required flags."""
    assert cmd[0] == expected_chrome
    assert f"--remote-debugging-port={expected_port}" in cmd
    assert "--no-first-run" in cmd
    assert "--no-default-browser-check" in cmd
    assert "--remote-allow-origins=*" in cmd
    user_data_args = [a for a in cmd if a.startswith("--user-data-dir=")]
    assert len(user_data_args) == 1, "Expected exactly one --user-data-dir flag"
    assert "chrome-debug" in user_data_args[0]


class TestChromeDebugArgs:
    def test_chrome_debug_args_uses_provided_user_data_dir(self):
        from hermes_cli.browser_connect import _chrome_debug_args
        args = _chrome_debug_args(9222, "/custom/path", "Linux", "/usr/bin/chrome")
        assert "--remote-debugging-port=9222" in args
        assert "--user-data-dir=/custom/path" in args
        assert "--remote-allow-origins=*" in args
        assert "--no-first-run" in args
        assert "--no-default-browser-check" in args


class TestChromeDebugLaunch:
    def test_windows_launch_uses_browser_found_on_path(self):
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return object()

        with patch("hermes_cli.browser_connect.shutil.which", side_effect=lambda name: r"C:\Chrome\chrome.exe" if name == "chrome.exe" else None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == r"C:\Chrome\chrome.exe"), \
             patch("subprocess.Popen", side_effect=fake_popen):
            assert HermesCLI._try_launch_chrome_debug(9333, "Windows") is True

        _assert_chrome_debug_cmd(captured["cmd"], r"C:\Chrome\chrome.exe", 9333)
        # Windows uses creationflags (POSIX-only start_new_session would raise).
        assert "start_new_session" not in captured["kwargs"]
        flags = captured["kwargs"].get("creationflags", 0)
        expected = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        assert flags == expected

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

        with patch("hermes_cli.browser_connect.shutil.which", return_value=None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == installed), \
             patch("subprocess.Popen", side_effect=fake_popen):
            assert HermesCLI._try_launch_chrome_debug(9222, "Windows") is True

        _assert_chrome_debug_cmd(captured["cmd"], installed, 9222)

    def test_manual_command_uses_detected_linux_browser(self):
        with patch("hermes_cli.browser_connect.shutil.which", side_effect=lambda name: "/usr/bin/chromium" if name == "chromium" else None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == "/usr/bin/chromium"):
            command = manual_chrome_debug_command(9222, "Linux")

        assert command is not None
        assert command.startswith("/usr/bin/chromium --remote-debugging-port=9222")

    def test_manual_command_uses_wsl_windows_chrome_when_available(self):
        chrome = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"

        with patch("hermes_cli.browser_connect.shutil.which", return_value=None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == chrome):
            command = manual_chrome_debug_command(9222, "Linux")

        assert command is not None
        # Linux/WSL uses POSIX shell quoting (single quotes around paths with spaces).
        assert command.startswith(f"'{chrome}' --remote-debugging-port=9222")

    def test_manual_command_uses_windows_quoting_on_windows(self):
        chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

        with patch("hermes_cli.browser_connect.shutil.which", side_effect=lambda name: chrome if name == "chrome.exe" else None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == chrome):
            command = manual_chrome_debug_command(9222, "Windows")

        assert command is not None
        # Windows uses cmd.exe-compatible quoting via subprocess.list2cmdline.
        assert command.startswith(f'"{chrome}" --remote-debugging-port=9222')
        assert "'" not in command

    def test_manual_command_returns_none_when_linux_browser_missing(self):
        with patch("hermes_cli.browser_connect.shutil.which", return_value=None), \
             patch("hermes_cli.browser_connect.os.path.isfile", return_value=False):
            assert manual_chrome_debug_command(9222, "Linux") is None


class TestWSLHelpers:
    def test_is_wsl_true_on_wsl2_proc_version(self):
        content = "Linux version 6.6.114.1-microsoft-standard-WSL2 (Ubuntu 22.04)"
        with patch("builtins.open", side_effect=open):
            with patch("builtins.open.read", return_value=content):
                from hermes_cli.browser_connect import _is_wsl
                with patch("hermes_cli.browser_connect.open", side_effect=open):
                    result = _is_wsl()
        assert result is True

    def test_is_wsl_true_on_microsoft_proc_version(self):
        content = "Linux version 5.15.0microsoft-WSL2"
        with patch("builtins.open", side_effect=open):
            with patch("hermes_cli.browser_connect.open", side_effect=open):
                with patch("hermes_cli.browser_connect.open.read", return_value=content):
                    from hermes_cli.browser_connect import _is_wsl
                    result = _is_wsl()
        assert result is True

    def test_is_wsl_false_on_native_linux(self):
        content = "Linux version 6.8.0-45-generic"
        with patch("hermes_cli.browser_connect.open", side_effect=FileNotFoundError("no /proc/version")):
            from hermes_cli.browser_connect import _is_wsl
            result = _is_wsl()
        assert result is False

    def test_is_windows_chrome_path_true_for_mnt_paths(self):
        from hermes_cli.browser_connect import _is_windows_chrome_path
        assert _is_windows_chrome_path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe") is True
        assert _is_windows_chrome_path("/mnt/d/Program Files/Edge/msedge.exe") is True

    def test_is_windows_chrome_path_false_for_unix_paths(self):
        from hermes_cli.browser_connect import _is_windows_chrome_path
        assert _is_windows_chrome_path("/usr/bin/chromium") is False
        assert _is_windows_chrome_path("/home/user/.local/bin/chrome") is False

    def test_translate_path_to_windows_success(self):
        from hermes_cli.browser_connect import _translate_path_to_windows
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = r"\\wsl.localhost\Ubuntu-22.04\home\nxy\.hermes\chrome-debug" "\n"
            result = _translate_path_to_windows("/home/nxy/.hermes/chrome-debug")
        assert result == r"\\wsl.localhost\Ubuntu-22.04\home\nxy\.hermes\chrome-debug"
        mock_run.assert_called_once_with(
            ["wslpath", "-w", "/home/nxy/.hermes/chrome-debug"],
            capture_output=True, text=True, timeout=5,
        )

    def test_translate_path_to_windows_failure(self):
        from hermes_cli.browser_connect import _translate_path_to_windows
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            result = _translate_path_to_windows("/home/nxy/.hermes/chrome-debug")
        assert result is None

    def test_translate_path_to_windows_exception(self):
        from hermes_cli.browser_connect import _translate_path_to_windows
        with patch("subprocess.run", side_effect=FileNotFoundError("wslpath not found")):
            result = _translate_path_to_windows("/home/nxy/.hermes/chrome-debug")
        assert result is None


class TestWSLIntegration:
    def test_try_launch_chrome_debug_wsl_windows_chrome_translates_path(self):
        captured = {}
        chrome = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return object()

        with patch("hermes_cli.browser_connect.shutil.which", return_value=None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == chrome), \
             patch("hermes_cli.browser_connect.open", side_effect=FileNotFoundError()), \
             patch("subprocess.Popen", side_effect=fake_popen):
            from hermes_cli.browser_connect import try_launch_chrome_debug
            with patch("hermes_cli.browser_connect._is_wsl", return_value=True), \
                 patch("hermes_cli.browser_connect._translate_path_to_windows", return_value=r"\\wsl.localhost\Ubuntu-22.04\home\nxy\.hermes\chrome-debug"):
                result = try_launch_chrome_debug(9222, "Linux")

        assert result is True
        user_data_args = [a for a in captured["cmd"] if a.startswith("--user-data-dir=")]
        assert len(user_data_args) == 1
        assert user_data_args[0] == r"--user-data-dir=\\wsl.localhost\Ubuntu-22.04\home\nxy\.hermes\chrome-debug"

    def test_try_launch_chrome_debug_wsl_native_linux_chrome_no_translation(self):
        captured = {}
        chrome = "/usr/bin/chromium"

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return object()

        with patch("hermes_cli.browser_connect.shutil.which", side_effect=lambda name: chrome if name == "chromium" else None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == chrome), \
             patch("hermes_cli.browser_connect._is_wsl", return_value=True), \
             patch("hermes_cli.browser_connect.chrome_debug_data_dir", return_value="/home/nxy/.hermes/chrome-debug"), \
             patch("subprocess.Popen", side_effect=fake_popen):
            from hermes_cli.browser_connect import try_launch_chrome_debug
            result = try_launch_chrome_debug(9222, "Linux")

        assert result is True
        user_data_args = [a for a in captured["cmd"] if a.startswith("--user-data-dir=")]
        assert len(user_data_args) == 1
        assert "\\" not in user_data_args[0]

    def test_try_launch_chrome_debug_native_linux_no_wsl_check(self):
        captured = {}
        chrome = "/usr/bin/chromium"

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return object()

        with patch("hermes_cli.browser_connect.shutil.which", side_effect=lambda name: chrome if name == "chromium" else None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == chrome), \
             patch("hermes_cli.browser_connect._is_wsl", return_value=False), \
             patch("subprocess.Popen", side_effect=fake_popen):
            from hermes_cli.browser_connect import try_launch_chrome_debug
            result = try_launch_chrome_debug(9222, "Linux")

        assert result is True
        user_data_args = [a for a in captured["cmd"] if a.startswith("--user-data-dir=")]
        assert len(user_data_args) == 1

    def test_manual_chrome_debug_command_wsl_windows_chrome_translated_path(self):
        chrome = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"

        with patch("hermes_cli.browser_connect.shutil.which", return_value=None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == chrome), \
             patch("hermes_cli.browser_connect._is_wsl", return_value=True), \
             patch("hermes_cli.browser_connect._translate_path_to_windows", return_value=r"\\wsl.localhost\Ubuntu-22.04\home\nxy\.hermes\chrome-debug"):
            command = manual_chrome_debug_command(9222, "Linux")

        assert command is not None
        assert r"\\wsl.localhost" in command
        assert "/home/nxy" not in command

    def test_manual_chrome_debug_command_wsl_windows_chrome_fallback_on_translate_fail(self):
        chrome = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"

        with patch("hermes_cli.browser_connect.shutil.which", return_value=None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == chrome), \
             patch("hermes_cli.browser_connect._is_wsl", return_value=True), \
             patch("hermes_cli.browser_connect._translate_path_to_windows", return_value=None), \
             patch("hermes_cli.browser_connect.chrome_debug_data_dir", return_value="/home/nxy/.hermes/chrome-debug"):
            command = manual_chrome_debug_command(9222, "Linux")

        assert command is not None
        assert "/home" in command
