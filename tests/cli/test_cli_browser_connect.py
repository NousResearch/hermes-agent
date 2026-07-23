"""Tests for CLI browser CDP auto-launch helpers."""

from contextlib import redirect_stdout
from io import StringIO
import os
from queue import Queue
import subprocess
from unittest.mock import patch

from cli import HermesCLI
from hermes_cli.browser_connect import (
    _wait_for_browser_debug_ready_or_exit,
    get_chrome_debug_candidates,
    is_browser_debug_ready,
    launch_chrome_debug,
    manual_chrome_debug_command,
)


def _assert_chrome_debug_cmd(cmd, expected_chrome, expected_port):
    """Verify the auto-launch command has all required flags."""
    assert cmd[0] == expected_chrome
    assert f"--remote-debugging-port={expected_port}" in cmd
    assert "--no-first-run" in cmd
    assert "--no-default-browser-check" in cmd
    user_data_args = [a for a in cmd if a.startswith("--user-data-dir=")]
    assert len(user_data_args) == 1, "Expected exactly one --user-data-dir flag"
    assert "chrome-debug" in user_data_args[0]


class _FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestChromeDebugLaunch:
    def test_browser_debug_ready_requires_http_cdp_endpoint(self):
        requested = []

        def fake_urlopen(url, timeout):
            requested.append(url)
            if url.endswith("/json/version"):
                return _FakeResponse()
            raise OSError("unexpected probe")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            assert is_browser_debug_ready("http://127.0.0.1:9222", timeout=0.1) is True

        assert requested == ["http://127.0.0.1:9222/json/version"]

    def test_browser_debug_ready_rejects_non_cdp_listener(self):
        with patch("urllib.request.urlopen", side_effect=OSError("not cdp")):
            assert is_browser_debug_ready("http://127.0.0.1:9222", timeout=0.1) is False

    def test_windows_launch_uses_browser_found_on_path(self):
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return object()

        with patch("hermes_cli.browser_connect.shutil.which", side_effect=lambda name: r"C:\Chrome\chrome.exe" if name == "chrome.exe" else None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == r"C:\Chrome\chrome.exe"), \
             patch("hermes_cli.browser_connect._wait_for_browser_debug_ready_or_exit", return_value="ready"), \
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
             patch("hermes_cli.browser_connect._wait_for_browser_debug_ready_or_exit", return_value="ready"), \
             patch("subprocess.Popen", side_effect=fake_popen):
            assert HermesCLI._try_launch_chrome_debug(9222, "Windows") is True

        _assert_chrome_debug_cmd(captured["cmd"], installed, 9222)

    def test_manual_command_uses_detected_linux_browser(self):
        with patch("hermes_cli.browser_connect.shutil.which", side_effect=lambda name: "/usr/bin/chromium" if name == "chromium" else None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == "/usr/bin/chromium"):
            command = manual_chrome_debug_command(9222, "Linux")

        assert command is not None
        assert command.startswith("/usr/bin/chromium --remote-debugging-port=9222")

    def test_linux_candidates_prefer_chrome_before_brave_when_both_exist(self):
        chrome = "/usr/bin/google-chrome"
        brave = "/usr/bin/brave-browser"

        def fake_which(name):
            return {"google-chrome": chrome, "brave-browser": brave}.get(name)

        with patch("hermes_cli.browser_connect.shutil.which", side_effect=fake_which), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path in {chrome, brave}):
            candidates = get_chrome_debug_candidates("Linux")
            command = manual_chrome_debug_command(9222, "Linux")

        assert candidates[:2] == [chrome, brave]
        assert command is not None
        assert command.startswith(f"{chrome} --remote-debugging-port=9222")

    def test_linux_candidates_prefer_chrome_install_path_before_brave_on_path(self):
        chrome = "/opt/google/chrome/chrome"
        brave = "/usr/bin/brave-browser"

        with patch("hermes_cli.browser_connect.shutil.which", side_effect=lambda name: brave if name == "brave-browser" else None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path in {chrome, brave}):
            candidates = get_chrome_debug_candidates("Linux")

        assert candidates[:2] == [chrome, brave]

    def test_windows_candidates_prefer_chrome_install_path_before_brave_on_path(self, monkeypatch):
        program_files = r"C:\Program Files"
        chrome = os.path.join(program_files, "Google", "Chrome", "Application", "chrome.exe")
        brave = r"C:\Brave\brave.exe"

        monkeypatch.setenv("ProgramFiles", program_files)
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)

        with patch("hermes_cli.browser_connect.shutil.which", side_effect=lambda name: brave if name == "brave.exe" else None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path in {chrome, brave}):
            candidates = get_chrome_debug_candidates("Windows")

        assert candidates[:2] == [chrome, brave]

    def test_linux_candidates_include_arch_brave_install_path(self):
        brave = "/opt/brave-bin/brave"

        with patch("hermes_cli.browser_connect.shutil.which", return_value=None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == brave):
            candidates = get_chrome_debug_candidates("Linux")
            command = manual_chrome_debug_command(9222, "Linux")

        assert candidates == [brave]
        assert command is not None
        assert command.startswith(f"{brave} --remote-debugging-port=9222")

    def test_linux_candidates_include_brave_binary_name(self):
        brave = "/usr/bin/brave"

        with patch("hermes_cli.browser_connect.shutil.which", side_effect=lambda name: brave if name == "brave" else None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == brave):
            candidates = get_chrome_debug_candidates("Linux")
            command = manual_chrome_debug_command(9222, "Linux")

        assert candidates == [brave]
        assert command is not None
        assert command.startswith(f"{brave} --remote-debugging-port=9222")

    def test_linux_candidates_include_official_brave_and_edge_stable_paths(self):
        brave = "/usr/bin/brave-browser-stable"
        edge = "/usr/bin/microsoft-edge-stable"

        with patch("hermes_cli.browser_connect.shutil.which", return_value=None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path in {brave, edge}):
            candidates = get_chrome_debug_candidates("Linux")

        assert candidates == [brave, edge]

    def test_launch_tries_next_browser_when_first_candidate_fails(self):
        brave = "/usr/bin/brave-browser"
        chrome = "/usr/bin/google-chrome"
        attempts = []

        def fake_popen(cmd, **kwargs):
            attempts.append(cmd[0])
            if cmd[0] == brave:
                raise OSError("broken brave install")
            return object()

        with patch("hermes_cli.browser_connect.get_chrome_debug_candidates", return_value=[brave, chrome]), \
             patch("hermes_cli.browser_connect._wait_for_browser_debug_ready_or_exit", return_value="ready"), \
             patch("subprocess.Popen", side_effect=fake_popen):
            assert HermesCLI._try_launch_chrome_debug(9222, "Linux") is True

        assert attempts == [brave, chrome]

    def test_wait_for_browser_debug_ready_or_exit_detects_early_exit(self, monkeypatch):
        class _Proc:
            def __init__(self):
                self.calls = 0

            def poll(self):
                self.calls += 1
                return 1 if self.calls >= 2 else None

        monkeypatch.setattr("hermes_cli.browser_connect.time.sleep", lambda _seconds: None)
        with patch("hermes_cli.browser_connect.is_browser_debug_ready", return_value=False):
            state = _wait_for_browser_debug_ready_or_exit(_Proc(), 9222, timeout=0.3, interval=0.01)

        assert state == "exited"

    def test_launch_tries_next_browser_when_first_candidate_exits_before_debug_ready(self):
        brave = "/usr/bin/brave-browser"
        chrome = "/usr/bin/google-chrome"
        attempts = []

        class _Proc:
            pass

        def fake_popen(cmd, **kwargs):
            attempts.append(cmd[0])
            return _Proc()

        with patch("hermes_cli.browser_connect.get_chrome_debug_candidates", return_value=[brave, chrome]), \
             patch("hermes_cli.browser_connect._wait_for_browser_debug_ready_or_exit", side_effect=["exited", "ready"]), \
             patch("subprocess.Popen", side_effect=fake_popen):
            assert HermesCLI._try_launch_chrome_debug(9222, "Linux") is True

        assert attempts == [brave, chrome]

    def test_launch_result_hints_singleton_forward_on_clean_exit(self, tmp_path, monkeypatch):
        """A candidate that exits code 0 without opening the port = an existing
        instance absorbed the launch (Chromium single-instance behavior)."""
        chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

        class _Proc:
            pid = 1234
            returncode = 0

            def poll(self):
                return 0

        monkeypatch.setattr(
            "hermes_cli.browser_connect.chrome_debug_data_dir", lambda: str(tmp_path)
        )
        with patch("hermes_cli.browser_connect.get_chrome_debug_candidates", return_value=[chrome]), \
             patch("hermes_cli.browser_connect.is_browser_debug_ready", return_value=False), \
             patch("subprocess.Popen", return_value=_Proc()):
            result = launch_chrome_debug(9222, "Windows")

        assert result.launched is False
        assert result.attempts[0].state == "exited"
        assert result.attempts[0].returncode == 0
        assert result.hint is not None
        assert "already-running" in result.hint
        assert "chrome.exe" in result.hint

    def test_launch_result_surfaces_stderr_tail_on_crash(self, tmp_path, monkeypatch):
        chrome = "/usr/bin/google-chrome"

        class _Proc:
            pid = 4321
            returncode = 127

            def __init__(self, stderr_path):
                # Simulate the browser writing to the redirected stderr file.
                with open(stderr_path, "w", encoding="utf-8") as fh:
                    fh.write("error while loading shared libraries: libnspr4.so\n")

            def poll(self):
                return 127

        monkeypatch.setattr(
            "hermes_cli.browser_connect.chrome_debug_data_dir", lambda: str(tmp_path)
        )
        stderr_path = tmp_path / "launch-stderr.log"
        with patch("hermes_cli.browser_connect.get_chrome_debug_candidates", return_value=[chrome]), \
             patch("hermes_cli.browser_connect.is_browser_debug_ready", return_value=False), \
             patch("subprocess.Popen", side_effect=lambda *a, **k: _Proc(stderr_path)):
            result = launch_chrome_debug(9222, "Linux")

        assert result.launched is False
        assert result.attempts[0].returncode == 127
        assert "libnspr4.so" in result.attempts[0].stderr_tail
        assert result.hint is not None
        assert "libnspr4.so" in result.hint

    def test_launch_result_no_hint_when_no_candidates(self):
        with patch("hermes_cli.browser_connect.get_chrome_debug_candidates", return_value=[]):
            result = launch_chrome_debug(9222, "Linux")

        assert result.launched is False
        assert result.attempts == []
        assert result.hint is None

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

    def test_connect_context_note_allows_expected_browser_use(self, monkeypatch):
        """`/browser connect` is an instruction to use the CDP browser.

        The queued context note must not tell the model to wait for a second
        permission step or imply that the attached browser is the user's main
        everyday Chrome profile.
        """
        cli = HermesCLI.__new__(HermesCLI)
        cli._pending_input = Queue()
        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

        with patch("hermes_cli.cli_commands_mixin.is_browser_debug_ready", return_value=True), \
             patch("tools.browser_tool.cleanup_all_browsers"), \
             patch("tools.browser_tool._ensure_cdp_supervisor"), \
             redirect_stdout(StringIO()):
            cli._handle_browser_command("/browser connect")

        note = cli._pending_input.get_nowait()
        assert "Chromium-family" in note
        assert "dev/debug" in note
        assert "using browser tools for their current browser-related request is expected" in note
        assert "live Chrome browser" not in note
        assert "real browser" not in note
        assert "Please await their instruction" not in note
