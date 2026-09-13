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
    get_darwin_browser_app_paths,
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



    def test_manual_command_uses_detected_linux_browser(self):
        with patch("hermes_cli.browser_connect.shutil.which", side_effect=lambda name: "/usr/bin/chromium" if name == "chromium" else None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == "/usr/bin/chromium"):
            command = manual_chrome_debug_command(9222, "Linux")

        assert command is not None
        assert command.startswith("/usr/bin/chromium --remote-debugging-port=9222")






    def test_linux_candidates_include_brave_origin_binary_name(self):
        brave = "/usr/bin/brave-origin"

        with patch("hermes_cli.browser_connect.shutil.which", side_effect=lambda name: brave if name == "brave-origin" else None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == brave):
            candidates = get_chrome_debug_candidates("Linux")
            command = manual_chrome_debug_command(9222, "Linux")

        assert candidates == [brave]
        assert command is not None
        assert command.startswith(f"{brave} --remote-debugging-port=9222")

    def test_linux_candidates_include_brave_origin_install_path(self):
        brave = "/opt/brave.com/brave-origin/brave-origin"

        with patch("hermes_cli.browser_connect.shutil.which", return_value=None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == brave):
            candidates = get_chrome_debug_candidates("Linux")
            command = manual_chrome_debug_command(9222, "Linux")

        assert candidates == [brave]
        assert command is not None
        assert command.startswith(f"{brave} --remote-debugging-port=9222")

    def test_linux_candidates_include_brave_origin_nightly_binary_name(self):
        brave = "/usr/bin/brave-origin-nightly"

        with patch("hermes_cli.browser_connect.shutil.which", side_effect=lambda name: brave if name == "brave-origin-nightly" else None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == brave):
            candidates = get_chrome_debug_candidates("Linux")
            command = manual_chrome_debug_command(9222, "Linux")

        assert candidates == [brave]
        assert command is not None
        assert command.startswith(f"{brave} --remote-debugging-port=9222")

    def test_linux_candidates_include_brave_origin_nightly_install_path(self):
        brave = "/opt/brave.com/brave-origin-nightly/brave-origin"

        with patch("hermes_cli.browser_connect.shutil.which", return_value=None), \
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

    def test_darwin_candidates_include_user_applications_in_stable_order(self, monkeypatch):
        home = "/Users/alice"
        system_chrome = os.path.join(
            "/Applications",
            "Google Chrome.app",
            "Contents",
            "MacOS",
            "Google Chrome",
        )
        user_chrome = os.path.join(
            home,
            "Applications",
            "Google Chrome.app",
            "Contents",
            "MacOS",
            "Google Chrome",
        )
        system_chromium = os.path.join(
            "/Applications",
            "Chromium.app",
            "Contents",
            "MacOS",
            "Chromium",
        )

        monkeypatch.setattr(
            "hermes_cli.browser_connect.os.path.expanduser",
            lambda value: home if value == "~" else value,
        )
        existing = {system_chrome, user_chrome, system_chromium}
        with patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path in existing):
            candidates = get_chrome_debug_candidates("Darwin")

        assert candidates == [system_chrome, user_chrome, system_chromium]
        assert len(candidates) == len({os.path.normcase(os.path.normpath(path)) for path in candidates})
        assert user_chrome in get_darwin_browser_app_paths(home)


    def test_wsl_install_candidates_keep_posix_separators_on_nt_host(self):
        expected = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"

        with patch("hermes_cli.browser_connect.shutil.which", return_value=None), \
             patch("hermes_cli.browser_connect.os.path.isfile", side_effect=lambda path: path == expected):
            candidates = get_chrome_debug_candidates("Linux")

        assert candidates == [expected]
        assert "\\" not in candidates[0]


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


