import sys
import threading
import time
from unittest.mock import patch, MagicMock
import pytest

from hermes_cli import web_server

class SynchronousThread:
    def __init__(self, target, args=(), kwargs=None, **other_kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


@pytest.fixture(autouse=True)
def _mock_thread_and_sleep():
    with patch("threading.Thread", SynchronousThread), patch("time.sleep", return_value=None):
        yield


def test_maybe_open_browser_headless_linux_skips():
    with patch("sys.platform", "linux"), \
         patch("hermes_constants.is_wsl", return_value=False), \
         patch.dict("os.environ", {}, clear=True), \
         patch("webbrowser.open") as mock_open:
        
        web_server._maybe_open_browser("127.0.0.1", 9119, open_browser=True, initial_profile="")
        mock_open.assert_not_called()


def test_maybe_open_browser_display_linux_uses_webbrowser():
    with patch("sys.platform", "linux"), \
         patch("hermes_constants.is_wsl", return_value=False), \
         patch.dict("os.environ", {"DISPLAY": ":0"}, clear=True), \
         patch("webbrowser.open") as mock_open:
        
        web_server._maybe_open_browser("127.0.0.1", 9119, open_browser=True, initial_profile="")
        mock_open.assert_called_once_with("http://127.0.0.1:9119")


def test_maybe_open_browser_wsl_uses_powershell():
    with patch("sys.platform", "linux"), \
         patch("hermes_constants.is_wsl", return_value=True), \
         patch.dict("os.environ", {}, clear=True), \
         patch("subprocess.Popen") as mock_popen, \
         patch("webbrowser.open") as mock_open:
        
        web_server._maybe_open_browser("127.0.0.1", 9119, open_browser=True, initial_profile="default")
        
        # Should call powershell.exe instead of webbrowser.open
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert "powershell.exe" in args
        assert "Start-Process 'http://127.0.0.1:9119/?profile=default'" in args[3]
        mock_open.assert_not_called()


def test_maybe_open_browser_wsl_powershell_fails_falls_back():
    # If powershell.exe fails (e.g. raises FileNotFoundError), it should fall back to webbrowser
    with patch("sys.platform", "linux"), \
         patch("hermes_constants.is_wsl", return_value=True), \
         patch.dict("os.environ", {}, clear=True), \
         patch("subprocess.Popen", side_effect=FileNotFoundError("powershell")), \
         patch("webbrowser.open") as mock_open:
        
        web_server._maybe_open_browser("127.0.0.1", 9119, open_browser=True, initial_profile="")
        
        mock_open.assert_called_once_with("http://127.0.0.1:9119")
