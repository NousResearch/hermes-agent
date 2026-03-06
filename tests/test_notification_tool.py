"""
Tests for notification_tool.
Run: python tests/test_notification_tool.py
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

registry_stub = MagicMock()
tools_stub = types.ModuleType("tools")
tools_stub.registry = types.ModuleType("tools.registry")
tools_stub.registry.registry = registry_stub
sys.modules.setdefault("tools", tools_stub)
sys.modules.setdefault("tools.registry", tools_stub.registry)

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.notification_tool import (
    _handle_notify,
    _handle_notify_sound,
    _applescript_quote,
    _check_notify,
    _check_sound,
)


class TestAppleScriptQuote(unittest.TestCase):
    def test_plain_string(self):
        assert _applescript_quote("Hello") == '"Hello"'

    def test_double_quote_escaped(self):
        result = _applescript_quote('Say "hi"')
        assert '\\"' in result

    def test_no_single_quote_issue(self):
        result = _applescript_quote("it's done")
        assert "it's done" in result


class TestHandleNotifyLinux(unittest.TestCase):
    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value="/usr/bin/notify-send")
    @patch("subprocess.run")
    def test_linux_success(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0)
        result = _handle_notify({"title": "Done", "message": "Task finished"})
        assert result["success"] is True
        assert result["backend"] == "linux"
        call_args = mock_run.call_args[0][0]
        assert isinstance(call_args, list)
        assert "Done" in call_args
        assert "Task finished" in call_args

    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value=None)
    @patch("builtins.print")
    def test_linux_fallback_to_bell(self, mock_print, mock_which):
        result = _handle_notify({"title": "Done", "message": "Hi"})
        assert result["success"] is True
        assert result["backend"] == "terminal_bell"
        mock_print.assert_called_with("\a", end="", flush=True)


class TestHandleNotifyMacOS(unittest.TestCase):
    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_macos_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = _handle_notify({"title": "Done", "message": "Task finished"})
        assert result["success"] is True
        assert result["backend"] == "darwin"
        call_args = mock_run.call_args[0][0]
        assert isinstance(call_args, list)
        assert call_args[0] == "osascript"

    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_macos_special_chars_safe(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = _handle_notify({
            "title": 'It\'s "done"',
            "message": 'Result: "OK"',
        })
        assert result["success"] is True


class TestHandleNotifyWindows(unittest.TestCase):
    @patch("sys.platform", "win32")
    @patch("subprocess.run")
    def test_windows_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = _handle_notify({"title": "Done", "message": "Finished"})
        assert result["success"] is True
        assert result["backend"] == "win32"
        call_args = mock_run.call_args[0][0]
        assert isinstance(call_args, list)
        assert "Done" in call_args
        assert "Finished" in call_args


class TestHandleSound(unittest.TestCase):
    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_macos_sound(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = _handle_notify_sound({"sound": "complete"})
        assert result["success"] is True
        call_args = mock_run.call_args[0][0]
        assert "Glass.aiff" in call_args[1]

    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value=None)
    @patch("builtins.print")
    def test_linux_no_player_fallback(self, mock_print, mock_which):
        result = _handle_notify_sound({})
        assert result["success"] is True
        assert result["backend"] == "terminal_bell"


class TestUrgencyValidation(unittest.TestCase):
    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value="/usr/bin/notify-send")
    @patch("subprocess.run")
    def test_invalid_urgency_defaults_to_normal(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0)
        result = _handle_notify({
            "title": "Hi",
            "message": "Hello",
            "urgency": "INVALID",
        })
        assert result["success"] is True
        call_args = mock_run.call_args[0][0]
        urgency_idx = call_args.index("--urgency")
        assert call_args[urgency_idx + 1] == "normal"


class TestCheckFunctions(unittest.TestCase):
    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value="/usr/bin/notify-send")
    def test_check_notify_linux_available(self, _):
        assert _check_notify() is True

    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value=None)
    def test_check_notify_linux_unavailable(self, _):
        assert _check_notify() is False

    @patch("sys.platform", "darwin")
    def test_check_notify_macos_always_true(self):
        assert _check_notify() is True


if __name__ == "__main__":
    unittest.main(verbosity=2)
