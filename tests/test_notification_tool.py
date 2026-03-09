"""
Tests for notification_tool.
Run: python -m pytest tests/test_notification_tool.py -v
"""

import importlib
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Insert repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Save real module if already loaded
_real_registry = sys.modules.get("tools.registry")

# Install stub
_registry_stub = MagicMock()
_registry_mod = types.ModuleType("tools.registry")
_registry_mod.registry = _registry_stub
sys.modules["tools.registry"] = _registry_mod

# Import notification_tool with stub active
import tools.notification_tool as _nt
from tools.notification_tool import (
    _handle_notify,
    _handle_notify_sound,
    _applescript_escape,
    _check_notify,
    _check_sound,
    _is_headless,
)

# Restore real registry module so other tests are not affected
if _real_registry is not None:
    sys.modules["tools.registry"] = _real_registry
else:
    del sys.modules["tools.registry"]


class TestAppleScriptEscape(unittest.TestCase):
    def test_plain_string(self):
        assert _applescript_escape("Hello") == "Hello"

    def test_double_quote_escaped(self):
        assert '\\"' in _applescript_escape('Say "hi"')

    def test_backslash_escaped(self):
        assert "\\\\" in _applescript_escape("a\\b")

    def test_single_quote_untouched(self):
        assert "it's done" in _applescript_escape("it's done")


class TestIsHeadless(unittest.TestCase):
    def test_ssh_connection_env(self):
        with patch.dict(os.environ, {"SSH_CONNECTION": "1.2.3.4 22 5.6.7.8 22"}):
            assert _is_headless() is True

    def test_ssh_tty_env(self):
        with patch.dict(os.environ, {"SSH_TTY": "/dev/pts/0"}):
            assert _is_headless() is True

    @patch("sys.platform", "linux")
    def test_linux_no_display(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("DISPLAY", "WAYLAND_DISPLAY", "SSH_CONNECTION", "SSH_TTY")}
        with patch.dict(os.environ, env, clear=True):
            assert _is_headless() is True

    @patch("sys.platform", "linux")
    def test_linux_with_display(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("SSH_CONNECTION", "SSH_TTY")}
        env["DISPLAY"] = ":0"
        with patch.dict(os.environ, env, clear=True):
            assert _is_headless() is False


class TestHandleNotifySSH(unittest.TestCase):
    @patch("tools.notification_tool._is_headless", return_value=True)
    @patch("builtins.print")
    def test_ssh_falls_back_to_bell(self, mock_print, _):
        result = _handle_notify({"title": "Done", "message": "Finished"})
        assert result["success"] is True
        assert result["backend"] == "terminal_bell"
        mock_print.assert_called_with("\a", end="", flush=True)


class TestHandleNotifyLinux(unittest.TestCase):
    @patch("tools.notification_tool._is_headless", return_value=False)
    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value="/usr/bin/notify-send")
    @patch("subprocess.run")
    def test_linux_success(self, mock_run, *_):
        mock_run.return_value = MagicMock(returncode=0)
        result = _handle_notify({"title": "Done", "message": "Task finished"})
        assert result["success"] is True
        assert result["backend"] == "linux"
        call_args = mock_run.call_args[0][0]
        assert "Done" in call_args and "Task finished" in call_args

    @patch("tools.notification_tool._is_headless", return_value=False)
    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value=None)
    @patch("builtins.print")
    def test_linux_fallback_to_bell(self, mock_print, *_):
        result = _handle_notify({"title": "Done", "message": "Hi"})
        assert result["success"] is True
        assert result["backend"] == "terminal_bell"
        mock_print.assert_called_with("\a", end="", flush=True)


class TestHandleNotifyMacOS(unittest.TestCase):
    @patch("tools.notification_tool._is_headless", return_value=False)
    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_macos_uses_osascript_stdin(self, mock_run, _):
        mock_run.return_value = MagicMock(returncode=0)
        result = _handle_notify({"title": "Done", "message": "Task finished"})
        assert result["success"] is True
        assert result["backend"] == "darwin"
        assert mock_run.call_args[0][0] == ["osascript"]
        stdin_input = mock_run.call_args[1].get("input", b"")
        assert b"Done" in stdin_input and b"Task finished" in stdin_input

    @patch("tools.notification_tool._is_headless", return_value=False)
    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_macos_special_chars_escaped(self, mock_run, _):
        mock_run.return_value = MagicMock(returncode=0)
        _handle_notify({"title": 'It\'s "done"', "message": 'Result: "OK"'})
        stdin_input = mock_run.call_args[1].get("input", b"")
        assert b'\\"' in stdin_input


class TestHandleNotifyWindows(unittest.TestCase):
    @patch("tools.notification_tool._is_headless", return_value=False)
    @patch("sys.platform", "win32")
    @patch("subprocess.run")
    def test_windows_uses_encoded_command(self, mock_run, _):
        mock_run.return_value = MagicMock(returncode=0)
        result = _handle_notify({"title": "Done", "message": "Finished"})
        assert result["success"] is True
        call_args = mock_run.call_args[0][0]
        assert "-EncodedCommand" in call_args
        assert "Done" in call_args
        assert "Finished" in call_args

    @patch("tools.notification_tool._is_headless", return_value=False)
    @patch("sys.platform", "win32")
    @patch("subprocess.run")
    def test_windows_injection_safe(self, mock_run, _):
        import base64
        mock_run.return_value = MagicMock(returncode=0)
        evil = '"; Remove-Item -Recurse C:\\ #'
        _handle_notify({"title": evil, "message": "ok"})
        call_args = mock_run.call_args[0][0]
        encoded_idx = call_args.index("-EncodedCommand") + 1
        decoded = base64.b64decode(call_args[encoded_idx]).decode("utf-16-le")
        assert evil not in decoded


class TestHandleSound(unittest.TestCase):
    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_macos_sound_complete(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = _handle_notify_sound({"sound": "complete"})
        assert result["success"] is True
        assert "Glass.aiff" in mock_run.call_args[0][0][1]

    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_macos_sound_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = _handle_notify_sound({"sound": "error"})
        assert "Basso.aiff" in mock_run.call_args[0][0][1]

    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value=None)
    @patch("builtins.print")
    def test_linux_no_player_fallback(self, mock_print, *_):
        result = _handle_notify_sound({})
        assert result["success"] is True
        assert result["backend"] == "terminal_bell"

    @patch("sys.platform", "win32")
    @patch("subprocess.run")
    def test_windows_sound_encoded_command(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = _handle_notify_sound({"sound": "default"})
        assert result["success"] is True
        assert "-EncodedCommand" in mock_run.call_args[0][0]


class TestUrgencyValidation(unittest.TestCase):
    @patch("tools.notification_tool._is_headless", return_value=False)
    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value="/usr/bin/notify-send")
    @patch("subprocess.run")
    def test_invalid_urgency_defaults_to_normal(self, mock_run, *_):
        mock_run.return_value = MagicMock(returncode=0)
        _handle_notify({"title": "Hi", "message": "Hello", "urgency": "INVALID"})
        call_args = mock_run.call_args[0][0]
        assert call_args[call_args.index("--urgency") + 1] == "normal"


class TestCheckFunctions(unittest.TestCase):
    @patch("tools.notification_tool._is_headless", return_value=True)
    def test_check_notify_headless_true(self, _):
        assert _check_notify() is True

    @patch("tools.notification_tool._is_headless", return_value=False)
    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value="/usr/bin/notify-send")
    def test_check_notify_linux_available(self, *_):
        assert _check_notify() is True

    @patch("tools.notification_tool._is_headless", return_value=False)
    @patch("sys.platform", "linux")
    @patch("shutil.which", return_value=None)
    def test_check_notify_linux_unavailable(self, *_):
        assert _check_notify() is False

    @patch("sys.platform", "darwin")
    def test_check_notify_macos_always_true(self):
        assert _check_notify() is True


if __name__ == "__main__":
    unittest.main(verbosity=2)
