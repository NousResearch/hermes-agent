"""Tests for Ctrl+Z suspend scope (#83006).

The CLI's Ctrl+Z binding used to suspend the *entire process group* via
``os.kill(0, SIGTSTP)``, which also stopped every background job sharing
the group (long-running terminal/OCR tasks spawned from the session) and
turned a stray 0x1A byte in pasted input into an apparent crash. The fix
signals only the current process, matching shell job-control semantics.
"""
import signal
from unittest.mock import patch

from cli import _suspend_cli_process


def test_suspend_targets_only_current_process():
    """Ctrl+Z must signal the current pid, never the process group (0)."""
    with patch("os.kill") as mock_kill, patch("os.getpid", return_value=4242):
        _suspend_cli_process()
    mock_kill.assert_called_once_with(4242, signal.SIGTSTP)
    called_pids = [call.args[0] for call in mock_kill.call_args_list]
    assert 0 not in called_pids, (
        "Ctrl+Z must not signal the whole process group (os.kill(0, ...))"
    )
