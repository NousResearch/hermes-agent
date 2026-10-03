"""`_spawn_detached_update` must survive a job object that forbids breakaway.

Windows detaches the updater with CREATE_BREAKAWAY_FROM_JOB so it outlives the gateway
restart it may trigger. A job without BREAKAWAY_OK (Desktop/Electron, scheduled tasks)
rejects that flag with OSError — `/update` then failed after `.update_pending.json` was
already written, the same shape that made every WhatsApp reconnect fail (#68128).
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gateway import slash_commands

_BREAKAWAY_BIT = 0x01000000


def _win_patches(mock_popen):
    return [
        patch.object(sys, "platform", "win32"),
        patch("hermes_cli._subprocess_compat.IS_WINDOWS", True),
        patch(
            "hermes_cli._subprocess_compat.windows_detach_flags",
            return_value=_BREAKAWAY_BIT | 0x08000000,
        ),
        patch(
            "hermes_cli._subprocess_compat.windows_detach_flags_without_breakaway",
            return_value=0x08000000,
        ),
        patch("subprocess.Popen", mock_popen),
    ]


def test_breakaway_denied_retries_without_breakaway(tmp_path):
    mock_popen = MagicMock()
    mock_popen.side_effect = [PermissionError(5, "Access is denied"), MagicMock()]

    patches = _win_patches(mock_popen)
    for p in patches:
        p.start()
    try:
        slash_commands._spawn_detached_update(
            ["hermes"], tmp_path / "out.txt", tmp_path / "exit_code"
        )
    finally:
        for p in patches:
            p.stop()

    assert mock_popen.call_count == 2
    first_flags = mock_popen.call_args_list[0].kwargs["creationflags"]
    second_flags = mock_popen.call_args_list[1].kwargs["creationflags"]
    assert first_flags & _BREAKAWAY_BIT  # breakaway attempted first
    assert not second_flags & _BREAKAWAY_BIT  # fallback drops only the breakaway bit
    # Same updater argv on both attempts; only the flags changed.
    first_argv = mock_popen.call_args_list[0].args[0]
    second_argv = mock_popen.call_args_list[1].args[0]
    assert first_argv == second_argv
    assert first_argv[3:5] == [str(tmp_path / "out.txt"), str(tmp_path / "exit_code")]
    assert first_argv[-2:] == ["update", "--gateway"]


def test_breakaway_allowed_single_spawn(tmp_path):
    mock_popen = MagicMock()

    patches = _win_patches(mock_popen)
    for p in patches:
        p.start()
    try:
        slash_commands._spawn_detached_update(
            ["hermes"], tmp_path / "out.txt", tmp_path / "exit_code"
        )
    finally:
        for p in patches:
            p.stop()

    mock_popen.assert_called_once()
    assert mock_popen.call_args.kwargs["creationflags"] & _BREAKAWAY_BIT
