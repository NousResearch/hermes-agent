"""Tests for PR #16621: gateway restart watcher using fcntl file-lock."""

import sys
from unittest.mock import patch

import pytest

from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_launch_detached_restart_spawns_python_not_bash():
    """The restart watcher must use Python+fcntl, not bash+kill -0."""
    runner = object.__new__(GatewayRunner)
    runner._background_tasks = set()

    with (
        patch("gateway.run._resolve_hermes_bin", return_value=["/usr/bin/hermes"]),
        patch("subprocess.Popen") as mock_popen,
        patch("gateway.status._get_gateway_lock_path", return_value="/tmp/gateway.lock"),
        patch("gateway.run.logger"),
    ):
        await runner._launch_detached_restart_command()

    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    popen_cmd = args[0] if args else kwargs.get("args", [])

    # The command should be Python, NOT bash
    assert popen_cmd[0] == sys.executable, (
        f"Expected Python ({sys.executable}), got {popen_cmd[0]}"
    )
    assert popen_cmd[1] == "-c", "Expected '-c' to run inline code"

    watcher_code = popen_cmd[2]

    # Must use fcntl.flock, not kill -0 polling
    assert "fcntl.flock" in watcher_code, (
        "Watcher must use fcntl.flock for lock-based waiting"
    )
    assert "kill -0" not in watcher_code, (
        "Old bash kill -0 pattern must not be present"
    )

    # Must handle duplicate restart via non-blocking lock
    assert "BlockingIOError" in watcher_code, (
        "Watcher must handle duplicate restart via BlockingIOError"
    )
    assert "LOCK_NB" in watcher_code, (
        "Watcher must use non-blocking lock (LOCK_NB) for dedup"
    )

    # Must run in a new session (detached)
    assert kwargs.get("start_new_session") is True, (
        "Watcher must run in a detached session"
    )


@pytest.mark.asyncio
async def test_launch_detached_restart_no_bash_invocation():
    """Verify no bash or setsid is invoked in the new watcher."""
    runner = object.__new__(GatewayRunner)
    runner._background_tasks = set()

    with (
        patch("gateway.run._resolve_hermes_bin", return_value=["/usr/bin/hermes"]),
        patch("subprocess.Popen") as mock_popen,
        patch("gateway.status._get_gateway_lock_path", return_value="/tmp/gateway.lock"),
        patch("gateway.run.logger"),
    ):
        await runner._launch_detached_restart_command()

    mock_popen.assert_called_once()
    args, _ = mock_popen.call_args
    popen_cmd = args[0]

    # No bash anywhere in the command
    for part in popen_cmd:
        assert "bash" not in str(part), (
            f"bash should not appear in restart command, found: {part}"
        )


@pytest.mark.asyncio
async def test_launch_detached_restart_graceful_missing_binary():
    """Should return silently (no crash) when hermes binary is not found."""
    runner = object.__new__(GatewayRunner)
    runner._background_tasks = set()

    with (
        patch("gateway.run._resolve_hermes_bin", return_value=None),
        patch("subprocess.Popen") as mock_popen,
        patch("gateway.run.logger"),
    ):
        await runner._launch_detached_restart_command()

    # Must NOT call subprocess.Popen when no binary
    mock_popen.assert_not_called()


@pytest.mark.asyncio
async def test_watcher_code_is_valid_python():
    """The generated watcher code must be syntactically valid Python."""
    runner = object.__new__(GatewayRunner)
    runner._background_tasks = set()

    with (
        patch("gateway.run._resolve_hermes_bin", return_value=["/usr/bin/hermes"]),
        patch("subprocess.Popen") as mock_popen,
        patch("gateway.status._get_gateway_lock_path", return_value="/tmp/gateway.lock"),
        patch("gateway.run.logger"),
    ):
        await runner._launch_detached_restart_command()

    args, _ = mock_popen.call_args
    watcher_code = args[0][2]

    # Must compile without syntax errors
    try:
        compile(watcher_code, "<watcher>", "exec")
    except SyntaxError as e:
        pytest.fail(f"Watcher code has syntax error: {e}")
