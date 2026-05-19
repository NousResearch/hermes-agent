"""Tests for exec-type quick commands bypassing the draining guard.

When the gateway is in draining state (e.g. after SIGTERM or repeated LLM
failures), exec-type quick commands (those using ``asyncio.create_subprocess_shell``)
should still execute because they do not depend on the agent loop or LLM backend.

See https://github.com/NousResearch/hermes-agent/issues/28663
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.restart_test_helpers import RestartTestAdapter, make_restart_runner, make_restart_source


def _make_quick_command_event(command_name: str = "syscheck") -> MessageEvent:
    """Create a MessageEvent for a quick command."""
    return MessageEvent(
        text=f"/{command_name}",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m-qc1",
    )


def _configure_exec_quick_command(runner, command_name: str = "syscheck", shell_cmd: str = "echo OK"):
    """Set up a quick command with type: exec on the runner's config."""
    runner.config.quick_commands = {
        command_name: {
            "type": "exec",
            "command": shell_cmd,
        },
    }


def _add_runner_mocks(runner):
    """Add common mocks that _handle_message expects on the runner."""
    runner._session_db = None


@pytest.mark.asyncio
async def test_exec_quick_command_runs_while_draining():
    """Exec-type quick commands bypass the _draining guard and execute successfully."""
    runner, _adapter = make_restart_runner()
    _add_runner_mocks(runner)
    runner._draining = True
    runner._restart_requested = True

    _configure_exec_quick_command(runner, "syscheck", "echo health-ok")
    event = _make_quick_command_event("syscheck")

    result = await runner._handle_message(event)

    assert result == "health-ok"


@pytest.mark.asyncio
async def test_exec_quick_command_timeout_while_draining():
    """Exec-type quick commands still respect the 30s timeout during draining."""
    runner, _adapter = make_restart_runner()
    _add_runner_mocks(runner)
    runner._draining = True

    _configure_exec_quick_command(runner, "slowcmd", "sleep 60")
    event = _make_quick_command_event("slowcmd")

    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
        result = await runner._handle_message(event)

    assert "timed out" in result


@pytest.mark.asyncio
async def test_exec_quick_command_error_while_draining():
    """Exec-type quick commands still report errors during draining."""
    runner, _adapter = make_restart_runner()
    _add_runner_mocks(runner)
    runner._draining = True

    _configure_exec_quick_command(runner, "failcmd", "nonexistent_command_xyz_12345")
    event = _make_quick_command_event("failcmd")

    result = await runner._handle_message(event)
    # Should NOT be the draining message — the exec command ran (and produced output)
    assert "not accepting new work" not in result


@pytest.mark.asyncio
async def test_exec_quick_command_with_empty_command_returns_error():
    """Exec quick command with empty 'command' field returns an error, even while draining."""
    runner, _adapter = make_restart_runner()
    _add_runner_mocks(runner)
    runner._draining = True

    _configure_exec_quick_command(runner, "emptycmd", "")
    event = _make_quick_command_event("emptycmd")

    result = await runner._handle_message(event)

    assert "no command defined" in result


@pytest.mark.asyncio
async def test_exec_quick_command_works_when_not_draining():
    """Exec-type quick commands work normally when not draining (regression check)."""
    runner, _adapter = make_restart_runner()
    _add_runner_mocks(runner)
    runner._draining = False

    _configure_exec_quick_command(runner, "uptimechk", "echo uptime-ok")
    event = _make_quick_command_event("uptimechk")

    result = await runner._handle_message(event)

    assert result == "uptime-ok"


@pytest.mark.asyncio
async def test_unknown_command_blocked_while_draining():
    """Unknown slash commands (not quick commands, not built-in) are blocked during draining."""
    runner, _adapter = make_restart_runner()
    _add_runner_mocks(runner)
    runner._draining = True
    runner._restart_requested = True

    event = MessageEvent(
        text="/xyz-nonexistent-cmd-abc",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m-unknown",
    )

    result = await runner._handle_message(event)

    assert "not accepting new work" in result
