"""Regression tests: chat /restart is fail-closed in every service context.

Only an external CLI/supervisor may own gateway successor startup. Environment
markers must not re-enable an in-process service or detached restart path.
"""
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from gateway.platforms.base import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _make_restart_event(update_id: int | None = 100) -> MessageEvent:
    return MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m1",
        platform_update_id=update_id,
    )


def _make_runner_with_mock_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    return runner


@pytest.mark.asyncio
async def test_restart_under_launchd_is_fail_closed(tmp_path, monkeypatch):
    """launchd context must not re-enable an in-gateway restart path."""
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)
    monkeypatch.setenv("XPC_SERVICE_NAME", "ai.hermes.gateway")

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_restart_in_interactive_macos_shell_is_fail_closed(
    tmp_path, monkeypatch
):
    """Interactive macOS context must not use a detached helper."""
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)
    monkeypatch.setenv("XPC_SERVICE_NAME", "0")

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_restart_without_service_env_is_fail_closed(tmp_path, monkeypatch):
    """No service-manager environment must not enable detached restart."""
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_restart_under_systemd_is_fail_closed(tmp_path, monkeypatch):
    """systemd context must not re-enable an in-gateway restart path."""
    runner = _make_runner_with_mock_restart(tmp_path, monkeypatch)
    monkeypatch.setenv("INVOCATION_ID", "abc123")

    await runner._handle_restart_command(_make_restart_event())

    runner.request_restart.assert_not_called()
