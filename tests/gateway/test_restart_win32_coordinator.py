"""Tests for Windows transactional restart coordinator integration in slash_commands.

Verifies that _handle_restart_command in GatewaySlashCommandsMixin correctly
routes through the Windows coordinator on win32 and preserves upstream behavior
on non-Windows platforms.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.platforms.base import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _make_event(update_id=None):
    return MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m1",
        platform_update_id=update_id,
    )


# ── Windows coordinator tests (sys.platform == "win32") ──────────────────


@pytest.mark.asyncio
async def test_win32_coordinator_scheduled_returns_restarting(tmp_path, monkeypatch):
    """win32 + coordinator scheduled=True → returns restarting, no legacy call."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")

    runner, _ = make_restart_runner()
    runner._restart_requested = False
    runner._draining = False
    runner.request_restart = MagicMock()

    with patch(
        "hermes_cli.gateway_windows_restart.schedule_restart_handoff",
        return_value={"scheduled": True, "request_id": "abc-123"},
    ) as mock_coord:
        result = await runner._handle_restart_command(_make_event())

    assert "Restarting" in str(result) or "restarting" in str(result).lower()
    mock_coord.assert_called_once_with(origin="slash-command", wait=False)
    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_win32_coordinator_scheduled_false_returns_failure(tmp_path, monkeypatch):
    """win32 + coordinator scheduled=False → returns failure message, no legacy call."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")

    runner, _ = make_restart_runner()
    runner._restart_requested = False
    runner._draining = False
    runner.request_restart = MagicMock()

    with patch(
        "hermes_cli.gateway_windows_restart.schedule_restart_handoff",
        return_value={"scheduled": False, "detail": "task not registered"},
    ):
        result = await runner._handle_restart_command(_make_event())

    assert "failed" in str(result).lower()
    assert "task not registered" in str(result)
    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_win32_coordinator_exception_returns_failure(tmp_path, monkeypatch):
    """win32 + coordinator raises → returns failure, no legacy call."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")

    runner, _ = make_restart_runner()
    runner._restart_requested = False
    runner._draining = False
    runner.request_restart = MagicMock()

    with patch(
        "hermes_cli.gateway_windows_restart.schedule_restart_handoff",
        side_effect=RuntimeError("boom"),
    ):
        result = await runner._handle_restart_command(_make_event())

    assert "failed" in str(result).lower()
    assert "boom" in str(result)
    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_win32_coordinator_with_active_agents_returns_draining(tmp_path, monkeypatch):
    """win32 + coordinator scheduled=True + active agents → draining message."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")

    runner, _ = make_restart_runner()
    runner._restart_requested = False
    runner._draining = False
    runner._running_agent_count = MagicMock(return_value=3)
    runner.request_restart = MagicMock()

    with patch(
        "hermes_cli.gateway_windows_restart.schedule_restart_handoff",
        return_value={"scheduled": True, "request_id": "abc-123"},
    ):
        result = await runner._handle_restart_command(_make_event())

    assert "3" in str(result) or "draining" in str(result).lower()
    runner.request_restart.assert_not_called()


# ── Non-Windows upstream behavior tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_nonwin_systemd_uses_service_restart(tmp_path, monkeypatch):
    """non-Windows + systemd → request_restart(detached=False, via_service=True)."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("INVOCATION_ID", "systemd-123")

    runner, _ = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    await runner._handle_restart_command(_make_event())
    runner.request_restart.assert_called_once_with(detached=False, via_service=True)


@pytest.mark.asyncio
async def test_nonwin_no_systemd_uses_detached(tmp_path, monkeypatch):
    """non-Windows + no systemd → request_restart(detached=True, via_service=False)."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    runner, _ = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    await runner._handle_restart_command(_make_event())
    runner.request_restart.assert_called_once_with(detached=True, via_service=False)


# ── Marker write tests (platform-independent) ────────────────────────────


@pytest.mark.asyncio
async def test_restart_notify_marker_written(tmp_path, monkeypatch):
    """_handle_restart_command always writes .restart_notify.json."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    runner, _ = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    await runner._handle_restart_command(_make_event())

    notify_path = tmp_path / ".restart_notify.json"
    assert notify_path.exists()


@pytest.mark.asyncio
async def test_restart_dedup_marker_written_and_blocks_redelivery(tmp_path, monkeypatch):
    """Dedup marker is written and prevents re-processing of same update_id."""
    import json

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    runner, _ = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    # First /restart with update_id=100
    result1 = await runner._handle_restart_command(_make_event(update_id=100))
    assert "Restarting" in str(result1) or "restarting" in str(result1).lower()

    # Second /restart with same update_id should be blocked
    runner.request_restart.reset_mock()
    result2 = await runner._handle_restart_command(_make_event(update_id=100))
    assert result2 == ""
    runner.request_restart.assert_not_called()
