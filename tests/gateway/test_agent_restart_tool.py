"""Tests for gateway restart handoff coordinator and post-restart notifications."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.fixture
def temp_hermes_home(tmp_path, monkeypatch):
    """Set HERMES_HOME to a temporary directory."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_dispatch_gateway_restart_success(temp_hermes_home):
    """Verify dispatch_gateway_restart writes marker files and requests restart."""
    runner, _ = make_restart_runner()
    runner._restart_requested = False
    runner._draining = False
    runner._running_agents = {}
    runner.request_restart = MagicMock(return_value=True)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123456",
        chat_type="dm",
        user_id="user-1",
        thread_id="t-1",
        message_id="msg-1",
    )

    success, _msg = await runner.dispatch_gateway_restart(
        source=source,
        reason="Testing restart",
        update_id=999,
        message_id="msg-1",
        origin="agent_tool",
    )

    assert success is True
    runner.request_restart.assert_called_once()

    # Check notify marker
    notify_file = temp_hermes_home / ".restart_notify.json"
    assert notify_file.exists()
    notify_data = json.loads(notify_file.read_text(encoding="utf-8"))
    assert notify_data["platform"] == "telegram"
    assert notify_data["chat_id"] == "123456"
    assert notify_data["thread_id"] == "t-1"
    assert notify_data["message_id"] == "msg-1"

    # Check dedup marker
    dedup_file = temp_hermes_home / ".restart_last_processed.json"
    assert dedup_file.exists()
    dedup_data = json.loads(dedup_file.read_text(encoding="utf-8"))
    assert dedup_data["platform"] == "telegram"
    assert dedup_data["origin"] == "agent_tool"
    assert dedup_data["update_id"] == 999
    assert dedup_data["chat_id"] == "123456"
    assert dedup_data["reason"] == "Testing restart"


@pytest.mark.asyncio
async def test_dispatch_gateway_restart_busy_rejection():
    """Verify dispatch refuses when gateway is already restarting or draining."""
    runner, _ = make_restart_runner()
    runner._restart_requested = True
    runner._draining = False
    runner._running_agents = {}

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
    success, msg = await runner.dispatch_gateway_restart(source=source)
    assert success is False
    assert "in progress" in msg.lower() or "draining" in msg.lower()


@pytest.mark.asyncio
async def test_handle_restart_command_delegation():
    """Verify /restart slash command delegates to dispatch_gateway_restart."""
    runner, _ = make_restart_runner()
    runner.dispatch_gateway_restart = AsyncMock(return_value=(True, "Restarting..."))
    runner._is_stale_restart_redelivery = MagicMock(return_value=False)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
    event = MessageEvent(
        text="/restart",
        source=source,
        platform_update_id=101,
        message_id=202,
    )

    reply = await runner._handle_restart_command(event)
    runner.dispatch_gateway_restart.assert_awaited_once_with(
        source=source,
        update_id=101,
        message_id="202",
        origin="slash_command",
    )
    assert getattr(reply, "text", str(reply)) == "Restarting..."


def test_is_stale_restart_redelivery_agent_tool(temp_hermes_home):
    """Verify agent_tool origin suppresses redelivered turn matching message_id."""
    runner, _ = make_restart_runner()

    dedup_file = temp_hermes_home / ".restart_last_processed.json"
    dedup_file.write_text(
        json.dumps({
            "origin": "agent_tool",
            "chat_id": "c-100",
            "thread_id": "t-200",
            "message_id": "m-300",
        }),
        encoding="utf-8",
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="c-100",
        thread_id="t-200",
    )
    matching_event = MessageEvent(text="restart please", source=source, message_id="m-300")
    unmatched_event = MessageEvent(text="other msg", source=source, message_id="m-999")

    assert runner._is_stale_restart_redelivery(matching_event) is True
    assert runner._is_stale_restart_redelivery(unmatched_event) is False


@pytest.mark.asyncio
async def test_send_restart_notification_native_delivery(temp_hermes_home):
    """Verify _send_restart_notification uses gateway.delivery.resolve_delivery_transport."""
    runner, _ = make_restart_runner()

    notify_file = temp_hermes_home / ".restart_notify.json"
    notify_file.write_text(
        json.dumps({
            "platform": "telegram",
            "chat_id": "chat-777",
            "thread_id": "thread-888",
        }),
        encoding="utf-8",
    )

    mock_transport = MagicMock()
    mock_transport.adapter = MagicMock()
    mock_transport.send = AsyncMock(return_value=SendResult(success=True))

    with patch("gateway.delivery.resolve_delivery_transport", return_value=mock_transport) as mock_resolve:
        result = await runner._send_restart_notification()
        assert result == ("telegram", "chat-777", "thread-888")
        mock_resolve.assert_called_once()
        mock_transport.send.assert_awaited_once()

        # Marker should be unlinked after send
        assert not notify_file.exists()
