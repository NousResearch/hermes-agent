"""Tests for /restart notification — the gateway notifies the requester on comeback."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import (
    make_restart_runner,
    make_restart_source,
)


# ── _handle_restart_command writes .restart_notify.json ──────────────────


@pytest.mark.asyncio
async def test_restart_command_writes_notify_file(tmp_path, monkeypatch):
    """When /restart fires, the requester's routing info is persisted to disk."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    source = make_restart_source(chat_id="42")
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )

    result = await runner._handle_restart_command(event)
    assert "Restarting" in result

    notify_path = tmp_path / ".restart_notify.json"
    assert notify_path.exists()
    data = json.loads(notify_path.read_text())
    assert data["platform"] == "telegram"
    assert data["chat_id"] == "42"
    assert "thread_id" not in data  # no thread → omitted


@pytest.mark.asyncio
async def test_restart_command_preserves_thread_id(tmp_path, monkeypatch):
    """Thread ID is saved when the requester is in a threaded chat."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    source = make_restart_source(chat_id="99")
    source.thread_id = "topic_7"

    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m2",
    )

    await runner._handle_restart_command(event)

    data = json.loads((tmp_path / ".restart_notify.json").read_text())
    assert data["thread_id"] == "topic_7"


# ── _send_restart_notification ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_restart_notification_delivers_and_cleans_up(tmp_path, monkeypatch):
    """On startup, the notification is sent and the file is removed."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(json.dumps({
        "platform": "telegram",
        "chat_id": "42",
    }))

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock()

    await runner._send_restart_notification()

    adapter.send.assert_called_once()
    call_args = adapter.send.call_args
    assert call_args[0][0] == "42"  # chat_id
    assert "restarted" in call_args[0][1].lower()
    assert call_args[1].get("metadata") is None  # no thread
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_with_thread(tmp_path, monkeypatch):
    """Thread ID is passed as metadata so the message lands in the right topic."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(json.dumps({
        "platform": "telegram",
        "chat_id": "99",
        "thread_id": "topic_7",
    }))

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock()

    await runner._send_restart_notification()

    call_args = adapter.send.call_args
    assert call_args[1]["metadata"] == {"thread_id": "topic_7"}
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_noop_when_no_file(tmp_path, monkeypatch):
    """Nothing happens if there's no pending restart notification."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock()

    await runner._send_restart_notification()

    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_send_restart_notification_skips_when_adapter_missing(tmp_path, monkeypatch):
    """If the requester's platform isn't connected, clean up without crashing."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(json.dumps({
        "platform": "discord",  # runner only has telegram adapter
        "chat_id": "42",
    }))

    runner, _adapter = make_restart_runner()

    await runner._send_restart_notification()

    # File cleaned up even though we couldn't send
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_cleans_up_on_send_failure(
    tmp_path, monkeypatch
):
    """If the adapter.send() raises, the file is still cleaned up."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(json.dumps({
        "platform": "telegram",
        "chat_id": "42",
    }))

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock(side_effect=RuntimeError("network down"))

    await runner._send_restart_notification()

    assert not notify_path.exists()  # cleaned up despite error
