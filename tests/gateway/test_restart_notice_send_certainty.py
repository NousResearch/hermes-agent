"""Restart replay requires adapter certainty, not merely a transient failure."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
import gateway.run_restart_notifications as notices
from gateway.config import PlatformConfig
from plugins.platforms.teams.adapter import TeamsAdapter
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.restart_test_helpers import make_restart_runner
from utils import atomic_json_write


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError, ConnectionError])
async def test_restart_notice_never_replays_teams_acceptance_with_lost_response(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    adapter = TeamsAdapter(PlatformConfig(enabled=True))
    adapter._running = True
    accepted = []

    async def sdk_send(chat_id, content):
        accepted.append((chat_id, content))
        if len(accepted) == 1:
            raise failure("response lost after Teams accepted the message")
        return SimpleNamespace(id="duplicate")

    adapter._app = SimpleNamespace(send=sdk_send)
    runner, _ = make_restart_runner(adapter)
    runner.adapters = {adapter.platform: adapter}
    runner.config.platforms = {adapter.platform: adapter.config}
    path = tmp_path / ".restart_notify.json"
    atomic_json_write(path, {"platform": "teams", "chat_id": "42", "request_id": "boot"})

    assert await runner._send_restart_notification() is None
    assert len(accepted) == 1
    assert not path.exists()
    assert await runner._send_restart_notification() is None
    fresh, _ = make_restart_runner(adapter)
    assert await fresh._send_restart_notification() is None
    assert len(accepted) == 1


@pytest.mark.asyncio
async def test_restart_notice_retries_real_telegram_pre_send_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._running = True
    adapter._rich_send_disabled = True
    adapter._wait_for_reconnection = AsyncMock(return_value=False)
    bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)))
    adapter.send_typing = AsyncMock()
    runner, _ = make_restart_runner(adapter)
    path = tmp_path / ".restart_notify.json"
    atomic_json_write(path, {"platform": "telegram", "chat_id": "42", "request_id": "boot"})
    waits = []

    async def recover(delay):
        # The real send() refused before reaching any provider; the marker is still owed.
        assert path.exists()
        bot.send_message.assert_not_awaited()
        adapter._wait_for_reconnection.assert_awaited_once()
        waits.append(delay)
        adapter._bot = bot
        await asyncio.sleep(0)

    monkeypatch.setattr(notices, "asyncio", SimpleNamespace(**{
        name: recover if name == "sleep" else getattr(asyncio, name) for name in dir(asyncio)
    }))
    assert await runner._send_restart_notification() == ("telegram", "42", None)
    assert waits == [1.0]
    bot.send_message.assert_awaited_once()
    assert not path.exists()
