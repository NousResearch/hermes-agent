"""Tests for /voice realtime gateway command handling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


def _make_runner(tmp_path):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._VOICE_MODE_PATH = tmp_path / "gateway_voice_mode.json"
    runner._session_db = None
    runner.session_store = MagicMock()
    runner._is_user_authorized = lambda source: True
    return runner


def _make_discord_event(text="/voice realtime", chat_id="123", guild_id=111):
    source = SessionSource(
        chat_id=chat_id,
        user_id="42",
        user_name="Hákon",
        platform=Platform.DISCORD,
        chat_type="group",
    )
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source)
    event.message_id = "msg42"
    event.raw_message = SimpleNamespace(guild_id=guild_id, guild=None)
    return event


@pytest.mark.asyncio
async def test_voice_realtime_alias_dispatches_to_realtime_join(tmp_path):
    runner = _make_runner(tmp_path)
    event = _make_discord_event("/voice live")
    runner._handle_voice_realtime_join = AsyncMock(return_value="joined realtime")

    result = await runner._handle_voice_command(event)

    assert result == "joined realtime"
    runner._handle_voice_realtime_join.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_voice_channel_realtime_alias_dispatches_to_realtime_join(tmp_path):
    runner = _make_runner(tmp_path)
    event = _make_discord_event("/voice channel realtime")
    runner._handle_voice_realtime_join = AsyncMock(return_value="joined realtime")

    result = await runner._handle_voice_command(event)

    assert result == "joined realtime"


@pytest.mark.asyncio
async def test_voice_realtime_join_success_builds_context(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path)
    event = _make_discord_event()
    channel = MagicMock()
    channel.name = "General"
    adapter = AsyncMock()
    adapter.get_user_voice_channel = AsyncMock(return_value=channel)
    adapter.join_realtime_voice_channel = AsyncMock(return_value=True)
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    runner.adapters[Platform.DISCORD] = adapter
    monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "test-key")

    with patch("hermes_cli.config.load_config", return_value={
        "voice": {
            "realtime": {
                "enabled": True,
                "model": "gpt-realtime-2",
                "voice": "marin",
                "tools_enabled": False,
            }
        }
    }):
        result = await runner._handle_voice_realtime_join(event)

    assert "realtime voice" in result.lower()
    adapter.join_realtime_voice_channel.assert_awaited_once()
    realtime_context = adapter.join_realtime_voice_channel.await_args.args[1]
    assert realtime_context.config.api_key == "test-key"
    assert realtime_context.config.model == "gpt-realtime-2"
    assert realtime_context.config.voice == "marin"
    assert realtime_context.config.tools_enabled is False
    assert adapter._voice_text_channels[111] == 123
    assert runner._voice_mode["discord:123"] == "realtime"


@pytest.mark.asyncio
async def test_voice_realtime_requires_openai_key(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path)
    event = _make_discord_event()
    channel = MagicMock()
    adapter = AsyncMock()
    adapter.get_user_voice_channel = AsyncMock(return_value=channel)
    runner.adapters[Platform.DISCORD] = adapter
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = await runner._handle_voice_realtime_join(event)

    assert "VOICE_TOOLS_OPENAI_KEY" in result
    adapter.get_user_voice_channel.assert_awaited_once()


@pytest.mark.asyncio
async def test_voice_status_reports_realtime_active(tmp_path):
    runner = _make_runner(tmp_path)
    event = _make_discord_event("/voice status")
    adapter = MagicMock()
    adapter.is_realtime_voice_active.return_value = True
    adapter.get_voice_channel_info.return_value = {
        "channel_name": "General",
        "member_count": 1,
        "members": [],
    }
    runner.adapters[Platform.DISCORD] = adapter
    runner._voice_mode["discord:123"] = "realtime"

    result = await runner._handle_voice_command(event)

    assert "realtime" in result.lower()
