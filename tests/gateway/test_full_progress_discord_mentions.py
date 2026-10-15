"""Full progress must not ping Discord users named in tool arguments or reply anchors."""

import asyncio
import queue
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

if "discord" not in sys.modules:
    discord_mock = MagicMock()
    discord_mock.DMChannel = type("DMChannel", (), {})
    discord_mock.Thread = type("Thread", (), {})
    discord_mock.ForumChannel = type("ForumChannel", (), {})
    sys.modules["discord"] = discord_mock
    sys.modules["discord.ext"] = MagicMock()
    sys.modules["discord.ext.commands"] = MagicMock()

from gateway.config import Platform, PlatformConfig
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from plugins.platforms.discord import adapter as discord_adapter_module
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.asyncio
async def test_full_progress_native_discord_send_and_edit_never_ping(monkeypatch):
    monkeypatch.setattr(discord_adapter_module.discord.AllowedMentions, "none", lambda: SimpleNamespace(
        everyone=False, roles=False, users=False, replied_user=False,
    ))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="synthetic"))
    wire = []
    message = SimpleNamespace(id=901, edit=AsyncMock())

    async def send(**kwargs):
        wire.append(kwargs)
        return message

    channel = SimpleNamespace(id=123, send=AsyncMock(side_effect=send),
                              get_partial_message=lambda _id: message)
    adapter._client = SimpleNamespace(get_channel=lambda _id: channel,
                                      fetch_channel=AsyncMock(return_value=channel))
    ctx = TurnContext(source=SimpleNamespace(platform=Platform.DISCORD, chat_id="123"),
                      _run_still_current=lambda: True, progress_mode="full",
                      progress_grouping="accumulate", tool_progress_enabled=True,
                      progress_queue=queue.Queue(), _progress_reply_to="456",
                      _progress_metadata={})
    turn = TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda _source: adapter), ctx)
    st = turn._progress_edit_state(adapter)
    original = "printf 'first line'\nprintf '<@123456789012345678>'"
    turn.progress_callback("tool.started", "terminal", "short", {"command": original})
    first = ctx.progress_queue.get_nowait()
    assert first[0] == "__full__"
    await turn._send_full_progress_entry(st, first[1])
    turn.progress_callback("tool.started", "execute_code", "short", {"code": "print('safe')"})
    second = ctx.progress_queue.get_nowait()
    await turn._send_full_progress_entry(st, second[1])

    assert len(wire) == 1 and wire[0]["reference"] is not None
    assert "<@123456789012345678>" in wire[0]["content"]
    assert message.edit.await_count == 1
    assert "<@123456789012345678>" in message.edit.await_args.kwargs["content"]
    for payload in (wire[0], message.edit.await_args.kwargs):
        allowed = payload["allowed_mentions"]
        assert allowed.everyone is False and allowed.roles is False
        assert allowed.users is False and allowed.replied_user is False

    # The ordinary adapter path retains configured mentions; no global mute.
    await adapter.send("123", "<@123456789012345678>")
    assert "allowed_mentions" not in wire[-1]
