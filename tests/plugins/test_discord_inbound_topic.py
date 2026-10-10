"""Discord inbound topics follow the actual channel or thread route."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["channel", "existing-thread", "new-thread"])
async def test_discord_topic_uses_the_routed_channel(monkeypatch, route):
    discord = pytest.importorskip("discord")
    from discord.ext import commands
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="test",
            extra={
                "require_mention": False,
                "auto_thread": route == "new-thread",
                "history_backfill": False,
            },
        )
    )
    adapter._client = commands.Bot(
        command_prefix="!", intents=discord.Intents.default()
    )
    await adapter._client._async_setup_hook()
    connection = adapter._client._connection
    connection.user = discord.ClientUser(
        state=connection,
        data={"id": "900", "username": "hermes", "discriminator": "0", "avatar": None},
    )
    guild = discord.Guild(
        data={"id": "999", "name": "test", "owner_id": "333"}, state=connection
    )
    connection._guilds[guild.id] = guild
    channel = discord.TextChannel(
        state=connection,
        guild=guild,
        data={
            "id": "555",
            "type": 0,
            "name": "parent",
            "topic": "Parent-channel instructions",
            "position": 0,
            "permission_overwrites": [],
        },
    )
    guild._channels[channel.id] = channel
    thread = discord.Thread(
        state=connection,
        guild=guild,
        data={
            "id": "777",
            "parent_id": "555",
            "owner_id": "333",
            "name": "conversation",
            "type": 11,
            "message_count": 1,
            "member_count": 1,
            "thread_metadata": {
                "archived": False,
                "auto_archive_duration": 1440,
                "archive_timestamp": "2026-10-01T00:00:00+00:00",
            },
        },
    )
    guild._threads[thread.id] = thread
    monkeypatch.setattr(adapter, "_auto_create_thread", AsyncMock(return_value=thread))
    monkeypatch.setattr(adapter._threads, "mark_async", AsyncMock())
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    message = discord.Message(
        state=connection,
        channel=thread if route == "existing-thread" else channel,
        data={
            "id": "101",
            "type": 0,
            "content": "authored input",
            "attachments": [],
            "embeds": [],
            "mentions": [],
            "mention_roles": [],
            "author": {
                "id": "333",
                "username": "sender",
                "discriminator": "0",
                "avatar": None,
            },
        },
    )
    try:
        dispatched = await adapter._handle_message(message)
        actual = [
            (
                event.source.chat_id,
                event.source.thread_id,
                event.source.parent_chat_id,
                event.source.chat_topic,
            )
            for (event,), _kwargs in adapter.handle_message.await_args_list
        ]
        expected = (
            ("555", None, None, channel.topic)
            if route == "channel"
            else ("777", "777", "555", None)
        )
        assert (dispatched, actual) == (True, [expected])
    finally:
        await adapter._client.close()
