from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, danger=3, green=1, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.platforms.discord import DiscordAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_send_retries_without_reference_when_reply_target_is_system_message():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    ref_msg = SimpleNamespace(id=99)
    sent_msg = SimpleNamespace(id=1234)
    send_calls = []

    async def fake_send(*, content, reference=None):
        send_calls.append({"content": content, "reference": reference})
        if len(send_calls) == 1:
            raise RuntimeError(
                "400 Bad Request (error code: 50035): Invalid Form Body\n"
                "In message_reference: Cannot reply to a system message"
            )
        return sent_msg

    channel = SimpleNamespace(
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("555", "hello", reply_to="99")

    assert result.success is True
    assert result.message_id == "1234"
    assert channel.fetch_message.await_count == 1
    assert channel.send.await_count == 2
    assert send_calls[0]["reference"] is ref_msg
    assert send_calls[1]["reference"] is None


@pytest.mark.asyncio
async def test_edit_message_updates_existing_message():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    message = SimpleNamespace(edit=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(),
    )

    result = await adapter.edit_message("555", "99", "updated")

    assert result.success is True
    assert result.message_id == "99"
    channel.fetch_message.assert_awaited_once_with(99)
    message.edit.assert_awaited_once_with(content="updated")


@pytest.mark.asyncio
async def test_edit_message_returns_not_connected_when_client_missing():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = None

    result = await adapter.edit_message("555", "99", "updated")

    assert result.success is False
    assert result.error == "Not connected"


@pytest.mark.asyncio
async def test_edit_message_returns_channel_error_for_missing_channel():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=None),
        fetch_channel=AsyncMock(return_value=None),
    )

    result = await adapter.edit_message("555", "99", "updated")

    assert result.success is False
    assert result.error == "Channel 555 not found"


@pytest.mark.asyncio
async def test_delete_message_delegates_to_message_delete():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    message = SimpleNamespace(delete=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(),
    )

    result = await adapter.delete_message("555", "99")

    assert result.success is True
    assert result.message_id == "99"
    message.delete.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_add_and_remove_reaction_delegate_to_message_methods():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    message = SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=999),
    )

    add_result = await adapter.add_reaction("555", "99", "🔥")
    remove_result = await adapter.remove_reaction("555", "99", "🔥")

    assert add_result.success is True
    assert remove_result.success is True
    message.add_reaction.assert_awaited_once_with("🔥")
    message.remove_reaction.assert_awaited_once_with("🔥", adapter._client.user)


def test_apply_runtime_policy_overrides_refreshes_cached_policy():
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={
                "allow_bots": "none",
                "free_response_channels": ["10"],
                "require_mention": True,
                "auto_thread": True,
            },
        )
    )

    original = adapter._get_discord_policy()
    updated = adapter.apply_runtime_policy_overrides(
        {
            "allow_bots": "mentions",
            "free_response_channels": ["77", "88"],
            "require_mention": False,
            "auto_thread": False,
        }
    )

    assert original.bot_filter_policy == "none"
    assert updated.bot_filter_policy == "mentions"
    assert updated.free_response_channels == {"77", "88"}
    assert updated.require_mention is False
    assert updated.auto_thread is False


class _FakeHistoryIterator:
    def __init__(self, messages):
        self._messages = list(messages)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._messages):
            raise StopAsyncIteration
        message = self._messages[self._index]
        self._index += 1
        return message


class _FakeHistoryChannel:
    def __init__(self, messages, readable=True):
        self.id = 123
        self.name = "general"
        self.guild = SimpleNamespace(me=SimpleNamespace(id=999), name="Hermes")
        self._messages = list(messages)
        self.permissions_for = MagicMock(
            return_value=SimpleNamespace(
                view_channel=readable,
                read_message_history=readable,
                send_messages=True,
                attach_files=True,
                embed_links=True,
                add_reactions=True,
                manage_threads=False,
                create_public_threads=False,
                create_private_threads=False,
            )
        )

    def history(self, *, limit, before=None, after=None):
        filtered = list(self._messages)
        if before is not None:
            filtered = [message for message in filtered if int(message.id) < int(before.id)]
        if after is not None:
            filtered = [message for message in filtered if int(message.id) > int(after.id)]
        return _FakeHistoryIterator(filtered[:limit])


def _history_message(
    message_id,
    *,
    content,
    author_id="42",
    author_name="Jezza",
    is_bot=False,
    timestamp=None,
):
    if timestamp is None:
        timestamp = datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        id=message_id,
        author=SimpleNamespace(id=author_id, name=author_name, display_name=author_name, bot=is_bot),
        content=content,
        created_at=timestamp,
        attachments=[],
        reference=None,
    )


@pytest.mark.asyncio
async def test_fetch_channel_history_returns_empty_for_missing_channel():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=None),
        fetch_channel=AsyncMock(return_value=None),
        user=SimpleNamespace(id=999),
    )

    result = await adapter.fetch_channel_history("123")

    assert result == []


@pytest.mark.asyncio
async def test_search_channel_history_serializes_messages():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(
            return_value=_FakeHistoryChannel(
                [
                    _history_message(10, content="Hermes status update"),
                    _history_message(9, content="unrelated"),
                ]
            )
        ),
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=999),
    )

    result = await adapter.search_channel_history("123", "hermes", limit=1)

    assert result == [
        {
            "id": "10",
            "author_id": "42",
            "author_name": "Jezza",
            "content": "Hermes status update",
            "timestamp": "2026-03-18T12:00:00+00:00",
            "is_bot": False,
            "attachments": [],
            "reply_to": None,
        }
    ]


@pytest.mark.asyncio
async def test_get_channel_permissions_serializes_dm_thread_flags_as_false():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    dm_channel = sys.modules["discord"].DMChannel()
    dm_channel.id = 321
    dm_channel.name = None
    dm_channel.guild = None
    dm_channel.recipient = SimpleNamespace(name="Alan")

    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=dm_channel),
        fetch_channel=AsyncMock(return_value=None),
        user=SimpleNamespace(id=999),
    )

    result = await adapter.get_channel_permissions("321")

    assert result == {
        "channel_id": "321",
        "channel_name": "Alan",
        "can_read": True,
        "can_send": True,
        "can_read_history": True,
        "can_attach_files": True,
        "can_embed_links": True,
        "can_add_reactions": True,
        "can_manage_threads": False,
        "can_create_threads": False,
    }


@pytest.mark.asyncio
async def test_get_accessible_channels_includes_normalized_target_metadata():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    perms = SimpleNamespace(
        view_channel=True,
        read_messages=True,
        send_messages=True,
        read_message_history=True,
        attach_files=True,
        embed_links=True,
        add_reactions=True,
        manage_threads=False,
        create_public_threads=True,
        create_private_threads=False,
    )
    guild = SimpleNamespace(id=777, name="Hermes")
    channel = SimpleNamespace(
        id=123,
        name="general",
        guild=guild,
        permissions_for=MagicMock(return_value=perms),
    )
    guild.me = SimpleNamespace(id=999)
    guild.text_channels = [channel]
    adapter._client = SimpleNamespace(guilds=[guild], user=SimpleNamespace(id=999))

    result = await adapter.get_accessible_channels()

    assert result == [
        {
            "channel_id": "123",
            "channel_name": "general",
            "guild_id": "777",
            "guild_name": "Hermes",
            "channel_kind": "channel",
            "qualified_name": "Hermes/general",
            "mention": "<#123>",
            "can_read": True,
            "can_send": True,
            "can_read_history": True,
            "can_attach_files": True,
            "can_embed_links": True,
            "can_add_reactions": True,
            "can_manage_threads": False,
            "can_create_threads": True,
        }
    ]


@pytest.mark.asyncio
async def test_list_threads_returns_serialized_threads():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    thread = sys.modules["discord"].Thread()
    thread.id = 77
    thread.name = "planning"
    thread.parent = SimpleNamespace(id=5, name="general")
    thread.guild = SimpleNamespace(id=1, name="Hermes")
    thread.archived = False
    thread.locked = False
    thread.message_count = 4
    thread.member_count = 2
    channel = SimpleNamespace(threads=[thread])
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(),
    )

    result = await adapter.list_threads("123")

    assert result == [
        {
            "id": "77",
            "name": "planning",
            "parent_id": "5",
            "parent_name": "general",
            "guild_id": "1",
            "guild_name": "Hermes",
            "archived": False,
            "locked": False,
            "message_count": 4,
            "member_count": 2,
        }
    ]


@pytest.mark.asyncio
async def test_reply_in_thread_sends_to_valid_thread():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    thread = sys.modules["discord"].Thread()
    thread.parent = SimpleNamespace(id=5)
    thread.fetch_message = AsyncMock(return_value=SimpleNamespace(id=99))
    sent = SimpleNamespace(id=101)
    thread.send = AsyncMock(return_value=sent)
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=thread),
        fetch_channel=AsyncMock(),
    )

    result = await adapter.reply_in_thread("123", "hello", reply_to="99")

    assert result.success is True
    assert result.message_id == "101"
    thread.fetch_message.assert_awaited_once_with(99)
    thread.send.assert_awaited_once_with(content="hello", reference=thread.fetch_message.return_value)


@pytest.mark.asyncio
async def test_reply_in_thread_rejects_non_thread_channel():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=SimpleNamespace(parent=None)),
        fetch_channel=AsyncMock(),
    )

    result = await adapter.reply_in_thread("123", "hello")

    assert result.success is False
    assert result.error == "Channel 123 is not a thread"


@pytest.mark.asyncio
async def test_list_pins_returns_serialized_messages():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    author = SimpleNamespace(id=42, name="Jezza", display_name="Jezza", bot=False)
    pinned = SimpleNamespace(
        id=7,
        author=author,
        content="important",
        created_at=datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc),
        attachments=[],
        reference=None,
    )
    channel = SimpleNamespace(pins=MagicMock(return_value=_FakeHistoryIterator([pinned])))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(),
    )

    result = await adapter.list_pins("123")

    assert result == [
        {
            "id": "7",
            "author_id": "42",
            "author_name": "Jezza",
            "content": "important",
            "timestamp": "2026-03-18T12:00:00+00:00",
            "is_bot": False,
            "attachments": [],
            "reply_to": None,
        }
    ]


@pytest.mark.asyncio
async def test_pin_and_unpin_message_delegate_to_message_methods():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    message = SimpleNamespace(pin=AsyncMock(), unpin=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(),
    )

    pin_result = await adapter.pin_message("123", "456", reason="keep")
    unpin_result = await adapter.unpin_message("123", "456", reason="drop")

    assert pin_result.success is True
    assert unpin_result.success is True
    message.pin.assert_awaited_once_with(reason="keep")
    message.unpin.assert_awaited_once_with(reason="drop")


@pytest.mark.asyncio
async def test_list_reactions_returns_serialized_summaries():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    reaction = SimpleNamespace(
        emoji="🔥",
        count=2,
        users=MagicMock(
            return_value=_FakeHistoryIterator([SimpleNamespace(id=1, username="alan", discriminator="1234")])
        ),
    )
    message = SimpleNamespace(reactions=[reaction])
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(),
    )

    result = await adapter.list_reactions("123", "456", limit=3)

    assert result == [
        {
            "emoji": {"id": None, "name": "🔥", "raw": "🔥"},
            "count": 2,
            "users": [{"id": "1", "username": "alan", "tag": "alan#1234"}],
        }
    ]
    reaction.users.assert_called_once_with(limit=3)
