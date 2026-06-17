from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin


class _AsyncMembers:
    def __init__(self, members):
        self._members = list(members)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._members):
            raise StopAsyncIteration
        member = self._members[self._index]
        self._index += 1
        return member


class _Guild:
    id = 123

    def __init__(self, members, *, fetch_error=None, cached=None):
        self._members = members
        self._fetch_error = fetch_error
        self.members = cached or []

    def fetch_members(self, limit=None):
        assert limit is None
        if self._fetch_error is not None:
            raise self._fetch_error
        return _AsyncMembers(self._members)


class _Client:
    def __init__(self, guild):
        self.guild = guild

    def get_guild(self, guild_id):
        return self.guild if guild_id == self.guild.id else None


class _Runner(GatewaySlashCommandsMixin):
    adapters: dict


def _event(guild_id="123", raw_message=None):
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="456",
        chat_type="group",
        user_id="789",
        guild_id=guild_id,
    )
    return MessageEvent(
        text="/server-users",
        message_type=MessageType.COMMAND,
        source=source,
        raw_message=raw_message,
    )


@pytest.mark.asyncio
async def test_server_users_lists_fetched_members_as_comma_separated_names():
    guild = _Guild([
        SimpleNamespace(display_name="Charlie"),
        SimpleNamespace(display_name="Alice, Admin"),
        SimpleNamespace(display_name="bob"),
    ])
    runner = _Runner()
    runner.adapters = {Platform.DISCORD: SimpleNamespace(_client=_Client(guild))}

    result = await runner._handle_server_users_command(_event())

    assert result == "Alice  Admin, bob, Charlie"


@pytest.mark.asyncio
async def test_server_users_falls_back_to_cached_members_when_fetch_fails():
    guild = _Guild(
        [],
        fetch_error=RuntimeError("missing intent"),
        cached=[SimpleNamespace(display_name="Cached User")],
    )
    runner = _Runner()
    runner.adapters = {Platform.DISCORD: SimpleNamespace(_client=_Client(guild))}

    result = await runner._handle_server_users_command(_event())

    assert result == "⚠️ Full member fetch failed; using the cached member list.\nCached User"


@pytest.mark.asyncio
async def test_server_users_rejects_non_discord_sources():
    runner = GatewaySlashCommandsMixin()
    event = MessageEvent(
        text="/server-users",
        message_type=MessageType.COMMAND,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1"),
    )

    result = await runner._handle_server_users_command(event)

    assert "only available from a Discord server" in result
