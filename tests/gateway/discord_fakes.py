"""Reusable lightweight Discord fakes for gateway tests.

These classes intentionally model only the small surface area used by tests.
They do not import discord.py and never require a real Discord token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from types import SimpleNamespace
from typing import Any, Iterable


_ids = count(1)


def _next_id() -> int:
    return next(_ids)


def _id_matches(actual: Any, expected: Any) -> bool:
    return str(actual) == str(expected)


@dataclass
class FakeRole:
    id: Any = field(default_factory=_next_id)
    name: str = "role"


@dataclass
class FakeUser:
    id: Any = field(default_factory=_next_id)
    name: str = "user"
    display_name: str | None = None
    bot: bool = False
    mention: str | None = None

    def __post_init__(self) -> None:
        if self.display_name is None:
            self.display_name = self.name
        if self.mention is None:
            self.mention = f"<@{self.id}>"


@dataclass
class FakeMember(FakeUser):
    roles: list[FakeRole] = field(default_factory=list)
    guild: Any = None


class FakeMessage:
    def __init__(
        self,
        content: str = "",
        *,
        id: Any | None = None,
        author: FakeUser | None = None,
        user: FakeUser | None = None,
        channel: Any = None,
        guild: Any = None,
        reference: Any = None,
        mentions: Iterable[Any] | None = None,
        attachments: Iterable[Any] | None = None,
        type: Any = None,
        **attrs: Any,
    ) -> None:
        self.id = _next_id() if id is None else id
        self.content = content
        self.author = author or user or FakeMember()
        self.user = user or self.author
        self.channel = channel
        self.guild = guild if guild is not None else getattr(channel, "guild", None)
        self.reference = reference
        self.mentions = list(mentions or [])
        self.attachments = list(attachments or [])
        self.type = type or "default"
        self.edits: list[dict[str, Any]] = []
        self.deleted = False
        for key, value in attrs.items():
            setattr(self, key, value)

    async def edit(self, **kwargs: Any) -> "FakeMessage":
        self.edits.append(dict(kwargs))
        if "content" in kwargs:
            self.content = kwargs["content"]
        for key, value in kwargs.items():
            if key != "content":
                setattr(self, key, value)
        return self

    async def delete(self) -> None:
        self.deleted = True

    async def reply(self, content: str | None = None, **kwargs: Any) -> "FakeMessage":
        if self.channel is None:
            raise RuntimeError("FakeMessage.reply requires a channel")
        return await self.channel.send(content=content, reference=self, **kwargs)


class FakeChannel:
    def __init__(
        self,
        id: Any | None = None,
        name: str = "channel",
        *,
        guild: Any = None,
        is_dm: bool = False,
        **attrs: Any,
    ) -> None:
        self.id = _next_id() if id is None else id
        self.name = name
        self.guild = guild
        self.is_dm = is_dm
        self.sent_messages: list[FakeMessage] = []
        self.edits: list[dict[str, Any]] = []
        for key, value in attrs.items():
            setattr(self, key, value)

    @property
    def mention(self) -> str:
        return f"<#{self.id}>"

    async def send(self, content: str | None = None, **kwargs: Any) -> FakeMessage:
        message = FakeMessage(
            content or "",
            channel=self,
            guild=self.guild,
            reference=kwargs.get("reference"),
        )
        for key, value in kwargs.items():
            if key != "reference":
                setattr(message, key, value)
        self.sent_messages.append(message)
        return message

    async def edit(self, **kwargs: Any) -> "FakeChannel":
        self.edits.append(dict(kwargs))
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    async def fetch_message(self, message_id: Any) -> FakeMessage:
        for message in self.sent_messages:
            if _id_matches(message.id, message_id):
                return message
        raise LookupError(f"message not found: {message_id}")


class FakeThread(FakeChannel):
    def __init__(self, id: Any | None = None, name: str = "thread", *, parent: Any = None, **attrs: Any) -> None:
        super().__init__(id=id, name=name, guild=getattr(parent, "guild", None), **attrs)
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)


class FakeForumChannel(FakeChannel):
    def __init__(self, id: Any | None = None, name: str = "forum", *, guild: Any = None, **attrs: Any) -> None:
        super().__init__(id=id, name=name, guild=guild, **attrs)
        self.created_threads: list[FakeThread] = []

    async def create_thread(self, name: str, content: str | None = None, **kwargs: Any) -> Any:
        thread = FakeThread(name=name, parent=self)
        for key, value in kwargs.items():
            setattr(thread, key, value)
        starter = await thread.send(content=content or "")
        self.created_threads.append(thread)
        return SimpleNamespace(thread=thread, message=starter)


class FakeGuild:
    def __init__(
        self,
        id: Any | None = None,
        name: str = "guild",
        *,
        roles: Iterable[FakeRole] | None = None,
        members: Iterable[FakeMember] | None = None,
        channels: Iterable[FakeChannel] | None = None,
        **attrs: Any,
    ) -> None:
        self.id = _next_id() if id is None else id
        self.name = name
        self.roles = list(roles or [])
        self.members = list(members or [])
        self.channels = list(channels or [])
        for member in self.members:
            member.guild = self
        for channel in self.channels:
            channel.guild = self
        for key, value in attrs.items():
            setattr(self, key, value)

    def get_role(self, role_id: Any) -> FakeRole | None:
        return next((role for role in self.roles if _id_matches(role.id, role_id)), None)

    def get_member(self, member_id: Any) -> FakeMember | None:
        return next((member for member in self.members if _id_matches(member.id, member_id)), None)

    def get_channel(self, channel_id: Any) -> FakeChannel | None:
        return next((channel for channel in self.channels if _id_matches(channel.id, channel_id)), None)


class FakeInteractionResponse:
    def __init__(self) -> None:
        self.deferred = False
        self.ephemeral = False
        self.sent_messages: list[FakeMessage] = []
        self.edits: list[dict[str, Any]] = []

    async def defer(self, *, ephemeral: bool = False, **kwargs: Any) -> None:
        self.deferred = True
        self.ephemeral = ephemeral

    async def send_message(self, content: str | None = None, *, ephemeral: bool = False, **kwargs: Any) -> FakeMessage:
        message = FakeMessage(content or "")
        message.ephemeral = ephemeral
        for key, value in kwargs.items():
            setattr(message, key, value)
        self.sent_messages.append(message)
        return message

    async def edit_message(self, **kwargs: Any) -> None:
        self.edits.append(dict(kwargs))


class FakeInteractionFollowup:
    def __init__(self) -> None:
        self.sent_messages: list[FakeMessage] = []

    async def send(self, content: str | None = None, *, ephemeral: bool = False, **kwargs: Any) -> FakeMessage:
        message = FakeMessage(content or "")
        message.ephemeral = ephemeral
        for key, value in kwargs.items():
            setattr(message, key, value)
        self.sent_messages.append(message)
        return message


class FakeInteraction:
    def __init__(
        self,
        *,
        id: Any | None = None,
        command_name: str | None = None,
        custom_id: str | None = None,
        user: FakeUser | None = None,
        channel: Any = None,
        guild: Any = None,
        data: dict[str, Any] | None = None,
        **attrs: Any,
    ) -> None:
        self.id = _next_id() if id is None else id
        self.user = user or FakeMember()
        self.channel = channel
        self.guild = guild if guild is not None else getattr(channel, "guild", None)
        self.command = SimpleNamespace(name=command_name) if command_name else None
        self.data = data or {}
        if command_name:
            self.data.setdefault("name", command_name)
        if custom_id:
            self.data.setdefault("custom_id", custom_id)
        self.response = FakeInteractionResponse()
        self.followup = FakeInteractionFollowup()
        for key, value in attrs.items():
            setattr(self, key, value)


class FakeClient:
    def __init__(
        self,
        *,
        user: FakeUser | None = None,
        channels: Iterable[FakeChannel] | None = None,
        guilds: Iterable[FakeGuild] | None = None,
    ) -> None:
        self.user = user or FakeMember(id="bot", name="bot", bot=True)
        self.channels = list(channels or [])
        self.guilds = list(guilds or [])

    def get_channel(self, channel_id: Any) -> FakeChannel | None:
        return next((channel for channel in self.channels if _id_matches(channel.id, channel_id)), None)

    async def fetch_channel(self, channel_id: Any) -> FakeChannel:
        channel = self.get_channel(channel_id)
        if channel is None:
            raise LookupError(f"channel not found: {channel_id}")
        return channel

    def get_guild(self, guild_id: Any) -> FakeGuild | None:
        return next((guild for guild in self.guilds if _id_matches(guild.id, guild_id)), None)
