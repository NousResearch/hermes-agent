"""Discord adapter contract for opt-in ambient room observation."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageDisposition
import plugins.platforms.discord.adapter as discord_platform
from plugins.platforms.discord.adapter import DiscordAdapter


class FakeDMChannel:
    def __init__(self, channel_id: int = 1):
        self.id = channel_id
        self.name = "dm"


class FakeTextChannel:
    def __init__(self, channel_id: int = 123, name: str = "bots"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(id=456, name="Test Guild")
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield

        return _iter()


class FakeThread(FakeTextChannel):
    def __init__(self, channel_id: int = 789, parent=None):
        super().__init__(channel_id=channel_id, name="thread")
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", self.guild)


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in (
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_AUTO_THREAD",
        "DISCORD_HISTORY_BACKFILL",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    config.extra.update(
        {
            "require_mention": True,
            "free_response_channels": [],
            "history_backfill": False,
        }
    )
    result = DiscordAdapter(config)
    setattr(result, "_client", SimpleNamespace(user=SimpleNamespace(id=999, bot=True, name="Hermes")))
    setattr(result, "_text_batch_delay_seconds", 0)
    setattr(result, "handle_message", AsyncMock())
    setattr(result, "_auto_create_thread", AsyncMock())
    setattr(result, "_cache_discord_image", AsyncMock())
    setattr(result, "_cache_discord_audio", AsyncMock())
    setattr(result, "_cache_discord_document", AsyncMock())
    return result


def make_message(
    *,
    adapter,
    channel=None,
    content: str = "ambient hello",
    mentions=None,
    role_mentions=None,
    author=None,
    reference=None,
    attachments=None,
    msg_type=None,
):
    channel = channel or FakeTextChannel()
    author = author or SimpleNamespace(
        id=42,
        display_name="Alice",
        name="Alice",
        bot=False,
    )
    return SimpleNamespace(
        id=12345,
        content=content,
        clean_content=content,
        mentions=list(mentions or []),
        role_mentions=list(role_mentions or []),
        attachments=list(attachments or []),
        message_snapshots=[],
        reference=reference,
        created_at=datetime(2026, 7, 10, 15, 0, tzinfo=timezone.utc),
        channel=channel,
        guild=getattr(channel, "guild", None),
        author=author,
        type=msg_type if msg_type is not None else discord_platform.discord.MessageType.default,
    )


def enable_observation(adapter, *channel_ids: str) -> None:
    adapter.config.extra["observe_unmentioned_group_messages"] = True
    adapter.config.extra["observed_channels"] = list(channel_ids)


def test_observation_config_defaults_off_and_rejects_wildcards_or_names(adapter):
    assert adapter._discord_observation_enabled() is False
    assert adapter._discord_observed_channel_ids() == set()

    adapter.config.extra["observe_unmentioned_group_messages"] = True
    adapter.config.extra["observed_channels"] = ["*", "bots", "#bots", "123"]

    assert adapter._discord_observation_enabled() is True
    assert adapter._discord_observed_channel_ids() == {"123"}


@pytest.mark.asyncio
async def test_durable_observation_suppresses_legacy_history_backfill(adapter):
    enable_observation(adapter, "123")
    adapter.config.extra["history_backfill"] = True
    adapter._fetch_channel_context = AsyncMock(
        return_value="[Recent channel messages]\n[Alice] duplicate context"
    )
    message = make_message(
        adapter=adapter,
        content="<@999> current request",
        mentions=[adapter._client.user],
    )

    await adapter._handle_message(message)

    adapter._fetch_channel_context.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "current request"


@pytest.mark.asyncio
async def test_legacy_history_backfill_remains_active_outside_observed_channel(adapter):
    enable_observation(adapter, "999")
    adapter.config.extra["history_backfill"] = True
    adapter._fetch_channel_context = AsyncMock(
        return_value="[Recent channel messages]\n[Alice] legacy context"
    )
    message = make_message(
        adapter=adapter,
        content="<@999> current request",
        mentions=[adapter._client.user],
    )

    await adapter._handle_message(message)

    adapter._fetch_channel_context.assert_awaited_once()
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_existing_channel_allow_and_ignore_gates_win_before_observation(
    adapter,
    monkeypatch,
):
    enable_observation(adapter, "123")
    message = make_message(adapter=adapter)

    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "123")
    assert await adapter._maybe_observe_room_message(message) is False

    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "999")
    assert await adapter._maybe_observe_room_message(message) is False

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_free_response_channel_keeps_dispatch_precedence(adapter):
    enable_observation(adapter, "123")
    adapter.config.extra["free_response_channels"] = ["123"]
    message = make_message(adapter=adapter)

    observed = await adapter._maybe_observe_room_message(message)
    if not observed:
        await adapter._handle_message(message)

    assert observed is False
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.disposition is MessageDisposition.DISPATCH


@pytest.mark.asyncio
async def test_ambient_text_in_unconfigured_room_remains_dropped(adapter):
    enable_observation(adapter, "999999")
    message = make_message(adapter=adapter, channel=FakeTextChannel(channel_id=123))

    observed = await adapter._maybe_observe_room_message(message)
    if not observed:
        await adapter._handle_message(message)

    assert observed is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_ambient_text_in_exact_observed_room_emits_one_observe_event(adapter):
    enable_observation(adapter, "123")
    message = make_message(adapter=adapter)

    observed = await adapter._maybe_observe_room_message(message, role_authorized=True)

    assert observed is True
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.disposition is MessageDisposition.OBSERVE
    assert event.text == "ambient hello"
    assert event.source.chat_id == "123"
    assert event.source.scope_id == "456"
    assert event.source.user_id == "42"
    assert event.source.user_name == "Alice"
    assert event.source.role_authorized is True
    assert event.addressing.direct_mention is False
    assert event.addressing.reply_to_self is False


@pytest.mark.asyncio
async def test_explicit_self_mention_dispatches_once_and_never_observes(adapter):
    enable_observation(adapter, "123")
    message = make_message(
        adapter=adapter,
        content="<@999> answer this",
        mentions=[adapter._client.user],
    )

    observed = await adapter._maybe_observe_room_message(message)
    if not observed:
        await adapter._handle_message(message)

    assert observed is False
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.disposition is MessageDisposition.DISPATCH
    assert event.addressing.direct_mention is True


@pytest.mark.asyncio
async def test_self_managed_role_mention_dispatches_once_and_never_observes(adapter):
    enable_observation(adapter, "123")
    self_role = SimpleNamespace(
        id=777,
        tags=SimpleNamespace(bot_id=adapter._client.user.id),
        managed=True,
    )
    message = make_message(
        adapter=adapter,
        content="<@&777> what did ciacon and Destruxus contribute?",
        role_mentions=[self_role],
    )

    observed = await adapter._maybe_observe_room_message(message)
    if not observed:
        await adapter._handle_message(message)

    assert observed is False
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.disposition is MessageDisposition.DISPATCH
    assert event.addressing.direct_mention is True
    assert event.text == "what did ciacon and Destruxus contribute?"


@pytest.mark.asyncio
async def test_raw_self_managed_role_resolves_from_guild_cache(adapter):
    enable_observation(adapter, "123")
    self_role = SimpleNamespace(
        id=777,
        tags=SimpleNamespace(bot_id=adapter._client.user.id),
        managed=True,
    )
    channel = FakeTextChannel()
    channel.guild.get_role = lambda role_id: self_role if role_id == 777 else None
    message = make_message(
        adapter=adapter,
        channel=channel,
        content="<@&777> cached role mention",
    )

    observed = await adapter._maybe_observe_room_message(message)
    if not observed:
        await adapter._handle_message(message)

    assert observed is False
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.disposition is MessageDisposition.DISPATCH
    assert event.addressing.direct_mention is True
    assert event.text == "cached role mention"


@pytest.mark.asyncio
async def test_unrelated_managed_role_mention_remains_ambient_observation(adapter):
    enable_observation(adapter, "123")
    other_role = SimpleNamespace(
        id=888,
        tags=SimpleNamespace(bot_id=123456),
        managed=True,
    )
    message = make_message(
        adapter=adapter,
        content="<@&888> unrelated role question",
        role_mentions=[other_role],
    )

    observed = await adapter._maybe_observe_room_message(message)

    assert observed is True
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.disposition is MessageDisposition.OBSERVE
    assert event.addressing.direct_mention is False
    assert event.text == "<@&888> unrelated role question"


@pytest.mark.asyncio
async def test_reply_to_self_dispatches_and_never_observes(adapter):
    enable_observation(adapter, "123")
    reference = SimpleNamespace(
        message_id=41,
        resolved=SimpleNamespace(author=adapter._client.user, content="Prior answer"),
    )
    message = make_message(
        adapter=adapter,
        content="follow up",
        mentions=[adapter._client.user],
        msg_type=discord_platform.discord.MessageType.reply,
        reference=reference,
    )

    observed = await adapter._maybe_observe_room_message(message)
    if not observed:
        await adapter._handle_message(message)

    assert observed is False
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.disposition is MessageDisposition.DISPATCH
    assert event.addressing.reply_to_self is True


def test_reply_to_self_bypasses_other_bot_multi_agent_filter(adapter):
    other_bot = SimpleNamespace(id=1234, bot=True, name="Other Bot")
    ordinary_message = make_message(adapter=adapter, mentions=[other_bot])
    assert adapter._discord_multi_agent_message_allowed(ordinary_message) is False

    reference = SimpleNamespace(
        message_id=41,
        resolved=SimpleNamespace(author=adapter._client.user, content="Prior answer"),
    )
    reply = make_message(
        adapter=adapter,
        content="follow up",
        mentions=[other_bot],
        msg_type=discord_platform.discord.MessageType.reply,
        reference=reference,
    )
    assert adapter._discord_multi_agent_message_allowed(reply) is True


@pytest.mark.asyncio
async def test_human_text_mentioning_another_bot_observes_without_dispatch(adapter):
    enable_observation(adapter, "123")
    other_bot = SimpleNamespace(id=555, display_name="OtherBot", name="OtherBot", bot=True)
    message = make_message(
        adapter=adapter,
        content="<@555> your turn",
        mentions=[other_bot],
    )

    observed = await adapter._maybe_observe_room_message(message)

    assert observed is True
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.disposition is MessageDisposition.OBSERVE
    assert event.addressing.mentions_other_bots is True
    assert event.addressing.direct_mention is False


@pytest.mark.asyncio
async def test_other_bot_messages_remain_unobserved_by_default(adapter):
    enable_observation(adapter, "123")
    other_bot = SimpleNamespace(id=555, display_name="OtherBot", name="OtherBot", bot=True)
    message = make_message(adapter=adapter, author=other_bot)

    observed = await adapter._maybe_observe_room_message(message)

    assert observed is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dm_never_enters_room_observation(adapter):
    enable_observation(adapter, "1")
    message = make_message(adapter=adapter, channel=FakeDMChannel(channel_id=1))

    observed = await adapter._maybe_observe_room_message(message)

    assert observed is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_passive_attachment_uses_metadata_without_read_or_cache(adapter):
    enable_observation(adapter, "123")
    attachment = SimpleNamespace(
        filename="evidence.png",
        content_type="image/png",
        size=2048,
        url="https://cdn.discord.invalid/evidence.png",
        read=AsyncMock(return_value=b"not-used"),
    )
    message = make_message(
        adapter=adapter,
        content="",
        attachments=[attachment],
    )

    observed = await adapter._maybe_observe_room_message(message)

    assert observed is True
    event = adapter.handle_message.await_args.args[0]
    assert event.disposition is MessageDisposition.OBSERVE
    assert event.text == "[Attachment: evidence.png; image/png; 2048 bytes]"
    assert event.media_urls == []
    assert event.media_types == []
    attachment.read.assert_not_awaited()
    adapter._cache_discord_image.assert_not_awaited()
    adapter._cache_discord_audio.assert_not_awaited()
    adapter._cache_discord_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_observation_path_never_attempts_auto_thread(adapter):
    enable_observation(adapter, "123")
    message = make_message(adapter=adapter)

    observed = await adapter._maybe_observe_room_message(message)

    assert observed is True
    adapter._auto_create_thread.assert_not_awaited()
