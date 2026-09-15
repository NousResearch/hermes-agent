"""Discord: the adapter's mention facts reach plugins on ``MessageEvent.metadata``.

The adapter already decides whether the bot was explicitly mentioned
(``_self_is_explicitly_mentioned`` / ``_raw_mentioned_user_ids``) for its own admission gate,
then strips the ``<@bot>`` token before building the event. A ``pre_gateway_dispatch`` plugin
sees only the stripped text, so it cannot tell an addressed message from ambient chatter, nor
which users or bots were mentioned. The facts are now published on the event and survive the
inbound text-batch merge.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

BOT_ID = 999
OTHER_BOT_ID = 555
HUMAN_ID = 7


class _TextChannel:
    def __init__(self, channel_id: int = 100):
        self.id = channel_id
        self.name = "general"
        self.guild = SimpleNamespace(name="Test Server", id=1)
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _empty():
            return
            yield
        return _empty()


def _user(user_id: int, *, bot: bool = False):
    return SimpleNamespace(id=user_id, display_name=f"u{user_id}", name=f"u{user_id}", bot=bot)


def _message(content: str, *, mentions=()):
    return SimpleNamespace(
        id=42, content=content, mentions=list(mentions), attachments=[], reference=None,
        message_snapshots=None, created_at=datetime.now(timezone.utc), channel=_TextChannel(),
        author=_user(HUMAN_ID), type=discord_platform.discord.MessageType.default,
    )


@pytest.fixture
def adapter(monkeypatch):
    for var in ("DISCORD_REQUIRE_MENTION", "DISCORD_AUTO_THREAD", "DISCORD_NO_THREAD_CHANNELS",
                "DISCORD_FREE_RESPONSE_CHANNELS", "DISCORD_ALLOWED_CHANNELS", "DISCORD_IGNORED_CHANNELS",
                "DISCORD_HISTORY_BACKFILL", "DISCORD_ALLOW_BOTS", "DISCORD_IGNORE_NO_MENTION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    a = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    a._client = SimpleNamespace(user=_user(BOT_ID, bot=True))
    a._text_batch_delay_seconds = 0
    a.handle_message = AsyncMock()
    return a


async def _dispatched_event(adapter, message) -> MessageEvent:
    await adapter._handle_message(message)
    assert adapter.handle_message.call_count == 1
    return adapter.handle_message.call_args.args[0]


@pytest.mark.asyncio
async def test_explicit_mention_is_published_even_though_the_token_is_stripped(adapter):
    bot = adapter._client.user
    event = await _dispatched_event(
        adapter, _message(f"<@{BOT_ID}> what do you think <@{OTHER_BOT_ID}>",
                          mentions=[bot, _user(OTHER_BOT_ID, bot=True)]))
    assert f"<@{BOT_ID}>" not in event.text                       # existing normalisation
    assert event.metadata["explicit_self_mention"] is True
    assert str(BOT_ID) in event.metadata["discord_mentioned_user_ids"]
    assert event.metadata["discord_mentioned_bot_ids"] == sorted([str(BOT_ID), str(OTHER_BOT_ID)])
    assert f"<@{BOT_ID}>" in event.metadata["raw_content"]


@pytest.mark.asyncio
async def test_ambient_message_is_not_marked_as_addressed(adapter):
    event = await _dispatched_event(adapter, _message("just chatting"))
    assert event.metadata["explicit_self_mention"] is False
    assert event.metadata["discord_mentioned_user_ids"] == []


class _BatchAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.DISCORD)
        self._text_batch_delay_seconds = 0.0
        self._text_batch_split_delay_seconds = 0.0
        self.dispatched = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, *a, **k) -> None:
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def handle_message(self, event: MessageEvent) -> None:
        self.dispatched.append(event)


def _batched(text: str, **metadata) -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT,
                        source=SessionSource(platform=Platform.DISCORD, chat_id="c", chat_type="group"),
                        metadata=dict(metadata))


@pytest.mark.asyncio
async def test_batch_merge_keeps_the_mention_from_the_later_chunk():
    import asyncio

    adapter = _BatchAdapter()
    adapter._enqueue_text_event(_batched("first part", explicit_self_mention=False,
                                         discord_mentioned_user_ids=["7"]))
    adapter._enqueue_text_event(_batched("second part", explicit_self_mention=True,
                                         discord_mentioned_user_ids=["999"],
                                         discord_mentioned_bot_ids=["999"]))
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())
    (merged,) = adapter.dispatched
    assert merged.text == "first part\nsecond part"
    assert merged.metadata["explicit_self_mention"] is True
    assert merged.metadata["discord_mentioned_user_ids"] == ["7", "999"]
    assert merged.metadata["discord_mentioned_bot_ids"] == ["999"]
