"""Tests for Discord inbound context enrichment.

Covers the three inbound-context gaps fixed together:

1. Embed-only messages (bot error relays, CI alerts) were invisible to
   reply context and history backfill — only ``content`` was read.
2. Threads spun off a channel message backfilled empty: the thread's own
   history only contains a ``thread_starter_message`` system pointer, while
   the real content lives in the parent channel under the thread's ID.
3. Forwarded messages (``message_snapshots``) were only parsed on the
   trigger message — never in backfill — and a forward can't carry an
   @mention, so in mention-gated channels the wrapper is always dropped
   before snapshot parsing runs.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
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
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
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

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter, _extract_embed_text  # noqa: E402


class FakeDMChannel:
    def __init__(self, channel_id: int = 1, name: str = "dm"):
        self.id = channel_id
        self.name = name


class FakeTextChannel:
    def __init__(self, channel_id: int = 1, name: str = "general", guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name)
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield
        return _iter()


class FakeThread:
    def __init__(self, channel_id: int = 1, name: str = "thread", parent=None, guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(name=guild_name)
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield
        return _iter()


class FakeHistoryChannel(FakeTextChannel):
    def __init__(self, history_messages, **kwargs):
        super().__init__(**kwargs)
        self._history_messages = list(history_messages)

    def history(self, *, limit, before, after=None, oldest_first=None):
        before_id = int(getattr(before, "id", before))
        after_id = int(getattr(after, "id", after)) if after is not None else None
        if oldest_first is None:
            oldest_first = after is not None

        messages = [
            message for message in self._history_messages
            if int(message.id) < before_id
            and (after_id is None or int(message.id) > after_id)
        ]
        messages.sort(key=lambda message: int(message.id), reverse=not oldest_first)

        async def _iter():
            for message in messages[:limit]:
                yield message

        return _iter()


class FakeHistoryThread(FakeThread):
    def __init__(self, history_messages, **kwargs):
        super().__init__(**kwargs)
        self._history_messages = list(history_messages)

    history = FakeHistoryChannel.history


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)

    for _var in (
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_THREAD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_AUTO_THREAD",
        "DISCORD_NO_THREAD_CHANNELS",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_HISTORY_BACKFILL",
        "DISCORD_HISTORY_BACKFILL_LIMIT",
        "DISCORD_ALLOW_BOTS",
    ):
        monkeypatch.delenv(_var, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0  # disable batching for tests
    adapter.handle_message = AsyncMock()
    return adapter


def make_embed(*, title=None, description=None, fields=None, footer_text=None):
    return SimpleNamespace(
        title=title,
        description=description,
        fields=list(fields or []),
        footer=SimpleNamespace(text=footer_text) if footer_text else None,
    )


def make_history_message(
    *,
    author,
    content: str,
    msg_id: int,
    msg_type=None,
    attachments=None,
    embeds=None,
    message_snapshots=None,
):
    return SimpleNamespace(
        id=msg_id,
        author=author,
        content=content,
        attachments=list(attachments or []),
        embeds=list(embeds or []),
        message_snapshots=list(message_snapshots or []),
        type=msg_type if msg_type is not None else discord_platform.discord.MessageType.default,
    )


def make_message(*, channel, content: str, mentions=None, msg_type=None, reference=None):
    author = SimpleNamespace(id=42, display_name="Jezza", name="Jezza")
    return SimpleNamespace(
        id=12345,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        reference=reference,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
        type=msg_type if msg_type is not None else discord_platform.discord.MessageType.default,
    )


# ---------------------------------------------------------------------------
# _extract_embed_text
# ---------------------------------------------------------------------------


def test_extract_embed_text_flattens_title_description_fields_footer():
    msg = SimpleNamespace(embeds=[
        make_embed(
            title="Log Security Scan",
            description="Scanned 10 namespaces",
            fields=[SimpleNamespace(name="kube-system/elastic-agent", value="157 matches")],
            footer_text="severity: warn",
        ),
    ])

    assert _extract_embed_text(msg) == (
        "Log Security Scan\n"
        "Scanned 10 namespaces\n"
        "kube-system/elastic-agent: 157 matches\n"
        "severity: warn"
    )


def test_extract_embed_text_empty_for_no_embeds():
    assert _extract_embed_text(SimpleNamespace(embeds=[])) == ""
    assert _extract_embed_text(SimpleNamespace()) == ""


# ---------------------------------------------------------------------------
# Backfill: embed-only and forwarded messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_channel_context_includes_embed_only_bot_message(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    adapter.config.extra["history_backfill_limit"] = 10

    error_bot = SimpleNamespace(id=55, display_name="Error Bot", name="errorbot", bot=True)

    channel = FakeHistoryChannel(
        [
            make_history_message(
                author=error_bot,
                content="",
                msg_id=3,
                embeds=[make_embed(title="Alert", description="disk full on node-7")],
            ),
        ],
        channel_id=123,
    )

    result = await adapter._fetch_channel_context(channel, before=make_message(channel=channel, content="trigger"))

    assert result == "[Recent channel messages]\n[Error Bot [bot]] Alert\ndisk full on node-7"


@pytest.mark.asyncio
async def test_fetch_channel_context_includes_forwarded_snapshot(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "none")
    adapter.config.extra["history_backfill_limit"] = 10

    human = SimpleNamespace(id=56, display_name="Alice", name="Alice", bot=False)
    snapshot = SimpleNamespace(
        content="original forwarded text",
        embeds=[make_embed(description="embedded detail")],
        attachments=[],
    )

    channel = FakeHistoryChannel(
        [
            make_history_message(author=human, content="", msg_id=3, message_snapshots=[snapshot]),
        ],
        channel_id=123,
    )

    result = await adapter._fetch_channel_context(channel, before=make_message(channel=channel, content="trigger"))

    assert result == (
        "[Recent channel messages]\n"
        "[Alice] [Forwarded] original forwarded text\nembedded detail"
    )


# ---------------------------------------------------------------------------
# Backfill: thread starter message
# ---------------------------------------------------------------------------


def _make_starter_parent(starter_msg):
    return SimpleNamespace(
        id=99,
        guild=SimpleNamespace(name="Hermes Server"),
        fetch_message=AsyncMock(return_value=starter_msg),
    )


@pytest.mark.asyncio
async def test_fetch_channel_context_fetches_thread_starter_from_parent(adapter, monkeypatch):
    """A thread spun off another bot's embed-only message backfills the starter."""
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "none")
    adapter.config.extra["history_backfill_limit"] = 10

    error_bot = SimpleNamespace(id=55, display_name="Error Bot", name="errorbot", bot=True)
    starter = make_history_message(
        author=error_bot,
        content="",
        msg_id=456,
        embeds=[make_embed(title="Log Security Scan", description="157 unauthorized matches")],
    )
    parent = _make_starter_parent(starter)
    thread = FakeHistoryThread([], channel_id=456, parent=parent)

    result = await adapter._fetch_channel_context(thread, before=make_message(channel=thread, content="trigger"))

    parent.fetch_message.assert_awaited_once_with(456)
    assert result == (
        "[Recent channel messages]\n"
        "[Error Bot [bot] — thread starter] Log Security Scan\n157 unauthorized matches"
    )


@pytest.mark.asyncio
async def test_thread_starter_precedes_thread_history_chronologically(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "none")
    adapter.config.extra["history_backfill_limit"] = 10

    human = SimpleNamespace(id=56, display_name="Alice", name="Alice", bot=False)
    starter = make_history_message(author=human, content="starter question", msg_id=456)
    parent = _make_starter_parent(starter)
    thread = FakeHistoryThread(
        [make_history_message(author=human, content="follow-up", msg_id=500)],
        channel_id=456,
        parent=parent,
    )

    result = await adapter._fetch_channel_context(thread, before=make_message(channel=thread, content="trigger"))

    assert result == (
        "[Recent channel messages]\n"
        "[Alice — thread starter] starter question\n"
        "[Alice] follow-up"
    )


@pytest.mark.asyncio
async def test_thread_starter_skipped_when_partition_hit(adapter, monkeypatch):
    """If the bot already spoke in the thread, the starter was surfaced when it
    first engaged — don't re-fetch it."""
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "none")
    adapter.config.extra["history_backfill_limit"] = 10

    human = SimpleNamespace(id=56, display_name="Alice", name="Alice", bot=False)
    parent = _make_starter_parent(make_history_message(author=human, content="starter", msg_id=456))
    thread = FakeHistoryThread(
        [
            make_history_message(author=human, content="new question", msg_id=600),
            make_history_message(author=adapter._client.user, content="our prior answer", msg_id=500),
        ],
        channel_id=456,
        parent=parent,
    )

    result = await adapter._fetch_channel_context(thread, before=make_message(channel=thread, content="trigger"))

    parent.fetch_message.assert_not_awaited()
    assert result == "[Recent channel messages]\n[Alice] new question"


@pytest.mark.asyncio
async def test_thread_starter_fetch_failure_is_silent(adapter, monkeypatch):
    """Standalone threads (no starter in the parent channel) 404 on fetch."""
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "none")
    adapter.config.extra["history_backfill_limit"] = 10

    parent = SimpleNamespace(
        id=99,
        guild=SimpleNamespace(name="Hermes Server"),
        fetch_message=AsyncMock(side_effect=Exception("404 Not Found")),
    )
    thread = FakeHistoryThread([], channel_id=456, parent=parent)

    result = await adapter._fetch_channel_context(thread, before=make_message(channel=thread, content="trigger"))

    assert result == ""


# ---------------------------------------------------------------------------
# Reply reference: embeds and attachments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_to_embed_only_message_yields_reply_text(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    referenced = SimpleNamespace(
        content="",
        embeds=[make_embed(title="Alert", description="disk full")],
        attachments=[],
    )
    reference = SimpleNamespace(message_id=777, resolved=referenced, type=None)
    message = make_message(
        channel=FakeTextChannel(channel_id=123),
        content="what's this?",
        reference=reference,
    )

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.reply_to_message_id == "777"
    assert event.reply_to_text == "Alert\ndisk full"


@pytest.mark.asyncio
async def test_reply_attachments_join_media_pipeline(adapter, monkeypatch):
    """Replying to a screenshot routes the referenced image through vision."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    adapter._cache_discord_image = AsyncMock(return_value="/cache/ref-screenshot.png")

    screenshot = SimpleNamespace(
        content_type="image/png", url="https://cdn.discordapp.com/x.png",
        filename="x.png", size=1024,
    )
    referenced = SimpleNamespace(content="error screenshot", embeds=[], attachments=[screenshot])
    reference = SimpleNamespace(message_id=778, resolved=referenced, type=None)
    message = make_message(
        channel=FakeTextChannel(channel_id=123),
        content="can you read this error?",
        reference=reference,
    )

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.media_urls == ["/cache/ref-screenshot.png"]
    assert event.reply_to_text == "error screenshot"


@pytest.mark.asyncio
async def test_reply_reference_fetched_when_not_resolved(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    referenced = SimpleNamespace(content="older message text", embeds=[], attachments=[])
    channel = FakeTextChannel(channel_id=123)
    channel.fetch_message = AsyncMock(return_value=referenced)
    reference = SimpleNamespace(message_id=779, resolved=None, type=None)
    message = make_message(channel=channel, content="see above", reference=reference)

    await adapter._handle_message(message)

    channel.fetch_message.assert_awaited_once_with(779)
    event = adapter.handle_message.await_args.args[0]
    assert event.reply_to_text == "older message text"


@pytest.mark.asyncio
async def test_forward_reference_not_treated_as_reply(adapter, monkeypatch):
    """Forwards populate message.reference (type=forward) pointing at the
    original — snapshot parsing owns those; reply handling must skip them."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    snapshot = SimpleNamespace(content="forwarded payload", embeds=[], attachments=[])
    reference = SimpleNamespace(message_id=780, resolved=None, type=SimpleNamespace(name="forward"))
    channel = FakeTextChannel(channel_id=123)
    channel.fetch_message = AsyncMock()
    message = make_message(channel=channel, content="", reference=reference)
    message.message_snapshots = [snapshot]

    await adapter._handle_message(message)

    channel.fetch_message.assert_not_awaited()
    event = adapter.handle_message.await_args.args[0]
    assert event.reply_to_message_id is None
    assert event.reply_to_text is None
    assert event.text == "forwarded payload"
