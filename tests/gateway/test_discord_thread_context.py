"""Tests for Discord thread starter/body context injection."""

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
    discord_mod.MessageType = SimpleNamespace(default="default", reply="reply")
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
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class FakeTextChannel:
    def __init__(self, channel_id: int = 1, name: str = "general", guild_name: str = "Hermes Server", messages=None):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(id=10, name=guild_name)
        self.topic = None
        self._messages = {str(getattr(message, "id", "")): message for message in (messages or [])}

    async def fetch_message(self, message_id):
        return self._messages[str(message_id)]

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield

        return _iter()


class FakeThread:
    def __init__(self, channel_id: int = 2, name: str = "email alert", parent=None, history_messages=None, starter_message=None):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(id=10, name="Hermes Server")
        self.topic = None
        self.starter_message = starter_message
        self._history_messages = list(history_messages or [])

    def history(self, *, limit=100, before=None, after=None, oldest_first=None):
        async def _iter():
            messages = self._history_messages[:limit]
            if oldest_first is False:
                messages = list(reversed(messages))
            for message in messages:
                yield message

        return _iter()


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    monkeypatch.setattr(discord_platform.discord, "DMChannel", type("FakeDMChannel", (), {}), raising=False)
    monkeypatch.setattr(discord_platform.discord, "MessageType", SimpleNamespace(default="default", reply="reply"), raising=False)
    for var in (
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_THREAD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_AUTO_THREAD",
        "DISCORD_NO_THREAD_CHANNELS",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_HISTORY_BACKFILL",
        "DISCORD_THREAD_CONTEXT_MESSAGES",
        "DISCORD_THREAD_CONTEXT_MAX_CHARS",
    ):
        monkeypatch.delenv(var, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    return adapter


def fake_message(message_id: int, content: str, *, author_name: str = "User", channel=None, msg_type="default", attachments=None):
    return SimpleNamespace(
        id=message_id,
        content=content,
        clean_content=content,
        mentions=[],
        attachments=list(attachments or []),
        message_snapshots=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=SimpleNamespace(id=message_id + 1000, display_name=author_name, name=author_name),
        guild=getattr(channel, "guild", None),
        type=msg_type,
    )


@pytest.mark.asyncio
async def test_build_thread_context_uses_starter_message(adapter):
    starter = fake_message(200, "# 고로켓 이메일 알림\n요약: TLS 인증서 확인 필요", author_name="Hermes")
    prior = fake_message(201, "이건 배포 환경 확인 필요", author_name="Doo")
    thread = FakeThread(channel_id=200, name="고로켓 이메일 알림", starter_message=starter, history_messages=[starter, prior])
    current = fake_message(202, "이거 무슨 내용이야?", channel=thread, author_name="Doo")

    context = await adapter._build_thread_context_injection(thread, current)

    assert context is not None
    assert context.startswith("[Discord thread context]")
    assert "Thread: 고로켓 이메일 알림" in context
    assert "[starter] Hermes: # 고로켓 이메일 알림" in context
    assert "TLS 인증서 확인 필요" in context
    assert "[message] Doo: 이건 배포 환경 확인 필요" in context
    assert "이거 무슨 내용이야?" not in context


@pytest.mark.asyncio
async def test_build_thread_context_fetches_parent_starter_fallback(adapter):
    starter = fake_message(300, "Parent channel starter content", author_name="Hermes")
    parent = FakeTextChannel(channel_id=100, messages=[starter])
    thread = FakeThread(channel_id=300, name="fallback-thread", parent=parent, starter_message=None)
    current = fake_message(301, "follow-up", channel=thread)

    context = await adapter._build_thread_context_injection(thread, current)

    assert context is not None
    assert "[starter] Hermes: Parent channel starter content" in context


@pytest.mark.asyncio
async def test_handle_message_injects_thread_context_for_non_command(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_HISTORY_BACKFILL", "false")
    starter = fake_message(400, "Cron alert body", author_name="Hermes")
    thread = FakeThread(channel_id=400, name="alert-thread", starter_message=starter, history_messages=[starter])
    message = fake_message(401, "이거 처리해줘", channel=thread, author_name="Doo")
    adapter._threads.mark(str(thread.id))

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text.startswith("[Discord thread context]")
    assert "[starter] Hermes: Cron alert body" in event.text
    assert event.text.endswith("이거 처리해줘")


@pytest.mark.asyncio
async def test_handle_message_skips_thread_context_for_commands(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_HISTORY_BACKFILL", "false")
    starter = fake_message(500, "Do not inject into command", author_name="Hermes")
    thread = FakeThread(channel_id=500, name="command-thread", starter_message=starter, history_messages=[starter])
    message = fake_message(501, "/status", channel=thread, author_name="Doo")
    adapter._threads.mark(str(thread.id))

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "/status"


@pytest.mark.asyncio
async def test_thread_context_can_be_disabled(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_CONTEXT_MESSAGES", "0")
    starter = fake_message(600, "Hidden starter", author_name="Hermes")
    thread = FakeThread(channel_id=600, name="disabled-thread", starter_message=starter, history_messages=[starter])
    current = fake_message(601, "follow-up", channel=thread)

    context = await adapter._build_thread_context_injection(thread, current)

    assert context is None
