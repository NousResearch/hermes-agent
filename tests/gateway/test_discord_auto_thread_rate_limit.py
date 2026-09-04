"""Auto-thread 429 handling: fail fast, actionable error, no orphan seed msg.

When Discord rate-limits the create-thread bucket (HTTP 429), the
auto-thread path must not burn through the seed-message fallback and a
backoff retry (both hit the same bucket, so both are guaranteed to fail),
and the user-facing error must carry an honest wait hint instead of
advising an immediate retry that cannot succeed.

Fixtures mirror tests/gateway/test_discord_channel_controls.py.
"""

from types import SimpleNamespace
from datetime import datetime, timezone
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
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


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


class FakeThread:
    def __init__(self, channel_id: int = 1, name: str = "thread", parent=None, guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(name=guild_name)
        self.topic = None


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0  # disable batching for tests
    adapter.handle_message = AsyncMock()
    return adapter


def make_message(*, channel, content: str, mentions=None):
    author = SimpleNamespace(id=42, display_name="TestUser", name="TestUser")
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
    )


class FakeRateLimited(Exception):
    """Duck-typed Discord 429 (recognised via name + retry_after)."""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Too many requests. Retry in {retry_after} seconds.")


def _make_auto_thread_message(channel):
    """Message whose create_thread raises a 429, on a channel that records sends."""
    message = make_message(channel=channel, content="hello")
    message.create_thread = AsyncMock(side_effect=FakeRateLimited(280.36))
    return message


@pytest.mark.asyncio
async def test_auto_thread_rate_limit_skips_seed_fallback(adapter):
    """A 429 on create_thread must not attempt the seed-message fallback.

    The seed message's create_thread call hits the same rate-limit bucket,
    so the fallback is guaranteed to fail — and it leaves an orphan
    "Thread created by Hermes" message in the channel when it does.
    """
    channel = FakeTextChannel(channel_id=800)
    channel.send = AsyncMock()
    message = _make_auto_thread_message(channel)

    thread = await adapter._auto_create_thread(message)

    assert thread is None
    # Exactly one create attempt: no backoff retry inside the 429 window.
    message.create_thread.assert_awaited_once()
    # No seed message posted to the channel.
    channel.send.assert_not_awaited()
    # retry_after captured for the caller's user-facing hint.
    assert adapter._auto_thread_retry_after == pytest.approx(280.36)


@pytest.mark.asyncio
async def test_auto_thread_rate_limit_error_message_is_actionable(adapter, monkeypatch):
    """On 429 the user-facing error carries a wait hint, not 'please retry'."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    monkeypatch.delenv("DISCORD_NO_THREAD_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    channel = FakeTextChannel(channel_id=800)
    channel.send = AsyncMock()
    message = _make_auto_thread_message(channel)

    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()
    channel.send.assert_awaited_once()
    sent_text = channel.send.await_args.args[0]
    assert "rate-limiting" in sent_text.lower()
    # 280.36s rounds up to ~5 minutes; must not advise an immediate retry.
    assert "~5 minute" in sent_text
    assert "existing thread" in sent_text.lower()
    assert "please retry." not in sent_text.lower()


@pytest.mark.asyncio
async def test_auto_thread_non_rate_limit_failure_keeps_generic_error(adapter, monkeypatch):
    """Non-429 failures keep the original generic error text."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    monkeypatch.delenv("DISCORD_NO_THREAD_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    adapter._auto_create_thread = AsyncMock(return_value=None)
    # No rate-limit info recorded by the stub.
    adapter._auto_thread_retry_after = None

    channel = FakeTextChannel(channel_id=800)
    channel.send = AsyncMock()
    message = make_message(channel=channel, content="hello")

    await adapter._handle_message(message)

    channel.send.assert_awaited_once()
    sent_text = channel.send.await_args.args[0]
    assert "could not create" in sent_text.lower()


@pytest.mark.asyncio
async def test_auto_thread_rate_limit_on_seed_fallback_skips_retry(adapter):
    """A 429 raised by the seed-message fallback also fails fast."""
    channel = FakeTextChannel(channel_id=800)
    # Direct create_thread fails with a non-429 error; the seed message send
    # then hits the rate limit.
    channel.send = AsyncMock(side_effect=FakeRateLimited(120.0))
    message = make_message(channel=channel, content="hello")
    message.create_thread = AsyncMock(side_effect=RuntimeError("boom"))

    thread = await adapter._auto_create_thread(message)

    assert thread is None
    # No second loop iteration: one direct attempt, one fallback attempt.
    message.create_thread.assert_awaited_once()
    channel.send.assert_awaited_once()
    assert adapter._auto_thread_retry_after == pytest.approx(120.0)


def test_format_retry_after_hint_rounds_up():
    """Wait hints round up so they never undersell the wait."""
    fmt = DiscordAdapter._format_retry_after_hint
    assert fmt(280.36) == "~5 minutes"
    assert fmt(176.74) == "~3 minutes"
    assert fmt(60) == "~1 minute"
    assert fmt(59.2) == "~1 minute"
    assert fmt(45) == "~45s"
    assert fmt(1) == "~1s"
    assert fmt("garbage") == "a few minutes"
