"""Tests for Discord auto-thread orphaned-seed-message cleanup.

When auto-threading is enabled, _auto_create_thread() first tries
``message.create_thread()``.  If that fails (commonly a Discord 429
rate-limit on thread creation), it falls back to posting a seed
announcement message ("🧵 Thread created by Hermes: ...") and creating
the thread *from that message*.

The bug: the seed announcement is posted **before** the fallback
``create_thread()`` is confirmed to succeed.  When the fallback also
fails (e.g. the rate-limit is still in effect), the announcement is left
orphaned in the channel and the caller responds inline — so users see a
"Thread created by Hermes" message with no thread behind it, and the
answer in the main channel.

Fix: on fallback failure, delete the orphaned seed message so the
announcement only ever survives when a real thread exists.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class _RateLimited(Exception):
    """Stand-in for discord.HTTPException 429."""


class _SeedMessage:
    """Fake seed message returned by channel.send()."""

    def __init__(self, create_thread_succeeds: bool, thread=None):
        self._create_thread_succeeds = create_thread_succeeds
        self._thread = thread
        self.deleted = False
        self.create_thread = AsyncMock(side_effect=self._create_thread)
        self.delete = AsyncMock(side_effect=self._delete)

    async def _create_thread(self, *args, **kwargs):
        if self._create_thread_succeeds:
            return self._thread
        raise _RateLimited("Too many requests. Retry in 222.97 seconds.")

    async def _delete(self, *args, **kwargs):
        self.deleted = True


class _Channel:
    def __init__(self, seed_message):
        self.id = 100
        self.name = "general"
        self.guild = SimpleNamespace(name="Test Server", id=1)
        self.send = AsyncMock(return_value=seed_message)


class _Thread:
    def __init__(self, thread_id=55555, parent=None):
        self.id = thread_id
        self.name = "thread"
        self.parent = parent


def _make_message(*, channel, content="hello bot", direct_create_succeeds=False,
                  thread=None):
    async def _create_thread(*args, **kwargs):
        if direct_create_succeeds:
            return thread
        raise _RateLimited("Too many requests. Retry in 278.53 seconds.")

    return SimpleNamespace(
        id=42,
        content=content,
        channel=channel,
        author=SimpleNamespace(id=7, display_name="Alice", name="Alice", bot=False),
        created_at=datetime.now(timezone.utc),
        type=discord_platform.discord.MessageType.default,
        create_thread=AsyncMock(side_effect=_create_thread),
    )


@pytest.fixture
def adapter(monkeypatch):
    for var in ("DISCORD_AUTO_THREAD", "DISCORD_NO_THREAD_CHANNELS"):
        monkeypatch.delenv(var, raising=False)
    config = PlatformConfig(enabled=True, token="***")
    a = DiscordAdapter(config)
    a._client = SimpleNamespace(user=SimpleNamespace(id=999, bot=True))
    return a


class TestAutoThreadOrphanSeed:
    @pytest.mark.asyncio
    async def test_fallback_failure_deletes_orphaned_seed(self, adapter):
        """Direct create 429s, fallback create 429s → seed message deleted, None returned."""
        seed = _SeedMessage(create_thread_succeeds=False)
        channel = _Channel(seed)
        message = _make_message(channel=channel, direct_create_succeeds=False)

        result = await adapter._auto_create_thread(message)

        assert result is None, "must return None when both create paths fail"
        channel.send.assert_awaited_once()  # seed announcement was posted
        assert seed.deleted is True, (
            "orphaned seed announcement must be deleted when fallback "
            "create_thread fails, so users never see a 'Thread created' "
            "message with no thread behind it"
        )

    @pytest.mark.asyncio
    async def test_fallback_success_keeps_seed(self, adapter):
        """Direct create 429s but fallback create succeeds → seed kept, thread returned."""
        thread = _Thread()
        seed = _SeedMessage(create_thread_succeeds=True, thread=thread)
        channel = _Channel(seed)
        message = _make_message(channel=channel, direct_create_succeeds=False)

        result = await adapter._auto_create_thread(message)

        assert result is thread
        channel.send.assert_awaited_once()
        assert seed.deleted is False, "seed must survive when a real thread was created"

    @pytest.mark.asyncio
    async def test_direct_success_posts_no_seed(self, adapter):
        """Direct create succeeds → no seed announcement, no fallback."""
        thread = _Thread()
        seed = _SeedMessage(create_thread_succeeds=True, thread=thread)
        channel = _Channel(seed)
        message = _make_message(
            channel=channel, direct_create_succeeds=True, thread=thread
        )

        result = await adapter._auto_create_thread(message)

        assert result is thread
        channel.send.assert_not_awaited()  # no seed message on the happy path
