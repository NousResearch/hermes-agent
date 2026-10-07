"""Tests for Discord auto-thread 429 handling.

The thread-create rate-limit bucket refills on the order of minutes, but
_auto_create_thread() previously treated every failure as transient: two
attempts with a 0.75s backoff, then the seed-message fallback (which doubles
the API load while rate-limited and leaves an orphan seed message in the
channel when the create also fails). The user message was then dropped with
"could not create a Discord thread".

Fix: on a Discord 429, wait out the server's Retry-After hint once (capped)
and retry the direct path; skip the seed-message fallback for rate limits.

Covers:
  1. Rate-limited once → waits Retry-After, direct retry succeeds, no seed
     message is ever sent.
  2. Rate-limited on both attempts → returns None, no seed message, and the
     wait is capped at the module maximum.
  3. Non-rate-limit failures keep the original transient-retry + fallback
     behavior.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import plugins.platforms.discord.adapter as discord_platform
from plugins.platforms.discord.adapter import (
    DiscordAdapter,
    _DISCORD_AUTO_THREAD_MAX_RATE_LIMIT_SLEEP_SECONDS,
)

from tests.gateway.test_discord_double_dispatch import (  # noqa: F401  (fixtures reused)
    _make_message,
    _TextChannel,
    adapter,
)


def _rate_limit_error(retry_after: float) -> "_RateLimitError":
    """Build an exception that _is_discord_rate_limit() recognizes (duck-typed,
    matching the adapter's own fallback for mocked transports)."""
    return _RateLimitError(retry_after)


class _RateLimitError(Exception):
    """Duck-typed rate limit: name contains 'ratelimit' + numeric retry_after."""

    def __init__(self, retry_after: float):
        super().__init__(f"Too many requests. Retry in {retry_after:.2f} seconds.")
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# 1. 429 once → wait Retry-After → direct retry succeeds
# ---------------------------------------------------------------------------

class TestRateLimitWaitAndRetry:
    @pytest.mark.asyncio
    async def test_429_waits_retry_after_then_succeeds_direct(self, adapter, monkeypatch):
        """First attempt 429s; after honoring Retry-After the direct retry
        succeeds and no fallback seed message is sent."""
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        attempts = {"n": 0}

        async def fake_create_thread(**kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _RateLimitError(30.0)
            return SimpleNamespace(id=555, name="t")

        message = _make_message(channel=_TextChannel())
        message.create_thread = AsyncMock(side_effect=fake_create_thread)
        message.channel.send = AsyncMock()

        result = await adapter._auto_create_thread(message)

        assert result is not None
        assert result.id == 555
        assert attempts["n"] == 2
        assert sleeps == [30.0]
        # Seed-message fallback must NOT be used for rate limits.
        message.channel.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_429_retry_after_is_capped(self, adapter, monkeypatch):
        """A huge Retry-After hint is capped at the module maximum so the
        user's message dispatch doesn't stall for many minutes."""
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        attempts = {"n": 0}

        async def fake_create_thread(**kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _RateLimitError(3600.0)
            return SimpleNamespace(id=556, name="t")

        message = _make_message(channel=_TextChannel())
        message.create_thread = AsyncMock(side_effect=fake_create_thread)
        message.channel.send = AsyncMock()

        result = await adapter._auto_create_thread(message)

        assert result is not None
        assert sleeps == [_DISCORD_AUTO_THREAD_MAX_RATE_LIMIT_SLEEP_SECONDS]

    @pytest.mark.asyncio
    async def test_429_without_retry_hint_uses_default_wait(self, adapter, monkeypatch):
        """A 429 with no parseable Retry-After backs off the conservative default."""
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        # Rate-limit-shaped exception (name fallback) whose retry_after is a
        # non-numeric value, so _extract_discord_retry_after() returns None.
        class RateLimitedError(Exception):
            retry_after = "soon"

        attempts = {"n": 0}

        async def fake_create_thread(**kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RateLimitedError("Too many requests")
            return SimpleNamespace(id=557, name="t")

        message = _make_message(channel=_TextChannel())
        message.create_thread = AsyncMock(side_effect=fake_create_thread)
        message.channel.send = AsyncMock()

        result = await adapter._auto_create_thread(message)

        assert result is not None
        assert sleeps == [_DISCORD_AUTO_THREAD_MAX_RATE_LIMIT_SLEEP_SECONDS]


# ---------------------------------------------------------------------------
# 2. Persistent 429 → give up WITHOUT seed messages
# ---------------------------------------------------------------------------

class TestPersistentRateLimit:
    @pytest.mark.asyncio
    async def test_persistent_429_returns_none_without_seed_message(self, adapter, monkeypatch):
        """When both attempts 429, _auto_create_thread returns None and never
        posts the seed message (no orphan 'Thread created by Hermes' left behind)."""
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        async def always_429(**kwargs):
            raise _RateLimitError(120.0)

        message = _make_message(channel=_TextChannel())
        message.create_thread = AsyncMock(side_effect=always_429)
        # The seed-message path rate-limits too: both the channel.send and the
        # seed's create_thread raise 429, so nothing can succeed.
        message.channel.send = AsyncMock(side_effect=_RateLimitError(120.0))

        result = await adapter._auto_create_thread(message)

        assert result is None
        # One capped Retry-After wait; the seed-message fallback never ran
        # (channel.send 429'd), so no transient 0.75s backoff either.
        assert sleeps == [90.0]
        message.channel.send.assert_awaited_once()


# ---------------------------------------------------------------------------
# 3. Non-rate-limit failures keep the original behavior
# ---------------------------------------------------------------------------

class TestTransientBehaviorUnchanged:
    @pytest.mark.asyncio
    async def test_transient_error_still_uses_fallback(self, adapter, monkeypatch):
        """A non-429 failure (e.g. connect error) still falls back to the
        seed-message path with the 0.75s transient backoff."""
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        direct_attempts = {"n": 0}
        seed_thread = SimpleNamespace(id=558, name="t")

        async def fake_direct(**kwargs):
            direct_attempts["n"] += 1
            raise RuntimeError("Cannot connect to host discord.com:443")

        async def fake_seed_create_thread(**kwargs):
            # The fallback seed path succeeds on the first attempt (original
            # behavior for transient errors — no full-loop exhaustion here).
            return seed_thread

        seed_msg = SimpleNamespace(create_thread=AsyncMock(side_effect=fake_seed_create_thread))

        message = _make_message(channel=_TextChannel())
        message.create_thread = AsyncMock(side_effect=fake_direct)
        message.channel.send = AsyncMock(return_value=seed_msg)

        result = await adapter._auto_create_thread(message)

        assert result is not None
        assert result.id == 558
        # Direct tried once, then the seed-message fallback succeeded.
        assert direct_attempts["n"] == 1
        message.channel.send.assert_awaited_once()
        assert sleeps == []
