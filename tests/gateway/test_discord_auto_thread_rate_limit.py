"""Tests for rate-limit-aware auto-thread creation.

When Discord rate-limits thread creation (HTTP 429), discord.py raises
``RateLimited`` only when the server-requested ``retry_after`` exceeds its
internal ``max_ratelimit_timeout``. The old retry loop slept a fixed 0.75s
and retried — guaranteed to fail against a long Retry-After (e.g. 260s),
so the user's message was dropped with a generic "please retry" notice.

The fix:
  1. Honors ``retry_after`` within ``_AUTO_THREAD_MAX_RATE_LIMIT_WAIT_SECONDS``
     by waiting out the bucket before retrying the direct path.
  2. Gives up immediately for longer rate limits and returns a per-call
     ``_AutoThreadRateLimited`` sentinel carrying the delay, so the caller
     surfaces a rate-limit-aware notice ("retry in ~Ns") instead of the
     generic one — without any shared adapter state.
"""

from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import AsyncMock
import sys

import pytest

from gateway.config import PlatformConfig

# tests/gateway/conftest.py installs a comprehensive discord mock at
# collection time, so ``discord`` here is that mock (no __file__). We define
# a real RateLimited stand-in and monkeypatch it onto the adapter's discord
# reference per-test so ``raise RateLimited(n)`` behaves like the real one.


class _RateLimited(Exception):
    """Stand-in for discord.RateLimited with the same retry_after contract."""

    def __init__(self, retry_after: float):
        self.retry_after = float(retry_after)
        super().__init__(f"Too many requests. Retry in {retry_after:.2f} seconds.")


import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import (  # noqa: E402
    DiscordAdapter,
    _AUTO_THREAD_MAX_RATE_LIMIT_WAIT_SECONDS,
    _AutoThreadRateLimited,
)


class FakeTextChannel:
    def __init__(self, channel_id: int = 1, name: str = "general", guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name)
        self.topic = None
        self.send = AsyncMock()


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
    # Make the adapter's discord.RateLimited a real exception class so the
    # production ``_is_discord_rate_limit`` isinstance check works.
    monkeypatch.setattr(discord_platform.discord, "RateLimited", _RateLimited, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    a = DiscordAdapter(config)
    a._client = SimpleNamespace(user=SimpleNamespace(id=999))
    a._text_batch_delay_seconds = 0  # disable batching for tests
    a.handle_message = AsyncMock()
    return a


def make_message(*, channel, content: str = "hello"):
    author = SimpleNamespace(id=42, display_name="TestUser", name="TestUser", bot=False)
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=[],
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
    )


def _patch_sleep(monkeypatch, sleeps):
    """Record asyncio.sleep calls without actually waiting."""
    async def _fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(discord_platform.asyncio, "sleep", _fake_sleep)


# ── direct: short rate limit is waited out, then retried ─────────────


@pytest.mark.asyncio
async def test_short_rate_limit_waits_retry_after_then_succeeds(adapter, monkeypatch):
    """A 429 with retry_after within the bound is waited out and retried.

    The direct ``create_thread`` raises ``RateLimited(3.0)`` once, then
    succeeds. The adapter must sleep ``retry_after`` (not the fixed 0.75s
    backoff) before retrying, and the retried call must return the thread.
    """
    sleeps = []
    _patch_sleep(monkeypatch, sleeps)

    channel = FakeTextChannel(channel_id=100)
    message = make_message(channel=channel)
    fake_thread = FakeThread(channel_id=100)
    calls = {"n": 0}

    async def flaky_create_thread(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _RateLimited(3.0)
        return fake_thread

    message.create_thread = flaky_create_thread

    result = await adapter._auto_create_thread(message)

    assert result is fake_thread
    assert calls["n"] == 2, "direct path should be retried once after the wait"
    assert sleeps == [3.0], "should sleep the server-requested retry_after, not 0.75s"
    # Seed-message fallback must not have run — the retry succeeded directly.
    channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_connect_error_uses_seed_fallback(adapter, monkeypatch):
    """Non-rate-limit direct error falls through to the seed-message fallback.

    The original #20243 behavior is preserved: transient connect errors on
    ``message.create_thread`` still try the seed-message fallback, which
    creates the thread from a posted message. No rate-limit state is set.
    """
    sleeps = []
    _patch_sleep(monkeypatch, sleeps)

    channel = FakeTextChannel(channel_id=100)
    message = make_message(channel=channel)
    fake_thread = FakeThread(channel_id=100)

    async def always_connect_error(**kwargs):
        raise RuntimeError("Cannot connect to host discord.com:443")

    async def seed_create_thread(**kwargs):
        return fake_thread

    message.create_thread = always_connect_error
    # channel.send returns the seed message whose create_thread succeeds.
    channel.send.return_value = SimpleNamespace(create_thread=seed_create_thread)

    result = await adapter._auto_create_thread(message)

    assert result is fake_thread
    assert sleeps == [], "fallback succeeded on the first attempt — no backoff needed"
    channel.send.assert_awaited_once()
    assert not isinstance(result, _AutoThreadRateLimited)


@pytest.mark.asyncio
async def test_two_short_rate_limits_give_up_skipping_fallback(adapter, monkeypatch):
    """Two short direct rate limits → give up; seed fallback is NOT attempted.

    The seed-message fallback would hit the same channel bucket and only
    spam a stray message into the channel, so it is skipped once the direct
    path has been rate-limited on both attempts.
    """
    sleeps = []
    _patch_sleep(monkeypatch, sleeps)

    channel = FakeTextChannel(channel_id=100)
    message = make_message(channel=channel)

    async def always_rate_limited(**kwargs):
        raise _RateLimited(2.0)

    message.create_thread = always_rate_limited

    result = await adapter._auto_create_thread(message)

    assert isinstance(result, _AutoThreadRateLimited)
    assert result.retry_after == 2.0
    assert sleeps == [2.0], "attempt 0 waits out the short limit; attempt 1 gives up"
    channel.send.assert_not_awaited(), "seed fallback must not spam a 429'd channel"


# ── direct: long rate limit gives up fast with recorded retry_after ──


@pytest.mark.asyncio
async def test_long_rate_limit_gives_up_and_records_retry_after(adapter, monkeypatch):
    """A 429 with retry_after beyond the bound gives up immediately.

    The adapter must NOT block for minutes. It returns None and records the
    delay so the caller can tell the user when to retry. The seed-message
    fallback is skipped too — it would hit the same bucket.
    """
    sleeps = []
    _patch_sleep(monkeypatch, sleeps)

    long_wait = _AUTO_THREAD_MAX_RATE_LIMIT_WAIT_SECONDS + 60  # e.g. 90s
    channel = FakeTextChannel(channel_id=100)
    message = make_message(channel=channel)

    async def always_rate_limited(**kwargs):
        raise _RateLimited(long_wait)

    message.create_thread = always_rate_limited

    result = await adapter._auto_create_thread(message)

    assert isinstance(result, _AutoThreadRateLimited)
    assert result.retry_after == long_wait
    assert sleeps == [], "long rate limit must not block the handler"
    channel.send.assert_not_awaited(), "fallback would also 429 — skip it"


# ── handle_message integration: rate-limit-aware user notice ─────────


@pytest.mark.asyncio
async def test_rate_limit_failure_notifies_user_with_retry_hint(adapter, monkeypatch):
    """Rate-limit failure surfaces a 'retry in ~Ns' notice, not the generic one."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    monkeypatch.delenv("DISCORD_NO_THREAD_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    channel = FakeTextChannel(channel_id=800)
    message = make_message(channel=channel, content="hello")

    async def rate_limited_create_thread(**kwargs):
        raise _RateLimited(120.0)

    message.create_thread = rate_limited_create_thread

    # Silence sleeps — 120s > bound, so none happen anyway, but keep it fast.
    _patch_sleep(monkeypatch, [])

    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()
    channel.send.assert_awaited_once()
    sent_text = channel.send.await_args.args[0]
    assert "rate-limiting" in sent_text.lower()
    assert "120" in sent_text  # tells the user roughly when to retry


@pytest.mark.asyncio
async def test_non_rate_limit_failure_keeps_generic_notice(adapter, monkeypatch):
    """A non-429 failure keeps the original generic 'could not create' notice."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    monkeypatch.delenv("DISCORD_NO_THREAD_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    channel = FakeTextChannel(channel_id=800)
    message = make_message(channel=channel, content="hello")

    async def failing_create_thread(**kwargs):
        raise RuntimeError("Cannot connect to host discord.com:443")

    message.create_thread = failing_create_thread

    # The seed-message fallback also fails (same connect error), but the
    # user-facing notice must still go through.
    async def failing_seed_create_thread(**kwargs):
        raise RuntimeError("Cannot connect to host discord.com:443")

    channel.send.return_value = SimpleNamespace(create_thread=failing_seed_create_thread)

    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()
    # Last channel.send is the user-facing notice (after 2 fallback seed
    # attempts). Verify it kept the original generic wording.
    channel.send.assert_awaited()
    sent_text = channel.send.await_args.args[0]
    assert "could not create" in sent_text.lower()
    assert "thread" in sent_text.lower()
    assert "rate-limiting" not in sent_text.lower()


# ── interleaving regression: per-message notice isolation (#review) ────


@pytest.mark.asyncio
async def test_interleaved_rate_limit_failures_get_own_notices(adapter, monkeypatch):
    """Concurrent auto-thread failures each surface their own retry hint.

    Regression for shared-state hazard: retry-after metadata travels in the
    per-call ``_AutoThreadRateLimited`` return value, never on adapter state,
    so two messages racing through ``_auto_create_thread`` (which awaits
    during retry sleeps) cannot overwrite each other's rate-limit notice.
    """
    import asyncio

    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    monkeypatch.delenv("DISCORD_NO_THREAD_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    sleeps = []
    _patch_sleep(monkeypatch, sleeps)

    channel_a = FakeTextChannel(channel_id=810)
    channel_b = FakeTextChannel(channel_id=811)
    msg_a = make_message(channel=channel_a, content="hello a")
    msg_b = make_message(channel=channel_b, content="hello b")

    async def rate_limited_a(**kwargs):
        raise _RateLimited(45.0)

    async def rate_limited_b(**kwargs):
        raise _RateLimited(90.0)

    msg_a.create_thread = rate_limited_a
    msg_b.create_thread = rate_limited_b

    await asyncio.gather(
        adapter._handle_message(msg_a),
        adapter._handle_message(msg_b),
    )

    # Each message got its OWN rate-limit notice with its own retry_after —
    # 45s for channel A, 90s for channel B. If the metadata lived on shared
    # adapter state, both could show the same (last-written) value.
    adapter.handle_message.assert_not_awaited()
    text_a = channel_a.send.await_args.args[0]
    text_b = channel_b.send.await_args.args[0]
    assert "45" in text_a and "rate-limiting" in text_a.lower()
    assert "90" in text_b and "rate-limiting" in text_b.lower()
    assert "45" not in text_b, "channel B must not inherit channel A's retry_after"
    assert "90" not in text_a, "channel A must not inherit channel B's retry_after"
