"""Minimal e2e tests for Discord mention stripping + /command detection.

Covers the fix for slash commands not being recognized when sent via
@mention in a channel, especially after auto-threading.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.e2e.conftest import (
    BOT_USER_ID,
    E2E_MESSAGE_SETTLE_DELAY,
    get_response_text,
    make_discord_message,
    make_fake_dm_channel,
    make_fake_thread,
)

pytestmark = pytest.mark.asyncio


async def dispatch(adapter, msg):
    await adapter._handle_message(msg)
    await asyncio.sleep(E2E_MESSAGE_SETTLE_DELAY)


class TestMentionStrippedCommandDispatch:
    async def test_mention_then_command(self, discord_adapter, bot_user):
        """<@BOT> /help → mention stripped, /help dispatched."""
        msg = make_discord_message(
            content=f"<@{BOT_USER_ID}> /help",
            mentions=[bot_user],
        )
        await dispatch(discord_adapter, msg)
        response = get_response_text(discord_adapter)
        assert response is not None
        assert "/new" in response

    async def test_nickname_mention_then_command(self, discord_adapter, bot_user):
        """<@!BOT> /help → nickname mention also stripped, /help works."""
        msg = make_discord_message(
            content=f"<@!{BOT_USER_ID}> /help",
            mentions=[bot_user],
        )
        await dispatch(discord_adapter, msg)
        response = get_response_text(discord_adapter)
        assert response is not None
        assert "/new" in response

    async def test_text_before_command_not_detected(self, discord_adapter, bot_user):
        """'<@BOT> something else /help' → mention stripped, but 'something else /help'
        doesn't start with / so it's treated as text, not a command."""
        msg = make_discord_message(
            content=f"<@{BOT_USER_ID}> something else /help",
            mentions=[bot_user],
        )
        await dispatch(discord_adapter, msg)
        # Message is accepted (not dropped by mention gate), but since it doesn't
        # start with / it's routed as text — no command output, and no agent in this
        # mock setup means no send call either.
        response = get_response_text(discord_adapter)
        assert response is None or "/new" not in response

    async def test_no_mention_in_channel_dropped(self, discord_adapter):
        """Message without @mention in server channel → silently dropped."""
        msg = make_discord_message(content="/help", mentions=[])
        await dispatch(discord_adapter, msg)
        assert get_response_text(discord_adapter) is None

    async def test_dm_no_mention_needed(self, discord_adapter):
        """DMs don't require @mention — /help works directly."""
        dm = make_fake_dm_channel()
        msg = make_discord_message(content="/help", channel=dm, mentions=[])
        await dispatch(discord_adapter, msg)
        response = get_response_text(discord_adapter)
        assert response is not None
        assert "/new" in response


class TestAutoThreadingPreservesCommand:
    async def test_command_detected_after_auto_thread(self, discord_adapter, bot_user, monkeypatch):
        """@mention /help in channel with auto-thread → thread created AND command dispatched."""
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
        fake_thread = make_fake_thread(thread_id=90001, name="help")
        msg = make_discord_message(
            content=f"<@{BOT_USER_ID}> /help",
            mentions=[bot_user],
        )

        # Simulate discord.py restoring the original raw content (with mention)
        # after create_thread(), which undoes any prior mention stripping.
        original_content = msg.content

        async def clobber_content(**kwargs):
            msg.content = original_content
            return fake_thread

        msg.create_thread = AsyncMock(side_effect=clobber_content)
        await dispatch(discord_adapter, msg)

        msg.create_thread.assert_awaited_once()
        response = get_response_text(discord_adapter)
        assert response is not None
        assert "/new" in response


class TestRepliedToMediaDispatch:
    async def test_reply_to_image_message_caches_referenced_attachment(
        self, discord_adapter, bot_user, monkeypatch
    ):
        """A text reply to an image-bearing Discord message should give the agent that image."""
        cached_path = "/tmp/replied-discord-image.png"

        async def fake_cache_image_from_url(url, *, ext=".jpg"):
            assert url == "https://cdn.discordapp.com/attachments/image.png"
            assert ext == ".png"
            return cached_path

        monkeypatch.setattr(
            "plugins.platforms.discord.adapter.cache_image_from_url",
            fake_cache_image_from_url,
        )
        discord_adapter.handle_message = AsyncMock()

        attachment = SimpleNamespace(
            content_type="image/png",
            filename="image.png",
            url="https://cdn.discordapp.com/attachments/image.png",
            size=1234,
        )
        referenced_message = SimpleNamespace(
            id=12345,
            content="",
            attachments=[attachment],
        )
        msg = make_discord_message(
            content=f"<@{BOT_USER_ID}> what's in this image?",
            mentions=[bot_user],
        )
        msg.type = 19
        msg.reference = SimpleNamespace(message_id=12345, resolved=referenced_message)

        await discord_adapter._handle_message(msg)

        discord_adapter.handle_message.assert_awaited_once()
        await_args = discord_adapter.handle_message.await_args
        assert await_args is not None
        event = await_args.args[0]
        assert event.reply_to_message_id == "12345"
        assert event.media_urls == [cached_path]
        assert event.media_types == ["image/png"]
        assert event.message_type.value == "photo"


class TestGuildSubscriptionPriming:
    """on_ready must wait for every guild to report ``available=True``
    before letting ``on_message`` through, otherwise MESSAGE_CREATE for
    new guilds is silently dropped until the next reconnect race (#58866).
    """

    async def test_wait_until_all_guilds_ready_returns_when_all_available(
        self, discord_adapter,
    ):
        client = discord_adapter._client
        if client is None:
            pytest.skip("discord_adapter fixture did not wire a client")
        # Replace the guild list with two mocks that are already available.
        class _G:
            def __init__(self, gid, available):
                self.id = gid
                self.available = available
        client.guilds = [_G("111", True), _G("222", True)]
        # No sleep — returns immediately on the first poll.
        await discord_adapter._wait_until_all_guilds_ready()

    async def test_wait_until_all_guilds_ready_returns_when_no_guilds(
        self, discord_adapter,
    ):
        """Zero-guild account (DM-only bot) must not block startup."""
        client = discord_adapter._client
        if client is None:
            pytest.skip("discord_adapter fixture did not wire a client")
        client.guilds = []
        await discord_adapter._wait_until_all_guilds_ready()

    async def test_wait_until_all_guilds_ready_times_out_and_warns(
        self, discord_adapter, monkeypatch,
    ):
        """If ``Guild.available`` never flips True within the timeout we
        still proceed (logged warning + return) instead of blocking startup
        forever. Fixture patches the constants to keep the test fast.
        """
        client = discord_adapter._client
        if client is None:
            pytest.skip("discord_adapter fixture did not wire a client")

        class _G:
            def __init__(self, gid):
                self.id = gid
                self.available = False
        client.guilds = [_G("999")]

        monkeypatch.setattr(
            discord_adapter, "_GUILD_READY_POLL_INTERVAL_S", 0.0,
        )
        monkeypatch.setattr(
            discord_adapter, "_GUILD_READY_TIMEOUT_S", 0.05,
        )
        # Should NOT raise; just timeout + return.
        await discord_adapter._wait_until_all_guilds_ready()
        assert client.guilds[0].available is False  # unchanged

    async def test_wait_until_all_guilds_ready_polled_until_available(
        self, discord_adapter, monkeypatch,
    ):
        """A guild flipping available after a couple of polls must clear
        the wait. Verifies the poll loop itself is wired correctly.
        """
        client = discord_adapter._client
        if client is None:
            pytest.skip("discord_adapter fixture did not wire a client")

        class _G:
            def __init__(self, gid):
                self.id = gid
                self.available = False

        class _FlippableGuild:
            def __init__(self):
                self.id = "flippable"
                self.available = False
            def flip(self):
                self.available = True

        flippable = _FlippableGuild()
        client.guilds = [flippable]

        # Flip available=True on the third poll iteration.
        polls = {"n": 0}
        real_sleep = discord_adapter._GUILD_READY_POLL_INTERVAL_S
        monkeypatch.setattr(discord_adapter, "_GUILD_READY_POLL_INTERVAL_S", 0.0)

        async def _maybe_flip():
            polls["n"] += 1
            if polls["n"] >= 3:
                flippable.flip()
        monkeypatch.setattr(
            "asyncio.sleep",
            lambda _delay: _maybe_flip(),
        )  # asyncio.sleep is a top-level import in adapter.py
        monkeypatch.setattr(
            discord_adapter, "_GUILD_READY_TIMEOUT_S", 5.0,
        )

        await discord_adapter._wait_until_all_guilds_ready()
        assert flippable.available is True

        # Restore real sleep constant so the test process doesn't leave
        # asyncio.sleep patched for any later test in the same session.
        monkeypatch.setattr(
            discord_adapter, "_GUILD_READY_POLL_INTERVAL_S", real_sleep,
        )
