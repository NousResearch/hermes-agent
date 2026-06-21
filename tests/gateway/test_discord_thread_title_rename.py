"""Tests for syncing session titles to Discord thread names."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def _make_discord_thread_source(*, thread_id: str = "4242") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=thread_id,
        chat_name="old thread name",
        chat_type="thread",
        user_id="12345",
        user_name="tester",
        thread_id=thread_id,
        parent_chat_id="999",
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    adapter = SimpleNamespace(rename_conversation=AsyncMock(return_value=True))
    runner.adapters = {Platform.DISCORD: adapter}
    runner._session_db = MagicMock()
    return runner


@pytest.mark.asyncio
async def test_auto_generated_title_renames_visible_conversation_via_adapter():
    runner = _make_runner()
    source = _make_discord_thread_source(thread_id="4242")

    await runner._rename_visible_conversation_for_session_title(
        source,
        "sess-discord",
        "  Discord   Thread Naming UX  ",
    )

    runner.adapters[Platform.DISCORD].rename_conversation.assert_awaited_once_with(
        source,
        "  Discord   Thread Naming UX  ",
        session_id="sess-discord",
        session_db=runner._session_db,
    )


@pytest.mark.asyncio
async def test_schedule_visible_conversation_rename_uses_running_loop():
    runner = _make_runner()
    source = _make_discord_thread_source(thread_id="777")
    seen = []

    async def _spy(copied_source, session_id, title):
        seen.append((copied_source, session_id, title))

    runner._rename_visible_conversation_for_session_title = _spy

    runner._schedule_visible_conversation_title_rename(
        source,
        "sess-discord",
        "Auto Generated Title",
    )
    await asyncio.sleep(0.05)

    assert len(seen) == 1
    copied_source, session_id, title = seen[0]
    assert copied_source is not source
    assert copied_source.thread_id == "777"
    assert session_id == "sess-discord"
    assert title == "Auto Generated Title"


def test_discord_adapter_sanitizes_thread_title():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)

    title = adapter._sanitize_conversation_title("  " + "A" * 140 + "  ")

    assert len(title) == 100
    assert title.endswith("...")


@pytest.mark.asyncio
async def test_discord_adapter_rename_conversation_edits_fetched_thread():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    fake_thread = SimpleNamespace(edit=AsyncMock())
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=None),
        fetch_channel=AsyncMock(return_value=fake_thread),
    )

    source = _make_discord_thread_source(thread_id="4242")
    assert await adapter.rename_conversation(source, "Readable Session Title") is True

    adapter._client.fetch_channel.assert_awaited_once_with(4242)
    fake_thread.edit.assert_awaited_once_with(
        name="Readable Session Title",
        reason="Hermes session auto-title",
    )


@pytest.mark.asyncio
async def test_discord_adapter_rename_conversation_skips_non_thread():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter._client = SimpleNamespace()
    source = _make_discord_thread_source()
    source.chat_type = "group"
    source.thread_id = None

    assert await adapter.rename_conversation(source, "Should Not Rename") is False
