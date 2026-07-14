"""Discord visible-fork thread creation and cleanup behavior."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest


def _adapter_for_parent(monkeypatch, parent, forum_type):
    from plugins.platforms.discord import adapter as adapter_mod
    from plugins.platforms.discord.adapter import DiscordAdapter

    client = SimpleNamespace(
        get_channel=lambda _channel_id: parent,
        fetch_channel=AsyncMock(return_value=parent),
    )
    discord_stub = SimpleNamespace(ForumChannel=forum_type)
    monkeypatch.setattr(adapter_mod, "DISCORD_AVAILABLE", True)
    monkeypatch.setattr(adapter_mod, "discord", discord_stub)

    adapter: Any = object.__new__(DiscordAdapter)
    adapter._client = client
    return adapter, client


@pytest.mark.asyncio
async def test_create_visible_fork_thread_supports_forum_channels(monkeypatch):
    class FakeForumChannel:
        create_thread: Any

    parent: Any = FakeForumChannel()
    parent.create_thread = AsyncMock(
        return_value=SimpleNamespace(thread=SimpleNamespace(id=777))
    )
    adapter, client = _adapter_for_parent(monkeypatch, parent, FakeForumChannel)

    thread_id = await adapter.create_visible_fork_thread("123", "Fork Lane")

    assert thread_id == "777"
    parent.create_thread.assert_awaited_once_with(
        name="Fork Lane",
        content="⑂ Hermes visible fork: **Fork Lane**",
        auto_archive_duration=1440,
        reason="Hermes visible session fork",
    )
    client.fetch_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_visible_fork_thread_rejects_non_forum_parent(monkeypatch):
    class FakeForumChannel:
        pass

    class FakeTextChannel:
        create_thread: Any

    parent: Any = FakeTextChannel()
    parent.create_thread = AsyncMock()
    adapter, _client = _adapter_for_parent(monkeypatch, parent, FakeForumChannel)

    with pytest.raises(RuntimeError, match="requires a forum thread"):
        await adapter.create_visible_fork_thread("123", "Fork Lane")

    parent.create_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_visible_fork_thread_removes_unbound_forum_post(monkeypatch):
    class FakeForumChannel:
        pass

    thread = SimpleNamespace(delete=AsyncMock())
    adapter, client = _adapter_for_parent(monkeypatch, thread, FakeForumChannel)

    removed = await adapter.delete_visible_fork_thread("777")

    assert removed is True
    thread.delete.assert_awaited_once_with(reason="Hermes visible fork setup failed")
    client.fetch_channel.assert_not_awaited()
