"""DiscordAdapter.send_typing / stop_typing lifecycle (typing-indicator leak).

Reported symptom: a channel kept showing "typing…" long after the reply was delivered.
Cause: ``_typing_loop``'s ``finally`` did an unconditional ``_typing_tasks.pop(chat_id)``, so a
cancelled loop A could delete the entry of a replacement loop B created by a late
``send_typing()``. B was then unreachable — nothing could cancel it — and kept POSTing typing
every 12 s for the life of the process.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import plugins.platforms.discord.adapter_typing as typing_mod
from gateway.config import Platform, PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter

CHAT = "1234567890"


@pytest.fixture(autouse=True)
def _fast_typing_interval(monkeypatch):
    """Keep the refresh cadence in the code, but make the test's tick 20 ms."""
    monkeypatch.setattr(typing_mod, "_TYPING_POST_INTERVAL_SECONDS", 0.02)


def _adapter():
    """Return an adapter wired to a fake Discord client."""
    client = MagicMock()
    client.http = MagicMock()
    client.http.request = AsyncMock()
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, token="x", extra={})
    adapter._client = client
    adapter._typing_tasks = {}
    adapter._typing_epoch = {}
    return adapter, client


@pytest.mark.asyncio
async def test_send_then_stop_ends_the_loop():
    adapter, client = _adapter()

    await adapter.send_typing(CHAT)
    task = adapter._typing_tasks.get(CHAT)
    assert task is not None

    await asyncio.sleep(0.01)
    assert client.http.request.await_count >= 1

    await adapter.stop_typing(CHAT)
    assert CHAT not in adapter._typing_tasks
    assert task.done()

    posts = client.http.request.await_count
    await asyncio.sleep(0.05)
    assert client.http.request.await_count == posts


@pytest.mark.asyncio
async def test_loop_finally_pops_only_its_own_entry():
    """A cancelled loop must not delete its replacement's map entry."""
    adapter, client = _adapter()

    await adapter.send_typing(CHAT)
    abandoned = adapter._typing_tasks[CHAT]

    replacement = asyncio.ensure_future(asyncio.sleep(1))
    adapter._typing_tasks[CHAT] = replacement

    abandoned.cancel()
    await asyncio.sleep(0.01)

    assert abandoned.done()
    assert adapter._typing_tasks.get(CHAT) is replacement
    replacement.cancel()


@pytest.mark.asyncio
async def test_orphaned_loop_exits_after_stop():
    """A loop whose map entry was lost must still die on the next stop_typing()."""
    adapter, client = _adapter()

    await adapter.send_typing(CHAT)
    orphan = adapter._typing_tasks.pop(CHAT)  # entry lost, loop still alive
    await asyncio.sleep(0.01)
    assert not orphan.done()

    await adapter.stop_typing(CHAT)  # nothing in the map; the epoch bump must reach it
    await asyncio.sleep(0.1)
    assert orphan.done()
