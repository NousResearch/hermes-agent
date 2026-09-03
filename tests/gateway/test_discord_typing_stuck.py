"""Regression test for the Discord typing-indicator stuck-on race.

The Discord adapter's ``send_typing`` starts a persistent background loop per
chat that POSTs ``/channels/{id}/typing`` every 12s; ``stop_typing`` cancels
it. The loop's ``finally`` previously popped the registry entry
unconditionally::

    finally:
        self._typing_tasks.pop(chat_id, None)

Race: ``stop_typing()`` pops the entry and cancels the old loop, but before
the old loop's ``finally`` runs, a fresh ``send_typing()`` (from the tool
progress path in ``gateway/run.py`` or the base ``_keep_typing`` refresh)
re-registers a new loop for the same chat. The old loop's ``finally`` then
pops the *new* loop out of the registry, orphaning it — it keeps POSTing
``/typing`` every 12s forever and ``stop_typing()`` can never cancel it again,
so Discord shows "is typing…" permanently.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


def _make_adapter() -> DiscordAdapter:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=None))
    return adapter


@pytest.mark.asyncio
async def test_cancelled_typing_loop_does_not_orphan_newer_loop() -> None:
    """A cancelled loop's ``finally`` must not pop a re-registered sibling.

    Reproduces the interleaving directly: stop_typing's pop happens first,
    a new send_typing re-registers loop B, then loop A's cancellation unwinds
    through its ``finally``. With the bug, A's ``finally`` pops B and B is
    orphaned (still tracked nowhere, still POSTing /typing, unstoppable).
    """
    adapter = _make_adapter()
    chat_id = "123456789012345678"

    # Signal once loop A has actually started POSTing, so the later cancel()
    # lands while A is inside its 12s sleep — not before the coroutine body
    # (and therefore its finally) has even begun.
    a_started = asyncio.Event()

    async def _request(_route) -> None:
        a_started.set()

    adapter._client.http.request = _request

    # Turn 1: send_typing registers loop A.
    await adapter.send_typing(chat_id)
    loop_a = adapter._typing_tasks[chat_id]
    assert loop_a is not None

    # Let loop A start and reach its first POST, then settle into sleep(12).
    await asyncio.wait_for(a_started.wait(), timeout=1.0)
    await asyncio.sleep(0)

    # stop_typing pops the entry, then cancels. We emulate the pop explicitly
    # so we can interleave a re-registration the way the real code does.
    popped = adapter._typing_tasks.pop(chat_id)
    assert popped is loop_a

    # A fresh send_typing re-registers loop B before A's finally runs.
    await adapter.send_typing(chat_id)
    loop_b = adapter._typing_tasks[chat_id]
    assert loop_b is not None and loop_b is not loop_a

    # Now A's cancellation unwinds through its finally.
    loop_a.cancel()
    try:
        await loop_a
    except asyncio.CancelledError:
        pass

    # B must still be tracked — otherwise it is orphaned and unstoppable.
    assert adapter._typing_tasks.get(chat_id) is loop_b

    # And stop_typing can still cancel B cleanly.
    await adapter.stop_typing(chat_id)
    assert adapter._typing_tasks.get(chat_id) is None


@pytest.mark.asyncio
async def test_stop_typing_clears_registry_in_normal_path() -> None:
    """The non-raced path still clears the registry entry."""
    adapter = _make_adapter()
    chat_id = "123456789012345678"

    async def _request(_route) -> None:
        return None

    adapter._client.http.request = _request

    await adapter.send_typing(chat_id)
    assert chat_id in adapter._typing_tasks

    await adapter.stop_typing(chat_id)
    assert chat_id not in adapter._typing_tasks
