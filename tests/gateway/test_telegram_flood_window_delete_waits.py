"""The delete wait loop: a window re-armed while it sleeps is re-checked, never slept inline past the
cap, and a window that closes during the last allowed wait still gets its delete."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.plugins.platforms.telegram import telegram_flood_window as fw
from hermes_agent.plugins.platforms.telegram.adapter import _FLOOD_INLINE_WAIT_CAP_SECS, TelegramAdapter
from hermes_agent.plugins.platforms.telegram.telegram_media_flood import _MAX_WINDOW_WAITS

_real_sleep = asyncio.sleep


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.delete_message = AsyncMock(return_value=True)
    return adapter


def _scripted_sleep(monkeypatch, on_wake):
    """Each sleep the delete loop takes returns at once; ``on_wake(n)`` is what other in-flight
    traffic did to the window meanwhile."""
    slept = []

    async def sleep(seconds, *a, **k):
        if seconds:
            slept.append(seconds)
            on_wake(len(slept))
        await _real_sleep(0)
    monkeypatch.setattr(fw.asyncio, "sleep", sleep)
    return slept


@pytest.mark.asyncio
async def test_a_window_closing_during_the_last_allowed_wait_still_deletes(monkeypatch):
    adapter = _adapter()
    adapter._record_send_flood_cooldown("4242", 1.0)

    def wake(n):
        if n <= _MAX_WINDOW_WAITS:
            adapter._record_send_flood_cooldown("4242", 1.0)  # re-armed by a sibling request
        else:
            adapter.__dict__["_telegram_send_cooldown_until"].clear()  # closed during the last sleep
    slept = _scripted_sleep(monkeypatch, wake)

    assert await adapter.delete_message("4242", "55") is True
    assert len(slept) == _MAX_WINDOW_WAITS + 1
    adapter._bot.delete_message.assert_awaited_once_with(chat_id=4242, message_id=55)


@pytest.mark.asyncio
async def test_a_window_extended_during_an_inline_wait_moves_to_the_background(monkeypatch):
    adapter = _adapter()
    adapter._record_send_flood_cooldown("4242", 1.0)  # short: waited inline
    slept = _scripted_sleep(monkeypatch, lambda n: adapter._record_send_flood_cooldown("4242", 240.0))

    assert await adapter.delete_message("4242", "55") is True  # owned by the background task now
    assert len(slept) == 1 and slept[0] <= _FLOOD_INLINE_WAIT_CAP_SECS  # never slept the long window inline
    assert adapter._bot.delete_message.await_count == 0
    assert [t for t in adapter._background_tasks if not t.done()]
    await adapter.cancel_background_tasks()
