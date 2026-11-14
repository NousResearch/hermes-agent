"""Tests for WhatsAppAdapter.send_or_update_status (issue #30045).

Mirrors tests/gateway/test_telegram_status_update.py's structure. The
status-update path must:
  1. Send a fresh message on the first call for a (chat_id, status_key) pair.
  2. Edit that same message on subsequent calls with the same key.
  3. Fall back to sending fresh when the cached message edit fails.
  4. Keep distinct keys independent (no cross-talk).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


@pytest.fixture
def adapter():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    a = WhatsAppAdapter(PlatformConfig(enabled=True))
    # Patch send / edit_message so tests can drive them directly, same as
    # the Telegram fixture — no bridge process involved.
    a.send = AsyncMock()
    a.edit_message = AsyncMock()
    return a


@pytest.mark.asyncio
async def test_first_call_sends_and_caches_message_id(adapter):
    """First call for a (chat, key) pair must send and remember the id."""
    adapter.send.return_value = SendResult(success=True, message_id="wamid.100")

    result = await adapter.send_or_update_status("chat-1", "lifecycle", "starting")

    assert result.success is True
    assert result.message_id == "wamid.100"
    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_not_awaited()
    assert adapter._status_message_ids[("chat-1", "lifecycle")] == "wamid.100"


@pytest.mark.asyncio
async def test_second_call_edits_same_message(adapter):
    """Second call with the same key must edit, not send fresh."""
    adapter.send.return_value = SendResult(success=True, message_id="wamid.100")
    adapter.edit_message.return_value = SendResult(success=True, message_id="wamid.100")

    await adapter.send_or_update_status("chat-1", "lifecycle", "starting")
    result = await adapter.send_or_update_status("chat-1", "lifecycle", "in progress")

    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_awaited_once_with("chat-1", "wamid.100", "in progress")
    assert result.success is True
    assert result.message_id == "wamid.100"


@pytest.mark.asyncio
async def test_edit_failure_falls_back_to_fresh_send(adapter):
    """If the cached message's edit fails, drop the cache and send fresh."""
    adapter.send.side_effect = [
        SendResult(success=True, message_id="wamid.100"),
        SendResult(success=True, message_id="wamid.200"),
    ]
    adapter.edit_message.return_value = SendResult(success=False, error="message not found")

    await adapter.send_or_update_status("chat-1", "lifecycle", "starting")
    result = await adapter.send_or_update_status("chat-1", "lifecycle", "in progress")

    assert adapter.send.await_count == 2
    adapter.edit_message.assert_awaited_once_with("chat-1", "wamid.100", "in progress")
    assert result.success is True
    assert result.message_id == "wamid.200"
    assert adapter._status_message_ids[("chat-1", "lifecycle")] == "wamid.200"


@pytest.mark.asyncio
async def test_distinct_status_keys_do_not_collide(adapter):
    """A different status_key gets its own message; the original isn't touched."""
    adapter.send.side_effect = [
        SendResult(success=True, message_id="wamid.100"),
        SendResult(success=True, message_id="wamid.200"),
    ]

    await adapter.send_or_update_status("chat-1", "lifecycle", "ctx pressure")
    await adapter.send_or_update_status("chat-1", "model-switch", "switched to opus")

    assert adapter.send.await_count == 2
    adapter.edit_message.assert_not_awaited()
    assert adapter._status_message_ids[("chat-1", "lifecycle")] == "wamid.100"
    assert adapter._status_message_ids[("chat-1", "model-switch")] == "wamid.200"
