"""Cold-boot pending-queue knob for the Telegram adapter.

Contract: a cold boot drops Telegram's server-side pending updates unless
``extra.drop_pending_on_cold_boot`` is false; a watcher reconnect always
preserves them. Conflict recovery is a separate path and is not covered here.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(extra=None) -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra or {}))


async def _capture_drop_pending(adapter: TelegramAdapter, *, is_reconnect: bool):
    """Run _start_polling_mode with staging mocked; return the forwarded flag."""
    captured = {}
    adapter._delete_webhook_best_effort = AsyncMock()

    async def _fake_resilient(*, drop_pending_updates, error_callback, require_progress=False):
        captured["drop_pending_updates"] = drop_pending_updates
        captured["require_progress"] = require_progress
        return True

    adapter._start_polling_resilient = _fake_resilient
    await adapter._start_polling_mode(is_reconnect=is_reconnect)
    return captured


@pytest.mark.asyncio
async def test_cold_boot_drops_queue_by_default():
    """Default config: cold boot drops, reconnect preserves."""
    adapter = _make_adapter()
    assert adapter._drop_pending_on_cold_boot is True

    cold = await _capture_drop_pending(adapter, is_reconnect=False)
    assert cold["drop_pending_updates"] is True

    warm = await _capture_drop_pending(adapter, is_reconnect=True)
    assert warm["drop_pending_updates"] is False


@pytest.mark.asyncio
async def test_cold_boot_preserves_queue_when_opted_out():
    """extra.drop_pending_on_cold_boot=false: cold boot preserves the backlog."""
    adapter = _make_adapter(extra={"drop_pending_on_cold_boot": False})
    assert adapter._drop_pending_on_cold_boot is False

    cold = await _capture_drop_pending(adapter, is_reconnect=False)
    assert cold["drop_pending_updates"] is False

    warm = await _capture_drop_pending(adapter, is_reconnect=True)
    assert warm["drop_pending_updates"] is False
