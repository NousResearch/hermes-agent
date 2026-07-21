"""#68502: Telegram inbound update_id dedup prevents duplicate agent turns."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_update(update_id: int, text: str = "hello"):
    msg = MagicMock()
    msg.text = text
    msg.chat = MagicMock(id=123)
    msg.from_user = MagicMock(id=456)
    msg.message_id = 1
    return SimpleNamespace(update_id=update_id, message=msg, effective_message=msg)


def test_duplicate_update_id_is_suppressed():
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from gateway.config import PlatformConfig, Platform

    config = PlatformConfig(enabled=True, token="test", extra={})
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    # Minimal init for dedup fields only
    adapter._seen_update_ids = {}
    adapter._seen_update_ids_lock = __import__("threading").Lock()
    adapter._seen_update_ids_max = 4096

    update = _make_update(1001)
    assert adapter._is_duplicate_update(update) is False
    # Same update_id → suppressed
    assert adapter._is_duplicate_update(update) is True
    # Different update_id → not suppressed
    update2 = _make_update(1002)
    assert adapter._is_duplicate_update(update2) is False


def test_none_update_id_is_not_duplicate():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._seen_update_ids = {}
    adapter._seen_update_ids_lock = __import__("threading").Lock()
    adapter._seen_update_ids_max = 4096

    update = SimpleNamespace(update_id=None)
    assert adapter._is_duplicate_update(update) is False
    # Calling again still False — None never tracked
    assert adapter._is_duplicate_update(update) is False


def test_dedup_evicts_oldest_when_cap_exceeded():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._seen_update_ids = {}
    adapter._seen_update_ids_lock = __import__("threading").Lock()
    adapter._seen_update_ids_max = 3

    for i in range(3):
        adapter._is_duplicate_update(_make_update(i))
    assert len(adapter._seen_update_ids) == 3
    # Adding a 4th evicts the oldest (update_id=0)
    adapter._is_duplicate_update(_make_update(3))
    assert len(adapter._seen_update_ids) == 3
    assert 0 not in adapter._seen_update_ids
    assert 3 in adapter._seen_update_ids