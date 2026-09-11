"""Tests for WhatsApp inbound message deduplication (#100481)."""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig


@pytest.mark.asyncio
async def test_whatsapp_dedup_ignores_duplicate_messages(tmp_path):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    session_path = tmp_path / "whatsapp_session"
    session_path.mkdir()
    config = PlatformConfig(enabled=True, extra={"session_path": str(session_path)})

    adapter = WhatsAppAdapter(config)

    assert not await adapter._is_duplicate("msg_123")
    assert await adapter._is_duplicate("msg_123")

    # Verify persistence file was written
    dedup_file = session_path / "whatsapp_seen_message_ids.json"
    assert dedup_file.exists()
    data = json.loads(dedup_file.read_text(encoding="utf-8"))
    assert "msg_123" in data.get("message_ids", {})

    # Create new adapter instance to verify loading from disk
    adapter2 = WhatsAppAdapter(config)
    assert await adapter2._is_duplicate("msg_123")
    assert not await adapter2._is_duplicate("msg_456")
