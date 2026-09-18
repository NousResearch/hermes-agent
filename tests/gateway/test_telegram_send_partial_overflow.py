"""send() must record landed chunks on a mid-split failure (#114396).

Premise: a mid-split failure in send() returns total failure with chunks
1..N-1 unrecorded, while the edit path already reports ``partial_overflow``.
These tests prove the gap for both mid-split failure modes: a verbatim
SendResult from _send_chunk_with_retries, and a raised exception.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter


CONTENT = "word " * 120


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    object.__setattr__(adapter, "MAX_MESSAGE_LENGTH", 160)
    return adapter


def _ok(message_id: int):
    return (SimpleNamespace(message_id=message_id), False)


@pytest.mark.asyncio
async def test_send_mid_split_sendresult_failure_records_delivered_chunks():
    """A verbatim failure SendResult keeps delivered ids + undelivered tail."""
    adapter = _adapter()
    failure = SendResult(success=False, error="flood_control:97.0", retry_after=97.0)
    with patch.object(
        TelegramAdapter, "_send_chunk_with_retries",
        new=AsyncMock(side_effect=[_ok(111), failure]),
    ):
        result = await adapter.send("12345", CONTENT, metadata={"expect_edits": True})

    assert result.success is False
    assert result.error == "flood_control:97.0"
    raw = result.raw_response
    assert isinstance(raw, dict) and raw.get("partial_overflow") is True
    assert raw["delivered_chunks"] == 1
    assert raw["total_chunks"] >= 2
    assert raw["last_message_id"] == "111"
    assert result.message_id == "111"
    assert raw["undelivered_tail"]


@pytest.mark.asyncio
async def test_send_mid_split_exception_records_delivered_chunks():
    """A raise on chunk 2 keeps chunk 1 recorded instead of total failure."""
    adapter = _adapter()
    with patch.object(
        TelegramAdapter, "_send_chunk_with_retries",
        new=AsyncMock(side_effect=[_ok(111), RuntimeError("boom")]),
    ):
        result = await adapter.send("12345", CONTENT, metadata={"expect_edits": True})

    assert result.success is False
    raw = result.raw_response
    assert isinstance(raw, dict) and raw.get("partial_overflow") is True
    assert raw["delivered_chunks"] == 1
    assert raw["total_chunks"] >= 2
    assert raw["last_message_id"] == "111"
    assert raw["undelivered_tail"]
