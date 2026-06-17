"""Stale streamed-preview cleanup after a gateway final re-send.

Regression tests for the 2026-06-12 Telegram duplicate-answer bug: the
finalize edit hit flood control, the gateway re-sent the full response,
and the raw streamed preview (asterisks + cursor) was left on screen
alongside the fresh copy.

GatewayRunner registers the preview message ids via
``register_stale_stream_preview``; the adapter deletes them only after a
successful final send and drops unconsumed registrations otherwise.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class StubAdapter(BasePlatformAdapter):
    """Minimal concrete adapter with delete tracking."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake"), Platform.DISCORD)
        self.deleted = []

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="msg1")

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def delete_message(self, chat_id, message_id):
        self.deleted.append((str(chat_id), str(message_id)))
        return True


def test_register_filters_sentinel_and_empty_ids():
    adapter = StubAdapter()
    adapter.register_stale_stream_preview("c1", ["10", None, "__no_edit__", "11"])
    assert adapter._stale_stream_previews == {"c1": ["10", "11"]}


def test_register_with_no_usable_ids_is_a_noop():
    adapter = StubAdapter()
    adapter.register_stale_stream_preview("c1", [None, "__no_edit__"])
    assert adapter._stale_stream_previews == {}


def test_delete_consumes_registration_and_deletes_all():
    adapter = StubAdapter()
    adapter.register_stale_stream_preview("c1", ["10", "11"])
    asyncio.run(adapter._delete_stale_stream_preview("c1"))
    assert adapter.deleted == [("c1", "10"), ("c1", "11")]
    assert adapter._stale_stream_previews == {}


def test_delete_without_registration_is_a_noop():
    adapter = StubAdapter()
    asyncio.run(adapter._delete_stale_stream_preview("c1"))
    assert adapter.deleted == []


def test_delete_survives_platform_errors():
    adapter = StubAdapter()

    async def _failing_delete(chat_id, message_id):
        adapter.deleted.append((str(chat_id), str(message_id)))
        raise RuntimeError("message too old")

    adapter.delete_message = _failing_delete
    adapter.register_stale_stream_preview("c1", ["10", "11"])
    asyncio.run(adapter._delete_stale_stream_preview("c1"))
    # Both deletes attempted despite the first raising; registry drained.
    assert adapter.deleted == [("c1", "10"), ("c1", "11")]
    assert adapter._stale_stream_previews == {}
