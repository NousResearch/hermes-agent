"""Regression tests: a turn that dies mid-stream (API error / interrupt / retry) must keep
every visible character of the streamed preview.

The finalize on failure used to be able to take the fresh-final lane (a fresh send plus a
best-effort DELETE of the live preview), so the message the user was reading vanished before
the retry.  The contract now is: append a short interruption note by editing the existing
message in place — never a fresh send, never a delete, never a delivery flag.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

NOTE = "⚠️ Interrupted (API error) — retrying…"


class _FreshFinalAdapter(MagicMock):
    """Telegram-like adapter: prefers the fresh-final send path on finalize."""

    def prefers_fresh_final_streaming(self, text, metadata=None, chat_id=None):  # noqa: ARG002
        return True


def _make_adapter() -> _FreshFinalAdapter:
    adapter = _FreshFinalAdapter()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="preview"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="preview"))
    adapter.delete_message = AsyncMock(return_value=True)
    return adapter


def _consumer(adapter) -> GatewayStreamConsumer:
    return GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat",
        config=StreamConsumerConfig(edit_interval=0.0, buffer_threshold=1),
    )


class TestInterruptedFinalizeKeepsPreview:
    @pytest.mark.asyncio
    async def test_interrupt_note_appends_in_place_never_deletes(self):
        """finish(interrupt_note=...) edits the preview with partial+note; no fresh send,
        no delete, and no delivery flag (the gateway still owns the failure notice)."""
        adapter = _make_adapter()
        consumer = _consumer(adapter)
        task = asyncio.create_task(consumer.run())
        consumer.on_delta("Partial answer the user is reading")
        await asyncio.sleep(0.2)
        consumer.finish(interrupt_note=NOTE)
        await asyncio.wait_for(task, timeout=5)

        # Exactly one send ever (the preview); the interrupt finalize never sends fresh.
        assert adapter.send.await_count == 1
        adapter.delete_message.assert_not_called()
        final_content = adapter.edit_message.await_args_list[-1].kwargs["content"]
        assert "Partial answer the user is reading" in final_content
        assert NOTE in final_content
        assert not consumer._final_response_sent
        assert not consumer._final_content_delivered
        assert consumer._suppress_fresh_final is True

    @pytest.mark.asyncio
    async def test_interrupt_note_with_no_preview_sends_nothing(self):
        """Nothing on screen (no message_id): the note is dropped — no orphan bubble."""
        adapter = _make_adapter()
        consumer = _consumer(adapter)
        task = asyncio.create_task(consumer.run())
        consumer.finish(interrupt_note=NOTE)
        await asyncio.wait_for(task, timeout=5)
        adapter.send.assert_not_called()
        adapter.edit_message.assert_not_called()
        adapter.delete_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancelled_task_finalizes_without_fresh_final_or_delete(self):
        """The cancel path (_on_cancelled) must not take the fresh-final lane either."""
        adapter = _make_adapter()
        consumer = _consumer(adapter)
        task = asyncio.create_task(consumer.run())
        consumer.on_delta("Partial under the user's eyes")
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.wait_for(task, timeout=5)
        assert adapter.send.await_count == 1  # preview only
        adapter.delete_message.assert_not_called()
        assert adapter.edit_message.await_count >= 1

    @pytest.mark.asyncio
    async def test_suppress_fresh_final_flag_blocks_fresh_final_lane(self):
        """_edit_existing honours _suppress_fresh_final even with a preferring adapter."""
        adapter = _make_adapter()
        consumer = _consumer(adapter)
        consumer._message_id = "preview"
        consumer._message_created_ts = 0.0
        consumer._last_sent_text = "old"
        consumer._suppress_fresh_final = True
        ok = await consumer._send_or_edit("brand new content", finalize=True, is_turn_final=False)
        assert ok is True
        adapter.send.assert_not_called()  # no fresh send
        adapter.edit_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_normal_finalize_still_takes_fresh_final(self):
        """Successful turns keep the legacy behaviour: fresh final + preview cleanup."""
        adapter = _make_adapter()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="fresh_final"))
        consumer = _consumer(adapter)
        consumer._message_id = "preview"
        consumer._message_created_ts = 0.0
        consumer._last_sent_text = "old"
        await consumer._send_or_edit("brand new content", finalize=True, is_turn_final=True)
        adapter.send.assert_awaited_once()  # the fresh final
        adapter.delete_message.assert_awaited_once()  # preview cleanup is wanted here
