"""Tests for ``cleanup_interim_segments``: once the turn-final answer is
confirmed delivered, best-effort delete every earlier segment bubble from
this turn (tool-call / commentary messages sent as their own messages) so
only the clean final answer remains visible.

Revived from a 6 Aug autostash after the 2026-09-18 update; the surrounding
transport code (``_stale_preview_ids`` / ``_delete_previews`` in
gateway/stream_consumer_transport.py) has since grown enough that the
cleanup helper is written directly against it rather than reinventing its
own id bookkeeping.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.delete_message = AsyncMock(return_value=True)
    return adapter


class TestCleanupInterimSegments:
    @pytest.mark.asyncio
    async def test_disabled_by_default_deletes_nothing(self):
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter=adapter, chat_id="chat", config=StreamConsumerConfig(),
        )
        consumer._message_id = "final"
        consumer._preview_message_ids = {"seg1", "seg2", "final"}
        await consumer._cleanup_interim_segment_messages()
        adapter.delete_message.assert_not_called()
        # Untouched when the feature is off.
        assert consumer._preview_message_ids == {"seg1", "seg2", "final"}

    @pytest.mark.asyncio
    async def test_enabled_deletes_every_id_except_the_final_message(self):
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter=adapter, chat_id="chat",
            config=StreamConsumerConfig(cleanup_interim_segments=True),
        )
        consumer._message_id = "final"
        consumer._preview_message_ids = {"seg1", "seg2", "final"}
        await consumer._cleanup_interim_segment_messages()
        deleted = {call.args[1] for call in adapter.delete_message.call_args_list}
        assert deleted == {"seg1", "seg2"}
        # Left-over bookkeeping reflects what's actually still on screen.
        assert consumer._preview_message_ids == {"final"}

    @pytest.mark.asyncio
    async def test_no_adapter_delete_support_is_a_silent_noop(self):
        adapter = _make_adapter()
        del adapter.delete_message  # adapter without the optional method
        consumer = GatewayStreamConsumer(
            adapter=adapter, chat_id="chat",
            config=StreamConsumerConfig(cleanup_interim_segments=True),
        )
        consumer._message_id = "final"
        consumer._preview_message_ids = {"seg1", "final"}
        await consumer._cleanup_interim_segment_messages()  # must not raise

    @pytest.mark.asyncio
    async def test_full_turn_multiple_segments_then_final_cleans_up(self):
        """End-to-end: two commentary segments (own bubbles) + a final edit,
        with cleanup enabled, leaves only the final message's id tracked and
        deletes the two superseded ones."""
        adapter = _make_adapter()
        message_ids = iter(["seg1", "seg2", "final"])
        adapter.send = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(success=True, message_id=next(message_ids))
        )
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="final"))
        consumer = GatewayStreamConsumer(
            adapter=adapter, chat_id="chat",
            config=StreamConsumerConfig(cleanup_interim_segments=True),
        )
        await consumer._deliver_commentary("Running tool A...")
        await consumer._deliver_commentary("Running tool B...")
        await consumer._send_or_edit("final answer", finalize=True)
        consumer._mark_final_delivered(record="final answer")
        await consumer._cleanup_interim_segment_messages()
        deleted = {call.args[1] for call in adapter.delete_message.call_args_list}
        assert deleted == {"seg1", "seg2"}
        assert consumer._preview_message_ids == {"final"}
