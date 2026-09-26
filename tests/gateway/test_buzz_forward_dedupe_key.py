"""Forward frame dedupe_key (item 7)."""
from __future__ import annotations

import asyncio
import hashlib

import pytest

from tests.gateway.buzz_forward_support import (
    CHANNEL,
    _event,
    _forward_dedupe_key,
    forward_adapter,
)

pytest_plugins = ["tests.gateway.buzz_forward_support"]


class TestDedupeKey:
    def test_sha256_of_event_direction_destination(self):
        event_id = "abc"
        direction = "inbound"
        destination = CHANNEL
        expected = hashlib.sha256(f"{event_id}{direction}{destination}".encode("utf-8")).hexdigest()
        assert _forward_dedupe_key(event_id, direction, destination) == expected

    @pytest.mark.asyncio
    async def test_frame_carries_dedupe_key(self, forward_socket):
        adapter = forward_adapter(forward_socket.path)
        event = _event("dedupe-1", content="@Chip hello", created_at=11)
        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)
        await asyncio.sleep(0.05)
        frame = forward_socket.frames[0]
        assert frame["dedupe_key"] == _forward_dedupe_key("dedupe-1", "inbound", CHANNEL)
