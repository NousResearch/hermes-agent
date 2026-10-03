"""Empty/invalid Buzz allowlist denies every sender in forward-only mode (item 5)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tests.gateway.buzz_forward_support import CHANNEL, SELF_PUBKEY, _event, _make_adapter

pytest_plugins = ["tests.gateway.buzz_forward_support"]


class TestEmptyAllowlistForwardDeny:
    @pytest.mark.asyncio
    async def test_invalid_only_allowlist_denies_every_sender(self, forward_socket):
        adapter = _make_adapter(
            {
                "forward_only": True,
                "forward_socket": forward_socket.path,
                "allowed_users": ["not-a-key", "npub1zzzz"],
                "allowed_destinations": [CHANNEL],
                "channels": [CHANNEL],
            }
        )
        adapter._self_pubkey = SELF_PUBKEY
        adapter._channel_state[CHANNEL] = adapter._new_channel_state("group")
        adapter._input_scope = {CHANNEL}
        adapter.handle_message = AsyncMock()
        assert adapter._allowed_pubkeys == set()
        await adapter._handle_event(
            CHANNEL, adapter._channel_state[CHANNEL], _event("x1", content="@Chip hi", created_at=9)
        )
        assert adapter.handle_message.await_count == 0
        assert forward_socket.frames == []
        assert "x1" in adapter._channel_state[CHANNEL]["seen"]
