"""Buzz forward-only unix-socket dispatch (item 4)."""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from tests.gateway.buzz_forward_support import (
    CHANNEL,
    BuzzAdapter,
    ForwardServer,
    _encode_len_prefixed,
    _event,
    forward_adapter,
    short_sock_path,
)

pytest_plugins = ["tests.gateway.buzz_forward_support"]


class TestForwardOnlyMode:
    @pytest.mark.asyncio
    async def test_dispatch_writes_frame_and_skips_agent(self, forward_socket):
        adapter = forward_adapter(forward_socket.path)
        event = _event("fwd-1", content="@Chip hello", created_at=50)
        event["sig"] = "signature-bytes"
        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)
        await asyncio.sleep(0.05)
        assert adapter.handle_message.await_count == 0
        assert adapter._message_handler.await_count == 0
        assert len(forward_socket.frames) == 1
        frame = forward_socket.frames[0]
        assert frame["direction"] == "inbound"
        assert frame["channel_id"] == CHANNEL
        assert frame["event"]["id"] == "fwd-1"
        assert frame["event"]["sig"] == "signature-bytes"
        assert "fwd-1" in adapter._channel_state[CHANNEL]["seen"]

    @pytest.mark.asyncio
    async def test_missing_ack_does_not_advance_cursor(self):
        server = ForwardServer(short_sock_path(), hang=True)
        server.start()
        adapter = forward_adapter(server.path)
        state = adapter._channel_state[CHANNEL]
        event = _event("lost-1", content="@Chip hello", created_at=80)
        await adapter._handle_event(CHANNEL, state, event)
        assert "lost-1" not in state["seen"]
        assert state["last_ts"] == 0
        assert adapter._forward_delivery_unknown is True
        server.close()

    @pytest.mark.asyncio
    async def test_missing_socket_refuses_connect(self, tmp_path):
        adapter = forward_adapter(str(tmp_path / "missing.sock"))
        adapter.cli_path = str(tmp_path / "buzz")
        Path(adapter.cli_path).write_text("x", encoding="utf-8")
        ok = await adapter.connect()
        assert ok is False
        assert adapter._fatal_error_code == "config_missing"

    def test_slash_and_handle_message_are_unreachable_in_source(self):
        source = inspect.getsource(BuzzAdapter._dispatch_message)
        handle_index = source.find("await self.handle_message")
        forward_index = source.find("if self._forward_only")
        assert forward_index != -1
        assert handle_index != -1
        assert forward_index < handle_index
        wire_source = inspect.getsource(BuzzAdapter.connect)
        assert "if not self._forward_only:" in wire_source
        assert "_wire_plugin_handlers" in wire_source

    @pytest.mark.asyncio
    async def test_unacked_event_is_refetched_after_later_ack(self):
        server = ForwardServer(short_sock_path(), hang_ids={"evt-a"})
        server.start()
        adapter = forward_adapter(server.path)
        state = adapter._channel_state[CHANNEL]
        evt_a = _event("evt-a", content="@Chip a", created_at=10)
        evt_b = _event("evt-b", content="@Chip b", created_at=20)
        await adapter._handle_events(CHANNEL, state, [evt_a, evt_b])
        assert "evt-a" not in state["seen"]
        assert "evt-b" in state["seen"]
        assert state["last_ts"] < 10
        server.close()

        server2 = ForwardServer(short_sock_path())
        server2.start()
        adapter._forward_socket = server2.path
        await adapter._handle_events(CHANNEL, state, [evt_a, evt_b])
        ids = [frame["event"]["id"] for frame in server2.frames]
        assert "evt-a" in ids
        assert "evt-b" not in ids
        assert "evt-a" in state["seen"]
        server2.close()

    @pytest.mark.asyncio
    async def test_forward_only_skips_reaction_only_handling(self):
        from unittest.mock import AsyncMock

        from tests.gateway.test_buzz_adapter import AGENT_PUBKEY, SELF_PUBKEY

        adapter = forward_adapter(short_sock_path())
        adapter.send_reaction = AsyncMock(return_value=True)
        adapter._allowed_pubkeys = {CHANNEL}
        adapter._reaction_only_pubkeys = {AGENT_PUBKEY}
        adapter._self_pubkey = SELF_PUBKEY
        event = _event("react-1", pubkey=AGENT_PUBKEY, content="@Chip coordinate", created_at=11)
        event["tags"].append(["p", SELF_PUBKEY])
        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)
        adapter.send_reaction.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,reason",
        [("accepted", None), ("duplicate", None), ("rejected", "schema")],
    )
    async def test_ack_status_advances_cursor(self, status, reason, caplog):
        payload = {"ack": "ack-1", "status": status}
        if reason is not None:
            payload["reason"] = reason
        server = ForwardServer(short_sock_path(), replies=[_encode_len_prefixed(payload)])
        server.start()
        adapter = forward_adapter(server.path)
        state = adapter._channel_state[CHANNEL]
        event = _event("ack-1", content="@Chip hello", created_at=40)
        await adapter._handle_event(CHANNEL, state, event)
        assert "ack-1" in state["seen"]
        assert state["last_ts"] == 40
        if status == "rejected":
            assert "forward rejected by consumer reason=schema" in caplog.text
        server.close()
