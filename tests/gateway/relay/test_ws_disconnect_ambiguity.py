"""A missing relay acknowledgement is not a rejection (#137963)."""

import asyncio
import contextlib
import json

import pytest

from gateway.relay.ws_transport import WebSocketRelayTransport
from tests.gateway.relay.test_relay_live_cards import _connected_adapter


class _BoundarySocket:
    def __init__(self, *, fail_write=False):
        self.frames = []
        self.write_started = asyncio.Event()
        self.release_write = asyncio.Event()
        self.incoming = asyncio.Queue()
        self.fail_write = fail_write

    async def send(self, data):
        self.write_started.set()
        if self.fail_write:
            await self.release_write.wait()
            raise ConnectionResetError("write refused")
        self.frames.append(json.loads(data))

    def __aiter__(self):
        return self

    async def __anext__(self):
        frame = await self.incoming.get()
        if frame is None:
            raise StopAsyncIteration
        return json.dumps(frame) + "\n"

    async def close(self):
        self.incoming.put_nowait(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["eof", "reader_cancel", "disconnect", "rejection"])
async def test_written_draft_retains_seal_obligation_until_confirmed_rejection(ending):
    adapter, connector = _connected_adapter()
    transport = WebSocketRelayTransport("ws://unused", "slack", "bot", outbound_timeout_s=30)
    socket = _BoundarySocket()
    transport._ws = socket
    transport._reader = asyncio.create_task(transport._read_loop())
    adapter._transport = transport
    metadata = {"message_id": "turn-1"}
    send = asyncio.create_task(adapter.send_draft("C1", 7, "partial", metadata=metadata))
    try:
        await asyncio.wait_for(socket.write_started.wait(), timeout=2)
        assert len(socket.frames) == 1
        if ending == "rejection":
            socket.incoming.put_nowait({
                "type": "outbound_result", "requestId": socket.frames[0]["requestId"],
                "result": {"success": False, "error": "connector rejected draft"},
            })
        elif ending == "disconnect":
            await transport.disconnect(budget_s=0)
        elif ending == "reader_cancel":
            transport._reader.cancel()
        else:
            await socket.close()
        result = await asyncio.wait_for(send, timeout=2)
        assert not result.success
        assert transport._pending == {}
        assert bool(adapter._open_draft_by_chat) is (ending != "rejection")
        # The next usable connection must seal the existing stream, not send a duplicate.
        adapter._transport = connector
        final = await adapter.send("C1", "complete", metadata=dict(metadata))
        assert final.success
        frame = connector.sent[-1]
        assert frame["op"] == ("send" if ending == "rejection" else "draft")
        if ending != "rejection":
            assert frame["final"] is True and frame["draft_id"] == 7
    finally:
        send.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await send
        await transport.disconnect(budget_s=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_sender", [False, True])
async def test_reader_loss_during_unfinished_write_does_not_claim_delivery(cancel_sender):
    transport = WebSocketRelayTransport("ws://unused", "slack", "bot", outbound_timeout_s=30)
    socket = _BoundarySocket(fail_write=True)
    transport._ws = socket
    transport._reader = asyncio.create_task(transport._read_loop())
    send = asyncio.create_task(transport.send_outbound({"op": "draft"}))
    try:
        await asyncio.wait_for(socket.write_started.wait(), timeout=2)
        await socket.close()
        await asyncio.wait_for(transport._reader, timeout=2)
        if cancel_sender:
            send.cancel()
            with pytest.raises(asyncio.CancelledError):
                await send
        else:
            socket.release_write.set()
            result = await asyncio.wait_for(send, timeout=2)
            assert result["success"] is False
            assert "ambiguous" not in result
        assert transport._pending == {} and socket.frames == []
        result = await transport.send_outbound({"op": "draft"})
        assert result == {"success": False, "error": "relay transport not connected"}
    finally:
        send.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await send
        await transport.disconnect(budget_s=0)
