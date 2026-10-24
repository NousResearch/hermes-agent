"""An awaited answer stays reachable past the client's event-overflow cap (#134889 F5).

The owner attaches a re-attached subscription before it answers ``session.resume``, so a burst larger
than the client's queue can sit ahead of the answer. Read past the cap, the reader used to park for
the consumer — which is the task awaiting that very answer — spend its stall budget and fail the RPC
as ``Gateway disconnected; turn outcome is unknown``, a message about a socket that was healthy.
The answer arrives; the frames the socket can no longer hold are declared as the one
``session.replay_gap`` they are, so the consumer re-attaches and replays the hole from the store.
"""
import asyncio
import json

import pytest

from hermes_cli import gateway_client
from hermes_cli.gateway_client import GatewayClient, GatewayClientError

QUEUE = 4096  # GatewayClient's own event queue (``EventQueue(maxsize=4096)``)
GAP = "session.replay_gap"


def event(seq, sid="s"):
    return {"jsonrpc": "2.0", "method": "event", "params": {
        "session_id": sid, "type": "message.delta", "replay_epoch": "e", "seq": seq,
        "payload": {"text": str(seq)}}}


class Wire:
    """A resume whose frames all land ahead of its answer (the owner's fanout holds 16,384 frames)."""

    def __init__(self, events):
        self.events = events
        self.frames = asyncio.Queue()

    async def send(self, raw):
        request = json.loads(raw)
        assert request["method"] == "session.resume"
        for frame in self.events:
            self.frames.put_nowait(frame)
        self.frames.put_nowait({"jsonrpc": "2.0", "id": request["id"], "result": {"resumed": "s"}})

    async def __aiter__(self):
        while True:
            yield json.dumps(await self.frames.get())


@pytest.mark.asyncio
async def test_a_resume_answer_behind_more_than_the_frame_cap_is_still_read():
    cap = gateway_client.EVENT_OVERFLOW_FRAMES
    wire = Wire([event(seq) for seq in range(2, 2 + QUEUE + cap + 10)])
    async with GatewayClient(wire) as client:
        answer = asyncio.create_task(client.rpc("session.resume", session_id="s"))
        # Well under EVENT_STALL_TIMEOUT: a reader parked for the consumer fails this RPC instead.
        assert await asyncio.wait_for(answer, 10) == {"resumed": "s"}
        assert client.events.qsize() == QUEUE, "the queue is full; the cap bounds what sits behind it"
        held = len(client.events.overflow)
        assert held == cap + 1, f"{held} held: {cap} frames + the single gap notice that replaces the rest"
        queued = [client.events.get_nowait() for _ in range(QUEUE + held)]
    assert not any(isinstance(frame, GatewayClientError) for frame in queued)
    assert queued[-1]["params"]["session_id"] == "s" and queued[-1]["params"]["type"] == GAP
    assert [frame["params"]["payload"]["text"] for frame in queued[:-1]] == [
        str(seq) for seq in range(2, 2 + QUEUE + cap)]


@pytest.mark.asyncio
async def test_frames_past_the_cap_are_declared_once_per_session(monkeypatch):
    """A shared overflow buries frames from several sessions: each gets one notice, not one per frame."""
    monkeypatch.setattr(gateway_client, "EVENT_OVERFLOW_FRAMES", 100)
    burst = []
    for seq in range(2, 2 + QUEUE + 150):
        burst += [event(seq, "s"), event(seq, "t")]
    wire = Wire(burst)
    async with GatewayClient(wire) as client:
        answer = asyncio.create_task(client.rpc("session.resume", session_id="s"))
        assert await asyncio.wait_for(answer, 10) == {"resumed": "s"}
        held = len(client.events.overflow)
        queued = [client.events.get_nowait() for _ in range(client.events.qsize() + held)]
    assert held == 102, f"{held} held: 100 frames + one notice for each session"
    assert [frame["params"]["session_id"] for frame in queued if frame["params"].get("type") == GAP] == ["s", "t"]
    assert not any(isinstance(frame, GatewayClientError) for frame in queued)
