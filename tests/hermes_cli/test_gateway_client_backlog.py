"""A client's own event backlog never fails its turn (#134889).

The frames a socket already buffered (a CLI resumed from Ctrl-Z) are read in one burst that never
yields to the loop, so the 4,096-frame client queue filled before its consumer had a turn: the
reader reported the turn's outcome as unknown, and the next prompt failed while the socket closed.
A Kanban worker, subscribed by ``kanban.run`` but reading only receipts, filled it the same way.
"""
import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from websockets.protocol import State

from hermes_cli import gateway_client
from hermes_cli.gateway_client import GatewayClient, GatewayClientError
from hermes_cli.gateway_connection import GatewayConnection

SID = "local-backlog"


def _frame(kind, text):
    return json.dumps({"jsonrpc": "2.0", "method": "event", "params": {
        "type": kind, "session_id": SID, "payload": {"text": text}}})


class Backlog:
    """A socket whose frames are all buffered already: iterating them never yields to the loop."""

    def __init__(self, frames):
        self.frames = frames

    async def __aiter__(self):
        for raw in self.frames:
            yield raw
        await asyncio.Event().wait()  # the socket stays open


def test_backlog_larger_than_the_client_queue_reaches_the_cli_intact():
    deltas = [str(i) for i in range(6000)]
    frames = [_frame("message.delta", text) for text in deltas] + [_frame("message.complete", "done")]

    @asynccontextmanager
    async def connect():
        async with GatewayClient(Backlog(frames)) as client:
            yield client

    conn = GatewayConnection(connect=connect)
    conn.start(timeout=5)
    try:
        seen = []
        while not seen or not (isinstance(seen[-1], Exception) or seen[-1]["params"]["type"] == "message.complete"):
            seen.append(conn.events.get(timeout=5))
    finally:
        conn.close()
    assert not isinstance(seen[-1], Exception), seen[-1]
    assert [frame["params"]["payload"]["text"] for frame in seen[:-1]] == deltas


@pytest.mark.asyncio
async def test_a_consumer_that_stops_draining_is_still_failed_with_the_queue_bounded(monkeypatch):
    """An awaited answer lets the reader read past a full queue, but only into a bounded overflow."""
    monkeypatch.setattr(gateway_client, "EVENT_STALL_TIMEOUT", 0.05, raising=False)
    monkeypatch.setattr(gateway_client, "EVENT_OVERFLOW_FRAMES", 100)
    client = GatewayClient(Backlog([_frame("message.delta", str(i)) for i in range(6000)]))
    pending = asyncio.get_running_loop().create_future()
    client.pending[1] = pending
    await asyncio.wait_for(client._read(), 5)
    with pytest.raises(GatewayClientError, match="outcome is unknown"):
        await pending
    assert client.events.full() and len(client.events.overflow) == 101  # 100 frames + the failure
    queued = [client.events.get_nowait() for _ in range(client.events.maxsize + 101)]
    assert client.events.empty() and isinstance(queued[-1], GatewayClientError)
    assert [frame["params"]["payload"]["text"] for frame in queued[:-1]] == [str(i) for i in range(4196)]


class CardOwner:
    """A Kanban card's owner as the worker's socket sees it: ``kanban.run`` subscribes the socket,
    the card's turn streams ``events`` frames, and the next receipt poll finds it terminal."""

    state = State.OPEN  # GatewayClient.rpc reads the socket state before it sends

    def __init__(self, events):
        self.events = events
        self.inbox = asyncio.Queue()

    async def send(self, raw):
        request = json.loads(raw)
        if request["method"] == "kanban.run":
            result = {"session_id": SID, "receipt": {"status": "running", "admission_id": "a1"}}
            frames = [{"jsonrpc": "2.0", "id": request["id"], "result": result}]
            frames += [json.loads(_frame("message.delta", str(i))) for i in range(self.events)]
        else:
            frames = [{"jsonrpc": "2.0", "id": request["id"], "result": {"status": "terminal", "admission_id": "a1"}}]
        for frame in frames:
            self.inbox.put_nowait(json.dumps(frame))

    async def __aiter__(self):
        while True:
            yield await self.inbox.get()


@pytest.mark.asyncio
async def test_kanban_worker_settles_a_card_whose_turn_outstreams_the_client_queue(monkeypatch, tmp_path):
    from gateway import session_kanban
    from hermes_cli import kanban_worker_client

    @asynccontextmanager
    async def connect():
        async with GatewayClient(CardOwner(6000)) as client:
            yield client

    monkeypatch.setattr(gateway_client, "connect_gateway", connect)
    monkeypatch.setattr(session_kanban, "worker_result", lambda *_: {"exit_code": 0, "last_output": "card closed"})
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    assert await asyncio.wait_for(kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db")), 5) == 0


@pytest.mark.asyncio
async def test_closing_as_a_waiting_frame_lands_still_ends_the_reader():
    """A close that lands just as the consumer makes room still ends the reader, so ``__aexit__``
    returns instead of waiting on the open socket forever."""
    client = await GatewayClient(Backlog([_frame("message.delta", str(i)) for i in range(4097)])).__aenter__()
    while not client.events.full():  # the reader yields only once it waits on the last frame
        await asyncio.sleep(0)
    client.events.get_nowait()
    closing = asyncio.create_task(client.__aexit__(None, None, None))
    try:
        await asyncio.wait_for(asyncio.shield(closing), 2)
    finally:
        client.reader.cancel()
        await asyncio.gather(closing, return_exceptions=True)
