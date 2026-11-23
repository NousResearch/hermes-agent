"""Expected keepalive close preserves an honest unknown outcome and no task traceback."""
import argparse
import asyncio
from contextlib import nullcontext
import json

import pytest
from websockets.exceptions import ConnectionClosedError

from hermes_cli.gateway_client import GatewayClient, GatewayClientError


@pytest.mark.asyncio
async def test_keepalive_close_releases_waiters_without_unhandled_reader_exception():
    class Disconnected:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ConnectionClosedError(None, None)

    client = GatewayClient(Disconnected())
    pending = asyncio.get_running_loop().create_future()
    client.pending[1] = pending
    await client._read()
    with pytest.raises(GatewayClientError, match='outcome is unknown'):
        await pending
    assert 'do not resend' in str(await client.events.get())


@pytest.mark.asyncio
async def test_idle_composer_exits_with_unknown_outcome_when_owner_dies_without_keypress(monkeypatch):
    """The owner dying while the REPL waits for input must end the viewer at once with the
    unknown-outcome hint; the pending line read is cancelled rather than awaited forever."""
    from websockets.asyncio.server import serve

    from hermes_cli import gateway_chat

    snapshot = {"stored_session_id": "stored", "execution_generation": 1, "pending": []}
    attached = asyncio.Event()
    reads = []

    async def peer(ws):
        async for raw in ws:
            request = json.loads(raw)
            result = snapshot if request["method"] == "session.resume" else {}
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}))

    class IdleInput:
        def __init__(self, **_kwargs):
            pass

        async def prompt_async(self, _prompt):
            attached.set()
            try:
                await asyncio.Future()  # nobody ever presses a key
            except asyncio.CancelledError:
                reads.append("cancelled")
                raise

    monkeypatch.setattr("prompt_toolkit.PromptSession", IdleInput)
    monkeypatch.setattr("prompt_toolkit.patch_stdout.patch_stdout", nullcontext)
    async with serve(peer, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", f"ws://127.0.0.1:{port}")
        viewer = asyncio.create_task(gateway_chat.run_gateway_chat(argparse.Namespace(resume="stored")))
        await asyncio.wait_for(attached.wait(), 5)
        server.close(close_connections=True)
        with pytest.raises(GatewayClientError, match="outcome is unknown"):
            await asyncio.wait_for(viewer, 3)
    assert reads == ["cancelled"]
