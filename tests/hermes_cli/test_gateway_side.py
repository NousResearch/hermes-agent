"""/bg and /btw on the shared gateway: the side task runs in the gateway (prompt.background /
prompt.btw) and its answer is routed off the event stream by task id, never built in the CLI."""
import asyncio
from contextlib import asynccontextmanager

import pytest

from hermes_cli import gateway_side
from hermes_cli.gateway_connection import GatewayConnection

SID = "local-side"


class _Client:
    def __init__(self, text):
        self.events = asyncio.Queue()
        self.calls = []
        self.text = text

    async def rpc(self, method, *, _timeout=None, **params):
        self.calls.append((method, params))
        kind = "background.complete" if method == "prompt.background" else "btw.complete"
        # The answer can land before the verb's reply reaches the caller.
        await self.events.put({"jsonrpc": "2.0", "method": "event", "params": {
            "type": kind, "session_id": SID, "payload": {"task_id": "bg_1", "text": self.text}}})
        return {"task_id": "bg_1"}


def _connection(client):
    @asynccontextmanager
    async def connect():
        yield client
    conn = GatewayConnection(connect=connect)
    conn.start(5)
    return conn


@pytest.mark.parametrize("verb", ["background", "btw"])
def test_side_task_runs_on_the_gateway_and_its_answer_is_routed_by_task_id(verb):
    client = _Client("DOG14 background answer")
    conn = _connection(client)
    try:
        task_id, wait = gateway_side.start(conn, SID, verb, "compute something")
        assert task_id == "bg_1"
        assert wait() == "DOG14 background answer"
        assert client.calls == [(f"prompt.{verb}", {"session_id": SID, "text": "compute something"})]
        assert conn.events.empty()  # never left for the turn pump / idle tick to drop
    finally:
        conn.close()


def test_a_failed_side_task_raises_the_gateways_error():
    conn = _connection(_Client("error: model unavailable"))
    try:
        _task_id, wait = gateway_side.start(conn, SID, "background", "x")
        with pytest.raises(RuntimeError, match="model unavailable"):
            wait()
    finally:
        conn.close()
