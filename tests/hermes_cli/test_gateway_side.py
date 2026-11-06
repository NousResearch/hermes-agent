"""/bg and /btw on the shared gateway: the side task runs in the gateway (prompt.background /
prompt.btw) and its answer is routed off the event stream by task id, never built in the CLI."""
import asyncio
from contextlib import asynccontextmanager

import pytest

from hermes_cli import gateway_side
from hermes_cli.gateway_client import GatewayRPCError
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


class _RefusingClient(_Client):
    """A client whose first side verb is refused: the socket holds no subscription for the
    session, so the owner answers ``permission_denied`` until the socket resumes it."""

    def __init__(self, text, reason="permission_denied"):
        super().__init__(text)
        self.reason = reason
        self.refused = False

    async def rpc(self, method, *, _timeout=None, **params):
        if method == "session.resume":
            self.calls.append((method, params))
            return {"session_id": SID, "stored_session_id": SID, "messages": [],
                    "subscription_id": "sub-1", "revision": 1, "execution_generation": 1}
        if not self.refused:
            self.refused = True
            self.calls.append((method, params))
            raise GatewayRPCError(self.reason)
        return await super().rpc(method, _timeout=_timeout, **params)


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


@pytest.mark.parametrize("reason", ["permission_denied", "not_found"])
def test_a_refused_side_task_reattaches_and_retries_once(reason):
    """The owner refuses a socket that holds no subscription for the session; one resume
    registers it, so the side task is admitted instead of surfacing as a refusal."""
    client = _RefusingClient("DOG14 answer after re-attach", reason)
    conn = _connection(client)
    try:
        task_id, wait = gateway_side.start(conn, SID, "btw", "compute something")
        assert task_id == "bg_1"
        assert wait() == "DOG14 answer after re-attach"
        methods = [method for method, _ in client.calls]
        assert methods == ["prompt.btw", "session.resume", "prompt.btw"]
    finally:
        conn.close()


def test_a_refusal_that_a_reattach_cannot_clear_is_raised():
    class _Gone(_Client):
        async def rpc(self, method, *, _timeout=None, **params):
            self.calls.append((method, params))
            raise GatewayRPCError("session_busy")

    client = _Gone("x")
    conn = _connection(client)
    try:
        with pytest.raises(GatewayRPCError, match="session_busy"):
            gateway_side.start(conn, SID, "btw", "compute something")
        assert [method for method, _ in client.calls] == ["prompt.btw"]
    finally:
        conn.close()


def test_a_refusal_never_surfaces_as_a_traceback():
    assert "no longer attached" in gateway_side.refusal_message(GatewayRPCError("permission_denied"))
    assert "no longer knows" in gateway_side.refusal_message(GatewayRPCError("not_found"))
    assert "session_busy" in gateway_side.refusal_message(GatewayRPCError("session_busy"))
