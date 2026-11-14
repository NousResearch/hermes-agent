"""``GatewayConnection.call_until`` sends its one request on a socket of its own (#134889 PLG-7): never
on the viewer's socket, never twice, and the socket is closed on the reply, on Ctrl+C (``wake``), on
``close()`` (CLI exit) and when the dial fails."""
import asyncio
from contextlib import asynccontextmanager
import threading

import pytest

from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.gateway_connection import CallStopped, GatewayConnection


class _Client:
    def __init__(self, hold, sent):
        self.events = asyncio.Queue()
        self.calls, self.closed, self.hold, self.sent = [], False, hold, sent

    async def rpc(self, method, *, _timeout=None, **params):
        self.calls.append((method, params, _timeout))
        self.sent.set()
        if self.hold:
            await asyncio.Event().wait()  # a handler that never returns; only a cancel ends it
        return {'type': 'exec', 'output': 'done'}


class _Gateway:
    """Every dial gets a new client: the first is the viewer's, the next ones command sockets."""
    def __init__(self, hold=False, refuse_commands=False):
        self.clients, self.hold, self.refuse = [], hold, refuse_commands
        self.command_sent = threading.Event()

    @asynccontextmanager
    async def dial(self):
        if self.refuse and self.clients:
            raise GatewayClientError('Gateway connection failed; no local fallback')
        command = bool(self.clients)
        client = _Client(self.hold and command, self.command_sent if command else threading.Event())
        self.clients.append(client)
        try:
            yield client
        finally:
            client.closed = True


def _threads():
    return [t for t in threading.enumerate() if t.name == 'hermes-gateway-command']


def _waiting(conn, wake):
    """``call_until`` on a thread: ``(thread, outcome)``, the outcome filled when it returns."""
    outcome = {}

    def run():
        try:
            outcome['reply'] = conn.call_until('slash.exec', wake, session_id='s', command='slow')
        except GatewayClientError as exc:
            outcome['error'] = exc
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def test_the_reply_comes_over_a_socket_of_its_own_that_then_closes():
    gateway = _Gateway()
    conn = GatewayConnection(connect=gateway.dial)
    try:
        reply = conn.call_until('slash.exec', threading.Event(), session_id='s', command='quick')
        viewer, command = gateway.clients
        assert reply == {'type': 'exec', 'output': 'done'}
        assert viewer.calls == [] and not viewer.closed  # the viewer's socket stays free and open
        assert command.calls == [('slash.exec', {'session_id': 's', 'command': 'quick'}, float('inf'))]
        assert command.closed and not _threads() and not conn._commands
    finally:
        conn.close()


@pytest.mark.parametrize('stop', ['ctrl_c', 'close'])
def test_a_stopped_wait_closes_the_command_socket_and_never_sends_again(stop):
    gateway = _Gateway(hold=True)
    conn = GatewayConnection(connect=gateway.dial)
    conn.start(5)
    wake = threading.Event()
    thread, outcome = _waiting(conn, wake)
    try:
        assert gateway.command_sent.wait(10)
        if stop == 'ctrl_c':
            wake.set()
        else:  # CLI exit
            conn.close()
        thread.join(10)
    finally:
        conn.close()
    assert not thread.is_alive()
    assert isinstance(outcome.get('error'), CallStopped), outcome
    viewer, command = gateway.clients
    assert viewer.calls == [] and len(command.calls) == 1 and command.closed
    assert not _threads() and not conn._commands


def test_a_failed_command_dial_sends_nothing_and_is_not_retried():
    gateway = _Gateway(refuse_commands=True)
    conn = GatewayConnection(connect=gateway.dial)
    try:
        with pytest.raises(GatewayClientError, match='connection failed') as raised:
            conn.call_until('slash.exec', threading.Event(), session_id='s', command='x')
        assert not isinstance(raised.value, CallStopped)
        assert len(gateway.clients) == 1 and gateway.clients[0].calls == []
        assert not _threads() and not conn._commands
    finally:
        conn.close()
