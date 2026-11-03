"""The viewer RPC client keeps an owner refusal's structured verdict (code + bounded data)."""
import asyncio
import json

import pytest

from websockets.exceptions import ConnectionClosedError
from websockets.protocol import State

from hermes_cli.gateway_client import NOT_SENT, GatewayClient, GatewayClientError, GatewayRPCError


class _Socket:
    """One scripted owner: answers each request with the frame ``reply(request)`` builds."""

    state = State.OPEN  # GatewayClient.rpc reads the socket state before it sends

    def __init__(self, reply):
        self.reply = reply
        self.inbox = asyncio.Queue()

    async def send(self, raw):
        frame = self.reply(json.loads(raw))
        if frame is not None:
            await self.inbox.put(json.dumps(frame))

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.inbox.get()


@pytest.mark.asyncio
async def test_rpc_error_keeps_numeric_code_and_bounded_data():
    errors = {
        'session.mutate': {'code': 4001, 'message': 'invalid_params', 'data': {
            'reason': 'invalid_params', 'fields': ['payload.title'],
            'trace': {'nested': 'remote diagnostics'}, 'blob': 'x' * 5000}},
        'nope': {'code': -32601, 'message': 'unknown method: nope', 'data': {'reason': 'unknown_method'}},
    }
    socket = _Socket(lambda req: {'jsonrpc': '2.0', 'id': req['id'], 'error': errors[req['method']]})
    async with GatewayClient(socket) as client:
        with pytest.raises(GatewayRPCError) as refused:
            await client.rpc('session.mutate', session_id='s')
        assert str(refused.value) == 'invalid_params' and refused.value.code == 4001
        # Scalar facts survive; nested objects and oversized strings never reach the caller.
        assert refused.value.data == {'reason': 'invalid_params', 'fields': ['payload.title']}
        with pytest.raises(GatewayRPCError) as unknown:
            await client.rpc('nope')
        # Free-text messages stay bounded: the owner's reason code, not the raw sentence.
        assert str(unknown.value) == 'unknown_method' and unknown.value.code == -32601


@pytest.mark.asyncio
async def test_compress_waits_past_the_default_rpc_budget(monkeypatch):
    """A canonical ``/compress`` answers only after its summary + commit (minutes on a long
    session). The viewer must not report failure at the 30 s default while the owner, which
    shields the mutation, still commits it. Time is scaled 1000x: 30 s -> 30 ms."""
    from hermes_cli.gateway_mutations import PreparedMutations
    real_wait_for = asyncio.wait_for
    monkeypatch.setattr(asyncio, 'wait_for', lambda fut, timeout: real_wait_for(fut, timeout / 1000))
    sent = []

    def reply(req):
        sent.append(req)
        if req['method'] == 'session.resume':
            return {'id': req['id'], 'result': {'revision': 1, 'execution_generation': 1}}

        async def commit():  # a 120 s summary (scaled) then the receipt
            await asyncio.sleep(0.12)  # health: allow HX013 -- the reply must land after the scaled 30 ms default budget expired
            await owner.inbox.put(json.dumps({'id': req['id'], 'result': {'status': 'applied'}}))
        asyncio.ensure_future(commit())

    owner = _Socket(reply)
    async with GatewayClient(owner) as client:
        result = await PreparedMutations().apply(client, 's', 'compress', {})
    assert result == {'status': 'applied'}
    # The budget is a client-side knob, never a wire parameter the closed contract would refuse.
    assert all('_timeout' not in req['params'] for req in sent)


@pytest.mark.asyncio
async def test_send_on_a_dead_socket_is_a_client_error_not_a_raw_websockets_one():
    """A keepalive timeout closes the socket under a poller (the dock's ``subagent.list``): the send
    must surface as the transport's own error, which every caller catches, never as websockets'."""
    from websockets.exceptions import ConnectionClosedError

    class _Dead(_Socket):
        async def send(self, raw):
            raise ConnectionClosedError(None, None)

    async with GatewayClient(_Dead(lambda req: None)) as client:
        with pytest.raises(GatewayClientError, match='turn outcome is unknown'):
            await client.rpc('subagent.list', session_id='s')
        assert client.pending == {}


@pytest.mark.asyncio
async def test_a_request_on_a_closing_socket_is_not_sent():
    """A keepalive timeout fails the socket (CLOSING) and websockets waits up to ``close_timeout`` for
    the peer before the reader ends. A send in that window is never written either."""
    sent = []

    class _Closing(_Socket):
        state = State.CLOSING

        async def send(self, raw):
            sent.append(raw)
            raise ConnectionClosedError(None, None)

    async with GatewayClient(_Closing(lambda req: None)) as client:
        assert not client.reader.done()
        with pytest.raises(GatewayClientError) as refused:
            await client.rpc('prompt.submit', session_id='s')
        assert str(refused.value) == NOT_SENT
        assert sent == [] and client.pending == {}


@pytest.mark.asyncio
async def test_a_request_after_the_reader_ended_is_not_sent():
    """The keepalive timeout closes the socket and ends the reader before the dock's next poll: that
    send is never written (websockets refuses a send on a closed socket), so the caller is told
    nothing was sent, not that a turn's outcome is unknown."""
    sent = []

    class _Closed(_Socket):
        async def send(self, raw):  # websockets: a send on a closed socket raises, nothing written
            sent.append(raw)
            raise ConnectionClosedError(None, None)

        async def __anext__(self):
            raise StopAsyncIteration  # the socket closed: iteration ends

    async with GatewayClient(_Closed(lambda req: None)) as client:
        await asyncio.wait({client.reader}, timeout=1)  # the reader sees the close
        assert client.reader.done()
        with pytest.raises(GatewayClientError) as refused:
            await client.rpc('prompt.submit', session_id='s')
        assert str(refused.value) == NOT_SENT
        assert sent == [] and client.pending == {}


@pytest.mark.asyncio
async def test_a_send_that_fails_after_the_reader_failed_its_request_logs_nothing():
    """websockets' send waits for the connection to be lost before it raises, so the reader fails the
    pending request first; rpc raises its own error and must retrieve that one, or asyncio logs
    "Future exception was never retrieved"."""
    import gc

    class _DiesMidSend(_Socket):
        def __init__(self, reply):
            super().__init__(reply)
            self.closed = asyncio.Event()

        async def send(self, raw):
            self.closed.set()
            await asyncio.wait({self.client.reader}, timeout=1)  # the reader fails the request first
            raise ConnectionClosedError(None, None)

        async def __anext__(self):
            await self.closed.wait()
            raise ConnectionClosedError(None, None)

    loop = asyncio.get_running_loop()
    logged, previous = [], loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: logged.append(context.get('message')))
    try:
        socket = _DiesMidSend(lambda req: None)
        async with GatewayClient(socket) as client:
            socket.client = client
            with pytest.raises(GatewayClientError):
                await client.rpc('subagent.list', session_id='s')
            assert client.reader.done()  # the reader failed the request before the send raised
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous)
    assert not [m for m in logged if 'never retrieved' in str(m)]
