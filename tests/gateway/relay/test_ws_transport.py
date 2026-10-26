"""WebSocketRelayTransport against a real in-process WebSocket server.

Exercises the production transport over an actual ``websockets`` server (no
mock socket): handshake (hello -> descriptor), inbound frame -> handler,
outbound request/response correlation, and follow_up routing. Proves the wire
framing (newline-delimited JSON) and the request/response future plumbing work
end to end on a live socket.

Skipped cleanly if the optional ``websockets`` dependency is absent.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import pytest_asyncio

from gateway.relay.ws_transport import WebSocketRelayTransport, WEBSOCKETS_AVAILABLE

pytestmark = pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")

if WEBSOCKETS_AVAILABLE:
    import websockets


DESCRIPTOR = {
    "contract_version": 1,
    "platform": "discord",
    "label": "Discord",
    "max_message_length": 2000,
    "supports_draft_streaming": False,
    "supports_edit": True,
    "supports_threads": True,
    "markdown_dialect": "discord",
    "len_unit": "chars",
}


class _StubConnectorServer:
    """Minimal connector: answers hello with a descriptor, echoes outbound."""

    def __init__(self):
        self.received: list[dict] = []
        self._server = None
        self.url = ""
        # Push channel: tests set this to a frame dict to deliver inbound.
        self._to_push: list[dict] = []

    async def start(self):
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        sock = next(iter(self._server.sockets))
        port = sock.getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws):
        async for raw in ws:
            for line in str(raw).split("\n"):
                if not line.strip():
                    continue
                frame = json.loads(line)
                self.received.append(frame)
                await self._on_frame(ws, frame)

    async def _on_frame(self, ws, frame):
        ftype = frame.get("type")
        if ftype == "hello":
            await ws.send(json.dumps({"type": "descriptor", "descriptor": DESCRIPTOR}) + "\n")
            # Deliver any queued inbound frames right after handshake.
            for f in self._to_push:
                await ws.send(json.dumps(f) + "\n")
        elif ftype == "outbound":
            action = frame.get("action", {})
            # Echo a successful result correlated by requestId.
            result = {"success": True, "message_id": f"srv-{action.get('op')}"}
            await ws.send(
                json.dumps({"type": "outbound_result", "requestId": frame["requestId"], "result": result})
                + "\n"
            )


@pytest_asyncio.fixture
async def server():
    srv = _StubConnectorServer()
    await srv.start()
    yield srv
    await srv.stop()


@pytest.mark.asyncio
async def test_handshake_negotiates_descriptor(server):
    t = WebSocketRelayTransport(server.url, "discord", "appShared")
    await t.connect()
    try:
        desc = await t.handshake()
        assert desc.platform == "discord"
        assert desc.max_message_length == 2000
        # The hello carried the platform + botId.
        hello = next(f for f in server.received if f["type"] == "hello")
        assert hello["platform"] == "discord"
        assert hello["botId"] == "appShared"
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_inbound_frame_reaches_handler(server):
    server._to_push = [
        {
            "type": "inbound",
            "event": {
                "text": "hello from connector",
                "message_type": "text",
                "source": {"platform": "discord", "chat_id": "chan1", "chat_type": "group", "scope_id": "guildA"},
            },
            "bufferId": "buf-1",
        }
    ]
    received = []
    t = WebSocketRelayTransport(server.url, "discord", "appShared")
    t.set_inbound_handler(lambda ev: received.append(ev) or asyncio.sleep(0))
    await t.connect()
    try:
        await t.handshake()
        # Give the reader a tick to deliver the pushed inbound frame.
        await asyncio.sleep(0.05)
        assert len(received) == 1
        assert received[0].text == "hello from connector"
        assert received[0].source.scope_id == "guildA"
    finally:
        await t.disconnect()


# ── Phase 7 Unit 7d-B: terminal 4401 (opt-out revocation) ────────────────────


class _Revoking4401Server:
    """Connector stub that, on hello, optionally sends a descriptor and then
    closes the socket with application code 4401 (unauthorized) — the shape of a
    connector that has revoked this gateway's per-gateway secret (opt-out)."""

    def __init__(self, *, send_descriptor_first: bool):
        self._server = None
        self.url = ""
        self._send_descriptor_first = send_descriptor_first

    async def start(self):
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        port = next(iter(self._server.sockets)).getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws):
        async for raw in ws:
            for line in str(raw).split("\n"):
                if not line.strip():
                    continue
                frame = json.loads(line)
                if frame.get("type") == "hello":
                    if self._send_descriptor_first:
                        await ws.send(
                            json.dumps({"type": "descriptor", "descriptor": DESCRIPTOR}) + "\n"
                        )
                        # Let the descriptor flush + be processed before the close.
                        await asyncio.sleep(0.05)
                    # Close with 4401 (the connector's "unauthorized" close).
                    await ws.close(code=4401, reason="unauthorized")
                    return


@pytest.mark.asyncio
async def test_4401_after_handshake_is_terminal_no_reconnect():
    """A 4401 close AFTER a successful handshake = a revoked credential (opt-out):
    the transport latches auth_revoked and does NOT spin the reconnect supervisor.

    Since the expired-token fix, the transport first re-dials ONCE with a fresh
    token; this stub 4401s every dial, so that retry is also refused and the
    latch still lands — the terminal contract is unchanged."""
    srv = _Revoking4401Server(send_descriptor_first=True)
    await srv.start()
    try:
        t = WebSocketRelayTransport(
            srv.url, "discord", "appShared",
            gateway_id="gw-x", upgrade_secret="secret-x",
            reconnect=True, reconnect_backoff_s=0.05,
        )
        await t.connect()
        await t.handshake()  # records _handshake_succeeded
        # Wait for the server's 4401 close to propagate through the read loop.
        for _ in range(100):
            if t.auth_revoked:
                break
            await asyncio.sleep(0.02)
        assert t.auth_revoked is True
        # Terminal: no reconnect supervisor was spawned.
        assert t._supervisor is None
        # Give a reconnect (if it were going to happen) time to NOT happen.
        await asyncio.sleep(0.2)
        assert t._supervisor is None
    finally:
        await t.disconnect()
        await srv.stop()


# ── Expired upgrade token vs revoked secret (incident 2026-09-02) ─────────────


class _ExpiredToken4401Server:
    """Connector stub for the expired-vs-revoked ambiguity. Every dial gets a
    descriptor on hello (the upgrade itself is never refused). After the
    descriptor the stub closes 4401 on the dials whose 1-based index is in
    ``close_4401_on`` (``None`` = every dial) with the given reason; other dials
    stay open and serve as a normal connector. ``dials`` counts connections."""

    def __init__(self, *, close_4401_on: set[int] | None, reason: str = "unauthorized"):
        self._server = None
        self.url = ""
        self.dials = 0
        self._close_on = close_4401_on
        self._reason = reason

    async def start(self):
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        port = next(iter(self._server.sockets)).getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws):
        self.dials += 1
        dial = self.dials
        async for raw in ws:
            for line in str(raw).split("\n"):
                if not line.strip():
                    continue
                frame = json.loads(line)
                if frame.get("type") == "hello":
                    await ws.send(json.dumps({"type": "descriptor", "descriptor": DESCRIPTOR}) + "\n")
                    if self._close_on is None or dial in self._close_on:
                        await asyncio.sleep(0.05)  # let the descriptor land first
                        await ws.close(code=4401, reason=self._reason)
                        return


def _transport(url: str) -> WebSocketRelayTransport:
    return WebSocketRelayTransport(
        url, "discord", "appShared",
        gateway_id="gw-x", upgrade_secret="secret-x",
        reconnect=True, reconnect_backoff_s=5.0,  # slow: proves the retry bypasses backoff
    )


async def _wait_until(pred, *, timeout_s: float = 2.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return pred()


@pytest.mark.asyncio
async def test_4401_after_handshake_redials_once_with_fresh_token_and_recovers():
    """A single post-handshake 4401 (an expired upgrade token after a scale-to-zero
    suspend) must NOT latch revocation: the transport re-dials immediately with a
    fresh token, the connector accepts it, and the relay stays live."""
    srv = _ExpiredToken4401Server(close_4401_on={1})
    await srv.start()
    t = _transport(srv.url)
    try:
        await t.connect()
        await t.handshake()
        assert srv.dials == 1
        # The 4401 close lands, the fresh-token re-dial connects (well inside the
        # 5s reconnect backoff, so this is the immediate retry, not the supervisor).
        assert await _wait_until(lambda: srv.dials == 2 and t._descriptor is not None)
        assert t.auth_revoked is False
        # The new reader is running against a live socket; no fatal state.
        assert t._reader is not None and not t._reader.done()
        assert t._ws is not None
        assert t._supervisor is None
        await asyncio.sleep(0.2)
        assert srv.dials == 2  # exactly one re-dial, no extra spinning
        assert t.auth_revoked is False
    finally:
        await t.disconnect()
        await srv.stop()


@pytest.mark.asyncio
async def test_4401_on_fresh_token_redial_latches_revocation_after_exactly_one_retry():
    """If the connector 4401s the FRESH token too, the secret really is gone:
    latch auth_revoked after exactly one retry (2 dials total) and stop."""
    srv = _ExpiredToken4401Server(close_4401_on=None)
    await srv.start()
    t = _transport(srv.url)
    try:
        await t.connect()
        await t.handshake()
        assert await _wait_until(lambda: t.auth_revoked)
        assert srv.dials == 2
        assert t._supervisor is None
        await asyncio.sleep(0.2)
        assert srv.dials == 2  # terminal: no further dials
        assert t._supervisor is None
    finally:
        await t.disconnect()
        await srv.stop()


@pytest.mark.asyncio
async def test_4401_with_reason_expired_never_latches_and_reconnects_normally():
    """A connector that labels the close reason 'expired' is explicitly saying the
    token aged out: never a revocation, even repeatedly — take the normal
    reconnect path (backoff supervisor) instead."""
    srv = _ExpiredToken4401Server(close_4401_on={1}, reason="expired")
    await srv.start()
    t = _transport(srv.url)
    t._reconnect_backoff_s = 0.05  # normal path: should reconnect via the supervisor
    try:
        await t.connect()
        await t.handshake()
        assert await _wait_until(lambda: t._supervisor is not None)
        assert t.auth_revoked is False
        assert await _wait_until(lambda: srv.dials == 2 and t._descriptor is not None)
        assert t.auth_revoked is False
        assert t._reader is not None and not t._reader.done()
    finally:
        await t.disconnect()
        await srv.stop()


@pytest.mark.asyncio
async def test_4401_with_reason_expired_repeated_still_never_latches():
    """Even a second 'expired' 4401 in a row is not a revocation."""
    srv = _ExpiredToken4401Server(close_4401_on={1, 2}, reason="expired")
    await srv.start()
    t = _transport(srv.url)
    t._reconnect_backoff_s = 0.05
    try:
        await t.connect()
        await t.handshake()
        assert await _wait_until(lambda: srv.dials == 3 and t._descriptor is not None)
        assert t.auth_revoked is False
    finally:
        await t.disconnect()
        await srv.stop()




@pytest.mark.asyncio
async def test_provisional_4401_never_starts_a_second_dialer_while_the_supervisor_is_mid_dial(monkeypatch):
    """Only ONE dialer may be live. A reader that dies with a provisional 4401
    while the backoff supervisor is already mid-dial (socket installed, hello in
    flight) must not start the fresh-token retry as a second concurrent dialer
    — two dialers replace each other's socket/reader and can leave the
    transport attached to the wrong socket. The supervisor's own re-dial carries
    the fresh token instead, so exactly two dials happen in total."""
    import gateway.relay.ws_transport as mod

    class _Close4401(Exception):
        code = 4401
        reason = "unauthorized"

    class _FirstWS:
        """The supervisor's first dial: its reader 4401s while hello is in flight,
        then the hello itself fails (the connector dropped this connection)."""

        def __init__(self, second_dial_done: asyncio.Event):
            self.release_reader = asyncio.Event()
            self.second_dial_done = second_dial_done

        async def send(self, _data):
            self.release_reader.set()
            await asyncio.wait_for(self.second_dial_done.wait(), 1)
            raise RuntimeError("dial lost during hello")

        def __aiter__(self):
            return self

        async def __anext__(self):
            await self.release_reader.wait()
            raise _Close4401()

        async def close(self):
            pass

    class _IdleWS:
        def __init__(self):
            self.stop = asyncio.Event()

        async def send(self, _data):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            await self.stop.wait()
            raise StopAsyncIteration

        async def close(self):
            self.stop.set()

    second_dial_done = asyncio.Event()
    sockets = [_FirstWS(second_dial_done), _IdleWS(), _IdleWS()]
    dials: list = []

    async def fake_connect(*_args, **_kwargs):
        ws = sockets[len(dials)]
        dials.append(ws)
        if len(dials) == 2:
            second_dial_done.set()
        return ws

    monkeypatch.setattr(mod.websockets, "connect", fake_connect)
    t = WebSocketRelayTransport(
        "ws://unused", "discord", "bot", reconnect=True,
        reconnect_backoff_s=0, reconnect_max_backoff_s=0,
    )
    t._handshake_succeeded = True
    t._supervisor = asyncio.create_task(t._reconnect_loop())
    try:
        assert await _wait_until(lambda: len(dials) >= 2)
        await asyncio.sleep(0.1)  # give a racing second dialer time to show itself
        assert len(dials) == 2, "a second dialer ran alongside the supervisor"
        assert t._ws is sockets[1], "the transport must be attached to the live socket"
        assert t._reader is not None and not t._reader.done()
        assert not t.auth_revoked
        # The supervisor's re-dial carried the fresh token: a 4401 on it is terminal.
        assert t._auth_retry_generation == t._dial_generation
    finally:
        await t.disconnect(budget_s=0)


class _FakeCloseError(Exception):
    """Stand-in for websockets' ConnectionClosed: carries a close code + reason."""

    def __init__(self, code: int, reason: str = ""):
        super().__init__(f"closed {code} {reason}")
        self.code = code
        self.reason = reason


class _FakeWS:
    """Scriptable socket for dialer-ownership tests. ``hello`` decides what
    happens when the transport sends its hello: a callable is awaited (and may
    raise); ``reader`` decides what the reader sees: an exception to raise once
    released, or None to idle until close()."""

    def __init__(self, *, reader: Exception | None = None, hello=None):
        self._reader_exc = reader
        self._hello = hello
        self.released = asyncio.Event()
        self.stopped = asyncio.Event()

    async def send(self, _data):
        self.released.set()
        if self._hello is not None:
            await self._hello(self)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self.released.wait()
        if self._reader_exc is not None:
            raise self._reader_exc
        await self.stopped.wait()
        raise StopAsyncIteration

    async def close(self):
        self.released.set()
        self.stopped.set()


def _scripted_connect(monkeypatch, outcomes: list):
    """Patch websockets.connect to pop ``outcomes`` in order: an Exception is
    raised (connect-time failure), anything else is returned as the socket.
    Returns the dial log."""
    import gateway.relay.ws_transport as mod

    dials: list = []

    async def fake_connect(*_args, **_kwargs):
        item = outcomes[len(dials)]
        dials.append(item)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(mod.websockets, "connect", fake_connect)
    return dials


def _fake_transport() -> WebSocketRelayTransport:
    t = WebSocketRelayTransport(
        "ws://unused", "discord", "bot", reconnect=True,
        reconnect_backoff_s=0, reconnect_max_backoff_s=0,
    )
    t._handshake_succeeded = True
    return t


@pytest.mark.asyncio
async def test_fresh_token_retry_socket_dropping_during_hello_still_reconnects(monkeypatch):
    """Mirror of the supervisor race: the auth-retry is the only live dialer and
    its fresh socket's reader dies (non-4401) after the hello was sent but before
    the dial returns. The reader must not arm a dialer (one is live) — so the
    DIAL itself must notice its reader is gone, report failure, and hand off to
    the supervisor. Otherwise the transport ends with no socket, no reader and
    no dialer: silently dead."""
    first = _FakeWS(reader=_FakeCloseError(1006, "lost"))

    async def hello_then_let_reader_die(ws):
        # hello send succeeds; the reader (released by send) raises and finishes
        # BEFORE _dial_and_start returns, so _ws is already None on return.
        await asyncio.sleep(0.02)

    first._hello = hello_then_let_reader_die
    second = _FakeWS()
    dials = _scripted_connect(monkeypatch, [first, second])
    t = _fake_transport()
    t._auth_retry_pending = True
    t._auth_retry = asyncio.create_task(t._redial_with_fresh_token())
    try:
        assert await _wait_until(lambda: len(dials) == 2 and t._ws is second)
        assert t._reader is not None and not t._reader.done()
        assert not t.auth_revoked
    finally:
        await t.disconnect(budget_s=0)


@pytest.mark.asyncio
async def test_network_failure_on_the_retry_dial_does_not_unmark_it(monkeypatch):
    """The fresh-token marker must survive connect-time failures that never
    reached authentication (network, timeout). If a blip consumed it, the NEXT
    dial — the one whose fresh token actually gets judged — would be unmarked
    and its 4401 read as another first strike, so a real revocation would never
    latch (unbounded provisional retries)."""
    refused = _FakeWS(reader=_FakeCloseError(4401, "unauthorized"))
    dials = _scripted_connect(monkeypatch, [RuntimeError("network blip"), refused, _FakeWS(), _FakeWS()])
    t = _fake_transport()
    t._auth_retry_pending = True
    t._auth_retry = asyncio.create_task(t._redial_with_fresh_token())
    try:
        assert await _wait_until(lambda: t.auth_revoked)
        assert len(dials) == 2, "the 4401 on the first judged fresh token must be terminal"
        assert not t._dialer_running()
    finally:
        await t.disconnect(budget_s=0)
