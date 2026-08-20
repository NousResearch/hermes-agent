"""Regression tests for the relay WS transport hardening fix.

Coatue incident 2026-08-18: WAN latency / event-loop stalls tripped the
websockets library's default 20s pong deadline, closing customer-gateway
sockets with `1011 keepalive ping timeout`. On top of the spurious close,
every in-flight outbound then hung for the full _outbound_timeout_s (~30s)
because only disconnect() failed pending futures — an unexpected socket drop
left them stranded — and sends issued while the reconnect supervisor was
backing off registered futures no reader could ever resolve.

Three hardening changes under test:
  1. _read_loop fails all in-flight _pending futures on ANY exit path with
     the dict shape callers expect ({"success": False, ...}).
  2. _request_response fails fast while the reconnect supervisor is
     mid-redial (live supervisor task = the redial window).
  3. connect() passes explicit WAN-friendly keepalive tuning
     (ping_interval=30, ping_timeout=60) to websockets.connect().
"""

from __future__ import annotations

import asyncio

import pytest

import gateway.relay.ws_transport as ws_transport_mod
from gateway.relay.ws_transport import WebSocketRelayTransport, WEBSOCKETS_AVAILABLE

pytestmark = pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")

if WEBSOCKETS_AVAILABLE:
    from websockets.exceptions import ConnectionClosedError


class _DroppingWS:
    """Fake socket: accepts sends, then the read loop dies mid-iteration —
    the shape of an unexpected close (e.g. 1011 keepalive ping timeout)."""

    def __init__(self, close_code: int | None = None):
        self.sent: list[str] = []
        # Reader blocks here until the test releases it, so the outbound
        # future is registered BEFORE the "socket" drops.
        self.drop = asyncio.Event()
        self._close_code = close_code

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self.drop.wait()
        if self._close_code is not None:
            from websockets.frames import Close

            raise ConnectionClosedError(Close(self._close_code, ""), None)
        raise ConnectionClosedError(None, None)

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_read_loop_exit_fails_pending_futures_promptly():
    """When the socket drops unexpectedly, in-flight _request_response callers
    must get {"success": False, ...} promptly — not block ~30s on a future
    only the (now dead) reader could have resolved."""
    t = WebSocketRelayTransport("ws://unused", "discord", "bot1", outbound_timeout_s=30.0)
    fake = _DroppingWS()
    t._ws = fake
    t._reader = asyncio.create_task(t._read_loop())

    send_task = asyncio.create_task(t.send_outbound({"op": "send_message", "text": "hi"}))
    # Let the outbound frame go out and its future register in _pending.
    for _ in range(50):
        if t._pending:
            break
        await asyncio.sleep(0.01)
    assert t._pending, "outbound future never registered"

    # Drop the socket: the read loop exits on ConnectionClosedError.
    fake.drop.set()

    result = await asyncio.wait_for(send_task, timeout=2.0)
    assert result == {"success": False, "error": "relay transport connection lost"}
    assert t._pending == {}
    await t._reader


@pytest.mark.asyncio
async def test_send_during_redial_window_fails_fast():
    """While the reconnect supervisor is backing off after a drop, a send must
    return an error dict immediately (no RuntimeError, no 30s timeout on an
    unresolvable future). Drives the REAL sequence — reader exit arms the
    supervisor and clears _ws — rather than hand-crafting a stale-_ws state
    the transport can no longer reach."""
    t = WebSocketRelayTransport(
        "ws://unused",
        "discord",
        "bot1",
        reconnect=True,
        reconnect_backoff_s=60.0,  # park the supervisor in backoff
        outbound_timeout_s=30.0,
    )
    fake = _DroppingWS()
    t._ws = fake
    await _run_reader_to_exit(t, fake)
    supervisor = t._supervisor
    try:
        assert supervisor is not None and not supervisor.done(), (
            "reader exit must arm the reconnect supervisor"
        )
        result = await asyncio.wait_for(
            t.send_outbound({"op": "send_message", "text": "hi"}), timeout=1.0
        )
        assert result["success"] is False
        assert t._pending == {}
    finally:
        if supervisor is not None:
            supervisor.cancel()
            try:
                await supervisor
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_send_allowed_once_redial_installs_fresh_socket(monkeypatch):
    """The moment _dial_and_start() installs a fresh socket and its reader,
    the transport is genuinely usable — even though the supervisor task has
    not finished unwinding (it is still awaiting the hello sends). A send in
    that window must be ACCEPTED, not rejected as 'reconnecting': gating
    sends on supervisor state rejected real traffic on a live socket."""

    class _LiveWS:
        def __init__(self):
            self.sent: list[str] = []
            self.hello_seen = asyncio.Event()
            self.release = asyncio.Event()

        async def send(self, data):
            self.sent.append(data)
            if '"hello"' in data:
                # Inside _dial_and_start, AFTER _ws and the reader are
                # installed. Park here to hold the window open.
                self.hello_seen.set()
                await self.release.wait()

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(3600)

        async def close(self):
            pass

    live = _LiveWS()

    async def _fake_connect(url, **kwargs):
        return live

    monkeypatch.setattr(ws_transport_mod.websockets, "connect", _fake_connect)

    t = WebSocketRelayTransport(
        "ws://unused",
        "discord",
        "bot1",
        reconnect=True,
        reconnect_backoff_s=0.01,
        outbound_timeout_s=5.0,
    )
    # Arm the supervisor exactly as the reader's fall-through does.
    t._supervisor = asyncio.create_task(t._reconnect_loop())
    await asyncio.wait_for(live.hello_seen.wait(), timeout=2.0)
    try:
        assert t._ws is live and not t._supervisor.done()

        send_task = asyncio.create_task(
            t.send_outbound({"op": "send_message", "text": "hi"})
        )
        # The send must reach the live socket (registered + frame written),
        # not fail fast: wait for the outbound frame to land.
        for _ in range(100):
            if any('"outbound"' in s for s in live.sent):
                break
            await asyncio.sleep(0.01)
        assert any('"outbound"' in s for s in live.sent), (
            "send was rejected during the post-dial window despite a live "
            "socket and running reader"
        )

        # Resolve it via the reader path shape: answer directly.
        rid = next(iter(t._pending))
        t._pending[rid].set_result({"success": True})
        assert (await asyncio.wait_for(send_task, timeout=2.0)) == {"success": True}
    finally:
        live.release.set()
        await asyncio.wait_for(t._supervisor, timeout=2.0)
        if t._reader is not None:
            t._reader.cancel()
            try:
                await t._reader
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_connect_passes_wan_keepalive_tuning(monkeypatch):
    """connect() must pass ping_interval=30 / ping_timeout=60 explicitly —
    the library defaults (20/20) caused spurious 1011 keepalive closes over
    WAN paths (Coatue 2026-08-18). Both call sites (with/without auth
    headers) are exercised."""
    captured: list[dict] = []

    class _IdleWS:
        async def send(self, data):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(3600)

        async def close(self):
            pass

    async def _fake_connect(url, **kwargs):
        captured.append(kwargs)
        return _IdleWS()

    monkeypatch.setattr(ws_transport_mod.websockets, "connect", _fake_connect)

    # Site 1: no upgrade secret -> the headerless connect() call.
    t = WebSocketRelayTransport("ws://unused", "discord", "bot1")
    await t.connect()
    await t.disconnect(budget_s=0)

    # Site 2: secret + gateway_id -> the additional_headers connect() call.
    t2 = WebSocketRelayTransport(
        "ws://unused", "discord", "bot1", gateway_id="gw-1", upgrade_secret="s3cret"
    )
    await t2.connect()
    await t2.disconnect(budget_s=0)

    assert len(captured) == 2
    no_header_kwargs, header_kwargs = captured
    assert "additional_headers" not in no_header_kwargs
    assert "additional_headers" in header_kwargs
    for kwargs in captured:
        assert kwargs.get("ping_interval") == 30
        assert kwargs.get("ping_timeout") == 60


async def _run_reader_to_exit(t: WebSocketRelayTransport, fake: _DroppingWS) -> None:
    """Start the reader on ``fake``, drop the socket, and wait for the reader
    to fully unwind — the state every post-drop assertion depends on."""
    t._reader = asyncio.create_task(t._read_loop())
    await asyncio.sleep(0)
    fake.drop.set()
    await t._reader


@pytest.mark.asyncio
async def test_read_loop_without_socket_still_fails_pending():
    """If the reader is ever scheduled with no socket (lifecycle bug), it must
    still settle in-flight waiters on its way out — the old `assert` escaped
    before the fail-pending cleanup and left them to the full 30s timeout."""
    t = WebSocketRelayTransport("ws://unused", "discord", "bot1", outbound_timeout_s=30.0)
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    t._pending["rid"] = fut
    t._ws = None

    await t._read_loop()  # must not raise

    assert fut.done()
    assert fut.result() == {"success": False, "error": "relay transport connection lost"}
    assert t._pending == {}


@pytest.mark.asyncio
async def test_send_after_terminal_4401_revocation_fails_fast():
    """A terminal 4401 revocation deliberately arms NO reconnect supervisor,
    so the reader's exit is the LAST liveness transition this transport will
    ever make. If _ws still points at the dead socket afterwards, the
    revocation path's own fatal-error notification send wedges for the full
    _outbound_timeout_s. The reader must leave _ws cleared so the
    not-connected guard answers instantly."""
    t = WebSocketRelayTransport(
        "ws://unused", "discord", "bot1", reconnect=True, outbound_timeout_s=30.0
    )
    fake = _DroppingWS(close_code=4401)
    t._ws = fake
    t._handshake_succeeded = True  # prior handshake -> 4401 is a revocation
    await _run_reader_to_exit(t, fake)

    assert t._auth_revoked is True
    assert t._supervisor is None  # revocation must not re-dial
    assert t._ws is None, "dead socket handle must not survive the reader"

    result = await asyncio.wait_for(
        t.send_outbound({"op": "send_message", "text": "hi"}), timeout=2.0
    )
    assert result["success"] is False
    assert t._pending == {}


@pytest.mark.asyncio
async def test_send_after_drop_with_reconnect_disabled_fails_fast():
    """reconnect=False transports never arm a supervisor either — the same
    stranded-_ws wedge as the revocation path, reachable by configuration."""
    t = WebSocketRelayTransport(
        "ws://unused", "discord", "bot1", reconnect=False, outbound_timeout_s=30.0
    )
    fake = _DroppingWS()
    t._ws = fake
    await _run_reader_to_exit(t, fake)

    assert t._ws is None, "dead socket handle must not survive the reader"

    result = await asyncio.wait_for(
        t.send_outbound({"op": "send_message", "text": "hi"}), timeout=2.0
    )
    assert result["success"] is False
    assert t._pending == {}
