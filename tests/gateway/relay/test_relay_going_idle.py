"""Phase 5 §5.3 — going-idle / buffered-flip primitive (gateway side).

Exercises the WebSocketRelayTransport's going_idle/ack handshake, the
buffered-inbound ack (a bufferId-carrying inbound is acked after the handler
runs), the NET-NEW reconnect loop (re-dial + re-handshake after an unexpected
close), and the RelayAdapter emitting going_idle from its existing drain
(disconnect) transition. All against a real in-process websockets server.
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


class _IdleAwareServer:
    """Connector stub: descriptor on hello, acks going_idle, records inbound_acks,
    and can push buffered inbound frames (with bufferId) after handshake."""

    def __init__(self):
        self.received: list[dict] = []
        self.inbound_acks: list[str] = []
        self.going_idle_count = 0
        self._server = None
        self.url = ""
        # Frames to push right after each handshake (e.g. buffered backlog replay).
        self._to_push: list[dict] = []
        self.connections = 0

    async def start(self):
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        sock = next(iter(self._server.sockets))
        self.url = f"ws://127.0.0.1:{sock.getsockname()[1]}"

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws):
        self.connections += 1
        try:
            async for raw in ws:
                for line in str(raw).split("\n"):
                    if not line.strip():
                        continue
                    frame = json.loads(line)
                    self.received.append(frame)
                    await self._on_frame(ws, frame)
        except Exception:
            pass

    async def _on_frame(self, ws, frame):
        ftype = frame.get("type")
        if ftype == "hello":
            await ws.send(json.dumps({"type": "descriptor", "descriptor": DESCRIPTOR}) + "\n")
            for f in self._to_push:
                await ws.send(json.dumps(f) + "\n")
        elif ftype == "going_idle":
            self.going_idle_count += 1
            await ws.send(json.dumps({"type": "going_idle_ack"}) + "\n")
        elif ftype == "inbound_ack":
            self.inbound_acks.append(frame.get("bufferId"))


@pytest_asyncio.fixture
async def server():
    srv = _IdleAwareServer()
    await srv.start()
    yield srv
    await srv.stop()


@pytest.mark.asyncio
async def test_go_idle_awaits_ack(server):
    t = WebSocketRelayTransport(server.url, "discord", "appShared")
    await t.connect()
    try:
        await t.handshake()
        acked = await t.go_idle(timeout_s=2)
        assert acked is True
        assert server.going_idle_count == 1
        assert any(f["type"] == "going_idle" for f in server.received)
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_go_idle_returns_false_on_timeout(server):
    # A server that never acks going_idle -> go_idle returns False (caller closes anyway).
    async def no_ack(ws, frame):
        if frame.get("type") == "hello":
            await ws.send(json.dumps({"type": "descriptor", "descriptor": DESCRIPTOR}) + "\n")
        # deliberately ignore going_idle

    server._on_frame = no_ack  # type: ignore[assignment]
    t = WebSocketRelayTransport(server.url, "discord", "appShared")
    await t.connect()
    try:
        await t.handshake()
        acked = await t.go_idle(timeout_s=0.3)
        assert acked is False
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_buffered_inbound_is_acked_after_handler(server):
    # A buffered delivery (bufferId present) is acked AFTER the handler runs; a
    # live delivery (no bufferId) is not acked.
    server._to_push = [
        {
            "type": "inbound",
            "event": {
                "text": "buffered",
                "message_type": "text",
                "source": {"platform": "discord", "chat_id": "c1", "chat_type": "dm"},
            },
            "bufferId": "buf-42",
        },
        {
            "type": "inbound",
            "event": {
                "text": "live",
                "message_type": "text",
                "source": {"platform": "discord", "chat_id": "c1", "chat_type": "dm"},
            },
        },
    ]
    seen = []

    async def handler(ev):
        seen.append(ev.text)

    t = WebSocketRelayTransport(server.url, "discord", "appShared")
    t.set_inbound_handler(handler)
    await t.connect()
    try:
        await t.handshake()
        await asyncio.sleep(0.1)
        assert "buffered" in seen and "live" in seen
        # Only the buffered (bufferId) delivery was acked.
        assert server.inbound_acks == ["buf-42"]
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_reconnect_redials_after_unexpected_close():
    # A server that drops the FIRST connection right after handshake; the
    # transport with reconnect=True re-dials and handshakes again.
    drops = {"n": 0}
    srv = _IdleAwareServer()

    async def handle(ws):
        srv.connections += 1
        async for raw in ws:
            for line in str(raw).split("\n"):
                if not line.strip():
                    continue
                frame = json.loads(line)
                if frame.get("type") == "hello":
                    await ws.send(json.dumps({"type": "descriptor", "descriptor": DESCRIPTOR}) + "\n")
                    if drops["n"] == 0:
                        drops["n"] += 1
                        await ws.close()  # force an unexpected close on the first connection
                        return

    srv._server = await websockets.serve(handle, "127.0.0.1", 0)
    sock = next(iter(srv._server.sockets))
    srv.url = f"ws://127.0.0.1:{sock.getsockname()[1]}"
    t = WebSocketRelayTransport(srv.url, "discord", "appShared", reconnect=True, reconnect_backoff_s=0.05)
    try:
        await t.connect()
        await t.handshake()
        # First connection is dropped server-side; the reconnect loop re-dials.
        await asyncio.sleep(0.5)
        assert srv.connections >= 2
    finally:
        await t.disconnect()
        srv._server.close()
        await srv._server.wait_closed()


@pytest.mark.asyncio
async def test_no_reconnect_after_deliberate_disconnect(server):
    t = WebSocketRelayTransport(server.url, "discord", "appShared", reconnect=True, reconnect_backoff_s=0.05)
    await t.connect()
    await t.handshake()
    before = server.connections
    await t.disconnect()
    await asyncio.sleep(0.3)
    # A deliberate disconnect must NOT trigger the reconnect loop.
    assert server.connections == before


@pytest.mark.asyncio
async def test_adapter_emits_going_idle_on_disconnect(server):
    # The RelayAdapter emits going_idle as part of its existing disconnect (drain)
    # transition, then tears down the transport.
    from gateway.config import PlatformConfig
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

    placeholder = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="discord",
        label="Relay",
        max_message_length=4096,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=False,
        markdown_dialect="plain",
        len_unit="chars",
    )
    transport = WebSocketRelayTransport(server.url, "discord", "appShared")
    adapter = RelayAdapter(PlatformConfig(), placeholder, transport=transport)
    await adapter.connect()
    await adapter.disconnect()
    assert server.going_idle_count == 1


# ── scale-to-zero go_dormant() (D12 / F14) ───────────────────────────────────


@pytest.mark.asyncio
async def test_go_dormant_emits_going_idle_and_closes_without_terminal_teardown(server):
    """go_dormant() flips the connector to buffered-only (going_idle->ack) AND
    closes the socket, but does NOT set the terminal _closing flag or cancel the
    reconnect supervisor — the F14 distinction from disconnect()."""
    t = WebSocketRelayTransport(
        server.url, "discord", "appShared", reconnect=True, reconnect_backoff_s=0.05
    )
    await t.connect()
    await t.handshake()
    try:
        acked = await t.go_dormant(timeout_s=2)
        assert acked is True
        assert server.going_idle_count == 1
        # The socket was closed (dormant), but NOT via the terminal path:
        assert t._closing is False  # disconnect() would set this True
        assert t._dormant is True
        # Not a revocation — the auth-revoked latch stays clear.
        assert t.auth_revoked is False
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_go_dormant_redials_on_wake_and_drains(server):
    """After go_dormant() the reconnect supervisor stays armed, so the gateway
    re-dials (simulating a wake) and the connector replays its buffered backlog
    on the new handshake. This is the wake->reconnect->drain contract (§3.4)."""
    # Queue a buffered inbound to be replayed on the NEXT (wake) handshake.
    server._to_push = [
        {
            "type": "inbound",
            "event": {
                "text": "while-asleep",
                "message_type": "text",
                "source": {"platform": "discord", "chat_id": "c1", "chat_type": "dm"},
            },
            "bufferId": "buf-wake-1",
        }
    ]
    seen: list[str] = []

    async def handler(ev):
        seen.append(ev.text)

    t = WebSocketRelayTransport(
        server.url, "discord", "appShared", reconnect=True, reconnect_backoff_s=5.0
    )
    # Dormant re-dial cadence is short so the test wakes promptly even though the
    # ordinary reconnect backoff is long (proves the dormant path uses its own).
    t._dormant_redial_s = 0.05
    t.set_inbound_handler(handler)
    await t.connect()
    await t.handshake()
    before = server.connections
    try:
        await t.go_dormant(timeout_s=2)
        # The supervisor was armed by the dormant close; it re-dials on the
        # dormant cadence (~0.05s), NOT the 5s reconnect backoff.
        for _ in range(50):
            if server.connections > before and "while-asleep" in seen:
                break
            await asyncio.sleep(0.05)
        assert server.connections > before  # re-dialed (woke)
        assert "while-asleep" in seen  # drained the buffered backlog on reconnect
        # The successful re-dial cleared the dormant flag.
        assert t._dormant is False
        # The buffered entry was acked (this stub re-pushes on every handshake, so
        # a long-lived dormant poll may ack it more than once; the invariant is
        # that it was drained at least once — a real connector stops replaying an
        # acked entry).
        assert "buf-wake-1" in server.inbound_acks
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_disconnect_cancels_supervisor_but_go_dormant_does_not(server):
    """Direct contrast (F14): disconnect() is terminal (cancels supervisor, no
    re-dial); go_dormant() keeps it armed. Guards against a future refactor that
    routes dormancy through disconnect()."""
    # disconnect(): terminal — no reconnect.
    t1 = WebSocketRelayTransport(
        server.url, "discord", "appShared", reconnect=True, reconnect_backoff_s=0.05
    )
    await t1.connect()
    await t1.handshake()
    after_first = server.connections
    await t1.disconnect()
    await asyncio.sleep(0.3)
    assert server.connections == after_first  # disconnect did NOT re-dial
    assert t1._closing is True

    # go_dormant(): armed — re-dials.
    t2 = WebSocketRelayTransport(
        server.url, "discord", "appShared", reconnect=True, reconnect_backoff_s=0.05
    )
    t2._dormant_redial_s = 0.05
    await t2.connect()
    await t2.handshake()
    before = server.connections
    try:
        await t2.go_dormant(timeout_s=2)
        for _ in range(50):
            if server.connections > before:
                break
            await asyncio.sleep(0.05)
        assert server.connections > before  # go_dormant stayed armed and re-dialed
        assert t2._closing is False
    finally:
        await t2.disconnect()


@pytest.mark.asyncio
async def test_go_dormant_noop_when_never_connected():
    """go_dormant() on a transport that never connected is a safe no-op (False),
    not a crash."""
    t = WebSocketRelayTransport("ws://127.0.0.1:1", "discord", "appShared")
    assert await t.go_dormant(timeout_s=0.1) is False


@pytest.mark.asyncio
async def test_adapter_go_dormant_delegates_to_transport(server):
    """RelayAdapter.go_dormant() drives the transport's go_dormant (going_idle +
    dormant close) without the terminal teardown disconnect() does."""
    from gateway.config import PlatformConfig
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

    placeholder = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="discord",
        label="Relay",
        max_message_length=4096,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=False,
        markdown_dialect="plain",
        len_unit="chars",
    )
    transport = WebSocketRelayTransport(
        server.url, "discord", "appShared", reconnect=True, reconnect_backoff_s=0.05
    )
    adapter = RelayAdapter(PlatformConfig(), placeholder, transport=transport)
    await adapter.connect()
    try:
        ok = await adapter.go_dormant()
        assert ok is True
        assert server.going_idle_count == 1
        assert transport._closing is False  # NOT the terminal teardown
        assert transport._dormant is True
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_adapter_go_dormant_noop_on_stub_transport():
    """An adapter whose transport lacks go_dormant (the stub) degrades to a safe
    no-op returning False, never raising."""
    from gateway.config import PlatformConfig
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

    placeholder = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="discord",
        label="Relay",
        max_message_length=4096,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=False,
        markdown_dialect="plain",
        len_unit="chars",
    )

    class _StubTransport:
        async def connect(self, *, is_reconnect: bool = False):
            return True

        def set_inbound_handler(self, h):
            pass

        async def handshake(self):
            return placeholder

    adapter = RelayAdapter(PlatformConfig(), placeholder, transport=_StubTransport())
    assert await adapter.go_dormant() is False
