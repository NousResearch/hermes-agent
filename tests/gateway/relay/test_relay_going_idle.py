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
        await asyncio.sleep(0.2)
        assert srv.connections >= 2
    finally:
        await t.disconnect()
        srv._server.close()
        await srv._server.wait_closed()


# ── scale-to-zero go_dormant() (D12 / F14) ───────────────────────────────────


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

def _relay_descriptor():
    """Placeholder descriptor for the adapter tests below."""
    from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

    return CapabilityDescriptor(
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



def test_adapter_reports_false_when_the_transport_hold_fails():
    """The runner suspends on the strength of this answer, so a transport that is missing the method or throws must read as 'not held' rather than pass for success on the mere absence of an exception."""
    from gateway.config import PlatformConfig
    from gateway.relay.adapter import RelayAdapter

    placeholder = _relay_descriptor()

    class Throwing:
        def hold_redial(self):
            raise RuntimeError("transport refused the hold")

        def release_redial(self):
            raise RuntimeError("transport refused the release")

    class Missing:
        pass

    for transport in (Throwing(), Missing()):
        adapter = RelayAdapter(PlatformConfig(), placeholder, transport=transport)
        assert adapter.hold_redial() is False, type(transport).__name__
        assert adapter.release_redial() is False, type(transport).__name__

    class Working:
        def __init__(self):
            self.calls = []

        def hold_redial(self):
            self.calls.append("hold")

        def release_redial(self):
            self.calls.append("release")

    working = Working()
    adapter = RelayAdapter(PlatformConfig(), placeholder, transport=working)
    assert adapter.hold_redial() is True
    assert working.calls == ["hold"]


@pytest.mark.asyncio
async def test_adapter_redial_hold_delegates_to_transport(server):
    """The runner only holds the adapter, so this delegation is the only thing
    wiring its suspend decision to the transport."""
    from gateway.config import PlatformConfig
    from gateway.relay.adapter import RelayAdapter

    placeholder = _relay_descriptor()
    transport = WebSocketRelayTransport(
        server.url, "discord", "appShared", reconnect=True, reconnect_backoff_s=0.05
    )
    adapter = RelayAdapter(PlatformConfig(), placeholder, transport=transport)
    await adapter.connect()
    try:
        adapter.hold_redial()
        assert transport._redial_held is True
        adapter.release_redial()
        assert transport._redial_held is False
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_go_dormant_leaves_the_socket_open_when_the_ack_is_missed(server):
    """No ack means nothing will suspend us, so closing would only cost a needless disconnect/reconnect cycle while the agent keeps serving."""
    t = WebSocketRelayTransport(
        server.url, "discord", "appShared", reconnect=True, reconnect_backoff_s=0.05
    )
    await t.connect()
    try:
        async def no_ack(timeout_s=None):
            return False

        t.go_idle = no_ack
        assert await t.go_dormant() is False
        assert t._dormant is False, "must not enter the dormant cadence"
        assert t._ws is not None, "socket must stay open"
        assert t._closing is False
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_adapter_go_dormant_delegates_to_transport(server):
    """RelayAdapter.go_dormant() drives the transport's go_dormant (going_idle +
    dormant close) without the terminal teardown disconnect() does."""
    from gateway.config import PlatformConfig
    from gateway.relay.adapter import RelayAdapter

    placeholder = _relay_descriptor()
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


