"""Buzz typing indicator: ``send_typing`` publishes an ephemeral kind-20002 event.

Buzz clients render a typing bubble from kind-20002 events (the cadence its own
``buzz-acp`` agents use is one every 3s). The adapter signs one with the profile
key and publishes it on the already-authenticated WebSocket, best-effort.
"""

import asyncio
import hashlib
import json
import logging
from types import SimpleNamespace

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz_mod = load_plugin_adapter("buzz")
BuzzAdapter = _buzz_mod.BuzzAdapter
nostr_auth = _buzz_mod._nostr_auth
typing_indicator = _buzz_mod._typing

# BIP-340 test vector 0 private key
TEST_PRIVATE_KEY = "00" * 31 + "03"
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"
ROOT = "a" * 64
PARENT = "b" * 64


class _Socket:
    def __init__(self):
        self.sent = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))


class _DeadSocket:
    async def send(self, raw):
        raise ConnectionError("socket closed")


def _make_adapter(extra=None, *, socket=None):
    from gateway.config import PlatformConfig

    adapter = BuzzAdapter(PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay", **(extra or {})}))
    adapter._private_key = TEST_PRIVATE_KEY
    adapter._self_pubkey = nostr_auth.public_key_hex(TEST_PRIVATE_KEY)
    adapter._ws = socket
    return adapter


def _bip340_verify(pubkey_hex: str, message: bytes, signature: bytes) -> bool:
    """Independent BIP-340 verification (the adapter module only signs)."""
    p, n = nostr_auth.FIELD_ORDER, nostr_auth.CURVE_ORDER
    x = int(pubkey_hex, 16)
    y = pow((pow(x, 3, p) + 7) % p, (p + 1) // 4, p)
    if pow(y, 2, p) != (pow(x, 3, p) + 7) % p:
        return False
    public_point = (x, y if y % 2 == 0 else p - y)
    r, s = int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:], "big")
    if r >= p or s >= n:
        return False
    challenge = int.from_bytes(
        nostr_auth._tagged_hash("BIP0340/challenge", signature[:32] + bytes.fromhex(pubkey_hex) + message), "big"
    ) % n
    point = nostr_auth._point_add(
        nostr_auth._point_multiply(s), nostr_auth._point_multiply(n - challenge, public_point)
    )
    return point is not None and point[1] % 2 == 0 and point[0] == r


@pytest.mark.asyncio
async def test_typing_publishes_signed_kind_20002_event():
    socket = _Socket()
    adapter = _make_adapter(socket=socket)

    await adapter.send_typing(CHANNEL)

    assert len(socket.sent) == 1, "send_typing published nothing on the authenticated WebSocket"
    verb, event = socket.sent[0]
    assert verb == "EVENT"
    assert event["kind"] == 20002
    assert event["content"] == ""
    assert event["tags"] == [["h", CHANNEL]]
    assert event["pubkey"] == adapter._self_pubkey
    serialized = json.dumps(
        [0, event["pubkey"], event["created_at"], 20002, event["tags"], ""], separators=(",", ":"), ensure_ascii=False
    ).encode()
    assert event["id"] == hashlib.sha256(serialized).hexdigest()
    assert _bip340_verify(adapter._self_pubkey, bytes.fromhex(event["id"]), bytes.fromhex(event["sig"]))


@pytest.mark.parametrize(
    "metadata, thread_tags",
    [
        ({"thread_id": ROOT, "reply_to_message_id": PARENT}, [["e", ROOT, "", "root"], ["e", PARENT, "", "reply"]]),
        ({"thread_id": ROOT}, [["e", ROOT, "", "reply"]]),
        ({"thread_id": ROOT, "reply_to_message_id": ROOT}, [["e", ROOT, "", "reply"]]),
        ({}, []),
    ],
)
@pytest.mark.asyncio
async def test_typing_in_a_thread_carries_nip10_tags(metadata, thread_tags):
    socket = _Socket()
    adapter = _make_adapter(socket=socket)

    await adapter.send_typing(CHANNEL, metadata=metadata)

    assert socket.sent[0][1]["tags"] == [["h", CHANNEL], *thread_tags]


@pytest.mark.asyncio
async def test_typing_stays_flat_when_replies_post_flat():
    socket = _Socket()
    adapter = _make_adapter({"reply_in_thread": False}, socket=socket)

    await adapter.send_typing(CHANNEL, metadata={"thread_id": ROOT, "reply_to_message_id": PARENT})

    assert socket.sent[0][1]["tags"] == [["h", CHANNEL]]


@pytest.mark.asyncio
async def test_typing_without_an_authenticated_socket_is_a_silent_skip():
    adapter = _make_adapter(socket=None)

    assert await adapter.send_typing(CHANNEL) is None


@pytest.mark.asyncio
async def test_typing_is_throttled_per_chat(monkeypatch):
    socket = _Socket()
    adapter = _make_adapter(socket=socket)

    await adapter.send_typing(CHANNEL)
    await adapter.send_typing(CHANNEL)
    assert len(socket.sent) == 1, "a second tick inside the interval must not publish"

    await adapter.send_typing("another-channel")
    assert [frame[1]["tags"][0][1] for frame in socket.sent] == [CHANNEL, "another-channel"]

    later = typing_indicator.time.monotonic() + typing_indicator.TYPING_INTERVAL
    monkeypatch.setattr(typing_indicator, "time", SimpleNamespace(monotonic=lambda: later))
    await adapter.send_typing(CHANNEL)
    assert len(socket.sent) == 3, "typing must resume once the interval has passed"


@pytest.mark.asyncio
async def test_typing_failures_never_raise_or_log_the_key(caplog):
    caplog.set_level(logging.DEBUG)
    adapter = _make_adapter(socket=_DeadSocket())
    await adapter.send_typing(CHANNEL)

    unsignable = "zz" * 32
    adapter = _make_adapter(socket=_Socket())
    adapter._private_key = unsignable
    await adapter.send_typing(CHANNEL)

    assert adapter._ws.sent == []
    assert TEST_PRIVATE_KEY not in caplog.text and unsignable not in caplog.text


class _LiveSocket(_Socket):
    """A ``websockets.connect()`` target: NIP-42 handshake, then frames from ``next_frame``."""

    def __init__(self, next_frame):
        super().__init__()
        self._next_frame = next_frame

    async def recv(self):
        if self.sent:
            return json.dumps(["OK", self.sent[0][1]["id"], True, "authenticated"])
        return json.dumps(["AUTH", "relay-challenge"])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._next_frame()


@pytest.mark.asyncio
async def test_typing_rides_the_live_socket_and_stops_when_it_closes(monkeypatch):
    adapter = _make_adapter()
    adapter._channel_state = {CHANNEL: {"last_ts": 1, "seen": {}}}

    async def type_then_close():
        await adapter.send_typing(CHANNEL)
        raise StopAsyncIteration

    sockets = []

    def fake_connect(*args, **kwargs):
        if sockets:
            raise asyncio.CancelledError()  # ends the reconnect loop
        sockets.append(_LiveSocket(type_then_close))
        return sockets[0]

    import websockets

    monkeypatch.setattr(websockets, "connect", fake_connect)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.create_task(adapter._websocket_loop()), 10.0)

    def typing_frames():
        return [f for f in sockets[0].sent if f[0] == "EVENT" and f[1]["kind"] == 20002]

    assert len(typing_frames()) == 1, "typing was not published on the authenticated socket"
    await adapter.send_typing("another-channel")
    assert len(typing_frames()) == 1, "typing must not publish on a closed socket"

