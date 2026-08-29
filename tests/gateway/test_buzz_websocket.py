"""Tests for the Buzz WebSocket transport (NIP-42) and Nostr signing module.

The signing module and WS transport were contributed in PR #73636 by
@ScaleLeanChris and consolidated onto the merged poll-based adapter; these
tests cover the crypto (against the official BIP-340 vector) and the WS
lifecycle as wired into BuzzAdapter.
"""

import asyncio
import json
import time

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz_mod = load_plugin_adapter("buzz")
BuzzAdapter = _buzz_mod.BuzzAdapter

import importlib.util as _ilu
from pathlib import Path as _Path

_auth_path = _Path(_buzz_mod.__file__).with_name("nostr_auth.py")
_spec = _ilu.spec_from_file_location("plugin_adapter_buzz_nostr_auth", _auth_path)
nostr_auth = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(nostr_auth)

SELF_PUBKEY = "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
# BIP-340 test vector 0 private key
TEST_PRIVATE_KEY = "00" * 31 + "03"
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"


def _make_adapter(extra=None):
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay", **(extra or {})})
    adapter = BuzzAdapter(cfg)
    adapter._self_pubkey = SELF_PUBKEY
    adapter._private_key = TEST_PRIVATE_KEY
    adapter._display_name = "Chip"
    return adapter


# ── nostr_auth: BIP-340 / NIP-42 ──────────────────────────────────────────


def test_schnorr_sign_matches_official_bip340_vector_zero():
    signature = nostr_auth.schnorr_sign(
        bytes(32), TEST_PRIVATE_KEY, auxiliary_randomness=bytes(32)
    )
    assert nostr_auth.public_key_hex(TEST_PRIVATE_KEY).upper() == (
        "F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9"
    )
    assert signature.hex().upper() == (
        "E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA8215"
        "25F66A4A85EA8B71E482A74F382D2CE5EBEEE8FDB2172F477DF4900D310536C0"
    )


def test_decode_private_key_rejects_bad_input():
    with pytest.raises(ValueError):
        nostr_auth.decode_private_key("not-a-key")
    with pytest.raises(ValueError):
        nostr_auth.decode_private_key("00" * 32)  # zero — outside range
    with pytest.raises(ValueError):
        nostr_auth.decode_private_key("nsec1qqqqqqqq")  # bad checksum/length


def test_build_auth_event_shape_and_owner_tag():
    tag = json.dumps(["auth", "b" * 64, "", "c" * 128])
    event = nostr_auth.build_auth_event(
        private_key=TEST_PRIVATE_KEY,
        challenge="challenge-1",
        relay_url="wss://relay.example",
        auth_tag_json=tag,
        created_at=1_700_000_000,
        auxiliary_randomness=bytes(32),
    )
    assert event["kind"] == 22242
    assert ["relay", "wss://relay.example"] in event["tags"]
    assert ["challenge", "challenge-1"] in event["tags"]
    assert ["auth", "b" * 64, "", "c" * 128] in event["tags"]
    assert len(bytes.fromhex(event["sig"])) == 64
    assert event["pubkey"] == nostr_auth.public_key_hex(TEST_PRIVATE_KEY)


# ── Adapter WS wiring ─────────────────────────────────────────────────────


class _FakeWebSocket:
    """Replays a NIP-42 handshake: AUTH challenge, then OK for the reply."""

    def __init__(self):
        self.sent = []

    async def recv(self):
        if self.sent:
            auth_event = self.sent[0][1]
            return json.dumps(["OK", auth_event["id"], True, "authenticated"])
        return json.dumps(["AUTH", "relay-challenge"])

    async def send(self, raw):
        self.sent.append(json.loads(raw))


# ── _websocket_loop: read-idle watchdog (#98097) ──────────────────────────


class _ScriptedWebSocket(_FakeWebSocket):
    """A connect() target whose event frames come from a scripted behavior.

    The auth handshake is the relay's (inherited from _FakeWebSocket); after
    it, each ``__anext__`` delegates to ``anext_behavior`` — a coroutine
    function returning the next raw frame or raising StopAsyncIteration for
    a clean close.
    """

    def __init__(self, anext_behavior):
        super().__init__()
        self._anext_behavior = anext_behavior
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        self.exited = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._anext_behavior()


@pytest.mark.asyncio
async def test_websocket_loop_reconnects_when_read_goes_silent(monkeypatch, caplog):
    """A relay close the transport never surfaces must not park the loop.

    Reproduces the #98097 shape: a socket stuck in CLOSE_WAIT yields no
    frame and no error, so without a read-side bound the loop would wait
    forever while the gateway keeps reporting "connected".
    """
    import logging

    adapter = _make_adapter()
    monkeypatch.setattr(_buzz_mod, "_WS_READ_IDLE_TIMEOUT", 0.05)
    caplog.set_level(logging.WARNING)

    sockets = []

    async def dead_anext():
        await asyncio.Event().wait()  # never yields, never raises

    def fake_connect(*args, **kwargs):
        ws = _ScriptedWebSocket(dead_anext)
        sockets.append(ws)
        return ws

    import websockets as _ws_mod

    monkeypatch.setattr(_ws_mod, "connect", fake_connect)

    task = asyncio.create_task(adapter._websocket_loop())
    try:
        deadline = time.monotonic() + 5.0
        while len(sockets) < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
    finally:
        task.cancel()
        try:
            await asyncio.wait_for(task, 5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    assert len(sockets) >= 2, "idle read watchdog did not force a reconnect"
    assert sockets[0].exited, "the silent connection was not closed before reconnecting"
    assert any("went silent" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_websocket_loop_dispatches_frames_and_closes_cleanly(monkeypatch):
    """The watchdog refactor preserves the healthy path: frames dispatch to
    _handle_event and a server-side close (StopAsyncIteration) exits the
    connection cleanly before the loop reconnects."""
    adapter = _make_adapter()
    adapter._channel_state = {CHANNEL: {"last_ts": 1, "seen": {}}}

    handled = []

    async def record_handle_event(channel_id, state, event):
        handled.append((channel_id, event))

    monkeypatch.setattr(adapter, "_handle_event", record_handle_event)

    frames = iter(
        [
            json.dumps(
                ["EVENT", "hermes-buzz-0", {"id": "e1", "kind": 9, "created_at": 2, "content": "hi"}]
            ),
        ]
    )

    async def scripted_anext():
        try:
            return next(frames)
        except StopIteration:
            raise StopAsyncIteration from None

    sockets = []

    def fake_connect(*args, **kwargs):
        # Second connect ends the loop: CancelledError re-raises out of the
        # loop's except-order, unlike a regular Exception which would retry.
        if len(sockets) == 1:
            raise asyncio.CancelledError()
        ws = _ScriptedWebSocket(scripted_anext)
        sockets.append(ws)
        return ws

    import websockets as _ws_mod

    monkeypatch.setattr(_ws_mod, "connect", fake_connect)

    task = asyncio.create_task(adapter._websocket_loop())
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 10.0)

    assert sockets[0].exited, "clean close did not exit the async-with block"
    assert handled and handled[0][0] == CHANNEL
    assert handled[0][1]["id"] == "e1"


@pytest.mark.asyncio
async def test_websocket_auth_raises_on_rejection():
    adapter = _make_adapter()

    class RejectingWs(_FakeWebSocket):
        async def recv(self):
            if self.sent:
                auth_event = self.sent[0][1]
                return json.dumps(["OK", auth_event["id"], False, "denied"])
            return json.dumps(["AUTH", "relay-challenge"])

    with pytest.raises(ConnectionError):
        await adapter._authenticate_websocket(RejectingWs())


@pytest.mark.asyncio
async def test_websocket_auth_uses_credentials_owner_tag():
    adapter = _make_adapter()
    adapter._auth_tag = json.dumps(["auth", "b" * 64, "", "c" * 128])
    websocket = _FakeWebSocket()
    await adapter._authenticate_websocket(websocket)
    assert ["auth", "b" * 64, "", "c" * 128] in websocket.sent[0][1]["tags"]


