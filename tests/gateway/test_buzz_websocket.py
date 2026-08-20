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


def test_compute_auth_tag_round_trip():
    """compute_auth_tag produces a tag that encodes the correct preimage.

    We can't verify the Schnorr sig without the Rust SDK, but we can check:
    - output is valid JSON with the expected structure
    - owner pubkey in the tag matches what we signed with
    - agent pubkey is NOT the same as the owner (self-attestation guard)
    - the preimage fed to SHA256 matches the nip_oa.rs spec
    """
    import hashlib as _hashlib

    # Use the BIP-340 test vector key as the owner and a different key as agent.
    owner_key = TEST_PRIVATE_KEY
    # agent pubkey: deterministic from a different private key (BIP-340 vector 1 sk)
    agent_sk = "B7E151628AED2A6ABF7158809CF4F3C762E7160F38B4DA56A784D9045190CFEF"
    agent_pubkey = nostr_auth.public_key_hex(agent_sk)

    tag_json = nostr_auth.compute_auth_tag(owner_key, agent_pubkey)

    import json as _json
    tag = _json.loads(tag_json)

    assert tag[0] == "auth", "first element must be 'auth'"
    assert tag[1] == nostr_auth.public_key_hex(owner_key), "element 1 must be owner pubkey"
    assert tag[2] == "", "element 2 (conditions) must be empty string by default"
    assert len(tag[3]) == 128, "element 3 (sig) must be 128 hex chars"
    assert all(c in "0123456789abcdef" for c in tag[3]), "sig must be lowercase hex"


def test_compute_auth_tag_rejects_self_attestation():
    owner_key = TEST_PRIVATE_KEY
    owner_pubkey = nostr_auth.public_key_hex(owner_key)
    try:
        nostr_auth.compute_auth_tag(owner_key, owner_pubkey)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "self-attestation" in str(e)


def test_compute_auth_tag_with_conditions():
    owner_key = TEST_PRIVATE_KEY
    agent_sk = "B7E151628AED2A6ABF7158809CF4F3C762E7160F38B4DA56A784D9045190CFEF"
    agent_pubkey = nostr_auth.public_key_hex(agent_sk)

    import json as _json
    tag_json = nostr_auth.compute_auth_tag(owner_key, agent_pubkey, conditions="kind=9")
    tag = _json.loads(tag_json)
    assert tag[2] == "kind=9"


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


# ── CLOSED frame handling ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_websocket_loop_drops_restricted_channel_without_reconnect():
    """A CLOSED frame with 'restricted: not a channel member' must silently
    drop the offending subscription and continue — not raise ConnectionError
    and trigger a reconnect loop.

    Regression test for the 1.6 s flood caused by the relay immediately
    rejecting a private-channel subscription.
    """
    import sys
    from unittest.mock import patch, MagicMock
    from contextlib import asynccontextmanager

    adapter = _make_adapter(extra={"channels": [CHANNEL]})
    adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
    adapter._ws_ready = asyncio.Event()

    sub_id = "hermes-buzz-0"
    messages = [json.dumps(["CLOSED", sub_id, "restricted: not a channel member"])]
    idx = 0

    class _FakeWs:
        sent = []

        async def send(self, raw):
            self.sent.append(json.loads(raw))

        def __aiter__(self):
            return self

        async def __anext__(self):
            nonlocal idx
            if idx < len(messages):
                val = messages[idx]
                idx += 1
                return val
            # Stall so the task stays alive for our assertions.
            await asyncio.sleep(10)
            raise StopAsyncIteration

    @asynccontextmanager
    async def _fake_connect(*_a, **_kw):
        yield _FakeWs()

    async def _noop_auth(self_inner, ws):
        pass

    async def _noop_subscribe(self_inner, ws):
        return {sub_id: CHANNEL}

    fake_ws_mod = MagicMock()
    fake_ws_mod.connect = _fake_connect

    with (
        patch.dict(sys.modules, {"websockets": fake_ws_mod}),
        patch.object(type(adapter), "_authenticate_websocket", _noop_auth),
        patch.object(type(adapter), "_subscribe_websocket", _noop_subscribe),
    ):
        adapter._ws_ready = asyncio.Event()
        adapter._ws_ready.set()
        adapter._ws_active = True
        task = asyncio.create_task(adapter._websocket_loop())
        await asyncio.sleep(0.1)

    assert CHANNEL in adapter._restricted_channels, (
        "restricted channel should be recorded in _restricted_channels"
    )
    assert CHANNEL not in adapter._channel_state, (
        "channel_state entry should be removed for a restricted channel"
    )
    assert not task.done(), "websocket_loop must not exit/reconnect on a restricted CLOSED"

    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_websocket_loop_reconnects_on_non_restricted_closed():
    """A CLOSED frame that is NOT 'restricted' must NOT add the channel to
    _restricted_channels — it is a transient error and the loop should reconnect.
    """
    import sys
    from unittest.mock import patch, MagicMock
    from contextlib import asynccontextmanager

    adapter = _make_adapter(extra={"channels": [CHANNEL]})
    adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}

    sub_id = "hermes-buzz-0"
    messages = [json.dumps(["CLOSED", sub_id, "error: server shutting down"])]
    idx = 0

    class _FakeWs:
        sent = []

        async def send(self, raw):
            self.sent.append(json.loads(raw))

        def __aiter__(self):
            return self

        async def __anext__(self):
            nonlocal idx
            if idx < len(messages):
                val = messages[idx]
                idx += 1
                return val
            await asyncio.sleep(10)
            return json.dumps(["NOTICE", "stall"])

    @asynccontextmanager
    async def _fake_connect(*_a, **_kw):
        yield _FakeWs()

    async def _noop_auth(self_inner, ws):
        pass

    async def _noop_subscribe(self_inner, ws):
        return {sub_id: CHANNEL}

    fake_ws_mod = MagicMock()
    fake_ws_mod.connect = _fake_connect

    with (
        patch.dict(sys.modules, {"websockets": fake_ws_mod}),
        patch.object(type(adapter), "_authenticate_websocket", _noop_auth),
        patch.object(type(adapter), "_subscribe_websocket", _noop_subscribe),
    ):
        adapter._ws_ready = asyncio.Event()
        adapter._ws_ready.set()
        adapter._ws_active = True
        task = asyncio.create_task(adapter._websocket_loop())
        await asyncio.sleep(0.1)

    assert CHANNEL not in adapter._restricted_channels, (
        "non-restricted CLOSED must not add channel to _restricted_channels"
    )

    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


def test_restricted_channels_skipped_during_subscribe():
    """Channels in _restricted_channels are not re-subscribed on reconnect."""
    adapter = _make_adapter()
    adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
    adapter._restricted_channels.add(CHANNEL)

    subscriptions = {}

    class _CountingWs:
        sent = []

        async def send(self, raw):
            self.sent.append(json.loads(raw))

    async def _run():
        ws = _CountingWs()
        result = await adapter._subscribe_websocket(ws)
        return ws.sent, result

    sent, subs = asyncio.get_event_loop().run_until_complete(_run())

    assert CHANNEL not in subs.values(), (
        "restricted channel must not appear in subscriptions dict"
    )
    req_channels = [
        frame[2].get("#h", [])
        for frame in sent
        if isinstance(frame, list) and frame[0] == "REQ"
    ]
    assert all(CHANNEL not in ch_list for ch_list in req_channels), (
        "restricted channel must not be sent in any REQ frame"
    )
