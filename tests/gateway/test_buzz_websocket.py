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


def test_build_auth_event_rejects_malformed_auth_tag():
    with pytest.raises(ValueError):
        nostr_auth.build_auth_event(
            private_key=TEST_PRIVATE_KEY,
            challenge="c",
            relay_url="wss://r",
            auth_tag_json='{"not": "a list"}',
        )


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


def test_transport_config_parsing():
    assert _make_adapter().transport == "auto"
    assert _make_adapter({"transport": "poll"}).transport == "poll"
    assert _make_adapter({"transport": "websocket"}).transport == "websocket"
    assert _make_adapter({"transport": "bogus"}).transport == "auto"


def test_websocket_url_converts_rest_schemes():
    adapter = _make_adapter()
    adapter.relay_url = "https://community.example/path"
    assert adapter._websocket_url() == "wss://community.example/path"
    adapter.relay_url = "http://localhost:3000"
    assert adapter._websocket_url() == "ws://localhost:3000"
    adapter.relay_url = "ftp://nope"
    with pytest.raises(ValueError):
        adapter._websocket_url()


@pytest.mark.asyncio
async def test_websocket_auth_signs_nip42_challenge(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setenv("BUZZ_AUTH_TAG", json.dumps(["auth", "b" * 64, "", "c" * 128]))
    ws = _FakeWebSocket()
    await adapter._authenticate_websocket(ws)
    frame = ws.sent[0]
    assert frame[0] == "AUTH"
    event = frame[1]
    assert event["kind"] == 22242
    assert ["challenge", "relay-challenge"] in event["tags"]
    assert ["auth", "b" * 64, "", "c" * 128] in event["tags"]


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
async def test_subscriptions_resume_from_channel_high_water_marks():
    adapter = _make_adapter()
    adapter._channel_state = {
        "general-id": {"chat_type": "group", "last_ts": 100, "seen": {}},
        "dm-one": {"chat_type": "dm", "last_ts": 200, "seen": {}},
    }
    adapter._membership_since = 300
    ws = _FakeWebSocket()
    subscriptions = await adapter._subscribe_websocket(ws)
    assert subscriptions["hermes-buzz-0"] == "general-id"
    assert subscriptions["hermes-buzz-1"] == "dm-one"
    assert subscriptions[_buzz_mod._WS_MEMBERSHIP_SUB_ID] is None
    assert ws.sent[0][2]["#h"] == ["general-id"]
    assert ws.sent[0][2]["since"] == 99
    assert ws.sent[1][2]["since"] == 199
    assert ws.sent[2][2]["kinds"] == [_buzz_mod._WS_MEMBERSHIP_KIND]
    assert ws.sent[2][2]["since"] == 299


@pytest.mark.asyncio
async def test_ws_events_route_through_handle_event_with_dedup():
    """WS events use the same _handle_event pipeline as the poll loop —
    same de-dupe, same mention gate, same DM semantics."""
    adapter = _make_adapter()
    adapter._channel_state = {
        CHANNEL: {"chat_type": "dm", "last_ts": 0, "seen": __import__("collections").OrderedDict()}
    }
    received = []

    async def handle_message(event):
        received.append(event)

    adapter.handle_message = handle_message
    adapter._message_handler = handle_message  # dispatch gate requires a handler
    adapter._user_names["b" * 64] = "user"
    event = {
        "id": "evt-1",
        "kind": 9,
        "pubkey": "b" * 64,
        "content": "hello",
        "created_at": 1_700_000_000,
        "tags": [],
    }
    state = adapter._channel_state[CHANNEL]
    await adapter._handle_event(CHANNEL, state, event)
    await adapter._handle_event(CHANNEL, state, event)
    assert len(received) == 1
    assert state["last_ts"] == 1_700_000_000


@pytest.mark.asyncio
async def test_start_websocket_times_out_and_cleans_up(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(_buzz_mod, "_WS_AUTH_TIMEOUT", 0.0)

    async def never_ready():
        await asyncio.sleep(3600)

    monkeypatch.setattr(adapter, "_websocket_loop", never_ready)
    assert await adapter._start_websocket() is False
    assert adapter._ws_task is None
    assert adapter._ws_active is False


@pytest.mark.asyncio
async def test_membership_event_subscribes_new_conversations():
    adapter = _make_adapter()
    adapter._channel_state = {"old-chan": {"chat_type": "group", "last_ts": 5, "seen": {}}}

    async def fake_discover(seed):
        adapter._channel_state["new-dm"] = {"chat_type": "dm", "last_ts": 0, "seen": {}}

    adapter._discover_dms = fake_discover
    ws = _FakeWebSocket()
    subscriptions = {"hermes-buzz-0": "old-chan"}
    await adapter._handle_membership_event(
        ws, subscriptions, {"created_at": int(time.time()), "tags": []}
    )
    assert "new-dm" in subscriptions.values()
    assert any(f[2].get("#h") == ["new-dm"] for f in ws.sent)
