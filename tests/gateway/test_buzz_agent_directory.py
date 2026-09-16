"""Tests for the Buzz WebSocket transport (NIP-42) and Nostr signing module.

The signing module and WS transport were contributed in PR #73636 by
@ScaleLeanChris and consolidated onto the merged poll-based adapter; these
tests cover the crypto (against the official BIP-340 vector) and the WS
lifecycle as wired into BuzzAdapter.
"""

import asyncio
import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter
from plugins.platforms.buzz import settings as buzz_settings

_buzz_mod = load_plugin_adapter("buzz")
BuzzAdapter = _buzz_mod.BuzzAdapter

import importlib.util as _ilu
from pathlib import Path as _Path

_auth_path = _Path(_buzz_mod.__file__).with_name("nostr_auth.py")
_spec = _ilu.spec_from_file_location("plugin_adapter_buzz_nostr_auth", _auth_path)
nostr_auth = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(nostr_auth)

# BIP-340 test vector 0 private key
TEST_PRIVATE_KEY = "00" * 31 + "03"
SELF_PUBKEY = nostr_auth.public_key_hex(TEST_PRIVATE_KEY)
OWNER_PRIVATE_KEY = "00" * 31 + "04"
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"


@pytest.fixture(autouse=True)
def _clean_transport_override(monkeypatch):
    monkeypatch.delenv("BUZZ_TRANSPORT", raising=False)
    monkeypatch.delenv("BUZZ_AUTH_TAG", raising=False)
    monkeypatch.delenv("BUZZ_CHANNELS", raising=False)


def _make_adapter(extra=None):
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay", **(extra or {})})
    adapter = BuzzAdapter(cfg)
    adapter._self_pubkey = SELF_PUBKEY
    adapter._private_key = TEST_PRIVATE_KEY
    adapter._display_name = "Chip"
    return adapter


def _owner_tag(agent_pubkey=None, conditions=""):
    agent_pubkey = agent_pubkey or nostr_auth.public_key_hex(TEST_PRIVATE_KEY)
    message = __import__("hashlib").sha256(
        f"nostr:agent-auth:{agent_pubkey}:{conditions}".encode()
    ).digest()
    return [
        "auth",
        nostr_auth.public_key_hex(OWNER_PRIVATE_KEY),
        conditions,
        nostr_auth.schnorr_sign(
            message, OWNER_PRIVATE_KEY, auxiliary_randomness=bytes(32)
        ).hex(),
    ]


@pytest.mark.asyncio
async def test_websocket_connection_republishes_directory_before_subscribing(
    monkeypatch,
):
    adapter = _make_adapter()
    websocket = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = websocket
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        MagicMock(connect=MagicMock(return_value=context)),
    )
    order = []
    adapter._authenticate_websocket = AsyncMock(
        side_effect=lambda _ws: order.append("auth")
    )
    adapter._publish_directory_websocket = AsyncMock(
        side_effect=lambda _ws, force=False: order.append(("publish", force))
    )

    async def cancel_after_subscribe(_ws):
        order.append("subscribe")
        raise asyncio.CancelledError

    adapter._subscribe_websocket = cancel_after_subscribe
    with pytest.raises(asyncio.CancelledError):
        await adapter._websocket_loop()

    assert order == ["auth", ("publish", True), "subscribe"]



def test_signed_directory_event_deterministic_vector_and_signature():
    event = nostr_auth.build_signed_event(
        private_key=TEST_PRIVATE_KEY,
        kind=10100,
        tags=[],
        content='{"status":"online"}',
        created_at=1_700_000_000,
        auxiliary_randomness=bytes(32),
    )
    assert event["id"] == "14073c61e18826eb743cbd4928a7ab30df8d1fed53846ef53c9fdeb287f34093"
    assert event["sig"] == (
        "2df61fd0276a03f6b1392a276a0ba40ef98fa9107f19def3263de9016a11fbae"
        "fdfe3ed458c86cdd6e507ed10ce6344390236ac491f1e1ad5fe2037f3ff06c16"
    )
    assert nostr_auth.schnorr_verify(
        bytes.fromhex(event["id"]), event["pubkey"], event["sig"]
    )


def test_nip_oa_verification_accepts_exact_agent_and_rejects_wrong_agent():
    tag = _owner_tag(conditions="kind=10100&created_at>0")
    assert nostr_auth.verify_auth_tag_for_event(
        tag,
        nostr_auth.public_key_hex(TEST_PRIVATE_KEY),
        kind=10100,
        created_at=1,
    )
    with pytest.raises(ValueError):
        nostr_auth.verify_auth_tag_for_event(
            tag,
            nostr_auth.public_key_hex("00" * 31 + "05"),
            kind=10100,
            created_at=1,
        )


@pytest.mark.parametrize(
    "conditions",
    [
        "kind=9",
        "created_at<1",
        "kind=10100&created_at>10&created_at<5",
    ],
)
def test_nip_oa_directory_conditions_reject_incompatible_event(conditions):
    with pytest.raises(ValueError, match="condition"):
        nostr_auth.verify_auth_tag_for_event(
            _owner_tag(conditions=conditions),
            nostr_auth.public_key_hex(TEST_PRIVATE_KEY),
            kind=10100,
            created_at=10,
        )


def test_nip_oa_directory_conditions_accept_compatible_kind_and_time():
    tag = _owner_tag(conditions="kind=10100&created_at>9&created_at<11")
    assert nostr_auth.verify_auth_tag_for_event(
        tag,
        nostr_auth.public_key_hex(TEST_PRIVATE_KEY),
        kind=10100,
        created_at=10,
    )


@pytest.mark.parametrize(
    "conditions",
    [
        "kind=10100 ",
        "kind=10100&",
        "&kind=10100",
        "kind!=10100",
        "kind=010100",
        "kind=65536",
        "created_at<4294967296",
    ],
)
def test_nip_oa_condition_grammar_rejects_noncanonical_forms(conditions):
    with pytest.raises(ValueError):
        nostr_auth.verify_auth_tag_for_event(
            _owner_tag(conditions=conditions),
            nostr_auth.public_key_hex(TEST_PRIVATE_KEY),
            kind=10100,
            created_at=10,
        )


@pytest.mark.parametrize("conditions", ["created_at<10", "created_at>10"])
def test_nip_oa_created_at_boundaries_are_strict(conditions):
    with pytest.raises(ValueError, match="created_at condition"):
        nostr_auth.verify_auth_tag_for_event(
            _owner_tag(conditions=conditions),
            nostr_auth.public_key_hex(TEST_PRIVATE_KEY),
            kind=10100,
            created_at=10,
        )


def test_nip_oa_rejects_self_attestation_and_non_lowercase_keys():
    agent_pubkey = nostr_auth.public_key_hex(TEST_PRIVATE_KEY)
    with pytest.raises(ValueError, match="self-attestation"):
        nostr_auth.verify_auth_tag(
            ["auth", agent_pubkey, "", "0" * 128], agent_pubkey
        )
    with pytest.raises(ValueError, match="invalid label or pubkey"):
        nostr_auth.verify_auth_tag(_owner_tag(), agent_pubkey.upper())
    uppercase_owner = _owner_tag()
    uppercase_owner[1] = uppercase_owner[1].upper()
    with pytest.raises(ValueError, match="invalid label or pubkey"):
        nostr_auth.verify_auth_tag(uppercase_owner, agent_pubkey)


def test_schnorr_verify_rejects_mutation_off_curve_and_scalar_bounds():
    message = bytes.fromhex("22" * 32)
    private_key = "00" * 31 + "02"
    public_key = nostr_auth.public_key_hex(private_key)
    signature = nostr_auth.schnorr_sign(
        message, private_key, auxiliary_randomness=bytes(32)
    )
    mutated = signature[:-1] + bytes([signature[-1] ^ 1])

    assert not nostr_auth.schnorr_verify(message, public_key, mutated.hex())
    assert not nostr_auth.schnorr_verify(message, "0" * 63 + "5", signature.hex())
    r_out_of_range = nostr_auth.FIELD_ORDER.to_bytes(32, "big") + signature[32:]
    s_out_of_range = signature[:32] + nostr_auth.CURVE_ORDER.to_bytes(32, "big")
    assert not nostr_auth.schnorr_verify(message, public_key, r_out_of_range.hex())
    assert not nostr_auth.schnorr_verify(message, public_key, s_out_of_range.hex())






def test_directory_projection_uses_live_policy_and_group_channels_only(monkeypatch):
    allowed = ["d" * 64, "a" * 64]
    monkeypatch.setattr(
        buzz_settings,
        "effective_authorization_policy",
        lambda *args, **kwargs: {"allowed_users": allowed, "allow_all_users": False},
    )
    adapter = _make_adapter()
    adapter._profile_name = "chip-agent"
    adapter._channel_state = {
        CHANNEL: {"chat_type": "group", "last_ts": 0, "seen": {}},
        "unjoined-id": {"chat_type": "group", "last_ts": 0, "seen": {}},
        "dm-id": {"chat_type": "dm", "last_ts": 0, "seen": {}},
        "relay-dm-id": {"chat_type": "group", "last_ts": 0, "seen": {}},
    }
    adapter._joined_channel_ids = {CHANNEL, "relay-dm-id"}
    adapter._channel_names = {
        CHANNEL: "general",
        "unjoined-id": "not joined",
        "dm-id": "DM",
        "relay-dm-id": "DM",
    }
    adapter._channel_meta = {
        CHANNEL: {"channel_id": CHANNEL, "name": "general", "description": "group"},
        "relay-dm-id": {
            "channel_id": "relay-dm-id",
            "name": "DM",
            "description": "",
        },
    }

    assert adapter._directory_content() == {
        "name": "chip-agent",
        "display_name": "Chip",
        "agent_type": "hermes-gateway",
        "capabilities": ["chat"],
        "status": "online",
        "respond_to": "allowlist",
        "respond_to_allowlist": sorted(allowed),
        "channels": ["general"],
        "channel_ids": [CHANNEL],
        "channel_add_policy": "owner_only",
    }


def test_directory_projection_never_uses_config_only_allow_all(monkeypatch):
    monkeypatch.delenv("BUZZ_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    adapter = _make_adapter({"allow_all_users": True})
    assert adapter._directory_content()["respond_to"] == "allowlist"
    assert adapter._directory_content()["respond_to_allowlist"] == []


def test_directory_projection_default_deny_is_buzz_picker_compatible(monkeypatch):
    for name in (
        "BUZZ_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(name, raising=False)
    content = _make_adapter()._directory_content()
    assert content["status"] == "online"
    assert content["respond_to"] == "allowlist"
    assert content["respond_to_allowlist"] == []


def test_directory_projection_never_widens_adapter_allowlist(monkeypatch):
    allowed = "a" * 64
    monkeypatch.setenv("BUZZ_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    adapter = _make_adapter({"allowed_users": [allowed]})

    projection = adapter._directory_content()

    assert projection["respond_to"] == "allowlist"
    assert projection["respond_to_allowlist"] == [allowed]


@pytest.mark.asyncio
async def test_publish_directory_signs_with_adapter_identity_and_requires_matching_ack(
    monkeypatch,
):
    monkeypatch.setattr(
        buzz_settings,
        "effective_authorization_policy",
        lambda *args, **kwargs: {"allowed_users": [], "allow_all_users": False},
    )
    monkeypatch.setattr(_buzz_mod.time, "time", lambda: 10)
    monkeypatch.setenv(
        "BUZZ_AUTH_TAG",
        json.dumps(_owner_tag(conditions="kind=10100&created_at>9")),
    )
    adapter = _make_adapter()
    adapter._self_pubkey = nostr_auth.public_key_hex(TEST_PRIVATE_KEY)

    class AckWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, raw):
            self.sent.append(json.loads(raw))

        async def recv(self):
            event = self.sent[-1][1]
            return json.dumps(["OK", event["id"], True, "stored"])

    websocket = AckWebSocket()
    assert await adapter._publish_directory_websocket(websocket) is True
    event = websocket.sent[0][1]
    assert event["kind"] == 10100
    assert event["pubkey"] == adapter._self_pubkey
    assert event["tags"] == [_owner_tag(conditions="kind=10100&created_at>9")]
    assert nostr_auth.schnorr_verify(
        bytes.fromhex(event["id"]), event["pubkey"], event["sig"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conditions",
    [
        "kind=9",
        "created_at<1",
        "kind=10100&created_at>10&created_at<5",
    ],
)
async def test_directory_publication_rejects_incompatible_owner_conditions(
    monkeypatch, conditions
):
    monkeypatch.setattr(_buzz_mod.time, "time", lambda: 10)
    monkeypatch.setenv("BUZZ_AUTH_TAG", json.dumps(_owner_tag(conditions=conditions)))
    adapter = _make_adapter()
    adapter._self_pubkey = nostr_auth.public_key_hex(TEST_PRIVATE_KEY)
    with pytest.raises(ValueError, match="condition"):
        await adapter._publish_directory_websocket(AsyncMock())


@pytest.mark.asyncio
async def test_directory_publication_accepts_compatible_owner_conditions(monkeypatch):
    monkeypatch.setattr(_buzz_mod.time, "time", lambda: 10)
    monkeypatch.setenv(
        "BUZZ_AUTH_TAG",
        json.dumps(_owner_tag(conditions="kind=10100&created_at>9&created_at<11")),
    )
    websocket = AsyncMock()
    websocket.recv.side_effect = lambda: json.dumps(
        ["OK", json.loads(websocket.send.await_args.args[0])[1]["id"], True, "stored"]
    )
    adapter = _make_adapter()
    adapter._self_pubkey = nostr_auth.public_key_hex(TEST_PRIVATE_KEY)
    assert await adapter._publish_directory_websocket(websocket) is True


@pytest.mark.asyncio
async def test_directory_publication_rejects_profile_signer_identity_mismatch(
    monkeypatch,
):
    profile_pubkey = nostr_auth.public_key_hex(TEST_PRIVATE_KEY)
    signer_private_key = "00" * 31 + "02"
    monkeypatch.setenv("BUZZ_AUTH_TAG", json.dumps(_owner_tag()))
    adapter = _make_adapter()
    adapter._self_pubkey = profile_pubkey
    adapter._private_key = signer_private_key
    websocket = AsyncMock()
    websocket.recv.side_effect = lambda: json.dumps(
        ["OK", json.loads(websocket.send.await_args.args[0])[1]["id"], True, "stored"]
    )

    with pytest.raises(ValueError, match="identity.*signing key"):
        await adapter._publish_directory_websocket(websocket)

    websocket.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_directory_ack_wait_defers_interleaved_subscription_frame():
    adapter = _make_adapter()
    interleaved = json.dumps(["EVENT", "hermes-buzz-0", {"id": "message"}])

    class InterleavedWebSocket:
        def __init__(self):
            self.event = None
            self.responses = [interleaved]

        async def send(self, raw):
            self.event = json.loads(raw)[1]

        async def recv(self):
            if self.responses:
                return self.responses.pop(0)
            return json.dumps(["OK", self.event["id"], True, "stored"])

    assert await adapter._publish_directory_websocket(InterleavedWebSocket()) is True
    assert list(adapter._ws_deferred_frames) == [interleaved]


@pytest.mark.asyncio
async def test_directory_ack_wait_uses_one_absolute_deadline_for_drip_frames(
    monkeypatch,
):
    monkeypatch.delenv("BUZZ_AUTH_TAG", raising=False)
    adapter = _make_adapter()
    adapter._self_pubkey = nostr_auth.public_key_hex(TEST_PRIVATE_KEY)
    unrelated = json.dumps(["NOTICE", "still waiting"])
    deadlines = []

    async def recv_before_deadline(_websocket, deadline, _message):
        deadlines.append(deadline)
        if len(deadlines) == 1:
            return unrelated
        raise TimeoutError("directory ACK timed out")

    adapter._recv_before_deadline = recv_before_deadline
    websocket = AsyncMock()
    websocket.recv.side_effect = [unrelated, TimeoutError("directory ACK timed out")]

    with pytest.raises(TimeoutError, match="directory ACK timed out"):
        await adapter._publish_directory_websocket(websocket)

    assert len(deadlines) == 2
    assert deadlines[0] == deadlines[1]


@pytest.mark.asyncio
async def test_directory_publication_deduplicates_projection_but_republishes_on_reconnect():
    adapter = _make_adapter()

    class AckWebSocket:
        def __init__(self):
            self.events = []

        async def send(self, raw):
            frame = json.loads(raw)
            if frame[0] == "EVENT":
                self.events.append(frame[1])

        async def recv(self):
            return json.dumps(["OK", self.events[-1]["id"], True, "stored"])

    websocket = AckWebSocket()
    assert await adapter._publish_directory_websocket(websocket) is True
    assert await adapter._publish_directory_websocket(websocket) is False
    assert await adapter._publish_directory_websocket(websocket, force=True) is True
    assert len(websocket.events) == 2


@pytest.mark.asyncio
async def test_poll_transport_uses_bounded_authenticated_websocket_for_directory(
    monkeypatch,
):
    adapter = _make_adapter({"transport": "poll"})
    websocket = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = websocket
    connect = MagicMock(return_value=context)
    monkeypatch.setitem(sys.modules, "websockets", MagicMock(connect=connect))
    adapter._authenticate_websocket = AsyncMock()
    adapter._publish_directory_websocket = AsyncMock(return_value=True)

    assert await adapter._publish_directory_fallback() is True
    adapter._authenticate_websocket.assert_awaited_once_with(websocket)
    adapter._publish_directory_websocket.assert_awaited_once_with(
        websocket, force=False
    )
    assert connect.call_args.kwargs["open_timeout"] == _buzz_mod._WS_AUTH_TIMEOUT
    assert connect.call_args.kwargs["max_size"] == _buzz_mod._WS_MAX_MESSAGE_BYTES


@pytest.mark.asyncio
async def test_directory_publication_ignores_nonmatching_positive_ack():
    adapter = _make_adapter()

    class WrongThenRejectedWebSocket:
        def __init__(self):
            self.event = None
            self.responses = 0

        async def send(self, raw):
            self.event = json.loads(raw)[1]

        async def recv(self):
            self.responses += 1
            if self.responses == 1:
                return json.dumps(["OK", "0" * 64, True, "stored another event"])
            return json.dumps(["OK", self.event["id"], False, "denied"])

    with pytest.raises(ConnectionError, match="directory publication rejected"):
        await adapter._publish_directory_websocket(WrongThenRejectedWebSocket())



@pytest.mark.asyncio
async def test_read_loop_drains_directory_deferred_frames_exactly_once():
    adapter = _make_adapter()
    event = {"id": "deferred", "created_at": 1}
    adapter._ws_deferred_frames.append(json.dumps(["EVENT", "room", event]))
    class Socket:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise StopAsyncIteration
    adapter._channel_state[CHANNEL] = adapter._new_channel_state("group")
    adapter._handle_events = AsyncMock()
    # A clean relay close is reported as a disconnect after the deferred frame was consumed.
    with pytest.raises(ConnectionError):
        await adapter._ws_read_loop(Socket(), {"room": CHANNEL})
    adapter._handle_events.assert_awaited_once_with(CHANNEL, adapter._channel_state[CHANNEL], [event])
    assert not adapter._ws_deferred_frames


def test_directory_uses_real_live_owner_authorization_not_mention_policy(tmp_path, monkeypatch):
    from tests.gateway.test_buzz_mention_settings import save, policy
    owner, routed = tmp_path / "owner", tmp_path / "routed"
    save(owner, policy(allow_all_users=True, allowed_users=[]))
    save(routed, policy(allow_all_users=False, allowed_users=["b" * 64]))
    monkeypatch.setenv("HERMES_HOME", str(owner))
    adapter = _make_adapter()
    monkeypatch.setenv("HERMES_HOME", str(routed))
    assert adapter._directory_content()["respond_to"] == "anyone"
    save(owner, policy(allow_all_users=False, allowed_users=["a" * 64]))
    assert adapter._directory_content()["respond_to_allowlist"] == ["a" * 64]
    save(owner, policy(allow_all_users=False, allowed_users=[]))
    content = adapter._directory_content()
    assert content["respond_to"] == "allowlist"
    assert content["respond_to_allowlist"] == []


@pytest.mark.asyncio
async def test_real_signed_reconnect_republishes_and_drains_interleaved_events(monkeypatch):
    adapter = _make_adapter()
    adapter._channel_state[CHANNEL] = adapter._new_channel_state("group")
    adapter._joined_channel_ids = {CHANNEL}
    adapter._handle_event = AsyncMock()
    sockets = []
    class Socket:
        def __init__(self, index):
            self.index = index
            self.sent = []
            self.replies = [json.dumps(["AUTH", "challenge"])]
        async def send(self, raw):
            frame = json.loads(raw)
            self.sent.append(frame)
            if frame[0] in ("AUTH", "EVENT"):
                event = frame[1]
                assert nostr_auth.schnorr_verify(bytes.fromhex(event["id"]), event["pubkey"], event["sig"])
                if frame[0] == "EVENT":
                    self.replies.append(json.dumps(["EVENT", "hermes-buzz-0", {"id": f"message-{self.index}", "created_at": 1}]))
                self.replies.append(json.dumps(["OK", event["id"], True, "stored"]))
        async def recv(self):
            assert self.replies, "unexpected second socket reader"
            return self.replies.pop(0)
        def __aiter__(self):
            return self
        async def __anext__(self):
            assert not self.replies, "ACK path left a frame behind"
            if self.index == 0:
                raise StopAsyncIteration
            raise asyncio.CancelledError
    class Connection:
        async def __aenter__(self):
            socket = Socket(len(sockets))
            sockets.append(socket)
            return socket
        async def __aexit__(self, *args):
            return False
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=lambda *a, **kw: Connection()))
    # The discovery loop imports the real exception type on each connection.
    monkeypatch.setitem(sys.modules, "websockets.exceptions", SimpleNamespace(ConnectionClosed=ConnectionError))
    with pytest.raises(asyncio.CancelledError):
        await adapter._websocket_loop()
    assert len(sockets) == 2
    for socket in sockets:
        kinds = [frame[1]["kind"] for frame in socket.sent if frame[0] in ("AUTH", "EVENT")]
        assert kinds == [22242, 10100]
        assert any(frame[0] == "REQ" for frame in socket.sent)
    assert [call.args[2]["id"] for call in adapter._handle_event.await_args_list] == ["message-0", "message-1"]
    assert not adapter._ws_deferred_frames


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, 1, "true"])
async def test_directory_rejection_does_not_commit_dedup_state(accepted):
    adapter = _make_adapter()
    socket = AsyncMock()
    socket.recv.side_effect = lambda: json.dumps(["OK", json.loads(socket.send.await_args.args[0])[1]["id"], accepted, "denied"])
    with pytest.raises(ConnectionError):
        await adapter._publish_directory_websocket(socket)
    assert adapter._last_directory_projection is None
    socket.recv.side_effect = lambda: json.dumps(["OK", json.loads(socket.send.await_args.args[0])[1]["id"], True, "stored"])
    assert await adapter._publish_directory_websocket(socket)
    assert socket.send.await_count == 2


@pytest.mark.asyncio
async def test_directory_cancel_is_propagated_without_committing_projection():
    adapter = _make_adapter()
    socket = AsyncMock()
    socket.recv.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await adapter._publish_directory_websocket(socket)
    assert adapter._last_directory_projection is None
