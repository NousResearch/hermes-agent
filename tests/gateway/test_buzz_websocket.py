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

import pytest
from unittest.mock import AsyncMock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz_mod = load_plugin_adapter("buzz")
BuzzAdapter = _buzz_mod.BuzzAdapter

import importlib.util as _ilu
from pathlib import Path as _Path

_auth_path = _Path(_buzz_mod.__file__).with_name("nostr_auth.py")
_spec = _ilu.spec_from_file_location("plugin_adapter_buzz_nostr_auth", _auth_path)
nostr_auth = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(nostr_auth)

SELF_PUBKEY = "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9"  # public_key_hex(TEST_PRIVATE_KEY)
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
    # Transport-only sockets here bypass AUTH and omit recv/ACK; directory publication is covered
    # by test_buzz_agent_directory.py with real signed events.
    adapter._publish_directory_websocket = AsyncMock(return_value=True)
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


def test_nip44_encrypt_matches_official_vector():
    """Observer payload encryption must be byte-compatible with NIP-44 v2."""
    sender_private_key = "00" * 31 + "01"
    recipient_private_key = "00" * 31 + "02"
    recipient_pubkey = nostr_auth.public_key_hex(recipient_private_key)

    payload = nostr_auth.nip44_encrypt(
        "a",
        private_key=sender_private_key,
        recipient_pubkey=recipient_pubkey,
        nonce=bytes.fromhex("00" * 31 + "01"),
    )

    assert payload == (
        "AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABee0G5VSK0/9YypIObAtD"
        "KfYEAjD35uVkHyB0F4DwrcNaCXlCWZKaArsGrY6M9wnuTMxWfp1RTN9Xga8no+"
        "kF5Vsb"
    )


def test_nip44_encrypt_rejects_non_spec_plaintext_above_65535_bytes():
    with pytest.raises(ValueError, match="1 to 65535 bytes"):
        nostr_auth.nip44_encrypt(
            "a" * 65_536,
            private_key="00" * 31 + "01",
            recipient_pubkey=nostr_auth.public_key_hex("00" * 31 + "02"),
            nonce=bytes(32),
        )


def test_build_observer_event_encrypts_and_signs_nip_ao_shape(monkeypatch):
    owner_private_key = "00" * 31 + "02"
    owner_pubkey = nostr_auth.public_key_hex(owner_private_key)
    captured = {}

    def fake_encrypt(plaintext, **kwargs):
        captured["plaintext"] = plaintext
        captured.update(kwargs)
        return "encrypted-observer-payload"

    monkeypatch.setattr(nostr_auth, "nip44_encrypt", fake_encrypt)
    payload = {
        "seq": 1,
        "timestamp": "2026-08-03T14:00:00.000Z",
        "kind": "turn_started",
        "agentIndex": None,
        "channelId": CHANNEL,
        "sessionId": "session-1",
        "turnId": "turn-1",
        "payload": {"source": "channel"},
    }

    event = nostr_auth.build_observer_event(
        private_key=TEST_PRIVATE_KEY,
        owner_pubkey=owner_pubkey,
        payload=payload,
        created_at=1_700_000_000,
        auxiliary_randomness=bytes(32),
    )

    assert event["kind"] == 24200
    assert event["content"] == "encrypted-observer-payload"
    assert ["p", owner_pubkey] in event["tags"]
    assert ["agent", event["pubkey"]] in event["tags"]
    assert ["frame", "telemetry"] in event["tags"]
    assert json.loads(captured["plaintext"]) == payload
    assert captured["private_key"] == TEST_PRIVATE_KEY
    assert captured["recipient_pubkey"] == owner_pubkey
    assert len(bytes.fromhex(event["sig"])) == 64


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
    forever while the gateway keeps reporting "connected". The receive here
    also ignores cancellation (#112049), which held the bound hostage as long
    as it was an ``asyncio.wait_for``.
    """
    import logging

    adapter = _make_adapter()
    monkeypatch.setattr(_buzz_mod, "_WS_READ_IDLE_TIMEOUT", 0.05)
    caplog.set_level(logging.WARNING)
    states = []
    monkeypatch.setattr(adapter, "_write_runtime_status_safe", lambda status, **kw: states.append(kw["platform_state"]))

    sockets = []
    release_parked_receive = asyncio.Event()

    async def dead_anext():
        # Some transports do not acknowledge cancellation while their receive
        # call is parked.  ``asyncio.wait_for`` then waits for this coroutine
        # to finish, so its own timeout cannot return.
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release_parked_receive.wait()
            raise

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
        release_parked_receive.set()
        task.cancel()
        try:
            await asyncio.wait_for(task, 5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    assert len(sockets) >= 2, "idle read watchdog did not force a reconnect"
    assert sockets[0].exited, "the silent connection was not closed before reconnecting"
    assert any("went silent" in record.message for record in caplog.records)
    assert states[:2] == ["retrying", "connected"], f"health must flip to retrying and back, got {states}"


@pytest.mark.asyncio
async def test_websocket_loop_reconnects_when_discovery_send_sees_closed_socket(monkeypatch):
    """A send-side ConnectionClosed proves the socket is dead even while the read is parked.

    #112049: the discovery sweep logged ``ConnectionClosedError`` for hours while
    ``_ws_read_loop`` sat on a receive that never returned; the sweep must end the
    connection so the reconnect path runs instead of retrying next tick forever.
    """
    from websockets.exceptions import ConnectionClosedError

    adapter = _make_adapter()
    adapter.poll_interval = 0.02
    monkeypatch.setattr(_buzz_mod, "_MIN_POLL_INTERVAL", 0.02)
    monkeypatch.setattr(_buzz_mod, "_DM_DISCOVERY_EVERY", 1)
    monkeypatch.setattr(_buzz_mod, "_WS_READ_IDLE_TIMEOUT", 30.0)  # far away: only the send side can end this

    sockets = []

    async def silent_anext():
        sockets[-1].dead = True  # the read loop is now parked; the relay side is gone
        await asyncio.Event().wait()

    async def rediscover(websocket, subscriptions):
        await websocket.send(json.dumps(["REQ", "hermes-buzz-dm-1", {}]))

    monkeypatch.setattr(adapter, "_rediscover_and_subscribe", rediscover)

    def fake_connect(*args, **kwargs):
        ws = _ScriptedWebSocket(silent_anext)
        ws.dead = False
        real_send = ws.send

        async def send(raw):
            if ws.dead:
                raise ConnectionClosedError(None, None)
            await real_send(raw)

        ws.send = send
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

    assert len(sockets) >= 2, "a closed socket seen by the discovery sweep did not force a reconnect"
    assert sockets[0].exited, "the dead connection was not closed before reconnecting"


@pytest.mark.asyncio
async def test_websocket_loop_backs_off_and_publishes_retrying_on_clean_relay_close(monkeypatch):
    """A relay that accepts, then cleanly closes after subscribe, is a disconnect like any other.

    StopAsyncIteration used to return from the read loop without raising, so the
    loop reconnected in a hot loop with no backoff and health stayed "connected".
    """
    adapter = _make_adapter()
    states = []
    monkeypatch.setattr(adapter, "_write_runtime_status_safe", lambda status, **kw: states.append(kw["platform_state"]))

    async def clean_close():
        raise StopAsyncIteration

    sockets = []

    def fake_connect(*args, **kwargs):
        ws = _ScriptedWebSocket(clean_close)
        sockets.append(ws)
        return ws

    import websockets as _ws_mod

    monkeypatch.setattr(_ws_mod, "connect", fake_connect)

    task = asyncio.create_task(adapter._websocket_loop())
    try:
        await asyncio.sleep(0.3)
    finally:
        task.cancel()
        try:
            await asyncio.wait_for(task, 5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    assert len(sockets) == 1, f"clean close must back off before reconnecting, got {len(sockets)} connects in 0.3s"
    assert sockets[0].exited, "the closed connection was not exited before backing off"
    assert states == ["retrying"], f"a clean relay close must publish retrying, got {states}"


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


# ── Fresh-conversation subscription window (#78429) ────────────────────────


@pytest.mark.asyncio
async def test_new_subscription_without_high_water_mark_has_no_since_floor():
    """A conversation adopted mid-run (last_ts == 0) must NOT subscribe with
    `since ≈ now` — that drops the message that created the conversation
    (#78429). It must request from the beginning with a bounded limit."""
    adapter = _make_adapter()
    adapter._channel_state[CHANNEL] = {"chat_type": "dm", "last_ts": 0, "seen": {}}

    class _Ws:
        def __init__(self):
            self.sent = []

        async def send(self, raw):
            self.sent.append(json.loads(raw))

    ws = _Ws()
    await adapter._send_channel_subscription(ws, "hermes-buzz-dm-1", CHANNEL)
    assert len(ws.sent) == 1
    req_filter = ws.sent[0][2]
    assert "since" not in req_filter, (
        "fresh conversation must not have a since floor (drops the opening message)"
    )
    assert req_filter.get("limit") == _buzz_mod._FETCH_LIMIT
    assert req_filter["#h"] == [CHANNEL]


@pytest.mark.asyncio
async def test_seeded_subscription_resumes_from_high_water_mark():
    """A channel with a real high-water mark keeps the since-resume contract
    (last_ts - 1, same-second overlap de-duped by id)."""
    adapter = _make_adapter()
    adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 1_700_000_000, "seen": {}}

    class _Ws:
        def __init__(self):
            self.sent = []

        async def send(self, raw):
            self.sent.append(json.loads(raw))

    ws = _Ws()
    await adapter._send_channel_subscription(ws, "hermes-buzz-0", CHANNEL)
    req_filter = ws.sent[0][2]
    assert req_filter["since"] == 1_699_999_999
    assert "limit" not in req_filter


# ── Membership-rejection phrasing (#97502 composed into #76850) ────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "detail",
    [
        "restricted: not a channel member",
        "not a channel member",
        "auth-required: subscription needs auth",
    ],
)
async def test_closed_membership_phrases_prune_without_reconnect(detail):
    """Every production-observed membership-rejection phrasing (#76850,
    #97502) prunes the subscription instead of tearing down the socket."""
    import sys
    from unittest.mock import patch, MagicMock
    from contextlib import asynccontextmanager

    adapter = _make_adapter(extra={"channels": [CHANNEL]})
    adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}

    sub_id = "hermes-buzz-0"
    messages = [json.dumps(["CLOSED", sub_id, detail])]
    idx = 0

    class _FakeWs:
        async def send(self, raw):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            nonlocal idx
            if idx < len(messages):
                val = messages[idx]
                idx += 1
                return val
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
        task = asyncio.create_task(adapter._websocket_loop())
        await asyncio.sleep(0.1)

    assert CHANNEL in adapter._restricted_channels
    assert CHANNEL not in adapter._channel_state
    assert not task.done(), "membership rejection must not reconnect the socket"

    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_restricted_channel_not_readopted_by_discovery():
    """The live-discovery paths must skip _restricted_channels; otherwise the
    next sweep silently re-adds the channel and re-triggers the rejection
    (the re-adoption hole flagged in the #76850 review)."""
    adapter = _make_adapter()
    adapter._restricted_channels.add(CHANNEL)

    calls = []

    async def scripted_cli(args, *, input_text=None):
        calls.append(list(args))
        if args[:2] == ["dms", "list"]:
            return 0, json.dumps([{"dm_id": CHANNEL}]), ""
        if args[:2] == ["channels", "list"]:
            return 0, json.dumps([{"channel_id": CHANNEL, "name": "DM", "description": ""}]), ""
        return 0, "[]", ""

    adapter._run_cli = scripted_cli
    await adapter._discover_dms(seed=False)
    assert CHANNEL not in adapter._channel_state, (
        "restricted channel must not be re-adopted by discovery"
    )


# ── WS periodic discovery (#93557 / #75107) ────────────────────────────────


@pytest.mark.asyncio
async def test_ws_discovery_loop_subscribes_newly_discovered_conversation(monkeypatch):
    """Without a kind-44100 membership event, the WS transport's periodic
    sweep must still find and subscribe a conversation opened mid-session
    (#93557)."""
    adapter = _make_adapter()
    adapter.poll_interval = 0.01

    new_dm = "0f0e0d0c-0b0a-4123-8123-cafecafecafe"

    async def fake_discover(*, seed):
        adapter._channel_state.setdefault(
            new_dm, {"chat_type": "dm", "last_ts": 0, "seen": {}}
        )

    monkeypatch.setattr(adapter, "_discover_dms", fake_discover)
    monkeypatch.setattr(_buzz_mod, "_MIN_POLL_INTERVAL", 0.01)

    class _Ws:
        def __init__(self):
            self.sent = []

        async def send(self, raw):
            self.sent.append(json.loads(raw))

    ws = _Ws()
    subscriptions = {"hermes-buzz-0": CHANNEL}
    task = asyncio.create_task(adapter._ws_discovery_loop(ws, subscriptions))
    try:
        deadline = time.monotonic() + 5.0
        while not ws.sent and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert new_dm in subscriptions.values(), "sweep did not subscribe the new conversation"
    req = ws.sent[0]
    assert req[0] == "REQ" and req[2]["#h"] == [new_dm]
    # Fresh conversation: no since floor (#78429 applies here too).
    assert "since" not in req[2]


@pytest.mark.asyncio
async def test_ws_discovery_task_cancelled_when_connection_exits(monkeypatch):
    """The companion discovery task must not outlive its connection."""
    adapter = _make_adapter()
    adapter.poll_interval = 10.0  # sweep never fires; we only test lifecycle
    adapter._channel_state = {CHANNEL: {"chat_type": "group", "last_ts": 1, "seen": {}}}

    started = []
    real_create_task = asyncio.create_task

    def tracking_create_task(coro, **kw):
        t = real_create_task(coro, **kw)
        if "_ws_discovery_loop" in repr(coro):
            started.append(t)
        return t

    monkeypatch.setattr(asyncio, "create_task", tracking_create_task)

    async def closed_anext():
        raise StopAsyncIteration

    sockets = []

    def fake_connect(*args, **kwargs):
        if sockets:
            raise asyncio.CancelledError()
        ws = _ScriptedWebSocket(closed_anext)
        sockets.append(ws)
        return ws

    import websockets as _ws_mod

    monkeypatch.setattr(_ws_mod, "connect", fake_connect)

    task = real_create_task(adapter._websocket_loop())
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 10.0)

    assert started, "discovery task was never started with the connection"
    assert all(t.done() for t in started), "discovery task outlived its connection"


@pytest.mark.asyncio
async def test_publish_activity_sends_encrypted_observer_event_over_active_websocket():
    owner_private_key = "00" * 31 + "02"
    owner_pubkey = nostr_auth.public_key_hex(owner_private_key)
    adapter = _make_adapter({"activity_owner_pubkey": owner_pubkey})
    websocket = _FakeWebSocket()
    adapter._ws_active = True
    adapter._ws_connection = websocket

    published = await adapter.publish_activity(
        "turn_started",
        channel_id=CHANNEL,
        session_id="session-1",
        turn_id="turn-1",
        started_at="2026-08-03T14:00:00.000Z",
        payload={"source": "channel"},
    )

    assert published is True
    await asyncio.wait_for(adapter._activity_queue.join(), timeout=1)
    assert len(websocket.sent) == 1
    frame = websocket.sent[0]
    assert frame[0] == "EVENT"
    assert frame[1]["kind"] == 24200
    assert ["p", owner_pubkey] in frame[1]["tags"]
    assert ["agent", frame[1]["pubkey"]] in frame[1]["tags"]
    assert ["frame", "telemetry"] in frame[1]["tags"]
    adapter._activity_sender_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await adapter._activity_sender_task


@pytest.mark.asyncio
async def test_publish_activity_does_not_wait_for_backpressured_websocket():
    owner_pubkey = nostr_auth.public_key_hex("2".zfill(64))
    adapter = _make_adapter(extra={"activity_owner_pubkey": owner_pubkey})

    class _BackpressuredWebSocket:
        async def send(self, _payload):
            await asyncio.Event().wait()

    adapter._ws_connection = _BackpressuredWebSocket()
    adapter._ws_active = True

    published = await asyncio.wait_for(
        adapter.publish_activity(
            "turn_started",
            channel_id="channel-1",
            session_id="session-1",
            turn_id="turn-1",
            payload={},
        ),
        timeout=0.05,
    )

    assert published is True
    assert adapter._activity_sender_task is not None
    adapter._activity_sender_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await adapter._activity_sender_task


def test_activity_owner_pubkey_rejects_malformed_config():
    with pytest.raises(ValueError, match="activity_owner_pubkey"):
        _make_adapter({"activity_owner_pubkey": "not-a-pubkey"})


def test_activity_owner_pubkey_rejects_poll_only_transport():
    owner_pubkey = nostr_auth.public_key_hex("00" * 31 + "02")
    with pytest.raises(ValueError, match="requires transport"):
        _make_adapter(
            {"activity_owner_pubkey": owner_pubkey, "transport": "poll"}
        )


@pytest.mark.asyncio
async def test_activity_auto_fallback_reports_websocket_requirement(
    monkeypatch, caplog
):
    owner_pubkey = nostr_auth.public_key_hex("00" * 31 + "02")
    adapter = _make_adapter({"activity_owner_pubkey": owner_pubkey})

    def invalid_websocket_url():
        raise ValueError("websocket unavailable")

    monkeypatch.setattr(adapter, "_websocket_url", invalid_websocket_url)
    with caplog.at_level("WARNING"):
        assert await adapter._start_websocket() is False

    assert "native activity is unavailable while using polling" in caplog.text


@pytest.mark.asyncio
async def test_terminal_send_failure_retries_once_on_same_live_socket(monkeypatch):
    owner_pubkey = nostr_auth.public_key_hex("00" * 31 + "02")
    adapter = _make_adapter({"activity_owner_pubkey": owner_pubkey})
    monkeypatch.setattr(_buzz_mod, "_ACTIVITY_TERMINAL_RETRY_DELAY", 0.01)

    event_counter = 0

    def build_event(**kwargs):
        nonlocal event_counter
        event_counter += 1
        return {
            "id": f"terminal-event-{event_counter}",
            "kind": 24200,
            "payload": kwargs["payload"],
        }

    monkeypatch.setattr(
        _buzz_mod,
        "_load_nostr_auth",
        lambda: SimpleNamespace(build_observer_event=build_event),
    )

    class FlakyWebSocket:
        def __init__(self):
            self.send_count = 0
            self.delivered = []

        async def send(self, raw):
            self.send_count += 1
            if self.send_count == 1:
                raise TimeoutError("relay backpressure")
            self.delivered.append(json.loads(raw))

    websocket = FlakyWebSocket()
    adapter._ws_active = True
    adapter._ws_connection = websocket
    generation = adapter._activity_ws_generation

    assert await adapter.publish_activity(
        "turn_completed",
        channel_id=CHANNEL,
        session_id="session-1",
        turn_id="turn-1",
    ) is True
    await asyncio.wait_for(adapter._activity_queue.join(), timeout=1)
    assert websocket.send_count == 1
    assert list(adapter._activity_terminal_replay) == ["turn-1"]
    assert not adapter._activity_pending_event_ids

    await asyncio.sleep(0.03)
    await asyncio.wait_for(adapter._activity_queue.join(), timeout=1)
    assert websocket.send_count == 2
    assert adapter._ws_connection is websocket
    assert adapter._activity_ws_generation == generation
    assert [frame[1]["payload"]["kind"] for frame in websocket.delivered] == [
        "turn_completed"
    ]
    assert not adapter._activity_terminal_replay

    await adapter._reset_activity_transport()


@pytest.mark.asyncio
async def test_reset_activity_transport_propagates_cancellation():
    adapter = _make_adapter()
    sender_blocked = asyncio.Event()
    adapter._activity_sender_task = asyncio.create_task(sender_blocked.wait())

    reset_task = asyncio.create_task(adapter._reset_activity_transport())
    await asyncio.sleep(0)
    reset_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await reset_task
    assert reset_task.cancelled()


@pytest.mark.asyncio
async def test_disconnect_completes_while_websocket_loop_is_resetting():
    adapter = _make_adapter()
    reset_entered = asyncio.Event()
    sender_blocked = asyncio.Event()
    adapter._activity_sender_task = asyncio.create_task(sender_blocked.wait())

    async def reconnecting_websocket_loop():
        while True:
            try:
                raise ConnectionError("relay disconnected")
            except Exception:
                reset_entered.set()
                await adapter._reset_activity_transport()
                await asyncio.sleep(3600)

    adapter._ws_task = asyncio.create_task(reconnecting_websocket_loop())
    await asyncio.wait_for(reset_entered.wait(), timeout=1)
    await asyncio.sleep(0)

    disconnect_task = asyncio.create_task(adapter.disconnect())
    try:
        await asyncio.wait_for(asyncio.shield(disconnect_task), timeout=0.1)
    finally:
        if not disconnect_task.done():
            if adapter._ws_task and not adapter._ws_task.done():
                adapter._ws_task.cancel()
            disconnect_task.cancel()
        try:
            await disconnect_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_observer_relay_rejection_is_correlated_and_logged(caplog):
    owner_pubkey = nostr_auth.public_key_hex("2".zfill(64))
    adapter = _make_adapter(extra={"activity_owner_pubkey": owner_pubkey})
    websocket = _FakeWebSocket()
    adapter._ws_connection = websocket
    adapter._ws_active = True

    assert await adapter.publish_activity(
        "turn_started",
        channel_id="channel-1",
        session_id="session-1",
        turn_id="turn-1",
        started_at="2026-08-03T14:00:00.000Z",
        payload={},
    ) is True
    await asyncio.wait_for(adapter._activity_queue.join(), timeout=1)
    event_id = websocket.sent[0][1]["id"]
    assert event_id in adapter._activity_pending_event_ids

    with caplog.at_level("WARNING"):
        assert adapter._handle_activity_ack(
            ["OK", event_id, False, "restricted: not authorized"]
        ) is True

    assert event_id not in adapter._activity_pending_event_ids
    assert "restricted: not authorized" in caplog.text
    adapter._activity_sender_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await adapter._activity_sender_task


@pytest.mark.asyncio
async def test_rejected_terminal_is_retained_and_retried_once(monkeypatch, caplog):
    owner_pubkey = nostr_auth.public_key_hex("2".zfill(64))
    adapter = _make_adapter(extra={"activity_owner_pubkey": owner_pubkey})
    websocket = _FakeWebSocket()
    adapter._ws_connection = websocket
    adapter._ws_active = True
    monkeypatch.setattr(_buzz_mod, "_ACTIVITY_TERMINAL_RETRY_DELAY", 0.01)

    event_counter = 0

    def build_event(**kwargs):
        nonlocal event_counter
        event_counter += 1
        return {
            "id": f"terminal-event-{event_counter}",
            "kind": 24200,
            "payload": kwargs["payload"],
        }

    monkeypatch.setattr(
        _buzz_mod,
        "_load_nostr_auth",
        lambda: SimpleNamespace(build_observer_event=build_event),
    )

    assert await adapter.publish_activity(
        "turn_completed",
        channel_id=CHANNEL,
        session_id="session-1",
        turn_id="turn-1",
    )
    await asyncio.wait_for(adapter._activity_queue.join(), timeout=1)

    with caplog.at_level("WARNING"):
        assert adapter._handle_activity_ack(
            ["OK", "terminal-event-1", False, "rate-limited: slow down"]
        ) is True

    assert list(adapter._activity_terminal_replay) == ["turn-1"]
    await asyncio.sleep(0.03)
    await asyncio.wait_for(adapter._activity_queue.join(), timeout=1)
    assert len(websocket.sent) == 2

    assert adapter._handle_activity_ack(
        ["OK", "terminal-event-2", False, "rate-limited: slow down"]
    ) is True
    await asyncio.sleep(0.03)
    assert len(websocket.sent) == 2
    assert list(adapter._activity_terminal_replay) == ["turn-1"]
    assert "rate-limited: slow down" in caplog.text

    await adapter._reset_activity_transport()


@pytest.mark.asyncio
async def test_disconnect_drops_stale_activity_queue_and_pending_acks():
    adapter = _make_adapter()
    adapter._track_activity_ack("event-id", adapter._activity_ws_generation)
    _, timer = adapter._activity_pending_event_ids["event-id"]
    adapter._activity_queue.put_nowait({"kind": "turn_liveness"})

    await adapter.disconnect()

    assert adapter._activity_queue.empty()
    assert not adapter._activity_pending_event_ids
    assert timer.cancelled()


@pytest.mark.asyncio
async def test_terminal_during_disconnect_replays_once_after_reconnect(monkeypatch):
    owner_pubkey = nostr_auth.public_key_hex("00" * 31 + "02")
    adapter = _make_adapter({"activity_owner_pubkey": owner_pubkey})
    old_websocket = _FakeWebSocket()
    adapter._ws_active = True
    adapter._ws_connection = old_websocket

    counter = 0

    def build_event(**kwargs):
        nonlocal counter
        counter += 1
        return {
            "id": f"event-{counter}",
            "kind": 24200,
            "payload": kwargs["payload"],
        }

    monkeypatch.setattr(
        _buzz_mod,
        "_load_nostr_auth",
        lambda: SimpleNamespace(build_observer_event=build_event),
    )

    assert adapter._enqueue_activity(
        "turn_started",
        channel_id=CHANNEL,
        session_id="session-1",
        turn_id="turn-1",
    )
    await asyncio.wait_for(adapter._activity_queue.join(), timeout=1)
    assert [frame[1]["payload"]["kind"] for frame in old_websocket.sent] == [
        "turn_started"
    ]

    adapter._ws_active = False
    adapter._ws_connection = None
    await adapter._reset_activity_transport()
    assert not adapter._enqueue_activity(
        "turn_liveness",
        channel_id=CHANNEL,
        session_id="session-1",
        turn_id="turn-1",
    )
    assert adapter._enqueue_activity(
        "turn_completed",
        channel_id=CHANNEL,
        session_id="session-1",
        turn_id="turn-1",
    )

    new_websocket = _FakeWebSocket()
    adapter._activity_ws_generation += 1
    adapter._ws_connection = new_websocket
    adapter._ws_active = True
    adapter._replay_terminal_activity()
    await asyncio.wait_for(adapter._activity_queue.join(), timeout=1)

    assert [frame[1]["payload"]["kind"] for frame in new_websocket.sent] == [
        "turn_completed"
    ]
    assert not adapter._activity_terminal_replay
    adapter._replay_terminal_activity()
    await asyncio.sleep(0)
    assert len(new_websocket.sent) == 1
    adapter._activity_sender_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await adapter._activity_sender_task


@pytest.mark.asyncio
async def test_websocket_loop_replays_terminal_after_real_reconnect(monkeypatch):
    owner_pubkey = nostr_auth.public_key_hex("00" * 31 + "02")
    adapter = _make_adapter({"activity_owner_pubkey": owner_pubkey})
    second_delivery = asyncio.Event()

    class RelaySocket(_FakeWebSocket):
        def __init__(self, *, disconnect_after_terminal):
            super().__init__()
            self.disconnect_after_terminal = disconnect_after_terminal
            self.terminal_sent = asyncio.Event()

        async def send(self, raw):
            frame = json.loads(raw)
            self.sent.append(frame)
            if frame[0] == "EVENT" and frame[1].get("kind") == 24200:
                self.terminal_sent.set()

        def __aiter__(self):
            async def frames():
                await self.terminal_sent.wait()
                if not self.disconnect_after_terminal:
                    second_delivery.set()
                    await asyncio.Future()
                if False:
                    yield ""

            return frames()

    first_socket = RelaySocket(disconnect_after_terminal=True)
    second_socket = RelaySocket(disconnect_after_terminal=False)
    sockets = iter((first_socket, second_socket))

    class RelayConnection:
        def __init__(self, websocket):
            self.websocket = websocket

        async def __aenter__(self):
            return self.websocket

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    def connect(*args, **kwargs):
        return RelayConnection(next(sockets))

    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=connect))
    terminal_payload = {
        "kind": "turn_completed",
        "seq": 1,
        "timestamp": "2026-08-03T14:00:00.000Z",
        "channelId": CHANNEL,
        "sessionId": "session-1",
        "turnId": "turn-1",
        "payload": {},
    }
    assert adapter._cache_terminal_activity(terminal_payload)

    websocket_task = asyncio.create_task(adapter._websocket_loop())
    try:
        await asyncio.wait_for(second_delivery.wait(), timeout=3)
        first_terminal = [
            frame for frame in first_socket.sent if frame[0] == "EVENT"
        ]
        second_terminal = [
            frame for frame in second_socket.sent if frame[0] == "EVENT"
        ]
        assert len(first_terminal) == 1
        assert len(second_terminal) == 1
        assert adapter._activity_ws_generation >= 3
    finally:
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            pass


def test_terminal_replay_is_bounded_and_keeps_latest_turns(monkeypatch):
    owner_pubkey = nostr_auth.public_key_hex("00" * 31 + "02")
    adapter = _make_adapter({"activity_owner_pubkey": owner_pubkey})
    monkeypatch.setattr(_buzz_mod, "_ACTIVITY_TERMINAL_REPLAY_CAP", 2)

    for turn_id in ("turn-1", "turn-2", "turn-3"):
        assert adapter._enqueue_activity(
            "turn_error",
            channel_id=CHANNEL,
            session_id="session-1",
            turn_id=turn_id,
            payload={"status": "failed"},
        )

    assert list(adapter._activity_terminal_replay) == ["turn-2", "turn-3"]


@pytest.mark.asyncio
async def test_unacked_terminal_is_recovered_when_socket_disconnects(monkeypatch):
    owner_pubkey = nostr_auth.public_key_hex("00" * 31 + "02")
    adapter = _make_adapter({"activity_owner_pubkey": owner_pubkey})
    old_websocket = _FakeWebSocket()
    adapter._ws_active = True
    adapter._ws_connection = old_websocket

    counter = 0

    def build_event(**kwargs):
        nonlocal counter
        counter += 1
        return {
            "id": f"event-{counter}",
            "kind": 24200,
            "payload": kwargs["payload"],
        }

    monkeypatch.setattr(
        _buzz_mod,
        "_load_nostr_auth",
        lambda: SimpleNamespace(build_observer_event=build_event),
    )

    assert adapter._enqueue_activity(
        "turn_completed",
        channel_id=CHANNEL,
        session_id="session-1",
        turn_id="turn-1",
    )
    await asyncio.wait_for(adapter._activity_queue.join(), timeout=1)
    assert "event-1" in adapter._activity_pending_event_ids

    adapter._ws_active = False
    adapter._ws_connection = None
    await adapter._reset_activity_transport()
    assert not adapter._activity_pending_event_ids
    assert list(adapter._activity_terminal_replay) == ["turn-1"]

    new_websocket = _FakeWebSocket()
    adapter._activity_ws_generation += 1
    adapter._ws_connection = new_websocket
    adapter._ws_active = True
    adapter._replay_terminal_activity()
    await asyncio.wait_for(adapter._activity_queue.join(), timeout=1)

    assert [frame[1]["payload"]["kind"] for frame in new_websocket.sent] == [
        "turn_completed"
    ]
    adapter._activity_sender_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await adapter._activity_sender_task


@pytest.mark.asyncio
async def test_activity_sender_drops_frame_when_websocket_generation_changes(monkeypatch):
    owner_pubkey = nostr_auth.public_key_hex("00" * 31 + "02")
    adapter = _make_adapter({"activity_owner_pubkey": owner_pubkey})
    old_websocket = _FakeWebSocket()
    new_websocket = _FakeWebSocket()
    adapter._ws_active = True
    adapter._ws_connection = old_websocket
    adapter._activity_ws_generation = 1

    def build_during_reconnect(**kwargs):
        adapter._activity_ws_generation = 2
        adapter._ws_connection = new_websocket
        return {"id": "event-id"}

    monkeypatch.setattr(
        _buzz_mod,
        "_load_nostr_auth",
        lambda: SimpleNamespace(build_observer_event=build_during_reconnect),
    )
    adapter._activity_queue.put_nowait((1, {"kind": "turn_liveness"}))
    adapter._activity_sender_task = asyncio.create_task(adapter._activity_sender_loop())

    await asyncio.wait_for(adapter._activity_queue.join(), timeout=1)
    adapter._activity_sender_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await adapter._activity_sender_task

    assert old_websocket.sent == []
    assert new_websocket.sent == []
    assert not adapter._activity_pending_event_ids


@pytest.mark.asyncio
async def test_activity_ack_expires_on_deadline_and_late_ack_is_ignored(monkeypatch):
    owner_pubkey = nostr_auth.public_key_hex("00" * 31 + "02")
    adapter = _make_adapter({"activity_owner_pubkey": owner_pubkey})
    websocket = _FakeWebSocket()
    adapter._ws_active = True
    adapter._ws_connection = websocket
    monkeypatch.setattr(_buzz_mod, "_ACTIVITY_ACK_TIMEOUT", 0.01)

    assert await adapter.publish_activity(
        "turn_started",
        channel_id=CHANNEL,
        session_id="session-1",
        turn_id="turn-1",
        payload={},
    )
    await asyncio.wait_for(adapter._activity_queue.join(), timeout=1)
    event_id = websocket.sent[0][1]["id"]
    assert event_id in adapter._activity_pending_event_ids

    await asyncio.sleep(0.03)
    assert event_id not in adapter._activity_pending_event_ids
    assert adapter._handle_activity_ack(["OK", event_id, True, "late"]) is False

    adapter._activity_sender_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await adapter._activity_sender_task


@pytest.mark.asyncio
async def test_unacked_terminal_is_retained_on_ack_timeout(monkeypatch):
    owner_pubkey = nostr_auth.public_key_hex("00" * 31 + "02")
    adapter = _make_adapter({"activity_owner_pubkey": owner_pubkey})
    websocket = _FakeWebSocket()
    adapter._ws_active = True
    adapter._ws_connection = websocket
    monkeypatch.setattr(_buzz_mod, "_ACTIVITY_ACK_TIMEOUT", 0.01)
    captured_payloads = []

    def build_event(**kwargs):
        captured_payloads.append(kwargs["payload"])
        return {"id": f"event-{len(captured_payloads)}"}

    monkeypatch.setattr(
        _buzz_mod,
        "_load_nostr_auth",
        lambda: SimpleNamespace(build_observer_event=build_event),
    )

    assert await adapter.publish_activity(
        "turn_completed",
        channel_id=CHANNEL,
        session_id="session-1",
        turn_id="turn-1",
    )
    await asyncio.wait_for(adapter._activity_queue.join(), timeout=1)
    event_id = websocket.sent[0][1]["id"]
    await asyncio.sleep(0.03)

    assert event_id not in adapter._activity_pending_event_ids
    assert list(adapter._activity_terminal_replay) == ["turn-1"]
    assert len(websocket.sent) == 2
    assert all("_hermesAckRetry" not in payload for payload in captured_payloads)
    adapter._activity_sender_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await adapter._activity_sender_task


@pytest.mark.asyncio
async def test_activity_pending_ack_cap_cancels_evicted_deadline(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(_buzz_mod, "_ACTIVITY_PENDING_CAP", 2)

    terminal_payload = {
        "kind": "turn_completed",
        "turnId": "turn-1",
        "sessionId": "session-1",
    }
    adapter._track_activity_ack("first", 1, terminal_payload)
    first_timer = adapter._activity_pending_event_ids["first"][1]
    adapter._track_activity_ack("second", 1)
    adapter._track_activity_ack("third", 1)

    assert list(adapter._activity_pending_event_ids) == ["second", "third"]
    assert first_timer.cancelled()
    assert list(adapter._activity_terminal_replay) == ["turn-1"]
    await adapter._reset_activity_transport()


def test_activity_owner_pubkey_rejects_non_curve_x_coordinate():
    with pytest.raises(ValueError, match="activity_owner_pubkey"):
        _make_adapter({"activity_owner_pubkey": "f" * 64})


def test_activity_owner_pubkey_is_config_only(monkeypatch):
    env_owner = nostr_auth.public_key_hex("00" * 31 + "02")
    monkeypatch.setenv("BUZZ_ACTIVITY_OWNER_PUBKEY", env_owner)

    adapter = _make_adapter()

    assert adapter.activity_owner_pubkey == ""
