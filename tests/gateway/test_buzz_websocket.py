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
        # Bounded poll, not a fixed settle: under full-suite load the inline
        # presence publish (executor + ~50ms schnorr) precedes the first frame
        # read, so the CLOSED frame can land after an arbitrary 0.1s.
        deadline = time.monotonic() + 5.0
        while CHANNEL not in adapter._restricted_channels and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

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
        # Wait for the CLOSED frame to be consumed (idx advances past it),
        # then confirm it did NOT restrict the channel — a negative assert
        # needs a positive observation point first.
        deadline = time.monotonic() + 5.0
        while idx < len(messages) and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        # Give the handler a turn to (wrongly) restrict, if it were going to.
        await asyncio.sleep(0.05)

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
        # Bounded poll (see drops_restricted test): the CLOSED frame is only
        # observed after the inline presence publish completes.
        deadline = time.monotonic() + 5.0
        while CHANNEL not in adapter._restricted_channels and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

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


# ── Presence heartbeat on the persistent socket ───────────────────────────


def test_build_presence_event_shape():
    event = nostr_auth.build_presence_event(
        private_key=TEST_PRIVATE_KEY,
        status="online",
        created_at=1_700_000_000,
        auxiliary_randomness=bytes(32),
    )
    assert event["kind"] == 20001
    assert event["content"] == "online"
    assert event["tags"] == [["status", "online"]]
    assert event["pubkey"] == nostr_auth.public_key_hex(TEST_PRIVATE_KEY)
    assert len(bytes.fromhex(event["sig"])) == 64
    # No p-tags — the Desktop's live path trusts author = subject.
    assert not any(tag[0] == "p" for tag in event["tags"])


def test_build_presence_event_rejects_bad_status():
    with pytest.raises(ValueError):
        nostr_auth.build_presence_event(private_key=TEST_PRIVATE_KEY, status="brb")


async def fail_cli(args, *, input_text=None):
    raise AssertionError(f"presence must not shell out to buzz, got {args}")


class _PresenceWebSocket(_ScriptedWebSocket):
    """Authenticating WS whose outbound presence events are observable."""

    def __init__(self):
        self.stream = asyncio.Queue()

        async def behavior():
            frame = await self.stream.get()
            if frame is None:
                raise StopAsyncIteration from None
            return frame

        super().__init__(behavior)

    @property
    def presence_events(self):
        return [
            message[1]
            for message in self.sent
            if isinstance(message, list)
            and message
            and message[0] == "EVENT"
            and isinstance(message[1], dict)
            and message[1].get("kind") == 20001
        ]


def _heartbeat_tasks():
    return [
        task
        for task in asyncio.all_tasks()
        if task.get_coro().__qualname__ == "BuzzAdapter._presence_heartbeat"
    ]


async def _force_close(*sockets):
    """Push a CLOSED frame so the adapter's reconnect loop parks in its
    backoff sleep — a cancellation point that settles cleanly on every
    code path (a cancel delivered inside the read-loop's cleanup during
    an immediate reconnect can be swallowed and resurrect the loop)."""
    for websocket in sockets:
        websocket.stream.put_nowait(
            json.dumps(["CLOSED", "hermes-buzz-membership", "error: connection reset"])
        )
    await asyncio.sleep(0.05)


async def _settle(*tasks):
    """Cancel tasks and await them so nothing leaks past a test."""
    for task in tasks:
        task.cancel()
    await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True), timeout=5
    )


@pytest.mark.asyncio
async def test_presence_heartbeat_runs_on_same_socket_without_cli(monkeypatch):
    """Presence rides the authenticated inbound connection: no second
    WebSocket, no ``buzz users set-presence`` subprocess."""
    adapter = _make_adapter()
    monkeypatch.setattr(_buzz_mod, "_PRESENCE_INTERVAL", 0.02)
    adapter._presence_interval = 0.02

    adapter._run_cli = fail_cli

    sockets = []

    def fake_connect(*args, **kwargs):
        ws = _PresenceWebSocket()
        sockets.append(ws)
        return ws

    import websockets as _ws_mod

    monkeypatch.setattr(_ws_mod, "connect", fake_connect)

    task = asyncio.create_task(adapter._websocket_loop())
    try:
        deadline = time.monotonic() + 5.0
        while not sockets and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        while sockets and len(sockets[0].presence_events) < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert len(sockets) == 1
    finally:
        await _settle(task)

    assert len(sockets) == 1, "presence spawned a second WebSocket"
    events = sockets[0].presence_events
    assert len(events) >= 2, "presence did not repeat on the persistent socket"
    online_events = [event for event in events if event["content"] == "online"]
    assert len(online_events) >= 2
    expected_pubkey = nostr_auth.public_key_hex(TEST_PRIVATE_KEY)
    assert all(event["pubkey"] == expected_pubkey for event in online_events)
    assert all(["status", "online"] in event["tags"] for event in online_events)


@pytest.mark.asyncio
async def test_presence_heartbeat_reconnect_replaces_task_exactly_once(monkeypatch):
    """A reconnect retires the old heartbeat with the old socket and keeps
    exactly one live heartbeat for the new connection."""
    adapter = _make_adapter()
    monkeypatch.setattr(_buzz_mod, "_PRESENCE_INTERVAL", 0.02)
    adapter._presence_interval = 0.02
    adapter._run_cli = fail_cli

    sockets = []

    def fake_connect(*args, **kwargs):
        ws = _PresenceWebSocket()
        sockets.append(ws)
        return ws

    import websockets as _ws_mod

    monkeypatch.setattr(_ws_mod, "connect", fake_connect)

    task = asyncio.create_task(adapter._websocket_loop())
    try:
        deadline = time.monotonic() + 5.0
        while not sockets and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        while (
            sockets
            and len(sockets[0].presence_events) < 1
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.01)
        assert sockets and sockets[0].presence_events, "first connection never heartbeated"
        assert len(_heartbeat_tasks()) == 1, "duplicate heartbeats on one connection"

        # Force a reconnect: CLOSED makes the read loop raise and back off.
        await _force_close(sockets[0])
        while len(sockets) < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert len(sockets) == 2, "relay CLOSED did not trigger a reconnect"
        assert sockets[0].exited, "old socket was not closed before reconnecting"
        # The retired heartbeat must not publish after its socket exited.
        retired_count = len(sockets[0].presence_events)
        await asyncio.sleep(0.1)
        assert len(sockets[0].presence_events) == retired_count, (
            "retired heartbeat kept publishing after its socket closed"
        )
        while (
            len(sockets[1].presence_events) < 1
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.01)
        assert len(_heartbeat_tasks()) == 1, "reconnect duplicated the heartbeat"
    finally:
        await _settle(task)

    assert sockets[1].presence_events, "reconnected connection never heartbeated"
    assert not _heartbeat_tasks(), "heartbeat task outlived its reconnect loop"


@pytest.mark.asyncio
async def test_cancel_during_first_presence_does_not_leak_heartbeat(monkeypatch):
    """Cancelling the loop while the first online publish is still in flight
    must not strand the heartbeat task (or any task) past shutdown."""
    adapter = _make_adapter()
    adapter._presence_interval = 0.02
    adapter._run_cli = fail_cli

    release = asyncio.Event()
    original_publish = adapter._publish_presence

    async def gated_publish(websocket, status):
        if status == "online" and not release.is_set():
            await release.wait()
        return await original_publish(websocket, status)

    monkeypatch.setattr(adapter, "_publish_presence", gated_publish)

    sockets = []

    def fake_connect(*args, **kwargs):
        ws = _PresenceWebSocket()
        sockets.append(ws)
        return ws

    import websockets as _ws_mod

    monkeypatch.setattr(_ws_mod, "connect", fake_connect)

    task = asyncio.create_task(adapter._websocket_loop())
    try:
        deadline = time.monotonic() + 5.0
        while not sockets and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert sockets, "connection never opened"
        # The loop is now parked waiting for the first online publish.
        # Cancel, then release the gate: the cancellation consumed at the
        # executor boundary must be honored before child tasks spawn, so
        # the loop settles promptly with no orphan heartbeat.
        task.cancel()
        release.set()
        # (asyncio.wait, not wait_for: on Python 3.11 wait_for converts a
        # completed-with-CancelledError gather into TimeoutError.)
        await asyncio.wait([task], timeout=2)
        assert task.done(), "websocket loop did not settle after cancel"
    finally:
        release.set()

    await asyncio.sleep(0.05)
    heartbeat = getattr(adapter, "_presence_task", None)
    assert heartbeat is None or heartbeat.done(), "heartbeat task leaked past shutdown"
    strays = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]
    assert not strays, f"leaked tasks: {[t.get_coro() for t in strays]}"


@pytest.mark.asyncio
async def test_presence_heartbeat_coexists_with_inbound_dispatch(monkeypatch):
    """Heartbeats keep flowing while inbound frames dispatch to handlers."""
    adapter = _make_adapter()
    monkeypatch.setattr(_buzz_mod, "_PRESENCE_INTERVAL", 0.02)
    adapter._presence_interval = 0.02
    adapter._run_cli = fail_cli
    adapter._channel_state = {CHANNEL: {"chat_type": "group", "last_ts": 1, "seen": {}}}

    handled = []

    async def record_handle_event(channel_id, state, event):
        handled.append(event["id"])

    monkeypatch.setattr(adapter, "_handle_event", record_handle_event)

    ws_refs = []

    def fake_connect(*args, **kwargs):
        ws = _PresenceWebSocket()
        ws_refs.append(ws)
        return ws

    import websockets as _ws_mod

    monkeypatch.setattr(_ws_mod, "connect", fake_connect)

    # Feed inbound frames continuously while heartbeats tick.
    async def feed():
        for index in range(20):
            await asyncio.sleep(0.01)
            for ws in list(ws_refs):
                ws.stream.put_nowait(
                    json.dumps(
                        ["EVENT", "hermes-buzz-0", {"id": f"in-{index}", "kind": 9, "created_at": 2, "content": "hi"}]
                    )
                )

    feeder = asyncio.create_task(feed())
    task = asyncio.create_task(adapter._websocket_loop())
    try:
        deadline = time.monotonic() + 5.0
        while not ws_refs and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        while (
            ws_refs
            and len(ws_refs[0].presence_events) < 3
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.01)
    finally:
        feeder.cancel()
        await _force_close(*ws_refs)
        await _settle(task, feeder)

    assert len(ws_refs) == 1
    assert len(ws_refs[0].presence_events) >= 3, "heartbeats stopped under inbound load"
    assert handled, "inbound frames were not dispatched alongside heartbeats"


@pytest.mark.asyncio
async def test_presence_heartbeat_survives_failed_send_and_retries(monkeypatch):
    """A failed sign/send is logged and retried on the next cadence; it must
    not kill the heartbeat."""
    adapter = _make_adapter()
    adapter._presence_interval = 0.02
    adapter._run_cli = fail_cli

    attempts = []
    sent = []
    ready = asyncio.Event()

    class FlakySend:
        async def send(self, raw):
            message = json.loads(raw)
            if message[0] == "EVENT" and message[1].get("kind") == 20001:
                attempts.append(message)
                if len(attempts) == 1:
                    raise ConnectionError("send failed")
                sent.append(message)
                ready.set()

    websocket = FlakySend()
    task = asyncio.create_task(adapter._presence_heartbeat(websocket))
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
    finally:
        await _settle(task)

    assert len(attempts) >= 2, "heartbeat died after the first failed send"
    assert len(sent) >= 1, "heartbeat never retried after the failed send"


@pytest.mark.asyncio
async def test_graceful_shutdown_publishes_offline_before_socket_close(monkeypatch):
    """Adapter shutdown must push a final ``offline`` on the live socket
    before the connection exits."""
    adapter = _make_adapter()
    adapter._presence_interval = 0.02
    adapter._run_cli = fail_cli

    sent_before_exit = {}

    class OfflineWebSocket(_PresenceWebSocket):
        async def __aexit__(self, *exc_info):
            sent_before_exit["offline"] = any(
                isinstance(message, list)
                and message
                and message[0] == "EVENT"
                and isinstance(message[1], dict)
                and message[1].get("content") == "offline"
                for message in self.sent
            )
            return await _ScriptedWebSocket.__aexit__(self, *exc_info)

    sockets = []

    def fake_connect(*args, **kwargs):
        ws = OfflineWebSocket()
        sockets.append(ws)
        return ws

    import websockets as _ws_mod

    monkeypatch.setattr(_ws_mod, "connect", fake_connect)

    task = asyncio.create_task(adapter._websocket_loop())
    try:
        deadline = time.monotonic() + 5.0
        while not sockets and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        while (
            sockets
            and len(sockets[0].presence_events) < 1
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.01)
        assert sockets and sockets[0].presence_events, "online heartbeat never landed"
        # Graceful shutdown through the adapter's own disconnect() — the same
        # path the gateway runs — so offline lands while the socket is open.
        await asyncio.wait_for(adapter.disconnect(), timeout=5)
    finally:
        await _settle(task)

    assert sent_before_exit.get("offline") is True, (
        "offline was not published before the socket exited"
    )
    heartbeat = getattr(adapter, "_presence_task", None)
    assert heartbeat is None or heartbeat.done(), "heartbeat leaked past disconnect()"
