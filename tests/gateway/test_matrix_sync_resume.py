"""Matrix sync positions must resume without replaying accepted user turns."""

import asyncio
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import secret_scope
from gateway.config import PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.matrix import adapter as matrix


NOW = 1_700_000_000.0


class SyncFailure(RuntimeError):
    errcode = "M_UNKNOWN_POS"


class SynapseSyncFailure(RuntimeError):
    errcode = "M_UNKNOWN"
    http_status = 400


@pytest.fixture(params=["fake", "mautrix"])
def transport(monkeypatch, request):
    requests = []
    responses = []

    class Client:
        def __init__(self, *, mxid, device_id, api, state_store, sync_store):
            self.mxid, self.device_id, self.api = mxid, device_id, api
            self.state_store, self.sync_store = state_store, sync_store
            self.crypto = None
            self.event_handlers, self.global_event_handlers = {}, {}
            self.event_middlewares = {}

        async def whoami(self):
            return SimpleNamespace(user_id=self.mxid, device_id=self.device_id)

        async def sync(self, **kwargs):
            requests.append(kwargs)
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        def add_event_handler(self, event_type, handler, wait_sync=False):
            self.event_handlers.setdefault(event_type, {})[handler] = wait_sync

        def add_dispatcher(self, _dispatcher):
            pass

        async def _catch_errors(self, handler, event):
            try:
                await handler(event)
            except Exception:
                pass

        def _dispatch_manual_event(
            self, event_type, event, _global, force_synchronous, source
        ):
            return [
                asyncio.create_task(self._catch_errors(handler, event))
                for handler in self.event_handlers.get(event_type, {})
            ]

        def handle_sync(self, response):
            tasks = []
            for room_id, room in response.get("rooms", {}).get("join", {}).items():
                for event in room.get("timeline", {}).get("events", []):
                    event = SimpleNamespace(
                        **event, room_id=room_id, timestamp=event["origin_server_ts"]
                    )
                    event_type = next(
                        key for key in self.event_handlers if str(key) == event.type
                    )
                    tasks.extend(
                        self._dispatch_manual_event(
                            event_type, event, True, False, None
                        )
                    )
            return tasks

    class HTTPAPI:
        def __init__(self, *, token, **kwargs):
            self.token = token
            self.session = SimpleNamespace(close=AsyncMock())

    class MemorySyncStore:
        def __init__(self):
            self.token = None

        async def get_next_batch(self):
            return self.token

        async def put_next_batch(self, token):
            self.token = token

    class InternalEventType(Enum):
        INVITE = "invite"
        DEVICE_OTK_COUNT = "device_otk_count"
        DEVICE_LISTS = "device_lists"

    modules = {
        "mautrix.api": {"HTTPAPI": HTTPAPI},
        "mautrix.client": {"Client": Client, "InternalEventType": InternalEventType},
        "mautrix.client.state_store": {
            "MemoryStateStore": SimpleNamespace,
            "MemorySyncStore": MemorySyncStore,
        },
        "mautrix.client.dispatcher": {"MembershipEventDispatcher": object},
    }
    if request.param == "mautrix":
        real_client = pytest.importorskip("mautrix.client").Client
        monkeypatch.setattr(real_client, "whoami", Client.whoami)
        monkeypatch.setattr(real_client, "sync", Client.sync)
    else:
        for name, values in modules.items():
            module = ModuleType(name)
            module.__dict__.update(values)
            monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(matrix, "_create_matrix_session", lambda _proxy: None)
    monkeypatch.setattr(matrix.time, "time", lambda: NOW)
    monkeypatch.setattr(matrix.MatrixAdapter, "_sync_loop", AsyncMock())
    monkeypatch.setattr(matrix.MatrixAdapter, "_refresh_dm_cache", AsyncMock())
    return requests, responses


def make_adapter(**extra):
    adapter = matrix.MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token=extra.pop("token", "test-token"),
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
                "device_id": "DEVICE",
                "encryption": False,
                **extra,
            },
        )
    )
    adapter._handle_text_message = AsyncMock(return_value=True)
    adapter._dispatch_reaction = AsyncMock()
    return adapter


def message(event_id, timestamp=int((NOW - 1) * 1000)):
    return {
        "type": str(matrix.EventType.ROOM_MESSAGE),
        "event_id": event_id,
        "sender": "@alice:example.org",
        "origin_server_ts": timestamp,
        "content": {"msgtype": "m.text", "body": event_id},
    }


def batch(token, *events):
    return {
        "next_batch": token,
        "rooms": {
            "join": {"!room:example.org": {"timeline": {"events": list(events)}}}
        },
    }


def encrypted_message(event_id):
    return {
        **message(event_id),
        "type": "m.room.encrypted",
        "content": {
            "algorithm": "m.megolm.v1.aes-sha2",
            "ciphertext": "ciphertext",
            "sender_key": "key",
            "device_id": "ALICE",
            "session_id": "session",
        },
    }


def reaction(event_id):
    return {
        **message(event_id),
        "type": "m.reaction",
        "content": {
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": "$final",
                "key": "👍",
            }
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
@pytest.mark.parametrize("unreadable", ["missing-key", "malformed"])
async def test_unreadable_ciphertext_allows_connection_and_native_traffic(
    tmp_path,
    monkeypatch,
    transport,
    unreadable,
):
    from mautrix.errors import DecryptionError, SessionNotFound

    _requests, responses = transport
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    error = (
        SessionNotFound("session", "key")
        if unreadable == "missing-key"
        else DecryptionError("Failed to decrypt megolm event")
    )
    decrypt = AsyncMock(side_effect=error)

    async def setup(_adapter, client, _api, _state_store):
        client._crypto = SimpleNamespace(
            decrypt_megolm_event=decrypt, share_keys=AsyncMock()
        )
        return True

    monkeypatch.setattr(matrix.MatrixAdapter, "_connect_setup_e2ee", setup)
    adapter = make_adapter(encryption=True)
    responses.append(
        batch(
            "s1",
            encrypted_message("$unreadable"),
            message("$plain"),
            reaction("$reaction"),
        )
    )
    try:
        assert await adapter.connect()
        assert await adapter._client.sync_store.get_next_batch() == "s1"
        await adapter._absorb_sync(
            adapter._client, batch("s2", message("$next"), reaction("$next-reaction"))
        )
        assert (
            [call.args[2] for call in adapter._handle_text_message.await_args_list],
            [call.args for call in adapter._dispatch_reaction.await_args_list],
            await adapter._client.sync_store.get_next_batch(),
        ) == (
            ["$plain", "$next"],
            [
                ("!room:example.org", "$final", "👍", "@alice:example.org", event_id)
                for event_id in ("$reaction", "$next-reaction")
            ],
            "s2",
        )
        decrypt.assert_awaited_once()
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
@pytest.mark.parametrize("kind", ["plain", "encrypted", "reaction"])
@pytest.mark.parametrize("failed_owner", ["hermes", "plugin"])
async def test_retry_clears_only_the_failed_intake_owners_deduplication(
    tmp_path,
    monkeypatch,
    transport,
    kind,
    failed_owner,
):
    from mautrix.types import Event

    _requests, responses = transport
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    async def setup(_adapter, client, _api, _state_store):
        client._crypto = SimpleNamespace(
            share_keys=AsyncMock(),
            decrypt_megolm_event=AsyncMock(
                side_effect=lambda event: Event.deserialize({
                    **message(str(event.event_id)),
                    "room_id": str(event.room_id),
                }),
            ),
        )
        return True

    monkeypatch.setattr(matrix.MatrixAdapter, "_connect_setup_e2ee", setup)
    adapter = make_adapter(encryption=kind == "encrypted")
    responses.append(batch("s1"))
    assert await adapter.connect()
    accepted = asyncio.Event()
    intake_attempts, accepted_ids, plugin_attempts = [], [], []

    async def intake(*args):
        event_id = args[4] if kind == "reaction" else args[2]
        intake_attempts.append(event_id)
        if failed_owner == "hermes" and len(intake_attempts) == 1:
            raise RuntimeError("Hermes intake failed")
        accepted_ids.append(event_id)
        accepted.set()
        return True

    async def plugin(event):
        plugin_attempts.append(str(event.event_id))
        if failed_owner == "plugin" and len(plugin_attempts) == 1:
            await accepted.wait()
            raise RuntimeError("plugin handler failed")

    handler_type = (
        matrix.EventType.REACTION
        if kind == "reaction"
        else matrix.EventType.ROOM_MESSAGE
    )
    adapter._client.add_event_handler(handler_type, plugin, wait_sync=False)
    if kind == "reaction":
        adapter._dispatch_reaction = intake
    else:
        adapter._handle_text_message = intake
    event = {"plain": message, "encrypted": encrypted_message, "reaction": reaction}[
        kind
    ]("$event")
    response = batch("s2", event)
    try:
        with pytest.raises(RuntimeError, match="intake failed|plugin handler failed"):
            await adapter._absorb_sync(adapter._client, response)
        assert await adapter._client.sync_store.get_next_batch() == "s1"
        await adapter._absorb_sync(adapter._client, response)
        assert (
            intake_attempts,
            accepted_ids,
            plugin_attempts,
            await adapter._client.sync_store.get_next_batch(),
        ) == (
            ["$event"] * (2 if failed_owner == "hermes" else 1),
            ["$event"],
            ["$event", "$event"],
            "s2",
        )
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_fresh_reconnect_and_restart_resume_per_profile(
    tmp_path, monkeypatch, transport
):
    requests, responses = transport
    homes = [tmp_path / "a", tmp_path / "b"]
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    secret_scope.set_multiplex_active(True)
    captured = []
    try:
        for home, expected_since, next_batch in [
            (homes[0], None, "a1"),
            (homes[1], None, "b1"),
            (homes[0], "a1", "a2"),
        ]:
            home_token = set_hermes_home_override(str(home))
            scope_token = secret_scope.set_secret_scope({}, profile_home=str(home))
            try:
                adapter = make_adapter()
                # A fresh full sync replays recent history; a resumed timeline contains only offline events.
                recent = message("$recent")
                offline = message("$offline", timestamp=int((NOW - 100) * 1000))
                reaction = {
                    "type": str(matrix.EventType.REACTION),
                    "event_id": "$reaction",
                    "sender": "@alice:example.org",
                    "origin_server_ts": int((NOW - 100) * 1000),
                    "content": {
                        "m.relates_to": {
                            "event_id": "$final",
                            "key": "✅",
                            "rel_type": "m.annotation",
                        }
                    },
                }
                responses.append(
                    batch(
                        next_batch,
                        *([recent] if expected_since is None else [offline, reaction]),
                    )
                )
                assert await adapter.connect()
                captured.append((
                    requests[-1].get("since"),
                    requests[-1].get("full_state"),
                    [
                        call.args[2]
                        for call in adapter._handle_text_message.await_args_list
                    ],
                    adapter._dispatch_reaction.await_args_list,
                ))
                if home == homes[0] and expected_since is None:
                    responses.append(batch("a1"))
                    assert await adapter.connect(is_reconnect=True)
                    assert requests[-1].get("since") == "a1"
                await adapter.disconnect()
            finally:
                secret_scope.reset_secret_scope(scope_token)
                reset_hermes_home_override(home_token)
    finally:
        secret_scope.set_multiplex_active(False)
    assert [
        (since, full_state, messages) for since, full_state, messages, _ in captured
    ] == [
        (None, True, ["$recent"]),
        (None, True, ["$recent"]),
        ("a1", True, ["$offline"]),
    ]
    assert [call.args for call in captured[-1][3]] == [
        ("!room:example.org", "$final", "✅", "@alice:example.org", "$reaction"),
    ]


@pytest.mark.asyncio
async def test_failed_dispatch_retries_without_acknowledging_or_repeating_siblings(
    tmp_path, monkeypatch, transport
):
    requests, responses = transport
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = make_adapter()
    responses.append(batch("s1"))
    assert await adapter.connect()
    entered, release = asyncio.Event(), asyncio.Event()
    accepted = []

    async def handle(_room, _sender, event_id, *_args):
        if event_id == "$failure" and not release.is_set():
            entered.set()
            await release.wait()
            raise RuntimeError("dispatch failed")
        accepted.append(event_id)

    adapter._handle_text_message = handle
    response = batch("s2", message("$sibling"), message("$failure"))
    dispatch = asyncio.create_task(adapter._absorb_sync(adapter._client, response))
    waiter = asyncio.create_task(entered.wait())
    await asyncio.wait([waiter, dispatch], return_when=asyncio.FIRST_COMPLETED)
    assert entered.is_set()
    assert await adapter._client.sync_store.get_next_batch() == "s1"
    release.set()
    with pytest.raises(Exception, match="dispatch failed"):
        await dispatch
    assert await adapter._client.sync_store.get_next_batch() == "s1"
    await adapter._absorb_sync(adapter._client, response)
    assert accepted == ["$sibling", "$failure"]
    assert await adapter._client.sync_store.get_next_batch() == "s2"
    await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        SyncFailure("expired since token"),
        SynapseSyncFailure("Invalid stream token"),
    ],
)
async def test_rejected_cursor_refreshes_state_with_startup_grace(
    tmp_path, monkeypatch, transport, error
):
    requests, responses = transport
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = make_adapter()
    responses.append(batch("s1", message("$recent")))
    assert await adapter.connect()
    responses.extend([
        error,
        batch("s2", message("$recent"), message("$offline", int((NOW - 100) * 1000))),
    ])
    assert await adapter.connect(is_reconnect=True)
    assert [
        (request.get("since"), request.get("full_state")) for request in requests
    ] == [
        (None, True),
        ("s1", True),
        (None, True),
    ]
    assert [call.args[2] for call in adapter._handle_text_message.await_args_list] == [
        "$recent",
    ]
    await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changed", "resume"),
    [
        ({"token": "replacement-token"}, True),
        ({"device_id": "OTHER"}, False),
        ({"user_id": "@other:example.org"}, False),
        ({"homeserver": "https://other.example.org"}, False),
    ],
)
async def test_cursor_cannot_cross_authenticated_identity(
    tmp_path, monkeypatch, transport, changed, resume
):
    requests, responses = transport
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for extra, expected_since in [
        ({}, None),
        (changed, "s1" if resume else None),
        ({}, "s1"),
    ]:
        adapter = make_adapter(**extra)
        responses.append(batch("s1"))
        assert await adapter.connect()
        assert requests[-1].get("since") == expected_since
        await adapter.disconnect()
    cursors = list(tmp_path.rglob("sync-*.json"))
    assert [path.read_text(encoding="utf-8") for path in cursors] == [
        '{"next_batch": "s1"}'
    ] * (1 if resume else 2)
    assert all(path.stat().st_mode & 0o077 == 0 for path in cursors)


@pytest.mark.asyncio
@pytest.mark.parametrize("intake_error", ["runtime", "decryption"])
async def test_queued_keys_room_state_and_decrypted_dispatch_precede_ack(
    tmp_path, monkeypatch, intake_error,
):
    pytest.importorskip("mautrix.client")
    types = pytest.importorskip("mautrix.types")
    from mautrix.client.state_store import MemoryStateStore
    from mautrix.client.dispatcher import MembershipEventDispatcher
    from mautrix.util.logging import TraceLogger
    from mautrix.errors import DecryptionError
    from plugins.platforms.matrix.sync_transport import (
        DurableSyncStore,
        create_sync_client,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = make_adapter()
    store = DurableSyncStore(
        tmp_path,
        "https://matrix.example.org",
        "@bot:example.org",
        "DEVICE",
        "test-token",
    )
    await store.put_next_batch("s1")
    client = create_sync_client(
        mxid=types.UserID("@bot:example.org"),
        device_id=types.DeviceID("DEVICE"),
        api=SimpleNamespace(log=TraceLogger("matrix-sync-test")),
        state_store=MemoryStateStore(),
        sync_store=store,
    )
    adapter._client = client
    invitations = []
    client.add_dispatcher(MembershipEventDispatcher)

    async def invited(event):
        invitations.append(str(event.room_id))

    client.add_event_handler(
        pytest.importorskip("mautrix.client").InternalEventType.INVITE,
        invited,
        wait_sync=True,
    )
    key_entered, key_ready, message_entered, message_ready = (
        asyncio.Event() for _ in range(4)
    )
    state_entered, state_ready = asyncio.Event(), asyncio.Event()
    update_state = client.state_store.update_state

    async def refresh_state(event):
        state_entered.set()
        await state_ready.wait()
        await update_state(event)

    monkeypatch.setattr(client.state_store, "update_state", refresh_state)
    imported_keys = []
    received_messages = []
    error_type = RuntimeError if intake_error == "runtime" else DecryptionError

    async def import_key(_event):
        imported_keys.append("key")
        assert imported_keys == ["key"]
        key_entered.set()
        await key_ready.wait()

    async def decrypt(_event):
        assert key_ready.is_set()
        members = await client.state_store.get_members(
            types.RoomID("!room:example.org")
        )
        assert set(members) == {"@bot:example.org", "@alice:example.org"}
        return types.Event.deserialize({
            **message("$encrypted"),
            "room_id": "!room:example.org",
        })

    async def receive(_event):
        received_messages.append("message")
        message_entered.set()
        await message_ready.wait()
        if len(received_messages) == 1:
            raise error_type("decrypted dispatch failed")

    client.add_event_handler(types.EventType.ROOM_KEY, import_key)
    client._crypto = SimpleNamespace(decrypt_megolm_event=decrypt)
    client.add_event_handler(
        types.EventType.ROOM_ENCRYPTED, client.hermes_sync.decrypt_sync_event
    )
    client.add_event_handler(types.EventType.ROOM_MESSAGE, receive, wait_sync=True)
    response = batch(
        "s2",
        {
            "type": "m.room.encrypted",
            "event_id": "$encrypted",
            "origin_server_ts": int((NOW - 1) * 1000),
            "sender": "@alice:example.org",
            "content": {
                "algorithm": "m.megolm.v1.aes-sha2",
                "ciphertext": "ciphertext",
                "sender_key": "key",
                "device_id": "ALICE",
                "session_id": "session",
            },
        },
    )
    response["to_device"] = {
        "events": [
            {
                "type": "m.room_key",
                "sender": "@alice:example.org",
                "content": {
                    "algorithm": "m.megolm.v1.aes-sha2",
                    "room_id": "!room:example.org",
                    "session_id": "session",
                    "session_key": "key",
                },
            }
        ]
    }
    response["rooms"]["invite"] = {
        "!invite:example.org": {
            "invite_state": {
                "events": [
                    {
                        "type": "m.room.member",
                        "sender": "@alice:example.org",
                        "state_key": "@bot:example.org",
                        "content": {"membership": "invite"},
                    }
                ]
            }
        }
    }
    response["rooms"]["join"]["!room:example.org"]["state"] = {
        "events": [
            {
                "type": "m.room.member",
                "event_id": f"$member-{user}",
                "origin_server_ts": int((NOW - 100) * 1000),
                "sender": user,
                "state_key": user,
                "content": {"membership": "join"},
            }
            for user in ("@bot:example.org", "@alice:example.org")
        ]
    }
    dispatch = asyncio.create_task(adapter._absorb_sync(client, response))

    async def checkpoint(entered):
        waiter = asyncio.create_task(entered.wait())
        await asyncio.wait([waiter, dispatch], return_when=asyncio.FIRST_COMPLETED)
        if not entered.is_set():
            await dispatch
        assert await store.get_next_batch() == "s1"

    await checkpoint(key_entered)
    key_ready.set()
    await checkpoint(state_entered)
    assert not message_entered.is_set()
    state_ready.set()
    await checkpoint(message_entered)
    message_ready.set()
    with pytest.raises(error_type, match="decrypted dispatch failed"):
        await dispatch
    assert await store.get_next_batch() == "s1"
    await adapter._absorb_sync(client, response)
    restarted = DurableSyncStore(
        tmp_path,
        "https://matrix.example.org",
        "@bot:example.org",
        "DEVICE",
        "test-token",
    )
    await restarted.load()
    assert await restarted.get_next_batch() == "s2"
    assert invitations == ["!invite:example.org", "!invite:example.org"]
@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
@pytest.mark.parametrize("cancel", [False, True, "disconnect-initial"])
async def test_native_middleware_is_owned_until_completion_or_cancellation(
    tmp_path,
    monkeypatch,
    transport,
    cancel,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    adapter = make_adapter()
    responses.append(batch("s1"))
    assert await adapter.connect()
    entered, release, finished = (asyncio.Event() for _ in range(3))

    async def middleware(_event):
        entered.set()
        try:
            await release.wait()
            return True
        finally:
            finished.set()

    adapter._client.add_event_middleware(matrix.EventType.ROOM_MESSAGE, middleware)
    dispatch = asyncio.create_task(
        adapter._absorb_sync(adapter._client, batch("s2", message("$middleware")))
    )
    try:
        waiter = asyncio.create_task(entered.wait())
        await asyncio.wait([waiter, dispatch], return_when=asyncio.FIRST_COMPLETED)
        assert (dispatch.done(), await adapter._client.sync_store.get_next_batch()) == (
            False,
            "s1",
        )
        if cancel:
            client = adapter._client
            if cancel == "disconnect-initial":
                await adapter.disconnect()
                assert finished.is_set()
            else:
                dispatch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await dispatch
            assert (finished.is_set(), adapter._handle_text_message.await_count) == (
                True,
                0,
            )
            assert await client.sync_store.get_next_batch() == "s1"
        else:
            release.set()
            await dispatch
            assert (finished.is_set(), adapter._handle_text_message.await_count) == (
                True,
                1,
            )
            assert await adapter._client.sync_store.get_next_batch() == "s2"
    finally:
        release.set()
        dispatch.cancel()
        await asyncio.gather(dispatch, return_exceptions=True)
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
@pytest.mark.parametrize("interruption", ["cancel", "failure"])
async def test_completed_intake_survives_fresh_adapter_with_unfinished_sibling(
    tmp_path,
    monkeypatch,
    transport,
    interruption,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    adapter = make_adapter()
    responses.append(batch("s1"))
    assert await adapter.connect()
    accepted, blocked, release, stopped = (asyncio.Event() for _ in range(4))
    intakes = []

    async def intake(_room, _sender, event_id, *_args):
        intakes.append(event_id)
        accepted.set()
        return True

    async def sibling(_event):
        await accepted.wait()
        blocked.set()
        try:
            await release.wait()
            raise RuntimeError("incomplete sibling")
        finally:
            stopped.set()

    adapter._handle_text_message = intake
    adapter._client.add_event_handler(matrix.EventType.ROOM_MESSAGE, sibling)
    response = batch("s2", message("$accepted"))
    dispatch = asyncio.create_task(adapter._absorb_sync(adapter._client, response))
    adapter._sync_task = dispatch
    await asyncio.wait_for(blocked.wait(), timeout=2)
    if interruption == "failure":
        release.set()
        with pytest.raises(RuntimeError, match="incomplete sibling"):
            await dispatch
    await adapter.disconnect()
    assert stopped.is_set()
    fresh = make_adapter()
    fresh._handle_text_message = intake
    responses.append(response)
    try:
        assert await fresh.connect()
        assert (intakes, await fresh._client.sync_store.get_next_batch()) == (
            ["$accepted"],
            "s2",
        )
    finally:
        await fresh.disconnect()


@pytest.mark.asyncio
async def test_pending_intake_checkpoint_is_scoped_bounded_and_removed_on_ack(
    tmp_path,
    monkeypatch,
):
    import json
    from plugins.platforms.matrix import sync_transport

    monkeypatch.setattr(sync_transport, "MAX_PENDING_INTAKES", 2)
    scopes = [
        (tmp_path / "a", "https://matrix.test", "@bot:matrix.test", "BOT"),
        (tmp_path / "b", "https://matrix.test", "@bot:matrix.test", "BOT"),
        (tmp_path / "a", "https://other.test", "@bot:matrix.test", "BOT"),
        (tmp_path / "a", "https://matrix.test", "@other:matrix.test", "BOT"),
        (tmp_path / "a", "https://matrix.test", "@bot:matrix.test", "OTHER"),
    ]
    original = sync_transport.DurableSyncStore(*scopes[0], "secret")
    await original.put_next_batch("before")
    for event_id in ("$accepted", "$second"):
        assert original.reserve_intake(event_id)
        await original.accept_intake(event_id)
        original.release_intake(event_id)
    with pytest.raises(RuntimeError, match="checkpoint is full"):
        original.reserve_intake("$never-admitted")
    for event_id in ("$" + "x" * 255, "$" + "é" * 128):
        with pytest.raises(ValueError, match="event ID exceeds"):
            original.reserve_intake(event_id)
    observed = []
    for scope in [*scopes, scopes[0]]:
        fresh = sync_transport.DurableSyncStore(*scope, "replacement-secret")
        await fresh.load()
        observed.append((
            await fresh.get_next_batch(),
            fresh.reserve_intake("$accepted"),
        ))
        fresh.release_intake("$accepted")
    assert observed == [("before", False), *[(None, True)] * 4, ("before", False)]
    assert json.loads(original.path.read_text(encoding="utf-8")) == {
        "next_batch": "before",
        "accepted_events": ["$accepted", "$second"],
    }
    await original.put_next_batch("after")
    assert json.loads(original.path.read_text(encoding="utf-8")) == {
        "next_batch": "after"
    }
    assert original.reserve_intake("$never-admitted")


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
async def test_unadmitted_intake_is_retryable_after_incomplete_batch(
    tmp_path,
    monkeypatch,
    transport,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    adapter = make_adapter()
    responses.append(batch("s1"))
    assert await adapter.connect()
    attempts = []

    async def refused_then_accepted(*_args):
        attempts.append("intake")
        return len(attempts) > 1

    async def failed_sibling(_event):
        raise RuntimeError("sibling failed")

    adapter._handle_text_message = refused_then_accepted
    adapter._client.add_event_handler(matrix.EventType.ROOM_MESSAGE, failed_sibling)
    response = batch("s2", message("$unadmitted"))
    try:
        with pytest.raises(RuntimeError, match="sibling failed"):
            await adapter._absorb_sync(adapter._client, response)
        adapter._client.remove_event_handler(
            matrix.EventType.ROOM_MESSAGE, failed_sibling
        )
        await adapter._absorb_sync(adapter._client, response)
        assert (attempts, await adapter._client.sync_store.get_next_batch()) == (
            ["intake", "intake"],
            "s2",
        )
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
@pytest.mark.parametrize("admitted", [False, True])
async def test_actual_admission_receipt_covers_inputs_outside_native_callback(
    tmp_path,
    monkeypatch,
    transport,
    admitted,
):
    from gateway.platforms.base import MessageEvent
    from plugins.platforms.matrix.sync_transport import DurableSyncStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    adapter = make_adapter()
    responses.append(batch("s1"))
    assert await adapter.connect()
    adapter._message_handler = AsyncMock() if admitted else None
    monkeypatch.setattr(adapter, "_start_session_processing", lambda *_args: True)
    event = MessageEvent(
        text="admitted input",
        message_id="$input",
        source=adapter.build_source(
            chat_id="!room:example.org", user_id="@alice:example.org"
        ),
    )
    try:
        await adapter.handle_message(event)
        store = adapter._client.sync_store
        fresh = DurableSyncStore(tmp_path, "unused", "unused", "unused", "unused")
        fresh.path = store.path
        await fresh.load()
        assert (event._gateway_accepted, fresh.reserve_intake("$input")) == (
            admitted,
            not admitted,
        )
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
async def test_restart_sibling_interrupts_only_the_admitted_watched_input(
    tmp_path,
    monkeypatch,
    transport,
):
    from tests.integration.matrix_live import restart_barriers

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    adapter = make_adapter()
    responses.append(batch("before"))
    assert await adapter.connect()
    adapter._client.crypto = SimpleNamespace(_receive_room_key=AsyncMock())
    restart_barriers.register(
        SimpleNamespace(
            register_platform_handler=lambda _platform, attach: attach(
                adapter._client, adapter
            )
        )
    )
    entered = asyncio.Event()
    interrupted = []

    async def barrier(_home, marker, event_id):
        interrupted.append((marker, event_id))
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(restart_barriers, "_barrier", barrier)
    notice = message("$thread-root")
    notice["content"]["msgtype"] = "m.notice"
    prime = message("$prime")
    prime["content"]["body"] = "Prime encrypted thread"
    watched = message("$watched")
    watched["content"]["body"] = "Watch the split answer"
    dispatch = None
    try:
        (tmp_path / "room-input-release").touch()
        await asyncio.wait_for(
            adapter._absorb_sync(adapter._client, batch("primed", notice, prime)),
            timeout=2,
        )
        dispatch = asyncio.create_task(
            adapter._absorb_sync(adapter._client, batch("watched", watched))
        )
        await asyncio.wait_for(entered.wait(), timeout=2)
        store = adapter._client.sync_store
        assert (
            interrupted,
            [call.args[2] for call in adapter._handle_text_message.await_args_list],
            await store.get_next_batch(),
            store.reserve_intake("$watched"),
            dispatch.done(),
        ) == (
            [("intake-blocked", "$watched")],
            ["$prime", "$watched"],
            "primed",
            False,
            False,
        )
    finally:
        if dispatch is not None:
            dispatch.cancel()
            await asyncio.gather(dispatch, return_exceptions=True)
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
@pytest.mark.parametrize("phase", ["initial", "secondary", "running", "restart"])
async def test_gateway_stop_awaits_native_import_before_closing_transport(
    tmp_path, monkeypatch, transport, phase
):
    from gateway.config import GatewayConfig, HomeChannel, Platform
    from mautrix.types import EventType
    from gateway.run import GatewayRunner
    from plugins.platforms.matrix.sync_transport import DurableSyncStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    adapter = make_adapter()
    runner = GatewayRunner(
        GatewayConfig(
            platforms={
                Platform.MATRIX: PlatformConfig(
                    enabled=True,
                    home_channel=HomeChannel(
                        Platform.MATRIX, "!room:example.org", "Home"
                    ),
                )
            },
            sessions_dir=tmp_path / "sessions",
        )
    )
    monkeypatch.setattr(runner, "_create_adapter", lambda *_args: adapter)
    for method in (
        "_start_log_startup_environment",
        "_start_recover_previous_run",
        "_run_free_tier_bootstrap",
        "_start_finish_wiring",
        "_stop_hosted_room_worker",
    ):
        monkeypatch.setattr(runner, method, AsyncMock())
    monkeypatch.setattr(runner, "_start_startup_warmup", lambda: None)
    monkeypatch.setattr(runner, "_start_spawn_background_watchers", lambda: None)
    monkeypatch.setattr(runner, "_start_install_faulthandler", lambda: None)
    monkeypatch.setattr(runner, "_start_check_access_policy", lambda: False)

    profile_home = tmp_path / "secondary" if phase == "secondary" else tmp_path
    if phase == "secondary":
        runner.config.platforms = {}
        monkeypatch.setattr(runner, "_multiplex_on", lambda: True)
        monkeypatch.setattr(
            runner,
            "_load_secondary_profile_config",
            AsyncMock(
                return_value=GatewayConfig(platforms={Platform.MATRIX: adapter.config})
            ),
        )

        async def secondary_start():
            return await runner._start_one_profile_adapters(
                "secondary", profile_home, {}
            )

        monkeypatch.setattr(
            runner, "_start_secondary_profile_adapters", secondary_start
        )

    from gateway.run import _profile_runtime_scope

    with _profile_runtime_scope(profile_home, hydrate_secrets=False):
        store = DurableSyncStore(
            adapter._resolve_store_dir(),
            adapter._homeserver,
            adapter._user_id,
            adapter._device_id,
            adapter._access_token,
        )
    await store.put_next_batch("before")
    entered, unwinding, release, finished = (asyncio.Event() for _ in range(4))
    lock = asyncio.Lock()
    imported = []
    clients = []
    connections = []
    owned = []

    async def import_key(_event):
        if finished.is_set():
            imported.append("key")
            return
        async with lock:
            imported.append("key")
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                unwinding.set()
                await release.wait()
                imported.clear()
                finished.set()

    async def middleware(_event):
        return True

    async def send_notice(*_args, **_kwargs):
        async with lock:
            return "$notice"

    def attach(client):
        clients.append(client)
        connections.append(asyncio.current_task())
        client.add_event_middleware(EventType.ROOM_KEY, middleware)
        client.add_event_handler(EventType.ROOM_KEY, import_key)
        monkeypatch.setattr(client, "send_message_event", send_notice)

    monkeypatch.setattr(adapter, "_wire_plugin_handlers", attach)
    response = batch("after", message("$after-key"))
    response["to_device"] = {
        "events": [
            {
                "type": "m.room_key",
                "sender": "@alice:example.org",
                "content": {
                    "algorithm": "m.megolm.v1.aes-sha2",
                    "room_id": "!room:example.org",
                    "session_id": "session",
                    "session_key": "key",
                },
            }
        ]
    }
    responses.append(response if phase != "running" else batch("before"))
    startup = asyncio.create_task(runner.start())
    dispatch = None
    stop = None
    waiter = None
    cleanup_waiter = None
    try:
        if phase == "running":
            assert await startup
            dispatch = asyncio.create_task(
                adapter._absorb_sync(adapter._client, response)
            )
            adapter._sync_task = dispatch
        await asyncio.wait_for(entered.wait(), timeout=2)
        client = clients[0]
        owned = list(client.hermes_sync._owned_tasks)
        cleanup_waiting = asyncio.Event()
        await_cleanup = runner._await_adapter_cleanup_with_timeout

        async def track_cleanup(awaitable, timeout):
            if asyncio.current_task() is runner._stop_task:
                cleanup_waiting.set()
            return await await_cleanup(awaitable, timeout)

        monkeypatch.setattr(runner, "_await_adapter_cleanup_with_timeout", track_cleanup)
        if phase == "restart":
            runner._restart_requested = True
            await asyncio.wait_for(unwinding.wait(), timeout=2)
        stop = asyncio.create_task(runner.stop())
        waiter = asyncio.create_task(unwinding.wait())
        await asyncio.wait(
            {stop, waiter}, timeout=2, return_when=asyncio.FIRST_COMPLETED
        )
        assert (unwinding.is_set(), stop.done()) == (True, False)
        if phase != "running":
            cleanup_waiter = asyncio.create_task(cleanup_waiting.wait())
            await asyncio.wait(
                {stop, cleanup_waiter}, timeout=2, return_when=asyncio.FIRST_COMPLETED
            )
            assert (cleanup_waiting.is_set(), stop.done()) == (True, False)
        release.set()
        await asyncio.wait_for(stop, timeout=2)
        assert (
            finished.is_set(),
            imported,
            [task.done() for task in owned],
            client.hermes_sync._owned_tasks,
            await client.sync_store.get_next_batch(),
            adapter._client,
            [task.done() for task in connections],
        ) == (True, [], [True] * len(owned), set(), "before", None, [True])
        assert await asyncio.wait_for(startup, timeout=2)

        fresh = make_adapter()
        monkeypatch.setattr(fresh, "_wire_plugin_handlers", attach)
        responses.append(response)
        try:
            with _profile_runtime_scope(profile_home, hydrate_secrets=False):
                assert await fresh.connect()
            assert (
                imported,
                fresh._handle_text_message.await_count,
                await fresh._client.sync_store.get_next_batch(),
            ) == (["key"], 1, "after")
        finally:
            await fresh.disconnect()
    finally:
        release.set()
        for task in (waiter, cleanup_waiter, dispatch, startup, stop):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (waiter, cleanup_waiter, dispatch, startup, stop) if task is not None),
            return_exceptions=True,
        )
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
@pytest.mark.parametrize(
    "boundary",
    [
        "merged",
        "restart",
        "reconnect",
        "refused",
        "application-failure",
        "middleware-failure",
        "middleware-ignore",
    ],
)
async def test_native_sync_checkpoints_only_completed_application_admission(
    tmp_path,
    monkeypatch,
    transport,
    boundary,
):
    from gateway.platforms.event import MessageEvent
    from plugins.platforms.matrix.sync_transport import DurableSyncStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    adapter = make_adapter()
    responses.append(batch("s1"))
    assert await adapter.connect()
    del adapter._handle_text_message
    adapter._source_session_key = lambda _source: "session"
    source = adapter.build_source(
        chat_id="!room:example.org", user_id="@alice:example.org"
    )
    adapter._build_inbound_event = AsyncMock(
        side_effect=lambda _room, _sender, event_id, body, *_args: MessageEvent(
            text=body, source=source, message_id=event_id
        )
    )
    adapter._message_handler = AsyncMock()
    entered, release, model_entered, model_release = (asyncio.Event() for _ in range(4))
    real_sleep = asyncio.sleep

    async def clock(delay):
        if delay == adapter._text_batch_delay_seconds:
            entered.set()
            await release.wait()
            return
        await real_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", clock)
    admitted = []

    async def model(event, _key):
        admitted.append(event.text)
        model_entered.set()
        await model_release.wait()

    adapter._process_message_background = model
    start_session = adapter._start_session_processing
    if boundary == "refused":
        adapter._message_handler = None
    if boundary == "application-failure":
        adapter._start_session_processing = lambda *_args: (_ for _ in ()).throw(
            RuntimeError("admission failed")
        )
    if boundary.startswith("middleware"):

        async def middleware(_event):
            if boundary == "middleware-failure":
                raise RuntimeError("middleware failed")
            return False

        adapter._client.add_event_middleware(matrix.EventType.ROOM_MESSAGE, middleware)
    response = batch("s2", message("$first"), message("$second"))
    client = adapter._client
    fresh = DurableSyncStore(tmp_path, "unused", "unused", "unused", "unused")
    fresh.path = client.sync_store.path

    async def persisted(*event_ids):
        await fresh.load()
        return (
            await fresh.get_next_batch(),
            [fresh.reserve_intake(event_id) for event_id in event_ids],
        )

    dispatch = asyncio.create_task(adapter._absorb_sync(client, response))
    try:
        if boundary.startswith("middleware"):
            if boundary == "middleware-failure":
                with pytest.raises(RuntimeError, match="middleware failed"):
                    await dispatch
                assert await client.sync_store.get_next_batch() == "s1"
            else:
                await dispatch
                assert await client.sync_store.get_next_batch() == "s2"
            assert admitted == []
            return
        # The response is absorbed while its text batch waits for the quiet period.
        assert await asyncio.wait_for(dispatch, timeout=2) == "s2"
        await entered.wait()
        assert (admitted, await persisted("$first", "$second")) == (
            [],
            ("s1", [True, True]),
        )
        if boundary in {"restart", "reconnect"}:
            await adapter.disconnect()
            assert await persisted("$first", "$second") == ("s1", [True, True])
            if boundary == "restart":
                adapter = make_adapter()
                del adapter._handle_text_message
                adapter._source_session_key = lambda _source: "session"
                adapter._build_inbound_event = AsyncMock(
                    side_effect=lambda _room, _sender, event_id, body, *_args: MessageEvent(
                        text=body, source=source, message_id=event_id
                    )
                )
                adapter._message_handler = AsyncMock()
                adapter._process_message_background = model
            release.set()
            responses.append(response)
            assert await adapter.connect(is_reconnect=boundary == "reconnect")
            await asyncio.wait_for(model_entered.wait(), timeout=2)
            await adapter._sync_checkpoints.settled()
            assert (admitted, await persisted("$first", "$second")) == (
                ["$first\n$second"],
                ("s2", [True, True]),
            )
            return
        release.set()
        await adapter._sync_checkpoints.settled()
        if boundary in {"refused", "application-failure"}:
            assert (admitted, await persisted("$first", "$second")) == (
                [],
                ("s1", [True, True]),
            )
            # The sync loop resumes from the saved cursor and hands both events over again.
            assert await adapter._rewind_failed_intake(client)
            adapter._message_handler = AsyncMock()
            adapter._start_session_processing = start_session
            release.clear()
            entered.clear()
            await adapter._absorb_sync(client, response)
            await entered.wait()
            release.set()
            await adapter._sync_checkpoints.settled()
            await model_entered.wait()
            assert (admitted, await persisted("$first", "$second")) == (
                ["$first\n$second"],
                ("s2", [True, True]),
            )
            return
        await model_entered.wait()
        assert (admitted, model_release.is_set(), await persisted("$first", "$second")) == (
            ["$first\n$second"],
            False,
            ("s2", [True, True]),
        )

        async def fail(_event):
            raise RuntimeError("sibling failed")

        client.add_event_handler(matrix.EventType.ROOM_MESSAGE, fail)
        release.clear()
        entered.clear()
        adapter._active_sessions.clear()
        response = batch("s3", message("$third"), message("$fourth"))
        with pytest.raises(RuntimeError, match="sibling failed"):
            await adapter._absorb_sync(client, response)
        await entered.wait()
        release.set()
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())
        assert await persisted("$third", "$fourth") == ("s2", [False, False])
    finally:
        release.set()
        model_release.set()
        dispatch.cancel()
        await asyncio.gather(dispatch, return_exceptions=True)
        await adapter.disconnect()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
@pytest.mark.parametrize("checkpoint", ["completed", "unfinished"])
async def test_startup_replay_executes_admitted_input_before_native_checkpoint(
    tmp_path, monkeypatch, transport, checkpoint
):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    adapter = make_adapter()
    responses.append(batch("s1"))
    assert await adapter.connect()
    del adapter._handle_text_message
    adapter._source_session_key = lambda _source: "session"
    source = adapter.build_source(
        chat_id="!room:example.org", user_id="@alice:example.org"
    )
    adapter._build_inbound_event = AsyncMock(
        side_effect=lambda _room, _sender, event_id, body, *_args: MessageEvent(
            text=body, source=source, message_id=event_id
        )
    )
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._startup_restore_in_progress = True
    runner._startup_restore_queue = []
    runner._startup_restore_tasks = []
    runner._sync_voice_mode_state_to_adapter = lambda _adapter: None
    runner._bind_voice_input_callback = lambda _adapter: None
    runner._await_startup_warmup = AsyncMock()
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._hm_pre_gateway_dispatch_hook = AsyncMock(
        side_effect=lambda event, _source: event
    )
    runner._is_user_authorized_for_source = lambda _source: True
    runner._admit_bot_message_for_source = lambda _source: True
    queued, sibling_entered, sibling_release, executed = (
        asyncio.Event() for _ in range(4)
    )
    seen = []

    async def handler(event):
        admitted = await runner._hm_admit_event(event)
        if admitted is None:
            queued.set()
            return
        seen.append((event.message_id, event.text))
        executed.set()

    adapter.set_message_handler(handler)
    real_sleep = asyncio.sleep

    async def clock(delay):
        if delay == adapter._text_batch_delay_seconds:
            return
        await real_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", clock)
    client = adapter._client
    accept = client.sync_store.accept_intakes
    accepted = asyncio.Event()

    async def receipt(event_ids):
        await accept(event_ids)
        accepted.set()

    client.sync_store.accept_intakes = receipt

    async def sibling(_event):
        if checkpoint == "completed":
            return
        await accepted.wait()
        sibling_entered.set()
        await sibling_release.wait()

    client.add_event_handler(matrix.EventType.ROOM_MESSAGE, sibling)
    dispatch = asyncio.create_task(
        adapter._absorb_sync(client, batch("s2", message("$queued")))
    )
    try:
        await asyncio.wait_for(queued.wait(), timeout=2)
        if checkpoint == "completed":
            await dispatch
        else:
            await asyncio.wait_for(sibling_entered.wait(), timeout=2)
        await asyncio.gather(*adapter._background_tasks)
        assert (seen, len(runner._startup_restore_queue)) == ([], 1)
        runner._publish_primary_adapter(Platform.MATRIX, adapter)
        await runner._finish_startup_restore()
        await asyncio.gather(*adapter._background_tasks)
        assert (seen, executed.is_set(), runner._startup_restore_queue) == (
            [("$queued", "$queued")],
            True,
            [],
        )
        assert await client.sync_store.get_next_batch() == (
            "s2" if checkpoint == "completed" else "s1"
        )
        if checkpoint == "unfinished":
            assert not client.sync_store.reserve_intake("$queued")
        sibling_release.set()
        await dispatch
        assert await client.sync_store.get_next_batch() == "s2"
    finally:
        sibling_release.set()
        dispatch.cancel()
        await asyncio.gather(dispatch, return_exceptions=True)
        await adapter.disconnect()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
@pytest.mark.parametrize("buffered", [False, True])
async def test_restart_fixture_buffers_prime_before_initial_checkpoint_and_watch(
    tmp_path, monkeypatch, transport, buffered
):
    from gateway.platforms.event import MessageEvent
    from plugins.platforms.matrix.sync_transport import DurableSyncStore
    from tests.integration.matrix_live import restart_barriers

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    blocked = asyncio.Event()
    interrupted = []
    admitted = []
    real_sleep = asyncio.sleep

    async def clock(delay):
        if delay == 0.6:
            return
        await real_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", clock)

    async def barrier(home, marker, event_id):
        (home / marker).write_text(event_id, encoding="utf-8")
        interrupted.append((marker, event_id))
        blocked.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(restart_barriers, "_barrier", barrier)

    def attach(adapter):
        adapter._client.crypto = SimpleNamespace(_receive_room_key=AsyncMock())
        restart_barriers.register(
            SimpleNamespace(
                register_platform_handler=lambda _platform, handler: handler(
                    adapter._client, adapter
                )
            )
        )

    def create():
        adapter = make_adapter()
        del adapter._handle_text_message
        adapter._source_session_key = lambda _source: "session"
        source = adapter.build_source(
            chat_id="!room:example.org", user_id="@alice:example.org"
        )
        adapter._build_inbound_event = AsyncMock(
            side_effect=lambda _room, _sender, event_id, body, *_args: MessageEvent(
                text=body, source=source, message_id=event_id
            )
        )

        async def handler(event):
            admitted.append((event.message_id, event.text))

        adapter.set_message_handler(handler)
        adapter._wire_plugin_handlers = lambda _client: attach(adapter)
        return adapter

    store = DurableSyncStore(
        tmp_path / "platforms/matrix/store",
        "https://matrix.example.org",
        "@bot:example.org",
        "DEVICE",
        "test-token",
    )
    await store.put_next_batch("before")
    (tmp_path / "room-input-release").touch()
    if buffered:
        (tmp_path / "text-buffered").write_text("$prime", encoding="utf-8")
    prime = message("$prime")
    prime["content"]["body"] = "Prime encrypted thread"
    responses.append(batch("primed", prime))
    adapter = create()
    connect = asyncio.create_task(adapter.connect())
    dispatch = None
    try:
        if not buffered:
            connected = await connect
            await asyncio.wait_for(blocked.wait(), timeout=2)
            assert (
                connected,
                interrupted,
                admitted,
                await adapter._client.sync_store.get_next_batch(),
            ) == (True, [("text-buffered", "$prime")], [], "before")
            await adapter.disconnect()
            await store.load()
            assert (await store.get_next_batch(), store.reserve_intake("$prime")) == (
                "before",
                True,
            )
            responses.append(batch("primed", prime))
            adapter = create()
            connect = asyncio.create_task(adapter.connect())
        assert await connect
        await asyncio.gather(*adapter._background_tasks)
        assert (admitted, await adapter._client.sync_store.get_next_batch()) == (
            [("$prime", "Prime encrypted thread")],
            "primed",
        )
        blocked.clear()
        watched = message("$watched")
        watched["content"]["body"] = "Watch the split answer"
        dispatch = asyncio.create_task(
            adapter._absorb_sync(adapter._client, batch("watched", watched))
        )
        waiter = asyncio.create_task(blocked.wait())
        await asyncio.wait([waiter, dispatch], return_when=asyncio.FIRST_COMPLETED)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await asyncio.gather(*adapter._background_tasks)
        assert (
            interrupted[-1],
            admitted,
            dispatch.done(),
            await adapter._client.sync_store.get_next_batch(),
            adapter._client.sync_store.reserve_intake("$watched"),
        ) == (
            ("intake-blocked", "$watched"),
            [
                ("$prime", "Prime encrypted thread"),
                ("$watched", "Watch the split answer"),
            ],
            False,
            "primed",
            False,
        )
    finally:
        connect.cancel()
        if dispatch is not None:
            dispatch.cancel()
        await asyncio.gather(
            connect,
            *(task for task in (dispatch,) if task is not None),
            return_exceptions=True,
        )
        await adapter.disconnect()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)


_SYNC_LOOP = matrix.MatrixAdapter._sync_loop


def gateway_intake(adapter, texts):
    """Route timeline text through the real Matrix and base adapter intake."""
    from gateway.platforms.event import MessageEvent, MessageType

    del adapter._handle_text_message
    adapter._source_session_key = lambda _source: "session"
    source = adapter.build_source(
        chat_id="!room:example.org", user_id="@alice:example.org"
    )

    def build(_room, _sender, event_id, body, *_args):
        text = texts.get(event_id, body)
        return MessageEvent(
            text=text,
            source=source,
            message_id=event_id,
            message_type=MessageType.COMMAND if text.startswith("/") else MessageType.TEXT,
        )

    adapter._build_inbound_event = AsyncMock(side_effect=build)
    return source


def occupy_session(adapter):
    running = asyncio.create_task(asyncio.Event().wait())
    adapter._active_sessions["session"] = asyncio.Event()
    adapter._session_tasks["session"] = running
    return running


BUSY_CONSUMED = [
    "redirected-follow-up",
    "unauthorised-sender",
    "approve-command",
    "stop-command",
    "clarify-reply",
    "unresolved-identity",
    "unresolved-identity-batched",
]


def consume_while_busy(adapter, monkeypatch, kind, handled):
    """Install the gateway path that consumes *kind* without queueing it."""
    from gateway.run import GatewayRunner

    async def message_handler(event):
        handled.append(event.text)

    async def new_turn(event, _key):
        handled.append(event.text)

    adapter.set_message_handler(message_handler)
    adapter._process_message_background = new_turn
    if kind == "redirected-follow-up":

        async def redirect(event, _key):
            # A successful redirect or steer returns True without queueing the event.
            handled.append(event.text)
            return True

        adapter.set_busy_session_handler(redirect)
    if kind == "unauthorised-sender":
        runner = object.__new__(GatewayRunner)
        runner._is_user_authorized_for_source = lambda _source: False

        async def unauthorised(event, key):
            handled.append(event.text)
            return await runner._handle_active_session_busy_message(event, key)

        adapter.set_busy_session_handler(unauthorised)
    if kind == "clarify-reply":
        from tools import clarify_gateway

        monkeypatch.setattr(
            clarify_gateway,
            "get_pending_for_session",
            lambda _key, **_kwargs: SimpleNamespace(awaiting_text=True),
        )
    if kind.startswith("unresolved-identity"):
        adapter._active_sessions.clear()

        def unresolved(event):
            handled.append(event.text)
            return True

        adapter._drop_unresolved = unresolved


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
@pytest.mark.parametrize("kind", BUSY_CONSUMED)
async def test_input_consumed_while_busy_is_acknowledged_once(
    tmp_path, monkeypatch, transport, kind
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    delay = "0.2" if kind.endswith("-batched") else "0"
    monkeypatch.setenv("HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS", delay)
    _, responses = transport
    adapter = make_adapter()
    responses.append(batch("s1"))
    assert await adapter.connect()
    commands = {"approve-command": "/approve", "stop-command": "/stop"}
    gateway_intake(adapter, {"$input": commands.get(kind, "also this")})
    running = occupy_session(adapter)
    handled = []
    consume_while_busy(adapter, monkeypatch, kind, handled)
    response = batch("s2", message("$input"))
    outcomes = []
    try:
        for _ in range(3):
            try:
                await adapter._absorb_sync(adapter._client, response)
                outcomes.append("acknowledged")
            except RuntimeError as exc:
                outcomes.append(str(exc))
        await asyncio.gather(
            *adapter._pending_text_batch_tasks.values(), return_exceptions=True
        )
        assert (
            handled,
            outcomes,
            await adapter._client.sync_store.get_next_batch(),
        ) == (
            [commands.get(kind, "also this")],
            ["acknowledged"] * 3,
            "s2",
        )
    finally:
        running.cancel()
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_busy_approve_runs_once_across_repeated_sync_requests(
    tmp_path, monkeypatch, transport
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MATRIX_TEXT_BATCH_DELAY_SECONDS", "0")
    _, responses = transport
    adapter = make_adapter()
    responses.append(batch("s1"))
    assert await adapter.connect()
    gateway_intake(adapter, {"$approve": "/approve"})
    running = occupy_session(adapter)
    handled = []
    consume_while_busy(adapter, monkeypatch, "approve-command", handled)
    real_sleep = asyncio.sleep

    async def retry_clock(delay):
        await real_sleep(0 if delay == 5 else delay)

    monkeypatch.setattr(asyncio, "sleep", retry_clock)
    requested, observed = [], asyncio.Event()

    async def homeserver(*, since=None, **_kwargs):
        requested.append(since)
        if len(requested) == 3:
            observed.set()
            await asyncio.Event().wait()
        if since == "s1":
            return batch("s2", message("$approve"))
        return batch(since)

    adapter._client.sync = homeserver
    adapter._sync_task = asyncio.create_task(_SYNC_LOOP(adapter))
    try:
        await asyncio.wait_for(observed.wait(), timeout=2)
        assert (handled, requested) == (["/approve"], ["s1", "s2", "s2"])
    finally:
        running.cancel()
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_text_batch_spans_sync_responses_before_the_cursor_advances(
    tmp_path, monkeypatch, transport
):
    from plugins.platforms.matrix.sync_transport import DurableSyncStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    adapter = make_adapter()
    responses.append(batch("s1"))
    assert await adapter.connect()
    gateway_intake(adapter, {})
    turns = []

    async def model(event, _key):
        turns.append(event.text)

    adapter.set_message_handler(AsyncMock())
    adapter._process_message_background = model
    quiet, release = asyncio.Event(), asyncio.Event()
    real_sleep = asyncio.sleep

    async def batch_clock(delay):
        if delay == adapter._text_batch_delay_seconds:
            quiet.set()
            await release.wait()
            return
        await real_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", batch_clock)
    client = adapter._client
    durable = DurableSyncStore(tmp_path, "unused", "unused", "unused", "unused")
    durable.path = client.sync_store.path

    async def persisted():
        await durable.load()
        return (
            await durable.get_next_batch(),
            [durable.reserve_intake(event_id) for event_id in ("$first", "$second")],
        )

    try:
        # The sync loop issues the next long poll only after this returns.
        await asyncio.wait_for(
            adapter._absorb_sync(client, batch("s2", message("$first"))), timeout=2
        )
        await asyncio.wait_for(
            adapter._absorb_sync(client, batch("s3", message("$second"))), timeout=2
        )
        await quiet.wait()
        buffered = (list(turns), await persisted())
        release.set()
        await adapter._sync_checkpoints.settled()
        assert (buffered, turns, await persisted()) == (
            ([], ("s1", [True, True])),
            ["$first\n$second"],
            ("s3", [True, True]),
        )
    finally:
        release.set()
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_sync_loop_retries_a_batch_that_the_gateway_did_not_consume(
    tmp_path, monkeypatch, transport
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    adapter = make_adapter()
    responses.append(batch("s1"))
    assert await adapter.connect()
    gateway_intake(adapter, {})
    turns = []

    async def model(event, _key):
        turns.append(event.text)

    adapter._process_message_background = model
    real_sleep = asyncio.sleep
    requested, polling, observed = [], asyncio.Event(), asyncio.Event()

    async def clock(delay):
        if delay == adapter._text_batch_delay_seconds:
            await polling.wait()
        await real_sleep(0 if delay in {5, adapter._text_batch_delay_seconds} else delay)

    monkeypatch.setattr(asyncio, "sleep", clock)

    async def homeserver(*, since=None, **_kwargs):
        requested.append(since)
        if len(requested) == 2:
            # The batch from s2 reaches the gateway before any handler is installed.
            polling.set()
            await adapter._sync_checkpoints.settled()
            adapter.set_message_handler(AsyncMock())
        if len(requested) == 4:
            observed.set()
            await asyncio.Event().wait()
        if since == "s1":
            return batch("s2", message("$first"))
        return batch(since)

    adapter._client.sync = homeserver
    adapter._sync_task = asyncio.create_task(_SYNC_LOOP(adapter))
    try:
        await asyncio.wait_for(observed.wait(), timeout=2)
        await adapter._sync_checkpoints.settled()
        assert (
            requested,
            turns,
            await adapter._client.sync_store.get_next_batch(),
        ) == (["s1", "s2", "s1", "s2"], ["$first"], "s2")
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected_by", ["connect", "sync-loop"])
async def test_rejected_cursor_after_restart_does_not_replay_handled_history(
    tmp_path, monkeypatch, transport, rejected_by
):
    requests, responses = transport
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = make_adapter()
    handled_yesterday = message("$handled-yesterday", int((NOW - 1) * 1000))
    responses.append(batch("s1", handled_yesterday))
    assert await first.connect()
    await first.disconnect()

    # A day later the gateway restarts and the homeserver rejects the saved cursor.
    monkeypatch.setattr(matrix.time, "time", lambda: NOW + 86400)
    restarted = make_adapter()
    offline = message("$sent-while-offline", int((NOW + 86399) * 1000))
    if rejected_by == "connect":
        responses.extend([
            SyncFailure("expired since token"),
            batch("s2", handled_yesterday),
        ])
        assert await restarted.connect()
    else:
        responses.append(batch("s2", offline))
        assert await restarted.connect()
        responses.extend([
            SyncFailure("expired since token"),
            SyncFailure("expired since token"),
            batch("s3", handled_yesterday),
        ])
        real_sleep = asyncio.sleep
        monkeypatch.setattr(asyncio, "sleep", lambda delay: real_sleep(0))
        restarted._sync_task = asyncio.create_task(_SYNC_LOOP(restarted))

        async def fallback_absorbed():
            # The sixth request follows the absorbed fallback response.
            while len(requests) < 6:
                await real_sleep(0)

        await asyncio.wait_for(fallback_absorbed(), timeout=2)
    handled = [c.args[2] for c in restarted._handle_text_message.await_args_list]
    await restarted.disconnect()
    assert (first._handle_text_message.await_count, handled) == (
        1,
        ["$sent-while-offline"] if rejected_by == "sync-loop" else [],
    )


@pytest.mark.asyncio
async def test_connect_purges_reaction_watches_that_expired_while_stopped(
    tmp_path, monkeypatch, transport
):
    import sqlite3
    from datetime import datetime

    from plugins.platforms.matrix.reaction_followups import (
        WATCH_SECONDS,
        ReactionWatchStore,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    adapter = make_adapter()
    path = adapter._followup_store_path()
    # The transport fixture freezes time.time for the adapter; the watch store keeps the real clock.
    for event_id, age in (("$expired", WATCH_SECONDS + 100), ("$live", 100)):
        ReactionWatchStore(path, clock=lambda age=age: datetime.now().timestamp() - age).arm(
            "turn" + event_id, (event_id,), profile="", room_id="!room:example.org",
            thread_id="", session_key="session", session_id="sid",
            requester="@alice:example.org", source={}, emoji_filter=(),
            delivery_event_id=event_id,
        )
    responses.append(batch("s1"))
    assert await adapter.connect()
    try:
        with sqlite3.connect(path) as db:
            remaining = db.execute("SELECT event_id FROM watches").fetchall()
        next_purge = adapter._watch_purge_handle.when() - asyncio.get_running_loop().time()
        assert (remaining, WATCH_SECONDS - 110 < next_purge <= WATCH_SECONDS - 100) == (
            [("$live",)],
            True,
        )
    finally:
        await adapter.disconnect()
    assert adapter._watch_purge_handle is None
