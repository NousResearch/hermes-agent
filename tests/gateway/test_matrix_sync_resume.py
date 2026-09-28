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
    adapter._handle_text_message = AsyncMock()
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
async def test_rejected_cursor_refreshes_state_and_preserves_offline_intake(
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
        "$offline",
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
