"""Image packs bind selections to current bot, room policy and profile."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from contextlib import ExitStack
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import quote, unquote

import pytest
from typing import Any

matrix_types = pytest.importorskip("mautrix.types")
StateEvent = matrix_types.StateEvent
RoomEncryptionStateEventContent = matrix_types.RoomEncryptionStateEventContent
RoomID = matrix_types.RoomID
MemoryStateStore = pytest.importorskip("mautrix.client.state_store").MemoryStateStore
ClientAPI = pytest.importorskip("mautrix.client.api").ClientAPI
MNotFound = pytest.importorskip("mautrix.errors").MNotFound

from agent.secret_scope import is_multiplex_active, set_multiplex_active
from gateway.run import _profile_runtime_scope
from gateway.config import PlatformConfig
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.matrix.adapter import MatrixAdapter
from plugins.platforms.matrix.image_packs import MAX_PACKS, MAX_STATE_EVENTS

ROOM = "!room:example.org"
BOT = "@hermes:example.org"
USER = "@alice:example.org"
PACK: dict[str, Any] = {
    "pack": {"display_name": "/new same label", "usage": ["sticker"]},
    "images": {
        "fox": {
            "url": "mxc://example.org/fox",
            "body": "Fox.png",
            "info": {"mimetype": "image/png", "size": 20, "w": 1, "h": 2},
        }
    },
}
CREATE = {
    "type": "m.room.create",
    "state_key": "",
    "room_id": ROOM,
    "sender": USER,
    "event_id": "$create",
    "origin_server_ts": 1000,
    "content": {"room_version": "10"},
}
SENT = {"success": True, "event_id": "$sent"}
UNAVAILABLE = {"error": "Matrix image selection is unavailable; list packs again"}
CONVERSATION_CHANGED = {
    "error": "Matrix image-pack conversation changed; list packs again"
}
OWNER_CHANGED = {"error": "Matrix image-pack owner or profile changed"}
ADMISSION_CHANGED = {"error": "Matrix room admission changed"}
IMAGE_CHANGED = {"error": "Matrix image selection changed; list packs again"}
REFERENCE_CHANGED = {"error": "Matrix account pack reference changed; list packs again"}


def make_adapter(home, event_type="m.room.image_pack"):
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test",
            extra={
                "homeserver": "https://example.org",
                "user_id": BOT,
                "require_mention": False,
            },
        )
    )
    adapter._joined_rooms = {ROOM, "!reference:example.org"}
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter.set_authorization_check(lambda *_args, **_kwargs: True)
    adapter._resolve_store_dir()
    state = StateEvent.deserialize({
        "type": event_type,
        "state_key": "",
        "room_id": ROOM,
        "sender": BOT,
        "event_id": "$pack",
        "origin_server_ts": 1000,
        "content": deepcopy(PACK),
    })

    async def state_event(room_id, kind, state_key="", query=None):
        if kind in {"m.room.image_pack", "im.ponies.room_emotes"}:
            return deepcopy(PACK)
        if kind == "m.room.create":
            return deepcopy(CREATE if query == {"format": "event"} else CREATE["content"])
        raise MNotFound(404, "no state")

    async def request(method, path, content=None, *, query_params=None, **kwargs):
        if str(method) != "GET":
            return await api.send(method, path, content, **kwargs)
        room_id, _, state = str(path).partition("/rooms/")[2].partition("/state/")
        kind, _, state_key = state.partition("/")
        return await api.state(
            unquote(room_id), unquote(kind), unquote(state_key), query=query_params
        )

    api = SimpleNamespace(
        get_txn_id=lambda: "txn",
        request=request,
        state=AsyncMock(side_effect=state_event),
        send=AsyncMock(return_value={"event_id": "$sent"}),
    )
    client = SimpleNamespace(
        mxid=BOT,
        crypto=None,
        get_state=AsyncMock(return_value=[state]),
        get_account_data=AsyncMock(return_value={}),
        api=api,
    )
    client.get_state_event = MethodType(ClientAPI.get_state_event, client)
    adapter._client = client
    return adapter, client


def selections_replaced(catalog):
    return {
        **catalog,
        "packs": [
            {
                **pack,
                "items": [
                    {**item, "selection_id": "selected"} for item in pack["items"]
                ],
            }
            for pack in catalog["packs"]
        ],
    }


@pytest.fixture
def multiplex_profiles():
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        yield
    finally:
        set_multiplex_active(previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["m.room.image_pack", "im.ponies.room_emotes"])
@pytest.mark.parametrize("catalog_case", ["valid", "bounds", "malformed", "usage"])
async def test_catalog_and_send_use_exact_native_selection_across_two_homes(
    tmp_path, event_type, catalog_case, multiplex_profiles
):
    homes = [tmp_path / "a", tmp_path / "b"]
    for home in homes:
        home.mkdir()
    adapters = {}
    selections = {}
    for home in [homes[0], homes[1], homes[0]]:
        scope = ExitStack()
        scope.enter_context(
            _profile_runtime_scope(
                home,
                prepared_secret_scope={
                    "MATRIX_ACCESS_TOKEN": f"test-{home.name}",
                    "MATRIX_ALLOWED_USERS": USER,
                },
            )
        )
        if home not in adapters:
            adapters[home] = make_adapter(home, event_type)
        adapter, client = adapters[home]
        reference_type = (
            "m.image_pack.rooms"
            if event_type == "m.room.image_pack"
            else "im.ponies.emote_rooms"
        )

        async def account(kind):
            if kind == reference_type:
                return {
                    "rooms": {
                        "!reference:example.org": {"": {}, "named": {}},
                        "!empty:example.org": {},
                        "!denied:example.org": {"": {}},
                    }
                }
            if kind == "im.ponies.user_emotes":
                return deepcopy(PACK)
            return {}

        client.get_account_data.side_effect = account
        tokens = set_session_vars(
            platform="matrix",
            chat_id=ROOM,
            user_id=USER,
            thread_id="$root",
            message_id="$question",
            session_key="session",
            session_id="conversation",
            transport_adapter=adapter,
        )
        try:
            catalog = await adapter.matrix_image_packs("list", ROOM, requester=USER)
            assert selections_replaced(catalog) == {
                "untrusted_data": True,
                "account_user_id": BOT,
                "truncated": False,
                "errors": [
                    {
                        "source": "account_reference",
                        "error": "Matrix room is not allowed or joined",
                    }
                ],
                "packs": [
                    {
                        "source": source,
                        "event_type": kind,
                        "room_id": room,
                        "state_key": key,
                        "account_user_id": account_user,
                        "display_name": "/new same label",
                        "description": None,
                        "attribution": None,
                        "items": [
                            {
                                "selection_id": "selected",
                                "shortcode": "fox",
                                **PACK["images"]["fox"],
                            }
                        ],
                    }
                    for source, kind, room, key, account_user in [
                        ("room", event_type, ROOM, "", None),
                        (
                            "account_reference",
                            event_type,
                            "!reference:example.org",
                            "",
                            BOT,
                        ),
                        (
                            "account_reference",
                            event_type,
                            "!reference:example.org",
                            "named",
                            BOT,
                        ),
                        ("bot_account", "im.ponies.user_emotes", None, None, BOT),
                    ]
                ],
            }
            assert not any(
                call.args[0] in ("!denied:example.org", "!empty:example.org")
                for call in client.api.state.await_args_list
            )
            item = catalog["packs"][0]["items"][0]
            assert {**item, "selection_id": "selected"} == {
                "selection_id": "selected",
                "shortcode": "fox",
                "body": "Fox.png",
                "url": "mxc://example.org/fox",
                "info": {"mimetype": "image/png", "size": 20, "w": 1, "h": 2},
            }
            for foreign_home, foreign_selection in selections.items():
                if foreign_home != home:
                    refused = await adapter.matrix_image_packs(
                        "send", ROOM, requester=USER, selection_id=foreign_selection
                    )
                    assert refused == {
                        "error": "Matrix image selection is unavailable; list packs again"
                    }
            foreign_scope = set_hermes_home_override(
                homes[1] if home == homes[0] else homes[0]
            )
            try:
                assert await adapter.matrix_image_packs(
                    "send", ROOM, requester=USER, selection_id=item["selection_id"]
                ) == {
                    "error": "Matrix image selection is unavailable; list packs again",
                }
            finally:
                reset_hermes_home_override(foreign_scope)
            continuation = home in selections
            selections[home] = item["selection_id"]
            result = await adapter.matrix_image_packs(
                "send",
                ROOM,
                requester=USER,
                selection_id=item["selection_id"],
                reply_to="$question",
                thread_id="$root",
            )
            assert result == {"success": True, "event_id": "$sent"}
            call = client.api.send.await_args
            assert (
                str(call.args[0]),
                str(call.args[1]),
                call.args[2],
                call.kwargs,
            ) == (
                "PUT",
                f"_matrix/client/v3/rooms/{quote(ROOM, safe='')}/send/m.sticker/txn",
                {
                    **PACK["images"]["fox"],
                    "m.relates_to": {
                        "m.in_reply_to": {"event_id": "$sent" if continuation else "$question"},
                        "rel_type": "m.thread",
                        "event_id": "$root",
                        "is_falling_back": continuation,
                    },
                },
                {"metrics_method": "sendMessageEvent"},
            )
            original_state = client.get_state.return_value
            varied = deepcopy(PACK)
            if catalog_case == "bounds":
                varied["images"] = {
                    f"fox{index}": PACK["images"]["fox"] for index in range(101)
                }
            if catalog_case == "malformed":
                varied["images"] = {
                    "http": {"url": "https://example.org/fox"},
                    "bool": {"url": "mxc://example.org/fox", "info": {"w": True}},
                    "array": {"url": "mxc://example.org/fox", "info": []},
                    "large": {
                        "url": "mxc://example.org/fox",
                        "info": {"size": adapter._max_media_bytes + 1},
                    },
                    "nested": {
                        "url": "mxc://example.org/fox",
                        "info": {"extra": [1] * 257},
                    },
                }
            if catalog_case == "usage":
                varied["images"] = {
                    "inherit": {"url": "mxc://example.org/fox", "usage": []},
                    "emoji": {"url": "mxc://example.org/fox", "usage": ["emoticon"]},
                    "native": {"url": "mxc://example.org/fox"},
                }
            if catalog_case != "valid":
                client.get_state.return_value = [
                    StateEvent.deserialize({
                        "type": event_type,
                        "state_key": "",
                        "room_id": ROOM,
                        "sender": BOT,
                        "event_id": "$pack",
                        "origin_server_ts": 1000,
                        "content": varied,
                    })
                ]
                checked = await adapter.matrix_image_packs("list", ROOM, requester=USER)
                if catalog_case == "bounds":
                    assert (
                        checked["truncated"],
                        [len(pack["items"]) for pack in checked["packs"]],
                    ) == (True, [100])
                if catalog_case == "malformed":
                    assert checked["packs"][0]["items"] == []
                    assert [error["error"] for error in checked["errors"]] == [
                        "image URL is not MXC",
                        "image w must be a non-negative integer",
                        "image info must be an object",
                        "image exceeds the Matrix media limit",
                        "image metadata exceeds the catalog budget",
                        "Matrix room is not allowed or joined",
                    ]
                if catalog_case == "usage":
                    assert [
                        item["shortcode"] for item in checked["packs"][0]["items"]
                    ] == (
                        ["inherit", "emoji", "native"]
                        if event_type == "m.room.image_pack"
                        else ["inherit", "native"]
                    )
                client.get_state.return_value = original_state
        finally:
            clear_session_vars(tokens)
            scope.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change,expected",
    [
        ("changed-image", IMAGE_CHANGED),
        ("removed-pack", IMAGE_CHANGED),
        ("revoked", ADMISSION_CHANGED),
        ("replaced-client", OWNER_CHANGED),
        ("power", {"error": "Matrix bot cannot send m.sticker in this room"}),
        ("expired", {"error": "Matrix image selection expired; list packs again"}),
        ("crypto-revoked", ADMISSION_CHANGED),
        ("crypto-owner", OWNER_CHANGED),
        (
            "crypto-power",
            {"error": "Matrix bot cannot send m.room.encrypted in this room"},
        ),
        ("server", {"error": "M_FORBIDDEN: server refused sticker"}),
        ("late-encryption", SENT),
        (
            "late-encryption-missing",
            {"error": "Matrix encryption keys are unavailable"},
        ),
        ("reference-removed", REFERENCE_CHANGED),
        (
            "after-write-owner",
            {
                **SENT,
                "warning": (
                    "Matrix image-pack owner or profile changed after the server"
                    " accepted the sticker"
                ),
            },
        ),
    ],
)
async def test_send_rechecks_selection_and_admission_after_await(
    tmp_path, monkeypatch, change, expected
):
    from plugins.platforms.matrix import image_packs

    clock = [1000.0]
    monkeypatch.setattr(
        image_packs, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    scope = set_hermes_home_override(tmp_path)
    adapter, client = make_adapter(tmp_path)
    tokens = set_session_vars(
        platform="matrix",
        chat_id=ROOM,
        user_id=USER,
        session_key="session",
        session_id="conversation",
        transport_adapter=adapter,
    )
    try:
        if change.startswith("crypto-") or change == "late-encryption":
            client.crypto = object()
        if change.startswith("late-encryption"):
            client.state_store = MemoryStateStore()
            assert await client.state_store.is_encrypted(RoomID(ROOM)) is None
            client.encrypt = AsyncMock(return_value={"ciphertext": "encrypted"})
        references = {"rooms": {"!reference:example.org": {"": {}}}}
        if change == "reference-removed":
            client.get_account_data.side_effect = lambda kind: (
                references if kind == "m.image_pack.rooms" else {}
            )
        catalog = await adapter.matrix_image_packs("list", ROOM, requester=USER)
        selected_pack = (
            catalog["packs"][1]
            if change == "reference-removed"
            else catalog["packs"][0]
        )
        selected = selected_pack["items"][0]["selection_id"]
        entered, release = asyncio.Event(), asyncio.Event()
        original = client.api.state.side_effect
        original_api = client.api
        pack_reads = [0]

        crypto_case = change.startswith("crypto-")
        if crypto_case:

            async def encrypt(room_id, kind, content):
                entered.set()
                await release.wait()
                return {"ciphertext": "encrypted"}

            client.encrypt = AsyncMock(side_effect=encrypt)

        async def paused(room_id, kind, state_key="", **kwargs):
            if crypto_case and str(kind) == "m.room.encryption":
                return {"algorithm": "m.megolm.v1.aes-sha2"}
            if str(kind) == "m.room.image_pack" and not crypto_case:
                pack_reads[0] += 1
                entered.set()
                await release.wait()
                if change == "changed-image":
                    return {**PACK, "images": {"fox": {"url": "mxc://example.org/new"}}}
                if change == "removed-pack":
                    return {}
            if (
                change.startswith("late-encryption")
                and str(kind) == "m.room.encryption"
            ):
                if await client.state_store.is_encrypted(RoomID(ROOM)):
                    return {"algorithm": "m.megolm.v1.aes-sha2"}
            if (
                change in ("power", "crypto-power")
                and str(kind) == "m.room.power_levels"
            ):
                return {
                    "events": {
                        "m.sticker": 50,
                        "m.room.encrypted": 50 if release.is_set() else 0,
                    },
                    "users_default": 0,
                }
            return await original(room_id, kind, state_key, **kwargs)

        client.api.state.side_effect = paused
        sending = asyncio.create_task(
            adapter.matrix_image_packs(
                "send", ROOM, requester=USER, selection_id=selected
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=2)
        if change in ("revoked", "crypto-revoked"):
            adapter.set_authorization_check(lambda *_args, **_kwargs: False)
        if change in ("replaced-client", "crypto-owner"):
            adapter._client = SimpleNamespace(mxid=BOT)
        if change == "expired":
            clock[0] += image_packs.SELECTION_TTL + 1
        if change == "server":
            client.api.send.side_effect = RuntimeError(
                "M_FORBIDDEN: server refused sticker"
            )
        if change == "after-write-owner":

            async def accepted(*args, **kwargs):
                adapter._client = SimpleNamespace(mxid=BOT)
                return {"event_id": "$sent"}

            client.api.send.side_effect = accepted
        if change == "reference-removed":
            references["rooms"].clear()
        if change.startswith("late-encryption"):
            previous_admit = adapter._is_dm_room
            learned = False

            async def learn_encryption(room):
                nonlocal learned
                if not learned and pack_reads[0] >= 2:
                    learned = True
                    await client.state_store.set_encryption_info(
                        RoomID(ROOM), RoomEncryptionStateEventContent()
                    )
                return await previous_admit(room)

            adapter._is_dm_room = learn_encryption
        release.set()
        result = await asyncio.wait_for(sending, timeout=2)
        assert result == expected
        if change == "late-encryption":
            assert str(client.api.send.await_args.args[1]).endswith(
                "/send/m.room.encrypted/txn"
            )
            assert client.api.send.await_args.args[2] == {"ciphertext": "encrypted"}
            client.encrypt.assert_awaited_once()
            return
        if change in ("server", "after-write-owner"):
            original_api.send.assert_awaited_once()
            return
        original_api.send.assert_not_awaited()
    finally:
        clear_session_vars(tokens)
        reset_hermes_home_override(scope)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["initial", "late"])
@pytest.mark.parametrize(
    "change,expected",
    [
        ("image", IMAGE_CHANGED),
        ("reference", REFERENCE_CHANGED),
        ("conversation", CONVERSATION_CHANGED),
        ("owner", OWNER_CHANGED),
        ("profile", OWNER_CHANGED),
    ],
)
async def test_crypto_preparation_revalidates_source_and_live_conversation(
    tmp_path, phase, change, expected
):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore

    scope = set_hermes_home_override(tmp_path)
    adapter, client = make_adapter(tmp_path)
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    source = SessionSource(
        platform=Platform.MATRIX, chat_id=ROOM, chat_type="group", user_id=USER
    )
    entry = store.get_or_create_session(source)
    adapter.set_session_store(store)
    client.crypto = object()
    client.state_store = MemoryStateStore()
    assert await client.state_store.is_encrypted(RoomID(ROOM)) is None
    if phase == "initial":
        await client.state_store.set_encryption_info(
            RoomID(ROOM), RoomEncryptionStateEventContent()
        )
    references = {"rooms": {"!reference:example.org": {"": {}}}}
    client.get_account_data.side_effect = lambda kind: (
        references if kind == "m.image_pack.rooms" else {}
    )
    tokens = set_session_vars(
        platform="matrix",
        chat_id=ROOM,
        user_id=USER,
        session_key=entry.session_key,
        session_id=entry.session_id,
        transport_adapter=adapter,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        catalog = await adapter.matrix_image_packs("list", ROOM, requester=USER)
        selected = catalog["packs"][1]["items"][0]["selection_id"]
        original = client.api.state.side_effect
        pack_reads = 0
        removed = False
        decisions = []

        async def state(room, kind, key="", **kwargs):
            nonlocal pack_reads
            if str(kind) == "m.room.image_pack":
                pack_reads += 1
                if removed:
                    return {"pack": PACK["pack"], "images": {}}
            if str(kind) == "m.room.encryption":
                encrypted = bool(await client.state_store.is_encrypted(RoomID(ROOM)))
                decisions.append(encrypted)
                if encrypted:
                    return {"algorithm": "m.megolm.v1.aes-sha2"}
            return await original(room, kind, key, **kwargs)

        async def admit(room):
            if phase == "late" and pack_reads >= 2:
                await client.state_store.set_encryption_info(
                    RoomID(ROOM), RoomEncryptionStateEventContent()
                )
            return False

        async def encrypt(*args):
            entered.set()
            await release.wait()
            return {"ciphertext": "encrypted"}

        client.api.state.side_effect = state
        adapter._is_dm_room = admit
        client.encrypt = AsyncMock(side_effect=encrypt)
        sending = asyncio.create_task(
            adapter.matrix_image_packs(
                "send", ROOM, requester=USER, selection_id=selected
            )
        )
        await asyncio.wait_for(entered.wait(), 2)
        assert decisions[0] is (phase == "initial")
        if change == "image":
            removed = True
        if change == "reference":
            references["rooms"].clear()
        if change == "conversation":
            reset = store.reset_session(entry.session_key, source=source)
            assert reset is not None and reset.session_id != entry.session_id
        if change == "owner":
            adapter._client = SimpleNamespace(mxid=BOT)
        if change == "profile":
            adapter._owner_profile = "other-profile"
        release.set()
        result = await asyncio.wait_for(sending, 2)
        assert result == expected
        client.api.send.assert_not_awaited()
    finally:
        release.set()
        clear_session_vars(tokens)
        reset_hermes_home_override(scope)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary,expected",
    [
        ("in-place", (SENT, SENT, 2)),
        ("reset", (UNAVAILABLE, SENT, 1)),
        ("compression-during-turn", (UNAVAILABLE, SENT, 1)),
        ("compression-after-turn", (UNAVAILABLE, SENT, 1)),
        ("unrelated-conversation", (CONVERSATION_CHANGED, CONVERSATION_CHANGED, 0)),
    ],
)
async def test_selections_follow_conversation_identity_not_route(
    tmp_path, boundary, expected
):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_context import scoped_current_session_id

    scope = set_hermes_home_override(tmp_path)
    adapter, client = make_adapter(tmp_path)
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    source = SessionSource(
        platform=Platform.MATRIX, chat_id=ROOM, chat_type="group", user_id=USER
    )
    entry = store.get_or_create_session(source)
    adapter.set_session_store(store)
    tokens = set_session_vars(
        platform="matrix",
        chat_id=ROOM,
        user_id=USER,
        session_key=entry.session_key,
        session_id=entry.session_id,
        transport_adapter=adapter,
    )
    try:
        catalog = await adapter.matrix_image_packs("list", ROOM, requester=USER)
        selected = catalog["packs"][0]["items"][0]["selection_id"]
        parent_id = entry.session_id
        current_id = parent_id
        if boundary == "reset":
            reset = store.reset_session(entry.session_key, source=source)
            assert reset is not None and reset.session_id != parent_id
            current_id = reset.session_id
        if boundary.startswith("compression"):
            db = store._db_for_key(entry.session_key)
            db.end_session(parent_id, "compression")
            current_id = db.create_session(
                "compression-child", source="matrix", parent_session_id=parent_id
            )
        if boundary == "compression-after-turn":
            assert store.advance_compression_session(
                entry.session_key, parent_id, current_id
            )
        if boundary == "unrelated-conversation":
            current_id = "unrelated-conversation"
        with scoped_current_session_id(current_id):
            stale = await adapter.matrix_image_packs(
                "send", ROOM, requester=USER, selection_id=selected
            )
            listed = await adapter.matrix_image_packs("list", ROOM, requester=USER)
            fresh = (
                listed
                if "error" in listed
                else await adapter.matrix_image_packs(
                    "send",
                    ROOM,
                    requester=USER,
                    selection_id=listed["packs"][0]["items"][0]["selection_id"],
                )
            )
        assert (stale, fresh, client.api.send.await_count) == expected
    finally:
        clear_session_vars(tokens)
        reset_hermes_home_override(scope)


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["m.room.image_pack", "im.ponies.room_emotes"])
async def test_gateway_rebinds_conversation_for_cached_turns_across_profiles(
    tmp_path, event_type, multiplex_profiles, monkeypatch
):
    import importlib
    import json

    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore, build_session_context
    from gateway.session_context import scoped_current_session_id
    from tools.registry import registry

    importlib.import_module("tools.matrix_image_packs_tool")
    homes = [tmp_path / "a", tmp_path / "b"]
    owners = {}
    for home in homes:
        home.mkdir()
    for home in [homes[0], homes[1], homes[0]]:
        with _profile_runtime_scope(
            home, prepared_secret_scope={"MATRIX_ACCESS_TOKEN": "test"}
        ):
            if home not in owners:
                adapter, client = make_adapter(home, event_type)
                config = GatewayConfig()
                store = SessionStore(sessions_dir=home / "sessions", config=config)
                source = SessionSource(
                    platform=Platform.MATRIX,
                    chat_id=ROOM,
                    chat_type="group",
                    user_id=USER,
                )
                entry = store.get_or_create_session(source)
                adapter.set_session_store(store)
                runner = object.__new__(GatewayRunner)
                runner._gateway_loop = asyncio.get_running_loop()
                monkeypatch.setattr(
                    runner,
                    "_delivery_adapter_for",
                    lambda _source, owner=adapter: owner,
                )
                owners[home] = (
                    adapter,
                    client,
                    runner,
                    build_session_context(source, config, entry),
                )
            adapter, client, runner, context = owners[home]
            tokens = runner._set_session_env(context)
            try:
                with scoped_current_session_id(context.session_id):
                    listed = await asyncio.to_thread(
                        registry.dispatch, "matrix_image_packs", {"action": "list"}
                    )
                assert isinstance(listed, str)
                catalog = json.loads(listed)
                selected = catalog["packs"][0]["items"][0]["selection_id"]
            finally:
                runner._clear_session_env(tokens)
            tokens = runner._set_session_env(context)
            try:
                sent = await asyncio.to_thread(
                    registry.dispatch,
                    "matrix_image_packs",
                    {"action": "send", "selection_id": selected},
                )
                assert isinstance(sent, str)
                assert json.loads(sent) == {"success": True, "event_id": "$sent"}
            finally:
                runner._clear_session_env(tokens)
            assert client.api.send.await_args.args[2] == PACK["images"]["fox"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,room_error,truncated",
    [
        (
            "oversized",
            "Matrix room state exceeds the image-pack discovery budget",
            True,
        ),
        ("timeout", "room state read failed: TimeoutError", False),
    ],
)
async def test_room_state_failure_still_lists_bot_account_packs(
    tmp_path, failure, room_error, truncated
):
    scope = set_hermes_home_override(tmp_path)
    adapter, client = make_adapter(tmp_path)
    if failure == "oversized":
        client.get_state.return_value = client.get_state.return_value * (
            MAX_STATE_EVENTS + 1
        )
    if failure == "timeout":
        client.get_state.side_effect = TimeoutError()
    client.get_account_data.side_effect = lambda kind: (
        deepcopy(PACK) if kind == "im.ponies.user_emotes" else {}
    )
    tokens = set_session_vars(
        platform="matrix",
        chat_id=ROOM,
        user_id=USER,
        session_key="session",
        session_id="conversation",
        transport_adapter=adapter,
    )
    try:
        catalog = await adapter.matrix_image_packs("list", ROOM, requester=USER)
    finally:
        clear_session_vars(tokens)
        reset_hermes_home_override(scope)
    assert selections_replaced(catalog) == {
        "packs": [
            {
                "source": "bot_account",
                "event_type": "im.ponies.user_emotes",
                "room_id": None,
                "state_key": None,
                "account_user_id": BOT,
                "display_name": "/new same label",
                "description": None,
                "attribution": None,
                "items": [
                    {
                        "selection_id": "selected",
                        "shortcode": "fox",
                        **PACK["images"]["fox"],
                    }
                ],
            }
        ],
        "errors": [{"source": "room", "error": room_error}],
        "truncated": truncated,
        "untrusted_data": True,
        "account_user_id": BOT,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage,expected",
    [
        (["sticker", "org.example.reaction"], ([["fox"]], [])),
        (["org.example.reaction"], ([[]], [])),
        (
            "sticker",
            (
                [],
                [
                    {
                        "source": "room",
                        "event_type": "m.room.image_pack",
                        "room_id": ROOM,
                        "state_key": "",
                        "account_user_id": None,
                        "error": "pack usage is malformed",
                    }
                ],
            ),
        ),
    ],
)
async def test_unknown_pack_usage_values_are_ignored(tmp_path, usage, expected):
    scope = set_hermes_home_override(tmp_path)
    adapter, client = make_adapter(tmp_path)
    content = deepcopy(PACK)
    content["pack"]["usage"] = usage
    client.get_state.return_value = [
        StateEvent.deserialize({
            "type": "m.room.image_pack",
            "state_key": "",
            "room_id": ROOM,
            "sender": BOT,
            "event_id": "$pack",
            "origin_server_ts": 1000,
            "content": content,
        })
    ]
    tokens = set_session_vars(
        platform="matrix",
        chat_id=ROOM,
        user_id=USER,
        session_key="session",
        session_id="conversation",
        transport_adapter=adapter,
    )
    try:
        catalog = await adapter.matrix_image_packs("list", ROOM, requester=USER)
    finally:
        clear_session_vars(tokens)
        reset_hermes_home_override(scope)
    assert (
        [[item["shortcode"] for item in pack["items"]] for pack in catalog["packs"]],
        catalog["errors"],
    ) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("room_packs", [1, MAX_PACKS])
async def test_list_reads_each_account_source_once_until_the_catalog_is_full(
    tmp_path, room_packs
):
    scope = set_hermes_home_override(tmp_path)
    adapter, client = make_adapter(tmp_path)
    client.get_state.return_value = [
        StateEvent.deserialize({
            "type": "m.room.image_pack",
            "state_key": f"pack{index}",
            "room_id": ROOM,
            "sender": BOT,
            "event_id": f"$pack{index}",
            "origin_server_ts": 1000,
            "content": deepcopy(PACK),
        })
        for index in range(room_packs)
    ]
    references = {"rooms": {"!reference:example.org": {"a": {}, "b": {}, "c": {}}}}
    client.get_account_data.side_effect = lambda kind: (
        references if kind == "m.image_pack.rooms" else {}
    )
    tokens = set_session_vars(
        platform="matrix",
        chat_id=ROOM,
        user_id=USER,
        session_key="session",
        session_id="conversation",
        transport_adapter=adapter,
    )
    try:
        catalog = await adapter.matrix_image_packs("list", ROOM, requester=USER)
    finally:
        clear_session_vars(tokens)
        reset_hermes_home_override(scope)
    full = room_packs == MAX_PACKS
    account_reads = sorted(
        str(call.args[0]) for call in client.get_account_data.await_args_list
    )
    reference_reads = [
        (call.args[0], str(call.args[1]), call.args[2])
        for call in client.api.state.await_args_list
    ]
    assert (
        account_reads,
        reference_reads,
        len(catalog["packs"]),
        catalog["truncated"],
    ) == (
        []
        if full
        else ["im.ponies.emote_rooms", "im.ponies.user_emotes", "m.image_pack.rooms"],
        []
        if full
        else [
            ("!reference:example.org", "m.room.image_pack", key)
            for key in ("a", "b", "c")
        ],
        MAX_PACKS if full else room_packs + 3,
        full,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "create,content_only,power,expected",
    [
        pytest.param(
            {"sender": BOT, "content": {"room_version": "12"}},
            False,
            {"users": {USER: 100}, "events": {"m.sticker": 100}},
            SENT,
            id="v12-bot-creator",
        ),
        pytest.param(
            {"sender": USER, "content": {"room_version": "12"}},
            False,
            {"users": {}, "events": {"m.sticker": 100}},
            {"error": "Matrix bot cannot send m.sticker in this room"},
            id="v12-user-creator",
        ),
        pytest.param(
            {"content": {"room_version": "10", "creator": USER}},
            True,
            None,
            SENT,
            id="v10-content-only",
        ),
        pytest.param(
            {"content": {"room_version": "12"}},
            True,
            None,
            SENT,
            id="v12-content-only",
        ),
    ],
)
async def test_send_reads_the_create_event_for_creator_power(
    tmp_path, create, content_only, power, expected
):
    scope = set_hermes_home_override(tmp_path)
    adapter, client = make_adapter(tmp_path)
    create = {**CREATE, **create}
    packs = client.api.state.side_effect

    async def state_event(room_id, kind, state_key="", **kwargs):
        if kind == "m.room.create":
            return deepcopy(create["content"] if content_only else create)
        if kind == "m.room.power_levels" and power is not None:
            return deepcopy(power)
        return await packs(room_id, kind, state_key, **kwargs)

    client.api.state.side_effect = state_event
    tokens = set_session_vars(
        platform="matrix",
        chat_id=ROOM,
        user_id=USER,
        session_key="session",
        session_id="conversation",
        transport_adapter=adapter,
    )
    try:
        catalog = await adapter.matrix_image_packs("list", ROOM, requester=USER)
        result = await adapter.matrix_image_packs(
            "send",
            ROOM,
            requester=USER,
            selection_id=catalog["packs"][0]["items"][0]["selection_id"],
        )
    finally:
        clear_session_vars(tokens)
        reset_hermes_home_override(scope)
    assert result == expected
