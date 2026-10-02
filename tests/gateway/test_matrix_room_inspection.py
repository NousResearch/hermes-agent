"""Matrix room inspection reads current state through the receiving adapter."""

import asyncio
import gc
import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest

from plugins.platforms.matrix.adapter import MatrixAdapter
from plugins.platforms.matrix.room_inspection import inspect_matrix_room
from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache


class _NotFound(Exception):
    errcode = "M_NOT_FOUND"


def _state_type(path: Any) -> str | None:
    parts = unquote(str(path)).rstrip("/").split("/state/")
    return parts[1] if len(parts) == 2 else None


def _state_api(state: dict[str, Any]) -> SimpleNamespace:
    async def request(_method, path, *, query_params=None, **_kwargs):
        event_type = _state_type(path)
        if event_type == "m.room.create":
            assert query_params == {"format": "event"}
        if event_type not in state:
            raise _NotFound()
        return state[event_type]

    return SimpleNamespace(request=AsyncMock(side_effect=request))


def _create_event(version: str, sender: str = "@alice:server", **content: Any) -> dict[str, Any]:
    return {
        "type": "m.room.create", "state_key": "", "sender": sender, "event_id": "$create",
        "room_id": "!room:server", "origin_server_ts": 1,
        "content": {"room_version": version, **content},
    }


def _inspection_adapter(**attributes: Any) -> MatrixAdapter:
    adapter = object.__new__(MatrixAdapter)
    adapter._allowed_room_ids = set()
    vars(adapter).update(attributes)
    return adapter


@pytest.mark.asyncio
async def test_room_inspection_reports_state_members_permissions_and_pins():
    state = {
        "m.room.name": {"name": "Planning"},
        "m.room.topic": {"topic": "Release notes"},
        "m.room.canonical_alias": {"alias": "#planning:server"},
        "m.room.join_rules": {"join_rule": "invite"},
        "m.room.history_visibility": {"history_visibility": "shared"},
        "m.room.encryption": {"algorithm": "m.megolm.v1.aes-sha2"},
        "m.room.power_levels": {
            "users": {"@bot:server": 50, "@alice:server": 100},
            "users_default": 0,
            "state_default": 50,
            "events": {"m.room.pinned_events": 75},
            "invite": 50,
        },
        "m.room.create": {"creator": "@alice:server", "room_version": "10"},
        "m.room.pinned_events": {"pinned": ["$first", "$second"]},
    }

    async def get_event(room_id, event_id):
        return {"event_id": event_id, "sender": "@alice:server", "type": "m.room.message",
                "content": {"msgtype": "m.text", "body": f"Pinned {event_id}"}}

    async def request(_method, path, **kwargs):
        if _state_type(path) is not None:
            return await _state_api(state).request(_method, path, **kwargs)
        return await get_event("!room:server", "$first")

    client = SimpleNamespace(
        get_joined_members=AsyncMock(return_value={
            "@bot:server": SimpleNamespace(displayname="Hermes", avatar_url=None),
            "@alice:server": SimpleNamespace(displayname="Alice", avatar_url="mxc://server/alice"),
        }),
        api=SimpleNamespace(request=AsyncMock(side_effect=request)),
        get_event=AsyncMock(side_effect=get_event), crypto=None,
    )
    adapter = _inspection_adapter(
        _event_context_cache=MatrixEventContextCache(),
        _client=client, _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=True),
        _is_sender_authorized=lambda user, **kw: user == "@alice:server",
    )

    result = {
        kind: await inspect_matrix_room(adapter, kind, "!room:server", 1, requester="@alice:server")
        for kind in ("state", "members", "permissions", "pins")
    }

    assert result == {
        "state": {"room_id": "!room:server", "name": "Planning", "topic": "Release notes",
                  "canonical_alias": "#planning:server", "join_rule": "invite",
                  "history_visibility": "shared", "encryption": "m.megolm.v1.aes-sha2"},
        "members": {"members": [{"user_id": "@alice:server", "display_name": "Alice",
                                 "avatar_url": "mxc://server/alice"}], "total": 2, "truncated": True},
        "permissions": {"requester": {"user_id": "@alice:server", "level": 100,
                                      "creator_override": False},
                        "bot": {"user_id": "@bot:server", "level": 50,
                                "creator_override": False},
                        "required": {"send_message": 0, "send_event_type": "m.room.encrypted",
                                     "edit_pins": 75, "invite": 50,
                                     "kick": 50, "ban": 50, "redact_other": 50, "send_redaction": 0},
                        "bot_can_edit_pins": False},
        "pins": {"events": [{"event_id": "$first", "sender": "@alice:server",
                             "body": "Pinned $first", "msgtype": "m.text", "thread_id": None,
                             "timestamp": None, "sender_authorized": True}],
                 "total": 2, "truncated": True, "errors": []},
    }


@pytest.mark.asyncio
async def test_room_inspection_rejects_unauthorized_requester_before_network():
    client = SimpleNamespace(api=SimpleNamespace(request=AsyncMock()), get_joined_members=AsyncMock())
    adapter = _inspection_adapter(
        _client=client, _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: False,
    )

    result = await inspect_matrix_room(adapter, "state", "!room:server", 20,
                                       requester="@alice:server")

    assert result == {"error": "Matrix requester is not authorized for this room"}
    client.api.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_room_permissions_include_version_12_creator_override():
    api = _state_api({
        "m.room.create": _create_event("12", "@bot:server"),
        "m.room.encryption": {"algorithm": "m.megolm.v1.aes-sha2"},
        "m.room.power_levels": {"users_default": 0, "state_default": 50,
                                "events": {"m.room.pinned_events": 75, "m.room.encrypted": 25}},
    })
    adapter = _inspection_adapter(
        _client=SimpleNamespace(api=api),
        _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: True,
    )

    result = await inspect_matrix_room(adapter, "permissions", "!room:server", 20,
                                       requester="@alice:server")

    assert result == {
        "requester": {"user_id": "@alice:server", "level": 0, "creator_override": False},
        "bot": {"user_id": "@bot:server", "level": 0, "creator_override": True},
        "required": {"send_message": 25, "send_event_type": "m.room.encrypted",
                     "edit_pins": 75, "invite": 0,
                     "kick": 50, "ban": 50, "redact_other": 50, "send_redaction": 0},
        "bot_can_edit_pins": True,
    }


@pytest.mark.asyncio
async def test_old_room_power_levels_accept_numeric_strings():
    power = {
        "users": {"@bot:server": " +100 ", "@alice:server": "-5"},
        "events": {"m.room.pinned_events": " 075 ", "m.room.message": "+10"},
        "invite": " 25 ",
    }
    api = _state_api({"m.room.create": _create_event("9"), "m.room.power_levels": power})
    adapter = _inspection_adapter(
        _client=SimpleNamespace(api=api),
        _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: True,
    )

    result = await inspect_matrix_room(
        adapter, "permissions", "!room:server", 20, requester="@alice:server"
    )

    assert result == {
        "requester": {"user_id": "@alice:server", "level": -5, "creator_override": False},
        "bot": {"user_id": "@bot:server", "level": 100, "creator_override": False},
        "required": {"send_message": 10, "send_event_type": "m.room.message",
                     "edit_pins": 75, "invite": 25,
                     "kick": 50, "ban": 50, "redact_other": 50, "send_redaction": 0},
        "bot_can_edit_pins": True,
    }


@pytest.mark.asyncio
async def test_version_12_ignores_legacy_creator_property():
    api = _state_api({
        "m.room.create": _create_event("12", creator="@bot:server"),
        "m.room.power_levels": {"users_default": 0, "events": {"m.room.pinned_events": 75}},
    })
    adapter = _inspection_adapter(
        _client=SimpleNamespace(api=api),
        _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: True,
    )

    result = await inspect_matrix_room(
        adapter, "permissions", "!room:server", 20, requester="@alice:server"
    )

    assert result == {
        "requester": {"user_id": "@alice:server", "level": 0, "creator_override": True},
        "bot": {"user_id": "@bot:server", "level": 0, "creator_override": False},
        "required": {"send_message": 0, "send_event_type": "m.room.message",
                     "edit_pins": 75, "invite": 0,
                     "kick": 50, "ban": 50, "redact_other": 50, "send_redaction": 0},
        "bot_can_edit_pins": False,
    }


def _permissions(requester: tuple[int | None, bool | None], bot: tuple[int | None, bool | None],
                 edit_pins: int, can_edit_pins: bool | None) -> dict[str, Any]:
    return {
        "requester": {"user_id": "@alice:server", "level": requester[0], "creator_override": requester[1]},
        "bot": {"user_id": "@bot:server", "level": bot[0], "creator_override": bot[1]},
        "required": {"send_message": 0, "send_event_type": "m.room.message", "edit_pins": edit_pins,
                     "invite": 0, "kick": 50, "ban": 50, "redact_other": 50, "send_redaction": 0},
        "bot_can_edit_pins": can_edit_pins,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("create", "power", "expected"), [
    pytest.param(_create_event("10"), {"users": {"@alice:server": 100}},
                 _permissions((100, False), (0, False), 50, False), id="power-levels-omit-defaults"),
    pytest.param(_create_event("10"), None,
                 _permissions((100, False), (0, False), 0, True), id="no-power-levels"),
    pytest.param({"room_version": "10", "creator": "@alice:server"}, None,
                 _permissions((100, False), (0, False), 0, True), id="content-only-create-v10"),
    pytest.param({"room_version": "11"}, {"users": {"@alice:server": 100}},
                 _permissions((100, False), (0, False), 50, False), id="content-only-create-v11"),
    pytest.param({"room_version": "11"}, None,
                 _permissions((None, False), (None, False), 0, None),
                 id="content-only-create-v11-no-power-levels"),
    pytest.param({"room_version": "12"}, {"users": {"@bot:server": 100}},
                 _permissions((0, None), (100, None), 50, True), id="content-only-create-v12"),
])
async def test_permissions_apply_spec_defaults_to_raw_server_state(
    create: dict[str, Any], power: dict[str, Any] | None, expected: dict[str, Any],
):
    pytest.importorskip("mautrix")
    from mautrix.client.api import ClientAPI
    from mautrix.errors import MNotFound

    async def request(_method, path, *_args, query_params=None, **_kwargs):
        event_type = unquote(str(path)).rstrip("/").rsplit("/", 1)[-1]
        if event_type == "m.room.create":
            full_event = "content" in create and query_params == {"format": "event"}
            return create if full_event else create.get("content", create)
        if event_type == "m.room.power_levels" and power is not None:
            return power
        raise MNotFound(404, "no state")

    client = ClientAPI("@bot:server", api=SimpleNamespace(request=request, log=None))
    adapter = _inspection_adapter(
        _client=client, _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: True,
    )

    result = await inspect_matrix_room(
        adapter, "permissions", "!room:server", 20, requester="@alice:server"
    )

    assert result == expected


@pytest.mark.asyncio
async def test_cancelled_inspection_stays_cancelled_when_final_admission_fails():
    started = asyncio.Event()

    async def get_joined_members(_room):
        started.set()
        await asyncio.Event().wait()

    adapter = _inspection_adapter(
        _client=SimpleNamespace(get_joined_members=get_joined_members), _closing=False,
        _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: True,
    )
    pending = asyncio.create_task(
        inspect_matrix_room(adapter, "members", "!room:server", 20, requester="@alice:server")
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    adapter._closing = True
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [
    "plain", "edited", "redacted", "encrypted", "encrypted-edited", "encrypted-redacted",
    "missing-key", "withdrawn-original", "withdrawn-edit",
])
async def test_pin_snapshots_expose_effective_state_after_sibling_await(state: str):
    mautrix_types = pytest.importorskip("mautrix.types")
    room, sender = "!room:server", "@alice:server"
    content = {"msgtype": "m.text", "body": "Original plan"}
    raw = {
        "room_id": room, "event_id": "$target", "sender": sender, "origin_server_ts": 1,
        "type": "m.room.message", "content": content,
    }
    encrypted = state.startswith("encrypted") or state == "missing-key"
    cipher = {
        "algorithm": "m.megolm.v1.aes-sha2", "ciphertext": "original", "session_id": "session",
        "sender_key": "key", "device_id": "device",
    }
    if encrypted:
        raw.update(type="m.room.encrypted", content=cipher)
    edit_content = {
        "msgtype": "m.text", "body": "* Revised plan",
        "m.new_content": {"msgtype": "m.text", "body": "Revised plan"},
        "m.relates_to": {"rel_type": "m.replace", "event_id": "$target"},
    }
    if state in {"edited", "encrypted-edited", "withdrawn-edit"}:
        replacement = {**raw, "event_id": "$edit", "content": edit_content}
        if encrypted:
            replacement["content"] = {**cipher, "ciphertext": "replacement",
                                      "m.relates_to": edit_content["m.relates_to"]}
        raw["unsigned"] = {"m.relations": {"m.replace": replacement}}
    if state in {"redacted", "encrypted-redacted"}:
        raw["unsigned"] = {"redacted_because": {"event_id": "$redaction"}}
    gate = {**raw, "event_id": "$gate", "type": "m.room.message",
            "content": {"msgtype": "m.text", "body": "Gate plan"}}
    gate.pop("unsigned", None)
    started, release = asyncio.Event(), asyncio.Event()

    async def request(_method, path, **_kwargs):
        if _state_type(path) == "m.room.pinned_events":
            return {"pinned": ["$target", "$gate"]}
        event_id = "$gate" if path.endswith("%24gate") else "$target"
        if event_id == "$gate":
            started.set()
            await release.wait()
            return gate
        return raw

    class SessionNotFound(Exception):
        pass

    async def decrypt(event):
        if state == "missing-key":
            raise SessionNotFound()
        clear = edit_content if event.event_id == "$edit" else content
        return mautrix_types.Event.deserialize({
            **raw, "event_id": str(event.event_id), "type": "m.room.message",
            "content": json.loads(json.dumps(clear)),
        })

    plaintext = json.dumps({"room_id": room, "type": "m.room.message", "content": edit_content})
    store = SimpleNamespace(get_group_session=AsyncMock(return_value=SimpleNamespace(
        decrypt=lambda _ciphertext: (plaintext, 0),
    )))
    async def get_event(_room, event_id):
        return await request(None, "/event/" + event_id.replace("$", "%24"))

    client = SimpleNamespace(
        api=SimpleNamespace(request=request),
        get_event=AsyncMock(side_effect=get_event),
        crypto=SimpleNamespace(decrypt_megolm_event=decrypt, crypto_store=store),
    )
    cache = MatrixEventContextCache(max_entries=1)
    adapter = _inspection_adapter(
        _client=client, _event_context_cache=cache, _joined_rooms={room}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True), _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda *_args, **_kwargs: True,
    )
    pending = asyncio.create_task(inspect_matrix_room(adapter, "pins", room, 2, requester=sender))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        withdrawals = {"withdrawn-original": "$target", "withdrawn-edit": "$edit"}
        if state in withdrawals:
            cache.redact(room, withdrawals[state])
            cache.store(room, "$pressure", MatrixEventContext(sender, "Pressure"))
            gc.collect()
    finally:
        release.set()
    result = await asyncio.wait_for(pending, timeout=2)
    target = {
        "event_id": "$target", "sender": sender, "body": "Original plan", "msgtype": "m.text",
        "thread_id": None, "timestamp": 1, "sender_authorized": True,
    }
    if state in {"edited", "encrypted-edited"}:
        target.update(body="Revised plan", edited=True)
    if state in {"redacted", "encrypted-redacted", "withdrawn-original"}:
        target.update(body="[redacted]", msgtype=None, redacted=True)
    if state == "withdrawn-edit":
        target.update(body="[event content unavailable]", msgtype=None)
    errors = {
        "missing-key": [{"event_id": "$target", "error": "missing decryption keys"}],
        "withdrawn-edit": [{"event_id": "$target", "error": "replacement was redacted"}],
    }.get(state, [])
    assert result == {
        "events": ([] if state == "missing-key" else [target]) + [{
            "event_id": "$gate", "sender": sender, "body": "Gate plan", "msgtype": "m.text",
            "thread_id": None, "timestamp": 1, "sender_authorized": True,
        }],
        "total": 2, "truncated": False, "errors": errors,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "change"), [
    (kind, change)
    for kind in ("state", "members", "permissions", "pins")
    for change in (
        "unchanged", "closing", "client", "account", "api", "api-token", "http-session",
        "room", "policy", "chat-type",
    )
] + [("state", "policy-last-missing-state")])
async def test_inspection_rechecks_owning_client_and_policy_after_await(
    tmp_path, monkeypatch, kind: str, change: str,
):
    from agent import secret_scope
    from gateway.run import _profile_runtime_scope
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.matrix_read_tool import _matrix_read

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    room, user = "!room:server", "@alice:server"
    started, release = asyncio.Event(), asyncio.Event()
    waiting = True
    mutation = lambda: None

    def make_adapter(label):
        async def network_barrier(event_type=None):
            nonlocal waiting
            final_missing = change == "policy-last-missing-state"
            if label == "A" and waiting and (not final_missing or event_type == "m.room.encryption"):
                waiting = False
                started.set()
                await release.wait()
                mutation()

        state = _state_api({
            "m.room.name": {"name": label},
            "m.room.create": _create_event("10", user),
            "m.room.pinned_events": {"pinned": ["$pin"]},
        })

        async def members(_room):
            await network_barrier()
            return {user: {"displayname": label}}

        async def request(_method, path, **kwargs):
            event_type = _state_type(path)
            if event_type is not None:
                await network_barrier(event_type)
                return await state.request(_method, path, **kwargs)
            return {
                "room_id": room, "event_id": "$pin", "sender": user,
                "type": "m.room.message", "content": {"msgtype": "m.text", "body": label},
            }

        client = SimpleNamespace(
            api=SimpleNamespace(request=request, base_url="https://server", token=label, session=object()),
            mxid=f"@bot-{label}:server", device_id=f"device-{label}", crypto=None,
            get_joined_members=members, get_event=AsyncMock(return_value=awaitable_pin(label)),
        )
        adapter = _inspection_adapter(
            _client=client, _event_context_cache=MatrixEventContextCache(), _joined_rooms={room},
            _user_id=client.mxid, _closing=False, _allowed=True, _dm=False,
            _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        )
        adapter._is_dm_room = AsyncMock(side_effect=lambda _room: adapter._dm)
        adapter._is_sender_authorized = lambda actor, **_kw: actor == user and adapter._allowed
        adapter.inspect_matrix_room = lambda *args, **kwargs: inspect_matrix_room(adapter, *args, **kwargs)
        return adapter

    def awaitable_pin(label):
        return {"room_id": room, "event_id": "$pin", "sender": user,
                "type": "m.room.message", "content": {"msgtype": "m.text", "body": label}}

    first, second = make_adapter("A"), make_adapter("B")
    first_client = first._client
    first_api = first_client.api
    first_http_session = first_api.session
    home_a, home_b = tmp_path / "A", tmp_path / "B"
    home_a.mkdir()
    home_b.mkdir()

    async def run(adapter, home):
        with _profile_runtime_scope(home, {}):
            tokens = set_session_vars(
                platform="matrix", chat_id=room, user_id=user, profile=home.name,
                session_key=f"matrix-{home.name}", transport_adapter=adapter,
                transport_loop=asyncio.get_running_loop(),
            )
            try:
                return json.loads(await _matrix_read({"kind": kind}))
            finally:
                clear_session_vars(tokens)

    changes = {
        "unchanged": lambda: None,
        "closing": lambda: setattr(first, "_closing", True),
        "client": lambda: setattr(first, "_client", second._client),
        "account": lambda: setattr(first_client, "mxid", "@different:server"),
        "api": lambda: setattr(first_client, "api", second._client.api),
        "api-token": lambda: setattr(first_client.api, "token", "changed"),
        "http-session": lambda: setattr(first_client.api, "session", object()),
        "room": lambda: first._joined_rooms.clear(),
        "policy": lambda: setattr(first, "_allowed", False),
        "policy-last-missing-state": lambda: setattr(first, "_allowed", False),
        "chat-type": lambda: setattr(first, "_dm", True),
    }
    mutation = changes[change]
    pending = asyncio.create_task(run(first, home_a))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        result_b = await asyncio.wait_for(run(second, home_b), timeout=2)
    finally:
        release.set()
    result_a = await asyncio.wait_for(pending, timeout=2)
    first._client = first_client
    first._closing = False
    first._allowed = True
    first._dm = False
    first._joined_rooms.add(room)
    first_client.mxid = "@bot-A:server"
    first_client.api = first_api
    first_api.token = "A"
    first_api.session = first_http_session
    again_a = await asyncio.wait_for(run(first, home_a), timeout=2)

    def expected(label):
        return {
            "state": {"room_id": room, "name": label, "topic": None, "canonical_alias": None,
                      "join_rule": None, "history_visibility": None, "encryption": None},
            "members": {"members": [{"user_id": user, "display_name": label, "avatar_url": None}],
                        "total": 1, "truncated": False},
            "permissions": {
                "requester": {"user_id": user, "level": 100, "creator_override": False},
                "bot": {"user_id": f"@bot-{label}:server", "level": 0, "creator_override": False},
                "required": {"send_message": 0, "send_event_type": "m.room.message", "edit_pins": 0,
                             "invite": 0, "kick": 50, "ban": 50, "redact_other": 50, "send_redaction": 0},
                "bot_can_edit_pins": True,
            },
            "pins": {"events": [{"event_id": "$pin", "sender": user, "body": label,
                                 "msgtype": "m.text", "thread_id": None, "timestamp": None,
                                 "sender_authorized": True}],
                     "total": 1, "truncated": False, "errors": []},
        }[kind]

    refusals = {
        "closing": {"error": "Matrix client is disconnected"},
        "room": {"error": "Matrix room is not allowed or joined"},
        "policy": {"error": "Matrix requester is not authorized for this room"},
        "policy-last-missing-state": {"error": "Matrix requester is not authorized for this room"},
    }
    expected_a = expected("A") if change == "unchanged" else refusals.get(change, {
        "error": "Matrix room inspection context changed",
    })
    assert (result_a, result_b, again_a) == (expected_a, expected("B"), expected("A"))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["members", "missing-state"])
@pytest.mark.parametrize(("chat_type", "change"), [
    ("group", "unchanged"), ("group", "joined"), ("group", "allowlist"),
    ("dm", "allowlist"), ("group", "sender"), ("group", "closing"),
])
async def test_final_admission_uses_current_matrix_policy_after_identity_await(
    kind: str, chat_type: str, change: str,
):
    from tests.gateway.test_matrix import _make_adapter

    room, requester = "!room:server", "@alice:server"
    adapter = _make_adapter()
    adapter._joined_rooms = {room}
    adapter._allowed_room_ids = {room}
    sender_allowed = True
    network_complete = False
    paused = False
    started, release = asyncio.Event(), asyncio.Event()

    async def identity(_room):
        nonlocal paused
        if network_complete and not paused:
            paused = True
            started.set()
            await release.wait()
        return chat_type == "dm"

    async def members(_room):
        nonlocal network_complete
        network_complete = True
        return {requester: {"displayname": "Private profile"}}

    state = _state_api({"m.room.name": {"name": "Private planning"}})

    async def request(_method, path, **kwargs):
        nonlocal network_complete
        if _state_type(path) == "m.room.encryption":
            network_complete = True
        return await state.request(_method, path, **kwargs)

    adapter._client = SimpleNamespace(get_joined_members=members, api=SimpleNamespace(request=request))
    adapter._is_dm_room = identity
    adapter._is_sender_authorized = lambda *_args, **_kwargs: sender_allowed
    pending = asyncio.create_task(inspect_matrix_room(
        adapter, "state" if kind == "missing-state" else kind, room, 20, requester=requester,
    ))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        if change == "joined":
            adapter._joined_rooms.clear()
        if change == "allowlist":
            adapter._allowed_room_ids = {"!other:server"}
        if change == "sender":
            sender_allowed = False
        if change == "closing":
            adapter._closing = True
    finally:
        release.set()
    result = await asyncio.wait_for(pending, timeout=2)
    accepted = {
        "members": {"members": [{"user_id": requester, "display_name": "Private profile",
                                 "avatar_url": None}], "total": 1, "truncated": False},
        "missing-state": {"room_id": room, "name": "Private planning", "topic": None,
                          "canonical_alias": None, "join_rule": None,
                          "history_visibility": None, "encryption": None},
    }[kind]
    expected = {
        "joined": {"error": "Matrix room is not allowed or joined"},
        "sender": {"error": "Matrix requester is not authorized for this room"},
        "closing": {"error": "Matrix client is disconnected"},
    }.get(change, accepted)
    if change == "allowlist" and chat_type != "dm":
        expected = {"error": "Matrix room is not allowed or joined"}
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize("read", ["m.room.encryption", "m.room.create"])
async def test_permission_inspection_uses_power_after_other_state_reads(encrypted, read):
    power = {"users": {"@alice:server": 50, "@bot:server": 100}}
    state = {"m.room.create": _create_event("10"), "m.room.power_levels": power}
    if encrypted:
        state["m.room.encryption"] = {"algorithm": "m.megolm.v1.aes-sha2"}
    api = _state_api(state)
    request = api.request

    async def changing_state(method, path, **kwargs):
        if _state_type(path) == read:
            power["users"]["@alice:server"] = 0
            power["events"] = {
                "m.room.pinned_events": 100, "m.room.message": 7, "m.room.encrypted": 9,
            }
        return deepcopy(await request(method, path, **kwargs))

    api.request = changing_state
    adapter = _inspection_adapter(
        _client=SimpleNamespace(api=api), _joined_rooms={"!room:server"}, _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda *_args, **_kwargs: True,
    )
    result = await inspect_matrix_room(adapter, "permissions", "!room:server", 20,
                                       requester="@alice:server")
    expected = _permissions((0, False), (100, False), 100, True)
    expected["required"].update(
        send_message=9 if encrypted else 7,
        send_event_type="m.room.encrypted" if encrypted else "m.room.message",
    )

    assert result == expected
