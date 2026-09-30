"""Matrix pin changes use the owning room state and server permissions."""

import asyncio
import functools
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest

from plugins.platforms.matrix.room_inspection import change_matrix_pin

_change_pin = functools.partial(
    change_matrix_pin, interrupt_check=lambda: False, before_write=lambda: None,
)

class _MissingState(RuntimeError):
    errcode = "M_NOT_FOUND"


_ALICE_CAN_PIN = {"users": {"@alice:server": 50}}


def _adapter(pins, send, *, power=_ALICE_CAN_PIN, room_version="10", create=None):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    async def request(_method, path, *, query_params=None, **_kwargs):
        room_id, event_type = unquote(str(path)).removeprefix("/_matrix/client/v3/rooms/").rstrip("/").split("/state/")
        if event_type == "m.room.pinned_events":
            return await pins(room_id, event_type)
        if event_type == "m.room.power_levels" and power is not None:
            return power
        if event_type == "m.room.create" and query_params == {"format": "event"}:
            return create or {"type": "m.room.create", "sender": "@creator:server",
                              "content": {"room_version": room_version}}
        raise _MissingState()

    adapter = object.__new__(MatrixAdapter)
    vars(adapter).update(
        _pin_state_lock=asyncio.Lock(),
        _client=SimpleNamespace(api=SimpleNamespace(request=request), send_state_event=send),
        _joined_rooms={"!room:server"}, _allowed_room_ids=set(), _user_id="@bot:server",
        _is_allowed_matrix_room_event=AsyncMock(return_value=True),
        _is_dm_room=AsyncMock(return_value=False),
        _is_sender_authorized=lambda user, **kw: user != "@mallory:server",
    )
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("nested_content", [{}, {"pinned": ["$decoy"]}, None])
async def test_pin_round_trip_preserves_other_content_fields(nested_content):
    original = {"pinned": ["$first"], "content": nested_content,
                "org.example.pin_metadata": {"label": "Important"}}
    current = dict(original)
    writes = []

    async def state(room_id, event_type):
        return dict(current)

    async def send(room_id, event_type, content):
        writes.append(content)
        current.update(content)
        return "$state"

    adapter = _adapter(state, send)
    result = [
        await _change_pin(adapter, action, "!room:server", "$new", requester="@alice:server")
        for action in ("pin", "unpin")
    ]

    assert (result, writes) == (
        [{"pinned": ["$first", "$new"], "state_event_id": "$state"},
         {"pinned": ["$first"], "state_event_id": "$state"}],
        [{**original, "pinned": ["$first", "$new"]}, original],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("state", "action", "written", "pinned"), [
    pytest.param("missing", "pin", {"pinned": ["$new"]}, ["$new"], id="missing"),
    pytest.param({}, "pin", {"pinned": ["$new"]}, ["$new"], id="redacted"),
    pytest.param({"pinned": None, "content": {}}, "pin", {"pinned": ["$new"], "content": {}}, ["$new"],
                 id="null-list"),
    pytest.param({"content": {"pinned": ["$decoy"]}}, "pin",
                 {"content": {"pinned": ["$decoy"]}, "pinned": ["$new"]}, ["$new"], id="nested-content"),
    pytest.param({"pinned": [7, "invalid", "$", "$first"]}, "pin",
                 {"pinned": [7, "invalid", "$", "$first", "$new"]}, ["$first", "$new"],
                 id="malformed-entries-pin"),
    pytest.param({"pinned": [7, "$new"]}, "unpin", {"pinned": [7]}, [], id="malformed-entries-unpin"),
    pytest.param({"pinned": "$first"}, "pin", None, None, id="non-list"),
])
async def test_pin_state_is_read_as_matrix_clients_read_it(state, action, written, pinned):
    pins = AsyncMock(return_value=state)
    if state == "missing":
        pins.side_effect = _MissingState()
    send = AsyncMock(return_value="$state")
    adapter = _adapter(pins, send)

    result = await _change_pin(adapter, action, "!room:server", "$new", requester="@alice:server")

    expected = ({"error": "Matrix pinned events state is invalid"}, [])
    if written is not None:
        expected = ({"pinned": pinned, "state_event_id": "$state"},
                    [(("!room:server", "m.room.pinned_events", written),)])
    assert (result, send.await_args_list) == expected


@pytest.mark.asyncio
async def test_pin_and_unpin_preserve_other_events_and_reject_unauthorized_requester():
    pinned = ["$first", "$second"]

    async def state(room_id, event_type):
        assert (room_id, event_type) == ("!room:server", "m.room.pinned_events")
        return {"pinned": pinned.copy()}

    async def send(room_id, event_type, content):
        assert (room_id, str(event_type)) == ("!room:server", "m.room.pinned_events")
        pinned[:] = content["pinned"]
        return "$state"

    adapter = _adapter(state, AsyncMock(side_effect=send))
    client = adapter._client

    result = [
        await _change_pin(adapter, "pin", "!room:server", "$third", requester="@alice:server"),
        await _change_pin(adapter, "pin", "!room:server", "$third", requester="@alice:server"),
        await _change_pin(adapter, "unpin", "!room:server", "$second", requester="@alice:server"),
        await _change_pin(adapter, "pin", "!room:server", "$fourth", requester="@mallory:server"),
    ]

    assert result == [
        {"pinned": ["$first", "$second", "$third"], "state_event_id": "$state"},
        {"pinned": ["$first", "$second", "$third"], "unchanged": True},
        {"pinned": ["$first", "$third"], "state_event_id": "$state"},
        {"error": "Matrix requester is not authorized for this room"},
    ]
    assert client.send_state_event.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(("requester", "power", "room_version", "refusal"), [
    ("@member:server", {"users": {"@bot:server": 100}}, "10", {"required": 50, "level": 0}),
    ("@member:server", {"users_default": 10, "events": {"m.room.pinned_events": 20}}, "10",
     {"required": 20, "level": 10}),
    ("@member:server", {"users": {"@member:server": 50}}, "10", None),
    ("@member:server", None, "10", None),
    ("@creator:server", {"users": {"@bot:server": 100}}, "12", None),
    # A server that ignores format=event returns only the content, which names no
    # creator from version 11, so without power levels the level is unknown.
    ("@member:server", None, "11-content-only", {"required": 0, "level": None}),
], ids=["state-default", "explicit-levels", "member-has-power", "no-power-levels", "v12-creator", "unknown-level"])
async def test_requester_needs_room_power_to_change_pins(requester, power, room_version, refusal):
    send = AsyncMock(return_value="$state")
    adapter = _adapter(
        AsyncMock(return_value={"pinned": ["$admin_pin"]}), send, power=power, room_version=room_version,
        create={"room_version": "11"} if room_version == "11-content-only" else None,
    )

    result = await _change_pin(adapter, "unpin", "!room:server", "$admin_pin", requester=requester)

    if refusal is None:
        assert (result, send.await_args_list) == (
            {"pinned": [], "state_event_id": "$state"},
            [(("!room:server", "m.room.pinned_events", {"pinned": []}),)],
        )
        return
    assert (result, send.await_count) == (
        {"error": "Matrix requester lacks permission to change pins", **refusal}, 0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("change", "refusal"), [
    ("requester", {"error": "Matrix requester is not authorized for this room"}),
    ("room", {"error": "Matrix room is not allowed or joined"}),
    ("client", {"error": "Matrix room inspection context changed"}),
])
async def test_pin_change_rechecks_access_after_reading_pins(change, refusal):
    send = AsyncMock(return_value="$state")

    async def pins(room_id, event_type):
        if change == "requester":
            adapter._is_sender_authorized = lambda user, **kw: False
        if change == "room":
            adapter._joined_rooms = set()
        if change == "client":
            adapter._client = SimpleNamespace(api=adapter._client.api, send_state_event=send)
        return {"pinned": []}

    adapter = _adapter(pins, send)

    result = await _change_pin(adapter, "pin", "!room:server", "$new", requester="@alice:server")

    assert (result, send.await_count) == (refusal, 0)


@pytest.mark.asyncio
async def test_pin_surfaces_server_permission_error():
    class Forbidden(RuntimeError):
        errcode = "M_FORBIDDEN"

    error = Forbidden("You don't have permission to post this state event")
    adapter = _adapter(AsyncMock(return_value={"pinned": []}), AsyncMock(side_effect=error))

    result = await _change_pin(adapter, "pin", "!room:server", "$event", requester="@alice:server")

    assert result == {"error": "Matrix pin update was rejected", "errcode": "M_FORBIDDEN",
                      "message": "You don't have permission to post this state event"}


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [asyncio.TimeoutError(), ConnectionResetError()], ids=["write-timeout", "transport"])
async def test_failure_after_the_write_was_sent_reports_an_unknown_outcome(error):
    dispatched = []
    adapter = _adapter(AsyncMock(return_value={"pinned": []}), AsyncMock(side_effect=error))

    result = await change_matrix_pin(
        adapter, "pin", "!room:server", "$event", requester="@alice:server",
        interrupt_check=lambda: False, before_write=lambda: dispatched.append(True),
    )

    assert (result, dispatched) == ({
        "error": "Matrix pin update failed after the change was sent to the homeserver",
        "outcome": "unknown",
        "next_step": "Read the current pins with matrix_read kind=pins before retrying",
    }, [True])


@pytest.mark.asyncio
async def test_parallel_pin_calls_keep_both_events():
    pinned = []

    async def state(room_id, event_type):
        return {"pinned": pinned.copy()}

    async def send(room_id, event_type, content):
        await asyncio.sleep(0)
        pinned[:] = content["pinned"]
        return "$state"

    adapter = _adapter(state, send)

    await asyncio.gather(*(
        _change_pin(adapter, "pin", "!room:server", event, requester="@alice:server")
        for event in ("$first", "$second")
    ))

    assert pinned == ["$first", "$second"]
