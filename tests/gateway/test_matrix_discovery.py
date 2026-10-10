"""Discovery uses joined membership, admission policy and bounded live reads."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.matrix import discovery
from plugins.platforms.matrix.adapter import MatrixAdapter, MNotFound


def _summary(room, room_type=None, name=None):
    return {
        "room_id": room,
        "room_type": room_type,
        "name": name,
        "topic": None,
        "canonical_alias": None,
    }


def _created(adapter):
    return [
        call.args[0]
        for call in adapter._client.get_state_event.await_args_list
        if call.args[1] == "m.room.create"
    ]


@pytest.fixture
def members():
    return {
        "!origin:server": {"@bot:server", "@alice:server", "@bob:server"},
        "!room:server": {"@bot:server", "@alice:server", "@carol:server"},
        "!space:server": {"@bot:server", "@alice:server", "@carol:server"},
    }


@pytest.fixture
def adapter(members):
    async def joined_members(room):
        users = members.get(str(room), {"@bot:server", "@alice:server", "@zed:server"})
        return {user: {} for user in users}

    client = SimpleNamespace(
        get_state_event=AsyncMock(return_value={}),
        get_joined_members=AsyncMock(side_effect=joined_members),
        api=SimpleNamespace(request=AsyncMock()),
        crypto=object(),
    )
    instance = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"homeserver": "https://server", "user_id": "@bot:server"},
        )
    )
    instance._client = client
    instance._joined_rooms = set(members)
    client.get_joined_rooms = AsyncMock(
        side_effect=lambda: sorted(instance._joined_rooms)
    )
    instance._user_id = "@bot:server"
    instance.set_authorization_check(
        lambda user, chat_type, chat_id: user == "@alice:server"
    )
    return instance


@pytest.fixture
def expire_discovery(monkeypatch):
    """Expire the discovery deadline from inside a pending request."""
    scopes = []
    timeout_factory = asyncio.timeout

    def timeout(seconds):
        scope = timeout_factory(seconds)
        scopes.append(scope)
        return scope

    monkeypatch.setattr(discovery.asyncio, "timeout", timeout)

    async def expire():
        scopes[0].reschedule(asyncio.get_running_loop().time())
        await asyncio.Event().wait()

    return expire


@pytest.mark.asyncio
async def test_joined_discovery_classifies_spaces_without_traversing_or_joining(
    adapter,
):
    async def state(room, event_type):
        content = {
            "m.room.create": {
                "type": "m.space" if room == "!space:server" else "org.example.room"
            },
            "m.room.name": {"name": room},
            "m.room.topic": {"topic": "t" * 1300},
            "m.room.canonical_alias": {"alias": "#plan:server"},
        }.get(event_type)
        if content is None:
            raise MNotFound(404, "Event not found.")
        return content

    adapter._client.get_state_event.side_effect = state
    rooms = await adapter.discover_matrix(
        "joined_rooms", "!origin:server", 20, requester="@alice:server"
    )
    spaces = await adapter.discover_matrix(
        "joined_spaces", "!origin:server", 20, requester="@alice:server"
    )

    def summary(room, room_type):
        return {
            "room_id": room,
            "room_type": room_type,
            "name": room,
            "topic": "t" * 1200,
            "canonical_alias": "#plan:server",
        }

    assert (rooms, spaces) == (
        {
            "rooms": [
                summary("!origin:server", "org.example.room"),
                summary("!room:server", "org.example.room"),
            ],
            "truncated": False,
            "errors": [],
        },
        {
            "rooms": [summary("!space:server", "m.space")],
            "truncated": False,
            "errors": [],
        },
    )
    assert {
        call.args[1] for call in adapter._client.get_state_event.await_args_list
    } == {
        "m.room.create",
        "m.room.name",
        "m.room.topic",
        "m.room.canonical_alias",
        "m.room.join_rules",
        "m.room.history_visibility",
        "m.room.encryption",
        "m.room.tombstone",
    }
    adapter._client.api.request.assert_not_awaited()
    assert adapter._joined_rooms == {"!origin:server", "!room:server", "!space:server"}


@pytest.mark.asyncio
async def test_discovery_lists_only_rooms_and_spaces_the_requester_has_joined(
    adapter, members
):
    members.update({
        "!bobdm:server": {"@bot:server", "@bob:server"},
        "!hr:server": {"@bot:server", "@bob:server", "@carol:server"},
        "!hrspace:server": {"@bot:server", "@bob:server", "@carol:server"},
    })
    adapter._joined_rooms = set(members)
    adapter.set_authorization_check(
        lambda user, chat_type, chat_id: user in {"@alice:server", "@bob:server"}
    )
    adapter._client.get_state_event.side_effect = lambda room, event_type: (
        {"type": "m.space"}
        if event_type == "m.room.create" and "space" in room
        else {}
    )

    rooms = await adapter.discover_matrix(
        "joined_rooms", "!origin:server", 20, requester="@alice:server"
    )
    spaces = await adapter.discover_matrix(
        "joined_spaces", "!origin:server", 20, requester="@alice:server"
    )

    assert (rooms, spaces) == (
        {
            "rooms": [_summary("!origin:server"), _summary("!room:server")],
            "truncated": False,
            "errors": [],
        },
        {
            "rooms": [_summary("!space:server", "m.space")],
            "truncated": False,
            "errors": [],
        },
    )
    assert not {"!bobdm:server", "!hr:server", "!hrspace:server"} & set(
        _created(adapter)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,admission", [("joined_rooms", False), ("joined_spaces", None), ("users", 1)]
)
async def test_discovery_requires_explicit_requester_admission(
    adapter, kind, admission
):
    adapter.set_authorization_check(lambda *args: admission)
    result = await adapter.discover_matrix(
        kind, "!origin:server", 20, requester="@alice:server", search_term="alice"
    )
    assert result == {"error": "Matrix requester is not authorized for this room"}
    adapter._client.api.request.assert_not_awaited()
    adapter._client.get_joined_rooms.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,refusal", [("joined_rooms", "left"), ("users", "allowlist")])
async def test_origin_room_policy_gates_directory_and_rooms(adapter, kind, refusal):
    if refusal == "left":
        adapter._joined_rooms.remove("!origin:server")
    else:
        adapter._allowed_room_ids = {"!elsewhere:server"}
    result = await adapter.discover_matrix(
        kind, "!origin:server", 20, requester="@alice:server", search_term="alice"
    )
    assert result == {"error": "Matrix room is not allowed or joined"}
    adapter._client.api.request.assert_not_awaited()
    adapter._client.get_joined_rooms.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["joined_rooms", "users"])
async def test_disconnected_client_is_reported_as_disconnected(adapter, kind):
    adapter._client = None
    result = await adapter.discover_matrix(
        kind, "!origin:server", 20, requester="@alice:server", search_term="alice"
    )
    assert result == {"error": "Matrix client is disconnected"}


@pytest.mark.asyncio
async def test_candidate_rooms_require_both_room_and_sender_policy(adapter):
    adapter._allowed_room_ids = {"!origin:server", "!space:server"}
    adapter.set_authorization_check(
        lambda user, chat_type, room: room != "!space:server"
    )
    result = await adapter.discover_matrix(
        "joined_rooms", "!origin:server", 20, requester="@alice:server"
    )
    assert result == {
        "rooms": [_summary("!origin:server")],
        "truncated": False,
        "errors": [],
    }
    assert _created(adapter) == ["!origin:server"]


@pytest.mark.asyncio
async def test_room_scan_and_results_are_bounded(adapter):
    adapter._joined_rooms.update(
        f"!r{i:03}:server" for i in range(discovery.MAX_ROOM_CANDIDATES + 20)
    )
    adapter._client.get_state_event.side_effect = lambda room, kind: (
        {"type": "m.space"} if kind == "m.room.create" else {}
    )
    spaces_only = await adapter.discover_matrix(
        "joined_rooms", "!origin:server", 1, requester="@alice:server"
    )
    scanned = _created(adapter)

    adapter._client.get_state_event.reset_mock(side_effect=True)
    adapter._client.get_state_event.return_value = {}
    first = await adapter.discover_matrix(
        "joined_rooms", "!origin:server", 1, requester="@alice:server"
    )

    adapter._client.get_joined_members.reset_mock()
    adapter._client.api.request.return_value = {
        "results": [{"user_id": "@nobody:server"}],
        "limited": False,
    }
    users = await adapter.discover_matrix(
        "users", "!origin:server", 20, requester="@alice:server", search_term="nobody"
    )

    assert (
        spaces_only,
        len(scanned),
        first,
        _created(adapter),
        users,
        adapter._client.get_joined_members.await_count,
    ) == (
        {"rooms": [], "truncated": True, "errors": []},
        discovery.MAX_ROOM_CANDIDATES,
        {"rooms": [_summary("!origin:server")], "truncated": True, "errors": []},
        ["!origin:server"],
        {"users": [], "truncated": True, "errors": []},
        discovery.MAX_ROOM_CANDIDATES,
    )


@pytest.mark.asyncio
async def test_directory_lists_only_users_who_share_a_room_with_the_requester(
    adapter, members
):
    members["!private:server"] = {"@bot:server", "@dave:server"}
    adapter._joined_rooms = set(members)
    adapter._client.api.request.return_value = {
        "results": [
            {"user_id": user}
            for user in (
                "@alice:server",
                "@bob:server",
                "@carol:server",
                "@dave:server",
                "@erin:server",
            )
        ],
        "limited": False,
    }
    result = await adapter.discover_matrix(
        "users", "!origin:server", 20, requester="@alice:server", search_term="server"
    )
    assert result == {
        "users": [
            {"user_id": user, "display_name": None, "avatar_url": None}
            for user in ("@alice:server", "@bob:server", "@carol:server")
        ],
        "truncated": False,
        "errors": [],
    }


@pytest.mark.asyncio
async def test_directory_bounds_fields_and_homeserver_truncation(adapter):
    adapter._client.api.request.return_value = {
        "results": [
            {
                "user_id": "@alice:server",
                "display_name": "A" * 1300,
                "avatar_url": "mxc://server/avatar",
                "access_token": "secret",
            },
            {"user_id": "@bob:server"},
        ],
        "limited": True,
    }
    result = await adapter.discover_matrix(
        "users", "!origin:server", 1, requester="@alice:server", search_term=" alice "
    )
    assert result == {
        "users": [
            {
                "user_id": "@alice:server",
                "display_name": "A" * 1200,
                "avatar_url": "mxc://server/avatar",
            }
        ],
        "truncated": True,
        "errors": [],
    }
    call = adapter._client.api.request.await_args
    assert call.args[0].value == "POST"
    assert call.args[1:] == (
        "/_matrix/client/v3/user_directory/search",
        {"search_term": "alice", "limit": 1},
    )
    assert call.kwargs == {"retry_count": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {},
        {"results": [], "limited": "false"},
        {"results": [{"user_id": "invalid"}], "limited": False},
    ],
)
async def test_directory_rejects_malformed_responses(adapter, response):
    adapter._client.api.request.return_value = response
    result = await adapter.discover_matrix(
        "users", "!origin:server", 20, requester="@alice:server", search_term="alice"
    )
    assert result == {
        "users": [],
        "truncated": True,
        "errors": [{"error": "Matrix user directory returned an invalid response"}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["joined_rooms", "users"])
async def test_discovery_failures_are_explicit_and_do_not_include_exception_secrets(
    adapter, kind
):
    adapter._client.get_state_event.side_effect = RuntimeError("access_token=secret")
    adapter._client.api.request.side_effect = RuntimeError("access_token=secret")
    result = await adapter.discover_matrix(
        kind, "!origin:server", 20, requester="@alice:server", search_term="alice"
    )
    if kind == "users":
        assert result == {
            "users": [],
            "truncated": True,
            "errors": [{"error": "Matrix user directory failed: RuntimeError"}],
        }
        return
    assert result == {
        "rooms": [],
        "truncated": True,
        "errors": [
            {"room_id": room, "error": "Matrix room discovery failed: RuntimeError"}
            for room in sorted(adapter._joined_rooms)
        ],
    }


@pytest.mark.asyncio
async def test_discovery_timeout_preserves_partial_results(adapter, expire_discovery):
    async def state(room, event_type):
        if room == "!room:server" and event_type == "m.room.create":
            await expire_discovery()
        return {}

    adapter._client.get_state_event.side_effect = state
    result = await adapter.discover_matrix(
        "joined_rooms", "!origin:server", 20, requester="@alice:server"
    )
    assert result == {
        "rooms": [_summary("!origin:server")],
        "truncated": True,
        "errors": [{"error": "Matrix discovery timed out"}],
    }
    assert _created(adapter) == ["!origin:server", "!room:server"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner,failure,stage",
    [
        ("replacement", "error", "room_state"),
        ("disconnected", "timeout", "directory"),
        ("replacement", "deadline", "candidate_policy"),
        ("current", "error", "candidate_policy"),
    ],
)
async def test_discovery_exception_results_belong_to_current_client(
    adapter, expire_discovery, owner, failure, stage
):
    original = adapter._client
    replacement = SimpleNamespace(get_state_event=AsyncMock())
    clients = {"current": original, "replacement": replacement, "disconnected": None}
    allowed = adapter._is_allowed_matrix_room_event

    async def fail(*args, **kwargs):
        try:
            if failure == "deadline":
                await expire_discovery()
            await asyncio.sleep(0)
            if failure == "timeout":
                raise TimeoutError
            raise RuntimeError("access_token=secret")
        finally:
            adapter._client = clients[owner]

    async def policy(room, **kwargs):
        if stage == "candidate_policy" and room == "!room:server":
            await fail()
        return await allowed(room, **kwargs)

    async def state(room, event_type):
        if stage == "room_state" and (room, event_type) == ("!room:server", "m.room.create"):
            await fail()
        return {}

    adapter._is_allowed_matrix_room_event = policy
    original.get_state_event.side_effect = state
    if stage == "directory":
        original.api.request.side_effect = fail

    result = await adapter.discover_matrix(
        "users" if stage == "directory" else "joined_rooms",
        "!origin:server",
        20,
        requester="@alice:server",
        search_term="alice",
    )
    expected = {"error": "Matrix discovery client changed"}
    if owner == "current":
        expected = {
            "rooms": [_summary("!origin:server")],
            "truncated": True,
            "errors": [{"error": "Matrix discovery failed: RuntimeError"}],
        }
    assert result == expected
    replacement.get_state_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_discovery_uses_live_joined_membership_not_the_adapter_cache(adapter):
    adapter._client.get_joined_rooms.side_effect = None
    adapter._client.get_joined_rooms.return_value = ["!origin:server", "!new:server"]
    before = adapter._joined_rooms.copy()
    result = await adapter.discover_matrix(
        "joined_rooms", "!origin:server", 20, requester="@alice:server"
    )
    assert result == {
        "rooms": [_summary("!new:server"), _summary("!origin:server")],
        "truncated": False,
        "errors": [],
    }
    assert _created(adapter) == ["!new:server", "!origin:server"]
    assert adapter._joined_rooms == before
    adapter._client.get_joined_rooms.assert_awaited_once()


@pytest.mark.asyncio
async def test_directory_refuses_a_session_room_the_bot_has_left(adapter):
    adapter._client.get_joined_rooms.side_effect = None
    adapter._client.get_joined_rooms.return_value = ["!room:server"]
    adapter._client.api.request.return_value = {
        "results": [{"user_id": "@alice:server"}],
        "limited": False,
    }
    result = await adapter.discover_matrix(
        "users", "!origin:server", 20, requester="@alice:server", search_term="alice"
    )
    assert result == {"error": "Matrix session room is no longer joined"}
    adapter._client.api.request.assert_not_awaited()


@pytest.fixture
def policy_adapter(adapter):
    profiles = {
        user: SimpleNamespace(displayname=user)
        for user in ("@bot:server", "@alice:server")
    }
    adapter._client.state_store = SimpleNamespace(
        has_full_member_list=AsyncMock(return_value=True),
        get_members=AsyncMock(return_value=set(profiles)),
        get_member_profiles=AsyncMock(return_value=profiles),
    )
    adapter._client.get_joined_members = AsyncMock(return_value=profiles)
    adapter._joined_rooms = {"!a-origin:server", "!candidate:server"}
    adapter._allowed_room_ids = {"!a-origin:server"}
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,entry,classification",
    [
        ("joined_rooms", "candidate", "malformed"),
        ("joined_spaces", "candidate", "dm"),
        ("joined_rooms", "origin", "error"),
        ("joined_rooms", "origin", "group"),
    ],
)
async def test_real_admission_distinguishes_unavailable_classification_from_denial(
    policy_adapter, monkeypatch, kind, entry, classification
):
    from plugins.platforms.matrix import adapter as matrix_adapter

    adapter = policy_adapter
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        matrix_adapter, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    origin = None
    if entry == "candidate":
        origin = await adapter._resolve_room_identity("!a-origin:server")
    else:
        adapter._allowed_room_ids = {"!allowed:server"}
    adapter._client.state_store.has_full_member_list.return_value = False
    adapter._client.state_store.get_members.return_value = {
        "@bot:server",
        "@alice:server",
    }

    async def members(room):
        clock.now += 1
        if classification == "error":
            raise RuntimeError("!candidate:server access_token=secret")
        if classification == "malformed":
            return None
        result = {"@bot:server": {}, "@alice:server": {}}
        if classification == "group":
            result["@bob:server"] = {}
        return result

    adapter._client.get_joined_members.side_effect = members
    adapter._client.get_state_event.side_effect = lambda room, event_type: (
        {"type": "m.space"}
        if kind == "joined_spaces" and event_type == "m.room.create"
        else {}
    )
    adapter._client.state_store.get_members.reset_mock()

    result = await adapter.discover_matrix(
        kind, "!a-origin:server", 20, requester="@alice:server"
    )
    room_type = "m.space" if kind == "joined_spaces" else None
    expected: dict[str, Any] = {
        ("candidate", "malformed"): {
            "rooms": [_summary("!a-origin:server", room_type)],
            "truncated": True,
            "errors": [{"error": "Matrix room classification is unavailable"}],
        },
        ("candidate", "dm"): {
            "rooms": [
                _summary("!a-origin:server", room_type),
                _summary("!candidate:server", room_type),
            ],
            "truncated": False,
            "errors": [],
        },
        ("origin", "error"): {
            "rooms": [],
            "truncated": True,
            "errors": [{"error": "Matrix room classification is unavailable"}],
        },
        ("origin", "group"): {"error": "Matrix room is not allowed or joined"},
    }[entry, classification]
    assert result == expected
    assert adapter._joined_rooms == {"!a-origin:server", "!candidate:server"}
    if origin is not None:
        assert adapter._room_identities["!a-origin:server"] == origin
    adapter._client.state_store.get_members.assert_not_awaited()
    assert _created(adapter) == [
        room["room_id"] for room in expected.get("rooms", [])
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry,owner,outcome,boundary",
    [
        ("origin", "replacement", "response", "name"),
        ("candidate", "disconnected", "error", "api_members"),
        ("candidate", "replacement", "response", "full_list"),
        ("read", "replacement", "error", "store_profiles"),
    ],
)
async def test_real_identity_admission_never_reads_or_caches_after_owner_change(
    policy_adapter, monkeypatch, entry, owner, outcome, boundary
):
    from plugins.platforms.matrix import adapter as matrix_adapter

    adapter = policy_adapter
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        matrix_adapter, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    original = adapter._client
    target = "!candidate:server" if entry == "candidate" else "!a-origin:server"
    if entry == "candidate":
        await adapter._resolve_room_identity("!a-origin:server")
    before = (
        dict(adapter._room_identities),
        dict(adapter._room_identity_cached_at),
        {room: dict(values) for room, values in adapter._room_state_values.items()},
    )
    calls = []
    entered = asyncio.Event()
    released = asyncio.Event()
    prefix = []
    replacement_calls = []

    async def replaced(*args, **kwargs):
        replacement_calls.append((args, kwargs))
        return {}

    replacement = SimpleNamespace(
        get_state_event=replaced,
        get_joined_members=replaced,
        state_store=SimpleNamespace(
            has_full_member_list=replaced,
            get_members=replaced,
            get_member_profiles=replaced,
        ),
    )

    async def response(stage, room, value):
        calls.append((stage, room))
        if stage == boundary and room == target:
            entered.set()
            await released.wait()
            if outcome == "error":
                raise RuntimeError("access_token=secret")
        return value

    async def state(room, event_type):
        stage = {
            "m.room.name": "name",
            "m.room.topic": "topic",
            "m.room.canonical_alias": "alias",
            "m.room.create": "create",
        }.get(event_type, event_type)
        return await response(stage, room, {})

    async def full_list(room):
        return await response("full_list", room, boundary != "api_members")

    async def store_members(room, **kwargs):
        return await response("store_members", room, {"@bot:server", "@alice:server"})

    async def profiles(room, **kwargs):
        return await response(
            "store_profiles",
            room,
            {
                "@bot:server": SimpleNamespace(displayname="Hermes"),
                "@alice:server": SimpleNamespace(displayname="Alice"),
            },
        )

    async def api_members(room):
        return await response("api_members", room, {"@bot:server": {}, "@alice:server": {}})

    original.get_state_event.side_effect = state
    original.state_store.has_full_member_list.side_effect = full_list
    original.state_store.get_members.side_effect = store_members
    original.state_store.get_member_profiles.side_effect = profiles
    original.get_joined_members.side_effect = api_members
    operation = (
        adapter.read_matrix_context(
            "recent", "!a-origin:server", None, 20, requester="@alice:server"
        )
        if entry == "read"
        else adapter.discover_matrix(
            "joined_rooms", "!a-origin:server", 20, requester="@alice:server"
        )
    )
    task = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        adapter._client = replacement if owner == "replacement" else None
        prefix[:] = calls
        released.set()
        result = await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert (
        result,
        calls,
        replacement_calls,
        adapter._room_identities,
        adapter._room_identity_cached_at,
        adapter._room_state_values,
        adapter._joined_rooms,
    ) == (
        {
            "error": "Matrix client changed"
            if entry == "read"
            else "Matrix discovery client changed"
        },
        prefix,
        [],
        *before,
        {"!a-origin:server", "!candidate:server"},
    )
