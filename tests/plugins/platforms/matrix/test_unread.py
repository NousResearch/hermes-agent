"""Unread observations and deliberate receipts preserve their native Matrix scope."""

from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from gateway.config import PlatformConfig
from gateway.session_context import clear_session_vars, set_session_vars
from plugins.platforms.matrix.adapter import MatrixAdapter
from tools.registry import registry


ROOM = "!room:server"
BOT = "@bot:server"
ALICE = "@alice:server"


class _DeniedError(RuntimeError):
    errcode = "M_FORBIDDEN"


def _adapter() -> MatrixAdapter:
    adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={
        "user_id": BOT, "read_receipts": "disabled", "reactions": False,
    }))
    adapter._joined_rooms.add(ROOM)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter.set_authorization_check(lambda user, *_args, **_kwargs: user == ALICE)
    adapter._refresh_dm_cache = AsyncMock()
    adapter._schedule_pending_invite_joins = MagicMock()
    adapter._client = SimpleNamespace(
        handle_sync=MagicMock(return_value=[]),
        sync_store=SimpleNamespace(put_next_batch=AsyncMock()),
        get_state_event=AsyncMock(return_value={"membership": "join"}),
        api=SimpleNamespace(request=AsyncMock(return_value={})),
        get_event=AsyncMock(), set_account_data=AsyncMock(), crypto=None,
    )
    return adapter


async def _sync(adapter: MatrixAdapter, room: dict, *, left: bool = False) -> None:
    await adapter._absorb_sync(adapter._client, {
        "next_batch": "token", "rooms": {"leave" if left else "join": {ROOM: room}},
    })


async def _tool(adapter: MatrixAdapter, name: str, args: dict) -> dict:
    importlib.import_module("tools.matrix_unread_tool")
    tokens = set_session_vars(
        platform="matrix", chat_id=ROOM, user_id=ALICE, transport_adapter=adapter,
    )
    try:
        result = await asyncio.to_thread(registry.dispatch, name, args)
        assert isinstance(result, str)
        return json.loads(result)
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize("delta", ["unknown", "zero", "partial", "omitted", "acknowledged", "left", "replacement", "stale", "policy", "sdk", "bounded", "profiles", "gates"])
async def test_sync_counts_are_observations_of_the_current_owner(monkeypatch, tmp_path, delta):
    from plugins.platforms.matrix.unread import MatrixUnreadState

    adapter = _adapter()
    clock = [100.0]
    adapter._unread = MatrixUnreadState(clock=lambda: clock[0])
    if delta == "gates":
        from hermes_cli.tools_config import _get_platform_tools
        from toolsets import resolve_multiple_toolsets

        expected = {"matrix_unread", "matrix_mark_read"}
        assert expected.issubset(resolve_multiple_toolsets(list(_get_platform_tools({}, "matrix"))))
        assert expected.isdisjoint(resolve_multiple_toolsets(list(_get_platform_tools({}, "telegram"))))
        assert expected.isdisjoint(resolve_multiple_toolsets(list(_get_platform_tools({"agent": {"disabled_toolsets": ["matrix_unread"]}}, "matrix"))))
        tokens = set_session_vars(platform="cli", chat_id=ROOM, user_id=ALICE)
        try:
            importlib.import_module("tools.matrix_unread_tool")
            result = registry.dispatch("matrix_unread", {})
            assert isinstance(result, str)
            assert json.loads(result) == {
                "error": "Matrix unread actions require a live Matrix session",
            }
        finally:
            clear_session_vars(tokens)
        return
    if delta == "profiles":
        from agent import secret_scope
        from gateway.run import _profile_runtime_scope
        from gateway.platforms._shared import get_scoped_secret

        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
        adapters = {}
        for label, count in (("A", 3), ("B", 7)):
            home = tmp_path / label
            home.mkdir()
            (home / ".env").write_text(f"MATRIX_ALLOWED_USERS={ALICE if label == 'A' else '@bob:server'}\n", encoding="utf-8")
            with _profile_runtime_scope(home):
                current = _adapter()
                current.set_authorization_check(lambda user, *_args, **_kwargs: user in get_scoped_secret("MATRIX_ALLOWED_USERS", "").split(","))
                await _sync(current, {"unread_notifications": {"notification_count": count, "highlight_count": 0}})
                adapters[label] = current
        observed = []
        for label in ("A", "B", "A"):
            with _profile_runtime_scope(tmp_path / label):
                observed.append(await _tool(adapters[label], "matrix_unread", {"thread_id": "main"}))
        assert [result.get("notification_count") for result in observed] == [3, None, 3]
        assert observed[1] == {"error": "Matrix requester is not authorized for this room"}
        return
    if delta == "bounded":
        adapter._unread = MatrixUnreadState(max_rooms=1, max_threads=1, clock=lambda: clock[0])
        await _sync(adapter, {"unread_thread_notifications": {
            "$old": {"notification_count": 3}, "$new": {"notification_count": 5},
        }})
        assert [adapter._unread.read(adapter._client, ROOM, root)["notification_count"] for root in ("$old", "$new", "$absent")] == [None, 5, None]
        await adapter._absorb_sync(adapter._client, {"rooms": {"join": {"!other:server": {}}}})
        assert adapter._unread.read(adapter._client, ROOM, "$new")["notification_count"] is None
        return
    if delta == "sdk":
        Client = pytest.importorskip("mautrix.client").Client
        from mautrix.api import HTTPAPI
        from mautrix.client.state_store import MemorySyncStore
        from mautrix.types import UserID

        api = HTTPAPI(base_url="http://127.0.0.1:1", client_session=MagicMock())
        native = Client(mxid=UserID(BOT), api=api, sync_store=MemorySyncStore())
        adapter._client.handle_sync = native.handle_sync

    await _sync(adapter, {} if delta == "unknown" else {
        "unread_notifications": {"notification_count": 4, "highlight_count": 1},
        "unread_thread_notifications": {"$root": {"notification_count": 2, "highlight_count": 0}},
        "account_data": {"events": [{"type": "m.marked_unread", "content": {"unread": True}}]},
    })
    if delta == "zero":
        await _sync(adapter, {"unread_notifications": {"notification_count": 0, "highlight_count": 0}})
    if delta == "partial":
        await _sync(adapter, {"unread_notifications": {"highlight_count": 0}, "unread_thread_notifications": {}})
    if delta == "omitted":
        await adapter._absorb_sync(adapter._client, {"rooms": {"join": {"!other:server": {}}}})
    if delta == "acknowledged":
        adapter._unread.receipt_sent(adapter._client, ROOM, "$root")
        await _sync(adapter, {
            "unread_notifications": {"notification_count": 4, "highlight_count": 1},
            "ephemeral": {"events": [{"type": "m.receipt", "content": {}}]},
        })
    if delta == "left":
        await _sync(adapter, {}, left=True)
    if delta == "replacement":
        adapter._client = SimpleNamespace(**vars(adapter._client))
    if delta == "stale":
        clock[0] += 91
    if delta == "policy":
        adapter.set_authorization_check(lambda *_args, **_kwargs: False)

    result = await _tool(adapter, "matrix_unread", {"thread_id": "main"})
    if delta in {"left", "policy"}:
        expected = {"error": "Matrix room is not allowed or joined"} if delta == "left" else {
            "error": "Matrix requester is not authorized for this room",
        }
        assert result == expected
        return
    unavailable = delta in {"unknown", "replacement"}
    assert result == {
        "room_id": ROOM, "account_user_id": BOT, "count_basis": "bot_account_push_rules",
        "thread_id": "main", "notification_count": None if unavailable else 0 if delta == "zero" else 4,
        "highlight_count": None if unavailable else 0 if delta in {"zero", "partial"} else 1,
        "marked_unread": None if unavailable else True,
        "status": "unavailable" if unavailable else "stale" if delta == "stale" else "observed",
        "observation_generation": None if delta == "replacement" else 2 if delta in {"zero", "partial", "acknowledged"} else 1,
        "last_sync_age_seconds": None if delta == "replacement" else 91.0 if delta == "stale" else 0.0,
    }
    if delta in {"zero", "partial", "omitted", "acknowledged"}:
        thread = await _tool(adapter, "matrix_unread", {"thread_id": "$root"})
        count = 2 if delta == "omitted" else 0
        assert thread == {**result, "thread_id": "$root", "notification_count": count, "highlight_count": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("scope,visibility,failure", [
    ("main", "public", None), ("$root", "private", None), ("room", "private", None),
    ("main", "private", None), ("$root", "public", None),
    ("room", "public", "marker"), ("main", "private", "receipt"),
    ("$root", "public", "thread"), ("main", "public", "room"),
    ("main", "public", "actor"), ("$root", "private", "keys"),
    ("main", "public", "replaced"), ("main", "public", "revoked"),
    ("room", "private", "marker_policy"), ("main", "public", "member"),
    ("main", "private", "uncertain"), ("$root", "public", "root"),
    ("main", "public", "edited_scope"), ("main", "public", "relation"),
    ("$root", "public", "edited_valid"), ("main", "public", "invalid_sync"),
])
async def test_explicit_receipts_never_broaden_or_hide_partial_success(scope, visibility, failure):
    adapter = _adapter()
    client = adapter._client
    await _sync(adapter, {"unread_notifications": {"notification_count": 4, "highlight_count": 1}})
    raw: dict[str, Any] = {
        "room_id": "!wrong:server" if failure == "room" else ROOM,
        "event_id": "$target", "sender": "@mallory:server" if failure == "actor" else ALICE,
        "type": "m.room.encrypted" if failure == "keys" else "m.room.message",
        "content": {"msgtype": "m.text", "body": "target", **({"m.relates_to": {
            "rel_type": "m.thread", "event_id": "$other" if failure == "thread" else "$root",
        }} if scope.startswith("$") else {})},
    }

    async def target(*_args):
        if failure == "replaced":
            adapter._client = SimpleNamespace(**vars(client))
        if failure == "revoked":
            adapter.set_authorization_check(lambda *_args, **_kwargs: False)
        return raw

    client.get_event.side_effect = target
    if failure == "member":
        client.get_state_event.return_value = {"membership": "leave"}
    if failure == "root":
        raw["event_id"] = "$root"
        raw["content"] = {"msgtype": "m.text", "body": "target", "m.relates_to": {
            "rel_type": "m.thread", "event_id": "$other",
        }}
    if failure in {"edited_scope", "edited_valid"}:
        raw["content"]["m.relates_to"] = {"rel_type": "m.thread", "event_id": "$root"}
        raw["unsigned"] = {"m.relations": {"m.replace": {"content": {
            "m.new_content": {"msgtype": "m.text", "body": "edited", "m.relates_to": {
                "rel_type": "m.replace", "event_id": "$target",
            }},
        }}}}
    if failure == "relation":
        raw["content"]["m.relates_to"] = {"rel_type": "m.thread"}
    if failure == "uncertain":
        client.api.request.side_effect = TimeoutError()
    if failure == "marker_policy":
        async def sent(*_args, **_kwargs):
            adapter.set_authorization_check(lambda *_args, **_kwargs: False)
            return {}

        client.api.request.side_effect = sent
    if failure in {"receipt", "marker"}:
        error = _DeniedError("denied")
        (client.api.request if failure == "receipt" else client.set_account_data).side_effect = error
    result = await _tool(adapter, "matrix_mark_read", {
        "event_id": "$root" if failure == "root" else "$target", "thread_id": scope, "visibility": visibility,
    })
    if failure not in {None, "edited_valid", "invalid_sync", "marker", "receipt", "marker_policy", "uncertain"}:
        assert "error" in result
        client.api.request.assert_not_awaited()
        client.set_account_data.assert_not_awaited()
        return
    if failure == "marker_policy":
        assert result == {
            "room_id": ROOM, "account_user_id": BOT, "event_id": "$target", "thread_id": scope,
            "visibility": visibility, "receipt_sent": True, "marked_unread_reset": False,
            "fully_read_marker_changed": False, "counts": "await_sync", "errors": [{
                "operation": "marked_unread", "error": "Matrix access changed after the receipt was sent",
            }],
        }
        client.api.request.assert_awaited_once()
        client.set_account_data.assert_not_awaited()
        return
    if failure == "uncertain":
        assert result == {
            "room_id": ROOM, "account_user_id": BOT, "event_id": "$target", "thread_id": scope,
            "visibility": visibility, "receipt_sent": None, "marked_unread_reset": False,
            "fully_read_marker_changed": False, "counts": "unknown", "errors": [{
                "operation": "receipt", "error": "TimeoutError",
            }],
        }
        client.api.request.assert_awaited_once()
        client.set_account_data.assert_not_awaited()
        return
    assert result == {
        "room_id": ROOM, "account_user_id": BOT, "event_id": "$target", "thread_id": scope,
        "visibility": visibility, "receipt_sent": failure != "receipt",
        "marked_unread_reset": scope == "room" and failure != "marker",
        "fully_read_marker_changed": False, "counts": "unchanged" if failure == "receipt" else "await_sync",
        "errors": [] if failure in {None, "edited_valid", "invalid_sync"} else [{"operation": "receipt" if failure == "receipt" else "marked_unread", "error": "M_FORBIDDEN"}],
    }
    if failure != "receipt":
        call = client.api.request.await_args
        assert (str(call.args[0]), str(call.args[1]), call.args[2], call.kwargs) == (
            "POST", f"/_matrix/client/v3/rooms/%21room%3Aserver/receipt/m.read{'.private' if visibility == 'private' else ''}/%24target",
            {} if scope == "room" else {"thread_id": scope}, {"retry_count": 0},
        )
    if scope == "room" and failure != "receipt":
        client.set_account_data.assert_awaited_once_with("m.marked_unread", {"unread": False}, room_id=ROOM)
    else:
        client.set_account_data.assert_not_awaited()
    if failure == "invalid_sync":
        await _sync(adapter, {"unread_notifications": {"notification_count": -1, "highlight_count": -1}})
    counts = await _tool(adapter, "matrix_unread", {"thread_id": "main"})
    if failure == "invalid_sync":
        assert counts["status"] == "await_sync"
    assert (counts["notification_count"], counts["highlight_count"]) == (4, 1)


def test_hermes_tools_toggles_the_toolset_only_on_matrix(capsys):
    from argparse import Namespace

    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _checklist_toolset_keys, _get_platform_tools, tools_disable_enable_command
    from toolsets import resolve_multiple_toolsets

    unread_tools = {"matrix_unread", "matrix_mark_read"}

    def matrix_state() -> tuple[object, bool]:
        config = load_config()
        saved = (config.get("platform_toolsets") or {}).get("matrix")
        enabled = unread_tools <= set(resolve_multiple_toolsets(list(_get_platform_tools(config, "matrix"))))
        return "matrix_unread" in saved if isinstance(saved, list) else saved, enabled

    observed = {"default": matrix_state()}
    for action in ("disable", "enable"):
        tools_disable_enable_command(Namespace(tools_action=action, platform="matrix", names=["matrix_unread"]))
        observed[action] = matrix_state()
    tools_disable_enable_command(Namespace(tools_action="disable", platform="telegram", names=["matrix_unread"]))

    assert observed == {"default": (None, True), "disable": (False, False), "enable": (True, True)}
    assert ("matrix_unread" in _checklist_toolset_keys("matrix"), "matrix_unread" in _checklist_toolset_keys("telegram")) == (True, False)
    assert (load_config().get("platform_toolsets") or {}).get("telegram") is None
    assert "Toolset 'matrix_unread' is not available on platform 'telegram' (only: matrix)" in capsys.readouterr().out


_RECEIPT_SCOPE_ERROR = {"error": "thread_id must be main, room, or a thread root event ID"}


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args,expected", [
    ("matrix_unread", {"thread_id": "room"}, {"error": "thread_id must be main or a thread root event ID"}),
    ("matrix_unread", {"thread_id": 5}, {"error": "thread_id must be main or a thread root event ID"}),
    ("matrix_mark_read", {"event_id": "$target", "visibility": "public"}, _RECEIPT_SCOPE_ERROR),
    ("matrix_mark_read", {"event_id": "$target", "thread_id": 5, "visibility": "public"}, _RECEIPT_SCOPE_ERROR),
    ("matrix_mark_read", {"event_id": "$target", "thread_id": "thread", "visibility": "public"}, _RECEIPT_SCOPE_ERROR),
    ("matrix_mark_read", {"thread_id": "main", "visibility": "public"}, {"error": "event_id is required"}),
    ("matrix_mark_read", {"event_id": 7, "thread_id": "main", "visibility": "public"}, {"error": "event_id is required"}),
    ("matrix_mark_read", {"event_id": "$target", "thread_id": "main", "visibility": "secret"}, {"error": "visibility must be public or private"}),
])
async def test_invalid_arguments_are_rejected_before_any_matrix_request(name, args, expected):
    adapter = _adapter()
    client = adapter._client

    result = await _tool(adapter, name, args)

    assert result == expected
    assert [mock.await_count for mock in (client.get_state_event, client.get_event, client.api.request, client.set_account_data)] == [0, 0, 0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", [None, "saved-position"])
async def test_initial_sync_requests_unread_counts_before_checkpointing(cursor):
    from mautrix.api import HTTPAPI, Method, Path
    from mautrix.client import Client
    from mautrix.client.state_store import MemoryStateStore, MemorySyncStore
    from mautrix.types import UserID
    from plugins.platforms.matrix.unread import MatrixUnreadState, SYNC_FILTER

    adapter = _adapter()
    api = MagicMock(spec=HTTPAPI)
    api.log = MagicMock()
    api.request = AsyncMock(return_value={
        "next_batch": "fresh-position",
        "rooms": {"join": {ROOM: {
            "unread_notifications": {"notification_count": 4, "highlight_count": 1},
        }}},
    })
    store = MemorySyncStore(cursor)
    client = Client(mxid=UserID(BOT), api=api, state_store=MemoryStateStore(), sync_store=store)
    adapter._client = client
    adapter._unread = MatrixUnreadState(clock=lambda: 10.0)

    await adapter._connect_initial_sync(client)

    assert api.request.await_args_list == [
        call(
            Method.GET, Path.v3.sync,
            query_params={"timeout": 10000, "full_state": "true", "filter": SYNC_FILTER,
                          **({"since": cursor} if cursor else {})},
            retry_count=0, metrics_method="sync",
        ),
    ]
    assert (await store.get_next_batch(), adapter._unread.read(client, ROOM, "main")) == (
        "fresh-position", {
            "notification_count": 4, "highlight_count": 1, "marked_unread": False,
            "status": "observed", "observation_generation": 1, "last_sync_age_seconds": 0.0,
        },
    )
