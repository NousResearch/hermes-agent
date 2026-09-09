"""Ordinary remote-room links and typed Bot commands share canonical navigation."""

import builtins
import copy
import sqlite3
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import hosted_room_controls as controls
from gateway import hosted_room_messaging as rooms
from gateway.choice_picker import ChoicePage
from gateway.platforms.base import SendResult
from tests.gateway.test_group_native_menu_navigation import english, token
from tests.gateway.test_hosted_room_file_access import file_state
from tests.gateway.test_hosted_room_messaging_files import consumer, event


@pytest.fixture
def remote(consumer, monkeypatch):
    state, runner, adapter = consumer
    controls.save_peer_control_link(
        state.db, room_id="workshop-release", member_id="barry",
        target_profile="default", room_name="Workshop release", member_count=2,
        home_url="https://vps.example.test", authority_gateway_id="install:vps",
        authority_epoch=3, control_token="T" * 43, expires_at=time.time() + 600,
    )
    summary = {
        "room": {
            "room_id": "workshop-release", "name": "Workshop release",
            "authority_gateway_id": "install:vps", "authority_epoch": 3,
            "members": [
                {"member_id": "barry", "display_name": "Barry", "handle": "barry"},
                {"member_id": "vps", "display_name": "VPS", "handle": "vps"},
            ],
        },
        "status": {"working": False, "blocked": False, "counts": {}},
        "events": [],
    }
    calls = []

    class RemoteClient:
        def __init__(self, link):
            assert link.room_id == "workshop-release"
            assert link.member_id == "barry"
            assert link.authority_gateway_id == "install:vps"

        def summary(self):
            calls.append("summary")
            return copy.deepcopy(summary)

    monkeypatch.setattr(rooms, "RoomControlHTTPClient", RemoteClient)
    # Exercise the real reference ledger and thin remote projection, not a room mock.
    rooms.list_messaging_rooms(state.backend)
    with sqlite3.connect(state.db) as conn:
        conn.execute(
            "UPDATE hosted_room_messaging_refs SET room_ref=2044 WHERE room_id=?",
            ("workshop-release",),
        )
    room = rooms.resolve_room(rooms.list_messaging_rooms(state.backend), "2044")
    assert room["_room_mode"] == "remote" and not room["members"]
    assert room["member_count"] == 2
    # File inventory is intentionally unavailable; Bots must remain navigable.
    monkeypatch.setattr(adapter, "send_document", None)
    return SimpleNamespace(
        state=state, runner=runner, adapter=adapter, summary=summary, calls=calls,
    )


def block_imports(monkeypatch, *modules):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name in modules:
            raise ImportError("optional consumer is not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["remote", "hosted"])
@pytest.mark.parametrize("verb", ["", "bots", "bot 1"])
@pytest.mark.parametrize("prefix", ["/", "!"])
@pytest.mark.parametrize("file_modules", [True, False])
async def test_typed_and_button_bot_paths_share_back_navigation(
    remote, monkeypatch, location, verb, prefix, file_modules,
):
    runner, adapter = remote.runner, remote.adapter
    reference, handle = ("2044", "barry") if location == "remote" else ("1", "reviewer")
    monkeypatch.setattr(runner, "_typed_command_prefix_for", lambda source: prefix)
    if not file_modules:
        block_imports(monkeypatch, "gateway.hosted_room_file_delivery",
                      "gateway.hosted_room_file_lookup")
    result = await runner._handle_rooms_command(event(f"{prefix}group {reference} {verb}"))
    assert result is None
    call = adapter.pages[-1]
    assert call["metadata"].get("choice_pages") is True
    menu = next(reversed(runner._group_file_menus.values()))
    page = ChoicePage(call["title"], call["choices"])
    callback = call["on_choice_selected"]
    if not verb:
        assert handle in page.title
        page = await callback("chat", token(menu, page, "bots"))
    if verb != "bot 1":
        bot_rows = [c for c in page.choices if menu.actions[c["value"]][0] == "bot"]
        assert len(bot_rows) == 2 and all(c["full_width"] for c in bot_rows)
        page = await callback("chat", bot_rows[0]["value"])
    assert f"{prefix}group {reference} send @{handle}" in page.title
    bots = await callback("chat", token(menu, page, "bots"))
    activity = await callback("chat", token(menu, bots, "room"))
    assert handle in activity.title
    groups = await callback("chat", token(menu, activity, "groups"))
    assert f"{prefix}group list" in groups.title
    row = next(c for c in groups.choices if c["label"].split(". ")[0].endswith(reference))
    reopened = await callback("chat", row["value"])
    assert token(menu, reopened, "bots")
    assert not adapter.documents and not adapter.notices
    assert bool(remote.calls) == (location == "remote")


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["bots", "bot 1"])
@pytest.mark.parametrize("fallback", [
    "old", "dynamic", "unknown", "missing", "old_consumer", "plain", "send_failed",
])
async def test_direct_bot_command_retains_core_fallback(remote, monkeypatch, verb, fallback):
    runner, adapter = remote.runner, remote.adapter
    old_room_calls = []
    if fallback in {"old", "dynamic", "unknown"}:
        monkeypatch.delattr(type(adapter), "supports_choice_pages")
        if fallback == "old":
            monkeypatch.setattr(type(adapter), "supports_choice_pages", False, raising=False)
        elif fallback == "dynamic":
            monkeypatch.setattr(adapter, "supports_choice_pages", True, raising=False)
    elif fallback == "missing":
        block_imports(monkeypatch, "gateway.hosted_room_messaging_files")
    elif fallback == "old_consumer":
        old_consumer = ModuleType("gateway.hosted_room_messaging_files")

        async def try_room_menu(runner, event, backend, room, command):
            old_room_calls.append(room["messaging_ref"])
            return False

        old_consumer.try_room_menu = try_room_menu
        monkeypatch.setitem(sys.modules, old_consumer.__name__, old_consumer)
    elif fallback == "plain":
        monkeypatch.setattr(adapter, "send_choice_picker", None)
    else:
        monkeypatch.setattr(adapter, "send_choice_picker", AsyncMock(
            return_value=SendResult(success=False, error="test transport unavailable"),
        ))
    result = await runner._handle_rooms_command(event(f"/group 2044 {verb}"))
    if verb == "bots" and fallback not in {"plain", "send_failed"}:
        assert result is None
        call = adapter.pages[-1]
        assert not call["metadata"].get("choice_pages")
        assert len(call["choices"]) == 2
        result = await call["on_choice_selected"]("chat", call["choices"][0]["value"])
        assert isinstance(result, str) and "/group 2044 send @barry" in result
    else:
        assert isinstance(result, str)
        assert "Barry" in result
    assert "/group 2044" in result
    assert not adapter.documents
    if fallback == "old_consumer":
        assert not old_room_calls
        result = await runner._handle_rooms_command(event("/group 2044"))
        assert "Workshop release" in result and "Barry" in result
        assert old_room_calls == [2044]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["authority", "members", "roster_during_render", "source_during_render"])
@pytest.mark.parametrize("verb", ["", "bots", "bot 1"])
async def test_remote_bot_pages_reject_unverified_or_changed_reads(remote, monkeypatch, change, verb):
    command = event(f"/group 2044 {verb}")
    if change == "authority":
        remote.summary["room"]["authority_epoch"] += 1
    elif change == "members":
        remote.summary["room"]["members"] = "unverified roster"
    else:
        name = {"": "format_room_detail", "bots": "format_room_bot_list", "bot 1": "format_room_bot_detail"}[verb]
        original = getattr(rooms, name)

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            if change == "roster_during_render":
                remote.summary["room"]["members"].pop(0)
            else:
                command.source.thread_id = "another-topic"
            return result

        monkeypatch.setattr(rooms, name, changed)
    result = await remote.runner._handle_rooms_command(command)
    assert not remote.adapter.pages
    assert isinstance(result, str) and "Barry" not in result and "send @" not in result
