"""The original disclosure scope fences the start of each post-await private read."""

import asyncio
from dataclasses import replace
from threading import Event

import pytest

from gateway import hosted_room_messaging as rooms
from gateway.config import Platform
from tests.gateway.test_group_home_consent import home, command
from tests.gateway.group_chat_picker_fixtures import picker_home, choose_first


async def accept(state):
    await command(state, "/group")
    await command(state, "/group confirm")
    assert state.runner._can_control_group_chats(state.event)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verb", ["1", "Release room", "1 bot 1", "1 bots", "1 approvals", "list", ""]
)
async def test_no_private_formatter_starts_after_listing_revokes_enrollment(
    home, monkeypatch, verb
):
    await accept(home)
    original = rooms.list_messaging_rooms
    fetched = []

    def revoke(*args, **kwargs):
        result = original(*args, **kwargs)
        home.runner.config.platforms[Platform.TELEGRAM].extra[
            "group_allow_admin_from"
        ] = ["other"]
        return result

    monkeypatch.setattr(rooms, "list_messaging_rooms", revoke)
    for name in (
        "format_room_detail",
        "format_room_bot_detail",
        "format_room_bot_list",
        "format_room_list",
        "room_picker_choices",
        "room_bot_picker_choices",
    ):
        function = getattr(rooms, name)

        def observed(*args, _function=function, _name=name, **kwargs):
            fetched.append(_name)
            return _function(*args, **kwargs)

        monkeypatch.setattr(rooms, name, observed)
    monkeypatch.setattr(
        home.service, "status", lambda room: pytest.fail("post-revocation status fetch")
    )
    result = await command(home, "/group " + verb)
    assert not fetched and "Release room" not in str(result)


@pytest.mark.asyncio
async def test_reselection_and_new_consent_cannot_reauthorize_an_old_detail_read(
    home, monkeypatch
):
    await accept(home)
    entered, release = Event(), Event()
    original = rooms.list_messaging_rooms
    armed = True
    details = []

    def held(*args, **kwargs):
        nonlocal armed
        result = original(*args, **kwargs)
        if armed:
            armed = False
            entered.set()
            assert release.wait(10)
        return result

    monkeypatch.setattr(rooms, "list_messaging_rooms", held)
    monkeypatch.setattr(
        rooms,
        "format_room_detail",
        lambda *args, **kwargs: details.append(args) or "PrivateDetailMarker",
    )
    pending = asyncio.create_task(command(home, "/group 1"))
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        await home.runner._handle_set_home_command(replace(home.event, text="/sethome"))
        await accept(home)
        assert home.runner._can_control_group_chats(home.event)
    finally:
        release.set()
    result = await pending
    assert not details and "PrivateDetailMarker" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["room", "bot", "approval"])
async def test_picker_home_post_listing_continuation_starts_no_new_status_or_decision(
    picker_home, monkeypatch, kind
):
    await accept(picker_home)
    picker_home.service.service = None
    picker_home.service.room_status["pending_actions"] = [
        {
            "kind": "approval",
            "member_id": "default",
            "task_id": "task-1",
            "request_id": "request-1",
            "execution_generation": 1,
            "authority_gateway_id": "install:test-gateway",
            "authority_epoch": 1,
            "approval": {
                "description": "PrivateDecisionMarker",
                "command": "echo private",
                "choices": ["once", "deny"],
            },
        }
    ]
    original = rooms.list_messaging_rooms
    armed = False
    revoked_reads = []

    def revoke(*args, **kwargs):
        result = original(*args, **kwargs)
        if armed:
            revoked_reads.append(True)
            picker_home.adapter.config.extra["group_allow_admin_from"] = ["other"]
        return result

    # Install before the picker captures its imported continuation functions.
    monkeypatch.setattr(rooms, "list_messaging_rooms", revoke)
    verb = {"room": "/group", "bot": "/group 1 bots", "approval": "/group 1 approvals"}[
        kind
    ]
    assert await command(picker_home, verb) is None
    armed = True
    monkeypatch.setattr(
        picker_home.service,
        "status",
        lambda room: pytest.fail("post-revocation status fetch"),
    )
    from gateway import hosted_room_messaging_approvals as approvals

    monkeypatch.setattr(
        approvals,
        "begin_approval_command",
        lambda *args, **kwargs: pytest.fail("post-revocation decision"),
    )
    result = await choose_first(picker_home)
    assert revoked_reads == [True]
    assert not picker_home.runner._can_control_group_chats(picker_home.event)
    assert "Release room" not in str(result)
    assert "PrivateDecisionMarker" not in str(result)
