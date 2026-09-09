"""Global Files navigation cannot retarget an open native group message."""

from types import SimpleNamespace

import pytest

from gateway import hosted_rooms, hosted_room_messaging as rooms
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.hosted_room_messaging_all_files import AllFilesMenu
from gateway.hosted_room_messaging_files import text as files_text
from gateway.native_reply_input import text as compose_text
from gateway.platforms.base import MessageType
from tests.gateway.test_hosted_room_messaging_all_files import share
from tests.plugins.platforms.telegram.test_group_compose import compose, message, reply


@pytest.mark.asyncio
async def test_global_files_to_group_compose_keeps_target_and_commits_once(compose, monkeypatch):
    state = compose
    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id", lambda: "install:test-gateway")
    backend = rooms.MessagingRoomBackend(db_path=state.backend.db_path)
    state.backend = backend
    monkeypatch.setattr(rooms, "current_room_backend", lambda: backend)
    room = hosted_rooms.room_state(backend.db_path, room_id="release-room")
    share(SimpleNamespace(
        store=HostedRoomAttachmentStore(backend.db_path), db=backend.db_path,
        authority=room["authority_gateway_id"],
    ), room, 1, name="release.md")
    state.original = state.adapter._build_message_event(message("/group files", number=101), MessageType.COMMAND)
    menu = AllFilesMenu(state.runner, state.original, backend, "/group")
    first = await menu.open_page()
    assert files_text("all_title") in first.title
    token = next(token for token, action in menu.actions.items() if action[0] == "groups")
    await menu.choose("100", token)
    selected = rooms._room_picker_value(room)
    token = next(token for token, action in menu.child.actions.items() if action == ("bind", selected))
    detail = await menu.choose("100", token)
    assert detail.choices[0]["label"] == compose_text("send")
    assert detail.choices[0]["full_width"] is True
    await menu.choose("100", detail.choices[0]["value"])
    prompt = state.outgoing[-1][1]
    assert prompt.text.startswith(compose_text("title", group="Release room"))

    menu.child.room = None
    await menu.child.bind("2")
    body = "@ops Check the shared release notes; preserve Café 日本語."
    actual_message = await reply(state, prompt, body)
    await reply(state, prompt, body)
    expected_id = rooms.messaging_event_id(state.adapter._build_message_event(actual_message, MessageType.TEXT))
    release = hosted_rooms.read_events(backend.db_path, room_id="release-room")["events"]
    research = hosted_rooms.read_events(backend.db_path, room_id="research-room")["events"]
    matching = [event for event in release if event["event_id"] == expected_id]
    assert len(matching) == 1 and matching[0]["payload"]["text"] == body
    assert all(event["event_id"] != expected_id for event in research)
    state.runner._hm_pending_reply_intercepts.assert_not_awaited()
