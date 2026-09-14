"""Initially authorized native actions and cancellation at real receipt phases."""

import os

import pytest

from gateway import hosted_rooms, hosted_room_messaging as rooms
from gateway.choice_picker import ChoicePage, ChoiceProgress
from gateway.config import Platform
from gateway import hosted_room_file_delivery as delivery
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.test_hosted_room_file_access import publish
from tests.gateway.test_hosted_room_file_access import file_state as file_state
from tests.gateway.test_hosted_room_messaging_files import NativeAdapter, choice
from tests.gateway.test_slash_access_dispatch import _make_runner


def real_runner(state, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42,43")
    for key in [
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
    ]:
        monkeypatch.delenv(key, raising=False)
    runner = _make_runner(
        platform=Platform.TELEGRAM, platform_extra={"allow_admin_from": ["42"]}
    )
    del runner.__dict__["_is_user_authorized"]
    adapter = NativeAdapter()
    adapter.config = runner.config.platforms[Platform.TELEGRAM]
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._primary_profile_name = "default"
    runner.pairing_store = None
    runner._normalize_source_for_session_key = lambda source: source
    state.backend.service.status = lambda room_id: {
        "working": False,
        "blocked": False,
        "counts": {},
        "pending_actions": [],
    }
    monkeypatch.setattr(rooms, "current_room_backend", lambda: state.backend)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42"
    )
    source.profile = "default"
    source.is_one_to_one = True
    event = MessageEvent(
        text="/group 1 files",
        message_type=MessageType.COMMAND,
        source=source,
        user_id="42",
        message_id="101",
    )
    assert runner._is_user_authorized_for_source(source) is True
    assert runner._can_control_group_chats(event) is True
    return runner, adapter, event


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["file", "reply"])
@pytest.mark.parametrize("phase", ["after-load", "after-sending-receipt"])
async def test_current_talk_revocation_before_native_submit_is_enforced(
    file_state, monkeypatch, kind, phase
):
    state = file_state
    runner, adapter, event = real_runner(state, monkeypatch)
    publish(state, "private.txt", b"Revoked sender must not receive these bytes")
    if kind == "reply":
        hosted_rooms.append_event(
            state.db,
            room_id="room-1",
            event_id="bot-reply",
            kind="message.member",
            actor={"kind": "member", "id": "ops"},
            authority_gateway_id=state.authority,
            authority_epoch=1,
            payload={
                "member_id": "ops",
                "thread_id": "t",
                "text": "# Private full reply\nNot for a revoked sender",
            },
        )
        event.text = "/group 1"
    await runner._handle_message(event)
    picker = adapter.pages[-1]
    page = ChoicePage(picker["title"], picker["choices"])
    selected = choice(page, "private.txt" if kind == "file" else "Get full reply")
    # The actual gateway auth check is still true when the native callback starts.
    assert runner._is_user_authorized_for_source(event.source) is True
    progress = await picker["on_choice_selected"]("42", selected)
    assert isinstance(progress, ChoiceProgress)
    revoked = []
    if phase == "after-load":
        method = "read_file" if kind == "file" else "read_shared_message"
        original = getattr(state.backend, method)

        def revoke_after_load(**kwargs):
            result = original(**kwargs)
            assert adapter.documents == []
            os.environ["TELEGRAM_ALLOWED_USERS"] = "43"
            revoked.append(True)
            return result

        monkeypatch.setattr(state.backend, method, revoke_after_load)
    else:
        original_mark = delivery.mark_delivery

        def revoke_after_sending(db, key, next_state):
            result = original_mark(db, key, next_state)
            if next_state == "sending":
                assert adapter.documents == []
                os.environ["TELEGRAM_ALLOWED_USERS"] = "43"
                revoked.append(True)
            return result

        monkeypatch.setattr(delivery, "mark_delivery", revoke_after_sending)
    outcome = await progress.complete()
    assert revoked
    assert runner._is_user_authorized_for_source(event.source) is False
    assert runner.config.platforms[Platform.TELEGRAM].extra["allow_admin_from"] == [
        "42"
    ]
    assert not list(state.db.parent.glob("group-file-delivery-tmp/send-*"))
    assert adapter.documents == [], (
        f"native submission occurred after talk revocation: {outcome!r}"
    )


@pytest.mark.asyncio
async def test_legitimate_callback_download_has_exact_destination_and_no_duplicate(
    file_state, monkeypatch
):
    state = file_state
    runner, adapter, event = real_runner(state, monkeypatch)
    publish(state, "legitimate.txt", b"exact native bytes")
    await runner._handle_message(event)
    picker = adapter.pages[-1]
    page = ChoicePage(picker["title"], picker["choices"])
    progress = await picker["on_choice_selected"]("42", choice(page, "legitimate.txt"))
    await progress.complete()
    await progress.complete()
    assert len(adapter.documents) == 1
    target, data, name, path, options = adapter.documents[0]
    assert (target, data, name) == ("42", b"exact native bytes", "legitimate.txt")
    assert not path.exists()
    assert options["metadata"]["group_file_delivery_id"]
