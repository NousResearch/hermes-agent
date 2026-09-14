"""Independent-review reproductions and canonical native-compose boundary controls."""

import asyncio
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway import hosted_rooms
from gateway import hosted_room_messaging as rooms
from gateway.native_reply_input import text
from tests.plugins.platforms.telegram.test_group_compose import compose, message, open_compose, reply
from tests.gateway.test_hosted_room_messaging import _TestHostedRoomService


@pytest.mark.asyncio
async def test_startup_replay_of_closed_prompt_does_not_enter_busy_bot_path(compose, monkeypatch):
    state = compose
    _, prompt, _ = await open_compose(state)
    state.adapter._native_reply_inputs.clear()
    state.runner._startup_restore_in_progress = True
    await reply(state, prompt, "Private Group Send draft must be closed.")
    assert len(state.runner._startup_restore_queue) == 1
    queued = state.runner._startup_restore_queue[0]
    assert getattr(queued, "_native_reply_submission", None) is not None
    session_key = state.adapter._event_session_key(queued)
    active_task = asyncio.create_task(asyncio.Event().wait())
    guard = asyncio.Event()
    state.adapter._active_sessions[session_key] = guard
    state.adapter._session_tasks[session_key] = active_task
    ordinary_busy = AsyncMock(return_value=True)
    state.adapter._busy_session_handler = ordinary_busy
    from tools import clarify_gateway
    monkeypatch.setattr(clarify_gateway, "get_pending_for_session", lambda *args, **kwargs: None)
    state.runner._startup_restore_in_progress = False
    try:
        assert await state.runner._drain_startup_restore_queue() == 1
        ordinary_busy.assert_not_awaited()
        assert state.adapter._active_sessions[session_key] is guard
        assert not active_task.done()
        assert not state.backend.sent
        assert state.adapter.send.await_args.kwargs["content"] == text("closed")
    finally:
        active_task.cancel()
        await asyncio.gather(active_task, return_exceptions=True)
        state.adapter._active_sessions.pop(session_key, None)
        state.adapter._session_tasks.pop(session_key, None)


def real_backend(state, tmp_path, monkeypatch, *, attached_service=True):
    authority = "install:test-gateway"
    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id", lambda: authority)
    service = _TestHostedRoomService(tmp_path / "real-room-state.db")
    service.create_room(room_id="release-room", name="Release room", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "ops", "profile": "ops", "handle": "ops"},
    ])
    state.backend = rooms.MessagingRoomBackend(db_path=service.db_path, service=service if attached_service else None)
    monkeypatch.setattr(rooms, "current_room_backend", lambda: state.backend)
    return service


def advance(service, *, event_id="new-term-during-claim"):
    hosted_rooms.claim_authority(
        service.db_path, room_id="release-room",
        expected_gateway_id="install:test-gateway", expected_epoch=1,
        new_gateway_id="install:test-gateway", event_id=event_id,
    )


def user_events(service):
    return [event for event in hosted_rooms.read_events(service.db_path, room_id="release-room")["events"]
            if event["kind"] == "message.user"]


@pytest.mark.asyncio
@pytest.mark.parametrize("attached_service", [False, True])
async def test_changed_authority_during_claim_cannot_send_under_new_term(compose, tmp_path, monkeypatch, attached_service):
    service = real_backend(compose, tmp_path, monkeypatch, attached_service=attached_service)
    menu, prompt, _ = await open_compose(compose)
    original = menu.compose_request.claim

    def move_term(event_id):
        result = original(event_id)
        advance(service)
        return result

    menu.compose_request.claim = move_term
    await reply(compose, prompt, "Do not cross the bound authority term.")
    assert not user_events(service)
    assert compose.adapter.send.await_args.kwargs["content"] == text("closed")
    await reply(compose, prompt, "Do not cross the bound authority term.")
    assert not user_events(service)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["service-entry", "service-append", "store-append"])
async def test_late_authority_change_is_fenced_at_actual_append(compose, tmp_path, monkeypatch, stage):
    service = real_backend(compose, tmp_path, monkeypatch, attached_service=stage != "store-append")
    _, prompt, _ = await open_compose(compose)
    attempts = []
    if stage == "service-entry":
        original = service.send_server_owned

        def mutate_before_service_read(**kwargs):
            attempts.append(kwargs["expected_authority"])
            advance(service)
            return original(**kwargs)

        monkeypatch.setattr(service, "send_server_owned", mutate_before_service_read)
    else:
        original = hosted_rooms.append_event

        def mutate_before_append(*args, **kwargs):
            if kwargs.get("kind") == "message.user":
                attempts.append((kwargs["authority_gateway_id"], kwargs["authority_epoch"]))
                advance(service)
            return original(*args, **kwargs)

        monkeypatch.setattr(hosted_rooms, "append_event", mutate_before_append)
    await reply(compose, prompt, "A boundary-fenced message")
    assert attempts == [("install:test-gateway", 1)]
    assert not user_events(service)
    assert compose.adapter.send.await_args.kwargs["content"] == text("closed")


@pytest.mark.asyncio
async def test_old_prompt_after_accepted_send_cannot_append_in_new_epoch(compose, tmp_path, monkeypatch):
    service = real_backend(compose, tmp_path, monkeypatch)
    _, prompt, _ = await open_compose(compose)
    await reply(compose, prompt, "One accepted turn")
    advance(service)
    await reply(compose, prompt, "One accepted turn")
    await reply(compose, prompt, "Another draft with the old prompt", number=202)
    events = user_events(service)
    assert len(events) == 1 and events[0]["authority_epoch"] == 1
    assert compose.adapter.send.await_args.kwargs["content"] == text("closed")


@pytest.mark.asyncio
@pytest.mark.parametrize("attached_service", [False, True])
async def test_normal_typed_send_keeps_current_authority_and_idempotency(compose, tmp_path, monkeypatch, attached_service):
    service = real_backend(compose, tmp_path, monkeypatch, attached_service=attached_service)
    advance(service)
    event = replace(compose.original, text="/group 1 send A normal typed message")
    first = await compose.runner._handle_room_command(event)
    second = await compose.runner._handle_room_command(event)
    assert first == second and first.startswith("Queued in Release room")
    events = user_events(service)
    assert len(events) == 1 and events[0]["authority_epoch"] == 2
    assert events[0]["payload"]["text"] == "A normal typed message"


@pytest.mark.parametrize("disbanding", [False, True])
def test_service_fenced_send_rejects_new_epoch_on_both_append_branches(compose, tmp_path, monkeypatch, disbanding):
    service = real_backend(compose, tmp_path, monkeypatch)
    advance(service)
    if disbanding:
        monkeypatch.setattr(service, "_room_is_disbanding", lambda room_id: True)
    with pytest.raises(hosted_rooms.AuthorityConflictError):
        service.send_server_owned(
            room_id="release-room", event_id="stale-bound-send",
            actor={"kind": "user", "id": "synthetic-owner"}, payload={"text": "Cannot cross terms"},
            expected_authority=("install:test-gateway", 1),
        )
    assert not user_events(service)


def test_backend_does_not_retry_legacy_service_without_fence():
    called = []

    class LegacyService:
        def send(self, *, room_id, event_id, actor, payload):
            called.append(event_id)
            return {"accepted": True}

    backend = rooms.MessagingRoomBackend(db_path=None, service=LegacyService())
    values = dict(room_id="room", event_id="bound-send", actor={"kind": "user", "id": "owner"}, payload={"text": "hello"})
    with pytest.raises(TypeError):
        backend.send(**values, expected_authority=("install:test-gateway", 1))
    assert called == []
    assert backend.send(**values) == {"accepted": True}
    assert called == ["bound-send"]


def test_frozen_remote_compose_cannot_borrow_a_replacement_epoch_link(compose, monkeypatch):
    import time
    from gateway import hosted_room_controls

    hosted_room_controls.save_peer_control_link(
        compose.backend.db_path, room_id="remote-room", member_id="reviewer", target_profile="default",
        room_name="Remote group", member_count=2, home_url="https://home.example.test",
        authority_gateway_id="install:remote-home", authority_epoch=2,
        control_token="A" * 43, expires_at=time.time() + 600,
    )
    room = {"room_id": "remote-room", "_remote_member_id": "reviewer", "_room_mode": "remote",
            "authority_gateway_id": "install:remote-home", "authority_epoch": 1}
    monkeypatch.setattr(rooms, "RoomControlHTTPClient", lambda *args: pytest.fail("borrowed a replacement control link"))
    with pytest.raises(hosted_rooms.AuthorityConflictError):
        rooms.send_to_room(compose.backend, room, compose.original, "Bound remote draft",
                           expected_authority=("install:remote-home", 1))


@pytest.mark.asyncio
async def test_startup_native_replay_rechecks_revoked_rights(compose):
    _, prompt, _ = await open_compose(compose)
    compose.runner._startup_restore_in_progress = True
    await reply(compose, prompt)
    compose.adapter.config.extra["allow_admin_from"] = ["222"]
    compose.adapter.handle_message = AsyncMock(side_effect=AssertionError("ordinary session path"))
    assert await compose.runner._drain_startup_restore_queue() == 1
    assert compose.adapter.send.await_args.kwargs["content"] == text("closed")
    assert not compose.backend.sent
    compose.runner._hm_pending_reply_intercepts.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_native_replay_keeps_routed_profile_scope(compose, tmp_path):
    from hermes_constants import get_hermes_home
    from gateway.platforms.base import MessageType
    profile_home = tmp_path / "profiles" / "ops"
    profile_home.mkdir(parents=True)
    compose.runner.config.multiplex_profiles = True
    compose.runner._profile_name_for_source = lambda source: "ops"
    compose.runner._resolve_profile_home_for_source = lambda source: profile_home
    compose.original = compose.adapter._build_message_event(message("/group 1", number=101), MessageType.COMMAND)
    compose.original.source._authorization_profile_home = tmp_path
    compose.adapter.set_message_handler(compose.runner._make_default_profile_message_handler())
    _, prompt, _ = await open_compose(compose)
    compose.runner._startup_restore_in_progress = True
    await reply(compose, prompt)
    assert not compose.backend.sent
    original = compose.backend.send
    scopes = []

    def scoped_send(**kwargs):
        scopes.append(Path(get_hermes_home()))
        return original(**kwargs)

    compose.backend.send = scoped_send
    compose.adapter.handle_message = AsyncMock(side_effect=AssertionError("ordinary session path"))
    assert await compose.runner._drain_startup_restore_queue() == 1
    assert scopes == [profile_home]
    assert len(compose.backend.sent) == 1


@pytest.mark.asyncio
async def test_startup_native_replay_rejects_replaced_receiving_adapter(compose):
    from gateway.config import Platform
    from gateway.platforms.base import SendResult
    from plugins.platforms.telegram.adapter import TelegramAdapter
    _, prompt, _ = await open_compose(compose)
    compose.runner._startup_restore_in_progress = True
    await reply(compose, prompt)
    replacement = TelegramAdapter(compose.adapter.config)
    replacement.gateway_runner = compose.runner
    replacement.set_message_handler(compose.runner._primary_message_handler())
    replacement.send = AsyncMock(return_value=SendResult(success=True, message_id="rejection"))
    replacement.handle_message = AsyncMock(side_effect=AssertionError("ordinary session path"))
    compose.runner.adapters[Platform.TELEGRAM] = replacement
    assert await compose.runner._drain_startup_restore_queue() == 1
    assert replacement.send.await_args.kwargs["content"] == text("closed")
    assert not compose.backend.sent


@pytest.mark.asyncio
@pytest.mark.parametrize("fake_marker", [None, {"token": "not-an-in-process-submission"}])
async def test_startup_normal_or_untyped_events_keep_normal_dispatch(compose, fake_marker):
    event = replace(compose.original, text="A normal message")
    event._native_reply_submission = fake_marker
    compose.runner._queue_startup_restore_event(event)
    compose.adapter.handle_message = AsyncMock()
    compose.adapter._dispatch_inline_reply = AsyncMock(side_effect=AssertionError("native bypass"))
    assert await compose.runner._drain_startup_restore_queue() == 1
    compose.adapter.handle_message.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_finish_failure_is_unknown_without_resend_or_draft_persistence(compose):
    menu, prompt, _ = await open_compose(compose)

    def unavailable(*args):
        raise sqlite3.OperationalError("temporary receipt failure")

    menu.compose_request.finish = unavailable
    draft = "NO_DRAFT_BODY_IN_NATIVE_RECEIPTS"
    await reply(compose, prompt, draft)
    assert compose.adapter.send.await_args.kwargs["content"] == text("unknown")
    await reply(compose, prompt, draft)
    assert len(compose.backend.sent) == 1
    with sqlite3.connect(menu.compose_request.path) as conn:
        stored = conn.execute("SELECT * FROM native_reply_inputs").fetchall()
    assert draft not in repr(stored)
    compose.runner._hm_pending_reply_intercepts.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_caller_keeps_claimed_work_tracked_and_never_resends(compose):
    _, prompt, _ = await open_compose(compose)
    started, release = threading.Event(), threading.Event()
    original = compose.backend.send

    def blocked(**kwargs):
        started.set()
        assert release.wait(5)
        return original(**kwargs)

    compose.backend.send = blocked
    pending = asyncio.create_task(reply(compose, prompt, "One accepted draft."))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert compose.runner._active_deferred_agent_worker_count() >= 1
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

        async def drained():
            while compose.runner._active_deferred_agent_worker_count():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(drained(), 3)
    await reply(compose, prompt, "One accepted draft.")
    assert len(compose.backend.sent) == 1
    compose.runner._hm_pending_reply_intercepts.assert_not_awaited()


@pytest.mark.asyncio
async def test_serialized_sdk_prompt_payload_is_exact_and_reply_bound(compose):
    _, prompt, _ = await open_compose(compose)
    sent = compose.outgoing[-1][0]
    payload = {key: value.to_dict() if hasattr(value, "to_dict") else value for key, value in sent.items()}
    assert payload["reply_parameters"] == {"message_id": 101, "allow_sending_without_reply": False}
    assert payload["reply_markup"]["force_reply"] is True
    assert payload["reply_markup"]["selective"] is True
    assert payload["text"] == prompt.text == text("title", group="Release room") + "\n\n" + text("prompt")
    assert payload["chat_id"] == 100 and payload["parse_mode"] is None
