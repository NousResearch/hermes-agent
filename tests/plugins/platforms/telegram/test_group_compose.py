"""Native picker -> ForceReply -> admitted inbound -> real group Send boundary."""

import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
pytest.importorskip("telegram")
from telegram import Chat, ForceReply, Message, User

from agent.i18n import SUPPORTED_LANGUAGES, t
from gateway import hosted_room_messaging as rooms
from gateway.hosted_room_messaging_files import FilesMenu
from gateway.native_reply_input import text
from gateway.platforms.base import MessageType, SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.choice_picker import cancel_choice_pages
from tests.gateway.test_hosted_room_messaging import _FakeService, _runner, _seed_rooms
from gateway.config import Platform


def message(body="Hello group", *, number=201, reply=None, user=111, chat=100, thread=None, chat_type="private", **kwargs):
    return Message(
        message_id=number, date=datetime.now(timezone.utc),
        chat=Chat(chat, chat_type), from_user=User(user, "Synthetic", user == 999),
        text=body, reply_to_message=reply, message_thread_id=thread, **kwargs,
    )


@pytest.fixture
def compose(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111,222")
    runner = _runner(platform=Platform.TELEGRAM, extra={
        "allow_from": ["111", "222"], "allow_admin_from": ["111"],
    })
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._hm_pre_gateway_dispatch_hook = lambda event, source: event
    runner._profile_name_for_source = lambda source: "default"
    runner._hm_pending_reply_intercepts = AsyncMock(return_value="ordinary chat")
    adapter = TelegramAdapter(runner.config.platforms[Platform.TELEGRAM])
    adapter.gateway_runner = runner
    runner.adapters[Platform.TELEGRAM] = adapter
    adapter.set_session_store(SimpleNamespace(sessions_dir=tmp_path / "sessions"))
    adapter.set_message_handler(runner._primary_message_handler())
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.TELEGRAM))
    adapter._ensure_forum_commands = AsyncMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="response"))
    outgoing = []

    async def send_message(**kwargs):
        sent = message(kwargs["text"], number=500 + len(outgoing), user=999,
                       thread=kwargs.get("message_thread_id"))
        outgoing.append((kwargs, sent))
        return sent

    adapter._bot = SimpleNamespace(id=999, username="synthetic_bot", send_message=AsyncMock(side_effect=send_message))
    db, _, _ = _seed_rooms(tmp_path)
    backend = _FakeService(db)
    monkeypatch.setattr(rooms, "current_room_backend", lambda: backend)
    original = adapter._build_message_event(message("/group 1", number=101), MessageType.COMMAND)
    state = SimpleNamespace(runner=runner, adapter=adapter, backend=backend, original=original, outgoing=outgoing)
    yield state
    cancel_choice_pages(adapter)


async def open_compose(state):
    menu = FilesMenu(state.runner, state.original, state.backend, "/group")
    await menu.bind("1")
    page = await menu.room_page()
    assert page.choices[0]["label"] == text("send")
    assert page.choices[0]["full_width"] is True
    assert all(not choice["full_width"] for choice in page.choices[1:])
    assert len(page.choices) <= 12
    assert await menu.send_page(page)
    rows = state.outgoing[-1][0]["reply_markup"].inline_keyboard
    assert len(rows[0]) == 1 and rows[0][0].text == text("send")
    assert len(rows[1]) == 2
    picker = state.adapter._choice_picker_state["100"]
    from gateway.choice_picker import choice_action
    callback = choice_action(picker["token"], picker["revision"], 0)
    sent_menu = state.outgoing[-1][1]
    query = SimpleNamespace(
        message=sent_menu, from_user=User(111, "Synthetic", False),
        answer=AsyncMock(), edit_message_text=AsyncMock(),
    )
    await state.adapter._handle_choice_picker_callback(query, callback, "100")
    query.answer.assert_awaited_once()
    prompt = state.outgoing[-1][1]
    assert isinstance(state.outgoing[-1][0]["reply_markup"], ForceReply)
    assert prompt.text == text("title", group="Release room") + "\n\n" + text("prompt")
    assert menu.compose_request.token not in prompt.text
    return menu, prompt, query


async def reply(state, prompt, body="Hello group", **kwargs):
    msg = message(body, reply=prompt, **kwargs)
    await state.adapter._handle_text_message(SimpleNamespace(message=msg, effective_message=msg, update_id=700), None)
    return msg


@pytest.mark.asyncio
async def test_real_send_uses_reply_identity_and_leaves_both_busy_guards_untouched(compose):
    state = compose
    menu, prompt, _ = await open_compose(state)
    sent = state.outgoing[-1][0]
    assert sent["reply_parameters"].message_id == 101
    assert sent["reply_parameters"].allow_sending_without_reply is False
    assert sent["reply_markup"].selective is True
    guard = object()
    session_key = state.adapter._event_session_key(state.original)
    state.adapter._active_sessions[session_key] = guard
    state.adapter._handle_message_while_active = AsyncMock(side_effect=AssertionError("busy guard"))
    state.runner._hm_handle_running_session_message = AsyncMock(side_effect=AssertionError("busy runner"))
    body = "@ops Review this\nKeep --flags, quotes and /paths unchanged."
    msg = await reply(state, prompt, body)
    assert len(state.backend.sent) == 1
    result = state.backend.sent[0]
    assert result["payload"]["text"] == body
    actual = state.adapter._build_message_event(msg, MessageType.TEXT)
    assert result["event_id"] == rooms.messaging_event_id(actual)
    assert result["event_id"] != rooms.messaging_event_id(state.original)
    assert result["room_id"] == "release-room"
    assert state.adapter._active_sessions[session_key] is guard
    assert not state.adapter._pending_text_batches
    state.runner._hm_pending_reply_intercepts.assert_not_awaited()
    assert state.adapter.send.await_args.kwargs["reply_to"] == "201"
    assert "Queued in Release room" in state.adapter.send.await_args.kwargs["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["user", "chat", "topic", "prompt", "edit", "machine", "profile", "adapter", "expired", "restart"])
async def test_foreign_expired_and_replayed_prompt_replies_never_become_bot_turns(compose, mismatch):
    state = compose
    menu, prompt, _ = await open_compose(state)
    kwargs = {}
    if mismatch == "user":
        kwargs["user"] = 222
    elif mismatch == "chat":
        kwargs["chat"] = 200
    elif mismatch == "topic":
        kwargs["thread"] = 77
        kwargs["is_topic_message"] = True
    elif mismatch == "prompt":
        prompt = message(prompt.text, number=9999, user=999)
    elif mismatch == "edit":
        kwargs["edit_date"] = datetime.now(timezone.utc)
    elif mismatch == "machine":
        state.runner._is_user_authorized_for_source = lambda source: True
        state.adapter._is_user_authorized_from_message = lambda msg: True
        kwargs["user"] = 999
    elif mismatch == "profile":
        state.runner._profile_name_for_source = lambda source: "other"
    elif mismatch == "adapter":
        state.runner.adapters[Platform.TELEGRAM] = SimpleNamespace(config=state.adapter.config)
    elif mismatch == "expired":
        menu.compose_request.deadline = 0
    else:
        state.adapter._native_reply_inputs.clear()
    await reply(state, prompt, **kwargs)
    assert not state.backend.sent
    state.runner._hm_pending_reply_intercepts.assert_not_awaited()
    assert not state.adapter._pending_text_batches


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["revoke", "home", "authority", "target", "navigate"])
async def test_fresh_target_and_authorization_while_composing(compose, monkeypatch, change):
    state = compose
    menu, prompt, _ = await open_compose(state)
    if change == "revoke":
        state.adapter.config.extra["allow_admin_from"] = ["222"]
    elif change == "home":
        from gateway.config import HomeChannel
        state.runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(Platform.TELEGRAM, "100", "New home")
    elif change == "navigate":
        menu.room = None
        await menu.bind("2")
    else:
        original = rooms.list_messaging_rooms

        def changed(*args, **kwargs):
            rows = original(*args, **kwargs)
            first = next(row for row in rows if row["room_id"] == "release-room")
            first["authority_epoch" if change == "authority" else "room_id"] = 900 if change == "authority" else "replacement"
            return rows

        monkeypatch.setattr(rooms, "list_messaging_rooms", changed)
    await reply(state, prompt)
    assert len(state.backend.sent) == (1 if change == "navigate" else 0)
    if state.backend.sent:
        assert state.backend.sent[0]["room_id"] == "release-room"


@pytest.mark.asyncio
async def test_repeat_same_receipt_and_second_message_do_not_send_twice(compose):
    _, prompt, _ = await open_compose(compose)
    await reply(compose, prompt)
    await reply(compose, prompt)
    await reply(compose, prompt, "A different message", number=202)
    assert len(compose.backend.sent) == 1
    assert compose.adapter.send.await_args.kwargs["content"] == text("closed")


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["   ", "x" * 4097], ids=["blank", "too-long"])
async def test_invalid_text_is_not_sent_or_routed_to_agent(compose, body):
    _, prompt, _ = await open_compose(compose)
    await reply(compose, prompt, body)
    assert not compose.backend.sent
    assert compose.adapter.send.await_args.kwargs["content"] == text("text_only")
    compose.runner._hm_pending_reply_intercepts.assert_not_awaited()
    await reply(compose, prompt, "valid", number=202)
    assert len(compose.backend.sent) == 1


@pytest.mark.asyncio
async def test_media_caption_is_rejected_before_download_or_bot_processing(compose):
    from telegram import Document
    _, prompt, _ = await open_compose(compose)
    msg = message(None, reply=prompt, caption="send this", document=Document("file", "unique"))
    await compose.adapter._handle_media_message(SimpleNamespace(message=msg, update_id=701), None)
    assert not compose.backend.sent
    assert compose.adapter.send.await_args.kwargs["content"] == text("text_only")
    compose.runner._hm_pending_reply_intercepts.assert_not_awaited()


@pytest.mark.asyncio
async def test_unrelated_text_and_reply_keep_existing_batching(compose):
    await open_compose(compose)
    compose.adapter._enqueue_text_event = lambda event: compose.outgoing.append(event)
    await reply(compose, message("Ordinary bot reply", user=999))
    assert compose.outgoing[-1].text == "Hello group"
    assert not compose.backend.sent


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop", "/group 1 stop", "/help", "/approve", "/deny"])
async def test_emergency_and_slash_commands_keep_normal_dispatch(compose, command):
    _, prompt, _ = await open_compose(compose)
    compose.adapter.handle_message = AsyncMock()
    msg = message(command, reply=prompt)
    await compose.adapter._handle_command(SimpleNamespace(message=msg, effective_message=msg, update_id=701), None)
    compose.adapter.handle_message.assert_awaited_once()
    assert compose.adapter.handle_message.await_args.args[0].text == command
    assert not compose.backend.sent


@pytest.mark.asyncio
async def test_drain_blocks_send_without_changing_active_work(compose):
    _, prompt, _ = await open_compose(compose)
    compose.runner._external_drain_active = True
    await reply(compose, prompt)
    assert not compose.backend.sent
    assert "maintenance" in compose.adapter.send.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_prompt_send_failure_is_honest_and_orphan_reply_is_closed(compose):
    state = compose
    menu = FilesMenu(state.runner, state.original, state.backend, "/group")
    await menu.bind("1")
    state.adapter._bot.send_message.side_effect = TimeoutError()
    from gateway.hosted_room_messaging_compose import begin
    assert await begin(menu) == text("unavailable")
    kwargs = state.adapter._bot.send_message.await_args.kwargs
    await reply(state, message(kwargs["text"], user=999, number=9999))
    assert not state.backend.sent
    assert state.adapter.send.await_args.kwargs["content"] == text("closed")
    state.runner._hm_pending_reply_intercepts.assert_not_awaited()


@pytest.mark.asyncio
async def test_backend_ambiguous_result_is_not_automatically_retried(compose):
    _, prompt, _ = await open_compose(compose)
    calls = []

    def unknown(**kwargs):
        calls.append(kwargs)
        raise TimeoutError()

    compose.backend.send = unknown
    await reply(compose, prompt)
    assert compose.adapter.send.await_args.kwargs["content"] == text("unknown")
    await reply(compose, prompt)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_cancel_and_native_input_only_on_room_view(compose):
    menu, prompt, _ = await open_compose(compose)
    cancel = next(token for token, action in menu.actions.items() if action[0] == "cancel_compose")
    await menu.choose("100", cancel)
    await reply(compose, prompt)
    assert not compose.backend.sent
    page = await menu.room_page(view="bots")
    assert text("send") not in [item["label"] for item in page.choices]
    page = await menu.room_page(view="bot", bot_query="ops")
    assert text("send") not in [item["label"] for item in page.choices]


@pytest.mark.asyncio
async def test_restart_recognizes_actual_prompt_id_without_quoted_text(compose):
    menu, prompt, _ = await open_compose(compose)
    assert menu.compose_request.path.stat().st_mode & 0o777 == 0o600
    compose.adapter._native_reply_inputs.clear()
    await reply(compose, message(None, number=prompt.message_id, user=999))
    assert not compose.backend.sent
    assert compose.adapter.send.await_args.kwargs["content"] == text("closed")
    compose.runner._hm_pending_reply_intercepts.assert_not_awaited()


@pytest.mark.asyncio
async def test_arbitrary_quoted_marker_is_not_authority_or_capture(compose):
    await open_compose(compose)
    captured = []
    compose.adapter._enqueue_text_event = captured.append
    await reply(compose, message("[reply:0123456789abcdef]", user=222))
    assert captured[0].text == "Hello group"
    assert not compose.backend.sent


@pytest.mark.asyncio
@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
async def test_localized_orphan_shape_only_closes_never_resolves_group(compose, monkeypatch, language):
    # No request or receipt exists; even a plausible title cannot infer a target.
    body = t("gateway.group_compose.title", lang=language, group="Release room")
    body += "\n\n" + t("gateway.group_compose.prompt", lang=language)
    monkeypatch.setattr(rooms, "list_messaging_rooms", lambda *a, **kw: pytest.fail("orphan inferred a group"))
    await reply(compose, message(body, user=999, number=9999))
    assert compose.adapter.send.await_args.kwargs["content"] == text("closed")
    assert not compose.backend.sent
    compose.runner._hm_pending_reply_intercepts.assert_not_awaited()
    assert not compose.adapter._pending_text_batches


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    "Message to Release room",
    "Message to Release room\n\nType your message and send. Extra text",
    "A quote: Message to Release room\n\nType your message and send.",
    "Message to Release room\nType your message and send.",
    "Message to \n\nType your message and send.",
    "Message to    \n\nType your message and send.",
    "Message to " + "x" * 81 + "\n\nType your message and send.",
    "[reply:0123456789abcdef]",
    "Message to Release\u200broom\n\nType your message and send.",
], ids=["title-only", "extra-text", "prefix", "wrong-spacing", "empty-label", "blank-label", "long-label", "old-marker", "format-control"])
async def test_own_bot_near_matches_stay_ordinary(compose, body):
    captured = []
    compose.adapter._enqueue_text_event = captured.append
    await reply(compose, message(body, user=999, number=9999))
    assert captured[0].text == "Hello group"
    assert not compose.backend.sent
    compose.adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_friendly_prompt_quoted_from_another_author_stays_ordinary(compose):
    _, prompt, _ = await open_compose(compose)
    captured = []
    compose.adapter._enqueue_text_event = captured.append
    await reply(compose, message(prompt.text, user=222, number=prompt.message_id))
    assert captured[0].text == "Hello group"
    assert not compose.backend.sent


@pytest.mark.asyncio
async def test_friendly_text_without_reply_is_ordinary(compose):
    body = text("title", group="Release room") + "\n\n" + text("prompt")
    captured = []
    compose.adapter._enqueue_text_event = captured.append
    await reply(compose, None, body)
    assert captured[0].text == body
    assert not compose.backend.sent


@pytest.mark.asyncio
async def test_receipt_error_rejects_known_input_without_capturing_other_replies(compose, monkeypatch):
    from plugins.platforms.telegram import reply_input
    _, prompt, _ = await open_compose(compose)

    def unavailable(*args):
        raise OSError("isolated test storage error")

    monkeypatch.setattr(reply_input, "prompt_token", unavailable)
    await reply(compose, message(None, number=prompt.message_id, user=999))
    assert compose.adapter.send.await_args.kwargs["content"] == text("closed")
    assert not compose.backend.sent
    captured = []
    compose.adapter._enqueue_text_event = captured.append
    await reply(compose, message("A normal Bot answer", number=9999, user=999))
    assert captured[0].text == "Hello group"


@pytest.mark.asyncio
async def test_concurrent_replies_claim_once_without_interrupting_active_work(compose):
    _, prompt, _ = await open_compose(compose)
    started, release = threading.Event(), threading.Event()
    original = compose.backend.send

    def blocked(**kwargs):
        started.set()
        assert release.wait(5)
        return original(**kwargs)

    compose.backend.send = blocked
    first = asyncio.create_task(reply(compose, prompt, "first"))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        await reply(compose, prompt, "second", number=202)
        assert compose.adapter.send.await_args.kwargs["content"] == text("closed")
        assert not first.done()
    finally:
        release.set()
        await first
    assert len(compose.backend.sent) == 1
    assert compose.backend.sent[0]["payload"]["text"] == "first"


@pytest.mark.asyncio
async def test_routed_profile_uses_receiving_adapter_and_live_profile_scope(compose, tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    profile_home = tmp_path / "profiles" / "ops"
    profile_home.mkdir(parents=True)
    compose.runner.config.multiplex_profiles = True
    compose.runner._profile_name_for_source = lambda source: "ops"
    compose.runner._resolve_profile_home_for_source = lambda source: profile_home
    compose.original = compose.adapter._build_message_event(message("/group 1", number=101), MessageType.COMMAND)
    compose.adapter.set_message_handler(compose.runner._make_default_profile_message_handler())
    # The receiving transport owns admission even though the room work runs under ops.
    compose.original.source._authorization_profile_home = tmp_path
    _, prompt, _ = await open_compose(compose)
    scopes = []
    original = compose.backend.send

    def capture(**kwargs):
        scopes.append(Path(get_hermes_home()))
        return original(**kwargs)

    compose.backend.send = capture
    await reply(compose, prompt)
    assert scopes == [profile_home]
    assert compose.backend.sent[0]["room_id"] == "release-room"


@pytest.mark.asyncio
async def test_six_member_bot_page_keeps_choice_budget(compose):
    from gateway import hosted_rooms
    hosted_rooms.create_room(
        compose.backend.db_path, room_id="six-room", name="Six members",
        authority_gateway_id="install:test-gateway",
        members=[{"member_id": member, "handle": member} for member in ["default", "ops", "a", "b", "c", "d"]],
    )
    current = next(row for row in rooms.list_messaging_rooms(compose.backend) if row["room_id"] == "six-room")
    menu = FilesMenu(compose.runner, compose.original, compose.backend, "/group")
    await menu.bind(rooms.room_reference(current))
    page = await menu.room_page(view="bots")
    assert sum(action[0] == "bot" for action in menu.actions.values()) == 6
    assert len(page.choices) <= 12
    assert all(action[0] != "compose" for action in menu.actions.values())


@pytest.mark.asyncio
async def test_unwired_adapter_retains_typed_send_fallback(compose, monkeypatch):
    monkeypatch.setattr(TelegramAdapter, "supports_reply_input", False)
    menu = FilesMenu(compose.runner, compose.original, compose.backend, "/group")
    await menu.bind("1")
    page = await menu.room_page()
    assert all(action[0] != "compose" for action in menu.actions.values())
    assert "/group 1 send <message>" in page.title


@pytest.mark.asyncio
async def test_shared_home_topic_input_requires_live_audience_consent(compose):
    from gateway.config import HomeChannel
    from gateway.group_home_identity import acknowledgement
    from gateway.hosted_room_messaging_compose import begin
    state = compose
    original = message("/group 1", number=101, chat_type="supergroup", thread=77, is_topic_message=True)
    state.original = state.adapter._build_message_event(original, MessageType.COMMAND)
    home = HomeChannel(Platform.TELEGRAM, "100", "Synthetic home", thread_id="77", user_id="111")
    home.group_audience_ack = acknowledgement(home)
    state.adapter.config.home_channel = home
    state.adapter.config.extra["group_allow_admin_from"] = ["111"]
    state.adapter.config.extra["group_allow_from"] = ["111", "222"]
    menu = FilesMenu(state.runner, state.original, state.backend, "/group")
    await menu.bind("1")
    await begin(menu)
    prompt = state.outgoing[-1][1]
    assert state.outgoing[-1][0]["message_thread_id"] == 77
    await reply(state, prompt, thread=77, chat_type="supergroup", is_topic_message=True)
    assert len(state.backend.sent) == 1
    await begin(menu)
    prompt = state.outgoing[-1][1]
    home.group_audience_ack = None
    await reply(state, prompt, thread=77, chat_type="supergroup", is_topic_message=True, number=202)
    assert len(state.backend.sent) == 1
    state.runner._hm_pending_reply_intercepts.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_trigger_gate_still_rejects_disallowed_chat(compose):
    _, prompt, _ = await open_compose(compose)
    compose.adapter._should_process_message = lambda msg: False
    await reply(compose, prompt)
    assert not compose.backend.sent
    compose.runner._hm_pending_reply_intercepts.assert_not_awaited()


@pytest.mark.asyncio
async def test_admission_rewrite_does_not_lose_native_reply_ownership(compose):
    from dataclasses import replace
    _, prompt, _ = await open_compose(compose)
    compose.runner._hm_pre_gateway_dispatch_hook = lambda event, source: replace(event)
    await reply(compose, prompt)
    assert len(compose.backend.sent) == 1
    compose.runner._hm_pending_reply_intercepts.assert_not_awaited()


@pytest.mark.asyncio
async def test_revocation_during_receipt_claim_still_prevents_backend_send(compose):
    menu, prompt, _ = await open_compose(compose)
    original = menu.compose_request.claim

    def revoke(event_id):
        result = original(event_id)
        compose.adapter.config.extra["allow_admin_from"] = ["222"]
        return result

    menu.compose_request.claim = revoke
    await reply(compose, prompt)
    assert not compose.backend.sent
    assert compose.adapter.send.await_args.kwargs["content"] == text("closed")


def test_locale_parity_and_placeholder_bound():
    import yaml
    from agent.i18n import _locales_dir
    locales = [yaml.safe_load(path.read_text(encoding="utf-8"))["gateway"]["group_compose"] for path in _locales_dir().glob("*.yaml")]
    expected = set(locales[0])
    assert all(set(locale) == expected for locale in locales)
    assert all(0 < len(locale["placeholder"]) <= 64 for locale in locales)
    assert all("{group}" in locale["title"] for locale in locales)
