"""Telegram interaction regressions for staged memory and skill writes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin
from gateway.write_approval_interactions import (
    WRITE_APPROVAL_REPLY_KEY,
    WriteApprovalReply,
    build_pending_surface,
    command_for_reply_intent,
)
from plugins.platforms.telegram import adapter as telegram_module
from plugins.platforms.telegram.adapter import TelegramAdapter


MEMORY_SURFACE = {
    "subsystems": ["memory"],
    "items": {"memory": ["abc12345"]},
}
SKILL_SURFACE = {
    "subsystems": ["skills"],
    "items": {"skills": ["def67890"]},
}


class _Button:
    def __init__(self, text, callback_data=None, **_kwargs):
        self.text = text
        self.callback_data = callback_data


class _Markup:
    def __init__(self, rows):
        self.inline_keyboard = rows


def _make_adapter(monkeypatch) -> TelegramAdapter:
    monkeypatch.setattr(telegram_module, "InlineKeyboardButton", _Button)
    monkeypatch.setattr(telegram_module, "InlineKeyboardMarkup", _Markup)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=77))
    return adapter


def _event(text: str, surface=None, *, message_type=MessageType.TEXT) -> MessageEvent:
    metadata = {}
    if surface is not None:
        metadata[WRITE_APPROVAL_REPLY_KEY] = surface
    return MessageEvent(
        text=text,
        message_type=message_type,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            user_id="42",
        ),
        reply_to_message_id="77" if surface is not None else None,
        metadata=metadata,
    )


def test_reply_intents_are_exact_and_scoped():
    assert (
        command_for_reply_intent("approve abc12345", MEMORY_SURFACE)
        == "/memory approve abc12345"
    )
    assert (
        command_for_reply_intent("одобри abc12345", MEMORY_SURFACE)
        == "/memory approve abc12345"
    )
    assert (
        command_for_reply_intent("покажи diff", SKILL_SURFACE)
        == "/skills diff def67890"
    )

    assert command_for_reply_intent("approve all", MEMORY_SURFACE) is None
    assert command_for_reply_intent("одобри всё", MEMORY_SURFACE) is None
    assert command_for_reply_intent("reject skills", SKILL_SURFACE) is None
    assert command_for_reply_intent("approve all", None) is None
    assert (
        command_for_reply_intent("can you approve all of this?", MEMORY_SURFACE) is None
    )
    assert (
        command_for_reply_intent(
            "approve all",
            {
                "subsystems": ["memory", "skills"],
                "items": {"memory": ["abc12345"], "skills": ["def67890"]},
            },
        )
        is None
    )


def test_pending_surface_contains_ids_only(monkeypatch):
    monkeypatch.setattr(
        "tools.write_approval.list_pending",
        lambda subsystem: (
            [
                {
                    "id": "abc12345",
                    "summary": "private content must not enter delivery metadata",
                    "payload": {"content": "secret"},
                }
            ]
            if subsystem == "memory"
            else []
        ),
    )

    assert build_pending_surface(("memory", "skills")) == MEMORY_SURFACE


@pytest.mark.asyncio
async def test_telegram_pending_reply_adds_buttons_and_records_owned_surface(
    monkeypatch,
):
    adapter = _make_adapter(monkeypatch)

    result = await adapter.send(
        "12345",
        "Pending memory writes (1): abc12345",
        metadata={"write_approval": MEMORY_SURFACE},
    )

    assert result.success is True
    kwargs = adapter._bot.send_message.call_args.kwargs
    callbacks = [
        button.callback_data
        for row in kwargs["reply_markup"].inline_keyboard
        for button in row
    ]
    assert "wa:m:a:all" not in callbacks
    assert "wa:m:r:all" not in callbacks
    assert "wa:m:a:abc12345" in callbacks
    assert "wa:m:r:abc12345" in callbacks
    assert adapter._lookup_write_approval_surface("12345", "77") == MEMORY_SURFACE


def test_only_known_approval_replies_are_rewritten(monkeypatch):
    adapter = _make_adapter(monkeypatch)

    scoped = _event("approve abc12345", MEMORY_SURFACE)
    assert adapter._rewrite_write_approval_reply(scoped) is True
    assert scoped.text == "/memory approve abc12345"
    assert scoped.message_type == MessageType.COMMAND

    unrelated = _event("can you approve this design?", MEMORY_SURFACE)
    assert adapter._rewrite_write_approval_reply(unrelated) is False
    assert unrelated.text == "can you approve this design?"

    unscoped = _event("approve all")
    assert adapter._rewrite_write_approval_reply(unscoped) is False


class _IntentRunner(GatewaySlashCommandsMixin):
    def __init__(self, denied=None):
        self.denied = denied
        self.events = []

    def _check_slash_access(self, _source, _command):
        return self.denied

    async def _handle_memory_command(self, event):
        self.events.append(event)
        return WriteApprovalReply("Approved memory", MEMORY_SURFACE)

    async def _handle_skills_command(self, event):
        self.events.append(event)
        return WriteApprovalReply("Approved skills", SKILL_SURFACE)


@pytest.mark.asyncio
async def test_voice_intent_dispatch_reuses_slash_handler_and_access_gate():
    runner = _IntentRunner()
    handled, response = await runner._dispatch_write_approval_reply_intent(
        _event("", MEMORY_SURFACE, message_type=MessageType.VOICE),
        "одобри abc12345",
    )
    assert handled is True
    assert response == "Approved memory"
    assert runner.events[0].text == "/memory approve abc12345"
    assert runner.events[0].message_type == MessageType.COMMAND

    denied_runner = _IntentRunner(denied="admin only")
    handled, response = await denied_runner._dispatch_write_approval_reply_intent(
        _event("", MEMORY_SURFACE, message_type=MessageType.VOICE),
        "approve abc12345",
    )
    assert handled is True
    assert response == "admin only"
    assert denied_runner.events == []


@pytest.mark.asyncio
async def test_voice_reply_is_intercepted_at_the_stt_boundary(monkeypatch):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False
    runner._check_slash_access = lambda _source, _command: None
    runner._handle_memory_command = AsyncMock(
        return_value=WriteApprovalReply("Approved memory", MEMORY_SURFACE)
    )

    event = _event("", MEMORY_SURFACE, message_type=MessageType.VOICE)
    event.media_urls = ["/tmp/approval-reply.ogg"]
    event.media_types = ["audio/ogg"]
    monkeypatch.setattr(
        runner,
        "_enrich_message_with_transcription",
        AsyncMock(return_value=("transcribed", ["approve abc12345"])),
    )

    response = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
        session_key="agent:main:telegram:dm:12345",
    )

    assert isinstance(response, WriteApprovalReply)
    assert response == "Approved memory"
    runner._handle_memory_command.assert_awaited_once()
    command_event = runner._handle_memory_command.call_args.args[0]
    assert command_event.text == "/memory approve abc12345"
    assert command_event.message_type == MessageType.COMMAND


class _CallbackRunner:
    def __init__(self, authorized=True):
        self.authorized = authorized
        self.events = []

    def _is_user_authorized(self, _source):
        return self.authorized

    async def _handle_message(self, event):
        self.events.append(event)
        return WriteApprovalReply("Approved 1 memory write(s).", MEMORY_SURFACE)


@pytest.mark.asyncio
async def test_callback_routes_through_runner_command_path(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    runner = _CallbackRunner()
    adapter.set_message_handler(runner._handle_message)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="78"))
    query = SimpleNamespace(
        data="wa:m:a:abc12345",
        from_user=SimpleNamespace(id=42, first_name="Joe"),
        message=SimpleNamespace(
            chat_id=12345,
            chat=SimpleNamespace(type="private"),
            message_thread_id=None,
            message_id=77,
        ),
        answer=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
    )

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    assert runner.events[0].text == "/memory approve abc12345"
    assert runner.events[0].message_type == MessageType.COMMAND
    adapter.send.assert_awaited_once()
    assert adapter.send.call_args.args[1] == "Approved 1 memory write(s)."
    assert adapter.send.call_args.kwargs["metadata"]["write_approval"] == MEMORY_SURFACE


@pytest.mark.asyncio
async def test_callback_fails_closed_for_unauthorized_user(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    runner = _CallbackRunner(authorized=False)
    adapter.set_message_handler(runner._handle_message)
    adapter.send = AsyncMock()
    query = SimpleNamespace(
        data="wa:m:r:abc12345",
        from_user=SimpleNamespace(id=99, first_name="Mallory"),
        message=SimpleNamespace(
            chat_id=12345,
            chat=SimpleNamespace(type="private"),
            message_thread_id=None,
            message_id=77,
        ),
        answer=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
    )

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    assert runner.events == []
    adapter.send.assert_not_awaited()
    assert "not authorized" in query.answer.call_args.kwargs["text"].lower()


@pytest.mark.asyncio
async def test_stale_bulk_callback_cannot_touch_later_pending_record(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    pending_ids = {"abc12345", "feed6789"}

    class PendingRunner(_CallbackRunner):
        async def _handle_message(self, event):
            self.events.append(event)
            if event.text.endswith(" all"):
                pending_ids.clear()
            else:
                pending_ids.discard(event.text.rsplit(" ", 1)[-1])
            return WriteApprovalReply("updated", MEMORY_SURFACE)

    runner = PendingRunner()
    adapter.set_message_handler(runner._handle_message)
    adapter.send = AsyncMock()
    query = SimpleNamespace(
        data="wa:m:a:all",
        from_user=SimpleNamespace(id=42, first_name="Joe"),
        message=SimpleNamespace(
            chat_id=12345,
            chat=SimpleNamespace(type="private"),
            message_thread_id=None,
            message_id=77,
        ),
        answer=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
    )

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    assert pending_ids == {"abc12345", "feed6789"}
    assert runner.events == []
    adapter.send.assert_not_awaited()
    assert "invalid" in query.answer.call_args.kwargs["text"].lower()
