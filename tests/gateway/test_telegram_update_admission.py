import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from gateway.platforms.telegram import ChatType, TelegramAdapter


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token", extra={}))
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    return adapter


def _dm_message(**overrides):
    msg = SimpleNamespace(
        text="hello",
        caption=None,
        entities=[],
        caption_entities=[],
        message_id=42,
        message_thread_id=None,
        is_topic_message=False,
        chat_id=123,
        chat=SimpleNamespace(id=123, type=ChatType.PRIVATE, title=None, full_name="User"),
        from_user=SimpleNamespace(id=123, full_name="User"),
        reply_to_message=None,
        quote=None,
        date=None,
    )
    for key, value in overrides.items():
        setattr(msg, key, value)
    return msg


def test_claim_update_rejects_duplicate_and_malformed_updates(caplog):
    caplog.set_level(logging.INFO)
    adapter = _make_adapter()

    assert adapter._claim_update_for_processing(SimpleNamespace(update_id=100), "text") is True
    assert adapter._claim_update_for_processing(SimpleNamespace(update_id=100), "text") is False
    assert adapter._claim_update_for_processing(SimpleNamespace(update_id=None), "text") is True
    assert adapter._claim_update_for_processing(SimpleNamespace(update_id="not-int"), "text") is False

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "duplicate update_id=100" in log_text
    assert "dedupe unavailable" in log_text
    assert "invalid update_id" in log_text


@pytest.mark.asyncio
async def test_text_handler_dedupes_before_enqueueing_agent_turn(monkeypatch):
    adapter = _make_adapter()
    enqueued = []

    monkeypatch.setattr(adapter, "_should_process_message", lambda message, is_command=False: True)
    monkeypatch.setattr(
        adapter,
        "_build_message_event",
        lambda message, msg_type, update_id=None: SimpleNamespace(text=message.text),
    )
    monkeypatch.setattr(adapter, "_clean_bot_trigger_text", lambda text: text)
    monkeypatch.setattr(adapter, "_enqueue_text_event", lambda event: enqueued.append(event.text))

    update = SimpleNamespace(update_id=200, message=_dm_message(text="hello"))

    await adapter._handle_text_message(update, SimpleNamespace())
    await adapter._handle_text_message(update, SimpleNamespace())

    assert enqueued == ["hello"]


def test_build_message_event_parses_group_topic_reply_and_update_id():
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="fake-token",
            extra={
                "group_topics": [
                    {
                        "chat_id": "-100123",
                        "topics": [
                            {"thread_id": "7", "name": "Ops", "skill": "ops-runbook"}
                        ],
                    }
                ]
            },
        )
    )
    message = SimpleNamespace(
        text="status?",
        caption=None,
        message_id=55,
        message_thread_id=7,
        is_topic_message=True,
        chat_id=-100123,
        chat=SimpleNamespace(
            id=-100123,
            type=ChatType.SUPERGROUP,
            title="Ops Group",
            full_name=None,
            is_forum=True,
        ),
        from_user=SimpleNamespace(id=321, full_name="Operator"),
        reply_to_message=SimpleNamespace(message_id=54, text="full prior message", caption=None),
        quote=SimpleNamespace(text="selected quote"),
        date=None,
    )

    event = adapter._build_message_event(message, MessageType.TEXT, update_id=202)

    assert event.text == "status?"
    assert event.message_id == "55"
    assert event.platform_update_id == 202
    assert event.reply_to_message_id == "54"
    assert event.reply_to_text == "selected quote"
    assert event.source.chat_id == "-100123"
    assert event.source.chat_type == "group"
    assert event.source.thread_id == "7"
    assert event.source.chat_topic == "Ops"
    assert event.auto_skill == "ops-runbook"


@pytest.mark.asyncio
async def test_command_handler_routes_normalized_event_to_hermes_agent(monkeypatch):
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    monkeypatch.setattr(adapter, "_should_process_message", lambda message, is_command=False: is_command)

    update = SimpleNamespace(update_id=201, message=_dm_message(text="/help"))

    await adapter._handle_command(update, SimpleNamespace())

    adapter.handle_message.assert_awaited_once()
    handle_call = adapter.handle_message.await_args
    assert handle_call is not None
    event = handle_call.args[0]
    assert event.text == "/help"
    assert event.message_type is MessageType.COMMAND
    assert event.platform_update_id == 201
    assert event.source.chat_id == "123"


@pytest.mark.asyncio
async def test_unsupported_user_payload_gets_clear_user_facing_reply(monkeypatch):
    adapter = _make_adapter()
    adapter.send = AsyncMock()
    monkeypatch.setattr(adapter, "_should_process_message", lambda message, is_command=False: True)

    update = SimpleNamespace(
        update_id=300,
        message=_dm_message(text=None, contact=SimpleNamespace(phone_number="example-phone")),
    )

    await adapter._handle_unsupported_message(update, SimpleNamespace())

    adapter.send.assert_awaited_once()
    send_call = adapter.send.await_args
    assert send_call is not None
    chat_id, content = send_call.args[:2]
    assert chat_id == "123"
    assert "contact card" in content
    assert "can't process" in content
    assert send_call.kwargs["reply_to"] == "42"


@pytest.mark.asyncio
async def test_unsupported_service_payload_is_logged_not_sent(monkeypatch):
    adapter = _make_adapter()
    adapter.send = AsyncMock()
    monkeypatch.setattr(adapter, "_should_process_message", lambda message, is_command=False: True)

    update = SimpleNamespace(
        update_id=301,
        message=_dm_message(text=None, new_chat_members=[SimpleNamespace(id=999)]),
    )

    await adapter._handle_unsupported_message(update, SimpleNamespace())

    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_formats_markdown_and_falls_back_to_plain_text(monkeypatch):
    adapter = _make_adapter()
    calls = []

    class FakeBot:
        async def send_message(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("parse_mode") is not None:
                raise Exception("can't parse MarkdownV2")
            return SimpleNamespace(message_id=777)

    adapter._bot = FakeBot()
    monkeypatch.setattr(adapter, "send_typing", AsyncMock())

    result = await adapter.send("123", "**done** (ok)", reply_to="42")

    assert result.success is True
    assert result.message_id == "777"
    assert len(calls) == 2
    assert calls[0]["parse_mode"] is not None
    assert calls[0]["text"] == "*done* \\(ok\\)"
    assert calls[0]["reply_to_message_id"] == 42
    assert calls[1]["parse_mode"] is None
    assert calls[1]["text"] == "done (ok)"
    assert calls[1]["reply_to_message_id"] == 42
