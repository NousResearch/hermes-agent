import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource, build_session_key


class _DummyTask:
    def __init__(self):
        self.cancelled = False

    def done(self):
        return False

    def cancel(self):
        self.cancelled = True


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="273403055",
        chat_type="dm",
        user_id="273403055",
        user_name="Maxim E.",
        thread_id="363402",
    )


def _text_event(source: SessionSource) -> MessageEvent:
    return MessageEvent(
        text="Is this a real study?",
        message_type=MessageType.TEXT,
        source=source,
    )


def _photo_event(source: SessionSource, path: str = "/tmp/alcohol-study.jpg") -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=[path],
        media_types=["image/jpeg"],
    )


def _document_event(
    source: SessionSource,
    path: str = "/root/.hermes/cache/documents/doc_abcd_guide.docx",
) -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.DOCUMENT,
        source=source,
        media_urls=[path],
        media_types=["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
        forward_origin={"type": "user", "sender_name": "Alice"},
    )


def _forwarded_text_event(source: SessionSource) -> MessageEvent:
    return MessageEvent(
        text="sk-or-v1-example\nsecond forwarded text",
        message_type=MessageType.TEXT,
        source=source,
        forward_origin={
            "type": "user",
            "sender_name": "Alina",
            "date": "2026-06-14T21:03:26+00:00",
        },
    )


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(enabled=True, token="fake")
    adapter._pending_messages = {}
    adapter._pending_photo_batches = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._media_group_events = {}
    adapter._media_group_tasks = {}
    adapter._media_downloads_in_progress_by_session = {}
    return adapter


def _make_runner(adapter: TelegramAdapter) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._pending_native_image_paths_by_session = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda **_kwargs: "native"
    runner._is_user_authorized = lambda _source: True
    return runner


def test_telegram_forward_origin_extraction_user_origin():
    adapter = _make_adapter()
    message = SimpleNamespace(
        is_automatic_forward=False,
        forward_origin=SimpleNamespace(
            type="user",
            date=datetime(2026, 6, 14, 9, 0, tzinfo=timezone.utc),
            sender_user=SimpleNamespace(
                id=42,
                full_name="Skippy",
                username="skippy_bot",
            ),
        ),
    )

    assert adapter._extract_forward_origin(message) == {
        "type": "user",
        "date": "2026-06-14T09:00:00+00:00",
        "sender_name": "Skippy",
        "sender_id": "42",
        "sender_username": "skippy_bot",
    }


@pytest.mark.asyncio
async def test_gateway_merges_buffered_photo_batch_before_image_routing():
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    runner = _make_runner(adapter)
    task = _DummyTask()
    adapter._pending_photo_batches[f"{session_key}:photo-burst"] = _photo_event(source)
    adapter._pending_photo_batch_tasks[f"{session_key}:photo-burst"] = task

    event = await runner._merge_startup_media_followups(
        _text_event(source),
        source,
        session_key,
    )
    message_text = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert message_text == "Is this a real study?"
    assert event.message_type == MessageType.PHOTO
    assert event.media_urls == ["/tmp/alcohol-study.jpg"]
    assert runner._consume_pending_native_image_paths(session_key) == ["/tmp/alcohol-study.jpg"]
    assert adapter._pending_photo_batches == {}
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_gateway_waits_for_in_progress_photo_download(monkeypatch):
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    runner = _make_runner(adapter)
    adapter._media_downloads_in_progress_by_session[session_key] = 1
    monkeypatch.setattr(runner, "_startup_media_grace_seconds", lambda: 0.2)

    async def finish_download():
        await asyncio.sleep(0.02)
        adapter._pending_photo_batches[f"{session_key}:photo-burst"] = _photo_event(
            source,
            "/tmp/late-photo.jpg",
        )
        adapter._media_downloads_in_progress_by_session.pop(session_key, None)

    producer = asyncio.create_task(finish_download())
    event = await runner._merge_startup_media_followups(
        _text_event(source),
        source,
        session_key,
    )
    await producer

    assert event.message_type == MessageType.PHOTO
    assert event.text == "Is this a real study?"
    assert event.media_urls == ["/tmp/late-photo.jpg"]
    assert adapter._pending_photo_batches == {}


@pytest.mark.asyncio
async def test_gateway_text_only_fast_path_does_not_wait_without_pending_media(monkeypatch):
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    runner = _make_runner(adapter)
    sleep = AsyncMock()
    monkeypatch.setattr("gateway.run.asyncio.sleep", sleep)

    event = await runner._merge_startup_media_followups(
        _text_event(source),
        source,
        session_key,
    )

    assert event.message_type == MessageType.TEXT
    assert event.text == "Is this a real study?"
    assert event.media_urls == []
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_merges_priority_queued_document_followup_before_first_model_call():
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    runner = _make_runner(adapter)
    adapter._pending_messages[session_key] = _document_event(source)

    event = await runner._merge_startup_media_followups(
        _text_event(source),
        source,
        session_key,
    )
    message_text = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert event.message_type == MessageType.DOCUMENT
    assert event.media_urls == ["/root/.hermes/cache/documents/doc_abcd_guide.docx"]
    assert event.forward_origin == {"type": "user", "sender_name": "Alice"}
    assert "The user sent a document: 'guide.docx'" in message_text
    assert "Is this a real study?" in message_text
    assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_gateway_merges_forwarded_text_batch_before_first_model_call():
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    runner = _make_runner(adapter)
    adapter._pending_messages[session_key] = _forwarded_text_event(source)

    event = await runner._merge_startup_media_followups(
        _text_event(source),
        source,
        session_key,
    )

    assert event.message_type == MessageType.TEXT
    assert event.forward_origin is None
    assert event.text.startswith("Is this a real study?")
    assert "[Forwarded message | From: Alina | Date: 2026-06-14T21:03:26+00:00]" in event.text
    assert "sk-or-v1-example" in event.text
    assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_gateway_queues_startup_forwarded_text_batch_without_interrupt_ack():
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    runner = _make_runner(adapter)
    runner._running_agents = {session_key: _AGENT_PENDING_SENTINEL}
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner._busy_ack_ts = {}
    runner._running_agents_ts = {}

    handled = await runner._handle_active_session_busy_message(
        _forwarded_text_event(source),
        session_key,
    )

    assert handled is True
    assert adapter._pending_messages[session_key].text == "sk-or-v1-example\nsecond forwarded text"


@pytest.mark.asyncio
async def test_forwarded_context_is_rendered_before_inbound_text():
    source = _source()
    adapter = _make_adapter()
    runner = _make_runner(adapter)
    event = MessageEvent(
        text="original text",
        message_type=MessageType.TEXT,
        source=source,
        forward_origin={
            "type": "user",
            "sender_name": "Skippy",
            "sender_username": "skippy_bot",
            "date": "2026-06-14T09:00:00+00:00",
        },
    )

    message_text = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert message_text.startswith(
        "[Forwarded message | From: Skippy (@skippy_bot) | Date: 2026-06-14T09:00:00+00:00]"
    )
    assert message_text.endswith("original text")
