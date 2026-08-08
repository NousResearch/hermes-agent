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


def _forwarded_photo_event(
    source: SessionSource,
    path: str = "/tmp/forwarded-post.jpg",
) -> MessageEvent:
    return MessageEvent(
        text="Forwarded post body",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=[path],
        media_types=["image/jpeg"],
        forward_origin={
            "type": "channel",
            "chat_name": "AI Channel",
            "chat_username": "ai_channel",
            "date": "2026-07-28T09:02:43+00:00",
        },
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
    adapter._startup_batch_events = {}
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._drop_delayed_deliveries = False
    adapter._text_batch_delay_seconds = 0.3
    adapter._text_batch_split_delay_seconds = 1.0
    adapter._apply_topic_recovery = lambda _event: None
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


def test_media_filter_includes_video_note(monkeypatch):
    class _Filter:
        def __init__(self, names):
            self.names = set(names)

        def __or__(self, other):
            return _Filter(self.names | other.names)

    fake_filters = SimpleNamespace(
        PHOTO=_Filter({"photo"}),
        VIDEO=_Filter({"video"}),
        AUDIO=_Filter({"audio"}),
        VOICE=_Filter({"voice"}),
        VIDEO_NOTE=_Filter({"video_note"}),
        Document=SimpleNamespace(ALL=_Filter({"document"})),
        Sticker=SimpleNamespace(ALL=_Filter({"sticker"})),
    )
    monkeypatch.setattr("plugins.platforms.telegram.adapter.filters", fake_filters)

    media_filter = TelegramAdapter._media_message_filter()

    assert "video_note" in media_filter.names


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


def test_telegram_forward_origin_extraction_channel_origin():
    adapter = _make_adapter()
    message = SimpleNamespace(
        is_automatic_forward=False,
        forward_origin=SimpleNamespace(
            type="channel",
            date=datetime(2026, 7, 28, 9, 2, 43, tzinfo=timezone.utc),
            chat=SimpleNamespace(
                id=-1003091706822,
                title="AI Channel",
                full_name=None,
                username="ai_channel",
            ),
            author_signature="Editor",
            message_id=382,
        ),
    )

    assert adapter._extract_forward_origin(message) == {
        "type": "channel",
        "date": "2026-07-28T09:02:43+00:00",
        "chat_name": "AI Channel",
        "chat_id": "-1003091706822",
        "chat_username": "ai_channel",
        "author_signature": "Editor",
        "message_id": "382",
    }


@pytest.mark.asyncio
async def test_multiple_forwarded_texts_preserve_each_origin_in_order():
    source = _source()
    adapter = _make_adapter()
    first = MessageEvent(
        text="first",
        message_type=MessageType.TEXT,
        source=source,
        forward_origin={"type": "user", "sender_name": "Alice"},
    )
    second = MessageEvent(
        text="second",
        message_type=MessageType.TEXT,
        source=source,
        forward_origin={"type": "user", "sender_name": "Bob"},
    )

    adapter._enqueue_text_event(first)
    adapter._enqueue_text_event(second)
    batch = adapter._pending_text_batches[adapter._text_batch_key(first)]

    assert batch.forward_origin is None
    assert batch.text == (
        "[Forwarded message | From: Alice]\n\nfirst\n"
        "[Forwarded message | From: Bob]\n\nsecond"
    )
    for task in adapter._pending_text_batch_tasks.values():
        task.cancel()


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
async def test_forwarded_photo_keeps_origin_on_forwarded_payload_not_user_question():
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    runner = _make_runner(adapter)
    adapter._pending_photo_batches[f"{session_key}:photo-burst"] = _forwarded_photo_event(source)

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

    assert event.forward_origin is None
    assert message_text.startswith("Is this a real study?")
    assert (
        "[Forwarded message | Chat: AI Channel (@ai_channel) | "
        "Date: 2026-07-28T09:02:43+00:00]"
    ) in message_text
    assert message_text.index("Is this a real study?") < message_text.index("[Forwarded message |")
    assert message_text.endswith("Forwarded post body")
    assert event.media_urls == ["/tmp/forwarded-post.jpg"]


@pytest.mark.asyncio
async def test_photo_burst_preserves_late_forward_origin_inline():
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    key = f"{session_key}:photo-burst"

    adapter._enqueue_photo_event(key, _photo_event(source, "/tmp/plain.jpg"))
    adapter._enqueue_photo_event(key, _forwarded_photo_event(source, "/tmp/forwarded.jpg"))

    batch = adapter._pending_photo_batches[key]
    assert batch.forward_origin is None
    assert batch.media_urls == ["/tmp/plain.jpg", "/tmp/forwarded.jpg"]
    assert "[Forwarded message | Chat: AI Channel (@ai_channel)" in batch.text
    adapter._pending_photo_batch_tasks[key].cancel()


@pytest.mark.asyncio
async def test_album_flush_waits_for_other_downloads():
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    adapter.MEDIA_GROUP_WAIT_SECONDS = 0
    adapter.handle_message = AsyncMock()
    adapter._media_group_events["album-1"] = _photo_event(source)
    adapter._media_downloads_in_progress_by_session[session_key] = 1

    async def finish_second_download():
        await asyncio.sleep(0.02)
        adapter._media_downloads_in_progress_by_session.pop(session_key, None)

    producer = asyncio.create_task(finish_second_download())
    await adapter._flush_media_group_event("album-1")
    await producer

    adapter.handle_message.assert_awaited_once()


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
async def test_non_photo_tracker_stays_live_until_startup_registration(monkeypatch):
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    runner = _make_runner(adapter)
    runner._running_agents = {session_key: _AGENT_PENDING_SENTINEL}
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner._busy_ack_ts = {}
    runner._running_agents_ts = {}
    monkeypatch.setattr(runner, "_startup_media_grace_seconds", lambda: 0.2)

    adapter._is_user_authorized_from_message = lambda _message: True
    adapter._should_process_message = lambda _message: True
    adapter._media_message_type = lambda _message: MessageType.VOICE
    adapter._build_message_event = lambda _message, _type, update_id=None: MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        platform_update_id=update_id,
    )
    adapter._apply_telegram_group_observe_attribution = lambda event: event

    async def delayed_registration(event):
        await asyncio.sleep(0.05)
        handled = await runner._handle_active_session_busy_message(event, session_key)
        assert handled is True

    adapter.handle_message = delayed_registration

    class _File:
        async def download_as_bytearray(self):
            await asyncio.sleep(0.01)
            return bytearray(b"voice")

    class _Voice:
        file_size = 5

        async def get_file(self):
            return _File()

    msg = SimpleNamespace(
        caption=None,
        sticker=None,
        photo=None,
        voice=_Voice(),
        audio=None,
        video_note=None,
        video=None,
        document=None,
        media_group_id=None,
    )
    update = SimpleNamespace(message=msg, update_id=41)
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.cache_audio_from_bytes",
        lambda _data, ext: "/tmp/registered-voice.ogg",
    )

    producer = asyncio.create_task(
        adapter._handle_media_message(update, SimpleNamespace())
    )
    await asyncio.sleep(0)
    merged = await runner._merge_startup_media_followups(
        _text_event(source),
        source,
        session_key,
    )
    await producer

    assert merged.message_type is MessageType.VOICE
    assert merged.media_urls == ["/tmp/registered-voice.ogg"]
    assert adapter._startup_batch_events == {}
    assert adapter._media_downloads_in_progress_by_session == {}


def test_media_session_key_applies_topic_recovery_and_adapter_profile():
    source = _source()
    source.profile = None
    source.thread_id = "raw-topic"
    adapter = _make_adapter()
    adapter._gateway_profile_name = "coder"
    adapter._apply_topic_recovery = lambda event: setattr(event.source, "thread_id", "recovered-topic")
    event = _photo_event(source)

    key = adapter._event_session_key(event)

    assert key == "agent:coder:telegram:dm:273403055:recovered-topic"
    assert event.source.profile == "coder"
    assert event.source.thread_id == "recovered-topic"


def test_text_batch_key_uses_adapter_profile_and_topic_recovery():
    source = _source()
    source.profile = None
    source.thread_id = "raw-topic"
    adapter = _make_adapter()
    adapter._gateway_profile_name = "coder"
    adapter._apply_topic_recovery = lambda event: setattr(event.source, "thread_id", "recovered-topic")

    key = adapter._text_batch_key(_forwarded_text_event(source))

    assert key == "agent:coder:telegram:dm:273403055:recovered-topic"


@pytest.mark.asyncio
async def test_base_adapter_busy_guard_uses_secondary_profile_key():
    source = _source()
    source.profile = None
    adapter = _make_adapter()
    adapter._gateway_profile_name = "coder"
    adapter._message_handler = AsyncMock()
    adapter._busy_session_handler = AsyncMock(return_value=True)
    adapter._active_sessions = {
        "agent:coder:telegram:dm:273403055:363402": asyncio.Event()
    }
    adapter._session_tasks = {}
    adapter._background_tasks = set()
    adapter._heal_stale_session_lock = lambda _key: None

    await adapter.handle_message(_forwarded_photo_event(source))

    adapter._busy_session_handler.assert_awaited_once()
    assert adapter._busy_session_handler.await_args.args[1] == (
        "agent:coder:telegram:dm:273403055:363402"
    )


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
async def test_gateway_consumes_forwarded_text_from_telegram_debounce():
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    runner = _make_runner(adapter)
    adapter._enqueue_text_event(_forwarded_text_event(source))

    event = await runner._merge_startup_media_followups(
        _text_event(source),
        source,
        session_key,
    )

    assert "[Forwarded message | From: Alina" in event.text
    assert "second forwarded text" in event.text
    assert session_key not in adapter._pending_text_batches
    assert session_key not in adapter._pending_text_batch_tasks


@pytest.mark.asyncio
async def test_gateway_does_not_consume_generic_fifo_document_head():
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    runner = _make_runner(adapter)
    queued = _document_event(source)
    adapter._pending_messages[session_key] = queued

    event = await runner._merge_startup_media_followups(
        _text_event(source),
        source,
        session_key,
    )

    assert event.message_type == MessageType.TEXT
    assert event.media_urls == []
    assert adapter._pending_messages[session_key] is queued


@pytest.mark.asyncio
async def test_gateway_merges_forwarded_text_batch_before_first_model_call():
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    runner = _make_runner(adapter)
    adapter._enqueue_text_event(_forwarded_text_event(source))

    event = await runner._merge_startup_media_followups(
        _text_event(source),
        source,
        session_key,
    )

    assert event.message_type == MessageType.TEXT
    assert event.forward_origin is None
    assert event.text.startswith("Is this a real study?")
    assert "[Forwarded message | From: Alina | Date: 2026-06-14T21:03:26+00:00]" in event.text
    assert "second forwarded text" in event.text
    assert session_key not in adapter._pending_text_batches
    assert session_key not in adapter._pending_text_batch_tasks


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
    assert adapter._startup_batch_events[session_key].text.endswith("second forwarded text")
    assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_startup_forward_does_not_absorb_existing_fifo_head():
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    runner = _make_runner(adapter)
    queued = MessageEvent(text="queued-first", message_type=MessageType.TEXT, source=source)
    adapter._pending_messages[session_key] = queued
    runner._running_agents = {session_key: _AGENT_PENDING_SENTINEL}
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner._busy_ack_ts = {}
    runner._running_agents_ts = {}

    handled = await runner._handle_active_session_busy_message(
        _forwarded_photo_event(source),
        session_key,
    )

    assert handled is True
    assert adapter._pending_messages[session_key] is queued
    assert adapter._pending_messages[session_key].text == "queued-first"
    assert adapter._startup_batch_events[session_key].media_urls == ["/tmp/forwarded-post.jpg"]


def test_startup_slot_preserves_each_forwarded_media_origin_inline():
    source = _source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    ordinary = MessageEvent(
        text="ordinary",
        message_type=MessageType.DOCUMENT,
        source=source,
        media_urls=["/tmp/ordinary.pdf"],
        media_types=["application/pdf"],
    )
    alice = MessageEvent(
        text="alice file",
        message_type=MessageType.DOCUMENT,
        source=source,
        media_urls=["/tmp/alice.pdf"],
        media_types=["application/pdf"],
        forward_origin={"type": "user", "sender_name": "Alice"},
    )
    bob = MessageEvent(
        text="bob file",
        message_type=MessageType.DOCUMENT,
        source=source,
        media_urls=["/tmp/bob.pdf"],
        media_types=["application/pdf"],
        forward_origin={"type": "user", "sender_name": "Bob"},
    )

    adapter.queue_startup_batch_event(session_key, ordinary)
    adapter.queue_startup_batch_event(session_key, alice)
    adapter.queue_startup_batch_event(session_key, bob)
    batch = adapter._startup_batch_events[session_key]

    assert batch.forward_origin is None
    assert batch.text.count("[Forwarded message | From: Alice]") == 1
    assert batch.text.count("[Forwarded message | From: Bob]") == 1
    assert batch.media_urls == ["/tmp/ordinary.pdf", "/tmp/alice.pdf", "/tmp/bob.pdf"]


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
