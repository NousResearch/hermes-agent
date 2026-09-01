import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="273403055",
        chat_type="dm",
        user_id="273403055",
        user_name="Maxim E.",
        thread_id="439655",
    )


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(enabled=True, token="fake")
    adapter.platform = Platform.TELEGRAM
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._pending_photo_batches = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._pending_voice_batches = {}
    adapter._pending_voice_batch_tasks = {}
    adapter._media_group_events = {}
    adapter._media_group_tasks = {}
    adapter._drop_delayed_deliveries = False
    adapter._media_batch_delay_seconds = 0.01
    adapter._text_batch_delay_seconds = 0.3
    adapter._text_batch_split_delay_seconds = 1.0
    adapter._apply_topic_recovery = lambda _event: None
    adapter._gateway_profile_name = None
    adapter._dm_topics = {}
    adapter._max_doc_bytes = 20 * 1024 * 1024
    adapter._should_process_message = lambda _message: True
    adapter._is_user_authorized_from_message = lambda _message: True
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    return adapter


def _telegram_voice_message(message_id: int, path: str, *, download_delay: float = 0.0):
    class Voice:
        file_size = 5

        async def get_file(self):
            async def download_as_bytearray():
                if download_delay:
                    await asyncio.sleep(download_delay)
                return b"voice"

            return SimpleNamespace(download_as_bytearray=download_as_bytearray)

    return SimpleNamespace(
        message_id=message_id,
        text=None,
        caption=None,
        date=None,
        chat=SimpleNamespace(
            id=273403055,
            type="private",
            title=None,
            full_name="Maxim E.",
        ),
        from_user=SimpleNamespace(id=273403055, full_name="Maxim E.", is_bot=False),
        reply_to_message=None,
        forum_topic_created=None,
        voice=Voice(),
        audio=None,
        photo=None,
        video=None,
        document=None,
        sticker=None,
        media_group_id=None,
        message_thread_id=439655,
        is_topic_message=True,
        forward_origin=SimpleNamespace(type="user", sender_user=SimpleNamespace(full_name=path)),
        is_automatic_forward=False,
    )


def test_build_message_event_preserves_forward_origin():
    adapter = _make_adapter()
    message = _telegram_voice_message(1, "Alice")

    event = adapter._build_message_event(message, MessageType.VOICE, update_id=10)

    assert event.forward_origin == {
        "type": "user",
        "sender_name": "Alice",
    }


@pytest.mark.asyncio
async def test_batch_forward_two_voice_notes_are_one_event_and_both_reach_stt(monkeypatch):
    adapter = _make_adapter()
    handled = []
    adapter.handle_message = AsyncMock(side_effect=lambda event: handled.append(event))
    cached_paths = iter(["/tmp/voice-one.ogg", "/tmp/voice-two.ogg"])
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.cache_audio_from_bytes",
        lambda _data, ext=".ogg": next(cached_paths),
    )

    # Deliberately complete the newer Telegram update first: final order must
    # follow source message_id, not network/download completion timing.
    first = SimpleNamespace(message=_telegram_voice_message(2, "Bob"), update_id=102)
    second = SimpleNamespace(
        message=_telegram_voice_message(
            1,
            "Alice",
            download_delay=adapter._media_batch_delay_seconds + 0.1,
        ),
        update_id=101,
    )

    await asyncio.gather(
        adapter._handle_media_message(first, SimpleNamespace()),
        adapter._handle_media_message(second, SimpleNamespace()),
    )
    await asyncio.sleep(adapter._media_batch_delay_seconds + 0.2)

    assert len(handled) == 1
    batch = handled[0]
    assert batch.message_id == "1"
    assert batch.media_urls == ["/tmp/voice-two.ogg", "/tmp/voice-one.ogg"]
    assert batch.media_types == ["audio/ogg", "audio/ogg"]
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    transcripts = iter(["первое", "второе"])
    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda _path, _model, _source: {
            "success": True,
            "transcript": next(transcripts),
            "provider": "deepgram",
        },
    )

    enriched, successful = await runner._enrich_message_with_transcription(
        batch.text,
        batch.media_urls,
    )

    assert successful == ["первое", "второе"]
    assert '"первое"' in enriched
    assert '"второе"' in enriched
