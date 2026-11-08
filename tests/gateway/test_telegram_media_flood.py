"""Flood-refused native media uploads are re-uploaded after the window, never degraded to text.

Drives the base dispatcher (``BasePlatformAdapter._deliver_media_attachments``, the MEDIA-tag path
every turn uses) into the real ``TelegramAdapter.send_document``; only the Bot API object is fake.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.waiting import async_wait_until

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import Platform, SessionSource
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


class RetryAfter(Exception):
    """PTB's RetryAfter shape: ``retry_after`` attribute + the Bot API wording."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


@pytest.fixture()
def flooded(monkeypatch, tmp_path):
    # Shrink the inline cap so a 0.2 s penalty counts as "long": the deferred path runs in real time.
    monkeypatch.setattr(tg_adapter, "_FLOOD_INLINE_WAIT_CAP_SECS", 0.05)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = SimpleNamespace(send_document=AsyncMock(), send_message=AsyncMock())
    adapter.send = AsyncMock()  # any text notice would land here
    doc = tmp_path / "report.txt"
    doc.write_text("hello")
    return adapter, doc


async def _deliver(adapter, doc):
    results = []
    event = MessageEvent(text="", source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345"))
    await adapter._deliver_media_attachments(
        event, [(str(doc), False)], [], force_document_attachments=False, human_delay=0,
        metadata={}, record_delivery=results.append)
    return results


@pytest.mark.asyncio
async def test_flood_refused_document_is_reuploaded_natively_after_the_window(flooded):
    adapter, doc = flooded
    adapter._bot.send_document.side_effect = [RetryAfter(0.2), SimpleNamespace(message_id=77)]

    results = await _deliver(adapter, doc)

    assert [r.error for r in results] == ["flood_control:0.2"]  # canonical fail-closed result
    adapter.send.assert_not_awaited()  # no text apology fired into the penalty
    await async_wait_until(lambda: adapter._bot.send_document.await_count == 2, timeout=5,
                           what="one native re-upload after the window")
    await asyncio.sleep(0)
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_document_still_flooded_after_the_window_reports_once(flooded):
    adapter, doc = flooded
    adapter._bot.send_document.side_effect = RetryAfter(0.2)

    await _deliver(adapter, doc)
    await async_wait_until(lambda: adapter.send.await_count, timeout=5, what="failure notice after the second window")

    assert adapter._bot.send_document.await_count == 2  # never a third request
    adapter.send.assert_awaited_once()  # the failure notice, once, after the second window
    assert "Couldn't deliver" in adapter.send.await_args.args[1]


def _ok(message_id=1):
    return SimpleNamespace(message_id=message_id)


@pytest.mark.asyncio
async def test_deferred_voice_survives_the_caller_deleting_its_tts_file(flooded, tmp_path):
    """Auto-TTS deletes the clip as soon as send_voice returns; the deferred re-upload still lands."""
    adapter, _doc = flooded
    clip = tmp_path / "reply.ogg"
    clip.write_bytes(b"OggS-voice")
    uploads = []

    async def send_voice(**kwargs):
        uploads.append(kwargs["voice"].read())
        if len(uploads) == 1:
            raise RetryAfter(0.2)
        return _ok(5)

    adapter._bot.send_voice = send_voice
    result = await adapter.send_voice("12345", str(clip))
    clip.unlink()  # what _send_voice_reply and the base TTS loop do on return

    assert result.error == "flood_control:0.2"
    await async_wait_until(lambda: len(uploads) == 2, timeout=5, what="deferred voice re-upload")
    assert uploads == [b"OggS-voice", b"OggS-voice"]
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_deferred_image_document_fallback_survives_the_temp_jpeg_cleanup(flooded, tmp_path, monkeypatch):
    """A compressed image refused as a photo falls back to send_document on the temp JPEG, which
    send_image_file deletes on return; the deferred document upload must still have the bytes."""
    adapter, _doc = flooded
    image = tmp_path / "chart.png"
    image.write_bytes(b"png-original")

    def compress(path):
        out = tmp_path / f"tg_compress_{len(list(tmp_path.glob('tg_compress_*')))}.jpg"
        out.write_bytes(b"jpeg-compressed")
        return str(out)

    monkeypatch.setattr(adapter, "_compress_image_to_jpeg", compress)
    adapter._bot.send_photo = AsyncMock(side_effect=Exception("Bad Request: PHOTO_INVALID_DIMENSIONS"))
    documents = []

    async def send_document(**kwargs):
        documents.append(kwargs["document"].read())
        if len(documents) == 1:
            raise RetryAfter(0.2)
        return _ok(9)

    adapter._bot.send_document = send_document
    result = await adapter.send_image_file("12345", str(image))

    assert result.error == "flood_control:0.2"
    await async_wait_until(lambda: len(documents) == 2, timeout=5, what="deferred document re-upload")
    assert documents == [b"jpeg-compressed", b"jpeg-compressed"]
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_deferred_upload_waits_out_a_window_a_sibling_extended(flooded, tmp_path):
    """Two files deferred to the same window: the first re-upload is refused again (longer window);
    the second must wait for the new deadline instead of spending its last attempt inside it."""
    adapter, doc = flooded
    second = tmp_path / "second.txt"
    second.write_text("two")
    loop = asyncio.get_running_loop()
    calls = []
    plan = [RetryAfter(0.2), RetryAfter(2.0), _ok(2), _ok(3)]

    async def send_document(**kwargs):
        calls.append(loop.time())
        outcome = plan.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    adapter._bot.send_document = send_document
    await _deliver(adapter, doc)       # refused → deferred 0.2 s
    await _deliver(adapter, second)    # inside the cooldown → deferred without a request

    await async_wait_until(lambda: len(calls) == 3, timeout=8, what="sibling re-upload")
    assert len(calls) == 3
    # The 1 s per-chat outbound budget spaces requests anyway; 2 s outlasts it.
    assert calls[2] - calls[1] >= 1.9  # no request inside the extended 2 s window


@pytest.mark.asyncio
async def test_flood_on_downloaded_image_upload_defers_instead_of_text_fallback(flooded, monkeypatch):
    """URL send refused for a non-flood reason, then the downloaded-bytes upload is flood-refused:
    the image is re-sent natively after the window, never replaced by its URL as text."""
    adapter, _doc = flooded
    from tools import url_safety

    class _Resp:
        content = b"image-bytes"

        def raise_for_status(self):
            return None

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url):
            return _Resp()

    monkeypatch.setattr(url_safety, "is_safe_url", lambda *a, **k: True)
    monkeypatch.setattr(url_safety, "create_ssrf_safe_async_client", lambda **kw: _Client())
    url_refused = Exception("Bad Request: wrong file identifier/HTTP URL specified")
    adapter._bot.send_photo = AsyncMock(side_effect=[url_refused, RetryAfter(0.2), url_refused, _ok(4)])

    result = await adapter.send_image("12345", "https://example.com/pic.png")

    assert result.error == "flood_control:0.2"
    adapter.send.assert_not_awaited()  # no URL-as-text into the penalty
    await async_wait_until(lambda: adapter._bot.send_photo.await_count == 4, timeout=5,
                           what="deferred native image re-send")
    await asyncio.sleep(0)
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_every_native_media_entry_waits_out_a_known_flood_window(flooded, tmp_path, monkeypatch):
    """A document's refusal arms the chat's window; a fresh voice clip, URL photo, animation and a
    queued sibling document then make no request inside it and each go out once after it."""
    from tools import url_safety
    monkeypatch.setattr(url_safety, "is_safe_url", lambda *a, **k: True)
    adapter, doc = flooded
    clip = tmp_path / "reply.ogg"
    clip.write_bytes(b"OggS-voice")
    first_request = asyncio.Event()
    loop = asyncio.get_running_loop()
    calls = []

    def recorder(kind, refuse=False):
        async def send(**kwargs):
            calls.append((kind, loop.time()))
            first_request.set()
            if refuse and len(calls) == 1:
                await asyncio.sleep(0)  # the sibling queues on the chat lock meanwhile
                raise RetryAfter(0.3)
            return _ok()
        return send

    adapter._bot.send_document = recorder("document", refuse=True)
    adapter._bot.send_voice = recorder("voice")
    adapter._bot.send_photo = recorder("photo")
    adapter._bot.send_animation = recorder("animation")
    first = asyncio.create_task(adapter.send_document("12345", str(doc)))
    await first_request.wait()
    sibling = asyncio.create_task(adapter.send_document("12345", str(doc)))
    refused_at = loop.time()
    results = [await first, await sibling,
               await adapter.send_voice("12345", str(clip)),
               await adapter.send_image("12345", "https://example.com/pic.png"),
               await adapter.send_animation("12345", "https://example.com/a.gif")]

    assert [r.success for r in results] == [False] * 5
    assert all(r.raw_response["deferred_upload"] for r in results)
    assert calls == [("document", calls[0][1])]  # nothing else inside the window
    await async_wait_until(lambda: len(calls) == 6, timeout=10, what="every deferred upload once")
    assert sorted(kind for kind, _ in calls) == ["animation", "document", "document", "document", "photo", "voice"]
    assert all(at - refused_at >= 0.29 for _, at in calls[1:])
    adapter.send.assert_not_awaited()
