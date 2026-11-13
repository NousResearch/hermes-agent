"""Flood-refused native media uploads are re-uploaded after the window, never degraded to text.

Drives the base dispatcher (``BasePlatformAdapter._deliver_media_attachments``, the MEDIA-tag path
every turn uses) into the real ``TelegramAdapter.send_document``; only the Bot API object is fake.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.waiting import async_wait_until

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.gateway.platforms.event import MessageEvent
from hermes_agent.gateway.session import Platform, SessionSource
from hermes_agent.plugins.platforms.telegram import adapter as tg_adapter
from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter


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
    from hermes_agent.tools import url_safety

    class _Resp:
        headers: dict = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield b"image-bytes"

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url):
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
    from hermes_agent.tools import url_safety
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


@pytest.mark.asyncio
async def test_short_flood_wait_keeps_sibling_uploads_out_of_the_window(flooded, monkeypatch):
    """A refusal short enough to wait out inline still closes the chat: a sibling upload sent while
    the first sleeps makes no request until the window is over, and the first is retried once."""
    adapter, doc = flooded
    monkeypatch.setattr(tg_adapter, "_FLOOD_INLINE_WAIT_CAP_SECS", 5.0)  # 0.3 s is waited out inline
    loop = asyncio.get_running_loop()
    first_request = asyncio.Event()
    calls = []

    async def send_document(**kwargs):
        calls.append(loop.time())
        first_request.set()
        if len(calls) == 1:
            raise RetryAfter(0.3)
        return _ok(len(calls))

    adapter._bot.send_document = send_document
    first = asyncio.create_task(adapter.send_document("12345", str(doc)))
    await first_request.wait()
    sibling = await adapter.send_document("12345", str(doc))

    assert sibling.raw_response["deferred_upload"]  # refused locally, no request into the window
    assert (await first).success
    await async_wait_until(lambda: len(calls) == 3, timeout=10, what="retry and sibling re-upload")
    assert all(at - calls[0] >= 0.29 for at in calls[1:]), calls
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_during_the_private_copy_leaves_no_orphaned_attachment(flooded, tmp_path, monkeypatch):
    """Shutdown cancelling the send while its private copy is still being written: the copy thread
    finishes anyway, and the copy it produced is removed instead of left on disk with no owner."""
    import shutil
    import threading

    from hermes_agent.plugins.platforms.telegram import telegram_media_flood as flood

    adapter, doc = flooded
    monkeypatch.setattr(flood.tempfile, "tempdir", str(tmp_path))
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_copy = shutil.copy2

    def gated_copy(src, dst):
        entered.set()
        release.wait(10)
        try:
            return real_copy(src, dst)
        finally:
            finished.set()

    monkeypatch.setattr(flood.shutil, "copy2", gated_copy)
    adapter._bot.send_document = AsyncMock(side_effect=RetryAfter(30))
    sending = asyncio.create_task(adapter.send_document("12345", str(doc)))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        sending.cancel()
        await asyncio.gather(sending, return_exceptions=True)
        release.set()
        assert await asyncio.to_thread(finished.wait, 10)
        await async_wait_until(lambda: not list(tmp_path.glob("hermes-tg-flood-*")), timeout=5,
                               what="the finished copy removed")
        assert not adapter._background_tasks  # no upload was scheduled for the cancelled send
    finally:
        release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_failed_private_copy_still_posts_the_promised_notice(flooded, tmp_path, monkeypatch):
    """The private copy fails (disk full) and the caller deletes its temp file on return: nothing is
    left to re-upload, so the failure notice the result promised is posted once after the window."""
    from hermes_agent.plugins.platforms.telegram import telegram_media_flood as flood

    adapter, _doc = flooded
    clip = tmp_path / "reply.ogg"
    clip.write_bytes(b"OggS-voice")

    def disk_full(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(flood.shutil, "copy2", disk_full)
    adapter._bot.send_voice = AsyncMock(side_effect=RetryAfter(0.2))
    result = await adapter.send_voice("12345", str(clip), metadata={"thread_id": "7"})
    assert result.error == "flood_control:0.2"  # deferred, not the immediate base notice
    adapter.send.assert_not_awaited()
    clip.unlink()  # what auto-TTS does as soon as send_voice returns

    assert result.raw_response["failure_notice_sent"]
    await async_wait_until(lambda: adapter.send.await_count, timeout=5, what="post-window failure notice")
    await async_wait_until(lambda: not adapter._background_tasks, timeout=5, what="deferred task done")
    adapter.send.assert_awaited_once()
    assert "Couldn't deliver" in adapter.send.await_args.args[1]
    assert adapter.send.await_args.kwargs["metadata"] == {"thread_id": "7"}
    assert adapter._bot.send_voice.await_count == 1  # no upload of a file that no longer exists


@pytest.mark.asyncio
async def test_every_lost_file_in_a_flooded_batch_gets_its_notice(flooded, tmp_path):
    """A batch of documents still refused after the window: no request goes out inside a known
    window, and each lost file gets exactly one notice through the real send()."""
    adapter, _doc = flooded
    del adapter.send  # the real send(): a notice refused by a re-armed window must not be lost
    loop = asyncio.get_running_loop()
    uploads, notices = [], []

    async def send_document(**kwargs):
        uploads.append(adapter._send_flood_cooldown_remaining(kwargs["chat_id"]) is None)
        raise RetryAfter(1.0)

    async def send_message(**kwargs):
        notices.append(loop.time())
        return _ok(len(notices))

    adapter._bot = SimpleNamespace(send_document=send_document, send_message=send_message)
    for i in range(5):
        path = tmp_path / f"part{i}.txt"
        path.write_text(str(i))
        assert (await adapter.send_document("12345", str(path))).raw_response["failure_notice_sent"]
    await async_wait_until(lambda: not adapter._background_tasks, timeout=60, what="every deferred upload settled")

    assert all(uploads), "a request went out inside a known flood window"
    assert len(notices) == 5


@pytest.mark.asyncio
async def test_promised_notice_survives_a_sibling_refused_as_it_goes_out(flooded, tmp_path, monkeypatch):
    """The deferred task owes a notice (its file is gone); a sibling send is refused by Telegram right
    as that notice goes out. The notice must wait out the new window, not be refused locally and lost."""
    from hermes_agent.plugins.platforms.telegram import telegram_media_flood as flood

    adapter, _doc = flooded
    del adapter.send  # the real send(): its local refusal inside a window is what loses the notice
    clip = tmp_path / "reply.ogg"
    clip.write_bytes(b"OggS-voice")

    def disk_full(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(flood.shutil, "copy2", disk_full)
    notices = []

    async def send_message(**kwargs):
        if kwargs["text"] == "sibling":
            raise RetryAfter(1.0)
        notices.append(kwargs["text"])
        return _ok(len(notices))

    adapter._bot = SimpleNamespace(send_voice=AsyncMock(side_effect=RetryAfter(0.2)), send_message=send_message)
    notify = adapter._notify_media_delivery_failure

    async def notify_as_a_sibling_is_refused(*args, **kwargs):
        sibling = asyncio.create_task(adapter.send("12345", "sibling"))
        await asyncio.wait([sibling], timeout=0.5)  # holding the chat lock, the notice keeps it from going first
        await notify(*args, **kwargs)

    monkeypatch.setattr(adapter, "_notify_media_delivery_failure", notify_as_a_sibling_is_refused)
    await adapter.send_voice("12345", str(clip))
    clip.unlink()  # what auto-TTS does as soon as send_voice returns
    await async_wait_until(lambda: not adapter._background_tasks, timeout=10, what="deferred task done")

    assert len(notices) == 1 and "Couldn't deliver" in notices[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("open_checks,notices", [(4, 1), (5, 0)], ids=["clears-in-last-wait", "still-open"])
async def test_lost_upload_notice_checks_the_window_after_its_last_wait(flooded, tmp_path, monkeypatch,
                                                                        open_checks, notices):
    """The notice loop is capped at _MAX_WINDOW_WAITS waits, but a window that clears during the last
    wait still gets its notice; only a window open on the check after it gives up."""
    from hermes_agent.plugins.platforms.telegram import telegram_media_flood as flood

    adapter, _doc = flooded
    clip = tmp_path / "reply.ogg"
    clip.write_bytes(b"OggS-voice")
    monkeypatch.setattr(flood.shutil, "copy2", Mock(side_effect=OSError("No space left on device")))
    adapter._bot.send_voice = AsyncMock(side_effect=RetryAfter(0.2))
    lost, checks = [], []
    real_error, real_remaining = flood.logger.error, adapter._send_flood_cooldown_remaining

    def error(msg, *args, **kwargs):
        lost.append(msg)
        real_error(msg, *args, **kwargs)

    def remaining(chat_id):
        if not any("lost" in m for m in lost):
            return real_remaining(chat_id)
        checks.append(chat_id)
        return 0.01 if len(checks) <= open_checks else None

    monkeypatch.setattr(flood.logger, "error", error)
    monkeypatch.setattr(adapter, "_send_flood_cooldown_remaining", remaining)
    await adapter.send_voice("12345", str(clip))
    clip.unlink()  # the caller deletes its file: only the notice is left to deliver
    await async_wait_until(lambda: not adapter._background_tasks, timeout=10, what="deferred task done")

    assert adapter.send.await_count == notices
    assert len(checks) == min(open_checks + 1, flood._MAX_WINDOW_WAITS + 2)

