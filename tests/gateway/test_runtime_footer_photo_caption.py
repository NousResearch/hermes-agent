"""The runtime footer rides the last photo's caption instead of a message of its own (#74547).

Where the footer would be its own message (the trailing send after a streamed reply, or a reply
whose only text is the footer) it goes on the last photo's caption, on adapters that declare
``media_caption_limit`` (Telegram: 1024) and only when it fits. Everything else is unchanged.
"""

from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType, SessionSource
from gateway.runtime_footer import FooteredReply
from tests.gateway.test_run_progress_topics import _make_runner

FOOTER = "model-x · 29%"


class _Adapter(BasePlatformAdapter):
    media_caption_limit = 1024

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent, self.batches = [], []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id="m1")

    async def send_multiple_images(self, chat_id, images, metadata=None, human_delay=0.0) -> SendResult:
        self.batches.append(list(images))
        return SendResult(success=True, message_id="p1")

    async def get_chat_info(self, chat_id):
        return {"name": "c", "type": "dm"}


def _event():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    return MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="1")


def _photo(tmp_path, monkeypatch, name="chart.png"):
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    monkeypatch.setattr(BasePlatformAdapter, "filter_media_delivery_paths", staticmethod(lambda files, **_k: files))
    return path


@pytest.mark.asyncio
async def test_photo_only_reply_puts_the_footer_on_the_caption(tmp_path, monkeypatch):
    photo = _photo(tmp_path, monkeypatch)
    adapter = _Adapter()
    extracted = await adapter._extract_response_content(
        FooteredReply(f"MEDIA:{photo}", FOOTER), _event(), "k", is_ephemeral_response=False)

    assert extracted.text_content == ""  # no footer-only message
    assert extracted.media_files == []
    assert extracted.images[-1][1] == FOOTER and extracted.images[-1][0].startswith("file://")


@pytest.mark.asyncio
async def test_footer_stays_in_text_when_there_is_text_or_it_does_not_fit(tmp_path, monkeypatch):
    photo = _photo(tmp_path, monkeypatch)
    adapter = _Adapter()
    with_text = await adapter._extract_response_content(
        FooteredReply(f"Here it is\nMEDIA:{photo}", FOOTER), _event(), "k", is_ephemeral_response=False)
    assert with_text.text_content.endswith(FOOTER) and with_text.media_files  # the text message carries it

    adapter.media_caption_limit = len(FOOTER) - 1
    too_long = await adapter._extract_response_content(
        FooteredReply(f"MEDIA:{photo}", FOOTER), _event(), "k", is_ephemeral_response=False)
    assert too_long.text_content == FOOTER and too_long.media_files  # today's behaviour

    adapter.media_caption_limit = 0  # an adapter that never declared a caption
    plain = await adapter._extract_response_content(
        FooteredReply(f"MEDIA:{photo}", FOOTER), _event(), "k", is_ephemeral_response=False)
    assert plain.text_content == FOOTER


async def _streamed_turn(adapter, response, footer):
    runner = _make_runner(adapter)
    event = _event()
    return await runner._hmwa_deliver_turn_response(
        event, event.source, SimpleNamespace(session_id="s1"), "k", None,
        {"already_sent": True}, [], response, footer, False)


@pytest.mark.asyncio
async def test_streamed_photo_reply_sends_no_trailing_footer_message(tmp_path, monkeypatch):
    photo = _photo(tmp_path, monkeypatch)
    adapter = _Adapter()
    monkeypatch.setattr("gateway.platforms.base.BasePlatformAdapter.filter_media_delivery_paths",
                        staticmethod(lambda files, **_k: files))

    await _streamed_turn(adapter, f"Here it is\nMEDIA:{photo}", FOOTER)

    assert adapter.batches and adapter.batches[-1][-1][1] == FOOTER
    assert FOOTER not in adapter.sent


@pytest.mark.asyncio
async def test_streamed_footer_falls_back_when_the_captioned_photo_failed(tmp_path, monkeypatch):
    """The album reports success when ANY photo landed: if the last one (with the footer) is in
    ``failed``, the footer still goes out as its trailing message."""
    first, last = _photo(tmp_path, monkeypatch, "a.png"), _photo(tmp_path, monkeypatch, "b.png")
    adapter = _Adapter()

    async def partly(chat_id, images, metadata=None, human_delay=0.0):
        adapter.batches.append(list(images))
        return SendResult(success=True, raw_response={"failed": [(str(last), "refused")]})

    adapter.send_multiple_images = partly
    await _streamed_turn(adapter, f"Here\nMEDIA:{first}\nMEDIA:{last}", FOOTER)

    assert adapter.batches[-1][-1][1] == FOOTER
    assert adapter.sent == [FOOTER]


@pytest.mark.asyncio
async def test_streamed_reply_without_a_photo_keeps_the_trailing_footer(tmp_path, monkeypatch):
    adapter = _Adapter()
    await _streamed_turn(adapter, "just text", FOOTER)
    assert adapter.sent == [FOOTER] and adapter.batches == []

    adapter.sent.clear()
    adapter.media_caption_limit = 0
    photo = _photo(tmp_path, monkeypatch)
    await _streamed_turn(adapter, f"Here it is\nMEDIA:{photo}", FOOTER)
    assert adapter.sent == [FOOTER] and adapter.batches[-1][-1][1] == ""


@pytest.mark.asyncio
async def test_non_streamed_footer_is_sent_when_the_captioned_photo_failed(tmp_path, monkeypatch):
    """Photo-only reply: the footer rides the photo's caption, so when that photo send fails the
    footer goes out as its own message, as it did before the caption move."""
    photo = _photo(tmp_path, monkeypatch)
    adapter = _Adapter()

    async def refused(chat_id, images, metadata=None, human_delay=0.0):
        adapter.batches.append(list(images))
        return SendResult(success=False, error="all images failed to send")

    adapter.send_multiple_images = refused
    outcomes = []
    extracted = await adapter._extract_response_content(
        FooteredReply(f"MEDIA:{photo}", FOOTER), _event(), "k", is_ephemeral_response=False)
    await adapter._deliver_attachments(_event(), extracted, {}, anything_sent=False, record_delivery=outcomes.append)

    assert adapter.batches[-1][-1][1] == FOOTER
    assert adapter.sent == [FOOTER]

    adapter.sent.clear()  # control: the photo landed, so no footer-only message
    adapter.send_multiple_images = _Adapter.send_multiple_images.__get__(adapter)
    await adapter._deliver_attachments(_event(), extracted, {}, anything_sent=False, record_delivery=outcomes.append)
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_streamed_footer_is_sent_when_the_photo_batch_raises(tmp_path, monkeypatch):
    photo = _photo(tmp_path, monkeypatch)
    adapter = _Adapter()

    async def boom(chat_id, images, metadata=None, human_delay=0.0):
        adapter.batches.append(list(images))
        raise RuntimeError("upload exploded")

    adapter.send_multiple_images = boom
    await _streamed_turn(adapter, f"Here it is\nMEDIA:{photo}", FOOTER)

    assert adapter.batches
    assert adapter.sent == [FOOTER]


@pytest.mark.asyncio
async def test_footer_never_rides_an_animation(tmp_path, monkeypatch):
    """A remote GIF goes out on its own path, whose failure the batch result does not list: the
    footer rides the last real photo, so a lost GIF cannot take the footer with it."""
    photo = _photo(tmp_path, monkeypatch)
    adapter = _Adapter()
    landed = []

    async def gif_lost(chat_id, images, metadata=None, human_delay=0.0):
        adapter.batches.append(list(images))
        landed.extend(alt for url, alt in images if not adapter._is_animation_url(url))
        return SendResult(success=True, message_id="p1")  # the photo landed, the GIF did not

    adapter.send_multiple_images = gif_lost
    await _streamed_turn(adapter, f"Here it is\nMEDIA:{photo}\n![cat](https://example.com/cat.gif)", FOOTER)

    assert adapter.batches[-1][-1] == ("https://example.com/cat.gif", "cat")
    assert FOOTER in [*landed, *adapter.sent]


@pytest.mark.asyncio
async def test_no_separate_footer_when_the_captioned_photo_is_queued_for_a_reupload(tmp_path, monkeypatch):
    """Flood control defers the photo to a later re-upload that keeps its caption (and the footer):
    a separate footer message now would show the footer twice."""
    photo = _photo(tmp_path, monkeypatch)
    adapter = _Adapter()

    async def deferred(chat_id, images, metadata=None, human_delay=0.0):
        adapter.batches.append(list(images))
        url = images[-1][0]
        return SendResult(success=False, error="flood_control:30",
                          raw_response={"failed": [(url, "flood_control:30")], "deferred": [url]})

    adapter.send_multiple_images = deferred
    extracted = await adapter._extract_response_content(
        FooteredReply(f"MEDIA:{photo}", FOOTER), _event(), "k", is_ephemeral_response=False)
    await adapter._deliver_attachments(_event(), extracted, {}, anything_sent=False, record_delivery=lambda _r: None)
    await _streamed_turn(adapter, f"Here it is\nMEDIA:{photo}", FOOTER)

    assert [batch[-1][1] for batch in adapter.batches] == [FOOTER, FOOTER]
    assert FOOTER not in adapter.sent
