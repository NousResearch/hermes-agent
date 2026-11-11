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
async def test_streamed_reply_without_a_photo_keeps_the_trailing_footer(tmp_path, monkeypatch):
    adapter = _Adapter()
    await _streamed_turn(adapter, "just text", FOOTER)
    assert adapter.sent == [FOOTER] and adapter.batches == []

    adapter.sent.clear()
    adapter.media_caption_limit = 0
    photo = _photo(tmp_path, monkeypatch)
    await _streamed_turn(adapter, f"Here it is\nMEDIA:{photo}", FOOTER)
    assert adapter.sent == [FOOTER] and adapter.batches[-1][-1][1] == ""
