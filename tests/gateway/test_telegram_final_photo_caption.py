"""Live Telegram final delivery: prose + ONE image ``MEDIA:`` tag is one photo message.

The non-streaming final path (``BasePlatformAdapter._process_message_background``) used to
always split such a turn into two Telegram messages — a text message followed by a standalone
photo. Telegram captions hold 1024 UTF-16 units, so the common "here's your chart" reply fits
in a single captioned photo.

Pinned here against a real ``TelegramAdapter`` driven by a fake Bot API object (no network):

* prose + exactly one image MEDIA tag, caption within budget -> ONE ``sendPhoto`` with the
  prose as caption, carrying the reply/thread anchors the text message would have carried;
* caption over 1024 UTF-16 units (astral chars cost two) -> standalone behaviour is preserved:
  a text message plus a separate photo;
* every other shape (two attachments, a non-image MEDIA tag, an image URL, ``[[as_document]]``,
  a voice tag) keeps the standalone split;
* a refused captioned photo falls back to text + standalone attachment — the prose is never lost.
"""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from plugins.platforms.telegram.adapter import TelegramAdapter


class _FakeBot:
    """Records Bot API calls instead of talking to Telegram."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.photo_error: Exception | None = None
        self._next_id = 100

    def _record(self, name, kwargs):
        self.calls.append((name, dict(kwargs)))
        self._next_id += 1
        return SimpleNamespace(message_id=self._next_id)

    async def send_photo(self, **kwargs):
        if self.photo_error is not None:
            raise self.photo_error
        return self._record("send_photo", kwargs)

    async def send_message(self, **kwargs):
        return self._record("send_message", kwargs)

    async def send_media_group(self, **kwargs):
        return [self._record("send_media_group", kwargs)]

    async def send_document(self, **kwargs):
        return self._record("send_document", kwargs)

    async def send_video(self, **kwargs):
        return self._record("send_video", kwargs)

    async def send_voice(self, **kwargs):
        return self._record("send_voice", kwargs)

    async def send_chat_action(self, **kwargs):
        return None

    def names(self):
        return [name for name, _ in self.calls]

    def kwargs_for(self, name):
        return [kw for n, kw in self.calls if n == name]


async def _hold_typing(_chat_id, interval=2.0, metadata=None, stop_event=None):
    if stop_event is not None:
        await stop_event.wait()
    else:
        await asyncio.Event().wait()


@pytest.fixture()
def bot(monkeypatch):
    """A connected TelegramAdapter whose transport is a fake bot, ledger disabled."""
    monkeypatch.setattr("gateway.delivery_ledger.ledger_enabled", lambda: False)
    return _FakeBot()


@pytest.fixture()
def adapter(bot):
    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    a._bot = bot
    a._keep_typing = _hold_typing
    # Bot API 10.1 rich sends are a separate transport; pin the legacy sendMessage path.
    a._should_attempt_rich = lambda *_a, **_kw: False
    return a


def _event(text="make me a chart"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm"),
        message_id="7",
    )


def _media_file(tmp_path, monkeypatch, name, payload=b"\x89PNG\r\n"):
    root = tmp_path / "media-cache"
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    return path.resolve()


async def _run(adapter, response):
    async def handler(_event):
        return response

    adapter.set_message_handler(handler)
    event = _event()
    await adapter._process_message_background(event, build_session_key(event.source))


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prose_plus_one_image_is_a_single_captioned_photo(adapter, bot, tmp_path, monkeypatch):
    png = _media_file(tmp_path, monkeypatch, "chart.png")
    await _run(adapter, f"Here is your **chart**.\nMEDIA:{png}")

    assert bot.names() == ["send_photo"], (
        f"expected one captioned photo, got {bot.names()}")
    caption = bot.kwargs_for("send_photo")[0]["caption"]
    assert "chart" in caption and "Here is your" in caption
    # MarkdownV2 rendering, same reason as the #32029 voice-caption ladder.
    assert caption.startswith("Here is your *chart*")


@pytest.mark.asyncio
async def test_captioned_photo_keeps_the_reply_anchor(adapter, bot, tmp_path, monkeypatch):
    """The merged message carries the anchor the text message would have carried."""
    png = _media_file(tmp_path, monkeypatch, "chart.png")

    async def handler(_event):
        return f"Done.\nMEDIA:{png}"

    adapter.set_message_handler(handler)
    event = MessageEvent(
        text="chart please",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group"),
        message_id="9",
    )
    await adapter._process_message_background(event, build_session_key(event.source))

    assert bot.names() == ["send_photo"]
    assert bot.kwargs_for("send_photo")[0]["reply_to_message_id"] == 9


@pytest.mark.asyncio
async def test_captioned_photo_keeps_forum_topic_routing(adapter, bot, tmp_path, monkeypatch):
    """Forum topics route by topic id and deliberately carry no reply anchor
    (``_reply_anchor_for_event``) — the merged message must not change that."""
    png = _media_file(tmp_path, monkeypatch, "chart.png")

    async def handler(_event):
        return f"Done.\nMEDIA:{png}"

    adapter.set_message_handler(handler)
    event = MessageEvent(
        text="in a topic",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", thread_id="55"),
        message_id="9",
    )
    await adapter._process_message_background(event, build_session_key(event.source))

    assert bot.names() == ["send_photo"]
    sent = bot.kwargs_for("send_photo")[0]
    assert sent["message_thread_id"] == 55
    assert sent["reply_to_message_id"] is None


# ---------------------------------------------------------------------------
# Shapes that MUST keep the standalone split
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_caption_over_1024_utf16_units_stays_split(adapter, bot, tmp_path, monkeypatch):
    """Astral chars cost two UTF-16 units — 513 emoji overflow a 1024-unit caption."""
    png = _media_file(tmp_path, monkeypatch, "chart.png")
    prose = "😀" * 513
    await _run(adapter, f"{prose}\nMEDIA:{png}")

    names = bot.names()
    assert "send_message" in names, f"long prose must keep its own text message: {names}"
    assert "send_photo" in names or "send_media_group" in names, (
        f"the photo must still be delivered standalone: {names}")


@pytest.mark.asyncio
async def test_caption_at_the_1024_utf16_boundary_still_merges(adapter, bot, tmp_path, monkeypatch):
    png = _media_file(tmp_path, monkeypatch, "chart.png")
    await _run(adapter, f"{'a' * 1024}\nMEDIA:{png}")

    assert bot.names() == ["send_photo"], f"1024 units must still merge: {bot.names()}"


@pytest.mark.asyncio
async def test_two_media_tags_stay_split(adapter, bot, tmp_path, monkeypatch):
    first = _media_file(tmp_path, monkeypatch, "one.png")
    second = first.parent / "two.png"
    second.write_bytes(b"\x89PNG\r\n")
    await _run(adapter, f"Two charts.\nMEDIA:{first}\nMEDIA:{second}")

    assert "send_message" in bot.names(), f"albums are out of scope: {bot.names()}"


@pytest.mark.asyncio
async def test_non_image_media_tag_stays_split(adapter, bot, tmp_path, monkeypatch):
    pdf = _media_file(tmp_path, monkeypatch, "report.pdf")
    await _run(adapter, f"The report.\nMEDIA:{pdf}")

    assert bot.names() == ["send_message", "send_document"], bot.names()


@pytest.mark.asyncio
async def test_as_document_marker_stays_split(adapter, bot, tmp_path, monkeypatch):
    png = _media_file(tmp_path, monkeypatch, "chart.png")
    await _run(adapter, f"Full quality.[[as_document]]\nMEDIA:{png}")

    assert "send_message" in bot.names() and "send_photo" not in bot.names(), bot.names()


@pytest.mark.asyncio
async def test_image_url_alongside_media_tag_stays_split(adapter, bot, tmp_path, monkeypatch):
    png = _media_file(tmp_path, monkeypatch, "chart.png")
    await _run(adapter, f"Both.\n![pic](https://example.com/pic.png)\nMEDIA:{png}")

    assert "send_message" in bot.names(), bot.names()


@pytest.mark.asyncio
async def test_text_only_response_is_unchanged(adapter, bot):
    await _run(adapter, "Just words.")

    assert bot.names() == ["send_message"], bot.names()


@pytest.mark.asyncio
async def test_media_only_response_is_unchanged(adapter, bot, tmp_path, monkeypatch):
    png = _media_file(tmp_path, monkeypatch, "chart.png")
    await _run(adapter, f"MEDIA:{png}")

    assert "send_message" not in bot.names(), bot.names()


# ---------------------------------------------------------------------------
# Failure fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_captioned_send_falls_back_to_text_plus_attachment(
        adapter, bot, tmp_path, monkeypatch):
    """An unsuccessful merged send must not swallow the prose: the text goes out on its own
    and the image returns to the standalone attachment lane."""
    png = _media_file(tmp_path, monkeypatch, "chart.png")

    async def _refuse(*_a, **_kw):
        return SendResult(success=False, error="boom")

    monkeypatch.setattr(adapter, "send_final_captioned_image", _refuse)
    await _run(adapter, f"Here it is.\nMEDIA:{png}")

    names = bot.names()
    assert "send_message" in names, f"prose lost after a refused merge: {names}"
    assert "send_photo" in names or "send_media_group" in names, (
        f"image lost after a refused merge: {names}")


@pytest.mark.asyncio
async def test_photo_refusal_keeps_the_prose_on_the_document_fallback(
        adapter, bot, tmp_path, monkeypatch):
    """Telegram refusing sendPhoto (bad dimensions) drops to sendDocument — which still
    carries the prose as its caption, so the turn stays one message."""
    png = _media_file(tmp_path, monkeypatch, "chart.png")
    bot.photo_error = RuntimeError("Photo_invalid_dimensions")
    await _run(adapter, f"Here it is.\nMEDIA:{png}")

    assert bot.names() == ["send_document"], bot.names()
    assert bot.kwargs_for("send_document")[0]["caption"] == "Here it is."


@pytest.mark.asyncio
async def test_non_telegram_adapter_never_merges(tmp_path, monkeypatch):
    """The merge is opt-in per adapter (``final_caption_media_limit``)."""
    from tests.gateway.test_73771_media_resend_dedup import _DummyAdapter  # noqa: PLC0415

    png = _media_file(tmp_path, monkeypatch, "chart.png")
    adapter = _DummyAdapter()
    adapter._keep_typing = _hold_typing

    async def handler(_event):
        return f"Here.\nMEDIA:{png}"

    adapter.set_message_handler(handler)
    event = MessageEvent(
        text="chart",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.DISCORD, chat_id="1", chat_type="dm"),
        message_id="1",
    )
    await adapter._process_message_background(event, build_session_key(event.source))

    assert [entry["content"] for entry in adapter.sent] == ["Here."]
    assert adapter.images_sent and str(png) in adapter.images_sent[0]
