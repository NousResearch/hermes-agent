"""Telegram Bot API 10.2 inline media inside rich-message reports."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


RICH_REPORT = "## Report\n\n| Metric | Value |\n|---|---|\n| risk | low |"


def _adapter(extra=None):
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={
            "rich_messages": True,
            "rich_inline_media": True,
            **(extra or {}),
        },
    )
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    bot.do_api_request = AsyncMock(return_value={"message_id": 321})
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot
    return adapter


class _FakeInputFile:
    counter = 0

    def __init__(self, file_obj, *, filename=None, attach=False):
        type(self).counter += 1
        self.file_obj = file_obj
        self.filename = filename
        self.attach_name = f"attach_{type(self).counter}"
        self.attach_uri = f"attach://{self.attach_name}" if attach else None


@pytest.mark.asyncio
async def test_new_rich_report_embeds_local_photo_in_single_send(tmp_path):
    adapter = _adapter()
    photo = tmp_path / "chart.png"
    photo.write_bytes(b"png")

    with patch("telegram.InputFile", _FakeInputFile):
        result = await adapter.send_rich_media(
            "12345",
            RICH_REPORT,
            [(str(photo), False)],
            metadata={"notify": True},
        )

    assert result is not None and result.success is True
    adapter._bot.do_api_request.assert_awaited_once()
    call = adapter._bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessage"
    payload = call.kwargs["api_kwargs"]
    rich = payload["rich_message"]
    assert "![](tg://photo?id=media_0)" in rich["markdown"]
    assert rich["media"][0]["id"] == "media_0"
    attach_uri = rich["media"][0]["media"]["media"]
    assert attach_uri.startswith("attach://")
    uploads = [value for value in payload.values() if isinstance(value, _FakeInputFile)]
    assert len(uploads) == 1
    assert uploads[0].attach_uri == attach_uri
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_streamed_report_can_be_edited_to_add_inline_media(tmp_path):
    adapter = _adapter()
    video = tmp_path / "proof.mp4"
    video.write_bytes(b"mp4")

    with patch("telegram.InputFile", _FakeInputFile):
        result = await adapter.edit_rich_media(
            "12345",
            "777",
            RICH_REPORT,
            [(str(video), False)],
            metadata={"notify": True},
        )

    assert result is not None and result.success is True
    call = adapter._bot.do_api_request.call_args
    assert call.args[0] == "editMessageText"
    payload = call.kwargs["api_kwargs"]
    assert payload["message_id"] == 777
    assert "![](tg://video?id=media_0)" in payload["rich_message"]["markdown"]


@pytest.mark.asyncio
async def test_rich_inline_media_opt_out_or_unsupported_file_falls_back(tmp_path):
    photo = tmp_path / "chart.png"
    photo.write_bytes(b"png")
    document = tmp_path / "report.pdf"
    document.write_bytes(b"pdf")

    disabled = _adapter(extra={"rich_inline_media": False})
    assert (
        await disabled.send_rich_media("12345", RICH_REPORT, [(str(photo), False)])
        is None
    )
    disabled._bot.do_api_request.assert_not_called()

    enabled = _adapter()
    assert (
        await enabled.send_rich_media("12345", RICH_REPORT, [(str(document), False)])
        is None
    )
    enabled._bot.do_api_request.assert_not_called()


@pytest.mark.asyncio
async def test_rich_inline_media_transient_failure_does_not_duplicate_send(tmp_path):
    class TimedOut(Exception):
        pass

    photo = tmp_path / "chart.png"
    photo.write_bytes(b"png")
    adapter = _adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=TimedOut("network timeout"))

    with patch("telegram.InputFile", _FakeInputFile):
        result = await adapter.send_rich_media(
            "12345", RICH_REPORT, [(str(photo), False)]
        )

    assert result is not None
    assert result.success is False
    assert result.retryable is True
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_post_stream_delivery_upgrades_existing_message_instead_of_resending_media(
    tmp_path,
):
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"png")
    adapter = MagicMock()
    adapter.name = "Telegram"
    adapter.extract_media.return_value = (
        [(str(chart), False)],
        RICH_REPORT,
    )
    adapter.extract_images.return_value = ([], RICH_REPORT)
    adapter.extract_local_files.return_value = ([], RICH_REPORT)
    adapter.edit_rich_media = AsyncMock(
        return_value=SendResult(success=True, message_id="777")
    )
    adapter.send_multiple_images = AsyncMock()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
    )
    event = MessageEvent(
        text="show report",
        message_type=MessageType.TEXT,
        source=source,
    )
    runner = object.__new__(GatewayRunner)

    await runner._deliver_media_from_response(
        f"{RICH_REPORT}\nMEDIA:{chart}",
        event,
        adapter,
        stream_message_id="777",
    )

    adapter.edit_rich_media.assert_awaited_once_with(
        "12345",
        "777",
        RICH_REPORT,
        [(str(chart), False)],
        metadata=None,
    )
    adapter.send_multiple_images.assert_not_called()


class _RecordingRichMediaAdapter(TelegramAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={"rich_messages": True, "rich_inline_media": True},
            )
        )
        self.rich_media_calls = []

    async def send_rich_media(
        self,
        chat_id,
        content,
        media_files,
        reply_to=None,
        metadata=None,
    ):
        self.rich_media_calls.append((
            chat_id,
            content,
            media_files,
            reply_to,
            metadata,
        ))
        return SendResult(success=True, message_id="321")


@pytest.mark.asyncio
async def test_background_delivery_combines_text_and_media_without_duplicates(tmp_path):
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"png")
    adapter = _RecordingRichMediaAdapter()
    adapter._send_with_retry = AsyncMock()
    adapter.send_multiple_images = AsyncMock()

    async def handler(_event):
        return f"{RICH_REPORT}\nMEDIA:{chart}"

    adapter.set_message_handler(handler)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="group",
    )
    event = MessageEvent(
        text="show report",
        message_type=MessageType.TEXT,
        source=source,
        message_id="12",
    )

    with patch.object(adapter, "_keep_typing", new=AsyncMock()):
        await adapter._process_message_background(event, "telegram:group:12345")

    assert len(adapter.rich_media_calls) == 1
    chat_id, content, media_files, reply_to, metadata = adapter.rich_media_calls[0]
    assert chat_id == "12345"
    assert content == RICH_REPORT
    assert media_files == [(str(chart), False)]
    assert reply_to == "12"
    assert metadata["notify"] is True
    adapter._send_with_retry.assert_not_awaited()
    adapter.send_multiple_images.assert_not_awaited()
