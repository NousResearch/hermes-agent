"""MEDIA-tag caption support: ``MEDIA:<path> | <caption>`` (Pdd daily-job requirement).

A MEDIA tag may carry an optional caption after a pipe. The caption rides on the
attachment bubble (Telegram ``sendPhoto`` caption) so an image and its question are
delivered together instead of as an unrelated album. Tags without a pipe keep working
exactly as before (empty caption), and a caption is never delivered as body text.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    _real_media_tag_spans,
    extract_media_captions,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource


# ── parsing / backward compatibility ────────────────────────────────────────────────────────


def _png(tmp_path, name="q.webp"):
    f = tmp_path / name
    f.write_bytes(b"\x89PNG\r\n\x1a\n")
    return f


def test_caption_parsed_from_pipe_suffix(tmp_path):
    f = _png(tmp_path)
    captions = extract_media_captions(f"MEDIA:{f} | Вопрос 1: что делать?")
    assert captions[str(f)] == "Вопрос 1: что делать?"


def test_tag_without_pipe_has_no_caption(tmp_path):
    f = _png(tmp_path)
    assert extract_media_captions(f"MEDIA:{f}") == {}


def test_caption_is_stripped_from_delivered_body(tmp_path):
    f = _png(tmp_path)
    media, cleaned = BasePlatformAdapter.extract_media(f"Смотри:\nMEDIA:{f} | Вопрос 1\nдальше")
    assert [p for p, _ in media] == [str(f)]
    assert "Вопрос 1" not in cleaned
    assert "MEDIA:" not in cleaned
    assert "дальше" in cleaned


def test_captions_are_per_tag_across_lines(tmp_path):
    a, b = _png(tmp_path, "a.webp"), _png(tmp_path, "b.webp")
    text = f"MEDIA:{a} | первый\nMEDIA:{b} | второй"
    captions = extract_media_captions(text)
    assert captions == {str(a): "первый", str(b): "второй"}


def test_backward_compatible_tag_without_caption_still_extracts(tmp_path):
    f = _png(tmp_path)
    media, cleaned = BasePlatformAdapter.extract_media(f"MEDIA:{f}")
    assert media == [(str(f), False)]
    assert "MEDIA:" not in cleaned


def test_extensionless_caption_parsed(tmp_path):
    f = tmp_path / "Caddyfile"
    f.write_text("x")
    captions = extract_media_captions(f"MEDIA:{f} | конфиг")
    assert captions.get(str(f)) == "конфиг"


def test_caption_suffix_is_part_of_cleanup_span(tmp_path):
    f = _png(tmp_path)
    text = f"x MEDIA:{f} | cap y"
    spans = _real_media_tag_spans(text)
    assert spans
    covered = text[spans[0][0]:spans[0][1]]
    assert covered.endswith("cap y")


def test_caption_is_not_shown_in_streamed_display(tmp_path):
    """The streamed/display lane strips the caption with the tag — it must never render as text."""
    f = _png(tmp_path)
    shown = BasePlatformAdapter.strip_media_directives_for_display(f"вопрос?\nMEDIA:{f} | Вопрос 1: кто уступает?")
    assert "Вопрос 1" not in shown
    assert "MEDIA:" not in shown
    assert "вопрос?" in shown


# ── post-stream gateway delivery (card's primary path) ──────────────────────────────────────


def _event():
    source = SessionSource(platform=Platform.SLACK, chat_id="C123CHAN", chat_type="group", thread_id=None)
    return MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="171.000001")


def _fake_runner(thread_meta):
    return SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )


def _adapter():
    return SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="imgs")),
    )


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    return media_file.resolve()


@pytest.mark.asyncio
async def test_post_stream_image_batch_carries_caption(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "q1.webp")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}),
        f"Билет 1\nMEDIA:{media_file} | Вопрос 1: кто уступает?",
        _event(),
        adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    images = adapter.send_multiple_images.await_args.kwargs["images"]
    assert len(images) == 1
    assert str(media_file) in images[0][0]
    assert images[0][1] == "Вопрос 1: кто уступает?"


@pytest.mark.asyncio
async def test_post_stream_image_without_caption_stays_empty(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "q2.webp")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"MEDIA:{media_file}", _event(), adapter,
    )

    images = adapter.send_multiple_images.await_args.kwargs["images"]
    assert images[0][1] == ""


@pytest.mark.asyncio
async def test_post_stream_document_carries_caption(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "ticket.pdf")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"MEDIA:{media_file} | Билет 1", _event(), adapter,
    )

    adapter.send_document.assert_awaited_once()
    assert adapter.send_document.await_args.kwargs["caption"] == "Билет 1"


@pytest.mark.asyncio
async def test_post_stream_document_without_caption_omits_caption_kwarg(tmp_path, monkeypatch):
    """No pipe → the dispatch call stays byte-identical to the pre-caption shape."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "ticket2.pdf")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"MEDIA:{media_file}", _event(), adapter,
    )

    adapter.send_document.assert_awaited_once()
    assert "caption" not in adapter.send_document.await_args.kwargs


# ── standalone `hermes send` plumbing (tools/send_message_tool) ─────────────────────────────


def _telegram_config():
    from types import SimpleNamespace as _NS
    telegram_cfg = _NS(enabled=True, token="***", extra={})
    return _NS(platforms={Platform.TELEGRAM: telegram_cfg}, get_home_channel=lambda _p: None), telegram_cfg


def _send_tool(tmp_path, monkeypatch, media_file, message):
    """Run the send_message tool against a telegram target with ``_send_to_platform`` stubbed."""
    import asyncio
    import json
    from unittest.mock import patch

    from tools.send_message_tool import send_message_tool

    config, _tg = _telegram_config()
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (media_file.parent,))
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=lambda coro: asyncio.run(coro)), \
         patch("tools.send_message_tool._send_to_platform",
               new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = json.loads(send_message_tool({"action": "send", "target": "telegram:12345", "message": message}))
    assert result["success"] is True
    return send_mock


def test_send_message_tool_forwards_tag_caption(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "q3.webp")
    send_mock = _send_tool(tmp_path, monkeypatch, media_file,
                           f"Билет 1\nMEDIA:{media_file} | Вопрос 1: кто уступает?")

    kwargs = send_mock.await_args.kwargs
    assert kwargs["media_captions"] == {str(media_file): "Вопрос 1: кто уступает?"}
    # Caption text is delivered on the bubble, never as body text.
    assert "Вопрос 1" not in send_mock.await_args.args[3]


def test_send_message_tool_without_caption_omits_kwarg(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "q4.webp")
    send_mock = _send_tool(tmp_path, monkeypatch, media_file, f"MEDIA:{media_file}")

    assert "media_captions" not in send_mock.await_args.kwargs
