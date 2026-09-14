"""Complementary coverage for terminal EOS after validated nonstandard MEDIA paths (#111046)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer


def _allow_media_root(monkeypatch, root):
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)


@pytest.mark.parametrize("filename", ["payload.weirdext", "Caddyfile"])
def test_terminal_eos_preserves_valid_nonstandard_media_paths(tmp_path, monkeypatch, filename):
    root = tmp_path / "media-cache"
    root.mkdir()
    media_file = root / filename
    media_file.write_bytes(b"media")
    _allow_media_root(monkeypatch, root)

    media, cleaned = BasePlatformAdapter.extract_media(
        f"Here is the file.\nMEDIA:{media_file}<|eos|>"
    )

    assert media == [(str(media_file.resolve()), False)]
    assert cleaned == "Here is the file."


def test_terminal_eos_nonstandard_media_is_removed_from_stream_display(tmp_path, monkeypatch):
    root = tmp_path / "media-cache"
    root.mkdir()
    media_file = root / "payload.weirdext"
    media_file.write_bytes(b"media")
    _allow_media_root(monkeypatch, root)

    cleaned = GatewayStreamConsumer._clean_for_display(
        f"Here is the file.\nMEDIA:{media_file}<|eos|>"
    )

    assert cleaned == "Here is the file."


def test_terminal_eos_preserves_spaced_nonstandard_media_path(tmp_path, monkeypatch):
    root = tmp_path / "media-cache"
    media_file = root / "My Documents" / "payload.weirdext"
    media_file.parent.mkdir(parents=True)
    media_file.write_bytes(b"media")
    _allow_media_root(monkeypatch, root)

    media, cleaned = BasePlatformAdapter.extract_media(
        f"Here is the file.\nMEDIA:{media_file}<|eos|>"
    )

    assert media == [(str(media_file.resolve()), False)]
    assert cleaned == "Here is the file."


@pytest.mark.parametrize("suffix", ["<|EOS|>", "<|not-eos|>"])
def test_nonexact_terminal_markup_is_not_a_media_boundary(tmp_path, monkeypatch, suffix):
    root = tmp_path / "media-cache"
    media_file = root / "payload.weirdext"
    root.mkdir()
    media_file.write_bytes(b"media")
    _allow_media_root(monkeypatch, root)
    content = f"MEDIA:{media_file}{suffix}"

    media, cleaned = BasePlatformAdapter.extract_media(content)

    assert media == []
    assert cleaned == content


def test_terminal_eos_does_not_extend_a_previous_nonstandard_media_span(tmp_path, monkeypatch):
    root = tmp_path / "media-cache"
    first = root / "first.weirdext"
    second = root / "second.weirdext"
    root.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    _allow_media_root(monkeypatch, root)

    media, cleaned = BasePlatformAdapter.extract_media(
        f"MEDIA:{first} MEDIA:{second}<|eos|>"
    )

    assert media == [(str(first.resolve()), False), (str(second.resolve()), False)]
    assert cleaned == ""


@pytest.mark.asyncio
async def test_terminal_eos_nonstandard_media_reaches_post_stream_document_send(tmp_path, monkeypatch):
    root = tmp_path / "media-cache"
    root.mkdir()
    media_file = root / "payload.weirdext"
    media_file.write_bytes(b"media")
    _allow_media_root(monkeypatch, root)
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="images")),
    )
    source = SessionSource(platform=Platform.SLACK, chat_id="C123CHAN", chat_type="group")
    event = MessageEvent(
        text="hi", message_type=MessageType.TEXT, source=source, message_id="171.000001"
    )
    runner = SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: {},
        _reply_anchor_for_event=lambda event: None,
    )

    await GatewayRunner._deliver_media_from_response(
        runner,
        f"Here is the file.\nMEDIA:{media_file}<|eos|>",
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once()
    adapter.send_multiple_images.assert_not_awaited()
    assert adapter.send_document.await_args.kwargs["file_path"] == str(media_file.resolve())
