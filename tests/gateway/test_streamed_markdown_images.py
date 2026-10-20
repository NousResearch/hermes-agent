"""A streamed reply's markdown/HTML images must arrive as attachments (#141806).

When streaming already delivered the answer, the gateway suppresses its normal final
send, and with it the ``extract_images`` step that turns ``![alt](url)`` / ``<img src>``
into a native attachment. A streamed reply that carried an image therefore showed the
tag as text and never sent the picture. The post-stream rescan has to deliver the URLs,
and the streamed display must not keep the literal tag on screen.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer


def _event():
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="D123CHAN",
        chat_type="group",
        thread_id=None,
    )
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=source,
        message_id="171.000001",
    )


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


@pytest.mark.asyncio
async def test_markdown_image_is_delivered_post_stream():
    """The report's shape: the reply is only a markdown image."""
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}),
        "![meme](https://api.memegen.link/images/drake/a/b.png)",
        _event(),
        adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    kwargs = adapter.send_multiple_images.await_args.kwargs
    assert kwargs["chat_id"] == "D123CHAN"
    assert kwargs["images"] == [("https://api.memegen.link/images/drake/a/b.png", "meme")]


@pytest.mark.asyncio
async def test_html_image_is_delivered_post_stream():
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}),
        'Chart follows.\n<img src="https://example.com/chart.png" />',
        _event(),
        adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    assert adapter.send_multiple_images.await_args.kwargs["images"] == [
        ("https://example.com/chart.png", "")
    ]


@pytest.mark.asyncio
async def test_url_image_and_media_file_share_one_batch(tmp_path, monkeypatch):
    """A reply with both contracts still produces a single attachment call."""
    root = tmp_path / "media-cache"
    media_file = root / "chart.png"
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    resolved = media_file.resolve()
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}),
        f"Two pictures.\nMEDIA:{resolved}\n![remote](https://example.com/remote.png)",
        _event(),
        adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    images = adapter.send_multiple_images.await_args.kwargs["images"]
    assert len(images) == 2
    assert any(url.endswith("remote.png") for url, _ in images)
    assert any(url.startswith("file://") for url, _ in images)


@pytest.mark.asyncio
async def test_plain_link_is_not_delivered():
    """Only the image form is an attachment; a plain markdown link stays text."""
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}),
        "See [the chart](https://example.com/chart.png) for details.",
        _event(),
        adapter,
    )

    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_document.assert_not_awaited()


def test_display_hides_the_image_tag_and_keeps_the_prose():
    shown = GatewayStreamConsumer._clean_for_display(
        "Here it is:\n\n![meme](https://api.memegen.link/images/drake/a/b.png)\n\nEnjoy."
    )

    assert "![meme]" not in shown
    assert "Here it is:" in shown
    assert "Enjoy." in shown


def test_display_leaves_plain_text_untouched():
    text = "No pictures here, just [a link](https://example.com/page)."
    assert GatewayStreamConsumer._clean_for_display(text) == text
