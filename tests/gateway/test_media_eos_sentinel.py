"""Regression coverage for terminal provider sentinels after MEDIA tags (#111046)."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer


def test_terminal_eos_does_not_hide_known_extension_media():
    media, cleaned = BasePlatformAdapter.extract_media(
        "Here is the image.\nMEDIA:/tmp/hermes-media-example.png<|eos|>"
    )

    assert media == [("/tmp/hermes-media-example.png", False)]
    assert cleaned == "Here is the image."


def test_clean_media_response_is_unchanged():
    media, cleaned = BasePlatformAdapter.extract_media(
        "Here is the image.\nMEDIA:/tmp/hermes-media-example.png"
    )

    assert media == [("/tmp/hermes-media-example.png", False)]
    assert cleaned == "Here is the image."


@pytest.mark.parametrize(
    "text",
    [
        "MEDIA:/tmp/hermes-media-example.png<unknown>",
        "MEDIA:/tmp/hermes-media-example.png<|EOS|>",
    ],
)
def test_unknown_adjacent_markup_is_not_reinterpreted_as_a_delimiter(text):

    media, cleaned = BasePlatformAdapter.extract_media(text)

    assert media == []
    assert cleaned == text


@pytest.mark.parametrize(
    "text",
    [
        "```text\nMEDIA:/tmp/example.png<|eos|>\n```",
        '{"example":"MEDIA:/tmp/example.png<|eos|>"}',
    ],
)
def test_protected_examples_do_not_extract_or_strip(text):
    media, cleaned = BasePlatformAdapter.extract_media(text)

    assert media == []
    assert cleaned == text
    assert BasePlatformAdapter.strip_media_directives_for_display(text) == text


def test_streamed_display_cleanup_handles_split_terminal_sentinel():
    partial = "Done.\nMEDIA:/tmp/example.png"
    complete = partial + "<|eos|>"

    assert GatewayStreamConsumer._clean_for_display(partial) == "Done."
    assert GatewayStreamConsumer._clean_for_display(complete) == "Done."


@pytest.mark.asyncio
async def test_post_stream_delivery_sends_terminal_eos_media_once(tmp_path, monkeypatch):
    root = tmp_path / "media-cache"
    image = root / "chart.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))

    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123CHAN",
        chat_type="group",
        thread_id=None,
    )
    event = MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=source,
        message_id="171.000001",
    )
    adapter = type(
        "Adapter",
        (),
        {
            "name": "test",
            "extract_media": staticmethod(BasePlatformAdapter.extract_media),
            "extract_images": staticmethod(BasePlatformAdapter.extract_images),
            "send_voice": AsyncMock(return_value=SendResult(success=True)),
            "send_document": AsyncMock(return_value=SendResult(success=True)),
            "send_image_file": AsyncMock(return_value=SendResult(success=True)),
            "send_video": AsyncMock(return_value=SendResult(success=True)),
            "send_multiple_images": AsyncMock(return_value=SendResult(success=True)),
        },
    )()
    runner = type(
        "Runner",
        (),
        {
            "_thread_metadata_for_source": lambda self, source, anchor=None: {},
            "_reply_anchor_for_event": lambda self, event: None,
        },
    )()

    await GatewayRunner._deliver_media_from_response(
        runner,
        f"Done.\nMEDIA:{image}<|eos|>",
        event,
        adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    adapter.send_image_file.assert_not_awaited()
    sent = adapter.send_multiple_images.await_args.kwargs["images"]
    assert [item[0] for item in sent] == [image.resolve().as_uri()]
