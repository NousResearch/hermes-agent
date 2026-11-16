"""A streamed reply's markdown/HTML images must arrive as attachments (#141806).

When streaming already delivered the answer, the gateway suppresses its normal final
send, and with it the ``extract_images`` step that turns ``![alt](url)`` / ``<img src>``
into a native attachment. A streamed reply that carried an image therefore showed the
tag as text and never sent the picture. The post-stream rescan has to deliver the URLs,
and the streamed display must not keep the literal tag on screen.
"""

import itertools
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


async def _stream_frames(deltas):
    """Run a real consumer over *deltas* (every chunk edit-eligible); return the shown frames."""
    import asyncio
    from unittest.mock import MagicMock

    from gateway.stream_consumer import StreamConsumerConfig
    from tests.waiting import async_wait_until

    adapter = MagicMock()
    frames = []

    async def _send(**kw):
        frames.append(kw["content"])
        return SimpleNamespace(success=True, message_id="m1")

    async def _edit(**kw):
        frames.append(kw["content"])
        return SimpleNamespace(success=True)

    adapter.send, adapter.edit_message = _send, _edit
    adapter.MAX_MESSAGE_LENGTH = 4096
    consumer = GatewayStreamConsumer(adapter, "chat_1", StreamConsumerConfig(edit_interval=0.0, buffer_threshold=1))
    task = asyncio.create_task(consumer.run())
    seen = ""
    for delta in deltas:
        consumer.on_delta(delta)
        seen += delta
        await async_wait_until(lambda: consumer._accumulated.endswith(seen[-8:]) and consumer._queue.empty())
        for _ in range(5):
            await asyncio.sleep(0)
    consumer.finish()
    await task
    cursor = (consumer.cfg.cursor or "").strip()
    return [f.replace(cursor, "") if cursor else f for f in frames]


@pytest.mark.parametrize("deltas, final", [
    (["Here is the render.\n\n!", "[barrel ren", "der](https://exa", "mple.com/r/b", "arrel.png)",
      "\n\nWant another ", "angle?"], "Here is the render.\n\nWant another angle?"),
    # "!" glued to prose: the extractor has no left boundary, so the "!" must not be shown first.
    (["Here is the chart:!", "[chart](https://example.com/chart.png) More"], "Here is the chart: More"),
    # The alt text may span lines; the hold covers the whole unfinished tag, not just the last line.
    (["Look: ![first\nsec", "ond](https://example.com/c.png)", "done"], "Look: done"),
])
@pytest.mark.asyncio
async def test_streamed_image_tag_never_shows_and_frames_stay_prefix_stable(deltas, final):
    """No frame shows any part of the tag, and each frame extends the previous one: showing
    ``![barrel](https://ex`` and removing it later would break prefix-stable draft frames."""
    frames = await _stream_frames(deltas)

    assert frames, "nothing was streamed"
    for frame in frames:
        assert "!" not in frame and "example.com" not in frame and "](" not in frame, frame
    interim = [f.rstrip() for f in frames[:-1]]
    for prev, nxt in itertools.pairwise(interim):
        assert nxt.startswith(prev), (prev, nxt)
    assert frames[-1].strip() == final


@pytest.mark.asyncio
async def test_queued_lane_sends_the_text_without_the_tag_and_the_image_once():
    """The queued follow-up lane sends the text itself, then runs the post-stream uploader: the
    tag it now attaches must not also stay in the text (one picture, no literal tag)."""
    adapter = _adapter()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="t1"))
    source = _event().source

    await GatewayRunner._deliver_queued_first_response(
        object.__new__(GatewayRunner), "Done.\n\n![chart](https://example.com/chart.png)",
        source=source, adapter=adapter, metadata={}, event_message_id=None,
        text_already_delivered=False, deliver_media=True)

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[1] == "Done."
    adapter.send_multiple_images.assert_awaited_once()
    assert adapter.send_multiple_images.await_args.kwargs["images"] == [("https://example.com/chart.png", "chart")]
