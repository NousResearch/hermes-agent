"""Gateway auto-detects local image paths typed in message text.

Regression tests for https://github.com/NousResearch/hermes-agent/issues/40533
"""

import pytest
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _make_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda **_: "native"
    return runner


def _source(chat_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="private",
        user_name=f"user-{chat_id}",
    )


@pytest.mark.asyncio
async def test_text_image_path_detected_and_buffered(tmp_path):
    """A local image path typed in message text is auto-detected and buffered."""
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG")

    runner = _make_runner()
    source = _source("chat-1")

    await runner._prepare_inbound_message_text(
        event=MessageEvent(text=f"see {img}", source=source),
        source=source,
        history=[],
    )

    key = build_session_key(source)
    paths = runner._consume_pending_native_image_paths(key)
    assert str(img) in paths


@pytest.mark.asyncio
async def test_text_http_image_url_detected_and_buffered():
    """An HTTP image URL typed in message text is buffered as a URL, not a path."""
    runner = _make_runner()
    source = _source("chat-2")

    await runner._prepare_inbound_message_text(
        event=MessageEvent(
            text="look at https://example.com/photo.jpg",
            source=source,
        ),
        source=source,
        history=[],
    )

    key = build_session_key(source)
    assert runner._consume_pending_native_image_paths(key) == []
    urls = runner._consume_pending_native_image_urls(key)
    assert "https://example.com/photo.jpg" in urls


@pytest.mark.asyncio
async def test_native_url_typed_in_text_becomes_content_part():
    """A typed HTTP URL survives native routing as an image_url content part.

    Regression for the review finding that URLs were appended to the local-path
    collection: ``build_native_content_parts`` skips non-files, so the URL was
    silently dropped in native mode.  The URL must travel the ``image_urls``
    parameter and reach the model as a real content part.
    """
    runner = _make_runner()
    source = _source("chat-7")
    url = "https://example.com/photo.jpg"

    await runner._prepare_inbound_message_text(
        event=MessageEvent(text=f"see {url}", source=source),
        source=source,
        history=[],
    )

    key = build_session_key(source)
    paths = runner._consume_pending_native_image_paths(key)
    urls = runner._consume_pending_native_image_urls(key)
    assert paths == []
    assert urls == [url]

    # Mirror the run_conversation consume site (gateway/run.py).
    from agent.image_routing import build_native_content_parts
    parts, skipped = build_native_content_parts("see", paths, image_urls=urls)
    assert url not in skipped
    assert any(
        p.get("type") == "image_url"
        and p.get("image_url", {}).get("url") == url
        for p in parts
    )


@pytest.mark.asyncio
async def test_url_only_message_still_routes_native():
    """A URL-only message (no local paths) still takes the routing path."""
    runner = _make_runner()
    source = _source("chat-9")
    url = "https://example.com/pic.png"

    await runner._prepare_inbound_message_text(
        event=MessageEvent(text=url, source=source),
        source=source,
        history=[],
    )

    key = build_session_key(source)
    assert runner._consume_pending_native_image_paths(key) == []
    assert runner._consume_pending_native_image_urls(key) == [url]


@pytest.mark.asyncio
async def test_channel_context_references_not_attached(tmp_path):
    """Image references that exist only in channel_context are NOT attached.

    Regression for the review finding that the scan ran on composed
    ``message_text``, which already includes ``event.channel_context`` — a
    historical channel/thread reference could become an attachment for a
    different trigger message.
    """
    hist_img = tmp_path / "historical.png"
    hist_img.write_bytes(b"\x89PNG")
    runner = _make_runner()
    source = _source("chat-8")
    text_url = "https://example.com/current.jpg"

    event = MessageEvent(
        text=f"see {text_url}",
        source=source,
        channel_context=(
            "[Recent channel messages]\n"
            "Alice: look at https://example.com/old.png\n"
            f"Bob: check {hist_img}"
        ),
    )

    await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    key = build_session_key(source)
    # Neither the historical URL nor the historical (existing) local path may
    # ride along with the current trigger message.
    assert runner._consume_pending_native_image_paths(key) == []
    urls = runner._consume_pending_native_image_urls(key)
    assert urls == [text_url]
    assert "https://example.com/old.png" not in urls


@pytest.mark.asyncio
async def test_text_image_paths_deduplicated_with_media_urls(tmp_path, monkeypatch):
    """Paths found in text are skipped if already present in event.media_urls."""
    img = tmp_path / "dup.png"
    img.write_bytes(b"\x89PNG")

    runner = _make_runner()
    source = _source("chat-3")

    event = MessageEvent(
        text=f"here is {img}",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=[str(img)],
        media_types=["image/png"],
    )

    await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    key = build_session_key(source)
    paths = runner._consume_pending_native_image_paths(key)
    # Should appear exactly once, not twice
    assert paths.count(str(img)) == 1


@pytest.mark.asyncio
async def test_text_url_deduplicated_with_media_url():
    """A URL already present in event.media_urls is buffered exactly once.

    Attached media that is already a remote URL rides the URL collection
    (native attachment passes URLs through verbatim), so the same URL found
    by the text scan is deduplicated against it.
    """
    url = "https://example.com/dup.png"
    runner = _make_runner()
    source = _source("chat-10")

    event = MessageEvent(
        text=f"see {url}",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=[url],
        media_types=["image/png"],
    )

    await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    key = build_session_key(source)
    paths = runner._consume_pending_native_image_paths(key)
    urls = runner._consume_pending_native_image_urls(key)
    # Buffered exactly once, via the URL collection — not doubled across the
    # path and URL buckets, and not dropped into the local-path bucket where
    # native attachment would reject it as a non-file.
    assert paths == []
    assert urls.count(url) == 1


@pytest.mark.asyncio
async def test_text_with_no_image_paths_unchanged():
    """Message text with no image references passes through normally."""
    runner = _make_runner()
    source = _source("chat-4")

    result = await runner._prepare_inbound_message_text(
        event=MessageEvent(text="just a plain message", source=source),
        source=source,
        history=[],
    )

    assert result is not None
    assert "just a plain message" in result
    # No image paths should be buffered
    key = build_session_key(source)
    assert runner._consume_pending_native_image_paths(key) == []
    assert runner._consume_pending_native_image_urls(key) == []


@pytest.mark.asyncio
async def test_text_image_path_in_code_block_ignored():
    """Image paths inside backtick code blocks are NOT auto-detected."""
    runner = _make_runner()
    source = _source("chat-5")

    await runner._prepare_inbound_message_text(
        event=MessageEvent(
            text="run `convert /tmp/image.png` to fix it",
            source=source,
        ),
        source=source,
        history=[],
    )

    key = build_session_key(source)
    paths = runner._consume_pending_native_image_paths(key)
    # Paths in backticks should be ignored by extract_image_refs
    assert "/tmp/image.png" not in paths


@pytest.mark.asyncio
async def test_extract_image_refs_failure_is_non_fatal(caplog):
    """If extract_image_refs raises, the message still processes normally."""
    runner = _make_runner()
    source = _source("chat-6")

    with patch(
        "agent.image_routing.extract_image_refs",
        side_effect=RuntimeError("boom"),
    ):
        result = await runner._prepare_inbound_message_text(
            event=MessageEvent(text="hello", source=source),
            source=source,
            history=[],
        )

    assert result is not None
    assert "hello" in result
