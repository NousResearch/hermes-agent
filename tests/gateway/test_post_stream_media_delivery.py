"""
Tests for post-stream media delivery in the gateway.

Verifies that _deliver_media_from_response() only uploads files explicitly
requested via MEDIA: directives, and does NOT upload bare local file paths
that happen to appear in the response text (e.g. from tool output).

Regression test for #20834.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class FakeSource:
    def __init__(self, chat_id="test_chat", thread_id=None):
        self.chat_id = chat_id
        self.thread_id = thread_id


class FakeEvent:
    def __init__(self, chat_id="test_chat", thread_id=None):
        self.source = FakeSource(chat_id, thread_id)


class FakeAdapter:
    """Minimal adapter mock that records delivery calls."""

    name = "test"

    def __init__(self, media_files=None, local_files=None):
        self._media_files = media_files or []
        self._local_files = local_files or []
        self.send_image_file = AsyncMock()
        self.send_video = AsyncMock()
        self.send_voice = AsyncMock()
        self.send_document = AsyncMock()

    def extract_media(self, response):
        """Return explicit MEDIA: directives."""
        return self._media_files, response

    def extract_images(self, response):
        """Return images and cleaned text."""
        return [], response

    def extract_local_files(self, content):
        """Return bare local file paths."""
        return self._local_files, content


@pytest.mark.asyncio
async def test_post_stream_ignores_bare_local_paths():
    """Bare local paths in response text must NOT be uploaded."""
    from gateway.run import GatewayRunner

    runner = MagicMock(spec=GatewayRunner)
    # Bind the real method
    runner._deliver_media_from_response = GatewayRunner._deliver_media_from_response.__get__(runner)

    # Adapter finds a bare local path but no explicit MEDIA:
    adapter = FakeAdapter(
        media_files=[],
        local_files=["/tmp/some_image.png"],
    )
    event = FakeEvent()

    await runner._deliver_media_from_response(
        response="The image at /tmp/some_image.png was analyzed.",
        event=event,
        adapter=adapter,
    )

    # No files should have been sent
    adapter.send_image_file.assert_not_called()
    adapter.send_document.assert_not_called()
    adapter.send_video.assert_not_called()
    adapter.send_voice.assert_not_called()


@pytest.mark.asyncio
async def test_post_stream_delivers_explicit_media():
    """Explicit MEDIA: directives must still be delivered."""
    from gateway.run import GatewayRunner

    runner = MagicMock(spec=GatewayRunner)
    runner._deliver_media_from_response = GatewayRunner._deliver_media_from_response.__get__(runner)

    adapter = FakeAdapter(
        media_files=[("/tmp/chart.png", False)],
        local_files=[],
    )
    event = FakeEvent()

    await runner._deliver_media_from_response(
        response="Here is the chart:\nMEDIA:/tmp/chart.png",
        event=event,
        adapter=adapter,
    )

    # Explicit MEDIA: should be delivered as image
    adapter.send_image_file.assert_called_once()
    call_kwargs = adapter.send_image_file.call_args
    assert call_kwargs.kwargs.get("image_path") == "/tmp/chart.png" or \
           (call_kwargs.args and call_kwargs.args[1] == "/tmp/chart.png")


@pytest.mark.asyncio
async def test_post_stream_mixed_explicit_and_bare():
    """When both explicit MEDIA: and bare paths exist, only MEDIA: is delivered."""
    from gateway.run import GatewayRunner

    runner = MagicMock(spec=GatewayRunner)
    runner._deliver_media_from_response = GatewayRunner._deliver_media_from_response.__get__(runner)

    adapter = FakeAdapter(
        media_files=[("/tmp/intended.png", False)],
        local_files=["/tmp/accidental.png", "/tmp/another.jpg"],
    )
    event = FakeEvent()

    await runner._deliver_media_from_response(
        response="Chart: MEDIA:/tmp/intended.png\nAlso found /tmp/accidental.png",
        event=event,
        adapter=adapter,
    )

    # Only the explicit MEDIA: file should be sent
    adapter.send_image_file.assert_called_once()
    adapter.send_document.assert_not_called()
