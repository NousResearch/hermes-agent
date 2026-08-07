"""Regression tests for /queue FIFO MEDIA delivery (#18539).

Before the fix, the queue-drain path in ``GatewayRunner._run_agent`` sent
the first response with ``adapter.send(raw_text)`` directly. Per-item
``MEDIA:/path`` markers leaked through as plain text and only the LAST
queue item's media file reached the platform — preceding items had their
attachments silently dropped (the file existed on disk but never
uploaded).

These tests target ``_send_queued_followup_first_response`` — the helper
the queue-drain path now delegates to — and assert that:

1. ``extract_media`` is called before ``send``.
2. The cleaned (MEDIA-tag-free) text is what reaches ``adapter.send``.
3. Each extracted file is uploaded via ``send_document`` or ``send_voice``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.run import GatewayRunner


def _stub_runner() -> GatewayRunner:
    """A bare GatewayRunner just for invoking instance methods in tests.

    ``object.__new__`` skips ``__init__`` so we don't need to stand up the
    full gateway config / adapter set. The methods under test here only
    touch their explicit arguments.
    """
    return object.__new__(GatewayRunner)


def _stub_source():
    return SimpleNamespace(chat_id="chat-1", platform="telegram", thread_id=None)


@pytest.mark.asyncio
async def test_extract_media_runs_before_send():
    """The cleaned text — not the raw MEDIA-tagged response — must reach
    ``adapter.send``. Otherwise users see literal ``MEDIA:/path`` lines
    instead of the upload."""
    runner = _stub_runner()
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.send_document = AsyncMock()
    adapter.send_voice = AsyncMock()
    adapter.extract_media = MagicMock(
        return_value=([("/tmp/report.pdf", False)], "Here is the report.")
    )

    await runner._send_queued_followup_first_response(
        adapter,
        _stub_source(),
        "Here is the report.\nMEDIA:/tmp/report.pdf",
        metadata=None,
        session_key="sess-1",
    )

    adapter.extract_media.assert_called_once_with(
        "Here is the report.\nMEDIA:/tmp/report.pdf"
    )
    adapter.send.assert_awaited_once_with(
        "chat-1",
        "Here is the report.",
        metadata=None,
    )


@pytest.mark.asyncio
async def test_documents_are_uploaded_via_send_document():
    runner = _stub_runner()
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.send_document = AsyncMock()
    adapter.send_voice = AsyncMock()
    adapter.extract_media = MagicMock(
        return_value=(
            [
                ("/tmp/a.pdf", False),
                ("/tmp/b.csv", False),
            ],
            "Two files attached.",
        )
    )

    await runner._send_queued_followup_first_response(
        adapter,
        _stub_source(),
        "raw response with media tags",
        metadata={"thread_id": "42"},
        session_key="sess-1",
    )

    assert adapter.send_document.await_count == 2
    adapter.send_document.assert_any_await(
        chat_id="chat-1", file_path="/tmp/a.pdf", metadata={"thread_id": "42"}
    )
    adapter.send_document.assert_any_await(
        chat_id="chat-1", file_path="/tmp/b.csv", metadata={"thread_id": "42"}
    )
    adapter.send_voice.assert_not_called()


@pytest.mark.asyncio
async def test_voice_flagged_files_route_to_send_voice():
    runner = _stub_runner()
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.send_document = AsyncMock()
    adapter.send_voice = AsyncMock()
    adapter.extract_media = MagicMock(
        return_value=([("/tmp/note.ogg", True)], "Voice note attached.")
    )

    await runner._send_queued_followup_first_response(
        adapter,
        _stub_source(),
        "Voice note attached. MEDIA:/tmp/note.ogg",
        metadata=None,
        session_key="sess-1",
    )

    adapter.send_voice.assert_awaited_once_with(
        chat_id="chat-1", audio_path="/tmp/note.ogg", metadata=None
    )
    adapter.send_document.assert_not_called()


@pytest.mark.asyncio
async def test_individual_file_failure_does_not_block_remaining_uploads():
    """If one media upload raises, remaining files must still attempt
    upload — losing one attachment is better than losing all."""
    runner = _stub_runner()
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.send_document = AsyncMock(side_effect=[RuntimeError("S3 down"), None])
    adapter.send_voice = AsyncMock()
    adapter.extract_media = MagicMock(
        return_value=(
            [
                ("/tmp/broken.pdf", False),
                ("/tmp/ok.pdf", False),
            ],
            "Two files.",
        )
    )

    await runner._send_queued_followup_first_response(
        adapter,
        _stub_source(),
        "raw",
        metadata=None,
        session_key="sess-1",
    )

    assert adapter.send_document.await_count == 2


@pytest.mark.asyncio
async def test_no_media_just_sends_text():
    """Responses with no MEDIA tags should behave exactly like the
    pre-fix path: a single send() call, no document uploads."""
    runner = _stub_runner()
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.send_document = AsyncMock()
    adapter.send_voice = AsyncMock()
    adapter.extract_media = MagicMock(return_value=([], "Plain text response."))

    await runner._send_queued_followup_first_response(
        adapter,
        _stub_source(),
        "Plain text response.",
        metadata=None,
        session_key="sess-1",
    )

    adapter.send.assert_awaited_once()
    adapter.send_document.assert_not_called()
    adapter.send_voice.assert_not_called()
