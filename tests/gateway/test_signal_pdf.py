"""Regression coverage for Signal PDF attachment handling."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_runner() -> "GatewayRunner":  # type: ignore[name-defined]
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False
    return runner


def _pdf_event(platform: Platform) -> tuple[MessageEvent, SessionSource]:
    source = SessionSource(platform=platform, chat_id="1", chat_type="dm")
    return (
        MessageEvent(
            text="",
            message_type=MessageType.DOCUMENT,
            source=source,
            media_urls=["/tmp/report.pdf"],
            media_types=["application/pdf"],
        ),
        source,
    )


@pytest.mark.asyncio
async def test_signal_pdf_attachment_includes_local_extracted_text():
    """Signal PDFs include locally extracted text in the agent turn."""
    runner = _make_runner()
    event, source = _pdf_event(Platform.SIGNAL)

    with (
        patch(
            "gateway.run._extract_pdf_text",
            new=AsyncMock(return_value="PDF says: hello world"),
        ) as extract_pdf,
        patch(
            "tools.credential_files.to_agent_visible_cache_path",
            side_effect=lambda path: path,
        ),
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    extract_pdf.assert_awaited_once_with("/tmp/report.pdf")
    assert result is not None
    assert "PDF says: hello world" in result
    assert "PDF document" in result
    assert "/tmp/report.pdf" in result


@pytest.mark.asyncio
async def test_non_signal_pdf_remains_a_regular_document_note():
    """The shared gateway path must not auto-extract PDFs from other platforms."""
    runner = _make_runner()
    event, source = _pdf_event(Platform.TELEGRAM)

    with (
        patch(
            "gateway.run._extract_pdf_text",
            new=AsyncMock(
                side_effect=AssertionError("PDF extraction must stay Signal-only")
            ),
        ) as extract_pdf,
        patch(
            "tools.credential_files.to_agent_visible_cache_path",
            side_effect=lambda path: path,
        ),
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    extract_pdf.assert_not_awaited()
    assert result is not None
    assert "The user sent a document" in result
    assert "extracted text" not in result


class _HangingProcess:
    """Subprocess stub that completes only after the code kills it."""

    def __init__(self) -> None:
        self.returncode = None
        self.killed = False
        self._released = asyncio.Event()

    async def communicate(self) -> tuple[bytes, bytes]:
        await self._released.wait()
        self.returncode = -9
        return b"", b"timed out"

    def kill(self) -> None:
        self.killed = True
        self._released.set()


@pytest.mark.asyncio
async def test_pdf_extraction_kills_and_reaps_timed_out_process(monkeypatch):
    """A timed-out pdftotext process must not linger in the gateway."""
    from gateway.run import _extract_pdf_text

    process = _HangingProcess()

    async def timeout_immediately(_awaitable, *, timeout):
        raise asyncio.TimeoutError

    monkeypatch.setattr(
        "gateway.run.asyncio.create_subprocess_exec", AsyncMock(return_value=process)
    )
    monkeypatch.setattr("gateway.run.asyncio.wait_for", timeout_immediately)

    result = await _extract_pdf_text("/tmp/report.pdf")

    assert result is None
    assert process.killed is True
    assert process.returncode == -9
