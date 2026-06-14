from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._thread_metadata_for_source = lambda source, _reply=None: (
        {"thread_id": source.thread_id} if source.thread_id else None
    )
    return runner


def _event() -> MessageEvent:
    return MessageEvent(
        text="next",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="273403055",
            chat_type="dm",
            user_id="273403055",
            thread_id="374476",
        ),
        message_id="25741",
    )


def _adapter():
    return SimpleNamespace(
        name="Telegram",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=lambda text: ([], text),
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(),
        send_voice=AsyncMock(),
        send_video=AsyncMock(),
        send_document=AsyncMock(),
        send_multiple_images=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_queued_first_response_strips_media_and_sends_document(tmp_path):
    html = tmp_path / "pivin-scenario.html"
    html.write_text("<!doctype html><title>Pivin</title>", encoding="utf-8")

    adapter = _adapter()
    response = f"Done.\n\nMEDIA:{html}"

    await _runner()._send_queued_first_response(
        response,
        _event(),
        adapter,
        metadata={"thread_id": "374476"},
    )

    adapter.send.assert_awaited_once_with(
        "273403055",
        "Done.",
        metadata={"thread_id": "374476"},
    )
    adapter.send_document.assert_awaited_once()
    assert adapter.send_document.await_args.kwargs["chat_id"] == "273403055"
    assert adapter.send_document.await_args.kwargs["file_path"] == str(html)
    assert adapter.send_document.await_args.kwargs["metadata"] == {"thread_id": "374476"}


@pytest.mark.asyncio
async def test_queued_first_response_all_media_sends_no_empty_text(tmp_path):
    html = tmp_path / "handoff.html"
    html.write_text("<!doctype html><title>Only media</title>", encoding="utf-8")

    adapter = _adapter()

    await _runner()._send_queued_first_response(
        f"MEDIA:{html}",
        _event(),
        adapter,
        metadata={"thread_id": "374476"},
    )

    adapter.send.assert_not_awaited()
    adapter.send_document.assert_awaited_once()
    assert adapter.send_document.await_args.kwargs["file_path"] == str(html)
