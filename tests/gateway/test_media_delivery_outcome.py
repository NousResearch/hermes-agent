"""Regression tests for media-only turn delivery outcomes (#106153).

Attachment sends (image batches and per-file media) must feed the
``delivery_attempted``/``delivery_succeeded`` accounting that decides
``ProcessingOutcome``, so a media-only reply that delivered its attachment
reports SUCCESS instead of FAILURE.
"""

import asyncio
from typing import Optional, Tuple, List, Dict, Any

import pytest

from gateway.config import PlatformConfig, Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.session import SessionSource, build_session_key


class MediaOutcomeAdapter(BasePlatformAdapter):
    """Minimal adapter whose attachment sends return configurable results."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.SLACK)
        self.processing_outcomes = []
        self.image_batch_result: Optional[SendResult] = SendResult(success=True)
        self.document_result = SendResult(success=True)

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="m-1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def send_document(self, chat_id, file_path, caption=None, metadata=None, **kwargs) -> SendResult:
        return self.document_result

    async def send_multiple_images(
        self, chat_id: str, images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None, human_delay: float = 0.0,
    ) -> Optional[SendResult]:
        return self.image_batch_result

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        self.processing_outcomes.append(outcome)


def _make_event() -> MessageEvent:
    return MessageEvent(
        text="send me the photo",
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            chat_type="channel",
            thread_id="171717",
            user_id="U123",
        ),
        message_id="m-1",
    )


async def _run_turn(adapter: MediaOutcomeAdapter, response: str):
    async def _handler(_event):
        return response

    adapter.set_message_handler(_handler)
    adapter._keep_typing = lambda *_args, **_kwargs: asyncio.Event().wait()
    event = _make_event()
    await adapter._process_message_background(event, build_session_key(event.source))


@pytest.mark.asyncio
async def test_media_only_image_delivery_reports_success(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0")
    adapter = MediaOutcomeAdapter()
    adapter.image_batch_result = SendResult(success=True)

    await _run_turn(adapter, f"MEDIA:{image}")

    assert adapter.processing_outcomes == [ProcessingOutcome.SUCCESS]


@pytest.mark.asyncio
async def test_media_only_image_failure_reports_failure(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0")
    adapter = MediaOutcomeAdapter()
    adapter.image_batch_result = SendResult(success=False, error="RPC send failed")

    await _run_turn(adapter, f"MEDIA:{image}")

    assert adapter.processing_outcomes == [ProcessingOutcome.FAILURE]


@pytest.mark.asyncio
async def test_media_only_document_delivery_reports_success(tmp_path):
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4")
    adapter = MediaOutcomeAdapter()
    adapter.document_result = SendResult(success=True)

    await _run_turn(adapter, f"MEDIA:{doc}")

    assert adapter.processing_outcomes == [ProcessingOutcome.SUCCESS]


@pytest.mark.asyncio
async def test_media_only_document_failure_reports_failure(tmp_path):
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4")
    adapter = MediaOutcomeAdapter()
    adapter.document_result = SendResult(success=False, error="file too large")

    await _run_turn(adapter, f"MEDIA:{doc}")

    assert adapter.processing_outcomes == [ProcessingOutcome.FAILURE]


@pytest.mark.asyncio
async def test_platform_opted_out_of_image_accounting_keeps_old_behaviour(tmp_path):
    # Overrides that still return None from send_multiple_images are opted out
    # of image-delivery accounting: a media-only image turn stays FAILURE for
    # them (pre-#106153 behaviour) rather than guessing success.
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0")
    adapter = MediaOutcomeAdapter()
    adapter.image_batch_result = None

    await _run_turn(adapter, f"MEDIA:{image}")

    assert adapter.processing_outcomes == [ProcessingOutcome.FAILURE]


@pytest.mark.asyncio
async def test_delivered_text_outranks_failed_image_batch(tmp_path):
    # The roll-up is ``delivery_succeeded or ...``: once the text went out, a
    # failed image batch must not flip the turn to FAILURE.
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0")
    adapter = MediaOutcomeAdapter()
    adapter.image_batch_result = SendResult(success=False, error="RPC send failed")

    await _run_turn(adapter, f"Here is the photo.\nMEDIA:{image}")

    assert adapter.processing_outcomes == [ProcessingOutcome.SUCCESS]
