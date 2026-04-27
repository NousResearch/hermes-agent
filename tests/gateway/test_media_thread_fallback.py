"""Regression test for media-attachment threading on top-level DMs.

Bug: when a user sends a top-level DM (no thread_ts), the gateway sets
``event.source.thread_id = None`` by design (so DM conversations share one
continuous session).  The text-reply path then threads the assistant's
response under the user's message via ``reply_to``, but the MEDIA delivery
path only consulted ``event.source.thread_id`` — so any file attachment
landed as a new top-level message in the DM, split away from the text
reply that referenced it.

Fix: the MEDIA delivery path should fall back to ``event.message_id``
(the user's original message ts) when ``thread_id`` is empty, so the file
upload lands in the same thread as the text reply.

This test exercises ``GatewayRunner._deliver_media_from_response`` with a minimal
stub adapter and asserts the ``metadata`` kwarg passed to ``send_document``
carries the expected thread id.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.run import GatewayRunner as _GatewayRunner

GatewayRunner = _GatewayRunner


def _make_pdf(tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "resume.pdf")
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4\n%stub\n")
    return path


def _make_event(*, thread_id: str | None, message_id: str) -> Any:
    event = MagicMock()
    event.source = MagicMock()
    event.source.thread_id = thread_id
    event.source.chat_id = "D0AQY7F15H8"
    event.message_id = message_id
    return event


def _make_adapter() -> Any:
    adapter = MagicMock()
    adapter.name = "slack"
    adapter.extract_media = MagicMock(return_value=([], []))
    adapter.extract_images = MagicMock(return_value=([], ""))
    adapter.extract_local_files = MagicMock(return_value=([], ""))
    adapter.send_voice = AsyncMock()
    adapter.send_video = AsyncMock()
    adapter.send_image_file = AsyncMock()
    adapter.send_document = AsyncMock()
    return adapter


def _call_deliver(response: str, event: Any, adapter: Any) -> None:
    gateway = GatewayRunner.__new__(GatewayRunner)
    bound = GatewayRunner._deliver_media_from_response.__get__(gateway, GatewayRunner)
    asyncio.get_event_loop().run_until_complete(bound(response, event, adapter))


def test_media_falls_back_to_message_id_when_thread_id_empty(tmp_path):
    """Top-level DM: thread_id=None must fall back to message_id."""
    pdf = _make_pdf(str(tmp_path))
    adapter = _make_adapter()
    adapter.extract_media.return_value = ([(pdf, False)], "")
    event = _make_event(thread_id=None, message_id="1776825784.764809")

    _call_deliver(f"Your resume is ready.\nMEDIA:{pdf}", event, adapter)

    adapter.send_document.assert_awaited_once()
    kwargs = adapter.send_document.await_args.kwargs
    assert kwargs["metadata"] == {"thread_id": "1776825784.764809"}, (
        f"expected fallback to message_id, got {kwargs.get('metadata')}"
    )


def test_media_prefers_thread_id_when_set(tmp_path):
    """Explicit thread_id wins over message_id fallback."""
    pdf = _make_pdf(str(tmp_path))
    adapter = _make_adapter()
    adapter.extract_media.return_value = ([(pdf, False)], "")
    event = _make_event(
        thread_id="1776687617.659629",
        message_id="1776825784.764809",
    )

    _call_deliver(f"MEDIA:{pdf}", event, adapter)

    kwargs = adapter.send_document.await_args.kwargs
    assert kwargs["metadata"] == {"thread_id": "1776687617.659629"}


def test_media_metadata_none_when_both_empty(tmp_path):
    """Defensive: if neither is set, metadata stays None (original behaviour)."""
    pdf = _make_pdf(str(tmp_path))
    adapter = _make_adapter()
    adapter.extract_media.return_value = ([(pdf, False)], "")
    event = _make_event(thread_id=None, message_id=None)

    _call_deliver(f"MEDIA:{pdf}", event, adapter)

    kwargs = adapter.send_document.await_args.kwargs
    assert kwargs["metadata"] is None
