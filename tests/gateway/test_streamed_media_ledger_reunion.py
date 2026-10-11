"""Regression: MEDIA: tags lost from final_response on a broken/reconciled stream.

A stream consumer hides ``MEDIA:`` tags from display while the gateway re-scans
``final_response`` post-stream to upload the files. If the stream breaks mid-way
and the agent loop reconciles a SHORTER ``final_response`` (the pre-break intro),
the cleaned payload still matches (tags are invisible to display), suppression
fires, and the raw tags simply vanish from the re-scan — attachments silently
never arrive.

Fix contract pinned here:

* the consumer records the UNSTRIPPED turn-final ledger (``delivered_final_raw``)
  alongside the cleaned payload it already records;
* the gateway stashes that ledger on the result (``_streamed_final_raw``) when it
  marks ``already_sent``;
* ``_deliver_media_from_response`` unions the ledger's MEDIA tags into the
  re-scan, deduped by exact path so normal turns stay single-shot.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.waiting import async_wait_until


# ---------------------------------------------------------------------------
# shared helpers (same shapes as test_post_stream_media_delivery.py)
# ---------------------------------------------------------------------------

def _event():
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123CHAN",
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


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    return media_file.resolve()


def _consumer_adapter():
    """BasePlatformAdapter subclass with stream-is-the-message drafts."""
    A = type("StreamIsMsgAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 39000})
    A.__abstractmethods__ = frozenset()
    a = A.__new__(A)
    a._typing_paused = set()
    a._fatal_error_message = None
    a.draft_stream_is_message = True
    a.send_calls = []

    def _supports(chat_type=None, metadata=None):
        return True
    a.supports_draft_streaming = _supports

    async def _send_draft(*, chat_id, draft_id, content, metadata=None):
        a.send_calls.append({"content": content, "draft": True})
        return SendResult(success=True, message_id=None)
    a.send_draft = _send_draft

    async def _send(chat_id, content, reply_to=None, metadata=None, **kw):
        a.send_calls.append({"content": content})
        return SendResult(success=True, message_id="sealed_ts_1")
    a.send = _send

    async def _edit(chat_id, message_id, content, **kw):
        return SendResult(success=True, message_id=message_id)
    a.edit_message = _edit
    return a


# ---------------------------------------------------------------------------
# consumer side: the raw ledger survives display stripping and broken finals
# ---------------------------------------------------------------------------

class TestConsumerRawLedger:
    @pytest.mark.asyncio
    async def test_clean_final_records_raw_and_clean_payloads(self, tmp_path):
        """Happy path: raw and cleaned payloads both recorded; the cleaned one
        has no directives (matching legacy behavior) while the raw keeps them."""
        adapter = _consumer_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=1, cursor="",
        )
        sc = GatewayStreamConsumer(adapter, "D1", cfg)
        task = asyncio.create_task(sc.run())
        sc.on_delta("here you go\nMEDIA:/tmp/boat.jpg")
        await async_wait_until(lambda: adapter.send_calls, what="delta flushed to the draft")
        sc.finish("here you go\nMEDIA:/tmp/boat.jpg")
        await task

        assert sc.delivered_final_raw is not None
        assert "MEDIA:/tmp/boat.jpg" in sc.delivered_final_raw
        # Legacy cleaned payload: directives hidden (what the user saw).
        assert "MEDIA:" not in (sc._delivered_final_text or "")
        # Reconcile semantics unchanged: cleaned final_response still matches.
        assert sc.delivered_final_matches("here you go\nMEDIA:/tmp/boat.jpg") is True

    @pytest.mark.asyncio
    async def test_broken_stream_final_keeps_streamed_raw_tags(self, tmp_path):
        """The 2026-10-10 incident shape: deltas stream a tag-bearing response,
        then the loop reconciles a shorter tag-less final_response. The cleaned
        payload matches the short final (suppression fires), and the raw ledger
        must still carry the tags for the post-stream re-scan to recover."""
        adapter = _consumer_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=1, cursor="",
        )
        sc = GatewayStreamConsumer(adapter, "D1", cfg)
        full = (
            "Four boats, one tailwind.\n"
            "MEDIA:/tmp/M1.jpg\nMEDIA:/tmp/M2.jpg\nMEDIA:/tmp/M3.jpg\nMEDIA:/tmp/M4.jpg"
        )
        task = asyncio.create_task(sc.run())
        sc.on_delta(full)
        await async_wait_until(lambda: adapter.send_calls, what="delta flushed to the draft")
        # Stream broke after delivering everything; reconciled final is the intro only.
        sc.finish("Four boats, one tailwind.")
        await task

        assert sc.final_content_delivered is True
        # Suppression stays valid: the cleaned payload matches the short final…
        assert sc.delivered_final_matches("Four boats, one tailwind.") is True
        # …but the raw ledger kept all four tags.
        assert sc.delivered_final_raw is not None
        for tag in ("M1.jpg", "M2.jpg", "M3.jpg", "M4.jpg"):
            assert tag in sc.delivered_final_raw


# ---------------------------------------------------------------------------
# gateway side: the union re-scan
# ---------------------------------------------------------------------------

class TestLedgerUnionRescan:
    @pytest.mark.asyncio
    async def test_tags_lost_from_final_response_are_recovered(self, tmp_path, monkeypatch):
        """response has no tags (reconciled intro only); the raw ledger does.
        The files must be delivered exactly once."""
        m1 = _allowed_media_path(tmp_path, monkeypatch, "M1.jpg")
        m2 = _allowed_media_path(tmp_path, monkeypatch, "M2.jpg")
        adapter = _adapter()
        ledger = f"Four boats.\nMEDIA:{m1}\nMEDIA:{m2}"

        await GatewayRunner._deliver_media_from_response(
            _fake_runner({}),
            "Four boats.",
            _event(),
            adapter,
            extra_text=ledger,
        )

        adapter.send_multiple_images.assert_awaited_once()
        images = adapter.send_multiple_images.await_args.kwargs["images"]
        assert len(images) == 2
        assert str(m1) in unquote(images[0][0])
        assert str(m2) in unquote(images[1][0])

    @pytest.mark.asyncio
    async def test_tags_in_both_sources_dedup_to_one_upload(self, tmp_path, monkeypatch):
        """Normal turn: final_response still carries the tags AND the ledger does.
        Exact-path dedup must keep the batch single-shot."""
        m1 = _allowed_media_path(tmp_path, monkeypatch, "M1.jpg")
        adapter = _adapter()
        text = f"Here you go.\nMEDIA:{m1}"

        await GatewayRunner._deliver_media_from_response(
            _fake_runner({}),
            text,
            _event(),
            adapter,
            extra_text=text,  # identical ledger (normal non-broken stream)
        )

        adapter.send_multiple_images.assert_awaited_once()
        images = adapter.send_multiple_images.await_args.kwargs["images"]
        assert len(images) == 1

    @pytest.mark.asyncio
    async def test_none_extra_text_keeps_current_behavior(self, tmp_path, monkeypatch):
        """No ledger stashed (non-streamed or pre-fix turns): unchanged scan of
        the response alone — bare paths still never upload post-stream (#20834)."""
        m1 = _allowed_media_path(tmp_path, monkeypatch, "M1.jpg")
        adapter = _adapter()

        await GatewayRunner._deliver_media_from_response(
            _fake_runner({}),
            f"The design lives at {m1} if you want it.",
            _event(),
            adapter,
            extra_text=None,
        )

        adapter.send_multiple_images.assert_not_awaited()
        adapter.send_image_file.assert_not_awaited()
        adapter.send_document.assert_not_awaited()