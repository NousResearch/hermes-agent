"""Post-stream media delivery is explicit-only (#20834).

``GatewayRunner._deliver_media_from_response`` runs AFTER streaming has sent
the visible reply. At that point a bare local filesystem path in the response
text is either text the user already saw, or stale inspected/tool content —
it is NOT an attachment request. Only explicit ``MEDIA:`` directives may
trigger post-stream uploads.

The non-streaming path (``gateway/platforms/base.py``) keeps its bare-path
auto-detect (``extract_local_files``) — that path controls what text is sent
and can strip the path from the visible reply, so auto-attach is intentional
there. This file pins the asymmetry.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


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
    runner = SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )
    # Plain-function attribute (no implicit self), so bind the mixin method explicitly.
    from functools import partial
    runner._ledger_flood_refused_media = partial(GatewayRunner._ledger_flood_refused_media, runner)
    return runner


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


@pytest.mark.asyncio
async def test_bare_local_path_in_streamed_reply_is_not_uploaded(tmp_path, monkeypatch):
    """The #20834 shape: visible reply contains a bare path (from inspected
    content), no MEDIA: directive — nothing may be uploaded post-stream."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "mockup.png")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}),
        f"The design lives at {media_file} if you want to look later.",
        _event(),
        adapter,
    )

    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_image_file.assert_not_awaited()
    adapter.send_document.assert_not_awaited()
    adapter.send_video.assert_not_awaited()
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_media_tag_still_delivers_post_stream(tmp_path, monkeypatch):
    """Explicit MEDIA: directives keep working after the #20834 fix."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "chart.png")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}),
        f"Here is the chart.\nMEDIA:{media_file}",
        _event(),
        adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    images_kwargs = adapter.send_multiple_images.await_args.kwargs
    assert images_kwargs["chat_id"] == "C123CHAN"
    assert str(media_file) in unquote(images_kwargs["images"][0][0])


def _patch_ledger(monkeypatch, *, enabled=True):
    """Capture ledger writes for the flood-notice path; returns (recorded, failed, scheduled)."""
    from gateway import delivery_ledger

    recorded, failed, scheduled = {}, {}, []
    monkeypatch.setattr(delivery_ledger, "ledger_enabled", lambda config=None: enabled)
    monkeypatch.setattr(delivery_ledger, "record_obligation", lambda **kwargs: recorded.update(kwargs))
    monkeypatch.setattr(delivery_ledger, "mark_failed", lambda oid, error="": failed.update(id=oid, error=error))
    return recorded, failed, scheduled


@pytest.mark.asyncio
async def test_flood_refused_post_stream_media_records_ledger_notice(tmp_path, monkeypatch):
    """A flood-refused upload must not vanish now the text fallback is gone (#125857): the refusal
    becomes a failed text obligation (a held-back notice) so the ledger's flood timer arms and
    redelivers the notice once the platform's penalty has passed."""
    from gateway.delivery_ledger import compute_obligation_id

    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.txt")
    adapter = _adapter()
    adapter.platform = Platform.SLACK
    adapter.send_document = AsyncMock(
        return_value=SendResult(success=False, error="flood_control:26.0", retry_after=26.0))
    recorded, failed, scheduled = _patch_ledger(monkeypatch)
    runner = _fake_runner({})
    runner._schedule_flood_redelivery = lambda platform, profile=None: scheduled.append((platform, profile))

    await GatewayRunner._deliver_media_from_response(
        runner, f"Here.\nMEDIA:{media_file}", _event(), adapter)

    adapter.send_document.assert_awaited_once()
    expected_content = ('📎 Attachment "report.txt" was refused by the platform rate limit '
                        "and was not delivered.")
    assert recorded["content"] == expected_content
    assert recorded["platform"] == "slack"
    assert recorded["chat_id"] == "C123CHAN"
    assert recorded["session_key"] == "slack:C123CHAN"
    assert failed == {"id": recorded["obligation_id"], "error": "flood_control:26.0"}
    assert failed["id"] == compute_obligation_id("slack:C123CHAN", "171.000001", expected_content)
    assert scheduled == [("slack", None)]


@pytest.mark.asyncio
async def test_flood_refused_media_notice_respects_ledger_gate(tmp_path, monkeypatch):
    """The notice follows the ``gateway.delivery_ledger`` config gate."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.txt")
    adapter = _adapter()
    adapter.platform = Platform.SLACK
    adapter.send_document = AsyncMock(
        return_value=SendResult(success=False, error="flood_control:26.0", retry_after=26.0))
    recorded, _, _ = _patch_ledger(monkeypatch, enabled=False)

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"Here.\nMEDIA:{media_file}", _event(), adapter)

    adapter.send_document.assert_awaited_once()
    assert recorded == {}


@pytest.mark.asyncio
async def test_delivered_media_send_is_not_ledgered(tmp_path, monkeypatch):
    """Only refusals are ledgered; a delivered upload records no obligation."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.txt")
    adapter = _adapter()
    adapter.platform = Platform.SLACK
    recorded, _, _ = _patch_ledger(monkeypatch)

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"Here.\nMEDIA:{media_file}", _event(), adapter)

    adapter.send_document.assert_awaited_once()
    assert recorded == {}


