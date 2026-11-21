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

import gateway.delivery_ledger as dl
from gateway.config import Platform
from persistence_assembly import profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.gateway.fixtures.storage_handles import storage_handle
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


# ---------------------------------------------------------------------------
# I-018 / #240: this lane must LEDGER the file before sending it, so a transport
# death leaves a row the reconnect/boot sweep can re-send by path — instead of a
# warning and a permanently lost attachment. The other half (a failed media row is
# actually re-sent by kind) is pinned in test_delivery_ledger_media.py.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    """Isolated state.db per test (same pattern as test_delivery_ledger.py)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_scope", lambda: profile_scope(home=home))
    yield


def _ledger_rows():
    holder = raw_holder(storage_handle(dl._scope().profile_dir))
    with holder._read_ctx() as conn:
        conn.row_factory = None
        return conn.execute(
            "SELECT media_path, media_kind, state FROM delivery_obligations "
            "WHERE media_path IS NOT NULL ORDER BY media_path").fetchall()


class _LedgerAdapter(BasePlatformAdapter):
    """Real ledger helpers, stubbed transport: only the methods the lane calls are faked."""

    def __init__(self, send_result: SendResult):
        self.platform = Platform.SLACK
        self.gateway_runner = SimpleNamespace()
        for method in ("send_document", "send_video", "send_voice", "send_image_file",
                       "send_multiple_images"):
            setattr(self, method, AsyncMock(return_value=send_result))

    async def connect(self, *, is_reconnect=False):  # pragma: no cover - abstract surface only
        return True

    async def disconnect(self):  # pragma: no cover - abstract surface only
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - abstract surface only
        return {"chat_id": chat_id}

    async def send(self, chat_id, content, **kwargs):  # pragma: no cover - abstract surface only
        return SendResult(success=True)


@pytest.mark.asyncio
async def test_post_stream_document_is_ledgered_before_send(tmp_path, monkeypatch):
    """The row must already exist (attempting) when send_document runs, and must close
    failed on send_path_degraded so the sweep re-sends the file."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    adapter = _LedgerAdapter(SendResult(success=True, message_id="doc"))
    seen = {}

    async def _send_document(chat_id, file_path, **kwargs):
        seen["at_send"] = _ledger_rows()
        return SendResult(success=False, error="send_path_degraded", retryable=True)

    adapter.send_document = _send_document

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"report attached\nMEDIA:{media_file}", _event(), adapter)

    assert seen["at_send"] == [(str(media_file), "document", "attempting")], seen
    assert _ledger_rows() == [(str(media_file), "document", "failed")], _ledger_rows()


@pytest.mark.asyncio
async def test_post_stream_document_marks_delivered_on_success(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "ok.pdf")
    adapter = _LedgerAdapter(SendResult(success=True, message_id="doc"))

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"MEDIA:{media_file}", _event(), adapter)

    assert adapter.send_document.await_count == 1
    assert _ledger_rows() == [(str(media_file), "document", "delivered")], _ledger_rows()


@pytest.mark.asyncio
async def test_post_stream_image_batch_ledgers_every_image(tmp_path, monkeypatch):
    """Images go out as one batch; each image still needs its own row, and a degraded
    batch must leave all of them failed (the sweep then re-sends one file per row)."""
    first = _allowed_media_path(tmp_path, monkeypatch, "one.png")
    second = tmp_path / "media-cache" / "two.png"
    second.write_bytes(b"media")
    second = second.resolve()
    adapter = _LedgerAdapter(SendResult(success=False, error="send_path_degraded", retryable=True))

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"MEDIA:{first}\nMEDIA:{second}", _event(), adapter)

    assert adapter.send_multiple_images.await_count == 1
    assert _ledger_rows() == [
        (str(first), "image", "failed"),
        (str(second), "image", "failed"),
    ], _ledger_rows()
