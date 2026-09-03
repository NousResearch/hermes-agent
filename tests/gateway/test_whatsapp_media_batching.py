"""Media album batching for the WhatsApp adapter (PR #75414).

WhatsApp delivers album images/videos as separate messages in rapid
succession.  Without batching each photo triggers a separate agent turn,
racing downloads and fragmenting replies.  This ports the Feishu
adapter's _enqueue_media_event / _flush_media_batch pattern.

Regression coverage:
- media events from the same session aggregate into a single batch
- media events from DIFFERENT profiles do not share a batch
- pending batches are cancelled on disconnect
"""

import asyncio

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def _make_adapter(**extra):
    base = {"session_name": "test"}
    base.update(extra)
    return WhatsAppAdapter(PlatformConfig(enabled=True, extra=base))


def _media_event(media_urls, profile=None, chat_type="dm", chat_id="chat123"):
    src = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id="user1",
        user_name="tester",
        profile=profile,
    )
    return MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=src,
        media_urls=media_urls,
        media_types=["image/jpeg"] * len(media_urls),
    )


def test_media_batch_delay_overridden_via_config_extra():
    adapter = _make_adapter(media_batch_delay_seconds="1.5")
    assert adapter._media_batch_delay_seconds == 1.5


def test_media_batch_key_includes_profile():
    """Two otherwise-identical events for separate profiles must NOT share a batch."""
    adapter = _make_adapter()
    e1 = _media_event(["/tmp/a.jpg"], profile="main")
    e2 = _media_event(["/tmp/b.jpg"], profile="work")
    assert adapter._media_batch_key(e1) != adapter._media_batch_key(e2)


def test_media_batch_key_same_profile_same_chat_aggregates():
    """Same profile + same chat => same batch key (album images merge)."""
    adapter = _make_adapter()
    e1 = _media_event(["/tmp/a.jpg"], profile="main")
    e2 = _media_event(["/tmp/b.jpg"], profile="main")
    assert adapter._media_batch_key(e1) == adapter._media_batch_key(e2)


def test_enqueue_media_event_aggregates_attachments():
    """Two rapid media events merge media_urls into the pending batch."""
    adapter = _make_adapter()
    e1 = _media_event(["/tmp/a.jpg"], profile="main")
    e2 = _media_event(["/tmp/b.jpg"], profile="main")

    async def run():
        await adapter._enqueue_media_event(e1)
        await adapter._enqueue_media_event(e2)
        pending = adapter._pending_media_batches
        # Both events should live under the same key with merged urls
        key = adapter._media_batch_key(e1)
        assert key in pending
        assert pending[key].media_urls == ["/tmp/a.jpg", "/tmp/b.jpg"]

    asyncio.run(run())


def test_flush_media_batch_delivers_single_event():
    """After the quiet period, the batch flushes as one event via handle_message."""
    adapter = _make_adapter(media_batch_delay_seconds=0.05)
    delivered = []

    async def fake_handle(event):
        delivered.append(event)

    adapter.handle_message = fake_handle
    e1 = _media_event(["/tmp/a.jpg"], profile="main")
    e2 = _media_event(["/tmp/b.jpg"], profile="main")

    async def run():
        await adapter._enqueue_media_event(e1)
        await adapter._enqueue_media_event(e2)
        key = adapter._media_batch_key(e1)
        await adapter._flush_media_batch(key)
        # Give the flush task a beat to finish
        await asyncio.sleep(0.1)

    asyncio.run(run())
    assert len(delivered) == 1
    assert delivered[0].media_urls == ["/tmp/a.jpg", "/tmp/b.jpg"]


def test_disconnect_cancels_pending_batches():
    """disconnect() cancels pending media-batch tasks and clears state."""
    adapter = _make_adapter()
    e1 = _media_event(["/tmp/a.jpg"], profile="main")

    async def run():
        await adapter._enqueue_media_event(e1)
        assert adapter._pending_media_batches
        assert adapter._pending_media_batch_tasks
        await adapter.disconnect()
        assert not adapter._pending_media_batches
        assert not adapter._pending_media_batch_tasks

    asyncio.run(run())


def test_disconnect_works_without_media_state(tmp_path):
    """Lightweight harnesses (__new__ + manual attrs) must not crash on disconnect."""
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra={"session_name": "test"})
    adapter._bridge_process = None
    adapter._http_session = None
    adapter._poll_task = None
    adapter._running = True
    adapter._session_lock_identity = None
    adapter._session_path = tmp_path
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._reply_prefix = None
    adapter._send_read_receipts = False
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._message_queue = asyncio.Queue()
    adapter._message_queue_state = None
    # Deliberately do NOT set _pending_media_batch_tasks / _pending_media_batches

    async def run():
        await adapter.disconnect()

    asyncio.run(run())
