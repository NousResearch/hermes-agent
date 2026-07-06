from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from gateway.run import GatewayRunner


class _PendingAdapter:
    def __init__(self):
        self._pending_messages = {}


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner.adapters = {Platform.TELEGRAM: _PendingAdapter()}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._is_user_authorized = lambda _source: True
    return runner


@pytest.mark.asyncio
async def test_handle_message_does_not_priority_interrupt_photo_followup():
    runner = _make_runner()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="u1")
    session_key = build_session_key(source)
    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    event = MessageEvent(
        text="caption",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/photo-a.jpg"],
        media_types=["image/jpeg"],
    )

    result = await runner._handle_message(event)

    assert result is None
    running_agent.interrupt.assert_not_called()
    assert runner.adapters[Platform.TELEGRAM]._pending_messages[session_key] is event


@pytest.mark.asyncio
async def test_queue_mode_photo_followups_do_not_collapse():
    """In queue mode, two photos sent one-after-another mid-run must each
    become their OWN FIFO turn — not merge into a single pending slot.

    Regression for the single-slot collapse: previously every mid-run photo
    went through merge_pending_message_event into adapter._pending_messages,
    so a second (distinct, non-album) photo merged into / overwrote the first
    and was silently lost. In queue mode each photo is a complete user turn
    and must be enqueued via the FIFO chain (slot + overflow).
    """
    runner = _make_runner()
    runner._busy_input_mode = "queue"
    runner._queued_events = {}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="u1")
    session_key = build_session_key(source)
    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    photo_a = MessageEvent(
        text="first", message_type=MessageType.PHOTO, source=source,
        media_urls=["/tmp/photo-a.jpg"], media_types=["image/jpeg"],
    )
    photo_b = MessageEvent(
        text="second", message_type=MessageType.PHOTO, source=source,
        media_urls=["/tmp/photo-b.jpg"], media_types=["image/jpeg"],
    )

    assert await runner._handle_message(photo_a) is None
    assert await runner._handle_message(photo_b) is None
    running_agent.interrupt.assert_not_called()

    adapter = runner.adapters[Platform.TELEGRAM]
    # First photo occupies the "next-up" slot, second lands in the overflow —
    # total FIFO depth is 2, and both distinct events survive independently.
    assert runner._queue_depth(session_key, adapter=adapter) == 2
    assert adapter._pending_messages[session_key] is photo_a
    assert runner._queued_events[session_key] == [photo_b]
    # The two events did NOT get merged: photo_a still carries only its own media.
    assert photo_a.media_urls == ["/tmp/photo-a.jpg"]
    assert photo_b.media_urls == ["/tmp/photo-b.jpg"]


@pytest.mark.asyncio
async def test_interrupt_mode_photo_burst_still_merges():
    """In non-queue (interrupt) mode, the legacy single-slot merge is kept so
    rapid photo bursts still coalesce into one pending event (album behavior).
    """
    runner = _make_runner()
    runner._busy_input_mode = "interrupt"
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="u1")
    session_key = build_session_key(source)
    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    photo_a = MessageEvent(
        text="a", message_type=MessageType.PHOTO, source=source,
        media_urls=["/tmp/photo-a.jpg"], media_types=["image/jpeg"],
    )
    photo_b = MessageEvent(
        text="b", message_type=MessageType.PHOTO, source=source,
        media_urls=["/tmp/photo-b.jpg"], media_types=["image/jpeg"],
    )

    assert await runner._handle_message(photo_a) is None
    assert await runner._handle_message(photo_b) is None
    running_agent.interrupt.assert_not_called()

    adapter = runner.adapters[Platform.TELEGRAM]
    merged = adapter._pending_messages[session_key]
    # Both photos merged into the single slot event (legacy album coalescing).
    assert merged.media_urls == ["/tmp/photo-a.jpg", "/tmp/photo-b.jpg"]
