"""Regression coverage for the Keryx SSE side-channel hub (``gateway.keryx_stream``).

Locks the behaviour contract the gateway relies on:

* stream events mirror, in order, to an attached subscriber;
* ``suppress_protocol_edits`` gates on subscriber presence (live subscriber →
  the consumer commits only the final message, no homeserver edit stream);
* subscriptions are keyed by ``(platform, chat_id)`` — one room never leaks
  into another;
* with no subscriber, publishing is a silent no-op and the Matrix
  ``FALLBACK_EDITS`` tier drops to the throttled ``m.replace`` path.

The hub captures the running loop at ``subscribe`` time and delivers via
``call_soon_threadsafe``, so each scenario runs inside ``asyncio.run`` and
yields once (``await asyncio.sleep(0)``) to let the scheduled callbacks fire.
"""

import asyncio
from enum import Enum

from gateway import keryx_stream as ks


class _Platform(Enum):
    MATRIX = "Matrix"


class _FakeAdapter:
    """Minimal stand-in for a gateway adapter (only ``platform`` is read)."""

    def __init__(self, platform: _Platform = _Platform.MATRIX) -> None:
        self.platform = platform
        self.name = str(platform.value).lower()


def _drain(queue: "asyncio.Queue") -> list:
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def test_side_channel_mirrors_events_and_gates_edit_suppression(monkeypatch):
    async def scenario():
        # Isolate from the module-global hub so cross-test state can't leak.
        monkeypatch.setattr(ks, "hub", ks.KeryxStreamHub())
        adapter = _FakeAdapter()
        chat = "!room:example.org"
        platform = ks._platform_of(adapter)
        assert platform == "matrix"

        # No subscriber: publishing is a silent no-op and edits are NOT forced off.
        ks.publish_delta(adapter, chat, "before-subscribe")  # must not raise
        assert ks.suppress_protocol_edits(adapter, chat, default_buffer_only=False) is False

        # Attach a subscriber (mimics GET /keryx/stream).
        sub = ks.hub.subscribe(platform, str(chat))
        assert ks.hub.has_subscribers(platform, str(chat)) is True

        # A live subscriber → consumer skips homeserver edits.
        assert ks.suppress_protocol_edits(adapter, chat, default_buffer_only=False) is True

        # Every stream event mirrors, in order, to the subscriber's queue.
        ks.publish_delta(adapter, chat, "hel")
        ks.publish_delta(adapter, chat, "lo")
        ks.publish_segment(adapter, chat)
        ks.publish_stop(adapter, chat)
        await asyncio.sleep(0)
        assert _drain(sub.queue) == [
            ("delta", "hel"),
            ("delta", "lo"),
            ("segment", None),
            ("stop", None),
        ]

        # Keyed isolation: a different chat_id shares no state.
        other = "!other:example.org"
        ks.publish_delta(adapter, other, "nope")
        await asyncio.sleep(0)
        assert _drain(sub.queue) == []
        assert ks.suppress_protocol_edits(adapter, other, default_buffer_only=False) is False

        # After detach, suppression relaxes again.
        ks.hub.unsubscribe(platform, str(chat), sub)
        assert ks.hub.has_subscribers(platform, str(chat)) is False
        assert ks.suppress_protocol_edits(adapter, chat, default_buffer_only=False) is False

    asyncio.run(scenario())


def test_matrix_fallback_tier_when_no_subscriber(monkeypatch):
    """Matrix + FALLBACK_EDITS + buffer-only default, no subscriber → throttled edits (False)."""
    async def scenario():
        monkeypatch.setattr(ks, "hub", ks.KeryxStreamHub())
        monkeypatch.setattr(ks, "FALLBACK_EDITS", True)
        adapter = _FakeAdapter()
        chat = "!room:example.org"

        # Buffer-only default on Matrix drops to the m.replace fallback tier.
        assert ks.suppress_protocol_edits(adapter, chat, default_buffer_only=True) is False
        # A non-buffer default is passed through unchanged.
        assert ks.suppress_protocol_edits(adapter, chat, default_buffer_only=False) is False

    asyncio.run(scenario())


def test_overflow_drops_without_blocking(monkeypatch):
    """A stalled subscriber whose queue is full drops events instead of blocking the publisher."""
    async def scenario():
        monkeypatch.setattr(ks, "hub", ks.KeryxStreamHub())
        adapter = _FakeAdapter()
        chat = "!room:example.org"
        platform = ks._platform_of(adapter)
        sub = ks.hub.subscribe(platform, str(chat))

        # Flood well past the bounded queue; publishing must never raise or block.
        for i in range(ks._QUEUE_MAX + 50):
            ks.publish_delta(adapter, chat, str(i))
        await asyncio.sleep(0)

        drained = _drain(sub.queue)
        assert len(drained) <= ks._QUEUE_MAX  # bounded — excess dropped, publisher unharmed
        ks.hub.unsubscribe(platform, str(chat), sub)

    asyncio.run(scenario())
