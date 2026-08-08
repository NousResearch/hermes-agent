"""Tests for the live-glass frame poller (AVA-17)."""
from __future__ import annotations

import base64
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from plugins.observability.live_glass.frame_poller import (
    FramePoller,
    FramePollerBackend,
)


class FakeBackend(FramePollerBackend):
    """In-memory backend for tests — no real desktop access."""

    def __init__(self):
        self.captures: list[dict] = []
        self._call_count = 0

    def capture_frame(self) -> dict | None:
        self._call_count += 1
        if self.captures:
            return self.captures.pop(0)
        return None

    def is_available(self) -> bool:
        return True


def _make_frame(mode="som", width=100, height=80):
    b64 = base64.b64encode(b"\x89PNG\r\n\x1a\ntestframe").decode()
    return {
        "image_url": f"data:image/png;base64,{b64}",
        "mime_type": "image/png",
        "mode": mode,
        "width": width,
        "height": height,
        "summary": f"capture mode={mode} {width}x{height}",
        "source": "poller",
    }


class TestFramePoller:
    def test_poll_emits_frame_event(self):
        backend = FakeBackend()
        frame = _make_frame()
        backend.captures.append(frame)

        from plugins.observability.live_glass import reset_event_bus_for_tests
        reset_event_bus_for_tests()

        poller = FramePoller(backend, interval=0.1)
        poller.start()
        time.sleep(0.3)
        poller.stop()

        from plugins.observability.live_glass import get_events
        events = get_events(event_type="frame")
        assert len(events) >= 1
        assert events[0]["payload"]["source"] == "poller"

    def test_no_frame_when_backend_returns_none(self):
        backend = FakeBackend()
        # No frames in the backend

        from plugins.observability.live_glass import reset_event_bus_for_tests
        reset_event_bus_for_tests()

        poller = FramePoller(backend, interval=0.1)
        poller.start()
        time.sleep(0.3)
        poller.stop()

        from plugins.observability.live_glass import get_events
        events = get_events(event_type="frame")
        assert len(events) == 0

    def test_poller_is_idempotent_start_stop(self):
        backend = FakeBackend()
        poller = FramePoller(backend, interval=0.1)
        poller.start()
        poller.start()  # second start is no-op
        assert poller.is_running()
        poller.stop()
        poller.stop()  # second stop is no-op
        assert not poller.is_running()

    def test_poller_backend_exception_is_caught(self):
        backend = FakeBackend()
        backend.capture_frame = MagicMock(side_effect=RuntimeError("boom"))

        from plugins.observability.live_glass import reset_event_bus_for_tests
        reset_event_bus_for_tests()

        poller = FramePoller(backend, interval=0.1)
        poller.start()
        time.sleep(0.3)
        assert poller.is_running()  # Still running despite error
        poller.stop()

        # No frames emitted
        from plugins.observability.live_glass import get_events
        assert len(get_events(event_type="frame")) == 0

    def test_poller_uses_minimum_interval(self):
        backend = FakeBackend()
        poller = FramePoller(backend, interval=0.01)
        # Should clamp to minimum (0.1s)
        assert poller.interval >= 0.1

    def test_poller_stops_on_context_manager(self):
        backend = FakeBackend()
        backend.captures.append(_make_frame())

        from plugins.observability.live_glass import reset_event_bus_for_tests
        reset_event_bus_for_tests()

        poller = FramePoller(backend, interval=0.1)
        with poller:
            time.sleep(0.25)
        assert not poller.is_running()

    def test_computer_use_backend_factory(self):
        """The factory should return the active computer_use backend."""
        from plugins.observability.live_glass.frame_poller import (
            computer_use_backend_factory,
        )
        backend = computer_use_backend_factory()
        # On macOS with cua-driver installed, should be available.
        # If cua-driver is not available, returns None.
        if backend is not None:
            assert backend.is_available()
