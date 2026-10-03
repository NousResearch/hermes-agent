"""Tests for EventEmitter — validated event dispatch with allowlist checking."""

import pytest

from agent.modules.event_emitter import EventEmitter


class TestEventEmitter:
    """EventEmitter test suite."""

    def test_valid_event_passes(self):
        """Valid event names pass through without raising."""
        events_captured = []

        def capture_sink(payload: dict) -> None:
            events_captured.append(payload)

        emitter = EventEmitter(sinks=[capture_sink])

        # Emit a valid event
        emitter.emit("hermes.identity.bootstrap", {"user_id": "test_user"})

        assert len(events_captured) == 1
        assert events_captured[0]["event"] == "hermes.identity.bootstrap"
        assert events_captured[0]["payload"]["user_id"] == "test_user"
        assert "timestamp" in events_captured[0]

    def test_unknown_event_rejected(self):
        """Unknown event names raise ValueError."""
        emitter = EventEmitter(sinks=[])

        with pytest.raises(ValueError) as exc_info:
            emitter.emit("unknown.event.name", {})

        assert "Unknown event" in str(exc_info.value)
        assert "unknown.event.name" in str(exc_info.value)

    def test_multi_sink_fanout(self):
        """Events are dispatched to all configured sinks."""
        sink1_events = []
        sink2_events = []

        def sink1(payload: dict) -> None:
            sink1_events.append(payload)

        def sink2(payload: dict) -> None:
            sink2_events.append(payload)

        emitter = EventEmitter(sinks=[sink1, sink2])

        emitter.emit("mission.submitted", {"mission_id": "m123"})

        assert len(sink1_events) == 1
        assert len(sink2_events) == 1
        assert sink1_events[0]["event"] == "mission.submitted"
        assert sink2_events[0]["event"] == "mission.submitted"
