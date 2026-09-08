"""Tests for the interim assistant message rate limit (issue #44926).

Covers: display-config resolution/normalisation of
``display.interim_assistant_min_interval_seconds`` (with per-platform overrides),
the ``_InterimRateGate`` unit behaviour, and the end-to-end gating of
``TurnRunner._setup_stream_consumer``'s ``interim_assistant_cb`` (second rapid
commentary within the window is dropped; finals are untouched by construction —
they never route through the callback).
"""

import threading
import time
from types import SimpleNamespace

import pytest

from gateway.display_config import resolve_display_setting


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

class TestMinIntervalResolution:
    """display.interim_assistant_min_interval_seconds resolves per the standard order."""

    def test_default_is_zero_unrestricted(self):
        assert resolve_display_setting({}, "telegram", "interim_assistant_min_interval_seconds") == 0

    def test_global_setting(self):
        config = {"display": {"interim_assistant_min_interval_seconds": 120}}
        assert resolve_display_setting(config, "telegram", "interim_assistant_min_interval_seconds") == 120

    def test_platform_override_wins(self):
        config = {
            "display": {
                "interim_assistant_min_interval_seconds": 120,
                "platforms": {"weixin": {"interim_assistant_min_interval_seconds": 90}},
            }
        }
        assert resolve_display_setting(config, "weixin", "interim_assistant_min_interval_seconds") == 90
        # Other platforms keep the global value.
        assert resolve_display_setting(config, "telegram", "interim_assistant_min_interval_seconds") == 120

    def test_string_value_normalised_to_int(self):
        config = {"display": {"interim_assistant_min_interval_seconds": "120"}}
        assert resolve_display_setting(config, "weixin", "interim_assistant_min_interval_seconds") == 120

    @pytest.mark.parametrize("bad", [-5, "-5", "abc", None])
    def test_invalid_values_normalise_to_zero(self, bad):
        config = {"display": {"interim_assistant_min_interval_seconds": bad}}
        assert resolve_display_setting(config, "weixin", "interim_assistant_min_interval_seconds") == 0

    def test_key_is_platform_overrideable(self):
        from gateway.display_config import OVERRIDEABLE_KEYS

        assert "interim_assistant_min_interval_seconds" in OVERRIDEABLE_KEYS


# ---------------------------------------------------------------------------
# _InterimRateGate unit behaviour
# ---------------------------------------------------------------------------

class TestInterimRateGate:
    @staticmethod
    def _gate(interval):
        from gateway.run_turn_runner import _InterimRateGate

        return _InterimRateGate(interval)

    def test_disabled_gate_always_allows(self):
        gate = self._gate(0)
        for _ in range(3):
            assert gate.allow("msg") is True
        assert gate.enabled is False

    def test_first_message_always_passes(self):
        gate = self._gate(120)
        assert gate.allow("first") is True

    def test_second_message_within_window_suppressed(self):
        gate = self._gate(120)
        assert gate.allow("first") is True
        assert gate.allow("second") is False

    def test_message_after_window_passes(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
        gate = self._gate(120)
        assert gate.allow("first") is True
        clock["t"] += 30
        assert gate.allow("second") is False
        clock["t"] += 91  # 121s after the first send
        assert gate.allow("third") is True

    def test_suppressed_message_does_not_consume_window(self, monkeypatch):
        """A dropped message must not extend the cooldown (no starvation under spam)."""
        clock = {"t": 1000.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
        gate = self._gate(60)
        assert gate.allow("first") is True
        for _ in range(50):
            clock["t"] += 1
            assert gate.allow("spam") is False
        clock["t"] += 10  # 61s after first (loop only advanced 50s)
        assert gate.allow("after-window") is True

    def test_concurrent_calls_admit_at_most_one(self):
        gate = self._gate(300)
        admitted = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def contender():
            barrier.wait()
            if gate.allow("racer"):
                with lock:
                    admitted.append(True)

        threads = [threading.Thread(target=contender) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(admitted) == 1

    def test_non_positive_intervals_disable(self):
        assert self._gate(-10).enabled is False
        assert self._gate(None).enabled is False

    def test_admission_at_monotonic_zero_still_gates(self, monkeypatch):
        """A send at monotonic t=0.0 must not leave the gate unset (0.0 is falsy)."""
        clock = {"t": 0.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
        gate = self._gate(120)
        assert gate.allow("at-zero") is True
        assert gate.allow("within-window") is False
        clock["t"] += 121
        assert gate.allow("after-window") is True

    @pytest.mark.parametrize("junk", [".inf", "-.inf", True, False])
    def test_yaml_edge_values_normalise_to_zero(self, junk):
        config = {"display": {"interim_assistant_min_interval_seconds": junk}}
        assert resolve_display_setting(config, "weixin", "interim_assistant_min_interval_seconds") == 0


# ---------------------------------------------------------------------------
# End-to-end: interim_assistant_cb honours the gate
# ---------------------------------------------------------------------------

class _RecordingConsumer:
    def __init__(self):
        self.commentary = []
        self.segment_breaks = 0

    def on_commentary(self, text):
        self.commentary.append(text)

    def on_segment_break(self):
        self.segment_breaks += 1


def _build_interim_cb(monkeypatch, min_interval):
    """Build the real interim_assistant_cb via TurnRunner._setup_stream_consumer.

    Stub TurnContext + runner carry only the attributes the wiring reads; the
    GatewayStreamConsumer symbol is swapped for a recording sink so no real
    transports start. Returns (callback, recording_consumer).
    """
    import gateway.stream_consumer as sc_mod
    from gateway.run_turn_runner import TurnRunner

    consumer = _RecordingConsumer()
    ctx = SimpleNamespace(
        streaming_tts_consumer_holder=[None],
        resolve_display_setting=lambda *a, **k: None,
        user_config={},
        interim_assistant_messages_enabled=True,
        interim_assistant_min_interval_seconds=min_interval,
        source=SimpleNamespace(chat_id="chat-1"),
        _status_thread_metadata=None,
        event_message_id=None,
        progress_queue=None,
        stream_consumer_holder=[None],
        _run_still_current=lambda: True,
        _status_adapter=None,
    )
    runner_stub = SimpleNamespace(
        config=SimpleNamespace(streaming=None),
        _adapter_for_source=lambda source: object(),
        _build_stream_consumer_config=lambda *a, **k: (object(), False),
    )
    turn_runner = TurnRunner(runner_stub, ctx)
    monkeypatch.setattr(sc_mod, "GatewayStreamConsumer", lambda *a, **k: consumer)

    _, _, interim_cb, want_interim = turn_runner._setup_stream_consumer("weixin")
    assert want_interim is True
    return interim_cb, consumer


class TestInterimCallbackGating:
    def test_rapid_second_message_suppressed(self, monkeypatch):
        cb, consumer = _build_interim_cb(monkeypatch, 120)
        cb("progress one")
        cb("progress two")
        assert consumer.commentary == ["progress one"]

    def test_interval_zero_delivers_both(self, monkeypatch):
        cb, consumer = _build_interim_cb(monkeypatch, 0)
        cb("one")
        cb("two")
        assert consumer.commentary == ["one", "two"]

    def test_already_streamed_commentary_also_gated(self, monkeypatch):
        cb, consumer = _build_interim_cb(monkeypatch, 120)
        cb("streamed one", already_streamed=True)
        cb("streamed two", already_streamed=True)
        assert consumer.commentary == []
        assert consumer.segment_breaks == 1

    def test_fresh_gate_per_turn_first_message_free(self, monkeypatch):
        """Each turn builds its own gate: turn 2's first commentary passes immediately."""
        cb1, consumer1 = _build_interim_cb(monkeypatch, 120)
        cb1("turn one")
        cb1("suppressed")
        cb2, consumer2 = _build_interim_cb(monkeypatch, 120)
        cb2("turn two first")
        assert consumer1.commentary == ["turn one"]
        assert consumer2.commentary == ["turn two first"]
