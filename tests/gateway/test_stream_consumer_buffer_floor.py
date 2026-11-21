"""The buffer_threshold early flush may not fire faster than once per second."""

from __future__ import annotations

from types import SimpleNamespace

import gateway.stream_consumer as sc
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _consumer(edit_interval: float, now: list[float]) -> GatewayStreamConsumer:
    consumer = GatewayStreamConsumer(
        SimpleNamespace(), "chat1",
        StreamConsumerConfig(buffer_threshold=5, edit_interval=edit_interval, cursor=""),
    )
    consumer._use_draft_streaming = False
    consumer._use_native_streaming = False
    consumer._accumulated = "A" * 200
    consumer._last_edit_time = now[0]
    return consumer


def test_full_buffer_flushes_at_one_second_not_before(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(sc.time, "monotonic", lambda: now[0])
    consumer = _consumer(10.0, now)  # the interval itself will not fire in this test
    tick = sc._Tick()

    now[0] = 100.5
    assert consumer._should_edit(tick) is False

    now[0] = 101.0
    assert consumer._should_edit(tick) is True


def test_flood_strike_still_blocks_the_early_flush(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(sc.time, "monotonic", lambda: now[0])
    consumer = _consumer(10.0, now)
    consumer._flood_strikes = 1

    now[0] = 105.0
    assert consumer._should_edit(sc._Tick()) is False
