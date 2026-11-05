"""A missed watchdog probe is observable before it becomes a process restart."""
import asyncio
import logging
import threading
from unittest.mock import MagicMock


def test_watchdog_reports_a_miss_and_recovery_without_exiting(monkeypatch, caplog):
    from gateway import loop_health, shutdown_watchdog

    monkeypatch.setattr(loop_health, '_latest', None)
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    started = threading.Event()
    observations = []
    holder = {}
    original = loop_health.record_probe

    def record(*, misses, response_seconds):
        original(misses=misses, response_seconds=response_seconds)
        observations.append(loop_health.probe_snapshot())
        if misses == 0:
            assert started.wait(5)
            holder['handle'].stop()

    def schedule(callback):
        if observations:
            callback()

    monkeypatch.setattr(loop_health, 'record_probe', record)
    hard_exit = MagicMock()
    monkeypatch.setattr(shutdown_watchdog, '_hard_exit', hard_exit)
    loop.call_soon_threadsafe.side_effect = schedule
    with caplog.at_level(logging.INFO, logger='gateway.loop_health'):
        handle = shutdown_watchdog.start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=3)
        assert handle is not None
        holder['handle'] = handle
        started.set()
        try:
            handle.join(5)
        finally:
            handle.stop()
            handle.join(5)
    assert not handle.is_alive()
    assert [sample.misses for sample in observations] == [1, 0]
    assert all(sample.response_seconds >= 0 for sample in observations)
    assert any(record.levelno == logging.WARNING and 'timed out' in record.message for record in caplog.records)
    assert any(record.levelno == logging.INFO and 'recovered' in record.message for record in caplog.records)
    hard_exit.assert_not_called()


def test_probe_diagnostic_has_a_specific_content_free_error_class():
    from agent.monitoring.gateway_health import classify_gateway_error

    assert classify_gateway_error('Gateway event loop liveness probe timed out') == 'loop_unresponsive'


def test_probe_observation_failure_does_not_disarm_the_watchdog(monkeypatch):
    from gateway import loop_health, shutdown_watchdog

    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    ready, exited = threading.Event(), threading.Event()
    holder = {}

    def fail_observation(**kwargs):
        raise RuntimeError('observation unavailable')

    def exit_gateway(code):
        assert ready.wait(5)
        holder['handle'].stop()
        exited.set()

    monkeypatch.setattr(loop_health, 'record_probe', fail_observation)
    monkeypatch.setattr(shutdown_watchdog, '_write_watchdog_dump', MagicMock())
    monkeypatch.setattr(shutdown_watchdog, '_mark_exited_quietly', MagicMock())
    monkeypatch.setattr(shutdown_watchdog, '_hard_exit', exit_gateway)
    handle = shutdown_watchdog.start_loop_liveness_watchdog(
        loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1)
    assert handle is not None
    holder['handle'] = handle
    ready.set()
    try:
        assert exited.wait(5)
    finally:
        handle.stop()
        handle.join(5)
    assert not handle.is_alive()
