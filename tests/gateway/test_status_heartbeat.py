"""Tests for gateway.status_heartbeat."""

from __future__ import annotations

import logging
import time

import pytest

from gateway import status_heartbeat as sh


@pytest.fixture(autouse=True)
def _ensure_heartbeat_stopped():
    sh.stop_status_heartbeat(timeout=1.0)
    yield
    sh.stop_status_heartbeat(timeout=1.0)


def test_heartbeat_once_calls_write_runtime_status(monkeypatch):
    calls = []
    monkeypatch.setattr(sh, "write_runtime_status", lambda: calls.append("tick"))

    sh.heartbeat_once()

    assert calls == ["tick"]


def test_start_emits_immediate_tick_and_returns_true(monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(sh, "write_runtime_status", lambda: calls.append("tick"))
    caplog.set_level(logging.INFO, logger="gateway.status_heartbeat")

    started = sh.start_status_heartbeat(interval_seconds=3600.0)

    assert started is True
    assert sh.is_running() is True
    assert calls == ["tick"]
    assert any("Periodic runtime heartbeat started" in r.getMessage() for r in caplog.records)


def test_double_start_is_noop(monkeypatch):
    monkeypatch.setattr(sh, "write_runtime_status", lambda: None)

    assert sh.start_status_heartbeat(interval_seconds=3600.0) is True
    assert sh.start_status_heartbeat(interval_seconds=3600.0) is False
    assert sh.is_running() is True


def test_stop_logs_shutdown(monkeypatch, caplog):
    monkeypatch.setattr(sh, "write_runtime_status", lambda: None)
    sh.start_status_heartbeat(interval_seconds=3600.0)
    caplog.clear()
    caplog.set_level(logging.INFO, logger="gateway.status_heartbeat")

    sh.stop_status_heartbeat(timeout=1.0)

    assert sh.is_running() is False
    assert any("Periodic runtime heartbeat stopped" in r.getMessage() for r in caplog.records)


def test_stop_without_start_is_noop():
    sh.stop_status_heartbeat(timeout=0.5)
    assert sh.is_running() is False


def test_periodic_timer_fires(monkeypatch):
    calls = []
    monkeypatch.setattr(sh, "write_runtime_status", lambda: calls.append(time.monotonic()))

    sh.start_status_heartbeat(interval_seconds=0.1)
    time.sleep(0.45)
    sh.stop_status_heartbeat(timeout=1.0)

    # Immediate baseline tick + at least 2 periodic ticks.
    assert len(calls) >= 3, calls


def test_thread_is_daemon(monkeypatch):
    monkeypatch.setattr(sh, "write_runtime_status", lambda: None)

    sh.start_status_heartbeat(interval_seconds=3600.0)

    assert sh._heartbeat_thread is not None
    assert sh._heartbeat_thread.daemon is True


def test_tick_failure_is_logged_at_debug(monkeypatch, caplog):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(sh, "write_runtime_status", _boom)
    caplog.set_level(logging.DEBUG, logger="gateway.status_heartbeat")

    sh.start_status_heartbeat(interval_seconds=0.1)
    time.sleep(0.15)
    sh.stop_status_heartbeat(timeout=1.0)

    assert any("Status heartbeat tick failed" in r.getMessage() for r in caplog.records)
