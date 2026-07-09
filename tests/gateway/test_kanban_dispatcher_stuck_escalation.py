"""Tests for BUILD-263: dispatcher-stuck Telegram escalation.

Two things are covered:

1. ``DispatcherStuckEscalationState`` — a pure state machine (no asyncio, no
   network) deciding WHEN to fire/re-fire/clear the escalation. Driven with a
   fake clock so timing assertions are exact and instant.
2. ``GatewayKanbanWatchersMixin._kanban_dispatcher_stuck_alert`` — the async
   send path, exercised against a minimal fake runner (fake ``config`` +
   fake ``adapters``) rather than a full ``GatewayRunner``, mirroring the
   existing "test the extracted pure helper directly" pattern used for
   ``_resolve_auto_decompose_settings`` in this same module.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.config import HomeChannel, Platform
from gateway.kanban_watchers import (
    DispatcherStuckEscalationState,
    GatewayKanbanWatchersMixin,
)


# ---------------------------------------------------------------------------
# DispatcherStuckEscalationState — pure state machine
# ---------------------------------------------------------------------------


def test_no_alert_below_threshold():
    state = DispatcherStuckEscalationState(escalate_after_ticks=12)
    assert state.should_alert(11, now=1000.0) is False


def test_alert_fires_at_threshold():
    state = DispatcherStuckEscalationState(escalate_after_ticks=12)
    assert state.should_alert(12, now=1000.0) is True


def test_alert_fires_above_threshold_when_not_yet_alerted():
    state = DispatcherStuckEscalationState(escalate_after_ticks=12)
    assert state.should_alert(50, now=1000.0) is True


def test_no_realert_before_interval_elapses():
    state = DispatcherStuckEscalationState(
        escalate_after_ticks=12, realert_seconds=3600,
    )
    assert state.should_alert(12, now=1000.0) is True
    state.mark_alerted(1000.0)
    # Still stuck, only 10 minutes later — must not re-alert yet.
    assert state.should_alert(20, now=1000.0 + 600) is False


def test_realert_after_interval_elapses():
    state = DispatcherStuckEscalationState(
        escalate_after_ticks=12, realert_seconds=3600,
    )
    state.mark_alerted(1000.0)
    # Exactly at the boundary and past it — both should re-alert.
    assert state.should_alert(50, now=1000.0 + 3600) is True
    assert state.should_alert(50, now=1000.0 + 4000) is True
    # Just under the boundary — must not re-alert yet.
    assert state.should_alert(50, now=1000.0 + 3599) is False


def test_recovery_resets_state_for_immediate_next_alert():
    """mark_recovered() must clear BOTH the alerted flag and the re-alert
    timer — otherwise a brand new stuck streak inherits the previous
    streak's cooldown and stays silent past its own threshold."""
    state = DispatcherStuckEscalationState(
        escalate_after_ticks=12, realert_seconds=3600,
    )
    state.mark_alerted(1000.0)
    assert state.should_alert(20, now=1010.0) is False  # still cooling down

    state.mark_recovered()

    # A fresh streak crossing the threshold shortly after recovery must
    # alert immediately — not wait out the old streak's hourly cadence.
    assert state.should_alert(12, now=1020.0) is True


def test_escalate_after_ticks_and_realert_seconds_are_clamped_to_at_least_one():
    state = DispatcherStuckEscalationState(escalate_after_ticks=0, realert_seconds=-5)
    assert state.escalate_after_ticks == 1
    assert state.realert_seconds == 1


# ---------------------------------------------------------------------------
# _kanban_dispatcher_stuck_alert — async send path
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, home=None):
        self._home = home

    def get_home_channel(self, platform):
        return self._home if platform == Platform.TELEGRAM else None


class _RecordingAdapter:
    def __init__(self, *, raises=None, send_result=None):
        self.sent = []
        self._raises = raises
        self._send_result = send_result

    async def send(self, chat_id, text, metadata=None):
        if self._raises is not None:
            raise self._raises
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
        return self._send_result


class _FakeRunner(GatewayKanbanWatchersMixin):
    """Minimal stand-in exposing only what `_kanban_dispatcher_stuck_alert`
    touches (`self.config`, `self.adapters`) — avoids constructing a full
    GatewayRunner just to exercise this one method."""

    def __init__(self, *, config, adapters):
        self.config = config
        self.adapters = adapters


def _run(coro):
    return asyncio.run(coro)


def test_alert_returns_false_when_no_home_channel_configured():
    runner = _FakeRunner(config=_FakeConfig(home=None), adapters={})
    sent = _run(runner._kanban_dispatcher_stuck_alert("stuck!"))
    assert sent is False


def test_alert_returns_false_when_telegram_adapter_not_connected():
    home = HomeChannel(platform=Platform.TELEGRAM, chat_id="123", name="Home")
    runner = _FakeRunner(config=_FakeConfig(home=home), adapters={})
    sent = _run(runner._kanban_dispatcher_stuck_alert("stuck!"))
    assert sent is False


def test_alert_sends_via_home_channel_chat_and_thread():
    home = HomeChannel(
        platform=Platform.TELEGRAM, chat_id="123", name="Home", thread_id="7",
    )
    adapter = _RecordingAdapter()
    runner = _FakeRunner(
        config=_FakeConfig(home=home), adapters={Platform.TELEGRAM: adapter},
    )
    sent = _run(runner._kanban_dispatcher_stuck_alert("stuck message"))
    assert sent is True
    assert len(adapter.sent) == 1
    assert adapter.sent[0]["chat_id"] == "123"
    assert adapter.sent[0]["text"] == "stuck message"
    assert adapter.sent[0]["metadata"]["thread_id"] == "7"


def test_alert_returns_false_when_send_result_reports_failure():
    from gateway.platforms.base import SendResult

    home = HomeChannel(platform=Platform.TELEGRAM, chat_id="123", name="Home")
    adapter = _RecordingAdapter(send_result=SendResult(success=False, error="chat not found"))
    runner = _FakeRunner(
        config=_FakeConfig(home=home), adapters={Platform.TELEGRAM: adapter},
    )
    sent = _run(runner._kanban_dispatcher_stuck_alert("stuck message"))
    assert sent is False


def test_alert_returns_false_when_send_raises():
    home = HomeChannel(platform=Platform.TELEGRAM, chat_id="123", name="Home")
    adapter = _RecordingAdapter(raises=RuntimeError("network down"))
    runner = _FakeRunner(
        config=_FakeConfig(home=home), adapters={Platform.TELEGRAM: adapter},
    )
    # Must not raise — the caller treats a failed send as "not delivered",
    # never as an unhandled exception that could crash the dispatcher tick.
    sent = _run(runner._kanban_dispatcher_stuck_alert("stuck message"))
    assert sent is False
