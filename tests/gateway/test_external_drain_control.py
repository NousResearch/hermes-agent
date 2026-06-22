"""Tests for the external drain-control marker contract + gateway state machine.

Task 2.2/2.3. Two layers:
  * drain_control.py — the presence-based marker contract (write/clear/read,
    HERMES_HOME-scoped, never-raises).
  * GatewayRunner enter/exit/watcher + the new-turn accept gate — the
    reversible state machine driven by the marker.

Mocked tests are necessary-not-sufficient here (the HARD live-validation gate,
Q-B, exercises a real `hermes gateway run`); these lock the unit contract.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gateway.drain_control as dc
from gateway.run import GatewayRunner
from gateway.platforms.base import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


# ---------------------------------------------------------------------------
# Marker contract (drain_control.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


class TestMarkerContract:
    def test_absent_by_default(self, home):
        assert dc.drain_requested() is False
        assert dc.read_drain_request() is None

    def test_write_then_present(self, home):
        payload = dc.write_drain_request(principal="nas")
        assert dc.drain_requested() is True
        assert payload["action"] == "drain"
        assert payload["principal"] == "nas"
        body = dc.read_drain_request()
        assert body is not None and body["principal"] == "nas"

    def test_clear_removes(self, home):
        dc.write_drain_request()
        assert dc.clear_drain_request() is True
        assert dc.drain_requested() is False
        # idempotent: clearing again is a no-op, returns False
        assert dc.clear_drain_request() is False

    def test_path_respects_hermes_home(self, home):
        assert dc.drain_request_path() == home / ".drain_request.json"

    def test_corrupt_marker_reads_as_present_contentless(self, home):
        # A half-written / malformed marker must still count as "drain active"
        # (fail-safe toward quiescing).
        dc.drain_request_path().write_text("{not valid json", encoding="utf-8")
        assert dc.drain_requested() is True
        assert dc.read_drain_request() == {}

    def test_write_is_atomic_json(self, home):
        dc.write_drain_request(principal="x")
        import json

        data = json.loads(dc.drain_request_path().read_text())
        assert data["action"] == "drain"


# ---------------------------------------------------------------------------
# Gateway state machine (enter / exit / idempotency)
# ---------------------------------------------------------------------------


def _drain_runner():
    runner, adapter = make_restart_runner()
    runner._external_drain_active = False
    # Bind the real methods under test.
    runner._enter_external_drain = GatewayRunner._enter_external_drain.__get__(
        runner, GatewayRunner
    )
    runner._exit_external_drain = GatewayRunner._exit_external_drain.__get__(
        runner, GatewayRunner
    )
    return runner, adapter


class TestDrainStateMachine:
    def test_enter_sets_flag_and_flips_state(self):
        runner, _ = _drain_runner()
        runner._enter_external_drain()
        assert runner._external_drain_active is True
        runner._update_runtime_status.assert_called_with("draining")

    def test_enter_idempotent(self):
        runner, _ = _drain_runner()
        runner._enter_external_drain()
        runner._update_runtime_status.reset_mock()
        runner._enter_external_drain()  # second call — no-op
        runner._update_runtime_status.assert_not_called()

    def test_exit_reverts_to_running(self):
        runner, _ = _drain_runner()
        runner._enter_external_drain()
        runner._update_runtime_status.reset_mock()
        runner._exit_external_drain()
        assert runner._external_drain_active is False
        runner._update_runtime_status.assert_called_with("running")

    def test_exit_idempotent_when_not_draining(self):
        runner, _ = _drain_runner()
        runner._exit_external_drain()  # never entered — no-op
        runner._update_runtime_status.assert_not_called()

    def test_exit_during_shutdown_does_not_revert_to_running(self):
        runner, _ = _drain_runner()
        runner._enter_external_drain()
        runner._update_runtime_status.reset_mock()
        # A shutdown drain is now in progress — exit must NOT resurrect running.
        runner._draining = True
        runner._exit_external_drain()
        assert runner._external_drain_active is False
        runner._update_runtime_status.assert_not_called()

    def test_exit_when_loop_stopped_does_not_revert(self):
        runner, _ = _drain_runner()
        runner._enter_external_drain()
        runner._update_runtime_status.reset_mock()
        runner._running = False
        runner._exit_external_drain()
        runner._update_runtime_status.assert_not_called()


# ---------------------------------------------------------------------------
# Watcher reconciliation
# ---------------------------------------------------------------------------


class TestDrainWatcher:
    @pytest.mark.asyncio
    async def test_watcher_enters_then_exits_with_marker(self, home):
        runner, _ = _drain_runner()
        runner._drain_control_watcher = GatewayRunner._drain_control_watcher.__get__(
            runner, GatewayRunner
        )
        # Drive a few ticks manually rather than spinning the loop.
        dc.write_drain_request()
        task = asyncio.create_task(runner._drain_control_watcher(interval=0.02))
        await asyncio.sleep(0.06)
        assert runner._external_drain_active is True
        dc.clear_drain_request()
        await asyncio.sleep(0.06)
        assert runner._external_drain_active is False
        runner._running = False
        await asyncio.sleep(0.04)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# New-turn accept gate
# ---------------------------------------------------------------------------


class TestNewTurnGate:
    @pytest.mark.asyncio
    async def test_new_turn_refused_during_external_drain(self):
        runner, _ = _drain_runner()
        runner._external_drain_active = True
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=make_restart_source(),
            message_id="m1",
        )
        result = await runner._handle_message(event)
        assert result is not None
        assert "draining" in result.lower()

    @pytest.mark.asyncio
    async def test_in_flight_turn_not_interrupted_by_drain(self):
        # Entering drain must NOT touch the running-agents set.
        runner, _ = _drain_runner()
        sentinel = MagicMock()
        runner._running_agents["k"] = sentinel
        runner._enter_external_drain()
        assert runner._running_agents.get("k") is sentinel
        sentinel.interrupt.assert_not_called()
