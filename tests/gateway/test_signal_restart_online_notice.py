"""Systemd restart must leave the planned-restart marker via the REAL unit path (#121937).

The generated systemd unit marks via ``ExecStop=gateway.systemd_stop_mark`` BEFORE
delivering SIGTERM, so a restart consumes the planned-stop marker in the registered
shutdown handler and leaves ``_signal_initiated_shutdown`` False. The persist phase
must still write ``.restart_pending.json`` for that signal-driven shutdown (a
terminal stop that is NOT promptly revived stays silent via the replay staleness
guard instead).

Drives the real writer (``write_planned_stop_marker``, the same call ExecStop
makes) into the real registered handler (``_start_gateway_make_shutdown_signal_handler``)
and then the real persist phase. No direct flag shortcuts.
"""

import asyncio
import json
import os
import signal as signal_mod
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

import gateway.run as gateway_run
import gateway.shutdown_forensics as forensics
from gateway import status as status_mod
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.run_shutdown import GatewayShutdownMixin
from tests.gateway.restart_test_helpers import make_restart_runner

ONLINE_NOTICE = "Gateway online"


def _finished_ctx():
    ctx = GatewayShutdownMixin._StopContext(deferred_count=lambda: 0)
    import time as _time

    ctx.started_at = _time.monotonic()
    return ctx


def _drive_execstop_sigterm(runner, marker):
    """Write a self-targeting planned-stop marker (what ExecStop does) and deliver
    SIGTERM through the real registered shutdown handler. Returns the signal flag."""
    assert status_mod.write_planned_stop_marker(os.getpid()) is True
    assert marker.exists()
    signal_flag = [False]
    handler = gateway_run._start_gateway_make_shutdown_signal_handler(runner, signal_flag)
    handler(signal_mod.SIGTERM)
    return signal_flag


@pytest.mark.asyncio
async def test_systemd_restart_marker_path_leaves_planned_restart_marker(tmp_path, monkeypatch):
    """ExecStop marker + SIGTERM (the real generated-unit restart) owes the next boot hello."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    marker = tmp_path / ".gateway-planned-stop.json"
    monkeypatch.setattr(status_mod, "_get_planned_stop_marker_path", lambda: marker)
    monkeypatch.setattr(status_mod, "consume_takeover_marker_for_self", lambda: False)
    monkeypatch.setattr(forensics, "snapshot_shutdown_context", lambda *a, **k: None)
    monkeypatch.setattr(gateway_run.asyncio, "create_task", lambda coro: coro.close())

    runner, _adapter = make_restart_runner()
    runner.stop = lambda **kwargs: asyncio.sleep(0)
    runner._stop_persist_exit_state = GatewayRunner._stop_persist_exit_state.__get__(
        runner, GatewayRunner
    )

    signal_flag = _drive_execstop_sigterm(runner, marker)

    # The review mechanism: the planned branch leaves the signal flag False ...
    assert signal_flag[0] is False
    assert not marker.exists(), "handler must consume the planned-stop marker"
    # ... but the shutdown is still signal-driven, which is what owes the hello.
    assert runner._stop_requested_by_signal is True

    await runner._stop_persist_exit_state(_finished_ctx())

    pending = tmp_path / ".restart_pending.json"
    assert pending.exists(), "signal-driven shutdown must leave .restart_pending.json"
    assert isinstance(json.loads(pending.read_text(encoding="utf-8-sig"))["requested_at"], float)


@pytest.mark.asyncio
async def test_plain_stop_leaves_no_restart_marker(tmp_path, monkeypatch):
    """A programmatic stop (no signal, no restart) stays silent on next boot."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, _adapter = make_restart_runner()
    runner._signal_initiated_shutdown = False
    runner._stop_requested_by_signal = False
    runner._restart_requested = False
    runner._restart_command_source = None
    runner._stop_persist_exit_state = GatewayRunner._stop_persist_exit_state.__get__(
        runner, GatewayRunner
    )
    await runner._stop_persist_exit_state(_finished_ctx())

    assert not (tmp_path / ".restart_pending.json").exists()


def _replay_runner(tmp_path, monkeypatch):
    """Minimal runner that can run the real planned-restart replay to a live home channel."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                gateway_restart_notification=True,
                home_channel=HomeChannel(
                    platform=Platform.DISCORD, chat_id="unit-test-home", name="Test home"
                ),
            ),
        }
    )
    adapter: Any = SimpleNamespace(
        send_path_degraded=False,
        send=AsyncMock(return_value=SendResult(success=True, message_id="unit-test-notice")),
    )
    runner.adapters = {}
    runner.adapters[Platform.DISCORD] = adapter
    runner._planned_restart_notice_lock = None
    runner._free_tier_startup_line = Mock(return_value=None)
    return runner, adapter


def _write_pending(tmp_path, *, age_s):
    pending = tmp_path / ".restart_pending.json"
    pending.write_text(
        json.dumps(
            {"requested_at": time.time() - age_s, "via_service": False, "detached": False}
        ),
        encoding="utf-8",
    )
    return pending


@pytest.mark.asyncio
async def test_stale_restart_marker_stays_silent(tmp_path, monkeypatch):
    """A terminal stop's marker found by a MUCH later start sends no online notice."""
    runner, adapter = _replay_runner(tmp_path, monkeypatch)
    pending = _write_pending(tmp_path, age_s=7200)

    await runner._replay_pending_planned_restart_notification()

    adapter.send.assert_not_called()
    assert not pending.exists(), "stale marker must be reaped, not replayed forever"


@pytest.mark.asyncio
async def test_fresh_restart_marker_sends_online_notice(tmp_path, monkeypatch):
    """Control: a promptly-revived restart (fresh marker) still says hello."""
    runner, adapter = _replay_runner(tmp_path, monkeypatch)
    pending = _write_pending(tmp_path, age_s=5)

    await runner._replay_pending_planned_restart_notification()

    adapter.send.assert_awaited_once()
    assert ONLINE_NOTICE in adapter.send.await_args.args[1]
    assert not pending.exists()
