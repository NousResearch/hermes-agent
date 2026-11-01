from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import HomeChannel, Platform
from gateway.run import GatewayRunner
from gateway.scheduled_restart import (
    MAX_SCHEDULED_LEAD_SECONDS,
    MIN_SCHEDULED_LEAD_SECONDS,
    format_countdown,
    parse_wallclock_string,
)
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _bind_scheduled_methods(runner):
    for name in (
        "schedule_restart", "cancel_scheduled_restart", "_cancel_pending_scheduled_restart",
        "_spawn_planned_announce", "_run_scheduled_restart_task",
        "_wait_until_monotonic", "_announce_planned_restart",
    ):
        setattr(runner, name, getattr(GatewayRunner, name).__get__(runner, GatewayRunner))
    runner._scheduled_restart = None
    return runner


def test_countdown_formatting():
    assert format_countdown(1) == "1 second"
    assert format_countdown(30) == "30 seconds"
    assert format_countdown(60) == "1 minute"
    assert format_countdown(125) == "2 minutes"
    assert format_countdown(7200) == "2 hours"


def test_wallclock_parser_accepts_local_hhmm_and_iso():
    assert parse_wallclock_string("04:15") is not None
    assert parse_wallclock_string("2026-10-25T04:15:00") is not None
    assert parse_wallclock_string("not-a-time") is None


@pytest.mark.asyncio
async def test_schedule_announces_then_triggers_restart_path(monkeypatch):
    runner, _adapter = make_restart_runner()
    _bind_scheduled_methods(runner)
    runner._announce_planned_restart = AsyncMock()
    runner.request_restart = MagicMock(return_value=True)

    monkeypatch.setattr("gateway.run_shutdown.MIN_SCHEDULED_LEAD_SECONDS", 0.01)
    monkeypatch.setattr("gateway.run_shutdown.REBROADCAST_LEAD_SECONDS", ())
    monkeypatch.setattr("gateway.restart.is_gateway_supervisor_process", lambda: False)
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)
    result = runner.schedule_restart(time.time() + 0.05, "test update")

    assert result["accepted"] is True
    await asyncio.sleep(0.09)
    runner.request_restart.assert_called_once_with(detached=True, via_service=False)
    assert runner._scheduled_restart is None
    assert runner._announce_planned_restart.await_count == 1
    notice = runner._announce_planned_restart.await_args.args[0]
    assert "test update" in notice
    assert "Downtime is expected to be brief" in notice


@pytest.mark.asyncio
async def test_rebroadcasts_then_fires_with_standard_relaunch_semantics(monkeypatch):
    runner, _adapter = make_restart_runner()
    _bind_scheduled_methods(runner)
    runner._announce_planned_restart = AsyncMock()
    runner.request_restart = MagicMock(return_value=True)

    monkeypatch.setattr("gateway.run_shutdown.MIN_SCHEDULED_LEAD_SECONDS", 0.01)
    monkeypatch.setattr("gateway.run_shutdown.REBROADCAST_LEAD_SECONDS", (2.0, 1.0))
    monkeypatch.setattr("gateway.restart.is_gateway_supervisor_process", lambda: False)
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)
    result = runner.schedule_restart(time.time() + 2.5, "test update")

    assert result["accepted"] is True
    await asyncio.sleep(3.0)
    runner.request_restart.assert_called_once_with(detached=True, via_service=False)
    messages = [call.args[0] for call in runner._announce_planned_restart.await_args_list]
    assert len(messages) == 3  # schedule-time + 2 rebroadcasts
    assert any("2 seconds" in message for message in messages)
    assert any("1 second" in message for message in messages)
    assert any("Reminder" in message for message in messages[1:])


@pytest.mark.asyncio
async def test_fire_uses_service_restart_under_supervisor(monkeypatch):
    runner, _adapter = make_restart_runner()
    _bind_scheduled_methods(runner)
    runner._announce_planned_restart = AsyncMock()
    runner.request_restart = MagicMock(return_value=True)

    monkeypatch.setattr("gateway.run_shutdown.MIN_SCHEDULED_LEAD_SECONDS", 0.01)
    monkeypatch.setattr("gateway.run_shutdown.REBROADCAST_LEAD_SECONDS", ())
    monkeypatch.setattr("gateway.restart.is_gateway_supervisor_process", lambda: True)
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)
    runner.schedule_restart(time.time() + 0.05, "test update")

    await asyncio.sleep(0.09)
    runner.request_restart.assert_called_once_with(detached=False, via_service=True)


@pytest.mark.asyncio
async def test_schedule_rejects_too_short_lead():
    runner, _adapter = make_restart_runner()
    _bind_scheduled_methods(runner)
    runner._announce_planned_restart = AsyncMock()

    result = runner.schedule_restart(time.time() + MIN_SCHEDULED_LEAD_SECONDS - 1)

    assert result["accepted"] is False
    assert result["error"] == "lead_too_short"
    runner._announce_planned_restart.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_rejects_too_long_lead():
    runner, _adapter = make_restart_runner()
    _bind_scheduled_methods(runner)
    runner._announce_planned_restart = AsyncMock()

    result = runner.schedule_restart(time.time() + MAX_SCHEDULED_LEAD_SECONDS + 60)

    assert result["accepted"] is False
    assert result["error"] == "lead_too_long"
    runner._announce_planned_restart.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_refuses_while_stopping():
    runner, _adapter = make_restart_runner()
    _bind_scheduled_methods(runner)
    runner._draining = True

    result = runner.schedule_restart(time.time() + 60)

    assert result["accepted"] is False
    assert result["error"] == "already_stopping"


@pytest.mark.asyncio
async def test_reschedule_cancels_old_timer_and_cancel_announces(monkeypatch):
    runner, _adapter = make_restart_runner()
    _bind_scheduled_methods(runner)
    runner._announce_planned_restart = AsyncMock()
    monkeypatch.setattr("gateway.run_shutdown.MIN_SCHEDULED_LEAD_SECONDS", 0.01)

    first = runner.schedule_restart(time.time() + 30, "first")
    first_task = runner._scheduled_restart.task
    second = runner.schedule_restart(time.time() + 40, "second")
    assert first["accepted"] and second["accepted"]
    assert first_task.cancelled() or first_task.cancelling()

    cancelled = runner.cancel_scheduled_restart()
    assert cancelled == {"cancelled": True, "had_scheduled": True}
    await asyncio.sleep(0)
    assert runner._scheduled_restart is None
    messages = [call.args[0] for call in runner._announce_planned_restart.await_args_list]
    assert any("cancelled" in message for message in messages)


@pytest.mark.asyncio
async def test_external_stop_cancels_scheduled_restart_without_notice(monkeypatch):
    runner, _adapter = make_restart_runner()
    _bind_scheduled_methods(runner)
    runner._announce_planned_restart = AsyncMock()
    monkeypatch.setattr("gateway.run_shutdown.MIN_SCHEDULED_LEAD_SECONDS", 0.01)

    runner.schedule_restart(time.time() + 30)
    assert runner._scheduled_restart is not None
    initial_announces = runner._announce_planned_restart.await_count

    runner._cancel_pending_scheduled_restart()

    assert runner._scheduled_restart is None
    # The schedule-time announcement happened; cancellation adds none.
    assert runner._announce_planned_restart.await_count == initial_announces


@pytest.mark.asyncio
async def test_announce_delivers_to_active_session_and_home_channel():
    from gateway.platforms.base import SendResult

    runner, adapter = make_restart_runner()
    _bind_scheduled_methods(runner)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="m"))
    source = make_restart_source(chat_id="active-42", chat_type="dm")
    session_key = build_session_key(source)
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    runner._running_agents[session_key] = object()
    runner.session_store._entries[session_key] = MagicMock(origin=source)

    await runner._announce_planned_restart("🕐 test notice")

    targets = [call.args[0] for call in adapter.send.await_args_list]
    assert targets == ["active-42", "home-42"]
    for call in adapter.send.await_args_list:
        assert call.kwargs["metadata"]["_interim_send"] is True


@pytest.mark.asyncio
async def test_announce_is_muted_when_flag_disabled():
    runner, adapter = make_restart_runner()
    _bind_scheduled_methods(runner)
    source = make_restart_source(chat_id="active-42", chat_type="group", thread_id="topic-7")
    session_key = build_session_key(source)
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    runner._running_agents[session_key] = object()
    runner.session_store._entries[session_key] = MagicMock(origin=source)
    adapter.send = AsyncMock()

    await runner._announce_planned_restart("🕐 test notice")

    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_control_handlers_marshal_schedule_and_cancel_to_loop():
    from gateway.run_bootstrap import _cancel_restart_handler, _schedule_restart_handler

    class ImmediateLoop:
        @staticmethod
        def call_soon_threadsafe(fn):
            fn()

    runner = SimpleNamespace(
        schedule_restart=MagicMock(return_value={"accepted": True, "scheduled_for": 123.0}),
        cancel_scheduled_restart=MagicMock(return_value={"cancelled": True, "had_scheduled": True}),
    )
    scheduled = _schedule_restart_handler(runner, ImmediateLoop())({"when": 123.0, "reason": "update"})
    cancelled = _cancel_restart_handler(runner, ImmediateLoop())({})

    assert scheduled["accepted"] is True
    assert cancelled["cancelled"] is True
    runner.schedule_restart.assert_called_once_with(123.0, "update")
    runner.cancel_scheduled_restart.assert_called_once_with()


@pytest.mark.asyncio
async def test_control_schedule_handler_rejects_missing_or_unparseable_when():
    from gateway.run_bootstrap import _schedule_restart_handler

    class ImmediateLoop:
        @staticmethod
        def call_soon_threadsafe(fn):
            fn()

    runner = SimpleNamespace(schedule_restart=MagicMock())
    handler = _schedule_restart_handler(runner, ImmediateLoop())

    assert handler({})["error"] == "missing_when"
    assert handler({"when": "not-a-time"})["error"] == "unparseable_when"
    runner.schedule_restart.assert_not_called()


# ── CLI (hermes gateway restart --in/--at/--cancel/--reason) ────────────────


def _restart_args(**overrides):
    args = {"in": None, "at": None, "cancel": False, "reason": "", "all": False,
            "system": False, "force": False, "detach": False}
    args.update(overrides)
    return SimpleNamespace(**args)


def test_cli_restart_scheduled_success(monkeypatch, capsys):
    from hermes_cli.gateway import _cmd_restart
    import gateway.control_socket as control

    monkeypatch.setattr(control, "schedule_gateway_restart",
                        lambda home, when, reason: {"accepted": True, "scheduled_for": when,
                                                    "lead_seconds": 600, "reason": reason})
    _cmd_restart(_restart_args(**{"in": "10m", "reason": "update"}))

    out = capsys.readouterr().out
    assert "scheduled for" in out
    assert "users were notified" in out


def test_cli_restart_scheduled_cancel(monkeypatch, capsys):
    from hermes_cli.gateway import _cmd_restart
    import gateway.control_socket as control

    monkeypatch.setattr(control, "cancel_gateway_restart",
                        lambda home: {"cancelled": True, "had_scheduled": True})
    _cmd_restart(_restart_args(cancel=True))

    out = capsys.readouterr().out
    assert "cancelled" in out.lower()


def test_cli_restart_scheduled_cancel_nothing_pending(monkeypatch, capsys):
    from hermes_cli.gateway import _cmd_restart
    import gateway.control_socket as control

    monkeypatch.setattr(control, "cancel_gateway_restart",
                        lambda home: {"cancelled": False, "had_scheduled": False})
    _cmd_restart(_restart_args(cancel=True))

    out = capsys.readouterr().out
    assert "No scheduled gateway restart is pending" in out


def test_cli_restart_scheduled_mutual_exclusion(monkeypatch, capsys):
    from hermes_cli.gateway import _cmd_restart

    with pytest.raises(SystemExit) as excinfo:
        _cmd_restart(_restart_args(**{"in": "10m", "at": "04:00"}))
    assert excinfo.value.code == 2
    assert "exactly one" in capsys.readouterr().err


def test_cli_restart_scheduled_invalid_at(monkeypatch, capsys):
    from hermes_cli.gateway import _cmd_restart

    with pytest.raises(SystemExit) as excinfo:
        _cmd_restart(_restart_args(at="not-a-time"))
    assert excinfo.value.code == 2
    assert "Invalid --at" in capsys.readouterr().err


def test_cli_restart_scheduled_gateway_unreachable(monkeypatch, capsys):
    from hermes_cli.gateway import _cmd_restart
    import gateway.control_socket as control

    monkeypatch.setattr(control, "schedule_gateway_restart", lambda *a, **k: None)
    with pytest.raises(SystemExit) as excinfo:
        _cmd_restart(_restart_args(**{"in": "10m"}))
    assert excinfo.value.code == 1
    assert "does not support scheduled restarts" in capsys.readouterr().err


def test_cli_restart_scheduled_gateway_refused(monkeypatch, capsys):
    from hermes_cli.gateway import _cmd_restart
    import gateway.control_socket as control

    monkeypatch.setattr(control, "schedule_gateway_restart",
                        lambda *a, **k: {"accepted": False, "error": "lead_too_long"})
    with pytest.raises(SystemExit) as excinfo:
        _cmd_restart(_restart_args(**{"in": "10m"}))
    assert excinfo.value.code == 1
    assert "lead_too_long" in capsys.readouterr().err
