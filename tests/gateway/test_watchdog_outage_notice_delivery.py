"""The recovered gateway delivers the Windows hang watchdog's outage notice (CORP-836).

The watchdog records the outage window in ``.restart_pending.json``; the boot/reconnect replay must
send the window + resend request instead of the plain "online" notice, exactly once per home, with
one recovery time across replays, and must survive a failed send like any planned-restart notice.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from agent.i18n import t
from gateway import watchdog_outage
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult

_T = 1_793_400_000.0


def _adapter(success=True):
    return SimpleNamespace(
        send_path_degraded=False,
        send=AsyncMock(return_value=SendResult(success=success, message_id="unit-test-notice")),
    )


def _home_config(platform, chat_id, *, notify=True):
    return GatewayConfig(platforms={platform: PlatformConfig(
        enabled=True, gateway_restart_notification=notify,
        home_channel=HomeChannel(platform=platform, chat_id=chat_id, name=chat_id))})


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = _home_config(Platform.SLACK, "C-home")
    runner.config.sessions_dir = tmp_path / "sessions"
    runner.adapters = {}
    runner._profile_configs = {"coder": _home_config(Platform.TELEGRAM, "coder-home")}
    runner._profile_adapters = {"coder": {}}
    runner._planned_restart_notice_lock = None
    watchdog_outage.record_outage(tmp_path, last_healthy_at=_T, detected_at=_T + 60)
    return runner, tmp_path / ".restart_pending.json"


@pytest.mark.asyncio
async def test_recovered_gateway_sends_the_outage_window_and_resend_request_once(runner, monkeypatch):
    runner, marker = runner
    monkeypatch.setattr("gateway.run_notifications.time.time", lambda: _T + 300)
    slack = _adapter()
    runner.adapters[Platform.SLACK] = slack

    await runner._replay_pending_planned_restart_notification()  # boot: coder's Telegram is down

    expected = t(watchdog_outage.NOTICE_KEY, start=watchdog_outage.format_time(_T),
                 end=watchdog_outage.format_time(_T + 300))
    assert slack.send.await_args.args[0] == "C-home"
    assert slack.send.await_args.args[1].startswith(expected)
    assert json.loads(marker.read_text(encoding="utf-8"))[watchdog_outage.OUTAGE_KEY]["recovered_at"] == _T + 300

    monkeypatch.setattr("gateway.run_notifications.time.time", lambda: _T + 900)
    coder = _adapter()
    runner._profile_adapters["coder"][Platform.TELEGRAM] = coder
    await runner._replay_pending_planned_restart_notification()  # reconnect

    assert coder.send.await_args.args[0] == "coder-home"
    assert coder.send.await_args.args[1].startswith(expected), "one recovery time across replays"
    assert slack.send.await_count == 1, "a reached home is never notified twice"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_a_failed_outage_send_keeps_it_owed_and_raises_nothing(runner):
    runner, marker = runner
    runner._profile_configs = {}
    runner.adapters[Platform.SLACK] = _adapter(success=False)

    await runner._replay_pending_planned_restart_notification()

    assert marker.exists(), "the outage notice is still owed"
    runner.adapters[Platform.SLACK] = slack = _adapter()
    await runner._replay_pending_planned_restart_notification()
    assert "{" not in slack.send.await_args.args[1] and watchdog_outage.format_time(_T) in slack.send.await_args.args[1]
    assert not marker.exists()


@pytest.mark.asyncio
async def test_the_restart_notification_opt_out_also_silences_the_outage_notice(runner):
    runner, _marker = runner
    runner.config = _home_config(Platform.SLACK, "C-home", notify=False)
    runner.config.sessions_dir = _marker.parent / "sessions"
    runner._profile_configs = {}
    runner.adapters[Platform.SLACK] = slack = _adapter()

    await runner._replay_pending_planned_restart_notification()

    slack.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_in_band_planned_restart_keeps_a_pending_outage_record(tmp_path, monkeypatch):
    import gateway.status as gateway_status
    from gateway.run_shutdown import GatewayShutdownMixin

    async def flushed(timeout=None):
        return True

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_status, "remove_pid_file", lambda *a, **k: None)
    monkeypatch.setattr(gateway_status, "release_gateway_runtime_lock", lambda *a, **k: None)
    monkeypatch.setattr(gateway_status, "flush_runtime_status_async", flushed)
    monkeypatch.setattr(gateway_run, "_shutdown_gateway_health_export", lambda runner: None)
    watchdog_outage.record_outage(tmp_path, last_healthy_at=_T, detected_at=_T + 60)
    runner = SimpleNamespace(
        _restart_requested=True, _restart_command_source=None, _restart_via_service=False,
        _restart_detached=False, _exit_reason=None, _draining=True,
        _update_runtime_status=lambda *a, **k: None)
    ctx = GatewayShutdownMixin._StopContext(deferred_count=lambda: 0, started_at=time.monotonic())

    await gateway_run.GatewayRunner._stop_persist_exit_state(runner, ctx)

    data = json.loads((tmp_path / ".restart_pending.json").read_text(encoding="utf-8"))
    assert data[watchdog_outage.OUTAGE_KEY] == {"last_healthy_at": _T, "detected_at": _T + 60}
    assert "requested_at" in data
