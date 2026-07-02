"""Tests for the cron ticker supervisor stall/restart home-channel alert.

PR #71 made the cron ticker self-heal after a stall, but the restart was
silent — the original 17h stall went unnoticed precisely because nothing
announced it. These tests lock in the behaviour of the alert that surfaces the
event: it fires exactly once per stall, respects the enable flag and cooldown,
and is fully exception-isolated so it can never block or delay ticker recovery.
"""
import threading
from unittest.mock import MagicMock, patch

import pytest


class TestCronSupervisorAlert:
    def _run_one_stall(self, monkeypatch, alerts_enabled=True, **supervisor_kwargs):
        """Drive the supervisor through exactly one stale-heartbeat detection.

        Returns the restart-count dict and the alert-send mock so callers can
        assert on both.
        """
        from gateway import run as gateway_run

        stop_event = threading.Event()
        restarts = {"n": 0}

        def fake_restart():
            restarts["n"] += 1
            stop_event.set()  # exit the supervisor loop after one restart

        monkeypatch.setattr(
            gateway_run, "_cron_supervisor_alerts_enabled", lambda: alerts_enabled
        )

        with patch("cron.scheduler.ticker_heartbeat_is_stale", return_value=True), \
             patch("cron.scheduler.get_ticker_heartbeat_age", return_value=1234.0):
            gateway_run._cron_ticker_supervisor(
                stop_event,
                fake_restart,
                interval=60,
                stale_multiplier=5.0,
                check_interval=0,
                **supervisor_kwargs,
            )

        return restarts

    def test_alert_sent_once_on_stale_restart(self, monkeypatch):
        """A stale heartbeat restart triggers exactly one alert send."""
        from gateway import run as gateway_run

        alert = MagicMock()
        monkeypatch.setattr(gateway_run, "_send_cron_supervisor_alert", alert)

        adapters = {"slack": MagicMock()}
        loop = MagicMock()
        restarts = self._run_one_stall(
            monkeypatch, adapters=adapters, loop=loop
        )

        assert restarts["n"] == 1
        assert alert.call_count == 1
        # Called with adapters, loop, stall age, threshold.
        args = alert.call_args.args
        assert args[0] is adapters
        assert args[1] is loop
        assert args[2] == pytest.approx(1234.0)
        assert args[3] == pytest.approx(300.0)  # interval * stale_multiplier

    def test_no_alert_when_healthy(self, monkeypatch):
        """A fresh heartbeat sends no alert and does not restart."""
        from gateway import run as gateway_run

        alert = MagicMock()
        monkeypatch.setattr(gateway_run, "_send_cron_supervisor_alert", alert)

        stop_event = threading.Event()
        restarts = {"n": 0}
        call_count = {"n": 0}

        def not_stale(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                stop_event.set()
            return False

        with patch("cron.scheduler.ticker_heartbeat_is_stale", side_effect=not_stale):
            gateway_run._cron_ticker_supervisor(
                stop_event,
                lambda: restarts.__setitem__("n", restarts["n"] + 1),
                adapters={"slack": MagicMock()},
                loop=MagicMock(),
                interval=60,
                stale_multiplier=5.0,
                check_interval=0,
            )

        assert restarts["n"] == 0
        assert alert.call_count == 0

    def test_restart_still_happens_when_alert_raises(self, monkeypatch):
        """If the alert helper raises, the ticker is still restarted and the
        supervisor does not crash (exception isolation)."""
        from gateway import run as gateway_run

        def exploding_alert(*args, **kwargs):
            raise RuntimeError("send exploded")

        monkeypatch.setattr(
            gateway_run, "_send_cron_supervisor_alert", exploding_alert
        )

        # restart_ticker runs BEFORE the alert, so restart must have incremented
        # even though the alert raises.
        restarts = self._run_one_stall(
            monkeypatch, adapters={"slack": MagicMock()}, loop=MagicMock()
        )

        assert restarts["n"] == 1  # restart happened despite alert failure

    def test_cooldown_suppresses_second_alert(self, monkeypatch):
        """Two stalls inside the cooldown window produce only one alert."""
        from gateway import run as gateway_run

        alert = MagicMock()
        monkeypatch.setattr(gateway_run, "_send_cron_supervisor_alert", alert)
        monkeypatch.setattr(
            gateway_run, "_cron_supervisor_alerts_enabled", lambda: True
        )

        stop_event = threading.Event()
        restarts = {"n": 0}

        def fake_restart():
            restarts["n"] += 1
            if restarts["n"] >= 2:
                stop_event.set()  # stop after two stalls/restarts

        # monotonic advances only slightly between the two stalls, well within
        # the cooldown window, so the second alert must be suppressed.
        times = iter([1000.0, 1000.5, 1001.0, 1001.5, 1002.0])
        monkeypatch.setattr(
            gateway_run.time, "monotonic", lambda: next(times, 2000.0)
        )

        with patch("cron.scheduler.ticker_heartbeat_is_stale", return_value=True), \
             patch("cron.scheduler.get_ticker_heartbeat_age", return_value=1234.0):
            gateway_run._cron_ticker_supervisor(
                stop_event,
                fake_restart,
                adapters={"slack": MagicMock()},
                loop=MagicMock(),
                interval=60,
                stale_multiplier=5.0,
                check_interval=0,
                alert_cooldown=900.0,
            )

        assert restarts["n"] == 2  # both stalls restarted the ticker
        assert alert.call_count == 1  # but only the first alerted

    def test_disabled_config_skips_alert_but_restarts(self, monkeypatch):
        """When cron.supervisor_alerts is False, no alert is sent but the ticker
        is still restarted."""
        from gateway import run as gateway_run

        alert = MagicMock()
        monkeypatch.setattr(gateway_run, "_send_cron_supervisor_alert", alert)

        restarts = self._run_one_stall(
            monkeypatch,
            alerts_enabled=False,
            adapters={"slack": MagicMock()},
            loop=MagicMock(),
        )

        assert restarts["n"] == 1  # restart still happens
        assert alert.call_count == 0  # alert suppressed by config


class TestSendCronSupervisorAlert:
    """Direct tests of the send helper's resolution + isolation behaviour."""

    def test_returns_quietly_without_loop(self):
        from gateway import run as gateway_run

        # No loop -> just logs and returns, no crash.
        gateway_run._send_cron_supervisor_alert(
            {"slack": MagicMock()}, None, 100.0, 300.0
        )

    def test_returns_quietly_without_adapters(self):
        from gateway import run as gateway_run

        gateway_run._send_cron_supervisor_alert(None, MagicMock(), 100.0, 300.0)

    def test_sends_to_resolved_home_channel(self, monkeypatch):
        from gateway import run as gateway_run

        adapter = MagicMock()
        # Coroutine-ish sentinel returned by adapter.send; the real send is a
        # coroutine but we mock the scheduling layer so a MagicMock is fine.
        adapter.send.return_value = MagicMock()
        adapters = {"slack": adapter}
        loop = MagicMock()

        future = MagicMock()
        future.result.return_value = MagicMock(success=True)

        with patch("cron.scheduler._iter_home_target_platforms", return_value=["slack"]), \
             patch("cron.scheduler._get_home_target_chat_id", return_value="C123"), \
             patch("cron.scheduler._get_home_target_thread_id", return_value=None), \
             patch(
                 "agent.async_utils.safe_schedule_threadsafe", return_value=future
             ):
            gateway_run._send_cron_supervisor_alert(adapters, loop, 1234.0, 300.0)

        adapter.send.assert_called_once()
        sent_chat_id, sent_text = adapter.send.call_args.args
        assert sent_chat_id == "C123"
        assert "stalled" in sent_text
        assert "restarted" in sent_text
        future.result.assert_called_once_with(timeout=30)

    def test_timeout_cancels_future_and_does_not_raise(self, monkeypatch):
        from gateway import run as gateway_run

        adapter = MagicMock()
        adapters = {"slack": adapter}
        future = MagicMock()
        future.result.side_effect = TimeoutError()

        with patch("cron.scheduler._iter_home_target_platforms", return_value=["slack"]), \
             patch("cron.scheduler._get_home_target_chat_id", return_value="C123"), \
             patch("cron.scheduler._get_home_target_thread_id", return_value=None), \
             patch(
                 "agent.async_utils.safe_schedule_threadsafe", return_value=future
             ):
            # Must not raise.
            gateway_run._send_cron_supervisor_alert(
                adapters, MagicMock(), 1234.0, 300.0
            )

        future.cancel.assert_called_once()
