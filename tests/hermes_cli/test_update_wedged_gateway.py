"""Tests for the wedged-gateway health probe + bounded escalation (#81642).

A gateway whose event loop is stalled cannot process a graceful shutdown, so
the updater's drain wait used to burn the full 180s budget ("Gateway PID X
still running after 180.0s — restart may fail") and could deadlock `hermes
update`. The fix probes the loop-liveness heartbeat file BEFORE draining and,
only when the loop is provably dead, escalates SIGTERM → SIGKILL bounded to
seconds. A busy-but-alive gateway (fresh heartbeat) must keep the full drain
path — including the in-flight cron drain floor from #86684.
"""

import json
import os
import time

import pytest

import hermes_cli.gateway as gateway_cli
from gateway.shutdown_watchdog import get_loop_heartbeat_path, write_loop_heartbeat


def _write_heartbeat(home, pid, age_s=0.0):
    """Write a heartbeat file for ``pid`` whose mtime is ``age_s`` old."""
    path = get_loop_heartbeat_path(home)
    write_loop_heartbeat(pid=pid, home=home)
    if age_s:
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))
    return path


class TestProbeGatewayLoopLiveness:
    def test_fresh_heartbeat_is_alive(self, tmp_path):
        """A gateway that refreshed its heartbeat recently is busy, not wedged."""
        _write_heartbeat(tmp_path, pid=4242, age_s=5.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(4242, home=tmp_path)
            == gateway_cli.GATEWAY_LOOP_ALIVE
        )

    def test_stale_heartbeat_is_wedged(self, tmp_path):
        """A heartbeat several missed beats old proves the loop is dead."""
        _write_heartbeat(tmp_path, pid=4242, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(4242, home=tmp_path)
            == gateway_cli.GATEWAY_LOOP_WEDGED
        )

    def test_heartbeat_just_inside_budget_is_alive(self, tmp_path):
        """Boundary: age below the stale budget must NOT classify as wedged."""
        _write_heartbeat(tmp_path, pid=4242, age_s=60.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(
                4242, stale_after=90.0, home=tmp_path
            )
            == gateway_cli.GATEWAY_LOOP_ALIVE
        )

    def test_missing_heartbeat_is_unknown(self, tmp_path):
        """No heartbeat file (older gateway, fresh start) is not evidence."""
        assert (
            gateway_cli.probe_gateway_loop_liveness(4242, home=tmp_path)
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )

    def test_pid_mismatch_is_unknown_even_when_stale(self, tmp_path):
        """A stale file from a PREVIOUS process must not condemn the new PID."""
        _write_heartbeat(tmp_path, pid=1111, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(4242, home=tmp_path)
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )

    def test_corrupt_heartbeat_is_unknown(self, tmp_path):
        path = get_loop_heartbeat_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        stamp = time.time() - 600.0
        os.utime(path, (stamp, stamp))
        assert (
            gateway_cli.probe_gateway_loop_liveness(4242, home=tmp_path)
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )

    def test_nonpositive_pid_is_unknown(self, tmp_path):
        _write_heartbeat(tmp_path, pid=4242, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(0, home=tmp_path)
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )

    def test_invalid_stale_after_falls_back_to_default(self, tmp_path):
        _write_heartbeat(tmp_path, pid=4242, age_s=600.0)
        assert (
            gateway_cli.probe_gateway_loop_liveness(
                4242, stale_after="bogus", home=tmp_path
            )
            == gateway_cli.GATEWAY_LOOP_WEDGED
        )

    def test_probe_never_raises_on_unreadable_path(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.shutdown_watchdog.get_loop_heartbeat_path",
            lambda home=None: (_ for _ in ()).throw(OSError("boom")),
        )
        assert (
            gateway_cli.probe_gateway_loop_liveness(4242)
            == gateway_cli.GATEWAY_LOOP_UNKNOWN
        )


class TestEscalateWedgedGateway:
    def test_sigterm_grace_suffices_without_sigkill(self, monkeypatch):
        """If SIGTERM lands (signal-handler thread alive), no SIGKILL is sent."""
        signals = []
        monkeypatch.setattr(
            gateway_cli,
            "terminate_pid",
            lambda pid, force=False: signals.append(("kill" if force else "term", pid)),
        )
        monkeypatch.setattr(
            gateway_cli, "_wait_for_pid_exit", lambda pid, timeout: True
        )

        assert gateway_cli._escalate_wedged_gateway(4242) is True
        assert signals == [("term", 4242)]

    def test_escalates_to_sigkill_when_sigterm_ignored(self, monkeypatch):
        signals = []
        waits = []

        def fake_wait(pid, timeout):
            waits.append(timeout)
            # First wait (SIGTERM grace) times out; second (post-SIGKILL) succeeds.
            return len(waits) > 1

        monkeypatch.setattr(
            gateway_cli,
            "terminate_pid",
            lambda pid, force=False: signals.append(("kill" if force else "term", pid)),
        )
        monkeypatch.setattr(gateway_cli, "_wait_for_pid_exit", fake_wait)

        assert gateway_cli._escalate_wedged_gateway(4242) is True
        assert signals == [("term", 4242), ("kill", 4242)]

    def test_total_wait_budget_is_bounded_well_under_drain(self, monkeypatch):
        """Worst case must be seconds, never the 180s drain budget."""
        waits = []
        monkeypatch.setattr(gateway_cli, "terminate_pid", lambda pid, force=False: None)
        monkeypatch.setattr(
            gateway_cli,
            "_wait_for_pid_exit",
            lambda pid, timeout: waits.append(timeout) or False,
        )

        assert gateway_cli._escalate_wedged_gateway(4242) is False
        assert sum(waits) < 30.0

    def test_process_already_gone_is_success(self, monkeypatch):
        def raise_gone(pid, force=False):
            raise ProcessLookupError

        monkeypatch.setattr(gateway_cli, "terminate_pid", raise_gone)
        monkeypatch.setattr(
            gateway_cli, "_wait_for_pid_exit", lambda pid, timeout: True
        )

        assert gateway_cli._escalate_wedged_gateway(4242) is True

    def test_sigkill_permission_error_does_not_raise(self, monkeypatch):
        calls = []

        def term(pid, force=False):
            calls.append(force)
            if force:
                raise PermissionError

        monkeypatch.setattr(gateway_cli, "terminate_pid", term)
        monkeypatch.setattr(
            gateway_cli, "_wait_for_pid_exit", lambda pid, timeout: False
        )

        assert gateway_cli._escalate_wedged_gateway(4242) is False
        assert calls == [False, True]


class TestLaunchdRestartWedgedIntegration:
    """launchd_restart must skip the 180s drain only for a wedged loop."""

    def _setup(self, monkeypatch, liveness):
        events = []
        monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
        monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
        monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 180.0)
        # Wait budget covers after-turn deferral + drain + headroom (#77184).
        monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 195.0)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: 4242)
        monkeypatch.setattr(
            gateway_cli, "_request_gateway_self_restart", lambda pid: False
        )
        monkeypatch.setattr(
            gateway_cli,
            "probe_gateway_loop_liveness",
            lambda pid, **kw: events.append("probe") or liveness,
        )
        monkeypatch.setattr(
            gateway_cli,
            "_escalate_wedged_gateway",
            lambda pid, **kw: events.append("escalate") or True,
        )
        monkeypatch.setattr(
            gateway_cli,
            "terminate_pid",
            lambda pid, force=False: events.append("sigterm"),
        )
        # Never let a real SIGUSR1 escape to PID 4242 during tests.
        monkeypatch.setattr(
            gateway_cli,
            "_graceful_restart_via_sigusr1",
            lambda pid, timeout: events.append(("drain", pid, timeout)) or True,
        )
        # KeepAlive revival observed instantly — avoids the real 15s poll
        # (mocked subprocess.run returns empty stdout, so the PID probe
        # would otherwise burn the full observation timeout in time.sleep).
        monkeypatch.setattr(
            gateway_cli,
            "_wait_for_launchd_service_pid",
            lambda label, old_pid, timeout=10.0, *, domain: events.append("observe")
            or True,
        )
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda *a, **k: events.append("kickstart")
            or __import__("types").SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        monkeypatch.setattr(
            gateway_cli, "_clear_launchd_unsupported_marker", lambda: None
        )
        return events

    def test_wedged_gateway_skips_drain_and_escalates(self, monkeypatch):
        events = self._setup(monkeypatch, gateway_cli.GATEWAY_LOOP_WEDGED)
        gateway_cli.launchd_restart()
        assert "escalate" in events
        # The 180s drain wait must never run for a wedged loop.
        assert not any(isinstance(e, tuple) and e[0] == "drain" for e in events)

    def test_busy_gateway_keeps_full_drain_budget(self, monkeypatch):
        """A busy-but-alive gateway (fresh heartbeat) must NOT be escalated —
        that would bypass the in-flight cron drain floor (#86684)."""
        events = self._setup(monkeypatch, gateway_cli.GATEWAY_LOOP_ALIVE)
        gateway_cli.launchd_restart()
        assert "escalate" not in events
        assert ("drain", 4242, 195.0) in events

    def test_unknown_liveness_keeps_full_drain_budget(self, monkeypatch):
        """Ambiguity (no heartbeat) must never trigger escalation."""
        events = self._setup(monkeypatch, gateway_cli.GATEWAY_LOOP_UNKNOWN)
        gateway_cli.launchd_restart()
        assert "escalate" not in events
        assert ("drain", 4242, 195.0) in events
