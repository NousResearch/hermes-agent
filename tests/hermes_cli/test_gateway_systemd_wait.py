"""Systemd restart readiness wait: a booting replacement is waited for, not declared dead (#137995)."""

import hermes_cli.gateway as gateway_cli
from hermes_cli import gateway_systemd_wait


class TestSystemdRestartWaitBootProgress:
    """A multiplexer serving many profiles connects their adapters one after another, so a healthy
    cold boot can outlast the static supervisor budget. The restart wait must keep waiting while the
    replacement publishes startup steps, and report a still-booting gateway as such (#137995)."""

    BUDGET = 155.0

    @staticmethod
    def _starting(pid, connected, *, writer_pid=None, gateway_state="starting"):
        writer = pid if writer_pid is None else writer_pid
        return {
            "pid": pid,
            "gateway_state": gateway_state,
            "platforms": {
                f"emp{i}:telegram": {"state": "connected", "writer_pid": writer}
                for i in range(connected)
            },
        }

    def _drive(self, monkeypatch, timeline, *, pid_alive=True):
        """Run the wait against a fake clock. ``timeline(now)`` returns ``(main_pid, runtime_state)``
        at monotonic time ``now``; every poll sleeps ~2s."""
        clock = [0.0]
        monkeypatch.setattr(gateway_cli.time, "monotonic", lambda: clock[0])

        def _sleep(seconds):
            clock[0] += seconds

        monkeypatch.setattr(gateway_cli.time, "sleep", _sleep)
        monkeypatch.setattr(
            gateway_cli,
            "_read_systemd_unit_properties",
            lambda system=False, properties=None: {
                "ActiveState": "active",
                "SubState": "running",
                "MainPID": str(timeline(clock[0])[0]),
            },
        )
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: timeline(clock[0])[0])
        monkeypatch.setattr(
            gateway_cli, "_read_gateway_runtime_status", lambda: timeline(clock[0])[1]
        )
        monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: pid_alive)
        result = gateway_cli._wait_for_systemd_service_restart(
            previous_pid=654, timeout=self.BUDGET
        )
        return result, clock[0]

    def test_serial_boot_longer_than_budget_reports_restarted(self, monkeypatch, capsys):
        """18 profiles at ~11.5s each finish ~210s in — past the 155s budget — and still succeed."""
        boot_done = 210.0

        def timeline(now):
            if now >= boot_done:
                return 777, self._starting(777, 18, gateway_state="running")
            return 777, self._starting(777, int(now // 11.5))

        result, elapsed = self._drive(monkeypatch, timeline)

        out = capsys.readouterr().out
        assert result is True
        assert elapsed >= boot_done > self.BUDGET
        assert "restarted (PID 777)" in out
        assert "did not become active" not in out

    def test_stalled_boot_is_not_extended(self, monkeypatch, capsys):
        """A boot that stops publishing steps gets only the remaining grace, not an open-ended wait."""

        def timeline(now):
            return 777, self._starting(777, min(int(now // 11.5), 3))

        result, elapsed = self._drive(monkeypatch, timeline)

        out = capsys.readouterr().out
        assert result is False
        # Last step at ~34.5s; one grace past it lands inside the base budget.
        assert elapsed <= self.BUDGET + 2.0
        assert "still starting" in out

    def test_extensions_are_capped(self, monkeypatch):
        """A boot that flips a platform forever cannot hold the CLI past the ceiling."""

        def timeline(now):
            return 777, self._starting(777, int(now // 10) % 2)

        result, elapsed = self._drive(monkeypatch, timeline)

        assert result is False
        ceiling = self.BUDGET + gateway_systemd_wait._RESTART_BOOT_PROGRESS_MAX_EXTENSION_S
        assert ceiling - 5.0 <= elapsed <= ceiling + 2.0

    def test_crash_looping_replacements_do_not_extend(self, monkeypatch, capsys):
        """Each new PID is a fresh boot after a death — that is not progress of the first one."""

        def timeline(now):
            pid = 1000 + int(now // 20)
            return pid, self._starting(pid, int(now % 20 // 5))

        result, elapsed = self._drive(monkeypatch, timeline)

        assert result is False
        assert elapsed <= self.BUDGET + 2.0
        assert "did not become active" in capsys.readouterr().out

    def test_previous_generation_entries_are_not_progress(self, monkeypatch):
        """Platform entries the old PID wrote (still in the file) never count for the new one."""

        def timeline(now):
            return 777, self._starting(777, int(now // 11.5), writer_pid=654)

        result, elapsed = self._drive(monkeypatch, timeline)

        assert result is False
        assert elapsed <= self.BUDGET + 2.0

    def test_dead_booting_pid_is_not_reported_as_still_starting(self, monkeypatch, capsys):
        def timeline(now):
            return 777, self._starting(777, 1)

        result, _elapsed = self._drive(monkeypatch, timeline, pid_alive=False)

        out = capsys.readouterr().out
        assert result is False
        assert "still starting" not in out
        assert "did not become active" in out
