"""/after — one power action once all agent work is done (hermes_cli/power_after.py).

Behavior contracts, driven on a fake clock with the watcher run inline (no thread, no real
power action):

- idle past the grace window -> countdown announced -> action fires exactly once, then disarmed
- /after cancel during the countdown -> nothing fires
- new work during the countdown -> aborts with a notice; fires only after the next idle window
- never fires while a turn or background work keeps the process busy
"""

import pytest

from hermes_cli import power_after
from hermes_cli.power_after import _COUNTDOWN_S, _GRACE_S, PowerAfter


class _Host:
    """Fake clock + work picture. ``schedule`` maps a clock time to a callback run once reached."""

    def __init__(self) -> None:
        self.t = 0.0
        self.quiet_since = 0.0
        self.busy = False
        self.fired: list[str] = []
        self.notes: list[str] = []
        self.schedule: dict[float, object] = {}

    def quiet_for(self) -> float:
        return 0.0 if self.busy else self.t - self.quiet_since

    def advance(self, seconds: float) -> None:
        self.t += seconds
        for at in sorted(k for k in self.schedule if k <= self.t):
            self.schedule.pop(at)()
        if self.t > 24 * 3600:
            raise AssertionError("watcher never settled")


@pytest.fixture
def host_and_pa():
    host, pa = _Host(), PowerAfter()
    pa._now = lambda: host.t
    pa._quiet_for = host.quiet_for
    pa._busy = lambda: host.busy
    pa._fire = host.fired.append
    pa._sleep = host.advance
    pa._start = lambda target: None  # the test drives _watch inline
    return host, pa


def test_fires_once_after_grace_and_countdown(host_and_pa):
    host, pa = host_and_pa
    pa.arm("shutdown", host.notes.append)
    pa._watch()
    assert host.fired == ["shutdown"]
    assert host.t >= _GRACE_S + _COUNTDOWN_S
    assert "Shutting down in 60s" in host.notes[0]
    assert pa.armed() is None


def test_cancel_during_countdown_fires_nothing(host_and_pa):
    host, pa = host_and_pa
    host.schedule[_GRACE_S + 10] = pa.cancel
    pa.arm("sleep", host.notes.append)
    pa._watch()
    assert host.fired == []
    assert pa.armed() is None


def test_new_work_during_countdown_waits_for_next_idle_window(host_and_pa):
    host, pa = host_and_pa
    work_starts, work_ends = _GRACE_S + 10, _GRACE_S + 400

    def start():
        host.busy = True

    def end():
        host.busy, host.quiet_since = False, host.t

    host.schedule.update({work_starts: start, work_ends: end})
    pa.arm("hibernate", host.notes.append)
    pa._watch()
    assert host.fired == ["hibernate"]
    assert any("New work started" in note for note in host.notes)
    assert host.t >= work_ends + _GRACE_S + _COUNTDOWN_S


def test_busy_background_work_blocks_firing(host_and_pa):
    host, pa = host_and_pa
    host.busy = True
    host.schedule[3 * 3600] = pa.cancel
    pa.arm("shutdown", host.notes.append)
    pa._watch()
    assert host.fired == [] and host.notes == []


def test_run_command_arms_reports_and_cancels(monkeypatch):
    pa = PowerAfter()
    pa._start = lambda target: None
    monkeypatch.setattr(power_after, "POWER_AFTER", pa)
    monkeypatch.setattr(power_after, "unsupported_reason", lambda action: None)

    assert power_after.run_command("").startswith("Nothing armed")
    assert power_after.run_command("reboot").startswith("Usage:")
    assert "Send /after cancel" in power_after.run_command("Shutdown")
    assert power_after.run_command("status") == "Armed: /after shutdown runs once all agent work is done."
    assert power_after.run_command("cancel") == "Cancelled /after shutdown."
    assert power_after.run_command("cancel") == "Nothing was armed."
