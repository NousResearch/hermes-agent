"""Stale-survivor escalation: supervision judged by outcome.

Regression for the desktop-update failure loop (three identical repros, 2026-09-27): the fleet
matrix flagged the serving gateway stale, the drain-first signal worked, the supervisor queued
the successor — and the update still printed "Stopped 1 manual gateway process(es) that had no
supervisor to respawn them" and exited 1, because the serving pid was compared against launchd's
registered pid (the osascript launcher sits above it). Convergence is now re-verified from the
same fleet snapshot the happy path uses; only survivors that truly never come back get the
manual-restart note.
"""

import contextlib
import io


class _Restart:
    def __init__(self):
        self.killed_pids = set()
        self.incomplete = False


def test_settle_converges_when_successor_appears():
    from hermes_cli.update_cmd_stale_survivors import settle_survivor_restarts

    stale = [{"profile": "default", "pid": 4242, "state": "stale", "code_sha": "a" * 40}]
    calls = {"n": 0}

    def collect():
        calls["n"] += 1
        if calls["n"] < 2:
            return stale
        return [{"profile": "default", "pid": 4343, "state": "current", "code_sha": "b" * 40}]

    converged, fresh = settle_survivor_restarts(stale, collect, budget_s=5.0)
    assert converged is True
    assert [row["state"] for row in fresh] == ["current"]


def test_settle_fails_without_a_successor():
    from hermes_cli.update_cmd_stale_survivors import settle_survivor_restarts

    stale = [{"profile": "default", "pid": 4242, "state": "stale", "code_sha": "a" * 40}]
    converged, _fresh = settle_survivor_restarts(stale, lambda: stale, budget_s=0.1)
    assert converged is False


def test_settle_fails_closed_when_reprobe_raises():
    from hermes_cli.update_cmd_stale_survivors import settle_survivor_restarts

    stale = [{"profile": "default", "pid": 4242, "state": "stale", "code_sha": "a" * 40}]

    def boom():
        raise RuntimeError("probe died")

    converged, fresh = settle_survivor_restarts(stale, boom, budget_s=0.1)
    assert converged is False
    assert fresh == []


def test_report_prints_manual_note_only_for_unrecovered():
    from hermes_cli.update_cmd_stale_survivors import report_unsettled_survivors

    stale = [
        {"profile": "default", "pid": 4242, "state": "stale"},
        {"profile": "coder", "pid": 5252, "state": "stale"},
    ]
    fresh = [
        {"profile": "default", "pid": 4343, "state": "current"},
        {"profile": "coder", "pid": 5252, "state": "stale"},
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_unsettled_survivors(stale, fresh)
    out = buf.getvalue()
    assert "Stopped 1 manual gateway process(es) that had no supervisor to respawn them" in out
    assert "Restart manually: hermes gateway run" in out


def test_report_silent_when_all_recovered():
    from hermes_cli.update_cmd_stale_survivors import report_unsettled_survivors

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_unsettled_survivors(
            [{"profile": "default", "pid": 4242, "state": "stale"}],
            [{"profile": "default", "pid": 4343, "state": "current"}],
        )
    assert buf.getvalue() == ""


def test_signal_no_longer_prints_the_manual_note(monkeypatch):
    """The escalation's own output must not carry the misleading manual-restart verdict."""
    import hermes_cli.update_cmd_fleet as fleet_mod
    from hermes_cli.update_cmd_stale_survivors import signal_stale_fleet_survivors

    monkeypatch.setattr(
        fleet_mod, "_drain_or_signal_gateway_for_update",
        lambda pid, budget, label: True,
    )
    restart = _Restart()
    fleet = [{"profile": "default", "pid": 99999, "state": "stale"}]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        signalled = signal_stale_fleet_survivors(fleet, restart, 1.0)
    assert signalled == [99999]
    assert restart.killed_pids == {99999}
    out = buf.getvalue()
    assert "manual gateway process" not in out
    assert "requesting a restart" in out
