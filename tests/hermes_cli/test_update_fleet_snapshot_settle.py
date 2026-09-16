"""Regression for #111272: wait for a supervised gateway to publish its state.

A systemd restart can report the unit active before the replacement gateway
writes ``gateway_state.json``. The old 30-second fleet snapshot window then
returned no rows, treated a successful restart as incomplete, and left
``fleet_restart_pending`` behind. This test uses a fake monotonic clock and a
late current row to pin the bounded readiness window without starting a live
service.
"""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli import update_cmd
import hermes_cli.update_cmd_fleet as update_cmd_fleet
from hermes_constants import get_hermes_home


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_snapshot_waits_for_late_current_gateway_state(monkeypatch) -> None:
    """A current successor published after 30 seconds still completes verification."""
    clock = _FakeClock()
    expected = {"profile": "default", "pid": 202, "code_sha": "new", "state": "current"}
    snapshots = iter([[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [expected]])

    monkeypatch.setattr(update_cmd_fleet._time, "monotonic", clock.monotonic)
    monkeypatch.setattr(update_cmd_fleet._time, "sleep", clock.sleep)
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **_kwargs: next(snapshots),
    )

    restart = SimpleNamespace(pre_restart_gateway_pids=[101])

    result = update_cmd_fleet._collect_fleet_snapshot(restart, rows_expected=True)

    assert result == [expected]
    assert clock.now > 30.0
    assert clock.now <= update_cmd_fleet._FLEET_PROBE_SETTLE_TIMEOUT_SECONDS


def test_snapshot_stops_waiting_once_the_restarted_unit_is_dead(monkeypatch) -> None:
    """A successor that exits (unit failed/inactive) fails closed at once, not at the 120s deadline."""
    clock = _FakeClock()
    monkeypatch.setattr(update_cmd_fleet._time, "monotonic", clock.monotonic)
    monkeypatch.setattr(update_cmd_fleet._time, "sleep", clock.sleep)
    monkeypatch.setattr("hermes_cli.update_receipt.collect_fleet_versions", lambda **_kwargs: [])
    monkeypatch.setattr(
        update_cmd_fleet, "_systemctl",
        lambda cmd, *, timeout: SimpleNamespace(stdout="failed\n", stderr="", returncode=3))

    restart = SimpleNamespace(pre_restart_gateway_pids=[101], restarted_scoped_units={"user/hermes-gateway.service"})
    assert update_cmd_fleet._collect_fleet_snapshot(restart, rows_expected=True) == []
    assert clock.now < 30.0


def test_verifier_clears_marker_after_late_current_gateway_state(monkeypatch) -> None:
    """A late current row reaches the normal success-only marker cleanup."""
    clock = _FakeClock()
    expected = {"profile": "default", "pid": 202, "code_sha": "new", "state": "current"}
    snapshots = iter([[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [expected]])
    marker = get_hermes_home() / "fleet_restart_pending"
    marker.write_text("started=1\npid=101\nexpected_sha=new\n", encoding="utf-8")

    monkeypatch.setattr(update_cmd_fleet._time, "monotonic", clock.monotonic)
    monkeypatch.setattr(update_cmd_fleet._time, "sleep", clock.sleep)
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **_kwargs: next(snapshots),
    )
    monkeypatch.setattr(
        "hermes_cli.update_receipt.print_fleet_version_matrix",
        lambda _fleet: False,
    )
    monkeypatch.setattr(update_cmd_fleet, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr(update_cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda _plan: [])
    monkeypatch.setattr(update_cmd, "_warn_stale_serve_runtimes", lambda _rows: None)
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(_fleet_probe_expected_runtimes=lambda *a: True))
    monkeypatch.setattr(update_cmd_fleet, "_clear_fleet_restart_pending_marker", lambda: marker.unlink())

    restart = SimpleNamespace(
        incomplete=False,
        pre_restart_gateway_pids=[101],
        restarted_services=["hermes-gateway.service"],
        relaunched_profiles=[],
        externally_supervised_profiles=[],
        killed_pids=set(),
        failed_or_stale_units=[],
        fleet_probe_signals=lambda: ([101], set()),
    )

    update_cmd_fleet._verify_fleet_after_update(
        restart,
        _pre_update_plan=None,
        _windows_gateway_resume=None,
        node_failures=[],
        update_complete=True,
    )

    assert restart.incomplete is False
    assert not marker.exists()
