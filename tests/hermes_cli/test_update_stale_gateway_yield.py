"""#117275: a stale-code ticker that yields every tick must never read as healthy, and the
updater must hand a proven-stale survivor to the request_restart path."""

import io
import contextlib
import os
import signal
import subprocess
import sys
import time

import pytest


@pytest.fixture
def cron_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".hermes").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    for name in list(sys.modules):
        if name.split(".")[0] in ("cron", "gateway", "hermes_cli", "hermes_constants"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return home / ".hermes"


def _status_block(pids):
    from hermes_cli.cron import _print_ticker_health

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_ticker_health(pids)
    return buf.getvalue()


@pytest.mark.parametrize("had_success_before_update", [False, True])
def test_cron_status_reports_stale_code_yield_as_unhealthy(cron_home, had_success_before_update):
    """Fresh heartbeat + persisted CronTickYielded == outage, whatever the success marker says."""
    from cron.jobs import _current_cron_store, ensure_dirs, record_ticker_error, record_ticker_heartbeat
    from cron.scheduler import CronTickYielded

    ensure_dirs()
    if had_success_before_update:
        (_current_cron_store().cron_dir / "ticker_last_success").write_text(str(time.time() - 600))
    exc = CronTickYielded("5eb99eb284", "2ed6387d87")
    record_ticker_error(f"{type(exc).__name__}: {exc}")
    record_ticker_heartbeat(success=False)

    out = _status_block([4242])

    assert "cron jobs will fire automatically" not in out
    assert "STALE code" in out
    assert "5eb99eb284" in out and "2ed6387d87" in out
    assert "hermes gateway restart" in out


def test_cron_status_still_green_after_a_clean_tick(cron_home):
    """Control: a ticker whose last cycle succeeded reads healthy."""
    from cron.jobs import clear_ticker_error, ensure_dirs, record_ticker_heartbeat

    ensure_dirs()
    record_ticker_heartbeat(success=True)
    clear_ticker_error()

    assert "cron jobs will fire automatically" in _status_block([4242])


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX drain-first restart signal")
def test_update_signals_proven_stale_gateway_survivor(cron_home, tmp_path):
    """A `stale` fleet-matrix row gets SIGUSR1 (request_restart); a `current` row is untouched."""
    from hermes_cli.update_cmd_fleet import _GatewayRestartOutcome
    from hermes_cli.update_cmd_stale_survivors import signal_stale_fleet_survivors

    marker = tmp_path / "got_sigusr1"
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import signal, sys, time\n"
         f"signal.signal(signal.SIGUSR1, lambda *_: (open({str(marker)!r}, 'w').write('x'), sys.exit(75)))\n"
         "print('ready', flush=True)\n"
         "time.sleep(120)"],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        restart = _GatewayRestartOutcome(
            incomplete=True, phase_errors=[], pre_restart_gateway_pids=[child.pid], restarted_services=[],
            failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
        )
        fleet = [
            {"profile": "default", "pid": child.pid, "state": "stale", "code_sha": "a" * 40},
            {"profile": "fresh", "pid": os.getpid(), "state": "current", "code_sha": "b" * 40},
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            signalled = signal_stale_fleet_survivors(fleet, restart, 10.0)
        assert child.wait(timeout=15) == 75
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        child.stdout.close()

    assert signalled == [child.pid]
    assert marker.exists()
    assert restart.killed_pids == {child.pid}


def test_verify_fleet_hands_stale_rows_to_survivor_signalling(monkeypatch):
    """The wiring: a stale fleet matrix reaches signal_stale_fleet_survivors before exit 1."""
    import hermes_cli.update_cmd_fleet as fleet_mod
    import hermes_cli.update_cmd_stale_survivors as surv
    import hermes_cli.update_cmd as update_cmd
    from hermes_cli.update_cmd_fleet import _GatewayRestartOutcome

    stale_fleet = [{"profile": "default", "pid": 4242, "state": "stale", "code_sha": "a" * 40}]
    seen = {}
    monkeypatch.setattr(fleet_mod, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr(update_cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr(fleet_mod, "_collect_fleet_snapshot", lambda restart, expected: stale_fleet)
    monkeypatch.setattr(surv, "signal_stale_fleet_survivors",
                        lambda fleet, restart, budget: seen.setdefault("fleet", fleet) and [])
    restart = _GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[4242], restarted_services=[],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
    )
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(SystemExit) as exc:
        fleet_mod._verify_fleet_after_update(
            restart, _pre_update_plan=None, _windows_gateway_resume=None, node_failures=[], update_complete=True,
        )
    assert exc.value.code == 1
    assert seen["fleet"] == stale_fleet
