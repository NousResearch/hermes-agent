"""#126177: the fleet-restart tail must bracket itself against a live Desktop.

An external ``hermes update`` (maintenance script, second profile's CLI, the
gateway's own /update) runs its fleet bounce in the detached completion child.
The Desktop's backend-spawn gate waits on the on-disk update marker, whose
owner — the parent updater — has already exited by then, so in that window a
Desktop backend respawn raced the bouncing gateway into ``localBackendLifecycle.shutdown()``
("Hermes Desktop is quitting.").

Two halves close the window, pinned here:

- the restart phase, when the shared platform-neutral Desktop-lifecycle probe
  sees a live Desktop supervising this install's control plane, re-holds the
  update marker for the tail (``held_marker_for_desktop`` on the outcome);
- the verification phase releases that hold on every settle path (success,
  stale exit 1, and the restart-phase abort recovery), so the gate opens again.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import hermes_cli.update_cmd_fleet as fleet_mod


@pytest.fixture
def marker_home(tmp_path, monkeypatch):
    """The update marker resolves into a temp home; nothing touches the real one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _marker_path() -> Path:
    from hermes_cli.update_lock import update_marker_path

    return update_marker_path()


def _read_live():
    from hermes_cli.update_lock import read_live_update

    return read_live_update()


def _quiesced_restart(monkeypatch, *, desktop_owns: bool):
    """The restart phase reduced to its bookkeeping: no units, no gateways, no probes."""
    monkeypatch.setattr(
        "hermes_cli.update_cmd_common.desktop_owns_gateway_lifecycle", lambda: desktop_owns)
    monkeypatch.setattr(fleet_mod, "_restart_systemd_gateway_units", lambda *a, **k: None)
    monkeypatch.setattr(fleet_mod, "_restart_manual_gateways", lambda *a, **k: None)
    monkeypatch.setattr(fleet_mod, "_warn_incomplete_gateway_fleet_restart", lambda *a, **k: None)
    monkeypatch.setattr(fleet_mod, "_force_kill_stuck_gateways", lambda *a, **k: None)
    monkeypatch.setattr(fleet_mod, "_gateway_drain_budget", lambda: 1.0)
    monkeypatch.setattr(
        "hermes_cli.update_host_obligation.mark_host_restart_completed", lambda sha: None)
    import hermes_cli.gateway as gateway
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda **k: [])
    monkeypatch.setattr(fleet_mod, "_scoped_manual_gateway_pids", lambda pids, **k: [])


def test_restart_phase_holds_the_marker_while_a_desktop_owns_the_lifecycle(marker_home, monkeypatch):
    """The reported window: parent updater gone, Desktop respawn racing the bounce.

    With a live Desktop supervising the control plane, the restart phase must
    re-hold the update marker (owned by the bouncing process) so the Desktop's
    gate parks instead of aborting into its quitting sentinel.
    """
    _quiesced_restart(monkeypatch, desktop_owns=True)
    # The parent updater already exited: the marker it wrote names a dead pid.
    _marker_path().write_text("4294967294\n1\n", encoding="utf-8")
    assert _read_live() is None, "precondition: dead-owner marker reads as no live update"

    out = fleet_mod._restart_gateway_fleet_after_update(None, gateway_mode=False)

    assert out.held_marker_for_desktop is True
    holder = _read_live()
    assert holder is not None, "the Desktop gate must see a live update while the fleet bounces"
    assert holder.pid == os.getpid(), "the hold is owned by the process bouncing the units"


def test_restart_phase_holds_nothing_without_a_live_desktop(marker_home, monkeypatch):
    """No Desktop, no hold: an ordinary headless update never re-arms the marker."""
    _quiesced_restart(monkeypatch, desktop_owns=False)
    assert not _marker_path().exists()

    out = fleet_mod._restart_gateway_fleet_after_update(None, gateway_mode=False)

    assert out.held_marker_for_desktop is False
    assert not _marker_path().exists()


def test_verify_releases_the_hold_on_success(marker_home, monkeypatch, tmp_path):
    """Verification done, fleet healthy: the gate opens the moment the tail settles."""
    _quiesced_restart(monkeypatch, desktop_owns=True)
    out = fleet_mod._restart_gateway_fleet_after_update(None, gateway_mode=False)
    assert _read_live() is not None, "precondition: the tail is holding the marker"

    monkeypatch.setattr(fleet_mod, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr("hermes_cli.update_cmd_maint._refresh_dashboard_after_update", lambda *a, **k: False)
    monkeypatch.setattr("hermes_cli.update_cmd._surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr("hermes_cli.update_cmd._m", lambda: type("M", (), {
        "_fleet_probe_expected_runtimes": staticmethod(lambda *a, **k: False),
    })(), raising=False)
    monkeypatch.setattr(fleet_mod, "_clear_fleet_restart_pending_marker", lambda: None)
    monkeypatch.setattr("hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: None)

    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        fleet_mod._verify_fleet_after_update(
            out, _pre_update_plan=None, _windows_gateway_resume=None, update_complete=True)

    assert _read_live() is None, "the hold is released when verification settles"
    assert not _marker_path().exists()


def test_verify_releases_the_hold_even_when_the_fleet_is_stale(marker_home, monkeypatch):
    """Exit 1 with a stale fleet still drops the hold: the armed host obligation
    record (which the Desktop gate also reads) is what carries the failure, not
    a stranded marker whose age ceiling would park the Desktop for 20 minutes."""
    _quiesced_restart(monkeypatch, desktop_owns=True)
    out = fleet_mod._restart_gateway_fleet_after_update(None, gateway_mode=False)
    out.incomplete = True  # a stale row / failed unit settled the verdict before verify
    assert _read_live() is not None

    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(SystemExit) as exc:
        fleet_mod._verify_fleet_after_update(
            out, _pre_update_plan=None, _windows_gateway_resume=None, update_complete=True)

    assert exc.value.code == 1
    assert _read_live() is None, "failure releases the hold too — the obligation record stays armed"


def test_abort_recovery_releases_the_hold_when_verification_never_runs(marker_home, monkeypatch):
    """A restart-phase exception means verification may never run in this process;
    abort recovery must not leave the Desktop parked on a re-held marker."""
    _quiesced_restart(monkeypatch, desktop_owns=True)
    out = fleet_mod._restart_gateway_fleet_after_update(None, gateway_mode=False)
    assert _read_live() is not None

    # Drive the real abort recovery with a minimal environment: it must not raise.
    monkeypatch.setattr(fleet_mod, "_surviving_gateway_pids_after_failed_restart", lambda: None)
    monkeypatch.setattr("hermes_cli.update_abort_recovery._owed_stale_serve_rows", lambda rows: [])
    monkeypatch.setattr("hermes_cli.update_cmd._surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr("hermes_cli.update_cmd._abort_recovery_is_complete", lambda **k: False)
    monkeypatch.setattr("hermes_cli.update_cmd._recover_gateway_restart_after_abort", lambda *a, **k: {})
    monkeypatch.setattr("hermes_cli.update_cmd._warn_stale_serve_runtimes", lambda rows: None)

    fleet_mod._recover_after_restart_phase_abort(
        RuntimeError("restart phase exploded"), None, out, gateway_mode=False, restarted_scoped_units=set())

    assert _read_live() is None, "abort recovery releases the tail's marker hold"
