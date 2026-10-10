"""``hermes doctor`` must catch a gateway that is serving dispatch code older than the file on disk.

Card t_50d090c0: the review leg verifies behaviour on an isolated scratch board in a fresh process, so
it passes while production runs the code it imported at boot. Doctor is the surface that judges the
LIVE gateway's own boot snapshot.

Every case here runs the real check (``_check_dispatch_freshness``) against a real
``gateway_state.json`` written into an isolated HERMES_HOME, and reads its actual stdout.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from gateway import dispatch_freshness as df
from gateway import status as gw_status
from hermes_cli import doctor_dispatch
from hermes_cli.doctor import DOCTOR_CHECKS


def _record(path: Path, **fields) -> dict:
    """A live-gateway runtime record: this process's own PID, so the liveness guard passes for real."""
    payload = {
        "pid": os.getpid(),
        "kind": "hermes-gateway",
        "start_time": gw_status._get_process_start_time(os.getpid()),
        "gateway_state": "running",
        "hermes_home": str(path.parent),
    }
    payload.update(fields)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    """One reachable runtime record for the check to judge."""
    path = tmp_path / "gateway_state.json"
    monkeypatch.setattr(doctor_dispatch, "_runtime_candidates", lambda: [("test home", path)])
    return path


def test_check_is_registered_in_the_doctor_run():
    assert ("Kanban Dispatch Plane", doctor_dispatch._check_dispatch_freshness) in DOCTOR_CHECKS


def test_stale_dispatch_code_warns_with_the_restart_command(state_file, capsys):
    """Criterion 1: code newer than the gateway's boot snapshot must be a loud, actionable warning."""
    _record(state_file, dispatch_code_max_mtime_ns=1, dispatch_code_files=9,
            dispatch_code_newest="hermes_cli/kanban_db_dispatch.py")

    finding = doctor_dispatch._check_dispatch_freshness(False)

    out = capsys.readouterr().out
    assert "Dispatch code on disk is newer than what gateway PID" in out
    assert "hermes gateway restart" in out
    # The watched-file count is the tree's, not a constant: the fleet's dispatch plane carries two
    # extra local modules (kanban_goal_policy.py, the Jev router script) at full strength, and this
    # assertion must not pin a number the next update can move. Assert what the check OWES: the whole
    # count, reported, with the file list truncated at 3.
    _watched = len(df.watched_paths())
    assert f"{_watched} of {_watched} watched file(s)" in out
    assert f"+{_watched - 3} more" in out, "the file list is truncated, not dumped into doctor output"
    assert len(finding.manual_issues) == 1 and finding.issues == []
    assert "Stale kanban dispatch plane" in finding.manual_issues[0]
    assert "hermes gateway restart" in finding.manual_issues[0]
    assert "kanban.dispatch_auto_reload" in finding.manual_issues[0]


def test_clean_boot_reports_no_warning(state_file, capsys):
    """Criterion 2 (doctor half): a gateway started after the last edit must not warn."""
    _record(state_file, dispatch_code_max_mtime_ns=int(time.time() * 1_000_000_000),
            dispatch_code_files=9, dispatch_code_newest="hermes_cli/kanban_db_dispatch.py")

    finding = doctor_dispatch._check_dispatch_freshness(False)

    out = capsys.readouterr().out
    assert "✓" in out and "matches gateway PID" in out
    assert "Dispatch code on disk is newer" not in out
    assert finding.issues == [] and finding.manual_issues == []


def test_stamp_wins_over_the_process_start_time(state_file, capsys):
    """The gateway's own snapshot is authoritative; the derived start time is only a fallback.

    This record's live ``start_time`` is *after* every watched file, so the fallback would read
    "fresh" — the stale verdict can only come from the stamp, which is the point.
    """
    _record(state_file, dispatch_code_max_mtime_ns=1)

    finding = doctor_dispatch._check_dispatch_freshness(False)

    out = capsys.readouterr().out
    assert "Dispatch code on disk is newer" in out
    assert "against the boot snapshot the gateway stamped at startup" in out
    assert "derived from the gateway process start time" not in out
    assert len(finding.manual_issues) == 1 and finding.issues == []


def test_record_without_a_boot_instant_is_reported_as_unknown(state_file, capsys):
    """No stamp and no readable start time: the check says so and warns about nothing."""
    _record(state_file, start_time=None)

    finding = doctor_dispatch._check_dispatch_freshness(False)

    out = capsys.readouterr().out
    assert "freshness not judged" in out
    assert "Dispatch code on disk is newer" not in out
    assert finding.issues == [] and finding.manual_issues == []


def test_record_with_only_a_start_time_falls_back_to_it(state_file, capsys):
    """A record written before this check existed is still judged, one step coarser."""
    _record(state_file)  # live pid, live start time, no stamp

    finding = doctor_dispatch._check_dispatch_freshness(False)

    assert "matches gateway PID" in capsys.readouterr().out
    assert finding.issues == [] and finding.manual_issues == []


def test_dead_gateway_record_is_ignored(state_file, capsys):
    _record(state_file, pid=99_999_999, dispatch_code_max_mtime_ns=1)

    finding = doctor_dispatch._check_dispatch_freshness(False)

    out = capsys.readouterr().out
    assert "No live gateway" in out
    assert "Dispatch code on disk is newer" not in out
    assert finding.issues == [] and finding.manual_issues == []


def test_stopped_gateway_record_is_ignored(state_file, capsys):
    _record(state_file, gateway_state="stopped", dispatch_code_max_mtime_ns=1)

    finding = doctor_dispatch._check_dispatch_freshness(False)

    out = capsys.readouterr().out
    assert "No live gateway" in out
    assert "Dispatch code on disk is newer" not in out
    assert finding.issues == [] and finding.manual_issues == []


def test_missing_state_file_is_silent_about_freshness(state_file, capsys):
    finding = doctor_dispatch._check_dispatch_freshness(False)

    out = capsys.readouterr().out
    assert "Dispatch code on disk is newer" not in out
    assert "No live gateway" in out
    assert finding.issues == [] and finding.manual_issues == []


def test_runtime_candidates_cover_the_host_root_for_a_served_profile(tmp_path, monkeypatch):
    """Multiplex-only: a served profile writes no record of its own, so the host root's must be read."""
    from hermes_constants import get_default_hermes_root

    home = tmp_path / "profiles" / "hephaestus"
    home.mkdir(parents=True)
    monkeypatch.setattr("hermes_cli.doctor.HERMES_HOME", home)

    candidates = doctor_dispatch._runtime_candidates()

    assert candidates[0] == (str(home), home / "gateway_state.json")
    assert (Path(get_default_hermes_root()) / "gateway_state.json") == candidates[-1][1]
    paths = [str(path) for _label, path in candidates]
    assert len(paths) == len(set(paths)), "the same record must not be judged twice"

