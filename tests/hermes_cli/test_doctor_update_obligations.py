"""``hermes doctor`` names the runtime restarts still owed to their owner (#107402).

The reminder under ``serve_restart_pending/`` is discharged only by the death of the process it
names, so before this check the only surface that listed one was ``hermes update``'s own output.
Read-only: doctor reports the debt and never settles it.
"""

from __future__ import annotations

import json

from hermes_cli import doctor, process_identity
from hermes_cli import doctor_state as ds
from hermes_constants import get_hermes_home


def _reminder(pid: int = 900, create_time: float = 1000.0, **extra):
    directory = get_hermes_home() / "serve_restart_pending"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{pid}-{float(create_time).hex()}.json"
    path.write_text(
        json.dumps({"kind": "serve", "profile": "work", "pid": pid, "create_time": create_time, **extra}),
        encoding="utf-8",
    )
    return path


def test_pending_restart_is_named_with_identity_and_checkout(monkeypatch, capsys):
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: True)
    monkeypatch.setattr(ds, "_update_obligation_sha", lambda: "868b9f24")
    _reminder()

    finding = ds._check_update_obligations(False)

    out = capsys.readouterr().out
    assert "serve [work] pid 900 (started 1970-01-01 00:16 UTC)" in out
    assert "manual restart still pending" in out
    assert "relaunch `hermes serve` / `hermes dashboard`" in out
    assert "868b9f24" in out
    # Informational: the update flow owns the escalation, so a debt that clears itself on the next
    # relaunch must not flip doctor's exit status.
    assert finding.manual_issues == []
    assert finding.issues == []


def test_clean_host_reports_no_obligation(monkeypatch, capsys):
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: True)

    ds._check_update_obligations(False)

    assert "No runtime is owed a manual restart" in capsys.readouterr().out


def test_dead_pid_prunes_the_reminder_rather_than_reporting_it(monkeypatch, capsys):
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: False)
    path = _reminder()

    ds._check_update_obligations(False)

    assert not path.exists()
    assert "No runtime is owed a manual restart" in capsys.readouterr().out


def test_unreadable_reminder_is_reported(monkeypatch, capsys):
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: True)
    directory = get_hermes_home() / "serve_restart_pending"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "9-1.0.json").write_text("{not json", encoding="utf-8")

    ds._check_update_obligations(False)

    out = capsys.readouterr().out
    assert "could not be read" in out
    assert "9-1.0.json" in out


def test_doctor_runs_the_check():
    assert any(check is doctor._check_update_obligations for _title, check in doctor.DOCTOR_CHECKS)
