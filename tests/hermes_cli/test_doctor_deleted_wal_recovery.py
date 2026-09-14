"""Tests for hermes doctor in-product recovery of deleted-WAL sidecar holders and retired generations (#110054)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.doctor_report import Finding
import hermes_cli.doctor_state as doctor_state
from hermes_state_errors import DeletedWalGenerationError


def test_doctor_flags_deleted_wal_holders_dry_run(tmp_path, monkeypatch):
    """When processes hold unlinked/retired WAL sidecars, doctor flags them and suggests --fix."""
    db = tmp_path / "state.db"
    db.touch()

    # Mock 2 processes holding deleted sidecars
    mock_holders = [(12345, str(tmp_path / "state.db-wal (deleted)")), (67890, str(tmp_path / "state.db-shm (deleted)"))]
    monkeypatch.setattr(doctor_state, "iter_deleted_sqlite_sidecar_holders", lambda _p: mock_holders, raising=False)
    monkeypatch.setattr("hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders", lambda _p: mock_holders)

    finding = Finding()
    handled = doctor_state._recover_retired_wal(finding, should_fix=False, state_db_path=db, _DHH="~/.hermes")

    assert handled is True
    assert finding.fixed == 0
    assert len(finding.issues) == 1
    issue = finding.issues[0]
    assert "holding a retired WAL generation" in issue
    assert "12345" in issue and "67890" in issue
    assert "doctor --fix" in issue


def test_doctor_fix_stops_holders_and_reopens(tmp_path, monkeypatch):
    """With --fix, doctor stops the holding processes and reopens state.db cleanly."""
    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, title TEXT)")
    conn.execute("INSERT INTO sessions VALUES ('s1', 'title')")
    conn.commit()
    conn.close()

    # Create durable capture artifact satisfying precondition 2
    capture_dir = tmp_path / "state.db.retired-wal-20260913-11111"
    capture_dir.mkdir()
    (capture_dir / "manifest.json").write_text(json.dumps({"pid": 11111, "main": {"mode": "copied"}}), encoding="utf-8")

    mock_holders = [(11111, str(tmp_path / "state.db-wal (deleted)"))]
    calls = []
    dead_pids = set()

    def fake_terminate(pid, force=False, **kwargs):
        calls.append((pid, force, kwargs.get("expected_start_time")))
        dead_pids.add(pid)
        state["holders"].clear()

    # After termination, holders become empty
    state = {"holders": list(mock_holders)}
    def fake_iter_holders(_p):
        return state["holders"]

    def fake_kill(pid, sig):
        if pid in dead_pids:
            raise ProcessLookupError("No such process")
        return None

    monkeypatch.setattr("gateway.status.terminate_pid", fake_terminate)
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: 12345 if pid not in dead_pids else None)
    monkeypatch.setattr("gateway.status._start_times_agree", lambda cur, exp: cur == exp)
    monkeypatch.setattr(doctor_state, "iter_deleted_sqlite_sidecar_holders", fake_iter_holders, raising=False)
    monkeypatch.setattr("hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders", fake_iter_holders)
    monkeypatch.setattr("hermes_state_holders._read_proc_argv", lambda _p: ["hermes", "gateway", "run"])
    monkeypatch.setattr("os.kill", fake_kill)

    finding = Finding()
    handled = doctor_state._recover_retired_wal(finding, should_fix=True, state_db_path=db, _DHH="~/.hermes")

    assert handled is True
    assert finding.fixed == 1
    assert len(calls) >= 1
    assert calls[0][0] == 11111
    assert calls[0][2] == 12345  # expected_start_time passed
    assert len(finding.manual_issues) == 0


def test_doctor_fix_reports_manual_issue_when_stop_fails(tmp_path, monkeypatch):
    """When a holder cannot be stopped (permission or zombie), doctor records it with PID and cmdline."""
    db = tmp_path / "state.db"
    db.touch()

    capture_dir = tmp_path / "state.db.retired-wal-20260913-99999"
    capture_dir.mkdir()
    (capture_dir / "manifest.json").write_text(json.dumps({"pid": 99999, "main": {"mode": "copied"}}), encoding="utf-8")

    mock_holders = [(99999, str(tmp_path / "state.db-wal (deleted)"))]

    def fake_terminate_fail(pid, **kwargs):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr("gateway.status.terminate_pid", fake_terminate_fail)
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: 54321)
    monkeypatch.setattr("gateway.status._start_times_agree", lambda cur, exp: cur == exp)
    monkeypatch.setattr(doctor_state, "iter_deleted_sqlite_sidecar_holders", lambda _p: mock_holders, raising=False)
    monkeypatch.setattr("hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders", lambda _p: mock_holders)
    monkeypatch.setattr("hermes_state_holders._read_proc_argv", lambda _p: ["/usr/bin/python", "daemon.py"])

    finding = Finding()
    handled = doctor_state._recover_retired_wal(finding, should_fix=True, state_db_path=db, _DHH="~/.hermes")

    assert handled is True
    assert finding.fixed == 0
    assert len(finding.manual_issues) == 1
    manual = finding.manual_issues[0]
    assert "99999" in manual
    assert "daemon.py" in manual
    assert "stop them manually" in manual


def test_doctor_identity_unavailable_fails_closed(tmp_path, monkeypatch):
    """When start-time fingerprint cannot be established, doctor refuses to signal and reports manual issue."""
    db = tmp_path / "state.db"
    db.touch()

    capture_dir = tmp_path / "state.db.retired-wal-20260913-77777"
    capture_dir.mkdir()
    (capture_dir / "manifest.json").write_text(json.dumps({"pid": 77777, "main": {"mode": "copied"}}), encoding="utf-8")

    mock_holders = [(77777, str(tmp_path / "state.db-wal (deleted)"))]
    signals_sent = []

    monkeypatch.setattr("gateway.status.terminate_pid", lambda pid, **kw: signals_sent.append(pid))
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: None)  # Unavailable
    monkeypatch.setattr("hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders", lambda _p: mock_holders)
    monkeypatch.setattr("hermes_state_holders._read_proc_argv", lambda _p: ["worker.py"])

    finding = Finding()
    handled = doctor_state._recover_retired_wal(finding, should_fix=True, state_db_path=db, _DHH="~/.hermes")

    assert handled is True
    assert len(signals_sent) == 0  # No signal sent
    assert finding.fixed == 0
    assert any("start-time identity unavailable" in issue for issue in finding.manual_issues)


def test_doctor_unlinked_wal_no_capture_fails_closed(tmp_path, monkeypatch):
    """When an unlinked WAL generation has no durable capture, doctor refuses to signal the owner."""
    db = tmp_path / "state.db"
    db.touch()

    mock_holders = [(88888, str(tmp_path / "state.db-wal (deleted)"))]
    signals_sent = []

    monkeypatch.setattr("gateway.status.terminate_pid", lambda pid, **kw: signals_sent.append(pid))
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: 8888)
    monkeypatch.setattr("hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders", lambda _p: mock_holders)
    monkeypatch.setattr("hermes_state_dbfile.iter_deleted_sqlite_sidecar_holder_descriptors", lambda _p: [
        {"pid": 88888, "target": str(tmp_path / "state.db-wal (deleted)"), "fd_path": "/proc/88888/fd/3",
         "identity": (1, 2), "suffix": "-wal"}
    ])
    monkeypatch.setattr("hermes_state_holders._read_proc_argv", lambda _p: ["worker.py"])

    finding = Finding()
    handled = doctor_state._recover_retired_wal(finding, should_fix=True, state_db_path=db, _DHH="~/.hermes")

    assert handled is True
    assert len(signals_sent) == 0  # Refused to signal because no capture exists
    assert finding.fixed == 0
    assert any("has no durable capture" in issue for issue in finding.manual_issues)


def test_doctor_holder_exits_before_term_no_signal_to_replacement(tmp_path, monkeypatch):
    """When a holder exits before TERM, doctor detects identity drift and sends no signal to replacement."""
    db = tmp_path / "state.db"
    db.touch()

    capture_dir = tmp_path / "state.db.retired-wal-20260913-66666"
    capture_dir.mkdir()
    (capture_dir / "manifest.json").write_text(json.dumps({"pid": 66666, "main": {"mode": "copied"}}), encoding="utf-8")

    mock_holders = [(66666, str(tmp_path / "state.db-wal (deleted)"))]
    signals_sent = []

    # Witness sees start time 100, but immediately before TERM the process exited and start time is None
    start_times = {66666: [100, None]}
    def fake_start_time(pid):
        vals = start_times.get(pid, [100])
        return vals.pop(0) if len(vals) > 1 else vals[0]

    monkeypatch.setattr("gateway.status.terminate_pid", lambda pid, **kw: signals_sent.append(pid))
    monkeypatch.setattr("gateway.status.get_process_start_time", fake_start_time)
    monkeypatch.setattr("gateway.status._start_times_agree", lambda cur, exp: cur == exp)
    monkeypatch.setattr("hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders", lambda _p: mock_holders)
    monkeypatch.setattr("hermes_state_holders._read_proc_argv", lambda _p: ["worker.py"])

    finding = Finding()
    handled = doctor_state._recover_retired_wal(finding, should_fix=True, state_db_path=db, _DHH="~/.hermes")

    assert handled is True
    assert len(signals_sent) == 0  # No signal sent to replacement!


def test_doctor_pid_recycled_between_term_and_kill_no_kill(tmp_path, monkeypatch):
    """When PID is recycled between TERM and KILL, doctor detects identity mismatch and refuses SIGKILL."""
    db = tmp_path / "state.db"
    db.touch()

    capture_dir = tmp_path / "state.db.retired-wal-20260913-44444"
    capture_dir.mkdir()
    (capture_dir / "manifest.json").write_text(json.dumps({"pid": 44444, "main": {"mode": "copied"}}), encoding="utf-8")

    mock_holders = [(44444, str(tmp_path / "state.db-wal (deleted)"))]
    signals_sent = []

    # Start time was 200 during witness & TERM, but before KILL PID was recycled to a process with start time 999
    call_count = {"val": 0}
    def fake_start_time(pid):
        call_count["val"] += 1
        if call_count["val"] <= 2:
            return 200
        return 999  # Recycled PID

    def fake_terminate(pid, force=False, **kw):
        signals_sent.append((pid, force))

    monkeypatch.setattr("gateway.status.terminate_pid", fake_terminate)
    monkeypatch.setattr("gateway.status.get_process_start_time", fake_start_time)
    monkeypatch.setattr("gateway.status._start_times_agree", lambda cur, exp: cur == exp)
    monkeypatch.setattr("hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders", lambda _p: mock_holders)
    monkeypatch.setattr("hermes_state_holders._read_proc_argv", lambda _p: ["worker.py"])
    monkeypatch.setattr("os.kill", lambda pid, sig: None)  # Reports process still alive

    finding = Finding()
    handled = doctor_state._recover_retired_wal(finding, should_fix=True, state_db_path=db, _DHH="~/.hermes")

    assert handled is True
    # TERM was sent (force=False), but force-kill (force=True) was REFUSED because start time changed to 999!
    assert any(sig[1] is False for sig in signals_sent)
    assert not any(sig[1] is True for sig in signals_sent)


def test_doctor_header_only_mode_guidance(tmp_path, monkeypatch):
    """When retired WAL capture is header_only, guidance states forensic only and does not advertise nonexistent state.db."""
    db = tmp_path / "state.db"
    db.touch()

    artifact_dir = tmp_path / "state.db.retired-wal-20260913T120000Z-9999"
    artifact_dir.mkdir()
    manifest = {
        "manifest_version": 1,
        "main": {"mode": "header_only", "file": "state.db.header", "bytes": 100},
        "wal": {"file": "state.db-wal", "bytes": 512},
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr("hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders", lambda _p: [])

    finding = Finding()
    infos = []
    monkeypatch.setattr(doctor_state, "check_info", lambda text: infos.append(text))

    doctor_state._recover_retired_wal(finding, should_fix=False, state_db_path=db, _DHH="~/.hermes")

    # Verify guidance states forensic artifact only and references manifest.json, NOT sessions recover state.db
    assert any("mode: header_only" in msg for msg in infos)
    assert any("forensic artifact only" in msg for msg in infos)
    assert not any("sessions recover --source" in msg for msg in infos)


def test_state_db_health_catches_deleted_wal_error(tmp_path, monkeypatch):
    """When _session_count raises DeletedWalGenerationError, health check delegates to recovery."""
    db = tmp_path / "state.db"
    db.touch()

    def fake_session_count(_p):
        raise DeletedWalGenerationError("FATAL: a live process holds a deleted state.db-wal")

    monkeypatch.setattr(doctor_state, "_session_count", fake_session_count)
    mock_holders = [(5555, str(tmp_path / "state.db-wal (deleted)"))]
    monkeypatch.setattr("hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders", lambda _p: mock_holders)

    finding = Finding()
    doctor_state._state_db_health(finding, should_fix=False, state_db_path=db, _DHH="~/.hermes")

    assert len(finding.issues) == 1
    assert "5555" in finding.issues[0]
    assert "doctor --fix" in finding.issues[0]


def test_doctor_surfaces_retired_wal_capture_artifacts(tmp_path, monkeypatch):
    """When retired-wal captures exist beside state.db, doctor inspects and reports them."""
    db = tmp_path / "state.db"
    db.touch()

    # Create a retired-wal artifact directory
    artifact_dir = tmp_path / "state.db.retired-wal-20260913T120000Z-4321"
    artifact_dir.mkdir()
    manifest = {
        "manifest_version": 1,
        "main": {"mode": "copied", "bytes": 1024},
        "wal": {"file": "state.db-wal", "bytes": 512},
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr("hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders", lambda _p: [])

    finding = Finding()
    infos = []
    monkeypatch.setattr(doctor_state, "check_info", lambda text: infos.append(text))

    doctor_state._recover_retired_wal(finding, should_fix=False, state_db_path=db, _DHH="~/.hermes")

    assert any("state.db.retired-wal-" in msg for msg in infos)
    assert any("mode: copied" in msg for msg in infos)
    assert any("sessions recover" in msg for msg in infos)


@pytest.mark.asyncio
async def test_ops_run_doctor_supports_fix_param(monkeypatch):
    """POST /api/ops/doctor forwards --fix to _spawn_action when fix=True."""
    from hermes_cli.web_models import DoctorRequest
    from hermes_cli.web_routers.ops import run_doctor

    spawned = []
    monkeypatch.setattr("hermes_cli.web_routers.ops._spawn_action", lambda args, name, **kwargs: spawned.append((args, name)))

    # Without fix
    await run_doctor(None)
    assert spawned[-1] == (["doctor"], "doctor")

    await run_doctor(DoctorRequest(fix=False))
    assert spawned[-1] == (["doctor"], "doctor")

    # With fix=True
    await run_doctor(DoctorRequest(fix=True))
    assert spawned[-1] == (["doctor", "--fix"], "doctor")


def test_doctor_excludes_current_pid_and_reports_pending_spool(tmp_path, monkeypatch):
    """Doctor never attempts to terminate its own PID, and reports pending spool files upon reopening."""
    import os
    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

    spool = tmp_path / "pending_messages"
    spool.mkdir()
    (spool / "pending-123.json").write_text("{}", encoding="utf-8")

    capture_dir = tmp_path / "state.db.retired-wal-20260913-22222"
    capture_dir.mkdir()
    (capture_dir / "manifest.json").write_text(json.dumps({"pid": 22222, "main": {"mode": "copied"}}), encoding="utf-8")

    current_pid = os.getpid()
    mock_holders = [
        (current_pid, str(tmp_path / "state.db-wal (deleted)")),
        (22222, str(tmp_path / "state.db-wal (deleted)")),
    ]

    terminated = []
    dead_pids = set()

    def fake_terminate(pid, force=False, **kwargs):
        terminated.append((pid, force))
        dead_pids.add(pid)
        state["holders"] = [(current_pid, str(tmp_path / "state.db-wal (deleted)"))]

    state = {"holders": list(mock_holders)}
    def fake_iter_holders(_p):
        return [h for h in state["holders"] if h[0] not in dead_pids]

    def fake_kill(pid, sig):
        if pid in dead_pids:
            raise ProcessLookupError("No such process")
        return None

    monkeypatch.setattr("gateway.status.terminate_pid", fake_terminate)
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: 2222)
    monkeypatch.setattr("gateway.status._start_times_agree", lambda cur, exp: cur == exp)
    monkeypatch.setattr("hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders", fake_iter_holders)
    monkeypatch.setattr("hermes_state_holders._read_proc_argv", lambda _p: ["hermes", "worker"])
    monkeypatch.setattr("os.kill", fake_kill)

    infos = []
    monkeypatch.setattr(doctor_state, "check_info", lambda text: infos.append(text))

    finding = Finding()
    handled = doctor_state._recover_retired_wal(finding, should_fix=True, state_db_path=db, _DHH="~/.hermes")

    assert handled is True
    # current_pid was excluded from termination
    assert all(pid != current_pid for pid, _ in terminated)
    assert 22222 in [pid for pid, _ in terminated]
    # Spool file was surfaced
    assert any("pending message spool file(s) preserved" in msg for msg in infos)


def test_corrupt_store_as_status_handles_deleted_wal_and_replaced_errors(tmp_path):
    """corrupt_store_as_status maps DeletedWalGenerationError and StateDbReplacedError to 503 HTTPException."""
    from fastapi import HTTPException
    from hermes_cli.web_routers._common import corrupt_store_as_status
    from hermes_state_errors import DeletedWalGenerationError, StateDbReplacedError

    db = tmp_path / "state.db"

    # Test DeletedWalGenerationError -> 503 with error: deleted_wal
    with pytest.raises(HTTPException) as exc_info:
        with corrupt_store_as_status(db):
            raise DeletedWalGenerationError("deleted wal generation")
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "deleted_wal"
    assert "doctor --fix" in exc_info.value.detail["message"]

    # Test StateDbReplacedError -> 503 with error: state_db_replaced
    with pytest.raises(HTTPException) as exc_info:
        with corrupt_store_as_status(db):
            raise StateDbReplacedError("database replaced")
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "state_db_replaced"
    assert "doctor --fix" in exc_info.value.detail["message"]
