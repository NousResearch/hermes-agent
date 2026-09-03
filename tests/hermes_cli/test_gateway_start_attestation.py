"""Tests for the #91675 gateway-start honesty fixes.

Two holes closed by the fix:

1. ``_wait_for_gateway_ready`` returned on the FIRST process-table hit, so a
   gateway that spawned and then died moments later (parent Job Object
   teardown) still earned a ✓.  The poll now requires the gateway to stay
   visible for a confirmation window before it is reported ready.
2. A death AFTER the CLI process exits can never be seen by any poll.  Every
   ✓ now persists a start-attestation marker; the next CLI invocation checks
   it and reports the silent death (once) unless the lifecycle ledger shows
   a clean exit.

All timing knobs are shrunk so no test sleeps longer than ~1s.
"""

import json

import pytest

import hermes_cli.gateway_windows as gateway_windows


# ---------------------------------------------------------------------------
# _wait_for_gateway_ready: confirmation window
# ---------------------------------------------------------------------------


def _install_pid_sequence(monkeypatch, snapshots):
    """find_gateway_pids returns successive snapshots (last one repeats)."""
    calls = {"n": 0}

    def _fake(*args, **kwargs):
        idx = min(calls["n"], len(snapshots) - 1)
        calls["n"] += 1
        return list(snapshots[idx])

    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", _fake)
    return calls


def test_ready_poll_rejects_gateway_that_dies_during_confirmation(monkeypatch):
    """First-hit-then-dead must NOT be reported ready (#91675 sabotage case).

    Pre-fix, the poll returned ``[4242]`` on the first snapshot and the CLI
    printed ✓ for a process that was already doomed.
    """
    _install_pid_sequence(monkeypatch, [[4242], [], [], []])
    monkeypatch.setattr(gateway_windows.time, "sleep", lambda s: None)

    pids = gateway_windows._wait_for_gateway_ready(
        timeout_s=0.5, interval_s=0.01, confirm_s=0.2
    )
    assert pids == []


def test_ready_poll_confirms_stable_gateway(monkeypatch):
    """A gateway that stays visible through the confirmation window is ready."""
    _install_pid_sequence(monkeypatch, [[4242]])
    monkeypatch.setattr(gateway_windows.time, "sleep", lambda s: None)

    pids = gateway_windows._wait_for_gateway_ready(
        timeout_s=0.5, interval_s=0.01, confirm_s=0.05
    )
    assert pids == [4242]


def test_ready_poll_recovers_when_gateway_respawns_within_deadline(monkeypatch):
    """Death during confirmation resumes polling; a later stable gateway wins."""
    # hit → dead (confirmation fails) → nothing → new stable pid
    _install_pid_sequence(monkeypatch, [[1], [], [], [2], [2], [2]])
    monkeypatch.setattr(gateway_windows.time, "sleep", lambda s: None)

    pids = gateway_windows._wait_for_gateway_ready(
        timeout_s=1.0, interval_s=0.01, confirm_s=0.03
    )
    assert pids == [2]


def test_report_gateway_start_failure_is_loud_not_checkmark(monkeypatch, tmp_path, capsys):
    """No stable gateway ⇒ ✗ failure line, never ✓ (#91675)."""
    monkeypatch.setattr(
        gateway_windows, "_wait_for_gateway_ready", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "hermes_cli.config.get_hermes_home", lambda: str(tmp_path)
    )
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_x")

    gateway_windows._report_gateway_start("direct spawn (PID 7)")
    out = capsys.readouterr().out
    assert "✓" not in out
    assert "FAILED" in out
    assert "schtasks /Run /TN Hermes_Gateway_x" in out


# ---------------------------------------------------------------------------
# Start attestation: report-async-death on the next CLI invocation
# ---------------------------------------------------------------------------


@pytest.fixture
def attest_home(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(tmp_path))
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)
    return tmp_path


def test_success_report_writes_attestation(monkeypatch, attest_home, capsys):
    monkeypatch.setattr(
        gateway_windows, "_wait_for_gateway_ready", lambda *a, **k: [321]
    )
    gateway_windows._LAST_SPAWN_BREAKAWAY_FALLBACK["fallback"] = False
    gateway_windows._report_gateway_start("direct spawn (PID 321)")
    assert "✓" in capsys.readouterr().out

    marker = attest_home / "state" / "gateway.start-attestation.json"
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["pids"] == [321]
    assert data["via"] == "direct spawn (PID 321)"


def test_attestation_reports_silent_death(attest_home):
    """Attested PIDs gone + no clean-exit record ⇒ warning, marker consumed."""
    gateway_windows._write_start_attestation([555], "direct spawn (PID 555)")

    warning = gateway_windows.check_start_attestation(current_pids=[])
    assert warning is not None
    assert "died without a clean shutdown record" in warning
    assert "555" in warning
    # Consumed: second check is silent.
    assert gateway_windows.check_start_attestation(current_pids=[]) is None


def test_attestation_silent_when_gateway_running(attest_home):
    gateway_windows._write_start_attestation([555], "direct spawn (PID 555)")
    assert gateway_windows.check_start_attestation(current_pids=[555]) is None
    # Marker cleared — a later dead scan must not resurrect the warning.
    assert gateway_windows.check_start_attestation(current_pids=[]) is None


def test_attestation_silent_after_clean_ledger_exit(attest_home):
    """A clean lifecycle-ledger exit for the attested PID is a planned stop."""
    gateway_windows._write_start_attestation([777], "direct spawn (PID 777)")
    state = attest_home / "state"
    state.mkdir(exist_ok=True)
    (state / "gateway.lifecycle.json").write_text(
        json.dumps({"phase": "exited", "pid": 777, "exit_reason": "graceful_shutdown"}),
        encoding="utf-8",
    )
    assert gateway_windows.check_start_attestation(current_pids=[]) is None


def test_attestation_warning_includes_schtasks_recovery(monkeypatch, attest_home):
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(
        gateway_windows, "get_task_name", lambda: "Hermes_Gateway_arthur_tutor"
    )
    gateway_windows._write_start_attestation([888], "direct spawn (PID 888)")
    warning = gateway_windows.check_start_attestation(current_pids=[])
    assert "schtasks /Run /TN Hermes_Gateway_arthur_tutor" in warning


def test_attestation_tolerates_missing_and_garbage_marker(attest_home):
    assert gateway_windows.check_start_attestation(current_pids=[]) is None
    marker = attest_home / "state" / "gateway.start-attestation.json"
    marker.parent.mkdir(exist_ok=True)
    marker.write_text("not json", encoding="utf-8")
    assert gateway_windows.check_start_attestation(current_pids=[]) is None
    marker.write_text(json.dumps({"pids": []}), encoding="utf-8")
    assert gateway_windows.check_start_attestation(current_pids=[]) is None
    assert not marker.exists()


def test_breakaway_fallback_warns_even_on_success(monkeypatch, attest_home, capsys):
    """When the spawn fell back to no-breakaway, the ✓ carries a Job warning."""
    monkeypatch.setattr(
        gateway_windows, "_wait_for_gateway_ready", lambda *a, **k: [99]
    )
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway")
    gateway_windows._LAST_SPAWN_BREAKAWAY_FALLBACK["fallback"] = True
    try:
        gateway_windows._report_gateway_start("direct spawn (PID 99)")
    finally:
        gateway_windows._LAST_SPAWN_BREAKAWAY_FALLBACK["fallback"] = False
    out = capsys.readouterr().out
    assert "✓" in out
    assert "could not break away" in out
    assert "schtasks /Run /TN Hermes_Gateway" in out
