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


# ---------------------------------------------------------------------------
# #109538: read-only death probe for the update path
# ---------------------------------------------------------------------------


def test_attested_probe_fails_closed_without_a_well_formed_dead_attestation(attest_home):
    """``hermes update`` uses this probe to override Desktop-owned lifecycle suppression
    (#109538), so only a *detectable* death may read True: no marker, malformed ``pids``,
    a gateway alive now, or a clean ledger exit all read False. The probe must also be
    read-only — consuming the marker here would silence the CLI-start warning that reports
    the same death to the user.
    """
    marker = attest_home / "state" / "gateway.start-attestation.json"
    assert gateway_windows.attested_death_generation(current_pids=[]) is None  # no marker yet

    marker.parent.mkdir(exist_ok=True)
    # Exact positive ints only: ``True`` is an int subclass, 0/-1 are not PIDs, and a single
    # malformed item taints the list (the writer never emits such values).
    for malformed in (
        '{"pids": null}', '{"pids": 555}', '["not", "a", "dict"]', "not json",
        '{"pids": [true]}', '{"pids": [0]}', '{"pids": [-1]}', '{"pids": [555, "556"]}', '{"pids": [555, 0]}',
    ):
        marker.write_text(malformed, encoding="utf-8")
        assert gateway_windows.attested_death_generation(current_pids=[]) is None, malformed

    gateway_windows._write_start_attestation([555], "cold-start after update")
    assert gateway_windows.attested_death_generation(current_pids=[555]) is None  # alive
    assert gateway_windows.attested_death_generation(current_pids=[]) is not None  # dead, unclean
    assert marker.exists()  # unconsumed — the CLI start below still reports it
    assert gateway_windows.check_start_attestation(current_pids=[]) is not None

    (marker.parent / "gateway.lifecycle.json").write_text(
        json.dumps({"phase": "exited", "pid": 556, "exit_reason": "graceful_shutdown"}),
        encoding="utf-8",
    )
    gateway_windows._write_start_attestation([556], "cold-start after update")
    assert gateway_windows.attested_death_generation(current_pids=[]) is None  # planned stop


def test_attested_probe_treats_a_marker_past_the_horizon_as_no_authority(attest_home):
    """#110020 review (d): a historical marker must not later override Desktop ownership into a
    duplicate gateway (#76129). Missing/unparsable/old ``ts`` all fail closed."""
    marker = attest_home / "state" / "gateway.start-attestation.json"
    marker.parent.mkdir(exist_ok=True)
    gateway_windows._write_start_attestation([555], "direct spawn (PID 555)")
    assert gateway_windows.attested_death_generation(current_pids=[]) is not None  # fresh
    data = json.loads(marker.read_text(encoding="utf-8"))
    for ts in (None, "not-a-date", "2020-01-01T00:00:00+00:00"):
        stale = {k: v for k, v in data.items() if k != "ts"} if ts is None else {**data, "ts": ts}
        marker.write_text(json.dumps(stale), encoding="utf-8")
        assert gateway_windows.attested_death_generation(current_pids=[]) is None, ts
        assert marker.exists()  # not consumed either


def _sentinel(attest_home, **fields):
    state = attest_home / "state"
    state.mkdir(exist_ok=True)
    (state / "gateway.lifecycle.json").write_text(json.dumps(fields), encoding="utf-8")


def test_attestation_bound_to_create_time_is_no_authority_once_the_sentinel_moved_on(monkeypatch, attest_home):
    """#110020 review (gateway_windows.py:937): the sentinel used to be matched by numeric PID only, so a
    stale marker for PID 111 flipped from clean to crash once an unrelated PID 222 lifecycle overwrote
    the sentinel. A marker bound to 111's process birth fails closed: another PID or another birth
    time reads as undecidable, never as dead. (Birth, not the ledger's ``start_time``: that is stamped
    seconds later, once imports finish.)"""
    monkeypatch.setattr("hermes_cli.process_identity._process_create_time", lambda pid=None: 1000.0)
    gateway_windows._write_start_attestation([111], "direct spawn (PID 111)")
    marker = json.loads((attest_home / "state" / "gateway.start-attestation.json").read_text(encoding="utf-8"))
    assert marker["create_times"] == {"111": 1000.0}
    for fields in (
        {"phase": "exited", "pid": 222, "create_time": 5000.0},  # unrelated lifecycle overwrote it
        {"phase": "running", "pid": 222, "create_time": 5000.0},
        {"phase": "running", "pid": 111, "create_time": 1003.0},  # PID reuse: different incarnation
    ):
        _sentinel(attest_home, **fields)
        assert gateway_windows.attested_death_generation(current_pids=[]) is None, fields
        assert gateway_windows.check_start_attestation(current_pids=[]) is None, fields
        gateway_windows._write_start_attestation([111], "direct spawn (PID 111)")  # consumed above


def test_attestation_bound_to_create_time_keeps_authority_for_its_own_incarnation(monkeypatch, attest_home):
    """A running sentinel for the same PID within 2s of the bound create time is the attested
    incarnation: gone with no clean exit → dead. Its own clean exit (create_time carried by
    ``mark_exited``) → planned stop. A sentinel from a gateway older than the identity stamp, and
    older markers without ``create_times``, keep PID-only matching."""
    monkeypatch.setattr("hermes_cli.process_identity._process_create_time", lambda pid=None: 1000.0)
    gateway_windows._write_start_attestation([111], "direct spawn (PID 111)")
    _sentinel(attest_home, phase="running", pid=111, create_time=1001.5)
    assert gateway_windows.attested_death_generation(current_pids=[]) is not None
    _sentinel(attest_home, phase="exited", pid=111, create_time=1001.5, exit_reason="graceful_shutdown")
    assert gateway_windows.attested_death_generation(current_pids=[]) is None
    # Pre-identity sentinel (no create_time, start_time is the later ledger claim): PID-only rule.
    _sentinel(attest_home, phase="running", pid=111, start_time=1007.0)
    assert gateway_windows.attested_death_generation(current_pids=[]) is not None
    # No sentinel at all: the process never booted far enough to claim one → dead.
    (attest_home / "state" / "gateway.lifecycle.json").unlink()
    assert gateway_windows.attested_death_generation(current_pids=[]) is not None
    # Legacy marker (no create_times): PID-only rule unchanged.
    path = attest_home / "state" / "gateway.start-attestation.json"
    legacy = {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if k != "create_times"}
    path.write_text(json.dumps(legacy), encoding="utf-8")
    _sentinel(attest_home, phase="running", pid=111, create_time=1003.0)
    assert gateway_windows.attested_death_generation(current_pids=[]) is not None
