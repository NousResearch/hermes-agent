"""§3.3 idempotency guarantees that need more than one module's fixtures (t_011da190).

Extends ``tests/hermes_cli/test_update_restart_orchestration.py`` (fixture-level A1–A7, landed by
``t_5d0ba21a``). Those cover one module's seams in isolation; the guarantees below only hold across
the update phases, so they are exercised the way ``update_cmd.py`` actually runs them —
``_restart_gateway_fleet_after_update`` (S0–S4) then ``_verify_fleet_after_update`` (S5–S8).

Design: ``build/gateway_restart/POST-UPDATE-GATEWAY-RESTART-DESIGN.md`` §3.1 (S0–S8), §3.3
(idempotency), §4. Implementation record: ``build/gateway_restart/IMPLEMENTATION-t_5d0ba21a.md``.

* §3.3 #4 — an interruption between S3 and S7 keeps the obligation armed and the receipt
  ``partial``; the next run resumes for the RECORDED runtime set only, and a bounded retry is armed.
* §3.3 #3 — a second updater arriving AFTER the first finished takes the lease and completes as a
  no-op under the same ``(sha, runtime-identity)`` key: zero further signals.
* Ordering — exactly ONE verdict, taken from the fleet-matrix predicate, written once into the
  receipt; a ``deferred_to_lease`` run judges nothing and discharges nothing.
* S3 order — the drain deadline becomes a durable record, and only then can no second actor
  SIGTERM; an open drain is defended on disk, not in one process's memory.
* Repeated restarts must not double-apply (``_run_pending_fleet_restart`` + the host stamp).
* A failing health predicate fails CLOSED: a stopped gateway is never reported recovered.

Nothing here touches the live host gateway: the lease/obligation live under a per-test
``HERMES_GATEWAY_LOCK_DIR``, every signal is a recorder, every restart actor is a stub, and the
receipt is captured instead of written.
"""

from __future__ import annotations

import json
import os
import signal as signal_mod
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_restart_orchestrator as orch

SHA = "b" * 40
OTHER_SHA = "a" * 40


def fleet_mod():
    from hermes_cli import update_cmd_fleet

    return update_cmd_fleet


def gateway_mod():
    import hermes_cli.gateway as gateway

    return gateway


def obligation_path() -> Path:
    from hermes_cli.update_host_obligation import host_obligation_path

    path = host_obligation_path()
    assert path is not None
    return path


def obligation_record() -> dict:
    return json.loads(obligation_path().read_text(encoding="utf-8"))


def lease_file() -> dict:
    path = orch.lease_path()
    assert path is not None
    return json.loads(path.read_text(encoding="utf-8"))


def arm_obligation(sha: str = SHA, runtimes=None, profile: str = "default") -> None:
    from hermes_cli.update_host_obligation import write_host_obligation

    runtimes = runtimes if runtimes is not None else [{"kind": "gateway", "profile": "default"}]
    assert write_host_obligation(expected_sha=sha, runtimes=runtimes, profile=profile)


def plan_for(*profiles: str) -> SimpleNamespace:
    """A pre-update plan carrying the gateway runtimes the phase will lease for."""
    return SimpleNamespace(
        runtimes=[SimpleNamespace(kind="gateway", profile=profile) for profile in profiles] or
        [SimpleNamespace(kind="gateway", profile="default")]
    )


class _Signals:
    """Every signal a code path sends, so "exactly one" stays an assertion."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def kill(self, pid: int, signum: int) -> None:
        self.calls.append((pid, signum))

    def sigusr1(self) -> int:
        return sum(1 for _pid, sig in self.calls if sig == signal_mod.SIGUSR1)

    def sigterms(self) -> int:
        return sum(1 for _pid, sig in self.calls if sig == signal_mod.SIGTERM)

    def names(self) -> list[str]:
        return [signal_mod.Signals(sig).name for _pid, sig in self.calls]


@pytest.fixture(autouse=True)
def isolated_host_state(tmp_path, monkeypatch):
    """Per-test lease/obligation dir; the design's 120 s lease wait shortened."""
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir()
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    # Late-bound in acquire_restart_lease, so a live holder never blocks a test for two minutes.
    monkeypatch.setattr(orch, "LEASE_WAIT_S", 0.05, raising=False)
    return lock_dir


@pytest.fixture
def signals(monkeypatch):
    """Route the orchestrator's single OS signal seam to a recorder (never os.kill)."""
    recorder = _Signals()
    monkeypatch.setattr(orch, "_send_signal", recorder.kill, raising=False)
    return recorder


@pytest.fixture
def receipt(monkeypatch):
    """An active update receipt, captured instead of persisted to the sandboxed home."""
    import hermes_cli.update_receipt as ur

    class _Receipt:
        def __init__(self) -> None:
            self.restart: list[dict] = []
            self.verdict: list[dict] = []
            self.finalize: list[str] = []

        def block(self) -> dict:
            """The live ``gateway_restart`` block — readable before AND after finalize."""
            return dict((ur._current.data.get("gateway_restart") or {}) if ur._current else {})

    captured = _Receipt()
    real_restart = ur.record_gateway_restart
    real_verdict = ur.record_gateway_restart_verdict

    def _restart(**kwargs):
        captured.restart.append(kwargs)
        return real_restart(**kwargs)

    def _verdict(**kwargs):
        captured.verdict.append(kwargs)
        return real_verdict(**kwargs)

    def _finalize(outcome, fleet=None, stop_reason=""):
        captured.finalize.append(outcome)
        return None

    monkeypatch.setattr(ur, "record_gateway_restart", _restart, raising=False)
    monkeypatch.setattr(ur, "record_gateway_restart_verdict", _verdict, raising=False)
    monkeypatch.setattr(ur, "finalize_update_receipt", _finalize, raising=False)
    ur.begin_update_receipt()
    try:
        yield captured
    finally:
        ur._current = None


def wire_restart_phase(monkeypatch, *, current_rows=None, interrupt=None, drain_budget=55.0):
    """Wire a restart phase whose abort recovery is the REAL one (fail-closed), with stubs only
    for the OS edges: signals, launchd/systemd/manual actors and the survivor probes."""
    import hermes_cli.update_abort_recovery as abort_mod
    import hermes_cli.update_cmd as cmd

    fleet = fleet_mod()
    monkeypatch.setattr(fleet, "_live_fleet_current_rows", lambda: current_rows)
    monkeypatch.setattr(fleet, "_restart_identity_sha", lambda: SHA)
    monkeypatch.setattr(fleet, "_gateway_drain_budget", lambda: drain_budget)
    monkeypatch.setattr(fleet, "_scoped_manual_gateway_pids", lambda pids, **kw: [4242])
    monkeypatch.setattr(fleet, "_restart_systemd_gateway_units", lambda *a, **k: None)
    monkeypatch.setattr(fleet, "_restart_manual_gateways", lambda *a, **k: None)
    monkeypatch.setattr(fleet, "_force_kill_stuck_gateways", lambda *a, **k: None)
    monkeypatch.setattr(fleet, "_warn_incomplete_gateway_fleet_restart", lambda *a, **k: None)
    monkeypatch.setattr(fleet, "_warn_gateway_restart_phase_aborted", lambda *a, **k: None)
    monkeypatch.setattr(fleet, "_surviving_gateway_pids_after_failed_restart", lambda: [])
    # is_macos() is read through the gateway module at call time; pinning it keeps the launchd
    # branch (the interruption injects there) exercised identically on Linux CI runners.
    monkeypatch.setattr(gateway_mod(), "is_macos", lambda: True, raising=False)
    monkeypatch.setattr(gateway_mod(), "find_gateway_pids", lambda all_profiles=False: [4242], raising=False)
    if interrupt is None:
        monkeypatch.setattr(fleet, "_restart_macos_launchd_gateways", lambda *a, **k: None)
    else:
        def _interrupted(*_a, **_k):
            raise interrupt

        monkeypatch.setattr(fleet, "_restart_macos_launchd_gateways", _interrupted)

    # Abort-recovery collaborators: the fail-closed DECISION stays real
    # (`_restart_phase_failure_is_incomplete` runs for real against an empty survivor probe).
    monkeypatch.setattr(abort_mod, "_owed_stale_serve_rows", lambda rows: False, raising=False)
    monkeypatch.setattr(cmd, "_abort_recovery_is_complete", lambda **kw: False, raising=False)
    monkeypatch.setattr(cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [], raising=False)
    monkeypatch.setattr(cmd, "_warn_stale_serve_runtimes", lambda rows: None, raising=False)
    monkeypatch.setattr(cmd, "_recover_gateway_restart_after_abort", lambda *a, **k: {"verified": []}, raising=False)
    monkeypatch.setattr(cmd, "_write_gateway_update_exit_code", lambda ok: None, raising=False)
    return fleet


def wire_verify(monkeypatch, *, rows, matrix_incomplete=None, expected=True):
    """Wire the verification phase: matrix + fleet probe stubbed, one verdict recorded for real.

    ``matrix_incomplete`` defaults to the predicate's own read of ``rows`` (any non-current,
    non-external row) so a test cannot accidentally assert a matrix flag its rows do not imply.
    """
    import hermes_cli.gateway_migrate as migrate_mod
    import hermes_cli.main as main_mod
    import hermes_cli.update_cmd as cmd
    import hermes_cli.update_cmd_stale_survivors as stale_mod
    import hermes_cli.update_receipt as ur

    fleet = fleet_mod()
    if matrix_incomplete is None:
        from hermes_cli.update_receipt import row_is_external

        matrix_incomplete = any(
            not row_is_external(row) and row.get("state") != "current" for row in rows
        )
    monkeypatch.setattr(fleet, "_collect_fleet_snapshot", lambda restart, expected_rows: list(rows))
    monkeypatch.setattr(fleet, "_restart_identity_sha", lambda: SHA)
    monkeypatch.setattr(fleet, "_print_legacy_units_warning", lambda: None, raising=False)
    monkeypatch.setattr(main_mod, "_fleet_probe_expected_runtimes", lambda *a, **k: expected, raising=False)
    monkeypatch.setattr(cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [], raising=False)
    monkeypatch.setattr(cmd, "_warn_stale_serve_runtimes", lambda rows: None, raising=False)
    monkeypatch.setattr(stale_mod, "signal_stale_fleet_survivors", lambda *a, **k: None)
    monkeypatch.setattr(migrate_mod, "maybe_auto_migrate_after_update", lambda *a, **k: None)
    monkeypatch.setattr(ur, "print_fleet_version_matrix", lambda fleet_rows: bool(matrix_incomplete))
    return matrix_incomplete


def run_verify(restart, *, update_complete=True):
    fleet = fleet_mod()
    return fleet._verify_fleet_after_update(
        restart,
        _pre_update_plan=None,
        _windows_gateway_resume=[],
        node_failures=[],
        update_complete=update_complete,
    )


class TestInterruptedRunStaysResumable:
    """§3.3 #4 — an interruption between S3 and S7 is a resume, never a silent success."""

    def test_phase_abort_keeps_the_obligation_armed_and_releases_the_lease(
            self, monkeypatch, signals, receipt, capsys):
        arm_obligation()
        wire_restart_phase(monkeypatch, interrupt=RuntimeError("interrupted between S3 and S7"))

        out = fleet_mod()._restart_gateway_fleet_after_update(plan_for("default"), gateway_mode=False)

        assert out.incomplete is True, "an interrupted restart must never report a healthy fleet"
        assert out.phase_errors == ["interrupted between S3 and S7"]
        assert obligation_path().is_file(), (
            "the obligation must stay armed: the fleet was drained and never verified"
        )
        assert receipt.block()["incomplete"] is True
        assert receipt.block()["phase_error"] == "interrupted between S3 and S7"
        assert signals.calls == [], "the phase aborted before any signal"
        # The phase is over: a holder that is no longer restarting anything must not block the
        # next actor or the catch-up.
        assert orch.restart_in_progress() is False
        assert orch.lease_path() is not None and not orch.lease_path().exists()

    def test_the_next_run_finalizes_partial_and_arms_one_escalation_plus_a_retry(
            self, monkeypatch, signals, receipt, capsys):
        arm_obligation()
        wire_restart_phase(monkeypatch, interrupt=RuntimeError("interrupted between S3 and S7"))
        out = fleet_mod()._restart_gateway_fleet_after_update(plan_for("default"), gateway_mode=False)

        # S5/S6 on the next pass: the fleet is read back and still on the pre-update code.
        wire_verify(monkeypatch, rows=[{"profile": "default", "state": "stale", "code_sha": OTHER_SHA, "pid": 4242}])
        with pytest.raises(SystemExit) as exit_info:
            run_verify(out)
        assert exit_info.value.code == 1

        assert receipt.finalize == ["partial"], "an unfinished restart is a partial receipt"
        assert receipt.block()["verdict"] == "stale"
        assert receipt.block()["incomplete"] is True
        assert receipt.block()["phase_error"] == "interrupted between S3 and S7", (
            "the verdict merges in place; the phase error survives it"
        )
        assert len(receipt.verdict) == 1, "one verdict, written once"

        record = obligation_record()
        assert obligation_path().is_file(), "a failed restart keeps the obligation armed"
        assert record["catch_up"]["attempt"] == 1, "S8: a bounded retry is armed"
        assert record["escalation"]["reason"] == "stale", (
            "the escalation names the matrix's failing state, not the phase flag"
        )

    def test_resume_covers_only_the_recorded_runtime_set(self, monkeypatch):
        """The obligation's inventory is the owed set: a live fleet that does not include every
        recorded gateway cannot discharge the restart, however healthy the rows it does show."""
        fleet = fleet_mod()
        arm_obligation(runtimes=[
            {"kind": "gateway", "profile": "default"},
            {"kind": "gateway", "profile": "prompter"},
        ])
        monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: SHA)

        import hermes_cli.update_receipt as ur

        covered = [{"profile": "default", "state": "current", "code_sha": SHA, "pid": 1}]
        monkeypatch.setattr(ur, "collect_fleet_versions", lambda **kw: list(covered))

        assert fleet._pending_fleet_restart_needed() is True, (
            "the recorded prompter gateway is absent — the restart is still owed"
        )
        assert fleet._marker_only_restart_obsolete() is False

        both = covered + [{"profile": "prompter", "state": "current", "code_sha": SHA, "pid": 2}]
        monkeypatch.setattr(ur, "collect_fleet_versions", lambda **kw: list(both))

        assert fleet._marker_only_restart_obsolete() is True
        assert fleet._pending_fleet_restart_needed() is False, "the recorded set now serves the code"
        assert not obligation_path().exists(), "discharge removes the obligation"

    def test_a_foreign_gateway_cannot_stand_in_for_the_recorded_one(self, monkeypatch):
        """Coverage is identity-matched: an unrelated profile serving the code is not evidence."""
        fleet = fleet_mod()
        arm_obligation(runtimes=[{"kind": "gateway", "profile": "default"}])
        monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: SHA)

        import hermes_cli.update_receipt as ur

        monkeypatch.setattr(ur, "collect_fleet_versions", lambda **kw: [
            {"profile": "coder", "state": "current", "code_sha": SHA, "pid": 7},
        ])
        assert fleet._marker_only_restart_obsolete() is False
        assert obligation_path().is_file()

    def test_failed_retry_reschedules_within_bounds_and_a_successful_one_retries_once(
            self, monkeypatch):
        """S8 retry: bounded attempts, then escalation; success is reachable without a human."""
        arm_obligation()
        attempts: list[str] = []

        assert orch.schedule_catch_up(sha=SHA, now=0.0) is not None
        assert orch.run_due_catch_up(runner=lambda: attempts.append("run") or False, now=10_000.0) == "failed"
        assert attempts == ["run"]
        assert obligation_record()["catch_up"]["attempt"] == 2, "a failed retry arms the next attempt"
        assert obligation_path().is_file(), "the obligation survives a failed retry"

        # The armed delay is what gates the next attempt — not wall-clock in the test.
        due_at = obligation_record()["catch_up"]["next_at"]
        assert orch.run_due_catch_up(runner=lambda: attempts.append("run") or True, now=due_at) == "ran"
        assert attempts == ["run", "run"], "exactly one attempt per due window"
        assert obligation_path().is_file(), (
            "a successful attempt is not a discharge: the fleet has to prove it serves the code"
        )

        # The proof path (S7): once every recorded gateway serves the pulled code the obligation is
        # discharged and the bounded catch-up stops firing — that, not a dedupe, is what ends the retry.
        fleet = fleet_mod()
        import hermes_cli.update_receipt as ur

        monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: SHA)
        monkeypatch.setattr(ur, "collect_fleet_versions", lambda **kw: [
            {"profile": "default", "state": "current", "code_sha": SHA, "pid": 4242},
        ])
        assert fleet._marker_only_restart_obsolete() is True
        assert fleet._pending_fleet_restart_needed() is False
        assert orch.run_due_catch_up(runner=lambda: pytest.fail("nothing is owed any more")) == "no-obligation"


class TestRepeatedRestartDoesNotDoubleApply:
    """The card's core promise: re-running a restart must not restart the one shared multiplexer."""

    def _wire_pending(self, monkeypatch, calls: dict):
        fleet = fleet_mod()
        gateway = gateway_mod()
        monkeypatch.setattr(fleet, "_restart_identity_sha", lambda: SHA)
        monkeypatch.setattr(gateway, "is_macos", lambda: True, raising=False)
        monkeypatch.setattr(gateway, "is_windows", lambda: False, raising=False)
        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False, raising=False)
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda all_profiles=False: [4242], raising=False)
        monkeypatch.setattr(gateway, "kill_gateway_processes", lambda **kw: calls.setdefault("kill", []).append(kw))
        monkeypatch.setattr(gateway, "_wait_for_gateway_exit", lambda **kw: True, raising=False)
        monkeypatch.setattr(fleet, "_live_fleet_current_rows", lambda: None)
        monkeypatch.setattr(fleet, "_restart_macos_launchd_gateways", lambda *a, **k: calls.setdefault("launchd", []).append(a))
        return fleet

    def test_already_completed_host_restart_is_a_noop_before_any_probe(self, monkeypatch, capsys):
        fleet = fleet_mod()
        arm_obligation()
        from hermes_cli.update_host_obligation import mark_host_restart_completed

        mark_host_restart_completed(SHA)
        monkeypatch.setattr(fleet, "_restart_identity_sha", lambda: SHA)
        monkeypatch.setattr(
            gateway_mod(), "find_gateway_pids",
            lambda all_profiles=False: pytest.fail("the host restart already happened"),
        )

        assert fleet._run_pending_fleet_restart() is True
        assert "already restarted for this update" in capsys.readouterr().out

    def test_second_run_of_the_catch_up_does_not_restart_again(self, monkeypatch, capsys):
        calls: dict = {}
        fleet = self._wire_pending(monkeypatch, calls)
        arm_obligation()

        assert fleet._run_pending_fleet_restart() is True
        assert len(calls.get("kill", [])) == 1, "the first catch-up stops the stale gateway once"
        assert obligation_record()["restarted"]["sha"] == SHA, "the host stamp is the proof"

        assert fleet._run_pending_fleet_restart() is True
        assert len(calls.get("kill", [])) == 1, "the second catch-up must not stop it again"
        assert len(calls.get("launchd", [])) == 1
        assert "already restarted for this update" in capsys.readouterr().out


class TestSecondUpdaterTakesOverAfterTheFirstFinishes:
    """§3.3 #3 — waiting is not the only outcome; the successor completes as a no-op."""

    def test_second_updater_acquires_the_lease_and_sends_nothing(self, monkeypatch, signals, capsys):
        from hermes_cli import update_cmd_fleet as fleet

        first = orch.acquire_restart_lease(
            sha=SHA, runtime_ids=["gateway:default"], trigger="hermes update (first)")
        assert first.acquired is True
        outcome = orch.request_graceful_restart(
            4242, budget_s=55.0, actor="hermes update (first)", label="default", key=first.key)
        assert outcome.sent is True and signals.sigusr1() == 1
        orch.record_restart_action(
            key=first.key, sha=SHA, runtime_ids=["gateway:default"], verdict="current")
        assert orch.release_restart_lease(key=first.key) is True

        # The successor finds the fleet already on the pulled code: it takes the lease and does
        # nothing with it (the no-op path), so the same work is never applied twice.
        wire_restart_phase(monkeypatch, current_rows=[{"profile": "default", "state": "current"}])
        second = fleet._restart_gateway_fleet_after_update(plan_for("default"), gateway_mode=False)

        assert second.deferred_to_lease is False, "the holder released; nothing to defer to"
        assert second.incomplete is False
        assert second.restarted_services == [] and second.failed_or_stale_units == []
        assert signals.sigusr1() == 1, "the second updater must send no further signal"
        assert "no restart actions taken" in capsys.readouterr().out
        assert not orch.restart_in_progress(), "the no-op releases its own lease"

    def test_the_two_updaters_share_one_idempotency_key(self):
        """Same pulled code × same runtime set is ONE action, whoever runs it (§3.3 #1/#3)."""
        first = orch.restart_action_key(SHA, ["gateway:default"])
        second = orch.restart_action_key(SHA, ["gateway:default"])
        assert first == second
        assert first != orch.restart_action_key(SHA, ["gateway:default", "gateway:prompter"])


class TestOrderingIsOneVerdictFromTheMatrix:
    """Ordering: S0–S4 first, then one verdict from the fleet-matrix predicate — never before."""

    def test_restart_then_verify_writes_exactly_one_verdict(self, monkeypatch, signals, receipt, capsys):
        arm_obligation()
        wire_restart_phase(monkeypatch)
        out = fleet_mod()._restart_gateway_fleet_after_update(plan_for("default"), gateway_mode=False)
        assert out.incomplete is False, "the phase itself has no failed units"

        wire_verify(monkeypatch, rows=[{"profile": "default", "state": "stale", "code_sha": OTHER_SHA, "pid": 4242}])
        with pytest.raises(SystemExit) as exit_info:
            run_verify(out)
        assert exit_info.value.code == 1

        assert len(receipt.verdict) == 1, "one verdict, written once (G3)"
        assert len(receipt.restart) == 1, "the phase-level record is written once too"
        block = receipt.block()
        assert block["verdict"] == "stale"
        assert block["incomplete"] is True
        assert block["verdict_expected_sha"] == SHA
        assert block["verdict_failing_states"] == ["stale"]
        assert receipt.finalize == ["partial"]
        assert obligation_path().is_file(), "a failing verdict never discharges the obligation"

    def test_a_clean_fleet_writes_one_current_verdict_and_discharges(self, monkeypatch, signals, receipt):
        arm_obligation()
        wire_restart_phase(monkeypatch)
        out = fleet_mod()._restart_gateway_fleet_after_update(plan_for("default"), gateway_mode=False)

        wire_verify(monkeypatch, rows=[{"profile": "default", "state": "current", "code_sha": SHA, "pid": 4242}])
        run_verify(out)

        assert len(receipt.verdict) == 1
        assert receipt.block()["verdict"] == "current" and receipt.block()["incomplete"] is False
        assert receipt.finalize == ["success"]
        assert not obligation_path().exists(), "success-only discharge (§3.1 S7)"

    def test_a_deferred_run_judges_nothing_and_discharges_nothing(self, monkeypatch, signals, receipt, capsys):
        """S0 stand-down: no matrix, no verdict, no discharge — the holder owns all three."""
        import hermes_cli.update_receipt as ur

        arm_obligation()
        holder = orch.acquire_restart_lease(
            sha=SHA, runtime_ids=["gateway:default"], trigger="hermes update (holder)")
        assert holder.acquired

        wire_restart_phase(monkeypatch, current_rows=None)
        out = fleet_mod()._restart_gateway_fleet_after_update(plan_for("default"), gateway_mode=False)
        assert out.deferred_to_lease is True
        assert signals.calls == [], "a deferring run sends nothing"
        assert orch.restart_in_progress() is True, "and never releases the holder's lease"

        wire_verify(monkeypatch, rows=[{"profile": "default", "state": "stale", "code_sha": OTHER_SHA}])
        monkeypatch.setattr(
            ur, "print_fleet_version_matrix",
            lambda fleet_rows: pytest.fail("a deferring run must not judge the fleet"),
        )
        run_verify(out)

        assert receipt.verdict == [], "no verdict may be published by a run that acted on nothing"
        assert receipt.finalize == ["success"]
        assert obligation_path().is_file(), "the obligation is the holder's to discharge"
        assert "escalation" not in obligation_record(), "no escalation for a run that failed nothing"
        assert orch.restart_in_progress() is True

    def test_health_predicate_failure_fails_closed(self):
        """The rollback decision: an unproven fleet is never reported as recovered."""
        fleet = fleet_mod()
        assert fleet._restart_phase_failure_is_incomplete(None, [4242]) is True
        assert fleet._restart_phase_failure_is_incomplete([4242], [4242]) is True
        assert fleet._restart_phase_failure_is_incomplete([], [4242]) is True, (
            "a pre-restart gateway that is gone now was stopped without a verified replacement"
        )
        assert fleet._restart_phase_failure_is_incomplete([], None) is True
        assert fleet._restart_phase_failure_is_incomplete([], []) is False, (
            "empty survivors prove safety only when nothing ran beforehand"
        )


class TestS3OrderingIsDurable:
    """S3 order: the signal is arbitrated, then the deadline is recorded — on disk, not in memory."""

    def test_the_drain_deadline_is_a_durable_record_for_the_next_actor(self, signals):
        lease = orch.acquire_restart_lease(sha=SHA, runtime_ids=["gateway:default"], trigger="actor-A")
        assert lease.acquired

        outcome = orch.request_graceful_restart(
            4242, budget_s=55.0, actor="actor-A", label="default", key=lease.key)
        assert outcome.sent is True and signals.names() == ["SIGUSR1"]

        recorded = lease_file()["drains"]["4242"]
        assert recorded["deadline_ts"] == pytest.approx(outcome.deadline_ts)
        assert recorded["actor"] == "actor-A"

        # A second actor reads the deadline off disk and may not shorten it (the 19:28:46 SIGTERM).
        refused = orch.escalate_to_sigterm(4242, actor="actor-B", reason="impatient")
        assert refused.sent is False and refused.reason == "drain-deadline-not-reached"
        assert signals.sigterms() == 0
        assert lease_file()["drains"]["4242"]["deadline_ts"] == pytest.approx(outcome.deadline_ts)

    def test_a_second_actor_may_signal_only_after_the_recorded_deadline(self, signals):
        lease = orch.acquire_restart_lease(sha=SHA, runtime_ids=["gateway:default"], trigger="actor-A")
        assert lease.acquired
        started = 1_000_000.0
        orch.mark_drain_started(4242, deadline_ts=started + 55.0, actor="actor-A", key=lease.key)

        assert orch.restart_signal_gate(4242, "SIGTERM", actor="actor-B", now=started + 54.9).sent is False
        assert orch.restart_signal_gate(4242, "SIGTERM", actor="actor-B", now=started + 55.0).sent is True


class TestLeaseContractForNonUpdateActors:
    """The seam every other restart actor must route through (wiring itself: ``t_559d31fb``).

    These pin the contract those actors inherit — attribution, wait-then-defer, refusal inside an
    open drain — so the actor-specific call sites can be added without a second lease design.
    """

    def test_a_non_update_actor_defers_and_is_attributed(self, signals, caplog):
        first = orch.acquire_restart_lease(sha=SHA, runtime_ids=["gateway:default"], trigger="hermes update")
        assert first.acquired

        second = orch.acquire_restart_lease(
            sha=SHA, runtime_ids=["gateway:default"], trigger="hermes gateway restart",
            requestor=orch.Requestor(
                pid=os.getpid(), argv=("hermes", "gateway", "restart"), cwd="/tmp",
                hermes_home="/tmp/home", trigger="hermes gateway restart"),
        )
        assert second.acquired is False and second.state == "held"
        assert second.holder["requestor"]["trigger"] == "hermes update"
        assert signals.calls == [], "the deferring actor restarts nothing"

        record = lease_file()["requestor"]
        assert record["trigger"] == "hermes update", "the lease names who is acting on the host"
        assert set(record) >= {"pid", "argv", "cwd", "hermes_home", "trigger"}

    def test_a_non_update_actor_cannot_release_or_shorten_the_holder(self, signals):
        holder = orch.acquire_restart_lease(sha=SHA, runtime_ids=["gateway:default"], trigger="hermes update")
        assert holder.acquired
        orch.mark_drain_started(4242, deadline_ts=orch.time.time() + 55.0, actor="hermes update")

        assert orch.release_restart_lease(key=orch.restart_action_key(SHA, ["gateway:other"])) is False
        assert orch.restart_in_progress() is True

        refused = orch.escalate_to_sigterm(4242, actor="launchctl kickstart", reason="manual restart")
        assert refused.sent is False and signals.sigterms() == 0
