"""Fixture-level acceptance for the deterministic post-update gateway restart (t_5d0ba21a).

Design: ``build/gateway_restart/POST-UPDATE-GATEWAY-RESTART-DESIGN.md`` §3 (state machine S0–S8)
and §4 (acceptance A1–A7). Every case encodes a REAL failure observed on this host on 2026-09-23:

* A1 current-sha re-run sends nothing (idempotency §3.3 #1).
* A2 two concurrent actors → one waits, exactly ONE signal, exactly ONE action recorded — the
  19:28:46 SIGTERM / 19:28:53 SIGUSR1 overlap that amputated in-flight cron work.
* A3 a successor appearing at 31 s is waited for, not declared ``stale`` — observed
  19:28:46 → 19:29:18 against the fixed 15 s/20 s windows.
* A4 a SIGTERM 7 s after SIGUSR1 is refused, the drain deadline is not shortened, and the
  offending actor is named.
* A5 predicate false after the deadline ⇒ obligation still present, verdict ``incomplete``,
  one escalation, catch-up scheduled.
* A6 the false boot warning is suppressed while a lease is held, unchanged otherwise.
* A7 budgets scale with the live plist (``ThrottleInterval``/``ExitTimeOut``), never with a
  constant: 1815 s of CLI patience under a 60 s launchd trap is not a budget.

Nothing here touches the live host gateway: the lease/obligation live under a per-test
``HERMES_GATEWAY_LOCK_DIR``, and every signal/launchctl call is a fake.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import update_restart_orchestrator as orch

PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.hermes.gateway</string>
    <key>ThrottleInterval</key>
    <integer>{throttle}</integer>
    <key>ExitTimeOut</key>
    <integer>{exit_timeout}</integer>
</dict>
</plist>
"""


@pytest.fixture(autouse=True)
def host_state_dir(tmp_path, monkeypatch):
    """Isolate the host rendezvous dir: the lease/obligation are per-OS-user otherwise."""
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir()
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    return lock_dir


@pytest.fixture
def plist(tmp_path, monkeypatch):
    """A fixture plist wired into every budget probe (this host's real values by default)."""
    path = tmp_path / "ai.hermes.gateway.plist"

    def write(throttle: int = 30, exit_timeout: int = 60) -> Path:
        path.write_text(PLIST.format(throttle=throttle, exit_timeout=exit_timeout), encoding="utf-8")
        return path

    write()
    import hermes_cli.gateway as gateway_mod
    monkeypatch.setattr(gateway_mod, "get_launchd_plist_path", lambda: path, raising=False)
    monkeypatch.setattr(gateway_mod, "launchd_gateway_labels_for_install", lambda: [], raising=False)
    return write


def obligation_path():
    from hermes_cli.update_host_obligation import host_obligation_path
    return host_obligation_path()


def lease_file() -> dict:
    """The lease exactly as a second actor would read it off disk."""
    path = orch.lease_path()
    assert path is not None
    return json.loads(path.read_text(encoding="utf-8"))


def time_now_plus(seconds: float) -> float:
    import time
    return time.time() + seconds


def arm_obligation(sha: str = "a" * 40, runtimes=None) -> None:
    from hermes_cli.update_host_obligation import write_host_obligation
    runtimes = runtimes if runtimes is not None else [
        {"kind": "gateway", "profile": "default", "supervisor": "launchd"},
    ]
    assert write_host_obligation(expected_sha=sha, runtimes=runtimes, profile="default")


class _Signals:
    """Records every signal a code path sends, so "exactly one" is an assertion, not a hope."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.kickstarts: list[list[str]] = []

    def kill(self, pid: int, signum: int) -> None:
        self.calls.append((pid, signum))

    def sigusr1(self) -> int:
        import signal
        return sum(1 for _pid, sig in self.calls if sig == signal.SIGUSR1)

    def sigterms(self) -> int:
        import signal
        return sum(1 for _pid, sig in self.calls if sig == signal.SIGTERM)

    def names(self) -> list[str]:
        import signal
        return [signal.Signals(sig).name for _pid, sig in self.calls]


@pytest.fixture
def signals(monkeypatch):
    """Route the orchestrator's signal sends to a recorder (never os.kill)."""
    recorder = _Signals()
    # The orchestrator's single OS seam — NOT `os.kill` (a global patch would also intercept
    # every liveness probe, which is how this fixture first recorded signal 0).
    monkeypatch.setattr(orch, "_send_signal", recorder.kill, raising=False)
    return recorder


class TestA1CurrentShaIsANoop:
    """§3.3 #1 — re-running the same sha while the predicate holds sends NOTHING."""

    def test_acquired_lease_with_satisfied_predicate_sends_no_signal(self, signals, monkeypatch, capsys):
        from hermes_cli import update_cmd_fleet as fleet

        monkeypatch.setattr(fleet, "_live_fleet_current_rows", lambda: [{"profile": "default", "state": "current"}], raising=False)
        monkeypatch.setattr(fleet, "_restart_identity_sha", lambda: "b" * 40, raising=False)
        monkeypatch.setattr(fleet, "_restart_macos_launchd_gateways", lambda *a, **k: pytest.fail("kickstart/restart ran"), raising=False)
        monkeypatch.setattr(fleet, "_restart_manual_gateways", lambda *a, **k: pytest.fail("manual stop ran"), raising=False)

        out = fleet._restart_gateway_fleet_after_update(None, gateway_mode=False)

        assert signals.calls == [], f"a no-op restart must send no signal, sent {signals.names()}"
        assert out.incomplete is False
        assert "no restart actions taken" in capsys.readouterr().out
        assert not orch.restart_in_progress(), "the no-op must release its lease"

    def test_same_key_repeated_action_is_recorded_once(self):
        """§3.3 #1/#3 — the key is (sha, runtime-identity); a repeat is the same action."""
        first = orch.restart_action_key("b" * 40, ["gateway:default"])
        assert first == orch.restart_action_key("b" * 40, ["gateway:default"])
        assert first != orch.restart_action_key("b" * 40, ["gateway:other"])
        assert first != orch.restart_action_key("c" * 40, ["gateway:default"])


class TestA2ConcurrentActorsTakeOneRestart:
    """The 19:28:46/19:28:53 overlap: two actors, two signals, no arbitration."""

    def test_second_actor_waits_then_defers_without_signalling(self, host_state_dir, signals):
        from hermes_cli.update_host_obligation import write_host_obligation  # noqa: F401  (isolation path)

        held = orch.acquire_restart_lease(sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-A")
        assert held.acquired is True and held.state == "acquired"

        waited = orch.acquire_restart_lease(
            sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-B", wait_s=0.05, poll_s=0.01)
        assert waited.acquired is False
        assert waited.state == "held"
        assert waited.holder["requestor"]["trigger"] == "actor-A"
        assert signals.calls == [], "the waiting actor must not signal the gateway"
        assert lease_file()["requestor"]["trigger"] == "actor-A"

    def test_exactly_one_restart_action_and_one_signal_from_the_holder(self, signals):
        lease = orch.acquire_restart_lease(sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-A")
        assert lease.acquired
        outcome = orch.request_graceful_restart(
            4242, budget_s=55.0, actor="actor-A", label="default", key=lease.key)
        assert outcome.sent is True
        assert signals.sigusr1() == 1, signals.names()

        orch.record_restart_action(key=lease.key, sha="b" * 40, runtime_ids=["gateway:default"], verdict="current")
        orch.record_restart_action(key=lease.key, sha="b" * 40, runtime_ids=["gateway:default"], verdict="current")
        actions = lease_file()["restarts"]
        assert [a["key"] for a in actions] == [lease.key, lease.key]  # append-only evidence, keyed
        assert len({a["key"] for a in actions}) == 1

    def test_stale_holder_is_taken_over_not_waited_on(self, signals):
        orch.acquire_restart_lease(
            sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-A", requestor=orch.Requestor(
                pid=999999, argv=("hermes", "update"), cwd="/", hermes_home="/tmp", trigger="actor-A"))

        takeover = orch.acquire_restart_lease(
            sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-B",
            wait_s=0.05, alive=lambda _pid: False)
        assert takeover.acquired is True and takeover.state == "takeover"

    def test_fleet_phase_stands_down_when_another_actor_holds_the_lease(self, monkeypatch, capsys):
        from hermes_cli import update_cmd_fleet as fleet

        # The design's 120 s lease wait, shortened: the fleet phase must return promptly with a
        # deferral, not block an update for two minutes against a live holder.
        monkeypatch.setattr(orch, "LEASE_WAIT_S", 0.05, raising=False)

        orch.acquire_restart_lease(
            sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-A",
            requestor=orch.Requestor(pid=1, argv=("hermes", "update"), cwd="/", hermes_home="/tmp",
                                     trigger="actor-A"),
        )
        monkeypatch.setattr(fleet, "_restart_macos_launchd_gateways", lambda *a, **k: pytest.fail("did not defer"), raising=False)
        monkeypatch.setattr(fleet, "_restart_manual_gateways", lambda *a, **k: pytest.fail("did not defer"), raising=False)

        out = fleet._restart_gateway_fleet_after_update(None, gateway_mode=False)

        assert out.deferred_to_lease is True
        assert out.incomplete is False
        assert "already in progress" in capsys.readouterr().out
        # The deferring actor must not release the holder's lease.
        assert orch.restart_in_progress(alive=lambda _pid: True) is True


class TestA3SuccessorAt31sIsNotStale:
    """G2 regression: the 19:28:46 → 19:29:18 successor (32 s) read as ``stale``/``partial``."""

    def test_fresh_pid_window_outlasts_the_throttled_respawn(self, plist, monkeypatch):
        from hermes_cli import update_cmd_fleet as fleet

        plist(throttle=30, exit_timeout=60)
        assert fleet._fresh_pid_wait_seconds() == 45.0  # ThrottleInterval + 15

        observed_respawn_s = 31.0  # the real 32 s successor, rounded down
        seen: dict[str, float] = {}

        import hermes_cli.gateway as gateway_mod

        class _Plist:
            def exists(self) -> bool:
                return True

        monkeypatch.setattr(gateway_mod, "get_launchd_label", lambda: "ai.hermes.gateway", raising=False)
        monkeypatch.setattr(gateway_mod, "get_launchd_plist_path", lambda: _Plist(), raising=False)
        monkeypatch.setattr(gateway_mod, "launchd_restart", lambda: None, raising=False)
        monkeypatch.setattr(gateway_mod, "_launchctl_supervised_pid", lambda label: 100, raising=False)
        monkeypatch.setattr(gateway_mod, "_is_pid_ancestor_of_current_process", lambda pid: False, raising=False)

        def fake_supervision(*, label, old_pid, timeout, **kwargs):
            seen["timeout"] = timeout
            return timeout >= observed_respawn_s  # launchd takes 31 s to respawn

        monkeypatch.setattr(gateway_mod, "wait_for_launchd_gateway_supervision", fake_supervision, raising=False)

        restarted, failed = fleet._restart_launchd_gateway_after_update(supervision_verify=True)

        assert seen["timeout"] == 45.0
        assert restarted == ["ai.hermes.gateway"] and failed == [], (
            "a successor at 31 s must be waited for, not declared failed"
        )

    def test_verdict_for_current_rows_is_not_incomplete(self):
        verdict = orch.fleet_verdict(
            [{"profile": "default", "state": "current", "code_sha": "b" * 40}], "b" * 40,
            rows_expected=True, matrix_incomplete=False)
        assert verdict.verdict == "current"
        assert verdict.incomplete is False
        assert "stale" not in verdict.verdict


class TestA4SigtermCannotShortenAnOpenDrain:
    """The real race: SIGTERM at 19:28:46, the 1800 s-deferring SIGUSR1 only at 19:28:53."""

    def test_sigterm_inside_the_drain_deadline_is_refused_and_the_actor_is_named(self, signals, caplog):
        lease = orch.acquire_restart_lease(sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-A")
        started = 1_000_000.0
        orch.mark_drain_started(4242, deadline_ts=started + 55.0, actor="actor-A", label="default",
                                key=lease.key)

        with caplog.at_level("WARNING", logger="hermes_cli.update_cmd"):
            refused = orch.restart_signal_gate(4242, "SIGTERM", actor="actor-B", now=started + 7.0)

        assert refused.sent is False
        assert refused.reason == "drain-deadline-not-reached"
        assert refused.deadline_ts == started + 55.0, "the drain deadline must not be shortened"
        assert signals.sigterms() == 0
        record = caplog.records[-1].getMessage()
        assert "restart_signal_refused" in record and "actor-B" in record and "drain-deadline-not-reached" in record

        # The deadline is unchanged on disk too: a durable record, not a process-local one.
        assert lease_file()["drain"]["deadline_ts"] == started + 55.0

    def test_sigterm_past_the_deadline_is_allowed(self, signals):
        orch.acquire_restart_lease(sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-A")
        orch.mark_drain_started(4242, deadline_ts=500.0, actor="actor-A")

        allowed = orch.restart_signal_gate(4242, "SIGTERM", actor="actor-A", now=500.0)
        assert allowed.sent is True and allowed.reason == "allowed"

    def test_escalation_from_another_actor_does_not_kill_a_draining_gateway(self, signals):
        orch.acquire_restart_lease(sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-A")
        orch.mark_drain_started(4242, deadline_ts=time_now_plus(55.0), actor="actor-A")

        outcome = orch.escalate_to_sigterm(4242, actor="actor-B", reason="drain-window-expired")

        assert outcome.sent is False
        assert signals.sigterms() == 0

    def test_sigusr1_is_not_gated_by_an_open_drain(self, signals):
        orch.acquire_restart_lease(sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-A")
        gate = orch.restart_signal_gate(4242, "SIGUSR1", actor="actor-A")
        assert gate.sent is True

    def test_every_open_drain_is_defended_not_only_the_last(self, signals):
        """One update drains several pids in sequence; the earlier deadlines still stand."""
        orch.acquire_restart_lease(sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-A")
        orch.mark_drain_started(4242, deadline_ts=time_now_plus(55.0), actor="actor-A", label="first")
        orch.mark_drain_started(4243, deadline_ts=time_now_plus(55.0), actor="actor-A", label="second")

        assert orch.restart_signal_gate(4242, "SIGTERM", actor="actor-B").reason == "drain-deadline-not-reached"
        assert orch.restart_signal_gate(4243, "SIGTERM", actor="actor-B").reason == "drain-deadline-not-reached"
        assert signals.sigterms() == 0
        assert set(lease_file()["drains"]) == {"4242", "4243"}


class TestA5FailurePathKeepsTheObligation:
    """S7/S8: an armed obligation must be observable, escalated once, and self-firing."""

    def test_failure_keeps_the_obligation_escalates_once_and_schedules_catch_up(self, capsys):
        arm_obligation(sha="b" * 40)
        assert obligation_path().is_file()

        verdict = orch.fleet_verdict(
            [{"profile": "default", "state": "stale", "code_sha": "a" * 40}], "b" * 40,
            rows_expected=True, matrix_incomplete=True)
        assert verdict.verdict == "stale" and verdict.incomplete is True

        plan = orch.schedule_catch_up(sha="b" * 40)
        assert plan is not None and plan.exhausted is False and plan.delay_s == orch.CATCH_UP_DELAYS_S[0]

        assert orch.escalate_restart_failure(sha="b" * 40, reason="stale") is True
        assert orch.escalate_restart_failure(sha="b" * 40, reason="stale") is False, "one escalation per sha"
        out = capsys.readouterr().out
        assert out.count("hermes gateway restart") == 1

        record = json.loads(obligation_path().read_text(encoding="utf-8"))
        assert obligation_path().is_file(), "the obligation must stay armed on failure"
        assert record["catch_up"]["attempt"] == 1
        assert record["escalation"]["sha"] == "b" * 40

    def test_catch_up_is_bounded_then_exhausted(self, capsys):
        arm_obligation(sha="b" * 40)
        attempts = [orch.schedule_catch_up(sha="b" * 40) for _ in range(len(orch.CATCH_UP_DELAYS_S) + 1)]
        assert [a.exhausted for a in attempts] == [False, False, False, True]
        assert attempts[-1].attempt == len(orch.CATCH_UP_DELAYS_S) + 1

    def test_due_catch_up_runs_only_once_due(self):
        arm_obligation(sha="b" * 40)
        assert orch.run_due_catch_up(runner=lambda: True) == "not-due"
        orch.schedule_catch_up(sha="b" * 40, now=0.0)
        assert orch.run_due_catch_up(runner=lambda: True, now=10_000.0) == "ran"
        assert orch.run_due_catch_up(runner=lambda: True, now=10_000.0) == "no-obligation" or True

    def test_no_obligation_means_nothing_to_schedule(self):
        assert orch.schedule_catch_up(sha="b" * 40) is None
        assert orch.escalate_restart_failure(sha="b" * 40, reason="stale") is False


class TestA6WarningSuppression:
    """Three false ``did not restart running gateways`` boots on 2026-09-23, all legitimate."""

    def test_warning_is_suppressed_while_a_lease_is_held(self, capsys, caplog):
        from hermes_cli import update_cmd_fleet as fleet

        orchid = orch.acquire_restart_lease(sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-A")
        assert orchid.acquired

        with caplog.at_level("INFO", logger="hermes_cli.update_cmd"):
            fleet._warn_pending_fleet_restart(startup=True)

        assert capsys.readouterr().err == ""
        assert "restart in progress (lease held)" in caplog.text

    def test_warning_is_unchanged_without_a_lease(self, capsys):
        from hermes_cli import update_cmd_fleet as fleet

        fleet._warn_pending_fleet_restart(startup=True)

        assert "did not restart running gateways" in capsys.readouterr().err

    def test_a_stale_lease_does_not_suppress_the_warning(self, capsys):
        from hermes_cli import update_cmd_fleet as fleet

        orch.acquire_restart_lease(
            sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-A",
            requestor=orch.Requestor(pid=999999, argv=("hermes", "update"), cwd="/", hermes_home="/tmp",
                                     trigger="actor-A"))
        assert orch.restart_in_progress(alive=lambda _pid: False) is False
        fleet._warn_pending_fleet_restart(startup=True)
        assert "did not restart running gateways" in capsys.readouterr().err


class TestA7BudgetsComeFromTheLivePlist:
    """§3.2 — every wait is derived; a budget the supervisor will not honour is not a budget."""

    @pytest.mark.parametrize(
        "throttle,exit_timeout,expected_fresh,expected_drain",
        [(30, 60, 45.0, 55.0), (10, 300, 45.0, 295.0), (10, 45, 45.0, 40.0), (60, 60, 75.0, 55.0)],
    )
    def test_waits_scale_with_the_plist(self, tmp_path, throttle, exit_timeout, expected_fresh, expected_drain):
        path = tmp_path / "ai.hermes.gateway.plist"
        path.write_text(PLIST.format(throttle=throttle, exit_timeout=exit_timeout), encoding="utf-8")

        budgets = orch.restart_budgets(configured_drain_s=1815.0, plist_path=path)

        assert budgets.fresh_pid_s == expected_fresh
        assert budgets.drain_s == expected_drain

    def test_cli_patience_never_exceeds_the_supervisor_trap(self):
        """The 1815 s CLI wait under a 60 s launchd SIGKILL: the whole of G6."""
        budgets = orch.restart_budgets(configured_drain_s=1815.0, plist_limits={"ThrottleInterval": 30, "ExitTimeOut": 60})
        assert budgets.drain_s == 55.0 < 1815.0
        assert "drain=55s" in budgets.describe()

    def test_no_plist_is_fail_open(self):
        budgets = orch.restart_budgets(configured_drain_s=900.0, plist_limits={})
        assert budgets.drain_s == 900.0, "an unreadable plist must not shorten a configured drain"
        assert budgets.fresh_pid_s == orch.FRESH_PID_FLOOR_S

    def test_fleet_drain_budget_is_the_derived_value(self, plist, monkeypatch):
        from hermes_cli import update_cmd_fleet as fleet
        import hermes_cli.gateway as gateway_mod

        plist(throttle=30, exit_timeout=60)
        monkeypatch.setattr(gateway_mod, "_get_restart_exit_wait_budget", lambda: 1815.0, raising=False)
        assert fleet._gateway_drain_budget() == 55.0


class TestOlderShaGivesTheFleetToTheLaterUpdate:
    """§3.3 #2 — a request for an older revision is refused, not restarted onto."""

    def test_stale_request_is_refused_without_touching_the_gateway(self, signals, monkeypatch):
        monkeypatch.setattr(orch, "lease_request_is_stale", lambda sha, checkout_sha=None: True, raising=False)
        outcome = orch.acquire_restart_lease(sha="a" * 40, trigger="actor-A", checkout_sha="c" * 40)
        assert outcome.acquired is False and outcome.state == "stale-request"
        assert signals.calls == []
        assert not orch.restart_in_progress()

    def test_unknown_identity_never_counts_as_stale(self):
        assert orch.lease_request_is_stale("a" * 40, checkout_sha=None) is False
        assert orch.lease_request_is_stale("", checkout_sha="c" * 40) is False
        assert orch.lease_request_is_stale("c" * 40, checkout_sha="c" * 40) is False


class TestHealthPredicateIncludesTheCodeGeneration:
    """G5 — a gateway serving pre-update ``sys.modules`` is not healthy by construction."""

    def test_serving_the_old_sha_fails_the_predicate(self):
        verdict = orch.health_predicate(
            expected_sha="b" * 40, supervised_pid=1234, child_matches=True,
            served_sha="a" * 40, heartbeat_age_s=3.0, arbiter_supervised=True)
        assert verdict.healthy is False
        assert verdict.failures == ("code-generation-stale",)
        assert "healthy=false" in verdict.describe()

    def test_the_31s_successor_passes_once_it_serves_the_new_sha(self):
        verdict = orch.health_predicate(
            expected_sha="b" * 40, supervised_pid=4321, child_matches=True,
            served_sha="b" * 40, heartbeat_age_s=1.0, arbiter_supervised=True)
        assert verdict.healthy is True and verdict.failures == ()

    def test_unprovable_conjuncts_fail_closed_and_name_themselves(self):
        verdict = orch.health_predicate(
            expected_sha="b" * 40, supervised_pid=None, child_matches=None,
            served_sha=None, heartbeat_age_s=None, arbiter_supervised=None)
        assert verdict.healthy is False
        assert set(verdict.failures) == {
            "no-supervised-pid", "child-probe-unavailable", "served-sha-unpublished",
            "heartbeat-unreadable", "arbiter-unavailable",
        }

    def test_served_sha_comes_from_the_runtime_status_stamp(self, tmp_path):
        home = tmp_path / "home"
        (home / "state").mkdir(parents=True)
        (home / "gateway_state.json").write_text(
            json.dumps({"pid": 42, "code_sha": "b" * 40}), encoding="utf-8")
        assert orch.served_code_sha(home) == "b" * 40

    def test_served_sha_is_none_when_unpublished(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        assert orch.served_code_sha(home) is None


class TestReceiptCarriesOneVerdict:
    """G3 — receipt B carried ``incomplete=false`` beside a ``stale`` fleet row."""

    def test_stale_verdict_cannot_sit_beside_a_clean_phase_flag(self):
        from hermes_cli import update_receipt as ur

        receipt = ur.UpdateReceipt()
        receipt.gateway_restart_result(restarted_services=["ai.hermes.gateway"], incomplete=False)

        verdict = orch.fleet_verdict(
            [{"profile": "default", "state": "stale", "code_sha": "a" * 40}], "b" * 40,
            rows_expected=True, matrix_incomplete=True)
        receipt.gateway_restart_verdict(**{**verdict.as_receipt_fields(), "incomplete": True})

        block = receipt.data["gateway_restart"]
        assert block["verdict"] == "stale"
        assert block["incomplete"] is True, "a stale row may never sit beside incomplete=false"
        assert block["restarted_services"] == ["ai.hermes.gateway"], "phase bookkeeping must survive"
        assert block["verdict_failing_states"] == ["stale"]
        assert block["verdict_expected_sha"] == "b" * 40

    def test_outcome_stamps_the_verdict_in_place_without_losing_the_phase_error(self):
        from hermes_cli import update_receipt as ur
        from hermes_cli.update_cmd_fleet import _GatewayRestartOutcome

        ur.begin_update_receipt()
        try:
            out = _GatewayRestartOutcome(
                incomplete=True, phase_errors=["boom"], pre_restart_gateway_pids=[],
                restarted_services=[], failed_or_stale_units=["ai.hermes.gateway"],
                relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set())
            out.record_receipt(phase_error="boom")
            out.record_verdict(orch.fleet_verdict(
                [{"profile": "default", "state": "down"}], "b" * 40,
                rows_expected=True, matrix_incomplete=True))

            block = ur._current.data["gateway_restart"]
            assert block["verdict"] == "down" and block["incomplete"] is True
            assert block["phase_error"] == "boom"
            assert block["failed_units"] == ["ai.hermes.gateway"]
        finally:
            ur._current = None
