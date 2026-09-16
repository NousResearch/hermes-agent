"""Truthfulness invariants for the update pipeline at the RECEIPT layer.

Bug class ("update reports success while reality disagrees" — #88654,
#88848, #91378, #91439, #91962, #92780, #92902): the updater's word must be
backed by evidence. This suite pins the honesty contract of the receipt
subsystem with REAL module functions against a temp HERMES_HOME — it is not
a fleet E2E (that lives in the CI install/update harness).

Invariants pinned, and WHERE each is enforced:

1. RECEIPT ALWAYS DURABLE — ``begin → steps → finalize`` writes a
   parseable receipt for success/failed/refused (#91283 made every
   post-begin run leave a record), and ``begin`` seeds it on disk
   immediately: the module singleton is the only other copy of a run, so
   a run whose singleton is lost (module eviction / re-import) used to
   leave NOTHING behind — which is #112465. The seeded record is
   ``outcome == "running"`` and a lost run is finalized from it at the
   command boundary, never left dangling. What a crash must never do is
   READ as success: the Desktop's reader (``read_latest_receipt``,
   surfaced via ``/api/hermes/update/receipt`` — #92780) sees either
   ``running`` or a terminal non-success. The HTTP-layer gating of an
   ``outcome == "running"`` receipt is pinned in
   test_update_receipt_endpoint.py and is deliberately not duplicated
   here.

2. SUCCESS IMPLIES ACCOUNTING — #92902 made the pre-update plan the
   restart worklist. The final refuse-success decision is INLINE in
   ``_cmd_update_impl`` (update_cmd.py ~8331-8373, a 8.5k-line flow), so
   the deepest extractable real functions are pinned instead:
   ``match_runtime_outcomes`` (plan-vs-bookkeeping reconciliation) and
   ``report_unaccounted_runtimes`` (the escalation decision). The
   command's outcome selection is exactly
   ``"partial" if incomplete else "success"`` (update_cmd.py ~8360); the
   tests drive that expression with the real decision function's return
   value, so sabotaging the enforcement (making it accept a missed
   runtime) fails these tests.

3. REFUSAL IS NOT FAILURE — #91439: an exit-2 preflight refusal must
   produce a receipt distinguishable from a failed update, and the
   reader must report it as ``refused``.

Only paths/env are monkeypatched; every receipt is produced by the real
``hermes_cli.update_receipt`` / ``hermes_cli.update_inventory`` API.
"""

import json
import os
import time

import pytest

import hermes_cli.update_receipt as ur
from hermes_cli.update_inventory import (
    RuntimeRecord,
    UpdatePlan,
    match_runtime_outcomes,
    record_plan_in_receipt,
    report_unaccounted_runtimes,
)


@pytest.fixture()
def receipt_home(tmp_path, monkeypatch):
    """Hermetic HERMES_HOME so receipts never touch the real profile."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(
        "hermes_cli.config.get_hermes_home", lambda: home, raising=False
    )
    ur._current = None
    yield home
    ur._current = None


def _receipt_files(home):
    directory = home / "logs" / "update_receipts"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def _per_run_files(home):
    """The per-run records (``update_<id>.json``); ``latest.json`` is a pointer."""
    return [path for path in _receipt_files(home) if path.name.startswith("update_")]


def _plan_with_runtimes(records):
    plan = UpdatePlan(install_method="git", updatable_in_place=True)
    plan.profiles = sorted({r.profile for r in records})
    plan.runtimes = list(records)
    return plan


_THREE_RUNTIMES = [
    RuntimeRecord(kind="gateway", profile="default", pid=101,
                  supervisor="manual", restart_via="manual"),
    RuntimeRecord(kind="gateway", profile="work", pid=102,
                  supervisor="manual", restart_via="manual"),
    RuntimeRecord(kind="serve", profile="ops", pid=103,
                  supervisor="manual", restart_via="manual"),
]


class TestReceiptAlwaysFinalized:
    """Invariant 1 — every finalized run leaves a parseable record; a crash
    leaves NO record the reader could mistake for success (#91283, #92780)."""

    @pytest.mark.parametrize("outcome", ["success", "failed", "refused"])
    def test_finalize_writes_parseable_receipt_for_outcome(
        self, receipt_home, outcome
    ):
        ur.begin_update_receipt()
        ur.record_step("git_pull", outcome == "success", "detail")
        path = ur.finalize_update_receipt(outcome)
        assert path is not None and path.is_file()

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["outcome"] == outcome
        assert payload["finished_at"] is not None
        assert payload["steps"][0]["name"] == "git_pull"

        latest = ur.read_latest_receipt()
        assert latest is not None
        assert latest["outcome"] == outcome

    def test_running_receipt_is_durable_before_finalize(self, receipt_home):
        """`begin` seeds the run on disk and every step rewrites it (#112465):
        losing in-memory state can no longer erase the fact that an update was
        attempted."""
        ur.begin_update_receipt()
        ur.record_step("pre_update_backup", True)

        records = _per_run_files(receipt_home)
        assert len(records) == 1
        # No pointer yet: `latest.json` still names the last FINALIZED run, so an in-flight
        # run cannot hide a previous interrupted update's obligation (#98022).
        assert not (receipt_home / "logs" / "update_receipts" / "latest.json").exists()

        payload = json.loads(records[0].read_text(encoding="utf-8"))
        assert payload["outcome"] == "running"
        assert payload["finished_at"] is None
        assert payload["steps"][0]["name"] == "pre_update_backup"
        # ...and the reader can therefore report no success at all for this run.
        assert ur.read_latest_receipt() is None

    def test_crash_without_finalize_never_claims_success(self, receipt_home):
        """Simulated crash: begin + steps, then the process dies (fresh module
        state). A durable record is fine; a false success is not — the reader
        the Desktop uses (#92780 reads read_latest_receipt) must never be able
        to interpret the crash as a completed update."""
        ur.begin_update_receipt()
        ur.record_step("git_pull", True)
        ur.record_step("pip_install", True)
        # Crash: module singleton is gone, finalize never ran.
        ur._current = None

        records = _per_run_files(receipt_home)
        assert len(records) == 1
        durable = json.loads(records[0].read_text(encoding="utf-8"))
        assert durable["outcome"] == "running"
        assert durable["outcome"] != "success"
        assert durable["finished_at"] is None
        # The reader must not see a success either way (no receipt at all, or a
        # non-terminal one — never `success`).
        latest = ur.read_latest_receipt()
        assert latest is None or latest.get("outcome") != "success"

    def test_boundary_safety_net_records_crash_as_not_success(
        self, receipt_home
    ):
        """When the command boundary DOES catch the unwind (#91283), a
        non-zero exit finalizes as failed — never success."""
        ur.begin_update_receipt()
        ur.record_step("git_pull", False, "network died")
        path = ur.finalize_pending_update_receipt(1, "sys.exit(1)")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["outcome"] == "failed"
        assert ur.read_latest_receipt()["outcome"] == "failed"


class TestFailedReceiptSurvivesStateLoss:
    """#112465 — a failed run must still leave a terminal receipt when module
    state is lost. ``_current`` is the only other copy of the run, so losing it
    used to mean the failed update vanished entirely (no ``update_*.json``, a
    ``latest.json`` still pointing at an older update)."""

    def _lost_singleton_run(self):
        """A run that got as far as a failed restart/settlement, then lost the
        module singleton before anything was finalized."""
        ur.begin_update_receipt()
        ur.record_step("git_pull", True, "updated checkout")
        ur.record_step("gateway_restart", False, "settlement check failed closed")
        ur._current = None  # module eviction / re-import before the boundary
        return ur._receipt_dir()

    def test_failed_receipt_is_recovered_and_finalized(self, receipt_home):
        self._lost_singleton_run()

        path = ur.finalize_pending_update_receipt(1, "sys.exit(1)")

        assert path is not None and path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["outcome"] == "failed"
        assert payload["exit_code"] == 1
        assert payload["stop_reason"] == "sys.exit(1)"
        assert payload["finished_at"] is not None
        # Receipt metadata survives the loss: identity, steps, reason, timestamp.
        assert payload["update_id"]
        assert payload["pid"] == os.getpid()
        assert payload["started_at"]
        assert [step["name"] for step in payload["steps"]] == ["git_pull", "gateway_restart"]
        assert payload["steps"][1]["detail"] == "settlement check failed closed"

        latest = ur.read_latest_receipt()
        assert latest is not None and latest["outcome"] == "failed"

    def test_recovery_rewrites_the_same_record_in_place(self, receipt_home):
        """No dangling ``running`` record is left next to the recovered one —
        the reader would keep polling a run that can never finish."""
        self._lost_singleton_run()
        seeded = _per_run_files(receipt_home)
        assert len(seeded) == 1

        ur.finalize_pending_update_receipt(1, "sys.exit(1)")

        assert _per_run_files(receipt_home) == seeded
        assert json.loads(seeded[0].read_text(encoding="utf-8"))["outcome"] == "failed"

    def test_refused_receipt_is_recovered_with_its_exit_code(self, receipt_home):
        """Exit 2 (preflight refusal) keeps its own vocabulary through recovery."""
        ur.begin_update_receipt()
        ur.record_step("windows_preflight", False, "another hermes.exe running")
        ur._current = None

        path = ur.finalize_pending_update_receipt(2, "sys.exit(2)")

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["outcome"] == "refused"
        assert payload["exit_code"] == 2
        assert payload["stop_reason"] == "sys.exit(2)"

    def test_recovery_never_resurrects_a_successful_receipt(self, receipt_home):
        """A finalized success must not be rewritten as failed by a later
        boundary call in the same process (the failure mode this fix could
        introduce if recovery were less narrow)."""
        ur.begin_update_receipt()
        ur.record_step("git_pull", True)
        success_path = ur.finalize_update_receipt("success")
        assert json.loads(success_path.read_text(encoding="utf-8"))["outcome"] == "success"

        ur._current = None
        assert ur.finalize_pending_update_receipt(1, "sys.exit(1)") is None

        assert json.loads(success_path.read_text(encoding="utf-8"))["outcome"] == "success"
        assert ur.read_latest_receipt()["outcome"] == "success"

    def test_recovery_ignores_another_process_running_record(self, receipt_home):
        """Only OUR pid's in-progress record is admissible: a concurrent
        updater's record must never be finalized by this process."""
        directory = receipt_home / "logs" / "update_receipts"
        directory.mkdir(parents=True)
        other_pid = os.getpid() + 1
        other = directory / f"update_20260101_000000_{other_pid}.json"
        other.write_text(json.dumps({
            "schema": 1, "update_id": f"20260101_000000_{other_pid}", "pid": other_pid,
            "started_at": "2026-01-01T00:00:00+00:00", "finished_at": None,
            "outcome": "running", "steps": [],
        }), encoding="utf-8")

        assert ur.finalize_pending_update_receipt(1, "sys.exit(1)") is None

        assert json.loads(other.read_text(encoding="utf-8"))["outcome"] == "running"
        assert ur.read_latest_receipt() is None

    def test_recovery_skips_corrupt_partial_and_foreign_records(self, receipt_home):
        """Corrupt/partial records degrade to "skipped", never to a crash, and
        never get rewritten — the healthy record for this pid is still found."""
        self._lost_singleton_run()
        directory = ur._receipt_dir()
        healthy = _per_run_files(receipt_home)[0]
        future = time.time() + 60  # newest: recovery meets the decoys first
        decoys = {
            "torn": directory / f"update_19990101_000000_{os.getpid()}.json",
            "list": directory / f"update_19990101_000001_{os.getpid()}.json",
            "foreign": directory / f"update_19990101_000002_{os.getpid() + 1}.json",
        }
        decoys["torn"].write_text('{"schema": 1, "pid": 1, "outcome": "run', encoding="utf-8")
        decoys["list"].write_text("[1, 2, 3]", encoding="utf-8")
        decoys["foreign"].write_text(json.dumps({
            "schema": 1, "pid": os.getpid() + 1, "outcome": "running", "finished_at": None,
        }), encoding="utf-8")
        for path in decoys.values():
            os.utime(path, (future, future))

        path = ur.finalize_pending_update_receipt(1, "sys.exit(1)")

        assert path == healthy
        assert json.loads(healthy.read_text(encoding="utf-8"))["outcome"] == "failed"
        assert decoys["torn"].read_text(encoding="utf-8").startswith('{"schema": 1, "pid": 1')
        assert json.loads(decoys["list"].read_text(encoding="utf-8")) == [1, 2, 3]
        assert json.loads(decoys["foreign"].read_text(encoding="utf-8"))["outcome"] == "running"

    def test_two_runs_in_one_second_keep_separate_records(self, receipt_home):
        """`update_id` is second-resolution and per-pid (the refusal path can run
        right before a real update), so a later run must not overwrite an
        earlier run's record — including a finalized one."""
        ur.begin_update_receipt()
        ur.record_step("venv_preflight", False, "another hermes holds the venv")
        refused_path = ur.finalize_pending_update_receipt(2, "sys.exit(2)")

        ur.begin_update_receipt()
        ur.record_step("git_pull", False, "remote unreachable")
        failed_path = ur.finalize_pending_update_receipt(1, "sys.exit(1)")

        assert refused_path != failed_path
        assert json.loads(refused_path.read_text(encoding="utf-8"))["outcome"] == "refused"
        assert json.loads(failed_path.read_text(encoding="utf-8"))["outcome"] == "failed"
        assert ur.read_latest_receipt()["outcome"] == "failed"
        assert len(_per_run_files(receipt_home)) == 2


    def test_in_flight_run_does_not_hide_the_previous_receipt(self, receipt_home):
        """`latest.json` names the last FINALIZED run: a new run starting up must
        not overwrite a previous interrupted update's stale-plan obligation
        (#98022)."""
        ur.begin_update_receipt()
        ur.record_step("git_pull", False, "network died")
        ur.finalize_update_receipt("failed", stop_reason="KeyboardInterrupt: ")

        ur.begin_update_receipt()
        ur.record_step("preflight", True)

        latest = ur.read_latest_receipt()
        assert latest is not None
        assert latest["outcome"] == "failed"
        assert latest["stop_reason"] == "KeyboardInterrupt: "

    def test_plan_joins_the_durable_record(self, receipt_home):
        """The pre-update plan is the restart worklist; recording it must reach
        disk, not just the in-memory copy that state loss takes away (#112465)."""
        from hermes_cli.update_inventory import UpdatePlan, record_plan_in_receipt

        plan = UpdatePlan(install_method="git", updatable_in_place=True)
        plan.profiles = ["default"]
        plan.runtimes = _THREE_RUNTIMES[:1]

        ur.begin_update_receipt()
        record_plan_in_receipt(plan)

        payload = json.loads(_per_run_files(receipt_home)[0].read_text(encoding="utf-8"))
        assert payload["plan"]["runtimes"][0]["pid"] == 101
        assert payload["outcome"] == "running"


class TestReceiptPersistenceIsAtomicAndVisible:
    """The durable record is the only copy of a run, so a write must be atomic
    and a FAILED write must be visible instead of collapsing into the same
    ``None`` as "no receipt was open" (#112465)."""

    def test_no_temp_file_or_partial_record_is_left_behind(self, receipt_home):
        ur.begin_update_receipt()
        ur.record_step("git_pull", True)
        path = ur.finalize_update_receipt("success")

        directory = path.parent
        assert sorted(p.name for p in directory.iterdir()) == sorted(["latest.json", path.name])
        assert list(directory.glob("*.tmp")) == []
        json.loads(path.read_text(encoding="utf-8"))
        json.loads((directory / "latest.json").read_text(encoding="utf-8"))

    def test_failed_write_leaves_the_previous_record_intact(
        self, receipt_home, monkeypatch, capsys
    ):
        """A write that dies mid-flight must not leave a torn record where a
        reader or recovery would parse it."""
        ur.begin_update_receipt()
        ur.record_step("git_pull", True)
        seeded = _per_run_files(receipt_home)[0]
        before = seeded.read_text(encoding="utf-8")

        def _no_replace(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(ur.os, "replace", _no_replace)

        assert ur.finalize_update_receipt("failed") is None

        assert "Update receipt write failed: disk full" in capsys.readouterr().out
        assert seeded.read_text(encoding="utf-8") == before
        assert json.loads(seeded.read_text(encoding="utf-8"))["outcome"] == "running"
        assert list(seeded.parent.glob("*.tmp")) == []

    def test_unwritable_receipt_dir_is_reported_not_swallowed(
        self, receipt_home, capsys
    ):
        """The dir cannot be created: the run still proceeds, both write
        attempts are visible, and ``None`` is not silently indistinguishable
        from "no receipt was open"."""
        logs = receipt_home / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "update_receipts").write_text("not a directory", encoding="utf-8")

        ur.begin_update_receipt()  # must not raise
        path = ur.finalize_update_receipt("failed")

        out = capsys.readouterr().out
        assert path is None
        assert "Update receipt could not be persisted" in out
        assert "Update receipt write failed" in out
        assert ur.read_latest_receipt() is None


class TestSuccessImpliesAccounting:
    """Invariant 2 — #92902: the plan is the worklist. A planned runtime
    with no restart bookkeeping forbids a success outcome."""

    def _drive_decision(self, plan, **bookkeeping):
        """The real #92902 flow: reconcile, decide, finalize — mirroring
        update_cmd.py ~8331-8361 with the real decision functions."""
        outcomes = match_runtime_outcomes(plan, **bookkeeping)
        incomplete = report_unaccounted_runtimes(outcomes)
        record_plan_in_receipt(plan)
        if ur._current is not None:
            ur._current.data["runtime_outcomes"] = outcomes
        # Exact outcome-selection expression from update_cmd.py ~8360.
        path = ur.finalize_update_receipt(
            "partial" if incomplete else "success"
        )
        return outcomes, incomplete, path

    def test_n_minus_one_confirmations_refuses_success(self, receipt_home):
        """Plan of 3 runtimes, bookkeeping accounts for only 2: the verify
        layer must escalate and the receipt must NOT say success."""
        ur.begin_update_receipt()
        plan = _plan_with_runtimes(_THREE_RUNTIMES)
        outcomes, incomplete, path = self._drive_decision(
            plan,
            restarted_services=[],
            relaunched_profiles=["work"],   # 102 accounted
            externally_supervised_profiles=[],
            killed_pids={101},              # 101 accounted; 103 missed
            failed_units=[],
        )
        assert incomplete is True, (
            "report_unaccounted_runtimes accepted a plan with a silently "
            "missed runtime — the #92902 invariant is broken"
        )
        missed = [o for o in outcomes if o["outcome"] == "unaccounted"]
        assert [o["pid"] for o in missed] == [103]

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["outcome"] != "success"
        assert payload["outcome"] == "partial"

    def test_full_accounting_permits_success_with_evidence(
        self, receipt_home
    ):
        """All planned runtimes accounted → success is allowed, and the
        receipt carries the plan + per-runtime outcomes as evidence."""
        ur.begin_update_receipt()
        plan = _plan_with_runtimes(_THREE_RUNTIMES)
        outcomes, incomplete, path = self._drive_decision(
            plan,
            # Serve/dashboard runtimes are reconciled in their own unit
            # vocabulary and never borrow a gateway relaunch (#100479).
            restarted_services=["hermes-serve-ops.service"],
            relaunched_profiles=["work"],
            externally_supervised_profiles=[],
            killed_pids={101},
            failed_units=[],
        )
        assert incomplete is False
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["outcome"] == "success"
        # SUCCESS IMPLIES ACCOUNTING: the fleet/verification fields exist
        # and cover every planned runtime, none unaccounted.
        assert len(payload["plan"]["runtimes"]) == 3
        assert len(payload["runtime_outcomes"]) == 3
        assert all(
            o["outcome"] != "unaccounted" for o in payload["runtime_outcomes"]
        )

    def test_failed_unit_also_refuses_success(self, receipt_home):
        """A runtime whose restart FAILED (not merely missed) must also
        surface — outcome 'failed' in the reconciliation rows."""
        ur.begin_update_receipt()
        plan = _plan_with_runtimes(_THREE_RUNTIMES[:1])
        outcomes = match_runtime_outcomes(
            plan,
            restarted_services=[],
            relaunched_profiles=[],
            externally_supervised_profiles=[],
            killed_pids=set(),
            failed_units=["hermes-gateway.service"],
        )
        assert outcomes[0]["outcome"] == "failed"
        ur.finalize_update_receipt("partial")


class TestRefusalIsNotFailure:
    """Invariant 3 — #91439: refusal (exit 2) and failure are distinct
    outcomes, and the reader reports which one happened."""

    def test_refusal_receipt_distinguishable_from_failure(
        self, receipt_home
    ):
        # Run 1: a real failure.
        ur.begin_update_receipt()
        ur.record_step("git_fetch", False, "remote unreachable")
        failed_path = ur.finalize_pending_update_receipt(1, "sys.exit(1)")
        failed = json.loads(failed_path.read_text(encoding="utf-8"))

        # Run 2: a preflight refusal (venv-holder / concurrent instance).
        ur.begin_update_receipt()
        ur.record_step("venv_preflight", False, "another hermes holds venv")
        refused_path = ur.finalize_pending_update_receipt(2, "sys.exit(2)")
        refused = json.loads(refused_path.read_text(encoding="utf-8"))

        assert failed["outcome"] == "failed" and failed["exit_code"] == 1
        assert refused["outcome"] == "refused" and refused["exit_code"] == 2
        assert refused["outcome"] != failed["outcome"]

        # The reader (Desktop path, #92780/#91439) reports refusal as
        # refusal — not failure, and certainly not success.
        latest = ur.read_latest_receipt()
        assert latest["outcome"] == "refused"
        assert latest["stop_reason"] == "sys.exit(2)"

    def test_refused_receipt_survives_with_its_steps(self, receipt_home):
        """#91439: the refused run's receipt keeps the evidence of WHY —
        the failing preflight step is preserved, not lost."""
        ur.begin_update_receipt()
        ur.record_step("windows_preflight", False, "hermes.exe running")
        path = ur.finalize_pending_update_receipt(2, "concurrent instance")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["steps"][0]["name"] == "windows_preflight"
        assert payload["steps"][0]["ok"] is False
        assert payload["outcome"] == "refused"
