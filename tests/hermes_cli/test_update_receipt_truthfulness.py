"""Truthfulness invariants for the update pipeline at the RECEIPT layer.

Bug class ("update reports success while reality disagrees" — #88654,
#88848, #91378, #91439, #91962, #92780, #92902): the updater's word must be
backed by evidence. This suite pins the honesty contract of the receipt
subsystem with REAL module functions against a temp HERMES_HOME — it is not
a fleet E2E (that lives in the CI install/update harness).

Invariants pinned, and WHERE each is enforced:

1. RECEIPT ALWAYS FINALIZED — ``begin → steps → finalize`` writes a
   parseable receipt for success/failed/refused (#91283 made every
   post-begin run leave a record). A begun-but-never-finalized run
   (simulated crash) writes NOTHING to disk, so the reader the Desktop
   uses (``read_latest_receipt``, surfaced via
   ``/api/hermes/update/receipt`` — #92780) can never interpret a crash
   as success. The HTTP-layer gating of an ``outcome == "running"``
   receipt is already pinned in test_update_receipt_endpoint.py and is
   deliberately not duplicated here.

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
    monkeypatch.setenv("HERMES_HOME", str(home))
    ur._current = None
    yield home
    ur._current = None


def _receipt_files(home):
    directory = home / "logs" / "update_receipts"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def _pinned_intent(target="a" * 40, prior="b" * 40, branch="main"):
    return {
        "target": target,
        "install_id": "1" * 32,
        "correlation_id": "c" * 32,
        "prior_sha": prior,
        "branch": branch,
    }


@pytest.mark.parametrize("outcome", ["success", "partial"])
def test_pinned_terminal_success_requires_exact_post_swap_proof(receipt_home, outcome):
    """Pinned terminal success is impossible without target/install proof."""
    intent = _pinned_intent()
    ur.begin_update_receipt(intent=intent)

    path = ur.finalize_update_receipt(outcome)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["outcome"] == "failed"
    assert "post-swap-proof-missing" in payload["failure_reasons"]


def test_pinned_post_swap_proof_must_match_immutable_intent(receipt_home):
    """Caller-supplied proof cannot certify a different target or install."""
    intent = _pinned_intent()
    ur.begin_update_receipt(intent=intent)
    ur.record_pinned_post_swap(
        post_sha="d" * 40, post_install_id="2" * 32, verified=True
    )

    path = ur.finalize_update_receipt("success")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["outcome"] == "failed"
    assert payload["pinned_post_verified"] is False
    assert "post-swap-proof-mismatch" in payload["failure_reasons"]


def test_receipt_home_is_production_persistence_target(receipt_home):
    """An empty-directory assertion must inspect the writer's actual target."""
    assert ur._receipt_dir().resolve() == (
        receipt_home / "logs" / "update_receipts"
    ).resolve()
    ur.begin_update_receipt()
    assert _receipt_files(receipt_home) == []
    path = ur.finalize_update_receipt("success")
    assert path is not None
    assert path in _receipt_files(receipt_home)
    assert path.parent == receipt_home / "logs" / "update_receipts"
    assert ur.read_latest_receipt() == json.loads(path.read_text(encoding="utf-8"))


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

    def test_nothing_on_disk_until_finalize(self, receipt_home):
        """The receipt is written atomically at finalize — a run that is
        still going (or that dies) has NO on-disk artifact to misread."""
        ur.begin_update_receipt()
        ur.record_step("pre_update_backup", True)
        assert _receipt_files(receipt_home) == []

    def test_crash_without_finalize_never_claims_success(self, receipt_home):
        """Simulated crash: begin + steps, then the process dies (fresh
        module state). The Desktop's reader (#92780 reads the receipt via
        read_latest_receipt) must see NO successful update."""
        ur.begin_update_receipt()
        ur.record_step("git_pull", True)
        ur.record_step("pip_install", True)
        # Crash: module singleton is gone, finalize never ran.
        ur._current = None

        assert _receipt_files(receipt_home) == []
        latest = ur.read_latest_receipt()
        # No receipt at all — the reader cannot report success. If this
        # ever returns a dict, it must not claim a completed success.
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


class TestPinnedIntentTruthfulness:
    """T4: correlation and exact post-swap evidence gate the receipt."""

    _INTENT = {
        "target": "a" * 40,
        "install_id": "0" * 32,
        "correlation_id": "r" * 32,
        "prior_sha": "b" * 40,
        "branch": "main",
    }

    def test_dependency_failure_cannot_be_rewritten_as_success(self, receipt_home):
        ur.begin_update_receipt(intent=self._INTENT)
        ur.record_failure("dependency-failure: pip")
        path = ur.finalize_update_receipt("success")

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["outcome"] == "failed"
        assert payload["update_intent"] == self._INTENT
        assert payload["requested_sha"] == self._INTENT["target"]
        assert payload["prior_sha"] == self._INTENT["prior_sha"]
        assert payload["correlation_id"] == self._INTENT["correlation_id"]

    def test_post_swap_identity_is_correlated_to_requested_target(self, receipt_home):
        ur.begin_update_receipt(intent=self._INTENT)
        ur.record_pinned_post_swap(
            post_sha=self._INTENT["target"],
            post_install_id=self._INTENT["install_id"],
            verified=True,
        )
        path = ur.finalize_update_receipt("success")

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["outcome"] == "success"
        assert payload["post_sha"] == payload["requested_sha"]
        assert payload["post_install_id"] == payload["install_id"]
        assert payload["correlation_id"] == self._INTENT["correlation_id"]



def test_pinned_intent_is_correlated_and_immutable_through_resume(receipt_home):
    intent = {
        "target": "a" * 40,
        "install_id": "0123456789abcdef0123456789abcdef",
        "correlation_id": "c" * 32,
        "prior_sha": "b" * 40,
        "branch": "main",
    }
    ur.begin_update_receipt(intent=intent)
    detached = ur.detach_update_receipt()
    intent["target"] = "d" * 40
    ur.resume_update_receipt(detached)
    path = ur.finalize_update_receipt("success")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["update_intent"]["target"] == "a" * 40
    assert payload["requested_sha"] == "a" * 40
    assert payload["pre_sha"] == "b" * 40
    assert payload["correlation_id"] == "c" * 32


@pytest.mark.parametrize("reason", ["refusal", "dependency", "restart", "import"])
def test_recorded_pinned_failure_cannot_be_reported_success(receipt_home, reason):
    intent = {
        "target": "a" * 40,
        "install_id": "0123456789abcdef0123456789abcdef",
        "correlation_id": "d" * 32,
        "prior_sha": "b" * 40,
        "branch": "main",
    }
    ur.begin_update_receipt(intent=intent)
    if reason == "refusal":
        ur.record_refusal("target refused")
    else:
        ur.record_failure(reason + " failure")

    path = ur.finalize_update_receipt("success")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["outcome"] == ("refused" if reason == "refusal" else "failed")
