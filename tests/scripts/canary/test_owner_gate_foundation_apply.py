from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts.canary import owner_gate_foundation as gate
from scripts.canary import owner_gate_foundation_apply as apply
from scripts.canary import owner_gate_foundation_journal as journal
from scripts.canary import owner_gate_package as package
from scripts.canary import owner_gate_trust as trust
from tests.scripts.canary import test_owner_gate_pre_foundation as helpers


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@pytest.fixture(autouse=True)
def _pin_release_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        helpers.RELEASE_KEY_ID,
    )


def _chain() -> apply.ValidatedFoundationAChain:
    authority, _plan, _evidence = helpers._authority()
    return apply.decode_validated_foundation_a_chain(
        pre_foundation_authority_raw=gate.canonical_json_bytes(authority),
        owner_reauthentication_receipt_raw=gate.canonical_json_bytes(
            helpers._owner_reauth_receipt()
        ),
        network_evidence_raw=gate.canonical_json_bytes(
            helpers._signed_network_evidence()
        ),
        project_ancestry_evidence_raw=helpers._signed_ancestry_raw(),
        release_public_key=helpers.RELEASE_KEY.public_key(),
        network_collector_public_key=helpers.NETWORK_KEY.public_key(),
        project_ancestry_collector_public_key=(
            helpers.NETWORK_KEY.public_key()
        ),
        now_unix=helpers.NOW + 1,
    )


def _journal_for_test(root: Path) -> journal.FoundationApplyJournal:
    parent = root.parent.stat()
    store = journal.FoundationApplyJournal(
        _root=root,
        _owner_uid=parent.st_uid,
        _owner_gid=parent.st_gid,
    )
    store._require_owner_process = lambda: None  # type: ignore[method-assign]
    return store


def _load_source_recovery(
    chain: apply.ValidatedFoundationAChain,
) -> apply.ValidatedFoundationApplyChain:
    return apply._load_validated_foundation_apply_chain_for_source_recovery(
        pre_foundation_authority_raw=chain.pre_foundation_authority_raw,
        owner_reauthentication_receipt_raw=(
            chain.owner_reauthentication_receipt_raw
        ),
        network_evidence_raw=chain.network_evidence_raw,
        project_ancestry_evidence_raw=chain.ancestry_evidence_raw,
        release_public_key=chain.release_public_key,
        network_collector_public_key=chain.network_collector_public_key,
        project_ancestry_collector_public_key=(
            chain.ancestry_collector_public_key
        ),
    )


def _policy_precondition(
    step: gate.PlanStep,
    plan: gate.OwnerGateFoundationPlan,
    *,
    etag: str,
    exact: bool,
) -> Mapping[str, Any] | None:
    contract = apply._iam_binding_contract(step, plan=plan)
    if contract is None:
        return None
    resource_name, role, member = contract
    bindings: list[Mapping[str, Any]] = [
        {"role": "roles/viewer", "members": ["group:auditors@example.com"]}
    ]
    if exact:
        bindings.append({"role": role, "members": [member]})
    return {
        "resource_name": resource_name,
        "policy_etag": etag,
        "policy_version": 3,
        "policy_bindings": apply._canonical_inventory(bindings),
        "policy_audit_configs": [{
            "service": "allServices",
            "auditLogConfigs": [{"logType": "ADMIN_READ"}],
        }],
    }


class _FakeProvider:
    def __init__(
        self,
        chain: apply.ValidatedFoundationAChain,
        *,
        fail_step: str | None = None,
        failure_state: str = "failed",
        rollback_failure_step: str | None = None,
        crash_step: str | None = None,
        crash_rollback_step: str | None = None,
        authority_drift_after_failure: bool = False,
        shared_execute_count: Any | None = None,
        first_execute_entered: Any | None = None,
        release_first_execute: Any | None = None,
    ) -> None:
        self.chain = chain
        self.plan = chain.plan
        self.states = {step.name: "absent" for step in self.plan.foundation_steps}
        self.policy_etags = {
            step.name: "etag-policy-0"
            for step in self.plan.foundation_steps
            if step.name.startswith("bind_narrow_")
        }
        self.fail_step = fail_step
        self.failure_state = failure_state
        self.rollback_failure_step = rollback_failure_step
        self.crash_step = crash_step
        self.crash_rollback_step = crash_rollback_step
        self.authority_drift_after_failure = authority_drift_after_failure
        self.failed = False
        self.execute_calls: list[str] = []
        self.rollback_calls: list[str] = []
        self.shared_execute_count = shared_execute_count
        self.first_execute_entered = first_execute_entered
        self.release_first_execute = release_first_execute

    def assert_stable(self) -> None:
        if self.authority_drift_after_failure and self.failed:
            raise apply.OwnerGateFoundationApplyError(
                "owner_gate_foundation_provider_runtime_changed"
            )

    def observe_ancestry_chain(self) -> list[Mapping[str, Any]]:
        return [dict(item) for item in self.chain.ancestry_evidence.ordered_chain]

    def inspect_resource(
        self,
        step: gate.PlanStep,
        *,
        plan: gate.OwnerGateFoundationPlan,
    ) -> apply.ResourceObservation:
        assert plan is self.plan
        state = self.states[step.name]
        if (
            state == "absent"
            and self.shared_execute_count is not None
            and self.shared_execute_count.value
            == len(self.plan.foundation_steps)
        ):
            state = "exact"
            if step.name.startswith("bind_narrow_"):
                self.policy_etags[step.name] = "etag-policy-1"
        receipt = _digest(f"inspect:{step.name}:{state}:{self.policy_etags.get(step.name)}")
        precondition = _policy_precondition(
            step,
            plan,
            etag=self.policy_etags.get(step.name, "unused"),
            exact=state == "exact",
        )
        if state == "exact":
            identity = helpers._resource_identity(step.name, plan)
            if step.name.startswith("bind_narrow_"):
                identity["policy_etag"] = self.policy_etags[step.name]
            return apply.ResourceObservation(
                "exact",
                receipt,
                resource_identity=identity,
                precondition=precondition,
            )
        return apply.ResourceObservation(
            "absent",
            receipt,
            precondition=precondition,
        )

    def execute_step(
        self,
        step: gate.PlanStep,
        *,
        plan: gate.OwnerGateFoundationPlan,
        attempt_id: str,
        precondition: Mapping[str, Any] | None,
    ) -> apply.OperationObservation:
        assert plan is self.plan
        self.execute_calls.append(step.name)
        if self.shared_execute_count is not None:
            with self.shared_execute_count.get_lock():
                self.shared_execute_count.value += 1
                first = self.shared_execute_count.value == 1
            if first and self.first_execute_entered is not None:
                self.first_execute_entered.set()
                assert self.release_first_execute.wait(10)
        if step.name == self.crash_step:
            raise SystemExit("simulated_process_death_after_intent")
        if step.name == self.fail_step:
            self.failed = True
            return apply.OperationObservation(
                self.failure_state,
                _digest(f"operation:{step.name}:{self.failure_state}"),
                attempt_id,
            )
        self.states[step.name] = "exact"
        pre_etag = None
        post_etag = None
        if step.name.startswith("bind_narrow_"):
            assert precondition is not None
            pre_etag = str(precondition["policy_etag"])
            post_etag = "etag-policy-1"
            self.policy_etags[step.name] = post_etag
        return apply.OperationObservation(
            "completed",
            _digest(f"operation:{step.name}:completed"),
            attempt_id,
            _digest(f"binding:{step.name}:completed"),
            pre_etag,
            post_etag,
        )

    def rollback_step(
        self,
        original_step: gate.PlanStep,
        rollback_step: gate.PlanStep,
        *,
        plan: gate.OwnerGateFoundationPlan,
        attempt_id: str,
        precondition: Mapping[str, Any] | None,
    ) -> apply.OperationObservation:
        assert plan is self.plan
        self.rollback_calls.append(original_step.name)
        if original_step.name == self.crash_rollback_step:
            raise SystemExit("simulated_process_death_after_rollback_intent")
        if original_step.name == self.rollback_failure_step:
            return apply.OperationObservation(
                "unknown",
                _digest(f"rollback:{original_step.name}:unknown"),
                attempt_id,
            )
        self.states[original_step.name] = "absent"
        pre_etag = None
        post_etag = None
        if original_step.name.startswith("bind_narrow_"):
            assert precondition is not None
            pre_etag = str(precondition["policy_etag"])
            post_etag = "etag-policy-2"
            self.policy_etags[original_step.name] = post_etag
        return apply.OperationObservation(
            "completed",
            _digest(f"rollback:{original_step.name}:completed"),
            attempt_id,
            _digest(f"rollback-binding:{original_step.name}:completed"),
            pre_etag,
            post_etag,
        )


class _PreflightRuntimeFailureProvider(_FakeProvider):
    def assert_stable(self) -> None:
        raise apply.OwnerGateFoundationApplyError(
            "owner_gate_foundation_provider_runtime_changed"
        )


class _DelayedVisibilityProvider(_FakeProvider):
    def __init__(
        self,
        chain: apply.ValidatedFoundationAChain,
        *,
        hidden_reads: int,
    ) -> None:
        super().__init__(chain)
        self.hidden_reads = hidden_reads

    def inspect_resource(
        self,
        step: gate.PlanStep,
        *,
        plan: gate.OwnerGateFoundationPlan,
    ) -> apply.ResourceObservation:
        if (
            step.name == self.plan.foundation_steps[0].name
            and self.states[step.name] == "exact"
            and self.hidden_reads > 0
        ):
            self.hidden_reads -= 1
            return apply.ResourceObservation(
                "absent",
                _digest(f"delayed:{step.name}:{self.hidden_reads}"),
            )
        return super().inspect_resource(step, plan=plan)


class _DriftingPostconditionProvider(_FakeProvider):
    def __init__(self, chain: apply.ValidatedFoundationAChain) -> None:
        super().__init__(chain)
        self.returned_drift = False

    def inspect_resource(
        self,
        step: gate.PlanStep,
        *,
        plan: gate.OwnerGateFoundationPlan,
    ) -> apply.ResourceObservation:
        if (
            step.name == self.plan.foundation_steps[0].name
            and self.states[step.name] == "exact"
            and not self.returned_drift
        ):
            self.returned_drift = True
            return apply.ResourceObservation(
                "drift",
                _digest(f"drift:{step.name}"),
            )
        return super().inspect_resource(step, plan=plan)


class _RollbackPostconditionProvider(_FakeProvider):
    def __init__(
        self,
        chain: apply.ValidatedFoundationAChain,
        *,
        rollback_post_states: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(chain, **kwargs)
        self.rollback_post_states = list(rollback_post_states)
        self.rollback_target: str | None = None
        self.rollback_inspections = 0

    def rollback_step(
        self,
        original_step: gate.PlanStep,
        rollback_step: gate.PlanStep,
        *,
        plan: gate.OwnerGateFoundationPlan,
        attempt_id: str,
        precondition: Mapping[str, Any] | None,
    ) -> apply.OperationObservation:
        operation = super().rollback_step(
            original_step,
            rollback_step,
            plan=plan,
            attempt_id=attempt_id,
            precondition=precondition,
        )
        self.rollback_target = original_step.name
        return operation

    def inspect_resource(
        self,
        step: gate.PlanStep,
        *,
        plan: gate.OwnerGateFoundationPlan,
    ) -> apply.ResourceObservation:
        if (
            step.name == self.rollback_target
            and self.rollback_post_states
        ):
            self.rollback_inspections += 1
            state = self.rollback_post_states.pop(0)
            if state == "exact":
                stored = self.states[step.name]
                self.states[step.name] = "exact"
                try:
                    return super().inspect_resource(step, plan=plan)
                finally:
                    self.states[step.name] = stored
            if state in {"drift", "unknown"}:
                return apply.ResourceObservation(
                    state,
                    _digest(
                        f"rollback-post:{step.name}:{state}:"
                        f"{self.rollback_inspections}"
                    ),
                )
            assert state == "absent"
        return super().inspect_resource(step, plan=plan)


def test_completed_create_waits_for_exact_provider_visibility(
    tmp_path: Path,
) -> None:
    chain = _chain()
    provider = _DelayedVisibilityProvider(chain, hidden_reads=3)
    waits: list[float] = []

    receipt = apply._apply_with_provider(
        chain=chain,
        private_key=helpers.RELEASE_KEY,
        provider=provider,
        journal=_journal_for_test(tmp_path / "journal"),
        now_unix=lambda: helpers.NOW + 2,
        postcondition_wait=waits.append,
    )

    assert len(receipt["applied_steps"]) == len(
        chain.plan.foundation_steps
    )
    assert waits == [apply._POSTCONDITION_VISIBILITY_DELAY_SECONDS] * 3
    assert provider.execute_calls == [
        step.name for step in chain.plan.foundation_steps
    ]


def test_completed_create_visibility_wait_is_finite_and_fail_closed(
    tmp_path: Path,
) -> None:
    chain = _chain()
    provider = _DelayedVisibilityProvider(
        chain,
        hidden_reads=apply._POSTCONDITION_VISIBILITY_ATTEMPTS + 1,
    )
    waits: list[float] = []

    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=provider,
            journal=_journal_for_test(tmp_path / "journal"),
            now_unix=lambda: helpers.NOW + 2,
            postcondition_wait=waits.append,
        )

    assert caught.value.receipt["failure_code"] == (
        "owner_gate_foundation_postcondition_not_exact"
    )
    assert caught.value.receipt["terminal_state"] == (
        "manual_reconciliation_required"
    )
    assert waits == [apply._POSTCONDITION_VISIBILITY_DELAY_SECONDS] * (
        apply._POSTCONDITION_VISIBILITY_ATTEMPTS - 1
    )


def test_completed_create_never_retries_postcondition_drift(
    tmp_path: Path,
) -> None:
    chain = _chain()
    provider = _DriftingPostconditionProvider(chain)
    waits: list[float] = []

    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=provider,
            journal=_journal_for_test(tmp_path / "journal"),
            now_unix=lambda: helpers.NOW + 2,
            postcondition_wait=waits.append,
        )

    assert caught.value.receipt["failure_code"] == (
        "owner_gate_foundation_postcondition_not_exact"
    )
    assert waits == []


def test_completed_rollback_waits_for_absent_provider_visibility(
    tmp_path: Path,
) -> None:
    chain = _chain()
    created = chain.plan.foundation_steps[0]
    failed = chain.plan.foundation_steps[1]
    provider = _RollbackPostconditionProvider(
        chain,
        rollback_post_states=["exact"] * 3,
        fail_step=failed.name,
        failure_state="failed",
    )
    waits: list[float] = []

    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=provider,
            journal=_journal_for_test(tmp_path / "journal"),
            now_unix=lambda: helpers.NOW + 2,
            postcondition_wait=waits.append,
        )

    assert provider.rollback_calls == [created.name]
    assert provider.rollback_inspections == 3
    assert waits == [apply._POSTCONDITION_VISIBILITY_DELAY_SECONDS] * 3
    assert caught.value.receipt["terminal_state"] == "rolled_back_clean"
    assert [
        item["disposition"]
        for item in caught.value.receipt["rollback_step_receipts"]
    ] == ["rolled_back"]


def test_completed_rollback_never_retries_postcondition_drift(
    tmp_path: Path,
) -> None:
    chain = _chain()
    failed = chain.plan.foundation_steps[1]
    provider = _RollbackPostconditionProvider(
        chain,
        rollback_post_states=["drift"],
        fail_step=failed.name,
        failure_state="failed",
    )
    waits: list[float] = []

    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=provider,
            journal=_journal_for_test(tmp_path / "journal"),
            now_unix=lambda: helpers.NOW + 2,
            postcondition_wait=waits.append,
        )

    assert provider.rollback_inspections == 1
    assert waits == []
    assert caught.value.receipt["terminal_state"] == (
        "manual_reconciliation_required"
    )
    assert caught.value.receipt["rollback_step_receipts"][0][
        "disposition"
    ] == "rollback_unknown"


def test_noncompleted_rollback_observes_once_without_waiting(
    tmp_path: Path,
) -> None:
    chain = _chain()
    created = chain.plan.foundation_steps[0]
    failed = chain.plan.foundation_steps[1]
    provider = _RollbackPostconditionProvider(
        chain,
        rollback_post_states=["exact", "absent"],
        fail_step=failed.name,
        failure_state="failed",
        rollback_failure_step=created.name,
    )
    waits: list[float] = []

    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=provider,
            journal=_journal_for_test(tmp_path / "journal"),
            now_unix=lambda: helpers.NOW + 2,
            postcondition_wait=waits.append,
        )

    assert provider.rollback_inspections == 1
    assert provider.rollback_post_states == ["absent"]
    assert waits == []
    assert caught.value.receipt["terminal_state"] == (
        "manual_reconciliation_required"
    )
    assert caught.value.receipt["rollback_step_receipts"][0][
        "disposition"
    ] == "rollback_unknown"


def test_rollback_visibility_recheck_fails_when_reauth_expires(
    tmp_path: Path,
) -> None:
    chain = _chain()
    failed = chain.plan.foundation_steps[1]
    provider = _RollbackPostconditionProvider(
        chain,
        rollback_post_states=["exact", "absent"],
        fail_step=failed.name,
        failure_state="failed",
    )
    expired = False
    expires_at = int(
        chain.owner_reauthentication_receipt["expires_at_unix"]
    )
    waits: list[float] = []

    def now_unix() -> int:
        return expires_at + 1 if expired else helpers.NOW + 2

    def wait(seconds: float) -> None:
        nonlocal expired
        waits.append(seconds)
        expired = True

    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=provider,
            journal=_journal_for_test(tmp_path / "journal"),
            now_unix=now_unix,
            postcondition_wait=wait,
        )

    assert provider.rollback_inspections == 1
    assert provider.rollback_post_states == ["absent"]
    assert waits == [apply._POSTCONDITION_VISIBILITY_DELAY_SECONDS]
    assert caught.value.receipt["terminal_state"] == (
        "manual_reconciliation_required"
    )
    assert caught.value.receipt["rollback_step_receipts"][0][
        "disposition"
    ] == "rollback_unknown"


def test_completed_rollback_visibility_wait_is_finite_and_manual(
    tmp_path: Path,
) -> None:
    chain = _chain()
    failed = chain.plan.foundation_steps[1]
    provider = _RollbackPostconditionProvider(
        chain,
        rollback_post_states=["exact"]
        * (apply._POSTCONDITION_VISIBILITY_ATTEMPTS + 1),
        fail_step=failed.name,
        failure_state="failed",
    )
    waits: list[float] = []

    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=provider,
            journal=_journal_for_test(tmp_path / "journal"),
            now_unix=lambda: helpers.NOW + 2,
            postcondition_wait=waits.append,
        )

    assert provider.rollback_inspections == (
        apply._POSTCONDITION_VISIBILITY_ATTEMPTS
    )
    assert provider.rollback_post_states == ["exact"]
    assert waits == [apply._POSTCONDITION_VISIBILITY_DELAY_SECONDS] * (
        apply._POSTCONDITION_VISIBILITY_ATTEMPTS - 1
    )
    assert caught.value.receipt["terminal_state"] == (
        "manual_reconciliation_required"
    )
    assert caught.value.receipt["rollback_step_receipts"][0][
        "disposition"
    ] == "rollback_unknown"


def test_iam_cas_add_and_remove_use_real_policy_etag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain()
    step = chain.plan.foundation_steps[2]
    provider = object.__new__(apply._TrustedGcloudFoundationProvider)
    captured: list[Mapping[str, Any]] = []

    def request(*, resource_name: str, policy: Mapping[str, Any]):
        captured.append({"resource_name": resource_name, "policy": policy})
        response = {**dict(policy), "etag": f"etag-new-{len(captured)}"}
        return apply._IamPolicyHttpResponse(
            200,
            json.dumps(response, separators=(",", ":")).encode("ascii"),
        )

    monkeypatch.setattr(provider, "_request_iam_policy_cas", request)
    before = _policy_precondition(
        step,
        chain.plan,
        etag="etag-old",
        exact=False,
    )
    created = provider._iam_binding_operation(
        step,
        plan=chain.plan,
        attempt_id="1" * 64,
        precondition=before,
        add=True,
        logical_argv=step.argv,
    )
    assert created.state == "completed"
    assert created.cas_precondition_etag == "etag-old"
    assert created.cas_postcondition_etag == "etag-new-1"
    assert captured[0]["policy"]["etag"] == "etag-old"
    assert captured[0]["policy"]["auditConfigs"]

    after = apply._normalize_iam_policy(
        {**captured[0]["policy"], "etag": "etag-new-1"},
        resource_name=str(captured[0]["resource_name"]),
    )
    removed = provider._iam_binding_operation(
        step,
        plan=chain.plan,
        attempt_id="2" * 64,
        precondition=after,
        add=False,
        logical_argv=apply._rollback_step_for(step, plan=chain.plan).argv,
    )
    assert removed.state == "completed"
    assert removed.cas_precondition_etag == "etag-new-1"
    assert removed.cas_postcondition_etag == "etag-new-2"
    assert all(
        item.get("role") != chain.plan.spec.read_only_iam_role
        for item in captured[1]["policy"]["bindings"]
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (apply._IamPolicyHttpResponse(409, b""), "unknown"),
        (apply._IamPolicyHttpResponse(412, b""), "unknown"),
        (apply._IamPolicyHttpResponse(302, b""), "unknown"),
        (apply._IamPolicyHttpResponse(403, b""), "failed"),
        (apply._IamPolicyHttpResponse(200, b"not-json"), "unknown"),
        (
            apply._IamPolicyHttpResponse(
                200,
                b'{"etag":"a","etag":"b","version":3,"bindings":[],"auditConfigs":[]}',
            ),
            "unknown",
        ),
    ],
)
def test_iam_cas_ambiguous_or_invalid_response_never_completes(
    monkeypatch: pytest.MonkeyPatch,
    response: apply._IamPolicyHttpResponse,
    expected: str,
) -> None:
    chain = _chain()
    step = chain.plan.foundation_steps[2]
    provider = object.__new__(apply._TrustedGcloudFoundationProvider)
    monkeypatch.setattr(
        provider,
        "_request_iam_policy_cas",
        lambda **_kwargs: response,
    )
    operation = provider._iam_binding_operation(
        step,
        plan=chain.plan,
        attempt_id="3" * 64,
        precondition=_policy_precondition(
            step,
            chain.plan,
            etag="etag-old",
            exact=False,
        ),
        add=True,
        logical_argv=step.argv,
    )
    assert operation.state == expected
    assert operation.provider_result_binding_sha256 is None


def test_rest_iam_cas_disables_redirects_and_never_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opens = 0
    handlers: tuple[Any, ...] = ()

    class FakeOpener:
        def open(self, *_args: Any, **_kwargs: Any) -> Any:
            nonlocal opens
            opens += 1
            raise urllib.error.HTTPError(
                "https://cloudresourcemanager.googleapis.com/",
                302,
                "redirect",
                {"Location": "https://attacker.invalid/"},
                None,
            )

    def build_opener(*values: Any) -> FakeOpener:
        nonlocal handlers
        handlers = values
        return FakeOpener()

    for name in apply._FORBIDDEN_NETWORK_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(apply.launcher, "_reject_custom_ca_environment", lambda: None)
    monkeypatch.setattr(
        apply.launcher,
        "_pinned_system_tls_context",
        lambda: object(),
    )
    monkeypatch.setattr(apply.urllib.request, "build_opener", build_opener)
    result = apply._resource_manager_set_iam_policy(
        "token",
        f"projects/{gate.PROJECT}",
        {"etag": "old", "version": 3, "bindings": [], "auditConfigs": []},
    )
    assert result.status == 302
    assert opens == 1
    assert any(isinstance(item, apply._NoRedirectHandler) for item in handlers)


def test_rest_iam_cas_rejects_ambient_ssl_cert_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SSL_CERT_DIR", "/tmp/attacker-ca")
    called = False

    def context() -> Any:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(apply.launcher, "_pinned_system_tls_context", context)
    with pytest.raises(
        apply.OwnerGateFoundationApplyError,
        match="owner_gate_foundation_iam_cas_tls_invalid",
    ):
        apply._resource_manager_set_iam_policy(
            "token",
            f"projects/{gate.PROJECT}",
            {"etag": "old", "version": 3, "bindings": [], "auditConfigs": []},
        )
    assert called is False


def test_rest_iam_cas_non_json_2xx_is_unknown_without_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in apply._FORBIDDEN_NETWORK_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)

    class Response:
        status = 200
        headers = {"Content-Type": "text/plain", "Content-Length": "2"}

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def geturl(self) -> str:
            return (
                "https://cloudresourcemanager.googleapis.com/v3/"
                f"projects/{gate.PROJECT}:setIamPolicy"
            )

        def read(self, _maximum: int) -> bytes:
            raise AssertionError("non-JSON response body must not be read")

    class Opener:
        def open(self, *_args: Any, **_kwargs: Any) -> Response:
            return Response()

    monkeypatch.setattr(apply.launcher, "_reject_custom_ca_environment", lambda: None)
    monkeypatch.setattr(
        apply.launcher,
        "_pinned_system_tls_context",
        lambda: object(),
    )
    monkeypatch.setattr(
        apply.urllib.request,
        "build_opener",
        lambda *_handlers: Opener(),
    )
    response = apply._resource_manager_set_iam_policy(
        "token",
        f"projects/{gate.PROJECT}",
        {"etag": "old", "version": 3, "bindings": [], "auditConfigs": []},
    )
    assert response.status == 200
    assert response.transport_unknown is True


@pytest.mark.parametrize(
    ("step_name", "items"),
    [
        (
            "create_dedicated_private_owner_gate_subnet",
            [{
                "name": gate.OWNER_GATE_SUBNET_NAME,
                "selfLink": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{gate.PROJECT}/regions/europe-west2/subnetworks/"
                    f"{gate.OWNER_GATE_SUBNET_NAME}"
                ),
            }],
        ),
        (
            "create_private_owner_gate_vm",
            [{
                "name": gate.VM_NAME,
                "selfLink": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{gate.PROJECT}/zones/europe-west1-c/instances/"
                    f"{gate.VM_NAME}"
                ),
            }],
        ),
        (
            "allow_private_web_upstream_from_current_caddy_host",
            [{
                "name": "muncho-owner-gate-web-from-production",
                "selfLink": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "other-project/global/firewalls/"
                    "muncho-owner-gate-web-from-production"
                ),
            }],
        ),
    ],
)
def test_provider_inventory_ignores_same_name_outside_exact_scope(
    monkeypatch: pytest.MonkeyPatch,
    step_name: str,
    items: list[Mapping[str, Any]],
) -> None:
    chain = _chain()
    provider = object.__new__(apply._TrustedGcloudFoundationProvider)
    provider._plan_sha256 = apply.pre_foundation.inert_plan_sha256(chain.plan)
    monkeypatch.setattr(
        provider,
        "_read_json",
        lambda _logical: (items, _digest(f"read:{step_name}")),
    )
    step = next(item for item in chain.plan.foundation_steps if item.name == step_name)
    observed = provider.inspect_resource(step, plan=chain.plan)
    assert observed.state == "absent"


@pytest.mark.parametrize(
    ("step_name", "exact_item"),
    [
        (
            "create_dedicated_private_owner_gate_subnet",
            {
                "name": gate.OWNER_GATE_SUBNET_NAME,
                "selfLink": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{gate.PROJECT}/regions/{gate.REGION}/subnetworks/"
                    f"{gate.OWNER_GATE_SUBNET_NAME}"
                ),
            },
        ),
        (
            "create_private_owner_gate_vm",
            {
                "name": gate.VM_NAME,
                "selfLink": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{gate.PROJECT}/zones/{gate.ZONE}/instances/{gate.VM_NAME}"
                ),
            },
        ),
        (
            "allow_private_web_upstream_from_current_caddy_host",
            {
                "name": "muncho-owner-gate-web-from-production",
                "selfLink": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{gate.PROJECT}/global/firewalls/"
                    "muncho-owner-gate-web-from-production"
                ),
            },
        ),
    ],
)
def test_provider_inventory_rejects_duplicate_exact_scope_matches(
    monkeypatch: pytest.MonkeyPatch,
    step_name: str,
    exact_item: Mapping[str, Any],
) -> None:
    chain = _chain()
    provider = object.__new__(apply._TrustedGcloudFoundationProvider)
    provider._plan_sha256 = apply.pre_foundation.inert_plan_sha256(chain.plan)
    monkeypatch.setattr(
        provider,
        "_read_json",
        lambda _logical: ([dict(exact_item), dict(exact_item)], _digest("read")),
    )
    step = next(item for item in chain.plan.foundation_steps if item.name == step_name)
    assert provider.inspect_resource(step, plan=chain.plan).state == "drift"


def test_manifest_only_preflight_failure_intent_crash_replays_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain()
    store = _journal_for_test(tmp_path / "journal")
    provider = _PreflightRuntimeFailureProvider(chain)

    def crash_before_failure_receipt(
        *_args: Any,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        raise SystemExit("simulated_process_death_after_failure_intent")

    with monkeypatch.context() as context:
        context.setattr(
            apply,
            "_sign_failure_receipt",
            crash_before_failure_receipt,
        )
        with pytest.raises(SystemExit):
            apply._apply_with_provider(
                chain=chain,
                private_key=helpers.RELEASE_KEY,
                provider=provider,
                journal=store,
                now_unix=lambda: helpers.NOW + 2,
            )

    transaction_id = apply._transaction_id(chain)
    assert frozenset(store.list(transaction_id)) == {
        "failure-intent",
        "manifest",
    }
    intent = apply._read_transition(
        journal=store,
        chain=chain,
        transaction_id=transaction_id,
        name="failure-intent",
        phase="failure_intent",
    )
    assert intent is not None
    assert intent["failed_step_name"] == "preflight_live_ancestry"
    assert intent["inherently_unknown"] is False

    successor = _FakeProvider(chain)
    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=successor,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )
    assert successor.execute_calls == []
    assert successor.rollback_calls == []
    assert caught.value.receipt["terminal_state"] == "rolled_back_clean"
    assert caught.value.receipt["partial_unknown_state"] is False
    assert caught.value.receipt["completed_step_receipts"] == []
    assert caught.value.receipt["rollback_step_receipts"] == []


@pytest.mark.parametrize(
    ("crash_after", "expected_artifacts"),
    [
        ("precondition", {"manifest", "s0-pre"}),
        ("intent", {"manifest", "s0-intent", "s0-pre"}),
    ],
)
def test_preflight_failure_with_step_artifact_remains_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_after: str,
    expected_artifacts: set[str],
) -> None:
    chain = _chain()
    store = _journal_for_test(tmp_path / f"journal-{crash_after}")
    target = chain.plan.foundation_steps[0]
    crashing = _FakeProvider(
        chain,
        crash_step=target.name if crash_after == "intent" else None,
    )

    if crash_after == "precondition":
        original_publish = apply._publish_transition

        def publish_then_crash(**kwargs: Any) -> Mapping[str, Any]:
            published = original_publish(**kwargs)
            if kwargs["name"] == "s0-pre":
                raise SystemExit(
                    "simulated_process_death_after_precondition"
                )
            return published

        with monkeypatch.context() as context:
            context.setattr(apply, "_publish_transition", publish_then_crash)
            with pytest.raises(SystemExit):
                apply._apply_with_provider(
                    chain=chain,
                    private_key=helpers.RELEASE_KEY,
                    provider=crashing,
                    journal=store,
                    now_unix=lambda: helpers.NOW + 2,
                )
    else:
        with pytest.raises(SystemExit):
            apply._apply_with_provider(
                chain=chain,
                private_key=helpers.RELEASE_KEY,
                provider=crashing,
                journal=store,
                now_unix=lambda: helpers.NOW + 2,
            )

    transaction_id = apply._transaction_id(chain)
    assert frozenset(store.list(transaction_id)) == expected_artifacts
    successor = _PreflightRuntimeFailureProvider(chain)
    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=successor,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )
    assert successor.execute_calls == []
    assert successor.rollback_calls == []
    assert caught.value.receipt["terminal_state"] == (
        "manual_reconciliation_required"
    )
    assert caught.value.receipt["partial_unknown_state"] is True


def test_known_mutation_failure_still_rolls_back_cleanly(
    tmp_path: Path,
) -> None:
    chain = _chain()
    failed = chain.plan.foundation_steps[1]
    created = chain.plan.foundation_steps[0]
    provider = _FakeProvider(
        chain,
        fail_step=failed.name,
        failure_state="failed",
    )
    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=provider,
            journal=_journal_for_test(tmp_path / "journal"),
            now_unix=lambda: helpers.NOW + 2,
        )
    assert provider.execute_calls == [created.name, failed.name]
    assert provider.rollback_calls == [created.name]
    assert caught.value.receipt["terminal_state"] == "rolled_back_clean"
    assert caught.value.receipt["partial_unknown_state"] is False
    assert [
        item["disposition"]
        for item in caught.value.receipt["rollback_step_receipts"]
    ] == ["rolled_back"]


def test_unknown_current_operation_dispatches_zero_rollbacks(tmp_path: Path) -> None:
    chain = _chain()
    provider = _FakeProvider(
        chain,
        fail_step=chain.plan.foundation_steps[2].name,
        failure_state="unknown",
    )
    store = _journal_for_test(tmp_path / "journal")
    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=provider,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )
    assert provider.rollback_calls == []
    assert caught.value.receipt["terminal_state"] == (
        "manual_reconciliation_required"
    )
    assert all(
        item["disposition"] == "not_attempted_manual"
        for item in caught.value.receipt["rollback_step_receipts"]
    )


def test_rollback_stops_after_first_unknown_and_preserves_dependencies(
    tmp_path: Path,
) -> None:
    chain = _chain()
    failed = chain.plan.foundation_steps[4]
    first_rollback = chain.plan.foundation_steps[3]
    provider = _FakeProvider(
        chain,
        fail_step=failed.name,
        rollback_failure_step=first_rollback.name,
    )
    store = _journal_for_test(tmp_path / "journal")
    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=provider,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )
    assert provider.rollback_calls == [first_rollback.name]
    dispositions = caught.value.receipt["rollback_step_receipts"]
    assert dispositions[0]["disposition"] == "rollback_unknown"
    assert all(
        item["disposition"] == "not_attempted_manual"
        for item in dispositions[1:]
    )


def test_authority_drift_before_rollback_dispatches_no_rollback(
    tmp_path: Path,
) -> None:
    chain = _chain()
    provider = _FakeProvider(
        chain,
        fail_step=chain.plan.foundation_steps[3].name,
        authority_drift_after_failure=True,
    )
    store = _journal_for_test(tmp_path / "journal")
    with pytest.raises(apply.FoundationApplyFailed):
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=provider,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )
    assert provider.rollback_calls == []


def _concurrent_apply_worker(
    chain: apply.ValidatedFoundationAChain,
    root: str,
    count: Any,
    entered: Any,
    release: Any,
    results: Any,
) -> None:
    provider = _FakeProvider(
        chain,
        shared_execute_count=count,
        first_execute_entered=entered,
        release_first_execute=release,
    )
    store = _journal_for_test(Path(root))
    try:
        receipt = apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=provider,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )
        results.put(("ok", receipt["foundation_apply_receipt_sha256"]))
    except BaseException as exc:
        results.put(("error", type(exc).__name__, str(exc)))


def test_two_process_apply_holds_one_transaction_lease_and_dispatches_once(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    chain = _chain()
    root = tmp_path / "journal"
    count = context.Value("i", 0)
    entered = context.Event()
    release = context.Event()
    results = context.Queue()
    first = context.Process(
        target=_concurrent_apply_worker,
        args=(chain, str(root), count, entered, release, results),
    )
    second = context.Process(
        target=_concurrent_apply_worker,
        args=(chain, str(root), count, entered, release, results),
    )
    first.start()
    assert entered.wait(10)
    second.start()
    time.sleep(0.2)
    assert second.is_alive()
    release.set()
    first.join(15)
    second.join(15)
    assert first.exitcode == 0
    assert second.exitcode == 0
    observed = [results.get(timeout=2), results.get(timeout=2)]
    assert [item[0] for item in observed] == ["ok", "ok"], observed
    assert observed[0][1] == observed[1][1]
    assert count.value == len(chain.plan.foundation_steps)


def test_interrupted_apply_intent_is_never_redispatched(tmp_path: Path) -> None:
    chain = _chain()
    target = chain.plan.foundation_steps[0]
    store = _journal_for_test(tmp_path / "journal")
    crashing = _FakeProvider(chain, crash_step=target.name)
    with pytest.raises(SystemExit):
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=crashing,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )
    successor = _FakeProvider(chain)
    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=successor,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )
    assert successor.execute_calls == []
    assert successor.rollback_calls == []
    assert caught.value.receipt["partial_unknown_state"] is True


def _kill_after_rollback_posts_worker(
    chain: apply.ValidatedFoundationAChain,
    root: str,
    failed_step_name: str,
) -> None:
    def kill_before_terminal(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        os.kill(os.getpid(), signal.SIGKILL)
        raise AssertionError("SIGKILL returned")

    apply._sign_failure_receipt = kill_before_terminal
    provider = _FakeProvider(chain, fail_step=failed_step_name)
    apply._apply_with_provider(
        chain=chain,
        private_key=helpers.RELEASE_KEY,
        provider=provider,
        journal=_journal_for_test(Path(root)),
        now_unix=lambda: helpers.NOW + 2,
    )


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("failed_index", [1, 3])
def test_sigkill_after_rollback_post_never_resumes_forward_success(
    tmp_path: Path,
    failed_index: int,
) -> None:
    context = multiprocessing.get_context("fork")
    chain = _chain()
    root = tmp_path / f"journal-{failed_index}"
    worker = context.Process(
        target=_kill_after_rollback_posts_worker,
        args=(chain, str(root), chain.plan.foundation_steps[failed_index].name),
    )
    worker.start()
    worker.join(15)
    assert worker.exitcode == -signal.SIGKILL

    successor = _FakeProvider(chain)
    store = _journal_for_test(root)
    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=successor,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )
    assert successor.execute_calls == []
    assert successor.rollback_calls == []
    assert store.read(apply._transaction_id(chain), "success") is None
    assert caught.value.receipt["terminal_state"] in {
        "rolled_back_clean",
        "manual_reconciliation_required",
    }


@pytest.mark.parametrize(
    ("preexisting_indices", "crash_index", "successor_exact_indices"),
    [
        ({0}, 1, set()),
        (set(), 1, set()),
        (set(), 3, {0, 1}),
        ({0, 1, 2}, 3, {0, 1}),
    ],
)
def test_restart_revalidates_journaled_exact_resource_and_iam_state(
    tmp_path: Path,
    preexisting_indices: set[int],
    crash_index: int,
    successor_exact_indices: set[int],
) -> None:
    chain = _chain()
    store = _journal_for_test(tmp_path / "journal")
    crashing = _FakeProvider(
        chain,
        crash_step=chain.plan.foundation_steps[crash_index].name,
    )
    for index in preexisting_indices:
        step = chain.plan.foundation_steps[index]
        crashing.states[step.name] = "exact"
        if step.name.startswith("bind_narrow_"):
            crashing.policy_etags[step.name] = "etag-policy-1"
    with pytest.raises(SystemExit):
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=crashing,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )

    successor = _FakeProvider(chain)
    for index in successor_exact_indices:
        step = chain.plan.foundation_steps[index]
        successor.states[step.name] = "exact"
        if step.name.startswith("bind_narrow_"):
            successor.policy_etags[step.name] = "etag-policy-1"
    with pytest.raises(apply.FoundationApplyFailed) as caught:
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=successor,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )
    assert successor.execute_calls == []
    assert successor.rollback_calls == []
    assert caught.value.receipt["partial_unknown_state"] is True


def test_terminal_success_replay_revalidates_live_resources(tmp_path: Path) -> None:
    chain = _chain()
    store = _journal_for_test(tmp_path / "journal")
    first = _FakeProvider(chain)
    apply._apply_with_provider(
        chain=chain,
        private_key=helpers.RELEASE_KEY,
        provider=first,
        journal=store,
        now_unix=lambda: helpers.NOW + 2,
    )
    drifted = _FakeProvider(chain)
    with pytest.raises(
        apply.OwnerGateFoundationApplyError,
        match="owner_gate_foundation_journaled_resource_not_exact",
    ):
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=drifted,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )
    assert drifted.execute_calls == []
    assert drifted.rollback_calls == []


@pytest.mark.parametrize("case", ["missing", "diverged"])
def test_terminal_failure_requires_exact_signed_failure_intent(
    tmp_path: Path,
    case: str,
) -> None:
    chain = _chain()
    store = _journal_for_test(tmp_path / "journal")
    provider = _FakeProvider(
        chain,
        fail_step=chain.plan.foundation_steps[0].name,
        failure_state="unknown",
    )
    with pytest.raises(apply.FoundationApplyFailed):
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=provider,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )
    transaction_id = apply._transaction_id(chain)
    body = apply._read_transition(
        journal=store,
        chain=chain,
        transaction_id=transaction_id,
        name="failure-intent",
        phase="failure_intent",
    )
    assert body is not None
    intent_path = (
        tmp_path
        / "journal"
        / transaction_id
        / "failure-intent.json"
    )
    intent_path.unlink()
    if case == "diverged":
        apply._publish_transition(
            journal=store,
            transaction_id=transaction_id,
            name="failure-intent",
            body={
                **body,
                "failure_code": (
                    "owner_gate_foundation_replayed_intent_mismatch"
                ),
            },
            private_key=helpers.RELEASE_KEY,
        )
    successor = _FakeProvider(chain)
    with pytest.raises(
        apply.OwnerGateFoundationApplyError,
        match=(
            "owner_gate_foundation_failure_intent_missing"
            if case == "missing"
            else "owner_gate_foundation_failure_receipt_invalid"
        ),
    ):
        apply._apply_with_provider(
            chain=chain,
            private_key=helpers.RELEASE_KEY,
            provider=successor,
            journal=store,
            now_unix=lambda: helpers.NOW + 2,
        )
    assert successor.execute_calls == []
    assert successor.rollback_calls == []


@pytest.mark.parametrize("step_index", [2, 5])
def test_iam_cas_rejects_project_or_organization_substitution(
    step_index: int,
) -> None:
    chain = _chain()
    step = chain.plan.foundation_steps[step_index]
    argv = list(step.argv)
    argv[2] = (
        "projects/substituted-project"
        if step_index == 2
        else "organizations/999999999999"
    )
    with pytest.raises(
        apply.OwnerGateFoundationApplyError,
        match="owner_gate_foundation_provider_step_forbidden",
    ):
        apply._iam_binding_contract(
            replace(step, argv=tuple(argv)),
            plan=chain.plan,
        )


def test_public_apply_signature_exposes_no_runtime_or_mutation_seams() -> None:
    parameters = inspect.signature(
        apply.apply_foundation_from_canonical_artifacts
    ).parameters
    assert set(parameters) == {
        "pre_foundation_authority_raw",
        "owner_reauthentication_receipt_raw",
        "network_evidence_raw",
        "project_ancestry_evidence_raw",
        "release_public_key",
        "network_collector_public_key",
        "project_ancestry_collector_public_key",
    }
    assert not {
        "provider",
        "runner",
        "journal",
        "private_key",
        "signer",
        "now_unix",
        "gcloud_executable",
        "gcloud_configuration",
        "output_path",
    } & set(parameters)


def test_public_apply_constructs_runtime_from_exact_chain_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain()
    runtime = object()
    configuration = object()
    provider = object()
    fixed_journal = object()
    captured: dict[str, Any] = {}

    def runtime_factory(*, release_sha: str) -> object:
        captured["release_sha"] = release_sha
        return runtime

    def provider_factory(**values: Any) -> object:
        captured.update(values)
        return provider

    monkeypatch.setattr(apply.time, "time", lambda: float(helpers.NOW + 2))
    monkeypatch.setattr(
        apply,
        "_load_fixed_release_private_key",
        lambda **_kwargs: helpers.RELEASE_KEY,
    )
    monkeypatch.setattr(
        apply.launcher,
        "TrustedGcloudExecutable",
        runtime_factory,
    )
    monkeypatch.setattr(
        apply.launcher,
        "PinnedGcloudConfiguration",
        lambda: configuration,
    )
    monkeypatch.setattr(
        apply,
        "_TrustedGcloudFoundationProvider",
        provider_factory,
    )
    monkeypatch.setattr(
        apply.foundation_journal,
        "FoundationApplyJournal",
        lambda: fixed_journal,
    )
    monkeypatch.setattr(
        apply,
        "_apply_with_provider",
        lambda **values: captured.update(values) or {"ok": True},
    )

    assert apply.apply_foundation_from_canonical_artifacts(
        pre_foundation_authority_raw=chain.pre_foundation_authority_raw,
        owner_reauthentication_receipt_raw=(
            chain.owner_reauthentication_receipt_raw
        ),
        network_evidence_raw=chain.network_evidence_raw,
        project_ancestry_evidence_raw=chain.ancestry_evidence_raw,
        release_public_key=chain.release_public_key,
        network_collector_public_key=chain.network_collector_public_key,
        project_ancestry_collector_public_key=(
            chain.ancestry_collector_public_key
        ),
    ) == {"ok": True}
    assert captured["release_sha"] == helpers.REVISION
    assert captured["expected_release_revision"] == helpers.REVISION
    assert captured["gcloud_executable"] is runtime
    assert captured["gcloud_configuration"] is configuration
    assert captured["provider"] is provider
    assert captured["journal"] is fixed_journal


def test_direct_support_release_is_bound_to_signed_chain_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain()
    monkeypatch.setattr(
        apply,
        "_OWNER_SUPPORT_BOOTSTRAP_RELEASE_SHA",
        "b" * 40,
    )
    monkeypatch.setattr(apply.time, "time", lambda: float(helpers.NOW + 2))

    with pytest.raises(
        apply.OwnerGateFoundationApplyError,
        match="owner_gate_foundation_direct_release_mismatch",
    ):
        apply.apply_foundation_from_canonical_artifacts(
            pre_foundation_authority_raw=chain.pre_foundation_authority_raw,
            owner_reauthentication_receipt_raw=(
                chain.owner_reauthentication_receipt_raw
            ),
            network_evidence_raw=chain.network_evidence_raw,
            project_ancestry_evidence_raw=chain.ancestry_evidence_raw,
            release_public_key=chain.release_public_key,
            network_collector_public_key=chain.network_collector_public_key,
            project_ancestry_collector_public_key=(
                chain.ancestry_collector_public_key
            ),
        )


def test_fixed_journal_loader_returns_only_validated_nested_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain()
    store = _journal_for_test(tmp_path / "journal")
    receipt = apply._apply_with_provider(
        chain=chain,
        private_key=helpers.RELEASE_KEY,
        provider=_FakeProvider(chain),
        journal=store,
        now_unix=lambda: helpers.NOW + 2,
    )
    monkeypatch.setattr(
        apply.foundation_journal,
        "FoundationApplyJournal",
        lambda: store,
    )
    monkeypatch.setattr(apply.time, "time", lambda: float(helpers.NOW + 3))
    monkeypatch.setattr(
        store,
        "read",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recovering journal read was used")
        ),
    )

    loaded = apply.load_validated_foundation_apply_chain(chain)

    assert type(loaded) is apply.ValidatedFoundationApplyChain
    assert loaded.foundation_a is chain
    assert loaded.apply_receipt == receipt
    assert loaded.apply_receipt_raw == gate.canonical_json_bytes(receipt)
    assert set(
        inspect.signature(apply.load_validated_foundation_apply_chain).parameters
    ) == {"foundation_a"}
    assert not hasattr(apply, "decode_validated_foundation_apply_chain")
    assert "_decode_validated_foundation_apply_chain" not in apply.__all__


def test_source_recovery_loader_uses_signed_historical_times_after_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain()
    store = _journal_for_test(tmp_path / "journal")
    receipt = apply._apply_with_provider(
        chain=chain,
        private_key=helpers.RELEASE_KEY,
        provider=_FakeProvider(chain),
        journal=store,
        now_unix=lambda: helpers.NOW + 2,
    )
    monkeypatch.setattr(
        apply.foundation_journal,
        "FoundationApplyJournal",
        lambda: store,
    )
    monkeypatch.setattr(apply.time, "time", lambda: float(helpers.NOW + 301))

    with pytest.raises(apply.OwnerGateFoundationApplyError):
        apply.decode_validated_foundation_a_chain(
            pre_foundation_authority_raw=chain.pre_foundation_authority_raw,
            owner_reauthentication_receipt_raw=(
                chain.owner_reauthentication_receipt_raw
            ),
            network_evidence_raw=chain.network_evidence_raw,
            project_ancestry_evidence_raw=chain.ancestry_evidence_raw,
            release_public_key=chain.release_public_key,
            network_collector_public_key=chain.network_collector_public_key,
            project_ancestry_collector_public_key=(
                chain.ancestry_collector_public_key
            ),
            now_unix=helpers.NOW + 301,
        )
    recovered = _load_source_recovery(chain)

    assert recovered.apply_receipt == receipt
    assert recovered.apply_receipt_raw == gate.canonical_json_bytes(receipt)
    assert recovered.foundation_a.authority == chain.authority
    assert recovered.foundation_a.owner_reauthentication_receipt == (
        chain.owner_reauthentication_receipt
    )


def test_source_recovery_loader_rejects_empty_fixed_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain()
    store = _journal_for_test(tmp_path / "journal")
    monkeypatch.setattr(
        apply.foundation_journal,
        "FoundationApplyJournal",
        lambda: store,
    )

    with pytest.raises(
        apply.OwnerGateFoundationApplyError,
        match="owner_gate_foundation_success_journal_invalid",
    ):
        _load_source_recovery(chain)


def test_fixed_journal_loader_never_recovers_pending_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain()
    store = _journal_for_test(tmp_path / "journal")
    apply._apply_with_provider(
        chain=chain,
        private_key=helpers.RELEASE_KEY,
        provider=_FakeProvider(chain),
        journal=store,
        now_unix=lambda: helpers.NOW + 2,
    )
    transaction_id = apply._transaction_id(chain)
    pending = store.root / transaction_id / ".success.pending"
    pending.write_bytes(b"{}")
    pending.chmod(0o600)
    monkeypatch.setattr(
        apply.foundation_journal,
        "FoundationApplyJournal",
        lambda: store,
    )
    monkeypatch.setattr(apply.time, "time", lambda: float(helpers.NOW + 3))

    with pytest.raises(
        apply.OwnerGateFoundationApplyError,
        match="owner_gate_foundation_journal_read_failed",
    ):
        apply.load_validated_foundation_apply_chain(chain)

    assert pending.read_bytes() == b"{}"


def test_fixed_journal_loader_rejects_conflicting_failure_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _chain()
    store = _journal_for_test(tmp_path / "journal")
    apply._apply_with_provider(
        chain=chain,
        private_key=helpers.RELEASE_KEY,
        provider=_FakeProvider(chain),
        journal=store,
        now_unix=lambda: helpers.NOW + 2,
    )
    transaction_id = apply._transaction_id(chain)
    apply._publish_transition(
        journal=store,
        transaction_id=transaction_id,
        name="failure-intent",
        body=apply._transition_body(
            chain=chain,
            transaction_id=transaction_id,
            phase="failure_intent",
            payload={"test_conflict": True},
        ),
        private_key=helpers.RELEASE_KEY,
    )
    monkeypatch.setattr(
        apply.foundation_journal,
        "FoundationApplyJournal",
        lambda: store,
    )
    monkeypatch.setattr(apply.time, "time", lambda: float(helpers.NOW + 3))

    with pytest.raises(
        apply.OwnerGateFoundationApplyError,
        match="owner_gate_foundation_success_journal_invalid",
    ):
        apply.load_validated_foundation_apply_chain(chain)
    with pytest.raises(
        apply.OwnerGateFoundationApplyError,
        match="owner_gate_foundation_success_journal_invalid",
    ):
        _load_source_recovery(chain)


def _cli_artifacts(tmp_path: Path) -> tuple[list[str], Mapping[Path, bytes]]:
    authority, _plan, _evidence = helpers._authority()
    paths = {
        "authority": tmp_path / "authority.json",
        "owner": tmp_path / "owner.json",
        "network": tmp_path / "network.json",
        "ancestry": tmp_path / "ancestry.json",
        "network_key": tmp_path / "network.pub",
        "ancestry_key": tmp_path / "ancestry.pub",
    }
    raw = {
        paths["authority"]: gate.canonical_json_bytes(authority),
        paths["owner"]: gate.canonical_json_bytes(
            helpers._owner_reauth_receipt()
        ),
        paths["network"]: gate.canonical_json_bytes(
            helpers._signed_network_evidence()
        ),
        paths["ancestry"]: helpers._signed_ancestry_raw(),
    }
    argv = [
        "--pre-foundation-authority",
        str(paths["authority"]),
        "--owner-reauth-receipt",
        str(paths["owner"]),
        "--network-evidence",
        str(paths["network"]),
        "--project-ancestry-evidence",
        str(paths["ancestry"]),
        "--network-collector-public-key",
        str(paths["network_key"]),
        "--project-ancestry-collector-public-key",
        str(paths["ancestry_key"]),
    ]
    return argv, raw


@pytest.mark.parametrize(
    "forbidden",
    ["--private-key", "--provider", "--runner", "--journal", "--output"],
)
def test_cli_has_no_caller_controlled_mutation_or_signer_seams(
    forbidden: str,
) -> None:
    with pytest.raises(SystemExit):
        apply.main([forbidden, "/tmp/value"])


@pytest.mark.parametrize("case", ["relative", "mutable", "symlink"])
def test_cli_input_reader_requires_absolute_immutable_owner_file(
    tmp_path: Path,
    case: str,
) -> None:
    target = tmp_path / "artifact.json"
    target.write_bytes(b"{}")
    target.chmod(0o444 if case != "mutable" else 0o600)
    path = target
    if case == "relative":
        path = Path("artifact.json")
    elif case == "symlink":
        path = tmp_path / "artifact-link.json"
        path.symlink_to(target)
    with pytest.raises(
        apply.OwnerGateFoundationApplyError,
        match="owner_gate_foundation_input_path_invalid",
    ):
        apply._read_owner_immutable(path, maximum=1024)


@pytest.mark.parametrize("failed", [False, True])
def test_cli_prints_only_canonical_terminal_hash_and_fixed_journal_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed: bool,
) -> None:
    argv, raw = _cli_artifacts(tmp_path)
    monkeypatch.setattr(apply.time, "time", lambda: float(helpers.NOW + 2))
    monkeypatch.setattr(
        apply.pre_foundation,
        "load_pinned_public_key",
        lambda *_args, **_kwargs: helpers.RELEASE_KEY.public_key(),
    )
    monkeypatch.setattr(
        apply,
        "_load_collector_public_key",
        lambda _path: helpers.NETWORK_KEY.public_key(),
    )
    monkeypatch.setattr(
        apply,
        "_read_owner_immutable",
        lambda path, **_kwargs: raw[path],
    )
    terminal_hash = "9" * 64
    if failed:
        failure = {
            "foundation_apply_failure_receipt_sha256": terminal_hash,
            "terminal_state": "manual_reconciliation_required",
        }

        def run(**_kwargs: Any) -> Mapping[str, Any]:
            raise apply.FoundationApplyFailed(failure)

    else:

        def run(**_kwargs: Any) -> Mapping[str, Any]:
            return {"foundation_apply_receipt_sha256": terminal_hash}

    monkeypatch.setattr(
        apply,
        "apply_foundation_from_canonical_artifacts",
        run,
    )
    assert apply.main(argv) == (2 if failed else 0)
    output = capsys.readouterr().out
    report = json.loads(output)
    assert output == gate.canonical_json_bytes(report).decode("ascii") + "\n"
    assert report["terminal_receipt_sha256"] == terminal_hash
    assert report["terminal_journal_path"].startswith(
        str(journal.DEFAULT_JOURNAL_ROOT) + "/"
    )
    assert report["terminal_journal_path"].endswith(
        "/failure.json" if failed else "/success.json"
    )
    lowered = output.casefold()
    assert "token" not in lowered
    assert "private_key" not in lowered
    assert "signature" not in lowered
    assert "network_evidence" not in lowered


def _write_origin_probe(path: Path, body: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "import sys\n"
        "print(f'SEALED_ORIGIN\\t{__name__}\\t{__file__}', "
        "file=sys.stderr)\n"
        f"{body}",
        encoding="utf-8",
    )


def _synthetic_sealed_owner_support(tmp_path: Path) -> Path:
    release_sha = "d" * 40
    root = tmp_path / f"owner-support-{release_sha}"
    source = root / "source"
    site = root / "site-packages"
    module = source / "scripts/canary/owner_gate_foundation_apply.py"
    module.parent.mkdir(parents=True)
    shutil.copyfile(Path(apply.__file__), module)
    (root / "owner-support.json").write_text("{}\n", encoding="ascii")
    _write_origin_probe(source / "scripts/__init__.py")
    _write_origin_probe(source / "scripts/canary/__init__.py")
    for name in (
        "full_canary_owner_launcher",
        "owner_gate_foundation",
        "owner_gate_foundation_journal",
        "owner_gate_owner_reauth",
        "owner_gate_pre_foundation",
        "owner_gate_project_ancestry",
        "owner_gate_trust",
        "owner_gate_trust_author",
    ):
        _write_origin_probe(source / f"scripts/canary/{name}.py")
    _write_origin_probe(site / "cryptography/__init__.py")
    _write_origin_probe(
        site / "cryptography/exceptions.py",
        "class InvalidSignature(Exception):\n    pass\n",
    )
    _write_origin_probe(site / "cryptography/hazmat/__init__.py")
    _write_origin_probe(site / "cryptography/hazmat/primitives/__init__.py")
    _write_origin_probe(
        site / "cryptography/hazmat/primitives/serialization.py",
        "def load_pem_public_key(_raw):\n    raise ValueError('probe only')\n",
    )
    _write_origin_probe(
        site / "cryptography/hazmat/primitives/asymmetric/__init__.py"
    )
    _write_origin_probe(
        site / "cryptography/hazmat/primitives/asymmetric/ed25519.py",
        "class Ed25519PrivateKey:\n    pass\n"
        "class Ed25519PublicKey:\n    pass\n",
    )
    for item in sorted(
        root.rglob("*"),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        item.chmod(0o500 if item.is_dir() else 0o400)
    root.chmod(0o500)
    return root


def _unseal_owner_support(root: Path) -> None:
    root.chmod(0o700)
    for item in root.rglob("*"):
        item.chmod(0o700 if item.is_dir() else 0o600)


def _run_direct_owner_support(
    script: str,
    *,
    cwd: Path,
    hostile_pythonpath: Path,
) -> subprocess.CompletedProcess[bytes]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(hostile_pythonpath)
    return subprocess.run(
        [sys.executable, "-I", "-S", "-B", script, "--help"],
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )


def test_owner_side_cli_runs_only_from_absolute_sealed_support_tree(
    tmp_path: Path,
) -> None:
    relative = "scripts/canary/owner_gate_foundation_apply.py"
    assert relative not in package.ROOT_RUNTIME_FILES
    assert all(
        "foundation-apply" not in entrypoint
        for entrypoint in package.REQUIRED_ENTRYPOINTS
    )
    repository = Path(__file__).parents[3]
    hostile = tmp_path / "hostile"
    _write_origin_probe(
        hostile / "scripts/__init__.py",
        "raise RuntimeError('ambient source imported')\n",
    )
    _write_origin_probe(
        hostile / "cryptography/__init__.py",
        "raise RuntimeError('ambient site imported')\n",
    )
    root = _synthetic_sealed_owner_support(tmp_path)
    script = root / "source" / relative
    try:
        completed = _run_direct_owner_support(
            str(script),
            cwd=hostile,
            hostile_pythonpath=hostile,
        )
        assert completed.returncode == 0, completed.stderr.decode(
            "utf-8", errors="replace"
        )
        help_text = completed.stdout.decode("utf-8", errors="strict")
        assert "--pre-foundation-authority" in help_text
        assert "--private-key" not in help_text
        assert "--output" not in help_text
        origins = [
            Path(line.split("\t", 2)[2])
            for line in completed.stderr.decode("utf-8").splitlines()
            if line.startswith("SEALED_ORIGIN\t")
        ]
        assert origins
        assert all(
            origin.is_relative_to(root / "source")
            or origin.is_relative_to(root / "site-packages")
            for origin in origins
        )

        worktree = _run_direct_owner_support(
            str(Path(apply.__file__)),
            cwd=repository,
            hostile_pythonpath=hostile,
        )
        assert worktree.returncode != 0
        assert b"owner_gate_foundation_direct_path_invalid" in worktree.stderr

        relative_run = _run_direct_owner_support(
            f"source/{relative}",
            cwd=root,
            hostile_pythonpath=hostile,
        )
        assert relative_run.returncode != 0
        assert (
            b"owner_gate_foundation_direct_path_invalid"
            in relative_run.stderr
        )

        alias_parent = tmp_path / "alias"
        alias_parent.mkdir()
        alias = alias_parent / root.name
        alias.symlink_to(root, target_is_directory=True)
        symlinked = _run_direct_owner_support(
            str(alias / "source" / relative),
            cwd=hostile,
            hostile_pythonpath=hostile,
        )
        assert symlinked.returncode != 0
        assert b"owner_gate_foundation_direct_path_invalid" in symlinked.stderr

        (root / "source").chmod(0o700)
        mutable = _run_direct_owner_support(
            str(script),
            cwd=hostile,
            hostile_pythonpath=hostile,
        )
        assert mutable.returncode != 0
        assert b"owner_gate_foundation_direct_tree_invalid" in mutable.stderr
        (root / "source").chmod(0o500)
    finally:
        _unseal_owner_support(root)
