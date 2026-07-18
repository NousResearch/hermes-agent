from __future__ import annotations

import inspect
import hashlib
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_deferred_mutation_iam as deferred
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_foundation_apply as foundation_apply
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_trust as trust
from tests.scripts.canary import test_owner_gate_foundation_apply as apply_fixture
from tests.scripts.canary import test_owner_gate_foundation as foundation_fixture
from tests.scripts.canary import test_owner_gate_inert_observation as inert_fixture
from tests.scripts.canary import test_owner_gate_pre_foundation as pre_fixture


@pytest.fixture(autouse=True)
def _pin_release_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        pre_fixture.RELEASE_KEY_ID,
    )


def _authority() -> deferred._DeferredMutationIamAuthority:
    plan = foundation_fixture._plan()
    return deferred._DeferredMutationIamAuthority._create(
        plan=plan,
        foundation_apply_chain=object(),  # type: ignore[arg-type]
        final_release_public_key=pre_fixture.RELEASE_KEY.public_key(),
        contract=deferred._fixed_contract(plan),
        lineage={
            "final_release_revision": plan.spec.release_revision,
            "final_source_tree_oid": "f" * 40,
            "final_package_sha256": "0" * 64,
            "final_plan_sha256": plan.sha256,
            "final_network_evidence_sha256": "1" * 64,
            "final_network_collector_public_key_id": "2" * 64,
            "foundation_source_revision": "b" * 40,
            "foundation_source_tree_oid": "c" * 40,
            "pre_foundation_authority_sha256": "3" * 64,
            "foundation_apply_receipt_sha256": "4" * 64,
            "final_release_public_key_id": pre_fixture.RELEASE_KEY_ID,
            "inert_evidence_set_sha256": "5" * 64,
        },
    )


def _activation_receipt(
    authority: deferred._DeferredMutationIamAuthority,
    *,
    expires_at_unix: int = pre_fixture.NOW + 300,
    runtime_sha256: str = "0" * 64,
) -> Mapping[str, Any]:
    base = pre_fixture._owner_reauth_receipt(
        expires_at_unix=expires_at_unix
    )
    body = {
        name: item
        for name, item in base.items()
        if name
        not in {
            "owner_reauthentication_receipt_sha256",
            "signature_ed25519_b64url",
        }
    }
    body["trusted_runtime_identity"] = {
        **dict(body["trusted_runtime_identity"]),
        "release_revision": authority.plan.spec.release_revision,
        "sealed_runtime_identity_sha256": runtime_sha256,
    }
    return owner_reauth._sign_owner_reauth_receipt(
        body,
        private_key=pre_fixture.RELEASE_KEY,
    )


def _activation_authorization(
    authority: deferred._DeferredMutationIamAuthority,
    *,
    now_unix: int = pre_fixture.NOW,
    expires_at_unix: int = pre_fixture.NOW + 300,
) -> deferred._ActivationAuthorization:
    return deferred._validated_activation_authorization(
        authority=authority,
        receipt=_activation_receipt(
            authority,
            expires_at_unix=expires_at_unix,
        ),
        expected_runtime_sha256="0" * 64,
        now_unix=now_unix,
    )


def _journal(root: Path) -> deferred.DeferredMutationIamJournal:
    owner = root.parent.stat()
    store = deferred.DeferredMutationIamJournal(
        _root=root,
        _owner_uid=owner.st_uid,
        _owner_gid=owner.st_gid,
    )
    store._require_owner_process = lambda: None  # type: ignore[method-assign]
    return store


def _policy(
    authority: deferred._DeferredMutationIamAuthority,
    *,
    present: bool,
    etag: str = "etag-0",
    extra_bindings: list[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    bindings: list[Mapping[str, Any]] = [
        {"role": "roles/viewer", "members": ["group:auditors@example.com"]}
    ]
    if present:
        bindings.append(authority.contract.binding())
    bindings.extend(extra_bindings or [])
    return deferred._normalized_policy(
        {
            "etag": etag,
            "version": 3,
            "bindings": bindings,
            "auditConfigs": [{
                "service": "allServices",
                "auditLogConfigs": [{"logType": "ADMIN_READ"}],
            }],
        },
        resource_name=authority.contract.resource_name,
    )


class _FakeProvider:
    def __init__(
        self,
        authority: deferred._DeferredMutationIamAuthority,
        *,
        present: bool = False,
    ) -> None:
        self.authority = authority
        self.policy = _policy(authority, present=present)
        self.observe_calls = 0
        self.mutate_calls: list[Mapping[str, Any]] = []
        self.lineage_calls = 0
        self.lineage_actions: list[str] = []
        self.foundation_drift = False
        self.crash_after_effect = False
        self.operation_state = "completed"
        self.unknown_after_effect = False
        self.change_etag_on_observe: int | None = None

    def assert_lineage(
        self,
        authority: deferred._DeferredMutationIamAuthority,
        *,
        action: str,
    ) -> None:
        assert authority is self.authority
        self.lineage_calls += 1
        self.lineage_actions.append(action)
        if self.foundation_drift and action == deferred.ACTION_ACTIVATE:
            raise deferred.OwnerGateDeferredMutationIamError(
                "simulated_live_foundation_drift"
            )

    def observe_policy(
        self,
        authority: deferred._DeferredMutationIamAuthority,
    ) -> deferred.DeferredMutationIamObservation:
        assert authority is self.authority
        self.observe_calls += 1
        if self.change_etag_on_observe == self.observe_calls:
            self.policy = deferred._normalized_policy(
                {**deferred._api_policy(self.policy), "etag": "etag-drift"},
                resource_name=authority.contract.resource_name,
            )
        state = deferred._classify_policy(
            self.policy,
            contract=authority.contract,
        )
        return deferred.DeferredMutationIamObservation(
            state,
            self.policy,
            deferred._sha256_json({
                "observe": self.observe_calls,
                "policy": self.policy,
            }),
        )

    def mutate_policy(
        self,
        authority: deferred._DeferredMutationIamAuthority,
        *,
        action: str,
        attempt_id: str,
        precondition: Mapping[str, Any],
        request_policy: Mapping[str, Any],
    ) -> foundation_apply.OperationObservation:
        assert authority is self.authority
        assert precondition == self.policy
        assert request_policy == deferred._edited_policy(
            precondition,
            contract=authority.contract,
            action=action,
        )
        self.mutate_calls.append({
            "action": action,
            "attempt_id": attempt_id,
            "precondition": precondition,
            "request_policy": request_policy,
        })
        receipt = deferred._sha256_json({
            "action": action,
            "attempt_id": attempt_id,
            "call": len(self.mutate_calls),
        })
        if self.operation_state == "unknown" and self.unknown_after_effect:
            self.policy = deferred._normalized_policy(
                {**dict(request_policy), "etag": "etag-unknown-effect"},
                resource_name=authority.contract.resource_name,
            )
        if self.operation_state in {"failed", "unknown"}:
            return foundation_apply.OperationObservation(
                self.operation_state,
                receipt,
                attempt_id,
                cas_precondition_etag=str(request_policy["etag"]),
            )
        post_etag = f"etag-{len(self.mutate_calls)}"
        self.policy = deferred._normalized_policy(
            {**dict(request_policy), "etag": post_etag},
            resource_name=authority.contract.resource_name,
        )
        if self.crash_after_effect:
            raise SystemExit("simulated_crash_after_cas")
        return foundation_apply.OperationObservation(
            "completed",
            receipt,
            attempt_id,
            deferred._sha256_json({"post": self.policy}),
            str(request_policy["etag"]),
            post_etag,
        )


def _execute(
    authority: deferred._DeferredMutationIamAuthority,
    provider: _FakeProvider,
    journal: deferred.DeferredMutationIamJournal,
    *,
    action: str = deferred.ACTION_ACTIVATE,
) -> Mapping[str, Any]:
    authorization = (
        _activation_authorization(authority)
        if action == deferred.ACTION_ACTIVATE
        else None
    )
    return deferred._execute_with_provider(
        authority=authority,
        action=action,
        provider=provider,
        journal=journal,
        activation_authorization_factory=(
            (lambda: authorization)
            if authorization is not None
            else None
        ),
        activation_authorization_validator=(
            (lambda _intent: None)
            if action == deferred.ACTION_ACTIVATE
            else None
        ),
    )


def _real_authority_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    foundation.OwnerGateFoundationPlan,
    foundation_apply.ValidatedFoundationApplyChain,
    foundation.ProductionNetworkEvidence,
    Ed25519PrivateKey,
    Mapping[str, Any],
]:
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        pre_fixture.RELEASE_KEY_ID,
    )
    foundation_a = apply_fixture._chain()
    apply_chain = foundation_apply.ValidatedFoundationApplyChain._create(
        foundation_a=foundation_a,
        apply_receipt={"foundation_apply_receipt_sha256": "a" * 64},
        apply_receipt_raw=b"{}",
    )
    final_revision = "d" * 40
    final_network_key = Ed25519PrivateKey.generate()
    final_network_key_id = hashlib.sha256(
        final_network_key.public_key().public_bytes_raw()
    ).hexdigest()
    evidence = foundation.ProductionNetworkEvidence.from_mapping(
        foundation_fixture._signed_network_evidence(
            final_network_key,
            collected_at=pre_fixture.NOW - 1,
        ),
        public_key=final_network_key.public_key(),
        expected_public_key_id=final_network_key_id,
        now_unix=pre_fixture.NOW,
    )
    final_spec = replace(
        foundation_a.plan.spec,
        release_revision=final_revision,
        source_tree_oid=None,
        boot_image_numeric_id=None,
        package_inventory_sha256="b" * 64,
        network_collector_public_key_id=final_network_key_id,
        cloud_collector_public_key_id="c" * 64,
        host_collector_public_key_id="d" * 64,
    )
    plan = foundation.build_plan(
        spec=final_spec,
        network_evidence=evidence,
        network_collector_public_key=final_network_key.public_key(),
        now_unix=pre_fixture.NOW,
    )
    base_receipt = pre_fixture._owner_reauth_receipt()
    body = {
        name: item
        for name, item in base_receipt.items()
        if name
        not in {
            "owner_reauthentication_receipt_sha256",
            "signature_ed25519_b64url",
        }
    }
    body["trusted_runtime_identity"] = {
        **dict(body["trusted_runtime_identity"]),
        "release_revision": final_revision,
        "sealed_runtime_identity_sha256": "e" * 64,
    }
    receipt = owner_reauth._sign_owner_reauth_receipt(
        body,
        private_key=pre_fixture.RELEASE_KEY,
    )
    return plan, apply_chain, evidence, final_network_key, receipt


def _authority_variant(
    authority: deferred._DeferredMutationIamAuthority,
    *,
    evidence_digit: str,
    network_digit: str,
) -> deferred._DeferredMutationIamAuthority:
    lineage = {
        **dict(authority.lineage),
        "final_plan_sha256": network_digit * 64,
        "final_network_evidence_sha256": network_digit * 64,
        "inert_evidence_set_sha256": evidence_digit * 64,
    }
    variant = deferred._DeferredMutationIamAuthority._create(
        plan=authority.plan,
        foundation_apply_chain=authority.foundation_apply_chain,
        final_release_public_key=authority.final_release_public_key,
        contract=authority.contract,
        lineage=lineage,
    )
    assert variant.transaction_id == authority.transaction_id
    return variant


def _distinct_release_authority(
    authority: deferred._DeferredMutationIamAuthority,
) -> deferred._DeferredMutationIamAuthority:
    release_revision = "e" * 40
    plan = replace(
        authority.plan,
        spec=replace(
            authority.plan.spec,
            release_revision=release_revision,
        ),
    )
    lineage = {
        **dict(authority.lineage),
        "final_release_revision": release_revision,
        "final_source_tree_oid": "a" * 40,
        "final_package_sha256": "b" * 64,
    }
    variant = deferred._DeferredMutationIamAuthority._create(
        plan=plan,
        foundation_apply_chain=authority.foundation_apply_chain,
        final_release_public_key=authority.final_release_public_key,
        contract=authority.contract,
        lineage=lineage,
    )
    assert variant.transaction_id != authority.transaction_id
    return variant


def _rotated_activation_receipt(
    authority: deferred._DeferredMutationIamAuthority,
    *,
    marker: str,
) -> Mapping[str, Any]:
    receipt = _activation_receipt(authority)
    unsigned = {
        name: item
        for name, item in receipt.items()
        if name
        not in {
            "owner_reauthentication_receipt_sha256",
            "signature_ed25519_b64url",
        }
    }
    unsigned["authenticated_probe"] = {
        **dict(unsigned["authenticated_probe"]),
        "output_sha256": hashlib.sha256(marker.encode("ascii")).hexdigest(),
    }
    return owner_reauth._sign_owner_reauth_receipt(
        unsigned,
        private_key=pre_fixture.RELEASE_KEY,
    )


def _frozen_stub(label: str) -> SimpleNamespace:
    stable_inputs = SimpleNamespace(
        assert_stable=lambda: None,
    )
    return SimpleNamespace(
        label=label,
        binding=object(),
        inputs=stable_inputs,
        assert_stable=lambda *, now_unix: None,
    )


def _pathless_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    *,
    transaction_id: str,
) -> tuple[
    deferred.launcher.TrustedGcloudExecutable,
    deferred.launcher.PinnedGcloudConfiguration,
    deferred.launcher.GcloudOwnerAccessToken,
]:
    executable = object.__new__(deferred.launcher.TrustedGcloudExecutable)
    configuration = object.__new__(
        deferred.launcher.PinnedGcloudConfiguration
    )
    configuration._account = owner_reauth.OWNER_ACCOUNT
    identity = object.__new__(deferred.launcher.GcloudOwnerAccessToken)
    identity._gcloud_executable = executable
    identity._gcloud_configuration = configuration
    runtime = {"identity_sha256": "0" * 64}
    monkeypatch.setattr(
        deferred,
        "_validate_owner_capabilities",
        lambda **_kwargs: runtime,
    )
    monkeypatch.setattr(
        deferred.launcher.TrustedGcloudExecutable,
        "sealed_runtime_identity",
        lambda self, *, expected_release_sha: runtime,
    )
    monkeypatch.setattr(
        deferred.launcher.PinnedGcloudConfiguration,
        "assert_stable",
        lambda self: None,
    )
    monkeypatch.setattr(
        deferred.launcher.GcloudOwnerAccessToken,
        "require_stable",
        lambda self: None,
    )
    monkeypatch.setattr(
        deferred.launcher,
        "require_trusted_owner_support_activation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        deferred.launcher,
        "require_local_launcher_provenance",
        lambda _release_revision: None,
    )
    monkeypatch.setattr(
        deferred.inert_observation,
        "_release_private_key",
        lambda _binding: pre_fixture.RELEASE_KEY,
    )
    monkeypatch.setattr(
        deferred,
        "_release_transaction_id",
        lambda _release_revision: transaction_id,
    )
    return executable, configuration, identity


def test_public_boundary_has_no_caller_selected_iam_fields() -> None:
    allowed = {
        "release_revision",
        "gcloud_executable",
        "gcloud_configuration",
        "owner_identity",
    }
    for function in (
        deferred.activate_deferred_mutation_iam,
        deferred.remove_deferred_mutation_iam,
    ):
        assert set(inspect.signature(function).parameters) == allowed
    assert not allowed.intersection({
        "project",
        "role",
        "member",
        "condition",
        "path",
        "receipt",
        "plan",
    })
    parser = deferred.launcher._cli_parser()
    activate = parser.parse_args([
        "--release-sha",
        pre_fixture.REVISION,
        "--activate-owner-gate-deferred-mutation-iam",
    ])
    remove = parser.parse_args([
        "--release-sha",
        pre_fixture.REVISION,
        "--remove-owner-gate-deferred-mutation-iam",
    ])
    assert activate.activate_owner_gate_deferred_mutation_iam is True
    assert remove.remove_owner_gate_deferred_mutation_iam is True
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--release-sha",
            pre_fixture.REVISION,
            "--activate-owner-gate-deferred-mutation-iam",
            "--remove-owner-gate-deferred-mutation-iam",
        ])
    for forbidden in ("--project", "--role", "--member", "--journal"):
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--release-sha",
                pre_fixture.REVISION,
                "--activate-owner-gate-deferred-mutation-iam",
                forbidden,
                "attacker-controlled",
            ])


def test_activate_is_cas_bound_and_replays_without_second_mutation(
    tmp_path: Path,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    journal = _journal(tmp_path / "journal")
    plan_before = authority.plan.report()

    first = _execute(authority, provider, journal)
    replay = _execute(authority, provider, journal)

    assert first == replay
    assert first["action"] == "activate"
    assert first["mutation_binding_present"] is True
    assert first["cloud_mutation_performed"] is True
    assert first["foundation_steps_extended"] is False
    assert len(provider.mutate_calls) == 1
    call = provider.mutate_calls[0]
    assert call["request_policy"]["etag"] == call["precondition"][
        "policy_etag"
    ]
    assert call["request_policy"]["version"] == 3
    assert authority.plan.report() == plan_before
    assert authority.plan.deferred_mutation_iam_steps[0] not in (
        authority.plan.foundation_steps
    )


def test_stable_lifecycle_id_is_r_not_f_and_excludes_rotated_reauth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, apply_chain, evidence, network_key, receipt = (
        _real_authority_inputs(monkeypatch)
    )
    authority = deferred._validated_authority(
        plan=plan,
        foundation_apply_chain=apply_chain,
        final_network_evidence=evidence,
        final_network_collector_public_key=network_key.public_key(),
        final_release_public_key=pre_fixture.RELEASE_KEY.public_key(),
        final_source_tree_oid="f" * 40,
        final_package_sha256="9" * 64,
        inert_evidence_set_sha256="8" * 64,
        now_unix=pre_fixture.NOW,
    )
    first = deferred._validated_activation_authorization(
        authority=authority,
        receipt=receipt,
        expected_runtime_sha256="e" * 64,
        now_unix=pre_fixture.NOW,
    )
    rotated_body = {
        name: item
        for name, item in receipt.items()
        if name
        not in {
            "owner_reauthentication_receipt_sha256",
            "signature_ed25519_b64url",
        }
    }
    rotated_body["authenticated_probe"] = {
        **dict(rotated_body["authenticated_probe"]),
        "output_sha256": "f" * 64,
    }
    rotated_receipt = owner_reauth._sign_owner_reauth_receipt(
        rotated_body,
        private_key=pre_fixture.RELEASE_KEY,
    )
    rotated = deferred._validated_activation_authorization(
        authority=authority,
        receipt=rotated_receipt,
        expected_runtime_sha256="e" * 64,
        now_unix=pre_fixture.NOW,
    )

    assert authority.plan.spec.release_revision == "d" * 40
    assert authority.foundation_apply_chain.foundation_source_revision == (
        pre_fixture.REVISION
    )
    assert authority.plan.spec.release_revision != (
        authority.foundation_apply_chain.foundation_source_revision
    )
    assert "owner_reauthentication_receipt_sha256" not in authority.lineage
    assert first.receipt_sha256 != rotated.receipt_sha256
    assert authority.transaction_id == deferred._stable_transaction_id(
        contract=authority.contract,
        lineage=authority.lineage,
    )


def test_activation_reauth_must_be_fresh_and_bound_to_final_r(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, apply_chain, evidence, network_key, receipt = (
        _real_authority_inputs(monkeypatch)
    )
    authority = deferred._validated_authority(
        plan=plan,
        foundation_apply_chain=apply_chain,
        final_network_evidence=evidence,
        final_network_collector_public_key=network_key.public_key(),
        final_release_public_key=pre_fixture.RELEASE_KEY.public_key(),
        final_source_tree_oid="f" * 40,
        final_package_sha256="9" * 64,
        inert_evidence_set_sha256="8" * 64,
        now_unix=pre_fixture.NOW,
    )
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="activation_authorization_invalid",
    ):
        deferred._validated_activation_authorization(
            authority=authority,
            receipt=pre_fixture._owner_reauth_receipt(),
            expected_runtime_sha256="0" * 64,
            now_unix=pre_fixture.NOW,
        )

    expired_body = {
        name: item
        for name, item in receipt.items()
        if name
        not in {
            "owner_reauthentication_receipt_sha256",
            "signature_ed25519_b64url",
        }
    }
    expired_body["expires_at_unix"] = pre_fixture.NOW + 1
    expired = owner_reauth._sign_owner_reauth_receipt(
        expired_body,
        private_key=pre_fixture.RELEASE_KEY,
    )
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="activation_authorization_invalid",
    ):
        deferred._validated_activation_authorization(
            authority=authority,
            receipt=expired,
            expected_runtime_sha256="e" * 64,
            now_unix=pre_fixture.NOW + 2,
        )


def test_expired_rotated_activation_reauth_does_not_block_exact_remove(
    tmp_path: Path,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    journal = _journal(tmp_path / "journal")
    authorization = _activation_authorization(
        authority,
        expires_at_unix=pre_fixture.NOW + 1,
    )
    author_calls = 0

    def author() -> deferred._ActivationAuthorization:
        nonlocal author_calls
        author_calls += 1
        return authorization

    activated = deferred._execute_with_provider(
        authority=authority,
        action=deferred.ACTION_ACTIVATE,
        provider=provider,
        journal=journal,
        activation_authorization_factory=author,
        activation_authorization_validator=lambda _intent: None,
    )
    with pytest.raises(owner_reauth.OwnerGateOwnerReauthError):
        owner_reauth.validate_owner_reauth_receipt(
            authorization.receipt,
            public_key=authority.final_release_public_key,
            now_unix=pre_fixture.NOW + 2,
        )
    rotated = _activation_receipt(authority, runtime_sha256="1" * 64)
    assert rotated["owner_reauthentication_receipt_sha256"] != (
        authorization.receipt_sha256
    )

    removed = deferred._execute_with_provider(
        authority=authority,
        action=deferred.ACTION_REMOVE,
        provider=provider,
        journal=journal,
    )

    assert activated["activation_owner_reauthentication_receipt_sha256"] == (
        authorization.receipt_sha256
    )
    assert removed["paired_activation_success_receipt_sha256"] == (
        activated["receipt_sha256"]
    )
    assert removed["mutation_binding_present"] is False
    assert author_calls == 1


def test_remove_recovers_journaled_activation_intent_with_exact_live_binding(
    tmp_path: Path,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    provider.crash_after_effect = True
    journal = _journal(tmp_path / "journal")

    with pytest.raises(SystemExit, match="simulated_crash_after_cas"):
        _execute(authority, provider, journal)
    assert set(journal.list(authority.transaction_id)) == {"activate-intent"}
    provider.crash_after_effect = False

    removed = deferred._execute_with_provider(
        authority=authority,
        action=deferred.ACTION_REMOVE,
        provider=provider,
        journal=journal,
    )

    assert removed["mutation_binding_present"] is False
    assert [item["action"] for item in provider.mutate_calls] == [
        "activate",
        "remove",
    ]
    assert {
        "activate-intent",
        "activate-success",
        "remove-intent",
        "remove-operation",
        "remove-success",
    }.issubset(journal.list(authority.transaction_id))


def test_crash_after_cas_reconciles_from_byte_exact_intent(
    tmp_path: Path,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    provider.crash_after_effect = True
    journal = _journal(tmp_path / "journal")

    with pytest.raises(SystemExit, match="simulated_crash_after_cas"):
        _execute(authority, provider, journal)
    artifacts = journal.list(authority.transaction_id)
    assert set(artifacts) == {"activate-intent"}
    intent_before = artifacts["activate-intent"]

    provider.crash_after_effect = False
    recovered = _execute(authority, provider, journal)

    assert recovered["disposition"] == "reconciled_after_interruption"
    assert recovered["cloud_mutation_performed"] is True
    assert len(provider.mutate_calls) == 1
    assert journal.list(authority.transaction_id)["activate-intent"] == intent_before


def test_cas_precondition_drift_fails_before_provider_mutation(
    tmp_path: Path,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    provider.change_etag_on_observe = 2
    journal = _journal(tmp_path / "journal")

    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="manual_reconciliation_required",
    ):
        _execute(authority, provider, journal)

    assert provider.mutate_calls == []
    assert set(journal.list(authority.transaction_id)) == {"activate-intent"}


def test_unknown_provider_outcome_keeps_intent_and_retries_exact_cas(
    tmp_path: Path,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    provider.operation_state = "unknown"
    journal = _journal(tmp_path / "journal")

    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="manual_reconciliation_required",
    ):
        _execute(authority, provider, journal)
    assert set(journal.list(authority.transaction_id)) == {"activate-intent"}
    assert len(provider.mutate_calls) == 1

    provider.operation_state = "completed"
    recovered = _execute(authority, provider, journal)

    assert recovered["ok"] is True
    assert recovered["mutation_binding_present"] is True
    assert len(provider.mutate_calls) == 2
    assert provider.mutate_calls[0] == provider.mutate_calls[1]


def test_unknown_after_effect_is_reconciled_and_never_cas_retried(
    tmp_path: Path,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    provider.operation_state = "unknown"
    provider.unknown_after_effect = True
    journal = _journal(tmp_path / "journal")

    result = _execute(authority, provider, journal)
    replay = _execute(authority, provider, journal)

    artifacts = journal.list(authority.transaction_id)
    operation = artifacts["activate-operation"]
    assert result == replay
    assert result["disposition"] == "reconciled_after_interruption"
    assert result["operation_sha256"] == operation["operation_sha256"]
    assert operation["state"] == "unknown"
    assert len(provider.mutate_calls) == 1


def test_fresh_stale_slot_context_never_reuses_journaled_reauth_or_cas(
    tmp_path: Path,
) -> None:
    authority = _authority()
    seed = _FakeProvider(authority)
    seed.operation_state = "unknown"
    journal = _journal(tmp_path / "journal")
    with pytest.raises(deferred.OwnerGateDeferredMutationIamError):
        _execute(authority, seed, journal)
    stale_provider = _FakeProvider(authority)
    authorization_calls = 0
    validation_calls = 0

    def authorize() -> deferred._ActivationAuthorization:
        nonlocal authorization_calls
        authorization_calls += 1
        return _activation_authorization(authority)

    def validate(_intent: Mapping[str, Any]) -> None:
        nonlocal validation_calls
        validation_calls += 1

    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="fresh_attempt_conflict",
    ):
        deferred._execute_with_provider(
            authority=authority,
            action=deferred.ACTION_ACTIVATE,
            provider=stale_provider,
            journal=journal,
            activation_authorization_factory=authorize,
            activation_authorization_validator=validate,
            activation_attempt_index=0,
            require_new_activation_intent=True,
        )

    assert authorization_calls == 0
    assert validation_calls == 0
    assert stale_provider.observe_calls == 0
    assert stale_provider.mutate_calls == []
    assert set(journal.list(authority.transaction_id)) == {
        "activate-intent"
    }


def test_fresh_retry_conflicts_with_prior_durable_operation(
    tmp_path: Path,
) -> None:
    authority = _authority()
    journal = _journal(tmp_path / "journal")
    observation = deferred.DeferredMutationIamObservation(
        "absent",
        _policy(authority, present=False),
        "7" * 64,
    )
    intent = deferred._build_intent(
        authority=authority,
        action=deferred.ACTION_ACTIVATE,
        observation=observation,
        activation_authorization=_activation_authorization(authority),
        activation_success=None,
        activation_attempt_index=0,
    )
    with journal.transaction_lease(authority.transaction_id):
        stored_intent = journal.publish(
            authority.transaction_id,
            "activate-intent",
            intent,
        )
        journal.publish(
            authority.transaction_id,
            "activate-operation",
            deferred._operation_artifact(
                intent=stored_intent,
                operation=foundation_apply.OperationObservation(
                    "failed",
                    "8" * 64,
                    str(stored_intent["attempt_id"]),
                    cas_precondition_etag=str(
                        stored_intent["request_policy"]["etag"]
                    ),
                ),
            ),
        )
    provider = _FakeProvider(authority)

    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="fresh_attempt_conflict",
    ):
        deferred._execute_with_provider(
            authority=authority,
            action=deferred.ACTION_ACTIVATE,
            provider=provider,
            journal=journal,
            activation_authorization_factory=lambda: (
                _activation_authorization(authority)
            ),
            activation_authorization_validator=lambda _intent: None,
            activation_attempt_index=1,
            require_new_activation_intent=True,
        )

    assert provider.observe_calls == 0
    assert provider.mutate_calls == []
    assert "activate-retry-1-intent" not in journal.list(
        authority.transaction_id
    )


def test_intent_operation_and_terminal_exact_cross_bindings_reject_tamper(
    tmp_path: Path,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    journal = _journal(tmp_path / "journal")
    _execute(authority, provider, journal)
    artifacts = journal.list(authority.transaction_id)
    intent = artifacts["activate-intent"]
    operation = deferred._validate_operation(
        artifacts["activate-operation"],
        intent=intent,
    )
    terminal = artifacts["activate-success"]
    deferred._validate_success(
        terminal,
        authority=authority,
        action=deferred.ACTION_ACTIVATE,
        operation=operation,
    )

    wrong_attempt = {**intent, "attempt_id": "f" * 64}
    wrong_attempt["intent_sha256"] = deferred._sha256_json({
        name: item
        for name, item in wrong_attempt.items()
        if name != "intent_sha256"
    })
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="intent_invalid",
    ):
        deferred._validate_intent(
            wrong_attempt,
            authority=authority,
            action=deferred.ACTION_ACTIVATE,
        )

    wrong_etag = {**operation, "cas_precondition_etag": "etag-other"}
    wrong_etag["operation_sha256"] = deferred._sha256_json({
        name: item
        for name, item in wrong_etag.items()
        if name != "operation_sha256"
    })
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="operation_invalid",
    ):
        deferred._validate_operation(wrong_etag, intent=intent)

    wrong_terminal_operation = {**terminal, "operation_sha256": "f" * 64}
    wrong_terminal_operation["receipt_sha256"] = deferred._sha256_json({
        name: item
        for name, item in wrong_terminal_operation.items()
        if name != "receipt_sha256"
    })
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="success_invalid",
    ):
        deferred._validate_success(
            wrong_terminal_operation,
            authority=authority,
            action=deferred.ACTION_ACTIVATE,
            operation=operation,
        )

    impossible_disposition = {
        **terminal,
        "disposition": "already_absent",
        "cloud_mutation_performed": False,
    }
    impossible_disposition["receipt_sha256"] = deferred._sha256_json({
        name: item
        for name, item in impossible_disposition.items()
        if name != "receipt_sha256"
    })
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="success_invalid",
    ):
        deferred._validate_success(
            impossible_disposition,
            authority=authority,
            action=deferred.ACTION_ACTIVATE,
            operation=operation,
        )
    exact_observation = provider.observe_policy(authority)
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="postcondition_invalid",
    ):
        deferred._success_receipt(
            authority=authority,
            action=deferred.ACTION_ACTIVATE,
            intent=intent,
            operation=None,
            observation=exact_observation,
            disposition="applied",
        )


def test_recovery_with_failed_operation_publishes_failure_not_success(
    tmp_path: Path,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    journal = _journal(tmp_path / "journal")
    initial = provider.observe_policy(authority)
    intent = deferred._build_intent(
        authority=authority,
        action=deferred.ACTION_ACTIVATE,
        observation=initial,
        activation_authorization=_activation_authorization(authority),
        activation_success=None,
        activation_attempt_index=0,
    )
    with journal.transaction_lease(authority.transaction_id):
        stored_intent = journal.publish(
            authority.transaction_id,
            "activate-intent",
            intent,
        )
        failed = foundation_apply.OperationObservation(
            "failed",
            "a" * 64,
            str(stored_intent["attempt_id"]),
            cas_precondition_etag=str(
                stored_intent["request_policy"]["etag"]
            ),
        )
        journal.publish(
            authority.transaction_id,
            "activate-operation",
            deferred._operation_artifact(
                intent=stored_intent,
                operation=failed,
            ),
        )
    provider.policy = _policy(authority, present=True, etag="etag-external")

    with pytest.raises(deferred.OwnerGateDeferredMutationIamFailed) as failure:
        _execute(authority, provider, journal)

    artifacts = journal.list(authority.transaction_id)
    assert failure.value.receipt["failure_code"] == (
        "owner_gate_deferred_mutation_iam_provider_failed"
    )
    assert "activate-failure" in artifacts
    assert "activate-success" not in artifacts
    assert provider.mutate_calls == []


def test_pathless_historical_absent_with_durable_failed_operation_never_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    journal = _journal(tmp_path / "journal")
    initial = provider.observe_policy(authority)
    intent = deferred._build_intent(
        authority=authority,
        action=deferred.ACTION_ACTIVATE,
        observation=initial,
        activation_authorization=_activation_authorization(authority),
        activation_success=None,
        activation_attempt_index=0,
    )
    with journal.transaction_lease(authority.transaction_id):
        stored_intent = journal.publish(
            authority.transaction_id,
            "activate-intent",
            intent,
        )
        failed = foundation_apply.OperationObservation(
            "failed",
            "a" * 64,
            str(stored_intent["attempt_id"]),
            cas_precondition_etag=str(
                stored_intent["request_policy"]["etag"]
            ),
        )
        journal.publish(
            authority.transaction_id,
            "activate-operation",
            deferred._operation_artifact(
                intent=stored_intent,
                operation=failed,
            ),
        )
    frozen = _frozen_stub("historical-failed-operation")

    @contextmanager
    def historical_context(**_kwargs: Any):
        descriptor = deferred._basic_activation_attempt_descriptor(
            artifacts=journal.list(authority.transaction_id),
            transaction_id=authority.transaction_id,
            release_revision=authority.plan.spec.release_revision,
        )
        assert descriptor is not None
        yield frozen, authority, descriptor

    @contextmanager
    def forbid_fresh(**_kwargs: Any):
        raise AssertionError("durable operation incorrectly advanced a retry")
        yield  # pragma: no cover

    monkeypatch.setattr(
        deferred,
        "_historical_remove_authority",
        historical_context,
    )
    monkeypatch.setattr(
        deferred,
        "_fresh_activation_authority_context",
        forbid_fresh,
    )
    monkeypatch.setattr(
        deferred,
        "_TrustedGcloudDeferredMutationIamProvider",
        lambda **_kwargs: provider,
    )
    monkeypatch.setattr(
        deferred.owner_reauth,
        "produce_owner_reauth_receipt",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("durable operation requested fresh reauth")
        ),
    )
    executable, configuration, identity = _pathless_capabilities(
        monkeypatch,
        transaction_id=authority.transaction_id,
    )

    with pytest.raises(deferred.OwnerGateDeferredMutationIamFailed):
        deferred._execute_pathless(
            action=deferred.ACTION_ACTIVATE,
            release_revision=authority.plan.spec.release_revision,
            gcloud_executable=executable,
            gcloud_configuration=configuration,
            owner_identity=identity,
            reauth_runner=SimpleNamespace(),  # type: ignore[arg-type]
            now_unix=lambda: pre_fixture.NOW,
            journal=journal,
        )

    artifacts = journal.list(authority.transaction_id)
    assert set(artifacts) == {
        "activate-intent",
        "activate-operation",
        "activate-failure",
    }
    assert provider.mutate_calls == []


def test_remove_intent_must_pair_to_actual_activation_success(
    tmp_path: Path,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    journal = _journal(tmp_path / "journal")
    activated = _execute(authority, provider, journal)
    observation = provider.observe_policy(authority)
    remove_intent = dict(deferred._build_intent(
        authority=authority,
        action=deferred.ACTION_REMOVE,
        observation=observation,
        activation_authorization=None,
        activation_success=activated,
        activation_attempt_index=None,
    ))
    remove_intent["paired_activation_success_receipt_sha256"] = "f" * 64
    remove_intent["attempt_id"] = deferred._intent_attempt_id(
        transaction_id=remove_intent["transaction_id"],
        action=remove_intent["action"],
        activation_attempt_index=remove_intent["activation_attempt_index"],
        intent_artifact_name=remove_intent["intent_artifact_name"],
        precondition=remove_intent["precondition"],
        request_policy=remove_intent["request_policy"],
        reauthentication_receipt_sha256=remove_intent[
            "activation_owner_reauthentication_receipt_sha256"
        ],
        paired_activation_success_receipt_sha256=remove_intent[
            "paired_activation_success_receipt_sha256"
        ],
    )
    remove_intent["intent_sha256"] = deferred._sha256_json({
        name: item
        for name, item in remove_intent.items()
        if name != "intent_sha256"
    })
    with journal.transaction_lease(authority.transaction_id):
        journal.publish(
            authority.transaction_id,
            "remove-intent",
            remove_intent,
        )

    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="journal_invalid",
    ):
        deferred._execute_with_provider(
            authority=authority,
            action=deferred.ACTION_REMOVE,
            provider=provider,
            journal=journal,
        )

    assert len(provider.mutate_calls) == 1
    assert "remove-operation" not in journal.list(authority.transaction_id)


def test_pathless_retry_uses_new_evidence_new_reauth_and_retry_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_a = _authority()
    authority_b = _authority_variant(
        authority_a,
        evidence_digit="6",
        network_digit="7",
    )
    authorities = {
        str(item.lineage["inert_evidence_set_sha256"]): item
        for item in (authority_a, authority_b)
    }
    frozen = {
        evidence: _frozen_stub(evidence)
        for evidence in authorities
    }
    journal = _journal(tmp_path / "journal")
    provider_a = _FakeProvider(authority_a)
    provider_a.operation_state = "unknown"
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="manual_reconciliation_required",
    ):
        _execute(authority_a, provider_a, journal)
    intent_a = journal.list(authority_a.transaction_id)["activate-intent"]
    intent_a_bytes = deferred._canonical(intent_a)
    provider_b = _FakeProvider(authority_b)
    providers = {
        str(authority_a.lineage["inert_evidence_set_sha256"]): provider_a,
        str(authority_b.lineage["inert_evidence_set_sha256"]): provider_b,
    }
    fresh_calls = 0

    @contextmanager
    def historical_context(**kwargs: Any):
        assert kwargs == {
            "release_revision": authority_a.plan.spec.release_revision,
            "journal": journal,
            "now_unix": pre_fixture.NOW,
        }
        artifacts = journal.list(authority_a.transaction_id)
        descriptor = deferred._basic_activation_attempt_descriptor(
            artifacts=artifacts,
            transaction_id=authority_a.transaction_id,
            release_revision=authority_a.plan.spec.release_revision,
        )
        assert descriptor is not None
        evidence = str(
            descriptor.intent["inert_evidence_set_sha256"]
        )
        yield frozen[evidence], authorities[evidence], descriptor

    @contextmanager
    def fresh_context(**kwargs: Any):
        nonlocal fresh_calls
        fresh_calls += 1
        assert kwargs == {
            "release_revision": authority_a.plan.spec.release_revision,
            "now_unix": pre_fixture.NOW,
        }
        evidence = str(authority_b.lineage["inert_evidence_set_sha256"])
        yield frozen[evidence], authority_b, None

    monkeypatch.setattr(
        deferred,
        "_historical_remove_authority",
        historical_context,
    )
    monkeypatch.setattr(
        deferred,
        "_fresh_activation_authority_context",
        fresh_context,
    )
    monkeypatch.setattr(
        deferred,
        "_TrustedGcloudDeferredMutationIamProvider",
        lambda *, authority, **_kwargs: providers[
            str(authority.lineage["inert_evidence_set_sha256"])
        ],
    )
    reauth_calls = 0
    receipt_b = _rotated_activation_receipt(
        authority_b,
        marker="retry-b",
    )

    def produce_reauthentication(**_kwargs: Any) -> Mapping[str, Any]:
        nonlocal reauth_calls
        reauth_calls += 1
        return receipt_b

    monkeypatch.setattr(
        deferred.owner_reauth,
        "produce_owner_reauth_receipt",
        produce_reauthentication,
    )
    executable, configuration, identity = _pathless_capabilities(
        monkeypatch,
        transaction_id=authority_a.transaction_id,
    )

    result = deferred._execute_pathless(
        action=deferred.ACTION_ACTIVATE,
        release_revision=authority_a.plan.spec.release_revision,
        gcloud_executable=executable,
        gcloud_configuration=configuration,
        owner_identity=identity,
        reauth_runner=SimpleNamespace(),  # type: ignore[arg-type]
        now_unix=lambda: pre_fixture.NOW,
        journal=journal,
    )

    artifacts = journal.list(authority_a.transaction_id)
    assert result["activation_attempt_index"] == 1
    assert result["intent_artifact_name"] == "activate-retry-1-intent"
    assert result["inert_evidence_set_sha256"] == (
        authority_b.lineage["inert_evidence_set_sha256"]
    )
    assert deferred._canonical(artifacts["activate-intent"]) == intent_a_bytes
    assert artifacts["activate-retry-1-intent"][
        "activation_owner_reauthentication_receipt_sha256"
    ] == receipt_b["owner_reauthentication_receipt_sha256"]
    assert artifacts["activate-success"]["intent_sha256"] == artifacts[
        "activate-retry-1-intent"
    ]["intent_sha256"]
    assert set(artifacts) == {
        "activate-intent",
        "activate-retry-1-intent",
        "activate-retry-1-operation",
        "activate-success",
    }
    assert len(provider_a.mutate_calls) == 1
    assert len(provider_b.mutate_calls) == 1
    assert fresh_calls == 1
    assert reauth_calls == 1


def test_pathless_activation_retry_slots_are_bounded_and_contiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_a = _authority()
    variants = (
        authority_a,
        _authority_variant(
            authority_a,
            evidence_digit="6",
            network_digit="7",
        ),
        _authority_variant(
            authority_a,
            evidence_digit="7",
            network_digit="8",
        ),
        _authority_variant(
            authority_a,
            evidence_digit="8",
            network_digit="9",
        ),
    )
    authorities = {
        str(item.lineage["inert_evidence_set_sha256"]): item
        for item in variants
    }
    frozen = {
        evidence: _frozen_stub(evidence)
        for evidence in authorities
    }
    providers = {
        evidence: _FakeProvider(authority)
        for evidence, authority in authorities.items()
    }
    for provider in providers.values():
        provider.operation_state = "unknown"
    journal = _journal(tmp_path / "journal")
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="manual_reconciliation_required",
    ):
        _execute(
            authority_a,
            providers[str(authority_a.lineage["inert_evidence_set_sha256"])],
            journal,
        )
    fresh_queue = list(variants[1:])

    @contextmanager
    def historical_context(**_kwargs: Any):
        artifacts = journal.list(authority_a.transaction_id)
        descriptor = deferred._basic_activation_attempt_descriptor(
            artifacts=artifacts,
            transaction_id=authority_a.transaction_id,
            release_revision=authority_a.plan.spec.release_revision,
        )
        assert descriptor is not None
        evidence = str(
            descriptor.intent["inert_evidence_set_sha256"]
        )
        yield frozen[evidence], authorities[evidence], descriptor

    @contextmanager
    def fresh_context(**_kwargs: Any):
        authority = fresh_queue.pop(0)
        evidence = str(authority.lineage["inert_evidence_set_sha256"])
        yield frozen[evidence], authority, None

    monkeypatch.setattr(
        deferred,
        "_historical_remove_authority",
        historical_context,
    )
    monkeypatch.setattr(
        deferred,
        "_fresh_activation_authority_context",
        fresh_context,
    )
    monkeypatch.setattr(
        deferred,
        "_TrustedGcloudDeferredMutationIamProvider",
        lambda *, authority, **_kwargs: providers[
            str(authority.lineage["inert_evidence_set_sha256"])
        ],
    )
    receipts = [
        _rotated_activation_receipt(authority, marker=f"retry-{index}")
        for index, authority in enumerate(variants[1:], start=1)
    ]
    produced: list[str] = []

    def produce_reauthentication(**_kwargs: Any) -> Mapping[str, Any]:
        receipt = receipts[len(produced)]
        produced.append(
            str(receipt["owner_reauthentication_receipt_sha256"])
        )
        return receipt

    monkeypatch.setattr(
        deferred.owner_reauth,
        "produce_owner_reauth_receipt",
        produce_reauthentication,
    )
    executable, configuration, identity = _pathless_capabilities(
        monkeypatch,
        transaction_id=authority_a.transaction_id,
    )
    arguments = {
        "action": deferred.ACTION_ACTIVATE,
        "release_revision": authority_a.plan.spec.release_revision,
        "gcloud_executable": executable,
        "gcloud_configuration": configuration,
        "owner_identity": identity,
        "reauth_runner": SimpleNamespace(),
        "now_unix": lambda: pre_fixture.NOW,
        "journal": journal,
    }
    for _index in range(1, deferred.MAX_ACTIVATION_ATTEMPTS):
        with pytest.raises(
            deferred.OwnerGateDeferredMutationIamError,
            match="manual_reconciliation_required",
        ):
            deferred._execute_pathless(**arguments)  # type: ignore[arg-type]

    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="attempts_exhausted",
    ):
        deferred._execute_pathless(**arguments)  # type: ignore[arg-type]

    artifacts = journal.list(authority_a.transaction_id)
    expected_intents = {
        deferred._intent_artifact_name(deferred.ACTION_ACTIVATE, index)
        for index in range(deferred.MAX_ACTIVATION_ATTEMPTS)
    }
    assert set(artifacts) == expected_intents
    assert fresh_queue == []
    assert len(produced) == deferred.MAX_ACTIVATION_ATTEMPTS - 1
    assert len(set(produced)) == len(produced)
    assert {
        str(value["inert_evidence_set_sha256"])
        for value in artifacts.values()
    } == set(authorities)

    retry = dict(artifacts["activate-retry-1-intent"])
    gap = {"activate-retry-1-intent": retry}
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="journal_invalid",
    ):
        deferred._basic_activation_attempt_descriptor(
            artifacts=gap,
            transaction_id=authority_a.transaction_id,
            release_revision=authority_a.plan.spec.release_revision,
        )
    duplicate = {
        **retry,
        "activation_owner_reauthentication_receipt_sha256": artifacts[
            "activate-intent"
        ][
            "activation_owner_reauthentication_receipt_sha256"
        ],
    }
    duplicate["intent_sha256"] = deferred._sha256_json({
        name: item
        for name, item in duplicate.items()
        if name != "intent_sha256"
    })
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="journal_invalid",
    ):
        deferred._basic_activation_attempt_descriptor(
            artifacts={
                "activate-intent": artifacts["activate-intent"],
                "activate-retry-1-intent": duplicate,
            },
            transaction_id=authority_a.transaction_id,
            release_revision=authority_a.plan.spec.release_revision,
        )


def test_pathless_retry_can_replay_same_still_fresh_evidence_with_new_reauth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    provider.operation_state = "unknown"
    journal = _journal(tmp_path / "journal")
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="manual_reconciliation_required",
    ):
        _execute(authority, provider, journal)
    original_intent = journal.list(authority.transaction_id)[
        "activate-intent"
    ]
    original_reauth = str(
        original_intent[
            "activation_owner_reauthentication_receipt_sha256"
        ]
    )
    frozen = _frozen_stub("same-still-fresh-evidence-a")

    @contextmanager
    def historical_context(**_kwargs: Any):
        descriptor = deferred._basic_activation_attempt_descriptor(
            artifacts=journal.list(authority.transaction_id),
            transaction_id=authority.transaction_id,
            release_revision=authority.plan.spec.release_revision,
        )
        assert descriptor is not None
        yield frozen, authority, descriptor

    @contextmanager
    def fresh_context(**_kwargs: Any):
        # The sealed fresh-evidence loader is allowed to replay the exact same
        # still-fresh immutable transaction A.
        yield frozen, authority, None

    provider.operation_state = "completed"
    monkeypatch.setattr(
        deferred,
        "_historical_remove_authority",
        historical_context,
    )
    monkeypatch.setattr(
        deferred,
        "_fresh_activation_authority_context",
        fresh_context,
    )
    monkeypatch.setattr(
        deferred,
        "_TrustedGcloudDeferredMutationIamProvider",
        lambda **_kwargs: provider,
    )
    fresh_receipt = _rotated_activation_receipt(
        authority,
        marker="same-evidence-new-reauth",
    )
    monkeypatch.setattr(
        deferred.owner_reauth,
        "produce_owner_reauth_receipt",
        lambda **_kwargs: fresh_receipt,
    )
    executable, configuration, identity = _pathless_capabilities(
        monkeypatch,
        transaction_id=authority.transaction_id,
    )

    result = deferred._execute_pathless(
        action=deferred.ACTION_ACTIVATE,
        release_revision=authority.plan.spec.release_revision,
        gcloud_executable=executable,
        gcloud_configuration=configuration,
        owner_identity=identity,
        reauth_runner=SimpleNamespace(),  # type: ignore[arg-type]
        now_unix=lambda: pre_fixture.NOW,
        journal=journal,
    )

    artifacts = journal.list(authority.transaction_id)
    retry = artifacts["activate-retry-1-intent"]
    assert result["activation_attempt_index"] == 1
    assert retry["inert_evidence_set_sha256"] == original_intent[
        "inert_evidence_set_sha256"
    ]
    assert retry[
        "activation_owner_reauthentication_receipt_sha256"
    ] != original_reauth
    assert retry[
        "activation_owner_reauthentication_receipt_sha256"
    ] == fresh_receipt["owner_reauthentication_receipt_sha256"]
    assert len(provider.mutate_calls) == 2

    replay = deferred._execute_pathless(
        action=deferred.ACTION_ACTIVATE,
        release_revision=authority.plan.spec.release_revision,
        gcloud_executable=executable,
        gcloud_configuration=configuration,
        owner_identity=identity,
        reauth_runner=SimpleNamespace(),  # type: ignore[arg-type]
        now_unix=lambda: pre_fixture.NOW,
        journal=journal,
    )
    removed = deferred._execute_pathless(
        action=deferred.ACTION_REMOVE,
        release_revision=authority.plan.spec.release_revision,
        gcloud_executable=executable,
        gcloud_configuration=configuration,
        owner_identity=identity,
        reauth_runner=None,
        now_unix=lambda: pre_fixture.NOW + 10_000,
        journal=journal,
    )
    assert replay == result
    assert removed["paired_activation_success_receipt_sha256"] == result[
        "receipt_sha256"
    ]
    assert len(provider.mutate_calls) == 3


def test_propagation_race_reconciles_historical_intent_without_retry_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_a = _authority()
    authority_b = _authority_variant(
        authority_a,
        evidence_digit="6",
        network_digit="7",
    )
    journal = _journal(tmp_path / "journal")
    seed = _FakeProvider(authority_a)
    seed.operation_state = "unknown"
    with pytest.raises(deferred.OwnerGateDeferredMutationIamError):
        _execute(authority_a, seed, journal)
    intent_before = deferred._canonical(
        journal.list(authority_a.transaction_id)["activate-intent"]
    )
    frozen_a = _frozen_stub("a")
    frozen_b = _frozen_stub("b")
    historical_calls = 0
    fresh_calls = 0

    @contextmanager
    def historical_context(**_kwargs: Any):
        nonlocal historical_calls
        historical_calls += 1
        descriptor = deferred._basic_activation_attempt_descriptor(
            artifacts=journal.list(authority_a.transaction_id),
            transaction_id=authority_a.transaction_id,
            release_revision=authority_a.plan.spec.release_revision,
        )
        assert descriptor is not None
        yield frozen_a, authority_a, descriptor

    @contextmanager
    def fresh_context(**_kwargs: Any):
        nonlocal fresh_calls
        fresh_calls += 1
        yield frozen_b, authority_b, None

    historical_absent = _FakeProvider(authority_a)
    historical_exact = _FakeProvider(authority_a, present=True)
    fresh_exact = _FakeProvider(authority_b, present=True)
    authority_a_provider_calls = 0

    def provider_factory(*, authority: Any, **_kwargs: Any) -> _FakeProvider:
        nonlocal authority_a_provider_calls
        if authority is authority_b:
            return fresh_exact
        assert authority is authority_a
        authority_a_provider_calls += 1
        return (
            historical_absent
            if authority_a_provider_calls == 1
            else historical_exact
        )

    monkeypatch.setattr(
        deferred,
        "_historical_remove_authority",
        historical_context,
    )
    monkeypatch.setattr(
        deferred,
        "_fresh_activation_authority_context",
        fresh_context,
    )
    monkeypatch.setattr(
        deferred,
        "_TrustedGcloudDeferredMutationIamProvider",
        provider_factory,
    )
    reauth_calls = 0

    def forbid_reauthentication(**_kwargs: Any) -> Mapping[str, Any]:
        nonlocal reauth_calls
        reauth_calls += 1
        raise AssertionError("propagation reconciliation requested reauth")

    monkeypatch.setattr(
        deferred.owner_reauth,
        "produce_owner_reauth_receipt",
        forbid_reauthentication,
    )
    executable, configuration, identity = _pathless_capabilities(
        monkeypatch,
        transaction_id=authority_a.transaction_id,
    )

    result = deferred._execute_pathless(
        action=deferred.ACTION_ACTIVATE,
        release_revision=authority_a.plan.spec.release_revision,
        gcloud_executable=executable,
        gcloud_configuration=configuration,
        owner_identity=identity,
        reauth_runner=SimpleNamespace(),  # type: ignore[arg-type]
        now_unix=lambda: pre_fixture.NOW,
        journal=journal,
    )

    artifacts = journal.list(authority_a.transaction_id)
    assert result["disposition"] == "reconciled_after_interruption"
    assert deferred._canonical(artifacts["activate-intent"]) == intent_before
    assert "activate-retry-1-intent" not in artifacts
    assert artifacts["activate-success"]["intent_sha256"] == artifacts[
        "activate-intent"
    ]["intent_sha256"]
    assert historical_calls == 2
    assert fresh_calls == 1
    assert reauth_calls == 0
    assert historical_absent.mutate_calls == []
    assert historical_exact.mutate_calls == []
    assert fresh_exact.mutate_calls == []


def test_pathless_remove_uses_historical_activation_after_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_a = _authority()
    authority_b = _authority_variant(
        authority_a,
        evidence_digit="6",
        network_digit="7",
    )
    provider = _FakeProvider(authority_a)
    journal = _journal(tmp_path / "journal")
    activated = _execute(authority_a, provider, journal)
    frozen_a = _frozen_stub("historical-a")
    historical_calls = 0

    @contextmanager
    def historical_context(**kwargs: Any):
        nonlocal historical_calls
        historical_calls += 1
        assert kwargs["now_unix"] == pre_fixture.NOW + 10_000
        descriptor = deferred._basic_activation_attempt_descriptor(
            artifacts=journal.list(authority_a.transaction_id),
            transaction_id=authority_a.transaction_id,
            release_revision=authority_a.plan.spec.release_revision,
        )
        assert descriptor is not None
        yield frozen_a, authority_a, descriptor

    @contextmanager
    def forbid_fresh(**_kwargs: Any):
        raise AssertionError(
            "remove tried to adopt differing fresh evidence B: "
            f"{authority_b.lineage['inert_evidence_set_sha256']}"
        )
        yield  # pragma: no cover

    monkeypatch.setattr(
        deferred,
        "_historical_remove_authority",
        historical_context,
    )
    monkeypatch.setattr(
        deferred,
        "_fresh_activation_authority_context",
        forbid_fresh,
    )
    monkeypatch.setattr(
        deferred,
        "_TrustedGcloudDeferredMutationIamProvider",
        lambda *, authority, **_kwargs: provider,
    )
    monkeypatch.setattr(
        deferred.owner_reauth,
        "produce_owner_reauth_receipt",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("remove requested a fresh reauth")
        ),
    )
    executable, configuration, identity = _pathless_capabilities(
        monkeypatch,
        transaction_id=authority_a.transaction_id,
    )

    removed = deferred._execute_pathless(
        action=deferred.ACTION_REMOVE,
        release_revision=authority_a.plan.spec.release_revision,
        gcloud_executable=executable,
        gcloud_configuration=configuration,
        owner_identity=identity,
        reauth_runner=None,
        now_unix=lambda: pre_fixture.NOW + 10_000,
        journal=journal,
    )

    assert removed["paired_activation_success_receipt_sha256"] == activated[
        "receipt_sha256"
    ]
    assert removed["inert_evidence_set_sha256"] == (
        authority_a.lineage["inert_evidence_set_sha256"]
    )
    assert removed["mutation_binding_present"] is False
    assert historical_calls == 1
    assert [call["action"] for call in provider.mutate_calls] == [
        "activate",
        "remove",
    ]


def test_global_contract_owner_blocks_distinct_release_until_paired_remove(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_r1 = _authority()
    authority_r2 = _distinct_release_authority(authority_r1)
    journal = _journal(tmp_path / "journal")
    provider_r1 = _FakeProvider(authority_r1)
    provider_r2 = _FakeProvider(authority_r2)
    activated_r1 = _execute(authority_r1, provider_r1, journal)
    frozen = {
        authority_r1.plan.spec.release_revision: _frozen_stub("r1"),
        authority_r2.plan.spec.release_revision: _frozen_stub("r2"),
    }
    authorities = {
        authority_r1.plan.spec.release_revision: authority_r1,
        authority_r2.plan.spec.release_revision: authority_r2,
    }

    @contextmanager
    def historical_context(*, release_revision: str, **_kwargs: Any):
        authority = authorities[release_revision]
        descriptor = deferred._basic_activation_attempt_descriptor(
            artifacts=journal.list(authority.transaction_id),
            transaction_id=authority.transaction_id,
            release_revision=release_revision,
        )
        if descriptor is None:
            raise deferred.OwnerGateDeferredMutationIamError(
                "owner_gate_deferred_mutation_iam_activation_missing"
            )
        yield frozen[release_revision], authority, descriptor

    @contextmanager
    def fresh_context(*, release_revision: str, **_kwargs: Any):
        assert release_revision == authority_r2.plan.spec.release_revision
        yield frozen[release_revision], authority_r2, None

    monkeypatch.setattr(
        deferred,
        "_historical_remove_authority",
        historical_context,
    )
    monkeypatch.setattr(
        deferred,
        "_fresh_activation_authority_context",
        fresh_context,
    )
    monkeypatch.setattr(
        deferred,
        "_TrustedGcloudDeferredMutationIamProvider",
        lambda *, authority, **_kwargs: (
            provider_r1 if authority is authority_r1 else provider_r2
        ),
    )
    monkeypatch.setattr(
        deferred.owner_reauth,
        "produce_owner_reauth_receipt",
        lambda **_kwargs: _activation_receipt(authority_r2),
    )
    executable, configuration, identity = _pathless_capabilities(
        monkeypatch,
        transaction_id=authority_r2.transaction_id,
    )
    common = {
        "release_revision": authority_r2.plan.spec.release_revision,
        "gcloud_executable": executable,
        "gcloud_configuration": configuration,
        "owner_identity": identity,
        "now_unix": lambda: pre_fixture.NOW,
        "journal": journal,
    }

    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="contract_owned",
    ):
        deferred._execute_pathless(
            action=deferred.ACTION_ACTIVATE,
            reauth_runner=SimpleNamespace(),  # type: ignore[arg-type]
            **common,  # type: ignore[arg-type]
        )
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="contract_owned",
    ):
        deferred._execute_pathless(
            action=deferred.ACTION_REMOVE,
            reauth_runner=None,
            **common,  # type: ignore[arg-type]
        )
    assert provider_r2.observe_calls == 0
    assert provider_r2.mutate_calls == []
    assert journal.list(authority_r2.transaction_id) == {}

    removed_r1 = _execute(
        authority_r1,
        provider_r1,
        journal,
        action=deferred.ACTION_REMOVE,
    )
    assert removed_r1["paired_activation_success_receipt_sha256"] == (
        activated_r1["receipt_sha256"]
    )

    activated_r2 = deferred._execute_pathless(
        action=deferred.ACTION_ACTIVATE,
        reauth_runner=SimpleNamespace(),  # type: ignore[arg-type]
        **common,  # type: ignore[arg-type]
    )

    assert activated_r2["transaction_id"] == authority_r2.transaction_id
    assert [call["action"] for call in provider_r1.mutate_calls] == [
        deferred.ACTION_ACTIVATE,
        deferred.ACTION_REMOVE,
    ]
    assert [call["action"] for call in provider_r2.mutate_calls] == [
        deferred.ACTION_ACTIVATE,
    ]


def test_historical_snapshot_accepts_expiry_but_rejects_changed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inert = deferred.inert_observation
    fixed = inert_fixture._fixed_evidence_store(tmp_path, monkeypatch)
    fixed.inputs.bundle_stream = object()
    fixed.inputs.assert_stable = lambda: None
    inert._publish_evidence(
        phase_root=fixed.phase_root,
        receipt=fixed.receipt,
        payloads=fixed.payloads,
    )
    monkeypatch.setattr(
        inert._PinnedObservationInputs,
        "load",
        lambda revision: fixed.inputs,
    )
    monkeypatch.setattr(
        inert,
        "_load_release_binding",
        lambda revision, stream: fixed.binding,
    )
    monkeypatch.setattr(
        inert,
        "_load_successful_foundation",
        lambda revision: fixed.loaded,
    )
    monkeypatch.setattr(
        inert,
        "_bind_release_to_foundation",
        lambda binding, loaded: None,
    )
    public_keys = {
        "network": fixed.network_key.public_key(),
        "cloud": fixed.cloud_key.public_key(),
        "host": fixed.host_key.public_key(),
    }
    monkeypatch.setattr(
        inert,
        "_collector_key",
        lambda revision, *, role, expected_key_id: public_keys[role],
    )
    evidence_set = str(fixed.receipt["evidence_set_sha256"])
    expired_at = pre_fixture.NOW + 10_000

    with inert._historical_inert_evidence_snapshot(
        release_revision=inert_fixture.REVISION,
        evidence_set_sha256=evidence_set,
        now_unix=expired_at,
    ) as historical:
        assert historical.receipt == fixed.receipt
        assert historical.network_evidence.evidence_sha256 == (
            fixed.network_evidence.evidence_sha256
        )
        assert historical.transaction_root.name == evidence_set

    receipt_path = fixed.phase_root / evidence_set / inert.RECEIPT_NAME
    original = receipt_path.read_bytes()
    receipt_path.chmod(0o600)
    receipt_path.write_bytes(original[:-2] + b"0\n")
    receipt_path.chmod(0o400)
    with pytest.raises(deferred.launcher.OwnerLauncherError):
        with inert._historical_inert_evidence_snapshot(
            release_revision=inert_fixture.REVISION,
            evidence_set_sha256=evidence_set,
            now_unix=expired_at,
        ):
            pytest.fail("changed historical bytes were accepted")


def test_exact_conditioned_binding_rejects_multi_member_same_role() -> None:
    authority = _authority()
    mixed = _policy(
        authority,
        present=False,
        extra_bindings=[{
            "role": authority.contract.role,
            "members": [authority.contract.member, "user:other@example.com"],
            "condition": dict(authority.contract.condition),
        }],
    )
    assert deferred._classify_policy(
        mixed,
        contract=authority.contract,
    ) == "drift"


def test_unjournaled_exact_binding_is_never_adopted(tmp_path: Path) -> None:
    authority = _authority()
    provider = _FakeProvider(authority, present=True)
    journal = _journal(tmp_path / "journal")

    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="unjournaled_binding_present",
    ):
        _execute(authority, provider, journal)

    assert provider.mutate_calls == []
    assert journal.list(authority.transaction_id) == {}


def test_paired_remove_is_cas_bound_and_activate_after_remove_is_refused(
    tmp_path: Path,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    journal = _journal(tmp_path / "journal")
    _execute(authority, provider, journal)

    removed = _execute(
        authority,
        provider,
        journal,
        action=deferred.ACTION_REMOVE,
    )
    replay = _execute(
        authority,
        provider,
        journal,
        action=deferred.ACTION_REMOVE,
    )

    assert removed == replay
    assert removed["mutation_binding_present"] is False
    assert [item["action"] for item in provider.mutate_calls] == [
        "activate",
        "remove",
    ]
    remove_call = provider.mutate_calls[1]
    assert remove_call["request_policy"]["etag"] == (
        remove_call["precondition"]["policy_etag"]
    )
    assert remove_call["request_policy"]["version"] == 3
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="lifecycle_complete",
    ):
        _execute(authority, provider, journal)


def test_paired_remove_survives_live_foundation_drift(
    tmp_path: Path,
) -> None:
    authority = _authority()
    provider = _FakeProvider(authority)
    journal = _journal(tmp_path / "journal")
    activated = _execute(authority, provider, journal)
    provider.foundation_drift = True
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="simulated_live_foundation_drift",
    ):
        deferred._observe(
            provider,
            authority,
            action=deferred.ACTION_ACTIVATE,
        )

    removed = _execute(
        authority,
        provider,
        journal,
        action=deferred.ACTION_REMOVE,
    )

    assert removed["paired_activation_success_receipt_sha256"] == activated[
        "receipt_sha256"
    ]
    assert removed["mutation_binding_present"] is False
    assert provider.lineage_actions[-2:] == [
        deferred.ACTION_REMOVE,
        deferred.ACTION_REMOVE,
    ]
    assert [call["action"] for call in provider.mutate_calls] == [
        deferred.ACTION_ACTIVATE,
        deferred.ACTION_REMOVE,
    ]


def test_remove_without_journaled_activation_is_refused(tmp_path: Path) -> None:
    authority = _authority()
    provider = _FakeProvider(authority, present=True)
    journal = _journal(tmp_path / "journal")

    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="activation_missing",
    ):
        _execute(
            authority,
            provider,
            journal,
            action=deferred.ACTION_REMOVE,
        )
    assert provider.mutate_calls == []


def test_fixed_plan_rejects_caller_selected_role_member_or_step() -> None:
    plan = foundation_fixture._plan()
    original = plan.deferred_mutation_iam_steps[0]
    bad = replace(
        plan,
        deferred_mutation_iam_steps=(
            replace(
                original,
                argv=tuple(
                    "--role=roles/owner" if item.startswith("--role=") else item
                    for item in original.argv
                ),
            ),
        ),
    )
    with pytest.raises(
        deferred.OwnerGateDeferredMutationIamError,
        match="plan_invalid",
    ):
        deferred._fixed_contract(bad)


def test_journal_inventory_is_closed_and_publication_is_byte_exact(
    tmp_path: Path,
) -> None:
    authority = _authority()
    journal = _journal(tmp_path / "journal")
    value = {"schema": "test", "value": 1}

    with journal.transaction_lease(authority.transaction_id):
        first = journal.publish(
            authority.transaction_id,
            "activate-intent",
            value,
        )
        assert journal.publish(
            authority.transaction_id,
            "activate-intent",
            value,
        ) == first
        with pytest.raises(RuntimeError, match="artifact_diverged"):
            journal.publish(
                authority.transaction_id,
                "activate-intent",
                {"schema": "test", "value": 2},
            )
        with pytest.raises(RuntimeError, match="artifact_name_invalid"):
            journal.publish(
                authority.transaction_id,
                "caller-selected",
                value,
            )


def test_contract_lease_rejects_non_transaction_root_entry(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path / "journal")
    with journal.contract_lease():
        pass
    garbage = journal.root / "caller-selected"
    garbage.write_text("not a transaction", encoding="ascii")

    with pytest.raises(
        RuntimeError,
        match="contract_inventory_invalid",
    ):
        with journal.contract_lease():
            pytest.fail("malformed contract root was accepted")
