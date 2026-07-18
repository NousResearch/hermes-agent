from __future__ import annotations

import copy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import passkey_v2_service as service
from scripts.canary import production_cutover_owner_launcher as owner
from scripts.canary import production_cutover_passkey as passkey
from tests.gateway.test_canonical_writer_production_cutover import (
    NOW,
    Services,
    _isolated_canary_goal_prerequisite,
    _runtime_attestation,
)
from tests.scripts.canary.test_production_cutover_owner_launcher import (
    REVISION,
    _collector_receipt,
)


def _freeze_publication() -> dict:
    _plan, _approval, publication = owner.author_freeze(
        collector_receipt=_collector_receipt(NOW, Services()),
        release_revision=REVISION,
        owner_subject_sha256="a" * 64,
        private_key=Ed25519PrivateKey.generate(),
        owner_runtime_attestation=_runtime_attestation(),
        isolated_canary_goal_prerequisite=(
            _isolated_canary_goal_prerequisite()
        ),
        truth_mode="start_new_truth_epoch",
        now_unix=NOW,
    )
    return publication


def _cutover_action(publication: dict) -> dict:
    return dict(passkey.build_cutover_action_envelope(
        freeze_publication=publication,
        authority_release_sha=REVISION,
        authority_manifest_sha256="1" * 64,
        authority_host_receipt_sha256="2" * 64,
        issued_at_unix=NOW,
    ))


def test_exact_cutover_action_is_fully_bound_and_has_no_generic_escape() -> None:
    publication = _freeze_publication()
    action = _cutover_action(publication)

    validated = passkey.validate_cutover_action_envelope(
        action,
        freeze_publication=publication,
    )
    facts = passkey.mechanical_approval_facts(validated)

    assert validated["action_payload"]["allowed_operations"] == list(
        passkey.ALLOWED_OPERATIONS
    )
    assert facts["production_vm_instance_id"] == (
        passkey.PRODUCTION_VM_INSTANCE_ID
    )
    assert facts["single_use"] is True
    assert facts["user_verification_required"] is True
    assert facts["totp_available"] is False
    assert facts["caller_selected_commands_allowed"] is False
    assert facts["caller_selected_paths_allowed"] is False
    assert facts["caller_selected_targets_allowed"] is False


def test_cutover_action_rejects_rehashed_caller_selected_target() -> None:
    action = _cutover_action(_freeze_publication())
    action["action_payload"]["caller_selected_targets_allowed"] = True
    action["envelope_sha256"] = protocol.sha256_json({
        name: item
        for name, item in action.items()
        if name != "envelope_sha256"
    })

    with pytest.raises(
        passkey.ProductionCutoverPasskeyError,
        match="production_cutover_passkey_action_invalid",
    ):
        passkey.validate_cutover_action_envelope(action)


def test_cutover_action_rejects_substituted_freeze_publication() -> None:
    first = _freeze_publication()
    second = _freeze_publication()
    action = _cutover_action(first)

    with pytest.raises(
        passkey.ProductionCutoverPasskeyError,
        match="production_cutover_passkey_publication_binding_invalid",
    ):
        passkey.validate_cutover_action_envelope(
            action,
            freeze_publication=second,
        )


def test_service_dispatch_is_closed_to_the_two_reviewed_action_schemas() -> None:
    action = _cutover_action(_freeze_publication())
    assert service._validate_authority_action(action) == action

    forbidden = copy.deepcopy(action)
    forbidden["action_payload"]["schema"] = "caller-selected-action.v1"
    with pytest.raises(
        service.PasskeyV2ServiceError,
        match="passkey_v2_action_schema_forbidden",
    ):
        service._validate_authority_action(forbidden)


@pytest.mark.parametrize("request_id", ("A" * 64, "a" * 63, "_" * 64))
def test_owner_boundary_rejects_non_sha256_cutover_request_id(
    request_id: str,
) -> None:
    class Transport:
        calls = 0

        def invoke_owner_gate(self, _frame: bytes) -> bytes:
            self.calls += 1
            raise AssertionError("invalid request reached owner gate")

    transport = Transport()
    boundary = passkey.ProductionCutoverPasskeyBoundary(
        REVISION,
        transport,
    )

    with pytest.raises(
        passkey.ProductionCutoverPasskeyError,
        match="production_cutover_consume_attempt_invalid",
    ):
        boundary.consume(
            freeze_publication={},
            request_id=request_id,
            consume_attempt_id="1" * 64,
        )

    assert transport.calls == 0


@pytest.mark.parametrize("request_id", ("A" * 64, "a" * 63, "_" * 64))
def test_service_consume_rejects_non_sha256_cutover_request_id(
    request_id: str,
) -> None:
    document = {
        "freeze_publication": {
            "documents": {"plan": {"plan_sha256": "1" * 64}},
        },
        "request_id": request_id,
        "consume_attempt_id": "2" * 64,
    }
    unsigned = {
        "schema": service.storage.REMOTE_FRAME_SCHEMA,
        "operation": "consume_production_cutover",
        "release_sha": REVISION,
        "document": document,
    }
    frame = {**unsigned, "frame_sha256": protocol.sha256_json(unsigned)}

    class Client:
        def call(self, *_args, **_kwargs) -> dict:
            raise AssertionError("invalid request reached local service")

    with pytest.raises(
        service.PasskeyV2ServiceError,
        match="passkey_v2_service_document_invalid",
    ):
        service.handle_intake_frame(
            frame,
            authority_client=Client(),
            executor_client=Client(),
            release_revision=REVISION,
            now_unix=NOW,
        )
