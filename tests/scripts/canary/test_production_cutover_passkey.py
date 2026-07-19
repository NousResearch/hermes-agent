from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import passkey_v2_service as service
from scripts.canary import owner_gate_package as owner_gate_package
from scripts.canary import production_cutover_owner_launcher as owner
from scripts.canary import production_cutover_passkey as passkey
from tests.gateway.test_canonical_writer_production_cutover import (
    NOW,
    Services,
    _database_recovery_receipt,
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
        database_recovery_receipt=_database_recovery_receipt(
            rechecked_at_unix=NOW
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


def test_real_publication_request_and_consume_under_isolated_source_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).parents[3]

    def local_blob(
        source_root: Path,
        release_revision: str,
        relative: str,
        *,
        required: bool,
    ):
        assert source_root == root
        assert release_revision == REVISION
        selected = source_root / relative
        if not selected.is_file():
            if required:
                raise owner_gate_package.OwnerGatePackageError(
                    "owner_gate_package_git_object_missing"
                )
            return None
        return selected.read_bytes(), "100644"

    monkeypatch.setattr(owner_gate_package, "_git_blob", local_blob)
    closure = owner_gate_package.resolve_runtime_source_closure(
        root,
        release_revision=REVISION,
    )
    sealed = tmp_path / "sealed-release"
    for relative in closure:
        destination = sealed / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    assert not any(
        relative.startswith(owner_gate_package.FORBIDDEN_RUNTIME_PREFIXES)
        for relative in closure
    )

    publication_path = tmp_path / "freeze-publication.json"
    publication_path.write_text(
        json.dumps(_freeze_publication(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    code = textwrap.dedent(f"""
        import json
        import sys
        sys.path.insert(0, {str(sealed)!r})
        from scripts.canary import passkey_v2_protocol as protocol
        from scripts.canary import passkey_v2_storage_growth as storage
        from scripts.canary import production_cutover_passkey as passkey

        with open({str(publication_path)!r}, encoding="utf-8") as handle:
            publication = json.load(handle)

        class Transport:
            def invoke_owner_gate(self, raw):
                frame = protocol.decode_canonical_json(raw)
                document = frame["document"]
                selected = document["freeze_publication"]
                action = passkey.build_cutover_action_envelope(
                    freeze_publication=selected,
                    authority_release_sha={REVISION!r},
                    authority_manifest_sha256="1" * 64,
                    authority_host_receipt_sha256="2" * 64,
                    issued_at_unix={NOW},
                )
                passkey.validate_cutover_action_envelope(
                    action,
                    freeze_publication=selected,
                )
                if frame["operation"] == "request_production_cutover":
                    result = {{
                        "request_id": action["request_id"],
                        "action_envelope_sha256": action["envelope_sha256"],
                        "challenge_record_sha256": "3" * 64,
                        "expires_at_unix": action["expires_at_unix"],
                        "release_sha": {REVISION!r},
                        "plan_sha256": selected["documents"]["plan"]["plan_sha256"],
                        "freeze_publication_sha256": selected["publication_sha256"],
                        "action_payload_sha256": action["action_payload_sha256"],
                        "transaction_id": action["transaction_id"],
                        "approval_url": f"{{protocol.PRODUCTION_ORIGIN}}/approve/{{action['request_id']}}",
                        "passkey_only": True,
                        "single_use": True,
                        "control_plane_mutation_performed": True,
                        "source_data_mutation_performed": False,
                        "production_host_mutation_performed": False,
                    }}
                else:
                    assert document["request_id"] == action["request_id"]
                    result = {{
                        "request_id": document["request_id"],
                        "consume_attempt_id": document["consume_attempt_id"],
                        "disposition": "authorized_once",
                        "passkey_proof": {{"portable_contract_validated": True}},
                        "release_sha": {REVISION!r},
                        "plan_sha256": selected["documents"]["plan"]["plan_sha256"],
                        "single_use": True,
                        "control_plane_mutation_performed": True,
                        "source_data_mutation_performed": False,
                        "production_host_mutation_performed": False,
                    }}
                unsigned = {{
                    "schema": storage.REMOTE_RESPONSE_SCHEMA,
                    "operation": frame["operation"],
                    "release_sha": {REVISION!r},
                    "ok": True,
                    "document": result,
                }}
                return protocol.canonical_json_bytes({{
                    **unsigned,
                    "response_sha256": protocol.sha256_json(unsigned),
                }})

        boundary = passkey.ProductionCutoverPasskeyBoundary(
            {REVISION!r}, Transport()
        )
        requested = boundary.request(publication)
        consumed = boundary.consume(
            freeze_publication=publication,
            request_id=requested["request_id"],
            consume_attempt_id="4" * 64,
        )
        assert consumed["passkey_proof"]["portable_contract_validated"] is True
        forbidden = ("agent", "gateway", "hermes_cli", "plugins")
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in forbidden
        )
    """)
    completed = subprocess.run(
        (sys.executable, "-I", "-B", "-c", code),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=20,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8")


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
