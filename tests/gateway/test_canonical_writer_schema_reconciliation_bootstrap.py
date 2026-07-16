from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import struct
import types
from dataclasses import dataclass
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_writer_schema_reconciliation_bootstrap as bootstrap
from gateway.canonical_writer_schema_reconciliation import (
    CANONICAL_QUARANTINE_ANCHORS,
    CANONICAL_TRUTH_RELATIONS,
    CANONICAL_TRUTH_RECEIPT_SCHEMA,
    RECONCILIATION_PREFLIGHT_SCHEMA,
    RECONCILIATION_RECEIPT_SCHEMA,
    CanonicalQuarantineAnchorReceipt,
    CanonicalRelationTruthReceipt,
    CanonicalTruthReceipt,
)
from gateway.canonical_writer_schema_reconciliation_db import (
    PostDeleteTerminalReceipt,
    WRITER_LOGIN,
)


NOW = 1_000
REVISION = "a" * 40
CREDENTIAL = b"C" * bootstrap.OPAQUE_CREDENTIAL_BYTES
ADMIN_USERNAME = "muncho_canary_admin_" + "b" * 16


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**value, field: hashlib.sha256(_canonical(value)).hexdigest()}


def _ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _public_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _fingerprint(public: bytes) -> str:
    blob = _ssh_string(b"ssh-ed25519") + _ssh_string(public)
    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode(
        "ascii"
    ).rstrip("=")


def _sshsig(
    key: Ed25519PrivateKey,
    message: bytes,
    *,
    namespace: str,
) -> str:
    public = _public_bytes(key)
    namespace_bytes = namespace.encode("ascii")
    signed = (
        b"SSHSIG"
        + _ssh_string(namespace_bytes)
        + _ssh_string(b"")
        + _ssh_string(b"sha512")
        + _ssh_string(hashlib.sha512(message).digest())
    )
    raw_signature = key.sign(signed)
    public_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(public)
    signature_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(raw_signature)
    envelope = (
        b"SSHSIG"
        + struct.pack(">I", 1)
        + _ssh_string(public_blob)
        + _ssh_string(namespace_bytes)
        + _ssh_string(b"")
        + _ssh_string(b"sha512")
        + _ssh_string(signature_blob)
    )
    encoded = base64.b64encode(envelope).decode("ascii")
    lines = [encoded[index : index + 70] for index in range(0, len(encoded), 70)]
    return (
        "-----BEGIN SSH SIGNATURE-----\n"
        + "\n".join(lines)
        + "\n-----END SSH SIGNATURE-----\n"
    )


def _frame(magic: bytes, value: Mapping[str, Any]) -> bytes:
    payload = _canonical(value)
    return magic + struct.pack(">I", len(payload)) + payload


def _journal(state: str) -> dict[str, Any]:
    if state == "empty":
        intent = None
        terminal = None
    elif state == "authorized_intent":
        intent = _digest("stored-authorized-intent")
        terminal = None
    elif state == "terminal":
        intent = _digest("stored-authorized-intent")
        terminal = _digest("stored-core-terminal")
    else:
        raise AssertionError(state)
    return _hashed(
        {
            "schema": bootstrap.JOURNAL_HEAD_SCHEMA,
            "state": state,
            "authorized_intent_sha256": intent,
            "terminal_receipt_sha256": terminal,
        },
        "head_sha256",
    )


def _gate(
    key: Ed25519PrivateKey,
    *,
    journal_state: str = "empty",
) -> dict[str, Any]:
    public = _public_bytes(key)
    unsigned = {
        "schema": bootstrap.GATE_SCHEMA,
        "ok": True,
        "state": "stopped_release_admin_preflight_ready",
        "release_revision": REVISION,
        "release_manifest_sha256": _digest("manifest"),
        "stopped_release_receipt_file_sha256": _digest(
            "stopped-release-file"
        ),
        "stopped_release_receipt_sha256": _digest("stopped-release"),
        "release_artifact_sha256": _digest("release-artifact"),
        "python_version": bootstrap.EXPECTED_PYTHON_VERSION,
        "interpreter_sha256": _digest("release-interpreter"),
        "activation_inventory_sha256": _digest("activation-inventory"),
        "plan_sha256": _digest("plan"),
        "base_artifact_sha256": _digest("base-artifact"),
        "target_asset_sha256": _digest("target-asset"),
        "expected_old_contract_sha256": _digest("old-contract"),
        "target_contract_sha256": _digest("target-contract"),
        "mutation_sql_sha256": _digest("mutation-sql"),
        "preflight_bridge_sql_sha256": _digest("preflight-bridge-sql"),
        "advisory_lock_key": 7_307_818_649,
        "host_identity_sha256": _digest("host-identity"),
        "services_stopped_sha256": _digest("services-stopped-state"),
        "project": bootstrap.EXPECTED_PROJECT,
        "sql_instance": bootstrap.EXPECTED_SQL_INSTANCE,
        "database": bootstrap.EXPECTED_DATABASE,
        "postgresql_major": bootstrap.EXPECTED_POSTGRESQL_MAJOR,
        "tls_server_name": "canary-pg18.example.invalid",
        "ca_file_sha256": _digest("ca-file"),
        "temporary_admin_username": ADMIN_USERNAME,
        "temporary_admin_username_sha256": hashlib.sha256(
            ADMIN_USERNAME.encode("ascii")
        ).hexdigest(),
        "owner_subject_sha256": _digest("owner-subject"),
        "owner_public_key_ed25519_hex": public.hex(),
        "owner_key_id": hashlib.sha256(public).hexdigest(),
        "owner_public_fingerprint": _fingerprint(public),
        "journal_head": _journal(journal_state),
        "run_nonce_sha256": _digest(f"run-nonce-{journal_state}"),
        "issued_at_unix": 900,
        "expires_at_unix": 1_100,
        "temporary_admin_required": True,
        "secret_material_recorded": False,
    }
    return _hashed(unsigned, "gate_sha256")


def _cloud_authority(gate: Mapping[str, Any]) -> dict[str, Any]:
    baseline_row = [
        "baseline-op",
        "CREATE_USER",
        "DONE",
        _digest("historic-actor"),
        True,
    ]
    authority_row = [
        "authority-op",
        "CREATE_USER",
        "DONE",
        gate["owner_subject_sha256"],
        True,
    ]
    unsigned = {
        "schema": bootstrap.CLOUD_ADMIN_AUTHORITY_SCHEMA,
        "project": gate["project"],
        "instance": gate["sql_instance"],
        "username_sha256": gate["temporary_admin_username_sha256"],
        "host": "",
        "type": "BUILT_IN",
        "user_present": True,
        "database_roles": list(
            bootstrap.SCHEMA_RECONCILIATION_DATABASE_ROLES
        ),
        "cloudsqlsuperuser_absent": True,
        "resource_etag_sha256": _digest("exact-cloud-sql-user-etag"),
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "mutation_context_sha256": gate["gate_sha256"],
        "baseline_operation_names": ["baseline-op"],
        "baseline_user_operations": [baseline_row],
        "authority_operation": authority_row,
    }
    return _hashed(unsigned, "receipt_sha256")


def _admin_preflight(
    key: Ed25519PrivateKey,
    gate: Mapping[str, Any],
    *,
    namespace: str = bootstrap.ADMIN_PREFLIGHT_OWNER_SSHSIG_NAMESPACE,
    changes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = _cloud_authority(gate)
    unsigned = {
        "schema": bootstrap.OWNER_ADMIN_PREFLIGHT_SCHEMA,
        "frame_schema": bootstrap.ADMIN_PREFLIGHT_FRAME_SCHEMA,
        "action": "authorize_temporary_admin_locked_preflight",
        "approved": True,
        "gate_sha256": gate["gate_sha256"],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "temporary_admin_username_sha256": gate[
            "temporary_admin_username_sha256"
        ],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "owner_key_id": gate["owner_key_id"],
        "cloud_sql_authority_receipt": authority,
        "cloud_sql_authority_receipt_sha256": authority["receipt_sha256"],
        "credential_present": True,
        "credential_length": bootstrap.OPAQUE_CREDENTIAL_BYTES,
        "issued_at_unix": 950,
        "expires_at_unix": 1_080,
        "nonce_sha256": _digest("admin-frame-nonce"),
        "secret_material_recorded": False,
    }
    if changes:
        unsigned.update(copy.deepcopy(dict(changes)))
    frame = {
        **unsigned,
        "authority_claim_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
        "signature_sshsig": "",
    }
    frame["signature_sshsig"] = _sshsig(
        key,
        bootstrap.admin_preflight_signature_payload(frame),
        namespace=namespace,
    )
    return frame


def _truth() -> CanonicalTruthReceipt:
    relations = tuple(
        CanonicalRelationTruthReceipt(
            relation=relation,
            row_count=3 if index == 0 else 0,
            chunk_count=1 if index == 0 else 0,
            chunk_manifest_sha256=_digest(f"canonical-chunks-{index}"),
        )
        for index, relation in enumerate(CANONICAL_TRUTH_RELATIONS)
    )
    return CanonicalTruthReceipt(
        row_count=3,
        canonical14_sha256=_digest("canonical14"),
        relation_receipts=relations,
        quarantine_anchor_receipts=tuple(
            CanonicalQuarantineAnchorReceipt(
                anchor=anchor,
                object_oid=7000 + index,
                owner="postgres",
                kind="n" if index == 1 else "r",
                persistence="" if index == 1 else "p",
                acl_sha256=_digest(f"quarantine-acl-{index}"),
            )
            for index, anchor in enumerate(
                CANONICAL_QUARANTINE_ANCHORS,
                start=1,
            )
        ),
    )


def _core_preflight(
    gate: Mapping[str, Any],
    *,
    state: str,
    observed_at_unix: int = 990,
) -> dict[str, Any]:
    observed = (
        gate["expected_old_contract_sha256"]
        if state == "exact_old_missing_one_helper"
        else gate["target_contract_sha256"]
    )
    unsigned = {
        "schema": RECONCILIATION_PREFLIGHT_SCHEMA,
        "ok": True,
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "base_artifact_sha256": gate["base_artifact_sha256"],
        "target_asset_sha256": gate["target_asset_sha256"],
        "postgresql_major": gate["postgresql_major"],
        "mutation_sql_sha256": gate["mutation_sql_sha256"],
        "observed_contract_sha256": observed,
        "truth_receipt_sha256": _truth().sha256,
        "expected_old_contract_sha256": gate["expected_old_contract_sha256"],
        "target_contract_sha256": gate["target_contract_sha256"],
        "state": state,
        "mutation_required": state == "exact_old_missing_one_helper",
        "observed_at_unix": observed_at_unix,
    }
    return _hashed(unsigned, "preflight_sha256")


def _bridge(observed_at_unix: int = 990) -> dict[str, Any]:
    unsigned = {
        "schema": bootstrap.TEMPORARY_OWNER_BRIDGE_SCHEMA,
        "transaction_isolation": "SERIALIZABLE",
        "database_roles": list(
            bootstrap.SCHEMA_RECONCILIATION_DATABASE_ROLES
        ),
        "provider_membership_count": 2,
        "admin_option": False,
        "inherit_option": True,
        "set_option": False,
        "cloudsqlsuperuser_absent": True,
        "canonical_truth_share_lock": True,
        "owner_authority_active_during_locked_collection": True,
        "current_user_remained_temporary_login": True,
        "exact_provider_memberships_during_locked_collection": True,
        "contract_collected_while_truth_lock_held": True,
        "canonical_truth_collected_while_truth_lock_held": True,
        "temporary_login_owned_objects": False,
        "memberships_remain_until_cloud_user_cleanup": True,
        "transaction_committed": True,
        "observed_at_unix": observed_at_unix,
        "secret_material_recorded": False,
    }
    return _hashed(unsigned, "receipt_sha256")


def _collection(
    gate: Mapping[str, Any],
    *,
    state: str,
) -> dict[str, Any]:
    return {
        "database_identity_sha256": _digest("database-identity"),
        "tls_peer_certificate_sha256": _digest("tls-peer"),
        "managed_hba_receipt_sha256": _digest("managed-hba"),
        "postgresql_major": gate["postgresql_major"],
        "temporary_owner_bridge_receipt": _bridge(),
        "preflight": _core_preflight(gate, state=state),
        "canonical_truth_receipt": _truth().value,
        "observed_at_unix": 990,
    }


def _challenge(
    gate: Mapping[str, Any],
    admin: Mapping[str, Any],
    *,
    state: str,
) -> Mapping[str, Any]:
    return bootstrap._build_preflight_challenge(
        gate=gate,
        admin_preflight=admin,
        collection=_collection(gate, state=state),
        issued_at_unix=NOW,
    )


_MODE_BY_STATE = {
    ("empty", "exact_old_missing_one_helper"): "reconcile_missing_helper",
    ("empty", "exact_target"): "adopt_existing_target",
    ("authorized_intent", "exact_old_missing_one_helper"): (
        "resume_durable_intent"
    ),
    ("authorized_intent", "exact_target"): "terminalize_durable_intent",
    ("terminal", "exact_target"): "reattest_terminal",
}


def _preflight_authorization(
    key: Ed25519PrivateKey,
    gate: Mapping[str, Any],
    admin: Mapping[str, Any],
    challenge: Mapping[str, Any],
    *,
    execution_mode: str | None = None,
    namespace: str = bootstrap.PREFLIGHT_AUTHORIZATION_OWNER_SSHSIG_NAMESPACE,
    issued_at_unix: int = NOW,
    expires_at_unix: int = 1_070,
    changes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    journal = gate["journal_head"]
    mode = execution_mode or _MODE_BY_STATE[(
        journal["state"],
        challenge["preflight"]["state"],
    )]
    nonce_sha256 = _digest(f"preflight-authorization-{mode}")
    frame: dict[str, Any] = {
        "schema": bootstrap.OWNER_PREFLIGHT_AUTHORIZATION_SCHEMA,
        "frame_schema": bootstrap.PREFLIGHT_AUTHORIZATION_FRAME_SCHEMA,
        "action": "apply_schema_reconciliation",
        "approved": True,
        "gate_sha256": gate["gate_sha256"],
        "authority_claim_sha256": admin["authority_claim_sha256"],
        "preflight_challenge_sha256": challenge[
            "preflight_challenge_sha256"
        ],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "journal_head_sha256": journal["head_sha256"],
        "execution_mode": mode,
        "preflight_sha256": challenge["preflight"]["preflight_sha256"],
        "preflight_state": challenge["preflight"]["state"],
        "observed_contract_sha256": challenge["preflight"][
            "observed_contract_sha256"
        ],
        "truth_receipt_sha256": challenge["canonical_truth_receipt"][
            "receipt_sha256"
        ],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "owner_key_id": gate["owner_key_id"],
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
        "nonce_sha256": nonce_sha256,
        "stored_authorized_intent_sha256": journal[
            "authorized_intent_sha256"
        ],
        "stored_terminal_receipt_sha256": journal[
            "terminal_receipt_sha256"
        ],
        "secret_material_recorded": False,
        "preflight_authorization_claim_sha256": "0" * 64,
        "signature_sshsig": "",
    }
    if changes:
        frame.update(copy.deepcopy(dict(changes)))
    unsigned = {
        name: frame[name]
        for name in bootstrap._PREFLIGHT_AUTHORIZATION_UNSIGNED_FIELDS
    }
    frame["preflight_authorization_claim_sha256"] = hashlib.sha256(
        _canonical(unsigned)
    ).hexdigest()
    frame["signature_sshsig"] = _sshsig(
        key,
        bootstrap.preflight_authorization_signature_payload(frame),
        namespace=namespace,
    )
    return frame


def _core_terminal(
    gate: Mapping[str, Any],
    challenge: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    authorized_intent_sha256: str,
) -> dict[str, Any]:
    execution_mode = authorization["execution_mode"]
    mode = (
        "reconcile_missing_helper"
        if execution_mode in {"reconcile_missing_helper", "resume_durable_intent"}
        else "adopt_existing_target"
    )
    initial_contract = (
        gate["expected_old_contract_sha256"]
        if mode == "reconcile_missing_helper"
        else gate["target_contract_sha256"]
    )
    core_authorization, owner_frame_receipt = bootstrap._build_core_admission(
        gate=gate,
        challenge=challenge,
        authorization=authorization,
    )
    authorization_sha256 = (
        core_authorization.sha256
        if core_authorization is not None
        else _digest("stored-core-authorization")
    )
    owner_frame_receipt_sha256 = (
        owner_frame_receipt["receipt_sha256"]
        if owner_frame_receipt is not None
        else _digest("stored-owner-frame-receipt")
    )
    unsigned = {
        "schema": RECONCILIATION_RECEIPT_SCHEMA,
        "ok": True,
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "base_artifact_sha256": gate["base_artifact_sha256"],
        "target_asset_sha256": gate["target_asset_sha256"],
        "postgresql_major": gate["postgresql_major"],
        "mutation_sql_sha256": gate["mutation_sql_sha256"],
        "expected_old_contract_sha256": gate["expected_old_contract_sha256"],
        "target_contract_sha256": gate["target_contract_sha256"],
        "initial_contract_sha256": initial_contract,
        "final_contract_sha256": gate["target_contract_sha256"],
        "initial_canonical_truth": challenge["canonical_truth_receipt"],
        "final_canonical_truth": challenge["canonical_truth_receipt"],
        "authorization_sha256": authorization_sha256,
        "preflight_sha256": challenge["preflight"]["preflight_sha256"],
        "owner_frame_receipt_sha256": owner_frame_receipt_sha256,
        "truth_receipt_sha256": challenge["canonical_truth_receipt"][
            "receipt_sha256"
        ],
        "authorized_intent_sha256": authorized_intent_sha256,
        "mode": mode,
        "mutation_applied": mode == "reconcile_missing_helper",
        "completed_at_unix": NOW,
    }
    return _hashed(unsigned, "receipt_sha256")


def _database_attestation(
    gate: Mapping[str, Any],
    challenge: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema": bootstrap.DATABASE_COMMIT_ATTESTATION_SCHEMA,
        "ok": True,
        "database_identity_sha256": challenge["database_identity_sha256"],
        "tls_peer_certificate_sha256": challenge[
            "tls_peer_certificate_sha256"
        ],
        "postgresql_major": gate["postgresql_major"],
        "observed_contract_sha256": gate["target_contract_sha256"],
        "canonical_truth_receipt": challenge["canonical_truth_receipt"],
        "transaction_committed": True,
        "temporary_owner_memberships_present": True,
        "temporary_login_owns_zero_objects": True,
        "trampoline_restored_before_commit": True,
        "cloud_user_cleanup_required": True,
        "database_session_closed": True,
        "re_attested_before_temporary_admin_delete": True,
        "observed_at_unix": NOW,
        "secret_material_recorded": False,
    }
    return _hashed(unsigned, "attestation_sha256")


def _apply_result(
    gate: Mapping[str, Any],
    challenge: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    journal_intent = gate["journal_head"]["authorized_intent_sha256"]
    intent = journal_intent or _digest("new-intent")
    return {
        "authorized_intent_sha256": intent,
        "core_terminal_receipt": _core_terminal(
            gate,
            challenge,
            authorization,
            authorized_intent_sha256=intent,
        ),
        "database_commit_attestation": _database_attestation(gate, challenge),
    }


def _intermediate(
    gate: Mapping[str, Any],
    admin: Mapping[str, Any],
    challenge: Mapping[str, Any],
    authorization: Mapping[str, Any],
    apply_result: Mapping[str, Any],
) -> Mapping[str, Any]:
    core_authorization, owner_frame_receipt = bootstrap._build_core_admission(
        gate=gate,
        challenge=challenge,
        authorization=authorization,
    )
    return bootstrap._build_database_intermediate(
        gate=gate,
        admin_preflight=admin,
        challenge=challenge,
        authorization=authorization,
        core_authorization=core_authorization,
        owner_frame_receipt=owner_frame_receipt,
        apply_result=apply_result,
        applied_at_unix=NOW,
    )


def _cloud_absence(
    gate: Mapping[str, Any],
    admin: Mapping[str, Any],
) -> dict[str, Any]:
    authority = admin["cloud_sql_authority_receipt"]
    authority_row = authority["authority_operation"]
    delete_row = [
        "delete-op",
        "DELETE_USER",
        "DONE",
        gate["owner_subject_sha256"],
        True,
    ]
    terminal_rows = sorted(
        [
            *authority["baseline_user_operations"],
            authority_row,
            delete_row,
        ],
        key=lambda row: row[0],
    )
    unsigned = {
        "schema": bootstrap.CLOUD_ADMIN_ABSENCE_SCHEMA,
        "temporary_admin_absent": True,
        "project": gate["project"],
        "instance": gate["sql_instance"],
        "username_sha256": gate["temporary_admin_username_sha256"],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "mutation_context_sha256": gate["gate_sha256"],
        "user_absent": True,
        "baseline_operation_names": authority["baseline_operation_names"],
        "baseline_user_operations": authority["baseline_user_operations"],
        "known_operation_names": [authority_row[0], delete_row[0]],
        "response_known_authority_operation_names": [authority_row[0]],
        "response_known_delete_operation_names": [delete_row[0]],
        "post_baseline_authority_operations": [authority_row],
        "response_known_candidate_observed": True,
        "post_baseline_authority_operation_count": 1,
        "terminal_user_operations": terminal_rows,
        "mutation_ambiguity_observed": False,
        "quiet_window_seconds": 180.0,
    }
    return _hashed(unsigned, "evidence_sha256")


def _cleanup(
    key: Ed25519PrivateKey,
    gate: Mapping[str, Any],
    admin: Mapping[str, Any],
    challenge: Mapping[str, Any],
    authorization: Mapping[str, Any],
    intermediate: Mapping[str, Any],
    *,
    namespace: str = bootstrap.ADMIN_CLEANUP_OWNER_SSHSIG_NAMESPACE,
    changes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    absence = _cloud_absence(gate, admin)
    unsigned = {
        "schema": bootstrap.OWNER_ADMIN_CLEANUP_SCHEMA,
        "frame_schema": bootstrap.ADMIN_CLEANUP_FRAME_SCHEMA,
        "action": "confirm_temporary_admin_absence",
        "approved": True,
        "gate_sha256": gate["gate_sha256"],
        "authority_claim_sha256": admin["authority_claim_sha256"],
        "preflight_challenge_sha256": challenge[
            "preflight_challenge_sha256"
        ],
        "preflight_authorization_claim_sha256": authorization[
            "preflight_authorization_claim_sha256"
        ],
        "database_intermediate_sha256": intermediate[
            "database_intermediate_sha256"
        ],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "temporary_admin_username_sha256": gate[
            "temporary_admin_username_sha256"
        ],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "owner_key_id": gate["owner_key_id"],
        "cloud_sql_absence_receipt": absence,
        "cloud_sql_absence_receipt_sha256": absence["evidence_sha256"],
        "issued_at_unix": NOW,
        "expires_at_unix": 1_060,
        "nonce_sha256": _digest("cleanup-nonce"),
        "secret_material_recorded": False,
    }
    if changes:
        unsigned.update(copy.deepcopy(dict(changes)))
    frame = {
        **unsigned,
        "cleanup_claim_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
        "signature_sshsig": "",
    }
    frame["signature_sshsig"] = _sshsig(
        key,
        bootstrap.admin_cleanup_signature_payload(frame),
        namespace=namespace,
    )
    return frame


def _fresh_managed_hba() -> dict[str, Any]:
    user = bootstrap.phase_b.SQL_USER
    database = "cloudsqladmin"
    return {
        "version": "managed-cloudsqladmin-hba-rejection-v2",
        "host": bootstrap.phase_b.SQL_HOST,
        "tls_server_name": bootstrap.phase_b.SQL_TLS_SERVER_NAME,
        "port": bootstrap.phase_b.SQL_PORT,
        "server_certificate_sha256": _digest("tls-peer"),
        "database": database,
        "user": user,
        "observed_at_unix": NOW,
        "expires_at_unix": NOW + 300,
        "sqlstate": "28000",
        "server_message": (
            f'no pg_hba.conf entry for host "{bootstrap.phase_b.SQL_HOST}", '
            f'user "{user}", database "{database}", SSL encryption'
        ),
        "result": "pg_hba_rejected",
        "tls_peer_verified": True,
    }


def _post_delete_terminal(
    gate: Mapping[str, Any],
    *,
    observed_at_unix: int = NOW,
) -> dict[str, Any]:
    request_id = "schema-reconciliation-post-delete-terminal-v1"
    response = {
        "ok": True,
        "result": {
            "service": "canonical_writer",
            "protocol": "v1",
            "database_identity": "canonical_brain_migration_owner",
            "request_id": request_id,
        },
    }
    return dict(
        PostDeleteTerminalReceipt(
            release_revision=gate["release_revision"],
            plan_sha256=gate["plan_sha256"],
            database=gate["database"],
            writer_login=WRITER_LOGIN,
            temporary_login=gate["temporary_admin_username"],
            temporary_login_sha256=gate[
                "temporary_admin_username_sha256"
            ],
            target_contract_sha256=gate["target_contract_sha256"],
            observed_contract_sha256=gate["target_contract_sha256"],
            writer_session_identity_exact=True,
            temporary_login_absent=True,
            temporary_login_inventory_empty=True,
            migration_owner_memberships_absent=True,
            writer_authority_exact=True,
            writer_ping_verified=True,
            writer_ping_request_id=request_id,
            writer_ping_response_sha256=hashlib.sha256(
                _canonical(response)
            ).hexdigest(),
            fresh_writer_session_closed=True,
            tls_peer_certificate_sha256=_digest("tls-peer"),
            managed_hba_receipt_sha256=hashlib.sha256(
                _canonical(_fresh_managed_hba())
            ).hexdigest(),
            pre_delete_canonical_truth_receipt_sha256=_truth().sha256,
            canonical_truth_observed=False,
            canonical_truth_limitation=(
                "writer_principal_has_no_direct_canonical_data_read_and_no_"
                "fixed_security_definer_full_truth_export"
            ),
            observed_at_unix=observed_at_unix,
        ).value
    )


def _post_cleanup(gate: Mapping[str, Any]) -> dict[str, Any]:
    post_delete = _post_delete_terminal(gate)
    fresh_hba = _fresh_managed_hba()
    unsigned = {
        "schema": bootstrap.POST_CLEANUP_OBSERVATION_SCHEMA,
        "release_manifest_sha256": gate["release_manifest_sha256"],
        "stopped_release_receipt_file_sha256": gate[
            "stopped_release_receipt_file_sha256"
        ],
        "stopped_release_receipt_sha256": gate[
            "stopped_release_receipt_sha256"
        ],
        "release_artifact_sha256": gate["release_artifact_sha256"],
        "python_version": gate["python_version"],
        "interpreter_sha256": gate["interpreter_sha256"],
        "activation_inventory_sha256": gate[
            "activation_inventory_sha256"
        ],
        "host_identity_sha256": gate["host_identity_sha256"],
        "services_stopped_sha256": gate["services_stopped_sha256"],
        "host_observation_receipt_sha256": _digest("fresh-host-observation"),
        "services_observation_receipt_sha256": _digest(
            "fresh-services-observation"
        ),
        "fresh_managed_hba_receipt": fresh_hba,
        "fresh_managed_hba_receipt_sha256": hashlib.sha256(
            _canonical(fresh_hba)
        ).hexdigest(),
        "post_delete_terminal_receipt": post_delete,
        "post_delete_terminal_receipt_sha256": post_delete[
            "receipt_sha256"
        ],
        "observed_at_unix": NOW,
        "secret_material_recorded": False,
    }
    return _hashed(unsigned, "observation_sha256")


@dataclass
class _Case:
    key: Ed25519PrivateKey
    gate: Mapping[str, Any]
    admin: Mapping[str, Any]
    challenge: Mapping[str, Any]
    authorization: Mapping[str, Any]
    apply_result: Mapping[str, Any]
    intermediate: Mapping[str, Any]
    cleanup: Mapping[str, Any]
    raw: bytes


def _case(
    *,
    state: str = "exact_old_missing_one_helper",
    journal_state: str = "empty",
) -> _Case:
    key = Ed25519PrivateKey.generate()
    gate = _gate(key, journal_state=journal_state)
    admin = _admin_preflight(key, gate)
    challenge = _challenge(gate, admin, state=state)
    authorization = _preflight_authorization(key, gate, admin, challenge)
    apply_result = _apply_result(gate, challenge, authorization)
    intermediate = _intermediate(
        gate,
        admin,
        challenge,
        authorization,
        apply_result,
    )
    cleanup = _cleanup(
        key,
        gate,
        admin,
        challenge,
        authorization,
        intermediate,
    )
    raw = (
        _frame(bootstrap.ADMIN_PREFLIGHT_MAGIC, admin)
        + CREDENTIAL
        + _frame(bootstrap.PREFLIGHT_AUTHORIZATION_MAGIC, authorization)
        + _frame(bootstrap.ADMIN_CLEANUP_MAGIC, cleanup)
    )
    return _Case(
        key=key,
        gate=gate,
        admin=admin,
        challenge=challenge,
        authorization=authorization,
        apply_result=apply_result,
        intermediate=intermediate,
        cleanup=cleanup,
        raw=raw,
    )


class _TrackingOutput(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


class _GateAwareInput(io.BytesIO):
    def __init__(self, raw: bytes, output: _TrackingOutput) -> None:
        super().__init__(raw)
        self.output = output
        self.first_read_checked = False

    def _check_first_read(self) -> None:
        if not self.first_read_checked:
            self.first_read_checked = True
            assert self.output.flush_count >= 1
            assert self.output.getvalue().endswith(b"\n")

    def read(self, size: int = -1) -> bytes:
        self._check_first_read()
        return super().read(size)

    def readinto(self, buffer) -> int:
        self._check_first_read()
        return super().readinto(buffer)


class _ReadBoundaryInput(_GateAwareInput):
    def __init__(self, raw: bytes, output: _TrackingOutput) -> None:
        super().__init__(raw, output)
        self.maximum_position = 0

    def read(self, size: int = -1) -> bytes:
        result = super().read(size)
        self.maximum_position = max(self.maximum_position, self.tell())
        return result

    def readinto(self, buffer) -> int:
        result = super().readinto(buffer)
        self.maximum_position = max(self.maximum_position, self.tell())
        return result


def _run(
    case: _Case,
    *,
    raw: bytes | None = None,
    preflight_callback=None,
    apply_callback=None,
    post_cleanup_callback=None,
    output: _TrackingOutput | None = None,
    input_type=_GateAwareInput,
):
    sink = output or _TrackingOutput()
    source = input_type(case.raw if raw is None else raw, sink)
    terminal = bootstrap.run_protocol_v2(
        case.gate,
        owner_public_key_ed25519_hex=_public_bytes(case.key).hex(),
        owner_public_fingerprint=_fingerprint(_public_bytes(case.key)),
        preflight_callback=(
            preflight_callback
            if preflight_callback is not None
            else lambda *_args: _collection(
                case.gate,
                state=case.challenge["preflight"]["state"],
            )
        ),
        apply_callback=(
            apply_callback
            if apply_callback is not None
            else lambda *_args: case.apply_result
        ),
        post_cleanup_callback=(
            post_cleanup_callback
            if post_cleanup_callback is not None
            else lambda *_args: _post_cleanup(case.gate)
        ),
        input_stream=source,
        output_stream=sink,
        now=lambda: NOW,
    )
    return terminal, sink, source


def test_v2_success_flushes_gate_then_emits_g0_p1_i2_t3_and_zeroizes() -> None:
    case = _case()
    captured: list[bytearray] = []
    apply_called = False

    def preflight(_gate, _admin, credential):
        assert credential == bytearray(CREDENTIAL)
        captured.append(credential)
        return _collection(case.gate, state="exact_old_missing_one_helper")

    def apply(
        _gate,
        _admin,
        challenge,
        authorization,
        core_authorization,
        owner_frame_receipt,
    ):
        nonlocal apply_called
        assert captured[0] == bytearray(bootstrap.OPAQUE_CREDENTIAL_BYTES)
        assert authorization["preflight_challenge_sha256"] == challenge[
            "preflight_challenge_sha256"
        ]
        assert core_authorization.value["owner_frame_sha256"] == hashlib.sha256(
            _canonical(authorization)
        ).hexdigest()
        assert owner_frame_receipt["signed_frame_sha256"] == (
            core_authorization.value["owner_frame_sha256"]
        )
        apply_called = True
        return case.apply_result

    terminal, output, source = _run(
        case,
        preflight_callback=preflight,
        apply_callback=apply,
    )

    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    assert source.first_read_checked is True
    assert output.flush_count == 4
    assert [line["schema"] for line in lines] == [
        bootstrap.GATE_SCHEMA,
        bootstrap.PREFLIGHT_CHALLENGE_SCHEMA,
        bootstrap.DATABASE_INTERMEDIATE_SCHEMA,
        bootstrap.TERMINAL_SCHEMA,
    ]
    assert terminal == lines[-1]
    assert apply_called is True
    assert captured[0] == bytearray(bootstrap.OPAQUE_CREDENTIAL_BYTES)
    assert CREDENTIAL not in output.getvalue()
    assert terminal[
        "database_re_attested_before_temporary_admin_delete"
    ] is True
    assert terminal[
        "fresh_writer_post_delete_authority_contract_and_behavior_proven"
    ] is True
    assert terminal["post_delete_canonical_truth_observed"] is False
    assert terminal["post_delete_terminal_receipt_sha256"] == terminal[
        "post_cleanup_observation"
    ]["post_delete_terminal_receipt_sha256"]
    assert terminal["fresh_managed_hba_receipt_sha256"] == terminal[
        "post_cleanup_observation"
    ]["fresh_managed_hba_receipt_sha256"]
    assert terminal["post_cleanup_observation"][
        "post_delete_terminal_receipt"
    ]["managed_hba_receipt_sha256"] == terminal[
        "fresh_managed_hba_receipt_sha256"
    ]


@pytest.mark.parametrize(
    ("failure_stage", "expected_wire_stage", "expected_head_field"),
    (
        ("preflight", "a1_to_p1", "gate_sha256"),
        ("apply", "a2_to_i2", "preflight_challenge_sha256"),
        ("post_cleanup", "c3_to_t3", "database_intermediate_sha256"),
    ),
)
def test_remote_failure_receipt_binds_each_wire_stage_and_transcript_head(
    failure_stage: str,
    expected_wire_stage: str,
    expected_head_field: str,
) -> None:
    case = _case()
    failure = bootstrap.SchemaReconciliationBootstrapError(
        "schema_reconciliation_runtime_post_cleanup_invalid"
    )

    def raise_failure(*_args):
        raise failure

    output = _TrackingOutput()
    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_runtime_post_cleanup_invalid",
    ):
        _run(
            case,
            preflight_callback=(
                raise_failure
                if failure_stage == "preflight"
                else lambda *_args: _collection(
                    case.gate,
                    state="exact_old_missing_one_helper",
                )
            ),
            apply_callback=(
                raise_failure
                if failure_stage == "apply"
                else lambda *_args: case.apply_result
            ),
            post_cleanup_callback=(
                raise_failure
                if failure_stage == "post_cleanup"
                else lambda *_args: _post_cleanup(case.gate)
            ),
            output=output,
        )

    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    receipt = lines[-1]
    expected_head = (
        case.gate[expected_head_field]
        if expected_wire_stage == "a1_to_p1"
        else case.challenge[expected_head_field]
        if expected_wire_stage == "a2_to_i2"
        else case.intermediate[expected_head_field]
    )
    unsigned = dict(receipt)
    del unsigned["receipt_sha256"]
    assert receipt == {
        **unsigned,
        "receipt_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }
    assert set(receipt) == bootstrap._REMOTE_FAILURE_FIELDS
    assert receipt["schema"] == bootstrap.REMOTE_FAILURE_SCHEMA
    assert receipt["ok"] is False
    assert receipt["wire_stage"] == expected_wire_stage
    assert receipt["error_code"] == failure.code
    assert receipt["gate_sha256"] == case.gate["gate_sha256"]
    assert receipt["release_revision"] == case.gate["release_revision"]
    assert receipt["plan_sha256"] == case.gate["plan_sha256"]
    assert receipt["transcript_head_sha256"] == expected_head
    assert receipt["secret_material_recorded"] is False


def test_generic_callback_failure_never_echoes_secret_detail() -> None:
    case = _case()
    output = _TrackingOutput()

    with pytest.raises(RuntimeError, match="owner-password-must-not-leak"):
        _run(
            case,
            preflight_callback=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("owner-password-must-not-leak")
            ),
            output=output,
        )

    assert b"owner-password-must-not-leak" not in output.getvalue()
    receipt = json.loads(output.getvalue().splitlines()[-1])
    assert receipt["error_code"] == "schema_reconciliation_remote_failed"
    assert receipt["wire_stage"] == "a1_to_p1"


def test_partial_protocol_output_failure_never_appends_failure_record() -> None:
    case = _case()

    class _PartialSecondWrite(_TrackingOutput):
        def __init__(self) -> None:
            super().__init__()
            self.write_count = 0

        def write(self, value: bytes) -> int:
            self.write_count += 1
            if self.write_count == 2:
                super().write(b"{")
                raise OSError("partial output")
            return super().write(value)

    output = _PartialSecondWrite()
    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_output_failed",
    ):
        _run(case, output=output)

    assert output.write_count == 2
    assert output.getvalue().count(b"\n") == 1
    assert output.getvalue().endswith(b"{")
    assert bootstrap.REMOTE_FAILURE_SCHEMA.encode("ascii") not in output.getvalue()


def test_gate_has_no_impossible_preflight_or_db_claims_and_exact_target_needs_secret() -> None:
    case = _case(state="exact_target")
    forbidden = {
        "preflight_sha256",
        "database_identity_sha256",
        "tls_peer_certificate_sha256",
        "mutation_required",
    }
    assert forbidden.isdisjoint(case.gate)
    seen: list[bytearray] = []

    terminal, _output, _source = _run(
        case,
        preflight_callback=lambda _gate, _admin, credential: (
            seen.append(credential)
            or _collection(case.gate, state="exact_target")
        ),
    )

    assert seen and seen[0] == bytearray(bootstrap.OPAQUE_CREDENTIAL_BYTES)
    assert terminal["final_canonical_truth"] == _truth().value


def test_invalid_a1_signature_is_rejected_before_any_credential_byte_is_read() -> None:
    case = _case()
    bad_admin = copy.deepcopy(dict(case.admin))
    bad_admin["signature_sshsig"] = bad_admin["signature_sshsig"].replace("A", "B", 1)
    admin_bytes = _frame(bootstrap.ADMIN_PREFLIGHT_MAGIC, bad_admin)
    raw = admin_bytes + CREDENTIAL
    output = _TrackingOutput()
    source = _ReadBoundaryInput(raw, output)

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_admin_preflight_signature_invalid",
    ):
        bootstrap.run_protocol_v2(
            case.gate,
            owner_public_key_ed25519_hex=_public_bytes(case.key).hex(),
            owner_public_fingerprint=_fingerprint(_public_bytes(case.key)),
            preflight_callback=lambda *_args: pytest.fail("must not authenticate"),
            apply_callback=lambda *_args: pytest.fail("must not apply"),
            post_cleanup_callback=lambda *_args: pytest.fail("must not finalize"),
            input_stream=source,
            output_stream=output,
            now=lambda: NOW,
        )

    assert source.maximum_position == len(admin_bytes)
    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(lines) == 2
    assert lines[-1]["schema"] == bootstrap.REMOTE_FAILURE_SCHEMA
    assert lines[-1]["wire_stage"] == "a1_to_p1"


@pytest.mark.parametrize(
    ("frame_kind", "error"),
    [
        ("a1_namespace", "schema_reconciliation_admin_preflight_signature_invalid"),
        ("a1_expired", "schema_reconciliation_admin_preflight_expired"),
        (
            "a2_namespace",
            "schema_reconciliation_preflight_authorization_signature_invalid",
        ),
        (
            "a2_expired",
            "schema_reconciliation_preflight_authorization_expired",
        ),
        (
            "a2_challenge",
            "schema_reconciliation_preflight_authorization_invalid",
        ),
    ],
)
def test_stale_tampered_or_wrong_namespace_frames_fail_closed(
    frame_kind: str,
    error: str,
) -> None:
    case = _case()
    admin = case.admin
    authorization = case.authorization
    if frame_kind == "a1_namespace":
        admin = _admin_preflight(
            case.key,
            case.gate,
            namespace="wrong-admin-preflight-owner-v2",
        )
    elif frame_kind == "a1_expired":
        admin = _admin_preflight(
            case.key,
            case.gate,
            changes={"issued_at_unix": 910, "expires_at_unix": 920},
        )
    elif frame_kind == "a2_namespace":
        authorization = _preflight_authorization(
            case.key,
            case.gate,
            case.admin,
            case.challenge,
            namespace="wrong-preflight-authorization-owner-v2",
        )
    elif frame_kind == "a2_expired":
        authorization = _preflight_authorization(
            case.key,
            case.gate,
            case.admin,
            case.challenge,
            issued_at_unix=980,
            expires_at_unix=990,
        )
    elif frame_kind == "a2_challenge":
        authorization = _preflight_authorization(
            case.key,
            case.gate,
            case.admin,
            case.challenge,
            changes={"preflight_challenge_sha256": _digest("wrong-challenge")},
        )
    raw = (
        _frame(bootstrap.ADMIN_PREFLIGHT_MAGIC, admin)
        + CREDENTIAL
        + _frame(bootstrap.PREFLIGHT_AUTHORIZATION_MAGIC, authorization)
    )
    apply_called = False

    def apply(*_args):
        nonlocal apply_called
        apply_called = True
        return case.apply_result

    with pytest.raises(bootstrap.SchemaReconciliationBootstrapError, match=error):
        _run(case, raw=raw, apply_callback=apply)
    assert apply_called is False


@pytest.mark.parametrize(
    ("journal_state", "preflight_state", "execution_mode"),
    [
        ("empty", "exact_old_missing_one_helper", "reconcile_missing_helper"),
        ("empty", "exact_target", "adopt_existing_target"),
        (
            "authorized_intent",
            "exact_old_missing_one_helper",
            "resume_durable_intent",
        ),
        ("authorized_intent", "exact_target", "terminalize_durable_intent"),
        ("terminal", "exact_target", "reattest_terminal"),
    ],
)
def test_fixed_replay_transition_table_accepts_only_exact_pairs(
    journal_state: str,
    preflight_state: str,
    execution_mode: str,
) -> None:
    key = Ed25519PrivateKey.generate()
    gate = _gate(key, journal_state=journal_state)
    admin = _admin_preflight(key, gate)
    challenge = _challenge(gate, admin, state=preflight_state)
    authorization = _preflight_authorization(
        key,
        gate,
        admin,
        challenge,
        execution_mode=execution_mode,
    )

    validated = bootstrap._validate_preflight_authorization(
        authorization,
        gate=gate,
        admin_preflight=admin,
        challenge=challenge,
        now_unix=NOW,
    )

    assert validated["execution_mode"] == execution_mode


@pytest.mark.parametrize(
    ("journal_state", "preflight_state", "wrong_mode"),
    [
        ("empty", "exact_target", "reconcile_missing_helper"),
        ("authorized_intent", "exact_target", "resume_durable_intent"),
        ("terminal", "exact_target", "adopt_existing_target"),
    ],
)
def test_fixed_transition_table_rejects_wrong_mode(
    journal_state: str,
    preflight_state: str,
    wrong_mode: str,
) -> None:
    key = Ed25519PrivateKey.generate()
    gate = _gate(key, journal_state=journal_state)
    admin = _admin_preflight(key, gate)
    challenge = _challenge(gate, admin, state=preflight_state)
    frame = _preflight_authorization(
        key,
        gate,
        admin,
        challenge,
        execution_mode=wrong_mode,
    )

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_preflight_authorization_invalid",
    ):
        bootstrap._validate_preflight_authorization(
            frame,
            gate=gate,
            admin_preflight=admin,
            challenge=challenge,
            now_unix=NOW,
        )


def test_terminal_journal_with_old_database_has_no_allowed_transition() -> None:
    key = Ed25519PrivateKey.generate()
    gate = _gate(key, journal_state="terminal")
    admin = _admin_preflight(key, gate)
    challenge = _challenge(gate, admin, state="exact_old_missing_one_helper")
    # A syntactically complete signed frame cannot create a missing table entry.
    frame = _preflight_authorization(
        key,
        gate,
        admin,
        challenge,
        execution_mode="reattest_terminal",
    )

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_preflight_authorization_invalid",
    ):
        bootstrap._validate_preflight_authorization(
            frame,
            gate=gate,
            admin_preflight=admin,
            challenge=challenge,
            now_unix=NOW,
        )


def test_apply_never_runs_before_exact_signed_p1_binding() -> None:
    case = _case()
    wrong = _preflight_authorization(
        case.key,
        case.gate,
        case.admin,
        case.challenge,
        changes={"plan_sha256": _digest("different-plan")},
    )
    raw = (
        _frame(bootstrap.ADMIN_PREFLIGHT_MAGIC, case.admin)
        + CREDENTIAL
        + _frame(bootstrap.PREFLIGHT_AUTHORIZATION_MAGIC, wrong)
    )
    called = False

    def apply(*_args):
        nonlocal called
        called = True
        return case.apply_result

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_preflight_authorization_invalid",
    ):
        _run(case, raw=raw, apply_callback=apply)
    assert called is False


@pytest.mark.parametrize("trailing", [b"", b"trailing"])
def test_cleanup_and_exact_eof_are_required(trailing: bytes) -> None:
    case = _case()
    prefix = (
        _frame(bootstrap.ADMIN_PREFLIGHT_MAGIC, case.admin)
        + CREDENTIAL
        + _frame(bootstrap.PREFLIGHT_AUTHORIZATION_MAGIC, case.authorization)
    )
    raw = prefix if not trailing else case.raw + trailing
    post_called = False

    def post(*_args):
        nonlocal post_called
        post_called = True
        return _post_cleanup(case.gate)

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_admin_cleanup_frame_invalid",
    ):
        _run(case, raw=raw, post_cleanup_callback=post)
    assert post_called is False


def test_full_cloud_absence_ledger_is_required_before_terminal() -> None:
    case = _case()
    absence = copy.deepcopy(dict(case.cleanup["cloud_sql_absence_receipt"]))
    absence["response_known_delete_operation_names"] = []
    unsigned_absence = dict(absence)
    unsigned_absence.pop("evidence_sha256")
    absence["evidence_sha256"] = hashlib.sha256(
        _canonical(unsigned_absence)
    ).hexdigest()
    bad_cleanup = _cleanup(
        case.key,
        case.gate,
        case.admin,
        case.challenge,
        case.authorization,
        case.intermediate,
        changes={
            "cloud_sql_absence_receipt": absence,
            "cloud_sql_absence_receipt_sha256": absence["evidence_sha256"],
        },
    )
    raw = (
        _frame(bootstrap.ADMIN_PREFLIGHT_MAGIC, case.admin)
        + CREDENTIAL
        + _frame(bootstrap.PREFLIGHT_AUTHORIZATION_MAGIC, case.authorization)
        + _frame(bootstrap.ADMIN_CLEANUP_MAGIC, bad_cleanup)
    )

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_cloud_absence_invalid",
    ):
        _run(case, raw=raw)


def test_cloud_absence_cannot_shorten_the_180_second_quiet_window() -> None:
    case = _case()
    absence = copy.deepcopy(dict(case.cleanup["cloud_sql_absence_receipt"]))
    absence["quiet_window_seconds"] = 179.999
    unsigned_absence = dict(absence)
    unsigned_absence.pop("evidence_sha256")
    absence["evidence_sha256"] = hashlib.sha256(
        _canonical(unsigned_absence)
    ).hexdigest()

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_cloud_absence_invalid",
    ):
        bootstrap._validate_cloud_absence(
            absence,
            gate=case.gate,
            authority=case.admin["cloud_sql_authority_receipt"],
        )


def test_cloud_absence_rejects_unaccounted_concurrent_delete_row() -> None:
    case = _case()
    absence = copy.deepcopy(dict(case.cleanup["cloud_sql_absence_receipt"]))
    absence["terminal_user_operations"].append([
        "unaccounted-delete-op",
        "DELETE_USER",
        "DONE",
        case.gate["owner_subject_sha256"],
        True,
    ])
    absence["terminal_user_operations"].sort(key=lambda row: row[0])
    unsigned_absence = dict(absence)
    unsigned_absence.pop("evidence_sha256")
    absence["evidence_sha256"] = hashlib.sha256(
        _canonical(unsigned_absence)
    ).hexdigest()

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_cloud_absence_invalid",
    ):
        bootstrap._validate_cloud_absence(
            absence,
            gate=case.gate,
            authority=case.admin["cloud_sql_authority_receipt"],
        )


def test_short_readinto_zeroizes_the_preallocated_credential_buffer() -> None:
    class ShortReadInto:
        def __init__(self) -> None:
            self.buffer: bytearray | None = None
            self.calls = 0

        def readinto(self, view) -> int:
            self.buffer = view.obj
            self.calls += 1
            if self.calls == 1:
                view[:3] = b"ABC"
                return 3
            return 0

    source = ShortReadInto()

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_admin_credential_invalid",
    ):
        bootstrap._read_exact_mutable(
            source,
            bootstrap.OPAQUE_CREDENTIAL_BYTES,
            code="schema_reconciliation_admin_credential_invalid",
        )

    assert source.buffer == bytearray(bootstrap.OPAQUE_CREDENTIAL_BYTES)


def test_post_cleanup_callback_requires_exact_post_delete_terminal_receipt() -> None:
    case = _case()
    value = _post_cleanup(case.gate)
    value.pop("post_delete_terminal_receipt")
    unsigned = dict(value)
    unsigned.pop("observation_sha256")
    value["observation_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    output = _TrackingOutput()

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_post_cleanup_observation_invalid",
    ):
        _run(
            case,
            post_cleanup_callback=lambda *_args: value,
            output=output,
        )
    schemas = [json.loads(line)["schema"] for line in output.getvalue().splitlines()]
    assert schemas == [
        bootstrap.GATE_SCHEMA,
        bootstrap.PREFLIGHT_CHALLENGE_SCHEMA,
        bootstrap.DATABASE_INTERMEDIATE_SCHEMA,
        bootstrap.REMOTE_FAILURE_SCHEMA,
    ]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("temporary_login_absent", False),
        ("migration_owner_memberships_absent", False),
        ("writer_authority_exact", False),
        ("writer_ping_verified", False),
        ("fresh_writer_session_closed", False),
        ("observed_contract_sha256", _digest("wrong-contract")),
        ("managed_hba_receipt_sha256", _digest("wrong-managed-hba")),
        ("tls_peer_certificate_sha256", _digest("wrong-tls")),
        (
            "pre_delete_canonical_truth_receipt_sha256",
            _digest("wrong-truth"),
        ),
        ("observed_at_unix", NOW - 1),
        ("canonical_truth_observed", True),
        ("canonical_truth_limitation", "broadened_writer_authority"),
    ],
)
def test_post_cleanup_rejects_tampered_post_delete_terminal_proof(
    field: str,
    replacement: Any,
) -> None:
    case = _case()
    value = _post_cleanup(case.gate)
    nested = copy.deepcopy(value["post_delete_terminal_receipt"])
    nested[field] = replacement
    nested_unsigned = dict(nested)
    nested_unsigned.pop("receipt_sha256")
    nested = _hashed(nested_unsigned, "receipt_sha256")
    value["post_delete_terminal_receipt"] = nested
    value["post_delete_terminal_receipt_sha256"] = nested["receipt_sha256"]
    outer_unsigned = dict(value)
    outer_unsigned.pop("observation_sha256")
    value = _hashed(outer_unsigned, "observation_sha256")

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_post_cleanup_observation_invalid",
    ):
        _run(case, post_cleanup_callback=lambda *_args: value)


@pytest.mark.parametrize("tamper", ("projection_digest", "projection_tls"))
def test_post_cleanup_rejects_rehashed_fresh_hba_projection(
    tamper: str,
) -> None:
    case = _case()
    value = _post_cleanup(case.gate)
    if tamper == "projection_digest":
        value["fresh_managed_hba_receipt_sha256"] = _digest(
            "wrong-fresh-hba"
        )
    else:
        fresh_hba = copy.deepcopy(value["fresh_managed_hba_receipt"])
        fresh_hba["server_certificate_sha256"] = _digest("wrong-tls")
        value["fresh_managed_hba_receipt"] = fresh_hba
        value["fresh_managed_hba_receipt_sha256"] = hashlib.sha256(
            _canonical(fresh_hba)
        ).hexdigest()
    unsigned = dict(value)
    unsigned.pop("observation_sha256")
    value = _hashed(unsigned, "observation_sha256")

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_post_cleanup_observation_invalid",
    ):
        _run(case, post_cleanup_callback=lambda *_args: value)


@pytest.mark.parametrize(
    "field",
    ("managed_hba_receipt_sha256", "tls_peer_certificate_sha256"),
)
def test_owner_terminal_rejects_fully_rehashed_nested_hba_tamper(
    field: str,
) -> None:
    case = _case()
    terminal = bootstrap._build_terminal(
        gate=case.gate,
        admin_preflight=case.admin,
        challenge=case.challenge,
        authorization=case.authorization,
        intermediate=case.intermediate,
        cleanup=case.cleanup,
        post_cleanup=_post_cleanup(case.gate),
        completed_at_unix=NOW,
    )
    value = copy.deepcopy(dict(terminal))
    post_cleanup = value["post_cleanup_observation"]
    nested = post_cleanup["post_delete_terminal_receipt"]
    nested[field] = _digest("attacker-rehashed-" + field)
    nested_unsigned = dict(nested)
    nested_unsigned.pop("receipt_sha256")
    nested = _hashed(nested_unsigned, "receipt_sha256")
    post_cleanup["post_delete_terminal_receipt"] = nested
    post_cleanup["post_delete_terminal_receipt_sha256"] = nested[
        "receipt_sha256"
    ]
    post_cleanup_unsigned = dict(post_cleanup)
    post_cleanup_unsigned.pop("observation_sha256")
    post_cleanup = _hashed(post_cleanup_unsigned, "observation_sha256")
    value["post_cleanup_observation"] = post_cleanup
    value["post_delete_terminal_receipt_sha256"] = nested["receipt_sha256"]
    value["post_cleanup_observation_sha256"] = post_cleanup[
        "observation_sha256"
    ]
    terminal_unsigned = dict(value)
    terminal_unsigned.pop("terminal_sha256")
    value = _hashed(terminal_unsigned, "terminal_sha256")

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_post_cleanup_observation_invalid",
    ):
        bootstrap.validate_terminal_for_owner(
            value,
            gate=case.gate,
            admin_preflight=case.admin,
            challenge=case.challenge,
            authorization=case.authorization,
            intermediate=case.intermediate,
            cleanup=case.cleanup,
            now_unix=NOW,
        )


def test_secret_like_apply_receipt_is_rejected_without_intermediate() -> None:
    case = _case()
    bad = dict(case.apply_result)
    bad["password"] = "must-not-appear"
    output = _TrackingOutput()

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_apply_result_invalid",
    ):
        _run(case, apply_callback=lambda *_args: bad, output=output)
    assert b"must-not-appear" not in output.getvalue()
    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(lines) == 3
    assert lines[-1]["schema"] == bootstrap.REMOTE_FAILURE_SCHEMA
    assert lines[-1]["wire_stage"] == "a2_to_i2"


def test_duplicate_a1_json_key_is_rejected_before_preflight() -> None:
    case = _case()
    raw_json = _canonical(case.admin)
    duplicate = b'{"schema":"duplicate",' + raw_json[1:]
    raw = (
        bootstrap.ADMIN_PREFLIGHT_MAGIC
        + struct.pack(">I", len(duplicate))
        + duplicate
        + CREDENTIAL
    )

    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_frame_json_invalid",
    ):
        _run(
            case,
            raw=raw,
            preflight_callback=lambda *_args: pytest.fail("must not authenticate"),
        )


def test_a2_has_no_nested_core_authorization_or_subset_binding() -> None:
    case = _case()

    assert "core_authorization" not in case.authorization
    assert "owner_approval_binding_sha256" not in case.authorization
    assert not hasattr(bootstrap, "preflight_authorization_owner_binding_payload")
    core_authorization, owner_frame_receipt = bootstrap._build_core_admission(
        gate=case.gate,
        challenge=case.challenge,
        authorization=case.authorization,
    )
    full_signed_digest = hashlib.sha256(
        _canonical(case.authorization)
    ).hexdigest()
    assert core_authorization is not None
    assert owner_frame_receipt is not None
    assert core_authorization.value["owner_frame_sha256"] == full_signed_digest
    assert owner_frame_receipt["signed_frame_sha256"] == full_signed_digest
    assert owner_frame_receipt["signature_sshsig_sha256"] == hashlib.sha256(
        case.authorization["signature_sshsig"].encode("utf-8")
    ).hexdigest()


def test_recovery_a2_is_outer_authority_and_does_not_replace_durable_bundle() -> None:
    case = _case(journal_state="authorized_intent")
    seen: list[tuple[Any, Any]] = []

    def apply(_gate, _admin, _challenge, authorization, core, owner_receipt):
        seen.append((core, owner_receipt))
        assert authorization["execution_mode"] == "resume_durable_intent"
        assert authorization["stored_authorized_intent_sha256"] == (
            case.gate["journal_head"]["authorized_intent_sha256"]
        )
        return case.apply_result

    terminal, _output, _source = _run(case, apply_callback=apply)

    assert seen == [(None, None)]
    assert terminal["authorized_intent_sha256"] == (
        case.gate["journal_head"]["authorized_intent_sha256"]
    )


def test_terminal_recovery_accepts_old_core_terminal_then_fresh_db_attestation() -> None:
    key = Ed25519PrivateKey.generate()
    seed_gate = _gate(key, journal_state="terminal")
    seed_admin = _admin_preflight(key, seed_gate)
    seed_challenge = _challenge(seed_gate, seed_admin, state="exact_target")
    seed_authorization = _preflight_authorization(
        key,
        seed_gate,
        seed_admin,
        seed_challenge,
    )
    core_terminal = _core_terminal(
        seed_gate,
        seed_challenge,
        seed_authorization,
        authorized_intent_sha256=seed_gate["journal_head"][
            "authorized_intent_sha256"
        ],
    )
    core_unsigned = dict(core_terminal)
    core_unsigned.pop("receipt_sha256")
    core_unsigned["completed_at_unix"] = 950
    core_terminal = _hashed(core_unsigned, "receipt_sha256")

    journal_unsigned = dict(seed_gate["journal_head"])
    journal_unsigned.pop("head_sha256")
    journal_unsigned["terminal_receipt_sha256"] = core_terminal[
        "receipt_sha256"
    ]
    gate_unsigned = dict(seed_gate)
    gate_unsigned.pop("gate_sha256")
    gate_unsigned["journal_head"] = _hashed(journal_unsigned, "head_sha256")
    gate = _hashed(gate_unsigned, "gate_sha256")
    admin = _admin_preflight(key, gate)
    challenge = _challenge(gate, admin, state="exact_target")
    authorization = _preflight_authorization(key, gate, admin, challenge)
    apply_result = {
        "authorized_intent_sha256": gate["journal_head"][
            "authorized_intent_sha256"
        ],
        "core_terminal_receipt": core_terminal,
        "database_commit_attestation": _database_attestation(gate, challenge),
    }

    intermediate = _intermediate(
        gate,
        admin,
        challenge,
        authorization,
        apply_result,
    )

    assert intermediate["core_terminal_receipt"]["completed_at_unix"] == 950
    assert intermediate["database_commit_attestation"]["observed_at_unix"] == NOW
    assert intermediate["database_session_closed"] is True


def test_gate_has_long_operation_ttl_but_owner_frames_remain_short_lived() -> None:
    key = Ed25519PrivateKey.generate()
    gate = _gate(key)
    gate_unsigned = dict(gate)
    gate_unsigned.pop("gate_sha256")
    gate_unsigned["expires_at_unix"] = 2_000
    long_gate = _hashed(gate_unsigned, "gate_sha256")

    validated = bootstrap.validate_gate(
        long_gate,
        owner_public_key_ed25519_hex=_public_bytes(key).hex(),
        owner_public_fingerprint=_fingerprint(_public_bytes(key)),
        now_unix=NOW,
    )
    assert validated["expires_at_unix"] - validated["issued_at_unix"] == 1_100

    too_long_unsigned = dict(long_gate)
    too_long_unsigned.pop("gate_sha256")
    too_long_unsigned["expires_at_unix"] = 2_701
    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_gate_expired",
    ):
        bootstrap.validate_gate(
            _hashed(too_long_unsigned, "gate_sha256"),
            owner_public_key_ed25519_hex=_public_bytes(key).hex(),
            owner_public_fingerprint=_fingerprint(_public_bytes(key)),
            now_unix=NOW,
        )

    admin = _admin_preflight(
        key,
        long_gate,
        changes={"issued_at_unix": 950, "expires_at_unix": 1_251},
    )
    with pytest.raises(
        bootstrap.SchemaReconciliationBootstrapError,
        match="schema_reconciliation_admin_preflight_expired",
    ):
        bootstrap._validate_admin_preflight(
            admin,
            gate=long_gate,
            now_unix=NOW,
        )


def test_bridge_receipt_states_truthful_collection_then_revoke_order() -> None:
    bridge = _bridge()

    assert bridge["owner_authority_active_during_locked_collection"] is True
    assert bridge["exact_provider_memberships_during_locked_collection"] is True
    assert bridge["current_user_remained_temporary_login"] is True
    assert bridge["memberships_remain_until_cloud_user_cleanup"] is True
    assert "role_reset_before_contract_collection" not in bridge
    assert "membership_absent_before_contract_collection" not in bridge


def test_public_owner_validators_accept_exact_p1_i2_t3_chain() -> None:
    case = _case()
    challenge = bootstrap.validate_preflight_challenge_for_owner(
        case.challenge,
        gate=case.gate,
        admin_preflight=case.admin,
        now_unix=NOW,
    )
    intermediate = bootstrap.validate_database_intermediate_for_owner(
        case.intermediate,
        gate=case.gate,
        admin_preflight=case.admin,
        challenge=challenge,
        authorization=case.authorization,
        now_unix=NOW,
    )
    post_cleanup = _post_cleanup(case.gate)
    terminal = bootstrap._build_terminal(
        gate=case.gate,
        admin_preflight=case.admin,
        challenge=challenge,
        authorization=case.authorization,
        intermediate=intermediate,
        cleanup=case.cleanup,
        post_cleanup=post_cleanup,
        completed_at_unix=NOW,
    )

    validated = bootstrap.validate_terminal_for_owner(
        terminal,
        gate=case.gate,
        admin_preflight=case.admin,
        challenge=challenge,
        authorization=case.authorization,
        intermediate=intermediate,
        cleanup=case.cleanup,
        now_unix=NOW,
    )

    assert validated == terminal
    assert validated["post_cleanup_observation"] == post_cleanup


@pytest.mark.parametrize("stage", ["p1", "i2", "t3"])
def test_public_owner_validators_reject_tampered_remote_receipts(stage: str) -> None:
    case = _case()
    if stage == "p1":
        value = copy.deepcopy(dict(case.challenge))
        value["database_identity_sha256"] = _digest("tampered-database")
        validator = lambda: bootstrap.validate_preflight_challenge_for_owner(
            value,
            gate=case.gate,
            admin_preflight=case.admin,
            now_unix=NOW,
        )
    elif stage == "i2":
        value = copy.deepcopy(dict(case.intermediate))
        value["database_session_closed"] = False
        validator = lambda: bootstrap.validate_database_intermediate_for_owner(
            value,
            gate=case.gate,
            admin_preflight=case.admin,
            challenge=case.challenge,
            authorization=case.authorization,
            now_unix=NOW,
        )
    else:
        terminal = bootstrap._build_terminal(
            gate=case.gate,
            admin_preflight=case.admin,
            challenge=case.challenge,
            authorization=case.authorization,
            intermediate=case.intermediate,
            cleanup=case.cleanup,
            post_cleanup=_post_cleanup(case.gate),
            completed_at_unix=NOW,
        )
        value = copy.deepcopy(dict(terminal))
        value[
            "fresh_writer_post_delete_authority_contract_and_behavior_proven"
        ] = False
        validator = lambda: bootstrap.validate_terminal_for_owner(
            value,
            gate=case.gate,
            admin_preflight=case.admin,
            challenge=case.challenge,
            authorization=case.authorization,
            intermediate=case.intermediate,
            cleanup=case.cleanup,
            now_unix=NOW,
        )
    with pytest.raises(bootstrap.SchemaReconciliationBootstrapError):
        validator()


def test_v1_wire_names_are_not_exposed() -> None:
    assert not hasattr(bootstrap, "REQUEST_MAGIC")
    assert not hasattr(bootstrap, "CLEANUP_MAGIC")
    assert not hasattr(bootstrap, "run_protocol")
    assert not hasattr(bootstrap, "OWNER_SSHSIG_NAMESPACE")


def test_cli_entry_point_accepts_only_run_and_preserves_runtime_output(
    monkeypatch,
    capsys,
) -> None:
    loaded: list[str] = []

    def runtime_run() -> Mapping[str, Any]:
        print('{"ok":true}')
        return {"ok": True}

    def import_module(name: str) -> object:
        loaded.append(name)
        return types.SimpleNamespace(run=runtime_run)

    monkeypatch.setattr(bootstrap, "_require_root_linux", lambda: None)
    monkeypatch.setattr(bootstrap.importlib, "import_module", import_module)

    assert bootstrap.main(["run"]) == 0
    captured = capsys.readouterr()
    assert captured.out == '{"ok":true}\n'
    assert captured.err == ""
    assert loaded == [
        "gateway.canonical_writer_schema_reconciliation_runtime"
    ]


@pytest.mark.parametrize(
    "arguments",
    [[], ["reconcile"], ["RUN"], ["run", "extra"]],
)
def test_cli_entry_point_rejects_every_other_command_before_runtime_import(
    monkeypatch,
    capsys,
    arguments: list[str],
) -> None:
    imported = False

    def import_module(_name: str) -> object:
        nonlocal imported
        imported = True
        raise AssertionError("runtime must remain lazy")

    monkeypatch.setattr(bootstrap, "_require_root_linux", lambda: None)
    monkeypatch.setattr(bootstrap.importlib, "import_module", import_module)

    assert bootstrap.main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "schema reconciliation bootstrap failed closed\n"
    assert imported is False


def test_cli_entry_point_runtime_failure_is_generic(
    monkeypatch,
    capsys,
) -> None:
    runtime = types.SimpleNamespace(
        run=lambda: (_ for _ in ()).throw(RuntimeError("secret detail"))
    )
    monkeypatch.setattr(bootstrap, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        bootstrap.importlib,
        "import_module",
        lambda _name: runtime,
    )

    assert bootstrap.main(["run"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "schema reconciliation bootstrap failed closed\n"
    assert "secret detail" not in captured.err
