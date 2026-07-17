"""One-time stopped-services bootstrap for the inert reconciliation control.

This module is deliberately separate from normal schema reconciliation.  Its
only mutation is the exact root-sealed
``canonical_writer_schema_reconciliation_control_v1.sql`` artifact.  The
temporary Cloud SQL login is broad enough to create the inert PostgreSQL role,
so the protocol closes its database session before it accepts the owner's
proof that the outer Cloud login was deleted.

Wire protocol::

    G0 -> MCB1 + u32 canonical JSON + 64 opaque credential -> I1
       -> MCC1 + u32 canonical JSON + EOF -> T2

No caller supplies SQL, paths, routine names, object names, or an action
selector.  All receipts are canonical, hash-bound, and secret-free.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import json
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable, Mapping, Sequence

from gateway import canonical_writer_foundation as foundation
from gateway import canonical_writer_foundation_phase_b as phase_b
from gateway import canonical_writer_phase_b_runtime as phase_b_runtime
from gateway import canonical_writer_schema_reconciliation_runtime as reconciliation_runtime
from gateway.canonical_writer_db import (
    CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY,
    CredentialSource,
    QueryResult,
)
from gateway.canonical_writer_schema_reconciliation import (
    _control_foundation_contract_sha256,
    _load_control_artifact,
)


CONTROL_BOOTSTRAP_INSTALL_OWNER_SSHSIG_NAMESPACE = (
    "muncho-canonical-writer-schema-reconciliation-control-install-owner-v1"
)
CONTROL_BOOTSTRAP_CLEANUP_OWNER_SSHSIG_NAMESPACE = (
    "muncho-canonical-writer-schema-reconciliation-control-cleanup-owner-v1"
)

INSTALL_MAGIC = b"MCB1"
CLEANUP_MAGIC = b"MCC1"
INSTALL_FRAME_SCHEMA = "MCB1-u32be-canonical-json-64byte-opaque.v1"
CLEANUP_FRAME_SCHEMA = "MCC1-u32be-canonical-json-no-secret-eof.v1"
OPAQUE_CREDENTIAL_BYTES = 64

GATE_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-control-bootstrap-gate.v1"
)
OWNER_INSTALL_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-control-owner-install.v1"
)
INTERMEDIATE_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-control-intermediate.v1"
)
OWNER_CLEANUP_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-control-owner-cleanup.v1"
)
TERMINAL_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-control-terminal.v1"
)
FAILURE_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-control-failure.v1"
)
FOUNDATION_OBSERVATION_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-control-foundation-observation.v1"
)
CLOUD_AUTHORITY_SCHEMA = (
    "muncho-cloud-sql-schema-reconciliation-control-admin-authority.v1"
)
CLOUD_ABSENCE_SCHEMA = (
    "muncho-cloud-sql-schema-reconciliation-control-admin-absence.v1"
)

CONTROL_ADMIN_PREFIX = "muncho_canary_control_"
INSTALL_ARTIFACT_FILENAME = (
    "canonical_writer_schema_reconciliation_control_v1.sql"
)
RETIRE_ARTIFACT_FILENAME = (
    "canonical_writer_schema_reconciliation_control_retire_v1.sql"
)
EXECUTOR_ROLE = "canonical_brain_schema_reconciler"
CONTROL_SCHEMA = "canonical_brain_reconciliation"
OBSERVER_SIGNATURE = (
    "canonical_brain_reconciliation."
    "observe_missing_discord_routeback_helper_v1()"
)
APPLY_SIGNATURE = (
    "canonical_brain_reconciliation."
    "apply_missing_discord_routeback_helper_v1()"
)
OBSERVER_PROSRC_SHA256 = (
    "47b63aa737d29e1d5b3a54fc824606d91c322a7869118b6f331040e0a3ef96fe"
)
OBSERVER_DEFINITION_SHA256 = (
    "7813ead62d79011f2f2c4e1895405bb35a8edc959e244a14fc22d1ab1be56974"
)
APPLY_PROSRC_SHA256 = (
    "2a28d4700d550bcc8ddc56ea870fc5f669f55a47f9abc7e1993b99b178db1719"
)
APPLY_DEFINITION_SHA256 = (
    "63d6388e50086bf2203bafb7d74291cbec32d04c1f2e05af4f007df4c1e9c8d6"
)

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_GATE_TTL_SECONDS = 1_800
MAX_OWNER_FRAME_TTL_SECONDS = 300
MIN_GATE_REMAINING_SECONDS = 900
MIN_CLOUD_ABSENCE_QUIET_WINDOW_SECONDS = 180
EXPECTED_PYTHON_VERSION = "3.11.15"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CONTROL_ADMIN = re.compile(r"^muncho_canary_control_[0-9a-f]{16}$")
_FINGERPRINT = re.compile(r"^SHA256:[A-Za-z0-9+/]{43}$")
_SAFE_OPERATION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~:/+\-]{0,511}$")
_URLSAFE_CREDENTIAL = re.compile(rb"^[A-Za-z0-9_-]{64}$")
_CONTROL_TEXT = re.compile(r"[\x00-\x1f\x7f]")
_REMOTE_STAGES = frozenset({"install_to_intermediate", "cleanup_to_terminal"})
_STABLE_ERROR = re.compile(r"^schema_reconciliation_control_[a-z0-9_]{2,80}$")

_GATE_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "release_revision",
    "release_manifest_sha256",
    "stopped_release_receipt_file_sha256",
    "stopped_release_receipt_sha256",
    "release_artifact_sha256",
    "python_version",
    "interpreter_sha256",
    "activation_inventory_sha256",
    "plan_sha256",
    "control_install_artifact_sha256",
    "control_retire_artifact_sha256",
    "control_foundation_contract_sha256",
    "advisory_lock_key",
    "host_identity_sha256",
    "services_stopped_sha256",
    "project",
    "sql_instance",
    "database",
    "postgresql_major",
    "tls_server_name",
    "ca_file_sha256",
    "temporary_control_admin_username",
    "temporary_control_admin_username_sha256",
    "owner_subject_sha256",
    "owner_public_key_ed25519_hex",
    "owner_key_id",
    "owner_public_fingerprint",
    "run_nonce_sha256",
    "issued_at_unix",
    "expires_at_unix",
    "services_stopped",
    "secret_material_recorded",
    "gate_sha256",
})

_CLOUD_AUTHORITY_FIELDS = frozenset({
    "schema",
    "project",
    "instance",
    "username_sha256",
    "host",
    "type",
    "user_present",
    "owner_subject_sha256",
    "mutation_context_sha256",
    "baseline_operation_names",
    "baseline_user_operations",
    "authority_operation",
    "broad_bootstrap_authority",
    "database_roles_requested",
    "normal_reconciliation_executor",
    "receipt_sha256",
})

_INSTALL_UNSIGNED_FIELDS = frozenset({
    "schema",
    "frame_schema",
    "action",
    "approved",
    "gate_sha256",
    "release_revision",
    "plan_sha256",
    "temporary_control_admin_username_sha256",
    "owner_subject_sha256",
    "owner_key_id",
    "control_install_artifact_sha256",
    "control_foundation_contract_sha256",
    "cloud_sql_authority_receipt",
    "cloud_sql_authority_receipt_sha256",
    "credential_present",
    "credential_length",
    "issued_at_unix",
    "expires_at_unix",
    "nonce_sha256",
    "secret_material_recorded",
})
_INSTALL_FIELDS = frozenset({
    *_INSTALL_UNSIGNED_FIELDS,
    "install_claim_sha256",
    "signature_sshsig",
})

_OBSERVATION_FIELDS = frozenset({
    "schema",
    "phase",
    "state",
    "database",
    "postgresql_major",
    "session_user_sha256",
    "control_admin_count",
    "control_admin_role_exact",
    "control_admin_forward_role_count",
    "control_admin_owned_object_count",
    "control_admin_shared_dependency_count",
    "foreign_client_session_count",
    "max_prepared_transactions",
    "cluster_prepared_xact_count",
    "non_template_database_inventory_exact",
    "all_connectable_database_inventory_exact",
    "latent_provider_exception_exact",
    "executor_database_effective_privileges_exact",
    "migration_owner_role_exact",
    "current_database_owner_exact",
    "executor_membership_count",
    "executor_owned_object_count",
    "executor_acl_dependency_count",
    "observer_prosrc_sha256",
    "observer_definition_sha256",
    "apply_prosrc_sha256",
    "apply_definition_sha256",
    "foundation_exact",
    "helper_absent",
    "helper_same_name_count",
    "observed_at_unix",
    "observation_sha256",
})

_INTERMEDIATE_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "gate_sha256",
    "release_revision",
    "plan_sha256",
    "install_claim_sha256",
    "control_install_artifact_sha256",
    "control_foundation_contract_sha256",
    "initial_foundation_state",
    "mutation_applied",
    "before_observation",
    "before_observation_sha256",
    "after_observation",
    "after_observation_sha256",
    "database_capability_terminated",
    "database_session_closed",
    "services_stopped_sha256",
    "observed_at_unix",
    "secret_material_recorded",
    "intermediate_sha256",
})

_CLOUD_ABSENCE_FIELDS = frozenset({
    "schema",
    "temporary_control_admin_absent",
    "project",
    "instance",
    "username_sha256",
    "owner_subject_sha256",
    "mutation_context_sha256",
    "user_absent",
    "baseline_operation_names",
    "baseline_user_operations",
    "known_operation_names",
    "response_known_authority_operation_names",
    "response_known_delete_operation_names",
    "post_baseline_authority_operations",
    "response_known_candidate_observed",
    "post_baseline_authority_operation_count",
    "terminal_user_operations",
    "mutation_ambiguity_observed",
    "quiet_window_seconds",
    "evidence_sha256",
})

_CLEANUP_UNSIGNED_FIELDS = frozenset({
    "schema",
    "frame_schema",
    "action",
    "approved",
    "gate_sha256",
    "release_revision",
    "plan_sha256",
    "temporary_control_admin_username_sha256",
    "owner_subject_sha256",
    "owner_key_id",
    "install_claim_sha256",
    "intermediate_sha256",
    "cloud_sql_absence_receipt",
    "cloud_sql_absence_receipt_sha256",
    "issued_at_unix",
    "expires_at_unix",
    "nonce_sha256",
    "secret_material_recorded",
})
_CLEANUP_FIELDS = frozenset({
    *_CLEANUP_UNSIGNED_FIELDS,
    "cleanup_claim_sha256",
    "signature_sshsig",
})

_TERMINAL_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "gate_sha256",
    "release_revision",
    "plan_sha256",
    "install_claim_sha256",
    "intermediate_sha256",
    "cleanup_claim_sha256",
    "control_install_artifact_sha256",
    "control_retire_artifact_sha256",
    "control_foundation_contract_sha256",
    "post_cleanup_observation",
    "post_cleanup_observation_sha256",
    "temporary_control_admin_absent",
    "executor_memberships_absent",
    "executor_owns_zero_objects",
    "fixed_routines_exact",
    "services_stopped_sha256",
    "completed_at_unix",
    "secret_material_recorded",
    "terminal_sha256",
})

_FAILURE_FIELDS = frozenset({
    "schema",
    "ok",
    "wire_stage",
    "error_code",
    "gate_sha256",
    "release_revision",
    "plan_sha256",
    "transcript_head_sha256",
    "secret_material_recorded",
    "receipt_sha256",
})


class ControlBootstrapError(RuntimeError):
    """Stable fail-closed error without reflected provider/database text."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise ControlBootstrapError(code)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ControlBootstrapError(
            "schema_reconciliation_control_json_invalid"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _hashed(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    unsigned = copy.deepcopy(dict(value))
    return {**unsigned, field: _sha256_json(unsigned)}


def _strict_mapping(value: Any, fields: frozenset[str], code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(code)
    return copy.deepcopy(dict(value))


def _hashed_mapping(
    value: Any,
    *,
    fields: frozenset[str],
    digest_field: str,
    code: str,
) -> dict[str, Any]:
    raw = _strict_mapping(value, fields, code)
    digest = raw.get(digest_field)
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        _fail(code)
    unsigned = {key: item for key, item in raw.items() if key != digest_field}
    if digest != _sha256_json(unsigned):
        _fail(code)
    return raw


def _signed_mapping(
    value: Any,
    *,
    fields: frozenset[str],
    unsigned_fields: frozenset[str],
    digest_field: str,
    code: str,
) -> dict[str, Any]:
    raw = _strict_mapping(value, fields, code)
    unsigned = {key: raw[key] for key in unsigned_fields}
    digest = raw.get(digest_field)
    if (
        not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or digest != _sha256_json(unsigned)
    ):
        _fail(code)
    return raw


def _require_sha256(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(code)
    return value


def _require_secret_free(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or _CONTROL_TEXT.search(key) is not None
                or any(
                    token in key.casefold()
                    for token in ("password", "credential_value", "secret_value")
                )
            ):
                _fail("schema_reconciliation_control_secret_material_forbidden")
            if key == "signature_sshsig":
                if (
                    not isinstance(item, str)
                    or not item.startswith("-----BEGIN SSH SIGNATURE-----\n")
                    or not item.endswith("\n-----END SSH SIGNATURE-----\n")
                    or len(item.encode("ascii", errors="ignore")) > 16_384
                ):
                    _fail(
                        "schema_reconciliation_control_secret_material_forbidden"
                    )
                continue
            _require_secret_free(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_secret_free(item)
    elif isinstance(value, str) and _CONTROL_TEXT.search(value) is not None:
        _fail("schema_reconciliation_control_secret_material_forbidden")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        _fail("schema_reconciliation_control_secret_material_forbidden")


def _validate_ttl(
    value: Mapping[str, Any],
    *,
    now_unix: int,
    code: str,
    maximum_seconds: int,
    not_before: int | None = None,
    not_after: int | None = None,
) -> None:
    issued = value.get("issued_at_unix")
    expires = value.get("expires_at_unix")
    if (
        type(now_unix) is not int
        or type(issued) is not int
        or type(expires) is not int
        or not issued <= now_unix < expires
        or not 1 <= expires - issued <= maximum_seconds
        or (not_before is not None and issued < not_before)
        or (not_after is not None and expires > not_after)
    ):
        _fail(code)


def _operation_name(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SAFE_OPERATION_NAME.fullmatch(value) is None:
        _fail(code)
    return value


def _operation_row(value: Any, code: str) -> list[Any]:
    if not isinstance(value, list) or len(value) != 5:
        _fail(code)
    name, operation_type, status, actor_sha256, succeeded = value
    _operation_name(name, code)
    if (
        operation_type not in {"CREATE_USER", "UPDATE_USER", "DELETE_USER"}
        or status != "DONE"
        or not isinstance(actor_sha256, str)
        or _SHA256.fullmatch(actor_sha256) is None
        or type(succeeded) is not bool
    ):
        _fail(code)
    return list(value)


def _operation_names(value: Any, code: str) -> list[str]:
    if not isinstance(value, list):
        _fail(code)
    result = [_operation_name(item, code) for item in value]
    if result != sorted(set(result)):
        _fail(code)
    return result


def _operation_rows(value: Any, code: str) -> list[list[Any]]:
    if not isinstance(value, list):
        _fail(code)
    rows = [_operation_row(item, code) for item in value]
    if [row[0] for row in rows] != sorted({row[0] for row in rows}):
        _fail(code)
    return rows


def _signature_payload(value: Any, fields: frozenset[str], code: str) -> bytes:
    raw = _strict_mapping(value, fields, code)
    return _canonical_bytes({
        key: item for key, item in raw.items() if key != "signature_sshsig"
    })


def owner_install_signature_payload(value: Mapping[str, Any]) -> bytes:
    return _signature_payload(
        value,
        _INSTALL_FIELDS,
        "schema_reconciliation_control_install_claim_invalid",
    )


def owner_cleanup_signature_payload(value: Mapping[str, Any]) -> bytes:
    return _signature_payload(
        value,
        _CLEANUP_FIELDS,
        "schema_reconciliation_control_cleanup_claim_invalid",
    )


def _verify_signature(
    signature: Any,
    *,
    message: bytes,
    public_key_ed25519_hex: str,
    namespace: str,
    code: str,
) -> None:
    try:
        phase_b.verify_phase_b_sshsig(
            signature,
            message=message,
            public_key_ed25519_hex=public_key_ed25519_hex,
            namespace=namespace,
        )
    except (TypeError, ValueError, phase_b.PhaseBError) as exc:
        raise ControlBootstrapError(code) from exc


def _validate_cloud_authority(
    value: Any,
    *,
    gate: Mapping[str, Any],
) -> Mapping[str, Any]:
    code = "schema_reconciliation_control_cloud_authority_invalid"
    raw = _hashed_mapping(
        value,
        fields=_CLOUD_AUTHORITY_FIELDS,
        digest_field="receipt_sha256",
        code=code,
    )
    baseline_names = _operation_names(raw.get("baseline_operation_names"), code)
    baseline_rows = _operation_rows(raw.get("baseline_user_operations"), code)
    authority = _operation_row(raw.get("authority_operation"), code)
    if (
        raw.get("schema") != CLOUD_AUTHORITY_SCHEMA
        or raw.get("project") != gate["project"]
        or raw.get("instance") != gate["sql_instance"]
        or raw.get("username_sha256")
        != gate["temporary_control_admin_username_sha256"]
        or raw.get("host") != ""
        or raw.get("type") != "BUILT_IN"
        or raw.get("user_present") is not True
        or raw.get("owner_subject_sha256") != gate["owner_subject_sha256"]
        or raw.get("mutation_context_sha256") != gate["gate_sha256"]
        or raw.get("broad_bootstrap_authority") is not True
        or raw.get("database_roles_requested") != []
        or raw.get("normal_reconciliation_executor") is not False
        or any(row[0] not in baseline_names for row in baseline_rows)
        or authority[0] in baseline_names
        or authority[1] not in {"CREATE_USER", "UPDATE_USER"}
        or authority[2] != "DONE"
        or authority[3] != gate["owner_subject_sha256"]
        or authority[4] is not True
    ):
        _fail(code)
    _require_secret_free(raw)
    return raw


def validate_gate_for_owner(
    value: Any,
    *,
    expected_release_revision: str,
    expected_owner_subject_sha256: str,
    owner_public_key_ed25519_hex: str,
    owner_public_fingerprint: str,
    now_unix: int,
) -> Mapping[str, Any]:
    code = "schema_reconciliation_control_gate_invalid"
    raw = _hashed_mapping(
        value,
        fields=_GATE_FIELDS,
        digest_field="gate_sha256",
        code=code,
    )
    username = raw.get("temporary_control_admin_username")
    try:
        _require_sha256(raw.get("control_install_artifact_sha256"), code)
        _require_sha256(raw.get("control_retire_artifact_sha256"), code)
        expected_control_contract_sha256 = (
            _control_foundation_contract_sha256(
                raw["control_install_artifact_sha256"],
                raw["control_retire_artifact_sha256"],
            )
        )
    except ControlBootstrapError:
        raise
    except BaseException as exc:
        raise ControlBootstrapError(code) from exc
    try:
        public_key = bytes.fromhex(owner_public_key_ed25519_hex)
    except (TypeError, ValueError):
        _fail(code)
    algorithm = b"ssh-ed25519"
    public_key_wire = (
        struct.pack(">I", len(algorithm))
        + algorithm
        + struct.pack(">I", len(public_key))
        + public_key
    )
    derived_fingerprint = "SHA256:" + base64.b64encode(
        hashlib.sha256(public_key_wire).digest()
    ).decode("ascii").rstrip("=")
    if (
        raw.get("schema") != GATE_SCHEMA
        or raw.get("ok") is not True
        or raw.get("state") != "stopped_release_control_bootstrap_ready"
        or raw.get("release_revision") != expected_release_revision
        or _REVISION.fullmatch(expected_release_revision or "") is None
        or raw.get("owner_subject_sha256") != expected_owner_subject_sha256
        or raw.get("owner_public_key_ed25519_hex")
        != owner_public_key_ed25519_hex
        or len(public_key) != 32
        or raw.get("owner_key_id") != _sha256_bytes(public_key)
        or raw.get("owner_public_fingerprint") != owner_public_fingerprint
        or derived_fingerprint != owner_public_fingerprint
        or _FINGERPRINT.fullmatch(owner_public_fingerprint or "") is None
        or not isinstance(username, str)
        or _CONTROL_ADMIN.fullmatch(username) is None
        or raw.get("temporary_control_admin_username_sha256")
        != _sha256_bytes(username.encode("ascii"))
        or raw.get("project") != foundation.PROJECT
        or raw.get("sql_instance") != foundation.SQL_INSTANCE
        or raw.get("database") != foundation.SQL_DATABASE
        or raw.get("postgresql_major") != 18
        or raw.get("python_version") != EXPECTED_PYTHON_VERSION
        or raw.get("advisory_lock_key")
        != CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        or raw.get("tls_server_name") != foundation.SQL_TLS_SERVER_NAME
        or raw.get("control_foundation_contract_sha256")
        != expected_control_contract_sha256
        or raw.get("services_stopped") is not True
        or raw.get("secret_material_recorded") is not False
    ):
        _fail(code)
    for name in _GATE_FIELDS - {
        "schema",
        "ok",
        "state",
        "release_revision",
        "python_version",
        "advisory_lock_key",
        "project",
        "sql_instance",
        "database",
        "postgresql_major",
        "tls_server_name",
        "temporary_control_admin_username",
        "owner_public_fingerprint",
        "issued_at_unix",
        "expires_at_unix",
        "services_stopped",
        "secret_material_recorded",
    }:
        _require_sha256(raw.get(name), code)
    _validate_ttl(
        raw,
        now_unix=now_unix,
        code=code,
        maximum_seconds=MAX_GATE_TTL_SECONDS,
    )
    _require_secret_free(raw)
    return raw


def build_owner_install_claim(
    *,
    gate: Mapping[str, Any],
    cloud_sql_authority_receipt: Mapping[str, Any],
    credential_length: int,
    issued_at_unix: int,
    expires_at_unix: int,
    nonce_sha256: str,
) -> Mapping[str, Any]:
    _require_sha256(nonce_sha256, "schema_reconciliation_control_install_claim_invalid")
    authority = _validate_cloud_authority(
        cloud_sql_authority_receipt,
        gate=gate,
    )
    unsigned = {
        "schema": OWNER_INSTALL_SCHEMA,
        "frame_schema": INSTALL_FRAME_SCHEMA,
        "action": "install_fixed_schema_reconciliation_control",
        "approved": True,
        "gate_sha256": gate["gate_sha256"],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "temporary_control_admin_username_sha256": gate[
            "temporary_control_admin_username_sha256"
        ],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "owner_key_id": gate["owner_key_id"],
        "control_install_artifact_sha256": gate[
            "control_install_artifact_sha256"
        ],
        "control_foundation_contract_sha256": gate[
            "control_foundation_contract_sha256"
        ],
        "cloud_sql_authority_receipt": authority,
        "cloud_sql_authority_receipt_sha256": authority["receipt_sha256"],
        "credential_present": True,
        "credential_length": credential_length,
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
        "nonce_sha256": nonce_sha256,
        "secret_material_recorded": False,
    }
    if (
        set(unsigned) != _INSTALL_UNSIGNED_FIELDS
        or credential_length != OPAQUE_CREDENTIAL_BYTES
    ):
        _fail("schema_reconciliation_control_install_claim_invalid")
    _validate_ttl(
        unsigned,
        now_unix=issued_at_unix,
        code="schema_reconciliation_control_install_claim_invalid",
        maximum_seconds=MAX_OWNER_FRAME_TTL_SECONDS,
        not_before=gate["issued_at_unix"],
        not_after=gate["expires_at_unix"],
    )
    return unsigned


def _validate_install_claim(
    value: Any,
    *,
    gate: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    code = "schema_reconciliation_control_install_claim_invalid"
    raw = _signed_mapping(
        value,
        fields=_INSTALL_FIELDS,
        unsigned_fields=_INSTALL_UNSIGNED_FIELDS,
        digest_field="install_claim_sha256",
        code=code,
    )
    authority = _validate_cloud_authority(
        raw.get("cloud_sql_authority_receipt"),
        gate=gate,
    )
    if (
        raw.get("schema") != OWNER_INSTALL_SCHEMA
        or raw.get("frame_schema") != INSTALL_FRAME_SCHEMA
        or raw.get("action") != "install_fixed_schema_reconciliation_control"
        or raw.get("approved") is not True
        or raw.get("gate_sha256") != gate["gate_sha256"]
        or raw.get("release_revision") != gate["release_revision"]
        or raw.get("plan_sha256") != gate["plan_sha256"]
        or raw.get("temporary_control_admin_username_sha256")
        != gate["temporary_control_admin_username_sha256"]
        or raw.get("owner_subject_sha256") != gate["owner_subject_sha256"]
        or raw.get("owner_key_id") != gate["owner_key_id"]
        or raw.get("control_install_artifact_sha256")
        != gate["control_install_artifact_sha256"]
        or raw.get("control_foundation_contract_sha256")
        != gate["control_foundation_contract_sha256"]
        or raw.get("cloud_sql_authority_receipt") != authority
        or raw.get("cloud_sql_authority_receipt_sha256")
        != authority["receipt_sha256"]
        or raw.get("credential_present") is not True
        or raw.get("credential_length") != OPAQUE_CREDENTIAL_BYTES
        or raw.get("secret_material_recorded") is not False
    ):
        _fail(code)
    _validate_ttl(
        raw,
        now_unix=now_unix,
        code=code,
        maximum_seconds=MAX_OWNER_FRAME_TTL_SECONDS,
        not_before=gate["issued_at_unix"],
        not_after=gate["expires_at_unix"],
    )
    _verify_signature(
        raw.get("signature_sshsig"),
        message=owner_install_signature_payload(raw),
        public_key_ed25519_hex=gate["owner_public_key_ed25519_hex"],
        namespace=CONTROL_BOOTSTRAP_INSTALL_OWNER_SSHSIG_NAMESPACE,
        code=code,
    )
    _require_secret_free(raw)
    return raw


def _validate_observation(value: Any, *, phase: str) -> Mapping[str, Any]:
    code = "schema_reconciliation_control_observation_invalid"
    raw = _hashed_mapping(
        value,
        fields=_OBSERVATION_FIELDS,
        digest_field="observation_sha256",
        code=code,
    )
    if (
        phase not in {"before_install", "after_install", "post_cleanup"}
        or raw.get("schema") != FOUNDATION_OBSERVATION_SCHEMA
        or raw.get("phase") != phase
        or raw.get("state") not in {"absent", "exact_installed"}
        or raw.get("database") != foundation.SQL_DATABASE
        or raw.get("postgresql_major") != 18
        or type(raw.get("control_admin_count")) is not int
        or type(raw.get("control_admin_role_exact")) is not bool
        or type(raw.get("control_admin_forward_role_count")) is not int
        or type(raw.get("control_admin_owned_object_count")) is not int
        or type(raw.get("control_admin_shared_dependency_count")) is not int
        or type(raw.get("foreign_client_session_count")) is not int
        or type(raw.get("max_prepared_transactions")) is not int
        or type(raw.get("cluster_prepared_xact_count")) is not int
        or type(raw.get("non_template_database_inventory_exact")) is not bool
        or type(raw.get("all_connectable_database_inventory_exact")) is not bool
        or type(raw.get("latent_provider_exception_exact"))
        is not bool
        or type(raw.get("executor_database_effective_privileges_exact"))
        is not bool
        or type(raw.get("migration_owner_role_exact")) is not bool
        or type(raw.get("current_database_owner_exact")) is not bool
        or type(raw.get("executor_membership_count")) is not int
        or type(raw.get("executor_owned_object_count")) is not int
        or type(raw.get("executor_acl_dependency_count")) is not int
        or type(raw.get("foundation_exact")) is not bool
        or type(raw.get("helper_absent")) is not bool
        or raw.get("helper_absent") is not True
        or type(raw.get("helper_same_name_count")) is not int
        or raw.get("helper_same_name_count") != 0
        or raw.get("non_template_database_inventory_exact") is not True
        or raw.get("all_connectable_database_inventory_exact") is not True
        or raw.get("latent_provider_exception_exact") is not True
        or raw.get("executor_database_effective_privileges_exact") is not True
        or raw.get("migration_owner_role_exact") is not True
        or raw.get("current_database_owner_exact") is not True
        or type(raw.get("observed_at_unix")) is not int
        or raw.get("observed_at_unix") < 0
    ):
        _fail(code)
    _require_sha256(raw.get("session_user_sha256"), code)
    for name in (
        "observer_prosrc_sha256",
        "observer_definition_sha256",
        "apply_prosrc_sha256",
        "apply_definition_sha256",
    ):
        item = raw.get(name)
        if item is not None and (
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
        ):
            _fail(code)
    if raw["state"] == "exact_installed":
        if (
            raw["foundation_exact"] is not True
            or raw["observer_prosrc_sha256"] != OBSERVER_PROSRC_SHA256
            or raw["observer_definition_sha256"]
            != OBSERVER_DEFINITION_SHA256
            or raw["apply_prosrc_sha256"] != APPLY_PROSRC_SHA256
            or raw["apply_definition_sha256"] != APPLY_DEFINITION_SHA256
            or raw["executor_owned_object_count"] != 0
            or raw["executor_acl_dependency_count"] != 4
        ):
            _fail(code)
    elif (
        raw["foundation_exact"] is not False
        or any(
            raw[name] is not None
            for name in (
                "observer_prosrc_sha256",
                "observer_definition_sha256",
                "apply_prosrc_sha256",
                "apply_definition_sha256",
            )
        )
        or raw["executor_membership_count"] != 0
        or raw["executor_owned_object_count"] != 0
        or raw["executor_acl_dependency_count"] != 0
    ):
        _fail(code)
    if phase in {"before_install", "after_install"}:
        if (
            raw["control_admin_count"] != 1
            or raw["control_admin_role_exact"] is not True
            or raw["control_admin_forward_role_count"]
            != 1 + raw["executor_membership_count"]
            or raw["control_admin_owned_object_count"] != 0
            or raw["control_admin_shared_dependency_count"] != 0
            or raw["foreign_client_session_count"] != 0
            or raw["max_prepared_transactions"] != 0
            or raw["cluster_prepared_xact_count"] != 0
        ):
            _fail(code)
    elif (
        raw["state"] != "exact_installed"
        or raw["control_admin_count"] != 0
        or raw["control_admin_role_exact"] is not False
        or raw["control_admin_forward_role_count"] != 0
        or raw["control_admin_owned_object_count"] != 0
        or raw["control_admin_shared_dependency_count"] != 0
        or raw["foreign_client_session_count"] != 0
        or raw["max_prepared_transactions"] != 0
        or raw["cluster_prepared_xact_count"] != 0
        or raw["executor_membership_count"] != 0
    ):
        _fail(code)
    _require_secret_free(raw)
    return raw


def validate_intermediate_for_owner(
    value: Any,
    *,
    gate: Mapping[str, Any],
    install_claim: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    code = "schema_reconciliation_control_intermediate_invalid"
    raw = _hashed_mapping(
        value,
        fields=_INTERMEDIATE_FIELDS,
        digest_field="intermediate_sha256",
        code=code,
    )
    before = _validate_observation(
        raw.get("before_observation"),
        phase="before_install",
    )
    after = _validate_observation(
        raw.get("after_observation"),
        phase="after_install",
    )
    if (
        raw.get("schema") != INTERMEDIATE_SCHEMA
        or raw.get("ok") is not True
        or raw.get("state")
        != "database_session_closed_awaiting_cloud_cleanup"
        or raw.get("gate_sha256") != gate["gate_sha256"]
        or raw.get("release_revision") != gate["release_revision"]
        or raw.get("plan_sha256") != gate["plan_sha256"]
        or raw.get("install_claim_sha256")
        != install_claim["install_claim_sha256"]
        or raw.get("control_install_artifact_sha256")
        != gate["control_install_artifact_sha256"]
        or raw.get("control_foundation_contract_sha256")
        != gate["control_foundation_contract_sha256"]
        or raw.get("initial_foundation_state") != before["state"]
        or type(raw.get("mutation_applied")) is not bool
        or raw["mutation_applied"] is not (before["state"] == "absent")
        or after["state"] != "exact_installed"
        or before["session_user_sha256"]
        != gate["temporary_control_admin_username_sha256"]
        or after["session_user_sha256"]
        != gate["temporary_control_admin_username_sha256"]
        or before["control_admin_count"] != 1
        or after["control_admin_count"] != 1
        or before["control_admin_role_exact"] is not True
        or after["control_admin_role_exact"] is not True
        or before["control_admin_forward_role_count"] != 1
        or after["control_admin_forward_role_count"]
        != 1 + after["executor_membership_count"]
        or before["control_admin_owned_object_count"] != 0
        or after["control_admin_owned_object_count"] != 0
        or before["control_admin_shared_dependency_count"] != 0
        or after["control_admin_shared_dependency_count"] != 0
        or before["foreign_client_session_count"] != 0
        or after["foreign_client_session_count"] != 0
        or before["max_prepared_transactions"] != 0
        or after["max_prepared_transactions"] != 0
        or before["cluster_prepared_xact_count"] != 0
        or after["cluster_prepared_xact_count"] != 0
        or before["executor_membership_count"] != 0
        or after["executor_membership_count"]
        != (1 if before["state"] == "absent" else 0)
        or raw.get("before_observation") != before
        or raw.get("before_observation_sha256")
        != before["observation_sha256"]
        or raw.get("after_observation") != after
        or raw.get("after_observation_sha256")
        != after["observation_sha256"]
        or raw.get("database_capability_terminated") is not True
        or raw.get("database_session_closed") is not True
        or raw.get("services_stopped_sha256")
        != gate["services_stopped_sha256"]
        or type(raw.get("observed_at_unix")) is not int
        or not gate["issued_at_unix"] <= raw["observed_at_unix"] <= now_unix
        or not gate["issued_at_unix"]
        <= before["observed_at_unix"]
        <= after["observed_at_unix"]
        <= raw["observed_at_unix"]
        or raw.get("secret_material_recorded") is not False
    ):
        _fail(code)
    _require_secret_free(raw)
    return raw


def _validate_cloud_absence(
    value: Any,
    *,
    gate: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    code = "schema_reconciliation_control_cloud_absence_invalid"
    raw = _hashed_mapping(
        value,
        fields=_CLOUD_ABSENCE_FIELDS,
        digest_field="evidence_sha256",
        code=code,
    )
    baseline_names = _operation_names(raw.get("baseline_operation_names"), code)
    baseline_rows = _operation_rows(raw.get("baseline_user_operations"), code)
    known_names = _operation_names(raw.get("known_operation_names"), code)
    known_authority = _operation_names(
        raw.get("response_known_authority_operation_names"), code
    )
    known_deletes = _operation_names(
        raw.get("response_known_delete_operation_names"), code
    )
    post_authority = _operation_rows(
        raw.get("post_baseline_authority_operations"), code
    )
    terminal_rows = _operation_rows(raw.get("terminal_user_operations"), code)
    authority_row = list(authority["authority_operation"])
    terminal_by_name = {row[0]: row for row in terminal_rows}
    quiet_window = raw.get("quiet_window_seconds")
    if (
        raw.get("schema") != CLOUD_ABSENCE_SCHEMA
        or raw.get("temporary_control_admin_absent") is not True
        or raw.get("project") != gate["project"]
        or raw.get("instance") != gate["sql_instance"]
        or raw.get("username_sha256")
        != gate["temporary_control_admin_username_sha256"]
        or raw.get("owner_subject_sha256") != gate["owner_subject_sha256"]
        or raw.get("mutation_context_sha256") != gate["gate_sha256"]
        or raw.get("user_absent") is not True
        or baseline_names != authority["baseline_operation_names"]
        or baseline_rows != authority["baseline_user_operations"]
        or known_authority != [authority_row[0]]
        or post_authority != [authority_row]
        or raw.get("response_known_candidate_observed") is not True
        or raw.get("post_baseline_authority_operation_count") != 1
        or not known_deletes
        or set(known_names) != {authority_row[0], *known_deletes}
        or set(terminal_by_name)
        != {*(row[0] for row in baseline_rows), *known_names}
        or terminal_by_name.get(authority_row[0]) != authority_row
        or any(
            terminal_by_name.get(name) is None
            or terminal_by_name[name][1] != "DELETE_USER"
            or terminal_by_name[name][2] != "DONE"
            or terminal_by_name[name][3] != gate["owner_subject_sha256"]
            or terminal_by_name[name][4] is not True
            for name in known_deletes
        )
        or any(terminal_by_name.get(row[0]) != row for row in baseline_rows)
        or type(raw.get("mutation_ambiguity_observed")) is not bool
        or not isinstance(quiet_window, (int, float))
        or isinstance(quiet_window, bool)
        or not MIN_CLOUD_ABSENCE_QUIET_WINDOW_SECONDS
        <= quiet_window
        <= 3_600
    ):
        _fail(code)
    _require_secret_free(raw)
    return raw


def build_owner_cleanup_claim(
    *,
    gate: Mapping[str, Any],
    install_claim_sha256: str,
    intermediate: Mapping[str, Any],
    cloud_sql_absence_receipt: Mapping[str, Any],
    issued_at_unix: int,
    expires_at_unix: int,
    nonce_sha256: str,
) -> Mapping[str, Any]:
    _require_sha256(
        install_claim_sha256,
        "schema_reconciliation_control_cleanup_claim_invalid",
    )
    _require_sha256(
        nonce_sha256,
        "schema_reconciliation_control_cleanup_claim_invalid",
    )
    authority = intermediate.get("_cloud_authority")
    if authority is None:
        # The owner-side intermediate deliberately contains no hidden state.
        # Authority is supplied through the gate-bound install claim when the
        # remote validator runs; accept the receipt structurally here and bind
        # its self-digest into the signed cleanup claim.
        absence = _hashed_mapping(
            cloud_sql_absence_receipt,
            fields=_CLOUD_ABSENCE_FIELDS,
            digest_field="evidence_sha256",
            code="schema_reconciliation_control_cleanup_claim_invalid",
        )
    else:  # pragma: no cover - defensive against non-canonical mappings
        _fail("schema_reconciliation_control_cleanup_claim_invalid")
    unsigned = {
        "schema": OWNER_CLEANUP_SCHEMA,
        "frame_schema": CLEANUP_FRAME_SCHEMA,
        "action": "confirm_control_admin_absence",
        "approved": True,
        "gate_sha256": gate["gate_sha256"],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "temporary_control_admin_username_sha256": gate[
            "temporary_control_admin_username_sha256"
        ],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "owner_key_id": gate["owner_key_id"],
        "install_claim_sha256": install_claim_sha256,
        "intermediate_sha256": intermediate["intermediate_sha256"],
        "cloud_sql_absence_receipt": absence,
        "cloud_sql_absence_receipt_sha256": absence["evidence_sha256"],
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
        "nonce_sha256": nonce_sha256,
        "secret_material_recorded": False,
    }
    if set(unsigned) != _CLEANUP_UNSIGNED_FIELDS:
        _fail("schema_reconciliation_control_cleanup_claim_invalid")
    _validate_ttl(
        unsigned,
        now_unix=issued_at_unix,
        code="schema_reconciliation_control_cleanup_claim_invalid",
        maximum_seconds=MAX_OWNER_FRAME_TTL_SECONDS,
        not_before=gate["issued_at_unix"],
        not_after=gate["expires_at_unix"],
    )
    return unsigned


def _validate_cleanup_claim(
    value: Any,
    *,
    gate: Mapping[str, Any],
    install_claim: Mapping[str, Any],
    intermediate: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    code = "schema_reconciliation_control_cleanup_claim_invalid"
    raw = _signed_mapping(
        value,
        fields=_CLEANUP_FIELDS,
        unsigned_fields=_CLEANUP_UNSIGNED_FIELDS,
        digest_field="cleanup_claim_sha256",
        code=code,
    )
    absence = _validate_cloud_absence(
        raw.get("cloud_sql_absence_receipt"),
        gate=gate,
        authority=install_claim["cloud_sql_authority_receipt"],
    )
    if (
        raw.get("schema") != OWNER_CLEANUP_SCHEMA
        or raw.get("frame_schema") != CLEANUP_FRAME_SCHEMA
        or raw.get("action") != "confirm_control_admin_absence"
        or raw.get("approved") is not True
        or raw.get("gate_sha256") != gate["gate_sha256"]
        or raw.get("release_revision") != gate["release_revision"]
        or raw.get("plan_sha256") != gate["plan_sha256"]
        or raw.get("temporary_control_admin_username_sha256")
        != gate["temporary_control_admin_username_sha256"]
        or raw.get("owner_subject_sha256") != gate["owner_subject_sha256"]
        or raw.get("owner_key_id") != gate["owner_key_id"]
        or raw.get("install_claim_sha256")
        != install_claim["install_claim_sha256"]
        or raw.get("intermediate_sha256")
        != intermediate["intermediate_sha256"]
        or raw.get("cloud_sql_absence_receipt") != absence
        or raw.get("cloud_sql_absence_receipt_sha256")
        != absence["evidence_sha256"]
        or raw.get("secret_material_recorded") is not False
    ):
        _fail(code)
    _validate_ttl(
        raw,
        now_unix=now_unix,
        code=code,
        maximum_seconds=MAX_OWNER_FRAME_TTL_SECONDS,
        not_before=intermediate["observed_at_unix"],
        not_after=gate["expires_at_unix"],
    )
    _verify_signature(
        raw.get("signature_sshsig"),
        message=owner_cleanup_signature_payload(raw),
        public_key_ed25519_hex=gate["owner_public_key_ed25519_hex"],
        namespace=CONTROL_BOOTSTRAP_CLEANUP_OWNER_SSHSIG_NAMESPACE,
        code=code,
    )
    _require_secret_free(raw)
    return raw


def validate_terminal_for_owner(
    value: Any,
    *,
    gate: Mapping[str, Any],
    install_claim: Mapping[str, Any],
    intermediate: Mapping[str, Any],
    cleanup_claim: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    code = "schema_reconciliation_control_terminal_invalid"
    raw = _hashed_mapping(
        value,
        fields=_TERMINAL_FIELDS,
        digest_field="terminal_sha256",
        code=code,
    )
    observation = _validate_observation(
        raw.get("post_cleanup_observation"),
        phase="post_cleanup",
    )
    if (
        raw.get("schema") != TERMINAL_SCHEMA
        or raw.get("ok") is not True
        or raw.get("state") != "control_installed_admin_absent_stopped"
        or raw.get("gate_sha256") != gate["gate_sha256"]
        or raw.get("release_revision") != gate["release_revision"]
        or raw.get("plan_sha256") != gate["plan_sha256"]
        or raw.get("install_claim_sha256")
        != install_claim["install_claim_sha256"]
        or raw.get("intermediate_sha256")
        != intermediate["intermediate_sha256"]
        or raw.get("cleanup_claim_sha256")
        != cleanup_claim["cleanup_claim_sha256"]
        or raw.get("control_install_artifact_sha256")
        != gate["control_install_artifact_sha256"]
        or raw.get("control_retire_artifact_sha256")
        != gate["control_retire_artifact_sha256"]
        or raw.get("control_foundation_contract_sha256")
        != gate["control_foundation_contract_sha256"]
        or raw.get("post_cleanup_observation") != observation
        or raw.get("post_cleanup_observation_sha256")
        != observation["observation_sha256"]
        or observation["session_user_sha256"]
        != _sha256_bytes(foundation.SQL_USER.encode("ascii"))
        or observation["control_admin_count"] != 0
        or observation["control_admin_role_exact"] is not False
        or observation["control_admin_forward_role_count"] != 0
        or observation["control_admin_owned_object_count"] != 0
        or observation["control_admin_shared_dependency_count"] != 0
        or observation["foreign_client_session_count"] != 0
        or observation["max_prepared_transactions"] != 0
        or observation["cluster_prepared_xact_count"] != 0
        or observation["executor_membership_count"] != 0
        or not cleanup_claim["issued_at_unix"]
        <= observation["observed_at_unix"]
        <= raw.get("completed_at_unix", -1)
        or raw.get("temporary_control_admin_absent") is not True
        or raw.get("executor_memberships_absent") is not True
        or raw.get("executor_owns_zero_objects") is not True
        or raw.get("fixed_routines_exact") is not True
        or raw.get("services_stopped_sha256")
        != gate["services_stopped_sha256"]
        or type(raw.get("completed_at_unix")) is not int
        or not intermediate["observed_at_unix"]
        <= raw["completed_at_unix"]
        <= now_unix
        or raw.get("secret_material_recorded") is not False
    ):
        _fail(code)
    _require_secret_free(raw)
    return raw


def validate_failure_for_owner(
    value: Any,
    *,
    gate: Mapping[str, Any],
    expected_wire_stage: str,
    expected_transcript_head_sha256: str,
) -> Mapping[str, Any]:
    code = "schema_reconciliation_control_failure_invalid"
    raw = _hashed_mapping(
        value,
        fields=_FAILURE_FIELDS,
        digest_field="receipt_sha256",
        code=code,
    )
    if (
        expected_wire_stage not in _REMOTE_STAGES
        or raw.get("schema") != FAILURE_SCHEMA
        or raw.get("ok") is not False
        or raw.get("wire_stage") != expected_wire_stage
        or not isinstance(raw.get("error_code"), str)
        or _STABLE_ERROR.fullmatch(raw["error_code"]) is None
        or raw.get("gate_sha256") != gate["gate_sha256"]
        or raw.get("release_revision") != gate["release_revision"]
        or raw.get("plan_sha256") != gate["plan_sha256"]
        or raw.get("transcript_head_sha256")
        != expected_transcript_head_sha256
        or raw.get("secret_material_recorded") is not False
    ):
        _fail(code)
    _require_secret_free(raw)
    return raw


def _zeroize(value: bytearray | None) -> None:
    if value is not None:
        for index in range(len(value)):
            value[index] = 0


def _read_exact_mutable(stream: BinaryIO, size: int, *, code: str) -> bytearray:
    if type(size) is not int or not 0 <= size <= MAX_JSON_BYTES:
        _fail(code)
    value = bytearray(size)
    view = memoryview(value)
    offset = 0
    try:
        while offset < size:
            read = stream.readinto(view[offset:])
            if type(read) is not int or read <= 0 or read > size - offset:
                _fail(code)
            offset += read
        return value
    except BaseException:
        _zeroize(value)
        raise
    finally:
        view.release()


def _read_exact(stream: BinaryIO, size: int, *, code: str) -> bytes:
    value = _read_exact_mutable(stream, size, code=code)
    try:
        return bytes(value)
    finally:
        _zeroize(value)


def _decode_canonical_mapping(raw: bytes, *, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ControlBootstrapError(code) from exc
    if not isinstance(value, Mapping) or _canonical_bytes(value) != raw:
        _fail(code)
    return value


def _read_mapping_frame(
    stream: BinaryIO,
    *,
    magic: bytes,
    code: str,
) -> Mapping[str, Any]:
    header = _read_exact(stream, 8, code=code)
    if header[:4] != magic:
        _fail(code)
    size = struct.unpack(">I", header[4:])[0]
    if not 2 <= size <= MAX_JSON_BYTES:
        _fail(code)
    return _decode_canonical_mapping(
        _read_exact(stream, size, code=code),
        code=code,
    )


def _emit_mapping(stream: BinaryIO, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value) + b"\n"
    try:
        written = stream.write(payload)
        if written is not None and written != len(payload):
            _fail("schema_reconciliation_control_output_failed")
        stream.flush()
    except ControlBootstrapError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ControlBootstrapError(
            "schema_reconciliation_control_output_failed"
        ) from exc


def _failure_receipt(
    *,
    gate: Mapping[str, Any],
    wire_stage: str,
    transcript_head_sha256: str,
    error: BaseException,
) -> Mapping[str, Any]:
    error_code = "schema_reconciliation_control_remote_failed"
    if isinstance(error, ControlBootstrapError) and _STABLE_ERROR.fullmatch(
        error.code
    ):
        error_code = error.code
    if (
        wire_stage not in _REMOTE_STAGES
        or _SHA256.fullmatch(transcript_head_sha256 or "") is None
    ):
        _fail("schema_reconciliation_control_remote_failed")
    return _hashed({
        "schema": FAILURE_SCHEMA,
        "ok": False,
        "wire_stage": wire_stage,
        "error_code": error_code,
        "gate_sha256": gate["gate_sha256"],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "transcript_head_sha256": transcript_head_sha256,
        "secret_material_recorded": False,
    }, "receipt_sha256")


InstallCallback = Callable[
    [Mapping[str, Any], Mapping[str, Any], bytearray],
    Mapping[str, Any],
]
PostCleanupCallback = Callable[
    [
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
    ],
    Mapping[str, Any],
]


def run_protocol(
    gate: Mapping[str, Any],
    *,
    install_callback: InstallCallback,
    post_cleanup_callback: PostCleanupCallback,
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
    now: Callable[[], int] = lambda: int(time.time()),
) -> Mapping[str, Any]:
    if not callable(install_callback) or not callable(post_cleanup_callback):
        _fail("schema_reconciliation_control_callbacks_invalid")
    source = sys.stdin.buffer if input_stream is None else input_stream
    sink = sys.stdout.buffer if output_stream is None else output_stream
    validated_gate = validate_gate_for_owner(
        gate,
        expected_release_revision=gate.get("release_revision"),
        expected_owner_subject_sha256=gate.get("owner_subject_sha256"),
        owner_public_key_ed25519_hex=gate.get("owner_public_key_ed25519_hex"),
        owner_public_fingerprint=gate.get("owner_public_fingerprint"),
        now_unix=now(),
    )
    _emit_mapping(sink, validated_gate)
    credential: bytearray | None = None
    wire_stage = "install_to_intermediate"
    transcript_head = validated_gate["gate_sha256"]
    output_unreliable = False
    try:
        install_raw = _read_mapping_frame(
            source,
            magic=INSTALL_MAGIC,
            code="schema_reconciliation_control_install_frame_invalid",
        )
        install_claim = _validate_install_claim(
            install_raw,
            gate=validated_gate,
            now_unix=now(),
        )
        transcript_head = install_claim["install_claim_sha256"]
        credential = _read_exact_mutable(
            source,
            OPAQUE_CREDENTIAL_BYTES,
            code="schema_reconciliation_control_credential_invalid",
        )
        if _URLSAFE_CREDENTIAL.fullmatch(credential) is None:
            _fail("schema_reconciliation_control_credential_invalid")
        try:
            intermediate = install_callback(
                validated_gate,
                install_claim,
                credential,
            )
        finally:
            _zeroize(credential)
        intermediate = validate_intermediate_for_owner(
            intermediate,
            gate=validated_gate,
            install_claim=install_claim,
            now_unix=now(),
        )
        _emit_mapping(sink, intermediate)
        wire_stage = "cleanup_to_terminal"
        transcript_head = intermediate["intermediate_sha256"]

        cleanup_raw = _read_mapping_frame(
            source,
            magic=CLEANUP_MAGIC,
            code="schema_reconciliation_control_cleanup_frame_invalid",
        )
        if source.read(1) != b"":
            _fail("schema_reconciliation_control_cleanup_frame_invalid")
        cleanup = _validate_cleanup_claim(
            cleanup_raw,
            gate=validated_gate,
            install_claim=install_claim,
            intermediate=intermediate,
            now_unix=now(),
        )
        transcript_head = cleanup["cleanup_claim_sha256"]
        terminal = post_cleanup_callback(
            validated_gate,
            install_claim,
            intermediate,
            cleanup,
        )
        terminal = validate_terminal_for_owner(
            terminal,
            gate=validated_gate,
            install_claim=install_claim,
            intermediate=intermediate,
            cleanup_claim=cleanup,
            now_unix=now(),
        )
        _emit_mapping(sink, terminal)
        return terminal
    except BaseException as error:
        if (
            isinstance(error, ControlBootstrapError)
            and error.code == "schema_reconciliation_control_output_failed"
        ):
            output_unreliable = True
        if not output_unreliable:
            try:
                _emit_mapping(
                    sink,
                    _failure_receipt(
                        gate=validated_gate,
                        wire_stage=wire_stage,
                        transcript_head_sha256=transcript_head,
                        error=error,
                    ),
                )
            except BaseException:
                output_unreliable = True
        raise
    finally:
        _zeroize(credential)


# The observation is fixed in this sealed module.  It accepts no identifiers,
# SQL fragments, or actions from either wire frame.  The shared session lock is
# acquired by Python before PostgreSQL begins the SERIALIZABLE transaction, so
# the catalog snapshot cannot predate a concurrently committing exclusive
# mutator.
_FOUNDATION_OBSERVATION_COLUMNS = (
    "database_name",
    "version_num",
    "session_user_name",
    "control_admin_count",
    "control_admin_role_exact",
    "control_admin_forward_role_count",
    "control_admin_owned_object_count",
    "control_admin_shared_dependency_count",
    "foreign_client_session_count",
    "max_prepared_transactions",
    "cluster_prepared_xact_count",
    "non_template_database_inventory_exact",
    "all_connectable_database_inventory_exact",
    "latent_provider_exception_exact",
    "executor_database_effective_privileges_exact",
    "migration_owner_role_exact",
    "current_database_owner_exact",
    "executor_membership_count",
    "executor_owned_object_count",
    "executor_acl_dependency_count",
    "observer_prosrc_sha256",
    "observer_definition_sha256",
    "apply_prosrc_sha256",
    "apply_definition_sha256",
    "foundation_state",
    "foundation_exact",
    "helper_absent",
    "helper_same_name_count",
)

_FOUNDATION_OBSERVATION_LOCK_SQL = (
    "SELECT pg_catalog.pg_advisory_lock_shared("
    f"{CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY})"
)
_FOUNDATION_OBSERVATION_SET_LOCK_TIMEOUT_SQL = "SET lock_timeout = '15s'"
_FOUNDATION_OBSERVATION_RESET_LOCK_TIMEOUT_SQL = "RESET lock_timeout"
_FOUNDATION_OBSERVATION_UNLOCK_SQL = (
    "SELECT pg_catalog.pg_advisory_unlock_shared("
    f"{CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY})"
)

_FOUNDATION_OBSERVATION_SQL = f"""
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY;
SET LOCAL TimeZone = 'UTC';
SET LOCAL DateStyle = 'ISO, YMD';
SET LOCAL IntervalStyle = 'postgres';
SET LOCAL extra_float_digits = 3;
SET LOCAL bytea_output = 'hex';
SET LOCAL search_path = pg_catalog, pg_temp;
SET LOCAL lock_timeout = '15s';
SET LOCAL statement_timeout = '5min';
WITH RECURSIVE executor AS MATERIALIZED (
    SELECT role.*
      FROM pg_catalog.pg_roles AS role
     WHERE role.rolname = '{EXECUTOR_ROLE}'
), session_role AS MATERIALIZED (
    SELECT role.*
      FROM pg_catalog.pg_roles AS role
     WHERE role.rolname = SESSION_USER
), relevant_session_edges AS MATERIALIZED (
    SELECT membership.*,
           granted.rolname AS granted_name,
           member.rolname AS member_name,
           grantor.rolname AS grantor_name
      FROM pg_catalog.pg_auth_members AS membership
      JOIN pg_catalog.pg_roles AS granted
        ON granted.oid = membership.roleid
      JOIN pg_catalog.pg_roles AS member
        ON member.oid = membership.member
      JOIN pg_catalog.pg_roles AS grantor
        ON grantor.oid = membership.grantor
     WHERE member.rolname = SESSION_USER
        OR granted.rolname = SESSION_USER
        OR grantor.rolname = SESSION_USER
), forward_role_closure(roleid) AS (
    SELECT membership.roleid
      FROM pg_catalog.pg_auth_members AS membership
      JOIN session_role ON session_role.oid = membership.member
    UNION
    SELECT membership.roleid
      FROM pg_catalog.pg_auth_members AS membership
      JOIN forward_role_closure AS reachable
        ON reachable.roleid = membership.member
), control_namespace AS MATERIALIZED (
    SELECT namespace.*
      FROM pg_catalog.pg_namespace AS namespace
     WHERE namespace.nspname = '{CONTROL_SCHEMA}'
), managed_database AS MATERIALIZED (
    SELECT database.*,
           pg_catalog.pg_get_userbyid(database.datdba) AS owner_name
      FROM pg_catalog.pg_database AS database
     WHERE database.datname = 'cloudsqladmin'
), managed_actual_database_acl AS MATERIALIZED (
    SELECT CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                ELSE pg_catalog.pg_get_userbyid(acl.grantee) END AS grantee,
           pg_catalog.pg_get_userbyid(acl.grantor) AS grantor,
           acl.privilege_type,
           acl.is_grantable
      FROM managed_database AS database
      CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
          database.datacl,
          pg_catalog.acldefault('d', database.datdba)
      )) AS acl
), managed_expected_database_acl(
    grantee, grantor, privilege_type, is_grantable
) AS (
    VALUES
      ('PUBLIC'::text, 'cloudsqladmin'::text, 'CONNECT'::text, false),
      ('PUBLIC'::text, 'cloudsqladmin'::text, 'TEMPORARY'::text, false),
      ('cloudsqladmin'::text, 'cloudsqladmin'::text, 'CREATE'::text, false),
      ('cloudsqladmin'::text, 'cloudsqladmin'::text, 'CONNECT'::text, false),
      ('cloudsqladmin'::text, 'cloudsqladmin'::text, 'TEMPORARY'::text, false)
), managed_cloudsqladmin_exception AS MATERIALIZED (
    SELECT (SELECT pg_catalog.count(*) = 1 AND COALESCE(
                       pg_catalog.bool_and(
                           datallowconn AND NOT datistemplate
                           AND owner_name = 'cloudsqladmin'
                       ), false
                   ) FROM managed_database)
       AND EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles
             WHERE rolname = 'cloudsqladmin'
               AND rolcanlogin AND rolsuper AND rolcreatedb AND rolcreaterole
               AND rolreplication AND rolbypassrls
       )
       AND EXISTS (
            SELECT 1 FROM pg_catalog.pg_roles
             WHERE rolname = 'cloudsqlsuperuser'
               AND rolcanlogin AND NOT rolsuper
               AND rolcreatedb AND rolcreaterole
               AND NOT rolreplication AND NOT rolbypassrls
       )
       AND NOT EXISTS (
            (SELECT * FROM managed_actual_database_acl
             EXCEPT SELECT * FROM managed_expected_database_acl)
            UNION ALL
            (SELECT * FROM managed_expected_database_acl
             EXCEPT SELECT * FROM managed_actual_database_acl)
       ) AS exact
), routine_facts AS MATERIALIZED (
    SELECT routine.oid,
           routine.proname,
           routine.proowner,
           routine.proacl,
           routine.proconfig,
           routine.pronargs,
           routine.prokind,
           routine.prosecdef,
           routine.provolatile,
           routine.proparallel,
           routine.proleakproof,
           routine.proisstrict,
           routine.proretset,
           language.lanname,
           pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
               routine.prosrc, 'UTF8'
           )), 'hex') AS prosrc_sha256,
           pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
               pg_catalog.pg_get_functiondef(routine.oid), 'UTF8'
           )), 'hex') AS definition_sha256
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_language AS language
        ON language.oid = routine.prolang
     WHERE routine.pronamespace = (
         SELECT oid FROM control_namespace
     )
), facts AS MATERIALIZED (
    SELECT pg_catalog.current_database()::text AS database_name,
           pg_catalog.current_setting('server_version_num')::text
               AS version_num,
           SESSION_USER::text AS session_user_name,
           (
               SELECT pg_catalog.count(*)::bigint
                 FROM pg_catalog.pg_roles
                WHERE rolname ~ '^muncho_canary_control_[0-9a-f]{{16}}$'
           ) AS control_admin_count,
           (
               SESSION_USER ~ '^muncho_canary_control_[0-9a-f]{{16}}$'
               AND CURRENT_USER = SESSION_USER
               AND (SELECT pg_catalog.count(*) FROM session_role) = 1
               AND (SELECT pg_catalog.bool_and(
                       rolcanlogin AND rolinherit AND NOT rolsuper
                       AND rolcreatedb AND rolcreaterole
                       AND NOT rolreplication AND NOT rolbypassrls
                       AND rolconnlimit = -1 AND rolvaliduntil IS NULL
                       AND rolconfig IS NULL
                   ) FROM session_role)
               AND (
                   SELECT pg_catalog.count(*) IN (1, 2)
                          AND pg_catalog.count(*) FILTER (
                              WHERE granted_name = 'cloudsqlsuperuser'
                                AND member_name = SESSION_USER
                                AND grantor_name = 'cloudsqladmin'
                                AND admin_option IS FALSE
                                AND inherit_option IS TRUE
                                AND set_option IS TRUE
                          ) = 1
                          AND pg_catalog.count(*) FILTER (
                              WHERE granted_name = '{EXECUTOR_ROLE}'
                                AND member_name = SESSION_USER
                                AND grantor_name = 'cloudsqladmin'
                                AND admin_option IS TRUE
                                AND inherit_option IS FALSE
                                AND set_option IS FALSE
                          ) = (
                              SELECT pg_catalog.count(*)
                                FROM pg_catalog.pg_auth_members
                               WHERE roleid = (SELECT oid FROM executor)
                          )
                          AND pg_catalog.bool_and(
                              (
                                  granted_name = 'cloudsqlsuperuser'
                                  AND member_name = SESSION_USER
                                  AND grantor_name = 'cloudsqladmin'
                                  AND admin_option IS FALSE
                                  AND inherit_option IS TRUE
                                  AND set_option IS TRUE
                              )
                              OR (
                                  granted_name = '{EXECUTOR_ROLE}'
                                  AND member_name = SESSION_USER
                                  AND grantor_name = 'cloudsqladmin'
                                  AND admin_option IS TRUE
                                  AND inherit_option IS FALSE
                                  AND set_option IS FALSE
                              )
                          )
                     FROM relevant_session_edges
               )
               AND (
                   SELECT pg_catalog.count(DISTINCT role.rolname) =
                              1 + (
                                  SELECT pg_catalog.count(*)
                                    FROM pg_catalog.pg_auth_members
                                   WHERE roleid = (SELECT oid FROM executor)
                              )
                          AND pg_catalog.bool_and(
                              role.rolname IN (
                                  'cloudsqlsuperuser', '{EXECUTOR_ROLE}'
                              )
                          )
                     FROM forward_role_closure AS closure
                     JOIN pg_catalog.pg_roles AS role
                       ON role.oid = closure.roleid
               )
           ) AS control_admin_role_exact,
           CASE
               WHEN SESSION_USER ~
                    '^muncho_canary_control_[0-9a-f]{{16}}$'
               THEN (
                   SELECT pg_catalog.count(DISTINCT roleid)::bigint
                     FROM forward_role_closure
               )
               ELSE 0::bigint
           END AS control_admin_forward_role_count,
           (
               SELECT pg_catalog.count(*)::bigint
                 FROM pg_catalog.pg_roles AS control_admin
                 JOIN pg_catalog.pg_shdepend AS dependency
                   ON dependency.refclassid =
                      'pg_catalog.pg_authid'::pg_catalog.regclass
                  AND dependency.refobjid = control_admin.oid
                WHERE control_admin.rolname ~
                      '^muncho_canary_control_[0-9a-f]{{16}}$'
                  AND dependency.deptype = 'o'
           ) AS control_admin_owned_object_count,
           (
               SELECT pg_catalog.count(*)::bigint
                 FROM pg_catalog.pg_roles AS control_admin
                 JOIN pg_catalog.pg_shdepend AS dependency
                   ON dependency.refclassid =
                      'pg_catalog.pg_authid'::pg_catalog.regclass
                  AND dependency.refobjid = control_admin.oid
                WHERE control_admin.rolname ~
                      '^muncho_canary_control_[0-9a-f]{{16}}$'
           ) AS control_admin_shared_dependency_count,
           (
               SELECT pg_catalog.count(*)::bigint
                 FROM pg_catalog.pg_stat_activity AS activity
                WHERE activity.backend_type = 'client backend'
                  AND activity.pid <> pg_catalog.pg_backend_pid()
                  AND (
                      activity.datname = pg_catalog.current_database()
                      OR activity.usename = SESSION_USER
                  )
           ) AS foreign_client_session_count,
           pg_catalog.current_setting(
               'max_prepared_transactions'
           )::bigint AS max_prepared_transactions,
           (
               SELECT pg_catalog.count(*)::bigint
                 FROM pg_catalog.pg_prepared_xacts
           ) AS cluster_prepared_xact_count,
           (
               SELECT pg_catalog.count(*) = 3
                      AND COALESCE(pg_catalog.string_agg(
                          database.datname::text,
                          ',' ORDER BY database.datname::text
                      ), '') = 'cloudsqladmin,muncho_canary_brain,postgres'
                 FROM pg_catalog.pg_database AS database
                WHERE database.datallowconn AND NOT database.datistemplate
           ) AS non_template_database_inventory_exact,
           (
               SELECT pg_catalog.count(*) = 4
                      AND COALESCE(pg_catalog.string_agg(
                          database.datname::text,
                          ',' ORDER BY database.datname::text
                      ), '') =
                          'cloudsqladmin,muncho_canary_brain,postgres,template1'
                 FROM pg_catalog.pg_database AS database
                WHERE database.datallowconn
           ) AS all_connectable_database_inventory_exact,
           (SELECT exact FROM managed_cloudsqladmin_exception)
               AS latent_provider_exception_exact,
           CASE
               WHEN NOT EXISTS (SELECT 1 FROM executor) THEN true
               ELSE
                   pg_catalog.has_database_privilege(
                       (SELECT oid FROM executor),
                       pg_catalog.current_database(), 'CONNECT'
                   )
                   AND NOT pg_catalog.has_database_privilege(
                       (SELECT oid FROM executor),
                       pg_catalog.current_database(), 'CREATE'
                   )
                   AND NOT pg_catalog.has_database_privilege(
                       (SELECT oid FROM executor),
                       pg_catalog.current_database(), 'TEMPORARY'
                   )
                   AND NOT EXISTS (
                       SELECT 1
                         FROM pg_catalog.pg_database AS database
                        WHERE database.datallowconn
                          AND database.datname <>
                              pg_catalog.current_database()
                          AND (
                              pg_catalog.pg_get_userbyid(database.datdba) =
                                  '{EXECUTOR_ROLE}'
                              OR pg_catalog.has_database_privilege(
                                  (SELECT oid FROM executor),
                                  database.oid, 'CONNECT'
                              )
                              OR pg_catalog.has_database_privilege(
                                  (SELECT oid FROM executor),
                                  database.oid, 'CREATE'
                              )
                              OR pg_catalog.has_database_privilege(
                                  (SELECT oid FROM executor),
                                  database.oid, 'TEMPORARY'
                              )
                          )
                          AND NOT (
                              database.datname = 'cloudsqladmin'
                              AND (
                                  SELECT exact
                                    FROM managed_cloudsqladmin_exception
                              )
                              AND pg_catalog.pg_get_userbyid(
                                  database.datdba
                              ) <> '{EXECUTOR_ROLE}'
                              AND pg_catalog.has_database_privilege(
                                  (SELECT oid FROM executor),
                                  database.oid, 'CONNECT'
                              )
                              AND NOT pg_catalog.has_database_privilege(
                                  (SELECT oid FROM executor),
                                  database.oid, 'CREATE'
                              )
                              AND pg_catalog.has_database_privilege(
                                  (SELECT oid FROM executor),
                                  database.oid, 'TEMPORARY'
                              )
                          )
                   )
           END AS executor_database_effective_privileges_exact,
           EXISTS (
               SELECT 1 FROM pg_catalog.pg_roles AS owner
                WHERE owner.rolname = 'canonical_brain_migration_owner'
                  AND NOT owner.rolcanlogin
                  AND NOT owner.rolinherit
                  AND NOT owner.rolsuper
                  AND NOT owner.rolcreatedb
                  AND NOT owner.rolcreaterole
                  AND NOT owner.rolreplication
                  AND NOT owner.rolbypassrls
                  AND owner.rolconnlimit = -1
                  AND owner.rolvaliduntil IS NULL
                  AND owner.rolconfig IS NULL
           ) AS migration_owner_role_exact,
           EXISTS (
               SELECT 1 FROM pg_catalog.pg_database AS database
                WHERE database.datname = pg_catalog.current_database()
                  AND pg_catalog.pg_get_userbyid(database.datdba) =
                      'cloudsqlsuperuser'
           ) AS current_database_owner_exact,
           (
               SELECT pg_catalog.count(*)::bigint
                 FROM pg_catalog.pg_auth_members AS membership
                WHERE membership.roleid = (SELECT oid FROM executor)
                   OR membership.member = (SELECT oid FROM executor)
                   OR membership.grantor = (SELECT oid FROM executor)
           ) AS executor_membership_count,
           (
               SELECT pg_catalog.count(*)::bigint
                 FROM pg_catalog.pg_shdepend AS dependency
                WHERE dependency.refclassid =
                      'pg_catalog.pg_authid'::pg_catalog.regclass
                  AND dependency.refobjid = (SELECT oid FROM executor)
                  AND dependency.deptype = 'o'
           ) AS executor_owned_object_count,
           (
               SELECT pg_catalog.count(*)::bigint
                 FROM pg_catalog.pg_shdepend AS dependency
                WHERE dependency.refclassid =
                      'pg_catalog.pg_authid'::pg_catalog.regclass
                  AND dependency.refobjid = (SELECT oid FROM executor)
                  AND dependency.deptype = 'a'
           ) AS executor_acl_dependency_count,
           (
               SELECT pg_catalog.max(prosrc_sha256)
                 FROM routine_facts
                WHERE proname = 'observe_missing_discord_routeback_helper_v1'
           ) AS observer_prosrc_sha256,
           (
               SELECT pg_catalog.max(definition_sha256)
                 FROM routine_facts
                WHERE proname = 'observe_missing_discord_routeback_helper_v1'
           ) AS observer_definition_sha256,
           (
               SELECT pg_catalog.max(prosrc_sha256)
                 FROM routine_facts
                WHERE proname = 'apply_missing_discord_routeback_helper_v1'
           ) AS apply_prosrc_sha256,
           (
               SELECT pg_catalog.max(definition_sha256)
                 FROM routine_facts
                WHERE proname = 'apply_missing_discord_routeback_helper_v1'
           ) AS apply_definition_sha256,
           NOT EXISTS (
               SELECT 1
                 FROM pg_catalog.pg_proc AS helper
                 JOIN pg_catalog.pg_namespace AS namespace
                   ON namespace.oid = helper.pronamespace
                WHERE namespace.nspname = 'canonical_brain'
                  AND helper.proname =
                      '_discord_guild_routeback_target_valid'
           ) AS helper_absent,
           (
               SELECT pg_catalog.count(*)::bigint
                 FROM pg_catalog.pg_proc AS helper
                 JOIN pg_catalog.pg_namespace AS namespace
                   ON namespace.oid = helper.pronamespace
                WHERE namespace.nspname = 'canonical_brain'
                  AND helper.proname =
                      '_discord_guild_routeback_target_valid'
           ) AS helper_same_name_count,
           (
               NOT EXISTS (SELECT 1 FROM executor)
               AND NOT EXISTS (SELECT 1 FROM control_namespace)
               AND NOT EXISTS (SELECT 1 FROM routine_facts)
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_roles
                    WHERE rolname ~ '^muncho_canary_reconciler_[0-9a-f]{{16}}$'
               )
               AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_event_trigger)
           ) AS absent_exact,
           (
               (SELECT pg_catalog.count(*) FROM executor) = 1
               AND (SELECT pg_catalog.bool_and(
                       NOT rolcanlogin
                       AND NOT rolinherit
                       AND NOT rolsuper
                       AND NOT rolcreatedb
                       AND NOT rolcreaterole
                       AND NOT rolreplication
                       AND NOT rolbypassrls
                       AND rolconnlimit = -1
                       AND rolvaliduntil IS NULL
                       AND rolconfig IS NULL
                   ) FROM executor)
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_auth_members
                    WHERE member = (SELECT oid FROM executor)
               )
               AND (
                   SELECT pg_catalog.count(*) <= 1
                          AND COALESCE(pg_catalog.bool_and(
                              membership.roleid = (SELECT oid FROM executor)
                              AND member.rolname ~
                                  '^muncho_canary_control_[0-9a-f]{{16}}$'
                              AND grantor.rolname = 'cloudsqladmin'
                              AND membership.admin_option IS TRUE
                              AND membership.inherit_option IS FALSE
                              AND membership.set_option IS FALSE
                          ), TRUE)
                     FROM pg_catalog.pg_auth_members AS membership
                     JOIN pg_catalog.pg_roles AS member
                       ON member.oid = membership.member
                     JOIN pg_catalog.pg_roles AS grantor
                       ON grantor.oid = membership.grantor
                    WHERE membership.roleid = (SELECT oid FROM executor)
                       OR membership.member = (SELECT oid FROM executor)
                       OR membership.grantor = (SELECT oid FROM executor)
               )
               AND (
                   SELECT pg_catalog.count(*) FROM pg_catalog.pg_shdepend
                    WHERE refclassid =
                          'pg_catalog.pg_authid'::pg_catalog.regclass
                      AND refobjid = (SELECT oid FROM executor)
                      AND deptype = 'o'
               ) = 0
               AND (
                   SELECT pg_catalog.count(*) = 4
                          AND pg_catalog.bool_and(
                              dependency.objsubid = 0
                              AND (
                                  (
                                      dependency.dbid = 0
                                      AND dependency.classid =
                                          'pg_catalog.pg_database'::pg_catalog.regclass
                                      AND dependency.objid = (
                                          SELECT oid FROM pg_catalog.pg_database
                                           WHERE datname = pg_catalog.current_database()
                                      )
                                  )
                                  OR (
                                      dependency.dbid = (
                                          SELECT oid FROM pg_catalog.pg_database
                                           WHERE datname = pg_catalog.current_database()
                                      )
                                      AND dependency.classid =
                                          'pg_catalog.pg_namespace'::pg_catalog.regclass
                                      AND dependency.objid =
                                          (SELECT oid FROM control_namespace)
                                  )
                                  OR (
                                      dependency.dbid = (
                                          SELECT oid FROM pg_catalog.pg_database
                                           WHERE datname = pg_catalog.current_database()
                                      )
                                      AND dependency.classid =
                                          'pg_catalog.pg_proc'::pg_catalog.regclass
                                      AND dependency.objid IN (
                                          SELECT oid FROM routine_facts
                                      )
                                  )
                              )
                          )
                     FROM pg_catalog.pg_shdepend AS dependency
                    WHERE refclassid =
                          'pg_catalog.pg_authid'::pg_catalog.regclass
                      AND refobjid = (SELECT oid FROM executor)
                      AND deptype = 'a'
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_shdepend AS dependency
                    WHERE dependency.refclassid =
                          'pg_catalog.pg_authid'::pg_catalog.regclass
                      AND dependency.refobjid = (SELECT oid FROM executor)
                      AND dependency.deptype <> 'a'
               )
               AND (SELECT pg_catalog.count(*) FROM control_namespace) = 1
               AND (
                   SELECT pg_catalog.bool_and(
                       pg_catalog.pg_get_userbyid(nspowner)
                           = 'canonical_brain_migration_owner'
                   ) FROM control_namespace
               )
               AND NOT EXISTS (
                   SELECT 1
                     FROM pg_catalog.pg_class AS relation
                    WHERE relation.relnamespace =
                          (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_type
                    WHERE typnamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_constraint
                    WHERE connamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_collation
                    WHERE collnamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_conversion
                    WHERE connamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_operator
                    WHERE oprnamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_opclass
                    WHERE opcnamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_opfamily
                    WHERE opfnamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_statistic_ext
                    WHERE stxnamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_ts_config
                    WHERE cfgnamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_ts_dict
                    WHERE dictnamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_ts_parser
                    WHERE prsnamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_ts_template
                    WHERE tmplnamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_extension
                    WHERE extnamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_default_acl
                    WHERE defaclnamespace = (SELECT oid FROM control_namespace)
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_publication_namespace
                    WHERE pnnspid = (SELECT oid FROM control_namespace)
               )
               AND NOT pg_catalog.has_schema_privilege(
                   '{EXECUTOR_ROLE}',
                   (SELECT oid FROM control_namespace),
                   'CREATE'
               )
               AND pg_catalog.has_schema_privilege(
                   '{EXECUTOR_ROLE}',
                   (SELECT oid FROM control_namespace),
                   'USAGE'
               )
               AND pg_catalog.has_database_privilege(
                   '{EXECUTOR_ROLE}',
                   'muncho_canary_brain',
                   'CONNECT'
               )
               AND NOT pg_catalog.has_database_privilege(
                   '{EXECUTOR_ROLE}',
                   'muncho_canary_brain',
                   'CREATE'
               )
               AND NOT pg_catalog.has_database_privilege(
                   '{EXECUTOR_ROLE}',
                   'muncho_canary_brain',
                   'TEMPORARY'
               )
               AND (
                   SELECT pg_catalog.count(*) = 1
                          AND pg_catalog.bool_and(
                              acl.grantee = (SELECT oid FROM executor)
                              AND acl.grantor = database.datdba
                              AND acl.privilege_type = 'CONNECT'
                              AND acl.is_grantable IS FALSE
                          )
                     FROM pg_catalog.pg_database AS database,
                          LATERAL pg_catalog.aclexplode(COALESCE(
                              database.datacl,
                              pg_catalog.acldefault('d', database.datdba)
                          )) AS acl
                    WHERE database.datname = pg_catalog.current_database()
                      AND acl.grantee = (SELECT oid FROM executor)
               )
               AND (
                   SELECT pg_catalog.count(*) = 3
                          AND pg_catalog.bool_and(
                              acl.grantor = namespace.nspowner
                              AND acl.is_grantable IS FALSE
                              AND (
                                  (
                                      acl.grantee = namespace.nspowner
                                      AND acl.privilege_type IN
                                          ('CREATE', 'USAGE')
                                  )
                                  OR (
                                      acl.grantee = (SELECT oid FROM executor)
                                      AND acl.privilege_type = 'USAGE'
                                  )
                              )
                          )
                     FROM control_namespace AS namespace,
                          LATERAL pg_catalog.aclexplode(COALESCE(
                              namespace.nspacl,
                              pg_catalog.acldefault('n', namespace.nspowner)
                          )) AS acl
               )
               AND (SELECT pg_catalog.count(*) FROM routine_facts) = 2
               AND (
                   SELECT pg_catalog.count(DISTINCT proname) = 2
                          AND pg_catalog.bool_and(
                              proname IN (
                                  'observe_missing_discord_routeback_helper_v1',
                                  'apply_missing_discord_routeback_helper_v1'
                              )
                              AND pg_catalog.pg_get_userbyid(proowner)
                                  = 'canonical_brain_migration_owner'
                              AND pronargs = 0
                              AND prokind = 'f'
                              AND prosecdef IS TRUE
                              AND provolatile = 'v'
                              AND proparallel = 'u'
                              AND proleakproof IS FALSE
                              AND proisstrict IS FALSE
                              AND proretset IS TRUE
                              AND lanname = 'plpgsql'
                              AND proconfig = ARRAY[
                                  'search_path=pg_catalog, pg_temp',
                                  'TimeZone=UTC',
                                  'DateStyle=ISO, YMD',
                                  'IntervalStyle=postgres',
                                  'extra_float_digits=3',
                                  'bytea_output=hex',
                                  'lock_timeout=15s',
                                  'statement_timeout=5min'
                              ]::text[]
                              AND (
                                  SELECT pg_catalog.count(*) = 2
                                         AND pg_catalog.bool_and(
                                             acl.grantor = routine.proowner
                                             AND acl.privilege_type = 'EXECUTE'
                                             AND acl.is_grantable IS FALSE
                                             AND acl.grantee IN (
                                                 routine.proowner,
                                                 (SELECT oid FROM executor)
                                             )
                                         )
                                    FROM pg_catalog.aclexplode(COALESCE(
                                        routine.proacl,
                                        pg_catalog.acldefault(
                                            'f', routine.proowner
                                        )
                                    )) AS acl
                              )
                          )
                     FROM routine_facts AS routine
               )
               AND (
                   SELECT pg_catalog.max(prosrc_sha256)
                     FROM routine_facts
                    WHERE proname =
                          'observe_missing_discord_routeback_helper_v1'
               ) = '{OBSERVER_PROSRC_SHA256}'
               AND (
                   SELECT pg_catalog.max(definition_sha256)
                     FROM routine_facts
                    WHERE proname =
                          'observe_missing_discord_routeback_helper_v1'
               ) = '{OBSERVER_DEFINITION_SHA256}'
               AND (
                   SELECT pg_catalog.max(prosrc_sha256)
                     FROM routine_facts
                    WHERE proname =
                          'apply_missing_discord_routeback_helper_v1'
               ) = '{APPLY_PROSRC_SHA256}'
               AND (
                   SELECT pg_catalog.max(definition_sha256)
                     FROM routine_facts
                    WHERE proname =
                          'apply_missing_discord_routeback_helper_v1'
               ) = '{APPLY_DEFINITION_SHA256}'
               AND NOT EXISTS (
                   SELECT 1
                     FROM pg_catalog.pg_auth_members AS membership
                     JOIN pg_catalog.pg_roles AS owner
                       ON owner.oid = membership.roleid
                     JOIN pg_catalog.pg_roles AS member
                       ON member.oid = membership.member
                    WHERE owner.rolname = 'canonical_brain_migration_owner'
                      AND member.rolname ~
                          '^muncho_canary_control_[0-9a-f]{{16}}$'
               )
               AND NOT EXISTS (
                   SELECT 1 FROM pg_catalog.pg_roles
                    WHERE rolname ~ '^muncho_canary_reconciler_[0-9a-f]{{16}}$'
               )
               AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_event_trigger)
           ) AS installed_exact
), classified AS MATERIALIZED (
    SELECT facts.*,
           (
               foreign_client_session_count = 0
               AND max_prepared_transactions = 0
               AND cluster_prepared_xact_count = 0
               AND control_admin_owned_object_count = 0
               AND control_admin_shared_dependency_count = 0
               AND CURRENT_USER = SESSION_USER
               AND (
                   (
                       session_user_name ~
                           '^muncho_canary_control_[0-9a-f]{{16}}$'
                       AND control_admin_count = 1
                       AND control_admin_role_exact
                       AND control_admin_forward_role_count =
                           1 + executor_membership_count
                   )
                   OR (
                       session_user_name = '{foundation.SQL_USER}'
                       AND control_admin_count = 0
                       AND NOT control_admin_role_exact
                       AND control_admin_forward_role_count = 0
                   )
               )
           ) AS session_boundary_exact
           ,(
               non_template_database_inventory_exact
               AND all_connectable_database_inventory_exact
               AND latent_provider_exception_exact
               AND executor_database_effective_privileges_exact
               AND migration_owner_role_exact
               AND current_database_owner_exact
           ) AS database_environment_exact
      FROM facts
)
SELECT database_name,
       version_num,
       session_user_name,
       control_admin_count::text AS control_admin_count,
       control_admin_role_exact::text AS control_admin_role_exact,
       control_admin_forward_role_count::text
           AS control_admin_forward_role_count,
       control_admin_owned_object_count::text
           AS control_admin_owned_object_count,
       control_admin_shared_dependency_count::text
           AS control_admin_shared_dependency_count,
       foreign_client_session_count::text AS foreign_client_session_count,
       max_prepared_transactions::text AS max_prepared_transactions,
       cluster_prepared_xact_count::text AS cluster_prepared_xact_count,
       non_template_database_inventory_exact::text
           AS non_template_database_inventory_exact,
       all_connectable_database_inventory_exact::text
           AS all_connectable_database_inventory_exact,
       latent_provider_exception_exact::text
           AS latent_provider_exception_exact,
       executor_database_effective_privileges_exact::text
           AS executor_database_effective_privileges_exact,
       migration_owner_role_exact::text AS migration_owner_role_exact,
       current_database_owner_exact::text AS current_database_owner_exact,
       executor_membership_count::text AS executor_membership_count,
       executor_owned_object_count::text AS executor_owned_object_count,
       executor_acl_dependency_count::text AS executor_acl_dependency_count,
       observer_prosrc_sha256,
       observer_definition_sha256,
       apply_prosrc_sha256,
       apply_definition_sha256,
       CASE
           WHEN absent_exact AND helper_same_name_count = 0
                AND session_boundary_exact
                AND database_environment_exact THEN 'absent'
           WHEN installed_exact AND helper_same_name_count = 0
                AND session_boundary_exact
                AND database_environment_exact THEN 'exact_installed'
           ELSE 'drift'
       END::text AS foundation_state,
       (
           installed_exact
           AND helper_same_name_count = 0
           AND session_boundary_exact
           AND database_environment_exact
       )::text AS foundation_exact,
       helper_absent::text AS helper_absent,
       helper_same_name_count::text AS helper_same_name_count
  FROM classified;
COMMIT;
""".strip()


def _parse_decimal(value: Any, code: str) -> int:
    if not isinstance(value, str) or not value.isdigit():
        _fail(code)
    return int(value)


def _parse_boolean(value: Any, code: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    _fail(code)


def _parse_foundation_observation_result(
    session: Any,
    result: Any,
    *,
    phase: str,
    observed_at_unix: int,
) -> Mapping[str, Any]:
    code = "schema_reconciliation_control_database_observation_invalid"
    if (
        not isinstance(result, QueryResult)
        or result.command_tag.upper() != "COMMIT"
        or result.columns != _FOUNDATION_OBSERVATION_COLUMNS
        or len(result.rows) != 1
        or len(result.rows[0]) != len(_FOUNDATION_OBSERVATION_COLUMNS)
    ):
        _fail(code)
    (
        database,
        version_text,
        session_user,
        control_admin_count,
        control_admin_role_exact,
        control_admin_forward_role_count,
        control_admin_owned_object_count,
        control_admin_shared_dependency_count,
        foreign_client_session_count,
        max_prepared_transactions,
        cluster_prepared_xact_count,
        non_template_database_inventory_exact,
        all_connectable_database_inventory_exact,
        latent_provider_exception_exact,
        executor_database_effective_privileges_exact,
        migration_owner_role_exact,
        current_database_owner_exact,
        executor_membership_count,
        executor_owned_object_count,
        executor_acl_dependency_count,
        observer_prosrc_sha256,
        observer_definition_sha256,
        apply_prosrc_sha256,
        apply_definition_sha256,
        state,
        foundation_exact,
        helper_absent,
        helper_same_name_count,
    ) = result.rows[0]
    try:
        version = int(str(version_text))
    except (TypeError, ValueError) as exc:
        raise ControlBootstrapError(code) from exc
    if (
        database != foundation.SQL_DATABASE
        or version // 10000 != 18
        or not isinstance(session_user, str)
        or not session_user
        or session_user != getattr(session, "username", None)
        or state not in {"absent", "exact_installed"}
    ):
        _fail(code)
    unsigned = {
        "schema": FOUNDATION_OBSERVATION_SCHEMA,
        "phase": phase,
        "state": state,
        "database": database,
        "postgresql_major": version // 10000,
        "session_user_sha256": _sha256_bytes(session_user.encode("utf-8")),
        "control_admin_count": _parse_decimal(control_admin_count, code),
        "control_admin_role_exact": _parse_boolean(
            control_admin_role_exact, code
        ),
        "control_admin_forward_role_count": _parse_decimal(
            control_admin_forward_role_count, code
        ),
        "control_admin_owned_object_count": _parse_decimal(
            control_admin_owned_object_count, code
        ),
        "control_admin_shared_dependency_count": _parse_decimal(
            control_admin_shared_dependency_count, code
        ),
        "foreign_client_session_count": _parse_decimal(
            foreign_client_session_count, code
        ),
        "max_prepared_transactions": _parse_decimal(
            max_prepared_transactions, code
        ),
        "cluster_prepared_xact_count": _parse_decimal(
            cluster_prepared_xact_count, code
        ),
        "non_template_database_inventory_exact": _parse_boolean(
            non_template_database_inventory_exact, code
        ),
        "all_connectable_database_inventory_exact": _parse_boolean(
            all_connectable_database_inventory_exact, code
        ),
        "latent_provider_exception_exact": _parse_boolean(
            latent_provider_exception_exact, code
        ),
        "executor_database_effective_privileges_exact": _parse_boolean(
            executor_database_effective_privileges_exact, code
        ),
        "migration_owner_role_exact": _parse_boolean(
            migration_owner_role_exact, code
        ),
        "current_database_owner_exact": _parse_boolean(
            current_database_owner_exact, code
        ),
        "executor_membership_count": _parse_decimal(
            executor_membership_count, code
        ),
        "executor_owned_object_count": _parse_decimal(
            executor_owned_object_count, code
        ),
        "executor_acl_dependency_count": _parse_decimal(
            executor_acl_dependency_count, code
        ),
        "observer_prosrc_sha256": observer_prosrc_sha256,
        "observer_definition_sha256": observer_definition_sha256,
        "apply_prosrc_sha256": apply_prosrc_sha256,
        "apply_definition_sha256": apply_definition_sha256,
        "foundation_exact": _parse_boolean(foundation_exact, code),
        "helper_absent": _parse_boolean(helper_absent, code),
        "helper_same_name_count": _parse_decimal(
            helper_same_name_count, code
        ),
        "observed_at_unix": observed_at_unix,
    }
    return _validate_observation(
        _hashed(unsigned, "observation_sha256"),
        phase=phase,
    )


def _observation_lock_result_exact(result: Any) -> bool:
    return (
        isinstance(result, QueryResult)
        and result.command_tag.upper() == "SELECT 1"
        and result.columns == ("pg_advisory_lock_shared",)
        and len(result.rows) == 1
        and len(result.rows[0]) == 1
    )


def _observation_command_result_exact(result: Any, command: str) -> bool:
    return (
        isinstance(result, QueryResult)
        and result.command_tag.upper() == command
        and result.columns == ()
        and result.rows == ()
    )


def _observation_unlock_result_exact(result: Any) -> bool:
    return (
        isinstance(result, QueryResult)
        and result.command_tag.upper() == "SELECT 1"
        and result.columns == ("pg_advisory_unlock_shared",)
        and result.rows == (("t",),)
    )


def _rollback_observation_quietly(session: Any) -> None:
    try:
        session.query("ROLLBACK", maximum_rows=0)
    except BaseException:
        pass


def _close_observation_session_quietly(session: Any) -> None:
    try:
        session.close()
    except BaseException:
        pass


def _observe_foundation(
    session: Any,
    *,
    phase: str,
    observed_at_unix: int | Callable[[], int],
) -> Mapping[str, Any]:
    """Observe one fresh SERIALIZABLE snapshot under the deployment lock."""

    code = "schema_reconciliation_control_database_observation_invalid"
    session_setting_started = False
    session_setting_restored = False
    lock_query_started = False
    lock_acquired = False
    observation_query_started = False
    primary: BaseException | None = None
    receipt: Mapping[str, Any] | None = None
    try:
        # A transaction-level lock inside the first SELECT would wait only
        # after PostgreSQL had already fixed that statement's snapshot.  The
        # session lock therefore has to complete before BEGIN is sent.
        session_setting_started = True
        timeout_result = session.query(
            _FOUNDATION_OBSERVATION_SET_LOCK_TIMEOUT_SQL,
            maximum_rows=0,
        )
        if not _observation_command_result_exact(timeout_result, "SET"):
            _fail(code)
        lock_query_started = True
        lock_result = session.query(
            _FOUNDATION_OBSERVATION_LOCK_SQL,
            maximum_rows=1,
        )
        lock_acquired = True
        if not _observation_lock_result_exact(lock_result):
            _fail(code)
        reset_result = session.query(
            _FOUNDATION_OBSERVATION_RESET_LOCK_TIMEOUT_SQL,
            maximum_rows=0,
        )
        if not _observation_command_result_exact(reset_result, "RESET"):
            _fail(code)
        session_setting_restored = True
        observation_query_started = True
        result = session.query(_FOUNDATION_OBSERVATION_SQL, maximum_rows=1)
        captured_at_unix = (
            observed_at_unix()
            if callable(observed_at_unix)
            else observed_at_unix
        )
        receipt = _parse_foundation_observation_result(
            session,
            result,
            phase=phase,
            observed_at_unix=captured_at_unix,
        )
    except BaseException as exc:
        primary = exc
        if observation_query_started:
            _rollback_observation_quietly(session)
    finally:
        if lock_acquired:
            try:
                unlock_result = session.query(
                    _FOUNDATION_OBSERVATION_UNLOCK_SQL,
                    maximum_rows=1,
                )
                if not _observation_unlock_result_exact(unlock_result):
                    _fail(code)
                lock_acquired = False
            except BaseException as exc:
                # Closing a PostgreSQL session releases every session-level
                # advisory lock even when the explicit unlock receipt is lost.
                _close_observation_session_quietly(session)
                if primary is None:
                    primary = exc
        elif lock_query_started and primary is not None:
            # A transport failure can hide whether PostgreSQL executed the
            # lock SELECT.  Terminate the capability instead of guessing.
            _close_observation_session_quietly(session)
        if session_setting_started and not session_setting_restored:
            _close_observation_session_quietly(session)
    if primary is not None:
        if isinstance(primary, ControlBootstrapError):
            raise primary
        raise ControlBootstrapError(code) from primary
    if receipt is None:
        _fail(code)
    return receipt


@dataclass(frozen=True)
class _ControlRuntimeDependencies:
    base_dependencies: reconciliation_runtime._RuntimeDependencies = field(
        default_factory=reconciliation_runtime._RuntimeDependencies
    )
    prepare_runtime: Callable[[Any], Any] = (
        reconciliation_runtime._prepare_runtime
    )
    load_control_artifact: Callable[..., Any] = _load_control_artifact
    protocol_runner: Callable[..., Mapping[str, Any]] = run_protocol


@dataclass
class _ControlRuntimeContext:
    base: Any
    gate: Mapping[str, Any]
    install_artifact: Any
    temporary_database_session_closed: bool = False
    install_callback_used: bool = False
    post_cleanup_callback_used: bool = False


def _prepare_control_runtime(
    dependencies: _ControlRuntimeDependencies,
) -> _ControlRuntimeContext:
    try:
        base = dependencies.prepare_runtime(dependencies.base_dependencies)
        plan = base.plan
        artifact = dependencies.load_control_artifact(
            base.revision,
            name="schema_reconciliation_control_install",
            filename=INSTALL_ARTIFACT_FILENAME,
        )
        if (
            artifact.name != "schema_reconciliation_control_install"
            or artifact.path.name != INSTALL_ARTIFACT_FILENAME
            or artifact.sha256
            != plan.value["control_install_artifact_sha256"]
            or not isinstance(artifact.payload, bytes)
            or _sha256_bytes(artifact.payload) != artifact.sha256
            or plan.value["control_foundation_contract_sha256"]
            != _control_foundation_contract_sha256(
                plan.value["control_install_artifact_sha256"],
                plan.value["control_retire_artifact_sha256"],
            )
        ):
            _fail("schema_reconciliation_control_release_invalid")
        issued_at = base.dependencies.now()
        nonce = base.dependencies.random_bytes(32)
        if (
            type(issued_at) is not int
            or not isinstance(nonce, bytes)
            or len(nonce) != 32
        ):
            _fail("schema_reconciliation_control_clock_invalid")
        username = CONTROL_ADMIN_PREFIX + plan.sha256[:16]
        unsigned_gate = {
            "schema": GATE_SCHEMA,
            "ok": True,
            "state": "stopped_release_control_bootstrap_ready",
            "release_revision": base.revision,
            **base.initial_release_binding,
            "plan_sha256": plan.sha256,
            "control_install_artifact_sha256": plan.value[
                "control_install_artifact_sha256"
            ],
            "control_retire_artifact_sha256": plan.value[
                "control_retire_artifact_sha256"
            ],
            "control_foundation_contract_sha256": plan.value[
                "control_foundation_contract_sha256"
            ],
            "advisory_lock_key": CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY,
            "host_identity_sha256": base.initial_host_state["state_sha256"],
            "services_stopped_sha256": base.initial_services_state[
                "state_sha256"
            ],
            "project": foundation.PROJECT,
            "sql_instance": foundation.SQL_INSTANCE,
            "database": foundation.SQL_DATABASE,
            "postgresql_major": 18,
            "tls_server_name": foundation.SQL_TLS_SERVER_NAME,
            "ca_file_sha256": base.gate["ca_file_sha256"],
            "temporary_control_admin_username": username,
            "temporary_control_admin_username_sha256": _sha256_bytes(
                username.encode("ascii")
            ),
            "owner_subject_sha256": (
                reconciliation_runtime.OWNER_SUBJECT_SHA256
            ),
            "owner_public_key_ed25519_hex": (
                reconciliation_runtime.OWNER_PUBLIC_KEY_ED25519_HEX
            ),
            "owner_key_id": reconciliation_runtime.OWNER_KEY_ID,
            "owner_public_fingerprint": (
                reconciliation_runtime.OWNER_PUBLIC_FINGERPRINT
            ),
            "run_nonce_sha256": _sha256_bytes(nonce),
            "issued_at_unix": issued_at,
            "expires_at_unix": issued_at + MAX_GATE_TTL_SECONDS,
            "services_stopped": True,
            "secret_material_recorded": False,
        }
        gate = _hashed(unsigned_gate, "gate_sha256")
        validate_gate_for_owner(
            gate,
            expected_release_revision=base.revision,
            expected_owner_subject_sha256=(
                reconciliation_runtime.OWNER_SUBJECT_SHA256
            ),
            owner_public_key_ed25519_hex=(
                reconciliation_runtime.OWNER_PUBLIC_KEY_ED25519_HEX
            ),
            owner_public_fingerprint=(
                reconciliation_runtime.OWNER_PUBLIC_FINGERPRINT
            ),
            now_unix=issued_at,
        )
        return _ControlRuntimeContext(
            base=base,
            gate=gate,
            install_artifact=artifact,
        )
    except ControlBootstrapError:
        raise
    except BaseException as exc:
        raise ControlBootstrapError(
            "schema_reconciliation_control_pre_gate_invalid"
        ) from exc


def _revalidate_stopped(context: _ControlRuntimeContext, code: str) -> None:
    try:
        reconciliation_runtime._revalidate_stopped_boundary(
            context.base,
            code=code,
        )
    except BaseException as exc:
        raise ControlBootstrapError(code) from exc


def _open_control_session(
    context: _ControlRuntimeContext,
    credential: bytearray,
) -> Any:
    code = "schema_reconciliation_control_authentication_failed"
    try:
        if (
            not isinstance(credential, bytearray)
            or len(credential) != OPAQUE_CREDENTIAL_BYTES
            or _URLSAFE_CREDENTIAL.fullmatch(credential) is None
        ):
            _fail("schema_reconciliation_control_credential_invalid")
        with phase_b_runtime._secret_descriptor(credential) as descriptor:
            config = phase_b_runtime._database_config(
                context.gate["temporary_control_admin_username"],
                credential=CredentialSource(
                    fd=descriptor,
                    expected_uid=0,
                    expected_gid=0,
                    allowed_modes=frozenset({0o400}),
                ),
                application_name="muncho-schema-reconciliation-control-bootstrap",
            )
            session = context.base.dependencies.open_session(config)
        if (
            getattr(session, "username", None)
            != context.gate["temporary_control_admin_username"]
        ):
            try:
                session.close()
            finally:
                _fail(code)
        return session
    except ControlBootstrapError:
        raise
    except BaseException as exc:
        raise ControlBootstrapError(code) from exc
    finally:
        _zeroize(credential)


def _close_database_session(session: Any) -> None:
    try:
        session.close()
    except BaseException as exc:
        raise ControlBootstrapError(
            "schema_reconciliation_control_database_close_failed"
        ) from exc


def _execute_install_artifact(
    context: _ControlRuntimeContext,
    session: Any,
) -> None:
    code = "schema_reconciliation_control_install_failed"
    artifact = context.install_artifact
    try:
        if (
            artifact.name != "schema_reconciliation_control_install"
            or artifact.path.name != INSTALL_ARTIFACT_FILENAME
            or artifact.sha256
            != context.gate["control_install_artifact_sha256"]
            or _sha256_bytes(artifact.payload) != artifact.sha256
        ):
            _fail("schema_reconciliation_control_release_invalid")
        sql = artifact.payload.decode("utf-8", errors="strict")
        result = session.query(sql, maximum_rows=1)
    except ControlBootstrapError:
        raise
    except BaseException as exc:
        raise ControlBootstrapError(code) from exc
    if not isinstance(result, QueryResult) or result.command_tag.upper() != "COMMIT":
        _fail(code)


def _revalidate_install_authorization(
    context: _ControlRuntimeContext,
    gate: Mapping[str, Any],
    install_claim: Mapping[str, Any],
) -> None:
    code = "schema_reconciliation_control_install_authorization_expired"
    try:
        now_unix = context.base.dependencies.now()
        if type(now_unix) is not int or now_unix < 0:
            _fail(code)
        validate_gate_for_owner(
            gate,
            expected_release_revision=gate["release_revision"],
            expected_owner_subject_sha256=gate["owner_subject_sha256"],
            owner_public_key_ed25519_hex=gate[
                "owner_public_key_ed25519_hex"
            ],
            owner_public_fingerprint=gate["owner_public_fingerprint"],
            now_unix=now_unix,
        )
        _validate_install_claim(
            install_claim,
            gate=gate,
            now_unix=now_unix,
        )
    except BaseException as exc:
        if isinstance(exc, ControlBootstrapError) and str(exc) == code:
            raise
        raise ControlBootstrapError(code) from exc


def _runtime_install_callback(
    context: _ControlRuntimeContext,
    gate: Mapping[str, Any],
    install_claim: Mapping[str, Any],
    credential: bytearray,
) -> Mapping[str, Any]:
    if gate != context.gate or context.install_callback_used:
        _fail("schema_reconciliation_control_install_state_invalid")
    context.install_callback_used = True
    _revalidate_stopped(
        context,
        "schema_reconciliation_control_install_stopped_boundary_drifted",
    )
    session: Any | None = None
    primary: BaseException | None = None
    try:
        session = _open_control_session(context, credential)
        before = _observe_foundation(
            session,
            phase="before_install",
            observed_at_unix=context.base.dependencies.now,
        )
        _revalidate_stopped(
            context,
            "schema_reconciliation_control_install_stopped_boundary_drifted",
        )
        _revalidate_install_authorization(context, gate, install_claim)
        mutation_applied = before["state"] == "absent"
        if mutation_applied:
            _revalidate_stopped(
                context,
                "schema_reconciliation_control_install_stopped_boundary_drifted",
            )
            _revalidate_install_authorization(context, gate, install_claim)
            _execute_install_artifact(context, session)
        after = _observe_foundation(
            session,
            phase="after_install",
            observed_at_unix=context.base.dependencies.now,
        )
        if after["state"] != "exact_installed":
            _fail("schema_reconciliation_control_install_reattestation_failed")
    except BaseException as exc:
        primary = exc
    finally:
        _zeroize(credential)
        if session is not None:
            try:
                _close_database_session(session)
                context.temporary_database_session_closed = True
            except BaseException as close_error:
                if primary is None:
                    primary = close_error
    if primary is not None:
        raise primary
    if not context.temporary_database_session_closed:
        _fail("schema_reconciliation_control_database_close_failed")
    _revalidate_stopped(
        context,
        "schema_reconciliation_control_install_stopped_boundary_drifted",
    )
    observed_at = context.base.dependencies.now()
    unsigned = {
        "schema": INTERMEDIATE_SCHEMA,
        "ok": True,
        "state": "database_session_closed_awaiting_cloud_cleanup",
        "gate_sha256": gate["gate_sha256"],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "install_claim_sha256": install_claim["install_claim_sha256"],
        "control_install_artifact_sha256": gate[
            "control_install_artifact_sha256"
        ],
        "control_foundation_contract_sha256": gate[
            "control_foundation_contract_sha256"
        ],
        "initial_foundation_state": before["state"],
        "mutation_applied": mutation_applied,
        "before_observation": before,
        "before_observation_sha256": before["observation_sha256"],
        "after_observation": after,
        "after_observation_sha256": after["observation_sha256"],
        "database_capability_terminated": True,
        "database_session_closed": True,
        "services_stopped_sha256": gate["services_stopped_sha256"],
        "observed_at_unix": observed_at,
        "secret_material_recorded": False,
    }
    return _hashed(unsigned, "intermediate_sha256")


def _runtime_post_cleanup_callback(
    context: _ControlRuntimeContext,
    gate: Mapping[str, Any],
    install_claim: Mapping[str, Any],
    intermediate: Mapping[str, Any],
    cleanup: Mapping[str, Any],
) -> Mapping[str, Any]:
    absence = cleanup.get("cloud_sql_absence_receipt")
    if (
        gate != context.gate
        or context.post_cleanup_callback_used
        or not context.install_callback_used
        or not context.temporary_database_session_closed
        or intermediate.get("database_capability_terminated") is not True
        or not isinstance(absence, Mapping)
        or absence.get("temporary_control_admin_absent") is not True
    ):
        _fail("schema_reconciliation_control_cleanup_order_invalid")
    context.post_cleanup_callback_used = True
    _revalidate_stopped(
        context,
        "schema_reconciliation_control_post_cleanup_stopped_boundary_drifted",
    )
    session: Any | None = None
    primary: BaseException | None = None
    try:
        config = context.base.dependencies.writer_config()
        if config.user != foundation.SQL_USER:
            _fail("schema_reconciliation_control_writer_identity_invalid")
        session = context.base.dependencies.open_session(config)
        if getattr(session, "username", None) != foundation.SQL_USER:
            _fail("schema_reconciliation_control_writer_identity_invalid")
        observation = _observe_foundation(
            session,
            phase="post_cleanup",
            observed_at_unix=context.base.dependencies.now,
        )
    except BaseException as exc:
        primary = exc
    finally:
        if session is not None:
            try:
                _close_database_session(session)
            except BaseException as close_error:
                if primary is None:
                    primary = close_error
    if primary is not None:
        if isinstance(primary, ControlBootstrapError):
            raise primary
        raise ControlBootstrapError(
            "schema_reconciliation_control_post_cleanup_observation_failed"
        ) from primary
    _revalidate_stopped(
        context,
        "schema_reconciliation_control_post_cleanup_stopped_boundary_drifted",
    )
    completed_at = context.base.dependencies.now()
    unsigned = {
        "schema": TERMINAL_SCHEMA,
        "ok": True,
        "state": "control_installed_admin_absent_stopped",
        "gate_sha256": gate["gate_sha256"],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "install_claim_sha256": install_claim["install_claim_sha256"],
        "intermediate_sha256": intermediate["intermediate_sha256"],
        "cleanup_claim_sha256": cleanup["cleanup_claim_sha256"],
        "control_install_artifact_sha256": gate[
            "control_install_artifact_sha256"
        ],
        "control_retire_artifact_sha256": gate[
            "control_retire_artifact_sha256"
        ],
        "control_foundation_contract_sha256": gate[
            "control_foundation_contract_sha256"
        ],
        "post_cleanup_observation": observation,
        "post_cleanup_observation_sha256": observation[
            "observation_sha256"
        ],
        "temporary_control_admin_absent": True,
        "executor_memberships_absent": True,
        "executor_owns_zero_objects": True,
        "fixed_routines_exact": True,
        "services_stopped_sha256": gate["services_stopped_sha256"],
        "completed_at_unix": completed_at,
        "secret_material_recorded": False,
    }
    return _hashed(unsigned, "terminal_sha256")


def run(
    *,
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
    _dependencies: _ControlRuntimeDependencies | None = None,
) -> Mapping[str, Any]:
    """Run the fixed stopped G0/MCB1/I1/MCC1/T2 bootstrap dialogue."""

    dependencies = _dependencies or _ControlRuntimeDependencies()
    context = _prepare_control_runtime(dependencies)
    return dependencies.protocol_runner(
        context.gate,
        install_callback=lambda gate, claim, credential: (
            _runtime_install_callback(
                context,
                gate,
                claim,
                credential,
            )
        ),
        post_cleanup_callback=lambda gate, claim, intermediate, cleanup: (
            _runtime_post_cleanup_callback(
                context,
                gate,
                claim,
                intermediate,
                cleanup,
            )
        ),
        input_stream=input_stream,
        output_stream=output_stream,
        now=context.base.dependencies.now,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Root-only packaged entry point; the only command is ``install``."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments != ["install"]:
            _fail("schema_reconciliation_control_arguments_invalid")
        effective_user_id = getattr(os, "geteuid", None)
        if not callable(effective_user_id) or effective_user_id() != 0:
            _fail("schema_reconciliation_control_root_required")
        run()
    except BaseException:
        print("schema reconciliation control bootstrap failed closed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLEANUP_FRAME_SCHEMA",
    "CLEANUP_MAGIC",
    "CONTROL_BOOTSTRAP_CLEANUP_OWNER_SSHSIG_NAMESPACE",
    "CONTROL_BOOTSTRAP_INSTALL_OWNER_SSHSIG_NAMESPACE",
    "FAILURE_SCHEMA",
    "GATE_SCHEMA",
    "INSTALL_FRAME_SCHEMA",
    "INSTALL_MAGIC",
    "ControlBootstrapError",
    "build_owner_cleanup_claim",
    "build_owner_install_claim",
    "main",
    "owner_cleanup_signature_payload",
    "owner_install_signature_payload",
    "run",
    "run_protocol",
    "validate_failure_for_owner",
    "validate_gate_for_owner",
    "validate_intermediate_for_owner",
    "validate_terminal_for_owner",
]
