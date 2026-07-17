"""Strict v2 owner protocol for stopped schema reconciliation.

This module is a mechanical, dependency-injected boundary.  It does not
discover Cloud resources, create or delete users, or choose a semantic action.
The only choices are exact transitions over a sealed journal head and the two
reviewed schema-contract hashes.

Wire dialogue (remote output is canonical NDJSON):

* G0: secret-free stopped-release gate.
* A1: ``MSA2 || u32be(json_size) || canonical_json || 64 credential bytes``.
* P1: locked PostgreSQL preflight challenge.
* A2: ``MSP2 || u32be(json_size) || canonical_json``.
* I2: committed/re-attested database receipt, with the DB session closed.
* C3: ``MSC2 || u32be(json_size) || canonical_json || EOF``.
* T3: composite terminal binding the pre-delete full-truth DB attestation to
  the post-delete Cloud SQL absence proof, a distinct fresh-writer authority /
  contract / behavior proof, and fresh stopped-host observations.  The writer
  proof explicitly records that it cannot re-read the privileged full-data
  truth without broadening permanent authority.

After a complete G0, a failed transition emits one secret-free failure receipt
bound to the last successfully emitted transcript head, then exits 2.  A
failed or partial output write is never followed by another wire record.

The credential is validated only after A1's signature and Cloud authority
receipt have been accepted.  It is passed once to the preflight callback and
zeroized immediately when that callback returns.  The callback therefore has
to authenticate the bounded DB session before returning P1; the apply callback
continues through that already-authenticated dependency-injected session and
never receives credential material.

A2 never embeds a core authorization.  After its signature is verified, an
empty-journal admission derives ``SchemaReconciliationAuthorization`` and the
core owner-frame receipt from SHA-256 of the complete canonical signed A2
mapping.  Recovery A2 frames instead authorize the outer transition and carry
the sealed journal digests; their apply callback receives ``None`` for both
derived core values and must load the byte-identical durable preflight,
authorization, and owner-frame receipt already admitted by the core.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import re
import struct
import sys
import time
from typing import Any, BinaryIO, Callable, Mapping, Sequence

from gateway import canonical_writer_foundation_phase_b as phase_b
from gateway.canonical_writer_db import (
    managed_cloudsqladmin_hba_receipt_from_mapping,
)
from gateway.canonical_writer_schema_reconciliation import (
    CANONICAL_TRUTH_RECEIPT_SCHEMA,
    RECONCILIATION_AUTHORIZATION_SCHEMA,
    RECONCILIATION_OWNER_A2_FRAME_SCHEMA,
    RECONCILIATION_OWNER_FRAME_RECEIPT_SCHEMA,
    RECONCILIATION_OWNER_SIGNATURE_NAMESPACE,
    RECONCILIATION_PREFLIGHT_SCHEMA,
    RECONCILIATION_RECEIPT_SCHEMA,
    CanonicalTruthReceipt,
    SchemaReconciliationAuthorization,
    SchemaReconciliationError,
)
from gateway.canonical_writer_schema_reconciliation_db import (
    parse_post_delete_terminal_receipt,
)


EXECUTOR_PREFLIGHT_OWNER_SSHSIG_NAMESPACE = (
    "muncho-canonical-writer-schema-reconciliation-executor-preflight-owner-v3"
)
PREFLIGHT_AUTHORIZATION_OWNER_SSHSIG_NAMESPACE = (
    RECONCILIATION_OWNER_SIGNATURE_NAMESPACE
)
EXECUTOR_CLEANUP_OWNER_SSHSIG_NAMESPACE = (
    "muncho-canonical-writer-schema-reconciliation-executor-cleanup-owner-v3"
)

EXECUTOR_PREFLIGHT_MAGIC = b"MSA2"
PREFLIGHT_AUTHORIZATION_MAGIC = b"MSP2"
EXECUTOR_CLEANUP_MAGIC = b"MSC2"
EXECUTOR_PREFLIGHT_FRAME_SCHEMA = "MSA2-u32be-canonical-json-64byte-opaque.v1"
PREFLIGHT_AUTHORIZATION_FRAME_SCHEMA = RECONCILIATION_OWNER_A2_FRAME_SCHEMA
EXECUTOR_CLEANUP_FRAME_SCHEMA = "MSC2-u32be-canonical-json-no-secret-eof.v1"

# Compatibility aliases for already-packaged v2 launcher imports.  The active
# protocol values and validators are the executor-v3 contracts above.
ADMIN_PREFLIGHT_OWNER_SSHSIG_NAMESPACE = (
    EXECUTOR_PREFLIGHT_OWNER_SSHSIG_NAMESPACE
)
ADMIN_CLEANUP_OWNER_SSHSIG_NAMESPACE = EXECUTOR_CLEANUP_OWNER_SSHSIG_NAMESPACE
ADMIN_PREFLIGHT_MAGIC = EXECUTOR_PREFLIGHT_MAGIC
ADMIN_CLEANUP_MAGIC = EXECUTOR_CLEANUP_MAGIC
ADMIN_PREFLIGHT_FRAME_SCHEMA = EXECUTOR_PREFLIGHT_FRAME_SCHEMA
ADMIN_CLEANUP_FRAME_SCHEMA = EXECUTOR_CLEANUP_FRAME_SCHEMA

GATE_SCHEMA = "muncho-canonical-writer-schema-reconciliation-gate.v3"
JOURNAL_HEAD_SCHEMA = "muncho-canonical-writer-schema-reconciliation-journal-head.v1"
OWNER_ADMIN_PREFLIGHT_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-owner-executor-preflight-authority.v3"
)
OWNER_EXECUTOR_PREFLIGHT_SCHEMA = OWNER_ADMIN_PREFLIGHT_SCHEMA
PREFLIGHT_CHALLENGE_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-locked-preflight-challenge.v2"
)
TEMPORARY_EXECUTOR_BOUNDARY_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-temporary-executor-boundary.v3"
)
OWNER_PREFLIGHT_AUTHORIZATION_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-owner-preflight-authorization.v2"
)
DATABASE_COMMIT_ATTESTATION_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-database-commit-attestation.v3"
)
DATABASE_INTERMEDIATE_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-database-terminal-awaiting-executor-cleanup.v3"
)
OWNER_ADMIN_CLEANUP_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-owner-executor-cleanup.v3"
)
OWNER_EXECUTOR_CLEANUP_SCHEMA = OWNER_ADMIN_CLEANUP_SCHEMA
POST_CLEANUP_OBSERVATION_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-post-cleanup-observation.v4"
)
TERMINAL_SCHEMA = "muncho-canonical-writer-schema-reconciliation-terminal.v5"
REMOTE_FAILURE_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-remote-failure.v1"
)

CLOUD_EXECUTOR_AUTHORITY_SCHEMA = "muncho-cloud-sql-temporary-executor-authority.v3"
CLOUD_ADMIN_AUTHORITY_SCHEMA = CLOUD_EXECUTOR_AUTHORITY_SCHEMA
SCHEMA_RECONCILIATION_DATABASE_ROLES = (
    "canonical_brain_schema_reconciler",
)
CLOUD_EXECUTOR_ABSENCE_SCHEMA = "muncho-cloud-sql-executor-absence-evidence.v2"
CLOUD_ADMIN_ABSENCE_SCHEMA = CLOUD_EXECUTOR_ABSENCE_SCHEMA

EXPECTED_PROJECT = phase_b.PROJECT
EXPECTED_SQL_INSTANCE = phase_b.SQL_INSTANCE
EXPECTED_DATABASE = phase_b.SQL_DATABASE
EXPECTED_POSTGRESQL_MAJOR = 18
EXPECTED_PYTHON_VERSION = "3.11.15"

OPAQUE_CREDENTIAL_BYTES = 64
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_OWNER_FRAME_TTL_SECONDS = 300
MAX_GATE_TTL_SECONDS = 1_800
MIN_CLOUD_ABSENCE_QUIET_WINDOW_SECONDS = 180

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_ADMIN = re.compile(r"^muncho_canary_reconciler_[0-9a-f]{16}$")
_FINGERPRINT = re.compile(r"^SHA256:[A-Za-z0-9+/]{43}$")
_SAFE_OPERATION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~:/+\-]{0,511}$")

_JOURNAL_HEAD_FIELDS = frozenset({
    "schema",
    "state",
    "authorized_intent_sha256",
    "terminal_receipt_sha256",
    "head_sha256",
})
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
    "base_artifact_sha256",
    "target_asset_sha256",
    "expected_old_contract_sha256",
    "target_contract_sha256",
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
    "temporary_executor_username",
    "temporary_executor_username_sha256",
    "owner_subject_sha256",
    "owner_public_key_ed25519_hex",
    "owner_key_id",
    "owner_public_fingerprint",
    "journal_head",
    "run_nonce_sha256",
    "issued_at_unix",
    "expires_at_unix",
    "temporary_executor_required",
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
    "database_roles",
    "cloudsqlsuperuser_absent",
    "resource_etag_sha256",
    "owner_subject_sha256",
    "mutation_context_sha256",
    "baseline_operation_names",
    "baseline_user_operations",
    "authority_operation",
    "receipt_sha256",
})
_ADMIN_PREFLIGHT_UNSIGNED_FIELDS = frozenset({
    "schema",
    "frame_schema",
    "action",
    "approved",
    "gate_sha256",
    "release_revision",
    "plan_sha256",
    "temporary_executor_username_sha256",
    "owner_subject_sha256",
    "owner_key_id",
    "cloud_sql_authority_receipt",
    "cloud_sql_authority_receipt_sha256",
    "credential_present",
    "credential_length",
    "issued_at_unix",
    "expires_at_unix",
    "nonce_sha256",
    "secret_material_recorded",
})
_ADMIN_PREFLIGHT_FIELDS = frozenset({
    *_ADMIN_PREFLIGHT_UNSIGNED_FIELDS,
    "authority_claim_sha256",
    "signature_sshsig",
})
_BRIDGE_FIELDS = frozenset({
    "schema",
    "transaction_isolation",
    "database_roles",
    "provider_membership_count",
    "admin_option",
    "inherit_option",
    "set_option",
    "cloudsqlsuperuser_absent",
    "canonical_truth_share_lock",
    "caller_has_no_owner_membership_or_set_path",
    "owner_writer_system_roles_unreachable",
    "executor_owns_zero_objects_clusterwide",
    "executor_cross_database_authority_hba_bounded",
    "connectable_database_inventory_exact",
    "connectable_non_template_database_inventory_exact",
    "connectable_template_authority_absent",
    "prepared_transactions_disabled_and_empty",
    "executor_managed_hba_receipt_sha256",
    "latent_provider_exception_databases",
    "latent_provider_exception_hba_receipt_sha256s",
    "roles_and_fence_recheck_required_before_authorization",
    "control_foundation_contract_sha256",
    "current_user_observed_as_temporary_login_before_and_after_locked_collection",
    "exact_provider_memberships_observed_before_and_after_locked_collection",
    "contract_collected_while_truth_lock_held",
    "canonical_truth_collected_while_truth_lock_held",
    "temporary_login_owned_objects",
    "temporary_login_has_zero_shared_dependencies",
    "memberships_observed_present_in_committed_preflight_snapshot_and_cleanup_required",
    "transaction_committed",
    "observed_at_unix",
    "secret_material_recorded",
    "receipt_sha256",
})
_PREFLIGHT_COLLECTION_FIELDS = frozenset({
    "database_identity_sha256",
    "tls_peer_certificate_sha256",
    "managed_hba_receipt_sha256",
    "executor_managed_hba_receipt",
    "executor_managed_hba_receipt_sha256",
    "postgresql_major",
    "temporary_executor_boundary_receipt",
    "preflight",
    "canonical_truth_receipt",
    "observed_at_unix",
})
_CORE_PREFLIGHT_FIELDS = frozenset({
    "schema",
    "ok",
    "release_revision",
    "plan_sha256",
    "base_artifact_sha256",
    "target_asset_sha256",
    "postgresql_major",
    "control_install_artifact_sha256",
    "control_retire_artifact_sha256",
    "control_foundation_contract_sha256",
    "observed_contract_sha256",
    "truth_receipt_sha256",
    "expected_old_contract_sha256",
    "target_contract_sha256",
    "state",
    "mutation_required",
    "observed_at_unix",
    "preflight_sha256",
})
_PREFLIGHT_CHALLENGE_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "gate_sha256",
    "authority_claim_sha256",
    "cloud_sql_authority_receipt_sha256",
    "release_revision",
    "plan_sha256",
    "journal_head_sha256",
    "database_identity_sha256",
    "tls_peer_certificate_sha256",
    "managed_hba_receipt_sha256",
    "executor_managed_hba_receipt",
    "executor_managed_hba_receipt_sha256",
    "postgresql_major",
    "temporary_executor_boundary_receipt",
    "preflight",
    "canonical_truth_receipt",
    "mutation_required",
    "issued_at_unix",
    "expires_at_unix",
    "secret_material_recorded",
    "preflight_challenge_sha256",
})
_PREFLIGHT_AUTHORIZATION_UNSIGNED_FIELDS = frozenset({
    "schema",
    "frame_schema",
    "action",
    "approved",
    "gate_sha256",
    "authority_claim_sha256",
    "preflight_challenge_sha256",
    "post_hba_temporary_executor_authority_receipt",
    "post_hba_temporary_executor_authority_receipt_sha256",
    "release_revision",
    "plan_sha256",
    "journal_head_sha256",
    "execution_mode",
    "preflight_sha256",
    "preflight_state",
    "observed_contract_sha256",
    "truth_receipt_sha256",
    "owner_subject_sha256",
    "owner_key_id",
    "issued_at_unix",
    "expires_at_unix",
    "nonce_sha256",
    "stored_authorized_intent_sha256",
    "stored_terminal_receipt_sha256",
    "secret_material_recorded",
})
_PREFLIGHT_AUTHORIZATION_FIELDS = frozenset({
    *_PREFLIGHT_AUTHORIZATION_UNSIGNED_FIELDS,
    "preflight_authorization_claim_sha256",
    "signature_sshsig",
})
_CORE_TERMINAL_FIELDS = frozenset({
    "schema",
    "ok",
    "release_revision",
    "plan_sha256",
    "base_artifact_sha256",
    "target_asset_sha256",
    "postgresql_major",
    "control_install_artifact_sha256",
    "control_retire_artifact_sha256",
    "control_foundation_contract_sha256",
    "expected_old_contract_sha256",
    "target_contract_sha256",
    "initial_contract_sha256",
    "final_contract_sha256",
    "initial_canonical_truth",
    "final_canonical_truth",
    "authorization_sha256",
    "preflight_sha256",
    "owner_frame_receipt_sha256",
    "truth_receipt_sha256",
    "authorized_intent_sha256",
    "mode",
    "mutation_applied",
    "completed_at_unix",
    "receipt_sha256",
})
_CORE_OWNER_FRAME_RECEIPT_FIELDS = frozenset({
    "schema",
    "frame_schema",
    "action",
    "approved",
    "signed_frame_sha256",
    "signature_sshsig_sha256",
    "signature_namespace",
    "signature_verified",
    "release_revision",
    "plan_sha256",
    "preflight_sha256",
    "observed_contract_sha256",
    "truth_receipt_sha256",
    "authorization_sha256",
    "owner_subject_sha256",
    "owner_key_id",
    "issued_at_unix",
    "expires_at_unix",
    "nonce",
    "receipt_sha256",
})
_DATABASE_ATTESTATION_FIELDS = frozenset({
    "schema",
    "ok",
    "database_identity_sha256",
    "tls_peer_certificate_sha256",
    "postgresql_major",
    "observed_contract_sha256",
    "canonical_truth_receipt",
    "transaction_committed",
    "temporary_login_owns_zero_objects",
    "fixed_control_routines_re_attested_before_commit",
    "inert_executor_membership_present",
    "cloud_user_cleanup_required",
    "database_session_closed",
    "re_attested_before_temporary_executor_delete",
    "observed_at_unix",
    "secret_material_recorded",
    "attestation_sha256",
})
_APPLY_RESULT_FIELDS = frozenset({
    "authorized_intent_sha256",
    "core_terminal_receipt",
    "database_commit_attestation",
})
_DATABASE_INTERMEDIATE_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "gate_sha256",
    "authority_claim_sha256",
    "preflight_challenge_sha256",
    "preflight_authorization_claim_sha256",
    "post_hba_temporary_executor_authority_receipt_sha256",
    "release_revision",
    "plan_sha256",
    "execution_mode",
    "authorized_intent_sha256",
    "core_terminal_receipt",
    "core_terminal_receipt_sha256",
    "database_commit_attestation",
    "database_commit_attestation_sha256",
    "initial_canonical_truth",
    "final_canonical_truth",
    "temporary_executor_cleanup_required",
    "database_session_closed",
    "applied_at_unix",
    "secret_material_recorded",
    "database_intermediate_sha256",
})
_CLOUD_ABSENCE_FIELDS = frozenset({
    "schema",
    "temporary_executor_absent",
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
_ADMIN_CLEANUP_UNSIGNED_FIELDS = frozenset({
    "schema",
    "frame_schema",
    "action",
    "approved",
    "gate_sha256",
    "authority_claim_sha256",
    "preflight_challenge_sha256",
    "preflight_authorization_claim_sha256",
    "database_intermediate_sha256",
    "release_revision",
    "plan_sha256",
    "temporary_executor_username_sha256",
    "owner_subject_sha256",
    "owner_key_id",
    "cloud_sql_absence_receipt",
    "cloud_sql_absence_receipt_sha256",
    "issued_at_unix",
    "expires_at_unix",
    "nonce_sha256",
    "secret_material_recorded",
})
_ADMIN_CLEANUP_FIELDS = frozenset({
    *_ADMIN_CLEANUP_UNSIGNED_FIELDS,
    "cleanup_claim_sha256",
    "signature_sshsig",
})
_POST_CLEANUP_FIELDS = frozenset({
    "schema",
    "release_manifest_sha256",
    "stopped_release_receipt_file_sha256",
    "stopped_release_receipt_sha256",
    "release_artifact_sha256",
    "python_version",
    "interpreter_sha256",
    "activation_inventory_sha256",
    "host_identity_sha256",
    "services_stopped_sha256",
    "host_observation_receipt_sha256",
    "services_observation_receipt_sha256",
    "fresh_managed_hba_receipt",
    "fresh_managed_hba_receipt_sha256",
    "post_delete_terminal_receipt",
    "post_delete_terminal_receipt_sha256",
    "observed_at_unix",
    "secret_material_recorded",
    "observation_sha256",
})
_TERMINAL_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "gate_sha256",
    "authority_claim_sha256",
    "preflight_challenge_sha256",
    "preflight_authorization_claim_sha256",
    "post_hba_temporary_executor_authority_receipt_sha256",
    "database_intermediate_sha256",
    "cleanup_claim_sha256",
    "release_revision",
    "plan_sha256",
    "release_manifest_sha256",
    "stopped_release_receipt_file_sha256",
    "stopped_release_receipt_sha256",
    "release_artifact_sha256",
    "python_version",
    "interpreter_sha256",
    "activation_inventory_sha256",
    "target_asset_sha256",
    "target_contract_sha256",
    "authorized_intent_sha256",
    "core_terminal_receipt_sha256",
    "database_commit_attestation_sha256",
    "final_canonical_truth",
    "temporary_executor_absence_receipt_sha256",
    "fresh_managed_hba_receipt_sha256",
    "post_delete_terminal_receipt_sha256",
    "post_cleanup_observation",
    "host_identity_sha256",
    "services_stopped_sha256",
    "host_observation_receipt_sha256",
    "services_observation_receipt_sha256",
    "post_cleanup_observation_sha256",
    "owner_subject_sha256",
    "owner_key_id",
    "database_re_attested_before_temporary_executor_delete",
    "fresh_writer_post_delete_authority_contract_and_behavior_proven",
    "post_delete_canonical_truth_observed",
    "secret_material_recorded",
    "completed_at_unix",
    "terminal_sha256",
})
_REMOTE_FAILURE_FIELDS = frozenset({
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
_REMOTE_FAILURE_WIRE_STAGES = frozenset({
    "a1_to_p1",
    "a2_to_i2",
    "c3_to_t3",
})
_STABLE_REMOTE_ERROR = re.compile(r"^[a-z][a-z0-9_]{2,95}$")
_GENERIC_REMOTE_ERROR = "schema_reconciliation_remote_failed"

_TRANSITIONS: Mapping[tuple[str, str], str] = {
    ("empty", "exact_old_missing_one_helper"): "reconcile_missing_helper",
    ("empty", "exact_target"): "adopt_existing_target",
    ("authorized_intent", "exact_old_missing_one_helper"): "resume_durable_intent",
    ("authorized_intent", "exact_target"): "terminalize_durable_intent",
    ("terminal", "exact_target"): "reattest_terminal",
}


class SchemaReconciliationBootstrapError(RuntimeError):
    """Stable, secret-free protocol failure."""

    def __init__(self, code: str) -> None:
        if (
            not isinstance(code, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{2,95}", code) is None
        ):
            code = "schema_reconciliation_bootstrap_failed"
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise SchemaReconciliationBootstrapError(code)


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SchemaReconciliationBootstrapError(
            "schema_reconciliation_value_not_canonical"
        ) from exc
    if not encoded or len(encoded) > MAX_JSON_BYTES:
        _fail("schema_reconciliation_value_not_canonical")
    return encoded


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("schema_reconciliation_frame_json_invalid")
        result[key] = value
    return result


def _decode_canonical_mapping(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_JSON_BYTES:
        _fail("schema_reconciliation_frame_json_invalid")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except SchemaReconciliationBootstrapError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SchemaReconciliationBootstrapError(
            "schema_reconciliation_frame_json_invalid"
        ) from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        _fail("schema_reconciliation_frame_json_invalid")
    return value


def _strict_mapping(
    value: Any,
    fields: frozenset[str],
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(code)
    return dict(value)


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(code)
    return value


def _hashed_mapping(
    value: Any,
    *,
    fields: frozenset[str],
    digest_field: str,
    code: str,
) -> dict[str, Any]:
    raw = _strict_mapping(value, fields, code)
    digest = raw.get(digest_field)
    unsigned = {key: item for key, item in raw.items() if key != digest_field}
    if not isinstance(digest, str) or digest != _sha256_json(unsigned):
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
    if raw.get(digest_field) != _sha256_json(unsigned):
        _fail(code)
    return raw


def _require_secret_free(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail("schema_reconciliation_secret_like_field")
            folded = key.casefold()
            if folded == "secret_material_recorded":
                if item is not False:
                    _fail("schema_reconciliation_secret_like_field")
            elif folded == "credential_present":
                if item is not True:
                    _fail("schema_reconciliation_secret_like_field")
            elif folded == "credential_length":
                if item != OPAQUE_CREDENTIAL_BYTES:
                    _fail("schema_reconciliation_secret_like_field")
            elif any(
                marker in folded
                for marker in (
                    "password",
                    "credential",
                    "private_key",
                    "secret",
                    "token",
                    "verifier",
                )
            ):
                _fail("schema_reconciliation_secret_like_field")
            _require_secret_free(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_secret_free(item)
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        _fail("schema_reconciliation_secret_like_field")


def _zeroize(value: bytearray | None) -> None:
    if value is None:
        return
    try:
        value[:] = b"\x00" * len(value)
    except (BufferError, TypeError, ValueError):
        pass


def _owner_fingerprint(public_key_ed25519_hex: str) -> tuple[str, str]:
    if (
        not isinstance(public_key_ed25519_hex, str)
        or _SHA256.fullmatch(public_key_ed25519_hex) is None
    ):
        _fail("schema_reconciliation_owner_key_invalid")
    try:
        public = bytes.fromhex(public_key_ed25519_hex)
    except ValueError as exc:
        raise SchemaReconciliationBootstrapError(
            "schema_reconciliation_owner_key_invalid"
        ) from exc
    if len(public) != 32:
        _fail("schema_reconciliation_owner_key_invalid")
    blob = (
        struct.pack(">I", len(b"ssh-ed25519"))
        + b"ssh-ed25519"
        + struct.pack(">I", len(public))
        + public
    )
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode(
        "ascii"
    ).rstrip("=")
    return _sha256_bytes(public), fingerprint


def _validate_ttl(
    value: Mapping[str, Any],
    *,
    now_unix: int,
    code: str,
    maximum_seconds: int = MAX_OWNER_FRAME_TTL_SECONDS,
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
        or type(maximum_seconds) is not int
        or not 1 <= expires - issued <= maximum_seconds
        or (not_before is not None and issued < not_before)
        or (not_after is not None and expires > not_after)
    ):
        _fail(code)


def _validate_journal_head(value: Any) -> Mapping[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_JOURNAL_HEAD_FIELDS,
        digest_field="head_sha256",
        code="schema_reconciliation_journal_head_invalid",
    )
    state = raw.get("state")
    intent = raw.get("authorized_intent_sha256")
    terminal = raw.get("terminal_receipt_sha256")
    if (
        raw.get("schema") != JOURNAL_HEAD_SCHEMA
        or state not in {"empty", "authorized_intent", "terminal"}
        or (
            state == "empty"
            and (intent is not None or terminal is not None)
        )
        or (
            state == "authorized_intent"
            and (
                not isinstance(intent, str)
                or _SHA256.fullmatch(intent) is None
                or terminal is not None
            )
        )
        or (
            state == "terminal"
            and (
                not isinstance(intent, str)
                or _SHA256.fullmatch(intent) is None
                or not isinstance(terminal, str)
                or _SHA256.fullmatch(terminal) is None
            )
        )
    ):
        _fail("schema_reconciliation_journal_head_invalid")
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def validate_gate(
    value: Mapping[str, Any],
    *,
    owner_public_key_ed25519_hex: str,
    owner_public_fingerprint: str,
    now_unix: int,
) -> Mapping[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_GATE_FIELDS,
        digest_field="gate_sha256",
        code="schema_reconciliation_gate_invalid",
    )
    key_id, fingerprint = _owner_fingerprint(owner_public_key_ed25519_hex)
    journal = _validate_journal_head(raw.get("journal_head"))
    if (
        raw.get("schema") != GATE_SCHEMA
        or raw.get("ok") is not True
        or raw.get("state") != "stopped_release_executor_preflight_ready"
        or not isinstance(raw.get("release_revision"), str)
        or _REVISION.fullmatch(raw["release_revision"]) is None
        or raw.get("project") != EXPECTED_PROJECT
        or raw.get("sql_instance") != EXPECTED_SQL_INSTANCE
        or raw.get("database") != EXPECTED_DATABASE
        or raw.get("postgresql_major") != EXPECTED_POSTGRESQL_MAJOR
        or raw.get("python_version") != EXPECTED_PYTHON_VERSION
        or not isinstance(raw.get("tls_server_name"), str)
        or not raw["tls_server_name"]
        or len(raw["tls_server_name"]) > 253
        or not isinstance(raw.get("temporary_executor_username"), str)
        or _ADMIN.fullmatch(raw["temporary_executor_username"]) is None
        or raw.get("temporary_executor_username_sha256")
        != _sha256_bytes(
            raw["temporary_executor_username"].encode("ascii")
        )
        or raw.get("owner_public_key_ed25519_hex")
        != owner_public_key_ed25519_hex
        or raw.get("owner_key_id") != key_id
        or raw.get("owner_public_fingerprint") != owner_public_fingerprint
        or fingerprint != owner_public_fingerprint
        or raw.get("temporary_executor_required") is not True
        or raw.get("secret_material_recorded") is not False
        or raw.get("journal_head") != journal
        or type(raw.get("advisory_lock_key")) is not int
    ):
        _fail("schema_reconciliation_gate_invalid")
    for name in (
        "release_manifest_sha256",
        "stopped_release_receipt_file_sha256",
        "stopped_release_receipt_sha256",
        "release_artifact_sha256",
        "interpreter_sha256",
        "activation_inventory_sha256",
        "plan_sha256",
        "base_artifact_sha256",
        "target_asset_sha256",
        "expected_old_contract_sha256",
        "target_contract_sha256",
        "control_install_artifact_sha256",
        "control_retire_artifact_sha256",
        "control_foundation_contract_sha256",
        "host_identity_sha256",
        "services_stopped_sha256",
        "ca_file_sha256",
        "temporary_executor_username_sha256",
        "owner_subject_sha256",
        "owner_public_key_ed25519_hex",
        "owner_key_id",
        "run_nonce_sha256",
    ):
        _digest(raw.get(name), "schema_reconciliation_gate_invalid")
    _validate_ttl(
        raw,
        now_unix=now_unix,
        code="schema_reconciliation_gate_expired",
        maximum_seconds=MAX_GATE_TTL_SECONDS,
    )
    _require_secret_free(raw)
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


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


def _validate_cloud_authority(
    value: Any,
    *,
    gate: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_CLOUD_AUTHORITY_FIELDS,
        digest_field="receipt_sha256",
        code="schema_reconciliation_cloud_authority_invalid",
    )
    baseline_names = _operation_names(
        raw.get("baseline_operation_names"),
        "schema_reconciliation_cloud_authority_invalid",
    )
    baseline_rows = _operation_rows(
        raw.get("baseline_user_operations"),
        "schema_reconciliation_cloud_authority_invalid",
    )
    authority = _operation_row(
        raw.get("authority_operation"),
        "schema_reconciliation_cloud_authority_invalid",
    )
    if (
        raw.get("schema") != CLOUD_ADMIN_AUTHORITY_SCHEMA
        or raw.get("project") != gate["project"]
        or raw.get("instance") != gate["sql_instance"]
        or raw.get("username_sha256")
        != gate["temporary_executor_username_sha256"]
        or raw.get("host") != ""
        or raw.get("type") != "BUILT_IN"
        or raw.get("user_present") is not True
        or raw.get("database_roles")
        != list(SCHEMA_RECONCILIATION_DATABASE_ROLES)
        or raw.get("cloudsqlsuperuser_absent") is not True
        or not _SHA256.fullmatch(str(raw.get("resource_etag_sha256", "")))
        or raw.get("owner_subject_sha256") != gate["owner_subject_sha256"]
        or raw.get("mutation_context_sha256") != gate["gate_sha256"]
        or any(row[0] not in baseline_names for row in baseline_rows)
        or authority[0] in baseline_names
        or authority[1] not in {"CREATE_USER", "UPDATE_USER"}
        or authority[2] != "DONE"
        or authority[3] != gate["owner_subject_sha256"]
        or authority[4] is not True
    ):
        _fail("schema_reconciliation_cloud_authority_invalid")
    _require_secret_free(raw)
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def admin_preflight_signature_payload(value: Mapping[str, Any]) -> bytes:
    raw = _strict_mapping(
        value,
        _ADMIN_PREFLIGHT_FIELDS,
        "schema_reconciliation_admin_preflight_invalid",
    )
    return _canonical_bytes({
        key: item for key, item in raw.items() if key != "signature_sshsig"
    })


executor_preflight_signature_payload = admin_preflight_signature_payload


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
        raise SchemaReconciliationBootstrapError(code) from exc


def _validate_admin_preflight(
    value: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    raw = _signed_mapping(
        value,
        fields=_ADMIN_PREFLIGHT_FIELDS,
        unsigned_fields=_ADMIN_PREFLIGHT_UNSIGNED_FIELDS,
        digest_field="authority_claim_sha256",
        code="schema_reconciliation_admin_preflight_invalid",
    )
    authority = _validate_cloud_authority(
        raw.get("cloud_sql_authority_receipt"),
        gate=gate,
    )
    if (
        raw.get("schema") != OWNER_ADMIN_PREFLIGHT_SCHEMA
        or raw.get("frame_schema") != ADMIN_PREFLIGHT_FRAME_SCHEMA
        or raw.get("action")
        != "authorize_temporary_executor_locked_preflight"
        or raw.get("approved") is not True
        or raw.get("gate_sha256") != gate["gate_sha256"]
        or raw.get("release_revision") != gate["release_revision"]
        or raw.get("plan_sha256") != gate["plan_sha256"]
        or raw.get("temporary_executor_username_sha256")
        != gate["temporary_executor_username_sha256"]
        or raw.get("owner_subject_sha256") != gate["owner_subject_sha256"]
        or raw.get("owner_key_id") != gate["owner_key_id"]
        or raw.get("cloud_sql_authority_receipt") != authority
        or raw.get("cloud_sql_authority_receipt_sha256")
        != authority["receipt_sha256"]
        or raw.get("credential_present") is not True
        or raw.get("credential_length") != OPAQUE_CREDENTIAL_BYTES
        or raw.get("secret_material_recorded") is not False
    ):
        _fail("schema_reconciliation_admin_preflight_invalid")
    _digest(raw.get("nonce_sha256"), "schema_reconciliation_admin_preflight_invalid")
    _validate_ttl(
        raw,
        now_unix=now_unix,
        code="schema_reconciliation_admin_preflight_expired",
        not_before=int(gate["issued_at_unix"]),
        not_after=int(gate["expires_at_unix"]),
    )
    _verify_signature(
        raw.get("signature_sshsig"),
        message=admin_preflight_signature_payload(raw),
        public_key_ed25519_hex=gate["owner_public_key_ed25519_hex"],
        namespace=EXECUTOR_PREFLIGHT_OWNER_SSHSIG_NAMESPACE,
        code="schema_reconciliation_admin_preflight_signature_invalid",
    )
    _require_secret_free(raw)
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def _canonical_truth(value: Any, code: str) -> CanonicalTruthReceipt:
    try:
        return CanonicalTruthReceipt.from_mapping(value)
    except (SchemaReconciliationError, TypeError, ValueError) as exc:
        raise SchemaReconciliationBootstrapError(code) from exc


def _validate_core_preflight(
    value: Any,
    *,
    gate: Mapping[str, Any],
    truth: CanonicalTruthReceipt,
) -> Mapping[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_CORE_PREFLIGHT_FIELDS,
        digest_field="preflight_sha256",
        code="schema_reconciliation_locked_preflight_invalid",
    )
    state = raw.get("state")
    expected_contract = (
        gate["expected_old_contract_sha256"]
        if state == "exact_old_missing_one_helper"
        else gate["target_contract_sha256"]
        if state == "exact_target"
        else None
    )
    if (
        raw.get("schema") != RECONCILIATION_PREFLIGHT_SCHEMA
        or raw.get("ok") is not True
        or raw.get("release_revision") != gate["release_revision"]
        or raw.get("plan_sha256") != gate["plan_sha256"]
        or raw.get("base_artifact_sha256") != gate["base_artifact_sha256"]
        or raw.get("target_asset_sha256") != gate["target_asset_sha256"]
        or raw.get("postgresql_major") != gate["postgresql_major"]
        or any(
            raw.get(name) != gate[name]
            for name in (
                "control_install_artifact_sha256",
                "control_retire_artifact_sha256",
                "control_foundation_contract_sha256",
            )
        )
        or raw.get("observed_contract_sha256") != expected_contract
        or raw.get("truth_receipt_sha256") != truth.sha256
        or raw.get("expected_old_contract_sha256")
        != gate["expected_old_contract_sha256"]
        or raw.get("target_contract_sha256") != gate["target_contract_sha256"]
        or raw.get("mutation_required")
        is not (state == "exact_old_missing_one_helper")
        or type(raw.get("observed_at_unix")) is not int
        or raw["observed_at_unix"] < 0
    ):
        _fail("schema_reconciliation_locked_preflight_invalid")
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def _validate_executor_boundary(
    value: Any,
    *,
    observed_at_unix: int,
) -> Mapping[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_BRIDGE_FIELDS,
        digest_field="receipt_sha256",
        code="schema_reconciliation_temporary_executor_boundary_invalid",
    )
    if (
        raw.get("schema") != TEMPORARY_EXECUTOR_BOUNDARY_SCHEMA
        or raw.get("transaction_isolation") != "SERIALIZABLE"
        or raw.get("database_roles")
        != list(SCHEMA_RECONCILIATION_DATABASE_ROLES)
        or raw.get("provider_membership_count") != 1
        or raw.get("admin_option") is not False
        or raw.get("inherit_option") is not True
        or raw.get("set_option") is not True
        or raw.get("cloudsqlsuperuser_absent") is not True
        or raw.get("canonical_truth_share_lock") is not True
        or raw.get("caller_has_no_owner_membership_or_set_path") is not True
        or raw.get("owner_writer_system_roles_unreachable") is not True
        or raw.get("executor_owns_zero_objects_clusterwide") is not True
        or raw.get("executor_cross_database_authority_hba_bounded") is not True
        or raw.get("connectable_database_inventory_exact") is not True
        or raw.get("connectable_non_template_database_inventory_exact")
        is not True
        or raw.get("connectable_template_authority_absent") is not True
        or raw.get("prepared_transactions_disabled_and_empty") is not True
        or not isinstance(
            raw.get("executor_managed_hba_receipt_sha256"), str
        )
        or _SHA256.fullmatch(
            raw["executor_managed_hba_receipt_sha256"]
        ) is None
        or raw.get("latent_provider_exception_databases")
        != ["cloudsqladmin"]
        or raw.get("latent_provider_exception_hba_receipt_sha256s")
        != [raw["executor_managed_hba_receipt_sha256"]]
        or raw.get("roles_and_fence_recheck_required_before_authorization")
        is not True
        or not isinstance(raw.get("control_foundation_contract_sha256"), str)
        or _SHA256.fullmatch(
            raw["control_foundation_contract_sha256"]
        ) is None
        or raw.get(
            "current_user_observed_as_temporary_login_before_and_after_locked_collection"
        )
        is not True
        or raw.get(
            "exact_provider_memberships_observed_before_and_after_locked_collection"
        )
        is not True
        or raw.get("contract_collected_while_truth_lock_held") is not True
        or raw.get("canonical_truth_collected_while_truth_lock_held") is not True
        or raw.get("temporary_login_owned_objects") is not False
        or raw.get("temporary_login_has_zero_shared_dependencies") is not True
        or raw.get(
            "memberships_observed_present_in_committed_preflight_snapshot_and_cleanup_required"
        )
        is not True
        or raw.get("transaction_committed") is not True
        or raw.get("observed_at_unix") != observed_at_unix
        or raw.get("secret_material_recorded") is not False
    ):
        _fail("schema_reconciliation_temporary_executor_boundary_invalid")
    _require_secret_free(raw)
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def _build_preflight_challenge(
    *,
    gate: Mapping[str, Any],
    admin_preflight: Mapping[str, Any],
    collection: Mapping[str, Any],
    issued_at_unix: int,
) -> Mapping[str, Any]:
    raw = _strict_mapping(
        collection,
        _PREFLIGHT_COLLECTION_FIELDS,
        "schema_reconciliation_preflight_collection_invalid",
    )
    if (
        type(issued_at_unix) is not int
        or not admin_preflight["issued_at_unix"] <= issued_at_unix
        < admin_preflight["expires_at_unix"]
        or issued_at_unix >= gate["expires_at_unix"]
        or raw.get("postgresql_major") != gate["postgresql_major"]
        or type(raw.get("observed_at_unix")) is not int
        or not admin_preflight["issued_at_unix"]
        <= raw["observed_at_unix"]
        <= issued_at_unix
    ):
        _fail("schema_reconciliation_preflight_collection_invalid")
    for name in (
        "database_identity_sha256",
        "tls_peer_certificate_sha256",
        "managed_hba_receipt_sha256",
        "executor_managed_hba_receipt_sha256",
    ):
        _digest(raw.get(name), "schema_reconciliation_preflight_collection_invalid")
    try:
        executor_hba = managed_cloudsqladmin_hba_receipt_from_mapping(
            raw.get("executor_managed_hba_receipt")
        )
    except (TypeError, ValueError) as exc:
        raise SchemaReconciliationBootstrapError(
            "schema_reconciliation_preflight_collection_invalid"
        ) from exc
    if (
        raw.get("executor_managed_hba_receipt") != executor_hba.as_dict()
        or raw.get("executor_managed_hba_receipt_sha256") != executor_hba.sha256
        or executor_hba.user != gate["temporary_executor_username"]
        or executor_hba.host != phase_b.SQL_HOST
        or executor_hba.tls_server_name != gate["tls_server_name"]
        or executor_hba.port != phase_b.SQL_PORT
        or executor_hba.database != "cloudsqladmin"
        or executor_hba.server_certificate_sha256
        != raw["tls_peer_certificate_sha256"]
        or executor_hba.observed_at_unix > raw["observed_at_unix"]
        or not executor_hba.is_fresh(issued_at_unix)
    ):
        _fail("schema_reconciliation_preflight_collection_invalid")
    truth = _canonical_truth(
        raw.get("canonical_truth_receipt"),
        "schema_reconciliation_preflight_collection_invalid",
    )
    preflight = _validate_core_preflight(raw.get("preflight"), gate=gate, truth=truth)
    if preflight["observed_at_unix"] != raw["observed_at_unix"]:
        _fail("schema_reconciliation_preflight_collection_invalid")
    boundary = _validate_executor_boundary(
        raw.get("temporary_executor_boundary_receipt"),
        observed_at_unix=raw["observed_at_unix"],
    )
    if (
        boundary["control_foundation_contract_sha256"]
        != gate["control_foundation_contract_sha256"]
        or boundary["executor_managed_hba_receipt_sha256"]
        != raw["executor_managed_hba_receipt_sha256"]
    ):
        _fail("schema_reconciliation_temporary_executor_boundary_invalid")
    unsigned = {
        "schema": PREFLIGHT_CHALLENGE_SCHEMA,
        "ok": True,
        "state": "locked_preflight_ready_for_owner_authorization",
        "gate_sha256": gate["gate_sha256"],
        "authority_claim_sha256": admin_preflight["authority_claim_sha256"],
        "cloud_sql_authority_receipt_sha256": admin_preflight[
            "cloud_sql_authority_receipt_sha256"
        ],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "journal_head_sha256": gate["journal_head"]["head_sha256"],
        "database_identity_sha256": raw["database_identity_sha256"],
        "tls_peer_certificate_sha256": raw["tls_peer_certificate_sha256"],
        "managed_hba_receipt_sha256": raw["managed_hba_receipt_sha256"],
        "executor_managed_hba_receipt": executor_hba.as_dict(),
        "executor_managed_hba_receipt_sha256": raw[
            "executor_managed_hba_receipt_sha256"
        ],
        "postgresql_major": raw["postgresql_major"],
        "temporary_executor_boundary_receipt": boundary,
        "preflight": preflight,
        "canonical_truth_receipt": truth.value,
        "mutation_required": preflight["mutation_required"],
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": gate["expires_at_unix"],
        "secret_material_recorded": False,
    }
    value = {
        **unsigned,
        "preflight_challenge_sha256": _sha256_json(unsigned),
    }
    if set(value) != _PREFLIGHT_CHALLENGE_FIELDS:
        _fail("schema_reconciliation_preflight_collection_invalid")
    _require_secret_free(value)
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def validate_preflight_challenge_for_owner(
    value: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    admin_preflight: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate remote P1 against the exact gate and signed A1."""

    code = "schema_reconciliation_preflight_challenge_invalid"
    gate = _strict_mapping(gate, _GATE_FIELDS, code)
    admin_preflight = _strict_mapping(
        admin_preflight,
        _ADMIN_PREFLIGHT_FIELDS,
        code,
    )
    raw = _hashed_mapping(
        value,
        fields=_PREFLIGHT_CHALLENGE_FIELDS,
        digest_field="preflight_challenge_sha256",
        code=code,
    )
    preflight = _strict_mapping(raw.get("preflight"), _CORE_PREFLIGHT_FIELDS, code)
    if type(now_unix) is not int or type(raw.get("issued_at_unix")) is not int:
        _fail(code)
    validated_gate = validate_gate(
        gate,
        owner_public_key_ed25519_hex=str(gate.get("owner_public_key_ed25519_hex")),
        owner_public_fingerprint=str(gate.get("owner_public_fingerprint")),
        now_unix=int(raw["issued_at_unix"]),
    )
    validated_admin = _validate_admin_preflight(
        admin_preflight,
        gate=validated_gate,
        now_unix=int(raw["issued_at_unix"]),
    )
    collection = {
        "database_identity_sha256": raw["database_identity_sha256"],
        "tls_peer_certificate_sha256": raw["tls_peer_certificate_sha256"],
        "managed_hba_receipt_sha256": raw["managed_hba_receipt_sha256"],
        "executor_managed_hba_receipt": raw[
            "executor_managed_hba_receipt"
        ],
        "executor_managed_hba_receipt_sha256": raw[
            "executor_managed_hba_receipt_sha256"
        ],
        "postgresql_major": raw["postgresql_major"],
        "temporary_executor_boundary_receipt": raw[
            "temporary_executor_boundary_receipt"
        ],
        "preflight": preflight,
        "canonical_truth_receipt": raw["canonical_truth_receipt"],
        "observed_at_unix": preflight["observed_at_unix"],
    }
    expected = _build_preflight_challenge(
        gate=validated_gate,
        admin_preflight=validated_admin,
        collection=collection,
        issued_at_unix=int(raw["issued_at_unix"]),
    )
    if (
        raw != expected
        or not raw["issued_at_unix"] <= now_unix < raw["expires_at_unix"]
    ):
        _fail(code)
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def preflight_authorization_signature_payload(value: Mapping[str, Any]) -> bytes:
    raw = _strict_mapping(
        value,
        _PREFLIGHT_AUTHORIZATION_FIELDS,
        "schema_reconciliation_preflight_authorization_invalid",
    )
    return _canonical_bytes({
        key: item for key, item in raw.items() if key != "signature_sshsig"
    })


def _validate_preflight_authorization(
    value: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    admin_preflight: Mapping[str, Any],
    challenge: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    raw = _signed_mapping(
        value,
        fields=_PREFLIGHT_AUTHORIZATION_FIELDS,
        unsigned_fields=_PREFLIGHT_AUTHORIZATION_UNSIGNED_FIELDS,
        digest_field="preflight_authorization_claim_sha256",
        code="schema_reconciliation_preflight_authorization_invalid",
    )
    journal = gate["journal_head"]
    preflight = challenge["preflight"]
    truth = _canonical_truth(
        challenge["canonical_truth_receipt"],
        "schema_reconciliation_preflight_authorization_invalid",
    )
    post_hba_authority = _validate_cloud_authority(
        raw.get("post_hba_temporary_executor_authority_receipt"),
        gate=gate,
    )
    expected_mode = _TRANSITIONS.get((journal["state"], preflight["state"]))
    if (
        expected_mode is None
        or raw.get("schema") != OWNER_PREFLIGHT_AUTHORIZATION_SCHEMA
        or raw.get("frame_schema") != PREFLIGHT_AUTHORIZATION_FRAME_SCHEMA
        or raw.get("action") != "apply_schema_reconciliation"
        or raw.get("approved") is not True
        or raw.get("gate_sha256") != gate["gate_sha256"]
        or raw.get("authority_claim_sha256")
        != admin_preflight["authority_claim_sha256"]
        or raw.get("preflight_challenge_sha256")
        != challenge["preflight_challenge_sha256"]
        or post_hba_authority
        != admin_preflight["cloud_sql_authority_receipt"]
        or raw.get("post_hba_temporary_executor_authority_receipt")
        != post_hba_authority
        or raw.get(
            "post_hba_temporary_executor_authority_receipt_sha256"
        )
        != post_hba_authority["receipt_sha256"]
        or raw.get("release_revision") != gate["release_revision"]
        or raw.get("plan_sha256") != gate["plan_sha256"]
        or raw.get("journal_head_sha256") != journal["head_sha256"]
        or raw.get("execution_mode") != expected_mode
        or raw.get("preflight_sha256") != preflight["preflight_sha256"]
        or raw.get("preflight_state") != preflight["state"]
        or raw.get("observed_contract_sha256")
        != preflight["observed_contract_sha256"]
        or raw.get("truth_receipt_sha256") != truth.sha256
        or raw.get("owner_subject_sha256") != gate["owner_subject_sha256"]
        or raw.get("owner_key_id") != gate["owner_key_id"]
        or raw.get("stored_authorized_intent_sha256")
        != journal["authorized_intent_sha256"]
        or raw.get("stored_terminal_receipt_sha256")
        != journal["terminal_receipt_sha256"]
        or raw.get("secret_material_recorded") is not False
    ):
        _fail("schema_reconciliation_preflight_authorization_invalid")
    _digest(
        raw.get("nonce_sha256"),
        "schema_reconciliation_preflight_authorization_invalid",
    )
    _validate_ttl(
        raw,
        now_unix=now_unix,
        code="schema_reconciliation_preflight_authorization_expired",
        not_before=int(challenge["issued_at_unix"]),
        not_after=int(gate["expires_at_unix"]),
    )

    _verify_signature(
        raw.get("signature_sshsig"),
        message=preflight_authorization_signature_payload(raw),
        public_key_ed25519_hex=gate["owner_public_key_ed25519_hex"],
        namespace=PREFLIGHT_AUTHORIZATION_OWNER_SSHSIG_NAMESPACE,
        code="schema_reconciliation_preflight_authorization_signature_invalid",
    )
    _require_secret_free(raw)
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def _build_core_admission(
    *,
    gate: Mapping[str, Any],
    challenge: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> tuple[
    SchemaReconciliationAuthorization | None,
    Mapping[str, Any] | None,
]:
    """Derive core admission only for a previously empty journal.

    Recovery frames authorize a fresh outer transition but cannot replace the
    byte-identical core authorization and owner-frame receipt already sealed
    in the durable intent.  The apply callback must load those durable values.
    """

    if gate["journal_head"]["state"] != "empty":
        return None, None
    preflight = challenge["preflight"]
    truth = _canonical_truth(
        challenge["canonical_truth_receipt"],
        "schema_reconciliation_preflight_authorization_invalid",
    )
    signed_frame_sha256 = _sha256_json(authorization)
    unsigned_authorization = {
        "schema": RECONCILIATION_AUTHORIZATION_SCHEMA,
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "preflight_sha256": preflight["preflight_sha256"],
        "preflight_state": preflight["state"],
        "observed_contract_sha256": preflight["observed_contract_sha256"],
        "truth_receipt_sha256": truth.sha256,
        "owner_frame_sha256": signed_frame_sha256,
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "owner_key_id": gate["owner_key_id"],
        "issued_at_unix": authorization["issued_at_unix"],
        "expires_at_unix": authorization["expires_at_unix"],
        "nonce": authorization["nonce_sha256"],
    }
    try:
        core_authorization = SchemaReconciliationAuthorization.from_mapping({
            **unsigned_authorization,
            "authorization_sha256": _sha256_json(unsigned_authorization),
        })
        signature = authorization["signature_sshsig"].encode(
            "utf-8", errors="strict"
        )
    except (SchemaReconciliationError, KeyError, TypeError, UnicodeError) as exc:
        raise SchemaReconciliationBootstrapError(
            "schema_reconciliation_preflight_authorization_invalid"
        ) from exc
    binding = core_authorization.value
    unsigned_receipt = {
        "schema": RECONCILIATION_OWNER_FRAME_RECEIPT_SCHEMA,
        "frame_schema": RECONCILIATION_OWNER_A2_FRAME_SCHEMA,
        "action": "apply_schema_reconciliation",
        "approved": True,
        "signed_frame_sha256": signed_frame_sha256,
        "signature_sshsig_sha256": _sha256_bytes(signature),
        "signature_namespace": RECONCILIATION_OWNER_SIGNATURE_NAMESPACE,
        "signature_verified": True,
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "preflight_sha256": preflight["preflight_sha256"],
        "observed_contract_sha256": preflight["observed_contract_sha256"],
        "truth_receipt_sha256": truth.sha256,
        "authorization_sha256": core_authorization.sha256,
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "owner_key_id": gate["owner_key_id"],
        "issued_at_unix": binding["issued_at_unix"],
        "expires_at_unix": binding["expires_at_unix"],
        "nonce": binding["nonce"],
    }
    owner_frame_receipt = _hashed_mapping(
        {
            **unsigned_receipt,
            "receipt_sha256": _sha256_json(unsigned_receipt),
        },
        fields=_CORE_OWNER_FRAME_RECEIPT_FIELDS,
        digest_field="receipt_sha256",
        code="schema_reconciliation_preflight_authorization_invalid",
    )
    _require_secret_free(owner_frame_receipt)
    return (
        core_authorization,
        json.loads(_canonical_bytes(owner_frame_receipt).decode("utf-8")),
    )


def _validate_core_terminal(
    value: Any,
    *,
    gate: Mapping[str, Any],
    challenge: Mapping[str, Any],
    authorization: Mapping[str, Any],
    core_authorization: SchemaReconciliationAuthorization | None,
    owner_frame_receipt: Mapping[str, Any] | None,
    authorized_intent_sha256: str,
    now_unix: int,
) -> Mapping[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_CORE_TERMINAL_FIELDS,
        digest_field="receipt_sha256",
        code="schema_reconciliation_core_terminal_invalid",
    )
    try:
        initial_truth = CanonicalTruthReceipt.from_mapping(
            raw["initial_canonical_truth"]
        )
        final_truth = CanonicalTruthReceipt.from_mapping(raw["final_canonical_truth"])
    except (SchemaReconciliationError, TypeError, ValueError) as exc:
        raise SchemaReconciliationBootstrapError(
            "schema_reconciliation_core_terminal_invalid"
        ) from exc
    challenge_truth = _canonical_truth(
        challenge["canonical_truth_receipt"],
        "schema_reconciliation_core_terminal_invalid",
    )
    mode = raw.get("mode")
    execution_mode = authorization["execution_mode"]
    journal = gate["journal_head"]
    expected_mode = (
        "reconcile_missing_helper"
        if execution_mode in {"reconcile_missing_helper", "resume_durable_intent"}
        else "adopt_existing_target"
        if execution_mode == "adopt_existing_target"
        else None
    )
    expected_initial_contract = (
        gate["expected_old_contract_sha256"]
        if mode == "reconcile_missing_helper"
        else gate["target_contract_sha256"]
        if mode == "adopt_existing_target"
        else None
    )
    if (
        raw.get("schema") != RECONCILIATION_RECEIPT_SCHEMA
        or raw.get("ok") is not True
        or raw.get("release_revision") != gate["release_revision"]
        or raw.get("plan_sha256") != gate["plan_sha256"]
        or raw.get("base_artifact_sha256") != gate["base_artifact_sha256"]
        or raw.get("target_asset_sha256") != gate["target_asset_sha256"]
        or raw.get("postgresql_major") != gate["postgresql_major"]
        or any(
            raw.get(name) != gate[name]
            for name in (
                "control_install_artifact_sha256",
                "control_retire_artifact_sha256",
                "control_foundation_contract_sha256",
            )
        )
        or raw.get("expected_old_contract_sha256")
        != gate["expected_old_contract_sha256"]
        or raw.get("target_contract_sha256") != gate["target_contract_sha256"]
        or raw.get("initial_contract_sha256") != expected_initial_contract
        or raw.get("final_contract_sha256") != gate["target_contract_sha256"]
        or raw.get("truth_receipt_sha256") != challenge_truth.sha256
        or initial_truth != challenge_truth
        or final_truth != challenge_truth
        or mode not in {"reconcile_missing_helper", "adopt_existing_target"}
        or raw.get("mutation_applied")
        is not (mode == "reconcile_missing_helper")
        or (
            journal["state"] == "empty"
            and expected_mode is not None
            and mode != expected_mode
        )
        or (
            execution_mode == "resume_durable_intent"
            and mode != "reconcile_missing_helper"
        )
        or type(raw.get("completed_at_unix")) is not int
        or not 0 <= raw["completed_at_unix"] <= now_unix
        or (
            journal["state"] != "terminal"
            and raw["completed_at_unix"] < challenge["issued_at_unix"]
        )
        or raw.get("authorized_intent_sha256") != authorized_intent_sha256
    ):
        _fail("schema_reconciliation_core_terminal_invalid")
    for name in (
        "authorization_sha256",
        "preflight_sha256",
        "owner_frame_receipt_sha256",
        "authorized_intent_sha256",
    ):
        _digest(raw.get(name), "schema_reconciliation_core_terminal_invalid")
    if journal["state"] == "empty":
        if (
            not isinstance(core_authorization, SchemaReconciliationAuthorization)
            or not isinstance(owner_frame_receipt, Mapping)
            or raw["authorization_sha256"] != core_authorization.sha256
            or raw["preflight_sha256"]
            != challenge["preflight"]["preflight_sha256"]
            or raw["owner_frame_receipt_sha256"]
            != owner_frame_receipt.get("receipt_sha256")
        ):
            _fail("schema_reconciliation_core_terminal_invalid")
    elif core_authorization is not None or owner_frame_receipt is not None:
        _fail("schema_reconciliation_core_terminal_invalid")
    if (
        journal["state"] != "empty"
        and raw["authorized_intent_sha256"]
        != journal["authorized_intent_sha256"]
    ):
        _fail("schema_reconciliation_core_terminal_invalid")
    if (
        gate["journal_head"]["state"] == "terminal"
        and raw["receipt_sha256"]
        != gate["journal_head"]["terminal_receipt_sha256"]
    ):
        _fail("schema_reconciliation_core_terminal_invalid")
    _require_secret_free(raw)
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def _validate_database_attestation(
    value: Any,
    *,
    gate: Mapping[str, Any],
    challenge: Mapping[str, Any],
    core_terminal: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_DATABASE_ATTESTATION_FIELDS,
        digest_field="attestation_sha256",
        code="schema_reconciliation_database_attestation_invalid",
    )
    truth = _canonical_truth(
        raw.get("canonical_truth_receipt"),
        "schema_reconciliation_database_attestation_invalid",
    )
    final_truth = _canonical_truth(
        core_terminal["final_canonical_truth"],
        "schema_reconciliation_database_attestation_invalid",
    )
    if (
        raw.get("schema") != DATABASE_COMMIT_ATTESTATION_SCHEMA
        or raw.get("ok") is not True
        or raw.get("database_identity_sha256")
        != challenge["database_identity_sha256"]
        or raw.get("tls_peer_certificate_sha256")
        != challenge["tls_peer_certificate_sha256"]
        or raw.get("postgresql_major") != gate["postgresql_major"]
        or raw.get("observed_contract_sha256") != gate["target_contract_sha256"]
        or truth != final_truth
        or raw.get("transaction_committed") is not True
        or raw.get("temporary_login_owns_zero_objects") is not True
        or raw.get("fixed_control_routines_re_attested_before_commit")
        is not True
        or raw.get("inert_executor_membership_present") is not True
        or raw.get("cloud_user_cleanup_required") is not True
        or raw.get("database_session_closed") is not True
        or raw.get("re_attested_before_temporary_executor_delete") is not True
        or type(raw.get("observed_at_unix")) is not int
        or not challenge["issued_at_unix"] <= raw["observed_at_unix"] <= now_unix
        or raw.get("secret_material_recorded") is not False
    ):
        _fail("schema_reconciliation_database_attestation_invalid")
    _require_secret_free(raw)
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def _build_database_intermediate(
    *,
    gate: Mapping[str, Any],
    admin_preflight: Mapping[str, Any],
    challenge: Mapping[str, Any],
    authorization: Mapping[str, Any],
    core_authorization: SchemaReconciliationAuthorization | None,
    owner_frame_receipt: Mapping[str, Any] | None,
    apply_result: Mapping[str, Any],
    applied_at_unix: int,
) -> Mapping[str, Any]:
    raw = _strict_mapping(
        apply_result,
        _APPLY_RESULT_FIELDS,
        "schema_reconciliation_apply_result_invalid",
    )
    _digest(
        raw.get("authorized_intent_sha256"),
        "schema_reconciliation_apply_result_invalid",
    )
    journal = gate["journal_head"]
    if (
        journal["state"] != "empty"
        and raw["authorized_intent_sha256"]
        != journal["authorized_intent_sha256"]
    ):
        _fail("schema_reconciliation_apply_result_invalid")
    core_terminal = _validate_core_terminal(
        raw.get("core_terminal_receipt"),
        gate=gate,
        challenge=challenge,
        authorization=authorization,
        core_authorization=core_authorization,
        owner_frame_receipt=owner_frame_receipt,
        authorized_intent_sha256=raw["authorized_intent_sha256"],
        now_unix=applied_at_unix,
    )
    database_attestation = _validate_database_attestation(
        raw.get("database_commit_attestation"),
        gate=gate,
        challenge=challenge,
        core_terminal=core_terminal,
        now_unix=applied_at_unix,
    )
    unsigned = {
        "schema": DATABASE_INTERMEDIATE_SCHEMA,
        "ok": True,
        "state": "database_target_re_attested_awaiting_executor_cleanup",
        "gate_sha256": gate["gate_sha256"],
        "authority_claim_sha256": admin_preflight["authority_claim_sha256"],
        "preflight_challenge_sha256": challenge[
            "preflight_challenge_sha256"
        ],
        "preflight_authorization_claim_sha256": authorization[
            "preflight_authorization_claim_sha256"
        ],
        "post_hba_temporary_executor_authority_receipt_sha256": (
            authorization[
                "post_hba_temporary_executor_authority_receipt_sha256"
            ]
        ),
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "execution_mode": authorization["execution_mode"],
        "authorized_intent_sha256": raw["authorized_intent_sha256"],
        "core_terminal_receipt": core_terminal,
        "core_terminal_receipt_sha256": core_terminal["receipt_sha256"],
        "database_commit_attestation": database_attestation,
        "database_commit_attestation_sha256": database_attestation[
            "attestation_sha256"
        ],
        "initial_canonical_truth": core_terminal["initial_canonical_truth"],
        "final_canonical_truth": core_terminal["final_canonical_truth"],
        "temporary_executor_cleanup_required": True,
        "database_session_closed": True,
        "applied_at_unix": applied_at_unix,
        "secret_material_recorded": False,
    }
    value = {
        **unsigned,
        "database_intermediate_sha256": _sha256_json(unsigned),
    }
    if set(value) != _DATABASE_INTERMEDIATE_FIELDS:
        _fail("schema_reconciliation_apply_result_invalid")
    _require_secret_free(value)
    return validate_database_intermediate_for_owner(
        value,
        gate=gate,
        admin_preflight=admin_preflight,
        challenge=challenge,
        authorization=authorization,
        now_unix=applied_at_unix,
    )


def validate_database_intermediate_for_owner(
    value: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    admin_preflight: Mapping[str, Any],
    challenge: Mapping[str, Any],
    authorization: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate remote I2 and its committed pre-delete DB attestation."""

    code = "schema_reconciliation_database_intermediate_invalid"
    gate = _strict_mapping(gate, _GATE_FIELDS, code)
    admin_preflight = _strict_mapping(
        admin_preflight,
        _ADMIN_PREFLIGHT_FIELDS,
        code,
    )
    challenge = _strict_mapping(challenge, _PREFLIGHT_CHALLENGE_FIELDS, code)
    authorization = _strict_mapping(
        authorization,
        _PREFLIGHT_AUTHORIZATION_FIELDS,
        code,
    )
    raw = _hashed_mapping(
        value,
        fields=_DATABASE_INTERMEDIATE_FIELDS,
        digest_field="database_intermediate_sha256",
        code=code,
    )
    if (
        type(now_unix) is not int
        or type(raw.get("applied_at_unix")) is not int
        or type(challenge.get("issued_at_unix")) is not int
        or type(authorization.get("issued_at_unix")) is not int
        or type(gate.get("expires_at_unix")) is not int
        or not challenge["issued_at_unix"]
        <= raw["applied_at_unix"]
        <= now_unix
        or raw["applied_at_unix"] >= gate["expires_at_unix"]
    ):
        _fail(code)
    validated_challenge = validate_preflight_challenge_for_owner(
        challenge,
        gate=gate,
        admin_preflight=admin_preflight,
        now_unix=challenge["issued_at_unix"],
    )
    validated_authorization = _validate_preflight_authorization(
        authorization,
        gate=gate,
        admin_preflight=admin_preflight,
        challenge=validated_challenge,
        now_unix=authorization["issued_at_unix"],
    )
    if not (
        validated_authorization["issued_at_unix"]
        <= raw["applied_at_unix"]
        < validated_authorization["expires_at_unix"]
    ):
        _fail(code)
    core_authorization, owner_frame_receipt = _build_core_admission(
        gate=gate,
        challenge=validated_challenge,
        authorization=validated_authorization,
    )
    _digest(
        raw.get("authorized_intent_sha256"),
        code,
    )
    journal = gate["journal_head"]
    if (
        raw.get("schema") != DATABASE_INTERMEDIATE_SCHEMA
        or raw.get("ok") is not True
        or raw.get("state")
        != "database_target_re_attested_awaiting_executor_cleanup"
        or raw.get("gate_sha256") != gate["gate_sha256"]
        or raw.get("authority_claim_sha256")
        != admin_preflight["authority_claim_sha256"]
        or raw.get("preflight_challenge_sha256")
        != validated_challenge["preflight_challenge_sha256"]
        or raw.get("preflight_authorization_claim_sha256")
        != validated_authorization["preflight_authorization_claim_sha256"]
        or raw.get(
            "post_hba_temporary_executor_authority_receipt_sha256"
        )
        != validated_authorization[
            "post_hba_temporary_executor_authority_receipt_sha256"
        ]
        or raw.get("release_revision") != gate["release_revision"]
        or raw.get("plan_sha256") != gate["plan_sha256"]
        or raw.get("execution_mode")
        != validated_authorization["execution_mode"]
        or (
            journal["state"] != "empty"
            and raw.get("authorized_intent_sha256")
            != journal["authorized_intent_sha256"]
        )
        or raw.get("temporary_executor_cleanup_required") is not True
        or raw.get("database_session_closed") is not True
        or raw.get("secret_material_recorded") is not False
    ):
        _fail(code)
    core_terminal = _validate_core_terminal(
        raw.get("core_terminal_receipt"),
        gate=gate,
        challenge=validated_challenge,
        authorization=validated_authorization,
        core_authorization=core_authorization,
        owner_frame_receipt=owner_frame_receipt,
        authorized_intent_sha256=raw["authorized_intent_sha256"],
        now_unix=raw["applied_at_unix"],
    )
    database_attestation = _validate_database_attestation(
        raw.get("database_commit_attestation"),
        gate=gate,
        challenge=validated_challenge,
        core_terminal=core_terminal,
        now_unix=raw["applied_at_unix"],
    )
    if (
        raw.get("core_terminal_receipt") != core_terminal
        or raw.get("core_terminal_receipt_sha256")
        != core_terminal["receipt_sha256"]
        or raw.get("database_commit_attestation") != database_attestation
        or raw.get("database_commit_attestation_sha256")
        != database_attestation["attestation_sha256"]
        or raw.get("initial_canonical_truth")
        != core_terminal["initial_canonical_truth"]
        or raw.get("final_canonical_truth")
        != core_terminal["final_canonical_truth"]
    ):
        _fail(code)
    _require_secret_free(raw)
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def _validate_cloud_absence(
    value: Any,
    *,
    gate: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_CLOUD_ABSENCE_FIELDS,
        digest_field="evidence_sha256",
        code="schema_reconciliation_cloud_absence_invalid",
    )
    baseline_names = _operation_names(
        raw.get("baseline_operation_names"),
        "schema_reconciliation_cloud_absence_invalid",
    )
    baseline_rows = _operation_rows(
        raw.get("baseline_user_operations"),
        "schema_reconciliation_cloud_absence_invalid",
    )
    known_names = _operation_names(
        raw.get("known_operation_names"),
        "schema_reconciliation_cloud_absence_invalid",
    )
    known_authority = _operation_names(
        raw.get("response_known_authority_operation_names"),
        "schema_reconciliation_cloud_absence_invalid",
    )
    known_deletes = _operation_names(
        raw.get("response_known_delete_operation_names"),
        "schema_reconciliation_cloud_absence_invalid",
    )
    post_authority = _operation_rows(
        raw.get("post_baseline_authority_operations"),
        "schema_reconciliation_cloud_absence_invalid",
    )
    terminal_rows = _operation_rows(
        raw.get("terminal_user_operations"),
        "schema_reconciliation_cloud_absence_invalid",
    )
    authority_row = list(authority["authority_operation"])
    terminal_by_name = {row[0]: row for row in terminal_rows}
    expected_terminal_names = {
        *(row[0] for row in baseline_rows),
        *known_names,
    }
    quiet_window = raw.get("quiet_window_seconds")
    if (
        raw.get("schema") != CLOUD_EXECUTOR_ABSENCE_SCHEMA
        or raw.get("temporary_executor_absent") is not True
        or raw.get("project") != gate["project"]
        or raw.get("instance") != gate["sql_instance"]
        or raw.get("username_sha256")
        != gate["temporary_executor_username_sha256"]
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
        or set(terminal_by_name) != expected_terminal_names
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
        _fail("schema_reconciliation_cloud_absence_invalid")
    _require_secret_free(raw)
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def admin_cleanup_signature_payload(value: Mapping[str, Any]) -> bytes:
    raw = _strict_mapping(
        value,
        _ADMIN_CLEANUP_FIELDS,
        "schema_reconciliation_admin_cleanup_invalid",
    )
    return _canonical_bytes({
        key: item for key, item in raw.items() if key != "signature_sshsig"
    })


executor_cleanup_signature_payload = admin_cleanup_signature_payload


def _validate_admin_cleanup(
    value: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    admin_preflight: Mapping[str, Any],
    challenge: Mapping[str, Any],
    authorization: Mapping[str, Any],
    intermediate: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    raw = _signed_mapping(
        value,
        fields=_ADMIN_CLEANUP_FIELDS,
        unsigned_fields=_ADMIN_CLEANUP_UNSIGNED_FIELDS,
        digest_field="cleanup_claim_sha256",
        code="schema_reconciliation_admin_cleanup_invalid",
    )
    absence = _validate_cloud_absence(
        raw.get("cloud_sql_absence_receipt"),
        gate=gate,
        authority=admin_preflight["cloud_sql_authority_receipt"],
    )
    if (
        raw.get("schema") != OWNER_EXECUTOR_CLEANUP_SCHEMA
        or raw.get("frame_schema") != EXECUTOR_CLEANUP_FRAME_SCHEMA
        or raw.get("action") != "confirm_temporary_executor_absence"
        or raw.get("approved") is not True
        or raw.get("gate_sha256") != gate["gate_sha256"]
        or raw.get("authority_claim_sha256")
        != admin_preflight["authority_claim_sha256"]
        or raw.get("preflight_challenge_sha256")
        != challenge["preflight_challenge_sha256"]
        or raw.get("preflight_authorization_claim_sha256")
        != authorization["preflight_authorization_claim_sha256"]
        or raw.get("database_intermediate_sha256")
        != intermediate["database_intermediate_sha256"]
        or raw.get("release_revision") != gate["release_revision"]
        or raw.get("plan_sha256") != gate["plan_sha256"]
        or raw.get("temporary_executor_username_sha256")
        != gate["temporary_executor_username_sha256"]
        or raw.get("owner_subject_sha256") != gate["owner_subject_sha256"]
        or raw.get("owner_key_id") != gate["owner_key_id"]
        or raw.get("cloud_sql_absence_receipt") != absence
        or raw.get("cloud_sql_absence_receipt_sha256")
        != absence["evidence_sha256"]
        or raw.get("secret_material_recorded") is not False
    ):
        _fail("schema_reconciliation_admin_cleanup_invalid")
    _digest(raw.get("nonce_sha256"), "schema_reconciliation_admin_cleanup_invalid")
    _validate_ttl(
        raw,
        now_unix=now_unix,
        code="schema_reconciliation_admin_cleanup_expired",
        not_before=int(intermediate["applied_at_unix"]),
        not_after=int(gate["expires_at_unix"]),
    )
    _verify_signature(
        raw.get("signature_sshsig"),
        message=admin_cleanup_signature_payload(raw),
        public_key_ed25519_hex=gate["owner_public_key_ed25519_hex"],
        namespace=EXECUTOR_CLEANUP_OWNER_SSHSIG_NAMESPACE,
        code="schema_reconciliation_admin_cleanup_signature_invalid",
    )
    _require_secret_free(raw)
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def _validate_post_cleanup_observation(
    value: Any,
    *,
    gate: Mapping[str, Any],
    challenge: Mapping[str, Any],
    intermediate: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_POST_CLEANUP_FIELDS,
        digest_field="observation_sha256",
        code="schema_reconciliation_post_cleanup_observation_invalid",
    )
    try:
        post_delete_terminal = parse_post_delete_terminal_receipt(
            raw.get("post_delete_terminal_receipt")
        )
        fresh_hba_raw = raw.get("fresh_managed_hba_receipt")
        if not isinstance(fresh_hba_raw, Mapping):
            raise TypeError
        fresh_hba = managed_cloudsqladmin_hba_receipt_from_mapping(
            fresh_hba_raw
        )
        pre_delete_truth = CanonicalTruthReceipt.from_mapping(
            intermediate["final_canonical_truth"]
        )
    except BaseException as exc:
        raise SchemaReconciliationBootstrapError(
            "schema_reconciliation_post_cleanup_observation_invalid"
        ) from exc
    if (
        raw.get("schema") != POST_CLEANUP_OBSERVATION_SCHEMA
        or raw.get("release_manifest_sha256")
        != gate["release_manifest_sha256"]
        or raw.get("stopped_release_receipt_file_sha256")
        != gate["stopped_release_receipt_file_sha256"]
        or raw.get("stopped_release_receipt_sha256")
        != gate["stopped_release_receipt_sha256"]
        or raw.get("release_artifact_sha256")
        != gate["release_artifact_sha256"]
        or raw.get("python_version") != gate["python_version"]
        or raw.get("interpreter_sha256") != gate["interpreter_sha256"]
        or raw.get("activation_inventory_sha256")
        != gate["activation_inventory_sha256"]
        or raw.get("host_identity_sha256") != gate["host_identity_sha256"]
        or raw.get("services_stopped_sha256") != gate["services_stopped_sha256"]
        or post_delete_terminal.release_revision != gate["release_revision"]
        or post_delete_terminal.plan_sha256 != gate["plan_sha256"]
        or post_delete_terminal.database != gate["database"]
        or post_delete_terminal.temporary_executor_login
        != gate["temporary_executor_username"]
        or post_delete_terminal.temporary_executor_login_sha256
        != gate["temporary_executor_username_sha256"]
        or post_delete_terminal.control_foundation_contract_sha256
        != gate["control_foundation_contract_sha256"]
        or post_delete_terminal.target_contract_sha256
        != gate["target_contract_sha256"]
        or post_delete_terminal.observed_contract_sha256
        != gate["target_contract_sha256"]
        or raw.get("fresh_managed_hba_receipt") != fresh_hba.as_dict()
        or raw.get("fresh_managed_hba_receipt_sha256") != fresh_hba.sha256
        or fresh_hba.host != phase_b.SQL_HOST
        or fresh_hba.tls_server_name != phase_b.SQL_TLS_SERVER_NAME
        or fresh_hba.port != phase_b.SQL_PORT
        or fresh_hba.user != phase_b.SQL_USER
        or fresh_hba.server_certificate_sha256
        != challenge["tls_peer_certificate_sha256"]
        or post_delete_terminal.managed_hba_receipt_sha256
        != fresh_hba.sha256
        or post_delete_terminal.tls_peer_certificate_sha256
        != fresh_hba.server_certificate_sha256
        or post_delete_terminal.tls_peer_certificate_sha256
        != challenge["tls_peer_certificate_sha256"]
        or post_delete_terminal.pre_delete_canonical_truth_receipt_sha256
        != pre_delete_truth.sha256
        or raw.get("post_delete_terminal_receipt")
        != post_delete_terminal.value
        or raw.get("post_delete_terminal_receipt_sha256")
        != post_delete_terminal.value["receipt_sha256"]
        or type(raw.get("observed_at_unix")) is not int
        or not cleanup["issued_at_unix"]
        <= post_delete_terminal.observed_at_unix
        <= raw["observed_at_unix"]
        or fresh_hba.observed_at_unix < cleanup["issued_at_unix"]
        or not fresh_hba.is_fresh(post_delete_terminal.observed_at_unix)
        or not fresh_hba.is_fresh(raw.get("observed_at_unix"))
        or not cleanup["issued_at_unix"] <= raw["observed_at_unix"] <= now_unix
        or raw.get("secret_material_recorded") is not False
    ):
        _fail("schema_reconciliation_post_cleanup_observation_invalid")
    for name in (
        "release_manifest_sha256",
        "stopped_release_receipt_file_sha256",
        "stopped_release_receipt_sha256",
        "release_artifact_sha256",
        "interpreter_sha256",
        "activation_inventory_sha256",
        "host_observation_receipt_sha256",
        "services_observation_receipt_sha256",
        "fresh_managed_hba_receipt_sha256",
        "post_delete_terminal_receipt_sha256",
    ):
        _digest(
            raw.get(name),
            "schema_reconciliation_post_cleanup_observation_invalid",
        )
    _require_secret_free(raw)
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def _build_terminal(
    *,
    gate: Mapping[str, Any],
    admin_preflight: Mapping[str, Any],
    challenge: Mapping[str, Any],
    authorization: Mapping[str, Any],
    intermediate: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    post_cleanup: Mapping[str, Any],
    completed_at_unix: int,
) -> Mapping[str, Any]:
    unsigned = {
        "schema": TERMINAL_SCHEMA,
        "ok": True,
        "state": (
            "pre_delete_full_truth_re_attested_then_temporary_executor_absence_"
            "and_fresh_writer_authority_contract_behavior_proven"
        ),
        "gate_sha256": gate["gate_sha256"],
        "authority_claim_sha256": admin_preflight["authority_claim_sha256"],
        "preflight_challenge_sha256": challenge[
            "preflight_challenge_sha256"
        ],
        "preflight_authorization_claim_sha256": authorization[
            "preflight_authorization_claim_sha256"
        ],
        "post_hba_temporary_executor_authority_receipt_sha256": (
            intermediate[
                "post_hba_temporary_executor_authority_receipt_sha256"
            ]
        ),
        "database_intermediate_sha256": intermediate[
            "database_intermediate_sha256"
        ],
        "cleanup_claim_sha256": cleanup["cleanup_claim_sha256"],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "release_manifest_sha256": post_cleanup["release_manifest_sha256"],
        "stopped_release_receipt_file_sha256": post_cleanup[
            "stopped_release_receipt_file_sha256"
        ],
        "stopped_release_receipt_sha256": post_cleanup[
            "stopped_release_receipt_sha256"
        ],
        "release_artifact_sha256": post_cleanup["release_artifact_sha256"],
        "python_version": post_cleanup["python_version"],
        "interpreter_sha256": post_cleanup["interpreter_sha256"],
        "activation_inventory_sha256": post_cleanup[
            "activation_inventory_sha256"
        ],
        "target_asset_sha256": gate["target_asset_sha256"],
        "target_contract_sha256": gate["target_contract_sha256"],
        "authorized_intent_sha256": intermediate["authorized_intent_sha256"],
        "core_terminal_receipt_sha256": intermediate[
            "core_terminal_receipt_sha256"
        ],
        "database_commit_attestation_sha256": intermediate[
            "database_commit_attestation_sha256"
        ],
        "final_canonical_truth": intermediate["final_canonical_truth"],
        "temporary_executor_absence_receipt_sha256": cleanup[
            "cloud_sql_absence_receipt_sha256"
        ],
        "fresh_managed_hba_receipt_sha256": post_cleanup[
            "fresh_managed_hba_receipt_sha256"
        ],
        "post_delete_terminal_receipt_sha256": post_cleanup[
            "post_delete_terminal_receipt_sha256"
        ],
        "post_cleanup_observation": post_cleanup,
        "host_identity_sha256": post_cleanup["host_identity_sha256"],
        "services_stopped_sha256": post_cleanup["services_stopped_sha256"],
        "host_observation_receipt_sha256": post_cleanup[
            "host_observation_receipt_sha256"
        ],
        "services_observation_receipt_sha256": post_cleanup[
            "services_observation_receipt_sha256"
        ],
        "post_cleanup_observation_sha256": post_cleanup["observation_sha256"],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "owner_key_id": gate["owner_key_id"],
        "database_re_attested_before_temporary_executor_delete": True,
        "fresh_writer_post_delete_authority_contract_and_behavior_proven": True,
        "post_delete_canonical_truth_observed": False,
        "secret_material_recorded": False,
        "completed_at_unix": completed_at_unix,
    }
    terminal = {**unsigned, "terminal_sha256": _sha256_json(unsigned)}
    if set(terminal) != _TERMINAL_FIELDS:
        _fail("schema_reconciliation_terminal_invalid")
    _require_secret_free(terminal)
    return json.loads(_canonical_bytes(terminal).decode("utf-8"))


def validate_terminal_for_owner(
    value: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    admin_preflight: Mapping[str, Any],
    challenge: Mapping[str, Any],
    authorization: Mapping[str, Any],
    intermediate: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate remote T3 against all prior validated frames and receipts."""

    code = "schema_reconciliation_terminal_invalid"
    gate = _strict_mapping(gate, _GATE_FIELDS, code)
    admin_preflight = _strict_mapping(
        admin_preflight,
        _ADMIN_PREFLIGHT_FIELDS,
        code,
    )
    challenge = _strict_mapping(challenge, _PREFLIGHT_CHALLENGE_FIELDS, code)
    authorization = _strict_mapping(
        authorization,
        _PREFLIGHT_AUTHORIZATION_FIELDS,
        code,
    )
    intermediate = _strict_mapping(
        intermediate,
        _DATABASE_INTERMEDIATE_FIELDS,
        code,
    )
    cleanup = _strict_mapping(cleanup, _ADMIN_CLEANUP_FIELDS, code)
    raw = _hashed_mapping(
        value,
        fields=_TERMINAL_FIELDS,
        digest_field="terminal_sha256",
        code=code,
    )
    if (
        type(now_unix) is not int
        or type(raw.get("completed_at_unix")) is not int
        or type(cleanup.get("issued_at_unix")) is not int
        or type(challenge.get("issued_at_unix")) is not int
        or type(authorization.get("issued_at_unix")) is not int
        or type(intermediate.get("applied_at_unix")) is not int
        or type(gate.get("expires_at_unix")) is not int
        or not cleanup["issued_at_unix"]
        <= raw["completed_at_unix"]
        <= now_unix
        or raw["completed_at_unix"] >= gate["expires_at_unix"]
    ):
        _fail(code)
    validated_challenge = validate_preflight_challenge_for_owner(
        challenge,
        gate=gate,
        admin_preflight=admin_preflight,
        now_unix=challenge["issued_at_unix"],
    )
    validated_authorization = _validate_preflight_authorization(
        authorization,
        gate=gate,
        admin_preflight=admin_preflight,
        challenge=validated_challenge,
        now_unix=authorization["issued_at_unix"],
    )
    validated_intermediate = validate_database_intermediate_for_owner(
        intermediate,
        gate=gate,
        admin_preflight=admin_preflight,
        challenge=validated_challenge,
        authorization=validated_authorization,
        now_unix=intermediate["applied_at_unix"],
    )
    validated_cleanup = _validate_admin_cleanup(
        cleanup,
        gate=gate,
        admin_preflight=admin_preflight,
        challenge=validated_challenge,
        authorization=validated_authorization,
        intermediate=validated_intermediate,
        now_unix=cleanup["issued_at_unix"],
    )
    post_cleanup = _validate_post_cleanup_observation(
        raw.get("post_cleanup_observation"),
        gate=gate,
        challenge=validated_challenge,
        intermediate=validated_intermediate,
        cleanup=validated_cleanup,
        now_unix=raw["completed_at_unix"],
    )
    expected = _build_terminal(
        gate=gate,
        admin_preflight=admin_preflight,
        challenge=validated_challenge,
        authorization=validated_authorization,
        intermediate=validated_intermediate,
        cleanup=validated_cleanup,
        post_cleanup=post_cleanup,
        completed_at_unix=raw["completed_at_unix"],
    )
    if raw != expected:
        _fail(code)
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def _read_exact_mutable(stream: BinaryIO, size: int, *, code: str) -> bytearray:
    if type(size) is not int or size < 0:
        _fail(code)
    result = bytearray(size)
    view = memoryview(result)
    offset = 0
    try:
        while offset < size:
            read = stream.readinto(view[offset:])
            if (
                type(read) is not int
                or read <= 0
                or read > size - offset
            ):
                _fail(code)
            offset += read
        return result
    except BaseException:
        _zeroize(result)
        raise
    finally:
        view.release()


def _read_exact(stream: BinaryIO, size: int, *, code: str) -> bytes:
    mutable = _read_exact_mutable(stream, size, code=code)
    try:
        return bytes(mutable)
    finally:
        _zeroize(mutable)


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
    if not 1 <= size <= MAX_JSON_BYTES:
        _fail(code)
    return _decode_canonical_mapping(_read_exact(stream, size, code=code))


def _emit_mapping(stream: BinaryIO, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value) + b"\n"
    try:
        written = stream.write(payload)
        if written is not None and written != len(payload):
            _fail("schema_reconciliation_output_failed")
        stream.flush()
    except SchemaReconciliationBootstrapError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise SchemaReconciliationBootstrapError(
            "schema_reconciliation_output_failed"
        ) from exc


def _remote_failure_receipt(
    *,
    gate: Mapping[str, Any],
    wire_stage: str,
    transcript_head_sha256: str,
    error: BaseException,
) -> Mapping[str, Any]:
    """Build one wire-only, secret-free failure bound to the emitted prefix."""

    error_code = _GENERIC_REMOTE_ERROR
    if isinstance(error, SchemaReconciliationBootstrapError):
        candidate = error.code
        if isinstance(candidate, str) and _STABLE_REMOTE_ERROR.fullmatch(candidate):
            error_code = candidate
    if (
        wire_stage not in _REMOTE_FAILURE_WIRE_STAGES
        or not isinstance(transcript_head_sha256, str)
        or _SHA256.fullmatch(transcript_head_sha256) is None
    ):
        _fail(_GENERIC_REMOTE_ERROR)
    unsigned = {
        "schema": REMOTE_FAILURE_SCHEMA,
        "ok": False,
        "wire_stage": wire_stage,
        "error_code": error_code,
        "gate_sha256": gate["gate_sha256"],
        "release_revision": gate["release_revision"],
        "plan_sha256": gate["plan_sha256"],
        "transcript_head_sha256": transcript_head_sha256,
        "secret_material_recorded": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
    if set(receipt) != _REMOTE_FAILURE_FIELDS:
        _fail(_GENERIC_REMOTE_ERROR)
    _require_secret_free(receipt)
    return json.loads(_canonical_bytes(receipt).decode("utf-8"))


PreflightCallback = Callable[
    [Mapping[str, Any], Mapping[str, Any], bytearray],
    Mapping[str, Any],
]
ApplyCallback = Callable[
    [
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        SchemaReconciliationAuthorization | None,
        Mapping[str, Any] | None,
    ],
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


def run_protocol_v2(
    gate: Mapping[str, Any],
    *,
    owner_public_key_ed25519_hex: str,
    owner_public_fingerprint: str,
    preflight_callback: PreflightCallback,
    apply_callback: ApplyCallback,
    post_cleanup_callback: PostCleanupCallback,
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
    now: Callable[[], int] = lambda: int(time.time()),
) -> Mapping[str, Any]:
    """Run the exact three-owner-frame v2 protocol.

    ``apply_callback`` receives ``(gate, A1, P1, A2, core_authorization,
    owner_frame_receipt)``.  The last two values are present only when G0's
    journal head is ``empty``; recovery callbacks must use durable core state.
    """

    if (
        not callable(preflight_callback)
        or not callable(apply_callback)
        or not callable(post_cleanup_callback)
        or not callable(now)
    ):
        _fail("schema_reconciliation_callbacks_invalid")
    source = sys.stdin.buffer if input_stream is None else input_stream
    sink = sys.stdout.buffer if output_stream is None else output_stream
    try:
        current = now()
    except BaseException as exc:
        raise SchemaReconciliationBootstrapError(
            "schema_reconciliation_clock_invalid"
        ) from exc
    validated_gate = validate_gate(
        gate,
        owner_public_key_ed25519_hex=owner_public_key_ed25519_hex,
        owner_public_fingerprint=owner_public_fingerprint,
        now_unix=current,
    )

    # This flush is the only authorization for the owner transport to create
    # and transmit the temporary-admin frame.  Failures before a complete G0
    # cannot be bound to an owner-visible transcript and therefore remain EOF.
    _emit_mapping(sink, validated_gate)

    credential: bytearray | None = None
    wire_stage = "a1_to_p1"
    transcript_head_sha256 = str(validated_gate["gate_sha256"])
    output_unreliable = False

    def emit_protocol_mapping(value: Mapping[str, Any]) -> None:
        nonlocal output_unreliable
        try:
            _emit_mapping(sink, value)
        except BaseException:
            # A failed write may already have emitted a partial JSON line.
            # Never append a second record to an unreliable wire prefix.
            output_unreliable = True
            raise

    try:
        admin_frame = _read_mapping_frame(
            source,
            magic=EXECUTOR_PREFLIGHT_MAGIC,
            code="schema_reconciliation_admin_preflight_frame_invalid",
        )
        validated_admin = _validate_admin_preflight(
            admin_frame,
            gate=validated_gate,
            now_unix=now(),
        )

        # No credential byte is read until the signed frame and full Cloud
        # authority receipt above have passed.
        credential = _read_exact_mutable(
            source,
            OPAQUE_CREDENTIAL_BYTES,
            code="schema_reconciliation_admin_credential_invalid",
        )
        try:
            collection = preflight_callback(
                validated_gate,
                validated_admin,
                credential,
            )
        finally:
            _zeroize(credential)
        if not isinstance(collection, Mapping):
            _fail("schema_reconciliation_preflight_collection_invalid")
        challenge = _build_preflight_challenge(
            gate=validated_gate,
            admin_preflight=validated_admin,
            collection=collection,
            issued_at_unix=now(),
        )
        emit_protocol_mapping(challenge)
        wire_stage = "a2_to_i2"
        transcript_head_sha256 = str(challenge["preflight_challenge_sha256"])

        authorization_frame = _read_mapping_frame(
            source,
            magic=PREFLIGHT_AUTHORIZATION_MAGIC,
            code="schema_reconciliation_preflight_authorization_frame_invalid",
        )
        validated_authorization = _validate_preflight_authorization(
            authorization_frame,
            gate=validated_gate,
            admin_preflight=validated_admin,
            challenge=challenge,
            now_unix=now(),
        )
        core_authorization, owner_frame_receipt = _build_core_admission(
            gate=validated_gate,
            challenge=challenge,
            authorization=validated_authorization,
        )

        apply_result = apply_callback(
            validated_gate,
            validated_admin,
            challenge,
            validated_authorization,
            core_authorization,
            owner_frame_receipt,
        )
        if not isinstance(apply_result, Mapping):
            _fail("schema_reconciliation_apply_result_invalid")
        intermediate = _build_database_intermediate(
            gate=validated_gate,
            admin_preflight=validated_admin,
            challenge=challenge,
            authorization=validated_authorization,
            core_authorization=core_authorization,
            owner_frame_receipt=owner_frame_receipt,
            apply_result=apply_result,
            applied_at_unix=now(),
        )
        emit_protocol_mapping(intermediate)
        wire_stage = "c3_to_t3"
        transcript_head_sha256 = str(intermediate["database_intermediate_sha256"])

        cleanup_frame = _read_mapping_frame(
            source,
            magic=EXECUTOR_CLEANUP_MAGIC,
            code="schema_reconciliation_admin_cleanup_frame_invalid",
        )
        if source.read(1) != b"":
            _fail("schema_reconciliation_admin_cleanup_frame_invalid")
        validated_cleanup = _validate_admin_cleanup(
            cleanup_frame,
            gate=validated_gate,
            admin_preflight=validated_admin,
            challenge=challenge,
            authorization=validated_authorization,
            intermediate=intermediate,
            now_unix=now(),
        )
        post_cleanup_value = post_cleanup_callback(
            validated_gate,
            challenge,
            intermediate,
            validated_cleanup,
        )
        post_cleanup = _validate_post_cleanup_observation(
            post_cleanup_value,
            gate=validated_gate,
            challenge=challenge,
            intermediate=intermediate,
            cleanup=validated_cleanup,
            now_unix=now(),
        )
        terminal = _build_terminal(
            gate=validated_gate,
            admin_preflight=validated_admin,
            challenge=challenge,
            authorization=validated_authorization,
            intermediate=intermediate,
            cleanup=validated_cleanup,
            post_cleanup=post_cleanup,
            completed_at_unix=now(),
        )
        terminal = validate_terminal_for_owner(
            terminal,
            gate=validated_gate,
            admin_preflight=validated_admin,
            challenge=challenge,
            authorization=validated_authorization,
            intermediate=intermediate,
            cleanup=validated_cleanup,
            now_unix=terminal["completed_at_unix"],
        )
        emit_protocol_mapping(terminal)
        return terminal
    except BaseException as error:
        if not output_unreliable:
            try:
                _emit_mapping(
                    sink,
                    _remote_failure_receipt(
                        gate=validated_gate,
                        wire_stage=wire_stage,
                        transcript_head_sha256=transcript_head_sha256,
                        error=error,
                    ),
                )
            except BaseException:
                # The original failure remains primary.  In particular, an
                # output failure must not trigger another write attempt.
                pass
        raise
    finally:
        _zeroize(credential)


def _require_root_linux() -> None:
    geteuid = getattr(os, "geteuid", None)
    if not sys.platform.startswith("linux") or not callable(geteuid) or geteuid() != 0:
        _fail("schema_reconciliation_root_linux_required")


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate the single packaged ``run`` action to the sealed runtime."""

    if argv is None:
        argv = sys.argv[1:]
    try:
        if list(argv) != ["run"]:
            _fail("schema_reconciliation_bootstrap_arguments_forbidden")
        _require_root_linux()
        runtime = importlib.import_module(
            "gateway.canonical_writer_schema_reconciliation_runtime"
        )
        runtime_run = getattr(runtime, "run", None)
        if not callable(runtime_run):
            _fail("schema_reconciliation_bootstrap_runtime_invalid")
        terminal = runtime_run()
        if not isinstance(terminal, Mapping):
            _fail("schema_reconciliation_bootstrap_runtime_invalid")
    except BaseException:
        print("schema reconciliation bootstrap failed closed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADMIN_CLEANUP_FRAME_SCHEMA",
    "ADMIN_CLEANUP_MAGIC",
    "ADMIN_CLEANUP_OWNER_SSHSIG_NAMESPACE",
    "ADMIN_PREFLIGHT_FRAME_SCHEMA",
    "ADMIN_PREFLIGHT_MAGIC",
    "ADMIN_PREFLIGHT_OWNER_SSHSIG_NAMESPACE",
    "CLOUD_ADMIN_ABSENCE_SCHEMA",
    "CLOUD_ADMIN_AUTHORITY_SCHEMA",
    "DATABASE_COMMIT_ATTESTATION_SCHEMA",
    "DATABASE_INTERMEDIATE_SCHEMA",
    "EXPECTED_DATABASE",
    "EXPECTED_POSTGRESQL_MAJOR",
    "EXPECTED_PYTHON_VERSION",
    "EXPECTED_PROJECT",
    "EXPECTED_SQL_INSTANCE",
    "GATE_SCHEMA",
    "JOURNAL_HEAD_SCHEMA",
    "MAX_GATE_TTL_SECONDS",
    "MAX_OWNER_FRAME_TTL_SECONDS",
    "MIN_CLOUD_ABSENCE_QUIET_WINDOW_SECONDS",
    "OPAQUE_CREDENTIAL_BYTES",
    "OWNER_ADMIN_CLEANUP_SCHEMA",
    "OWNER_ADMIN_PREFLIGHT_SCHEMA",
    "OWNER_PREFLIGHT_AUTHORIZATION_SCHEMA",
    "SCHEMA_RECONCILIATION_DATABASE_ROLES",
    "ApplyCallback",
    "PostCleanupCallback",
    "PreflightCallback",
    "POST_CLEANUP_OBSERVATION_SCHEMA",
    "PREFLIGHT_AUTHORIZATION_FRAME_SCHEMA",
    "PREFLIGHT_AUTHORIZATION_MAGIC",
    "PREFLIGHT_AUTHORIZATION_OWNER_SSHSIG_NAMESPACE",
    "PREFLIGHT_CHALLENGE_SCHEMA",
    "REMOTE_FAILURE_SCHEMA",
    "SchemaReconciliationBootstrapError",
    "TEMPORARY_OWNER_BRIDGE_SCHEMA",
    "TERMINAL_SCHEMA",
    "admin_cleanup_signature_payload",
    "admin_preflight_signature_payload",
    "main",
    "preflight_authorization_signature_payload",
    "run_protocol_v2",
    "validate_database_intermediate_for_owner",
    "validate_gate",
    "validate_preflight_challenge_for_owner",
    "validate_terminal_for_owner",
]
