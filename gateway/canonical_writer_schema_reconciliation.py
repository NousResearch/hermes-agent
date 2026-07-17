"""Release-bound repair for one missing Canonical Brain helper routine.

This module is deliberately smaller than a migration runner.  It accepts only
the exact old contract produced by removing
``_discord_guild_routeback_target_valid(jsonb)`` from a reviewed target
attestation.  Every other routine identity, owner, ACL, role privilege, event
log identity, and private-schema identity must remain byte-for-byte equal.

The database transport is injected.  Its transaction boundary must acquire
the supplied PostgreSQL advisory lock, roll back on an exception, and commit
only after the fixed owner-owned apply routine and post-apply observation
succeed.  Credential transport and Cloud SQL authority stay outside this
mechanical module.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import json
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Mapping, Protocol, Sequence

from gateway.canonical_writer_config_collector import _attestation_projection
from gateway.canonical_writer_db import (
    CANONICAL_PRIVATE_WRITER_TABLES,
    CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY,
    CanonicalEventLogIdentity,
    CanonicalPrivateRelationIdentity,
    CanonicalPrivateSchemaIdentity,
    ManagedCloudSQLAdminHBAReceipt,
    PrivilegeAttestation,
    PrivilegeAttestationError,
    RoutineIdentity,
    SequencePrivilegeGrant,
    TablePrivilegeGrant,
    WriterDBConfig,
    WriterPrivilegePolicy,
    _collect_privilege_attestation,
    _owner_membership_danger_sql,
    validate_privilege_attestation,
)
from gateway.canonical_writer_foundation import (
    SealedSQLArtifact,
    _effective_gid,
    _effective_uid,
    _filesystem_identity,
    _fsync_directory,
    _list_xattrs,
    _load_sealed_artifacts,
    _read_sealed_artifact,
    _same_inode,
    _secure_directory,
)
from gateway.canonical_writer_planner import load_release_manifest
from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
    CANONICAL_WRITER_ROLE,
    CANONICAL_WRITER_SCHEMA,
    EXPECTED_HELPER_ROUTINE_SIGNATURES,
    EXPECTED_ROUTINE_SIGNATURES,
)


RECONCILIATION_PLAN_SCHEMA = "muncho-canonical-writer-schema-reconciliation-plan.v2"
RECONCILIATION_PREFLIGHT_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-preflight.v2"
)
RECONCILIATION_AUTHORIZATION_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-authorization.v1"
)
RECONCILIATION_OWNER_A2_FRAME_SCHEMA = (
    "MSP2-u32be-canonical-json-no-secret.v1"
)
RECONCILIATION_OWNER_FRAME_RECEIPT_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-owner-a2-frame-receipt.v1"
)
RECONCILIATION_OWNER_SIGNATURE_NAMESPACE = (
    "muncho-canonical-writer-schema-reconciliation-"
    "preflight-authorization-owner-v2"
)
RECONCILIATION_AUTHORIZED_INTENT_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-authorized-intent.v3"
)
RECONCILIATION_RECEIPT_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-receipt.v3"
)
SCHEMA_CONTRACT_SCHEMA = "muncho-canonical-writer-schema-contract.v1"
SCHEMA_CONTRACT_ASSET_SCHEMA = (
    "muncho-canonical-writer-schema-contract-asset.v1"
)
CANONICAL_TRUTH_RECEIPT_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-canonical-data.v3"
)
MISSING_HELPER_SIGNATURE = (
    "canonical_brain._discord_guild_routeback_target_valid(jsonb)"
)
DATABASE = "muncho_canary_brain"
BASE_ARTIFACT_NAME = "base_migration"
BASE_ARTIFACT_FILENAME = "canonical_writer_v1.sql"
CONTROL_INSTALL_ARTIFACT_FILENAME = (
    "canonical_writer_schema_reconciliation_control_v1.sql"
)
CONTROL_RETIRE_ARTIFACT_FILENAME = (
    "canonical_writer_schema_reconciliation_control_retire_v1.sql"
)
SCHEMA_CONTRACT_ASSET_RELATIVE_PATH = Path(
    "gateway/assets/canonical_writer_schema_contract_v1.json"
)
POSTGRESQL_MAJOR = 18
EVIDENCE_ROOT = Path(
    "/var/lib/muncho-writer-canary-evidence/schema-reconciliation"
)
CANONICAL_TRUTH_RELATIONS = (
    "public.canonical_event_log",
    *(f"canonical_brain.{name}" for name in CANONICAL_PRIVATE_WRITER_TABLES),
)
CANONICAL_TRUTH_LOCK_SQL = (
    "LOCK TABLE " + ", ".join(CANONICAL_TRUTH_RELATIONS) + " IN SHARE MODE"
)
CANONICAL_QUARANTINE_ANCHORS = (
    "schema:canonical_brain_legacy_quarantine:postgres:owner-only",
    "table:canonical_brain_legacy_quarantine."
    "canonical_event_log_legacy_v1:postgres:r:p:owner-only",
    "table:canonical_brain_legacy_quarantine."
    "reconciliation_receipts:postgres:r:p:owner-only",
)
_CANONICAL_QUARANTINE_ANCHOR_EXPECTATIONS = (
    (CANONICAL_QUARANTINE_ANCHORS[0], "postgres", "n", ""),
    (CANONICAL_QUARANTINE_ANCHORS[1], "postgres", "r", "p"),
    (CANONICAL_QUARANTINE_ANCHORS[2], "postgres", "r", "p"),
)

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_AUTHORIZATION_LIFETIME_SECONDS = 900
_MAX_PREFLIGHT_TO_AUTHORIZATION_SECONDS = 300
_SAFE_SEARCH_PATH = ("search_path=pg_catalog, canonical_brain",)
_EXPECTED_HELPER_PROSRC_SHA256 = (
    "e82ee5b2240d61c1e7c60d76ec87729d9d87e134d4b2083d5cd7b447f5ef093c"
)
_EXPECTED_HELPER_DEFINITION_SHA256 = (
    "2fc8e5334784ae791398033c8b460ccb170324f4c8cc1b3d0387f907e9cf7737"
)
_PLAN_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "database",
        "helper_signature",
        "base_artifact_sha256",
        "target_asset_sha256",
        "postgresql_major",
        "control_install_artifact_sha256",
        "control_retire_artifact_sha256",
        "control_foundation_contract_sha256",
        "helper_catalog_identity_sha256",
        "expected_old_contract_sha256",
        "target_contract_sha256",
        "advisory_lock_key",
        "transaction_isolation",
        "canonical_truth_lock",
        "canonical_truth_preservation_required",
        "plan_sha256",
    }
)
_AUTHORIZED_INTENT_FIELDS = frozenset(
    {
        "schema",
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
        "preflight",
        "owner_authorization_frame",
        "authorization",
        "initial_contract_sha256",
        "initial_canonical_truth",
        "authorization_sha256",
        "preflight_sha256",
        "owner_frame_receipt_sha256",
        "truth_receipt_sha256",
        "mode",
        "mutation_required",
        "admitted_at_unix",
        "authorized_intent_sha256",
    }
)
_PREFLIGHT_FIELDS = frozenset(
    {
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
    }
)
_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "plan_sha256",
        "preflight_sha256",
        "preflight_state",
        "observed_contract_sha256",
        "truth_receipt_sha256",
        "owner_frame_sha256",
        "owner_subject_sha256",
        "owner_key_id",
        "issued_at_unix",
        "expires_at_unix",
        "nonce",
        "authorization_sha256",
    }
)
_OWNER_FRAME_RECEIPT_FIELDS = frozenset(
    {
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
    }
)
_RECEIPT_FIELDS = frozenset(
    {
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
    }
)


class SchemaReconciliationError(RuntimeError):
    """Fail-closed public error without database or credential detail."""

    def __init__(self, code: str) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]{2,95}", code) is None:
            code = "canonical_writer_schema_reconciliation_failed"
        self.code = code
        super().__init__(code)


class _UnpublishedStageDiscarded(RuntimeError):
    """Internal signal that a secure partial stage was removed under lock."""


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
        raise SchemaReconciliationError("schema_reconciliation_value_not_canonical") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _strict_json(value: bytes, code: str) -> Mapping[str, Any]:
    def reject_duplicates(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = item
        return result

    try:
        decoded = value.decode("utf-8", errors="strict")
        parsed = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _item: (_ for _ in ()).throw(
                ValueError("non-JSON constant")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SchemaReconciliationError(code) from exc
    if not isinstance(parsed, Mapping) or _canonical_bytes(parsed) != value:
        raise SchemaReconciliationError(code)
    return parsed


def _is_complete_canonical_json(value: bytes) -> bool:
    """Return true for any complete canonical JSON value, not just records."""

    def reject_duplicates(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = item
        return result

    try:
        decoded = value.decode("utf-8", errors="strict")
        parsed = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _item: (_ for _ in ()).throw(
                ValueError("non-JSON constant")
            ),
        )
        return _canonical_bytes(parsed) == value
    except (
        SchemaReconciliationError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _exact_mapping(
    value: Any,
    fields: frozenset[str],
    code: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SchemaReconciliationError(code)
    return value


def _exact_sequence(value: Any, code: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        raise SchemaReconciliationError(code)
    return value


def _exact_text(value: Any, code: str) -> str:
    if not isinstance(value, str):
        raise SchemaReconciliationError(code)
    return value


def _exact_boolean(value: Any, code: str) -> bool:
    if type(value) is not bool:
        raise SchemaReconciliationError(code)
    return value


def _exact_integer(value: Any, code: str) -> int:
    if type(value) is not int:
        raise SchemaReconciliationError(code)
    return value


def _exact_strings(value: Any, code: str) -> tuple[str, ...]:
    items = _exact_sequence(value, code)
    if any(not isinstance(item, str) for item in items):
        raise SchemaReconciliationError(code)
    return tuple(items)


def _table_grants_from_value(value: Any) -> tuple[TablePrivilegeGrant, ...]:
    code = "schema_reconciliation_contract_invalid"
    result: list[TablePrivilegeGrant] = []
    for raw in _exact_sequence(value, code):
        item = _exact_mapping(raw, frozenset({"table", "privileges"}), code)
        result.append(
            TablePrivilegeGrant(
                table=_exact_text(item["table"], code),
                privileges=_exact_strings(item["privileges"], code),
            )
        )
    return tuple(result)


def _sequence_grants_from_value(
    value: Any,
) -> tuple[SequencePrivilegeGrant, ...]:
    code = "schema_reconciliation_contract_invalid"
    result: list[SequencePrivilegeGrant] = []
    for raw in _exact_sequence(value, code):
        item = _exact_mapping(raw, frozenset({"sequence", "privileges"}), code)
        result.append(
            SequencePrivilegeGrant(
                sequence=_exact_text(item["sequence"], code),
                privileges=_exact_strings(item["privileges"], code),
            )
        )
    return tuple(result)


def _routine_identities_from_value(value: Any) -> tuple[RoutineIdentity, ...]:
    code = "schema_reconciliation_contract_invalid"
    fields = frozenset(
        {
            "signature",
            "owner",
            "security_definer",
            "language",
            "configuration",
            "definition_sha256",
            "owner_dangerous",
        }
    )
    result: list[RoutineIdentity] = []
    for raw in _exact_sequence(value, code):
        item = _exact_mapping(raw, fields, code)
        result.append(
            RoutineIdentity(
                signature=_exact_text(item["signature"], code),
                owner=_exact_text(item["owner"], code),
                security_definer=_exact_boolean(
                    item["security_definer"], code
                ),
                language=_exact_text(item["language"], code),
                configuration=_exact_strings(item["configuration"], code),
                definition_sha256=_exact_text(
                    item["definition_sha256"], code
                ),
                owner_dangerous=_exact_boolean(
                    item["owner_dangerous"], code
                ),
            )
        )
    return tuple(result)


def _event_log_identity_from_value(value: Any) -> CanonicalEventLogIdentity:
    code = "schema_reconciliation_contract_invalid"
    fields = frozenset(
        {
            "table",
            "owner",
            "owner_dangerous",
            "relation_kind",
            "persistence",
            "is_partition",
            "access_method",
            "tablespace_oid",
            "row_security",
            "force_row_security",
            "replica_identity",
            "relation_options",
            "columns",
            "constraints",
            "user_triggers",
            "rewrite_rules",
            "policies",
            "inheritance",
            "non_owner_acl_grants",
            "index_count",
            "primary_index_exact",
        }
    )
    item = _exact_mapping(value, fields, code)
    return CanonicalEventLogIdentity(
        table=_exact_text(item["table"], code),
        owner=_exact_text(item["owner"], code),
        owner_dangerous=_exact_boolean(item["owner_dangerous"], code),
        relation_kind=_exact_text(item["relation_kind"], code),
        persistence=_exact_text(item["persistence"], code),
        is_partition=_exact_boolean(item["is_partition"], code),
        access_method=_exact_text(item["access_method"], code),
        tablespace_oid=_exact_integer(item["tablespace_oid"], code),
        row_security=_exact_boolean(item["row_security"], code),
        force_row_security=_exact_boolean(item["force_row_security"], code),
        replica_identity=_exact_text(item["replica_identity"], code),
        relation_options=_exact_strings(item["relation_options"], code),
        columns=_exact_strings(item["columns"], code),
        constraints=_exact_strings(item["constraints"], code),
        user_triggers=_exact_strings(item["user_triggers"], code),
        rewrite_rules=_exact_strings(item["rewrite_rules"], code),
        policies=_exact_strings(item["policies"], code),
        inheritance=_exact_boolean(item["inheritance"], code),
        non_owner_acl_grants=_exact_strings(
            item["non_owner_acl_grants"], code
        ),
        index_count=_exact_integer(item["index_count"], code),
        primary_index_exact=_exact_boolean(
            item["primary_index_exact"], code
        ),
    )


def _private_relation_from_value(
    value: Any,
) -> CanonicalPrivateRelationIdentity:
    code = "schema_reconciliation_contract_invalid"
    fields = frozenset(
        {
            "name",
            "owner",
            "owner_dangerous",
            "relation_kind",
            "persistence",
            "is_partition",
            "access_method",
            "tablespace_oid",
            "row_security",
            "force_row_security",
            "replica_identity",
            "relation_options",
            "columns",
            "constraints",
            "indexes",
            "index_owners",
            "user_triggers",
            "rewrite_rules",
            "policies",
            "inheritance",
        }
    )
    item = _exact_mapping(value, fields, code)
    return CanonicalPrivateRelationIdentity(
        name=_exact_text(item["name"], code),
        owner=_exact_text(item["owner"], code),
        owner_dangerous=_exact_boolean(item["owner_dangerous"], code),
        relation_kind=_exact_text(item["relation_kind"], code),
        persistence=_exact_text(item["persistence"], code),
        is_partition=_exact_boolean(item["is_partition"], code),
        access_method=_exact_text(item["access_method"], code),
        tablespace_oid=_exact_integer(item["tablespace_oid"], code),
        row_security=_exact_boolean(item["row_security"], code),
        force_row_security=_exact_boolean(item["force_row_security"], code),
        replica_identity=_exact_text(item["replica_identity"], code),
        relation_options=_exact_strings(item["relation_options"], code),
        columns=_exact_strings(item["columns"], code),
        constraints=_exact_strings(item["constraints"], code),
        indexes=_exact_strings(item["indexes"], code),
        index_owners=_exact_strings(item["index_owners"], code),
        user_triggers=_exact_strings(item["user_triggers"], code),
        rewrite_rules=_exact_strings(item["rewrite_rules"], code),
        policies=_exact_strings(item["policies"], code),
        inheritance=_exact_boolean(item["inheritance"], code),
    )


def _private_schema_from_value(value: Any) -> CanonicalPrivateSchemaIdentity:
    code = "schema_reconciliation_contract_invalid"
    item = _exact_mapping(
        value,
        frozenset({"schema", "owner", "owner_dangerous", "relations"}),
        code,
    )
    return CanonicalPrivateSchemaIdentity(
        schema=_exact_text(item["schema"], code),
        owner=_exact_text(item["owner"], code),
        owner_dangerous=_exact_boolean(item["owner_dangerous"], code),
        relations=tuple(
            _private_relation_from_value(raw)
            for raw in _exact_sequence(item["relations"], code)
        ),
    )


def _attestation_from_value(value: Any) -> PrivilegeAttestation:
    code = "schema_reconciliation_contract_invalid"
    fields = frozenset(
        {
            "role",
            "dangerous_attributes",
            "table_grants",
            "sequence_grants",
            "executable_routines",
            "routine_identities",
            "helper_routine_identities",
            "schema_privileges",
            "database_privileges",
            "role_memberships",
            "unexpected_privileges",
            "public_acl_grants",
            "canonical_non_owner_acl_grants",
            "canonical_writer_role_inheritors",
            "canonical_event_log_identity",
            "canonical_private_schema_identity",
        }
    )
    item = _exact_mapping(value, fields, code)
    dangerous = _exact_mapping(
        item["dangerous_attributes"],
        frozenset(
            {
                "superuser",
                "createdb",
                "createrole",
                "replication",
                "bypassrls",
                "table_owner",
                "routine_owner",
            }
        ),
        code,
    )
    try:
        attestation = PrivilegeAttestation(
            role=_exact_text(item["role"], code),
            superuser=_exact_boolean(dangerous["superuser"], code),
            createdb=_exact_boolean(dangerous["createdb"], code),
            createrole=_exact_boolean(dangerous["createrole"], code),
            replication=_exact_boolean(dangerous["replication"], code),
            bypassrls=_exact_boolean(dangerous["bypassrls"], code),
            table_owner=_exact_boolean(dangerous["table_owner"], code),
            routine_owner=_exact_boolean(dangerous["routine_owner"], code),
            table_grants=_table_grants_from_value(item["table_grants"]),
            sequence_grants=_sequence_grants_from_value(
                item["sequence_grants"]
            ),
            executable_routines=_exact_strings(
                item["executable_routines"], code
            ),
            routine_identities=_routine_identities_from_value(
                item["routine_identities"]
            ),
            dependency_routine_identities=_routine_identities_from_value(
                item["helper_routine_identities"]
            ),
            schema_privileges=_exact_strings(item["schema_privileges"], code),
            database_privileges=_exact_strings(
                item["database_privileges"], code
            ),
            role_memberships=_exact_strings(item["role_memberships"], code),
            unexpected_privileges=_exact_strings(
                item["unexpected_privileges"], code
            ),
            public_acl_grants=_exact_strings(item["public_acl_grants"], code),
            canonical_non_owner_acl_grants=_exact_strings(
                item["canonical_non_owner_acl_grants"], code
            ),
            canonical_writer_role_inheritors=_exact_strings(
                item["canonical_writer_role_inheritors"], code
            ),
            canonical_event_log_identity=_event_log_identity_from_value(
                item["canonical_event_log_identity"]
            ),
            canonical_private_schema_identity=_private_schema_from_value(
                item["canonical_private_schema_identity"]
            ),
        )
    except SchemaReconciliationError:
        raise
    except (TypeError, ValueError) as exc:
        raise SchemaReconciliationError(code) from exc
    if _canonical_bytes(value) != _canonical_bytes(
        _attestation_projection(attestation)
    ):
        raise SchemaReconciliationError(code)
    return attestation


def _target_policy(attestation: PrivilegeAttestation) -> WriterPrivilegePolicy:
    private = attestation.canonical_private_schema_identity
    if private is None:
        raise SchemaReconciliationError("schema_reconciliation_private_schema_missing")
    try:
        return WriterPrivilegePolicy(
            schema=CANONICAL_WRITER_SCHEMA,
            table_grants=attestation.table_grants,
            sequence_grants=attestation.sequence_grants,
            executable_routines=attestation.executable_routines,
            routine_identities=attestation.routine_identities,
            dependency_routine_identities=attestation.dependency_routine_identities,
            schema_privileges=attestation.schema_privileges,
            database_privileges=attestation.database_privileges,
            role_memberships=attestation.role_memberships,
            canonical_owner_role=CANONICAL_WRITER_MIGRATION_OWNER,
            canonical_acl_grantee_role=CANONICAL_WRITER_ROLE,
            private_schema_identity_sha256=private.sha256,
        )
    except (TypeError, ValueError) as exc:
        raise SchemaReconciliationError("schema_reconciliation_contract_invalid") from exc


@dataclass(frozen=True)
class HelperRoutineCatalogIdentity:
    """Exact PostgreSQL 18 catalog identity of the one reviewed helper."""

    signature: str
    owner: str
    owner_dangerous: bool
    kind: str
    security_definer: bool
    volatility: str
    parallel: str
    leakproof: bool
    strict: bool
    returns_set: bool
    language: str
    argument_types: tuple[str, ...]
    return_type: str
    configuration: tuple[str, ...]
    prosrc_sha256: str
    definition_sha256: str
    non_owner_acl_grants: tuple[str, ...] = ()
    writer_role_execute: bool = False
    writer_login_execute: bool = False

    def __post_init__(self) -> None:
        if (
            self.signature != MISSING_HELPER_SIGNATURE
            or self.owner != CANONICAL_WRITER_MIGRATION_OWNER
            or self.owner_dangerous
            or self.kind != "f"
            or self.security_definer
            or self.volatility != "i"
            or self.parallel != "u"
            or self.leakproof
            or self.strict
            or self.returns_set
            or self.language != "sql"
            or self.argument_types != ("jsonb",)
            or self.return_type != "boolean"
            or self.configuration != _SAFE_SEARCH_PATH
            or self.prosrc_sha256 != _EXPECTED_HELPER_PROSRC_SHA256
            or self.definition_sha256 != _EXPECTED_HELPER_DEFINITION_SHA256
            or self.non_owner_acl_grants
            or self.writer_role_execute
            or self.writer_login_execute
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_helper_catalog_identity_invalid"
            )

    @property
    def value(self) -> Mapping[str, Any]:
        return {
            "signature": self.signature,
            "owner": self.owner,
            "owner_dangerous": self.owner_dangerous,
            "kind": self.kind,
            "security_definer": self.security_definer,
            "volatility": self.volatility,
            "parallel": self.parallel,
            "leakproof": self.leakproof,
            "strict": self.strict,
            "returns_set": self.returns_set,
            "language": self.language,
            "argument_types": list(self.argument_types),
            "return_type": self.return_type,
            "configuration": list(self.configuration),
            "prosrc_sha256": self.prosrc_sha256,
            "definition_sha256": self.definition_sha256,
            "non_owner_acl_grants": list(self.non_owner_acl_grants),
            "writer_role_execute": self.writer_role_execute,
            "writer_login_execute": self.writer_login_execute,
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.value)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "HelperRoutineCatalogIdentity":
        code = "schema_reconciliation_helper_catalog_identity_invalid"
        fields = frozenset(
            {
                "signature",
                "owner",
                "owner_dangerous",
                "kind",
                "security_definer",
                "volatility",
                "parallel",
                "leakproof",
                "strict",
                "returns_set",
                "language",
                "argument_types",
                "return_type",
                "configuration",
                "prosrc_sha256",
                "definition_sha256",
                "non_owner_acl_grants",
                "writer_role_execute",
                "writer_login_execute",
            }
        )
        raw = _exact_mapping(value, fields, code)
        try:
            identity = cls(
                signature=_exact_text(raw["signature"], code),
                owner=_exact_text(raw["owner"], code),
                owner_dangerous=_exact_boolean(raw["owner_dangerous"], code),
                kind=_exact_text(raw["kind"], code),
                security_definer=_exact_boolean(
                    raw["security_definer"], code
                ),
                volatility=_exact_text(raw["volatility"], code),
                parallel=_exact_text(raw["parallel"], code),
                leakproof=_exact_boolean(raw["leakproof"], code),
                strict=_exact_boolean(raw["strict"], code),
                returns_set=_exact_boolean(raw["returns_set"], code),
                language=_exact_text(raw["language"], code),
                argument_types=_exact_strings(raw["argument_types"], code),
                return_type=_exact_text(raw["return_type"], code),
                configuration=_exact_strings(raw["configuration"], code),
                prosrc_sha256=_exact_text(raw["prosrc_sha256"], code),
                definition_sha256=_exact_text(
                    raw["definition_sha256"], code
                ),
                non_owner_acl_grants=_exact_strings(
                    raw["non_owner_acl_grants"], code
                ),
                writer_role_execute=_exact_boolean(
                    raw["writer_role_execute"], code
                ),
                writer_login_execute=_exact_boolean(
                    raw["writer_login_execute"], code
                ),
            )
        except SchemaReconciliationError:
            raise
        except (TypeError, ValueError) as exc:
            raise SchemaReconciliationError(code) from exc
        if _canonical_bytes(value) != _canonical_bytes(identity.value):
            raise SchemaReconciliationError(code)
        return identity


EXPECTED_MISSING_HELPER_CATALOG_IDENTITY = HelperRoutineCatalogIdentity(
    signature=MISSING_HELPER_SIGNATURE,
    owner=CANONICAL_WRITER_MIGRATION_OWNER,
    owner_dangerous=False,
    kind="f",
    security_definer=False,
    volatility="i",
    parallel="u",
    leakproof=False,
    strict=False,
    returns_set=False,
    language="sql",
    argument_types=("jsonb",),
    return_type="boolean",
    configuration=_SAFE_SEARCH_PATH,
    prosrc_sha256=_EXPECTED_HELPER_PROSRC_SHA256,
    definition_sha256=_EXPECTED_HELPER_DEFINITION_SHA256,
)


@dataclass(frozen=True)
class CanonicalRelationTruthReceipt:
    """Deterministic multiset identity for one fixed canonical relation."""

    relation: str
    row_count: int
    chunk_count: int
    chunk_manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            self.relation not in CANONICAL_TRUTH_RELATIONS
            or type(self.row_count) is not int
            or self.row_count < 0
            or type(self.chunk_count) is not int
            or self.chunk_count < 0
            or self.chunk_count != (
                0 if self.row_count == 0 else (self.row_count - 1) // 4096 + 1
            )
            or not isinstance(self.chunk_manifest_sha256, str)
            or _SHA256.fullmatch(self.chunk_manifest_sha256) is None
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_invalid"
            )

    @property
    def value(self) -> Mapping[str, Any]:
        return {
            "relation": self.relation,
            "row_count": self.row_count,
            "chunk_count": self.chunk_count,
            "chunk_manifest_sha256": self.chunk_manifest_sha256,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "CanonicalRelationTruthReceipt":
        if not isinstance(value, Mapping) or set(value) != {
            "relation",
            "row_count",
            "chunk_count",
            "chunk_manifest_sha256",
        }:
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_invalid"
            )
        receipt = cls(
            relation=value.get("relation"),
            row_count=value.get("row_count"),
            chunk_count=value.get("chunk_count"),
            chunk_manifest_sha256=value.get("chunk_manifest_sha256"),
        )
        if _canonical_bytes(value) != _canonical_bytes(receipt.value):
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_invalid"
            )
        return receipt


@dataclass(frozen=True)
class CanonicalQuarantineAnchorReceipt:
    """Database-observed identity for one immutable quarantine anchor."""

    anchor: str
    object_oid: int
    owner: str
    kind: str
    persistence: str
    acl_sha256: str

    def __post_init__(self) -> None:
        expected = next(
            (
                item
                for item in _CANONICAL_QUARANTINE_ANCHOR_EXPECTATIONS
                if item[0] == self.anchor
            ),
            None,
        )
        if (
            expected is None
            or type(self.object_oid) is not int
            or self.object_oid <= 0
            or (self.owner, self.kind, self.persistence) != expected[1:]
            or not isinstance(self.acl_sha256, str)
            or _SHA256.fullmatch(self.acl_sha256) is None
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_quarantine_anchor_invalid"
            )

    @property
    def value(self) -> Mapping[str, Any]:
        return {
            "anchor": self.anchor,
            "object_oid": self.object_oid,
            "owner": self.owner,
            "kind": self.kind,
            "persistence": self.persistence,
            "acl_sha256": self.acl_sha256,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "CanonicalQuarantineAnchorReceipt":
        if not isinstance(value, Mapping) or set(value) != {
            "anchor",
            "object_oid",
            "owner",
            "kind",
            "persistence",
            "acl_sha256",
        }:
            raise SchemaReconciliationError(
                "schema_reconciliation_quarantine_anchor_invalid"
            )
        receipt = cls(
            anchor=value.get("anchor"),
            object_oid=value.get("object_oid"),
            owner=value.get("owner"),
            kind=value.get("kind"),
            persistence=value.get("persistence"),
            acl_sha256=value.get("acl_sha256"),
        )
        if _canonical_bytes(value) != _canonical_bytes(receipt.value):
            raise SchemaReconciliationError(
                "schema_reconciliation_quarantine_anchor_invalid"
            )
        return receipt


@dataclass(frozen=True)
class CanonicalTruthReceipt:
    """Canonical data plus DB-derived immutable quarantine identities."""

    row_count: int
    canonical14_sha256: str
    relation_receipts: tuple[CanonicalRelationTruthReceipt, ...]
    quarantine_anchor_receipts: tuple[CanonicalQuarantineAnchorReceipt, ...]

    def __post_init__(self) -> None:
        relations = self.relation_receipts
        if (
            type(self.row_count) is not int
            or self.row_count < 0
            or not isinstance(self.canonical14_sha256, str)
            or _SHA256.fullmatch(self.canonical14_sha256) is None
            or not isinstance(relations, tuple)
            or not all(
                isinstance(item, CanonicalRelationTruthReceipt)
                for item in relations
            )
            or tuple(item.relation for item in relations)
            != CANONICAL_TRUTH_RELATIONS
            or relations[0].row_count != self.row_count
            or not isinstance(self.quarantine_anchor_receipts, tuple)
            or not all(
                isinstance(item, CanonicalQuarantineAnchorReceipt)
                for item in self.quarantine_anchor_receipts
            )
            or tuple(
                item.anchor for item in self.quarantine_anchor_receipts
            )
            != CANONICAL_QUARANTINE_ANCHORS
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_invalid"
            )
        object.__setattr__(self, "relation_receipts", relations)

    @property
    def canonical_data_row_count(self) -> int:
        return sum(item.row_count for item in self.relation_receipts)

    @property
    def canonical_data_sha256(self) -> str:
        return _sha256_json([item.value for item in self.relation_receipts])

    @property
    def quarantine_anchors_sha256(self) -> str:
        return _sha256_json(
            [item.value for item in self.quarantine_anchor_receipts]
        )

    @property
    def value(self) -> Mapping[str, Any]:
        unsigned = {
            "schema": CANONICAL_TRUTH_RECEIPT_SCHEMA,
            "table": "public.canonical_event_log",
            "row_count": self.row_count,
            "canonical14_sha256": self.canonical14_sha256,
            "relation_count": len(self.relation_receipts),
            "canonical_data_row_count": self.canonical_data_row_count,
            "canonical_data_sha256": self.canonical_data_sha256,
            "quarantine_anchors": [
                item.value for item in self.quarantine_anchor_receipts
            ],
            "quarantine_anchors_sha256": self.quarantine_anchors_sha256,
            "relation_receipts": [
                item.value for item in self.relation_receipts
            ],
        }
        return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}

    @property
    def sha256(self) -> str:
        return str(self.value["receipt_sha256"])

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CanonicalTruthReceipt":
        fields = {
            "schema",
            "table",
            "row_count",
            "canonical14_sha256",
            "relation_count",
            "canonical_data_row_count",
            "canonical_data_sha256",
            "quarantine_anchors",
            "quarantine_anchors_sha256",
            "relation_receipts",
            "receipt_sha256",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value.get("schema") != CANONICAL_TRUTH_RECEIPT_SCHEMA
            or value.get("table") != "public.canonical_event_log"
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_invalid"
            )
        relations_raw = value.get("relation_receipts")
        quarantine_raw = value.get("quarantine_anchors")
        if not isinstance(relations_raw, list) or not isinstance(
            quarantine_raw, list
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_invalid"
            )
        receipt = cls(
            row_count=value.get("row_count"),
            canonical14_sha256=value.get("canonical14_sha256"),
            relation_receipts=tuple(
                CanonicalRelationTruthReceipt.from_mapping(item)
                for item in relations_raw
            ),
            quarantine_anchor_receipts=tuple(
                CanonicalQuarantineAnchorReceipt.from_mapping(item)
                for item in quarantine_raw
            ),
        )
        if _canonical_bytes(value) != _canonical_bytes(receipt.value):
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_invalid"
            )
        return receipt


@dataclass(frozen=True)
class SchemaReconciliationAuthorization:
    """One short-lived owner authorization for one exact preflight head."""

    value: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "SchemaReconciliationAuthorization":
        code = "schema_reconciliation_authorization_invalid"
        if not isinstance(value, Mapping) or set(value) != _AUTHORIZATION_FIELDS:
            raise SchemaReconciliationError(code)
        raw = copy.deepcopy(dict(value))
        unsigned = {
            key: item for key, item in raw.items() if key != "authorization_sha256"
        }
        digest_fields = (
            "plan_sha256",
            "preflight_sha256",
            "observed_contract_sha256",
            "truth_receipt_sha256",
            "owner_frame_sha256",
            "owner_subject_sha256",
            "owner_key_id",
            "nonce",
            "authorization_sha256",
        )
        if (
            raw.get("schema") != RECONCILIATION_AUTHORIZATION_SCHEMA
            or not isinstance(raw.get("release_revision"), str)
            or _REVISION.fullmatch(str(raw.get("release_revision"))) is None
            or any(
                not isinstance(raw.get(name), str)
                or _SHA256.fullmatch(str(raw.get(name))) is None
                for name in digest_fields
            )
            or raw.get("preflight_state")
            not in {"exact_old_missing_one_helper", "exact_target"}
            or type(raw.get("issued_at_unix")) is not int
            or type(raw.get("expires_at_unix")) is not int
            or raw["issued_at_unix"] < 0
            or not raw["issued_at_unix"] < raw["expires_at_unix"]
            or not 1
            <= raw["expires_at_unix"] - raw["issued_at_unix"]
            <= _MAX_AUTHORIZATION_LIFETIME_SECONDS
            or raw.get("authorization_sha256") != _sha256_json(unsigned)
        ):
            raise SchemaReconciliationError(code)
        return cls(json.loads(_canonical_bytes(raw).decode("utf-8")))

    @classmethod
    def build(
        cls,
        *,
        plan: "SchemaReconciliationPlan",
        preflight: Mapping[str, Any],
        truth: CanonicalTruthReceipt,
        owner_frame_sha256: str,
        owner_subject_sha256: str,
        owner_key_id: str,
        issued_at_unix: int,
        expires_at_unix: int,
        nonce: str,
    ) -> "SchemaReconciliationAuthorization":
        _validate_preflight(plan, preflight)
        if (
            not isinstance(truth, CanonicalTruthReceipt)
            or preflight["truth_receipt_sha256"] != truth.sha256
            or type(issued_at_unix) is not int
            or not preflight["observed_at_unix"] <= issued_at_unix
            or issued_at_unix - preflight["observed_at_unix"]
            > _MAX_PREFLIGHT_TO_AUTHORIZATION_SECONDS
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_authorization_binding_invalid"
            )
        unsigned = {
            "schema": RECONCILIATION_AUTHORIZATION_SCHEMA,
            "release_revision": plan.revision,
            "plan_sha256": plan.sha256,
            "preflight_sha256": preflight["preflight_sha256"],
            "preflight_state": preflight["state"],
            "observed_contract_sha256": preflight[
                "observed_contract_sha256"
            ],
            "truth_receipt_sha256": truth.sha256,
            "owner_frame_sha256": owner_frame_sha256,
            "owner_subject_sha256": owner_subject_sha256,
            "owner_key_id": owner_key_id,
            "issued_at_unix": issued_at_unix,
            "expires_at_unix": expires_at_unix,
            "nonce": nonce,
        }
        return cls.from_mapping(
            {
                **unsigned,
                "authorization_sha256": _sha256_json(unsigned),
            }
        )

    @property
    def sha256(self) -> str:
        return str(self.value["authorization_sha256"])

    def validate_binding(
        self,
        *,
        plan: "SchemaReconciliationPlan",
        preflight: Mapping[str, Any],
        truth: CanonicalTruthReceipt,
    ) -> None:
        canonical = type(self).from_mapping(self.value)
        value = canonical.value
        _validate_preflight(plan, preflight)
        if (
            value["release_revision"] != plan.revision
            or value["plan_sha256"] != plan.sha256
            or value["preflight_sha256"] != preflight["preflight_sha256"]
            or value["preflight_state"] != preflight["state"]
            or value["observed_contract_sha256"]
            != preflight["observed_contract_sha256"]
            or value["truth_receipt_sha256"] != truth.sha256
            or preflight["truth_receipt_sha256"] != truth.sha256
            or not preflight["observed_at_unix"]
            <= value["issued_at_unix"]
            or value["issued_at_unix"] - preflight["observed_at_unix"]
            > _MAX_PREFLIGHT_TO_AUTHORIZATION_SECONDS
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_authorization_binding_invalid"
            )


def _validate_owner_authorization_frame(
    plan: "SchemaReconciliationPlan",
    *,
    preflight: Mapping[str, Any],
    truth: CanonicalTruthReceipt,
    authorization: SchemaReconciliationAuthorization,
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    code = "schema_reconciliation_owner_frame_invalid"
    if (
        not isinstance(value, Mapping)
        or set(value) != _OWNER_FRAME_RECEIPT_FIELDS
        or not isinstance(authorization, SchemaReconciliationAuthorization)
    ):
        raise SchemaReconciliationError(code)
    authorization.validate_binding(
        plan=plan,
        preflight=preflight,
        truth=truth,
    )
    raw = copy.deepcopy(dict(value))
    unsigned = {
        key: item for key, item in raw.items() if key != "receipt_sha256"
    }
    digest_fields = (
        "signed_frame_sha256",
        "signature_sshsig_sha256",
        "plan_sha256",
        "preflight_sha256",
        "observed_contract_sha256",
        "truth_receipt_sha256",
        "authorization_sha256",
        "owner_subject_sha256",
        "owner_key_id",
        "nonce",
        "receipt_sha256",
    )
    if (
        raw.get("schema") != RECONCILIATION_OWNER_FRAME_RECEIPT_SCHEMA
        or raw.get("frame_schema") != RECONCILIATION_OWNER_A2_FRAME_SCHEMA
        or raw.get("action") != "apply_schema_reconciliation"
        or raw.get("approved") is not True
        or raw.get("signature_namespace")
        != RECONCILIATION_OWNER_SIGNATURE_NAMESPACE
        or raw.get("signature_verified") is not True
        or not isinstance(raw.get("release_revision"), str)
        or _REVISION.fullmatch(str(raw.get("release_revision"))) is None
        or any(
            not isinstance(raw.get(name), str)
            or _SHA256.fullmatch(str(raw.get(name))) is None
            for name in digest_fields
        )
        or type(raw.get("issued_at_unix")) is not int
        or type(raw.get("expires_at_unix")) is not int
        or raw.get("receipt_sha256") != _sha256_json(unsigned)
    ):
        raise SchemaReconciliationError(code)
    binding = authorization.value
    if (
        raw["release_revision"] != plan.revision
        or raw["plan_sha256"] != plan.sha256
        or raw["preflight_sha256"] != preflight["preflight_sha256"]
        or raw["observed_contract_sha256"]
        != preflight["observed_contract_sha256"]
        or raw["truth_receipt_sha256"] != truth.sha256
        or raw["authorization_sha256"] != authorization.sha256
        or raw["signed_frame_sha256"] != binding["owner_frame_sha256"]
        or raw["owner_subject_sha256"] != binding["owner_subject_sha256"]
        or raw["owner_key_id"] != binding["owner_key_id"]
        or raw["issued_at_unix"] != binding["issued_at_unix"]
        or raw["expires_at_unix"] != binding["expires_at_unix"]
        or raw["nonce"] != binding["nonce"]
    ):
        raise SchemaReconciliationError(
            "schema_reconciliation_owner_frame_binding_invalid"
        )
    return json.loads(_canonical_bytes(raw).decode("utf-8"))


def build_schema_reconciliation_owner_frame_receipt(
    *,
    plan: "SchemaReconciliationPlan",
    preflight: Mapping[str, Any],
    truth: CanonicalTruthReceipt,
    authorization: SchemaReconciliationAuthorization,
    signed_frame_sha256: str,
    signature_sshsig_sha256: str,
) -> Mapping[str, Any]:
    """Project one already-verified secret-free owner A2 frame.

    ``signed_frame_sha256`` is over the canonical JSON bytes of the complete
    signed A2 mapping. ``signature_sshsig_sha256`` is over the exact UTF-8
    bytes of that mapping's sshsig value.  Signature verification and durable
    publication of the full A2 frame remain the bootstrap's responsibility.
    """

    if (
        not isinstance(authorization, SchemaReconciliationAuthorization)
        or not isinstance(truth, CanonicalTruthReceipt)
    ):
        raise SchemaReconciliationError(
            "schema_reconciliation_owner_frame_invalid"
        )
    _validate_preflight(plan, preflight)
    binding = authorization.value
    unsigned = {
        "schema": RECONCILIATION_OWNER_FRAME_RECEIPT_SCHEMA,
        "frame_schema": RECONCILIATION_OWNER_A2_FRAME_SCHEMA,
        "action": "apply_schema_reconciliation",
        "approved": True,
        "signed_frame_sha256": signed_frame_sha256,
        "signature_sshsig_sha256": signature_sshsig_sha256,
        "signature_namespace": RECONCILIATION_OWNER_SIGNATURE_NAMESPACE,
        "signature_verified": True,
        "release_revision": plan.revision,
        "plan_sha256": plan.sha256,
        "preflight_sha256": preflight.get("preflight_sha256"),
        "observed_contract_sha256": preflight.get(
            "observed_contract_sha256"
        ),
        "truth_receipt_sha256": truth.sha256,
        "authorization_sha256": authorization.sha256,
        "owner_subject_sha256": binding["owner_subject_sha256"],
        "owner_key_id": binding["owner_key_id"],
        "issued_at_unix": binding["issued_at_unix"],
        "expires_at_unix": binding["expires_at_unix"],
        "nonce": binding["nonce"],
    }
    value = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
    return _validate_owner_authorization_frame(
        plan,
        preflight=preflight,
        truth=truth,
        authorization=authorization,
        value=value,
    )


@dataclass(frozen=True)
class SchemaContract:
    """Complete secret-free database contract observed by the trusted collector."""

    attestation: PrivilegeAttestation
    database: str = DATABASE
    schema_owner: str = CANONICAL_WRITER_MIGRATION_OWNER
    default_acl_non_owner_grants: tuple[str, ...] = ()
    helper_catalog_identity: HelperRoutineCatalogIdentity | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.attestation, PrivilegeAttestation)
            or self.database != DATABASE
            or self.schema_owner != CANONICAL_WRITER_MIGRATION_OWNER
            or self.default_acl_non_owner_grants
        ):
            raise SchemaReconciliationError("schema_reconciliation_contract_invalid")
        public_signatures = tuple(
            sorted(identity.signature for identity in self.attestation.routine_identities)
        )
        helper_signatures = tuple(
            sorted(
                identity.signature
                for identity in self.attestation.dependency_routine_identities
            )
        )
        old_helpers = tuple(
            signature
            for signature in EXPECTED_HELPER_ROUTINE_SIGNATURES
            if signature != MISSING_HELPER_SIGNATURE
        )
        if (
            tuple(sorted(self.attestation.executable_routines))
            != EXPECTED_ROUTINE_SIGNATURES
            or public_signatures != EXPECTED_ROUTINE_SIGNATURES
            or helper_signatures
            not in {EXPECTED_HELPER_ROUTINE_SIGNATURES, old_helpers}
        ):
            raise SchemaReconciliationError("schema_reconciliation_routine_set_invalid")
        is_target = helper_signatures == EXPECTED_HELPER_ROUTINE_SIGNATURES
        if is_target is not isinstance(
            self.helper_catalog_identity,
            HelperRoutineCatalogIdentity,
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_helper_catalog_identity_invalid"
            )
        identities = (
            *self.attestation.routine_identities,
            *self.attestation.dependency_routine_identities,
        )
        if any(
            identity.owner != CANONICAL_WRITER_MIGRATION_OWNER
            or identity.owner_dangerous
            or identity.configuration != _SAFE_SEARCH_PATH
            for identity in identities
        ):
            raise SchemaReconciliationError("schema_reconciliation_routine_identity_invalid")
        if self.helper_catalog_identity is not None:
            helper = next(
                (
                    identity
                    for identity in self.attestation.dependency_routine_identities
                    if identity.signature == MISSING_HELPER_SIGNATURE
                ),
                None,
            )
            if (
                helper is None
                or helper.owner != self.helper_catalog_identity.owner
                or helper.owner_dangerous
                    is not self.helper_catalog_identity.owner_dangerous
                or helper.security_definer
                    is not self.helper_catalog_identity.security_definer
                or helper.language != self.helper_catalog_identity.language
                or helper.configuration
                    != self.helper_catalog_identity.configuration
                or helper.definition_sha256
                    != self.helper_catalog_identity.definition_sha256
            ):
                raise SchemaReconciliationError(
                    "schema_reconciliation_helper_catalog_identity_invalid"
                )
        if any(
            not identity.security_definer
            for identity in self.attestation.routine_identities
        ) or any(
            identity.security_definer
            for identity in self.attestation.dependency_routine_identities
        ):
            raise SchemaReconciliationError("schema_reconciliation_routine_identity_invalid")
        try:
            validate_privilege_attestation(
                self.attestation,
                _target_policy(self.attestation),
                expected_user="muncho_canary_writer_login",
            )
        except (PrivilegeAttestationError, TypeError, ValueError) as exc:
            raise SchemaReconciliationError("schema_reconciliation_contract_invalid") from exc
        # Canonical round-trip also rejects unserialisable or non-finite values.
        _canonical_bytes(self.value)

    @property
    def value(self) -> Mapping[str, Any]:
        return {
            "schema": SCHEMA_CONTRACT_SCHEMA,
            "database": self.database,
            "schema_owner": self.schema_owner,
            "default_acl_non_owner_grants": list(
                self.default_acl_non_owner_grants
            ),
            "helper_catalog_identity": (
                None
                if self.helper_catalog_identity is None
                else self.helper_catalog_identity.value
            ),
            "attestation": _attestation_projection(self.attestation),
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.value)

    @property
    def is_target(self) -> bool:
        return tuple(
            sorted(
                identity.signature
                for identity in self.attestation.dependency_routine_identities
            )
        ) == EXPECTED_HELPER_ROUTINE_SIGNATURES

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SchemaContract":
        code = "schema_reconciliation_contract_invalid"
        raw = _exact_mapping(
            value,
            frozenset(
                {
                    "schema",
                    "database",
                    "schema_owner",
                    "default_acl_non_owner_grants",
                    "helper_catalog_identity",
                    "attestation",
                }
            ),
            code,
        )
        if raw["schema"] != SCHEMA_CONTRACT_SCHEMA:
            raise SchemaReconciliationError(code)
        helper_raw = raw["helper_catalog_identity"]
        helper = (
            None
            if helper_raw is None
            else HelperRoutineCatalogIdentity.from_mapping(
                _exact_mapping(
                    helper_raw,
                    frozenset(
                        EXPECTED_MISSING_HELPER_CATALOG_IDENTITY.value
                    ),
                    code,
                )
            )
        )
        try:
            contract = cls(
                attestation=_attestation_from_value(raw["attestation"]),
                database=_exact_text(raw["database"], code),
                schema_owner=_exact_text(raw["schema_owner"], code),
                default_acl_non_owner_grants=_exact_strings(
                    raw["default_acl_non_owner_grants"], code
                ),
                helper_catalog_identity=helper,
            )
        except SchemaReconciliationError:
            raise
        except (TypeError, ValueError) as exc:
            raise SchemaReconciliationError(code) from exc
        if _canonical_bytes(value) != _canonical_bytes(contract.value):
            raise SchemaReconciliationError(code)
        return contract


@dataclass(frozen=True)
class SchemaContractAsset:
    """Reviewed PostgreSQL-18 target contract shipped in the sealed release."""

    base_artifact_sha256: str
    contract: SchemaContract
    postgresql_major: int = POSTGRESQL_MAJOR
    base_artifact_filename: str = BASE_ARTIFACT_FILENAME

    def __post_init__(self) -> None:
        if (
            type(self.postgresql_major) is not int
            or self.postgresql_major != POSTGRESQL_MAJOR
            or self.base_artifact_filename != BASE_ARTIFACT_FILENAME
            or not isinstance(self.base_artifact_sha256, str)
            or _SHA256.fullmatch(self.base_artifact_sha256) is None
            or not isinstance(self.contract, SchemaContract)
            or not self.contract.is_target
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_target_asset_invalid"
            )

    @property
    def unsigned_value(self) -> Mapping[str, Any]:
        return {
            "schema": SCHEMA_CONTRACT_ASSET_SCHEMA,
            "postgresql_major": self.postgresql_major,
            "base_artifact_filename": self.base_artifact_filename,
            "base_artifact_sha256": self.base_artifact_sha256,
            "contract": self.contract.value,
            "contract_sha256": self.contract.sha256,
        }

    @property
    def value(self) -> Mapping[str, Any]:
        unsigned = self.unsigned_value
        return {**unsigned, "asset_sha256": _sha256_json(unsigned)}

    @property
    def sha256(self) -> str:
        return str(self.value["asset_sha256"])

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.value) + b"\n"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SchemaContractAsset":
        code = "schema_reconciliation_target_asset_invalid"
        raw = _exact_mapping(
            value,
            frozenset(
                {
                    "schema",
                    "postgresql_major",
                    "base_artifact_filename",
                    "base_artifact_sha256",
                    "contract",
                    "contract_sha256",
                    "asset_sha256",
                }
            ),
            code,
        )
        if raw["schema"] != SCHEMA_CONTRACT_ASSET_SCHEMA:
            raise SchemaReconciliationError(code)
        try:
            contract = SchemaContract.from_mapping(
                _exact_mapping(
                    raw["contract"],
                    frozenset(
                        {
                            "schema",
                            "database",
                            "schema_owner",
                            "default_acl_non_owner_grants",
                            "helper_catalog_identity",
                            "attestation",
                        }
                    ),
                    code,
                )
            )
            asset = cls(
                postgresql_major=_exact_integer(
                    raw["postgresql_major"], code
                ),
                base_artifact_filename=_exact_text(
                    raw["base_artifact_filename"], code
                ),
                base_artifact_sha256=_exact_text(
                    raw["base_artifact_sha256"], code
                ),
                contract=contract,
            )
        except SchemaReconciliationError:
            raise
        except (TypeError, ValueError) as exc:
            raise SchemaReconciliationError(code) from exc
        if (
            raw["contract_sha256"] != contract.sha256
            or raw["asset_sha256"] != asset.sha256
            or _canonical_bytes(value) != _canonical_bytes(asset.value)
        ):
            raise SchemaReconciliationError(code)
        return asset

    @classmethod
    def from_bytes(cls, value: bytes) -> "SchemaContractAsset":
        code = "schema_reconciliation_target_asset_invalid"
        if (
            not isinstance(value, bytes)
            or not value.endswith(b"\n")
            or value.endswith(b"\n\n")
            or len(value) > _MAX_JSON_BYTES
        ):
            raise SchemaReconciliationError(code)
        parsed = _strict_json(value[:-1], code)
        asset = cls.from_mapping(parsed)
        if value != asset.canonical_bytes:
            raise SchemaReconciliationError(code)
        return asset


def load_release_schema_contract_asset(revision: str) -> SchemaContractAsset:
    """Load the one target contract from the root-sealed release manifest."""

    try:
        from gateway.canonical_writer_planner import (
            _read_trusted_root_file,
            load_release_manifest,
        )

        manifest, _manifest_raw = load_release_manifest(revision)
        relative = SCHEMA_CONTRACT_ASSET_RELATIVE_PATH.as_posix()
        entries = [entry for entry in manifest.entries if entry.path == relative]
        if (
            len(entries) != 1
            or entries[0].kind != "file"
            or entries[0].mode != "0444"
            or not 0 < entries[0].size <= _MAX_JSON_BYTES
            or _SHA256.fullmatch(entries[0].sha256) is None
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_target_asset_manifest_invalid"
            )
        root = Path(manifest.artifact_root)
        raw = _read_trusted_root_file(
            root / SCHEMA_CONTRACT_ASSET_RELATIVE_PATH,
            allowed_modes=frozenset({0o444}),
            maximum=_MAX_JSON_BYTES,
        )
        if (
            len(raw) != entries[0].size
            or _sha256_bytes(raw) != entries[0].sha256
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_target_asset_manifest_invalid"
            )
        asset = SchemaContractAsset.from_bytes(raw)
        artifact = _load_sealed_artifacts(revision)[BASE_ARTIFACT_NAME]
        if asset.base_artifact_sha256 != artifact.sha256:
            raise SchemaReconciliationError(
                "schema_reconciliation_target_asset_base_mismatch"
            )
        return asset
    except SchemaReconciliationError:
        raise
    except BaseException as exc:
        raise SchemaReconciliationError(
            "schema_reconciliation_target_asset_load_failed"
        ) from exc


class SchemaContractCollectionSession(Protocol):
    def query(self, sql: str, *, maximum_rows: int) -> Any: ...


def _postgres_boolean(value: Any) -> bool:
    if value in (True, "t", "true", "TRUE", "1"):
        return True
    if value in (False, "f", "false", "FALSE", "0"):
        return False
    raise SchemaReconciliationError("schema_reconciliation_catalog_invalid")


def collect_schema_contract(
    session: SchemaContractCollectionSession,
    *,
    config: WriterDBConfig,
    policy: WriterPrivilegePolicy,
    managed_hba_receipt: ManagedCloudSQLAdminHBAReceipt | None,
    subject_user: str = "muncho_canary_writer_login",
    allow_missing_helper: bool = False,
) -> SchemaContract:
    """Collect the full target through one already-open trusted DB session."""

    if type(allow_missing_helper) is not bool:
        raise SchemaReconciliationError(
            "schema_reconciliation_contract_collection_failed"
        )
    try:
        attestation = _collect_privilege_attestation(
            session,
            config=config,
            policy=policy,
            managed_hba_receipt=managed_hba_receipt,
            subject_user=subject_user,
        )
        environment = session.query(
            "SELECT pg_catalog.current_database(), "
            "pg_catalog.current_setting('server_version_num'), "
            "pg_catalog.pg_get_userbyid(database.datdba), "
            "pg_catalog.pg_get_userbyid(namespace.nspowner), "
            "(SELECT pg_catalog.count(*) FROM pg_catalog.pg_default_acl AS defaults "
            "CROSS JOIN LATERAL pg_catalog.aclexplode(defaults.defaclacl) AS acl "
            "WHERE defaults.defaclrole = (SELECT role.oid FROM pg_catalog.pg_roles "
            "AS role WHERE role.rolname = 'canonical_brain_migration_owner') "
            "AND acl.grantee <> defaults.defaclrole)::text "
            "FROM pg_catalog.pg_database AS database "
            "JOIN pg_catalog.pg_namespace AS namespace ON namespace.nspname = "
            "'canonical_brain' WHERE database.datname = "
            "pg_catalog.current_database()",
            maximum_rows=1,
        )
        if len(environment.rows) != 1 or len(environment.rows[0]) != 5:
            raise SchemaReconciliationError(
                "schema_reconciliation_environment_invalid"
            )
        row = environment.rows[0]
        if (
            row[0] != DATABASE
            or not isinstance(row[1], str)
            or not row[1].isdigit()
            or int(row[1]) // 10000 != POSTGRESQL_MAJOR
            or row[2] != "cloudsqlsuperuser"
            or row[3] != CANONICAL_WRITER_MIGRATION_OWNER
            or row[4] != "0"
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_environment_invalid"
            )
        helper_result = session.query(
            "SELECT owner.rolname, (owner.rolcanlogin OR owner.rolsuper OR "
            "owner.rolcreatedb OR owner.rolcreaterole OR owner.rolreplication OR "
            "owner.rolbypassrls OR "
            + _owner_membership_danger_sql(
                "owner",
                "helper_owner_membership",
            )
            + "), "
            "routine.prokind::text, routine.prosecdef, routine.provolatile::text, "
            "routine.proparallel::text, routine.proleakproof, routine.proisstrict, "
            "routine.proretset, language.lanname, "
            "pg_catalog.oidvectortypes(routine.proargtypes), "
            "pg_catalog.format_type(routine.prorettype, NULL), "
            "pg_catalog.array_to_json(COALESCE(routine.proconfig, "
            "ARRAY[]::text[]))::text, "
            "pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to("
            "routine.prosrc, 'UTF8')), 'hex'), "
            "pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to("
            "pg_catalog.pg_get_functiondef(routine.oid), 'UTF8')), 'hex'), "
            "EXISTS (SELECT 1 FROM LATERAL pg_catalog.aclexplode(COALESCE("
            "routine.proacl, pg_catalog.acldefault('f', routine.proowner))) AS acl "
            "WHERE acl.grantee <> routine.proowner), "
            "pg_catalog.has_function_privilege('canonical_brain_writer', "
            "routine.oid, 'EXECUTE'), "
            "pg_catalog.has_function_privilege('muncho_canary_writer_login', "
            "routine.oid, 'EXECUTE') FROM pg_catalog.pg_proc AS routine "
            "JOIN pg_catalog.pg_language AS language ON language.oid = "
            "routine.prolang JOIN pg_catalog.pg_roles AS owner ON owner.oid = "
            "routine.proowner WHERE routine.oid = pg_catalog.to_regprocedure("
            "'canonical_brain._discord_guild_routeback_target_valid(jsonb)')",
            maximum_rows=1,
        )
        if not helper_result.rows and allow_missing_helper:
            helper = None
        elif len(helper_result.rows) != 1 or len(helper_result.rows[0]) != 18:
            raise SchemaReconciliationError(
                "schema_reconciliation_helper_catalog_identity_invalid"
            )
        else:
            helper_row = helper_result.rows[0]
            try:
                configuration_raw = json.loads(helper_row[12] or "")
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SchemaReconciliationError(
                    "schema_reconciliation_helper_catalog_identity_invalid"
                ) from exc
            if not isinstance(configuration_raw, list) or any(
                not isinstance(item, str) for item in configuration_raw
            ):
                raise SchemaReconciliationError(
                    "schema_reconciliation_helper_catalog_identity_invalid"
                )
            helper_owner_dangerous = _postgres_boolean(helper_row[1])
            helper = HelperRoutineCatalogIdentity(
                signature=MISSING_HELPER_SIGNATURE,
                owner=helper_row[0] or "",
                owner_dangerous=helper_owner_dangerous,
                kind=helper_row[2] or "",
                security_definer=_postgres_boolean(helper_row[3]),
                volatility=helper_row[4] or "",
                parallel=helper_row[5] or "",
                leakproof=_postgres_boolean(helper_row[6]),
                strict=_postgres_boolean(helper_row[7]),
                returns_set=_postgres_boolean(helper_row[8]),
                language=helper_row[9] or "",
                argument_types=(helper_row[10] or "",),
                return_type=helper_row[11] or "",
                configuration=tuple(configuration_raw),
                prosrc_sha256=helper_row[13] or "",
                definition_sha256=helper_row[14] or "",
                non_owner_acl_grants=(
                    ("present",) if _postgres_boolean(helper_row[15]) else ()
                ),
                writer_role_execute=_postgres_boolean(helper_row[16]),
                writer_login_execute=_postgres_boolean(helper_row[17]),
            )
        return SchemaContract(
            attestation=attestation,
            helper_catalog_identity=helper,
        )
    except SchemaReconciliationError:
        raise
    except BaseException as exc:
        raise SchemaReconciliationError(
            "schema_reconciliation_contract_collection_failed"
        ) from exc


def _old_contract_value(target: SchemaContract) -> Mapping[str, Any]:
    if not target.is_target:
        raise SchemaReconciliationError("schema_reconciliation_target_contract_invalid")
    value = copy.deepcopy(dict(target.value))
    attestation = value.get("attestation")
    if not isinstance(attestation, dict):
        raise SchemaReconciliationError("schema_reconciliation_target_contract_invalid")
    helpers = attestation.get("helper_routine_identities")
    if not isinstance(helpers, list):
        raise SchemaReconciliationError("schema_reconciliation_target_contract_invalid")
    filtered = [
        identity
        for identity in helpers
        if isinstance(identity, Mapping)
        and identity.get("signature") != MISSING_HELPER_SIGNATURE
    ]
    if len(helpers) != len(filtered) + 1:
        raise SchemaReconciliationError("schema_reconciliation_target_contract_invalid")
    attestation["helper_routine_identities"] = filtered
    value["helper_catalog_identity"] = None
    return value


CONTROL_FOUNDATION_CONTRACT_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-control-contract.v1"
)


def _control_foundation_contract_sha256(
    install_artifact_sha256: str,
    retire_artifact_sha256: str,
) -> str:
    """Bind the fixed inert role, routines, and both rollback artifacts."""

    if (
        not isinstance(install_artifact_sha256, str)
        or _SHA256.fullmatch(install_artifact_sha256) is None
        or not isinstance(retire_artifact_sha256, str)
        or _SHA256.fullmatch(retire_artifact_sha256) is None
    ):
        raise SchemaReconciliationError(
            "schema_reconciliation_control_contract_invalid"
        )
    return _sha256_json(
        {
            "schema": CONTROL_FOUNDATION_CONTRACT_SCHEMA,
            "database": DATABASE,
            "executor_role": "canonical_brain_schema_reconciler",
            "control_schema": "canonical_brain_reconciliation",
            "observer_signature": (
                "canonical_brain_reconciliation."
                "observe_missing_discord_routeback_helper_v1()"
            ),
            "apply_signature": (
                "canonical_brain_reconciliation."
                "apply_missing_discord_routeback_helper_v1()"
            ),
            "control_install_artifact_sha256": install_artifact_sha256,
            "control_retire_artifact_sha256": retire_artifact_sha256,
            "helper_catalog_identity_sha256": (
                EXPECTED_MISSING_HELPER_CATALOG_IDENTITY.sha256
            ),
        }
    )


def _load_control_artifact(
    revision: str,
    *,
    name: str,
    filename: str,
) -> SealedSQLArtifact:
    """Load one exact control artifact from the root-sealed release only."""

    manifest, _raw = load_release_manifest(revision)
    root = Path(manifest.artifact_root)
    if root != Path("/opt/muncho-canary-releases") / revision:
        raise SchemaReconciliationError(
            "schema_reconciliation_release_invalid"
        )
    relative = f"scripts/sql/{filename}"
    entries = {entry.path: entry for entry in manifest.entries}
    entry = entries.get(relative)
    if (
        entry is None
        or entry.kind != "file"
        or entry.mode != "0444"
        or entry.size <= 0
        or _SHA256.fullmatch(entry.sha256) is None
    ):
        raise SchemaReconciliationError(
            "schema_reconciliation_release_invalid"
        )
    return _read_sealed_artifact(
        name,
        root / relative,
        expected_sha256=entry.sha256,
        expected_size=entry.size,
        require_root_sealed=True,
    )


@dataclass(frozen=True)
class SchemaReconciliationPlan:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "SchemaReconciliationPlan":
        if not isinstance(value, Mapping) or set(value) != _PLAN_FIELDS:
            raise SchemaReconciliationError("schema_reconciliation_plan_invalid")
        raw = copy.deepcopy(dict(value))
        unsigned = {key: item for key, item in raw.items() if key != "plan_sha256"}
        if (
            raw["schema"] != RECONCILIATION_PLAN_SCHEMA
            or not isinstance(raw["release_revision"], str)
            or _REVISION.fullmatch(raw["release_revision"]) is None
            or raw["database"] != DATABASE
            or raw["helper_signature"] != MISSING_HELPER_SIGNATURE
            or any(
                not isinstance(raw[name], str) or _SHA256.fullmatch(raw[name]) is None
                for name in (
                    "base_artifact_sha256",
                    "target_asset_sha256",
                    "control_install_artifact_sha256",
                    "control_retire_artifact_sha256",
                    "control_foundation_contract_sha256",
                    "helper_catalog_identity_sha256",
                    "expected_old_contract_sha256",
                    "target_contract_sha256",
                )
            )
            or raw["helper_catalog_identity_sha256"]
            != EXPECTED_MISSING_HELPER_CATALOG_IDENTITY.sha256
            or raw["postgresql_major"] != POSTGRESQL_MAJOR
            or raw["advisory_lock_key"] != CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
            or raw["transaction_isolation"] != "SERIALIZABLE"
            or raw["canonical_truth_lock"] != CANONICAL_TRUTH_LOCK_SQL
            or raw["canonical_truth_preservation_required"] is not True
            or raw["control_foundation_contract_sha256"]
            != _control_foundation_contract_sha256(
                raw["control_install_artifact_sha256"],
                raw["control_retire_artifact_sha256"],
            )
            or raw["plan_sha256"] != _sha256_json(unsigned)
        ):
            raise SchemaReconciliationError("schema_reconciliation_plan_invalid")
        return cls(json.loads(_canonical_bytes(raw).decode("utf-8")))

    @property
    def sha256(self) -> str:
        return str(self.value["plan_sha256"])

    @property
    def revision(self) -> str:
        return str(self.value["release_revision"])


def _build_plan_from_artifact(
    revision: str,
    target: SchemaContract,
    artifact: SealedSQLArtifact,
    *,
    target_asset_sha256: str = "0" * 64,
    control_install_artifact_sha256: str,
    control_retire_artifact_sha256: str,
    postgresql_major: int = POSTGRESQL_MAJOR,
) -> SchemaReconciliationPlan:
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise SchemaReconciliationError("schema_reconciliation_revision_invalid")
    if not isinstance(target, SchemaContract) or not target.is_target:
        raise SchemaReconciliationError("schema_reconciliation_target_contract_invalid")
    if (
        not isinstance(target_asset_sha256, str)
        or _SHA256.fullmatch(target_asset_sha256) is None
        or not isinstance(control_install_artifact_sha256, str)
        or _SHA256.fullmatch(control_install_artifact_sha256) is None
        or not isinstance(control_retire_artifact_sha256, str)
        or _SHA256.fullmatch(control_retire_artifact_sha256) is None
        or type(postgresql_major) is not int
        or postgresql_major != POSTGRESQL_MAJOR
    ):
        raise SchemaReconciliationError("schema_reconciliation_target_asset_invalid")
    if (
        not isinstance(artifact, SealedSQLArtifact)
        or artifact.name != BASE_ARTIFACT_NAME
        or artifact.path.name != BASE_ARTIFACT_FILENAME
        or not artifact.payload
        or _sha256_bytes(artifact.payload) != artifact.sha256
    ):
        raise SchemaReconciliationError(
            "schema_reconciliation_artifact_invalid"
        )
    control_contract_sha256 = _control_foundation_contract_sha256(
        control_install_artifact_sha256,
        control_retire_artifact_sha256,
    )
    unsigned = {
        "schema": RECONCILIATION_PLAN_SCHEMA,
        "release_revision": revision,
        "database": DATABASE,
        "helper_signature": MISSING_HELPER_SIGNATURE,
        "base_artifact_sha256": artifact.sha256,
        "target_asset_sha256": target_asset_sha256,
        "postgresql_major": postgresql_major,
        "control_install_artifact_sha256": (
            control_install_artifact_sha256
        ),
        "control_retire_artifact_sha256": (
            control_retire_artifact_sha256
        ),
        "control_foundation_contract_sha256": control_contract_sha256,
        "helper_catalog_identity_sha256": (
            EXPECTED_MISSING_HELPER_CATALOG_IDENTITY.sha256
        ),
        "expected_old_contract_sha256": _sha256_json(_old_contract_value(target)),
        "target_contract_sha256": target.sha256,
        "advisory_lock_key": CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY,
        "transaction_isolation": "SERIALIZABLE",
        "canonical_truth_lock": CANONICAL_TRUTH_LOCK_SQL,
        "canonical_truth_preservation_required": True,
    }
    return SchemaReconciliationPlan.from_mapping(
        {**unsigned, "plan_sha256": _sha256_json(unsigned)}
    )


def build_schema_reconciliation_plan(
    revision: str,
) -> SchemaReconciliationPlan:
    """Build only from the root-sealed target and reviewed SQL artifacts."""

    try:
        target_asset = load_release_schema_contract_asset(revision)
        artifact = _load_sealed_artifacts(revision)[BASE_ARTIFACT_NAME]
        control_install = _load_control_artifact(
            revision,
            name="schema_reconciliation_control_install",
            filename=CONTROL_INSTALL_ARTIFACT_FILENAME,
        )
        control_retire = _load_control_artifact(
            revision,
            name="schema_reconciliation_control_retire",
            filename=CONTROL_RETIRE_ARTIFACT_FILENAME,
        )
    except BaseException as exc:
        raise SchemaReconciliationError("schema_reconciliation_release_invalid") from exc
    if artifact.sha256 != target_asset.base_artifact_sha256:
        raise SchemaReconciliationError(
            "schema_reconciliation_target_asset_base_mismatch"
        )
    return _build_plan_from_artifact(
        revision,
        target_asset.contract,
        artifact,
        target_asset_sha256=target_asset.sha256,
        control_install_artifact_sha256=control_install.sha256,
        control_retire_artifact_sha256=control_retire.sha256,
        postgresql_major=target_asset.postgresql_major,
    )


def preflight_schema_reconciliation(
    plan: SchemaReconciliationPlan,
    *,
    target: SchemaContract,
    observed: SchemaContract,
    truth: CanonicalTruthReceipt,
    observed_at_unix: int,
) -> Mapping[str, Any]:
    """Return a release-bound read-only proof for old or terminal state.

    The only mutable-state candidate is the exact target projection with the
    reviewed helper identity removed.  Equality is over the complete canonical
    contract, not merely the helper signature list.
    """

    if (
        not isinstance(plan, SchemaReconciliationPlan)
        or not isinstance(target, SchemaContract)
        or not target.is_target
        or not isinstance(observed, SchemaContract)
        or not isinstance(truth, CanonicalTruthReceipt)
        or type(observed_at_unix) is not int
        or observed_at_unix < 0
        or target.sha256 != plan.value["target_contract_sha256"]
        or _sha256_json(_old_contract_value(target))
        != plan.value["expected_old_contract_sha256"]
    ):
        raise SchemaReconciliationError("schema_reconciliation_plan_binding_invalid")
    if observed.sha256 == plan.value["expected_old_contract_sha256"]:
        state = "exact_old_missing_one_helper"
        mutation_required = True
    elif observed.sha256 == plan.value["target_contract_sha256"]:
        state = "exact_target"
        mutation_required = False
    else:
        raise SchemaReconciliationError(
            "schema_reconciliation_unreviewed_database_drift"
        )
    unsigned = {
        "schema": RECONCILIATION_PREFLIGHT_SCHEMA,
        "ok": True,
        "release_revision": plan.revision,
        "plan_sha256": plan.sha256,
        "base_artifact_sha256": plan.value["base_artifact_sha256"],
        "target_asset_sha256": plan.value["target_asset_sha256"],
        "postgresql_major": plan.value["postgresql_major"],
        "control_install_artifact_sha256": plan.value[
            "control_install_artifact_sha256"
        ],
        "control_retire_artifact_sha256": plan.value[
            "control_retire_artifact_sha256"
        ],
        "control_foundation_contract_sha256": plan.value[
            "control_foundation_contract_sha256"
        ],
        "observed_contract_sha256": observed.sha256,
        "truth_receipt_sha256": truth.sha256,
        "expected_old_contract_sha256": plan.value[
            "expected_old_contract_sha256"
        ],
        "target_contract_sha256": plan.value["target_contract_sha256"],
        "state": state,
        "mutation_required": mutation_required,
        "observed_at_unix": observed_at_unix,
    }
    value = {**unsigned, "preflight_sha256": _sha256_json(unsigned)}
    if set(value) != _PREFLIGHT_FIELDS:
        raise SchemaReconciliationError("schema_reconciliation_preflight_invalid")
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _validate_preflight(
    plan: SchemaReconciliationPlan,
    value: Mapping[str, Any],
) -> None:
    code = "schema_reconciliation_preflight_invalid"
    if not isinstance(plan, SchemaReconciliationPlan) or not isinstance(
        value, Mapping
    ) or set(value) != _PREFLIGHT_FIELDS:
        raise SchemaReconciliationError(code)
    unsigned = {
        key: item for key, item in value.items() if key != "preflight_sha256"
    }
    state = value.get("state")
    expected_contract = (
        plan.value["expected_old_contract_sha256"]
        if state == "exact_old_missing_one_helper"
        else plan.value["target_contract_sha256"]
        if state == "exact_target"
        else None
    )
    if (
        value.get("schema") != RECONCILIATION_PREFLIGHT_SCHEMA
        or value.get("ok") is not True
        or value.get("release_revision") != plan.revision
        or value.get("plan_sha256") != plan.sha256
        or value.get("base_artifact_sha256")
        != plan.value["base_artifact_sha256"]
        or value.get("target_asset_sha256")
        != plan.value["target_asset_sha256"]
        or value.get("postgresql_major") != plan.value["postgresql_major"]
        or any(
            value.get(name) != plan.value[name]
            for name in (
                "control_install_artifact_sha256",
                "control_retire_artifact_sha256",
                "control_foundation_contract_sha256",
            )
        )
        or value.get("observed_contract_sha256") != expected_contract
        or not isinstance(value.get("truth_receipt_sha256"), str)
        or _SHA256.fullmatch(str(value.get("truth_receipt_sha256"))) is None
        or value.get("expected_old_contract_sha256")
        != plan.value["expected_old_contract_sha256"]
        or value.get("target_contract_sha256")
        != plan.value["target_contract_sha256"]
        or value.get("mutation_required")
        is not (state == "exact_old_missing_one_helper")
        or type(value.get("observed_at_unix")) is not int
        or value["observed_at_unix"] < 0
        or value.get("preflight_sha256") != _sha256_json(unsigned)
    ):
        raise SchemaReconciliationError(code)


class SchemaReconciliationTransaction(Protocol):
    def lock_canonical_truth(self) -> None: ...

    def observe_contract(self) -> SchemaContract: ...

    def observe_canonical_truth(self) -> CanonicalTruthReceipt: ...

    def apply_missing_helper(
        self,
        *,
        authorized_intent_sha256: str,
    ) -> None: ...


class SchemaReconciliationDatabase(Protocol):
    def transaction(
        self,
        *,
        advisory_lock_key: int,
    ) -> ContextManager[SchemaReconciliationTransaction]: ...


class AppendOnlySchemaReconciliationJournal:
    """Crash-recoverable immutable intent and terminal receipt publication."""

    def __init__(
        self,
        root: Path = EVIDENCE_ROOT,
        *,
        strict_root: bool = True,
        publication_fault_injector: Callable[[str, str], None] | None = None,
    ) -> None:
        self.root = root
        self.strict_root = strict_root
        self.publication_fault_injector = publication_fault_injector
        self._lock_held = False

    def _fault(self, kind: str, point: str) -> None:
        if self.publication_fault_injector is not None:
            self.publication_fault_injector(kind, point)

    def _ensure_directory(self, path: Path, *, kind: str) -> None:
        _secure_directory(
            path,
            strict_root=self.strict_root,
            mkdir_callback=lambda: self._fault(kind, "after_mkdir"),
        )

    def _plan_root(self, plan: SchemaReconciliationPlan) -> Path:
        return self.root / plan.revision / plan.sha256

    def _paths(self, plan: SchemaReconciliationPlan, name: str) -> tuple[Path, Path]:
        root = self._plan_root(plan)
        return root / "staging" / f"{name}.json", root / f"{name}.json"

    def _read(self, path: Path, *, expected_links: int, code: str) -> bytes:
        try:
            before = path.lstat()
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                opened = os.fstat(descriptor)
                raw = bytearray()
                while len(raw) <= _MAX_JSON_BYTES:
                    chunk = os.read(
                        descriptor,
                        min(1024 * 1024, _MAX_JSON_BYTES + 1),
                    )
                    if not chunk:
                        break
                    raw.extend(chunk)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            reachable = path.lstat()
        except OSError as exc:
            raise SchemaReconciliationError(code) from exc
        expected_uid = 0 if self.strict_root else _effective_uid()
        expected_gid = 0 if self.strict_root else _effective_gid()
        payload = bytes(raw)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o400
            or opened.st_uid != expected_uid
            or opened.st_gid != expected_gid
            or opened.st_nlink != expected_links
            or not payload
            or len(payload) > _MAX_JSON_BYTES
            or len(payload) != opened.st_size
            or _list_xattrs(path)
            or _filesystem_identity(before) != _filesystem_identity(opened)
            or _filesystem_identity(before) != _filesystem_identity(after)
            or _filesystem_identity(before) != _filesystem_identity(reachable)
        ):
            raise SchemaReconciliationError(code)
        return payload

    def _read_unpublished_stage_candidate(
        self,
        path: Path,
    ) -> tuple[bytes, tuple[int, ...]]:
        """Read a single-link stage while separating metadata from JSON state."""

        code = "schema_reconciliation_staged_receipt_invalid"
        try:
            before = path.lstat()
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                opened = os.fstat(descriptor)
                raw = bytearray()
                while len(raw) <= _MAX_JSON_BYTES:
                    chunk = os.read(
                        descriptor,
                        min(1024 * 1024, _MAX_JSON_BYTES + 1),
                    )
                    if not chunk:
                        break
                    raw.extend(chunk)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            reachable = path.lstat()
        except OSError as exc:
            raise SchemaReconciliationError(code) from exc
        expected_uid = 0 if self.strict_root else _effective_uid()
        expected_gid = 0 if self.strict_root else _effective_gid()
        payload = bytes(raw)
        identity = _filesystem_identity(before)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o400
            or opened.st_uid != expected_uid
            or opened.st_gid != expected_gid
            or opened.st_nlink != 1
            or len(payload) > _MAX_JSON_BYTES
            or len(payload) != opened.st_size
            or _list_xattrs(path)
            or identity != _filesystem_identity(opened)
            or identity != _filesystem_identity(after)
            or identity != _filesystem_identity(reachable)
        ):
            raise SchemaReconciliationError(code)
        return payload, identity

    def _discard_partial_unpublished_stage(
        self,
        stage: Path,
        final: Path,
        *,
        payload: bytes,
        identity: tuple[int, ...],
    ) -> None:
        code = "schema_reconciliation_staged_receipt_invalid"
        if _is_complete_canonical_json(payload) or not self._lock_held:
            raise SchemaReconciliationError(code)
        try:
            reached = stage.lstat()
            expected_uid = 0 if self.strict_root else _effective_uid()
            expected_gid = 0 if self.strict_root else _effective_gid()
            if (
                os.path.lexists(final)
                or not stat.S_ISREG(reached.st_mode)
                or stat.S_ISLNK(reached.st_mode)
                or stat.S_IMODE(reached.st_mode) != 0o400
                or reached.st_uid != expected_uid
                or reached.st_gid != expected_gid
                or reached.st_nlink != 1
                or _list_xattrs(stage)
                or _filesystem_identity(reached) != identity
            ):
                raise SchemaReconciliationError(code)
            stage.unlink()
            _fsync_directory(stage.parent)
        except SchemaReconciliationError:
            raise
        except OSError as exc:
            raise SchemaReconciliationError(code) from exc
        raise _UnpublishedStageDiscarded

    @staticmethod
    def _fsync_file(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _finish_publication(self, stage: Path, final: Path, *, kind: str) -> bytes:
        stage_exists = os.path.lexists(stage)
        final_exists = os.path.lexists(final)
        if not stage_exists and not final_exists:
            raise FileNotFoundError
        if stage_exists and not final_exists:
            payload, identity = self._read_unpublished_stage_candidate(stage)
            if not _is_complete_canonical_json(payload):
                self._discard_partial_unpublished_stage(
                    stage,
                    final,
                    payload=payload,
                    identity=identity,
                )
            self._fsync_file(stage)
            _fsync_directory(stage.parent)
            os.link(stage, final, follow_symlinks=False)
            self._fault(kind, "after_publish")
            _fsync_directory(final.parent)
            final_exists = True
        if stage_exists and final_exists:
            staged = self._read(
                stage,
                expected_links=2,
                code="schema_reconciliation_staged_receipt_invalid",
            )
            published = self._read(
                final,
                expected_links=2,
                code="schema_reconciliation_receipt_invalid",
            )
            if staged != published or not _same_inode(stage.lstat(), final.lstat()):
                raise SchemaReconciliationError(
                    "schema_reconciliation_receipt_link_state_invalid"
                )
            stage.unlink()
            self._fault(kind, "after_unlink")
            _fsync_directory(stage.parent)
            _fsync_directory(final.parent)
        return self._read(
            final,
            expected_links=1,
            code="schema_reconciliation_receipt_invalid",
        )

    def _publish(self, plan: SchemaReconciliationPlan, name: str, payload: bytes) -> bytes:
        stage, final = self._paths(plan, name)
        self._ensure_directory(stage.parent, kind=name)
        self._ensure_directory(final.parent, kind=name)
        if os.path.lexists(stage) or os.path.lexists(final):
            try:
                existing = self._finish_publication(stage, final, kind=name)
            except _UnpublishedStageDiscarded:
                if os.path.lexists(stage) or os.path.lexists(final):
                    raise SchemaReconciliationError(
                        "schema_reconciliation_staged_receipt_invalid"
                    ) from None
            else:
                if existing != payload:
                    raise SchemaReconciliationError(
                        "schema_reconciliation_receipt_collision"
                    )
                return existing
        descriptor = os.open(
            stage,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        try:
            if self.strict_root:
                os.fchown(descriptor, 0, 0)
            else:
                os.fchown(descriptor, -1, _effective_gid())
            os.fchmod(descriptor, 0o400)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("receipt write stalled")
                offset += written
            self._fault(name, "after_write")
            os.fsync(descriptor)
            self._fault(name, "after_fsync")
        finally:
            os.close(descriptor)
        _fsync_directory(stage.parent)
        existing = self._finish_publication(stage, final, kind=name)
        if existing != payload:
            raise SchemaReconciliationError("schema_reconciliation_receipt_drifted")
        return existing

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        self._ensure_directory(self.root, kind="lock")
        path = self.root / ".reconciliation.lock"
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        created = False
        try:
            descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
        except FileExistsError:
            descriptor = os.open(path, flags)
        try:
            if created:
                if self.strict_root:
                    os.fchown(descriptor, 0, 0)
                else:
                    os.fchown(descriptor, -1, _effective_gid())
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
                _fsync_directory(path.parent)
            before = path.lstat()
            opened = os.fstat(descriptor)
            reachable = path.lstat()
            expected_uid = 0 if self.strict_root else _effective_uid()
            expected_gid = 0 if self.strict_root else _effective_gid()
            if (
                not stat.S_ISREG(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != 0o600
                or opened.st_uid != expected_uid
                or opened.st_gid != expected_gid
                or opened.st_nlink != 1
                or opened.st_size != 0
                or _list_xattrs(path)
                or _filesystem_identity(before) != _filesystem_identity(opened)
                or _filesystem_identity(before) != _filesystem_identity(reachable)
            ):
                raise SchemaReconciliationError("schema_reconciliation_lock_invalid")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            if self._lock_held:
                raise SchemaReconciliationError("schema_reconciliation_lock_invalid")
            self._lock_held = True
            try:
                yield
            finally:
                self._lock_held = False
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _load(
        self,
        plan: SchemaReconciliationPlan,
        name: str,
    ) -> Mapping[str, Any] | None:
        stage, final = self._paths(plan, name)
        if not os.path.lexists(stage) and not os.path.lexists(final):
            return None
        try:
            payload = self._finish_publication(stage, final, kind=name)
        except _UnpublishedStageDiscarded:
            return None
        return _strict_json(payload, "schema_reconciliation_receipt_invalid")

    def load_authorized_intent(
        self,
        plan: SchemaReconciliationPlan,
    ) -> Mapping[str, Any] | None:
        value = self._load(plan, "authorized_intent")
        if value is None:
            return None
        _validate_authorized_intent(plan, value)
        return value

    def append_authorized_intent(
        self,
        plan: SchemaReconciliationPlan,
        *,
        initial_contract_sha256: str,
        initial_canonical_truth: CanonicalTruthReceipt,
        authorization: SchemaReconciliationAuthorization,
        preflight: Mapping[str, Any],
        owner_authorization_frame: Mapping[str, Any],
        admitted_at_unix: int,
    ) -> Mapping[str, Any]:
        if initial_contract_sha256 == plan.value["expected_old_contract_sha256"]:
            mode = "reconcile_missing_helper"
            mutation_required = True
        elif initial_contract_sha256 == plan.value["target_contract_sha256"]:
            mode = "adopt_existing_target"
            mutation_required = False
        else:
            raise SchemaReconciliationError("schema_reconciliation_initial_contract_invalid")
        if not isinstance(initial_canonical_truth, CanonicalTruthReceipt):
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_invalid"
            )
        if not isinstance(authorization, SchemaReconciliationAuthorization):
            raise SchemaReconciliationError(
                "schema_reconciliation_authorization_invalid"
            )
        authorization.validate_binding(
            plan=plan,
            preflight=preflight,
            truth=initial_canonical_truth,
        )
        owner_frame = _validate_owner_authorization_frame(
            plan,
            preflight=preflight,
            truth=initial_canonical_truth,
            authorization=authorization,
            value=owner_authorization_frame,
        )
        if preflight["observed_contract_sha256"] != initial_contract_sha256:
            raise SchemaReconciliationError(
                "schema_reconciliation_authorization_binding_invalid"
            )
        unsigned = {
            "schema": RECONCILIATION_AUTHORIZED_INTENT_SCHEMA,
            "release_revision": plan.revision,
            "plan_sha256": plan.sha256,
            "base_artifact_sha256": plan.value["base_artifact_sha256"],
            "target_asset_sha256": plan.value["target_asset_sha256"],
            "postgresql_major": plan.value["postgresql_major"],
            "control_install_artifact_sha256": plan.value[
                "control_install_artifact_sha256"
            ],
            "control_retire_artifact_sha256": plan.value[
                "control_retire_artifact_sha256"
            ],
            "control_foundation_contract_sha256": plan.value[
                "control_foundation_contract_sha256"
            ],
            "expected_old_contract_sha256": plan.value[
                "expected_old_contract_sha256"
            ],
            "target_contract_sha256": plan.value["target_contract_sha256"],
            "preflight": copy.deepcopy(dict(preflight)),
            "owner_authorization_frame": owner_frame,
            "authorization": copy.deepcopy(dict(authorization.value)),
            "initial_contract_sha256": initial_contract_sha256,
            "initial_canonical_truth": initial_canonical_truth.value,
            "authorization_sha256": authorization.sha256,
            "preflight_sha256": preflight["preflight_sha256"],
            "owner_frame_receipt_sha256": owner_frame["receipt_sha256"],
            "truth_receipt_sha256": initial_canonical_truth.sha256,
            "mode": mode,
            "mutation_required": mutation_required,
            "admitted_at_unix": admitted_at_unix,
        }
        value = {
            **unsigned,
            "authorized_intent_sha256": _sha256_json(unsigned),
        }
        _validate_authorized_intent(plan, value)
        existing = self.load_authorized_intent(plan)
        if existing is not None:
            if existing != value:
                raise SchemaReconciliationError(
                    "schema_reconciliation_authorization_replayed"
                )
            return existing
        self._publish(
            plan,
            "authorized_intent",
            _canonical_bytes(value),
        )
        published = self.load_authorized_intent(plan)
        if published is None:
            raise SchemaReconciliationError(
                "schema_reconciliation_authorized_intent_publication_failed"
            )
        return published

    def load_terminal(self, plan: SchemaReconciliationPlan) -> Mapping[str, Any] | None:
        value = self._load(plan, "terminal")
        if value is None:
            return None
        _validate_terminal(plan, value)
        intent = self.load_authorized_intent(plan)
        if (
            intent is None
            or value["authorization_sha256"]
            != intent["authorization_sha256"]
            or value["preflight_sha256"] != intent["preflight_sha256"]
            or value["owner_frame_receipt_sha256"]
            != intent["owner_frame_receipt_sha256"]
            or value["truth_receipt_sha256"]
            != intent["truth_receipt_sha256"]
            or value["authorized_intent_sha256"]
            != intent["authorized_intent_sha256"]
            or value["initial_contract_sha256"]
            != intent["initial_contract_sha256"]
            or value["initial_canonical_truth"]
            != intent["initial_canonical_truth"]
            or value["final_canonical_truth"]
            != value["initial_canonical_truth"]
            or value["mode"] != intent["mode"]
            or value["mutation_applied"] is not intent["mutation_required"]
            or value["completed_at_unix"] < intent["admitted_at_unix"]
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_terminal_intent_mismatch"
            )
        return value

    def append_terminal(
        self,
        plan: SchemaReconciliationPlan,
        *,
        intent: Mapping[str, Any],
        final_canonical_truth: CanonicalTruthReceipt,
        completed_at_unix: int,
    ) -> Mapping[str, Any]:
        existing = self.load_terminal(plan)
        if existing is not None:
            return existing
        _validate_authorized_intent(plan, intent)
        initial_truth = CanonicalTruthReceipt.from_mapping(
            intent["initial_canonical_truth"]
        )
        if (
            not isinstance(final_canonical_truth, CanonicalTruthReceipt)
            or final_canonical_truth != initial_truth
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_canonical_truth_changed"
            )
        unsigned = {
            "schema": RECONCILIATION_RECEIPT_SCHEMA,
            "ok": True,
            "release_revision": plan.revision,
            "plan_sha256": plan.sha256,
            "base_artifact_sha256": plan.value["base_artifact_sha256"],
            "target_asset_sha256": plan.value["target_asset_sha256"],
            "postgresql_major": plan.value["postgresql_major"],
            "control_install_artifact_sha256": plan.value[
                "control_install_artifact_sha256"
            ],
            "control_retire_artifact_sha256": plan.value[
                "control_retire_artifact_sha256"
            ],
            "control_foundation_contract_sha256": plan.value[
                "control_foundation_contract_sha256"
            ],
            "expected_old_contract_sha256": plan.value[
                "expected_old_contract_sha256"
            ],
            "target_contract_sha256": plan.value["target_contract_sha256"],
            "initial_contract_sha256": intent["initial_contract_sha256"],
            "final_contract_sha256": plan.value["target_contract_sha256"],
            "initial_canonical_truth": initial_truth.value,
            "final_canonical_truth": final_canonical_truth.value,
            "authorization_sha256": intent["authorization_sha256"],
            "preflight_sha256": intent["preflight_sha256"],
            "owner_frame_receipt_sha256": intent[
                "owner_frame_receipt_sha256"
            ],
            "truth_receipt_sha256": intent["truth_receipt_sha256"],
            "authorized_intent_sha256": intent[
                "authorized_intent_sha256"
            ],
            "mode": intent["mode"],
            "mutation_applied": intent["mutation_required"],
            "completed_at_unix": completed_at_unix,
        }
        value = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
        self._publish(plan, "terminal", _canonical_bytes(value))
        return self.load_terminal(plan) or {}


def _timestamp(value: Any, code: str) -> int:
    if type(value) is not int or value < 0:
        raise SchemaReconciliationError(code)
    return value


def _require_live_authorization(
    authorization: SchemaReconciliationAuthorization,
    *,
    now_unix: int,
) -> None:
    current = _timestamp(now_unix, "schema_reconciliation_clock_invalid")
    if not (
        authorization.value["issued_at_unix"]
        <= current
        < authorization.value["expires_at_unix"]
    ):
        raise SchemaReconciliationError(
            "schema_reconciliation_authorization_expired"
        )


def _validate_authorized_intent(
    plan: SchemaReconciliationPlan,
    value: Mapping[str, Any],
) -> None:
    code = "schema_reconciliation_authorized_intent_invalid"
    if not isinstance(value, Mapping) or set(value) != _AUTHORIZED_INTENT_FIELDS:
        raise SchemaReconciliationError(code)
    unsigned = {
        key: item
        for key, item in value.items()
        if key != "authorized_intent_sha256"
    }
    mode = value.get("mode")
    expected_initial = (
        plan.value["expected_old_contract_sha256"]
        if mode == "reconcile_missing_helper"
        else plan.value["target_contract_sha256"]
        if mode == "adopt_existing_target"
        else None
    )
    try:
        preflight = value.get("preflight")
        _validate_preflight(plan, preflight)
        initial_truth = CanonicalTruthReceipt.from_mapping(
            value.get("initial_canonical_truth")
        )
        authorization = SchemaReconciliationAuthorization.from_mapping(
            value.get("authorization")
        )
        owner_frame = _validate_owner_authorization_frame(
            plan,
            preflight=preflight,
            truth=initial_truth,
            authorization=authorization,
            value=value.get("owner_authorization_frame"),
        )
    except (SchemaReconciliationError, TypeError, AttributeError):
        raise SchemaReconciliationError(code) from None
    admitted = _timestamp(value.get("admitted_at_unix"), code)
    if (
        value.get("schema") != RECONCILIATION_AUTHORIZED_INTENT_SCHEMA
        or value.get("release_revision") != plan.revision
        or value.get("plan_sha256") != plan.sha256
        or value.get("base_artifact_sha256") != plan.value["base_artifact_sha256"]
        or value.get("target_asset_sha256") != plan.value["target_asset_sha256"]
        or value.get("postgresql_major") != plan.value["postgresql_major"]
        or any(
            value.get(name) != plan.value[name]
            for name in (
                "control_install_artifact_sha256",
                "control_retire_artifact_sha256",
                "control_foundation_contract_sha256",
            )
        )
        or value.get("expected_old_contract_sha256")
        != plan.value["expected_old_contract_sha256"]
        or value.get("target_contract_sha256")
        != plan.value["target_contract_sha256"]
        or value.get("initial_contract_sha256") != expected_initial
        or value.get("preflight") != preflight
        or value.get("owner_authorization_frame") != owner_frame
        or value.get("authorization") != authorization.value
        or preflight["observed_contract_sha256"] != expected_initial
        or value.get("authorization_sha256") != authorization.sha256
        or value.get("preflight_sha256") != preflight["preflight_sha256"]
        or value.get("owner_frame_receipt_sha256")
        != owner_frame["receipt_sha256"]
        or value.get("truth_receipt_sha256") != initial_truth.sha256
        or value.get("mutation_required")
        is not (mode == "reconcile_missing_helper")
        or not authorization.value["issued_at_unix"]
        <= admitted
        < authorization.value["expires_at_unix"]
        or value.get("authorized_intent_sha256") != _sha256_json(unsigned)
    ):
        raise SchemaReconciliationError(code)


def _validate_terminal(plan: SchemaReconciliationPlan, value: Mapping[str, Any]) -> None:
    if set(value) != _RECEIPT_FIELDS:
        raise SchemaReconciliationError("schema_reconciliation_terminal_invalid")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    mode = value.get("mode")
    expected_initial = (
        plan.value["expected_old_contract_sha256"]
        if mode == "reconcile_missing_helper"
        else plan.value["target_contract_sha256"]
        if mode == "adopt_existing_target"
        else None
    )
    try:
        initial_truth = CanonicalTruthReceipt.from_mapping(
            value.get("initial_canonical_truth")
        )
        final_truth = CanonicalTruthReceipt.from_mapping(
            value.get("final_canonical_truth")
        )
    except (SchemaReconciliationError, TypeError):
        raise SchemaReconciliationError(
            "schema_reconciliation_terminal_invalid"
        ) from None
    if (
        value.get("schema") != RECONCILIATION_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("release_revision") != plan.revision
        or value.get("plan_sha256") != plan.sha256
        or value.get("base_artifact_sha256") != plan.value["base_artifact_sha256"]
        or value.get("target_asset_sha256") != plan.value["target_asset_sha256"]
        or value.get("postgresql_major") != plan.value["postgresql_major"]
        or any(
            value.get(name) != plan.value[name]
            for name in (
                "control_install_artifact_sha256",
                "control_retire_artifact_sha256",
                "control_foundation_contract_sha256",
            )
        )
        or value.get("expected_old_contract_sha256")
        != plan.value["expected_old_contract_sha256"]
        or value.get("target_contract_sha256")
        != plan.value["target_contract_sha256"]
        or value.get("initial_contract_sha256") != expected_initial
        or value.get("final_contract_sha256")
        != plan.value["target_contract_sha256"]
        or initial_truth != final_truth
        or not isinstance(value.get("authorization_sha256"), str)
        or _SHA256.fullmatch(str(value.get("authorization_sha256"))) is None
        or not isinstance(value.get("preflight_sha256"), str)
        or _SHA256.fullmatch(str(value.get("preflight_sha256"))) is None
        or not isinstance(value.get("owner_frame_receipt_sha256"), str)
        or _SHA256.fullmatch(
            str(value.get("owner_frame_receipt_sha256"))
        )
        is None
        or value.get("truth_receipt_sha256") != initial_truth.sha256
        or not isinstance(value.get("authorized_intent_sha256"), str)
        or _SHA256.fullmatch(
            str(value.get("authorized_intent_sha256"))
        )
        is None
        or value.get("mutation_applied")
        is not (mode == "reconcile_missing_helper")
        or value.get("receipt_sha256") != _sha256_json(unsigned)
    ):
        raise SchemaReconciliationError("schema_reconciliation_terminal_invalid")
    _timestamp(value.get("completed_at_unix"), "schema_reconciliation_terminal_invalid")


def _observe(transaction: SchemaReconciliationTransaction) -> SchemaContract:
    try:
        value = transaction.observe_contract()
    except SchemaReconciliationError:
        raise
    except BaseException as exc:
        raise SchemaReconciliationError("schema_reconciliation_observation_failed") from exc
    if not isinstance(value, SchemaContract):
        raise SchemaReconciliationError("schema_reconciliation_observation_invalid")
    return value


def _lock_canonical_truth(transaction: SchemaReconciliationTransaction) -> None:
    try:
        transaction.lock_canonical_truth()
    except SchemaReconciliationError:
        raise
    except BaseException as exc:
        raise SchemaReconciliationError(
            "schema_reconciliation_canonical_truth_lock_failed"
        ) from exc


def _observe_canonical_truth(
    transaction: SchemaReconciliationTransaction,
) -> CanonicalTruthReceipt:
    try:
        value = transaction.observe_canonical_truth()
    except SchemaReconciliationError:
        raise
    except BaseException as exc:
        raise SchemaReconciliationError(
            "schema_reconciliation_canonical_truth_observation_failed"
        ) from exc
    if not isinstance(value, CanonicalTruthReceipt):
        raise SchemaReconciliationError(
            "schema_reconciliation_canonical_truth_invalid"
        )
    return value


def execute_schema_reconciliation(
    plan: SchemaReconciliationPlan,
    *,
    target: SchemaContract,
    preflight: Mapping[str, Any],
    authorization: SchemaReconciliationAuthorization,
    owner_authorization_frame: Mapping[str, Any],
    database: SchemaReconciliationDatabase,
    journal: AppendOnlySchemaReconciliationJournal | None = None,
    now: Callable[[], int] = lambda: int(time.time()),
) -> Mapping[str, Any]:
    """Apply or replay the exact one-helper reconciliation plan.

    One authorized intent containing the full preflight, verified owner-frame
    receipt, core authorization, and initial truth is published atomically
    before SQL.  Admission requires live authority; a byte-identical durable
    retry remains authorized after expiry.  Terminal replay is re-attested.
    """

    if (
        not isinstance(plan, SchemaReconciliationPlan)
        or not isinstance(target, SchemaContract)
        or not target.is_target
        or not isinstance(authorization, SchemaReconciliationAuthorization)
        or target.sha256 != plan.value["target_contract_sha256"]
        or _sha256_json(_old_contract_value(target))
        != plan.value["expected_old_contract_sha256"]
    ):
        raise SchemaReconciliationError("schema_reconciliation_plan_binding_invalid")
    _validate_preflight(plan, preflight)
    writer = journal or AppendOnlySchemaReconciliationJournal()
    with writer.lock():
        terminal = writer.load_terminal(plan)
        if terminal is not None:
            intent = writer.load_authorized_intent(plan)
            if intent is None:
                raise SchemaReconciliationError(
                    "schema_reconciliation_authorized_intent_missing"
                )
            terminal_truth = CanonicalTruthReceipt.from_mapping(
                terminal["final_canonical_truth"]
            )
            authorization.validate_binding(
                plan=plan,
                preflight=preflight,
                truth=terminal_truth,
            )
            owner_frame = _validate_owner_authorization_frame(
                plan,
                preflight=preflight,
                truth=terminal_truth,
                authorization=authorization,
                value=owner_authorization_frame,
            )
            if (
                intent["authorization"] != authorization.value
                or intent["preflight"] != preflight
                or intent["owner_authorization_frame"] != owner_frame
                or terminal["authorization_sha256"] != authorization.sha256
                or terminal["preflight_sha256"]
                != preflight["preflight_sha256"]
                or terminal["owner_frame_receipt_sha256"]
                != owner_frame["receipt_sha256"]
            ):
                raise SchemaReconciliationError(
                    "schema_reconciliation_authorization_replayed"
                )
            with database.transaction(
                advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
            ) as transaction:
                _lock_canonical_truth(transaction)
                if _observe(transaction).sha256 != target.sha256:
                    raise SchemaReconciliationError(
                        "schema_reconciliation_terminal_contract_drifted"
                    )
                if _observe_canonical_truth(transaction) != terminal_truth:
                    raise SchemaReconciliationError(
                        "schema_reconciliation_terminal_truth_drifted"
                    )
            return terminal

        intent = writer.load_authorized_intent(plan)
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as transaction:
            _lock_canonical_truth(transaction)
            observed = _observe(transaction)
            initial_truth = _observe_canonical_truth(transaction)
            if observed.sha256 not in {
                plan.value["expected_old_contract_sha256"],
                plan.value["target_contract_sha256"],
            }:
                raise SchemaReconciliationError(
                    "schema_reconciliation_unreviewed_database_drift"
                )
            authorization.validate_binding(
                plan=plan,
                preflight=preflight,
                truth=initial_truth,
            )
            owner_frame = _validate_owner_authorization_frame(
                plan,
                preflight=preflight,
                truth=initial_truth,
                authorization=authorization,
                value=owner_authorization_frame,
            )
            if intent is None:
                if preflight["observed_contract_sha256"] != observed.sha256:
                    raise SchemaReconciliationError(
                        "schema_reconciliation_authorization_stale"
                    )
                current = _timestamp(
                    now(), "schema_reconciliation_clock_invalid"
                )
                _require_live_authorization(
                    authorization,
                    now_unix=current,
                )
                intent = writer.append_authorized_intent(
                    plan,
                    initial_contract_sha256=observed.sha256,
                    initial_canonical_truth=initial_truth,
                    authorization=authorization,
                    preflight=preflight,
                    owner_authorization_frame=owner_frame,
                    admitted_at_unix=current,
                )
            else:
                _validate_authorized_intent(plan, intent)
                if (
                    intent["authorization"] != authorization.value
                    or intent["preflight"] != preflight
                    or intent["owner_authorization_frame"] != owner_frame
                    or intent["authorization_sha256"] != authorization.sha256
                    or intent["truth_receipt_sha256"] != initial_truth.sha256
                    or intent["initial_contract_sha256"]
                    != preflight["observed_contract_sha256"]
                ):
                    raise SchemaReconciliationError(
                        "schema_reconciliation_authorization_replayed"
                    )
                if CanonicalTruthReceipt.from_mapping(
                    intent["initial_canonical_truth"]
                ) != initial_truth:
                    raise SchemaReconciliationError(
                        "schema_reconciliation_canonical_truth_changed"
                    )
                if (
                    intent["mode"] == "adopt_existing_target"
                    and observed.sha256 != target.sha256
                ):
                    raise SchemaReconciliationError(
                        "schema_reconciliation_intent_contract_drifted"
                    )
            if observed.sha256 == plan.value["expected_old_contract_sha256"]:
                if intent["mode"] != "reconcile_missing_helper":
                    raise SchemaReconciliationError(
                        "schema_reconciliation_intent_contract_drifted"
                    )
                try:
                    transaction.apply_missing_helper(
                        authorized_intent_sha256=intent[
                            "authorized_intent_sha256"
                        ],
                    )
                except SchemaReconciliationError:
                    raise
                except BaseException as exc:
                    raise SchemaReconciliationError(
                        "schema_reconciliation_database_apply_failed"
                    ) from exc
                if _observe(transaction).sha256 != target.sha256:
                    raise SchemaReconciliationError(
                        "schema_reconciliation_post_apply_contract_invalid"
                    )
            elif intent["mode"] not in {
                "reconcile_missing_helper",
                "adopt_existing_target",
            }:
                raise SchemaReconciliationError(
                    "schema_reconciliation_intent_contract_drifted"
                )
            final_truth = _observe_canonical_truth(transaction)
            if final_truth != initial_truth:
                raise SchemaReconciliationError(
                    "schema_reconciliation_canonical_truth_changed"
                )

        return writer.append_terminal(
            plan,
            intent=intent,
            final_canonical_truth=final_truth,
            completed_at_unix=_timestamp(
                now(), "schema_reconciliation_clock_invalid"
            ),
        )


__all__ = [
    "AppendOnlySchemaReconciliationJournal",
    "CANONICAL_QUARANTINE_ANCHORS",
    "CANONICAL_TRUTH_LOCK_SQL",
    "CANONICAL_TRUTH_RELATIONS",
    "CANONICAL_TRUTH_RECEIPT_SCHEMA",
    "CanonicalQuarantineAnchorReceipt",
    "CanonicalRelationTruthReceipt",
    "CanonicalTruthReceipt",
    "EXPECTED_MISSING_HELPER_CATALOG_IDENTITY",
    "HelperRoutineCatalogIdentity",
    "MISSING_HELPER_SIGNATURE",
    "POSTGRESQL_MAJOR",
    "RECONCILIATION_AUTHORIZED_INTENT_SCHEMA",
    "RECONCILIATION_OWNER_A2_FRAME_SCHEMA",
    "RECONCILIATION_OWNER_FRAME_RECEIPT_SCHEMA",
    "RECONCILIATION_OWNER_SIGNATURE_NAMESPACE",
    "SchemaContract",
    "SchemaContractAsset",
    "SCHEMA_CONTRACT_ASSET_RELATIVE_PATH",
    "SchemaReconciliationAuthorization",
    "SchemaReconciliationDatabase",
    "SchemaReconciliationError",
    "SchemaReconciliationPlan",
    "SchemaReconciliationTransaction",
    "build_schema_reconciliation_owner_frame_receipt",
    "build_schema_reconciliation_plan",
    "collect_schema_contract",
    "execute_schema_reconciliation",
    "load_release_schema_contract_asset",
    "preflight_schema_reconciliation",
]
