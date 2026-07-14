"""Deterministic isolated-canary Phase-B foundation workflow.

This module owns no semantic decision.  It validates an owner-authored exact
plan and mechanically composes injected, fixed-scope boundaries.  In
particular it cannot choose a database, user, role, SQL payload, service, or
recovery strategy.  Production transport/wiring is intentionally absent until
every injected boundary has its own live preflight.

The persistent bootstrap login is never deleted here.  Its provisional
password is used once through a dedicated self-session boundary, disabled by
``ALTER ROLE CURRENT_USER PASSWORD NULL``, and then proved unusable by a fresh
authentication attempt.  Only the plan-derived temporary Cloud SQL admin is
deleted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

try:  # The workflow can be imported on Windows, but its hardened journal is POSIX-only.
    import fcntl
except ImportError:  # pragma: no cover - exercised by Windows import CI.
    fcntl = None  # type: ignore[assignment]

if sys.platform == "win32":
    # The production boundary is the fixed POSIX Cloud VM.  Keep import and
    # plan inspection available to Windows tooling without importing the
    # POSIX-only Phase-A implementation (which itself owns hardened flock and
    # credential primitives).  Mutation execution is rejected explicitly.
    _FOUNDATION_RUNTIME_AVAILABLE = False
    PROJECT = "adventico-ai-platform"
    VM_NAME = "muncho-canary-v2-01"
    VM_INSTANCE_ID = "9153645328899914617"
    SQL_INSTANCE = "muncho-canary-pg18-v2"
    SQL_HOST = "10.91.0.3"
    SQL_TLS_SERVER_NAME = (
        "14-0d81ef63-2cac-4a64-84ad-c4f58c0cfd56.europe-west3.sql.goog"
    )
    SQL_PORT = 5432
    SQL_DATABASE = "muncho_canary_brain"
    SQL_USER = "muncho_canary_writer_login"
    DATABASE_OWNER_ROLE = "cloudsqlsuperuser"
    PERSISTENT_MEMBERSHIP_GRANTOR = "cloudsqladmin"
    MIGRATION_OWNER_ROLE = "canonical_brain_migration_owner"
    WRITER_ROLE = "canonical_brain_writer"
    CANARY_BOOTSTRAP_ROLE = "canonical_brain_canary_bootstrap"
    CANARY_BOOTSTRAP_LOGIN = "canonical_brain_canary_bootstrap_login"
    WRITER_UID = 999
    WRITER_GID = 994
    DATABASE_CREDENTIAL_PATH = Path(
        "/etc/muncho/credentials/canonical-writer-db-password"
    )

    class FoundationObservation:  # pragma: no cover - Windows import shim.
        pass

    class SealedSQLArtifact:  # pragma: no cover - Windows import shim.
        pass

else:
    _FOUNDATION_RUNTIME_AVAILABLE = True
    from gateway.canonical_writer_foundation import (
        CANARY_BOOTSTRAP_LOGIN,
        CANARY_BOOTSTRAP_ROLE,
        DATABASE_CREDENTIAL_PATH,
        DATABASE_OWNER_ROLE,
        MIGRATION_OWNER_ROLE,
        PERSISTENT_MEMBERSHIP_GRANTOR,
        PROJECT,
        SQL_DATABASE,
        SQL_HOST,
        SQL_INSTANCE,
        SQL_PORT,
        SQL_TLS_SERVER_NAME,
        SQL_USER,
        VM_INSTANCE_ID,
        VM_NAME,
        WRITER_GID,
        WRITER_ROLE,
        WRITER_UID,
        FoundationObservation,
        SealedSQLArtifact,
    )


PHASE_B_PREFLIGHT_SCHEMA = "muncho-canonical-writer-phase-b-preflight.v1"
PHASE_B_RECOVERY_SCHEMA = "muncho-canonical-writer-phase-b-recovery.v1"
PHASE_B_PLAN_SCHEMA = "muncho-canonical-writer-phase-b-plan.v1"
PHASE_B_APPROVAL_SCHEMA = "muncho-canonical-writer-phase-b-approval.v1"
PHASE_B_JOURNAL_SCHEMA = "muncho-canonical-writer-phase-b-journal.v1"
PHASE_B_ROLE_RECEIPT_SCHEMA = (
    "muncho-canonical-writer-foundation-phase-b-role-preterminal.v1"
)
PHASE_B_HBA_RECEIPT_SCHEMA = "muncho-canonical-writer-phase-b-hba-rejection.v1"
PHASE_B_SELF_DISABLE_SCHEMA = (
    "muncho-canonical-writer-phase-b-bootstrap-self-disable.v1"
)
PHASE_B_PREDELETE_SCHEMA = "muncho-canonical-writer-phase-b-predelete.v1"
PHASE_B_TERMINAL_OBSERVATION_SCHEMA = (
    "muncho-canonical-writer-phase-b-terminal-observation.v1"
)
PHASE_B_TERMINAL_RECEIPT_SCHEMA = (
    "muncho-canonical-writer-phase-b-terminal.v1"
)

TEMPORARY_ADMIN_AUTHORITY_RECEIPT_SCHEMA = (
    "muncho-cloud-sql-temporary-admin-authority.v1"
)
BOOTSTRAP_AUTHORITY_RECEIPT_SCHEMA = (
    "muncho-cloud-sql-bootstrap-login-authority.v1"
)
TEMPORARY_ADMIN_ABSENCE_RECEIPT_SCHEMA = (
    "muncho-cloud-sql-admin-absence-evidence.v1"
)

SELF_DISABLE_SQL = "ALTER ROLE CURRENT_USER PASSWORD NULL"
ROLE_ARTIFACT_NAME = "phase_b_role"
ROLE_ARTIFACT_PATH = (
    "scripts/sql/canonical_writer_foundation_phase_b_role_v1.sql"
)
PREFLIGHT_ARTIFACT_PATH = (
    "scripts/sql/canonical_writer_foundation_phase_b_preflight_v1.sql"
)
TEMPORARY_ADMIN_PREFIX = "muncho_canary_admin_"

SERVICE_UNITS = (
    "muncho-canary-discord-edge.service",
    "muncho-discord-egress.service",
    "muncho-canonical-writer.service",
    "muncho-canonical-writer-export.service",
    "muncho-canonical-writer-export.timer",
    "hermes-cloud-gateway.service",
)

JOURNAL_EVENT_ORDER = (
    "intent",
    "services_stopped",
    "temporary_admin_authority",
    "role_ready",
    "bootstrap_authority",
    "bootstrap_hba_rejected",
    "bootstrap_password_disabled",
    "predelete_verified",
    "temporary_admin_closed",
    "temporary_admin_predelete_authority",
    "temporary_admin_absent",
    "terminal_observed",
    "terminal",
)
JOURNAL_EVENTS = frozenset(JOURNAL_EVENT_ORDER)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_ADMIN_RE = re.compile(r"^muncho_canary_admin_[0-9a-f]{16}$")
_OPERATION_NAME_RE = re.compile(r"^[A-Za-z0-9._~-]{1,256}$")
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_JOURNAL_MODE = 0o700
_JOURNAL_FILE_MODE = 0o600
_PROCESS_INSTANCE_SHA256 = hashlib.sha256(secrets.token_bytes(32)).hexdigest()


class PhaseBError(RuntimeError):
    """Fail-closed public error with a non-secret stable code."""

    def __init__(self, code: str) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]{2,95}", code) is None:
            code = "canonical_writer_phase_b_failed"
        self.code = code
        super().__init__(code)


def _require_foundation_runtime() -> None:
    if not _FOUNDATION_RUNTIME_AVAILABLE or fcntl is None:
        raise PhaseBError("phase_b_posix_runtime_unavailable")


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
        raise PhaseBError("phase_b_value_not_canonical") from exc
    if len(encoded) > _MAX_JSON_BYTES:
        raise PhaseBError("phase_b_value_too_large")
    return encoded


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _digest(value: Any, code: str = "phase_b_digest_invalid") -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PhaseBError(code)
    return value


def _revision(value: Any) -> str:
    if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
        raise PhaseBError("phase_b_release_revision_invalid")
    return value


def _strict_mapping(value: Any, fields: frozenset[str], code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PhaseBError(code)
    return copy.deepcopy(dict(value))


def _hashed_mapping(
    value: Any,
    *,
    fields: frozenset[str],
    digest_field: str,
    code: str,
) -> dict[str, Any]:
    raw = _strict_mapping(value, fields, code)
    digest = _digest(raw[digest_field], code)
    unsigned = {name: item for name, item in raw.items() if name != digest_field}
    if _sha256_json(unsigned) != digest:
        raise PhaseBError(code)
    return raw


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _zeroize(value: bytearray | None) -> None:
    if value is None:
        return
    try:
        value[:] = b"\x00" * len(value)
    except (BufferError, TypeError, ValueError):
        pass


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        raise PhaseBError("phase_b_posix_identity_unavailable")
    return int(getter())


def _effective_gid() -> int:
    getter = getattr(os, "getegid", None)
    if not callable(getter):
        raise PhaseBError("phase_b_posix_identity_unavailable")
    return int(getter())


def _require_secret(value: Any) -> bytearray:
    if (
        not isinstance(value, bytearray)
        or not 32 <= len(value) <= 128
        or any(byte < 0x21 or byte > 0x7E for byte in value)
    ):
        raise PhaseBError("phase_b_provisional_secret_invalid")
    return value


def _require_secret_free(value: Any) -> None:
    """Reject evidence fields capable of carrying password material.

    Boolean contract names such as ``password_disabled`` and
    ``secret_material_recorded`` are allowed, but the latter must be false.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PhaseBError("phase_b_evidence_not_secret_free")
            folded = key.casefold()
            if folded in {
                "password",
                "password_sha256",
                "password_digest",
                "secret",
                "secret_sha256",
                "secret_digest",
                "verifier",
                "verifier_sha256",
            }:
                raise PhaseBError("phase_b_evidence_not_secret_free")
            if folded in {
                "secret_material_recorded",
                "password_or_digest_recorded",
                "content_or_digest_recorded",
            } and item is not False:
                raise PhaseBError("phase_b_evidence_not_secret_free")
            _require_secret_free(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_secret_free(item)
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise PhaseBError("phase_b_evidence_not_secret_free")


_SERVICE_FIELDS = frozenset(
    {
        "name",
        "load_state",
        "active_state",
        "sub_state",
        "unit_file_state",
        "main_pid",
        "fragment_path",
        "drop_in_paths",
        "triggered_by",
        "triggers",
        "next_elapse_unix_usec",
    }
)
_SERVICES_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "services",
        "services_stopped_and_disabled",
        "observed_at_unix",
        "attestation_sha256",
    }
)


def _validate_services(value: Any, *, release_revision: str) -> dict[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_SERVICES_FIELDS,
        digest_field="attestation_sha256",
        code="phase_b_services_attestation_invalid",
    )
    if (
        raw["schema"] != "muncho-canonical-writer-phase-b-services-stopped.v1"
        or raw["release_revision"] != release_revision
        or raw["services_stopped_and_disabled"] is not True
        or type(raw["observed_at_unix"]) is not int
        or raw["observed_at_unix"] <= 0
        or not isinstance(raw["services"], list)
    ):
        raise PhaseBError("phase_b_services_attestation_invalid")
    services: list[dict[str, Any]] = []
    for item in raw["services"]:
        service = _strict_mapping(
            item, _SERVICE_FIELDS, "phase_b_service_snapshot_invalid"
        )
        if (
            service["name"] not in SERVICE_UNITS
            or service["active_state"] != "inactive"
            or service["sub_state"] not in {"dead", "exited"}
            or service["unit_file_state"]
            not in {"disabled", "masked", "not-found", "static"}
            or service["main_pid"] != 0
            or not isinstance(service["drop_in_paths"], list)
            or not isinstance(service["triggered_by"], list)
            or not isinstance(service["triggers"], list)
            or service["triggered_by"]
            or service["triggers"]
            or service["next_elapse_unix_usec"] not in {None, 0}
        ):
            raise PhaseBError("phase_b_service_snapshot_invalid")
        if service["load_state"] == "not-found":
            if (
                service["unit_file_state"] != "not-found"
                or service["fragment_path"] is not None
                or service["drop_in_paths"]
            ):
                raise PhaseBError("phase_b_service_snapshot_invalid")
        elif (
            service["load_state"] != "loaded"
            or not isinstance(service["fragment_path"], str)
            or not service["fragment_path"].startswith("/")
        ):
            raise PhaseBError("phase_b_service_snapshot_invalid")
        services.append(service)
    if [item["name"] for item in services] != list(SERVICE_UNITS):
        raise PhaseBError("phase_b_service_set_invalid")
    return raw


_CREDENTIAL_FIELDS = frozenset(
    {
        "state",
        "path",
        "device",
        "inode",
        "owner_uid",
        "group_gid",
        "mode",
        "link_count",
        "modification_time_ns",
        "change_time_ns",
        "content_or_digest_recorded",
    }
)


def _validate_credential(value: Any) -> dict[str, Any]:
    raw = _strict_mapping(
        value, _CREDENTIAL_FIELDS, "phase_b_writer_credential_invalid"
    )
    if (
        raw["state"] != "installed"
        or raw["path"] != str(DATABASE_CREDENTIAL_PATH)
        or type(raw["device"]) is not int
        or raw["device"] <= 0
        or type(raw["inode"]) is not int
        or raw["inode"] <= 0
        or raw["owner_uid"] != WRITER_UID
        or raw["group_gid"] != WRITER_GID
        or raw["mode"] != "0400"
        or raw["link_count"] != 1
        or type(raw["modification_time_ns"]) is not int
        or type(raw["change_time_ns"]) is not int
        or raw["content_or_digest_recorded"] is not False
    ):
        raise PhaseBError("phase_b_writer_credential_invalid")
    return raw


_MEMBERSHIP_FIELDS = frozenset(
    {
        "granted_role",
        "member_role",
        "grantor",
        "admin_option",
        "inherit_option",
        "set_option",
    }
)
_DB_PREFLIGHT_FIELDS = frozenset(
    {
        "schema",
        "preflight",
        "terminal",
        "database",
        "database_owner",
        "postgres_version_num",
        "session_user",
        "current_user",
        "roles",
        "memberships",
        "temporary_admin_roles",
        "bootstrap_role_absent",
        "bootstrap_login_absent",
        "namespaces",
        "event_log",
        "writer_ping",
        "legacy_archive",
        "target_database",
        "other_connectable_databases",
        "managed_cloudsqladmin",
        "secret_material_recorded",
        "unsigned_receipt_jsonb_text",
        "receipt_sha256",
    }
)

_ROLE_ROW_FIELDS = frozenset(
    {
        "oid",
        "name",
        "can_login",
        "inherits",
        "superuser",
        "create_database",
        "create_role",
        "replication",
        "bypass_row_security",
        "connection_limit",
        "validity_is_unbounded",
        "configuration_is_empty",
    }
)
_ACL_ROW_FIELDS = frozenset(
    {
        "grantor_oid",
        "grantor",
        "grantee_oid",
        "grantee",
        "privilege",
        "grantable",
    }
)
_DATABASE_ROW_FIELDS = frozenset(
    {
        "oid",
        "name",
        "owner_oid",
        "owner",
        "allow_connections",
        "is_template",
        "connection_limit",
        "acl_is_null",
        "acl",
        "effective_public_connect",
        "effective_public_temporary",
    }
)


def _oid(value: Any, code: str = "phase_b_database_preflight_invalid") -> str:
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise PhaseBError(code)
    return value


def _validate_database_receipt_envelope(value: Any, *, code: str) -> dict[str, Any]:
    """Validate the exact PostgreSQL JSONB text rather than reconstructing it.

    PostgreSQL's JSONB text ordering is not the canonical JSON encoding used by
    the outer plan.  The SQL receipt therefore carries its exact unsigned text;
    preserving and hashing those bytes is the only replay-safe verification.
    """

    raw = _strict_mapping(value, _DB_PREFLIGHT_FIELDS, code)
    unsigned_text = raw["unsigned_receipt_jsonb_text"]
    if not isinstance(unsigned_text, str):
        raise PhaseBError(code)
    try:
        encoded = unsigned_text.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise PhaseBError(code) from exc
    if not encoded or len(encoded) > _MAX_JSON_BYTES:
        raise PhaseBError(code)
    if _sha256_bytes(encoded) != _digest(raw["receipt_sha256"], code):
        raise PhaseBError(code)

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            unsigned_text,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non-finite JSON number")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PhaseBError(code) from exc
    unsigned = {
        name: item
        for name, item in raw.items()
        if name not in {"unsigned_receipt_jsonb_text", "receipt_sha256"}
    }
    if not isinstance(parsed, dict) or parsed != unsigned:
        raise PhaseBError(code)
    return raw


def _validate_role_row(value: Any) -> dict[str, Any]:
    raw = _strict_mapping(value, _ROLE_ROW_FIELDS, "phase_b_role_row_invalid")
    _oid(raw["oid"], "phase_b_role_row_invalid")
    if (
        not isinstance(raw["name"], str)
        or any(type(raw[field]) is not bool for field in (
            "can_login",
            "inherits",
            "superuser",
            "create_database",
            "create_role",
            "replication",
            "bypass_row_security",
            "validity_is_unbounded",
            "configuration_is_empty",
        ))
        or type(raw["connection_limit"]) is not int
    ):
        raise PhaseBError("phase_b_role_row_invalid")
    return raw


def _validate_acl_rows(
    value: Any,
    *,
    allowed_privileges: frozenset[str] = frozenset(
        {"CONNECT", "CREATE", "TEMPORARY"}
    ),
    code: str = "phase_b_database_acl_invalid",
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PhaseBError(code)
    result: list[dict[str, Any]] = []
    for item in value:
        row = _strict_mapping(item, _ACL_ROW_FIELDS, code)
        _oid(row["grantor_oid"], code)
        if row["grantee"] == "PUBLIC":
            if row["grantee_oid"] != "0":
                raise PhaseBError(code)
        else:
            _oid(row["grantee_oid"], code)
        if (
            not isinstance(row["grantor"], str)
            or not isinstance(row["grantee"], str)
            or row["privilege"] not in allowed_privileges
            or type(row["grantable"]) is not bool
        ):
            raise PhaseBError(code)
        result.append(row)
    if result != sorted(
        result,
        key=lambda row: (row["grantee"], row["privilege"], row["grantor"]),
    ):
        raise PhaseBError(code)
    return result


def _validate_database_row(value: Any, *, target: bool) -> dict[str, Any]:
    raw = _strict_mapping(
        value, _DATABASE_ROW_FIELDS, "phase_b_database_scope_invalid"
    )
    _oid(raw["oid"], "phase_b_database_scope_invalid")
    _oid(raw["owner_oid"], "phase_b_database_scope_invalid")
    acl = _validate_acl_rows(raw["acl"])
    if (
        not isinstance(raw["name"], str)
        or not raw["name"]
        or not isinstance(raw["owner"], str)
        or type(raw["allow_connections"]) is not bool
        or type(raw["is_template"]) is not bool
        or type(raw["connection_limit"]) is not int
        or type(raw["acl_is_null"]) is not bool
        or type(raw["effective_public_connect"]) is not bool
        or type(raw["effective_public_temporary"]) is not bool
    ):
        raise PhaseBError("phase_b_database_scope_invalid")
    if target:
        writer_acl = [
            row
            for row in acl
            if row["grantee"] == WRITER_ROLE and row["privilege"] == "CONNECT"
        ]
        if (
            raw["name"] != SQL_DATABASE
            or raw["owner"] != DATABASE_OWNER_ROLE
            or raw["allow_connections"] is not True
            or raw["is_template"] is not False
            or raw["effective_public_connect"] is not False
            or raw["effective_public_temporary"] is not False
            or len(writer_acl) != 1
        ):
            raise PhaseBError("phase_b_target_database_invalid")
        writer_connect = writer_acl[0]
        if (
            writer_connect["grantor"] != DATABASE_OWNER_ROLE
            or writer_connect["grantee"] != WRITER_ROLE
            or writer_connect["privilege"] != "CONNECT"
            or writer_connect["grantable"] is not False
        ):
            raise PhaseBError("phase_b_target_database_invalid")
        if any(
            row["grantee"] in {"PUBLIC", SQL_USER, MIGRATION_OWNER_ROLE}
            for row in acl
        ):
            raise PhaseBError("phase_b_target_database_invalid")
    else:
        managed_hba_boundary = raw["name"] == "cloudsqladmin"
        if any(
            row["grantee"]
            in {
                SQL_USER,
                WRITER_ROLE,
                MIGRATION_OWNER_ROLE,
                CANARY_BOOTSTRAP_ROLE,
                CANARY_BOOTSTRAP_LOGIN,
            }
            for row in acl
        ) or (
            not managed_hba_boundary
            and (
                raw["effective_public_connect"] is not False
                or raw["effective_public_temporary"] is not False
                or any(row["grantee"] == "PUBLIC" for row in acl)
            )
        ):
            raise PhaseBError("phase_b_cross_database_authority_invalid")
    return raw


_EVENT_COLUMNS = (
    ("event_id", "uuid"),
    ("schema_version", "text"),
    ("event_type", "text"),
    ("occurred_at", "timestamp with time zone"),
    ("case_id", "text"),
    ("source", "jsonb"),
    ("actor", "jsonb"),
    ("subject", "jsonb"),
    ("evidence", "jsonb"),
    ("decision", "jsonb"),
    ("status", "jsonb"),
    ("next_action", "jsonb"),
    ("safety", "jsonb"),
    ("payload", "jsonb"),
)

_TYPE_OIDS = {
    "integer": "23",
    "text": "25",
    "timestamp with time zone": "1184",
    "uuid": "2950",
    "jsonb": "3802",
}
_ARCHIVE_COLUMNS = (
    ("event_id", "uuid", True, False),
    ("schema_version", "text", True, False),
    ("event_type", "text", True, False),
    ("occurred_at", "timestamp with time zone", True, False),
    ("case_id", "text", True, False),
    ("source", "jsonb", True, True),
    ("actor", "jsonb", True, True),
    ("subject", "jsonb", True, True),
    ("evidence", "jsonb", True, True),
    ("decision", "jsonb", True, True),
    ("status", "jsonb", True, True),
    ("next_action", "jsonb", True, True),
    ("safety", "jsonb", True, True),
    ("payload", "jsonb", True, True),
    ("inserted_at", "timestamp with time zone", True, True),
    ("idempotency_key", "text", False, False),
    ("source_spool", "text", False, False),
    ("spool_line_number", "integer", False, False),
    ("raw_event_sha256", "text", False, False),
)
_NAMESPACE_FIELDS = frozenset(
    {"oid", "name", "owner_oid", "owner", "acl_is_null", "acl"}
)
_RELATION_FIELDS = frozenset(
    {
        "namespace_oid",
        "oid",
        "owner_oid",
        "owner",
        "relation_kind",
        "persistence",
        "is_partition",
        "access_method",
        "tablespace_oid",
        "row_security",
        "force_row_security",
        "replica_identity",
        "options_are_empty",
        "attribute_slots",
        "relation_acl_is_null",
        "relation_acl",
        "columns",
        "constraints",
        "indexes",
        "user_triggers",
        "rules",
        "policies",
        "inheritance",
    }
)
_LEGACY_OWNER_FIELDS = frozenset(
    {
        "owner_superuser",
        "owner_create_database",
        "owner_create_role",
        "owner_replication",
        "owner_bypass_row_security",
        "owner_connection_limit",
        "owner_validity_is_unbounded",
        "owner_configuration_is_empty",
    }
)
_RELATION_COLUMN_FIELDS = frozenset(
    {
        "position",
        "name",
        "type_oid",
        "type",
        "not_null",
        "has_default",
        "default_expression_sha256",
        "identity",
        "generated",
        "has_missing",
        "is_local",
        "inheritance_count",
        "array_dimensions",
        "collation_is_type_default",
        "storage_is_type_default",
        "statistics_target",
        "options_are_empty",
        "fdw_options_are_empty",
        "acl_is_null",
        "acl",
    }
)
_CONSTRAINT_FIELDS = frozenset(
    {
        "oid",
        "name",
        "type",
        "validated",
        "deferrable",
        "initially_deferred",
        "no_inherit",
        "index_oid",
        "parent_constraint_oid",
        "column_numbers",
        "column_names",
        "definition_sha256",
    }
)
_INDEX_FIELDS = frozenset(
    {
        "oid",
        "name",
        "owner_oid",
        "owner",
        "relation_kind",
        "persistence",
        "access_method",
        "tablespace_oid",
        "options_are_empty",
        "unique",
        "nulls_not_distinct",
        "primary",
        "exclusion",
        "immediate",
        "clustered",
        "valid",
        "check_xmin",
        "ready",
        "live",
        "replica_identity",
        "key_attribute_count",
        "attribute_count",
        "key_columns",
        "operator_classes",
        "collation_oids",
        "index_options",
        "expressions_present",
        "expressions_sha256",
        "predicate_present",
        "predicate_sha256",
        "constraint_oids",
    }
)


def _catalog_oid(value: Any, *, allow_zero: bool, code: str) -> str:
    pattern = r"(?:0|[1-9][0-9]*)" if allow_zero else r"[1-9][0-9]*"
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise PhaseBError(code)
    return value


def _validate_namespaces(value: Any) -> Mapping[str, Mapping[str, Any]]:
    code = "phase_b_namespace_identity_invalid"
    if not isinstance(value, list) or len(value) != 3:
        raise PhaseBError(code)
    rows: list[dict[str, Any]] = []
    for item in value:
        row = _strict_mapping(item, _NAMESPACE_FIELDS, code)
        _oid(row["oid"], code)
        _oid(row["owner_oid"], code)
        acl = _validate_acl_rows(
            row["acl"],
            allowed_privileges=frozenset({"CREATE", "USAGE"}),
            code=code,
        )
        if (
            not isinstance(row["name"], str)
            or not isinstance(row["owner"], str)
            or row["acl_is_null"] is not False
            or not acl
        ):
            raise PhaseBError(code)
        rows.append(row)
    if rows != sorted(rows, key=lambda row: row["name"]):
        raise PhaseBError(code)
    by_name = {row["name"]: row for row in rows}
    if len(by_name) != 3 or set(by_name) != {
        "public",
        "canonical_brain",
        "canonical_brain_legacy_quarantine",
    }:
        raise PhaseBError(code)
    if len({row["oid"] for row in rows}) != 3:
        raise PhaseBError(code)

    public = by_name["public"]
    canonical = by_name["canonical_brain"]
    legacy = by_name["canonical_brain_legacy_quarantine"]
    public_acl = public["acl"]
    canonical_acl = canonical["acl"]
    legacy_acl = legacy["acl"]
    if (
        canonical["owner"] != MIGRATION_OWNER_ROLE
        or not isinstance(legacy["owner"], str)
        or legacy["owner"] in {
            "postgres",
            "cloudsqladmin",
            DATABASE_OWNER_ROLE,
            MIGRATION_OWNER_ROLE,
            WRITER_ROLE,
            CANARY_BOOTSTRAP_ROLE,
            CANARY_BOOTSTRAP_LOGIN,
            SQL_USER,
        }
        or _ADMIN_RE.fullmatch(legacy["owner"]) is not None
        or any(
            acl_row["grantee"] == "PUBLIC"
            for namespace_row in rows
            for acl_row in namespace_row["acl"]
        )
        or any(
            row["grantee"] not in {public["owner"], MIGRATION_OWNER_ROLE}
            for row in public_acl
        )
        or not any(
            row["grantee"] == MIGRATION_OWNER_ROLE
            and row["privilege"] == "USAGE"
            and row["grantable"] is False
            for row in public_acl
        )
        or any(
            row["grantee"] not in {MIGRATION_OWNER_ROLE, WRITER_ROLE}
            for row in canonical_acl
        )
        or not any(
            row["grantee"] == WRITER_ROLE
            and row["privilege"] == "USAGE"
            and row["grantable"] is False
            for row in canonical_acl
        )
        or any(row["grantee"] != legacy["owner"] for row in legacy_acl)
    ):
        raise PhaseBError(code)
    return by_name


def _validate_relation_column(
    value: Any,
    *,
    expected_position: int,
    expected_name: str,
    expected_type: str,
    expected_not_null: bool,
    expected_default: bool,
    code: str,
) -> dict[str, Any]:
    raw = _strict_mapping(value, _RELATION_COLUMN_FIELDS, code)
    _oid(raw["type_oid"], code)
    acl = _validate_acl_rows(
        raw["acl"],
        allowed_privileges=frozenset({"INSERT", "REFERENCES", "SELECT", "UPDATE"}),
        code=code,
    )
    boolean_fields = (
        "not_null",
        "has_default",
        "has_missing",
        "is_local",
        "collation_is_type_default",
        "storage_is_type_default",
        "options_are_empty",
        "fdw_options_are_empty",
        "acl_is_null",
    )
    if (
        raw["position"] != expected_position
        or raw["name"] != expected_name
        or raw["type"] != expected_type
        or raw["type_oid"] != _TYPE_OIDS[expected_type]
        or any(type(raw[field]) is not bool for field in boolean_fields)
        or raw["not_null"] is not expected_not_null
        or raw["has_default"] is not expected_default
        or raw["identity"] != ""
        or raw["generated"] != ""
        or raw["has_missing"] is not False
        or raw["is_local"] is not True
        or raw["inheritance_count"] != 0
        or raw["array_dimensions"] != 0
        or raw["collation_is_type_default"] is not True
        or raw["storage_is_type_default"] is not True
        or raw["statistics_target"] is not None
        or raw["options_are_empty"] is not True
        or raw["fdw_options_are_empty"] is not True
        or raw["acl_is_null"] is not True
        or acl != []
    ):
        raise PhaseBError(code)
    if expected_default:
        _digest(raw["default_expression_sha256"], code)
    elif raw["default_expression_sha256"] is not None:
        raise PhaseBError(code)
    return raw


def _validate_primary_constraint(
    value: Any,
    *,
    expected_name: str,
    code: str,
) -> dict[str, Any]:
    raw = _strict_mapping(value, _CONSTRAINT_FIELDS, code)
    _oid(raw["oid"], code)
    _oid(raw["index_oid"], code)
    _catalog_oid(raw["parent_constraint_oid"], allow_zero=True, code=code)
    _digest(raw["definition_sha256"], code)
    if (
        raw["name"] != expected_name
        or raw["type"] != "p"
        or raw["validated"] is not True
        or raw["deferrable"] is not False
        or raw["initially_deferred"] is not False
        or type(raw["no_inherit"]) is not bool
        or raw["parent_constraint_oid"] != "0"
        or raw["column_numbers"] != [1]
        or raw["column_names"] != ["event_id"]
    ):
        raise PhaseBError(code)
    return raw


def _validate_index(
    value: Any, *, owner: str, owner_oid: str, code: str
) -> dict[str, Any]:
    raw = _strict_mapping(value, _INDEX_FIELDS, code)
    _oid(raw["oid"], code)
    _oid(raw["owner_oid"], code)
    _catalog_oid(raw["tablespace_oid"], allow_zero=True, code=code)
    boolean_fields = (
        "options_are_empty",
        "unique",
        "nulls_not_distinct",
        "primary",
        "exclusion",
        "immediate",
        "clustered",
        "valid",
        "check_xmin",
        "ready",
        "live",
        "replica_identity",
        "expressions_present",
        "predicate_present",
    )
    if (
        not isinstance(raw["name"], str)
        or not raw["name"]
        or raw["owner"] != owner
        or raw["owner_oid"] != owner_oid
        or raw["relation_kind"] != "i"
        or raw["persistence"] != "p"
        or raw["access_method"] != "btree"
        or raw["tablespace_oid"] != "0"
        or any(type(raw[field]) is not bool for field in boolean_fields)
        or raw["options_are_empty"] is not True
        or raw["nulls_not_distinct"] is not False
        or raw["exclusion"] is not False
        or raw["immediate"] is not True
        or raw["clustered"] is not False
        or raw["valid"] is not True
        or raw["check_xmin"] is not False
        or raw["ready"] is not True
        or raw["live"] is not True
        or raw["replica_identity"] is not False
        or raw["key_attribute_count"] != 1
        or raw["attribute_count"] != 1
        or not isinstance(raw["key_columns"], list)
        or len(raw["key_columns"]) != 1
        or not isinstance(raw["operator_classes"], list)
        or len(raw["operator_classes"]) != 1
        or not isinstance(raw["collation_oids"], list)
        or len(raw["collation_oids"]) != 1
        or not isinstance(raw["index_options"], list)
        or len(raw["index_options"]) != 1
        or raw["expressions_present"] is not False
        or raw["expressions_sha256"] is not None
        or not isinstance(raw["constraint_oids"], list)
    ):
        raise PhaseBError(code)
    key = _strict_mapping(
        raw["key_columns"][0],
        frozenset({"position", "attribute_number", "name"}),
        code,
    )
    operator = _strict_mapping(
        raw["operator_classes"][0],
        frozenset({"position", "oid", "schema", "name"}),
        code,
    )
    _oid(operator["oid"], code)
    for collation_oid in raw["collation_oids"]:
        _catalog_oid(collation_oid, allow_zero=True, code=code)
    if (
        key["position"] != 1
        or type(key["attribute_number"]) is not int
        or key["attribute_number"] <= 0
        or not isinstance(key["name"], str)
        or not key["name"]
        or operator["position"] != 1
        or not isinstance(operator["schema"], str)
        or not isinstance(operator["name"], str)
        or any(type(item) is not int for item in raw["index_options"])
    ):
        raise PhaseBError(code)
    for oid in raw["constraint_oids"]:
        _oid(oid, code)
    if raw["constraint_oids"] != sorted(
        set(raw["constraint_oids"]), key=lambda item: int(item)
    ):
        raise PhaseBError(code)
    if raw["predicate_present"]:
        _digest(raw["predicate_sha256"], code)
    elif raw["predicate_sha256"] is not None:
        raise PhaseBError(code)
    return raw


def _validate_relation_identity(
    value: Any,
    *,
    namespace_oid: str,
    owner: str,
    owner_oid: str,
    archive: bool,
) -> dict[str, Any]:
    code = (
        "phase_b_legacy_archive_invalid"
        if archive
        else "phase_b_event_log_invalid"
    )
    fields = _RELATION_FIELDS | (_LEGACY_OWNER_FIELDS if archive else frozenset())
    raw = _strict_mapping(value, fields, code)
    _oid(raw["namespace_oid"], code)
    _oid(raw["oid"], code)
    _oid(raw["owner_oid"], code)
    _catalog_oid(raw["tablespace_oid"], allow_zero=True, code=code)
    expected_columns: Sequence[tuple[str, str, bool, bool]] = (
        _ARCHIVE_COLUMNS
        if archive
        else tuple((name, type_name, True, False) for name, type_name in _EVENT_COLUMNS)
    )
    boolean_fields = (
        "is_partition",
        "row_security",
        "force_row_security",
        "options_are_empty",
        "relation_acl_is_null",
    )
    if (
        raw["namespace_oid"] != namespace_oid
        or raw["owner"] != owner
        or raw["owner_oid"] != owner_oid
        or raw["relation_kind"] != "r"
        or raw["persistence"] != "p"
        or raw["is_partition"] is not False
        or raw["access_method"] != "heap"
        or raw["tablespace_oid"] != "0"
        or raw["row_security"] is not False
        or raw["force_row_security"] is not False
        or raw["replica_identity"] != "d"
        or raw["options_are_empty"] is not True
        or raw["attribute_slots"] != len(expected_columns)
        or raw["relation_acl_is_null"] is not False
        or any(type(raw[field]) is not bool for field in boolean_fields)
        or not isinstance(raw["columns"], list)
        or len(raw["columns"]) != len(expected_columns)
    ):
        raise PhaseBError(code)
    relation_acl = _validate_acl_rows(
        raw["relation_acl"],
        allowed_privileges=frozenset(
            {
                "DELETE",
                "INSERT",
                "MAINTAIN",
                "REFERENCES",
                "SELECT",
                "TRIGGER",
                "TRUNCATE",
                "UPDATE",
            }
        ),
        code=code,
    )
    if {
        row["privilege"] for row in relation_acl
    } != {
        "DELETE",
        "INSERT",
        "MAINTAIN",
        "REFERENCES",
        "SELECT",
        "TRIGGER",
        "TRUNCATE",
        "UPDATE",
    } or any(
        row["grantee"] != owner
        or row["grantor"] != owner
        or row["grantee_oid"] != owner_oid
        or row["grantor_oid"] != owner_oid
        or row["grantable"] is not False
        for row in relation_acl
    ):
        raise PhaseBError(code)
    for position, (column, expected) in enumerate(
        zip(raw["columns"], expected_columns, strict=True), start=1
    ):
        _validate_relation_column(
            column,
            expected_position=position,
            expected_name=expected[0],
            expected_type=expected[1],
            expected_not_null=expected[2],
            expected_default=expected[3],
            code=code,
        )

    if not isinstance(raw["constraints"], list) or len(raw["constraints"]) != 1:
        raise PhaseBError(code)
    expected_pkey = (
        "canonical_event_log_legacy_v1_pkey"
        if archive
        else "canonical_event_log_pkey"
    )
    primary_constraint = _validate_primary_constraint(
        raw["constraints"][0], expected_name=expected_pkey, code=code
    )
    if not isinstance(raw["indexes"], list):
        raise PhaseBError(code)
    indexes = [
        _validate_index(item, owner=owner, owner_oid=owner_oid, code=code)
        for item in raw["indexes"]
    ]
    if indexes != sorted(indexes, key=lambda item: int(item["oid"])):
        raise PhaseBError(code)
    index_by_name = {item["name"]: item for item in indexes}
    expected_index_keys = (
        {
            "canonical_event_log_legacy_v1_pkey": ("event_id", 1, "uuid_ops"),
            "legacy_case_id_idx": ("case_id", 5, "text_ops"),
            "legacy_event_type_idx": ("event_type", 3, "text_ops"),
            "legacy_occurred_at_idx": (
                "occurred_at",
                4,
                "timestamptz_ops",
            ),
            "legacy_idempotency_key_idx": (
                "idempotency_key",
                16,
                "text_ops",
            ),
        }
        if archive
        else {"canonical_event_log_pkey": ("event_id", 1, "uuid_ops")}
    )
    if len(index_by_name) != len(indexes) or set(index_by_name) != set(
        expected_index_keys
    ):
        raise PhaseBError(code)
    for name, (expected_key, attribute_number, operator_name) in (
        expected_index_keys.items()
    ):
        index = index_by_name[name]
        is_primary = name == expected_pkey
        is_partial_unique = archive and name == "legacy_idempotency_key_idx"
        if (
            index["key_columns"][0]["name"] != expected_key
            or index["key_columns"][0]["attribute_number"] != attribute_number
            or index["operator_classes"][0]["schema"] != "pg_catalog"
            or index["operator_classes"][0]["name"] != operator_name
            or index["index_options"] != [0]
            or index["primary"] is not is_primary
            or index["unique"] is not (is_primary or is_partial_unique)
            or index["predicate_present"] is not is_partial_unique
            or index["constraint_oids"]
            != ([primary_constraint["oid"]] if is_primary else [])
        ):
            raise PhaseBError(code)
        if is_primary and index["oid"] != primary_constraint["index_oid"]:
            raise PhaseBError(code)
    for field in ("user_triggers", "rules", "policies", "inheritance"):
        if raw[field] != []:
            raise PhaseBError(code)
    if archive and (
        raw["owner_superuser"] is not False
        or raw["owner_create_database"] is not False
        or raw["owner_create_role"] is not False
        or raw["owner_replication"] is not False
        or raw["owner_bypass_row_security"] is not False
        or raw["owner_connection_limit"] != -1
        or raw["owner_validity_is_unbounded"] is not True
        or raw["owner_configuration_is_empty"] is not True
    ):
        raise PhaseBError(code)
    return raw


def _validate_event_log(
    value: Any,
    *,
    namespaces: Mapping[str, Mapping[str, Any]],
    migration_owner_oid: str,
) -> None:
    code = "phase_b_event_log_invalid"
    raw = _strict_mapping(value, frozenset({"cardinality", "identity"}), code)
    if raw["cardinality"] != 1:
        raise PhaseBError(code)
    _validate_relation_identity(
        raw["identity"],
        namespace_oid=str(namespaces["public"]["oid"]),
        owner=MIGRATION_OWNER_ROLE,
        owner_oid=migration_owner_oid,
        archive=False,
    )


def _validate_legacy_archive(
    value: Any, *, namespaces: Mapping[str, Mapping[str, Any]]
) -> None:
    code = "phase_b_legacy_archive_invalid"
    raw = _strict_mapping(value, frozenset({"cardinality", "identity"}), code)
    if raw["cardinality"] != 1 or not isinstance(raw["identity"], Mapping):
        raise PhaseBError(code)
    owner = raw["identity"].get("owner")
    legacy_namespace = namespaces["canonical_brain_legacy_quarantine"]
    if (
        not isinstance(owner, str)
        or re.fullmatch(r"[a-z_][a-z0-9_-]{0,63}", owner) is None
        or legacy_namespace["owner"] != owner
    ):
        raise PhaseBError(code)
    _validate_relation_identity(
        raw["identity"],
        namespace_oid=str(legacy_namespace["oid"]),
        owner=owner,
        owner_oid=str(legacy_namespace["owner_oid"]),
        archive=True,
    )


def _database_preflight_projection(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "database_preflight_receipt_sha256": raw["receipt_sha256"],
        "event_log_identity_sha256": _sha256_json(raw["event_log"]),
        "writer_ping_identity_sha256": _sha256_json(raw["writer_ping"]),
        "legacy_archive_identity_sha256": _sha256_json(raw["legacy_archive"]),
        "cross_database_acl_sha256": _sha256_json(
            {
                "target_database": raw["target_database"],
                "other_connectable_databases": raw["other_connectable_databases"],
                "managed_cloudsqladmin": raw["managed_cloudsqladmin"],
            }
        ),
    }


def _validate_writer_ping(
    value: Any,
    *,
    namespaces: Mapping[str, Mapping[str, Any]],
    migration_owner_oid: str,
    writer_role_oid: str,
) -> None:
    raw = _strict_mapping(
        value, frozenset({"cardinality", "routines"}), "phase_b_writer_ping_invalid"
    )
    if raw["cardinality"] != 1 or not isinstance(raw["routines"], list) or len(raw["routines"]) != 1:
        raise PhaseBError("phase_b_writer_ping_invalid")
    routine = _strict_mapping(
        raw["routines"][0],
        frozenset(
            {
                "oid",
                "namespace_oid",
                "owner_oid",
                "owner",
                "language",
                "kind",
                "argument_types",
                "return_type",
                "returns_set",
                "security_definer",
                "leakproof",
                "strict",
                "volatility",
                "parallel",
                "configuration_count",
                "configuration_is_exact",
                "acl_is_null",
                "acl",
                "implementation_sha256",
            }
        ),
        "phase_b_writer_ping_invalid",
    )
    _oid(routine["oid"], "phase_b_writer_ping_invalid")
    _oid(routine["namespace_oid"], "phase_b_writer_ping_invalid")
    _oid(routine["owner_oid"], "phase_b_writer_ping_invalid")
    _digest(routine["implementation_sha256"], "phase_b_writer_ping_invalid")
    if (
        routine["namespace_oid"] != namespaces["canonical_brain"]["oid"]
        or routine["owner"] != MIGRATION_OWNER_ROLE
        or routine["owner_oid"] != migration_owner_oid
        or routine["language"] != "plpgsql"
        or routine["kind"] != "f"
        or routine["argument_types"]
        != [
            {"position": 1, "schema": "pg_catalog", "name": "jsonb"},
            {"position": 2, "schema": "pg_catalog", "name": "jsonb"},
        ]
        or routine["return_type"] != {"schema": "pg_catalog", "name": "jsonb"}
        or routine["returns_set"] is not False
        or routine["security_definer"] is not True
        or routine["leakproof"] is not False
        or routine["strict"] is not False
        or routine["volatility"] != "v"
        or routine["parallel"] != "u"
        or routine["configuration_count"] != 1
        or routine["configuration_is_exact"] is not True
        or routine["acl_is_null"] is not False
        or not isinstance(routine["acl"], list)
    ):
        raise PhaseBError("phase_b_writer_ping_invalid")
    acl = _validate_acl_rows(
        routine["acl"],
        allowed_privileges=frozenset({"EXECUTE"}),
        code="phase_b_writer_ping_invalid",
    )
    execute_grantees = {
        row["grantee"]
        for row in acl
        if row["privilege"] == "EXECUTE" and row["grantable"] is False
    }
    if execute_grantees != {MIGRATION_OWNER_ROLE, WRITER_ROLE} or any(
        row["privilege"] != "EXECUTE"
        or row["grantor"] != MIGRATION_OWNER_ROLE
        or row["grantor_oid"] != migration_owner_oid
        for row in acl
    ):
        raise PhaseBError("phase_b_writer_ping_invalid")
    grantee_oids = {row["grantee"]: row["grantee_oid"] for row in acl}
    if grantee_oids != {
        MIGRATION_OWNER_ROLE: migration_owner_oid,
        WRITER_ROLE: writer_role_oid,
    }:
        raise PhaseBError("phase_b_writer_ping_invalid")


def _validate_managed_cloudsqladmin(
    value: Any,
    *,
    database_scope: Sequence[Mapping[str, Any]],
) -> None:
    raw = _strict_mapping(
        value,
        frozenset({"role_cardinality", "role", "database_privileges"}),
        "phase_b_managed_cloudsqladmin_invalid",
    )
    role = _validate_role_row(raw["role"])
    if (
        raw["role_cardinality"] != 1
        or role["name"] != "cloudsqladmin"
        or not isinstance(raw["database_privileges"], list)
    ):
        raise PhaseBError("phase_b_managed_cloudsqladmin_invalid")
    expected = [(row["name"], row["oid"]) for row in database_scope]
    observed: list[tuple[str, str]] = []
    for item in raw["database_privileges"]:
        row = _strict_mapping(
            item,
            frozenset(
                {
                    "database_oid",
                    "database",
                    "effective_connect",
                    "effective_temporary",
                    "direct_acl",
                }
            ),
            "phase_b_managed_cloudsqladmin_invalid",
        )
        _oid(row["database_oid"], "phase_b_managed_cloudsqladmin_invalid")
        if (
            not isinstance(row["database"], str)
            or row["database"] in {name for name, _oid_value in observed}
            or row["effective_connect"] is not True
            or row["effective_temporary"] is not True
            or not isinstance(row["direct_acl"], list)
        ):
            raise PhaseBError("phase_b_managed_cloudsqladmin_invalid")
        observed.append((row["database"], row["database_oid"]))
        for direct in row["direct_acl"]:
            acl = _strict_mapping(
                direct,
                frozenset({"grantor_oid", "grantor", "privilege", "grantable"}),
                "phase_b_managed_cloudsqladmin_invalid",
            )
            _oid(acl["grantor_oid"], "phase_b_managed_cloudsqladmin_invalid")
            if (
                not isinstance(acl["grantor"], str)
                or acl["privilege"] not in {"CONNECT", "CREATE", "TEMPORARY"}
                or acl["grantable"] is not False
            ):
                raise PhaseBError("phase_b_managed_cloudsqladmin_invalid")
    if observed != sorted(expected):
        raise PhaseBError("phase_b_managed_cloudsqladmin_invalid")


def _validate_initial_foundation(value: Any) -> dict[str, Any]:
    raw = _validate_database_receipt_envelope(
        value, code="phase_b_database_preflight_invalid"
    )
    if not isinstance(raw["roles"], list):
        raise PhaseBError("phase_b_database_preflight_invalid")
    roles = [_validate_role_row(item) for item in raw["roles"]]
    role_by_name = {row["name"]: row for row in roles}
    expected_role_shapes = {
        MIGRATION_OWNER_ROLE: (False, False),
        WRITER_ROLE: (False, True),
        SQL_USER: (True, True),
    }
    if (
        set(role_by_name) != set(expected_role_shapes)
        or len(role_by_name) != len(roles)
        or roles != sorted(roles, key=lambda row: row["name"])
        or len({row["oid"] for row in roles}) != len(roles)
    ):
        raise PhaseBError("phase_b_initial_roles_invalid")
    for name, (can_login, inherits) in expected_role_shapes.items():
        role = role_by_name[name]
        if (
            role["can_login"] is not can_login
            or role["inherits"] is not inherits
            or any(role[field] is not False for field in (
                "superuser",
                "create_database",
                "create_role",
                "replication",
                "bypass_row_security",
            ))
            or role["connection_limit"] != -1
            or role["validity_is_unbounded"] is not True
            or role["configuration_is_empty"] is not True
        ):
            raise PhaseBError("phase_b_initial_roles_invalid")
    if (
        raw["schema"]
        != "muncho-canonical-writer-foundation-phase-b-db-preflight.v1"
        or raw["preflight"] is not True
        or raw["terminal"] is not False
        or raw["database"] != SQL_DATABASE
        or raw["database_owner"] != DATABASE_OWNER_ROLE
        or type(raw["postgres_version_num"]) is not int
        or raw["postgres_version_num"] // 10000 != 18
        or raw["session_user"] != SQL_USER
        or raw["current_user"] != SQL_USER
        or raw["temporary_admin_roles"] != []
        or raw["bootstrap_role_absent"] is not True
        or raw["bootstrap_login_absent"] is not True
        or raw["secret_material_recorded"] is not False
        or not isinstance(raw["memberships"], list)
        or len(raw["memberships"]) != 1
    ):
        raise PhaseBError("phase_b_database_preflight_invalid")
    membership = _strict_mapping(
        raw["memberships"][0],
        _MEMBERSHIP_FIELDS,
        "phase_b_database_preflight_invalid",
    )
    if membership != {
        "granted_role": WRITER_ROLE,
        "member_role": SQL_USER,
        "grantor": PERSISTENT_MEMBERSHIP_GRANTOR,
        "admin_option": False,
        "inherit_option": True,
        "set_option": False,
    }:
        raise PhaseBError("phase_b_database_preflight_invalid")
    namespaces = _validate_namespaces(raw["namespaces"])
    migration_owner_oid = str(role_by_name[MIGRATION_OWNER_ROLE]["oid"])
    writer_role_oid = str(role_by_name[WRITER_ROLE]["oid"])
    expected_namespace_grantee_oids = {
        MIGRATION_OWNER_ROLE: migration_owner_oid,
        WRITER_ROLE: writer_role_oid,
    }
    if (
        namespaces["canonical_brain"]["owner_oid"] != migration_owner_oid
        or any(
            row["grantee"] in expected_namespace_grantee_oids
            and row["grantee_oid"]
            != expected_namespace_grantee_oids[row["grantee"]]
            for namespace in namespaces.values()
            for row in namespace["acl"]
        )
        or any(
            row["grantee_oid"] != namespace["owner_oid"]
            for namespace in namespaces.values()
            for row in namespace["acl"]
            if row["grantee"] == namespace["owner"]
        )
    ):
        raise PhaseBError("phase_b_namespace_identity_invalid")
    _validate_event_log(
        raw["event_log"],
        namespaces=namespaces,
        migration_owner_oid=migration_owner_oid,
    )
    _validate_writer_ping(
        raw["writer_ping"],
        namespaces=namespaces,
        migration_owner_oid=migration_owner_oid,
        writer_role_oid=writer_role_oid,
    )
    _validate_legacy_archive(raw["legacy_archive"], namespaces=namespaces)
    target_database = _validate_database_row(raw["target_database"], target=True)
    writer_connect = next(
        row
        for row in target_database["acl"]
        if row["grantee"] == WRITER_ROLE and row["privilege"] == "CONNECT"
    )
    if (
        writer_connect["grantee_oid"] != role_by_name[WRITER_ROLE]["oid"]
        or writer_connect["grantor_oid"] != target_database["owner_oid"]
    ):
        raise PhaseBError("phase_b_target_database_invalid")
    if not isinstance(raw["other_connectable_databases"], list):
        raise PhaseBError("phase_b_cross_database_authority_invalid")
    other = [
        _validate_database_row(item, target=False)
        for item in raw["other_connectable_databases"]
    ]
    if [item["name"] for item in other] != sorted(item["name"] for item in other):
        raise PhaseBError("phase_b_cross_database_authority_invalid")
    if len({item["name"] for item in other}) != len(other):
        raise PhaseBError("phase_b_cross_database_authority_invalid")
    _validate_managed_cloudsqladmin(
        raw["managed_cloudsqladmin"],
        database_scope=sorted([target_database, *other], key=lambda row: row["name"]),
    )
    _require_secret_free(raw)
    return raw


_CLOUD_PREFLIGHT_FIELDS = frozenset(
    {
        "project",
        "instance",
        "visible_users",
        "user_inventory_sha256",
        "bootstrap_login_absent",
        "temporary_admin_users",
        "user_operations_quiescent",
        "operation_ledger_sha256",
    }
)


def _validate_initial_cloud(value: Any) -> dict[str, Any]:
    raw = _strict_mapping(value, _CLOUD_PREFLIGHT_FIELDS, "phase_b_cloud_invalid")
    if (
        raw["project"] != PROJECT
        or raw["instance"] != SQL_INSTANCE
        or raw["visible_users"] != sorted([SQL_USER, "postgres"])
        or raw["bootstrap_login_absent"] is not True
        or raw["temporary_admin_users"] != []
        or raw["user_operations_quiescent"] is not True
    ):
        raise PhaseBError("phase_b_cloud_invalid")
    _digest(raw["user_inventory_sha256"], "phase_b_cloud_invalid")
    _digest(raw["operation_ledger_sha256"], "phase_b_cloud_invalid")
    return raw


_DATABASE_FIELDS = frozenset(
    {
        "project",
        "instance",
        "host",
        "port",
        "database",
        "database_owner",
        "postgres_version_num",
        "tls_server_name",
        "tls_peer_certificate_sha256",
        "session_user",
        "current_user",
        "session_identity_sha256",
    }
)


def _validate_database(value: Any, *, writer: bool) -> dict[str, Any]:
    raw = _strict_mapping(value, _DATABASE_FIELDS, "phase_b_database_invalid")
    if (
        raw["project"] != PROJECT
        or raw["instance"] != SQL_INSTANCE
        or raw["host"] != SQL_HOST
        or raw["port"] != SQL_PORT
        or raw["database"] != SQL_DATABASE
        or raw["database_owner"] != DATABASE_OWNER_ROLE
        or type(raw["postgres_version_num"]) is not int
        or raw["postgres_version_num"] // 10000 != 18
        or raw["tls_server_name"] != SQL_TLS_SERVER_NAME
        or raw["current_user"] != raw["session_user"]
    ):
        raise PhaseBError("phase_b_database_invalid")
    if writer:
        if raw["session_user"] != SQL_USER:
            raise PhaseBError("phase_b_database_invalid")
    elif not isinstance(raw["session_user"], str) or _ADMIN_RE.fullmatch(
        raw["session_user"]
    ) is None:
        raise PhaseBError("phase_b_database_invalid")
    _digest(raw["tls_peer_certificate_sha256"], "phase_b_database_invalid")
    _digest(raw["session_identity_sha256"], "phase_b_database_invalid")
    return raw


_PREFLIGHT_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "release_manifest_sha256",
        "release_artifacts",
        "release_artifact_set_sha256",
        "database",
        "foundation",
        "credential",
        "services",
        "cloud_sql",
        "observed_at_unix",
        "observation_sha256",
    }
)


@dataclass(frozen=True)
class PhaseBPreflight:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "PhaseBPreflight":
        raw = _hashed_mapping(
            value,
            fields=_PREFLIGHT_FIELDS,
            digest_field="observation_sha256",
            code="phase_b_preflight_invalid",
        )
        revision = _revision(raw["release_revision"])
        _digest(raw["release_manifest_sha256"], "phase_b_preflight_invalid")
        if not isinstance(raw["release_artifacts"], Mapping):
            raise PhaseBError("phase_b_preflight_invalid")
        artifacts = dict(raw["release_artifacts"])
        if (
            not {PREFLIGHT_ARTIFACT_PATH, ROLE_ARTIFACT_PATH}.issubset(artifacts)
            or list(artifacts) != sorted(artifacts)
            or any(
                not isinstance(name, str)
                or not name.startswith(("gateway/", "scripts/", "tests/"))
                or _SHA256_RE.fullmatch(str(digest)) is None
                for name, digest in artifacts.items()
            )
            or _sha256_json(artifacts) != raw["release_artifact_set_sha256"]
        ):
            raise PhaseBError("phase_b_release_artifacts_invalid")
        _validate_database(raw["database"], writer=True)
        _validate_initial_foundation(raw["foundation"])
        _validate_credential(raw["credential"])
        _validate_services(raw["services"], release_revision=revision)
        _validate_initial_cloud(raw["cloud_sql"])
        if type(raw["observed_at_unix"]) is not int or raw["observed_at_unix"] <= 0:
            raise PhaseBError("phase_b_preflight_invalid")
        _require_secret_free(raw)
        return cls(_json_copy(raw))

    @property
    def sha256(self) -> str:
        return str(self.value["observation_sha256"])

    @property
    def revision(self) -> str:
        return str(self.value["release_revision"])

    @property
    def stable_projection(self) -> Mapping[str, Any]:
        value = self.to_mapping()
        value.pop("observed_at_unix")
        value.pop("observation_sha256")
        value["database"].pop("session_identity_sha256")
        value["services"].pop("observed_at_unix")
        value["services"].pop("attestation_sha256")
        return value

    @property
    def stable_sha256(self) -> str:
        return _sha256_json(self.stable_projection)

    def to_mapping(self) -> dict[str, Any]:
        return _json_copy(self.value)


_PLAN_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "owner_subject_sha256",
        "intent_sha256",
        "temporary_admin_username",
        "target",
        "initial_preflight",
        "initial_preflight_sha256",
        "stable_preflight_sha256",
        "preflight_artifact_sha256",
        "role_artifact_sha256",
        "bootstrap_role",
        "bootstrap_login",
        "self_disable_sql_sha256",
        "states",
        "secret_material_recorded",
        "plan_sha256",
    }
)


@dataclass(frozen=True)
class PhaseBPlan:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "PhaseBPlan":
        raw = _hashed_mapping(
            value,
            fields=_PLAN_FIELDS,
            digest_field="plan_sha256",
            code="phase_b_plan_invalid",
        )
        preflight = PhaseBPreflight.from_mapping(raw["initial_preflight"])
        owner = _digest(raw["owner_subject_sha256"], "phase_b_plan_invalid")
        if (
            raw["schema"] != PHASE_B_PLAN_SCHEMA
            or raw["release_revision"] != preflight.revision
            or raw["initial_preflight_sha256"] != preflight.sha256
            or raw["stable_preflight_sha256"] != preflight.stable_sha256
            or raw["bootstrap_role"] != CANARY_BOOTSTRAP_ROLE
            or raw["bootstrap_login"] != CANARY_BOOTSTRAP_LOGIN
            or raw["self_disable_sql_sha256"]
            != _sha256_bytes(SELF_DISABLE_SQL.encode("ascii"))
            or raw["states"] != list(JOURNAL_EVENT_ORDER)
            or raw["secret_material_recorded"] is not False
        ):
            raise PhaseBError("phase_b_plan_invalid")
        target = raw["target"]
        expected_target = {
            "project": PROJECT,
            "vm": VM_NAME,
            "vm_instance_id": VM_INSTANCE_ID,
            "sql_instance": SQL_INSTANCE,
            "host": SQL_HOST,
            "port": SQL_PORT,
            "database": SQL_DATABASE,
            "writer_login": SQL_USER,
        }
        if target != expected_target:
            raise PhaseBError("phase_b_plan_target_invalid")
        artifacts = preflight.value["release_artifacts"]
        if (
            raw["preflight_artifact_sha256"]
            != artifacts[PREFLIGHT_ARTIFACT_PATH]
            or raw["role_artifact_sha256"] != artifacts[ROLE_ARTIFACT_PATH]
        ):
            raise PhaseBError("phase_b_plan_artifact_invalid")
        intent = {
            "release_revision": preflight.revision,
            "owner_subject_sha256": owner,
            "initial_preflight_sha256": preflight.sha256,
            "stable_preflight_sha256": preflight.stable_sha256,
            "preflight_artifact_sha256": raw["preflight_artifact_sha256"],
            "role_artifact_sha256": raw["role_artifact_sha256"],
            "target": expected_target,
        }
        intent_sha = _sha256_json(intent)
        if (
            raw["intent_sha256"] != intent_sha
            or raw["temporary_admin_username"]
            != TEMPORARY_ADMIN_PREFIX + intent_sha[:16]
        ):
            raise PhaseBError("phase_b_plan_intent_invalid")
        _require_secret_free(raw)
        return cls(_json_copy(raw))

    @property
    def sha256(self) -> str:
        return str(self.value["plan_sha256"])

    @property
    def revision(self) -> str:
        return str(self.value["release_revision"])

    @property
    def temporary_admin_username(self) -> str:
        return str(self.value["temporary_admin_username"])

    @property
    def owner_subject_sha256(self) -> str:
        return str(self.value["owner_subject_sha256"])

    @property
    def preflight(self) -> PhaseBPreflight:
        return PhaseBPreflight.from_mapping(self.value["initial_preflight"])

    def to_mapping(self) -> dict[str, Any]:
        return _json_copy(self.value)


def build_phase_b_plan(
    preflight: PhaseBPreflight,
    *,
    owner_subject_sha256: str,
) -> PhaseBPlan:
    if not isinstance(preflight, PhaseBPreflight):
        raise TypeError("PhaseBPreflight is required")
    owner = _digest(owner_subject_sha256, "phase_b_owner_subject_invalid")
    artifacts = preflight.value["release_artifacts"]
    target = {
        "project": PROJECT,
        "vm": VM_NAME,
        "vm_instance_id": VM_INSTANCE_ID,
        "sql_instance": SQL_INSTANCE,
        "host": SQL_HOST,
        "port": SQL_PORT,
        "database": SQL_DATABASE,
        "writer_login": SQL_USER,
    }
    intent = {
        "release_revision": preflight.revision,
        "owner_subject_sha256": owner,
        "initial_preflight_sha256": preflight.sha256,
        "stable_preflight_sha256": preflight.stable_sha256,
        "preflight_artifact_sha256": artifacts[PREFLIGHT_ARTIFACT_PATH],
        "role_artifact_sha256": artifacts[ROLE_ARTIFACT_PATH],
        "target": target,
    }
    intent_sha = _sha256_json(intent)
    unsigned = {
        "schema": PHASE_B_PLAN_SCHEMA,
        "release_revision": preflight.revision,
        "owner_subject_sha256": owner,
        "intent_sha256": intent_sha,
        "temporary_admin_username": TEMPORARY_ADMIN_PREFIX + intent_sha[:16],
        "target": target,
        "initial_preflight": preflight.to_mapping(),
        "initial_preflight_sha256": preflight.sha256,
        "stable_preflight_sha256": preflight.stable_sha256,
        "preflight_artifact_sha256": artifacts[PREFLIGHT_ARTIFACT_PATH],
        "role_artifact_sha256": artifacts[ROLE_ARTIFACT_PATH],
        "bootstrap_role": CANARY_BOOTSTRAP_ROLE,
        "bootstrap_login": CANARY_BOOTSTRAP_LOGIN,
        "self_disable_sql_sha256": _sha256_bytes(SELF_DISABLE_SQL.encode("ascii")),
        "states": list(JOURNAL_EVENT_ORDER),
        "secret_material_recorded": False,
    }
    return PhaseBPlan.from_mapping(
        {**unsigned, "plan_sha256": _sha256_json(unsigned)}
    )


_APPROVAL_FIELDS = frozenset(
    {
        "schema",
        "plan_sha256",
        "owner_subject_sha256",
        "approval_source_sha256",
        "approved",
        "issued_at_unix",
        "expires_at_unix",
        "secret_material_recorded",
        "approval_sha256",
    }
)


@dataclass(frozen=True)
class PhaseBApproval:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        plan: PhaseBPlan,
        now_unix: int | None = None,
    ) -> "PhaseBApproval":
        if not isinstance(plan, PhaseBPlan):
            raise TypeError("PhaseBPlan is required")
        raw = _hashed_mapping(
            value,
            fields=_APPROVAL_FIELDS,
            digest_field="approval_sha256",
            code="phase_b_approval_invalid",
        )
        current = int(time.time()) if now_unix is None else now_unix
        if (
            raw["schema"] != PHASE_B_APPROVAL_SCHEMA
            or raw["plan_sha256"] != plan.sha256
            or raw["owner_subject_sha256"] != plan.owner_subject_sha256
            or raw["approved"] is not True
            or type(raw["issued_at_unix"]) is not int
            or type(raw["expires_at_unix"]) is not int
            or not raw["issued_at_unix"] <= current < raw["expires_at_unix"]
            or raw["expires_at_unix"] - raw["issued_at_unix"] > 3600
            or raw["secret_material_recorded"] is not False
        ):
            raise PhaseBError("phase_b_approval_invalid")
        _digest(raw["approval_source_sha256"], "phase_b_approval_invalid")
        _require_secret_free(raw)
        return cls(_json_copy(raw))

    @property
    def sha256(self) -> str:
        return str(self.value["approval_sha256"])

    def to_mapping(self) -> dict[str, Any]:
        return _json_copy(self.value)


def _validate_operation_row(
    value: Any,
    *,
    owner_subject_sha256: str | None = None,
    expected_types: frozenset[str] | None = None,
) -> list[Any]:
    if (
        not isinstance(value, list)
        or len(value) != 5
        or not isinstance(value[0], str)
        or _OPERATION_NAME_RE.fullmatch(value[0]) is None
        or not isinstance(value[1], str)
        or value[2] not in {"PENDING", "RUNNING", "DONE"}
        or _SHA256_RE.fullmatch(str(value[3])) is None
        or type(value[4]) is not bool
    ):
        raise PhaseBError("phase_b_cloud_operation_invalid")
    if expected_types is not None and value[1] not in expected_types:
        raise PhaseBError("phase_b_cloud_operation_invalid")
    if owner_subject_sha256 is not None and value[3] != owner_subject_sha256:
        raise PhaseBError("phase_b_cloud_operation_actor_invalid")
    if value[2] == "DONE" and value[4] is not True:
        raise PhaseBError("phase_b_cloud_operation_failed")
    return list(value)


def _validate_temporary_admin_authority(
    value: Any,
    *,
    plan: PhaseBPlan,
) -> dict[str, Any]:
    fields = frozenset(
        {
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
            "receipt_sha256",
        }
    )
    raw = _hashed_mapping(
        value,
        fields=fields,
        digest_field="receipt_sha256",
        code="phase_b_temporary_admin_authority_invalid",
    )
    if (
        raw["schema"] != TEMPORARY_ADMIN_AUTHORITY_RECEIPT_SCHEMA
        or raw["project"] != PROJECT
        or raw["instance"] != SQL_INSTANCE
        or raw["username_sha256"]
        != _sha256_bytes(plan.temporary_admin_username.encode("ascii"))
        or raw["host"] != ""
        or raw["type"] != "BUILT_IN"
        or raw["user_present"] is not True
        or raw["owner_subject_sha256"] != plan.owner_subject_sha256
        or raw["mutation_context_sha256"] != plan.sha256
        or not isinstance(raw["baseline_operation_names"], list)
        or raw["baseline_operation_names"]
        != sorted(set(raw["baseline_operation_names"]))
        or not isinstance(raw["baseline_user_operations"], list)
    ):
        raise PhaseBError("phase_b_temporary_admin_authority_invalid")
    for row in raw["baseline_user_operations"]:
        _validate_operation_row(row)
    authority = _validate_operation_row(
        raw["authority_operation"],
        owner_subject_sha256=plan.owner_subject_sha256,
        expected_types=frozenset({"CREATE_USER", "UPDATE_USER"}),
    )
    if authority[2:] == ["DONE", plan.owner_subject_sha256, True]:
        pass
    elif authority[2] != "DONE" or authority[4] is not True:
        raise PhaseBError("phase_b_temporary_admin_authority_invalid")
    _require_secret_free(raw)
    return raw


def _validate_bootstrap_authority(
    value: Any,
    *,
    plan: PhaseBPlan,
) -> dict[str, Any]:
    fields = frozenset(
        {
            "schema",
            "project",
            "instance",
            "name",
            "host",
            "type",
            "database_roles",
            "etag",
            "resource_projection_sha256",
            "operation_name",
            "operation_type",
            "owner_subject_sha256",
            "mutation_context_sha256",
            "receipt_sha256",
        }
    )
    raw = _hashed_mapping(
        value,
        fields=fields,
        digest_field="receipt_sha256",
        code="phase_b_bootstrap_authority_invalid",
    )
    normalized_resource = {
        "databaseRoles": [CANARY_BOOTSTRAP_ROLE],
        "etag": raw["etag"],
        "host": "",
        "instance": SQL_INSTANCE,
        "name": CANARY_BOOTSTRAP_LOGIN,
        "project": PROJECT,
        "type": "BUILT_IN",
    }
    if (
        raw["schema"] != BOOTSTRAP_AUTHORITY_RECEIPT_SCHEMA
        or raw["project"] != PROJECT
        or raw["instance"] != SQL_INSTANCE
        or raw["name"] != CANARY_BOOTSTRAP_LOGIN
        or raw["host"] != ""
        or raw["type"] != "BUILT_IN"
        or raw["database_roles"] != [CANARY_BOOTSTRAP_ROLE]
        or not isinstance(raw["etag"], str)
        or not 1 <= len(raw["etag"]) <= 1024
        or any(
            ord(character) < 0x21 or ord(character) > 0x7E
            for character in raw["etag"]
        )
        or not isinstance(raw["operation_name"], str)
        or _OPERATION_NAME_RE.fullmatch(raw["operation_name"]) is None
        or raw["operation_type"] not in {"CREATE_USER", "UPDATE_USER"}
        or raw["owner_subject_sha256"] != plan.owner_subject_sha256
        or raw["mutation_context_sha256"] != plan.sha256
        or raw["resource_projection_sha256"]
        != _sha256_json(normalized_resource)
    ):
        raise PhaseBError("phase_b_bootstrap_authority_invalid")
    _digest(raw["resource_projection_sha256"], "phase_b_bootstrap_authority_invalid")
    _require_secret_free(raw)
    return raw


_ROLE_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "phase",
        "preterminal",
        "database",
        "postgres_version_num",
        "session_user",
        "role",
        "role_outcome",
        "role_contract",
        "connect_contract",
        "temporary_auto_membership",
        "temporary_admin_delete_required",
        "release_revision",
        "artifact_sha256",
        "initial_observation_sha256",
        "approved_plan_sha256",
        "secret_material_recorded",
        "receipt_sha256",
    }
)


def _validate_role_receipt(value: Any, *, plan: PhaseBPlan) -> dict[str, Any]:
    raw = _strict_mapping(value, _ROLE_RECEIPT_FIELDS, "phase_b_role_receipt_invalid")
    if (
        raw["schema"] != PHASE_B_ROLE_RECEIPT_SCHEMA
        or raw["phase"] != "phase_b_role_and_connect"
        or raw["preterminal"] is not True
        or raw["database"] != SQL_DATABASE
        or type(raw["postgres_version_num"]) is not int
        or raw["postgres_version_num"] // 10000 != 18
        or raw["session_user"] != plan.temporary_admin_username
        or raw["role"] != CANARY_BOOTSTRAP_ROLE
        or raw["role_outcome"]
        not in {"created", "adopted_same_admin_predelete", "adopted_zero_membership"}
        or raw["role_contract"]
        != {
            "can_login": False,
            "inherits": False,
            "superuser": False,
            "create_database": False,
            "create_role": False,
            "replication": False,
            "bypass_row_security": False,
            "connection_limit": -1,
            "validity_is_unbounded": True,
            "configuration_is_empty": True,
        }
        or raw["connect_contract"]
        != {
            "database": SQL_DATABASE,
            "privilege": "CONNECT",
            "grantor": DATABASE_OWNER_ROLE,
            "grantable": False,
            "managed_cloudsqladmin_hba_boundary_separate": True,
        }
        or raw["temporary_admin_delete_required"] is not True
        or raw["release_revision"] != plan.revision
        or raw["artifact_sha256"] != plan.value["role_artifact_sha256"]
        or raw["initial_observation_sha256"] != plan.preflight.sha256
        or raw["approved_plan_sha256"] != plan.sha256
        or raw["secret_material_recorded"] is not False
    ):
        raise PhaseBError("phase_b_role_receipt_invalid")
    expected_bridge = None
    if raw["role_outcome"] in {"created", "adopted_same_admin_predelete"}:
        expected_bridge = {
            "granted_role": CANARY_BOOTSTRAP_ROLE,
            "member_role": plan.temporary_admin_username,
            "grantor": PERSISTENT_MEMBERSHIP_GRANTOR,
            "admin_option": True,
            "inherit_option": False,
            "set_option": False,
        }
    if raw["temporary_auto_membership"] != expected_bridge:
        raise PhaseBError("phase_b_role_receipt_invalid")
    _digest(raw["receipt_sha256"], "phase_b_role_receipt_invalid")
    _require_secret_free(raw)
    return _json_copy(raw)


_HBA_FIELDS = frozenset(
    {
        "schema",
        "plan_sha256",
        "bootstrap_authority_receipt_sha256",
        "host",
        "port",
        "tls_server_name",
        "tls_peer_certificate_sha256",
        "user",
        "database",
        "rejected",
        "sqlstate",
        "observed_at_unix",
        "expires_at_unix",
        "secret_material_recorded",
        "receipt_sha256",
    }
)


def _validate_hba_receipt(value: Any, *, plan: PhaseBPlan, authority: Mapping[str, Any]) -> dict[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_HBA_FIELDS,
        digest_field="receipt_sha256",
        code="phase_b_hba_receipt_invalid",
    )
    if (
        raw["schema"] != PHASE_B_HBA_RECEIPT_SCHEMA
        or raw["plan_sha256"] != plan.sha256
        or raw["bootstrap_authority_receipt_sha256"] != authority["receipt_sha256"]
        or raw["host"] != SQL_HOST
        or raw["port"] != SQL_PORT
        or raw["tls_server_name"] != SQL_TLS_SERVER_NAME
        or raw["user"] != CANARY_BOOTSTRAP_LOGIN
        or raw["database"] != "cloudsqladmin"
        or raw["rejected"] is not True
        or raw["sqlstate"] not in {"28000", "28P01"}
        or type(raw["observed_at_unix"]) is not int
        or type(raw["expires_at_unix"]) is not int
        or not raw["observed_at_unix"] < raw["expires_at_unix"]
        or raw["secret_material_recorded"] is not False
    ):
        raise PhaseBError("phase_b_hba_receipt_invalid")
    _digest(raw["tls_peer_certificate_sha256"], "phase_b_hba_receipt_invalid")
    _require_secret_free(raw)
    return raw


_SELF_DISABLE_FIELDS = frozenset(
    {
        "schema",
        "plan_sha256",
        "bootstrap_authority_receipt_sha256",
        "hba_rejection_receipt_sha256",
        "user",
        "database",
        "tls_peer_certificate_sha256",
        "authenticated_as_self",
        "statement_sha256",
        "command_tag",
        "password_disabled",
        "login_remains_true",
        "fresh_denial_connection",
        "denial_sqlstate",
        "password_or_digest_recorded",
        "receipt_sha256",
    }
)


def _validate_self_disable_receipt(
    value: Any,
    *,
    plan: PhaseBPlan,
    authority: Mapping[str, Any],
    hba: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_SELF_DISABLE_FIELDS,
        digest_field="receipt_sha256",
        code="phase_b_self_disable_receipt_invalid",
    )
    if (
        raw["schema"] != PHASE_B_SELF_DISABLE_SCHEMA
        or raw["plan_sha256"] != plan.sha256
        or raw["bootstrap_authority_receipt_sha256"] != authority["receipt_sha256"]
        or raw["hba_rejection_receipt_sha256"] != hba["receipt_sha256"]
        or raw["user"] != CANARY_BOOTSTRAP_LOGIN
        or raw["database"] != SQL_DATABASE
        or raw["tls_peer_certificate_sha256"]
        != hba["tls_peer_certificate_sha256"]
        or raw["authenticated_as_self"] is not True
        or raw["statement_sha256"]
        != _sha256_bytes(SELF_DISABLE_SQL.encode("ascii"))
        or raw["command_tag"] != "ALTER ROLE"
        or raw["password_disabled"] is not True
        or raw["login_remains_true"] is not True
        or raw["fresh_denial_connection"] is not True
        or raw["denial_sqlstate"] not in {"28000", "28P01"}
        or raw["password_or_digest_recorded"] is not False
    ):
        raise PhaseBError("phase_b_self_disable_receipt_invalid")
    _require_secret_free(raw)
    return raw


_PREDELETE_FIELDS = frozenset(
    {
        "schema",
        "plan_sha256",
        "foundation_observation",
        "database_preflight",
        "bootstrap_connect_acl",
        "temporary_auto_membership",
        "other_temporary_admin_references",
        "role_receipt_sha256",
        "bootstrap_authority_receipt_sha256",
        "self_disable_receipt_sha256",
        "temporary_admin_delete_required",
        "preterminal",
        "safe_to_start",
        "secret_material_recorded",
        "receipt_sha256",
    }
)


def _validate_predelete_receipt(
    value: Any,
    *,
    plan: PhaseBPlan,
    role: Mapping[str, Any],
    authority: Mapping[str, Any],
    self_disable: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_PREDELETE_FIELDS,
        digest_field="receipt_sha256",
        code="phase_b_predelete_receipt_invalid",
    )
    observation = FoundationObservation.from_mapping(raw["foundation_observation"])
    terminal_database_state = _validate_recovery_database_preflight(
        raw["database_preflight"], plan=plan
    )
    expected_acl = {
        "database": SQL_DATABASE,
        "grantee": CANARY_BOOTSTRAP_ROLE,
        "grantor": DATABASE_OWNER_ROLE,
        "privilege": "CONNECT",
        "grantable": False,
    }
    expected_bridge = {
        "granted_role": CANARY_BOOTSTRAP_ROLE,
        "member_role": plan.temporary_admin_username,
        "grantor": PERSISTENT_MEMBERSHIP_GRANTOR,
        "admin_option": True,
        "inherit_option": False,
        "set_option": False,
    }
    if role["role_outcome"] == "adopted_zero_membership":
        expected_bridge = None
    expected_database_state = {
        "bootstrap_role_present": True,
        "bootstrap_login_present": True,
        "temporary_admin_users": [plan.temporary_admin_username],
        "temporary_auto_membership_present": expected_bridge is not None,
    }
    if (
        raw["schema"] != PHASE_B_PREDELETE_SCHEMA
        or raw["plan_sha256"] != plan.sha256
        or observation.value["session_user"] != plan.temporary_admin_username
        or observation.value["temporary_admin_roles"]
        != [plan.temporary_admin_username]
        or not observation.membership_ready
        or raw["bootstrap_connect_acl"] != expected_acl
        or raw["temporary_auto_membership"] != expected_bridge
        or terminal_database_state != expected_database_state
        or raw["other_temporary_admin_references"] != []
        or raw["role_receipt_sha256"] != role["receipt_sha256"]
        or raw["bootstrap_authority_receipt_sha256"] != authority["receipt_sha256"]
        or raw["self_disable_receipt_sha256"] != self_disable["receipt_sha256"]
        or raw["temporary_admin_delete_required"] is not True
        or raw["preterminal"] is not True
        or raw["safe_to_start"] is not False
        or raw["secret_material_recorded"] is not False
    ):
        raise PhaseBError("phase_b_predelete_receipt_invalid")
    _require_same_credential(plan.preflight.value["credential"], observation.value["credential"])
    _require_secret_free(raw)
    return raw


def _require_same_credential(initial: Any, final: Any) -> None:
    expected = _validate_credential(initial)
    observed = dict(final)
    observed.pop("stage_path", None)
    normalized = _validate_credential(observed)
    if normalized != expected:
        raise PhaseBError("phase_b_writer_credential_changed")


_ABSENCE_FIELDS = frozenset(
    {
        "schema",
        "temporary_admin_absent",
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
    }
)


def _validate_admin_absence_receipt(
    value: Any,
    *,
    plan: PhaseBPlan,
    fresh_authority: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_ABSENCE_FIELDS,
        digest_field="evidence_sha256",
        code="phase_b_admin_absence_receipt_invalid",
    )
    if (
        raw["schema"] != TEMPORARY_ADMIN_ABSENCE_RECEIPT_SCHEMA
        or raw["temporary_admin_absent"] is not True
        or raw["project"] != PROJECT
        or raw["instance"] != SQL_INSTANCE
        or raw["username_sha256"]
        != _sha256_bytes(plan.temporary_admin_username.encode("ascii"))
        or raw["owner_subject_sha256"] != plan.owner_subject_sha256
        or raw["mutation_context_sha256"] != plan.sha256
        or raw["user_absent"] is not True
        or not isinstance(raw["baseline_operation_names"], list)
        or raw["baseline_operation_names"]
        != sorted(set(raw["baseline_operation_names"]))
        or not isinstance(raw["baseline_user_operations"], list)
        or not isinstance(raw["known_operation_names"], list)
        or raw["known_operation_names"] != sorted(set(raw["known_operation_names"]))
        or not isinstance(raw["response_known_authority_operation_names"], list)
        or not isinstance(raw["response_known_delete_operation_names"], list)
        or not raw["response_known_delete_operation_names"]
        or not isinstance(raw["post_baseline_authority_operations"], list)
        or raw["post_baseline_authority_operation_count"]
        != len(raw["post_baseline_authority_operations"])
        or raw["post_baseline_authority_operation_count"] != 1
        or raw["response_known_candidate_observed"] is not True
        or not isinstance(raw["terminal_user_operations"], list)
        or type(raw["mutation_ambiguity_observed"]) is not bool
        or not isinstance(raw["quiet_window_seconds"], (int, float))
        or isinstance(raw["quiet_window_seconds"], bool)
        or raw["quiet_window_seconds"] <= 0
    ):
        raise PhaseBError("phase_b_admin_absence_receipt_invalid")
    baseline_rows = [
        _validate_operation_row(row) for row in raw["baseline_user_operations"]
    ]
    if len({row[0] for row in baseline_rows}) != len(baseline_rows):
        raise PhaseBError("phase_b_admin_absence_receipt_invalid")
    authority_rows = [
        _validate_operation_row(
            row,
            owner_subject_sha256=plan.owner_subject_sha256,
            expected_types=frozenset({"CREATE_USER", "UPDATE_USER"}),
        )
        for row in raw["post_baseline_authority_operations"]
    ]
    terminal_rows = [_validate_operation_row(row) for row in raw["terminal_user_operations"]]
    if len({row[0] for row in terminal_rows}) != len(terminal_rows):
        raise PhaseBError("phase_b_admin_absence_receipt_invalid")
    terminal_by_name = {row[0]: row for row in terminal_rows}
    authority_operation = list(fresh_authority["authority_operation"])
    if (
        any(row[2] != "DONE" or row[4] is not True for row in authority_rows)
        or authority_rows != [authority_operation]
        or set(raw["response_known_authority_operation_names"])
        != {row[0] for row in authority_rows}
    ):
        raise PhaseBError("phase_b_admin_absence_receipt_invalid")
    for name in raw["response_known_delete_operation_names"]:
        row = terminal_by_name.get(name)
        if (
            row is None
            or row[1] != "DELETE_USER"
            or row[2] != "DONE"
            or row[3] != plan.owner_subject_sha256
            or row[4] is not True
        ):
            raise PhaseBError("phase_b_admin_absence_receipt_invalid")
    baseline_by_name = {row[0]: row for row in baseline_rows}
    if any(terminal_by_name.get(name) != row for name, row in baseline_by_name.items()):
        raise PhaseBError("phase_b_admin_absence_receipt_invalid")
    response_names = set(raw["response_known_authority_operation_names"]) | set(
        raw["response_known_delete_operation_names"]
    )
    if set(raw["known_operation_names"]) != response_names:
        raise PhaseBError("phase_b_admin_absence_receipt_invalid")
    post_baseline = {
        name: row for name, row in terminal_by_name.items()
        if name not in set(raw["baseline_operation_names"])
    }
    if set(post_baseline) != response_names:
        raise PhaseBError("phase_b_admin_absence_receipt_invalid")
    if any(
        row[2] != "DONE"
        or row[3] != plan.owner_subject_sha256
        or row[4] is not True
        or row[1] not in {"CREATE_USER", "UPDATE_USER", "DELETE_USER"}
        for row in post_baseline.values()
    ):
        raise PhaseBError("phase_b_admin_absence_receipt_invalid")
    _require_secret_free(raw)
    return raw


_RECOVERY_FIELDS = frozenset(
    {
        "schema",
        "plan_sha256",
        "database",
        "database_preflight",
        "credential",
        "services",
        "cloud_sql",
        "observed_at_unix",
        "observation_sha256",
    }
)


def _validate_recovery_cloud(value: Any, *, plan: PhaseBPlan) -> None:
    raw = _strict_mapping(
        value,
        frozenset(
            {
                "project",
                "instance",
                "bootstrap_resource",
                "temporary_admin_users",
                "user_inventory_sha256",
                "user_operations_quiescent",
                "operation_ledger_sha256",
            }
        ),
        "phase_b_recovery_cloud_invalid",
    )
    if (
        raw["project"] != PROJECT
        or raw["instance"] != SQL_INSTANCE
        or raw["temporary_admin_users"]
        not in ([], [plan.temporary_admin_username])
        or raw["user_operations_quiescent"] is not True
    ):
        raise PhaseBError("phase_b_recovery_cloud_invalid")
    _digest(raw["user_inventory_sha256"], "phase_b_recovery_cloud_invalid")
    _digest(raw["operation_ledger_sha256"], "phase_b_recovery_cloud_invalid")
    if raw["bootstrap_resource"] is not None:
        _validate_bootstrap_resource(raw["bootstrap_resource"])


def _validate_recovery_database_preflight(
    value: Any,
    *,
    plan: PhaseBPlan,
) -> Mapping[str, Any]:
    raw = _validate_database_receipt_envelope(
        value, code="phase_b_recovery_database_preflight_invalid"
    )
    if (
        raw["schema"]
        != "muncho-canonical-writer-foundation-phase-b-db-preflight.v1"
        or raw["preflight"] is not True
        or raw["terminal"] is not False
        or raw["database"] != SQL_DATABASE
        or raw["database_owner"] != DATABASE_OWNER_ROLE
        or type(raw["postgres_version_num"]) is not int
        or raw["postgres_version_num"] // 10000 != 18
        or raw["session_user"] != SQL_USER
        or raw["current_user"] != SQL_USER
        or raw["secret_material_recorded"] is not False
        or not isinstance(raw["roles"], list)
        or not isinstance(raw["memberships"], list)
        or not isinstance(raw["temporary_admin_roles"], list)
    ):
        raise PhaseBError("phase_b_recovery_database_preflight_invalid")
    initial = plan.preflight.value["foundation"]
    initial_roles = {row["name"]: row for row in initial["roles"]}
    roles = [_validate_role_row(row) for row in raw["roles"]]
    role_by_name = {row["name"]: row for row in roles}
    if (
        len(role_by_name) != len(roles)
        or roles != sorted(roles, key=lambda row: row["name"])
        or len({row["oid"] for row in roles}) != len(roles)
    ):
        raise PhaseBError("phase_b_recovery_roles_invalid")
    for name in (MIGRATION_OWNER_ROLE, WRITER_ROLE, SQL_USER):
        if role_by_name.get(name) != initial_roles[name]:
            raise PhaseBError("phase_b_recovery_writer_drifted")
    role_present = CANARY_BOOTSTRAP_ROLE in role_by_name
    login_present = CANARY_BOOTSTRAP_LOGIN in role_by_name
    if login_present and not role_present:
        raise PhaseBError("phase_b_recovery_roles_invalid")
    if raw["bootstrap_role_absent"] is not (not role_present) or raw[
        "bootstrap_login_absent"
    ] is not (not login_present):
        raise PhaseBError("phase_b_recovery_roles_invalid")
    expected_names = {MIGRATION_OWNER_ROLE, WRITER_ROLE, SQL_USER}
    if role_present:
        expected_names.add(CANARY_BOOTSTRAP_ROLE)
        role = role_by_name[CANARY_BOOTSTRAP_ROLE]
        if (
            role["can_login"] is not False
            or role["inherits"] is not False
            or any(role[field] is not False for field in (
                "superuser", "create_database", "create_role", "replication", "bypass_row_security"
            ))
            or role["connection_limit"] != -1
            or role["validity_is_unbounded"] is not True
            or role["configuration_is_empty"] is not True
        ):
            raise PhaseBError("phase_b_recovery_roles_invalid")
    if login_present:
        expected_names.add(CANARY_BOOTSTRAP_LOGIN)
        role = role_by_name[CANARY_BOOTSTRAP_LOGIN]
        if (
            role["can_login"] is not True
            or role["inherits"] is not True
            or any(role[field] is not False for field in (
                "superuser", "create_database", "create_role", "replication", "bypass_row_security"
            ))
            or role["connection_limit"] != -1
            or role["validity_is_unbounded"] is not True
            or role["configuration_is_empty"] is not True
        ):
            raise PhaseBError("phase_b_recovery_roles_invalid")
    if set(role_by_name) != expected_names:
        raise PhaseBError("phase_b_recovery_roles_invalid")
    temp_roles = [_validate_role_row(row) for row in raw["temporary_admin_roles"]]
    if (
        len(temp_roles) > 1
        or temp_roles != sorted(temp_roles, key=lambda row: row["name"])
        or len({row["oid"] for row in [*roles, *temp_roles]})
        != len(roles) + len(temp_roles)
    ):
        raise PhaseBError("phase_b_recovery_temporary_admin_invalid")
    if temp_roles:
        temp = temp_roles[0]
        if (
            temp["name"] != plan.temporary_admin_username
            or temp["can_login"] is not True
            or temp["inherits"] is not True
            or temp["superuser"] is not False
            or temp["create_database"] is not True
            or temp["create_role"] is not True
            or temp["replication"] is not False
            or temp["bypass_row_security"] is not False
            or temp["connection_limit"] != -1
            or temp["validity_is_unbounded"] is not True
            or temp["configuration_is_empty"] is not True
        ):
            raise PhaseBError("phase_b_recovery_temporary_admin_invalid")
    expected_memberships = [
        {
            "granted_role": WRITER_ROLE,
            "member_role": SQL_USER,
            "grantor": PERSISTENT_MEMBERSHIP_GRANTOR,
            "admin_option": False,
            "inherit_option": True,
            "set_option": False,
        }
    ]
    if login_present:
        expected_memberships.append(
            {
                "granted_role": CANARY_BOOTSTRAP_ROLE,
                "member_role": CANARY_BOOTSTRAP_LOGIN,
                "grantor": PERSISTENT_MEMBERSHIP_GRANTOR,
                "admin_option": False,
                "inherit_option": True,
                "set_option": False,
            }
        )
    optional_auto_bridge: Mapping[str, Any] | None = None
    if temp_roles:
        expected_memberships.append(
            {
                "granted_role": DATABASE_OWNER_ROLE,
                "member_role": plan.temporary_admin_username,
                "grantor": PERSISTENT_MEMBERSHIP_GRANTOR,
                "admin_option": False,
                "inherit_option": True,
                "set_option": True,
            }
        )
        if role_present:
            optional_auto_bridge = {
                "granted_role": CANARY_BOOTSTRAP_ROLE,
                "member_role": plan.temporary_admin_username,
                "grantor": PERSISTENT_MEMBERSHIP_GRANTOR,
                "admin_option": True,
                "inherit_option": False,
                "set_option": False,
            }
    memberships = [
        _strict_mapping(row, _MEMBERSHIP_FIELDS, "phase_b_recovery_memberships_invalid")
        for row in raw["memberships"]
    ]
    sort_key = lambda row: (row["granted_role"], row["member_role"], row["grantor"])
    if memberships != sorted(memberships, key=sort_key):
        raise PhaseBError("phase_b_recovery_memberships_invalid")
    allowed_memberships = [expected_memberships]
    if optional_auto_bridge is not None:
        allowed_memberships.append([*expected_memberships, optional_auto_bridge])
    if sorted(memberships, key=sort_key) not in [
        sorted(candidate, key=sort_key) for candidate in allowed_memberships
    ]:
        raise PhaseBError("phase_b_recovery_memberships_invalid")
    if (
        raw["namespaces"] != initial["namespaces"]
        or raw["event_log"] != initial["event_log"]
        or raw["writer_ping"] != initial["writer_ping"]
        or raw["legacy_archive"] != initial["legacy_archive"]
    ):
        raise PhaseBError("phase_b_recovery_canonical_truth_drifted")
    target = _validate_database_row(raw["target_database"], target=True)
    initial_target = initial["target_database"]
    for field in _DATABASE_ROW_FIELDS - {"acl", "acl_is_null"}:
        if field.startswith("effective_public_"):
            continue
        if target[field] != initial_target[field]:
            raise PhaseBError("phase_b_recovery_target_database_drifted")
    bootstrap_acl = [
        row
        for row in target["acl"]
        if row["grantee"] == CANARY_BOOTSTRAP_ROLE
    ]
    if role_present:
        if (
            len(bootstrap_acl) != 1
            or bootstrap_acl[0]["grantor"] != DATABASE_OWNER_ROLE
            or bootstrap_acl[0]["grantor_oid"] != target["owner_oid"]
            or bootstrap_acl[0]["grantee_oid"]
            != role_by_name[CANARY_BOOTSTRAP_ROLE]["oid"]
            or bootstrap_acl[0]["privilege"] != "CONNECT"
            or bootstrap_acl[0]["grantable"] is not False
        ):
            raise PhaseBError("phase_b_recovery_bootstrap_connect_invalid")
    elif bootstrap_acl:
        raise PhaseBError("phase_b_recovery_bootstrap_connect_invalid")
    if raw["other_connectable_databases"] != initial["other_connectable_databases"] or raw[
        "managed_cloudsqladmin"
    ] != initial["managed_cloudsqladmin"]:
        raise PhaseBError("phase_b_recovery_cross_database_drifted")
    _require_secret_free(raw)
    return {
        "bootstrap_role_present": role_present,
        "bootstrap_login_present": login_present,
        "temporary_admin_users": [row["name"] for row in temp_roles],
        "temporary_auto_membership_present": (
            optional_auto_bridge is not None and optional_auto_bridge in memberships
        ),
    }


@dataclass(frozen=True)
class PhaseBRecoveryObservation:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        plan: PhaseBPlan,
    ) -> "PhaseBRecoveryObservation":
        raw = _hashed_mapping(
            value,
            fields=_RECOVERY_FIELDS,
            digest_field="observation_sha256",
            code="phase_b_recovery_observation_invalid",
        )
        _validate_database(raw["database"], writer=True)
        _validate_credential(raw["credential"])
        _validate_services(raw["services"], release_revision=plan.revision)
        if (
            raw["schema"] != PHASE_B_RECOVERY_SCHEMA
            or raw["plan_sha256"] != plan.sha256
            or type(raw["observed_at_unix"]) is not int
            or raw["observed_at_unix"] <= 0
        ):
            raise PhaseBError("phase_b_recovery_observation_invalid")
        _require_same_credential(plan.preflight.value["credential"], raw["credential"])
        database_state = _validate_recovery_database_preflight(
            raw["database_preflight"],
            plan=plan,
        )
        _validate_recovery_cloud(raw["cloud_sql"], plan=plan)
        cloud = raw["cloud_sql"]
        if (
            database_state["bootstrap_login_present"]
            is not (cloud["bootstrap_resource"] is not None)
            or database_state["temporary_admin_users"]
            != cloud["temporary_admin_users"]
        ):
            raise PhaseBError("phase_b_recovery_cloud_database_mismatch")
        _require_secret_free(raw)
        return cls(_json_copy(raw))


_TERMINAL_OBSERVATION_FIELDS = frozenset(
    {
        "schema",
        "plan_sha256",
        "foundation_observation",
        "database_preflight",
        "session_identity_sha256",
        "writer_ping_identity_sha256",
        "event_log_identity_sha256",
        "legacy_archive_identity_sha256",
        "cross_database_acl_sha256",
        "bootstrap_connect_acl",
        "temporary_admin_references",
        "cloud_sql",
        "services",
        "observed_at_unix",
        "observation_sha256",
    }
)


def _validate_terminal_observation(
    value: Any,
    *,
    plan: PhaseBPlan,
    execution_preflight_session_sha256: str | None,
) -> dict[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_TERMINAL_OBSERVATION_FIELDS,
        digest_field="observation_sha256",
        code="phase_b_terminal_observation_invalid",
    )
    observation = FoundationObservation.from_mapping(raw["foundation_observation"])
    terminal_database_state = _validate_recovery_database_preflight(
        raw["database_preflight"], plan=plan
    )
    initial_foundation = plan.preflight.value["foundation"]
    initial_projection = _database_preflight_projection(initial_foundation)
    current_projection = _database_preflight_projection(raw["database_preflight"])
    initial_database = plan.preflight.value["database"]
    if (
        raw["schema"] != PHASE_B_TERMINAL_OBSERVATION_SCHEMA
        or raw["plan_sha256"] != plan.sha256
        or observation.value["session_user"] != SQL_USER
        or observation.value["temporary_admin_roles"] != []
        or not observation.membership_ready
        or terminal_database_state
        != {
            "bootstrap_role_present": True,
            "bootstrap_login_present": True,
            "temporary_admin_users": [],
            "temporary_auto_membership_present": False,
        }
        or (
            execution_preflight_session_sha256 is not None
            and raw["session_identity_sha256"]
            == execution_preflight_session_sha256
        )
        or raw["session_identity_sha256"] == initial_database["session_identity_sha256"]
        or raw["writer_ping_identity_sha256"]
        != current_projection["writer_ping_identity_sha256"]
        or current_projection["writer_ping_identity_sha256"]
        != initial_projection["writer_ping_identity_sha256"]
        or raw["event_log_identity_sha256"]
        != current_projection["event_log_identity_sha256"]
        or current_projection["event_log_identity_sha256"]
        != initial_projection["event_log_identity_sha256"]
        or raw["legacy_archive_identity_sha256"]
        != current_projection["legacy_archive_identity_sha256"]
        or current_projection["legacy_archive_identity_sha256"]
        != initial_projection["legacy_archive_identity_sha256"]
        or raw["cross_database_acl_sha256"]
        != current_projection["cross_database_acl_sha256"]
        or raw["bootstrap_connect_acl"]
        != {
            "database": SQL_DATABASE,
            "grantee": CANARY_BOOTSTRAP_ROLE,
            "grantor": DATABASE_OWNER_ROLE,
            "privilege": "CONNECT",
            "grantable": False,
        }
        or raw["temporary_admin_references"] != []
        or type(raw["observed_at_unix"]) is not int
        or raw["observed_at_unix"] <= 0
    ):
        raise PhaseBError("phase_b_terminal_observation_invalid")
    _digest(raw["session_identity_sha256"], "phase_b_terminal_observation_invalid")
    _require_same_credential(plan.preflight.value["credential"], observation.value["credential"])
    _validate_services(raw["services"], release_revision=plan.revision)
    cloud = _strict_mapping(
        raw["cloud_sql"],
        frozenset(
            {
                "project",
                "instance",
                "bootstrap_resource",
                "temporary_admin_absent",
                "temporary_admin_username_sha256",
                "user_operations_quiescent",
                "operation_ledger_sha256",
            }
        ),
        "phase_b_terminal_cloud_invalid",
    )
    if (
        cloud["project"] != PROJECT
        or cloud["instance"] != SQL_INSTANCE
        or cloud["temporary_admin_absent"] is not True
        or cloud["temporary_admin_username_sha256"]
        != _sha256_bytes(plan.temporary_admin_username.encode("ascii"))
        or cloud["user_operations_quiescent"] is not True
    ):
        raise PhaseBError("phase_b_terminal_cloud_invalid")
    _digest(cloud["operation_ledger_sha256"], "phase_b_terminal_cloud_invalid")
    _validate_bootstrap_resource(cloud["bootstrap_resource"])
    if terminal_database_state["bootstrap_login_present"] is not True:
        raise PhaseBError("phase_b_terminal_cloud_database_mismatch")
    _require_secret_free(raw)
    return raw


def _validate_bootstrap_resource(value: Any) -> dict[str, Any]:
    raw = _strict_mapping(
        value,
        frozenset(
            {"databaseRoles", "etag", "host", "instance", "name", "project", "type"}
        ),
        "phase_b_bootstrap_resource_invalid",
    )
    if (
        raw["databaseRoles"] != [CANARY_BOOTSTRAP_ROLE]
        or not isinstance(raw["etag"], str)
        or not 1 <= len(raw["etag"]) <= 1024
        or any(
            ord(character) < 0x21 or ord(character) > 0x7E
            for character in raw["etag"]
        )
        or raw["host"] != ""
        or raw["instance"] != SQL_INSTANCE
        or raw["name"] != CANARY_BOOTSTRAP_LOGIN
        or raw["project"] != PROJECT
        or raw["type"] != "BUILT_IN"
    ):
        raise PhaseBError("phase_b_bootstrap_resource_invalid")
    _require_secret_free(raw)
    return raw


_JOURNAL_FIELDS = frozenset(
    {
        "schema",
        "sequence",
        "plan_sha256",
        "event",
        "idempotency_key",
        "previous_entry_sha256",
        "evidence",
        "preterminal",
        "safe_to_start",
        "recorded_at_unix",
        "entry_sha256",
    }
)


@dataclass(frozen=True)
class PhaseBJournalEntry:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        plan: PhaseBPlan,
        expected_sequence: int,
        expected_previous: str | None,
    ) -> "PhaseBJournalEntry":
        raw = _hashed_mapping(
            value,
            fields=_JOURNAL_FIELDS,
            digest_field="entry_sha256",
            code="phase_b_journal_entry_invalid",
        )
        terminal = raw["event"] == "terminal"
        if (
            raw["schema"] != PHASE_B_JOURNAL_SCHEMA
            or raw["sequence"] != expected_sequence
            or raw["plan_sha256"] != plan.sha256
            or raw["event"] not in JOURNAL_EVENTS
            or not isinstance(raw["idempotency_key"], str)
            or not raw["idempotency_key"]
            or len(raw["idempotency_key"]) > 160
            or raw["previous_entry_sha256"] != expected_previous
            or not isinstance(raw["evidence"], Mapping)
            or raw["preterminal"] is terminal
            or raw["safe_to_start"] is not terminal
            or type(raw["recorded_at_unix"]) is not int
            or raw["recorded_at_unix"] <= 0
        ):
            raise PhaseBError("phase_b_journal_entry_invalid")
        _require_secret_free(raw)
        return cls(_json_copy(raw))

    @property
    def event(self) -> str:
        return str(self.value["event"])

    @property
    def sha256(self) -> str:
        return str(self.value["entry_sha256"])

    @property
    def evidence(self) -> Mapping[str, Any]:
        return self.value["evidence"]


class AppendOnlyPhaseBJournal:
    """Plan-addressed, fsynced append-only receipt chain."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path) or not root.is_absolute():
            raise PhaseBError("phase_b_journal_root_invalid")
        self.root = root
        self._lock_depth = 0

    def _plan_root(self, plan: PhaseBPlan) -> Path:
        return self.root / plan.sha256

    def _entries_root(self, plan: PhaseBPlan) -> Path:
        return self._plan_root(plan) / "entries"

    def _staging_root(self, plan: PhaseBPlan) -> Path:
        return self._plan_root(plan) / "staging"

    def _lock_path(self, plan: PhaseBPlan) -> Path:
        return self._plan_root(plan) / ".lock"

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        created = False
        try:
            path.mkdir(mode=_JOURNAL_MODE, parents=True, exist_ok=False)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise PhaseBError("phase_b_journal_unavailable") from exc
        try:
            status = path.lstat()
            if (
                created
                and status.st_uid == _effective_uid()
                and status.st_gid != _effective_gid()
            ):
                os.chown(path, -1, _effective_gid(), follow_symlinks=False)
                status = path.lstat()
        except OSError as exc:
            raise PhaseBError("phase_b_journal_unavailable") from exc
        if (
            stat.S_ISLNK(status.st_mode)
            or not stat.S_ISDIR(status.st_mode)
            or stat.S_IMODE(status.st_mode) != _JOURNAL_MODE
            or status.st_uid != _effective_uid()
            or status.st_gid != _effective_gid()
        ):
            raise PhaseBError("phase_b_journal_directory_untrusted")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @contextmanager
    def lock(self, plan: PhaseBPlan) -> Iterator[None]:
        if fcntl is None:
            raise PhaseBError("phase_b_posix_lock_unavailable")
        self._ensure_directory(self.root)
        self._ensure_directory(self._plan_root(plan))
        path = self._lock_path(plan)
        lock_existed = False
        try:
            path.lstat()
            lock_existed = True
        except FileNotFoundError:
            pass
        try:
            descriptor = os.open(
                path,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                _JOURNAL_FILE_MODE,
            )
        except OSError as exc:
            raise PhaseBError("phase_b_journal_lock_untrusted") from exc
        try:
            status = os.fstat(descriptor)
            if (
                not lock_existed
                and status.st_uid == _effective_uid()
                and status.st_gid != _effective_gid()
            ):
                os.fchown(descriptor, -1, _effective_gid())
                status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(status.st_mode)
                or stat.S_IMODE(status.st_mode) != _JOURNAL_FILE_MODE
                or status.st_nlink != 1
                or status.st_uid != _effective_uid()
                or status.st_gid != _effective_gid()
            ):
                raise PhaseBError("phase_b_journal_lock_untrusted")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._lock_depth += 1
            yield
        finally:
            try:
                self._lock_depth = max(0, self._lock_depth - 1)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def load(self, plan: PhaseBPlan) -> list[PhaseBJournalEntry]:
        if self.root.exists():
            self._ensure_directory(self.root)
        if self._plan_root(plan).exists():
            self._ensure_directory(self._plan_root(plan))
        entries_root = self._entries_root(plan)
        if not entries_root.exists():
            return []
        self._ensure_directory(entries_root)
        paths = sorted(entries_root.iterdir(), key=lambda path: path.name)
        if any(re.fullmatch(r"[0-9]{8}\.json", path.name) is None for path in paths):
            raise PhaseBError("phase_b_journal_path_invalid")
        entries: list[PhaseBJournalEntry] = []
        previous: str | None = None
        seen_keys: dict[str, PhaseBJournalEntry] = {}
        for sequence, path in enumerate(paths):
            if path.name != f"{sequence:08d}.json":
                raise PhaseBError("phase_b_journal_sequence_invalid")
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
                    chunks = bytearray()
                    while len(chunks) <= _MAX_JSON_BYTES:
                        chunk = os.read(
                            descriptor,
                            min(1024 * 1024, _MAX_JSON_BYTES + 1 - len(chunks)),
                        )
                        if not chunk:
                            break
                        chunks.extend(chunk)
                    raw = bytes(chunks)
                    after = os.fstat(descriptor)
                finally:
                    os.close(descriptor)
            except OSError as exc:
                raise PhaseBError("phase_b_journal_read_failed") from exc
            if (
                (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_uid,
                    before.st_gid,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                    before.st_nlink,
                )
                != (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_mode,
                    opened.st_uid,
                    opened.st_gid,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                    opened.st_nlink,
                )
                or (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_mode,
                    opened.st_uid,
                    opened.st_gid,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                    opened.st_nlink,
                )
                != (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_uid,
                    after.st_gid,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                    after.st_nlink,
                )
                or stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != _JOURNAL_FILE_MODE
                or before.st_nlink != 1
                or before.st_uid != _effective_uid()
                or before.st_gid != _effective_gid()
                or not raw
                or len(raw) > _MAX_JSON_BYTES
            ):
                raise PhaseBError("phase_b_journal_file_untrusted")
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise PhaseBError("phase_b_journal_json_invalid") from exc
            if _canonical_bytes(decoded) != raw:
                raise PhaseBError("phase_b_journal_not_canonical")
            entry = PhaseBJournalEntry.from_mapping(
                decoded,
                plan=plan,
                expected_sequence=sequence,
                expected_previous=previous,
            )
            key = str(entry.value["idempotency_key"])
            if key in seen_keys:
                raise PhaseBError("phase_b_journal_idempotency_key_reused")
            if entries and entries[-1].event == "terminal":
                raise PhaseBError("phase_b_journal_after_terminal")
            seen_keys[key] = entry
            entries.append(entry)
            previous = entry.sha256
        self._validate_event_prerequisites(entries)
        return entries

    @staticmethod
    def _validate_event_prerequisites(entries: Sequence[PhaseBJournalEntry]) -> None:
        seen: set[str] = set()
        for index, entry in enumerate(entries):
            event = entry.event
            if index == 0 and event != "intent":
                raise PhaseBError("phase_b_journal_intent_missing")
            if index > 0 and event == "intent":
                raise PhaseBError("phase_b_journal_intent_repeated")
            required = {
                "role_ready": {"temporary_admin_authority"},
                "bootstrap_authority": {"role_ready"},
                "bootstrap_hba_rejected": {"bootstrap_authority"},
                "bootstrap_password_disabled": {"bootstrap_hba_rejected"},
                "predelete_verified": {"role_ready", "bootstrap_password_disabled"},
                "temporary_admin_closed": {"predelete_verified"},
                "temporary_admin_predelete_authority": {"temporary_admin_closed"},
                "temporary_admin_absent": {
                    "temporary_admin_closed",
                    "temporary_admin_predelete_authority",
                },
                "terminal_observed": {"temporary_admin_absent"},
                "terminal": {"terminal_observed"},
            }.get(event, set())
            if not required.issubset(seen):
                raise PhaseBError("phase_b_journal_prerequisite_missing")
            seen.add(event)

    def append(
        self,
        plan: PhaseBPlan,
        *,
        event: str,
        idempotency_key: str,
        evidence: Mapping[str, Any],
        now_unix: int | None = None,
    ) -> PhaseBJournalEntry:
        if event not in JOURNAL_EVENTS:
            raise PhaseBError("phase_b_journal_event_invalid")
        if self._lock_depth != 1:
            raise PhaseBError("phase_b_journal_lock_required")
        _require_secret_free(evidence)
        self._ensure_directory(self._entries_root(plan))
        self._ensure_directory(self._staging_root(plan))
        entries = self.load(plan)
        for existing in entries:
            if existing.value["idempotency_key"] == idempotency_key:
                if existing.event != event or existing.evidence != evidence:
                    raise PhaseBError("phase_b_journal_idempotency_conflict")
                return existing
        terminal = event == "terminal"
        unsigned = {
            "schema": PHASE_B_JOURNAL_SCHEMA,
            "sequence": len(entries),
            "plan_sha256": plan.sha256,
            "event": event,
            "idempotency_key": idempotency_key,
            "previous_entry_sha256": entries[-1].sha256 if entries else None,
            "evidence": _json_copy(evidence),
            "preterminal": not terminal,
            "safe_to_start": terminal,
            "recorded_at_unix": int(time.time()) if now_unix is None else now_unix,
        }
        entry = PhaseBJournalEntry.from_mapping(
            {**unsigned, "entry_sha256": _sha256_json(unsigned)},
            plan=plan,
            expected_sequence=len(entries),
            expected_previous=entries[-1].sha256 if entries else None,
        )
        self._validate_event_prerequisites([*entries, entry])
        payload = _canonical_bytes(entry.value)
        stage = self._staging_root(plan) / (
            f"{len(entries):08d}.{secrets.token_hex(16)}.stage"
        )
        final = self._entries_root(plan) / f"{len(entries):08d}.json"
        descriptor = os.open(
            stage,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            _JOURNAL_FILE_MODE,
        )
        try:
            status = os.fstat(descriptor)
            if status.st_uid == _effective_uid() and status.st_gid != _effective_gid():
                os.fchown(descriptor, -1, _effective_gid())
            written = 0
            while written < len(payload):
                count = os.write(descriptor, payload[written:])
                if count <= 0:
                    raise PhaseBError("phase_b_journal_write_failed")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(stage, final)
            self._fsync_directory(self._entries_root(plan))
        except FileExistsError as exc:
            raise PhaseBError("phase_b_journal_append_collision") from exc
        finally:
            try:
                stage.unlink()
            except FileNotFoundError:
                pass
            self._fsync_directory(self._staging_root(plan))
        loaded = self.load(plan)
        if not loaded or loaded[-1].sha256 != entry.sha256:
            raise PhaseBError("phase_b_journal_readback_failed")
        return loaded[-1]

    def events(self, plan: PhaseBPlan) -> frozenset[str]:
        return frozenset(entry.event for entry in self.load(plan))

    def last_evidence(self, plan: PhaseBPlan, event: str) -> Mapping[str, Any] | None:
        matching = [entry.evidence for entry in self.load(plan) if entry.event == event]
        return None if not matching else matching[-1]


class ClosableSession(Protocol):
    username: str

    def close(self) -> None: ...


class RoleArtifactSession(ClosableSession, Protocol):
    def execute_phase_b_role_artifact(
        self,
        artifact: SealedSQLArtifact,
        *,
        bindings: Mapping[str, str],
    ) -> Mapping[str, Any]: ...


class TemporaryAdminBoundary(Protocol):
    def begin_mutation_observation(
        self,
        *,
        expected_owner_subject_sha256: str,
        expected_mutation_context_sha256: str,
    ) -> None: ...

    def create_or_rotate_recovery(self, username: str, password: str) -> None: ...

    def mutation_reconciliation_required(self) -> bool: ...

    def require_current_authority(self, username: str) -> None: ...

    def temporary_admin_authority_receipt(self, username: str) -> Mapping[str, Any]: ...

    def delete_and_confirm_absent(self, username: str) -> None: ...

    def reconciliation_receipt(self) -> Mapping[str, Any]: ...


class BootstrapLoginBoundary(Protocol):
    def describe(self) -> Mapping[str, Any] | None: ...

    def create_or_rotate_recovery(self, provisional_password: str) -> None: ...

    def mutation_reconciliation_required(self) -> bool: ...

    def require_current_authority(self) -> None: ...

    def authority_receipt(self) -> Mapping[str, Any]: ...


class BootstrapSelfDisableBoundary(Protocol):
    def disable_and_prove_denied(
        self,
        *,
        plan: PhaseBPlan,
        provisional_password: bytearray,
        authority_receipt: Mapping[str, Any],
        hba_rejection_receipt: Mapping[str, Any],
        statement: str,
    ) -> Mapping[str, Any]: ...


WriterSessionFactory = Callable[[], ClosableSession]
PristinePreflightCollector = Callable[[ClosableSession], Mapping[str, Any]]
RecoveryCollector = Callable[
    [ClosableSession, PhaseBPlan, frozenset[str]], Mapping[str, Any]
]
TemporaryAdminFactory = Callable[[PhaseBPlan], TemporaryAdminBoundary]
BootstrapLoginFactory = Callable[[PhaseBPlan], BootstrapLoginBoundary]
AdminSessionFactory = Callable[
    [PhaseBPlan, str, bytearray], RoleArtifactSession
]
HBACollector = Callable[
    [PhaseBPlan, bytearray, Mapping[str, Any]], Mapping[str, Any]
]
PredeleteCollector = Callable[
    [RoleArtifactSession, PhaseBPlan, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any],
]
TerminalCollector = Callable[
    [ClosableSession, PhaseBPlan, Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any],
]
ServicesCollector = Callable[[PhaseBPlan, str], Mapping[str, Any]]
SecretFactory = Callable[[], bytearray]


@dataclass(frozen=True)
class PhaseBDependencies:
    writer_session_factory: WriterSessionFactory
    pristine_preflight_collector: PristinePreflightCollector
    recovery_collector: RecoveryCollector
    temporary_admin_factory: TemporaryAdminFactory
    bootstrap_login_factory: BootstrapLoginFactory
    admin_session_factory: AdminSessionFactory
    bootstrap_self_disable: BootstrapSelfDisableBoundary
    hba_collector: HBACollector
    predelete_collector: PredeleteCollector
    terminal_collector: TerminalCollector
    services_collector: ServicesCollector
    secret_factory: SecretFactory


def _default_secret_factory() -> bytearray:
    import base64

    return bytearray(base64.urlsafe_b64encode(secrets.token_bytes(48)))


def _require_dependencies(value: Any) -> PhaseBDependencies:
    if not isinstance(value, PhaseBDependencies):
        raise PhaseBError("phase_b_adapters_missing")
    for field in (
        "writer_session_factory",
        "pristine_preflight_collector",
        "recovery_collector",
        "temporary_admin_factory",
        "bootstrap_login_factory",
        "admin_session_factory",
        "hba_collector",
        "predelete_collector",
        "terminal_collector",
        "services_collector",
        "secret_factory",
    ):
        if not callable(getattr(value, field, None)):
            raise PhaseBError("phase_b_adapters_missing")
    if not callable(
        getattr(value.bootstrap_self_disable, "disable_and_prove_denied", None)
    ):
        raise PhaseBError("phase_b_adapters_missing")
    return value


def _validate_role_artifact(
    artifact: Any,
    *,
    plan: PhaseBPlan,
) -> SealedSQLArtifact:
    if (
        not isinstance(artifact, SealedSQLArtifact)
        or artifact.name != ROLE_ARTIFACT_NAME
        or artifact.path.name != Path(ROLE_ARTIFACT_PATH).name
        or artifact.sha256 != plan.value["role_artifact_sha256"]
        or not artifact.payload
        or len(artifact.payload) > _MAX_ARTIFACT_BYTES
        or _sha256_bytes(artifact.payload) != artifact.sha256
        or b"\x00" in artifact.payload
    ):
        raise PhaseBError("phase_b_role_artifact_invalid")
    try:
        artifact.payload.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise PhaseBError("phase_b_role_artifact_invalid") from exc
    return artifact


def _service_stable_projection(value: Mapping[str, Any]) -> Mapping[str, Any]:
    result = _json_copy(value)
    result.pop("observed_at_unix")
    result.pop("attestation_sha256")
    return result


def _recheck_services(
    plan: PhaseBPlan,
    dependencies: PhaseBDependencies,
    journal: AppendOnlyPhaseBJournal,
    *,
    transition: str,
    clock: Callable[[], float],
) -> Mapping[str, Any]:
    services = _validate_services(
        dependencies.services_collector(plan, transition),
        release_revision=plan.revision,
    )
    initial = _validate_services(
        plan.preflight.value["services"], release_revision=plan.revision
    )
    if _service_stable_projection(services) != _service_stable_projection(initial):
        raise PhaseBError("phase_b_services_changed_after_approval")
    journal.append(
        plan,
        event="services_stopped",
        idempotency_key=f"services:{transition}:{services['attestation_sha256']}",
        evidence={
            "transition": transition,
            "services_attestation": services,
            "preterminal": True,
            "safe_to_start": False,
        },
        now_unix=int(clock()),
    )
    return services


def _revalidate_approval(
    approval: PhaseBApproval,
    plan: PhaseBPlan,
    clock: Callable[[], float],
) -> None:
    PhaseBApproval.from_mapping(
        approval.value,
        plan=plan,
        now_unix=int(clock()),
    )


def _provision_temporary_admin(
    boundary: TemporaryAdminBoundary,
    plan: PhaseBPlan,
    secret: bytearray,
    *,
    before_mutation: Callable[[], None],
) -> Mapping[str, Any]:
    boundary.begin_mutation_observation(
        expected_owner_subject_sha256=plan.owner_subject_sha256,
        expected_mutation_context_sha256=plan.sha256,
    )
    encoded = secret.decode("ascii")
    try:
        before_mutation()
        boundary.create_or_rotate_recovery(plan.temporary_admin_username, encoded)
    except Exception:
        if not boundary.mutation_reconciliation_required():
            raise
        before_mutation()
        boundary.create_or_rotate_recovery(plan.temporary_admin_username, encoded)
    boundary.require_current_authority(plan.temporary_admin_username)
    return _validate_temporary_admin_authority(
        boundary.temporary_admin_authority_receipt(
            plan.temporary_admin_username
        ),
        plan=plan,
    )


def _provision_bootstrap_login(
    boundary: BootstrapLoginBoundary,
    plan: PhaseBPlan,
    secret: bytearray,
    *,
    before_mutation: Callable[[], None],
) -> Mapping[str, Any]:
    encoded = secret.decode("ascii")
    try:
        before_mutation()
        boundary.create_or_rotate_recovery(encoded)
    except Exception:
        if not boundary.mutation_reconciliation_required():
            raise
        before_mutation()
        boundary.create_or_rotate_recovery(encoded)
    boundary.require_current_authority()
    return _validate_bootstrap_authority(boundary.authority_receipt(), plan=plan)


def _safe_close(session: ClosableSession | None) -> None:
    if session is None:
        return
    try:
        session.close()
    except Exception as exc:
        raise PhaseBError("phase_b_session_close_failed") from exc


def _last_required(
    journal: AppendOnlyPhaseBJournal,
    plan: PhaseBPlan,
    event: str,
    code: str,
) -> Mapping[str, Any]:
    value = journal.last_evidence(plan, event)
    if value is None:
        raise PhaseBError(code)
    return value


_TERMINAL_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "ok",
        "state",
        "safe_to_start",
        "release_revision",
        "plan_sha256",
        "approval_sha256",
        "initial_preflight_sha256",
        "stable_preflight_sha256",
        "initial_temporary_admin_authority_receipt_sha256",
        "predelete_temporary_admin_authority_receipt_sha256",
        "role_receipt_sha256",
        "bootstrap_authority_receipt_sha256",
        "post_disable_bootstrap_authority_receipt_sha256",
        "hba_rejection_receipt_sha256",
        "self_disable_receipt_sha256",
        "predelete_receipt_sha256",
        "temporary_admin_absence_receipt",
        "temporary_admin_absence_receipt_sha256",
        "terminal_observation_sha256",
        "services_attestation_sha256",
        "bootstrap_login_retained",
        "bootstrap_login_password_disabled",
        "temporary_admin_absent",
        "legacy_archive_preserved",
        "writer_credential_inode_preserved",
        "secret_material_recorded",
        "terminal_at_unix",
        "receipt_sha256",
    }
)


def _terminal_component_digest(value: Any, field: str) -> str:
    if not isinstance(value, Mapping):
        raise PhaseBError("phase_b_terminal_receipt_invalid")
    return _digest(value.get(field), "phase_b_terminal_receipt_invalid")


def _validate_terminal_receipt(
    value: Any,
    *,
    plan: PhaseBPlan,
    approval: PhaseBApproval,
    initial_admin_authority: Mapping[str, Any],
    deletion_admin_authority: Mapping[str, Any],
    role_receipt: Mapping[str, Any],
    bootstrap_authority: Mapping[str, Any],
    post_disable_authority: Mapping[str, Any],
    hba_receipt: Mapping[str, Any],
    self_disable_receipt: Mapping[str, Any],
    predelete_receipt: Mapping[str, Any],
    absence_receipt: Mapping[str, Any],
    terminal_observation: Mapping[str, Any],
    services: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_TERMINAL_RECEIPT_FIELDS,
        digest_field="receipt_sha256",
        code="phase_b_terminal_receipt_invalid",
    )
    initial_authority_sha256 = _terminal_component_digest(
        initial_admin_authority, "receipt_sha256"
    )
    deletion_authority_sha256 = _terminal_component_digest(
        deletion_admin_authority, "receipt_sha256"
    )
    bootstrap_authority_sha256 = _terminal_component_digest(
        bootstrap_authority, "receipt_sha256"
    )
    post_disable_authority_sha256 = _terminal_component_digest(
        post_disable_authority, "receipt_sha256"
    )
    expected_absence = _json_copy(absence_receipt)
    terminal_at = raw["terminal_at_unix"]
    if (
        raw["schema"] != PHASE_B_TERMINAL_RECEIPT_SCHEMA
        or raw["ok"] is not True
        or raw["state"] != "terminal"
        or raw["safe_to_start"] is not True
        or raw["release_revision"] != plan.revision
        or raw["plan_sha256"] != plan.sha256
        or raw["approval_sha256"] != approval.sha256
        or raw["initial_preflight_sha256"] != plan.preflight.sha256
        or raw["stable_preflight_sha256"] != plan.preflight.stable_sha256
        or raw["initial_temporary_admin_authority_receipt_sha256"]
        != initial_authority_sha256
        or raw["predelete_temporary_admin_authority_receipt_sha256"]
        != deletion_authority_sha256
        or initial_authority_sha256 == deletion_authority_sha256
        or raw["role_receipt_sha256"]
        != _terminal_component_digest(role_receipt, "receipt_sha256")
        or raw["bootstrap_authority_receipt_sha256"]
        != bootstrap_authority_sha256
        or raw["post_disable_bootstrap_authority_receipt_sha256"]
        != post_disable_authority_sha256
        or bootstrap_authority_sha256 != post_disable_authority_sha256
        or raw["hba_rejection_receipt_sha256"]
        != _terminal_component_digest(hba_receipt, "receipt_sha256")
        or raw["self_disable_receipt_sha256"]
        != _terminal_component_digest(self_disable_receipt, "receipt_sha256")
        or raw["predelete_receipt_sha256"]
        != _terminal_component_digest(predelete_receipt, "receipt_sha256")
        or raw["temporary_admin_absence_receipt"] != expected_absence
        or raw["temporary_admin_absence_receipt_sha256"]
        != _terminal_component_digest(absence_receipt, "evidence_sha256")
        or raw["terminal_observation_sha256"]
        != _terminal_component_digest(terminal_observation, "observation_sha256")
        or raw["services_attestation_sha256"]
        != _terminal_component_digest(services, "attestation_sha256")
        or raw["bootstrap_login_retained"] is not True
        or raw["bootstrap_login_password_disabled"] is not True
        or raw["temporary_admin_absent"] is not True
        or raw["legacy_archive_preserved"] is not True
        or raw["writer_credential_inode_preserved"] is not True
        or raw["secret_material_recorded"] is not False
        or type(terminal_at) is not int
        or terminal_at < approval.value["issued_at_unix"]
    ):
        raise PhaseBError("phase_b_terminal_receipt_invalid")
    _require_secret_free(raw)
    return raw


def _last_loaded_evidence(
    entries: Sequence[PhaseBJournalEntry],
    event: str,
    code: str,
) -> Mapping[str, Any]:
    for entry in reversed(entries):
        if entry.event == event:
            return entry.evidence
    raise PhaseBError(code)


def _terminal_replay_bindings(
    entries: Sequence[PhaseBJournalEntry],
    *,
    plan: PhaseBPlan,
) -> dict[str, Mapping[str, Any]]:
    initial_event = _last_loaded_evidence(
        entries,
        "temporary_admin_authority",
        "phase_b_initial_admin_authority_missing",
    )
    initial_admin_authority = _validate_temporary_admin_authority(
        initial_event.get("authority_receipt"), plan=plan
    )
    role_event = _last_loaded_evidence(
        entries, "role_ready", "phase_b_role_receipt_missing"
    )
    role_receipt = _validate_role_receipt(role_event.get("role_receipt"), plan=plan)
    disabled = _last_loaded_evidence(
        entries,
        "bootstrap_password_disabled",
        "phase_b_bootstrap_disable_receipt_missing",
    )
    bootstrap_authority = _validate_bootstrap_authority(
        disabled.get("initial_authority_receipt"), plan=plan
    )
    post_disable_authority = _validate_bootstrap_authority(
        disabled.get("post_disable_authority_receipt"), plan=plan
    )
    if post_disable_authority != bootstrap_authority:
        raise PhaseBError("phase_b_bootstrap_authority_changed_after_self_disable")
    hba_receipt = _validate_hba_receipt(
        disabled.get("hba_receipt"),
        plan=plan,
        authority=bootstrap_authority,
    )
    self_disable_receipt = _validate_self_disable_receipt(
        disabled.get("self_disable_receipt"),
        plan=plan,
        authority=bootstrap_authority,
        hba=hba_receipt,
    )
    predelete_event = _last_loaded_evidence(
        entries, "predelete_verified", "phase_b_predelete_receipt_missing"
    )
    predelete_receipt = _validate_predelete_receipt(
        predelete_event.get("predelete_receipt"),
        plan=plan,
        role=role_receipt,
        authority=post_disable_authority,
        self_disable=self_disable_receipt,
    )
    predelete_authority_event = _last_loaded_evidence(
        entries,
        "temporary_admin_predelete_authority",
        "phase_b_delete_authority_missing",
    )
    deletion_admin_authority = _validate_temporary_admin_authority(
        predelete_authority_event.get("authority_receipt"), plan=plan
    )
    absent_event = _last_loaded_evidence(
        entries,
        "temporary_admin_absent",
        "phase_b_admin_absence_receipt_missing",
    )
    absent_authority = _validate_temporary_admin_authority(
        absent_event.get("fresh_predelete_authority_receipt"), plan=plan
    )
    if absent_authority != deletion_admin_authority:
        raise PhaseBError("phase_b_delete_authority_changed")
    absence_receipt = _validate_admin_absence_receipt(
        absent_event.get("full_absence_receipt"),
        plan=plan,
        fresh_authority=deletion_admin_authority,
    )
    terminal_event = _last_loaded_evidence(
        entries,
        "terminal_observed",
        "phase_b_terminal_observation_missing",
    )
    terminal_observation = _validate_terminal_observation(
        terminal_event.get("terminal_observation"),
        plan=plan,
        execution_preflight_session_sha256=None,
    )
    terminal_service_events = [
        entry.evidence
        for entry in entries
        if entry.event == "services_stopped"
        and entry.evidence.get("transition") == "terminal_observation"
    ]
    if not terminal_service_events:
        raise PhaseBError("phase_b_terminal_services_missing")
    services = _validate_services(
        terminal_service_events[-1].get("services_attestation"),
        release_revision=plan.revision,
    )
    if _service_stable_projection(
        terminal_observation["services"]
    ) != _service_stable_projection(services):
        raise PhaseBError("phase_b_terminal_services_drifted")
    return {
        "initial_admin_authority": initial_admin_authority,
        "deletion_admin_authority": deletion_admin_authority,
        "role_receipt": role_receipt,
        "bootstrap_authority": bootstrap_authority,
        "post_disable_authority": post_disable_authority,
        "hba_receipt": hba_receipt,
        "self_disable_receipt": self_disable_receipt,
        "predelete_receipt": predelete_receipt,
        "absence_receipt": absence_receipt,
        "terminal_observation": terminal_observation,
        "services": services,
    }


def _terminal_receipt(
    *,
    plan: PhaseBPlan,
    approval: PhaseBApproval,
    initial_admin_authority: Mapping[str, Any],
    deletion_admin_authority: Mapping[str, Any],
    role_receipt: Mapping[str, Any],
    bootstrap_authority: Mapping[str, Any],
    post_disable_authority: Mapping[str, Any],
    hba_receipt: Mapping[str, Any],
    self_disable_receipt: Mapping[str, Any],
    predelete_receipt: Mapping[str, Any],
    absence_receipt: Mapping[str, Any],
    terminal_observation: Mapping[str, Any],
    services: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    if initial_admin_authority["receipt_sha256"] == deletion_admin_authority[
        "receipt_sha256"
    ]:
        raise PhaseBError("phase_b_delete_authority_not_fresh")
    unsigned = {
        "schema": PHASE_B_TERMINAL_RECEIPT_SCHEMA,
        "ok": True,
        "state": "terminal",
        "safe_to_start": True,
        "release_revision": plan.revision,
        "plan_sha256": plan.sha256,
        "approval_sha256": approval.sha256,
        "initial_preflight_sha256": plan.preflight.sha256,
        "stable_preflight_sha256": plan.preflight.stable_sha256,
        "initial_temporary_admin_authority_receipt_sha256": (
            initial_admin_authority["receipt_sha256"]
        ),
        "predelete_temporary_admin_authority_receipt_sha256": (
            deletion_admin_authority["receipt_sha256"]
        ),
        "role_receipt_sha256": role_receipt["receipt_sha256"],
        "bootstrap_authority_receipt_sha256": bootstrap_authority["receipt_sha256"],
        "post_disable_bootstrap_authority_receipt_sha256": (
            post_disable_authority["receipt_sha256"]
        ),
        "hba_rejection_receipt_sha256": hba_receipt["receipt_sha256"],
        "self_disable_receipt_sha256": self_disable_receipt["receipt_sha256"],
        "predelete_receipt_sha256": predelete_receipt["receipt_sha256"],
        "temporary_admin_absence_receipt": _json_copy(absence_receipt),
        "temporary_admin_absence_receipt_sha256": absence_receipt[
            "evidence_sha256"
        ],
        "terminal_observation_sha256": terminal_observation["observation_sha256"],
        "services_attestation_sha256": services["attestation_sha256"],
        "bootstrap_login_retained": True,
        "bootstrap_login_password_disabled": True,
        "temporary_admin_absent": True,
        "legacy_archive_preserved": True,
        "writer_credential_inode_preserved": True,
        "secret_material_recorded": False,
        "terminal_at_unix": now_unix,
    }
    _require_secret_free(unsigned)
    return _validate_terminal_receipt(
        {**unsigned, "receipt_sha256": _sha256_json(unsigned)},
        plan=plan,
        approval=approval,
        initial_admin_authority=initial_admin_authority,
        deletion_admin_authority=deletion_admin_authority,
        role_receipt=role_receipt,
        bootstrap_authority=bootstrap_authority,
        post_disable_authority=post_disable_authority,
        hba_receipt=hba_receipt,
        self_disable_receipt=self_disable_receipt,
        predelete_receipt=predelete_receipt,
        absence_receipt=absence_receipt,
        terminal_observation=terminal_observation,
        services=services,
    )


def execute_approved_phase_b(
    plan: PhaseBPlan,
    *,
    approval: Mapping[str, Any],
    role_artifact: SealedSQLArtifact,
    journal: AppendOnlyPhaseBJournal,
    dependencies: PhaseBDependencies,
    _clock: Callable[[], float] = time.time,
) -> Mapping[str, Any]:
    """Apply or resume the exact approved Phase-B workflow.

    Every mutation is preceded by a freshly fsynced stopped-services entry.
    A failed run intentionally leaves its exact journal and any recoverable
    external authority in place; a subsequent run rotates the same fixed
    accounts and resumes.  There is no best-effort cleanup path and no call
    capable of deleting the persistent bootstrap login.
    """

    _require_foundation_runtime()
    if not isinstance(plan, PhaseBPlan):
        raise TypeError("PhaseBPlan is required")
    if not isinstance(journal, AppendOnlyPhaseBJournal):
        raise PhaseBError("phase_b_journal_missing")
    deps = _require_dependencies(dependencies)
    approved = PhaseBApproval.from_mapping(
        approval,
        plan=plan,
        now_unix=int(_clock()),
    )
    artifact = _validate_role_artifact(role_artifact, plan=plan)

    with journal.lock(plan):
        entries = journal.load(plan)
        if entries and entries[-1].event == "terminal":
            receipt = entries[-1].evidence.get("terminal_receipt")
            if not isinstance(receipt, Mapping):
                raise PhaseBError("phase_b_terminal_receipt_missing")
            bindings = _terminal_replay_bindings(entries, plan=plan)
            return _json_copy(
                _validate_terminal_receipt(
                    receipt,
                    plan=plan,
                    approval=approved,
                    **bindings,
                )
            )
        if not entries:
            journal.append(
                plan,
                event="intent",
                idempotency_key="approved-intent",
                evidence={
                    "approval_sha256": approved.sha256,
                    "plan_sha256": plan.sha256,
                    "preterminal": True,
                    "safe_to_start": False,
                    "secret_material_recorded": False,
                },
                now_unix=int(_clock()),
            )
        events = journal.events(plan)

        execution_writer: ClosableSession | None = None
        try:
            execution_writer = deps.writer_session_factory()
            if not isinstance(getattr(execution_writer, "username", None), str) or execution_writer.username != SQL_USER:
                raise PhaseBError("phase_b_writer_session_invalid")
            # A stopped-services receipt means a mutation boundary may have
            # been entered even if its authority receipt was never published.
            # Only an intent-only journal is still eligible for the pristine
            # collector; every other replay must inventory recoverable state.
            if events == {"intent"}:
                current = PhaseBPreflight.from_mapping(
                    deps.pristine_preflight_collector(execution_writer)
                )
                if current.stable_projection != plan.preflight.stable_projection:
                    raise PhaseBError("phase_b_preflight_changed_after_approval")
                execution_session_sha = str(
                    current.value["database"]["session_identity_sha256"]
                )
            else:
                recovery = PhaseBRecoveryObservation.from_mapping(
                    deps.recovery_collector(execution_writer, plan, events),
                    plan=plan,
                )
                execution_session_sha = str(
                    recovery.value["database"]["session_identity_sha256"]
                )
        finally:
            _safe_close(execution_writer)

        initial_admin_boundary: TemporaryAdminBoundary | None = None
        initial_admin_secret: bytearray | None = None
        admin_session: RoleArtifactSession | None = None
        initial_admin_authority: Mapping[str, Any] | None = None
        role_receipt: Mapping[str, Any] | None = None
        predelete_receipt: Mapping[str, Any] | None = None
        predelete_process_instance_sha256: str | None = None
        needs_sql_admin = not {"role_ready", "predelete_verified"}.issubset(events)
        try:
            if needs_sql_admin:
                _recheck_services(
                    plan, deps, journal,
                    transition="temporary_admin_initial_authority",
                    clock=_clock,
                )
                initial_admin_boundary = deps.temporary_admin_factory(plan)
                initial_admin_secret = _require_secret(deps.secret_factory())
                _revalidate_approval(approved, plan, _clock)
                initial_admin_authority = _provision_temporary_admin(
                    initial_admin_boundary,
                    plan,
                    initial_admin_secret,
                    before_mutation=lambda: _revalidate_approval(
                        approved, plan, _clock
                    ),
                )
                journal.append(
                    plan,
                    event="temporary_admin_authority",
                    idempotency_key=(
                        "temporary-admin-initial:"
                        + initial_admin_authority["receipt_sha256"]
                    ),
                    evidence={
                        "purpose": "foundation_sql",
                        "authority_receipt": initial_admin_authority,
                        "preterminal": True,
                        "safe_to_start": False,
                    },
                    now_unix=int(_clock()),
                )
                admin_session = deps.admin_session_factory(
                    plan, plan.temporary_admin_username, initial_admin_secret
                )
                if admin_session.username != plan.temporary_admin_username:
                    raise PhaseBError("phase_b_admin_session_invalid")
            else:
                authority_event = _last_required(
                    journal, plan, "temporary_admin_authority",
                    "phase_b_initial_admin_authority_missing",
                )
                initial_admin_authority = _validate_temporary_admin_authority(
                    authority_event["authority_receipt"], plan=plan
                )

            if "role_ready" not in events:
                if admin_session is None:
                    raise PhaseBError("phase_b_admin_session_missing")
                _recheck_services(
                    plan, deps, journal,
                    transition="phase_b_role_artifact",
                    clock=_clock,
                )
                _revalidate_approval(approved, plan, _clock)
                role_receipt = _validate_role_receipt(
                    admin_session.execute_phase_b_role_artifact(
                        artifact,
                        bindings={
                            "muncho.canonical_writer_phase_b_release_revision": plan.revision,
                            "muncho.canonical_writer_phase_b_role_artifact_sha256": artifact.sha256,
                            "muncho.canonical_writer_phase_b_initial_observation_sha256": plan.preflight.sha256,
                            "muncho.canonical_writer_phase_b_approved_plan_sha256": plan.sha256,
                        },
                    ),
                    plan=plan,
                )
                journal.append(
                    plan,
                    event="role_ready",
                    idempotency_key="role:" + role_receipt["receipt_sha256"],
                    evidence={
                        "role_receipt": role_receipt,
                        "preterminal": True,
                        "safe_to_start": False,
                    },
                    now_unix=int(_clock()),
                )
                events = journal.events(plan)
            else:
                role_event = _last_required(
                    journal, plan, "role_ready", "phase_b_role_receipt_missing"
                )
                role_receipt = _validate_role_receipt(
                    role_event["role_receipt"], plan=plan
                )

            bootstrap_authority: Mapping[str, Any]
            post_disable_authority: Mapping[str, Any]
            hba_receipt: Mapping[str, Any]
            self_disable_receipt: Mapping[str, Any]
            bootstrap_mutation_boundary: BootstrapLoginBoundary | None = None
            if "bootstrap_password_disabled" not in events:
                _recheck_services(
                    plan, deps, journal,
                    transition="bootstrap_login_authority",
                    clock=_clock,
                )
                bootstrap_boundary = deps.bootstrap_login_factory(plan)
                bootstrap_mutation_boundary = bootstrap_boundary
                bootstrap_secret = _require_secret(deps.secret_factory())
                try:
                    _revalidate_approval(approved, plan, _clock)
                    bootstrap_authority = _provision_bootstrap_login(
                        bootstrap_boundary,
                        plan,
                        bootstrap_secret,
                        before_mutation=lambda: _revalidate_approval(
                            approved, plan, _clock
                        ),
                    )
                    journal.append(
                        plan,
                        event="bootstrap_authority",
                        idempotency_key=(
                            "bootstrap-authority:"
                            + bootstrap_authority["receipt_sha256"]
                        ),
                        evidence={
                            "authority_receipt": bootstrap_authority,
                            "preterminal": True,
                            "safe_to_start": False,
                        },
                        now_unix=int(_clock()),
                    )
                    hba_receipt = _validate_hba_receipt(
                        deps.hba_collector(plan, bootstrap_secret, bootstrap_authority),
                        plan=plan,
                        authority=bootstrap_authority,
                    )
                    now = int(_clock())
                    if not hba_receipt["observed_at_unix"] <= now < hba_receipt[
                        "expires_at_unix"
                    ]:
                        raise PhaseBError("phase_b_hba_receipt_stale")
                    journal.append(
                        plan,
                        event="bootstrap_hba_rejected",
                        idempotency_key="hba:" + hba_receipt["receipt_sha256"],
                        evidence={
                            "hba_receipt": hba_receipt,
                            "preterminal": True,
                            "safe_to_start": False,
                        },
                        now_unix=now,
                    )
                    _recheck_services(
                        plan, deps, journal,
                        transition="bootstrap_password_self_disable",
                        clock=_clock,
                    )
                    _revalidate_approval(approved, plan, _clock)
                    self_disable_receipt = _validate_self_disable_receipt(
                        deps.bootstrap_self_disable.disable_and_prove_denied(
                            plan=plan,
                            provisional_password=bootstrap_secret,
                            authority_receipt=bootstrap_authority,
                            hba_rejection_receipt=hba_receipt,
                            statement=SELF_DISABLE_SQL,
                        ),
                        plan=plan,
                        authority=bootstrap_authority,
                        hba=hba_receipt,
                    )
                    post_disable_authority = _validate_bootstrap_authority(
                        bootstrap_boundary.authority_receipt(), plan=plan
                    )
                    if post_disable_authority != bootstrap_authority:
                        raise PhaseBError(
                            "phase_b_bootstrap_authority_changed_after_self_disable"
                        )
                    journal.append(
                        plan,
                        event="bootstrap_password_disabled",
                        idempotency_key=(
                            "bootstrap-disabled:"
                            + self_disable_receipt["receipt_sha256"]
                        ),
                        evidence={
                            "initial_authority_receipt": bootstrap_authority,
                            "post_disable_authority_receipt": post_disable_authority,
                            "hba_receipt": hba_receipt,
                            "self_disable_receipt": self_disable_receipt,
                            "preterminal": True,
                            "safe_to_start": False,
                        },
                        now_unix=int(_clock()),
                    )
                finally:
                    _zeroize(bootstrap_secret)
                events = journal.events(plan)
            else:
                disabled = _last_required(
                    journal, plan, "bootstrap_password_disabled",
                    "phase_b_bootstrap_disable_receipt_missing",
                )
                bootstrap_authority = _validate_bootstrap_authority(
                    disabled["initial_authority_receipt"], plan=plan
                )
                post_disable_authority = _validate_bootstrap_authority(
                    disabled["post_disable_authority_receipt"], plan=plan
                )
                hba_receipt = _validate_hba_receipt(
                    disabled["hba_receipt"], plan=plan,
                    authority=bootstrap_authority,
                )
                self_disable_receipt = _validate_self_disable_receipt(
                    disabled["self_disable_receipt"], plan=plan,
                    authority=bootstrap_authority, hba=hba_receipt,
                )

            if "predelete_verified" not in events:
                if admin_session is None or role_receipt is None:
                    raise PhaseBError("phase_b_admin_session_missing")
                predelete_receipt = _validate_predelete_receipt(
                    deps.predelete_collector(
                        admin_session,
                        plan,
                        role_receipt,
                        post_disable_authority,
                        self_disable_receipt,
                    ),
                    plan=plan,
                    role=role_receipt,
                    authority=post_disable_authority,
                    self_disable=self_disable_receipt,
                )
                journal.append(
                    plan,
                    event="predelete_verified",
                    idempotency_key="predelete:" + predelete_receipt["receipt_sha256"],
                    evidence={
                        "predelete_receipt": predelete_receipt,
                        "process_instance_sha256": _PROCESS_INSTANCE_SHA256,
                        "preterminal": True,
                        "safe_to_start": False,
                    },
                    now_unix=int(_clock()),
                )
                predelete_process_instance_sha256 = _PROCESS_INSTANCE_SHA256
            else:
                predelete_event = _last_required(
                    journal, plan, "predelete_verified",
                    "phase_b_predelete_receipt_missing",
                )
                predelete_receipt = _validate_predelete_receipt(
                    predelete_event["predelete_receipt"],
                    plan=plan,
                    role=role_receipt,
                    authority=post_disable_authority,
                    self_disable=self_disable_receipt,
                )
                predelete_process_instance_sha256 = _digest(
                    predelete_event.get("process_instance_sha256"),
                    "phase_b_predelete_process_identity_missing",
                )
        finally:
            primary_error = sys.exc_info()[1]
            close_error: Exception | None = None
            try:
                _safe_close(admin_session)
            except Exception as exc:
                close_error = exc
            _zeroize(initial_admin_secret)
            if (
                admin_session is not None
                and close_error is None
                and predelete_receipt is not None
            ):
                journal.append(
                    plan,
                    event="temporary_admin_closed",
                    idempotency_key=(
                        "temporary-admin-closed:"
                        + str(predelete_receipt["receipt_sha256"] if predelete_receipt else "missing")
                    ),
                    evidence={
                        "predelete_receipt_sha256": (
                            predelete_receipt["receipt_sha256"] if predelete_receipt else None
                        ),
                        "admin_session_closed": True,
                        "process_instance_sha256": _PROCESS_INSTANCE_SHA256,
                        "provisional_secret_zeroized": True,
                        "preterminal": True,
                        "safe_to_start": False,
                        "secret_material_recorded": False,
                    },
                    now_unix=int(_clock()),
                )
            if close_error is not None and primary_error is None:
                raise close_error

        events = journal.events(plan)
        if "temporary_admin_closed" not in events:
            # A resumed run may have journaled predelete before process loss but
            # cannot claim deletion until the old SQL session is known gone.
            if (
                predelete_process_instance_sha256 is None
                or predelete_process_instance_sha256 == _PROCESS_INSTANCE_SHA256
            ):
                raise PhaseBError("phase_b_admin_session_close_unproven")
            journal.append(
                plan,
                event="temporary_admin_closed",
                idempotency_key="temporary-admin-closed:recovered-process",
                evidence={
                    "predelete_receipt_sha256": predelete_receipt["receipt_sha256"],
                    "admin_session_closed": True,
                    "provisional_secret_zeroized": True,
                    "process_recovery_boundary": True,
                    "predelete_process_instance_sha256": (
                        predelete_process_instance_sha256
                    ),
                    "recovery_process_instance_sha256": (
                        _PROCESS_INSTANCE_SHA256
                    ),
                    "preterminal": True,
                    "safe_to_start": False,
                    "secret_material_recorded": False,
                },
                now_unix=int(_clock()),
            )
            events = journal.events(plan)

        if "temporary_admin_absent" not in events:
            _recheck_services(
                plan, deps, journal,
                transition="temporary_admin_predelete_reacquire",
                clock=_clock,
            )
            deletion_boundary = deps.temporary_admin_factory(plan)
            if deletion_boundary is initial_admin_boundary:
                raise PhaseBError("phase_b_delete_boundary_not_fresh")
            deletion_secret = _require_secret(deps.secret_factory())
            try:
                _revalidate_approval(approved, plan, _clock)
                deletion_admin_authority = _provision_temporary_admin(
                    deletion_boundary,
                    plan,
                    deletion_secret,
                    before_mutation=lambda: _revalidate_approval(
                        approved, plan, _clock
                    ),
                )
                if initial_admin_authority is None or deletion_admin_authority[
                    "receipt_sha256"
                ] == initial_admin_authority["receipt_sha256"]:
                    raise PhaseBError("phase_b_delete_authority_not_fresh")
                journal.append(
                    plan,
                    event="temporary_admin_predelete_authority",
                    idempotency_key=(
                        "temporary-admin-predelete:"
                        + deletion_admin_authority["receipt_sha256"]
                    ),
                    evidence={
                        "purpose": "fresh_predelete_authority",
                        "authority_receipt": deletion_admin_authority,
                        "preterminal": True,
                        "safe_to_start": False,
                    },
                    now_unix=int(_clock()),
                )
            finally:
                _zeroize(deletion_secret)
            _recheck_services(
                plan, deps, journal,
                transition="temporary_admin_delete",
                clock=_clock,
            )
            _revalidate_approval(approved, plan, _clock)
            deletion_boundary.delete_and_confirm_absent(
                plan.temporary_admin_username
            )
            absence_receipt = _validate_admin_absence_receipt(
                deletion_boundary.reconciliation_receipt(),
                plan=plan,
                fresh_authority=deletion_admin_authority,
            )
            journal.append(
                plan,
                event="temporary_admin_absent",
                idempotency_key="admin-absent:" + absence_receipt["evidence_sha256"],
                evidence={
                    "fresh_predelete_authority_receipt": deletion_admin_authority,
                    "full_absence_receipt": absence_receipt,
                    "preterminal": True,
                    "safe_to_start": False,
                },
                now_unix=int(_clock()),
            )
        else:
            absent_event = _last_required(
                journal, plan, "temporary_admin_absent",
                "phase_b_admin_absence_receipt_missing",
            )
            deletion_admin_authority = _validate_temporary_admin_authority(
                absent_event["fresh_predelete_authority_receipt"], plan=plan
            )
            absence_receipt = _validate_admin_absence_receipt(
                absent_event["full_absence_receipt"],
                plan=plan,
                fresh_authority=deletion_admin_authority,
            )

        fresh_bootstrap_boundary = deps.bootstrap_login_factory(plan)
        if (
            bootstrap_mutation_boundary is not None
            and fresh_bootstrap_boundary is bootstrap_mutation_boundary
        ):
            raise PhaseBError("phase_b_bootstrap_observer_not_fresh")
        bootstrap_resource = fresh_bootstrap_boundary.describe()
        if bootstrap_resource is None:
            raise PhaseBError("phase_b_bootstrap_login_missing")
        bootstrap_resource = _validate_bootstrap_resource(bootstrap_resource)
        if (
            _sha256_json(bootstrap_resource)
            != post_disable_authority["resource_projection_sha256"]
            or bootstrap_resource["etag"] != post_disable_authority["etag"]
        ):
            raise PhaseBError("phase_b_bootstrap_resource_changed")

        terminal_writer: ClosableSession | None = None
        try:
            terminal_writer = deps.writer_session_factory()
            if terminal_writer.username != SQL_USER or terminal_writer is execution_writer:
                raise PhaseBError("phase_b_terminal_writer_session_not_fresh")
            terminal_observation = _validate_terminal_observation(
                deps.terminal_collector(
                    terminal_writer, plan, bootstrap_resource, absence_receipt
                ),
                plan=plan,
                execution_preflight_session_sha256=execution_session_sha,
            )
        finally:
            _safe_close(terminal_writer)
        terminal_services = _recheck_services(
            plan, deps, journal,
            transition="terminal_observation",
            clock=_clock,
        )
        if _service_stable_projection(terminal_observation["services"]) != _service_stable_projection(
            terminal_services
        ):
            raise PhaseBError("phase_b_terminal_services_drifted")
        journal.append(
            plan,
            event="terminal_observed",
            idempotency_key="terminal-observed:" + terminal_observation["observation_sha256"],
            evidence={
                "terminal_observation": terminal_observation,
                "preterminal": True,
                "safe_to_start": False,
            },
            now_unix=int(_clock()),
        )
        if initial_admin_authority is None or role_receipt is None or predelete_receipt is None:
            raise PhaseBError("phase_b_terminal_evidence_incomplete")
        terminal_receipt = _terminal_receipt(
            plan=plan,
            approval=approved,
            initial_admin_authority=initial_admin_authority,
            deletion_admin_authority=deletion_admin_authority,
            role_receipt=role_receipt,
            bootstrap_authority=bootstrap_authority,
            post_disable_authority=post_disable_authority,
            hba_receipt=hba_receipt,
            self_disable_receipt=self_disable_receipt,
            predelete_receipt=predelete_receipt,
            absence_receipt=absence_receipt,
            terminal_observation=terminal_observation,
            services=terminal_services,
            now_unix=int(_clock()),
        )
        journal.append(
            plan,
            event="terminal",
            idempotency_key="terminal:" + terminal_receipt["receipt_sha256"],
            evidence={"terminal_receipt": terminal_receipt},
            now_unix=int(_clock()),
        )
        return _json_copy(terminal_receipt)


__all__ = [
    "AppendOnlyPhaseBJournal",
    "PHASE_B_APPROVAL_SCHEMA",
    "PHASE_B_HBA_RECEIPT_SCHEMA",
    "PHASE_B_PLAN_SCHEMA",
    "PHASE_B_PREFLIGHT_SCHEMA",
    "PHASE_B_PREDELETE_SCHEMA",
    "PHASE_B_RECOVERY_SCHEMA",
    "PHASE_B_SELF_DISABLE_SCHEMA",
    "PHASE_B_TERMINAL_OBSERVATION_SCHEMA",
    "PHASE_B_TERMINAL_RECEIPT_SCHEMA",
    "PhaseBApproval",
    "PhaseBDependencies",
    "PhaseBError",
    "PhaseBPlan",
    "PhaseBPreflight",
    "PhaseBRecoveryObservation",
    "SELF_DISABLE_SQL",
    "SERVICE_UNITS",
    "build_phase_b_plan",
    "execute_approved_phase_b",
]
