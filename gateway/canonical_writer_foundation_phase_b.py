"""Deterministic isolated-canary Phase-B foundation workflow.

This module owns no semantic decision.  It validates an owner-authored exact
plan and mechanically composes injected, fixed-scope boundaries.  In
particular it cannot choose a database, user, role, SQL payload, service, or
recovery strategy.  Packaged production wiring lives at the privileged edge in
``gateway.canonical_writer_phase_b_runtime`` so this foundation remains free of
transport and owner-identity policy.

The persistent bootstrap login is never deleted here.  Its provisional
password is used once through a dedicated self-session boundary, disabled by
``ALTER ROLE CURRENT_USER PASSWORD NULL``, and then proved unusable by a fresh
authentication attempt.  Only the plan-derived temporary Cloud SQL admin is
deleted.
"""

from __future__ import annotations

import copy
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import struct
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from gateway.canonical_canary_host_identity import (
    DEDICATED_CANARY_INSTANCE_ID,
    DEDICATED_CANARY_INSTANCE_NAME,
    DEDICATED_CANARY_PROJECT_ID,
    DEDICATED_CANARY_PROJECT_NUMBER,
    DEDICATED_CANARY_SERVICE_ACCOUNT,
    DEDICATED_CANARY_ZONE,
    FULL_CANARY_HOST_IDENTITY_SCHEMA,
)

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
PHASE_B_APPROVAL_SCHEMA = "muncho-canonical-writer-phase-b-approval.v2"
PHASE_B_SOURCE_AUTH_SCHEMA = (
    "muncho-canonical-writer-phase-b-owner-source-auth.v1"
)
PHASE_B_APPROVAL_SSHSIG_NAMESPACE = (
    "muncho-canonical-writer-phase-b-owner-v2"
)
PHASE_B_SOURCE_AUTH_SSHSIG_NAMESPACE = (
    "muncho-canonical-writer-phase-b-source-auth-v1"
)
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
PHASE_B_DURABLE_FOUNDATION_SCHEMA = (
    "muncho-canonical-writer-foundation-phase-b-durable.v1"
)
PHASE_B_READINESS_OBSERVATION_SCHEMA = (
    "muncho-canonical-writer-foundation-phase-b-readiness-observation.v1"
)
PHASE_B_READINESS_RECEIPT_SCHEMA = (
    "muncho-canonical-writer-foundation-phase-b-readiness.v1"
)
PHASE_B_BOOTSTRAP_CONTINUITY_SCHEMA = (
    "muncho-canonical-writer-foundation-phase-b-bootstrap-continuity.v1"
)
PHASE_B_READINESS_MAX_AGE_SECONDS = 300
_PHASE_B_READINESS_ROOT = Path(
    "/var/lib/muncho/canonical-writer-phase-b-readiness"
)

# Readiness is startup authority, not a caller-owned cache.  Production
# publication is therefore rooted in a fixed root-owned journal.  Tests may
# replace these private process constants, but the public reader never accepts
# a path or an authority identity from its caller.
_READINESS_AUTHORITY_UID = 0
_READINESS_AUTHORITY_GID = 0
PHASE_B_VM_OAUTH_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
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


def _validate_exact_unsigned_json_text(
    *,
    unsigned_text: Any,
    unsigned_value: Mapping[str, Any],
    digest: Any,
    code: str,
) -> None:
    """Bind a receipt to exact producer-authored UTF-8 JSON bytes.

    PostgreSQL JSONB text ordering is intentionally not recreated here.  The
    producer carries the exact ``jsonb::text`` bytes; Python only hashes those
    bytes and parses them to prove that they describe the returned unsigned
    value without duplicate keys or non-finite numbers.
    """

    if not isinstance(unsigned_text, str):
        raise PhaseBError(code)
    try:
        encoded = unsigned_text.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise PhaseBError(code) from exc
    if not encoded or len(encoded) > _MAX_JSON_BYTES:
        raise PhaseBError(code)
    if _sha256_bytes(encoded) != _digest(digest, code):
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
    if not isinstance(parsed, dict) or parsed != unsigned_value:
        raise PhaseBError(code)


def _validate_database_receipt_envelope(value: Any, *, code: str) -> dict[str, Any]:
    """Validate the exact PostgreSQL JSONB text rather than reconstructing it."""

    raw = _strict_mapping(value, _DB_PREFLIGHT_FIELDS, code)
    unsigned = {
        name: item
        for name, item in raw.items()
        if name not in {"unsigned_receipt_jsonb_text", "receipt_sha256"}
    }
    _validate_exact_unsigned_json_text(
        unsigned_text=raw["unsigned_receipt_jsonb_text"],
        unsigned_value=unsigned,
        digest=raw["receipt_sha256"],
        code=code,
    )
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
        "owner_resume_public_key_ed25519_hex",
        "owner_resume_key_id",
        "owner_resume_public_key_file_sha256",
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
        public_hex = raw["owner_resume_public_key_ed25519_hex"]
        if not isinstance(public_hex, str) or re.fullmatch(r"[0-9a-f]{64}", public_hex) is None:
            raise PhaseBError("phase_b_plan_owner_key_invalid")
        public_bytes = bytes.fromhex(public_hex)
        if (
            raw["owner_resume_key_id"] != hashlib.sha256(public_bytes).hexdigest()
            or _SHA256_RE.fullmatch(
                str(raw["owner_resume_public_key_file_sha256"])
            )
            is None
        ):
            raise PhaseBError("phase_b_plan_owner_key_invalid")
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
            "owner_resume_public_key_ed25519_hex": public_hex,
            "owner_resume_key_id": raw["owner_resume_key_id"],
            "owner_resume_public_key_file_sha256": raw[
                "owner_resume_public_key_file_sha256"
            ],
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
    owner_resume_public_key_ed25519_hex: str,
    owner_resume_public_key_file_sha256: str,
) -> PhaseBPlan:
    if not isinstance(preflight, PhaseBPreflight):
        raise TypeError("PhaseBPreflight is required")
    owner = _digest(owner_subject_sha256, "phase_b_owner_subject_invalid")
    if (
        not isinstance(owner_resume_public_key_ed25519_hex, str)
        or re.fullmatch(r"[0-9a-f]{64}", owner_resume_public_key_ed25519_hex)
        is None
    ):
        raise PhaseBError("phase_b_plan_owner_key_invalid")
    owner_resume_public_key_file_sha256 = _digest(
        owner_resume_public_key_file_sha256,
        "phase_b_plan_owner_key_invalid",
    )
    owner_resume_key_id = hashlib.sha256(
        bytes.fromhex(owner_resume_public_key_ed25519_hex)
    ).hexdigest()
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
        "owner_resume_public_key_ed25519_hex": (
            owner_resume_public_key_ed25519_hex
        ),
        "owner_resume_key_id": owner_resume_key_id,
        "owner_resume_public_key_file_sha256": (
            owner_resume_public_key_file_sha256
        ),
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
        "owner_resume_public_key_ed25519_hex": (
            owner_resume_public_key_ed25519_hex
        ),
        "owner_resume_key_id": owner_resume_key_id,
        "owner_resume_public_key_file_sha256": (
            owner_resume_public_key_file_sha256
        ),
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


_SSHSIG_BEGIN = "-----BEGIN SSH SIGNATURE-----"
_SSHSIG_END = "-----END SSH SIGNATURE-----"
_MAX_SSHSIG_ASCII_BYTES = 4096


def _ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _read_ssh_string(payload: bytes, offset: int) -> tuple[bytes, int]:
    if offset + 4 > len(payload):
        raise PhaseBError("phase_b_approval_signature_invalid")
    size = struct.unpack(">I", payload[offset : offset + 4])[0]
    start = offset + 4
    end = start + size
    if size > _MAX_SSHSIG_ASCII_BYTES or end > len(payload):
        raise PhaseBError("phase_b_approval_signature_invalid")
    return payload[start:end], end


def verify_phase_b_sshsig(
    signature: str,
    *,
    message: bytes,
    public_key_ed25519_hex: str,
    namespace: str,
) -> None:
    """Strictly verify OpenSSH SSHSIG/Ed25519 over SHA-512(message)."""

    if (
        not isinstance(signature, str)
        or not isinstance(message, bytes)
        or not isinstance(namespace, str)
        or not namespace
        or len(signature.encode("ascii", errors="ignore"))
        > _MAX_SSHSIG_ASCII_BYTES
        or not signature.startswith(_SSHSIG_BEGIN + "\n")
        or not signature.endswith("\n" + _SSHSIG_END + "\n")
        or re.fullmatch(r"[0-9a-f]{64}", public_key_ed25519_hex or "") is None
    ):
        raise PhaseBError("phase_b_approval_signature_invalid")
    lines = signature.splitlines()
    if (
        len(lines) < 3
        or lines[0] != _SSHSIG_BEGIN
        or lines[-1] != _SSHSIG_END
        or any(re.fullmatch(r"[A-Za-z0-9+/=]{1,70}", line) is None for line in lines[1:-1])
        or any(len(line) != 70 for line in lines[1:-2])
    ):
        raise PhaseBError("phase_b_approval_signature_invalid")
    try:
        envelope = base64.b64decode("".join(lines[1:-1]), validate=True)
    except (ValueError, UnicodeError) as exc:
        raise PhaseBError("phase_b_approval_signature_invalid") from exc
    if not envelope.startswith(b"SSHSIG"):
        raise PhaseBError("phase_b_approval_signature_invalid")
    offset = 6
    if offset + 4 > len(envelope) or struct.unpack(">I", envelope[offset:offset+4])[0] != 1:
        raise PhaseBError("phase_b_approval_signature_invalid")
    offset += 4
    public_blob, offset = _read_ssh_string(envelope, offset)
    observed_namespace, offset = _read_ssh_string(envelope, offset)
    reserved, offset = _read_ssh_string(envelope, offset)
    hash_algorithm, offset = _read_ssh_string(envelope, offset)
    signature_blob, offset = _read_ssh_string(envelope, offset)
    if offset != len(envelope):
        raise PhaseBError("phase_b_approval_signature_invalid")
    key_type, key_offset = _read_ssh_string(public_blob, 0)
    public_bytes, key_offset = _read_ssh_string(public_blob, key_offset)
    algorithm, signature_offset = _read_ssh_string(signature_blob, 0)
    raw_signature, signature_offset = _read_ssh_string(signature_blob, signature_offset)
    try:
        expected_public = bytes.fromhex(public_key_ed25519_hex)
    except ValueError as exc:
        raise PhaseBError("phase_b_approval_signature_invalid") from exc
    if (
        key_offset != len(public_blob)
        or signature_offset != len(signature_blob)
        or key_type != b"ssh-ed25519"
        or algorithm != b"ssh-ed25519"
        or public_bytes != expected_public
        or len(public_bytes) != 32
        or len(raw_signature) != 64
        or observed_namespace != namespace.encode("ascii")
        or reserved != b""
        or hash_algorithm != b"sha512"
    ):
        raise PhaseBError("phase_b_approval_signature_invalid")
    signed = (
        b"SSHSIG"
        + _ssh_string(observed_namespace)
        + _ssh_string(reserved)
        + _ssh_string(hash_algorithm)
        + _ssh_string(hashlib.sha512(message).digest())
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(
            raw_signature,
            signed,
        )
    except (InvalidSignature, ValueError) as exc:
        raise PhaseBError("phase_b_approval_signature_invalid") from exc


_APPROVAL_FIELDS = frozenset(
    {
        "schema",
        "purpose",
        "sequence",
        "previous_approval_sha256",
        "plan_sha256",
        "intent_sha256",
        "owner_subject_sha256",
        "approval_source_sha256",
        "owner_public_key_ed25519_hex",
        "owner_key_id",
        "owner_public_key_file_sha256",
        "source_authentication_sha256",
        "approved",
        "issued_at_unix",
        "expires_at_unix",
        "secret_material_recorded",
        "signature_sshsig",
        "approval_sha256",
    }
)

_APPROVAL_SIGNATURE_FIELDS = _APPROVAL_FIELDS - {
    "signature_sshsig",
    "approval_sha256",
}


def phase_b_approval_signature_payload(value: Mapping[str, Any]) -> bytes:
    if set(value) != _APPROVAL_FIELDS:
        raise PhaseBError("phase_b_approval_invalid")
    return _canonical_bytes(
        {name: _json_copy(value[name]) for name in _APPROVAL_SIGNATURE_FIELDS}
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
        sequence = raw["sequence"]
        previous = raw["previous_approval_sha256"]
        purpose = raw["purpose"]
        public_hex = raw["owner_public_key_ed25519_hex"]
        signature = raw["signature_sshsig"]
        if (
            raw["schema"] != PHASE_B_APPROVAL_SCHEMA
            or type(sequence) is not int
            or sequence < 0
            or purpose
            != ("initial_apply" if sequence == 0 else "resume_incomplete")
            or (sequence == 0 and previous is not None)
            or (
                sequence > 0
                and (
                    not isinstance(previous, str)
                    or _SHA256_RE.fullmatch(previous) is None
                )
            )
            or raw["plan_sha256"] != plan.sha256
            or raw["intent_sha256"] != plan.value["intent_sha256"]
            or raw["owner_subject_sha256"] != plan.owner_subject_sha256
            or public_hex
            != plan.value["owner_resume_public_key_ed25519_hex"]
            or raw["owner_key_id"] != plan.value["owner_resume_key_id"]
            or raw["owner_public_key_file_sha256"]
            != plan.value["owner_resume_public_key_file_sha256"]
            or raw["approved"] is not True
            or type(raw["issued_at_unix"]) is not int
            or type(raw["expires_at_unix"]) is not int
            or not raw["issued_at_unix"] <= current < raw["expires_at_unix"]
            or raw["expires_at_unix"] - raw["issued_at_unix"] > 3600
            or raw["secret_material_recorded"] is not False
            or not isinstance(signature, str)
        ):
            raise PhaseBError("phase_b_approval_invalid")
        _digest(raw["approval_source_sha256"], "phase_b_approval_invalid")
        _digest(raw["source_authentication_sha256"], "phase_b_approval_invalid")
        verify_phase_b_sshsig(
            signature,
            message=phase_b_approval_signature_payload(raw),
            public_key_ed25519_hex=public_hex,
            namespace=PHASE_B_APPROVAL_SSHSIG_NAMESPACE,
        )
        _require_secret_free(raw)
        return cls(_json_copy(raw))

    @property
    def sha256(self) -> str:
        return str(self.value["approval_sha256"])

    @property
    def sequence(self) -> int:
        return int(self.value["sequence"])

    def to_mapping(self) -> dict[str, Any]:
        return _json_copy(self.value)


_SOURCE_AUTH_FIELDS = frozenset(
    {
        "schema",
        "authority_kind",
        "purpose",
        "sequence",
        "previous_approval_sha256",
        "plan_sha256",
        "intent_sha256",
        "owner_subject_sha256",
        "approval_source_sha256",
        "owner_key_id",
        "requested_at_unix",
        "expires_at_unix",
        "nonce_sha256",
        "signature_sshsig",
        "receipt_sha256",
    }
)
_SOURCE_AUTH_SIGNATURE_FIELDS = _SOURCE_AUTH_FIELDS - {
    "signature_sshsig",
    "receipt_sha256",
}


def phase_b_source_authentication_signature_payload(
    value: Mapping[str, Any],
) -> bytes:
    if set(value) != _SOURCE_AUTH_FIELDS:
        raise PhaseBError("phase_b_source_authentication_invalid")
    return _canonical_bytes(
        {
            name: _json_copy(value[name])
            for name in _SOURCE_AUTH_SIGNATURE_FIELDS
        }
    )


def validate_phase_b_source_authentication(
    value: Any,
    *,
    plan: PhaseBPlan,
    approval: PhaseBApproval,
) -> Mapping[str, Any]:
    raw = validate_phase_b_pending_source_authentication(
        value,
        plan=plan,
        expected_sequence=approval.sequence,
        expected_previous_approval_sha256=approval.value[
            "previous_approval_sha256"
        ],
        expected_purpose=str(approval.value["purpose"]),
        expected_approval_source_sha256=str(
            approval.value["approval_source_sha256"]
        ),
    )
    if (
        raw["purpose"] != approval.value["purpose"]
        or raw["requested_at_unix"] != approval.value["issued_at_unix"]
        or raw["expires_at_unix"] != approval.value["expires_at_unix"]
        or approval.value["source_authentication_sha256"]
        != raw["receipt_sha256"]
    ):
        raise PhaseBError("phase_b_source_authentication_invalid")
    return raw


def validate_phase_b_pending_source_authentication(
    value: Any,
    *,
    plan: PhaseBPlan,
    expected_sequence: int,
    expected_previous_approval_sha256: str | None,
    expected_purpose: str = "resume_incomplete",
    expected_approval_source_sha256: str,
) -> Mapping[str, Any]:
    raw = _hashed_mapping(
        value,
        fields=_SOURCE_AUTH_FIELDS,
        digest_field="receipt_sha256",
        code="phase_b_source_authentication_invalid",
    )
    if (
        raw["schema"] != PHASE_B_SOURCE_AUTH_SCHEMA
        or raw["authority_kind"] != "skyvision_mac_ops_ed25519"
        or expected_purpose not in {"initial_apply", "resume_incomplete"}
        or raw["purpose"] != expected_purpose
        or raw["sequence"] != expected_sequence
        or raw["previous_approval_sha256"]
        != expected_previous_approval_sha256
        or raw["plan_sha256"] != plan.sha256
        or raw["intent_sha256"] != plan.value["intent_sha256"]
        or raw["owner_subject_sha256"] != plan.owner_subject_sha256
        or raw["approval_source_sha256"]
        != expected_approval_source_sha256
        or raw["owner_key_id"] != plan.value["owner_resume_key_id"]
        or type(raw["requested_at_unix"]) is not int
        or type(raw["expires_at_unix"]) is not int
        or not 1 <= raw["expires_at_unix"] - raw["requested_at_unix"] <= 3600
    ):
        raise PhaseBError("phase_b_source_authentication_invalid")
    _digest(raw["nonce_sha256"], "phase_b_source_authentication_invalid")
    verify_phase_b_sshsig(
        raw["signature_sshsig"],
        message=phase_b_source_authentication_signature_payload(raw),
        public_key_ed25519_hex=plan.value[
            "owner_resume_public_key_ed25519_hex"
        ],
        namespace=PHASE_B_SOURCE_AUTH_SSHSIG_NAMESPACE,
    )
    _require_secret_free(raw)
    return _json_copy(raw)


def validate_phase_b_approval_chain(
    values: Sequence[Mapping[str, Any]],
    *,
    plan: PhaseBPlan,
    now_unix: int | None = None,
    require_fresh_head: bool = False,
) -> tuple[PhaseBApproval, ...]:
    """Validate one contiguous, same-authority approval chain.

    Historical members are validated at their issue time.  Freshness is a
    property of the current head only and is requested explicitly by the
    mutation boundary; durable replay never silently upgrades old approval.
    """

    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray, memoryview))
        or not values
        or type(require_fresh_head) is not bool
    ):
        raise PhaseBError("phase_b_approval_chain_invalid")
    result: list[PhaseBApproval] = []
    previous: PhaseBApproval | None = None
    for sequence, value in enumerate(values):
        if not isinstance(value, Mapping) or type(value.get("issued_at_unix")) is not int:
            raise PhaseBError("phase_b_approval_chain_invalid")
        approval = PhaseBApproval.from_mapping(
            value,
            plan=plan,
            now_unix=int(value["issued_at_unix"]),
        )
        if (
            approval.sequence != sequence
            or approval.value["previous_approval_sha256"]
            != (None if previous is None else previous.sha256)
            or (
                previous is not None
                and (
                    approval.value["intent_sha256"]
                    != previous.value["intent_sha256"]
                    or approval.value["owner_subject_sha256"]
                    != previous.value["owner_subject_sha256"]
                    or approval.value["approval_source_sha256"]
                    != previous.value["approval_source_sha256"]
                    or approval.value["owner_public_key_ed25519_hex"]
                    != previous.value["owner_public_key_ed25519_hex"]
                    or approval.value["owner_key_id"]
                    != previous.value["owner_key_id"]
                    or approval.value["owner_public_key_file_sha256"]
                    != previous.value["owner_public_key_file_sha256"]
                    or approval.value["issued_at_unix"]
                    <= previous.value["issued_at_unix"]
                )
            )
        ):
            raise PhaseBError("phase_b_approval_chain_invalid")
        result.append(approval)
        previous = approval
    assert previous is not None
    if require_fresh_head:
        current = int(time.time()) if now_unix is None else now_unix
        PhaseBApproval.from_mapping(
            previous.to_mapping(),
            plan=plan,
            now_unix=current,
        )
    return tuple(result)


def _active_approval_for_time(
    approvals: Sequence[PhaseBApproval],
    *,
    recorded_at_unix: int,
) -> PhaseBApproval:
    candidates = [
        approval
        for approval in approvals
        if approval.value["issued_at_unix"] <= recorded_at_unix
    ]
    if not candidates:
        raise PhaseBError("phase_b_approval_invalid")
    active = candidates[-1]
    if recorded_at_unix >= active.value["expires_at_unix"]:
        raise PhaseBError("phase_b_approval_invalid")
    return active


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


_CLOUD_USER_RESOURCE_FIELDS = frozenset(
    {
        "databaseRoles",
        "etag",
        "host",
        "instance",
        "name",
        "project",
        "type",
    }
)


def _validate_cloud_user_resource(value: Any) -> dict[str, Any]:
    """Validate one complete, secret-free stable Cloud SQL user projection."""

    raw = _strict_mapping(
        value,
        _CLOUD_USER_RESOURCE_FIELDS,
        "phase_b_cloud_user_resource_invalid",
    )
    roles = raw["databaseRoles"]
    etag = raw["etag"]
    if (
        not isinstance(roles, list)
        or roles != sorted(set(roles))
        or any(not isinstance(role, str) or not role for role in roles)
        or not isinstance(etag, str)
        or not 1 <= len(etag) <= 1024
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in etag)
        or raw["host"] != ""
        or raw["instance"] != SQL_INSTANCE
        or not isinstance(raw["name"], str)
        or not raw["name"]
        or raw["project"] != PROJECT
        or raw["type"] != "BUILT_IN"
    ):
        raise PhaseBError("phase_b_cloud_user_resource_invalid")
    _require_secret_free(raw)
    return raw


def _validate_terminal_cloud_inventory(
    value: Any,
    *,
    bootstrap_resource: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PhaseBError("phase_b_cloud_user_inventory_invalid")
    resources = [_validate_cloud_user_resource(item) for item in value]
    if resources != sorted(resources, key=lambda item: item["name"]):
        raise PhaseBError("phase_b_cloud_user_inventory_invalid")
    by_name = {item["name"]: item for item in resources}
    expected_names = {SQL_USER, "postgres", CANARY_BOOTSTRAP_LOGIN}
    if len(by_name) != len(resources) or set(by_name) != expected_names:
        raise PhaseBError("phase_b_cloud_user_inventory_invalid")
    expected_bootstrap = _validate_bootstrap_resource(bootstrap_resource)
    if by_name[CANARY_BOOTSTRAP_LOGIN] != expected_bootstrap:
        raise PhaseBError("phase_b_cloud_user_inventory_invalid")
    for name in (SQL_USER, "postgres"):
        if by_name[name]["databaseRoles"] != []:
            raise PhaseBError("phase_b_cloud_user_inventory_invalid")
    return resources


def _validate_terminal_cloud_operations(value: Any) -> list[list[Any]]:
    if not isinstance(value, list):
        raise PhaseBError("phase_b_cloud_operation_horizon_invalid")
    rows = [_validate_operation_row(row) for row in value]
    if (
        rows != sorted(rows, key=lambda row: row[0])
        or len({row[0] for row in rows}) != len(rows)
        or any(row[2] != "DONE" or row[4] is not True for row in rows)
    ):
        raise PhaseBError("phase_b_cloud_operation_horizon_invalid")
    return rows


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


_ROLE_RECEIPT_BODY_FIELDS = frozenset(
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
_ROLE_RECEIPT_FIELDS = frozenset(
    {*_ROLE_RECEIPT_BODY_FIELDS, "unsigned_receipt_jsonb_text"}
)
_ROLE_RECEIPT_SQL_ENVELOPE_FIELDS = frozenset(
    {"unsigned_receipt_jsonb_text", "receipt"}
)


def _validate_role_receipt(value: Any, *, plan: PhaseBPlan) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PhaseBError("phase_b_role_receipt_invalid")
    if set(value) == _ROLE_RECEIPT_SQL_ENVELOPE_FIELDS:
        envelope = _strict_mapping(
            value,
            _ROLE_RECEIPT_SQL_ENVELOPE_FIELDS,
            "phase_b_role_receipt_invalid",
        )
        body = _strict_mapping(
            envelope["receipt"],
            _ROLE_RECEIPT_BODY_FIELDS,
            "phase_b_role_receipt_invalid",
        )
        raw = {
            **body,
            "unsigned_receipt_jsonb_text": envelope[
                "unsigned_receipt_jsonb_text"
            ],
        }
    else:
        raw = _strict_mapping(
            value,
            _ROLE_RECEIPT_FIELDS,
            "phase_b_role_receipt_invalid",
        )
        body = {
            name: item
            for name, item in raw.items()
            if name != "unsigned_receipt_jsonb_text"
        }
    unsigned = {
        name: item for name, item in body.items() if name != "receipt_sha256"
    }
    _validate_exact_unsigned_json_text(
        unsigned_text=raw["unsigned_receipt_jsonb_text"],
        unsigned_value=unsigned,
        digest=body["receipt_sha256"],
        code="phase_b_role_receipt_invalid",
    )
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

_READINESS_RUNTIME_CLOUD_FIELDS = frozenset(
    {
        "project",
        "instance",
        "instance_projection",
        "instance_projection_sha256",
        "user_state_authority",
        "bootstrap_role_present",
        "bootstrap_login_present",
        "temporary_admin_absent",
        "temporary_admin_username_sha256",
        "temporary_admin_users",
        "user_operations_quiescent",
        "relevant_user_operations",
        "operation_ledger_sha256",
        "observed_at_unix",
    }
)
_READINESS_INSTANCE_PROJECTION = {
    "backendType": "SECOND_GEN",
    "connectionName": (
        "adventico-ai-platform:europe-west3:muncho-canary-pg18-v2"
    ),
    "databaseVersion": "POSTGRES_18",
    "ipAddresses": [{"ipAddress": "10.91.0.3", "type": "PRIVATE"}],
    "name": SQL_INSTANCE,
    "project": PROJECT,
    "region": "europe-west3",
    "state": "RUNNABLE",
}
_READINESS_INSTANCE_PROJECTION_SHA256 = (
    "c7979c4b0a97724a0ac6ac3217a67977a9584622546143aef093b146b7061139"
)


def _validate_readiness_runtime_cloud(
    value: Any,
    *,
    plan: PhaseBPlan,
    observed_at_unix: int,
) -> dict[str, Any]:
    raw = _strict_mapping(
        value,
        _READINESS_RUNTIME_CLOUD_FIELDS,
        "phase_b_readiness_cloud_invalid",
    )
    projection = raw["instance_projection"]
    operations = _validate_terminal_cloud_operations(
        raw["relevant_user_operations"]
    )
    if (
        raw["project"] != PROJECT
        or raw["instance"] != SQL_INSTANCE
        or projection != _READINESS_INSTANCE_PROJECTION
        or raw["instance_projection_sha256"]
        != _READINESS_INSTANCE_PROJECTION_SHA256
        or raw["instance_projection_sha256"]
        != _sha256_bytes(_canonical_bytes(projection) + b"\n")
        or raw["user_state_authority"] != "postgres_pg_roles"
        or raw["bootstrap_role_present"] is not True
        or raw["bootstrap_login_present"] is not True
        or raw["temporary_admin_absent"] is not True
        or raw["temporary_admin_username_sha256"]
        != _sha256_bytes(plan.temporary_admin_username.encode("ascii"))
        or raw["temporary_admin_users"] != []
        or raw["user_operations_quiescent"] is not True
        or raw["operation_ledger_sha256"] != _sha256_json(operations)
        or raw["observed_at_unix"] != observed_at_unix
        or any(row[2] != "DONE" or row[4] is not True for row in operations)
    ):
        raise PhaseBError("phase_b_readiness_cloud_invalid")
    return raw


def _validate_terminal_observation(
    value: Any,
    *,
    plan: PhaseBPlan,
    execution_preflight_session_sha256: str | None,
    services_release_revision: str | None = None,
    expected_bootstrap_authority: Mapping[str, Any] | None = None,
    expected_absence_receipt: Mapping[str, Any] | None = None,
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
    services = _validate_services(
        raw["services"],
        release_revision=(
            plan.revision
            if services_release_revision is None
            else _revision(services_release_revision)
        ),
    )
    runtime_cloud = set(raw["cloud_sql"]) == _READINESS_RUNTIME_CLOUD_FIELDS
    bootstrap_resource: Mapping[str, Any] | None = None
    if runtime_cloud:
        if (
            services_release_revision is None
            or expected_bootstrap_authority is not None
            or expected_absence_receipt is not None
        ):
            raise PhaseBError("phase_b_terminal_cloud_invalid")
        cloud = _validate_readiness_runtime_cloud(
            raw["cloud_sql"],
            plan=plan,
            observed_at_unix=raw["observed_at_unix"],
        )
        operations = _validate_terminal_cloud_operations(
            cloud["relevant_user_operations"]
        )
    else:
        cloud = _strict_mapping(
            raw["cloud_sql"],
            frozenset(
                {
                    "project",
                    "instance",
                    "bootstrap_resource",
                    "temporary_admin_absent",
                    "temporary_admin_username_sha256",
                    "user_inventory",
                    "user_inventory_sha256",
                    "user_operations_quiescent",
                    "relevant_user_operations",
                    "operation_ledger_sha256",
                    "observed_at_unix",
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
            or type(cloud["observed_at_unix"]) is not int
            or cloud["observed_at_unix"] <= 0
            or cloud["observed_at_unix"] != raw["observed_at_unix"]
            or services["observed_at_unix"] != raw["observed_at_unix"]
        ):
            raise PhaseBError("phase_b_terminal_cloud_invalid")
        bootstrap_resource = _validate_bootstrap_resource(
            cloud["bootstrap_resource"]
        )
        inventory = _validate_terminal_cloud_inventory(
            cloud["user_inventory"],
            bootstrap_resource=bootstrap_resource,
        )
        operations = _validate_terminal_cloud_operations(
            cloud["relevant_user_operations"]
        )
        if (
            cloud["user_inventory_sha256"] != _sha256_json(inventory)
            or cloud["operation_ledger_sha256"] != _sha256_json(operations)
        ):
            raise PhaseBError("phase_b_terminal_cloud_invalid")
    if expected_bootstrap_authority is not None:
        if bootstrap_resource is None:
            raise PhaseBError("phase_b_terminal_bootstrap_resource_drifted")
        authority = _validate_bootstrap_authority(
            expected_bootstrap_authority,
            plan=plan,
        )
        if (
            _sha256_json(bootstrap_resource)
            != authority["resource_projection_sha256"]
            or bootstrap_resource["etag"] != authority["etag"]
        ):
            raise PhaseBError("phase_b_terminal_bootstrap_resource_drifted")
        operation_by_name = {row[0]: row for row in operations}
        bootstrap_operation = operation_by_name.get(authority["operation_name"])
        if bootstrap_operation != [
            authority["operation_name"],
            authority["operation_type"],
            "DONE",
            plan.owner_subject_sha256,
            True,
        ]:
            raise PhaseBError("phase_b_terminal_bootstrap_operation_missing")
    if expected_absence_receipt is not None:
        absence = _strict_mapping(
            expected_absence_receipt,
            _ABSENCE_FIELDS,
            "phase_b_terminal_absence_receipt_drifted",
        )
        absence_rows = [
            _validate_operation_row(row)
            for row in absence["terminal_user_operations"]
        ]
        operation_by_name = {row[0]: row for row in operations}
        if any(operation_by_name.get(row[0]) != row for row in absence_rows):
            raise PhaseBError("phase_b_terminal_absence_receipt_drifted")
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
        "approval_sha256",
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
            or _SHA256_RE.fullmatch(str(raw["approval_sha256"])) is None
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
        try:
            self.root = root.parent.resolve(strict=True) / root.name
        except OSError as exc:
            raise PhaseBError("phase_b_journal_root_invalid") from exc
        self._lock_depth = 0
        self._active_plan_sha256: str | None = None
        self._active_root_fd: int | None = None
        self._active_plan_fd: int | None = None

    def _plan_root(self, plan: PhaseBPlan) -> Path:
        return self.root / plan.sha256

    def _entries_root(self, plan: PhaseBPlan) -> Path:
        return self._plan_root(plan) / "entries"

    def _staging_root(self, plan: PhaseBPlan) -> Path:
        return self._plan_root(plan) / "staging"

    def _lock_path(self, plan: PhaseBPlan) -> Path:
        return self._plan_root(plan) / ".lock"

    @staticmethod
    def _directory_flags() -> int:
        return (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )

    @staticmethod
    def _trusted_directory_status(status: os.stat_result) -> bool:
        return (
            stat.S_ISDIR(status.st_mode)
            and not stat.S_ISLNK(status.st_mode)
            and stat.S_IMODE(status.st_mode) == _JOURNAL_MODE
            and status.st_uid == _effective_uid()
            and status.st_gid == _effective_gid()
        )

    @classmethod
    def _open_absolute_parent(cls, path: Path) -> tuple[int, str]:
        """Open every existing parent component without following symlinks."""

        parts = path.parts
        if not path.is_absolute() or len(parts) < 2 or not path.name:
            raise PhaseBError("phase_b_journal_root_invalid")
        try:
            descriptor = os.open(parts[0], cls._directory_flags())
            for component in parts[1:-1]:
                child = os.open(
                    component,
                    cls._directory_flags(),
                    dir_fd=descriptor,
                )
                status = os.fstat(child)
                if not stat.S_ISDIR(status.st_mode):
                    os.close(child)
                    raise PhaseBError("phase_b_journal_directory_untrusted")
                os.close(descriptor)
                descriptor = child
        except PhaseBError:
            try:
                os.close(descriptor)
            except (OSError, UnboundLocalError):
                pass
            raise
        except OSError as exc:
            try:
                os.close(descriptor)
            except (OSError, UnboundLocalError):
                pass
            raise PhaseBError("phase_b_journal_unavailable") from exc
        return descriptor, path.name

    @classmethod
    def _open_child_directory(
        cls,
        parent_fd: int,
        name: str,
        *,
        missing_ok: bool = False,
    ) -> int | None:
        descriptor: int | None = None
        observed = False
        try:
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            observed = True
            descriptor = os.open(
                name,
                cls._directory_flags(),
                dir_fd=parent_fd,
            )
            opened = os.fstat(descriptor)
            after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if descriptor is not None:
                os.close(descriptor)
            if missing_ok and not observed:
                return None
            raise PhaseBError("phase_b_journal_unavailable") from None
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            raise PhaseBError("phase_b_journal_unavailable") from exc
        assert descriptor is not None

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_uid,
                value.st_gid,
            )

        if (
            identity(before) != identity(opened)
            or identity(opened) != identity(after)
            or not cls._trusted_directory_status(opened)
        ):
            os.close(descriptor)
            raise PhaseBError("phase_b_journal_directory_untrusted")
        return descriptor

    @classmethod
    def _ensure_child_directory(cls, parent_fd: int, name: str) -> int:
        created = False
        try:
            os.mkdir(name, _JOURNAL_MODE, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise PhaseBError("phase_b_journal_unavailable") from exc
        if created:
            try:
                descriptor = os.open(
                    name,
                    cls._directory_flags(),
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                raise PhaseBError("phase_b_journal_unavailable") from exc
        else:
            opened = cls._open_child_directory(parent_fd, name)
            assert opened is not None
            descriptor = opened
        try:
            status = os.fstat(descriptor)
            if (
                created
                and status.st_uid == _effective_uid()
                and status.st_gid != _effective_gid()
            ):
                os.fchown(descriptor, -1, _effective_gid())
                status = os.fstat(descriptor)
            if not cls._trusted_directory_status(status):
                raise PhaseBError("phase_b_journal_directory_untrusted")
            # Always retry both durability barriers.  If a previous mkdir
            # succeeded but its parent fsync failed, seeing FileExistsError on
            # retry must not skip the missing durability proof.
            os.fsync(descriptor)
            os.fsync(parent_fd)
        except PhaseBError:
            os.close(descriptor)
            raise
        except OSError as exc:
            os.close(descriptor)
            raise PhaseBError("phase_b_journal_unavailable") from exc
        return descriptor

    def _open_root_handle(self, *, create: bool) -> int | None:
        parent_fd, name = self._open_absolute_parent(self.root)
        try:
            if create:
                return self._ensure_child_directory(parent_fd, name)
            return self._open_child_directory(parent_fd, name, missing_ok=True)
        finally:
            os.close(parent_fd)

    @staticmethod
    def _open_lock_at(plan_fd: int, *, create: bool) -> int:
        flags = (
            (os.O_RDWR if create else os.O_RDONLY)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        if create:
            flags |= os.O_CREAT
        existed = True
        try:
            os.stat(".lock", dir_fd=plan_fd, follow_symlinks=False)
        except FileNotFoundError:
            existed = False
        try:
            descriptor = os.open(
                ".lock",
                flags,
                _JOURNAL_FILE_MODE,
                dir_fd=plan_fd,
            )
            status = os.fstat(descriptor)
            if (
                create
                and not existed
                and status.st_uid == _effective_uid()
                and status.st_gid != _effective_gid()
            ):
                os.fchown(descriptor, -1, _effective_gid())
                status = os.fstat(descriptor)
            after = os.stat(".lock", dir_fd=plan_fd, follow_symlinks=False)
        except OSError as exc:
            try:
                os.close(descriptor)
            except (OSError, UnboundLocalError):
                pass
            raise PhaseBError("phase_b_journal_lock_untrusted") from exc
        if (
            (status.st_dev, status.st_ino)
            != (after.st_dev, after.st_ino)
            or not stat.S_ISREG(status.st_mode)
            or stat.S_IMODE(status.st_mode) != _JOURNAL_FILE_MODE
            or status.st_nlink != 1
            or status.st_uid != _effective_uid()
            or status.st_gid != _effective_gid()
        ):
            os.close(descriptor)
            raise PhaseBError("phase_b_journal_lock_untrusted")
        if create:
            try:
                os.fsync(descriptor)
                os.fsync(plan_fd)
            except OSError as exc:
                os.close(descriptor)
                raise PhaseBError("phase_b_journal_lock_untrusted") from exc
        return descriptor

    @staticmethod
    def _read_canonical_entry_at(
        directory_fd: int,
        name: str,
        *,
        expected_link_count: int,
    ) -> Mapping[str, Any]:
        try:
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
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
                payload = bytes(chunks)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise PhaseBError("phase_b_journal_read_failed") from exc

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_uid,
                value.st_gid,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
                value.st_nlink,
            )

        if (
            identity(before) != identity(opened)
            or identity(opened) != identity(after)
            or not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != _JOURNAL_FILE_MODE
            or opened.st_nlink != expected_link_count
            or opened.st_uid != _effective_uid()
            or opened.st_gid != _effective_gid()
            or not payload
            or len(payload) > _MAX_JSON_BYTES
        ):
            raise PhaseBError("phase_b_journal_file_untrusted")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PhaseBError("phase_b_journal_json_invalid") from exc
        if not isinstance(decoded, Mapping) or _canonical_bytes(decoded) != payload:
            raise PhaseBError("phase_b_journal_not_canonical")
        return _json_copy(decoded)

    def _load_entries_from_fd(
        self,
        plan: PhaseBPlan,
        entries_fd: int,
        *,
        linked_sequence: int | None = None,
    ) -> list[PhaseBJournalEntry]:
        try:
            names = sorted(os.listdir(entries_fd))
        except OSError as exc:
            raise PhaseBError("phase_b_journal_unavailable") from exc
        if any(re.fullmatch(r"[0-9]{8}\.json", name) is None for name in names):
            raise PhaseBError("phase_b_journal_path_invalid")
        entries: list[PhaseBJournalEntry] = []
        previous: str | None = None
        seen_keys: set[str] = set()
        for sequence, name in enumerate(names):
            if name != f"{sequence:08d}.json":
                raise PhaseBError("phase_b_journal_sequence_invalid")
            decoded = self._read_canonical_entry_at(
                entries_fd,
                name,
                expected_link_count=(
                    2 if linked_sequence == sequence else 1
                ),
            )
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
            seen_keys.add(key)
            entries.append(entry)
            previous = entry.sha256
        self._validate_event_prerequisites(entries)
        return entries

    def _recover_staging_locked(self, plan: PhaseBPlan, plan_fd: int) -> None:
        if self._lock_depth != 1:
            raise PhaseBError("phase_b_journal_lock_required")
        staging_fd = self._open_child_directory(
            plan_fd,
            "staging",
            missing_ok=True,
        )
        if staging_fd is None:
            return
        entries_fd = self._open_child_directory(
            plan_fd,
            "entries",
            missing_ok=True,
        )
        try:
            stage_names = sorted(os.listdir(staging_fd))
            if not stage_names:
                return
            if len(stage_names) != 1 or entries_fd is None:
                raise PhaseBError("phase_b_journal_staging_conflict")
            stage_name = stage_names[0]
            match = re.fullmatch(
                r"([0-9]{8})\.([0-9a-f]{32})\.stage",
                stage_name,
            )
            if match is None:
                raise PhaseBError("phase_b_journal_staging_conflict")
            sequence = int(match.group(1))
            try:
                descriptor = os.open(
                    stage_name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=staging_fd,
                )
                stage_status = os.fstat(descriptor)
            except OSError as exc:
                raise PhaseBError("phase_b_journal_staging_conflict") from exc
            finally:
                try:
                    os.close(descriptor)
                except (OSError, UnboundLocalError):
                    pass
            if (
                not stat.S_ISREG(stage_status.st_mode)
                or stat.S_IMODE(stage_status.st_mode) != _JOURNAL_FILE_MODE
                or stage_status.st_uid != _effective_uid()
                or stage_status.st_gid != _effective_gid()
                or stage_status.st_nlink not in {1, 2}
            ):
                raise PhaseBError("phase_b_journal_staging_conflict")
            final_names = sorted(os.listdir(entries_fd))
            same_final = [
                name for name in final_names if name == f"{sequence:08d}.json"
            ]
            if stage_status.st_nlink == 1:
                if same_final or sequence != len(final_names):
                    raise PhaseBError("phase_b_journal_staging_conflict")
                os.unlink(stage_name, dir_fd=staging_fd)
                os.fsync(staging_fd)
                return
            if (
                same_final != [f"{sequence:08d}.json"]
                or sequence != len(final_names) - 1
            ):
                raise PhaseBError("phase_b_journal_staging_conflict")
            final_status = os.stat(
                same_final[0],
                dir_fd=entries_fd,
                follow_symlinks=False,
            )
            if (
                final_status.st_nlink != 2
                or (final_status.st_dev, final_status.st_ino)
                != (stage_status.st_dev, stage_status.st_ino)
            ):
                raise PhaseBError("phase_b_journal_staging_conflict")
            entries = self._load_entries_from_fd(
                plan,
                entries_fd,
                linked_sequence=sequence,
            )
            staged = self._read_canonical_entry_at(
                staging_fd,
                stage_name,
                expected_link_count=2,
            )
            if not entries or staged != entries[-1].value:
                raise PhaseBError("phase_b_journal_staging_conflict")
            # Retry the publication durability barrier before removing the
            # only evidence that distinguishes a linked crash window.
            os.fsync(entries_fd)
            os.unlink(stage_name, dir_fd=staging_fd)
            os.fsync(staging_fd)
            os.fsync(entries_fd)
        except OSError as exc:
            raise PhaseBError("phase_b_journal_staging_conflict") from exc
        finally:
            if entries_fd is not None:
                os.close(entries_fd)
            os.close(staging_fd)

    def _load_from_plan_fd(
        self,
        plan: PhaseBPlan,
        plan_fd: int,
    ) -> list[PhaseBJournalEntry]:
        try:
            names = set(os.listdir(plan_fd))
        except OSError as exc:
            raise PhaseBError("phase_b_journal_unavailable") from exc
        if not names.issubset({".lock", "entries", "staging"}):
            raise PhaseBError("phase_b_journal_path_invalid")
        staging_fd = self._open_child_directory(
            plan_fd,
            "staging",
            missing_ok=True,
        )
        if staging_fd is not None:
            try:
                if os.listdir(staging_fd):
                    raise PhaseBError("phase_b_journal_staging_residue")
            finally:
                os.close(staging_fd)
        entries_fd = self._open_child_directory(
            plan_fd,
            "entries",
            missing_ok=True,
        )
        if entries_fd is None:
            return []
        try:
            return self._load_entries_from_fd(plan, entries_fd)
        finally:
            os.close(entries_fd)

    @contextmanager
    def lock(self, plan: PhaseBPlan) -> Iterator[None]:
        if fcntl is None:
            raise PhaseBError("phase_b_posix_lock_unavailable")
        if self._lock_depth != 0:
            raise PhaseBError("phase_b_journal_lock_reentrant")
        root_fd = self._open_root_handle(create=True)
        assert root_fd is not None
        plan_fd: int | None = None
        descriptor: int | None = None
        try:
            plan_fd = self._ensure_child_directory(root_fd, plan.sha256)
            descriptor = self._open_lock_at(plan_fd, create=True)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._lock_depth = 1
            self._active_plan_sha256 = plan.sha256
            self._active_root_fd = root_fd
            self._active_plan_fd = plan_fd
            self._recover_staging_locked(plan, plan_fd)
            yield
        finally:
            try:
                self._lock_depth = 0
                self._active_plan_sha256 = None
                self._active_root_fd = None
                self._active_plan_fd = None
                if descriptor is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                if plan_fd is not None:
                    os.close(plan_fd)
                os.close(root_fd)

    def load(self, plan: PhaseBPlan) -> list[PhaseBJournalEntry]:
        if (
            self._lock_depth == 1
            and self._active_plan_sha256 == plan.sha256
            and self._active_plan_fd is not None
        ):
            return self._load_from_plan_fd(plan, self._active_plan_fd)
        if fcntl is None:
            raise PhaseBError("phase_b_posix_lock_unavailable")
        root_fd = self._open_root_handle(create=False)
        if root_fd is None:
            return []
        plan_fd = self._open_child_directory(
            root_fd,
            plan.sha256,
            missing_ok=True,
        )
        if plan_fd is None:
            os.close(root_fd)
            return []
        descriptor: int | None = None
        try:
            descriptor = self._open_lock_at(plan_fd, create=False)
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            return self._load_from_plan_fd(plan, plan_fd)
        finally:
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
            os.close(plan_fd)
            os.close(root_fd)

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
        approval: PhaseBApproval,
        event: str,
        idempotency_key: str,
        evidence: Mapping[str, Any],
        now_unix: int | None = None,
    ) -> PhaseBJournalEntry:
        if event not in JOURNAL_EVENTS:
            raise PhaseBError("phase_b_journal_event_invalid")
        if not isinstance(approval, PhaseBApproval):
            raise PhaseBError("phase_b_journal_approval_invalid")
        if self._lock_depth != 1:
            raise PhaseBError("phase_b_journal_lock_required")
        if (
            self._active_plan_sha256 != plan.sha256
            or self._active_plan_fd is None
        ):
            raise PhaseBError("phase_b_journal_lock_required")
        _require_secret_free(evidence)
        recorded_at = int(time.time()) if now_unix is None else now_unix
        PhaseBApproval.from_mapping(
            approval.to_mapping(),
            plan=plan,
            now_unix=recorded_at,
        )
        plan_fd = self._active_plan_fd
        self._recover_staging_locked(plan, plan_fd)
        entries_fd = self._ensure_child_directory(plan_fd, "entries")
        try:
            staging_fd = self._ensure_child_directory(plan_fd, "staging")
        except BaseException:
            os.close(entries_fd)
            raise
        try:
            entries = self._load_entries_from_fd(plan, entries_fd)
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
                "approval_sha256": approval.sha256,
                "event": event,
                "idempotency_key": idempotency_key,
                "previous_entry_sha256": (
                    entries[-1].sha256 if entries else None
                ),
                "evidence": _json_copy(evidence),
                "preterminal": not terminal,
                "safe_to_start": terminal,
                "recorded_at_unix": (
                    recorded_at
                ),
            }
            entry = PhaseBJournalEntry.from_mapping(
                {**unsigned, "entry_sha256": _sha256_json(unsigned)},
                plan=plan,
                expected_sequence=len(entries),
                expected_previous=(entries[-1].sha256 if entries else None),
            )
            self._validate_event_prerequisites([*entries, entry])
            payload = _canonical_bytes(entry.value)
            stage_name = (
                f"{len(entries):08d}.{secrets.token_hex(16)}.stage"
            )
            final_name = f"{len(entries):08d}.json"
            descriptor = os.open(
                stage_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                _JOURNAL_FILE_MODE,
                dir_fd=staging_fd,
            )
            try:
                try:
                    status = os.fstat(descriptor)
                    if (
                        status.st_uid == _effective_uid()
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
                        raise PhaseBError("phase_b_journal_file_untrusted")
                    written = 0
                    while written < len(payload):
                        count = os.write(descriptor, payload[written:])
                        if count <= 0:
                            raise PhaseBError("phase_b_journal_write_failed")
                        written += count
                    os.fsync(descriptor)
                except OSError as exc:
                    raise PhaseBError("phase_b_journal_write_failed") from exc
            finally:
                os.close(descriptor)
            try:
                # Persist the recovery marker before publishing a hard link.
                # After this barrier a crash leaves either a discardable
                # one-link stage or an authenticated two-link publication.
                os.fsync(staging_fd)
                os.link(
                    stage_name,
                    final_name,
                    src_dir_fd=staging_fd,
                    dst_dir_fd=entries_fd,
                    follow_symlinks=False,
                )
                # If this fails, preserve both links.  Recovery retries this
                # exact durability barrier before it removes the stage.
                os.fsync(entries_fd)
            except FileExistsError as exc:
                raise PhaseBError("phase_b_journal_append_collision") from exc
            except OSError as exc:
                raise PhaseBError("phase_b_journal_publish_failed") from exc
            try:
                os.unlink(stage_name, dir_fd=staging_fd)
                os.fsync(staging_fd)
                os.fsync(entries_fd)
            except OSError as exc:
                raise PhaseBError("phase_b_journal_publish_failed") from exc
            loaded = self._load_entries_from_fd(plan, entries_fd)
            if not loaded or loaded[-1].sha256 != entry.sha256:
                raise PhaseBError("phase_b_journal_readback_failed")
            return loaded[-1]
        finally:
            os.close(staging_fd)
            os.close(entries_fd)

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

    def create_or_rotate_recovery(self, username: str) -> bytearray: ...

    def mutation_reconciliation_required(self) -> bool: ...

    def require_current_authority(self, username: str) -> None: ...

    def temporary_admin_authority_receipt(self, username: str) -> Mapping[str, Any]: ...

    def delete_and_confirm_absent(self, username: str) -> None: ...

    def reconciliation_receipt(self) -> Mapping[str, Any]: ...


class BootstrapLoginBoundary(Protocol):
    def describe(self) -> Mapping[str, Any] | None: ...

    def create_or_rotate_recovery(self) -> bytearray: ...

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
    approval: PhaseBApproval,
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
    recorded_at = int(clock())
    if services["observed_at_unix"] > recorded_at:
        raise PhaseBError("phase_b_services_observed_in_future")
    service_attempt = sum(
        1
        for entry in journal.load(plan)
        if entry.event == "services_stopped"
        and entry.evidence.get("transition") == transition
    )
    journal.append(
        plan,
        approval=approval,
        event="services_stopped",
        idempotency_key=(
            f"services:{transition}:{service_attempt:08d}:"
            f"{services['attestation_sha256']}"
        ),
        evidence={
            "transition": transition,
            "services_attestation": services,
            "preterminal": True,
            "safe_to_start": False,
        },
        now_unix=recorded_at,
    )
    return services


def _pending_service_transition(
    journal: AppendOnlyPhaseBJournal,
    plan: PhaseBPlan,
) -> str | None:
    entries = journal.load(plan)
    if not entries or entries[-1].event != "services_stopped":
        return None
    transition = entries[-1].evidence.get("transition")
    if not isinstance(transition, str) or not transition:
        raise PhaseBError("phase_b_journal_services_invalid")
    return transition


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
    *,
    before_mutation: Callable[[], None],
) -> tuple[Mapping[str, Any], bytearray]:
    boundary.begin_mutation_observation(
        expected_owner_subject_sha256=plan.owner_subject_sha256,
        expected_mutation_context_sha256=plan.sha256,
    )
    candidate: Any = None
    try:
        before_mutation()
        candidate = boundary.create_or_rotate_recovery(
            plan.temporary_admin_username
        )
    except Exception:
        if not boundary.mutation_reconciliation_required():
            raise
        before_mutation()
        candidate = boundary.create_or_rotate_recovery(
            plan.temporary_admin_username
        )
    try:
        secret = _require_secret(candidate)
        boundary.require_current_authority(plan.temporary_admin_username)
        authority = _validate_temporary_admin_authority(
            boundary.temporary_admin_authority_receipt(
                plan.temporary_admin_username
            ),
            plan=plan,
        )
    except BaseException:
        if isinstance(candidate, bytearray):
            _zeroize(candidate)
        raise
    return authority, secret


def _provision_bootstrap_login(
    boundary: BootstrapLoginBoundary,
    plan: PhaseBPlan,
    *,
    before_mutation: Callable[[], None],
) -> tuple[Mapping[str, Any], bytearray]:
    candidate: Any = None
    try:
        before_mutation()
        candidate = boundary.create_or_rotate_recovery()
    except Exception:
        if not boundary.mutation_reconciliation_required():
            raise
        before_mutation()
        candidate = boundary.create_or_rotate_recovery()
    try:
        secret = _require_secret(candidate)
        boundary.require_current_authority()
        authority = _validate_bootstrap_authority(
            boundary.authority_receipt(),
            plan=plan,
        )
    except BaseException:
        if isinstance(candidate, bytearray):
            _zeroize(candidate)
        raise
    return authority, secret


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
        or terminal_at >= approval.value["expires_at_unix"]
        or terminal_observation["observed_at_unix"] > terminal_at
        or terminal_observation["cloud_sql"]["observed_at_unix"] > terminal_at
        or services["observed_at_unix"] > terminal_at
        or hba_receipt["observed_at_unix"] > terminal_at
        or terminal_at >= hba_receipt["expires_at_unix"]
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
        expected_bootstrap_authority=post_disable_authority,
        expected_absence_receipt=absence_receipt,
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


def _validate_terminal_journal_chain(
    entries: Sequence[PhaseBJournalEntry],
    *,
    plan: PhaseBPlan,
    approvals: Sequence[PhaseBApproval],
) -> tuple[dict[str, Mapping[str, Any]], PhaseBApproval]:
    """Replay every durable evidence envelope without invoking a boundary."""

    if not entries or entries[-1].event != "terminal":
        raise PhaseBError("phase_b_durable_terminal_missing")
    if any(
        entries[index].value["recorded_at_unix"]
        < entries[index - 1].value["recorded_at_unix"]
        for index in range(1, len(entries))
    ):
        raise PhaseBError("phase_b_journal_time_regressed")
    if not approvals:
        raise PhaseBError("phase_b_approval_chain_invalid")

    bindings = _terminal_replay_bindings(entries, plan=plan)
    initial_services = _validate_services(
        plan.preflight.value["services"],
        release_revision=plan.revision,
    )
    bootstrap_authorities: list[tuple[int, Mapping[str, Any]]] = []
    bootstrap_operation_names: set[str] = set()
    bootstrap_receipt_sha256s: set[str] = set()
    bootstrap_disable_index: int | None = None
    latest_hba_authority_sha256: str | None = None
    initial_admin_authorities: list[Mapping[str, Any]] = []
    deletion_admin_authorities: list[Mapping[str, Any]] = []
    admin_operation_names: set[str] = set()
    terminal_observations: list[Mapping[str, Any]] = []
    pending_service_transition: str | None = None
    pending_service_observed_at: int | None = None
    role_ready_seen = False
    temporary_admin_absent_seen = False
    transition_required = {
        "role_ready": "phase_b_role_artifact",
        "bootstrap_password_disabled": "bootstrap_password_self_disable",
        "temporary_admin_absent": "temporary_admin_delete",
        "terminal_observed": "terminal_observation",
    }
    known_service_transitions = frozenset(
        {
            *transition_required.values(),
            "temporary_admin_initial_authority",
            "temporary_admin_predelete_reacquire",
            "bootstrap_login_authority",
            "bootstrap_login_rotation_authority",
        }
    )

    for entry_index, entry in enumerate(entries):
        event = entry.event
        evidence = entry.evidence
        active_approval = _active_approval_for_time(
            approvals,
            recorded_at_unix=int(entry.value["recorded_at_unix"]),
        )
        if entry.value["approval_sha256"] != active_approval.sha256:
            raise PhaseBError("phase_b_journal_approval_head_mismatch")
        required_transition = transition_required.get(event)
        consumed_service_transition: str | None = None
        if event != "services_stopped":
            if event == "bootstrap_authority":
                allowed_bootstrap_transitions = {
                    "bootstrap_login_authority",
                    "bootstrap_login_rotation_authority",
                }
                if (
                    bootstrap_authorities
                    and latest_hba_authority_sha256
                    == bootstrap_authorities[-1][1]["receipt_sha256"]
                ):
                    allowed_bootstrap_transitions.add(
                        "bootstrap_password_self_disable"
                    )
                if pending_service_transition not in allowed_bootstrap_transitions:
                    raise PhaseBError("phase_b_journal_service_boundary_missing")
                consumed_service_transition = pending_service_transition
                pending_service_transition = None
                pending_service_observed_at = None
            elif event == "temporary_admin_authority":
                allowed_admin_transitions = {
                    "temporary_admin_initial_authority"
                }
                if initial_admin_authorities and not role_ready_seen:
                    allowed_admin_transitions.add("phase_b_role_artifact")
                if pending_service_transition not in allowed_admin_transitions:
                    raise PhaseBError("phase_b_journal_service_boundary_missing")
                consumed_service_transition = pending_service_transition
                pending_service_transition = None
                pending_service_observed_at = None
            elif event == "temporary_admin_predelete_authority":
                allowed_delete_transitions = {
                    "temporary_admin_predelete_reacquire"
                }
                if deletion_admin_authorities and not temporary_admin_absent_seen:
                    allowed_delete_transitions.add("temporary_admin_delete")
                if pending_service_transition not in allowed_delete_transitions:
                    raise PhaseBError("phase_b_journal_service_boundary_missing")
                consumed_service_transition = pending_service_transition
                pending_service_transition = None
                pending_service_observed_at = None
            elif required_transition is not None:
                if pending_service_transition != required_transition:
                    raise PhaseBError("phase_b_journal_service_boundary_missing")
                consumed_service_transition = pending_service_transition
                pending_service_transition = None
                pending_service_observed_at = None
            elif pending_service_transition is not None:
                raise PhaseBError("phase_b_journal_service_boundary_stale")

        if event == "intent":
            raw = _strict_mapping(
                evidence,
                frozenset(
                    {
                        "approval_sha256",
                        "plan_sha256",
                        "preterminal",
                        "safe_to_start",
                        "secret_material_recorded",
                    }
                ),
                "phase_b_journal_intent_invalid",
            )
            if (
                raw["approval_sha256"] != active_approval.sha256
                or raw["plan_sha256"] != plan.sha256
                or raw["preterminal"] is not True
                or raw["safe_to_start"] is not False
                or raw["secret_material_recorded"] is not False
                or not active_approval.value["issued_at_unix"]
                <= entry.value["recorded_at_unix"]
                < active_approval.value["expires_at_unix"]
            ):
                raise PhaseBError("phase_b_journal_intent_invalid")
        elif event == "services_stopped":
            raw = _strict_mapping(
                evidence,
                frozenset(
                    {
                        "transition",
                        "services_attestation",
                        "preterminal",
                        "safe_to_start",
                    }
                ),
                "phase_b_journal_services_invalid",
            )
            services = _validate_services(
                raw["services_attestation"],
                release_revision=plan.revision,
            )
            if (
                not isinstance(raw["transition"], str)
                or not raw["transition"]
                or len(raw["transition"]) > 96
                or raw["transition"] not in known_service_transitions
                or raw["preterminal"] is not True
                or raw["safe_to_start"] is not False
                or _service_stable_projection(services)
                != _service_stable_projection(initial_services)
                or services["observed_at_unix"]
                > entry.value["recorded_at_unix"]
            ):
                raise PhaseBError("phase_b_journal_services_invalid")
            # A crash retry may repeat the exact pending transition with fresh,
            # non-regressing observation evidence.  A different transition
            # cannot erase an unconsumed boundary: that would let an inserted
            # journal row manufacture causality for the following mutation.
            if (
                pending_service_transition is not None
                and (
                    raw["transition"] != pending_service_transition
                    or (
                        pending_service_observed_at is not None
                        and services["observed_at_unix"]
                        < pending_service_observed_at
                    )
                )
            ):
                raise PhaseBError("phase_b_journal_service_boundary_stale")
            pending_service_transition = raw["transition"]
            pending_service_observed_at = services["observed_at_unix"]
        elif event in {
            "temporary_admin_authority",
            "temporary_admin_predelete_authority",
        }:
            raw = _strict_mapping(
                evidence,
                frozenset(
                    {
                        "purpose",
                        "authority_receipt",
                        "preterminal",
                        "safe_to_start",
                    }
                ),
                "phase_b_journal_admin_authority_invalid",
            )
            expected_purpose = (
                "foundation_sql"
                if event == "temporary_admin_authority"
                else "fresh_predelete_authority"
            )
            authority = _validate_temporary_admin_authority(
                raw["authority_receipt"],
                plan=plan,
            )
            if (
                (
                    event == "temporary_admin_predelete_authority"
                    and temporary_admin_absent_seen
                )
                or raw["purpose"] != expected_purpose
                or raw["preterminal"] is not True
                or raw["safe_to_start"] is not False
            ):
                raise PhaseBError("phase_b_journal_admin_authority_invalid")
            operation_name = str(authority["authority_operation"][0])
            if operation_name in admin_operation_names:
                raise PhaseBError("phase_b_journal_admin_authority_reused")
            admin_operation_names.add(operation_name)
            target = (
                initial_admin_authorities
                if event == "temporary_admin_authority"
                else deletion_admin_authorities
            )
            target.append(authority)
        elif event == "role_ready":
            if role_ready_seen:
                raise PhaseBError("phase_b_journal_role_repeated")
            raw = _strict_mapping(
                evidence,
                frozenset({"role_receipt", "preterminal", "safe_to_start"}),
                "phase_b_journal_role_invalid",
            )
            role = _validate_role_receipt(raw["role_receipt"], plan=plan)
            if (
                role != bindings["role_receipt"]
                or raw["preterminal"] is not True
                or raw["safe_to_start"] is not False
            ):
                raise PhaseBError("phase_b_journal_role_invalid")
            role_ready_seen = True
        elif event == "bootstrap_authority":
            raw = _strict_mapping(
                evidence,
                frozenset(
                    {"authority_receipt", "preterminal", "safe_to_start"}
                ),
                "phase_b_journal_bootstrap_authority_invalid",
            )
            authority = _validate_bootstrap_authority(
                raw["authority_receipt"],
                plan=plan,
            )
            if bootstrap_disable_index is not None:
                raise PhaseBError("phase_b_journal_bootstrap_authority_after_disable")
            operation_name = str(authority["operation_name"])
            receipt_sha256 = str(authority["receipt_sha256"])
            if (
                operation_name in bootstrap_operation_names
                or receipt_sha256 in bootstrap_receipt_sha256s
            ):
                raise PhaseBError("phase_b_journal_bootstrap_authority_reused")
            bootstrap_operation_names.add(operation_name)
            bootstrap_receipt_sha256s.add(receipt_sha256)
            valid_create = (
                authority["operation_type"] == "CREATE_USER"
                and consumed_service_transition == "bootstrap_login_authority"
                and not bootstrap_authorities
            )
            valid_update = (
                authority["operation_type"] == "UPDATE_USER"
                and (
                    (
                        consumed_service_transition
                        in {
                            "bootstrap_login_authority",
                            "bootstrap_login_rotation_authority",
                        }
                        and (
                            not bootstrap_authorities
                            or consumed_service_transition
                            == "bootstrap_login_rotation_authority"
                        )
                    )
                    or (
                        consumed_service_transition
                        == "bootstrap_password_self_disable"
                        and bootstrap_authorities
                        and latest_hba_authority_sha256
                        == bootstrap_authorities[-1][1]["receipt_sha256"]
                    )
                )
            )
            if not (valid_create or valid_update):
                raise PhaseBError(
                    "phase_b_journal_bootstrap_authority_transition_invalid"
                )
            bootstrap_authorities.append((entry_index, authority))
            latest_hba_authority_sha256 = None
            if raw["preterminal"] is not True or raw["safe_to_start"] is not False:
                raise PhaseBError("phase_b_journal_bootstrap_authority_invalid")
        elif event == "bootstrap_hba_rejected":
            if bootstrap_disable_index is not None:
                raise PhaseBError("phase_b_journal_hba_after_disable")
            raw = _strict_mapping(
                evidence,
                frozenset({"hba_receipt", "preterminal", "safe_to_start"}),
                "phase_b_journal_hba_invalid",
            )
            hba_value = _strict_mapping(
                raw["hba_receipt"],
                _HBA_FIELDS,
                "phase_b_journal_hba_invalid",
            )
            if not bootstrap_authorities:
                raise PhaseBError("phase_b_journal_hba_invalid")
            authority = bootstrap_authorities[-1][1]
            if (
                hba_value["bootstrap_authority_receipt_sha256"]
                != authority["receipt_sha256"]
            ):
                raise PhaseBError("phase_b_journal_hba_invalid")
            _validate_hba_receipt(raw["hba_receipt"], plan=plan, authority=authority)
            if (
                raw["hba_receipt"]["observed_at_unix"]
                > entry.value["recorded_at_unix"]
                or raw["preterminal"] is not True
                or raw["safe_to_start"] is not False
            ):
                raise PhaseBError("phase_b_journal_hba_invalid")
            latest_hba_authority_sha256 = str(authority["receipt_sha256"])
        elif event == "bootstrap_password_disabled":
            raw = _strict_mapping(
                evidence,
                frozenset(
                    {
                        "initial_authority_receipt",
                        "post_disable_authority_receipt",
                        "hba_receipt",
                        "self_disable_receipt",
                        "preterminal",
                        "safe_to_start",
                    }
                ),
                "phase_b_journal_bootstrap_disable_invalid",
            )
            authority = _validate_bootstrap_authority(
                raw["initial_authority_receipt"],
                plan=plan,
            )
            post_authority = _validate_bootstrap_authority(
                raw["post_disable_authority_receipt"],
                plan=plan,
            )
            hba = _validate_hba_receipt(
                raw["hba_receipt"],
                plan=plan,
                authority=authority,
            )
            disabled = _validate_self_disable_receipt(
                raw["self_disable_receipt"],
                plan=plan,
                authority=authority,
                hba=hba,
            )
            if (
                not bootstrap_authorities
                or authority != bootstrap_authorities[-1][1]
                or authority != post_authority
                or authority != bindings["bootstrap_authority"]
                or hba != bindings["hba_receipt"]
                or disabled != bindings["self_disable_receipt"]
                or hba["observed_at_unix"] > entry.value["recorded_at_unix"]
                or raw["preterminal"] is not True
                or raw["safe_to_start"] is not False
            ):
                raise PhaseBError("phase_b_journal_bootstrap_disable_invalid")
            if bootstrap_disable_index is not None:
                raise PhaseBError("phase_b_journal_bootstrap_disable_repeated")
            bootstrap_disable_index = entry_index
        elif event == "predelete_verified":
            raw = _strict_mapping(
                evidence,
                frozenset(
                    {
                        "predelete_receipt",
                        "process_instance_sha256",
                        "preterminal",
                        "safe_to_start",
                    }
                ),
                "phase_b_journal_predelete_invalid",
            )
            _digest(
                raw["process_instance_sha256"],
                "phase_b_journal_predelete_invalid",
            )
            if (
                raw["predelete_receipt"] != bindings["predelete_receipt"]
                or raw["preterminal"] is not True
                or raw["safe_to_start"] is not False
            ):
                raise PhaseBError("phase_b_journal_predelete_invalid")
        elif event == "temporary_admin_closed":
            common = {
                "predelete_receipt_sha256",
                "admin_session_closed",
                "provisional_secret_zeroized",
                "preterminal",
                "safe_to_start",
                "secret_material_recorded",
            }
            direct = frozenset({*common, "process_instance_sha256"})
            recovered = frozenset(
                {
                    *common,
                    "process_recovery_boundary",
                    "predelete_process_instance_sha256",
                    "recovery_process_instance_sha256",
                }
            )
            if not isinstance(evidence, Mapping) or set(evidence) not in {
                direct,
                recovered,
            }:
                raise PhaseBError("phase_b_journal_admin_close_invalid")
            raw = copy.deepcopy(dict(evidence))
            if set(raw) == direct:
                _digest(
                    raw["process_instance_sha256"],
                    "phase_b_journal_admin_close_invalid",
                )
            else:
                before = _digest(
                    raw["predelete_process_instance_sha256"],
                    "phase_b_journal_admin_close_invalid",
                )
                after = _digest(
                    raw["recovery_process_instance_sha256"],
                    "phase_b_journal_admin_close_invalid",
                )
                if raw["process_recovery_boundary"] is not True or before == after:
                    raise PhaseBError("phase_b_journal_admin_close_invalid")
            if (
                raw["predelete_receipt_sha256"]
                != bindings["predelete_receipt"]["receipt_sha256"]
                or raw["admin_session_closed"] is not True
                or raw["provisional_secret_zeroized"] is not True
                or raw["preterminal"] is not True
                or raw["safe_to_start"] is not False
                or raw["secret_material_recorded"] is not False
            ):
                raise PhaseBError("phase_b_journal_admin_close_invalid")
        elif event == "temporary_admin_absent":
            if temporary_admin_absent_seen:
                raise PhaseBError("phase_b_journal_admin_absence_repeated")
            raw = _strict_mapping(
                evidence,
                frozenset(
                    {
                        "fresh_predelete_authority_receipt",
                        "full_absence_receipt",
                        "preterminal",
                        "safe_to_start",
                    }
                ),
                "phase_b_journal_admin_absence_invalid",
            )
            authority = _validate_temporary_admin_authority(
                raw["fresh_predelete_authority_receipt"],
                plan=plan,
            )
            absence = _validate_admin_absence_receipt(
                raw["full_absence_receipt"],
                plan=plan,
                fresh_authority=authority,
            )
            if (
                authority != bindings["deletion_admin_authority"]
                or absence != bindings["absence_receipt"]
                or raw["preterminal"] is not True
                or raw["safe_to_start"] is not False
            ):
                raise PhaseBError("phase_b_journal_admin_absence_invalid")
            temporary_admin_absent_seen = True
        elif event == "terminal_observed":
            raw = _strict_mapping(
                evidence,
                frozenset(
                    {"terminal_observation", "preterminal", "safe_to_start"}
                ),
                "phase_b_journal_terminal_observation_invalid",
            )
            terminal = _validate_terminal_observation(
                raw["terminal_observation"],
                plan=plan,
                execution_preflight_session_sha256=None,
                expected_bootstrap_authority=bindings["post_disable_authority"],
                expected_absence_receipt=bindings["absence_receipt"],
            )
            if (
                raw["preterminal"] is not True
                or raw["safe_to_start"] is not False
                or terminal["observed_at_unix"]
                > entry.value["recorded_at_unix"]
            ):
                raise PhaseBError("phase_b_journal_terminal_observation_invalid")
            terminal_observations.append(terminal)
        elif event == "terminal":
            terminal_evidence = _strict_mapping(
                evidence,
                frozenset({"terminal_receipt"}),
                "phase_b_terminal_receipt_invalid",
            )
            terminal_value = _strict_mapping(
                terminal_evidence["terminal_receipt"],
                _TERMINAL_RECEIPT_FIELDS,
                "phase_b_terminal_receipt_invalid",
            )
            if terminal_value["terminal_at_unix"] != entry.value["recorded_at_unix"]:
                raise PhaseBError("phase_b_terminal_receipt_invalid")
        else:  # pragma: no cover - PhaseBJournalEntry already rejects this.
            raise PhaseBError("phase_b_journal_event_invalid")

    if pending_service_transition is not None:
        raise PhaseBError("phase_b_journal_service_boundary_stale")
    if not bootstrap_authorities or bootstrap_disable_index is None:
        raise PhaseBError("phase_b_journal_bootstrap_authority_invalid")
    if bootstrap_authorities[-1][1]["operation_type"] != "UPDATE_USER":
        raise PhaseBError("phase_b_journal_bootstrap_authority_not_latest_update")
    if (
        not initial_admin_authorities
        or initial_admin_authorities[-1] != bindings["initial_admin_authority"]
        or not deletion_admin_authorities
        or deletion_admin_authorities[-1] != bindings["deletion_admin_authority"]
        or not terminal_observations
        or terminal_observations[-1] != bindings["terminal_observation"]
    ):
        raise PhaseBError("phase_b_journal_terminal_binding_invalid")
    terminal_approval = _active_approval_for_time(
        approvals,
        recorded_at_unix=int(entries[-1].value["recorded_at_unix"]),
    )
    if entries[-1].value["approval_sha256"] != terminal_approval.sha256:
        raise PhaseBError("phase_b_journal_approval_head_mismatch")
    return bindings, terminal_approval


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


_DURABLE_FOUNDATION_FIELDS = frozenset(
    {
        "schema",
        "foundation_release_revision",
        "plan_sha256",
        "approval_sha256",
        "approval_sequence",
        "approval_chain_sha256",
        "journal_entry_count",
        "terminal_entry_sequence",
        "terminal_entry_sha256",
        "terminal_receipt_sha256",
        "terminal_observation_sha256",
        "terminal_at_unix",
        "generation_sha256",
    }
)


class PhaseBDurableFoundation:
    """Validated immutable Phase-B generation, independent of later releases."""

    __slots__ = (
        "_plan_bytes",
        "_approval_bytes",
        "_generation_bytes",
        "_terminal_receipt_bytes",
        "_terminal_observation_bytes",
        "_authority_seal",
        "_authority_tag",
    )

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "PhaseBDurableFoundation":
        raise TypeError(
            "PhaseBDurableFoundation is issued only by "
            "load_durable_phase_b_foundation"
        )

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("PhaseBDurableFoundation is immutable")

    def _mapping(self, attribute: str) -> dict[str, Any]:
        _require_durable_foundation(self)
        encoded = getattr(self, attribute)
        try:
            decoded = json.loads(encoded.decode("utf-8"))
        except (AttributeError, UnicodeError, json.JSONDecodeError) as exc:
            raise PhaseBError("phase_b_durable_foundation_untrusted") from exc
        if not isinstance(decoded, dict) or _canonical_bytes(decoded) != encoded:
            raise PhaseBError("phase_b_durable_foundation_untrusted")
        return decoded

    @property
    def plan(self) -> PhaseBPlan:
        return PhaseBPlan.from_mapping(self._mapping("_plan_bytes"))

    @property
    def approval(self) -> Mapping[str, Any]:
        return self._mapping("_approval_bytes")

    @property
    def generation(self) -> Mapping[str, Any]:
        return self._mapping("_generation_bytes")

    @property
    def generation_sha256(self) -> str:
        return str(self.generation["generation_sha256"])

    @property
    def terminal_receipt(self) -> Mapping[str, Any]:
        return self._mapping("_terminal_receipt_bytes")

    @property
    def terminal_observation(self) -> Mapping[str, Any]:
        return self._mapping("_terminal_observation_bytes")

    @property
    def terminal_at_unix(self) -> int:
        return int(self.generation["terminal_at_unix"])

    @property
    def terminal_entry_sha256(self) -> str:
        return str(self.generation["terminal_entry_sha256"])

    def to_mapping(self) -> dict[str, Any]:
        return {
            "plan": self._mapping("_plan_bytes"),
            "approval": self._mapping("_approval_bytes"),
            "generation": self._mapping("_generation_bytes"),
            "terminal_receipt": self._mapping("_terminal_receipt_bytes"),
            "terminal_observation": self._mapping(
                "_terminal_observation_bytes"
            ),
        }


def _approval_chain_values(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [_json_copy(value)]
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray, memoryview))
        or not value
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise PhaseBError("phase_b_approval_chain_invalid")
    return [_json_copy(item) for item in value]


def _load_durable_phase_b_components(
    plan: PhaseBPlan,
    *,
    approval: Any,
    journal: AppendOnlyPhaseBJournal,
) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    """Read and semantically replay one completed Phase-B generation.

    This loader never invokes a collector, transport, or mutation boundary.
    The historical approval is evaluated at the authenticated terminal time,
    so normal present-day expiry cannot erase a completed generation.
    """

    _require_foundation_runtime()
    if not isinstance(plan, PhaseBPlan):
        raise TypeError("PhaseBPlan is required")
    if not isinstance(journal, AppendOnlyPhaseBJournal):
        raise PhaseBError("phase_b_journal_missing")
    validated_plan = PhaseBPlan.from_mapping(plan.to_mapping())
    entries = journal.load(validated_plan)
    if not entries or entries[-1].event != "terminal":
        raise PhaseBError("phase_b_durable_terminal_missing")

    terminal_evidence = _strict_mapping(
        entries[-1].evidence,
        frozenset({"terminal_receipt"}),
        "phase_b_terminal_receipt_invalid",
    )
    approval_values = _approval_chain_values(approval)
    approvals = validate_phase_b_approval_chain(
        approval_values,
        plan=validated_plan,
    )
    bindings, approved = _validate_terminal_journal_chain(
        entries,
        plan=validated_plan,
        approvals=approvals,
    )
    terminal_receipt = _validate_terminal_receipt(
        terminal_evidence["terminal_receipt"],
        plan=validated_plan,
        approval=approved,
        **bindings,
    )
    terminal_observation = bindings["terminal_observation"]
    terminal_at = terminal_receipt["terminal_at_unix"]
    if entries[-1].value["recorded_at_unix"] != terminal_at:
        raise PhaseBError("phase_b_durable_terminal_time_invalid")

    unsigned = {
        "schema": PHASE_B_DURABLE_FOUNDATION_SCHEMA,
        "foundation_release_revision": validated_plan.revision,
        "plan_sha256": validated_plan.sha256,
        "approval_sha256": approved.sha256,
        "approval_sequence": approved.sequence,
        "approval_chain_sha256": _sha256_json(approval_values),
        "journal_entry_count": len(entries),
        "terminal_entry_sequence": int(entries[-1].value["sequence"]),
        "terminal_entry_sha256": entries[-1].sha256,
        "terminal_receipt_sha256": terminal_receipt["receipt_sha256"],
        "terminal_observation_sha256": terminal_observation[
            "observation_sha256"
        ],
        "terminal_at_unix": terminal_at,
    }
    generation = _hashed_mapping(
        {**unsigned, "generation_sha256": _sha256_json(unsigned)},
        fields=_DURABLE_FOUNDATION_FIELDS,
        digest_field="generation_sha256",
        code="phase_b_durable_foundation_invalid",
    )
    _require_secret_free(generation)
    return (
        _canonical_bytes(validated_plan.to_mapping()),
        _canonical_bytes(approved.to_mapping()),
        _canonical_bytes(generation),
        _canonical_bytes(terminal_receipt),
        _canonical_bytes(terminal_observation),
    )


def _install_durable_foundation_boundary() -> tuple[Callable[..., Any], Callable[[Any], None]]:
    """Install a closure-sealed loader; no mapping-only mint factory escapes."""

    authority_seal = object()
    authentication_key = secrets.token_bytes(32)

    def authenticate(value: Any) -> bytes:
        digest = hmac.new(authentication_key, digestmod=hashlib.sha256)
        for attribute in (
            "_plan_bytes",
            "_approval_bytes",
            "_generation_bytes",
            "_terminal_receipt_bytes",
            "_terminal_observation_bytes",
        ):
            encoded = getattr(value, attribute, None)
            if not isinstance(encoded, bytes):
                return b""
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.digest()

    def require(value: Any) -> None:
        if (
            type(value) is not PhaseBDurableFoundation
            or getattr(value, "_authority_seal", None) is not authority_seal
        ):
            raise PhaseBError("phase_b_durable_foundation_untrusted")
        expected = authenticate(value)
        observed = getattr(value, "_authority_tag", None)
        if (
            not isinstance(observed, bytes)
            or not expected
            or not hmac.compare_digest(observed, expected)
        ):
            raise PhaseBError("phase_b_durable_foundation_untrusted")

    def load(
        plan: PhaseBPlan,
        *,
        approval: Any,
        journal: AppendOnlyPhaseBJournal,
    ) -> PhaseBDurableFoundation:
        components = _load_durable_phase_b_components(
            plan,
            approval=approval,
            journal=journal,
        )
        issued = object.__new__(PhaseBDurableFoundation)
        for attribute, encoded in zip(
            (
                "_plan_bytes",
                "_approval_bytes",
                "_generation_bytes",
                "_terminal_receipt_bytes",
                "_terminal_observation_bytes",
            ),
            components,
            strict=True,
        ):
            object.__setattr__(issued, attribute, encoded)
        object.__setattr__(issued, "_authority_seal", authority_seal)
        object.__setattr__(issued, "_authority_tag", authenticate(issued))
        require(issued)
        return issued

    return load, require


load_durable_phase_b_foundation, _require_durable_foundation = (
    _install_durable_foundation_boundary()
)


_BOOTSTRAP_CONTINUITY_FIELDS = frozenset(
    {
        "schema",
        "source_kind",
        "foundation_generation_sha256",
        "source_receipt_sha256",
        "bootstrap_resource_sha256",
        "operation_ledger_sha256",
        "observed_at_unix",
        "continuity_sha256",
    }
)


@dataclass(frozen=True)
class PhaseBBootstrapRuntimeContinuity:
    """First-generation continuity proved by the immutable Phase-B terminal.

    A later bootstrap rotation must not be accepted from caller-authored
    digests.  A future bootstrap-v2 adapter may add a capability-validated
    constructor; this public v1 parser deliberately accepts only the original
    Phase-B terminal generation.
    """

    _value: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        foundation: PhaseBDurableFoundation,
        bootstrap_resource: Mapping[str, Any],
        operation_ledger_sha256: str,
        observed_at_unix: int,
    ) -> "PhaseBBootstrapRuntimeContinuity":
        _require_durable_foundation(foundation)
        resource = _validate_bootstrap_resource(bootstrap_resource)
        ledger = _digest(
            operation_ledger_sha256,
            "phase_b_bootstrap_continuity_invalid",
        )
        raw = _hashed_mapping(
            value,
            fields=_BOOTSTRAP_CONTINUITY_FIELDS,
            digest_field="continuity_sha256",
            code="phase_b_bootstrap_continuity_invalid",
        )
        terminal = foundation.terminal_observation
        historical_cloud = _strict_mapping(
            terminal["cloud_sql"],
            frozenset(
                {
                    "project",
                    "instance",
                    "bootstrap_resource",
                    "temporary_admin_absent",
                    "temporary_admin_username_sha256",
                    "user_inventory",
                    "user_inventory_sha256",
                    "user_operations_quiescent",
                    "relevant_user_operations",
                    "operation_ledger_sha256",
                    "observed_at_unix",
                }
            ),
            "phase_b_bootstrap_continuity_invalid",
        )
        historical_resource = _validate_bootstrap_resource(
            historical_cloud["bootstrap_resource"]
        )
        terminal_receipt = foundation.terminal_receipt
        if (
            raw["schema"] != PHASE_B_BOOTSTRAP_CONTINUITY_SCHEMA
            or raw["source_kind"] != "phase_b_terminal"
            or raw["foundation_generation_sha256"]
            != foundation.generation_sha256
            or raw["source_receipt_sha256"]
            != terminal_receipt["receipt_sha256"]
            or raw["bootstrap_resource_sha256"] != _sha256_json(resource)
            or raw["operation_ledger_sha256"] != ledger
            or raw["observed_at_unix"] != observed_at_unix
            or type(observed_at_unix) is not int
            or observed_at_unix <= 0
            or resource != historical_resource
            or ledger != historical_cloud["operation_ledger_sha256"]
        ):
            raise PhaseBError("phase_b_bootstrap_continuity_invalid")
        _require_secret_free(raw)
        return cls(_json_copy(raw))

    @property
    def sha256(self) -> str:
        return str(self._value["continuity_sha256"])

    def to_mapping(self) -> dict[str, Any]:
        return _json_copy(self._value)


def build_phase_b_terminal_bootstrap_continuity(
    foundation: PhaseBDurableFoundation,
    *,
    bootstrap_resource: Mapping[str, Any],
    operation_ledger_sha256: str,
    observed_at_unix: int,
) -> PhaseBBootstrapRuntimeContinuity:
    """Build only the immutable first-generation bootstrap continuity proof."""

    _require_durable_foundation(foundation)
    resource = _validate_bootstrap_resource(bootstrap_resource)
    ledger = _digest(
        operation_ledger_sha256,
        "phase_b_bootstrap_continuity_invalid",
    )
    unsigned = {
        "schema": PHASE_B_BOOTSTRAP_CONTINUITY_SCHEMA,
        "source_kind": "phase_b_terminal",
        "foundation_generation_sha256": foundation.generation_sha256,
        "source_receipt_sha256": foundation.terminal_receipt["receipt_sha256"],
        "bootstrap_resource_sha256": _sha256_json(resource),
        "operation_ledger_sha256": ledger,
        "observed_at_unix": observed_at_unix,
    }
    return PhaseBBootstrapRuntimeContinuity.from_mapping(
        {**unsigned, "continuity_sha256": _sha256_json(unsigned)},
        foundation=foundation,
        bootstrap_resource=resource,
        operation_ledger_sha256=ledger,
        observed_at_unix=observed_at_unix,
    )


_READINESS_HOST_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "collector_authority",
        "project_id",
        "project_number",
        "zone",
        "instance_name",
        "instance_id",
        "service_account_email",
        "gce_identity_sha256",
        "machine_id_sha256",
        "hostname_sha256",
        "host_identity_sha256",
        "boot_id_sha256",
        "observed_at_unix",
        "receipt_sha256",
        "oauth_scopes",
        "oauth_scopes_sha256",
    }
)


def _validate_readiness_host_identity(value: Any, *, observed_at_unix: int) -> dict[str, Any]:
    raw = _strict_mapping(
        value,
        _READINESS_HOST_IDENTITY_FIELDS,
        "phase_b_readiness_host_identity_invalid",
    )
    receipt_unsigned = {
        name: item
        for name, item in raw.items()
        if name not in {"receipt_sha256", "oauth_scopes", "oauth_scopes_sha256"}
    }
    scopes = raw["oauth_scopes"]
    if (
        raw["schema"] != FULL_CANARY_HOST_IDENTITY_SCHEMA
        or raw["collector_authority"]
        != "trusted_root_read_only_host_collector"
        or raw["project_id"] != DEDICATED_CANARY_PROJECT_ID
        or raw["project_number"] != DEDICATED_CANARY_PROJECT_NUMBER
        or raw["zone"] != DEDICATED_CANARY_ZONE
        or raw["instance_name"] != DEDICATED_CANARY_INSTANCE_NAME
        or raw["instance_id"] != DEDICATED_CANARY_INSTANCE_ID
        or raw["service_account_email"] != DEDICATED_CANARY_SERVICE_ACCOUNT
        or raw["observed_at_unix"] != observed_at_unix
        or raw["receipt_sha256"] != _sha256_json(receipt_unsigned)
        or scopes != list(PHASE_B_VM_OAUTH_SCOPES)
        or raw["oauth_scopes_sha256"] != _sha256_json(scopes)
        or any(
            _SHA256_RE.fullmatch(str(raw[name])) is None
            for name in (
                "gce_identity_sha256",
                "machine_id_sha256",
                "hostname_sha256",
                "host_identity_sha256",
                "boot_id_sha256",
            )
        )
    ):
        raise PhaseBError("phase_b_readiness_host_identity_invalid")
    _require_secret_free(raw)
    return raw


_READINESS_OBSERVATION_FIELDS = frozenset(
    {
        "schema",
        "current_release_revision",
        "foundation_generation_sha256",
        "foundation_terminal_receipt_sha256",
        "terminal_observation",
        "host_identity",
        "cloud_sql",
        "credential",
        "services",
        "bootstrap_runtime_continuity",
        "observed_at_unix",
        "observation_sha256",
    }
)
class _PhaseBReadinessObservation:
    """Parsed readiness data, never authority merely because it is valid.

    The installed closure-sealed issuer gathers fixed external evidence itself;
    this parser only validates data and never attaches authority.  Accepting a
    caller-authored mapping as authority here would turn schema validation into
    startup permission.
    """

    __slots__ = ("_value_bytes", "_authority_seal", "_authority_tag")

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "_PhaseBReadinessObservation":
        raise TypeError(
            "_PhaseBReadinessObservation is parsed only by from_mapping"
        )

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("_PhaseBReadinessObservation is immutable")

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        foundation: PhaseBDurableFoundation,
        current_release_revision: str,
        now_unix: int | None = None,
    ) -> "_PhaseBReadinessObservation":
        _require_durable_foundation(foundation)
        revision = _revision(current_release_revision)
        current = int(time.time()) if now_unix is None else now_unix
        if type(current) is not int or current <= 0:
            raise PhaseBError("phase_b_readiness_time_invalid")
        raw = _hashed_mapping(
            value,
            fields=_READINESS_OBSERVATION_FIELDS,
            digest_field="observation_sha256",
            code="phase_b_readiness_observation_invalid",
        )
        plan = foundation.plan
        historical_terminal = foundation.terminal_observation
        historical_session = _digest(
            historical_terminal.get("session_identity_sha256"),
            "phase_b_readiness_database_invalid",
        )
        terminal = _validate_terminal_observation(
            raw["terminal_observation"],
            plan=plan,
            execution_preflight_session_sha256=historical_session,
            services_release_revision=revision,
        )
        host_identity = _validate_readiness_host_identity(
            raw["host_identity"],
            observed_at_unix=raw["observed_at_unix"],
        )
        cloud_evidence = _strict_mapping(
            raw["cloud_sql"],
            frozenset({"observation", "observed_at_unix"}),
            "phase_b_readiness_cloud_invalid",
        )
        cloud = _validate_readiness_runtime_cloud(
            cloud_evidence["observation"],
            plan=plan,
            observed_at_unix=raw["observed_at_unix"],
        )
        operations = _validate_terminal_cloud_operations(
            cloud["relevant_user_operations"]
        )
        credential_evidence = _strict_mapping(
            raw["credential"],
            frozenset({"identity", "observed_at_unix"}),
            "phase_b_readiness_credential_invalid",
        )
        credential = _validate_credential(credential_evidence["identity"])
        _require_same_credential(
            plan.preflight.value["credential"],
            credential,
        )
        services = _validate_services(
            raw["services"],
            release_revision=revision,
        )
        observed_at = raw["observed_at_unix"]
        if (
            raw["schema"] != PHASE_B_READINESS_OBSERVATION_SCHEMA
            or raw["current_release_revision"] != revision
            or raw["foundation_generation_sha256"]
            != foundation.generation_sha256
            or raw["foundation_terminal_receipt_sha256"]
            != foundation.terminal_receipt["receipt_sha256"]
            or type(observed_at) is not int
            or not foundation.terminal_at_unix <= observed_at <= current
            or current - observed_at > PHASE_B_READINESS_MAX_AGE_SECONDS
            or cloud_evidence["observed_at_unix"] != observed_at
            or credential_evidence["observed_at_unix"] != observed_at
            or terminal["observed_at_unix"] != observed_at
            or services["observed_at_unix"] != observed_at
            or terminal["services"] != services
            or host_identity["observed_at_unix"] != observed_at
        ):
            raise PhaseBError("phase_b_readiness_observation_invalid")

        terminal_cloud = terminal["cloud_sql"]
        terminal_foundation = FoundationObservation.from_mapping(
            terminal["foundation_observation"]
        )
        if terminal_cloud != cloud:
            raise PhaseBError("phase_b_readiness_cloud_mismatch")
        _require_same_credential(
            credential,
            terminal_foundation.value["credential"],
        )
        PhaseBBootstrapRuntimeContinuity.from_mapping(
            raw["bootstrap_runtime_continuity"],
            foundation=foundation,
            bootstrap_resource=historical_terminal["cloud_sql"][
                "bootstrap_resource"
            ],
            operation_ledger_sha256=cloud["operation_ledger_sha256"],
            observed_at_unix=observed_at,
        )
        _require_secret_free(raw)
        parsed = object.__new__(cls)
        object.__setattr__(parsed, "_value_bytes", _canonical_bytes(raw))
        object.__setattr__(parsed, "_authority_seal", None)
        object.__setattr__(parsed, "_authority_tag", None)
        return parsed

    def _mapping(self) -> dict[str, Any]:
        try:
            decoded = json.loads(self._value_bytes.decode("utf-8"))
        except (AttributeError, UnicodeError, json.JSONDecodeError) as exc:
            raise PhaseBError("phase_b_readiness_observation_untrusted") from exc
        if (
            not isinstance(decoded, dict)
            or _canonical_bytes(decoded) != self._value_bytes
        ):
            raise PhaseBError("phase_b_readiness_observation_untrusted")
        return decoded

    @property
    def sha256(self) -> str:
        return str(self._mapping()["observation_sha256"])

    @property
    def observed_at_unix(self) -> int:
        return int(self._mapping()["observed_at_unix"])

    @property
    def current_release_revision(self) -> str:
        return str(self._mapping()["current_release_revision"])

    def to_mapping(self) -> dict[str, Any]:
        return self._mapping()


def _install_readiness_observation_boundary() -> tuple[
    Callable[..., _PhaseBReadinessObservation],
    Callable[[Any], None],
]:
    """Install a fixed-collector issuer and its matching consumer.

    The issuer deliberately accepts no evidence mapping or collector from its
    caller.  It invokes the packaged fixed-target collector from inside this
    closure, validates the resulting observation, and only then attaches the
    process-private capability seal.  Consequently schema-valid caller data,
    ``object.__new__`` instances, and monkey-patched writer calls remain unable
    to authorize startup publication.
    """

    authority_seal = object()
    authentication_key = secrets.token_bytes(32)

    def require(value: Any) -> None:
        if (
            type(value) is not _PhaseBReadinessObservation
            or getattr(value, "_authority_seal", None) is not authority_seal
        ):
            raise PhaseBError("phase_b_readiness_observation_untrusted")
        encoded = getattr(value, "_value_bytes", None)
        observed = getattr(value, "_authority_tag", None)
        if not isinstance(encoded, bytes) or not isinstance(observed, bytes):
            raise PhaseBError("phase_b_readiness_observation_untrusted")
        expected = hmac.new(
            authentication_key,
            encoded,
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(observed, expected):
            raise PhaseBError("phase_b_readiness_observation_untrusted")

    def collect(
        foundation: PhaseBDurableFoundation,
        *,
        current_release_revision: str,
        now_unix: int | None = None,
    ) -> _PhaseBReadinessObservation:
        _require_durable_foundation(foundation)
        revision = _revision(current_release_revision)
        current = int(time.time()) if now_unix is None else now_unix
        if type(current) is not int or current <= 0:
            raise PhaseBError("phase_b_readiness_time_invalid")
        # Import lazily to keep this foundation module usable for offline plan
        # validation and Windows packaging inspection.  The called function is
        # zero-input with respect to evidence selection: all external targets
        # and collectors are compile-time fixed in the packaged runtime.
        try:
            from gateway.canonical_writer_phase_b_runtime import (
                _collect_fixed_phase_b_readiness_mapping,
            )

            raw = _collect_fixed_phase_b_readiness_mapping(
                foundation,
                current_release_revision=revision,
                observed_at_unix=current,
            )
        except PhaseBError:
            raise
        except BaseException as exc:
            raise PhaseBError("phase_b_readiness_collection_failed") from exc
        parsed = _PhaseBReadinessObservation.from_mapping(
            raw,
            foundation=foundation,
            current_release_revision=revision,
            now_unix=current,
        )
        issued = object.__new__(_PhaseBReadinessObservation)
        object.__setattr__(issued, "_value_bytes", parsed._value_bytes)
        object.__setattr__(issued, "_authority_seal", authority_seal)
        object.__setattr__(
            issued,
            "_authority_tag",
            hmac.new(
                authentication_key,
                parsed._value_bytes,
                hashlib.sha256,
            ).digest(),
        )
        require(issued)
        return issued

    return collect, require


(
    _collect_trusted_readiness_observation,
    _require_trusted_readiness_observation,
) = (
    _install_readiness_observation_boundary()
)


_READINESS_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "ok",
        "state",
        "safe_to_start",
        "current_release_revision",
        "foundation_generation_sha256",
        "foundation_terminal_receipt_sha256",
        "sequence",
        "previous_receipt_sha256",
        "readiness_observation",
        "readiness_observation_sha256",
        "observed_at_unix",
        "issued_at_unix",
        "expires_at_unix",
        "secret_material_recorded",
        "receipt_sha256",
    }
)


class PhaseBReadinessReceipt:
    """Opaque readiness authority issued only from the fixed durable journal."""

    __slots__ = ("_value_bytes", "_authority_seal", "_authority_tag")

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "PhaseBReadinessReceipt":
        raise TypeError(
            "PhaseBReadinessReceipt is issued only from the fixed readiness journal"
        )

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("PhaseBReadinessReceipt is immutable")

    def _mapping(self) -> dict[str, Any]:
        _require_readiness_receipt(self)
        try:
            decoded = json.loads(self._value_bytes.decode("utf-8"))
        except (AttributeError, UnicodeError, json.JSONDecodeError) as exc:
            raise PhaseBError("phase_b_readiness_receipt_untrusted") from exc
        if not isinstance(decoded, dict) or _canonical_bytes(decoded) != self._value_bytes:
            raise PhaseBError("phase_b_readiness_receipt_untrusted")
        return decoded

    @property
    def sha256(self) -> str:
        return str(self._mapping()["receipt_sha256"])

    @property
    def sequence(self) -> int:
        return int(self._mapping()["sequence"])

    def to_mapping(self) -> dict[str, Any]:
        return self._mapping()


def _validate_phase_b_readiness_receipt_mapping(
    value: Any,
    *,
    foundation: PhaseBDurableFoundation,
    current_release_revision: str,
    expected_sequence: int,
    expected_previous_receipt_sha256: str | None,
    now_unix: int | None = None,
) -> dict[str, Any]:
    _require_durable_foundation(foundation)
    revision = _revision(current_release_revision)
    current = int(time.time()) if now_unix is None else now_unix
    if (
        type(current) is not int
        or current <= 0
        or type(expected_sequence) is not int
        or expected_sequence < 0
        or (expected_sequence == 0)
        is not (expected_previous_receipt_sha256 is None)
    ):
        raise PhaseBError("phase_b_readiness_chain_invalid")
    if expected_previous_receipt_sha256 is not None:
        _digest(
            expected_previous_receipt_sha256,
            "phase_b_readiness_chain_invalid",
        )
    raw = _hashed_mapping(
        value,
        fields=_READINESS_RECEIPT_FIELDS,
        digest_field="receipt_sha256",
        code="phase_b_readiness_receipt_invalid",
    )
    observation = _PhaseBReadinessObservation.from_mapping(
        raw["readiness_observation"],
        foundation=foundation,
        current_release_revision=revision,
        now_unix=current,
    )
    observed_at = raw["observed_at_unix"]
    issued_at = raw["issued_at_unix"]
    expires_at = raw["expires_at_unix"]
    if (
        raw["schema"] != PHASE_B_READINESS_RECEIPT_SCHEMA
        or raw["ok"] is not True
        or raw["state"] != "ready"
        or raw["safe_to_start"] is not True
        or raw["current_release_revision"] != revision
        or raw["foundation_generation_sha256"]
        != foundation.generation_sha256
        or raw["foundation_terminal_receipt_sha256"]
        != foundation.terminal_receipt["receipt_sha256"]
        or raw["sequence"] != expected_sequence
        or raw["previous_receipt_sha256"]
        != expected_previous_receipt_sha256
        or raw["readiness_observation_sha256"] != observation.sha256
        or observed_at != observation.observed_at_unix
        or type(issued_at) is not int
        or type(expires_at) is not int
        or not observed_at <= issued_at <= current < expires_at
        or expires_at - observed_at > PHASE_B_READINESS_MAX_AGE_SECONDS
        or raw["secret_material_recorded"] is not False
    ):
        raise PhaseBError("phase_b_readiness_receipt_invalid")
    _require_secret_free(raw)
    return raw


class _PhaseBReadinessJournalFoundation:
    """Fixed-identity mechanics shared by the reader and root writer."""

    def __init__(self) -> None:
        self.root = _PHASE_B_READINESS_ROOT
        self._lock_depth = 0
        self._lock_exclusive = False
        self._active_root_fd: int | None = None
        self._active_generation_fd: int | None = None

    def _generation_root(self, foundation: PhaseBDurableFoundation) -> Path:
        return self.root / foundation.generation_sha256

    def _entries_root(self, foundation: PhaseBDurableFoundation) -> Path:
        return self._generation_root(foundation) / "entries"

    def _staging_root(self, foundation: PhaseBDurableFoundation) -> Path:
        return self._generation_root(foundation) / "staging"

    @staticmethod
    def _require_writer_authority() -> None:
        if _effective_uid() != _READINESS_AUTHORITY_UID:
            raise PhaseBError("phase_b_readiness_writer_authority_required")

    @staticmethod
    def _directory_flags() -> int:
        return (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )

    @staticmethod
    def _trusted_directory_status(status: os.stat_result) -> bool:
        return (
            stat.S_ISDIR(status.st_mode)
            and not stat.S_ISLNK(status.st_mode)
            and stat.S_IMODE(status.st_mode) == _JOURNAL_MODE
            and status.st_uid == _READINESS_AUTHORITY_UID
            and status.st_gid == _READINESS_AUTHORITY_GID
        )

    @staticmethod
    def _directory_identity(status: os.stat_result) -> tuple[int, ...]:
        return (
            status.st_dev,
            status.st_ino,
            status.st_mode,
            status.st_uid,
            status.st_gid,
        )

    @staticmethod
    def _fsync_directory_handle(descriptor: int, _path: Path) -> None:
        """Persist the already-open directory, never re-resolve its path."""

        os.fsync(descriptor)

    @classmethod
    def _open_absolute_parent(cls, path: Path) -> tuple[int, str]:
        parts = path.parts
        if not path.is_absolute() or len(parts) < 2 or not path.name:
            raise PhaseBError("phase_b_readiness_journal_root_invalid")
        descriptor: int | None = None
        try:
            descriptor = os.open(parts[0], cls._directory_flags())
            for component in parts[1:-1]:
                child = os.open(
                    component,
                    cls._directory_flags(),
                    dir_fd=descriptor,
                )
                opened = os.fstat(child)
                if not stat.S_ISDIR(opened.st_mode):
                    os.close(child)
                    raise PhaseBError(
                        "phase_b_readiness_journal_directory_untrusted"
                    )
                os.close(descriptor)
                descriptor = child
        except PhaseBError:
            if descriptor is not None:
                os.close(descriptor)
            raise
        except OSError as exc:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise PhaseBError(
                "phase_b_readiness_journal_unavailable"
            ) from exc
        assert descriptor is not None
        return descriptor, path.name

    @classmethod
    def _open_child_directory_at(
        cls,
        parent_fd: int,
        name: str,
        *,
        missing_ok: bool = False,
    ) -> int | None:
        descriptor: int | None = None
        observed = False
        try:
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            observed = True
            descriptor = os.open(
                name,
                cls._directory_flags(),
                dir_fd=parent_fd,
            )
            opened = os.fstat(descriptor)
            after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if descriptor is not None:
                os.close(descriptor)
            if missing_ok and not observed:
                return None
            raise PhaseBError(
                "phase_b_readiness_journal_unavailable"
            ) from None
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            raise PhaseBError(
                "phase_b_readiness_journal_unavailable"
            ) from exc
        assert descriptor is not None
        if (
            cls._directory_identity(before)
            != cls._directory_identity(opened)
            or cls._directory_identity(opened)
            != cls._directory_identity(after)
            or not cls._trusted_directory_status(opened)
        ):
            os.close(descriptor)
            raise PhaseBError(
                "phase_b_readiness_journal_directory_untrusted"
            )
        return descriptor

    @classmethod
    def _ensure_child_directory_at(
        cls,
        parent_fd: int,
        name: str,
        *,
        parent_path: Path,
    ) -> int:
        cls._require_writer_authority()
        created = False
        try:
            os.mkdir(name, _JOURNAL_MODE, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise PhaseBError(
                "phase_b_readiness_journal_unavailable"
            ) from exc
        if created:
            try:
                descriptor = os.open(
                    name,
                    cls._directory_flags(),
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                raise PhaseBError(
                    "phase_b_readiness_journal_unavailable"
                ) from exc
        else:
            opened = cls._open_child_directory_at(parent_fd, name)
            assert opened is not None
            descriptor = opened
        try:
            opened = os.fstat(descriptor)
            if (
                created
                and opened.st_uid == _READINESS_AUTHORITY_UID
                and opened.st_gid != _READINESS_AUTHORITY_GID
            ):
                os.fchown(
                    descriptor,
                    -1,
                    _READINESS_AUTHORITY_GID,
                )
                opened = os.fstat(descriptor)
            if not cls._trusted_directory_status(opened):
                raise PhaseBError(
                    "phase_b_readiness_journal_directory_untrusted"
                )
            # Both barriers are retried after EEXIST.  mkdir may have reached
            # disk while the previous parent fsync was the only failed step.
            cls._fsync_directory_handle(descriptor, parent_path / name)
            cls._fsync_directory_handle(parent_fd, parent_path)
        except PhaseBError:
            os.close(descriptor)
            raise
        except OSError as exc:
            os.close(descriptor)
            raise PhaseBError(
                "phase_b_readiness_journal_unavailable"
            ) from exc
        return descriptor

    def _open_root_handle(self, *, create: bool) -> int | None:
        parent_fd, name = self._open_absolute_parent(self.root)
        try:
            if create:
                return self._ensure_child_directory_at(
                    parent_fd,
                    name,
                    parent_path=self.root.parent,
                )
            return self._open_child_directory_at(
                parent_fd,
                name,
                missing_ok=True,
            )
        finally:
            os.close(parent_fd)

    @classmethod
    def _open_lock_at(
        cls,
        generation_fd: int,
        *,
        create: bool,
        generation_path: Path,
    ) -> int:
        flags = (
            (os.O_RDWR if create else os.O_RDONLY)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        if create:
            flags |= os.O_CREAT
        descriptor: int | None = None
        existed = True
        try:
            os.stat(".lock", dir_fd=generation_fd, follow_symlinks=False)
        except FileNotFoundError:
            existed = False
        try:
            descriptor = os.open(
                ".lock",
                flags,
                _JOURNAL_FILE_MODE,
                dir_fd=generation_fd,
            )
            opened = os.fstat(descriptor)
            if (
                create
                and not existed
                and opened.st_uid == _READINESS_AUTHORITY_UID
                and opened.st_gid != _READINESS_AUTHORITY_GID
            ):
                os.fchown(descriptor, -1, _READINESS_AUTHORITY_GID)
                opened = os.fstat(descriptor)
            after = os.stat(
                ".lock",
                dir_fd=generation_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise PhaseBError("phase_b_readiness_journal_lock_untrusted") from exc
        assert descriptor is not None
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_gid,
            value.st_nlink,
        )
        if (
            identity(opened) != identity(after)
            or not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != _JOURNAL_FILE_MODE
            or opened.st_nlink != 1
            or opened.st_uid != _READINESS_AUTHORITY_UID
            or opened.st_gid != _READINESS_AUTHORITY_GID
        ):
            os.close(descriptor)
            raise PhaseBError("phase_b_readiness_journal_lock_untrusted")
        if create:
            try:
                os.fsync(descriptor)
                cls._fsync_directory_handle(
                    generation_fd,
                    generation_path,
                )
            except OSError as exc:
                os.close(descriptor)
                raise PhaseBError(
                    "phase_b_readiness_journal_lock_untrusted"
                ) from exc
        return descriptor

    @contextmanager
    def _exclusive_lock(self, foundation: PhaseBDurableFoundation) -> Iterator[None]:
        _require_durable_foundation(foundation)
        if fcntl is None:
            raise PhaseBError("phase_b_posix_lock_unavailable")
        self._require_writer_authority()
        if self._lock_depth != 0:
            raise PhaseBError("phase_b_readiness_journal_lock_reentrant")
        root_fd = self._open_root_handle(create=True)
        assert root_fd is not None
        generation_path = self._generation_root(foundation)
        generation_fd: int | None = None
        descriptor: int | None = None
        try:
            generation_fd = self._ensure_child_directory_at(
                root_fd,
                foundation.generation_sha256,
                parent_path=self.root,
            )
            descriptor = self._open_lock_at(
                generation_fd,
                create=True,
                generation_path=generation_path,
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._lock_depth = 1
            self._lock_exclusive = True
            self._active_root_fd = root_fd
            self._active_generation_fd = generation_fd
            self._recover_staging(foundation, generation_fd=generation_fd)
            yield
        finally:
            try:
                self._lock_depth = 0
                self._lock_exclusive = False
                self._active_root_fd = None
                self._active_generation_fd = None
                if descriptor is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                if generation_fd is not None:
                    os.close(generation_fd)
                os.close(root_fd)

    @contextmanager
    def _shared_lock(self, foundation: PhaseBDurableFoundation) -> Iterator[None]:
        _require_durable_foundation(foundation)
        if fcntl is None:
            raise PhaseBError("phase_b_posix_lock_unavailable")
        if self._lock_depth != 0:
            raise PhaseBError("phase_b_readiness_journal_lock_reentrant")
        root_fd = self._open_root_handle(create=False)
        if root_fd is None:
            raise PhaseBError("phase_b_readiness_receipt_missing")
        generation_fd = self._open_child_directory_at(
            root_fd,
            foundation.generation_sha256,
            missing_ok=True,
        )
        if generation_fd is None:
            os.close(root_fd)
            raise PhaseBError("phase_b_readiness_receipt_missing")
        descriptor: int | None = None
        try:
            descriptor = self._open_lock_at(
                generation_fd,
                create=False,
                generation_path=self._generation_root(foundation),
            )
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            self._lock_depth = 1
            self._lock_exclusive = False
            self._active_root_fd = root_fd
            self._active_generation_fd = generation_fd
            yield
        finally:
            try:
                self._lock_depth = 0
                self._active_root_fd = None
                self._active_generation_fd = None
                if descriptor is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                os.close(generation_fd)
                os.close(root_fd)

    @staticmethod
    def _read_canonical_file_at(
        directory_fd: int,
        name: str,
        *,
        expected_link_count: int = 1,
    ) -> Mapping[str, Any]:
        if expected_link_count not in {1, 2}:
            raise PhaseBError("phase_b_readiness_journal_file_untrusted")
        try:
            before = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
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
                payload = bytes(chunks)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise PhaseBError("phase_b_readiness_journal_read_failed") from exc
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            value.st_nlink,
        )
        if (
            identity(before) != identity(opened)
            or identity(opened) != identity(after)
            or stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != _JOURNAL_FILE_MODE
            or before.st_nlink != expected_link_count
            or before.st_uid != _READINESS_AUTHORITY_UID
            or before.st_gid != _READINESS_AUTHORITY_GID
            or not payload
            or len(payload) > _MAX_JSON_BYTES
        ):
            raise PhaseBError("phase_b_readiness_journal_file_untrusted")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PhaseBError("phase_b_readiness_journal_json_invalid") from exc
        if not isinstance(decoded, Mapping) or _canonical_bytes(decoded) != payload:
            raise PhaseBError("phase_b_readiness_journal_not_canonical")
        return _json_copy(decoded)

    def _recover_staging(
        self,
        foundation: PhaseBDurableFoundation,
        *,
        generation_fd: int,
    ) -> None:
        if self._lock_depth != 1 or not self._lock_exclusive:
            raise PhaseBError("phase_b_readiness_journal_lock_required")
        staging_fd = self._open_child_directory_at(
            generation_fd,
            "staging",
            missing_ok=True,
        )
        if staging_fd is None:
            return
        entries_fd = self._open_child_directory_at(
            generation_fd,
            "entries",
            missing_ok=True,
        )
        generation_path = self._generation_root(foundation)
        staging_path = generation_path / "staging"
        entries_path = generation_path / "entries"
        try:
            stages = sorted(os.listdir(staging_fd))
            if not stages:
                return
            if len(stages) != 1 or entries_fd is None:
                raise PhaseBError(
                    "phase_b_readiness_journal_staging_conflict"
                )
            stage_name = stages[0]
            match = re.fullmatch(
                r"([0-9]{8})\.([0-9a-f]{32})\.stage",
                stage_name,
            )
            if match is None:
                raise PhaseBError(
                    "phase_b_readiness_journal_staging_conflict"
                )
            sequence = int(match.group(1))
            try:
                before = os.stat(
                    stage_name,
                    dir_fd=staging_fd,
                    follow_symlinks=False,
                )
                descriptor = os.open(
                    stage_name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=staging_fd,
                )
                try:
                    opened = os.fstat(descriptor)
                finally:
                    os.close(descriptor)
                after = os.stat(
                    stage_name,
                    dir_fd=staging_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise PhaseBError(
                    "phase_b_readiness_journal_read_failed"
                ) from exc
            identity = lambda value: (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_uid,
                value.st_gid,
                value.st_size,
                value.st_nlink,
            )
            if (
                identity(before) != identity(opened)
                or identity(opened) != identity(after)
                or not stat.S_ISREG(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != _JOURNAL_FILE_MODE
                or opened.st_uid != _READINESS_AUTHORITY_UID
                or opened.st_gid != _READINESS_AUTHORITY_GID
                or opened.st_nlink not in {1, 2}
            ):
                raise PhaseBError(
                    "phase_b_readiness_journal_staging_conflict"
                )
            final_names = sorted(os.listdir(entries_fd))
            same_sequence = [
                name
                for name in final_names
                if name.startswith(f"{sequence:08d}-")
            ]
            if opened.st_nlink == 1:
                if same_sequence or sequence != len(final_names):
                    raise PhaseBError(
                        "phase_b_readiness_journal_staging_conflict"
                    )
                os.unlink(stage_name, dir_fd=staging_fd)
                self._fsync_directory_handle(staging_fd, staging_path)
                return

            if len(same_sequence) != 1 or sequence != len(final_names) - 1:
                raise PhaseBError(
                    "phase_b_readiness_journal_staging_conflict"
                )
            value = self._read_canonical_file_at(
                staging_fd,
                stage_name,
                expected_link_count=2,
            )
            issued_at = value.get("issued_at_unix")
            revision = value.get("current_release_revision")
            if type(issued_at) is not int or not isinstance(revision, str):
                raise PhaseBError(
                    "phase_b_readiness_journal_staging_conflict"
                )
            loaded = self._load_receipt_bytes_from_fd(
                foundation,
                entries_fd,
                linked_sequence=sequence,
            )
            if not loaded or json.loads(loaded[-1].decode("utf-8")) != value:
                raise PhaseBError(
                    "phase_b_readiness_journal_staging_conflict"
                )
            receipt_sha256 = str(value.get("receipt_sha256"))
            final_name = f"{sequence:08d}-{receipt_sha256}.json"
            final_status = os.stat(
                final_name,
                dir_fd=entries_fd,
                follow_symlinks=False,
            )
            if (
                same_sequence != [final_name]
                or final_status.st_nlink != 2
                or (opened.st_dev, opened.st_ino)
                != (final_status.st_dev, final_status.st_ino)
            ):
                raise PhaseBError(
                    "phase_b_readiness_journal_staging_conflict"
                )
            # Retry the exact final publication barrier before discarding the
            # only marker that distinguishes the linked crash window.
            self._fsync_directory_handle(entries_fd, entries_path)
            os.unlink(stage_name, dir_fd=staging_fd)
            self._fsync_directory_handle(staging_fd, staging_path)
            self._fsync_directory_handle(entries_fd, entries_path)
        except OSError as exc:
            raise PhaseBError(
                "phase_b_readiness_journal_staging_conflict"
            ) from exc
        finally:
            if entries_fd is not None:
                os.close(entries_fd)
            os.close(staging_fd)

    def _load_receipt_bytes_from_fd(
        self,
        foundation: PhaseBDurableFoundation,
        entries_fd: int,
        *,
        linked_sequence: int | None = None,
    ) -> list[bytes]:
        _require_durable_foundation(foundation)
        try:
            names = sorted(os.listdir(entries_fd))
        except OSError as exc:
            raise PhaseBError(
                "phase_b_readiness_journal_unavailable"
            ) from exc
        receipts: list[bytes] = []
        previous: str | None = None
        previous_issued_at: int | None = None
        previous_observed_at: int | None = None
        for sequence, name in enumerate(names):
            match = re.fullmatch(
                r"([0-9]{8})-([0-9a-f]{64})\.json",
                name,
            )
            if match is None or int(match.group(1)) != sequence:
                raise PhaseBError("phase_b_readiness_journal_sequence_invalid")
            value = self._read_canonical_file_at(
                entries_fd,
                name,
                expected_link_count=(
                    2 if linked_sequence == sequence else 1
                ),
            )
            issued_at = value.get("issued_at_unix")
            revision = value.get("current_release_revision")
            if type(issued_at) is not int or not isinstance(revision, str):
                raise PhaseBError("phase_b_readiness_receipt_invalid")
            receipt = _validate_phase_b_readiness_receipt_mapping(
                value,
                foundation=foundation,
                current_release_revision=revision,
                expected_sequence=sequence,
                expected_previous_receipt_sha256=previous,
                now_unix=issued_at,
            )
            receipt_sha256 = str(receipt["receipt_sha256"])
            if match.group(2) != receipt_sha256:
                raise PhaseBError("phase_b_readiness_journal_path_invalid")
            observed_at = int(value["observed_at_unix"])
            if (
                previous_issued_at is not None
                and (
                    issued_at < previous_issued_at
                    or (
                        previous_observed_at is not None
                        and observed_at < previous_observed_at
                    )
                )
            ):
                raise PhaseBError("phase_b_readiness_journal_time_regressed")
            receipts.append(_canonical_bytes(receipt))
            previous = receipt_sha256
            previous_issued_at = issued_at
            previous_observed_at = observed_at
        return receipts

    def _load_receipt_bytes_unlocked(
        self,
        foundation: PhaseBDurableFoundation,
    ) -> list[bytes]:
        _require_durable_foundation(foundation)
        if (
            self._lock_depth != 1
            or self._active_generation_fd is None
        ):
            raise PhaseBError("phase_b_readiness_journal_lock_required")
        generation_fd = self._active_generation_fd
        try:
            names = set(os.listdir(generation_fd))
        except OSError as exc:
            raise PhaseBError(
                "phase_b_readiness_journal_unavailable"
            ) from exc
        if not names.issubset({".lock", "entries", "staging"}):
            raise PhaseBError("phase_b_readiness_journal_path_invalid")
        staging_fd = self._open_child_directory_at(
            generation_fd,
            "staging",
            missing_ok=True,
        )
        if staging_fd is not None:
            try:
                if os.listdir(staging_fd):
                    raise PhaseBError(
                        "phase_b_readiness_journal_staging_residue"
                    )
            finally:
                os.close(staging_fd)
        entries_fd = self._open_child_directory_at(
            generation_fd,
            "entries",
            missing_ok=True,
        )
        if entries_fd is None:
            return []
        try:
            return self._load_receipt_bytes_from_fd(
                foundation,
                entries_fd,
            )
        finally:
            os.close(entries_fd)

    def _load_unlocked(
        self,
        foundation: PhaseBDurableFoundation,
    ) -> list[PhaseBReadinessReceipt]:
        return _load_sealed_readiness_receipts(self, foundation)

    def _append_unlocked(
        self,
        foundation: PhaseBDurableFoundation,
        *,
        observation: _PhaseBReadinessObservation,
        current_release_revision: str,
        now_unix: int,
    ) -> PhaseBReadinessReceipt:
        if (
            self._lock_depth != 1
            or not self._lock_exclusive
            or self._active_generation_fd is None
        ):
            raise PhaseBError("phase_b_readiness_journal_lock_required")
        self._require_writer_authority()
        if not isinstance(observation, _PhaseBReadinessObservation):
            raise TypeError("_PhaseBReadinessObservation is required")
        _require_trusted_readiness_observation(observation)
        generation_fd = self._active_generation_fd
        generation_path = self._generation_root(foundation)
        entries_path = self._entries_root(foundation)
        staging_path = self._staging_root(foundation)
        entries_fd = self._ensure_child_directory_at(
            generation_fd,
            "entries",
            parent_path=generation_path,
        )
        try:
            staging_fd = self._ensure_child_directory_at(
                generation_fd,
                "staging",
                parent_path=generation_path,
            )
        except BaseException:
            os.close(entries_fd)
            raise
        try:
            self._recover_staging(
                foundation,
                generation_fd=generation_fd,
            )
            receipt_bytes = self._load_receipt_bytes_from_fd(
                foundation,
                entries_fd,
            )
            previous = None
            if receipt_bytes:
                previous_value = json.loads(
                    receipt_bytes[-1].decode("utf-8")
                )
                previous = str(previous_value["receipt_sha256"])
            receipt = _build_phase_b_readiness_receipt_mapping(
                foundation,
                observation=observation,
                current_release_revision=current_release_revision,
                sequence=len(receipt_bytes),
                previous_receipt_sha256=previous,
                now_unix=now_unix,
            )
            payload = _canonical_bytes(receipt)
            stage_name = (
                f"{len(receipt_bytes):08d}."
                f"{secrets.token_hex(16)}.stage"
            )
            final_name = (
                f"{len(receipt_bytes):08d}-"
                f"{receipt['receipt_sha256']}.json"
            )
            descriptor = os.open(
                stage_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                _JOURNAL_FILE_MODE,
                dir_fd=staging_fd,
            )
            try:
                try:
                    status = os.fstat(descriptor)
                    if (
                        status.st_uid == _READINESS_AUTHORITY_UID
                        and status.st_gid != _READINESS_AUTHORITY_GID
                    ):
                        os.fchown(
                            descriptor,
                            -1,
                            _READINESS_AUTHORITY_GID,
                        )
                        status = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(status.st_mode)
                        or stat.S_IMODE(status.st_mode)
                        != _JOURNAL_FILE_MODE
                        or status.st_nlink != 1
                        or status.st_uid != _READINESS_AUTHORITY_UID
                        or status.st_gid != _READINESS_AUTHORITY_GID
                    ):
                        raise PhaseBError(
                            "phase_b_readiness_journal_file_untrusted"
                        )
                    offset = 0
                    while offset < len(payload):
                        written = os.write(descriptor, payload[offset:])
                        if written <= 0:
                            raise PhaseBError(
                                "phase_b_readiness_journal_write_failed"
                            )
                        offset += written
                    os.fsync(descriptor)
                except OSError as exc:
                    raise PhaseBError(
                        "phase_b_readiness_journal_write_failed"
                    ) from exc
            finally:
                os.close(descriptor)
            try:
                self._fsync_directory_handle(staging_fd, staging_path)
                os.link(
                    stage_name,
                    final_name,
                    src_dir_fd=staging_fd,
                    dst_dir_fd=entries_fd,
                    follow_symlinks=False,
                )
                self._fsync_directory_handle(entries_fd, entries_path)
            except FileExistsError as exc:
                raise PhaseBError(
                    "phase_b_readiness_journal_fork"
                ) from exc
            except OSError as exc:
                raise PhaseBError(
                    "phase_b_readiness_journal_publish_failed"
                ) from exc
            try:
                os.unlink(stage_name, dir_fd=staging_fd)
                self._fsync_directory_handle(staging_fd, staging_path)
                # Removing the second link changes the final inode metadata.
                # Retry this barrier after a crash before accepting readback.
                self._fsync_directory_handle(entries_fd, entries_path)
            except OSError as exc:
                raise PhaseBError(
                    "phase_b_readiness_journal_publish_failed"
                ) from exc
            loaded_bytes = self._load_receipt_bytes_from_fd(
                foundation,
                entries_fd,
            )
            if (
                len(loaded_bytes) != len(receipt_bytes) + 1
                or json.loads(loaded_bytes[-1].decode("utf-8"))
                != receipt
            ):
                raise PhaseBError(
                    "phase_b_readiness_journal_readback_failed"
                )
            loaded = self._load_unlocked(foundation)
            if not loaded or loaded[-1].sha256 != receipt["receipt_sha256"]:
                raise PhaseBError(
                    "phase_b_readiness_journal_readback_failed"
                )
            return loaded[-1]
        finally:
            os.close(staging_fd)
            os.close(entries_fd)

    def _latest_fresh_unlocked(
        self,
        foundation: PhaseBDurableFoundation,
        *,
        expected_current_release_revision: str,
        now_unix: int | None = None,
    ) -> PhaseBReadinessReceipt:
        revision = _revision(expected_current_release_revision)
        receipts = self._load_unlocked(foundation)
        if not receipts:
            raise PhaseBError("phase_b_readiness_receipt_missing")
        latest = receipts[-1]
        value = latest.to_mapping()
        if value["current_release_revision"] != revision:
            raise PhaseBError("phase_b_readiness_release_mismatch")
        previous = receipts[-2].sha256 if len(receipts) > 1 else None
        validated = _validate_phase_b_readiness_receipt_mapping(
            value,
            foundation=foundation,
            current_release_revision=revision,
            expected_sequence=len(receipts) - 1,
            expected_previous_receipt_sha256=previous,
            now_unix=now_unix,
        )
        if validated != value:
            raise PhaseBError("phase_b_readiness_receipt_invalid")
        return latest


def _install_readiness_receipt_boundary() -> tuple[
    Callable[[Any, PhaseBDurableFoundation], list[PhaseBReadinessReceipt]],
    Callable[[Any], None],
]:
    """Seal receipts only after the fixed journal has authenticated its files."""

    authority_seal = object()
    authentication_key = secrets.token_bytes(32)

    def authenticate(value: Any) -> bytes:
        encoded = getattr(value, "_value_bytes", None)
        if not isinstance(encoded, bytes):
            return b""
        return hmac.new(
            authentication_key,
            encoded,
            hashlib.sha256,
        ).digest()

    def require(value: Any) -> None:
        if (
            type(value) is not PhaseBReadinessReceipt
            or getattr(value, "_authority_seal", None) is not authority_seal
        ):
            raise PhaseBError("phase_b_readiness_receipt_untrusted")
        expected = authenticate(value)
        observed = getattr(value, "_authority_tag", None)
        if (
            not isinstance(observed, bytes)
            or not expected
            or not hmac.compare_digest(observed, expected)
        ):
            raise PhaseBError("phase_b_readiness_receipt_untrusted")

    def load(
        journal: Any,
        foundation: PhaseBDurableFoundation,
    ) -> list[PhaseBReadinessReceipt]:
        if type(journal) is not _PhaseBReadinessJournalFoundation:
            raise PhaseBError("phase_b_readiness_journal_missing")
        _require_durable_foundation(foundation)
        encoded_values = (
            _PhaseBReadinessJournalFoundation._load_receipt_bytes_unlocked(
                journal,
                foundation,
            )
        )
        receipts: list[PhaseBReadinessReceipt] = []
        for encoded in encoded_values:
            issued = object.__new__(PhaseBReadinessReceipt)
            object.__setattr__(issued, "_value_bytes", encoded)
            object.__setattr__(issued, "_authority_seal", authority_seal)
            object.__setattr__(issued, "_authority_tag", authenticate(issued))
            require(issued)
            receipts.append(issued)
        return receipts

    return load, require


_load_sealed_readiness_receipts, _require_readiness_receipt = (
    _install_readiness_receipt_boundary()
)


class AppendOnlyPhaseBReadinessJournal:
    """Read-only view of the fixed root-owned readiness authority head."""

    __slots__ = ("_journal",)

    def __init__(self) -> None:
        self._journal = _PhaseBReadinessJournalFoundation()

    @property
    def root(self) -> Path:
        return self._journal.root

    def latest_fresh(
        self,
        foundation: PhaseBDurableFoundation,
        *,
        expected_current_release_revision: str,
        now_unix: int | None = None,
    ) -> PhaseBReadinessReceipt:
        with self._journal._shared_lock(foundation):
            return self._journal._latest_fresh_unlocked(
                foundation,
                expected_current_release_revision=(
                    expected_current_release_revision
                ),
                now_unix=now_unix,
            )


class _PhaseBReadinessWriterBoundary:
    """Non-exported root-authority publication boundary.

    The packaged fixed-target collector is invoked only by the closure-sealed
    issuer above.  Ordinary application callers receive only the fixed-journal
    reader and cannot turn a mapping into a publishable observation.
    """

    __slots__ = ("_journal",)

    def __init__(self) -> None:
        self._journal = _PhaseBReadinessJournalFoundation()

    @property
    def root(self) -> Path:
        return self._journal.root

    def publish(
        self,
        foundation: PhaseBDurableFoundation,
        *,
        observation: _PhaseBReadinessObservation,
        current_release_revision: str,
        now_unix: int,
    ) -> PhaseBReadinessReceipt:
        self._journal._require_writer_authority()
        if not isinstance(observation, _PhaseBReadinessObservation):
            raise TypeError("_PhaseBReadinessObservation is required")
        _require_trusted_readiness_observation(observation)
        with self._journal._exclusive_lock(foundation):
            return self._journal._append_unlocked(
                foundation,
                observation=observation,
                current_release_revision=current_release_revision,
                now_unix=now_unix,
            )


def _build_phase_b_readiness_receipt_mapping(
    foundation: PhaseBDurableFoundation,
    *,
    observation: _PhaseBReadinessObservation,
    current_release_revision: str,
    sequence: int,
    previous_receipt_sha256: str | None,
    now_unix: int | None = None,
) -> dict[str, Any]:
    """Build validated bytes for publication, never an in-memory authority."""

    _require_durable_foundation(foundation)
    if not isinstance(observation, _PhaseBReadinessObservation):
        raise TypeError("_PhaseBReadinessObservation is required")
    _require_trusted_readiness_observation(observation)
    current = int(time.time()) if now_unix is None else now_unix
    revision = _revision(current_release_revision)
    validated_observation = _PhaseBReadinessObservation.from_mapping(
        observation.to_mapping(),
        foundation=foundation,
        current_release_revision=revision,
        now_unix=current,
    )
    expires_at = (
        validated_observation.observed_at_unix
        + PHASE_B_READINESS_MAX_AGE_SECONDS
    )
    unsigned = {
        "schema": PHASE_B_READINESS_RECEIPT_SCHEMA,
        "ok": True,
        "state": "ready",
        "safe_to_start": True,
        "current_release_revision": revision,
        "foundation_generation_sha256": foundation.generation_sha256,
        "foundation_terminal_receipt_sha256": foundation.terminal_receipt[
            "receipt_sha256"
        ],
        "sequence": sequence,
        "previous_receipt_sha256": previous_receipt_sha256,
        "readiness_observation": validated_observation.to_mapping(),
        "readiness_observation_sha256": validated_observation.sha256,
        "observed_at_unix": validated_observation.observed_at_unix,
        "issued_at_unix": current,
        "expires_at_unix": expires_at,
        "secret_material_recorded": False,
    }
    return _validate_phase_b_readiness_receipt_mapping(
        {**unsigned, "receipt_sha256": _sha256_json(unsigned)},
        foundation=foundation,
        current_release_revision=revision,
        expected_sequence=sequence,
        expected_previous_receipt_sha256=previous_receipt_sha256,
        now_unix=current,
    )


def validate_published_phase_b_readiness_receipt(
    value: Any,
    *,
    foundation: PhaseBDurableFoundation,
    journal: AppendOnlyPhaseBReadinessJournal,
    expected_current_release_revision: str,
    now_unix: int | None = None,
) -> PhaseBReadinessReceipt:
    """Validate only the current head authenticated by the trusted journal."""

    if not isinstance(journal, AppendOnlyPhaseBReadinessJournal):
        raise PhaseBError("phase_b_readiness_journal_missing")
    with journal._journal._shared_lock(foundation):
        latest = journal._journal._latest_fresh_unlocked(
            foundation,
            expected_current_release_revision=(
                expected_current_release_revision
            ),
            now_unix=now_unix,
        )
        if (
            not isinstance(value, Mapping)
            or _json_copy(value) != latest.to_mapping()
        ):
            raise PhaseBError("phase_b_readiness_receipt_not_current_head")
        return latest


def execute_approved_phase_b(
    plan: PhaseBPlan,
    *,
    approval: Any,
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
    approval_values = _approval_chain_values(approval)
    entries = journal.load(plan)
    if entries and entries[-1].event == "terminal":
        durable = load_durable_phase_b_foundation(
            plan,
            approval=approval_values,
            journal=journal,
        )
        return _json_copy(durable.terminal_receipt)

    approvals = validate_phase_b_approval_chain(
        approval_values,
        plan=plan,
        now_unix=int(_clock()),
        require_fresh_head=True,
    )
    approved = approvals[-1]
    deps = _require_dependencies(dependencies)
    artifact = _validate_role_artifact(role_artifact, plan=plan)

    with journal.lock(plan):
        entries = journal.load(plan)
        if entries and entries[-1].event == "terminal":
            durable = load_durable_phase_b_foundation(
                plan,
                approval=approval_values,
                journal=journal,
            )
            return _json_copy(durable.terminal_receipt)
        if not entries:
            journal.append(
                plan,
                approval=approved,
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
        pending_transition = _pending_service_transition(journal, plan)

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
        needs_sql_admin = "role_ready" not in events
        try:
            if needs_sql_admin:
                initial_admin_transition = (
                    "phase_b_role_artifact"
                    if pending_transition == "phase_b_role_artifact"
                    else "temporary_admin_initial_authority"
                )
                _recheck_services(
                    plan, deps, journal,
                    approval=approved,
                    transition=initial_admin_transition,
                    clock=_clock,
                )
                initial_admin_boundary = deps.temporary_admin_factory(plan)
                _revalidate_approval(approved, plan, _clock)
                (
                    initial_admin_authority,
                    initial_admin_secret,
                ) = _provision_temporary_admin(
                    initial_admin_boundary,
                    plan,
                    before_mutation=lambda: _revalidate_approval(
                        approved, plan, _clock
                    ),
                )
                journal.append(
                    plan,
                    approval=approved,
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
                    approval=approved,
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
                    approval=approved,
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
                bootstrap_transition = (
                    "bootstrap_password_self_disable"
                    if pending_transition == "bootstrap_password_self_disable"
                    else (
                        "bootstrap_login_rotation_authority"
                        if "bootstrap_authority" in events
                        else "bootstrap_login_authority"
                    )
                )
                _recheck_services(
                    plan, deps, journal,
                    approval=approved,
                    transition=bootstrap_transition,
                    clock=_clock,
                )
                bootstrap_boundary = deps.bootstrap_login_factory(plan)
                bootstrap_mutation_boundary = bootstrap_boundary
                bootstrap_secret: bytearray | None = None
                try:
                    _revalidate_approval(approved, plan, _clock)
                    (
                        bootstrap_authority,
                        bootstrap_secret,
                    ) = _provision_bootstrap_login(
                        bootstrap_boundary,
                        plan,
                        before_mutation=lambda: _revalidate_approval(
                            approved, plan, _clock
                        ),
                    )
                    journal.append(
                        plan,
                        approval=approved,
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
                    if bootstrap_authority["operation_type"] == "CREATE_USER":
                        # The create proves existence; a separately observed
                        # rotation is the fresh authority consumed by HBA and
                        # self-disable.  Both Cloud mutations remain journaled.
                        _recheck_services(
                            plan,
                            deps,
                            journal,
                            approval=approved,
                            transition="bootstrap_login_rotation_authority",
                            clock=_clock,
                        )
                        _revalidate_approval(approved, plan, _clock)
                        previous_bootstrap_secret = bootstrap_secret
                        (
                            bootstrap_authority,
                            bootstrap_secret,
                        ) = _provision_bootstrap_login(
                            bootstrap_boundary,
                            plan,
                            before_mutation=lambda: _revalidate_approval(
                                approved, plan, _clock
                            ),
                        )
                        if bootstrap_secret is previous_bootstrap_secret:
                            raise PhaseBError(
                                "phase_b_bootstrap_secret_replay_forbidden"
                            )
                        _zeroize(previous_bootstrap_secret)
                        if bootstrap_authority["operation_type"] != "UPDATE_USER":
                            raise PhaseBError(
                                "phase_b_bootstrap_rotation_authority_missing"
                            )
                        journal.append(
                            plan,
                            approval=approved,
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
                        deps.hba_collector(
                            plan,
                            _require_secret(bootstrap_secret),
                            bootstrap_authority,
                        ),
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
                        approval=approved,
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
                        approval=approved,
                        transition="bootstrap_password_self_disable",
                        clock=_clock,
                    )
                    _revalidate_approval(approved, plan, _clock)
                    self_disable_receipt = _validate_self_disable_receipt(
                        deps.bootstrap_self_disable.disable_and_prove_denied(
                            plan=plan,
                            provisional_password=_require_secret(
                                bootstrap_secret
                            ),
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
                        approval=approved,
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
                if admin_session is None:
                    _recheck_services(
                        plan,
                        deps,
                        journal,
                        approval=approved,
                        transition="temporary_admin_initial_authority",
                        clock=_clock,
                    )
                    initial_admin_boundary = deps.temporary_admin_factory(plan)
                    _revalidate_approval(approved, plan, _clock)
                    (
                        initial_admin_authority,
                        initial_admin_secret,
                    ) = _provision_temporary_admin(
                        initial_admin_boundary,
                        plan,
                        before_mutation=lambda: _revalidate_approval(
                            approved, plan, _clock
                        ),
                    )
                    journal.append(
                        plan,
                        approval=approved,
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
                        plan,
                        plan.temporary_admin_username,
                        initial_admin_secret,
                    )
                    if admin_session.username != plan.temporary_admin_username:
                        raise PhaseBError("phase_b_admin_session_invalid")
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
                    approval=approved,
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
                    approval=approved,
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
                approval=approved,
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
            deletion_reacquire_transition = (
                "temporary_admin_delete"
                if _pending_service_transition(journal, plan)
                == "temporary_admin_delete"
                else "temporary_admin_predelete_reacquire"
            )
            _recheck_services(
                plan, deps, journal,
                approval=approved,
                transition=deletion_reacquire_transition,
                clock=_clock,
            )
            deletion_boundary = deps.temporary_admin_factory(plan)
            if deletion_boundary is initial_admin_boundary:
                raise PhaseBError("phase_b_delete_boundary_not_fresh")
            deletion_secret: bytearray | None = None
            try:
                _revalidate_approval(approved, plan, _clock)
                (
                    deletion_admin_authority,
                    deletion_secret,
                ) = _provision_temporary_admin(
                    deletion_boundary,
                    plan,
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
                    approval=approved,
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
                approval=approved,
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
                approval=approved,
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
                expected_bootstrap_authority=post_disable_authority,
                expected_absence_receipt=absence_receipt,
            )
        finally:
            _safe_close(terminal_writer)
        terminal_services = _recheck_services(
            plan, deps, journal,
            approval=approved,
            transition="terminal_observation",
            clock=_clock,
        )
        if _service_stable_projection(terminal_observation["services"]) != _service_stable_projection(
            terminal_services
        ):
            raise PhaseBError("phase_b_terminal_services_drifted")
        terminal_observed_at = int(_clock())
        PhaseBApproval.from_mapping(
            approved.to_mapping(),
            plan=plan,
            now_unix=terminal_observed_at,
        )
        if terminal_observation["observed_at_unix"] > terminal_observed_at:
            raise PhaseBError("phase_b_terminal_observation_in_future")
        journal.append(
            plan,
            approval=approved,
            event="terminal_observed",
            idempotency_key="terminal-observed:" + terminal_observation["observation_sha256"],
            evidence={
                "terminal_observation": terminal_observation,
                "preterminal": True,
                "safe_to_start": False,
            },
            now_unix=terminal_observed_at,
        )
        if initial_admin_authority is None or role_receipt is None or predelete_receipt is None:
            raise PhaseBError("phase_b_terminal_evidence_incomplete")
        terminal_at = int(_clock())
        PhaseBApproval.from_mapping(
            approved.to_mapping(),
            plan=plan,
            now_unix=terminal_at,
        )
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
            now_unix=terminal_at,
        )
        journal.append(
            plan,
            approval=approved,
            event="terminal",
            idempotency_key="terminal:" + terminal_receipt["receipt_sha256"],
            evidence={"terminal_receipt": terminal_receipt},
            now_unix=terminal_at,
        )
        durable = load_durable_phase_b_foundation(
            plan,
            approval=approval_values,
            journal=journal,
        )
        return _json_copy(durable.terminal_receipt)


__all__ = [
    "AppendOnlyPhaseBJournal",
    "AppendOnlyPhaseBReadinessJournal",
    "PHASE_B_APPROVAL_SCHEMA",
    "PHASE_B_APPROVAL_SSHSIG_NAMESPACE",
    "PHASE_B_BOOTSTRAP_CONTINUITY_SCHEMA",
    "PHASE_B_DURABLE_FOUNDATION_SCHEMA",
    "PHASE_B_HBA_RECEIPT_SCHEMA",
    "PHASE_B_PLAN_SCHEMA",
    "PHASE_B_PREFLIGHT_SCHEMA",
    "PHASE_B_PREDELETE_SCHEMA",
    "PHASE_B_READINESS_MAX_AGE_SECONDS",
    "PHASE_B_READINESS_OBSERVATION_SCHEMA",
    "PHASE_B_READINESS_RECEIPT_SCHEMA",
    "PHASE_B_RECOVERY_SCHEMA",
    "PHASE_B_SELF_DISABLE_SCHEMA",
    "PHASE_B_SOURCE_AUTH_SCHEMA",
    "PHASE_B_SOURCE_AUTH_SSHSIG_NAMESPACE",
    "PHASE_B_TERMINAL_OBSERVATION_SCHEMA",
    "PHASE_B_TERMINAL_RECEIPT_SCHEMA",
    "PhaseBApproval",
    "PhaseBBootstrapRuntimeContinuity",
    "PhaseBDependencies",
    "PhaseBDurableFoundation",
    "PhaseBError",
    "PhaseBPlan",
    "PhaseBPreflight",
    "PhaseBRecoveryObservation",
    "PhaseBReadinessReceipt",
    "SELF_DISABLE_SQL",
    "SERVICE_UNITS",
    "build_phase_b_plan",
    "build_phase_b_terminal_bootstrap_continuity",
    "execute_approved_phase_b",
    "load_durable_phase_b_foundation",
    "phase_b_approval_signature_payload",
    "phase_b_source_authentication_signature_payload",
    "validate_phase_b_approval_chain",
    "validate_phase_b_pending_source_authentication",
    "validate_phase_b_source_authentication",
    "validate_published_phase_b_readiness_receipt",
    "verify_phase_b_sshsig",
]
