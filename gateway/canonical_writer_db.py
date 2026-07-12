"""Least-privilege PostgreSQL adapter for the Canonical Brain writer service.

This module deliberately exposes only a catalog of fixed statements.  A
gateway or model payload can select a handler-owned statement and supply typed
values; it cannot submit SQL or retrieve the database credential.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import ipaddress
import json
import os
import re
import socket
import ssl
import stat
import struct
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence


_SSL_REQUEST_CODE = 80877103
_PROTOCOL_VERSION = 196608
_MAX_CREDENTIAL_BYTES = 16_384
_MAX_FRAME_BYTES = 16 * 1024 * 1024
_MAX_QUERY_BYTES = 128 * 1024
_MAX_RESULT_BYTES = 8 * 1024 * 1024
_MAX_FIELD_BYTES = 1024 * 1024
_MAX_ATTESTATION_ROWS = 2048
_MAX_FIXED_TRANSACTION_ATTEMPTS = 3
_RETRYABLE_TRANSACTION_ERRORS = frozenset(
    {
        "database_error_sqlstate:40001",
        "database_error_sqlstate:40P01",
    }
)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_PLACEHOLDER = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_SAFE_FIXED_SQL = re.compile(
    r"^\s*(?:SELECT\s+(?:\*\s+FROM\s+)?|CALL\s+)"
    r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\s*\(\s*"
    r"(?:\{\{[A-Za-z_][A-Za-z0-9_]*\}\}"
    r"(?:\s*,\s*\{\{[A-Za-z_][A-Za-z0-9_]*\}\})*)?\s*\)\s*$",
    re.IGNORECASE,
)
_TABLE_PRIVILEGES = (
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "TRUNCATE",
    "REFERENCES",
    "TRIGGER",
)
_DATABASE_PRIVILEGES = ("CONNECT", "CREATE", "TEMP")
_SCHEMA_PRIVILEGES = ("USAGE", "CREATE")
_SEQUENCE_PRIVILEGES = ("USAGE", "SELECT", "UPDATE")
CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY = 4_841_739_663_211_427_921
CANONICAL_EVENT_LOG_TABLE = "public.canonical_event_log"
CANONICAL_EVENT_LOG_OWNER = "canonical_brain_migration_owner"
CANONICAL_EVENT_LOG_COLUMNS = (
    "event_id:uuid:t::",
    "schema_version:text:t::",
    "event_type:text:t::",
    "occurred_at:timestamp with time zone:t::",
    "case_id:text:t::",
    "source:jsonb:t::",
    "actor:jsonb:t::",
    "subject:jsonb:t::",
    "evidence:jsonb:t::",
    "decision:jsonb:t::",
    "status:jsonb:t::",
    "next_action:jsonb:t::",
    "safety:jsonb:t::",
    "payload:jsonb:t::",
)
CANONICAL_PRIVATE_WRITER_TABLES = (
    "writer_capability_consumptions",
    "writer_capability_grants",
    "writer_capability_revocation_scopes",
    "writer_capability_revocations",
    "writer_event_provenance",
    "writer_public_routeback_targets",
    "writer_routeback_authorizations",
    "writer_routeback_lifecycle_terminals",
    "writer_routeback_terminals",
)


class CanonicalWriterDBError(RuntimeError):
    """Safe-to-report database boundary failure."""


class CredentialSecurityError(CanonicalWriterDBError):
    """Credential source violates the service boundary."""


class PrivilegeAttestationError(CanonicalWriterDBError):
    """The connected role is more privileged than its declared policy."""


class FixedStatementError(CanonicalWriterDBError):
    """A fixed statement or its typed parameters are invalid."""


class PostgresProtocolError(CanonicalWriterDBError):
    """The bounded PostgreSQL protocol exchange failed."""


@dataclass(frozen=True)
class CredentialSource:
    """An explicit service-owned password file or already-open descriptor."""

    expected_uid: int
    path: Path | None = None
    fd: int | None = None
    expected_gid: int | None = None
    allowed_modes: frozenset[int] = frozenset({0o400, 0o600})

    def __post_init__(self) -> None:
        if (self.path is None) == (self.fd is None):
            raise ValueError("credential source requires exactly one of path or fd")
        if self.expected_uid < 0 or (
            self.expected_gid is not None and self.expected_gid < 0
        ):
            raise ValueError("credential owner identity is invalid")
        if not self.allowed_modes or not self.allowed_modes <= {0o400, 0o600}:
            raise ValueError("credential mode policy is invalid")


def _validate_secret_stat(
    file_stat: os.stat_result,
    source: CredentialSource,
) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise CredentialSecurityError("credential_source_not_regular_file")
    if file_stat.st_uid != source.expected_uid:
        raise CredentialSecurityError("credential_owner_mismatch")
    if source.expected_gid is not None and file_stat.st_gid != source.expected_gid:
        raise CredentialSecurityError("credential_group_mismatch")
    mode = stat.S_IMODE(file_stat.st_mode)
    if mode not in source.allowed_modes:
        raise CredentialSecurityError("credential_mode_not_allowed")


def _read_fd_from_start(fd: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset <= _MAX_CREDENTIAL_BYTES:
        chunk = os.pread(fd, min(4096, _MAX_CREDENTIAL_BYTES + 1 - offset), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _read_credential(source: CredentialSource) -> str:
    descriptor: int
    if source.path is not None:
        path = os.fspath(source.path)
        try:
            initial_stat = os.lstat(path)
        except OSError as exc:
            raise CredentialSecurityError("credential_file_unavailable") from exc
        if stat.S_ISLNK(initial_stat.st_mode):
            raise CredentialSecurityError("credential_source_not_regular_file")
        _validate_secret_stat(initial_stat, source)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise CredentialSecurityError("credential_file_unavailable") from exc
    else:
        try:
            descriptor = os.dup(int(source.fd))
            os.set_inheritable(descriptor, False)
        except (OSError, TypeError, ValueError) as exc:
            raise CredentialSecurityError("credential_fd_unavailable") from exc

    try:
        file_stat = os.fstat(descriptor)
        _validate_secret_stat(file_stat, source)
        if source.path is not None:
            try:
                path_stat = os.lstat(source.path)
            except OSError as exc:
                raise CredentialSecurityError("credential_file_unavailable") from exc
            if stat.S_ISLNK(path_stat.st_mode) or (
                path_stat.st_dev,
                path_stat.st_ino,
            ) != (file_stat.st_dev, file_stat.st_ino):
                raise CredentialSecurityError("credential_file_identity_changed")
        raw = _read_fd_from_start(descriptor)
    finally:
        os.close(descriptor)

    if not raw or len(raw) > _MAX_CREDENTIAL_BYTES:
        raise CredentialSecurityError("credential_length_invalid")
    if b"\x00" in raw:
        raise CredentialSecurityError("credential_contains_nul")
    try:
        password = raw.decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise CredentialSecurityError("credential_not_utf8") from exc
    if not password:
        raise CredentialSecurityError("credential_empty")
    return password


def _valid_identifier(value: str, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {label}")
    return value


def _validate_ca_file(path: Path, writer_uid: int) -> None:
    try:
        path_stat = os.lstat(path)
    except OSError as exc:
        raise CredentialSecurityError("database_ca_file_unavailable") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise CredentialSecurityError("database_ca_file_not_regular")
    if path_stat.st_uid not in {0, writer_uid}:
        raise CredentialSecurityError("database_ca_owner_invalid")
    if stat.S_IMODE(path_stat.st_mode) & 0o022:
        raise CredentialSecurityError("database_ca_is_writable_by_group_or_world")


@dataclass(frozen=True)
class WriterDBConfig:
    """Fixed connection coordinates supplied by the writer service."""

    host: str
    port: int
    database: str
    user: str
    ca_file: Path
    credential: CredentialSource
    connect_timeout_seconds: float = 5.0
    io_timeout_seconds: float = 10.0
    application_name: str = "muncho-canonical-writer"

    def __post_init__(self) -> None:
        if not self.host or any(ord(char) < 32 for char in self.host):
            raise ValueError("database host is invalid")
        if not 1 <= self.port <= 65535:
            raise ValueError("database port is invalid")
        _valid_identifier(self.database, "database")
        _valid_identifier(self.user, "database user")
        if self.user.casefold() == "postgres":
            raise PrivilegeAttestationError("postgres_role_forbidden")
        if not 0.1 <= self.connect_timeout_seconds <= 30:
            raise ValueError("database connect timeout is invalid")
        if not 0.1 <= self.io_timeout_seconds <= 60:
            raise ValueError("database IO timeout is invalid")
        if (
            not self.application_name
            or len(self.application_name.encode("utf-8")) > 63
        ):
            raise ValueError("database application name is invalid")
        if any(ord(char) < 32 for char in self.application_name):
            raise ValueError("database application name is invalid")


@dataclass(frozen=True, order=True)
class TablePrivilegeGrant:
    table: str
    privileges: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        parts = self.table.split(".")
        if len(parts) != 2:
            raise ValueError("table must be schema-qualified")
        for part in parts:
            _valid_identifier(part, "table name")
        normalized = tuple(sorted({item.upper() for item in self.privileges}))
        if any(item not in _TABLE_PRIVILEGES for item in normalized):
            raise ValueError("unknown table privilege")
        object.__setattr__(self, "privileges", normalized)


@dataclass(frozen=True, order=True)
class SequencePrivilegeGrant:
    sequence: str
    privileges: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        parts = self.sequence.split(".")
        if len(parts) != 2:
            raise ValueError("sequence must be schema-qualified")
        for part in parts:
            _valid_identifier(part, "sequence name")
        normalized = tuple(sorted({item.upper() for item in self.privileges}))
        if any(item not in _SEQUENCE_PRIVILEGES for item in normalized):
            raise ValueError("unknown sequence privilege")
        object.__setattr__(self, "privileges", normalized)


@dataclass(frozen=True, order=True)
class RoutineIdentity:
    """Deployment-pinned identity of one executable or dependency routine."""

    signature: str
    owner: str
    security_definer: bool
    language: str
    configuration: tuple[str, ...]
    definition_sha256: str
    owner_dangerous: bool = False

    def __post_init__(self) -> None:
        if (
            "(" not in self.signature
            or any(char in self.signature for char in ";\x00\r\n")
            or len(self.signature) > 512
        ):
            raise ValueError("routine identity signature is invalid")
        _valid_identifier(self.owner, "routine owner")
        _valid_identifier(self.language, "routine language")
        if self.owner.casefold() == "postgres":
            raise ValueError("routine owner postgres is forbidden")
        if not re.fullmatch(r"[0-9a-f]{64}", self.definition_sha256):
            raise ValueError("routine definition digest is invalid")
        configuration = tuple(sorted(set(self.configuration)))
        if any(
            not item
            or len(item) > 1024
            or any(ord(char) < 32 for char in item)
            for item in configuration
        ):
            raise ValueError("routine configuration is invalid")
        object.__setattr__(self, "configuration", configuration)


@dataclass(frozen=True)
class CanonicalEventLogIdentity:
    """Pinned structural and authority identity of the append-only truth table."""

    table: str
    owner: str
    owner_dangerous: bool
    relation_kind: str
    persistence: str
    is_partition: bool
    access_method: str
    tablespace_oid: int
    row_security: bool
    force_row_security: bool
    replica_identity: str
    relation_options: tuple[str, ...]
    columns: tuple[str, ...]
    constraints: tuple[str, ...]
    user_triggers: tuple[str, ...]
    rewrite_rules: tuple[str, ...]
    policies: tuple[str, ...]
    inheritance: bool
    non_owner_acl_grants: tuple[str, ...]
    index_count: int
    primary_index_exact: bool

    def __post_init__(self) -> None:
        if self.table != CANONICAL_EVENT_LOG_TABLE:
            raise ValueError("canonical event log table identity is invalid")
        _valid_identifier(self.owner, "canonical event log owner")
        for name, values in (
            ("columns", self.columns),
            ("constraints", self.constraints),
            ("user_triggers", self.user_triggers),
            ("rewrite_rules", self.rewrite_rules),
            ("policies", self.policies),
            ("non_owner_acl_grants", self.non_owner_acl_grants),
            ("relation_options", self.relation_options),
        ):
            normalized = tuple(values)
            if any(
                not isinstance(item, str)
                or not item
                or len(item) > 1024
                or any(ord(char) < 32 for char in item)
                for item in normalized
            ):
                raise ValueError(f"canonical event log {name} identity is invalid")
            object.__setattr__(self, name, normalized)


@dataclass(frozen=True, order=True)
class CanonicalPrivateRelationIdentity:
    """Canonical structural identity for one writer-owned table or sequence."""

    name: str
    owner: str
    owner_dangerous: bool
    relation_kind: str
    persistence: str
    is_partition: bool
    access_method: str
    tablespace_oid: int
    row_security: bool
    force_row_security: bool
    replica_identity: str
    relation_options: tuple[str, ...]
    columns: tuple[str, ...]
    constraints: tuple[str, ...]
    indexes: tuple[str, ...]
    index_owners: tuple[str, ...]
    user_triggers: tuple[str, ...]
    rewrite_rules: tuple[str, ...]
    policies: tuple[str, ...]
    inheritance: bool

    def __post_init__(self) -> None:
        _valid_identifier(self.name, "private relation name")
        _valid_identifier(self.owner, "private relation owner")
        if self.relation_kind not in {"r", "p", "S", "v", "m", "f", "c"}:
            raise ValueError("private relation kind is invalid")
        if self.persistence not in {"p", "u", "t"}:
            raise ValueError("private relation persistence is invalid")
        if not isinstance(self.tablespace_oid, int) or self.tablespace_oid < 0:
            raise ValueError("private relation tablespace is invalid")
        for name in (
            "relation_options",
            "columns",
            "constraints",
            "indexes",
            "index_owners",
            "user_triggers",
            "rewrite_rules",
            "policies",
        ):
            normalized = tuple(getattr(self, name))
            if any(
                not isinstance(item, str)
                or not item
                or len(item) > 64 * 1024
                or "\x00" in item
                for item in normalized
            ):
                raise ValueError(f"private relation {name} identity is invalid")
            object.__setattr__(self, name, normalized)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "owner": self.owner,
            "owner_dangerous": self.owner_dangerous,
            "relation_kind": self.relation_kind,
            "persistence": self.persistence,
            "is_partition": self.is_partition,
            "access_method": self.access_method,
            "tablespace_oid": self.tablespace_oid,
            "row_security": self.row_security,
            "force_row_security": self.force_row_security,
            "replica_identity": self.replica_identity,
            "relation_options": self.relation_options,
            "columns": self.columns,
            "constraints": self.constraints,
            "indexes": self.indexes,
            "index_owners": self.index_owners,
            "user_triggers": self.user_triggers,
            "rewrite_rules": self.rewrite_rules,
            "policies": self.policies,
            "inheritance": self.inheritance,
        }


@dataclass(frozen=True)
class CanonicalPrivateSchemaIdentity:
    """Root-config-pinned identity of the complete private writer schema."""

    schema: str
    owner: str
    owner_dangerous: bool
    relations: tuple[CanonicalPrivateRelationIdentity, ...]

    def __post_init__(self) -> None:
        _valid_identifier(self.schema, "private schema")
        _valid_identifier(self.owner, "private schema owner")
        relations = tuple(sorted(self.relations))
        if len({relation.name for relation in relations}) != len(relations):
            raise ValueError("private schema contains duplicate relation identities")
        object.__setattr__(self, "relations", relations)

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            {
                "schema": self.schema,
                "owner": self.owner,
                "owner_dangerous": self.owner_dangerous,
                "relations": [relation.as_dict() for relation in self.relations],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class WriterPrivilegePolicy:
    """Exact role privileges allowed for this writer deployment."""

    schema: str
    table_grants: tuple[TablePrivilegeGrant, ...] = ()
    sequence_grants: tuple[SequencePrivilegeGrant, ...] = ()
    executable_routines: tuple[str, ...] = ()
    routine_identities: tuple[RoutineIdentity, ...] = ()
    dependency_routine_identities: tuple[RoutineIdentity, ...] = ()
    schema_privileges: tuple[str, ...] = ("USAGE",)
    database_privileges: tuple[str, ...] = ("CONNECT",)
    role_memberships: tuple[str, ...] = ()
    canonical_owner_role: str = "canonical_brain_migration_owner"
    canonical_acl_grantee_role: str = "canonical_brain_writer"
    private_schema_identity_sha256: str = ""
    deployment_lock_key: int = CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY

    def __post_init__(self) -> None:
        _valid_identifier(self.schema, "schema")
        tables = tuple(sorted(self.table_grants))
        if len({grant.table for grant in tables}) != len(tables):
            raise ValueError("duplicate table privilege policy")
        sequences = tuple(sorted(self.sequence_grants))
        if len({grant.sequence for grant in sequences}) != len(sequences):
            raise ValueError("duplicate sequence privilege policy")
        routines = tuple(sorted(set(self.executable_routines)))
        for routine in routines:
            if not routine.startswith(f"{self.schema}.") or "(" not in routine:
                raise ValueError("routine must use canonical schema signature")
            if any(char in routine for char in ";\x00\r\n"):
                raise ValueError("routine signature is invalid")
        identities = tuple(sorted(self.routine_identities))
        if len({identity.signature for identity in identities}) != len(identities):
            raise ValueError("duplicate routine identity policy")
        if {identity.signature for identity in identities} != set(routines):
            raise ValueError("every executable routine requires a pinned identity")
        dependencies = tuple(sorted(self.dependency_routine_identities))
        dependency_signatures = {
            identity.signature for identity in dependencies
        }
        if len(dependency_signatures) != len(dependencies):
            raise ValueError("duplicate dependency routine identity policy")
        if dependency_signatures.intersection(routines):
            raise ValueError("dependency routines must not be executable")
        if any(
            not identity.signature.startswith(f"{self.schema}.")
            for identity in dependencies
        ):
            raise ValueError("dependency routine must use canonical schema signature")
        schema_privileges = tuple(
            sorted({item.upper() for item in self.schema_privileges})
        )
        database_privileges = tuple(
            sorted({item.upper() for item in self.database_privileges})
        )
        if any(item not in _SCHEMA_PRIVILEGES for item in schema_privileges):
            raise ValueError("unknown schema privilege")
        if any(item not in _DATABASE_PRIVILEGES for item in database_privileges):
            raise ValueError("unknown database privilege")
        memberships = tuple(sorted(set(self.role_memberships)))
        for role in memberships:
            _valid_identifier(role, "role membership")
        _valid_identifier(self.canonical_owner_role, "canonical owner role")
        _valid_identifier(
            self.canonical_acl_grantee_role,
            "canonical ACL grantee role",
        )
        if self.canonical_owner_role == self.canonical_acl_grantee_role:
            raise ValueError("canonical owner and ACL grantee roles must differ")
        if not re.fullmatch(r"[0-9a-f]{64}", self.private_schema_identity_sha256):
            raise ValueError("private writer schema identity digest is invalid")
        if (
            isinstance(self.deployment_lock_key, bool)
            or not -(1 << 63) <= self.deployment_lock_key < (1 << 63)
        ):
            raise ValueError("deployment advisory lock key is invalid")
        object.__setattr__(self, "table_grants", tables)
        object.__setattr__(self, "sequence_grants", sequences)
        object.__setattr__(self, "executable_routines", routines)
        object.__setattr__(self, "routine_identities", identities)
        object.__setattr__(self, "dependency_routine_identities", dependencies)
        object.__setattr__(self, "schema_privileges", schema_privileges)
        object.__setattr__(self, "database_privileges", database_privileges)
        object.__setattr__(self, "role_memberships", memberships)


@dataclass(frozen=True)
class PrivilegeAttestation:
    role: str
    superuser: bool = False
    createdb: bool = False
    createrole: bool = False
    replication: bool = False
    bypassrls: bool = False
    table_owner: bool = False
    routine_owner: bool = False
    table_grants: tuple[TablePrivilegeGrant, ...] = ()
    sequence_grants: tuple[SequencePrivilegeGrant, ...] = ()
    executable_routines: tuple[str, ...] = ()
    routine_identities: tuple[RoutineIdentity, ...] = ()
    dependency_routine_identities: tuple[RoutineIdentity, ...] = ()
    schema_privileges: tuple[str, ...] = ()
    database_privileges: tuple[str, ...] = ()
    role_memberships: tuple[str, ...] = ()
    unexpected_privileges: tuple[str, ...] = ()
    public_acl_grants: tuple[str, ...] = ()
    canonical_non_owner_acl_grants: tuple[str, ...] = ()
    canonical_writer_role_inheritors: tuple[str, ...] = ()
    canonical_event_log_identity: CanonicalEventLogIdentity | None = None
    canonical_private_schema_identity: CanonicalPrivateSchemaIdentity | None = None


def _expected_canonical_non_owner_acl_grants(
    policy: WriterPrivilegePolicy,
) -> tuple[str, ...]:
    grants = [
        ":".join((
            "schema",
            policy.schema,
            "",
            policy.canonical_owner_role,
            policy.canonical_acl_grantee_role,
            "USAGE",
            "f",
        ))
    ]
    grants.extend(
        ":".join((
            "function",
            identity.signature,
            "",
            policy.canonical_owner_role,
            policy.canonical_acl_grantee_role,
            "EXECUTE",
            "f",
        ))
        for identity in policy.routine_identities
    )
    return tuple(sorted(grants))


def _validate_canonical_event_log_identity(
    identity: CanonicalEventLogIdentity | None,
) -> None:
    if identity is None:
        raise PrivilegeAttestationError(
            "database_canonical_event_log_identity_missing"
        )
    if (
        identity.owner != CANONICAL_EVENT_LOG_OWNER
        or identity.owner_dangerous
        or identity.relation_kind != "r"
        or identity.persistence != "p"
        or identity.is_partition
        or identity.access_method != "heap"
        or identity.tablespace_oid != 0
        or identity.row_security
        or identity.force_row_security
        or identity.replica_identity != "d"
        or identity.relation_options
        or identity.columns != CANONICAL_EVENT_LOG_COLUMNS
        or identity.constraints != ("PRIMARY KEY (event_id)",)
        or identity.user_triggers
        or identity.rewrite_rules
        or identity.policies
        or identity.inheritance
        or identity.non_owner_acl_grants
        or identity.index_count != 1
        or not identity.primary_index_exact
    ):
        raise PrivilegeAttestationError(
            "database_canonical_event_log_identity_mismatch"
        )


def _validate_private_writer_schema_identity(
    identity: CanonicalPrivateSchemaIdentity | None,
    policy: WriterPrivilegePolicy,
) -> None:
    """Reject owner, relation-surface, and structural drift in private ledgers."""

    if identity is None:
        raise PrivilegeAttestationError(
            "database_private_schema_identity_missing"
        )
    if (
        identity.schema != policy.schema
        or identity.owner != policy.canonical_owner_role
        or identity.owner_dangerous
    ):
        raise PrivilegeAttestationError(
            "database_private_schema_owner_identity_mismatch"
        )
    if tuple(relation.name for relation in identity.relations) != (
        CANONICAL_PRIVATE_WRITER_TABLES
    ):
        raise PrivilegeAttestationError(
            "database_private_schema_relation_set_mismatch"
        )
    expected_index_owner = f"{policy.canonical_owner_role}:f"
    for relation in identity.relations:
        if (
            relation.owner != policy.canonical_owner_role
            or relation.owner_dangerous
            or relation.relation_kind != "r"
            or relation.persistence != "p"
            or relation.is_partition
            or relation.access_method != "heap"
            or relation.tablespace_oid != 0
            or relation.row_security
            or relation.force_row_security
            or relation.replica_identity != "d"
            or relation.relation_options
            or not relation.columns
            or not relation.constraints
            or not relation.indexes
            or len(relation.index_owners) != len(relation.indexes)
            or any(
                owner != expected_index_owner
                for owner in relation.index_owners
            )
            or relation.user_triggers
            or relation.rewrite_rules
            or relation.policies
            or relation.inheritance
        ):
            raise PrivilegeAttestationError(
                "database_private_relation_identity_mismatch:"
                + relation.name
            )
    if identity.sha256 != policy.private_schema_identity_sha256:
        raise PrivilegeAttestationError(
            "database_private_schema_structure_digest_mismatch"
        )


def validate_privilege_attestation(
    attestation: PrivilegeAttestation,
    policy: WriterPrivilegePolicy,
    *,
    expected_user: str,
) -> None:
    """Reject dangerous attributes and every privilege outside exact policy."""

    if attestation.role != expected_user:
        raise PrivilegeAttestationError("database_role_identity_mismatch")
    if attestation.role.casefold() == "postgres":
        raise PrivilegeAttestationError("postgres_role_forbidden")
    _validate_canonical_event_log_identity(
        attestation.canonical_event_log_identity
    )
    _validate_private_writer_schema_identity(
        attestation.canonical_private_schema_identity,
        policy,
    )
    dangerous = {
        "superuser": attestation.superuser,
        "createdb": attestation.createdb,
        "createrole": attestation.createrole,
        "replication": attestation.replication,
        "bypassrls": attestation.bypassrls,
        "table_owner": attestation.table_owner,
        "routine_owner": attestation.routine_owner,
    }
    enabled = sorted(name for name, value in dangerous.items() if value)
    if enabled:
        raise PrivilegeAttestationError(
            "database_role_dangerous_attribute:" + ",".join(enabled)
        )

    actual_tables = tuple(sorted(attestation.table_grants))
    if actual_tables != policy.table_grants:
        raise PrivilegeAttestationError("database_table_privileges_mismatch")
    if tuple(sorted(attestation.sequence_grants)) != policy.sequence_grants:
        raise PrivilegeAttestationError("database_sequence_privileges_mismatch")
    if tuple(sorted(set(attestation.executable_routines))) != policy.executable_routines:
        raise PrivilegeAttestationError("database_routine_privileges_mismatch")
    identities = tuple(sorted(attestation.routine_identities))
    dependencies = tuple(sorted(attestation.dependency_routine_identities))
    if any(
        identity.owner_dangerous or identity.owner.casefold() == "postgres"
        for identity in identities + dependencies
    ):
        raise PrivilegeAttestationError("database_routine_owner_is_dangerous")
    if identities != policy.routine_identities:
        raise PrivilegeAttestationError("database_routine_identity_mismatch")
    if dependencies != policy.dependency_routine_identities:
        raise PrivilegeAttestationError(
            "database_dependency_routine_identity_mismatch"
        )
    for identity in identities + dependencies:
        search_paths = [
            item.split("=", 1)[1]
            for item in identity.configuration
            if item.casefold().startswith("search_path=")
        ]
        if len(search_paths) != 1:
            raise PrivilegeAttestationError(
                "database_routine_search_path_missing"
            )
        path_items = tuple(
            part.strip().strip('"').casefold()
            for part in search_paths[0].split(",")
        )
        if (
            path_items != ("pg_catalog", policy.schema.casefold())
            or any(
                part in {"public", "pg_temp", "$user"}
                for part in path_items
            )
        ):
            raise PrivilegeAttestationError(
                "database_routine_search_path_unsafe"
            )
    if tuple(sorted(set(attestation.schema_privileges))) != policy.schema_privileges:
        raise PrivilegeAttestationError("database_schema_privileges_mismatch")
    if tuple(sorted(set(attestation.database_privileges))) != policy.database_privileges:
        raise PrivilegeAttestationError("database_privileges_mismatch")
    if tuple(sorted(set(attestation.role_memberships))) != policy.role_memberships:
        raise PrivilegeAttestationError("database_role_memberships_mismatch")
    if attestation.unexpected_privileges:
        raise PrivilegeAttestationError("database_out_of_scope_privileges_present")
    if attestation.public_acl_grants:
        raise PrivilegeAttestationError("database_public_acl_grants_present")
    if tuple(sorted(attestation.canonical_non_owner_acl_grants)) != (
        _expected_canonical_non_owner_acl_grants(policy)
    ):
        raise PrivilegeAttestationError(
            "database_canonical_non_owner_acl_grants_mismatch"
        )
    expected_inheritors = (f"{expected_user}:1:t:f",)
    if tuple(sorted(attestation.canonical_writer_role_inheritors)) != (
        expected_inheritors
    ):
        raise PrivilegeAttestationError(
            "database_canonical_writer_role_inheritors_mismatch"
        )


class ParameterKind(str, Enum):
    TEXT = "text"
    JSON = "json"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    UUID = "uuid"


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: ParameterKind = ParameterKind.TEXT
    maximum_bytes: int = 64 * 1024
    nullable: bool = False

    def __post_init__(self) -> None:
        _valid_identifier(self.name, "parameter")
        if not 1 <= self.maximum_bytes <= _MAX_QUERY_BYTES:
            raise ValueError("parameter bound is invalid")


@dataclass(frozen=True)
class FixedStatement:
    """A service-handler-owned statement; callers supply values, never SQL."""

    name: str
    sql_template: str
    parameters: tuple[ParameterSpec, ...] = ()
    returns_rows: bool = True
    command_prefixes: tuple[str, ...] = ("SELECT",)
    maximum_rows: int = 100

    def __post_init__(self) -> None:
        _valid_identifier(self.name, "statement name")
        if len(self.sql_template.encode("utf-8")) > _MAX_QUERY_BYTES:
            raise ValueError("fixed SQL exceeds hard bound")
        if ";" in self.sql_template or "--" in self.sql_template or "/*" in self.sql_template:
            raise ValueError("fixed SQL must be one comment-free statement")
        if not _SAFE_FIXED_SQL.match(self.sql_template):
            raise ValueError("fixed SQL may only invoke a schema-qualified routine")
        names = tuple(spec.name for spec in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("duplicate fixed statement parameter")
        placeholders = tuple(_PLACEHOLDER.findall(self.sql_template))
        if sorted(placeholders) != sorted(names) or len(placeholders) != len(names):
            raise ValueError("fixed SQL placeholders must exactly match parameters")
        prefixes = tuple(prefix.strip().upper() for prefix in self.command_prefixes)
        if not prefixes or any(not prefix or "\x00" in prefix for prefix in prefixes):
            raise ValueError("command prefix policy is invalid")
        if not 0 <= self.maximum_rows <= 10_000:
            raise ValueError("row bound is invalid")
        object.__setattr__(self, "command_prefixes", prefixes)

    @property
    def sha256(self) -> str:
        payload = {
            "name": self.name,
            "sql_template": self.sql_template,
            "parameters": [
                {
                    "name": spec.name,
                    "kind": spec.kind.value,
                    "maximum_bytes": spec.maximum_bytes,
                    "nullable": spec.nullable,
                }
                for spec in self.parameters
            ],
            "returns_rows": self.returns_rows,
            "command_prefixes": self.command_prefixes,
            "maximum_rows": self.maximum_rows,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class StatementCatalog:
    """Immutable name-to-statement catalog constructed by the writer service."""

    def __init__(self, statements: Iterable[FixedStatement]) -> None:
        items = tuple(statements)
        mapping = {statement.name: statement for statement in items}
        if len(mapping) != len(items):
            raise ValueError("duplicate fixed statement name")
        self._statements = MappingProxyType(mapping)

    def get(self, name: str) -> FixedStatement:
        try:
            return self._statements[name]
        except (KeyError, TypeError) as exc:
            raise FixedStatementError("fixed_statement_not_allowed") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._statements))

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            [self._statements[name].sha256 for name in sorted(self._statements)],
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[str | None, ...], ...]
    command_tag: str


def _sql_literal(spec: ParameterSpec, value: Any) -> str:
    if value is None:
        if not spec.nullable:
            raise FixedStatementError(f"parameter_not_nullable:{spec.name}")
        return "NULL"
    if spec.kind is ParameterKind.BOOLEAN:
        if type(value) is not bool:
            raise FixedStatementError(f"parameter_type_invalid:{spec.name}")
        return "TRUE" if value else "FALSE"
    if spec.kind is ParameterKind.INTEGER:
        if type(value) is not int:
            raise FixedStatementError(f"parameter_type_invalid:{spec.name}")
        return str(value)
    if spec.kind is ParameterKind.UUID:
        try:
            rendered = str(uuid.UUID(str(value)))
        except (TypeError, ValueError, AttributeError) as exc:
            raise FixedStatementError(f"parameter_type_invalid:{spec.name}") from exc
        return "'" + rendered + "'::uuid"
    if spec.kind is ParameterKind.JSON:
        try:
            rendered = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise FixedStatementError(f"parameter_type_invalid:{spec.name}") from exc
        suffix = "::jsonb"
    else:
        if not isinstance(value, str):
            raise FixedStatementError(f"parameter_type_invalid:{spec.name}")
        rendered = value
        suffix = "::text"
    encoded = rendered.encode("utf-8")
    if len(encoded) > spec.maximum_bytes or "\x00" in rendered:
        raise FixedStatementError(f"parameter_bound_invalid:{spec.name}")
    # Base64 is used instead of PostgreSQL string escaping so safety does not
    # depend on the server's standard_conforming_strings setting.
    encoded_literal = base64.b64encode(encoded).decode("ascii")
    return (
        "convert_from(decode('"
        + encoded_literal
        + "','base64'),'UTF8')"
        + suffix
    )


def _render_fixed(statement: FixedStatement, parameters: Mapping[str, Any]) -> str:
    expected = {spec.name for spec in statement.parameters}
    if set(parameters) != expected:
        raise FixedStatementError("fixed_statement_parameters_mismatch")
    specs = {spec.name: spec for spec in statement.parameters}
    rendered = _PLACEHOLDER.sub(
        lambda match: _sql_literal(specs[match.group(1)], parameters[match.group(1)]),
        statement.sql_template,
    )
    if len(rendered.encode("utf-8")) > _MAX_QUERY_BYTES:
        raise FixedStatementError("fixed_statement_render_exceeds_bound")
    return rendered


class _Session(Protocol):
    def query(self, sql: str, *, maximum_rows: int) -> QueryResult: ...

    def close(self) -> None: ...


SessionFactory = Callable[[WriterDBConfig], _Session]


class FixedReadOnlyTransaction:
    """One catalog-bound statement executed inside one attested DB snapshot.

    The statement is selected by trusted service code before the transaction is
    opened.  Callers can supply only its typed parameters; they cannot change
    the routine name or submit SQL.  A scope is invalid as soon as its context
    manager exits.
    """

    def __init__(self, session: _Session, statement: FixedStatement) -> None:
        self._session = session
        self._statement = statement
        self._active = True
        self._failed = False

    @property
    def failed(self) -> bool:
        return self._failed

    def query(self, parameters: Mapping[str, Any]) -> QueryResult:
        if not self._active:
            raise FixedStatementError("fixed_read_transaction_not_active")
        try:
            sql = _render_fixed(self._statement, parameters)
            result = self._session.query(
                sql,
                maximum_rows=self._statement.maximum_rows,
            )
            command = result.command_tag.upper()
            if not command.startswith("SELECT"):
                raise PostgresProtocolError("database_command_tag_not_allowed")
            if len(result.rows) > self._statement.maximum_rows:
                raise PostgresProtocolError("database_result_row_bound_exceeded")
            return result
        except BaseException:
            self._failed = True
            raise

    def _invalidate(self) -> None:
        self._active = False


class CanonicalWriterDB:
    """Attested fixed-statement interface used only by writer handlers."""

    def __init__(
        self,
        *,
        config: WriterDBConfig,
        privilege_policy: WriterPrivilegePolicy,
        statements: StatementCatalog,
        _session_factory: SessionFactory | None = None,
    ) -> None:
        self._config = config
        self._policy = privilege_policy
        self._statements = statements
        self._session_factory = _session_factory or _open_postgres_session
        self._startup_attested = False

    @property
    def statement_names(self) -> tuple[str, ...]:
        return self._statements.names

    @property
    def statement_catalog_sha256(self) -> str:
        return self._statements.sha256

    def startup_attest(self) -> PrivilegeAttestation:
        self._startup_attested = False
        session = self._session_factory(self._config)
        try:
            _begin_locked_transaction(session, self._policy)
            try:
                attestation = _collect_privilege_attestation(
                    session,
                    config=self._config,
                    policy=self._policy,
                )
                validate_privilege_attestation(
                    attestation,
                    self._policy,
                    expected_user=self._config.user,
                )
            except BaseException:
                _rollback_quietly(session)
                raise
            _require_command(session, "COMMIT", "COMMIT")
        finally:
            session.close()
        self._startup_attested = True
        return attestation

    def query_fixed(
        self,
        statement_name: str,
        parameters: Mapping[str, Any],
    ) -> QueryResult:
        statement = self._statements.get(statement_name)
        if not statement.returns_rows:
            raise FixedStatementError("fixed_statement_is_execute_only")
        return self._run_fixed(statement, parameters)

    def execute_fixed(
        self,
        statement_name: str,
        parameters: Mapping[str, Any],
    ) -> QueryResult:
        return self._run_fixed(self._statements.get(statement_name), parameters)

    @contextlib.contextmanager
    def projection_read_transaction(
        self,
    ) -> Iterator[FixedReadOnlyTransaction]:
        """Bind the exact projection statement to one SERIALIZABLE snapshot."""

        if not self._startup_attested:
            raise PrivilegeAttestationError("database_startup_attestation_required")
        statement = self._statements.get("op_projection_read_events")
        if not statement.returns_rows or statement.command_prefixes != ("SELECT",):
            raise FixedStatementError("fixed_statement_is_not_read_only")

        session = self._session_factory(self._config)
        transaction_open = True
        scope: FixedReadOnlyTransaction | None = None
        try:
            _begin_locked_transaction(session, self._policy, read_only=True)
            attestation = _collect_privilege_attestation(
                session,
                config=self._config,
                policy=self._policy,
            )
            validate_privilege_attestation(
                attestation,
                self._policy,
                expected_user=self._config.user,
            )
            scope = FixedReadOnlyTransaction(session, statement)
            yield scope
            scope._invalidate()
            if scope.failed:
                raise PostgresProtocolError("fixed_read_transaction_failed")
            _require_command(session, "COMMIT", "COMMIT")
            transaction_open = False
        except BaseException:
            if scope is not None:
                scope._invalidate()
            if transaction_open:
                _rollback_quietly(session)
            raise
        finally:
            if scope is not None:
                scope._invalidate()
            session.close()

    def _run_fixed(
        self,
        statement: FixedStatement,
        parameters: Mapping[str, Any],
    ) -> QueryResult:
        if not self._startup_attested:
            raise PrivilegeAttestationError("database_startup_attestation_required")
        sql = _render_fixed(statement, parameters)
        for attempt in range(_MAX_FIXED_TRANSACTION_ATTEMPTS):
            session = self._session_factory(self._config)
            try:
                _begin_locked_transaction(session, self._policy)
                attestation = _collect_privilege_attestation(
                    session,
                    config=self._config,
                    policy=self._policy,
                )
                validate_privilege_attestation(
                    attestation,
                    self._policy,
                    expected_user=self._config.user,
                )
                result = session.query(sql, maximum_rows=statement.maximum_rows)
                command = result.command_tag.upper()
                if not any(
                    command.startswith(prefix)
                    for prefix in statement.command_prefixes
                ):
                    raise PostgresProtocolError("database_command_tag_not_allowed")
                if len(result.rows) > statement.maximum_rows:
                    raise PostgresProtocolError("database_result_row_bound_exceeded")
                _require_command(session, "COMMIT", "COMMIT")
            except PostgresProtocolError as exc:
                _rollback_quietly(session)
                if (
                    str(exc) in _RETRYABLE_TRANSACTION_ERRORS
                    and attempt + 1 < _MAX_FIXED_TRANSACTION_ATTEMPTS
                ):
                    continue
                raise
            except BaseException:
                _rollback_quietly(session)
                raise
            finally:
                session.close()
            return result
        raise AssertionError("fixed transaction retry bound is unreachable")


def _require_command(
    session: _Session,
    sql: str,
    expected_prefix: str,
) -> QueryResult:
    result = session.query(sql, maximum_rows=0)
    if not result.command_tag.upper().startswith(expected_prefix):
        raise PostgresProtocolError("database_transaction_command_invalid")
    return result


def _begin_locked_transaction(
    session: _Session,
    policy: WriterPrivilegePolicy,
    *,
    read_only: bool = False,
) -> None:
    begin = "BEGIN ISOLATION LEVEL SERIALIZABLE"
    if read_only:
        begin += " READ ONLY"
    _require_command(session, begin, "BEGIN")
    result = session.query(
        "SELECT pg_catalog.pg_advisory_xact_lock_shared("
        + str(policy.deployment_lock_key)
        + ")",
        maximum_rows=1,
    )
    if not result.command_tag.upper().startswith("SELECT") or len(result.rows) != 1:
        raise PostgresProtocolError("database_deployment_lock_failed")


def _rollback_quietly(session: _Session) -> None:
    try:
        _require_command(session, "ROLLBACK", "ROLLBACK")
    except Exception:
        pass


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _bool(value: str | None) -> bool:
    if value == "t":
        return True
    if value == "f":
        return False
    raise PostgresProtocolError("database_attestation_boolean_invalid")


def _collect_privilege_attestation(
    session: _Session,
    *,
    config: WriterDBConfig,
    policy: WriterPrivilegePolicy,
) -> PrivilegeAttestation:
    role_result = session.query(
        "SELECT current_user, r.rolsuper, r.rolcreatedb, r.rolcreaterole, "
        "r.rolreplication, r.rolbypassrls FROM pg_catalog.pg_roles r "
        "WHERE r.rolname = current_user",
        maximum_rows=1,
    )
    if len(role_result.rows) != 1 or len(role_result.rows[0]) != 6:
        raise PostgresProtocolError("database_role_attestation_missing")
    role_row = role_result.rows[0]
    role = role_row[0] or ""
    schema = _sql_string(policy.schema)

    private_schema_result = session.query(
        "SELECT owner.rolname, (owner.rolcanlogin OR owner.rolsuper OR "
        "owner.rolcreatedb OR owner.rolcreaterole OR owner.rolreplication OR "
        "owner.rolbypassrls OR EXISTS (SELECT 1 FROM "
        "pg_catalog.pg_auth_members membership WHERE "
        "membership.roleid = owner.oid OR membership.member = owner.oid)) "
        "FROM pg_catalog.pg_namespace namespace JOIN pg_catalog.pg_roles owner "
        "ON owner.oid = namespace.nspowner WHERE namespace.nspname = "
        + schema,
        maximum_rows=1,
    )
    if (
        len(private_schema_result.rows) != 1
        or len(private_schema_result.rows[0]) != 2
        or not private_schema_result.rows[0][0]
    ):
        raise PostgresProtocolError("database_private_schema_attestation_missing")

    event_log_result = session.query(
        "SELECT owner.rolname, "
        "(owner.rolcanlogin OR owner.rolsuper OR owner.rolcreatedb OR "
        "owner.rolcreaterole OR owner.rolreplication OR owner.rolbypassrls OR "
        "EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members membership "
        "WHERE membership.roleid = owner.oid OR membership.member = owner.oid)), "
        "c.relkind::text, c.relpersistence::text, c.relispartition, "
        "table_method.amname, c.reltablespace::text, c.relrowsecurity, "
        "c.relforcerowsecurity, c.relreplident::text, "
        "array_to_json(coalesce(c.reloptions, ARRAY[]::text[]))::text, "
        "coalesce((SELECT json_agg(format('%s:%s:%s:%s:%s', a.attname, "
        "format_type(a.atttypid, a.atttypmod), a.attnotnull, a.attidentity, "
        "a.attgenerated) ORDER BY a.attnum)::text FROM pg_catalog.pg_attribute a "
        "WHERE a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped), '[]'), "
        "coalesce((SELECT json_agg(pg_get_constraintdef(con.oid, true) "
        "ORDER BY con.oid)::text FROM pg_catalog.pg_constraint con "
        "WHERE con.conrelid = c.oid AND con.contype <> 'n'), '[]'), "
        "coalesce((SELECT json_agg(t.tgname ORDER BY t.tgname)::text "
        "FROM pg_catalog.pg_trigger t WHERE t.tgrelid = c.oid "
        "AND NOT t.tgisinternal), '[]'), "
        "coalesce((SELECT json_agg(r.rulename ORDER BY r.rulename)::text "
        "FROM pg_catalog.pg_rewrite r WHERE r.ev_class = c.oid), '[]'), "
        "coalesce((SELECT json_agg(p.polname ORDER BY p.polname)::text "
        "FROM pg_catalog.pg_policy p WHERE p.polrelid = c.oid), '[]'), "
        "EXISTS (SELECT 1 FROM pg_catalog.pg_inherits i "
        "WHERE i.inhrelid = c.oid OR i.inhparent = c.oid), "
        "coalesce((SELECT json_agg(grant_row.value ORDER BY grant_row.value)::text "
        "FROM (SELECT format('table:%s:%s:%s:%s', "
        "coalesce(grantor.rolname, 'PUBLIC'), "
        "coalesce(grantee.rolname, 'PUBLIC'), acl.privilege_type, "
        "acl.is_grantable) AS value "
        "FROM aclexplode(coalesce(c.relacl, acldefault('r', c.relowner))) acl "
        "LEFT JOIN pg_catalog.pg_roles grantor ON grantor.oid = acl.grantor "
        "LEFT JOIN pg_catalog.pg_roles grantee ON grantee.oid = acl.grantee "
        "WHERE acl.grantee <> c.relowner UNION ALL "
        "SELECT format('column:%s:%s:%s:%s:%s', attribute.attname, "
        "coalesce(grantor.rolname, 'PUBLIC'), "
        "coalesce(grantee.rolname, 'PUBLIC'), acl.privilege_type, "
        "acl.is_grantable) AS value "
        "FROM pg_catalog.pg_attribute attribute CROSS JOIN LATERAL "
        "aclexplode(attribute.attacl) acl "
        "LEFT JOIN pg_catalog.pg_roles grantor ON grantor.oid = acl.grantor "
        "LEFT JOIN pg_catalog.pg_roles grantee ON grantee.oid = acl.grantee "
        "WHERE attribute.attrelid = c.oid AND attribute.attnum > 0 "
        "AND NOT attribute.attisdropped AND acl.grantee <> c.relowner) grant_row), "
        "'[]'), "
        "(SELECT count(*)::text FROM pg_catalog.pg_index event_index "
        "WHERE event_index.indrelid = c.oid), "
        "coalesce((SELECT count(*) = 1 AND bool_and("
        "event_index.indisprimary AND event_index.indisunique AND "
        "NOT event_index.indisexclusion AND event_index.indimmediate AND "
        "event_index.indisvalid AND event_index.indisready AND "
        "event_index.indislive AND NOT event_index.indisclustered AND "
        "NOT event_index.indisreplident AND NOT event_index.indcheckxmin AND "
        "event_index.indnkeyatts = 1 AND event_index.indnatts = 1 AND "
        "event_index.indexprs IS NULL AND event_index.indpred IS NULL AND "
        "ARRAY(SELECT key_part.attnum FROM unnest(event_index.indkey) "
        "WITH ORDINALITY key_part(attnum, ordinal) ORDER BY key_part.ordinal) "
        "= ARRAY[(SELECT a.attnum FROM "
        "pg_catalog.pg_attribute a WHERE a.attrelid = c.oid AND "
        "a.attname = 'event_id')::smallint]::smallint[] AND "
        "index_method.amname = 'btree' AND "
        "ARRAY(SELECT key_part.opclass_oid FROM unnest(event_index.indclass) "
        "WITH ORDINALITY key_part(opclass_oid, ordinal) ORDER BY key_part.ordinal) "
        "= ARRAY[(SELECT operator_class.oid "
        "FROM pg_catalog.pg_opclass operator_class JOIN pg_catalog.pg_am "
        "operator_method ON operator_method.oid = operator_class.opcmethod "
        "WHERE operator_method.amname = 'btree' AND "
        "operator_class.opcintype = 'uuid'::regtype AND "
        "operator_class.opcdefault)::oid]::oid[] AND "
        "ARRAY(SELECT key_part.collation_oid FROM "
        "unnest(event_index.indcollation) WITH ORDINALITY "
        "key_part(collation_oid, ordinal) ORDER BY key_part.ordinal) "
        "= ARRAY[0::oid]::oid[] AND "
        "ARRAY(SELECT key_part.option_value FROM unnest(event_index.indoption) "
        "WITH ORDINALITY key_part(option_value, ordinal) "
        "ORDER BY key_part.ordinal) = ARRAY[0::smallint]::smallint[] AND "
        "index_class.relpersistence = 'p' AND index_class.reloptions IS NULL AND "
        "index_class.reltablespace = 0 AND "
        "pg_catalog.pg_get_userbyid(index_class.relowner) = "
        "'canonical_brain_migration_owner') FROM pg_catalog.pg_index event_index "
        "JOIN pg_catalog.pg_class index_class ON "
        "index_class.oid = event_index.indexrelid JOIN pg_catalog.pg_am "
        "index_method ON index_method.oid = index_class.relam "
        "WHERE event_index.indrelid = c.oid), false) "
        "FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n "
        "ON n.oid = c.relnamespace JOIN pg_catalog.pg_roles owner "
        "ON owner.oid = c.relowner JOIN pg_catalog.pg_am table_method "
        "ON table_method.oid = c.relam WHERE n.nspname = 'public' "
        "AND c.relname = 'canonical_event_log'",
        maximum_rows=1,
    )
    if len(event_log_result.rows) != 1 or len(event_log_result.rows[0]) != 20:
        raise PostgresProtocolError(
            "database_canonical_event_log_attestation_missing"
        )
    event_log_row = event_log_result.rows[0]

    def _identity_items(raw: str | None, label: str) -> tuple[str, ...]:
        try:
            value = json.loads(raw or "")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PostgresProtocolError(
                "database_canonical_event_log_" + label + "_invalid"
            ) from exc
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise PostgresProtocolError(
                "database_canonical_event_log_" + label + "_invalid"
            )
        return tuple(value)

    try:
        canonical_event_log_identity = CanonicalEventLogIdentity(
            table=CANONICAL_EVENT_LOG_TABLE,
            owner=event_log_row[0] or "",
            owner_dangerous=_bool(event_log_row[1]),
            relation_kind=event_log_row[2] or "",
            persistence=event_log_row[3] or "",
            is_partition=_bool(event_log_row[4]),
            access_method=event_log_row[5] or "",
            tablespace_oid=int(event_log_row[6] or "-1"),
            row_security=_bool(event_log_row[7]),
            force_row_security=_bool(event_log_row[8]),
            replica_identity=event_log_row[9] or "",
            relation_options=_identity_items(event_log_row[10], "options"),
            columns=_identity_items(event_log_row[11], "columns"),
            constraints=_identity_items(event_log_row[12], "constraints"),
            user_triggers=_identity_items(event_log_row[13], "triggers"),
            rewrite_rules=_identity_items(event_log_row[14], "rules"),
            policies=_identity_items(event_log_row[15], "policies"),
            inheritance=_bool(event_log_row[16]),
            non_owner_acl_grants=_identity_items(event_log_row[17], "acl"),
            index_count=int(event_log_row[18] or "-1"),
            primary_index_exact=_bool(event_log_row[19]),
        )
    except (TypeError, ValueError) as exc:
        raise PostgresProtocolError(
            "database_canonical_event_log_identity_invalid"
        ) from exc

    def _identity_json_items(raw: str | None, label: str) -> tuple[str, ...]:
        try:
            value = json.loads(raw or "")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PostgresProtocolError(
                "database_private_schema_" + label + "_invalid"
            ) from exc
        if not isinstance(value, list):
            raise PostgresProtocolError(
                "database_private_schema_" + label + "_invalid"
            )
        result: list[str] = []
        for item in value:
            if not isinstance(item, (dict, str)):
                raise PostgresProtocolError(
                    "database_private_schema_" + label + "_invalid"
                )
            try:
                result.append(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if isinstance(item, dict)
                    else item
                )
            except (TypeError, ValueError) as exc:
                raise PostgresProtocolError(
                    "database_private_schema_" + label + "_invalid"
                ) from exc
        return tuple(result)

    private_relations_result = session.query(
        "SELECT class.relname, owner.rolname, (owner.rolcanlogin OR "
        "owner.rolsuper OR owner.rolcreatedb OR owner.rolcreaterole OR "
        "owner.rolreplication OR owner.rolbypassrls OR EXISTS (SELECT 1 "
        "FROM pg_catalog.pg_auth_members membership WHERE "
        "membership.roleid = owner.oid OR membership.member = owner.oid)), "
        "class.relkind::text, class.relpersistence::text, class.relispartition, "
        "coalesce(method.amname, ''), class.reltablespace::text, "
        "class.relrowsecurity, class.relforcerowsecurity, class.relreplident::text, "
        "array_to_json(coalesce(class.reloptions, ARRAY[]::text[]))::text, "
        "coalesce((SELECT json_agg(column_identity.value ORDER BY "
        "column_identity.ordinal)::text FROM (SELECT attribute.attnum AS ordinal, "
        "json_build_object('name', attribute.attname, 'type', "
        "pg_catalog.format_type(attribute.atttypid, attribute.atttypmod), "
        "'not_null', attribute.attnotnull, 'identity', attribute.attidentity, "
        "'generated', attribute.attgenerated, 'has_default', attribute.atthasdef, "
        "'default', coalesce(pg_catalog.pg_get_expr(default_value.adbin, "
        "default_value.adrelid, true), ''), 'has_missing', attribute.atthasmissing, "
        "'missing_value', coalesce(attribute.attmissingval::text, ''), "
        "'is_local', attribute.attislocal, 'inheritance_count', "
        "attribute.attinhcount, 'dimensions', attribute.attndims, 'collation', "
        "CASE WHEN attribute.attcollation = 0 THEN '' ELSE "
        "pg_catalog.format('%I.%I', collation_namespace.nspname, "
        "collation_row.collname) END, 'storage', attribute.attstorage::text, "
        "'statistics_target', "
        "attribute.attstattarget, 'options', coalesce(attribute.attoptions, "
        "ARRAY[]::text[]), 'fdw_options', coalesce(attribute.attfdwoptions, "
        "ARRAY[]::text[])) AS value FROM pg_catalog.pg_attribute attribute "
        "LEFT JOIN pg_catalog.pg_attrdef default_value ON "
        "default_value.adrelid = attribute.attrelid AND "
        "default_value.adnum = attribute.attnum LEFT JOIN "
        "pg_catalog.pg_collation collation_row ON collation_row.oid = "
        "attribute.attcollation LEFT JOIN pg_catalog.pg_namespace "
        "collation_namespace ON collation_namespace.oid = "
        "collation_row.collnamespace WHERE attribute.attrelid = class.oid AND "
        "attribute.attnum > 0 AND NOT attribute.attisdropped) column_identity), "
        "'[]'), coalesce((SELECT json_agg(constraint_identity.value ORDER BY "
        "constraint_identity.sort_key)::text FROM (SELECT "
        "constraint_row.contype::text || ':' || "
        "pg_catalog.pg_get_constraintdef(constraint_row.oid, true) AS sort_key, "
        "json_build_object('type', constraint_row.contype::text, 'definition', "
        "pg_catalog.pg_get_constraintdef(constraint_row.oid, true), 'key_columns', "
        "coalesce((SELECT json_agg(key_attribute.attname ORDER BY "
        "key_part.ordinal) FROM pg_catalog.unnest(constraint_row.conkey) WITH "
        "ORDINALITY key_part(attnum, ordinal) JOIN pg_catalog.pg_attribute "
        "key_attribute ON key_attribute.attrelid = constraint_row.conrelid AND "
        "key_attribute.attnum = key_part.attnum), '[]'::json), "
        "'referenced_relation', coalesce((SELECT "
        "pg_catalog.format('%I.%I', referenced_namespace.nspname, "
        "referenced_class.relname) FROM pg_catalog.pg_class referenced_class "
        "JOIN pg_catalog.pg_namespace referenced_namespace ON "
        "referenced_namespace.oid = referenced_class.relnamespace WHERE "
        "referenced_class.oid = constraint_row.confrelid), ''), "
        "'referenced_columns', coalesce((SELECT json_agg(ref_attribute.attname "
        "ORDER BY ref_part.ordinal) FROM pg_catalog.unnest(constraint_row.confkey) "
        "WITH ORDINALITY ref_part(attnum, ordinal) JOIN "
        "pg_catalog.pg_attribute ref_attribute ON ref_attribute.attrelid = "
        "constraint_row.confrelid AND ref_attribute.attnum = ref_part.attnum), "
        "'[]'::json), 'deferrable', constraint_row.condeferrable, 'deferred', "
        "constraint_row.condeferred, 'validated', constraint_row.convalidated, "
        "'no_inherit', constraint_row.connoinherit, 'is_local', "
        "constraint_row.conislocal, 'inheritance_count', constraint_row.coninhcount, "
        "'parent_oid_zero', constraint_row.conparentid = 0, 'update_action', "
        "constraint_row.confupdtype::text, 'delete_action', "
        "constraint_row.confdeltype::text, 'match_type', "
        "constraint_row.confmatchtype::text) AS value FROM "
        "pg_catalog.pg_constraint constraint_row WHERE constraint_row.conrelid = "
        "class.oid) constraint_identity), '[]'), coalesce((SELECT "
        "json_agg(index_identity.value ORDER BY index_identity.sort_key)::text "
        "FROM (SELECT coalesce(owning_constraint.contype::text, '') || ':' || "
        "coalesce(pg_catalog.pg_get_constraintdef(owning_constraint.oid, true), "
        "'') || ':' || CASE WHEN owning_constraint.oid IS NULL THEN "
        "index_class.relname ELSE '' "
        "END AS sort_key, json_build_object('constraint_type', "
        "coalesce(owning_constraint.contype::text, ''), 'name', CASE WHEN "
        "owning_constraint.oid IS NULL THEN index_class.relname ELSE '' END, "
        "'access_method', index_method.amname, 'unique', index.indisunique, "
        "'primary', index.indisprimary, 'exclusion', index.indisexclusion, "
        "'immediate', index.indimmediate, 'valid', index.indisvalid, 'ready', "
        "index.indisready, 'live', index.indislive, 'clustered', "
        "index.indisclustered, 'replica_identity', index.indisreplident, "
        "'check_xmin', index.indcheckxmin, 'key_attribute_count', "
        "index.indnkeyatts, 'attribute_count', index.indnatts, 'key_columns', "
        "coalesce((SELECT json_agg(coalesce(key_attribute.attname, '') ORDER BY "
        "key_part.ordinal) FROM pg_catalog.unnest(index.indkey) WITH ORDINALITY "
        "key_part(attnum, ordinal) LEFT JOIN pg_catalog.pg_attribute "
        "key_attribute ON key_attribute.attrelid = index.indrelid AND "
        "key_attribute.attnum = key_part.attnum), '[]'::json), 'expressions', "
        "coalesce(pg_catalog.pg_get_expr(index.indexprs, index.indrelid, true), "
        "''), 'predicate', coalesce(pg_catalog.pg_get_expr(index.indpred, "
        "index.indrelid, true), ''), 'operator_classes', coalesce((SELECT "
        "json_agg(pg_catalog.format('%I.%I', operator_namespace.nspname, "
        "operator_class.opcname) ORDER BY operator_part.ordinal) FROM "
        "pg_catalog.unnest(index.indclass) WITH ORDINALITY "
        "operator_part(opclass_oid, ordinal) JOIN pg_catalog.pg_opclass "
        "operator_class ON operator_class.oid = operator_part.opclass_oid JOIN "
        "pg_catalog.pg_namespace operator_namespace ON operator_namespace.oid = "
        "operator_class.opcnamespace), '[]'::json), 'collations', coalesce((SELECT "
        "json_agg(CASE WHEN collation_part.collation_oid = 0 THEN '' ELSE "
        "pg_catalog.format('%I.%I', index_collation_namespace.nspname, "
        "index_collation.collname) END ORDER BY collation_part.ordinal) FROM "
        "pg_catalog.unnest(index.indcollation) WITH ORDINALITY "
        "collation_part(collation_oid, ordinal) LEFT JOIN "
        "pg_catalog.pg_collation index_collation ON index_collation.oid = "
        "collation_part.collation_oid LEFT JOIN pg_catalog.pg_namespace "
        "index_collation_namespace ON index_collation_namespace.oid = "
        "index_collation.collnamespace), '[]'::json), 'options', coalesce((SELECT "
        "json_agg(option_part.option_value ORDER BY option_part.ordinal) FROM "
        "pg_catalog.unnest(index.indoption) WITH ORDINALITY "
        "option_part(option_value, ordinal)), '[]'::json), 'persistence', "
        "index_class.relpersistence::text, 'tablespace_oid', "
        "index_class.reltablespace, 'relation_options', "
        "coalesce(index_class.reloptions, ARRAY[]::text[])) AS value FROM "
        "pg_catalog.pg_index index JOIN pg_catalog.pg_class index_class ON "
        "index_class.oid = index.indexrelid JOIN pg_catalog.pg_am index_method "
        "ON index_method.oid = index_class.relam LEFT JOIN "
        "pg_catalog.pg_constraint owning_constraint ON "
        "owning_constraint.conindid = index.indexrelid WHERE index.indrelid = "
        "class.oid) index_identity), '[]'), coalesce((SELECT "
        "json_agg(pg_catalog.format('%s:%s', index_owner.rolname, CASE WHEN "
        "index_owner.rolcanlogin OR index_owner.rolsuper OR "
        "index_owner.rolcreatedb OR index_owner.rolcreaterole OR "
        "index_owner.rolreplication OR index_owner.rolbypassrls OR EXISTS "
        "(SELECT 1 FROM pg_catalog.pg_auth_members index_membership WHERE "
        "index_membership.roleid = index_owner.oid OR "
        "index_membership.member = index_owner.oid) THEN 't' ELSE 'f' END) "
        "ORDER BY coalesce(owning_constraint.contype::text, ''), "
        "coalesce(pg_catalog.pg_get_constraintdef(owning_constraint.oid, true), "
        "''), CASE WHEN owning_constraint.oid IS NULL THEN index_class.relname "
        "ELSE '' END)::text "
        "FROM pg_catalog.pg_index index JOIN pg_catalog.pg_class index_class "
        "ON index_class.oid = index.indexrelid JOIN pg_catalog.pg_roles "
        "index_owner ON index_owner.oid = index_class.relowner LEFT JOIN "
        "pg_catalog.pg_constraint owning_constraint ON "
        "owning_constraint.conindid = index.indexrelid WHERE index.indrelid = "
        "class.oid), '[]'), coalesce((SELECT json_agg(trigger.tgname ORDER BY "
        "trigger.tgname)::text FROM pg_catalog.pg_trigger trigger WHERE "
        "trigger.tgrelid = class.oid AND NOT trigger.tgisinternal), '[]'), "
        "coalesce((SELECT json_agg(rewrite.rulename ORDER BY rewrite.rulename)::text "
        "FROM pg_catalog.pg_rewrite rewrite WHERE rewrite.ev_class = class.oid), "
        "'[]'), coalesce((SELECT json_agg(policy_row.polname ORDER BY "
        "policy_row.polname)::text FROM pg_catalog.pg_policy policy_row WHERE "
        "policy_row.polrelid = class.oid), '[]'), EXISTS (SELECT 1 FROM "
        "pg_catalog.pg_inherits inheritance WHERE inheritance.inhrelid = "
        "class.oid OR inheritance.inhparent = class.oid) FROM "
        "pg_catalog.pg_class class JOIN pg_catalog.pg_namespace namespace ON "
        "namespace.oid = class.relnamespace JOIN pg_catalog.pg_roles owner ON "
        "owner.oid = class.relowner LEFT JOIN pg_catalog.pg_am method ON "
        "method.oid = class.relam WHERE namespace.nspname = "
        + schema
        + " AND class.relkind IN ('r','p','S','v','m','f','c') "
        "ORDER BY class.relname",
        maximum_rows=_MAX_ATTESTATION_ROWS,
    )
    private_relations: list[CanonicalPrivateRelationIdentity] = []
    for row in private_relations_result.rows:
        if len(row) != 20 or not row[0] or not row[1]:
            raise PostgresProtocolError(
                "database_private_relation_attestation_invalid"
            )
        try:
            private_relations.append(
                CanonicalPrivateRelationIdentity(
                    name=row[0],
                    owner=row[1],
                    owner_dangerous=_bool(row[2]),
                    relation_kind=row[3] or "",
                    persistence=row[4] or "",
                    is_partition=_bool(row[5]),
                    access_method=row[6] or "",
                    tablespace_oid=int(row[7] or "-1"),
                    row_security=_bool(row[8]),
                    force_row_security=_bool(row[9]),
                    replica_identity=row[10] or "",
                    relation_options=_identity_items(row[11], "options"),
                    columns=_identity_json_items(row[12], "columns"),
                    constraints=_identity_json_items(row[13], "constraints"),
                    indexes=_identity_json_items(row[14], "indexes"),
                    index_owners=_identity_json_items(row[15], "index_owners"),
                    user_triggers=_identity_items(row[16], "triggers"),
                    rewrite_rules=_identity_items(row[17], "rules"),
                    policies=_identity_items(row[18], "policies"),
                    inheritance=_bool(row[19]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise PostgresProtocolError(
                "database_private_relation_identity_invalid"
            ) from exc
    try:
        private_schema_identity = CanonicalPrivateSchemaIdentity(
            schema=policy.schema,
            owner=private_schema_result.rows[0][0] or "",
            owner_dangerous=_bool(private_schema_result.rows[0][1]),
            relations=tuple(private_relations),
        )
    except (TypeError, ValueError) as exc:
        raise PostgresProtocolError(
            "database_private_schema_identity_invalid"
        ) from exc

    unexpected: list[str] = []
    tables_result = session.query(
        "SELECT format('%I.%I', n.nspname, c.relname), "
        "pg_get_userbyid(c.relowner) = current_user, "
        + ", ".join(
            f"has_table_privilege(current_user, c.oid, '{privilege}')"
            for privilege in _TABLE_PRIVILEGES
        )
        + " FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n "
        "ON n.oid = c.relnamespace WHERE n.nspname !~ '^pg_' "
        "AND n.nspname <> 'information_schema' "
        "AND c.relkind IN ('r','p','v','m','f') ORDER BY 1",
        maximum_rows=_MAX_ATTESTATION_ROWS,
    )
    table_grants: list[TablePrivilegeGrant] = []
    table_owner = False
    for row in tables_result.rows:
        if len(row) != 2 + len(_TABLE_PRIVILEGES) or not row[0]:
            raise PostgresProtocolError("database_table_attestation_invalid")
        table_owner = table_owner or _bool(row[1])
        privileges = tuple(
            privilege
            for privilege, raw in zip(_TABLE_PRIVILEGES, row[2:], strict=True)
            if _bool(raw)
        )
        if privileges:
            try:
                table_grants.append(TablePrivilegeGrant(row[0], privileges))
            except ValueError:
                unexpected.append("table:" + row[0])

    sequences_result = session.query(
        "SELECT format('%I.%I', n.nspname, c.relname), "
        "pg_get_userbyid(c.relowner) = current_user, "
        + ", ".join(
            f"has_sequence_privilege(current_user, c.oid, '{privilege}')"
            for privilege in _SEQUENCE_PRIVILEGES
        )
        + " FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n "
        "ON n.oid = c.relnamespace WHERE n.nspname !~ '^pg_' "
        "AND n.nspname <> 'information_schema' AND c.relkind = 'S' ORDER BY 1",
        maximum_rows=_MAX_ATTESTATION_ROWS,
    )
    sequence_grants: list[SequencePrivilegeGrant] = []
    for row in sequences_result.rows:
        if len(row) != 2 + len(_SEQUENCE_PRIVILEGES) or not row[0]:
            raise PostgresProtocolError("database_sequence_attestation_invalid")
        if _bool(row[1]):
            unexpected.append("sequence_owner:" + row[0])
        privileges = tuple(
            privilege
            for privilege, raw in zip(_SEQUENCE_PRIVILEGES, row[2:], strict=True)
            if _bool(raw)
        )
        if privileges:
            try:
                sequence_grants.append(SequencePrivilegeGrant(row[0], privileges))
            except ValueError:
                unexpected.append("sequence:" + row[0])

    routines_result = session.query(
        "SELECT format('%I.%I(%s)', n.nspname, p.proname, "
        "pg_catalog.oidvectortypes(p.proargtypes)), "
        "pg_get_userbyid(p.proowner) = current_user, "
        "has_function_privilege(current_user, p.oid, 'EXECUTE'), "
        "owner.rolname, (owner.rolcanlogin OR owner.rolsuper OR EXISTS ("
        "SELECT 1 FROM pg_catalog.pg_auth_members owner_membership WHERE "
        "owner_membership.roleid = owner.oid OR "
        "owner_membership.member = owner.oid)), owner.rolcreatedb, "
        "owner.rolcreaterole, owner.rolreplication, owner.rolbypassrls, "
        "p.prosecdef, language.lanname, "
        "array_to_json(coalesce(p.proconfig, ARRAY[]::text[]))::text, "
        "pg_get_functiondef(p.oid), n.nspname "
        "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n "
        "ON n.oid = p.pronamespace JOIN pg_catalog.pg_roles owner "
        "ON owner.oid = p.proowner JOIN pg_catalog.pg_language language "
        "ON language.oid = p.prolang WHERE n.nspname !~ '^pg_' "
        "AND n.nspname <> 'information_schema' ORDER BY 1",
        maximum_rows=_MAX_ATTESTATION_ROWS,
    )
    routines: list[str] = []
    routine_identities: list[RoutineIdentity] = []
    dependency_routine_identities: list[RoutineIdentity] = []
    routine_owner = False
    for row in routines_result.rows:
        if (
            len(row) != 14
            or not row[0]
            or not row[3]
            or row[11] is None
            or row[12] is None
            or not row[13]
        ):
            raise PostgresProtocolError("database_routine_attestation_invalid")
        routine_owner = routine_owner or _bool(row[1])
        try:
            configuration_value = json.loads(row[11])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PostgresProtocolError(
                "database_routine_configuration_invalid"
            ) from exc
        if not isinstance(configuration_value, list) or any(
            not isinstance(item, str) for item in configuration_value
        ):
            raise PostgresProtocolError(
                "database_routine_configuration_invalid"
            )
        try:
            identity = RoutineIdentity(
                signature=row[0],
                owner=row[3],
                security_definer=_bool(row[9]),
                language=row[10],
                configuration=tuple(configuration_value),
                definition_sha256=hashlib.sha256(
                    row[12].encode("utf-8")
                ).hexdigest(),
                owner_dangerous=any(_bool(value) for value in row[4:9]),
            )
        except (ValueError, UnicodeError) as exc:
            raise PostgresProtocolError(
                "database_routine_identity_invalid"
            ) from exc
        if _bool(row[2]):
            routines.append(row[0])
            routine_identities.append(identity)
        elif row[13] == policy.schema:
            dependency_routine_identities.append(identity)

    membership_result = session.query(
        "WITH RECURSIVE memberships(oid) AS ("
        "SELECT roleid FROM pg_catalog.pg_auth_members "
        "WHERE member = (SELECT oid FROM pg_catalog.pg_roles WHERE rolname = current_user) "
        "UNION SELECT m.roleid FROM pg_catalog.pg_auth_members m "
        "JOIN memberships inherited ON inherited.oid = m.member) "
        "SELECT r.rolname FROM memberships JOIN pg_catalog.pg_roles r "
        "ON r.oid = memberships.oid ORDER BY 1",
        maximum_rows=_MAX_ATTESTATION_ROWS,
    )
    memberships = tuple(row[0] or "" for row in membership_result.rows)

    writer_role = _sql_string(policy.canonical_acl_grantee_role)
    writer_inheritors_result = session.query(
        "WITH RECURSIVE inheritors(oid, depth, admin_path) AS ("
        "SELECT membership.member, 1, membership.admin_option "
        "FROM pg_catalog.pg_auth_members membership "
        "WHERE membership.roleid = (SELECT oid FROM pg_catalog.pg_roles "
        "WHERE rolname = "
        + writer_role
        + ") UNION ALL SELECT membership.member, inherited.depth + 1, "
        "inherited.admin_path OR membership.admin_option "
        "FROM pg_catalog.pg_auth_members membership JOIN inheritors inherited "
        "ON membership.roleid = inherited.oid) "
        "SELECT role.rolname, inheritors.depth, role.rolcanlogin, "
        "inheritors.admin_path FROM inheritors JOIN pg_catalog.pg_roles role "
        "ON role.oid = inheritors.oid ORDER BY 1, 2",
        maximum_rows=_MAX_ATTESTATION_ROWS,
    )
    canonical_writer_role_inheritors: list[str] = []
    for row in writer_inheritors_result.rows:
        if len(row) != 4 or not row[0] or not str(row[1] or "").isdigit():
            raise PostgresProtocolError(
                "database_canonical_writer_role_inheritors_invalid"
            )
        canonical_writer_role_inheritors.append(
            ":".join((
                row[0],
                str(row[1]),
                "t" if _bool(row[2]) else "f",
                "t" if _bool(row[3]) else "f",
            ))
        )

    schema_result = session.query(
        "SELECT n.nspname, pg_get_userbyid(n.nspowner) = current_user, "
        + ", ".join(
            f"has_schema_privilege(current_user, n.oid, '{privilege}')"
            for privilege in _SCHEMA_PRIVILEGES
        )
        + " FROM pg_catalog.pg_namespace n WHERE n.nspname !~ '^pg_' "
        "AND n.nspname <> 'information_schema' ORDER BY 1",
        maximum_rows=_MAX_ATTESTATION_ROWS,
    )
    schema_privileges: tuple[str, ...] = ()
    for row in schema_result.rows:
        if len(row) != 2 + len(_SCHEMA_PRIVILEGES) or not row[0]:
            raise PostgresProtocolError("database_schema_attestation_invalid")
        privileges = tuple(
            privilege
            for privilege, raw in zip(_SCHEMA_PRIVILEGES, row[2:], strict=True)
            if _bool(raw)
        )
        if row[0] == policy.schema:
            schema_privileges = privileges
            if _bool(row[1]):
                unexpected.append("schema_owner:" + row[0])
        elif privileges or _bool(row[1]):
            unexpected.append("schema:" + row[0])

    database_result = session.query(
        "SELECT d.datname, d.datname = current_database(), "
        "pg_get_userbyid(d.datdba) = current_user, "
        + ", ".join(
            f"has_database_privilege(current_user, d.oid, '{privilege}')"
            for privilege in _DATABASE_PRIVILEGES
        )
        + " FROM pg_catalog.pg_database d WHERE d.datallowconn ORDER BY 1",
        maximum_rows=_MAX_ATTESTATION_ROWS,
    )
    database_privileges: tuple[str, ...] = ()
    for row in database_result.rows:
        if len(row) != 3 + len(_DATABASE_PRIVILEGES) or not row[0]:
            raise PostgresProtocolError("database_scope_attestation_invalid")
        privileges = tuple(
            privilege
            for privilege, raw in zip(_DATABASE_PRIVILEGES, row[3:], strict=True)
            if _bool(raw)
        )
        if _bool(row[1]):
            database_privileges = privileges
            if _bool(row[2]):
                unexpected.append("database_owner:" + row[0])
        elif privileges or _bool(row[2]):
            unexpected.append("database:" + row[0])

    tablespace_result = session.query(
        "SELECT s.spcname, pg_get_userbyid(s.spcowner) = current_user, "
        "has_tablespace_privilege(current_user, s.oid, 'CREATE') "
        "FROM pg_catalog.pg_tablespace s ORDER BY 1",
        maximum_rows=_MAX_ATTESTATION_ROWS,
    )
    for row in tablespace_result.rows:
        if len(row) != 3 or not row[0]:
            raise PostgresProtocolError("database_tablespace_attestation_invalid")
        if _bool(row[1]) or _bool(row[2]):
            unexpected.append("tablespace:" + row[0])

    canonical_acl_result = session.query(
        "SELECT object_acl FROM ("
        "SELECT format('schema:%s::%s:%s:%s:%s', n.nspname, "
        "pg_get_userbyid(a.grantor), CASE WHEN a.grantee = 0 THEN 'PUBLIC' "
        "ELSE pg_get_userbyid(a.grantee) END, a.privilege_type, "
        "a.is_grantable) AS object_acl "
        "FROM pg_catalog.pg_namespace n CROSS JOIN LATERAL "
        "aclexplode(coalesce(n.nspacl, acldefault('n', n.nspowner))) a "
        "WHERE n.nspname = "
        + schema
        + " AND a.grantee <> n.nspowner UNION ALL "
        "SELECT format('%s:%s::%s:%s:%s:%s', "
        "CASE WHEN c.relkind = 'S' THEN 'sequence' ELSE 'table' END, "
        "format('%I.%I', n.nspname, c.relname), pg_get_userbyid(a.grantor), "
        "CASE WHEN a.grantee = 0 THEN 'PUBLIC' ELSE pg_get_userbyid(a.grantee) END, "
        "a.privilege_type, a.is_grantable) "
        "FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n "
        "ON n.oid = c.relnamespace CROSS JOIN LATERAL "
        "aclexplode(coalesce(c.relacl, acldefault("
        "CASE WHEN c.relkind = 'S' THEN 'S'::\"char\" ELSE 'r'::\"char\" END, "
        "c.relowner))) a WHERE n.nspname = "
        + schema
        + " AND c.relkind IN ('r','p','S') AND a.grantee <> c.relowner "
        "UNION ALL SELECT format('column:%s:%s:%s:%s:%s:%s', "
        "format('%I.%I', n.nspname, c.relname), attribute.attname, "
        "pg_get_userbyid(a.grantor), CASE WHEN a.grantee = 0 THEN 'PUBLIC' "
        "ELSE pg_get_userbyid(a.grantee) END, a.privilege_type, a.is_grantable) "
        "FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n "
        "ON n.oid = c.relnamespace JOIN pg_catalog.pg_attribute attribute "
        "ON attribute.attrelid = c.oid AND attribute.attnum > 0 "
        "AND NOT attribute.attisdropped CROSS JOIN LATERAL "
        "aclexplode(attribute.attacl) a "
        "WHERE n.nspname = "
        + schema
        + " AND c.relkind IN ('r','p') AND a.grantee <> c.relowner "
        "UNION ALL SELECT format('%s:%s::%s:%s:%s:%s', "
        "CASE WHEN p.prokind = 'p' THEN 'procedure' ELSE 'function' END, "
        "format('%I.%I(%s)', n.nspname, p.proname, "
        "pg_catalog.oidvectortypes(p.proargtypes)), pg_get_userbyid(a.grantor), "
        "CASE WHEN a.grantee = 0 THEN 'PUBLIC' ELSE pg_get_userbyid(a.grantee) END, "
        "a.privilege_type, a.is_grantable) "
        "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n "
        "ON n.oid = p.pronamespace CROSS JOIN LATERAL "
        "aclexplode(coalesce(p.proacl, acldefault('f', p.proowner))) a "
        "WHERE n.nspname = "
        + schema
        + " AND p.prokind IN ('f','p') AND a.grantee <> p.proowner) grants "
        "ORDER BY object_acl",
        maximum_rows=_MAX_ATTESTATION_ROWS,
    )
    canonical_acl_grants = tuple(
        row[0] or "" for row in canonical_acl_result.rows
    )
    public_acl_grants = tuple(
        grant for grant in canonical_acl_grants if ":PUBLIC:" in grant
    )
    return PrivilegeAttestation(
        role=role,
        superuser=_bool(role_row[1]),
        createdb=_bool(role_row[2]),
        createrole=_bool(role_row[3]),
        replication=_bool(role_row[4]),
        bypassrls=_bool(role_row[5]),
        table_owner=table_owner,
        routine_owner=routine_owner,
        table_grants=tuple(sorted(table_grants)),
        sequence_grants=tuple(sorted(sequence_grants)),
        executable_routines=tuple(sorted(routines)),
        routine_identities=tuple(sorted(routine_identities)),
        dependency_routine_identities=tuple(
            sorted(dependency_routine_identities)
        ),
        schema_privileges=schema_privileges,
        database_privileges=database_privileges,
        role_memberships=memberships,
        unexpected_privileges=tuple(sorted(set(unexpected))),
        public_acl_grants=public_acl_grants,
        canonical_non_owner_acl_grants=canonical_acl_grants,
        canonical_writer_role_inheritors=tuple(
            canonical_writer_role_inheritors
        ),
        canonical_event_log_identity=canonical_event_log_identity,
        canonical_private_schema_identity=private_schema_identity,
    )


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    if count < 0 or count > _MAX_FRAME_BYTES:
        raise PostgresProtocolError("database_frame_bound_invalid")
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        try:
            chunk = connection.recv(remaining)
        except (OSError, TimeoutError) as exc:
            raise PostgresProtocolError("database_socket_read_failed") from exc
        if not chunk:
            raise PostgresProtocolError("database_connection_closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(connection: socket.socket) -> tuple[bytes, bytes]:
    message_type = _recv_exact(connection, 1)
    length = struct.unpack("!I", _recv_exact(connection, 4))[0]
    if length < 4 or length - 4 > _MAX_FRAME_BYTES:
        raise PostgresProtocolError("database_frame_exceeds_bound")
    return message_type, _recv_exact(connection, length - 4)


def _send_message(connection: socket.socket, message_type: bytes, payload: bytes) -> None:
    if len(message_type) != 1 or len(payload) > _MAX_FRAME_BYTES:
        raise PostgresProtocolError("database_outbound_frame_exceeds_bound")
    try:
        connection.sendall(message_type + struct.pack("!I", len(payload) + 4) + payload)
    except (OSError, TimeoutError) as exc:
        raise PostgresProtocolError("database_socket_write_failed") from exc


def _error_sqlstate(payload: bytes) -> str:
    offset = 0
    while offset < len(payload) and payload[offset] != 0:
        code = payload[offset : offset + 1]
        end = payload.find(b"\x00", offset + 1)
        if end < 0:
            break
        if code == b"C":
            value = payload[offset + 1 : end].decode("ascii", errors="ignore")
            return value if re.fullmatch(r"[0-9A-Z]{5}", value) else "unknown"
        offset = end + 1
    return "unknown"


def _sasl_fields(value: bytes) -> dict[str, str]:
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PostgresProtocolError("database_scram_payload_invalid") from exc
    fields: dict[str, str] = {}
    for item in text.split(","):
        key, separator, nested = item.partition("=")
        if not separator or len(key) != 1 or key in fields:
            raise PostgresProtocolError("database_scram_payload_invalid")
        fields[key] = nested
    return fields


def _scram_continue(
    connection: socket.socket,
    *,
    password: str,
    client_nonce: str,
    client_first_bare: str,
    server_first_raw: bytes,
) -> bytes:
    fields = _sasl_fields(server_first_raw)
    server_nonce = fields.get("r", "")
    if not server_nonce.startswith(client_nonce) or len(server_nonce) <= len(client_nonce):
        raise PostgresProtocolError("database_scram_nonce_invalid")
    try:
        salt = base64.b64decode(fields.get("s", ""), validate=True)
        iterations = int(fields.get("i", "0"))
    except (ValueError, TypeError) as exc:
        raise PostgresProtocolError("database_scram_parameters_invalid") from exc
    if not salt or not 4096 <= iterations <= 1_000_000:
        raise PostgresProtocolError("database_scram_parameters_invalid")

    client_final_without_proof = f"c=biws,r={server_nonce}"
    try:
        server_first = server_first_raw.decode("utf-8", errors="strict")
        password_bytes = password.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise PostgresProtocolError("database_scram_encoding_invalid") from exc
    authentication_message = (
        f"{client_first_bare},{server_first},{client_final_without_proof}"
    ).encode("utf-8")
    salted_password = hashlib.pbkdf2_hmac(
        "sha256",
        password_bytes,
        salt,
        iterations,
    )
    client_key = hmac.new(salted_password, b"Client Key", hashlib.sha256).digest()
    stored_key = hashlib.sha256(client_key).digest()
    client_signature = hmac.new(
        stored_key,
        authentication_message,
        hashlib.sha256,
    ).digest()
    proof = bytes(
        left ^ right for left, right in zip(client_key, client_signature, strict=True)
    )
    server_key = hmac.new(salted_password, b"Server Key", hashlib.sha256).digest()
    expected_server_signature = hmac.new(
        server_key,
        authentication_message,
        hashlib.sha256,
    ).digest()
    final = (
        client_final_without_proof
        + ",p="
        + base64.b64encode(proof).decode("ascii")
    ).encode("utf-8")
    _send_message(connection, b"p", final)
    return expected_server_signature


def _authenticate(
    connection: socket.socket,
    *,
    user: str,
    password: str,
) -> None:
    client_nonce = ""
    client_first_bare = ""
    expected_server_signature: bytes | None = None
    authenticated = False
    while True:
        message_type, payload = _recv_message(connection)
        if message_type == b"E":
            raise PostgresProtocolError(
                "database_error_sqlstate:" + _error_sqlstate(payload)
            )
        if message_type == b"R":
            if len(payload) < 4:
                raise PostgresProtocolError("database_auth_frame_invalid")
            auth_type = struct.unpack("!I", payload[:4])[0]
            auth_payload = payload[4:]
            if auth_type == 0:
                authenticated = True
            elif auth_type == 3:
                _send_message(connection, b"p", password.encode("utf-8") + b"\x00")
            elif auth_type == 5:
                if len(auth_payload) != 4:
                    raise PostgresProtocolError("database_md5_auth_frame_invalid")
                inner = hashlib.md5(
                    password.encode("utf-8") + user.encode("utf-8"),
                    usedforsecurity=False,
                ).hexdigest()
                outer = hashlib.md5(
                    inner.encode("ascii") + auth_payload,
                    usedforsecurity=False,
                ).hexdigest()
                _send_message(connection, b"p", b"md5" + outer.encode("ascii") + b"\x00")
            elif auth_type == 10:
                mechanisms = auth_payload.rstrip(b"\x00").split(b"\x00")
                if b"SCRAM-SHA-256" not in mechanisms:
                    raise PostgresProtocolError("database_scram_sha256_required")
                client_nonce = base64.b64encode(os.urandom(24)).decode("ascii")
                client_first_bare = f"n=,r={client_nonce}"
                first = ("n,," + client_first_bare).encode("utf-8")
                initial = b"SCRAM-SHA-256\x00" + struct.pack("!I", len(first)) + first
                _send_message(connection, b"p", initial)
            elif auth_type == 11:
                if not client_nonce or not client_first_bare:
                    raise PostgresProtocolError("database_scram_state_invalid")
                expected_server_signature = _scram_continue(
                    connection,
                    password=password,
                    client_nonce=client_nonce,
                    client_first_bare=client_first_bare,
                    server_first_raw=auth_payload,
                )
            elif auth_type == 12:
                fields = _sasl_fields(auth_payload)
                if "e" in fields or expected_server_signature is None:
                    raise PostgresProtocolError("database_scram_server_rejected")
                try:
                    actual = base64.b64decode(fields.get("v", ""), validate=True)
                except ValueError as exc:
                    raise PostgresProtocolError(
                        "database_scram_signature_invalid"
                    ) from exc
                if not hmac.compare_digest(actual, expected_server_signature):
                    raise PostgresProtocolError("database_scram_signature_mismatch")
            else:
                raise PostgresProtocolError("database_auth_method_not_allowed")
        elif message_type == b"Z":
            if not authenticated:
                raise PostgresProtocolError("database_authentication_incomplete")
            return
        elif message_type not in {b"S", b"K", b"N"}:
            raise PostgresProtocolError("database_startup_message_unexpected")


class _PostgresWireSession:
    def __init__(self, connection: socket.socket) -> None:
        self._connection = connection
        self._closed = False

    def query(self, sql: str, *, maximum_rows: int) -> QueryResult:
        if self._closed:
            raise PostgresProtocolError("database_session_closed")
        try:
            encoded = sql.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise PostgresProtocolError("database_query_encoding_invalid") from exc
        if not encoded or len(encoded) > _MAX_QUERY_BYTES or b"\x00" in encoded:
            raise PostgresProtocolError("database_query_bound_invalid")
        if not 0 <= maximum_rows <= 10_000:
            raise PostgresProtocolError("database_row_bound_invalid")
        _send_message(self._connection, b"Q", encoded + b"\x00")

        columns: tuple[str, ...] = ()
        rows: list[tuple[str | None, ...]] = []
        command_tag = ""
        result_bytes = 0
        while True:
            message_type, payload = _recv_message(self._connection)
            if message_type == b"E":
                raise PostgresProtocolError(
                    "database_error_sqlstate:" + _error_sqlstate(payload)
                )
            if message_type == b"T":
                columns = self._parse_row_description(payload)
            elif message_type == b"D":
                if len(rows) >= maximum_rows:
                    raise PostgresProtocolError("database_result_row_bound_exceeded")
                row, consumed = self._parse_data_row(payload)
                result_bytes += consumed
                if result_bytes > _MAX_RESULT_BYTES:
                    raise PostgresProtocolError("database_result_byte_bound_exceeded")
                if columns and len(row) != len(columns):
                    raise PostgresProtocolError("database_result_shape_invalid")
                rows.append(row)
            elif message_type == b"C":
                if not payload.endswith(b"\x00"):
                    raise PostgresProtocolError("database_command_tag_invalid")
                command_tag = payload[:-1].decode("ascii", errors="strict")
            elif message_type == b"I":
                command_tag = "EMPTY"
            elif message_type == b"Z":
                if not command_tag:
                    raise PostgresProtocolError("database_command_tag_missing")
                return QueryResult(columns, tuple(rows), command_tag)
            elif message_type not in {b"N", b"S"}:
                raise PostgresProtocolError("database_query_message_unexpected")

    @staticmethod
    def _parse_row_description(payload: bytes) -> tuple[str, ...]:
        if len(payload) < 2:
            raise PostgresProtocolError("database_row_description_invalid")
        count = struct.unpack("!H", payload[:2])[0]
        if count > 1024:
            raise PostgresProtocolError("database_column_bound_exceeded")
        offset = 2
        columns: list[str] = []
        for _ in range(count):
            end = payload.find(b"\x00", offset)
            if end < 0 or end - offset > 1024 or end + 19 > len(payload):
                raise PostgresProtocolError("database_row_description_invalid")
            try:
                columns.append(payload[offset:end].decode("utf-8", errors="strict"))
            except UnicodeDecodeError as exc:
                raise PostgresProtocolError(
                    "database_column_name_encoding_invalid"
                ) from exc
            offset = end + 19
        if offset != len(payload):
            raise PostgresProtocolError("database_row_description_invalid")
        return tuple(columns)

    @staticmethod
    def _parse_data_row(payload: bytes) -> tuple[tuple[str | None, ...], int]:
        if len(payload) < 2:
            raise PostgresProtocolError("database_data_row_invalid")
        count = struct.unpack("!H", payload[:2])[0]
        if count > 1024:
            raise PostgresProtocolError("database_column_bound_exceeded")
        offset = 2
        values: list[str | None] = []
        consumed = 0
        for _ in range(count):
            if offset + 4 > len(payload):
                raise PostgresProtocolError("database_data_row_invalid")
            length = struct.unpack("!i", payload[offset : offset + 4])[0]
            offset += 4
            if length == -1:
                values.append(None)
                continue
            if length < 0 or length > _MAX_FIELD_BYTES or offset + length > len(payload):
                raise PostgresProtocolError("database_field_bound_exceeded")
            raw = payload[offset : offset + length]
            try:
                values.append(raw.decode("utf-8", errors="strict"))
            except UnicodeDecodeError as exc:
                raise PostgresProtocolError(
                    "database_field_encoding_invalid"
                ) from exc
            offset += length
            consumed += length
        if offset != len(payload):
            raise PostgresProtocolError("database_data_row_invalid")
        return tuple(values), consumed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            _send_message(self._connection, b"X", b"")
        except CanonicalWriterDBError:
            pass
        finally:
            try:
                self._connection.close()
            except OSError:
                pass


def _tls_server_name(host: str) -> str:
    normalized = host.strip()
    if normalized.startswith("[") or normalized.endswith("]"):
        raise ValueError("database host must not use URL brackets")
    try:
        return str(ipaddress.ip_address(normalized))
    except ValueError:
        if len(normalized) > 253 or normalized.endswith("."):
            raise ValueError("database TLS hostname is invalid")
        labels = normalized.split(".")
        if any(
            not label
            or len(label) > 63
            or label.startswith("-")
            or label.endswith("-")
            or not re.fullmatch(r"[A-Za-z0-9-]+", label)
            for label in labels
        ):
            raise ValueError("database TLS hostname is invalid")
        return normalized


def _open_postgres_session(config: WriterDBConfig) -> _PostgresWireSession:
    _validate_ca_file(config.ca_file, config.credential.expected_uid)
    password = _read_credential(config.credential)
    server_name = _tls_server_name(config.host)
    raw: socket.socket | None = None
    protected: socket.socket | None = None
    ownership_transferred = False
    try:
        raw = socket.create_connection(
            (config.host, config.port),
            timeout=config.connect_timeout_seconds,
        )
        raw.settimeout(config.io_timeout_seconds)
        raw.sendall(struct.pack("!II", 8, _SSL_REQUEST_CODE))
        if _recv_exact(raw, 1) != b"S":
            raise PostgresProtocolError("database_server_refused_tls")

        context = ssl.create_default_context(
            ssl.Purpose.SERVER_AUTH,
            cafile=os.fspath(config.ca_file),
        )
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        protected = context.wrap_socket(raw, server_hostname=server_name)
        raw = None
        protected.settimeout(config.io_timeout_seconds)

        parameters = {
            "user": config.user,
            "database": config.database,
            "application_name": config.application_name,
            "client_encoding": "UTF8",
            # A deterministic catalog rendering is required for pinned
            # routine signatures; no user schema is visible during attestation.
            "options": "-c search_path=pg_catalog -c standard_conforming_strings=on",
        }
        startup_payload = struct.pack("!I", _PROTOCOL_VERSION) + b"".join(
            key.encode("ascii") + b"\x00" + value.encode("utf-8") + b"\x00"
            for key, value in parameters.items()
        ) + b"\x00"
        protected.sendall(struct.pack("!I", len(startup_payload) + 4) + startup_payload)
        _authenticate(protected, user=config.user, password=password)
        ownership_transferred = True
        return _PostgresWireSession(protected)
    except (CanonicalWriterDBError, OSError, ssl.SSLError, UnicodeError) as exc:
        if isinstance(exc, CanonicalWriterDBError):
            raise
        raise PostgresProtocolError("database_connection_failed") from exc
    finally:
        password = ""
        if raw is not None:
            try:
                raw.close()
            except OSError:
                pass
        if protected is not None and not ownership_transferred:
            try:
                protected.close()
            except OSError:
                pass
