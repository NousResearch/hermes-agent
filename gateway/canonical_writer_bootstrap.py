#!/usr/bin/env python3
"""Fail-closed bootstrap for the privileged Canonical Writer service.

The service consumes one explicit, root-owned, secret-free JSON configuration
file. Database password and Discord capability key material remain in separate
strictly owned files; no credential is discovered from environment or remote
secret services.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import stat
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from gateway.canonical_writer_db import (
    CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY,
    CanonicalWriterDB,
    CredentialSource,
    ManagedCloudSQLAdminHBAReceipt,
    RoutineIdentity,
    TablePrivilegeGrant,
    WriterDBConfig,
    WriterPrivilegePolicy,
    managed_cloudsqladmin_hba_receipt_from_mapping,
)
from gateway.canonical_writer_boundary import (
    DEFAULT_GATEWAY_UNIT,
    DEFAULT_SOCKET_PATH,
    current_process_hardening_state,
    harden_current_process_against_dumping,
)
from gateway.canonical_writer_handlers import (
    CanonicalWriterHandlers,
    CanonicalWriterTypedDispatcher,
    ProjectorReadRequest,
    RuntimeContext,
)
from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
    CANONICAL_WRITER_ROLE,
    CANONICAL_WRITER_SCHEMA,
    EXPECTED_HELPER_ROUTINE_SIGNATURES,
    EXPECTED_ROUTINE_SIGNATURES,
    PRODUCTION_STATEMENT_CATALOG,
    PostgresCanonicalWriterBackend,
)
from gateway.discord_edge_protocol import ed25519_public_key_id
from gateway.discord_edge_writer_authority import CanonicalWriterDiscordAuthority
from gateway.canonical_writer_service import (
    CanonicalWriterServer,
    SystemdCgroupV2MainPidProvider,
    SystemdMainPidAuthorizer,
)
from gateway.canonical_writer_readiness import (
    boot_identity,
    current_python_runtime_identity,
    module_file_identity,
    notify_systemd_attestation,
    process_start_time_ticks,
    readiness_receipt_sha256,
    write_runtime_attestation,
)
import gateway.canonical_writer_service as canonical_writer_service_module


_MAX_CONFIG_BYTES = 64 * 1024
_MAX_KEY_BYTES = 8 * 1024
WRITER_RUNTIME_ATTESTATION_VERSION = "canonical-writer-runtime-attestation-v1"
DEFAULT_WRITER_RUNTIME_ATTESTATION_PATH = Path(
    "/run/muncho-canonical-writer/runtime-attestation.json"
)
_SYSTEMD_UNIT = re.compile(r"^[A-Za-z0-9_.@:-]+\.service$")
_FORBIDDEN_SECRET_KEYS = frozenset(
    {
        "password",
        "password_value",
        "secret",
        "secret_value",
        "token",
        "api_key",
        "credential_value",
    }
)
_ROOT_KEYS = frozenset(
    {"service", "database", "privileges", "discord_edge_authority"}
)
_SERVICE_KEYS = frozenset(
    {
        "socket_path",
        "gateway_unit",
        "gateway_uid",
        "writer_uid",
        "writer_gid",
        "socket_gid",
        "projector_gid",
        "owner_discord_user_ids",
        "connection_timeout_seconds",
        "max_connections",
    }
)
_DATABASE_KEYS = frozenset(
    {
        "host",
        "tls_server_name",
        "port",
        "database",
        "user",
        "ca_file",
        "credential_file",
        "connect_timeout_seconds",
        "io_timeout_seconds",
    }
)
_PRIVILEGE_KEYS = frozenset({
    "schema",
    "table_grants",
    "routine_identities",
    "helper_routine_identities",
    "schema_privileges",
    "database_privileges",
    "role_memberships",
    "private_schema_identity_sha256",
    "managed_cloudsqladmin_hba_rejection_receipt",
    "managed_cloudsqladmin_hba_rejection_sha256",
    "deployment_lock_key",
})
_DISCORD_EDGE_AUTHORITY_KEYS = frozenset(
    {
        "enabled",
        "capability_private_key_file",
        "edge_receipt_public_key_file",
        "edge_receipt_public_key_id",
        "request_timeout_seconds",
    }
)


@dataclass(frozen=True)
class DiscordEdgeWriterAuthorityConfig:
    enabled: bool
    capability_private_key_file: Path | None
    edge_receipt_public_key_file: Path | None
    edge_receipt_public_key_id: str
    request_timeout_seconds: int
    capability_private_key: Ed25519PrivateKey | None = field(
        repr=False,
        compare=False,
    )
    edge_receipt_public_key: Ed25519PublicKey | None = field(
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class CanonicalWriterServiceConfig:
    socket_path: Path
    gateway_unit: str
    gateway_uid: int
    writer_uid: int
    writer_gid: int
    socket_gid: int
    projector_gid: int
    owner_discord_user_ids: frozenset[str]
    connection_timeout_seconds: float
    max_connections: int
    database: WriterDBConfig
    privileges: WriterPrivilegePolicy
    discord_edge_authority: DiscordEdgeWriterAuthorityConfig


@dataclass(frozen=True)
class CanonicalWriterBootstrap:
    config: CanonicalWriterServiceConfig
    database: CanonicalWriterDB
    backend: PostgresCanonicalWriterBackend
    handlers: CanonicalWriterHandlers
    server: CanonicalWriterServer


def _strict_mapping(
    value: Any,
    *,
    label: str,
    allowed: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {','.join(unknown)}")
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or any(ord(char) < 32 for char in result):
        raise ValueError(f"{label} is invalid")
    return result


def _required_exact_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} is outside its bound")
    return value


def _number(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} is outside its bound")
    return result


def _absolute_path(value: Any, label: str) -> Path:
    path = Path(_required_text(value, label))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings")
    return tuple(value)


def _routine_identities(
    value: Any,
    *,
    label: str,
) -> tuple[RoutineIdentity, ...]:
    if not isinstance(value, list):
        raise ValueError(f"privileges.{label} must be a list")
    identities = []
    allowed = frozenset(
        {
            "signature",
            "owner",
            "security_definer",
            "language",
            "configuration",
            "definition_sha256",
        }
    )
    for raw in value:
        item = _strict_mapping(raw, label=label, allowed=allowed)
        if type(item.get("security_definer")) is not bool:
            raise ValueError("routine identity security_definer must be boolean")
        identities.append(
            RoutineIdentity(
                signature=_required_text(
                    item.get("signature"),
                    "routine identity signature",
                ),
                owner=_required_text(item.get("owner"), "routine identity owner"),
                security_definer=item["security_definer"],
                language=_required_text(
                    item.get("language"),
                    "routine identity language",
                ),
                configuration=_strings(
                    item.get("configuration"),
                    "routine identity configuration",
                ),
                definition_sha256=_required_text(
                    item.get("definition_sha256"),
                    "routine identity definition_sha256",
                ),
            )
        )
    return tuple(identities)


def _reject_embedded_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().casefold()
            if normalized in _FORBIDDEN_SECRET_KEYS:
                raise ValueError("writer config must not contain secret material")
            _reject_embedded_secrets(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_embedded_secrets(nested)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("writer config contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"writer config contains non-JSON constant: {value}")


def _validate_trusted_config_path(
    path: Path,
    *,
    expected_owner_uid: int,
    require_root_owned_parents: bool,
) -> os.stat_result:
    if not path.is_absolute():
        raise ValueError("writer config path must be absolute")
    try:
        file_stat = os.lstat(path)
    except OSError as exc:
        raise ValueError("writer config is unavailable") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError("writer config must be a regular non-symlink file")
    if file_stat.st_uid != expected_owner_uid:
        raise ValueError("writer config owner is not trusted")
    if stat.S_IMODE(file_stat.st_mode) not in {0o400, 0o440}:
        raise ValueError("writer config mode must be 0400 or 0440")
    if require_root_owned_parents:
        current = path.parent
        while True:
            parent_stat = os.lstat(current)
            if (
                stat.S_ISLNK(parent_stat.st_mode)
                or not stat.S_ISDIR(parent_stat.st_mode)
                or parent_stat.st_uid != expected_owner_uid
                or stat.S_IMODE(parent_stat.st_mode) & 0o022
            ):
                raise ValueError("writer config parent path is not root-controlled")
            if current == current.parent:
                break
            current = current.parent
    return file_stat


def _read_config_bytes(path: Path, expected: os.stat_result) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("writer config cannot be opened") from exc
    try:
        actual = os.fstat(descriptor)
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("writer config identity changed during open")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_CONFIG_BYTES:
            chunk = os.read(descriptor, min(4096, _MAX_CONFIG_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    if not raw or len(raw) > _MAX_CONFIG_BYTES:
        raise ValueError("writer config size is invalid")
    return raw


def _validate_trusted_key_path(
    path: Path,
    *,
    label: str,
    expected_owner_uid: int,
    expected_gid: int,
    expected_mode: int,
    trusted_parent_owner_uid: int,
    require_trusted_parents: bool,
) -> os.stat_result:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be an absolute normalized path")
    try:
        file_stat = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    if file_stat.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one filesystem link")
    if file_stat.st_uid != expected_owner_uid:
        raise ValueError(f"{label} owner is not trusted")
    if file_stat.st_gid != expected_gid:
        raise ValueError(f"{label} group is not the writer group")
    if stat.S_IMODE(file_stat.st_mode) != expected_mode:
        raise ValueError(f"{label} mode must be {expected_mode:04o}")
    if require_trusted_parents:
        current = path.parent
        while True:
            try:
                parent_stat = os.lstat(current)
            except OSError as exc:
                raise ValueError(f"{label} parent path is unavailable") from exc
            if (
                stat.S_ISLNK(parent_stat.st_mode)
                or not stat.S_ISDIR(parent_stat.st_mode)
                or parent_stat.st_uid != trusted_parent_owner_uid
                or stat.S_IMODE(parent_stat.st_mode) & 0o022
            ):
                raise ValueError(f"{label} parent path is not root-controlled")
            if current == current.parent:
                break
            current = current.parent
    return file_stat


def _read_trusted_key_bytes(
    path: Path,
    expected: os.stat_result,
    *,
    label: str,
    expected_owner_uid: int,
    expected_gid: int,
    expected_mode: int,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened") from exc
    try:
        actual = os.fstat(descriptor)
        if (
            (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)
            or not stat.S_ISREG(actual.st_mode)
            or actual.st_nlink != 1
            or actual.st_uid != expected_owner_uid
            or actual.st_gid != expected_gid
            or stat.S_IMODE(actual.st_mode) != expected_mode
        ):
            raise ValueError(f"{label} identity or policy changed during open")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_KEY_BYTES:
            chunk = os.read(descriptor, min(4096, _MAX_KEY_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    if not raw or len(raw) > _MAX_KEY_BYTES:
        raise ValueError(f"{label} size is invalid")
    return raw


def _load_discord_edge_authority_config(
    value: Any,
    *,
    writer_uid: int,
    writer_gid: int,
    trusted_config_owner_uid: int,
    require_trusted_parents: bool,
) -> DiscordEdgeWriterAuthorityConfig:
    raw = _strict_mapping(
        value,
        label="discord_edge_authority",
        allowed=_DISCORD_EDGE_AUTHORITY_KEYS,
    )
    if type(raw.get("enabled")) is not bool:
        raise ValueError("discord_edge_authority.enabled must be boolean")
    enabled = raw["enabled"]
    if not enabled:
        if set(raw) != {"enabled"}:
            raise ValueError(
                "disabled discord_edge_authority must contain only enabled=false"
            )
        return DiscordEdgeWriterAuthorityConfig(
            enabled=False,
            capability_private_key_file=None,
            edge_receipt_public_key_file=None,
            edge_receipt_public_key_id="",
            request_timeout_seconds=15,
            capability_private_key=None,
            edge_receipt_public_key=None,
        )

    if set(raw) != _DISCORD_EDGE_AUTHORITY_KEYS:
        missing = sorted(_DISCORD_EDGE_AUTHORITY_KEYS - set(raw))
        raise ValueError(
            "enabled discord_edge_authority requires exact key configuration:"
            + ",".join(missing)
        )
    private_path = _absolute_path(
        raw["capability_private_key_file"],
        "discord_edge_authority.capability_private_key_file",
    )
    public_path = _absolute_path(
        raw["edge_receipt_public_key_file"],
        "discord_edge_authority.edge_receipt_public_key_file",
    )
    if private_path == public_path:
        raise ValueError("Discord edge writer and receipt keys require distinct files")
    pinned_edge_key_id = _required_text(
        raw["edge_receipt_public_key_id"],
        "discord_edge_authority.edge_receipt_public_key_id",
    )
    if not re.fullmatch(r"[0-9a-f]{64}", pinned_edge_key_id):
        raise ValueError(
            "discord_edge_authority.edge_receipt_public_key_id must be lowercase SHA-256"
        )
    timeout = _integer(
        raw["request_timeout_seconds"],
        "discord_edge_authority.request_timeout_seconds",
        minimum=1,
        maximum=30,
    )
    private_stat = _validate_trusted_key_path(
        private_path,
        label="Discord writer capability private key",
        expected_owner_uid=writer_uid,
        expected_gid=writer_gid,
        expected_mode=0o400,
        trusted_parent_owner_uid=trusted_config_owner_uid,
        require_trusted_parents=require_trusted_parents,
    )
    public_stat = _validate_trusted_key_path(
        public_path,
        label="Discord edge receipt public key",
        expected_owner_uid=trusted_config_owner_uid,
        expected_gid=writer_gid,
        expected_mode=0o440,
        trusted_parent_owner_uid=trusted_config_owner_uid,
        require_trusted_parents=require_trusted_parents,
    )
    private_bytes = _read_trusted_key_bytes(
        private_path,
        private_stat,
        label="Discord writer capability private key",
        expected_owner_uid=writer_uid,
        expected_gid=writer_gid,
        expected_mode=0o400,
    )
    public_bytes = _read_trusted_key_bytes(
        public_path,
        public_stat,
        label="Discord edge receipt public key",
        expected_owner_uid=trusted_config_owner_uid,
        expected_gid=writer_gid,
        expected_mode=0o440,
    )
    try:
        private_key = serialization.load_pem_private_key(
            private_bytes,
            password=None,
        )
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise ValueError(
            "Discord writer capability private key is not unencrypted PEM"
        ) from exc
    try:
        public_key = serialization.load_pem_public_key(public_bytes)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise ValueError("Discord edge receipt public key is not PEM") from exc
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError("Discord writer capability private key must be Ed25519")
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError("Discord edge receipt public key must be Ed25519")
    if private_bytes != private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ):
        raise ValueError(
            "Discord writer capability private key must use exact PKCS#8 PEM encoding"
        )
    if public_bytes != public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ):
        raise ValueError(
            "Discord edge receipt public key must use exact SubjectPublicKeyInfo PEM encoding"
        )
    observed_edge_key_id = ed25519_public_key_id(public_key)
    if observed_edge_key_id != pinned_edge_key_id:
        raise ValueError("Discord edge receipt public key does not match pinned key ID")
    if ed25519_public_key_id(private_key.public_key()) == observed_edge_key_id:
        raise ValueError("Discord writer and edge receipt keys must be distinct")
    return DiscordEdgeWriterAuthorityConfig(
        enabled=True,
        capability_private_key_file=private_path,
        edge_receipt_public_key_file=public_path,
        edge_receipt_public_key_id=pinned_edge_key_id,
        request_timeout_seconds=timeout,
        capability_private_key=private_key,
        edge_receipt_public_key=public_key,
    )


def load_service_config(
    path: str | os.PathLike[str],
    *,
    _expected_owner_uid: int = 0,
    _require_root_owned_parents: bool = True,
) -> CanonicalWriterServiceConfig:
    """Load strict secret-free config from an explicit trusted file."""

    config_path = Path(path)
    trusted_stat = _validate_trusted_config_path(
        config_path,
        expected_owner_uid=_expected_owner_uid,
        require_root_owned_parents=_require_root_owned_parents,
    )
    raw = _read_config_bytes(config_path, trusted_stat)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("writer config is not strict UTF-8 JSON") from exc
    root = _strict_mapping(value, label="config", allowed=_ROOT_KEYS)
    if set(root) != _ROOT_KEYS:
        raise ValueError(
            "writer config requires service, database, privileges, and "
            "discord_edge_authority"
        )
    _reject_embedded_secrets(root)

    service = _strict_mapping(
        root["service"],
        label="service",
        allowed=_SERVICE_KEYS,
    )
    database = _strict_mapping(
        root["database"],
        label="database",
        allowed=_DATABASE_KEYS,
    )
    privileges = _strict_mapping(
        root["privileges"],
        label="privileges",
        allowed=_PRIVILEGE_KEYS,
    )

    writer_uid = _integer(
        service.get("writer_uid"),
        "service.writer_uid",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    writer_gid = _integer(
        service.get("writer_gid"),
        "service.writer_gid",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    socket_gid = _integer(
        service.get("socket_gid"),
        "service.socket_gid",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    projector_gid = _integer(
        service.get("projector_gid"),
        "service.projector_gid",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    gateway_uid = _integer(
        service.get("gateway_uid"),
        "service.gateway_uid",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    if gateway_uid == writer_uid:
        raise ValueError("gateway and writer UIDs must be distinct")
    if len({writer_gid, socket_gid, projector_gid}) != 3:
        raise ValueError(
            "writer, gateway socket, and projector groups must be distinct"
        )
    if (
        stat.S_IMODE(trusted_stat.st_mode) == 0o440
        and trusted_stat.st_gid != writer_gid
    ):
        raise ValueError("group-readable writer config must use writer GID")
    discord_edge_authority = _load_discord_edge_authority_config(
        root["discord_edge_authority"],
        writer_uid=writer_uid,
        writer_gid=writer_gid,
        trusted_config_owner_uid=_expected_owner_uid,
        require_trusted_parents=_require_root_owned_parents,
    )

    gateway_unit = _required_text(service.get("gateway_unit"), "service.gateway_unit")
    if not _SYSTEMD_UNIT.fullmatch(gateway_unit):
        raise ValueError("service.gateway_unit is invalid")
    if gateway_unit != DEFAULT_GATEWAY_UNIT:
        raise ValueError("service.gateway_unit must match the pinned gateway unit")
    schema = _required_text(privileges.get("schema"), "privileges.schema")
    if schema != CANONICAL_WRITER_SCHEMA:
        raise ValueError("privilege schema must match immutable production catalog")

    raw_table_grants = privileges.get("table_grants")
    if not isinstance(raw_table_grants, list):
        raise ValueError("privileges.table_grants must be a list")
    table_grants = []
    for raw_grant in raw_table_grants:
        grant = _strict_mapping(
            raw_grant,
            label="table grant",
            allowed=frozenset({"table", "privileges"}),
        )
        table_grants.append(
            TablePrivilegeGrant(
                table=_required_text(grant.get("table"), "table grant name"),
                privileges=_strings(
                    grant.get("privileges"),
                    "table grant privileges",
                ),
            )
        )
    if table_grants:
        raise ValueError("privileges.table_grants must be empty")

    routine_identities = _routine_identities(
        privileges.get("routine_identities"),
        label="routine_identities",
    )
    helper_identities = _routine_identities(
        privileges.get("helper_routine_identities"),
        label="helper_routine_identities",
    )
    if {identity.signature for identity in helper_identities} != set(
        EXPECTED_HELPER_ROUTINE_SIGNATURES
    ):
        raise ValueError(
            "privileges.helper_routine_identities must match the pinned helper catalog"
        )
    if any(
        identity.owner != CANONICAL_WRITER_MIGRATION_OWNER
        for identity in routine_identities + helper_identities
    ):
        raise ValueError(
            "all writer routine identities must use the pinned migration owner"
        )
    if any(not identity.security_definer for identity in routine_identities):
        raise ValueError("public writer routines must be SECURITY DEFINER")
    if any(identity.security_definer for identity in helper_identities):
        raise ValueError("dependency helper routines must remain SECURITY INVOKER")
    if any(
        identity.configuration != ("search_path=pg_catalog, canonical_brain",)
        for identity in routine_identities + helper_identities
    ):
        raise ValueError(
            "all writer routines must use the exact pinned safe search_path"
        )

    schema_privileges = tuple(
        sorted(
            (
                value.upper()
                for value in _strings(
                    privileges.get("schema_privileges"),
                    "privileges.schema_privileges",
                )
            )
        )
    )
    database_privileges = tuple(
        sorted(
            (
                value.upper()
                for value in _strings(
                    privileges.get("database_privileges"),
                    "privileges.database_privileges",
                )
            )
        )
    )
    role_memberships = tuple(
        sorted(
            _strings(
                privileges.get("role_memberships"),
                "privileges.role_memberships",
            )
        )
    )
    if schema_privileges != ("USAGE",):
        raise ValueError("privileges.schema_privileges must be exactly USAGE")
    if database_privileges != ("CONNECT",):
        raise ValueError("privileges.database_privileges must be exactly CONNECT")
    if role_memberships != (CANONICAL_WRITER_ROLE,):
        raise ValueError(
            "privileges.role_memberships must be exactly the dedicated writer role"
        )
    private_schema_identity_sha256 = _required_text(
        privileges.get("private_schema_identity_sha256"),
        "privileges.private_schema_identity_sha256",
    )
    if not re.fullmatch(r"[0-9a-f]{64}", private_schema_identity_sha256):
        raise ValueError(
            "privileges.private_schema_identity_sha256 must be lowercase SHA-256"
        )
    managed_hba_receipt_raw = privileges.get(
        "managed_cloudsqladmin_hba_rejection_receipt"
    )
    managed_hba_receipt: ManagedCloudSQLAdminHBAReceipt | None = None
    if managed_hba_receipt_raw is not None:
        if not isinstance(managed_hba_receipt_raw, Mapping):
            raise ValueError(
                "privileges.managed_cloudsqladmin_hba_rejection_receipt "
                "must be an object"
            )
        managed_hba_receipt = managed_cloudsqladmin_hba_receipt_from_mapping(
            managed_hba_receipt_raw
        )
    managed_hba_digest = privileges.get(
        "managed_cloudsqladmin_hba_rejection_sha256", ""
    )
    if managed_hba_digest:
        managed_hba_digest = _required_text(
            managed_hba_digest,
            "privileges.managed_cloudsqladmin_hba_rejection_sha256",
        )
        if not re.fullmatch(r"[0-9a-f]{64}", managed_hba_digest):
            raise ValueError(
                "managed cloudsqladmin HBA receipt must be lowercase SHA-256"
            )

    credential_path = _absolute_path(
        database.get("credential_file"),
        "database.credential_file",
    )
    db_config = WriterDBConfig(
        host=_required_exact_text(database.get("host"), "database.host"),
        tls_server_name=_required_exact_text(
            database.get("tls_server_name"),
            "database.tls_server_name",
        ),
        port=_integer(
            database.get("port"),
            "database.port",
            minimum=1,
            maximum=65535,
        ),
        database=_required_text(database.get("database"), "database.database"),
        user=_required_text(database.get("user"), "database.user"),
        ca_file=_absolute_path(database.get("ca_file"), "database.ca_file"),
        credential=CredentialSource(
            path=credential_path,
            expected_uid=writer_uid,
            expected_gid=writer_gid,
            allowed_modes=frozenset({0o400}),
        ),
        connect_timeout_seconds=_number(
            database.get("connect_timeout_seconds", 5.0),
            "database.connect_timeout_seconds",
            minimum=0.1,
            maximum=30,
        ),
        io_timeout_seconds=_number(
            database.get("io_timeout_seconds", 10.0),
            "database.io_timeout_seconds",
            minimum=0.1,
            maximum=60,
        ),
    )
    if managed_hba_receipt is not None and (
        managed_hba_receipt.host != db_config.host
        or managed_hba_receipt.tls_server_name != db_config.tls_server_name
        or managed_hba_receipt.port != db_config.port
        or managed_hba_receipt.user != db_config.user
    ):
        raise ValueError(
            "managed cloudsqladmin HBA receipt does not match database coordinates"
        )
    deployment_lock_key = _integer(
        privileges.get("deployment_lock_key"),
        "privileges.deployment_lock_key",
        minimum=-(1 << 63),
        maximum=(1 << 63) - 1,
    )
    if deployment_lock_key != CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY:
        raise ValueError(
            "privileges.deployment_lock_key must match the pinned writer lock"
        )
    policy = WriterPrivilegePolicy(
        schema=schema,
        table_grants=(),
        sequence_grants=(),
        executable_routines=EXPECTED_ROUTINE_SIGNATURES,
        routine_identities=routine_identities,
        dependency_routine_identities=helper_identities,
        schema_privileges=("USAGE",),
        database_privileges=("CONNECT",),
        role_memberships=(CANONICAL_WRITER_ROLE,),
        private_schema_identity_sha256=private_schema_identity_sha256,
        managed_cloudsqladmin_hba_rejection_receipt=managed_hba_receipt,
        managed_cloudsqladmin_hba_rejection_sha256=managed_hba_digest,
        deployment_lock_key=deployment_lock_key,
    )
    socket_path = _absolute_path(
        service.get("socket_path"),
        "service.socket_path",
    )
    if socket_path != DEFAULT_SOCKET_PATH:
        raise ValueError("service.socket_path must match the pinned writer socket")
    return CanonicalWriterServiceConfig(
        socket_path=socket_path,
        gateway_unit=gateway_unit,
        gateway_uid=gateway_uid,
        writer_uid=writer_uid,
        writer_gid=writer_gid,
        socket_gid=socket_gid,
        projector_gid=projector_gid,
        owner_discord_user_ids=frozenset(
            _required_text(value, "service.owner_discord_user_ids item")
            for value in _strings(
                service.get("owner_discord_user_ids"),
                "service.owner_discord_user_ids",
            )
        ),
        connection_timeout_seconds=_number(
            service.get("connection_timeout_seconds", 30.0),
            "service.connection_timeout_seconds",
            minimum=1,
            maximum=300,
        ),
        max_connections=_integer(
            service.get("max_connections", 8),
            "service.max_connections",
            minimum=1,
            maximum=64,
        ),
        database=db_config,
        privileges=policy,
        discord_edge_authority=discord_edge_authority,
    )


DatabaseFactory = Callable[..., CanonicalWriterDB]


def export_projection_events(
    bootstrap: CanonicalWriterBootstrap,
    output_path: str | os.PathLike[str],
    *,
    limit: int = 200_000,
) -> int:
    """Atomically export bounded canonical rows from inside the writer service.

    The unprivileged projector consumes this derived JSON file and never opens
    a database connection.  External IPC requests cannot set
    ``RuntimeContext.service_internal``; only this in-process writer job can
    request the all-case cursor.
    """

    if isinstance(limit, bool) or not 1 <= int(limit) <= 1_000_000:
        raise ValueError("projection export limit must be between 1 and 1000000")
    target = Path(output_path)
    if not target.is_absolute() or ".." in target.parts:
        raise ValueError("projection export path must be absolute and normalized")
    try:
        parent_stat = target.parent.lstat()
    except OSError as exc:
        raise ValueError("projection export parent is unavailable") from exc
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != bootstrap.config.writer_uid
        or stat.S_IMODE(parent_stat.st_mode) & 0o022
    ):
        raise ValueError("projection export parent is not writer-owned and protected")
    if target.exists() or target.is_symlink():
        target_stat = target.lstat()
        if stat.S_ISLNK(target_stat.st_mode) or not stat.S_ISREG(target_stat.st_mode):
            raise ValueError("projection export target must be a regular file")
        if target_stat.st_uid != bootstrap.config.writer_uid:
            raise ValueError("projection export target owner is invalid")

    projection_scope = getattr(bootstrap.backend, "projection_export_scope", None)
    if not callable(projection_scope):
        raise RuntimeError("writer backend cannot provide a projection snapshot")

    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    count = 0
    seen_event_ids: set[str] = set()
    try:
        descriptor = os.open(temporary, flags, 0o640)
        os.fchown(descriptor, -1, bootstrap.config.projector_gid)
        os.fchmod(descriptor, 0o640)
        with (
            projection_scope() as projection,
            os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle,
        ):
            descriptor = -1
            handle.write('{"events":[')
            cursor = ""
            first = True
            while count < int(limit):
                page_limit = min(500, int(limit) - count)
                response = projection.projector_read(
                    ProjectorReadRequest(
                        case_id="",
                        after_event_id=cursor,
                        limit=page_limit,
                    ),
                    RuntimeContext(
                        request_id=f"projection-export:{os.getpid()}:{count}",
                        platform="writer_service",
                        service_internal=True,
                    ),
                )
                events = response.get("events")
                if not isinstance(events, list):
                    raise RuntimeError("writer projection routine returned no events array")
                if len(events) > page_limit:
                    raise RuntimeError("writer projection routine exceeded its page limit")
                for event in events:
                    if not isinstance(event, Mapping):
                        raise RuntimeError("writer projection routine returned an invalid row")
                    event_id = str(event.get("event_id") or "").strip()
                    try:
                        parsed_event_id = uuid.UUID(event_id)
                    except (AttributeError, ValueError):
                        raise RuntimeError(
                            "writer projection row has an invalid event_id"
                        ) from None
                    if (
                        parsed_event_id.int == 0
                        or str(parsed_event_id) != event_id
                        or event_id in seen_event_ids
                    ):
                        raise RuntimeError(
                            "writer projection row has a duplicate or noncanonical event_id"
                        )
                    seen_event_ids.add(event_id)
                    if not first:
                        handle.write(",")
                    handle.write(
                        json.dumps(
                            dict(event),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    first = False
                    count += 1
                has_more = response.get("has_more")
                if type(has_more) is not bool:
                    raise RuntimeError(
                        "writer projection routine returned invalid pagination metadata"
                    )
                if has_more and not events:
                    raise RuntimeError(
                        "writer projection routine returned an empty nonterminal page"
                    )
                if has_more and count >= int(limit):
                    raise RuntimeError(
                        "writer projection export exceeds its global event limit"
                    )
                if not has_more:
                    break
                next_cursor = str(
                    response.get("next_after_event_id")
                    or events[-1].get("event_id")
                    or ""
                ).strip()
                if (
                    not next_cursor
                    or next_cursor == cursor
                    or next_cursor != str(events[-1].get("event_id") or "").strip()
                ):
                    raise RuntimeError("writer projection cursor did not advance")
                cursor = next_cursor
            handle.write("]}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return count


def build_service(
    config: CanonicalWriterServiceConfig,
    *,
    _database_factory: DatabaseFactory = CanonicalWriterDB,
) -> CanonicalWriterBootstrap:
    """Assemble and startup-attest the privileged service without serving."""

    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if not callable(getuid) or not callable(getgid):
        raise RuntimeError("privileged Canonical writer requires POSIX UID/GID")
    if getuid() != config.writer_uid or getgid() != config.writer_gid:
        raise PermissionError("writer service process UID/GID does not match config")
    authority_config = config.discord_edge_authority
    discord_edge_authority: CanonicalWriterDiscordAuthority | None = None
    if authority_config.enabled:
        if (
            not isinstance(
                authority_config.capability_private_key,
                Ed25519PrivateKey,
            )
            or not isinstance(
                authority_config.edge_receipt_public_key,
                Ed25519PublicKey,
            )
        ):
            raise RuntimeError(
                "enabled Discord route-back lacks writer-owned Ed25519 authority"
            )
        discord_edge_authority = CanonicalWriterDiscordAuthority(
            capability_private_key=authority_config.capability_private_key,
            edge_receipt_public_key=(
                authority_config.edge_receipt_public_key
            ),
            request_timeout_seconds=authority_config.request_timeout_seconds,
        )
    database = _database_factory(
        config=config.database,
        privilege_policy=config.privileges,
        statements=PRODUCTION_STATEMENT_CATALOG,
    )
    database.startup_attest()
    backend = PostgresCanonicalWriterBackend(database)
    handlers = CanonicalWriterHandlers(
        backend,
        discord_edge_authority=discord_edge_authority,
    )
    dispatcher = CanonicalWriterTypedDispatcher(
        handlers,
        owner_user_ids=config.owner_discord_user_ids,
    )
    authorizer = SystemdMainPidAuthorizer(
        config.gateway_unit,
        SystemdCgroupV2MainPidProvider(),
        expected_uid=config.gateway_uid,
    )
    server = CanonicalWriterServer(
        config.socket_path,
        authorizer=authorizer,
        dispatcher=dispatcher,
        expected_socket_gid=config.socket_gid,
        recover_stale_socket=True,
        socket_mode=0o660,
        connection_timeout_seconds=config.connection_timeout_seconds,
        max_connections=config.max_connections,
    )
    return CanonicalWriterBootstrap(
        config=config,
        database=database,
        backend=backend,
        handlers=handlers,
        server=server,
    )


def publish_writer_runtime_readiness(
    bootstrap: CanonicalWriterBootstrap,
    *,
    receipt_path: Path = DEFAULT_WRITER_RUNTIME_ATTESTATION_PATH,
    _now_unix: Callable[[], float] = time.time,
    _boot_identity_provider: Callable[[], tuple[str, int]] = boot_identity,
    _process_start_time: Callable[[int], int] = process_start_time_ticks,
    _notify: Callable[..., bool] = notify_systemd_attestation,
    _process_hardening_provider: Callable[[], tuple[bool, int, int]] = (
        current_process_hardening_state
    ),
    _python_runtime_provider: Callable[[], Mapping[str, Any]] = (
        current_python_runtime_identity
    ),
) -> Mapping[str, Any]:
    """Write and systemd-publish one live, secret-free writer attestation."""

    if bootstrap.server.fileno < 0:
        raise RuntimeError("writer listener is not active")
    socket_stat = bootstrap.config.socket_path.lstat()
    if (
        not stat.S_ISSOCK(socket_stat.st_mode)
        or socket_stat.st_uid != bootstrap.config.writer_uid
        or socket_stat.st_gid != bootstrap.config.socket_gid
        or stat.S_IMODE(socket_stat.st_mode) != 0o660
    ):
        raise RuntimeError("writer socket identity is invalid")
    boot_id_sha256, boottime_ns = _boot_identity_provider()
    pid = os.getpid()
    dumpable, core_soft, core_hard = _process_hardening_provider()
    if dumpable is not False or core_soft != 0 or core_hard != 0:
        raise RuntimeError("writer process hardening attestation is invalid")
    python_runtime = dict(_python_runtime_provider())
    if set(python_runtime) != {
        "effective_import_paths",
        "unexpected_import_paths",
        "loaded_module_origins",
        "unexpected_import_origins",
        "loaded_module_origins_complete",
        "effective_environment_variable_names",
        "effective_environment_variable_value_sha256",
    } or python_runtime.get("loaded_module_origins_complete") is not True:
        raise RuntimeError("writer Python runtime attestation is incomplete")
    for name in (
        "effective_import_paths",
        "unexpected_import_paths",
        "loaded_module_origins",
        "unexpected_import_origins",
        "effective_environment_variable_names",
    ):
        values = python_runtime.get(name)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or values != sorted(set(values))
        ):
            raise RuntimeError("writer Python runtime attestation is invalid")
    environment_names = python_runtime["effective_environment_variable_names"]
    environment_value_sha256 = python_runtime.get(
        "effective_environment_variable_value_sha256"
    )
    if (
        not isinstance(environment_value_sha256, dict)
        or list(environment_value_sha256) != environment_names
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in environment_value_sha256.values()
        )
    ):
        raise RuntimeError("writer Python environment attestation is invalid")
    bootstrap_origin, bootstrap_sha256 = module_file_identity(__file__)
    service_origin, service_sha256 = module_file_identity(
        canonical_writer_service_module.__file__
    )
    observed_at_unix = int(_now_unix())
    if observed_at_unix < 0:
        raise RuntimeError("writer runtime attestation clock is invalid")
    receipt = {
        "version": WRITER_RUNTIME_ATTESTATION_VERSION,
        "observed_at_unix": observed_at_unix,
        "observed_at_boottime_ns": boottime_ns,
        "boot_id_sha256": boot_id_sha256,
        "writer_pid": pid,
        "writer_start_time_ticks": _process_start_time(pid),
        "bootstrap_module_origin": bootstrap_origin,
        "bootstrap_module_sha256": bootstrap_sha256,
        "service_module_origin": service_origin,
        "service_module_sha256": service_sha256,
        "statement_catalog_sha256": bootstrap.database.statement_catalog_sha256,
        "database_identity": CANONICAL_WRITER_MIGRATION_OWNER,
        "database_role": bootstrap.config.database.user,
        "private_schema_identity_sha256": (
            bootstrap.config.privileges.private_schema_identity_sha256
        ),
        "managed_hba_baseline_sha256": (
            bootstrap.config.privileges
            .managed_cloudsqladmin_hba_rejection_sha256
        ),
        "discord_edge_authority_enabled": (
            bootstrap.config.discord_edge_authority.enabled
        ),
        "socket_path": str(bootstrap.config.socket_path),
        "socket_inode": socket_stat.st_ino,
        "socket_device": socket_stat.st_dev,
        "socket_owner_uid": socket_stat.st_uid,
        "socket_group_gid": socket_stat.st_gid,
        "socket_mode": "0660",
        "writer_dumpable": dumpable,
        "writer_core_soft_limit": core_soft,
        "writer_core_hard_limit": core_hard,
        **python_runtime,
    }
    write_runtime_attestation(receipt_path, receipt)
    digest = readiness_receipt_sha256(receipt)
    if not _notify(
        WRITER_RUNTIME_ATTESTATION_VERSION,
        digest,
        ready=True,
    ):
        raise RuntimeError("writer service requires systemd Type=notify")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    harden_current_process_against_dumping()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="absolute root-owned JSON config")
    parser.add_argument(
        "--export-events",
        help="absolute writer-owned atomic event export path instead of serving",
    )
    parser.add_argument("--export-limit", type=int, default=200_000)
    arguments = parser.parse_args(argv)
    config = load_service_config(arguments.config)
    bootstrap = build_service(config)
    if arguments.export_events:
        count = export_projection_events(
            bootstrap,
            arguments.export_events,
            limit=arguments.export_limit,
        )
        print(json.dumps({"success": True, "event_count": count}, sort_keys=True))
        return 0
    bootstrap.server.start()
    try:
        publish_writer_runtime_readiness(bootstrap)
    except BaseException:
        bootstrap.server.shutdown()
        raise

    def _shutdown(_signum, _frame):
        bootstrap.server.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    bootstrap.server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
