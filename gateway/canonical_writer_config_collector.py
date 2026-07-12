#!/usr/bin/env python3
"""Collect and stage one secret-free Canonical Writer canary configuration.

The collector is a root-only, packaged bootstrap boundary.  It reads the
already-provisioned database credential only through :mod:`canonical_writer_db`
and never serializes the value, a content digest, or an error derived from it.
It performs a verified-TLS HBA probe and one serializable, shared-lock,
read-only catalog transaction before staging fixed configuration files.

This module does not manage identities, systemd, services, Discord, or release
artifacts.  It contains no semantic routing or user-text interpretation.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import ctypes
import dataclasses
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway.canonical_writer_bootstrap import load_service_config
from gateway.canonical_writer_boundary import (
    DEFAULT_GATEWAY_UNIT,
    DEFAULT_SOCKET_PATH,
)
from gateway.canonical_writer_db import (
    CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY,
    CredentialSource,
    ManagedCloudSQLAdminHBAReceipt,
    PrivilegeAttestation,
    RoutineIdentity,
    WriterDBConfig,
    WriterPrivilegePolicy,
    _begin_locked_transaction,
    _collect_privilege_attestation,
    _open_postgres_session,
    _require_command,
    _rollback_quietly,
    collect_managed_cloudsqladmin_hba_receipt,
    validate_privilege_attestation,
    validate_tls_server_name,
)
from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
    CANONICAL_WRITER_ROLE,
    CANONICAL_WRITER_SCHEMA,
    EXPECTED_HELPER_ROUTINE_SIGNATURES,
    EXPECTED_ROUTINE_SIGNATURES,
)


COLLECTOR_RECEIPT_SCHEMA = "muncho-writer-config-collector-receipt.v1"
RELEASE_SCHEMA = "muncho-writer-only-release.v1"

SQL_PRIVATE_IP = "10.91.0.3"
SQL_PORT = 5432
SQL_DATABASE = "muncho_canary_brain"
SQL_USER = "muncho_canary_writer_login"
WRITER_UID = 999
WRITER_GID = 994
GATEWAY_UID = 993
SOCKET_CLIENT_GID = 990
PROJECTOR_GID = 991

DATABASE_CA_PATH = Path("/etc/muncho/trust/cloudsql-server-ca.pem")
DATABASE_CREDENTIAL_PATH = Path(
    "/etc/muncho/credentials/canonical-writer-db-password"
)
RELEASE_BASE = Path("/opt/muncho-canary-releases")
STAGING_ROOT = Path("/etc/muncho/writer-activation/staged")
STAGED_WRITER_CONFIG_PATH = STAGING_ROOT / "writer.json"
STAGED_GATEWAY_CONFIG_PATH = STAGING_ROOT / "gateway.yaml"
EVIDENCE_ROOT = Path(
    "/var/lib/muncho-writer-canary-evidence/config-collector"
)

GATEWAY_CONFIG_BYTES = (
    b"canonical_brain:\n"
    b"  writer_boundary:\n"
    b"    enabled: true\n"
    b"  discord_edge:\n"
    b"    enabled: false\n"
    b"plugins:\n"
    b"  enabled: []\n"
    b"  disabled: []\n"
    b"cron:\n"
    b"  provider: builtin\n"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_DISCORD_ID_RE = re.compile(r"^[1-9][0-9]{5,31}$")
_CLOUDSQL_TLS_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.europe-west3\.sql\.goog$"
)
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_MAX_CREDENTIAL_BYTES = 4096
_MAX_RELEASE_FILE_BYTES = 128 * 1024 * 1024
_SAFE_SEARCH_PATH = ("search_path=pg_catalog, canonical_brain",)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_COLLECTOR_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "release_revision",
        "release_artifact_sha256",
        "release_manifest_path",
        "release_manifest_file_sha256",
        "writer_config_path",
        "writer_config_sha256",
        "gateway_config_path",
        "gateway_config_sha256",
        "database",
        "credential_provenance",
        "catalog_attestation_sha256",
        "public_routine_count",
        "helper_routine_count",
        "private_schema_identity_sha256",
        "managed_hba_receipt_sha256",
        "server_certificate_sha256",
        "hba_observed_at_unix",
        "hba_expires_at_unix",
        "discord_edge_enabled",
        "credential_content_or_digest_recorded",
        "collected_at_unix",
        "receipt_sha256",
    }
)
_COLLECTOR_DATABASE_KEYS = frozenset(
    {"host", "tls_server_name", "port", "database", "user", "ca_path", "ca_sha256"}
)
_COLLECTOR_CREDENTIAL_KEYS = frozenset(
    {
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


SessionFactory = Callable[[WriterDBConfig], Any]
HBACollector = Callable[[WriterDBConfig], ManagedCloudSQLAdminHBAReceipt]


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("collector value is not canonical JSON") from exc
    return rendered.encode("utf-8", errors="strict")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _exact_mapping(
    value: Any,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are not exact")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be an exact nonnegative integer")
    return value


def _require_root_linux() -> None:
    if os.geteuid() != 0:
        raise PermissionError("canonical_writer_config_collector_requires_uid_0")
    if sys.platform != "linux":
        raise RuntimeError("canonical_writer_config_collector_requires_linux")


def _list_xattrs(path: Path) -> tuple[str, ...]:
    lister = getattr(os, "listxattr", None)
    if not callable(lister):
        if sys.platform == "linux":
            raise RuntimeError("collector xattr inspection is unavailable")
        return ()
    try:
        return tuple(lister(path, follow_symlinks=False))
    except OSError as exc:
        raise RuntimeError("collector xattr inspection failed") from exc


def _validate_protected_ancestor_chain(path: Path) -> None:
    current = path
    while True:
        item = os.lstat(current)
        if (
            stat.S_ISLNK(item.st_mode)
            or not stat.S_ISDIR(item.st_mode)
            or item.st_uid != 0
            or stat.S_IMODE(item.st_mode) & 0o022
            or _list_xattrs(current)
        ):
            raise PermissionError("collector parent path is not root-controlled")
        if current == current.parent:
            return
        current = current.parent


def _validate_exact_directory(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    item = os.lstat(path)
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISDIR(item.st_mode)
        or item.st_uid != uid
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != mode
        or _list_xattrs(path)
    ):
        raise PermissionError("collector directory identity is not exact")


def _ensure_exact_directory(
    path: Path,
    *,
    uid: int = 0,
    gid: int = 0,
    mode: int = 0o700,
) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("collector directory path is invalid")
    missing: list[Path] = []
    current = path
    while not os.path.lexists(current):
        missing.append(current)
        current = current.parent
    _validate_protected_ancestor_chain(current)
    for item in reversed(missing):
        os.mkdir(item, mode)
        os.chown(item, uid, gid)
        os.chmod(item, mode)
        _fsync_directory(item.parent)
    _validate_exact_directory(path, uid=uid, gid=gid, mode=mode)
    _validate_protected_ancestor_chain(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _trusted_regular_stat(
    path: Path,
    *,
    uid: int,
    gid: int,
    modes: frozenset[int],
    maximum_bytes: int,
    allow_empty: bool = False,
) -> os.stat_result:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("collector trusted path is invalid")
    item = os.lstat(path)
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISREG(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != uid
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) not in modes
        or item.st_size > maximum_bytes
        or (item.st_size == 0 and not allow_empty)
        or _list_xattrs(path)
    ):
        raise PermissionError("collector trusted file identity is not exact")
    _validate_protected_ancestor_chain(path.parent)
    return item


def _read_trusted_public_file(
    path: Path,
    *,
    uid: int,
    gid: int,
    modes: frozenset[int],
    maximum_bytes: int,
) -> bytes:
    before = _trusted_regular_stat(
        path,
        uid=uid,
        gid=gid,
        modes=modes,
        maximum_bytes=maximum_bytes,
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        reached = os.fstat(descriptor)
        if (reached.st_dev, reached.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("collector trusted file changed during open")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(path)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or (before.st_dev, before.st_ino, before.st_size)
        != (current.st_dev, current.st_ino, current.st_size)
    ):
        raise RuntimeError("collector trusted file changed during read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size or len(raw) > maximum_bytes:
        raise RuntimeError("collector trusted file read is incomplete")
    return raw


def _credential_identity() -> tuple[os.stat_result, dict[str, Any]]:
    item = _trusted_regular_stat(
        DATABASE_CREDENTIAL_PATH,
        uid=WRITER_UID,
        gid=WRITER_GID,
        modes=frozenset({0o400}),
        maximum_bytes=_MAX_CREDENTIAL_BYTES,
    )
    # Deliberately omit size and every content-derived value.
    projection = {
        "path": str(DATABASE_CREDENTIAL_PATH),
        "device": int(item.st_dev),
        "inode": int(item.st_ino),
        "owner_uid": int(item.st_uid),
        "group_gid": int(item.st_gid),
        "mode": "0400",
        "link_count": int(item.st_nlink),
        "modification_time_ns": int(item.st_mtime_ns),
        "change_time_ns": int(item.st_ctime_ns),
        "content_or_digest_recorded": False,
    }
    return item, projection


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_nlink,
        left.st_uid,
        left.st_gid,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_nlink,
        right.st_uid,
        right.st_gid,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


@contextlib.contextmanager
def _pinned_credential() -> Any:
    before, projection = _credential_identity()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(DATABASE_CREDENTIAL_PATH, flags)
    try:
        if not _same_file_identity(before, os.fstat(descriptor)):
            raise RuntimeError("collector credential changed during pinned open")
        yield descriptor, projection
        after, after_projection = _credential_identity()
        if (
            not _same_file_identity(before, os.fstat(descriptor))
            or not _same_file_identity(before, after)
            or after_projection != projection
        ):
            raise RuntimeError(
                "collector credential identity changed during attestation"
            )
    finally:
        os.close(descriptor)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("collector JSON contains duplicate keys")
        result[name] = value
    return result


def _release_relative_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or ".." in Path(value).parts
        or Path(value).as_posix() != value
        or _CONTROL_RE.search(value) is not None
        or value in {"release-manifest.json", ".release-build-incomplete"}
    ):
        raise ValueError("collector release entry path is invalid")
    return value


def _hash_release_file(path: Path, before: os.stat_result) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    total = 0
    try:
        reached = os.fstat(descriptor)
        if not _same_file_identity(before, reached):
            raise RuntimeError("collector release file changed during open")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_RELEASE_FILE_BYTES:
                raise ValueError("collector release file exceeds its bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(path)
    if (
        total != before.st_size
        or not _same_file_identity(before, after)
        or not _same_file_identity(before, current)
    ):
        raise RuntimeError("collector release file changed during hashing")
    return digest.hexdigest()


def _live_release_paths(root: Path) -> list[str]:
    paths: list[str] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        for name in [*directories, *files]:
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            if relative == "release-manifest.json":
                continue
            if relative == ".release-build-incomplete":
                raise RuntimeError("collector release artifact is incomplete")
            paths.append(_release_relative_path(relative))
    paths.sort()
    if len(paths) != len(set(paths)):
        raise RuntimeError("collector release tree contains duplicate paths")
    return paths


def _verify_release_entries(
    root: Path,
    entries: Any,
    *,
    expected_uid: int = 0,
    expected_gid: int = 0,
) -> None:
    if not isinstance(entries, list) or not entries:
        raise ValueError("collector release manifest entries are invalid")
    root_resolved = root.resolve(strict=True)
    root_before = os.lstat(root)
    declared: list[str] = []
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise ValueError("collector release manifest entry is invalid")
        kind = raw.get("kind")
        fields = {
            "directory": {"path", "kind", "mode"},
            "file": {"path", "kind", "mode", "size", "sha256"},
            "symlink": {"path", "kind", "mode", "target"},
        }.get(kind)
        if fields is None or set(raw) != fields:
            raise ValueError("collector release manifest entry fields are invalid")
        relative = _release_relative_path(raw.get("path"))
        declared.append(relative)
        path = root / relative
        item = os.lstat(path)
        if (
            item.st_uid != expected_uid
            or item.st_gid != expected_gid
            or raw.get("mode") != f"{stat.S_IMODE(item.st_mode):04o}"
            or _list_xattrs(path)
        ):
            raise PermissionError("collector release entry protection drifted")
        if kind == "directory":
            if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
                raise ValueError("collector release directory identity drifted")
        elif kind == "file":
            if (
                stat.S_ISLNK(item.st_mode)
                or not stat.S_ISREG(item.st_mode)
                or item.st_nlink != 1
                or type(raw.get("size")) is not int
                or not 0 <= raw["size"] <= _MAX_RELEASE_FILE_BYTES
                or item.st_size != raw["size"]
                or _digest(raw.get("sha256"), "collector release file")
                != _hash_release_file(path, item)
            ):
                raise ValueError("collector release file identity drifted")
        else:
            target = raw.get("target")
            if (
                not stat.S_ISLNK(item.st_mode)
                or not isinstance(target, str)
                or not target
                or _CONTROL_RE.search(target) is not None
                or os.readlink(path) != target
            ):
                raise ValueError("collector release symlink identity drifted")
            resolved = path.resolve(strict=True)
            if resolved != root_resolved and root_resolved not in resolved.parents:
                raise ValueError("collector release symlink escaped the artifact")
    if declared != sorted(declared) or len(declared) != len(set(declared)):
        raise ValueError("collector release manifest paths are not exact sorted set")
    if declared != _live_release_paths(root):
        raise ValueError("collector release manifest and live paths differ")
    if not _same_file_identity(root_before, os.lstat(root)):
        raise RuntimeError("collector release root changed during attestation")


def _load_release_binding(
    *,
    revision: str,
    expected_artifact_sha256: str,
    expected_manifest_file_sha256: str,
) -> Mapping[str, Any]:
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("collector release revision is invalid")
    _digest(expected_artifact_sha256, "release artifact")
    _digest(expected_manifest_file_sha256, "release manifest file")
    root = RELEASE_BASE / revision
    root_item = os.lstat(root)
    if (
        stat.S_ISLNK(root_item.st_mode)
        or not stat.S_ISDIR(root_item.st_mode)
        or root_item.st_uid != 0
        or root_item.st_gid != 0
        or stat.S_IMODE(root_item.st_mode) != 0o555
        or _list_xattrs(root)
    ):
        raise PermissionError("collector release root is not sealed")
    _validate_protected_ancestor_chain(root.parent)
    path = root / "release-manifest.json"
    raw = _read_trusted_public_file(
        path,
        uid=0,
        gid=0,
        modes=frozenset({0o400}),
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    if _sha256_bytes(raw) != expected_manifest_file_sha256:
        raise ValueError("collector release manifest file digest drifted")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant:{value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("collector release manifest is invalid JSON") from exc
    fields = {
        "schema",
        "revision",
        "artifact_root",
        "python_version",
        "interpreter",
        "writer_module",
        "writer_module_origin",
        "gateway_module",
        "gateway_module_origin",
        "entries",
        "artifact_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("collector release manifest fields are not exact")
    unsigned = {name: copy.deepcopy(item) for name, item in value.items() if name != "artifact_sha256"}
    if (
        value["schema"] != RELEASE_SCHEMA
        or value["revision"] != revision
        or value["artifact_root"] != str(root)
        or value["artifact_sha256"] != expected_artifact_sha256
        or _sha256_json(unsigned) != expected_artifact_sha256
        or raw not in {_canonical_bytes(value), _canonical_bytes(value) + b"\n"}
    ):
        raise ValueError("collector release manifest identity drifted")
    _verify_release_entries(root, value["entries"])
    return value


def _validate_tls_server_name(value: str) -> str:
    validate_tls_server_name(value)
    if value != value.lower() or _CLOUDSQL_TLS_RE.fullmatch(value) is None:
        raise ValueError("collector Cloud SQL TLS SAN is not exact")
    return value


def _owner_ids(values: Sequence[str]) -> tuple[str, ...]:
    if any(_DISCORD_ID_RE.fullmatch(item) is None for item in values):
        raise ValueError("collector owner Discord user IDs are not exact")
    result = tuple(sorted(set(values)))
    if len(result) != len(values):
        raise ValueError("collector owner Discord user IDs contain duplicates")
    return result


def _database_config(
    tls_server_name: str,
    *,
    credential_fd: int | None = None,
) -> WriterDBConfig:
    credential = (
        CredentialSource(
            path=DATABASE_CREDENTIAL_PATH,
            expected_uid=WRITER_UID,
            expected_gid=WRITER_GID,
            allowed_modes=frozenset({0o400}),
        )
        if credential_fd is None
        else CredentialSource(
            fd=credential_fd,
            expected_uid=WRITER_UID,
            expected_gid=WRITER_GID,
            allowed_modes=frozenset({0o400}),
        )
    )
    return WriterDBConfig(
        host=SQL_PRIVATE_IP,
        tls_server_name=_validate_tls_server_name(tls_server_name),
        port=SQL_PORT,
        database=SQL_DATABASE,
        user=SQL_USER,
        ca_file=DATABASE_CA_PATH,
        credential=credential,
        connect_timeout_seconds=5.0,
        io_timeout_seconds=10.0,
    )


def _seed_identity(signature: str, *, public: bool) -> RoutineIdentity:
    return RoutineIdentity(
        signature=signature,
        owner=CANONICAL_WRITER_MIGRATION_OWNER,
        security_definer=public,
        language="plpgsql" if public else "sql",
        configuration=_SAFE_SEARCH_PATH,
        definition_sha256="0" * 64,
    )


def _seed_policy(
    hba_receipt: ManagedCloudSQLAdminHBAReceipt,
) -> WriterPrivilegePolicy:
    return WriterPrivilegePolicy(
        schema=CANONICAL_WRITER_SCHEMA,
        table_grants=(),
        sequence_grants=(),
        executable_routines=EXPECTED_ROUTINE_SIGNATURES,
        routine_identities=tuple(
            _seed_identity(signature, public=True)
            for signature in EXPECTED_ROUTINE_SIGNATURES
        ),
        dependency_routine_identities=tuple(
            _seed_identity(signature, public=False)
            for signature in EXPECTED_HELPER_ROUTINE_SIGNATURES
        ),
        schema_privileges=("USAGE",),
        database_privileges=("CONNECT",),
        role_memberships=(CANONICAL_WRITER_ROLE,),
        private_schema_identity_sha256="0" * 64,
        managed_cloudsqladmin_hba_rejection_receipt=hba_receipt,
        managed_cloudsqladmin_hba_rejection_sha256=hba_receipt.sha256,
        deployment_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY,
    )


def _require_routine_contract(attestation: PrivilegeAttestation) -> None:
    public = tuple(sorted(attestation.routine_identities))
    helpers = tuple(sorted(attestation.dependency_routine_identities))
    if tuple(identity.signature for identity in public) != EXPECTED_ROUTINE_SIGNATURES:
        raise ValueError("collector public routine signature set drifted")
    if tuple(identity.signature for identity in helpers) != EXPECTED_HELPER_ROUTINE_SIGNATURES:
        raise ValueError("collector helper routine signature set drifted")
    for identity in (*public, *helpers):
        if (
            identity.owner != CANONICAL_WRITER_MIGRATION_OWNER
            or identity.owner_dangerous
            or identity.configuration != _SAFE_SEARCH_PATH
            or identity.language not in {"sql", "plpgsql"}
        ):
            raise ValueError("collector routine authority identity drifted")
    if any(not identity.security_definer or identity.language != "plpgsql" for identity in public):
        raise ValueError("collector public routine execution identity drifted")
    if any(identity.security_definer for identity in helpers):
        raise ValueError("collector helper routine execution identity drifted")


def _policy_from_attestation(
    attestation: PrivilegeAttestation,
    *,
    hba_receipt: ManagedCloudSQLAdminHBAReceipt,
) -> WriterPrivilegePolicy:
    _require_routine_contract(attestation)
    if (
        attestation.table_grants
        or attestation.sequence_grants
        or tuple(sorted(attestation.executable_routines)) != EXPECTED_ROUTINE_SIGNATURES
        or tuple(sorted(attestation.schema_privileges)) != ("USAGE",)
        or tuple(sorted(attestation.database_privileges)) != ("CONNECT",)
        or tuple(sorted(attestation.role_memberships)) != (CANONICAL_WRITER_ROLE,)
        or attestation.unexpected_privileges
        or attestation.public_acl_grants
        or attestation.canonical_event_log_identity is None
        or attestation.canonical_private_schema_identity is None
    ):
        raise ValueError("collector privilege surface is not exact")
    policy = WriterPrivilegePolicy(
        schema=CANONICAL_WRITER_SCHEMA,
        table_grants=(),
        sequence_grants=(),
        executable_routines=EXPECTED_ROUTINE_SIGNATURES,
        routine_identities=tuple(sorted(attestation.routine_identities)),
        dependency_routine_identities=tuple(
            sorted(attestation.dependency_routine_identities)
        ),
        schema_privileges=("USAGE",),
        database_privileges=("CONNECT",),
        role_memberships=(CANONICAL_WRITER_ROLE,),
        private_schema_identity_sha256=(
            attestation.canonical_private_schema_identity.sha256
        ),
        managed_cloudsqladmin_hba_rejection_receipt=hba_receipt,
        managed_cloudsqladmin_hba_rejection_sha256=hba_receipt.sha256,
        deployment_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY,
    )
    validate_privilege_attestation(attestation, policy, expected_user=SQL_USER)
    return policy


def _collect_live_policy(
    config: WriterDBConfig,
    *,
    hba_collector: HBACollector = collect_managed_cloudsqladmin_hba_receipt,
    session_factory: SessionFactory = _open_postgres_session,
) -> tuple[WriterPrivilegePolicy, PrivilegeAttestation, ManagedCloudSQLAdminHBAReceipt]:
    hba_receipt = hba_collector(config)
    if not isinstance(hba_receipt, ManagedCloudSQLAdminHBAReceipt):
        raise TypeError("collector HBA evidence is invalid")
    if (
        hba_receipt.host != SQL_PRIVATE_IP
        or hba_receipt.tls_server_name != config.tls_server_name
        or hba_receipt.port != SQL_PORT
        or hba_receipt.user != SQL_USER
        or hba_receipt.tls_peer_verified is not True
        or not hba_receipt.is_fresh(int(time.time()))
    ):
        raise ValueError("collector HBA evidence binding is invalid")
    seed = _seed_policy(hba_receipt)
    session = session_factory(config)
    try:
        _begin_locked_transaction(session, seed, read_only=True)
        try:
            attestation = _collect_privilege_attestation(
                session,
                config=config,
                policy=seed,
                managed_hba_receipt=hba_receipt,
            )
            policy = _policy_from_attestation(
                attestation,
                hba_receipt=hba_receipt,
            )
        except BaseException:
            _rollback_quietly(session)
            raise
        _require_command(session, "COMMIT", "COMMIT")
    finally:
        session.close()
    return policy, attestation, hba_receipt


def _routine_mapping(identity: RoutineIdentity) -> dict[str, Any]:
    return {
        "signature": identity.signature,
        "owner": identity.owner,
        "security_definer": identity.security_definer,
        "language": identity.language,
        "configuration": list(identity.configuration),
        "definition_sha256": identity.definition_sha256,
    }


def _writer_config_mapping(
    *,
    config: WriterDBConfig,
    policy: WriterPrivilegePolicy,
    owner_discord_user_ids: Sequence[str],
) -> dict[str, Any]:
    receipt = policy.managed_cloudsqladmin_hba_rejection_receipt
    if receipt is None:
        raise ValueError("collector policy lacks managed HBA evidence")
    return {
        "service": {
            "socket_path": str(DEFAULT_SOCKET_PATH),
            "gateway_unit": DEFAULT_GATEWAY_UNIT,
            "gateway_uid": GATEWAY_UID,
            "writer_uid": WRITER_UID,
            "writer_gid": WRITER_GID,
            "socket_gid": SOCKET_CLIENT_GID,
            "projector_gid": PROJECTOR_GID,
            "owner_discord_user_ids": list(owner_discord_user_ids),
            "connection_timeout_seconds": 30.0,
            "max_connections": 8,
        },
        "database": {
            "host": config.host,
            "tls_server_name": config.tls_server_name,
            "port": config.port,
            "database": config.database,
            "user": config.user,
            "ca_file": str(config.ca_file),
            "credential_file": str(DATABASE_CREDENTIAL_PATH),
            "connect_timeout_seconds": config.connect_timeout_seconds,
            "io_timeout_seconds": config.io_timeout_seconds,
        },
        "privileges": {
            "schema": policy.schema,
            "table_grants": [],
            "routine_identities": [
                _routine_mapping(identity) for identity in policy.routine_identities
            ],
            "helper_routine_identities": [
                _routine_mapping(identity)
                for identity in policy.dependency_routine_identities
            ],
            "schema_privileges": list(policy.schema_privileges),
            "database_privileges": list(policy.database_privileges),
            "role_memberships": list(policy.role_memberships),
            "private_schema_identity_sha256": policy.private_schema_identity_sha256,
            "managed_cloudsqladmin_hba_rejection_receipt": receipt.as_dict(),
            "managed_cloudsqladmin_hba_rejection_sha256": receipt.sha256,
            "deployment_lock_key": policy.deployment_lock_key,
        },
        "discord_edge_authority": {"enabled": False},
    }


def _attestation_projection(attestation: PrivilegeAttestation) -> dict[str, Any]:
    return {
        "role": attestation.role,
        "dangerous_attributes": {
            "superuser": attestation.superuser,
            "createdb": attestation.createdb,
            "createrole": attestation.createrole,
            "replication": attestation.replication,
            "bypassrls": attestation.bypassrls,
            "table_owner": attestation.table_owner,
            "routine_owner": attestation.routine_owner,
        },
        "table_grants": [dataclasses.asdict(item) for item in attestation.table_grants],
        "sequence_grants": [
            dataclasses.asdict(item) for item in attestation.sequence_grants
        ],
        "executable_routines": list(attestation.executable_routines),
        "routine_identities": [
            dataclasses.asdict(item) for item in attestation.routine_identities
        ],
        "helper_routine_identities": [
            dataclasses.asdict(item)
            for item in attestation.dependency_routine_identities
        ],
        "schema_privileges": list(attestation.schema_privileges),
        "database_privileges": list(attestation.database_privileges),
        "role_memberships": list(attestation.role_memberships),
        "unexpected_privileges": list(attestation.unexpected_privileges),
        "public_acl_grants": list(attestation.public_acl_grants),
        "canonical_non_owner_acl_grants": list(
            attestation.canonical_non_owner_acl_grants
        ),
        "canonical_writer_role_inheritors": list(
            attestation.canonical_writer_role_inheritors
        ),
        "canonical_event_log_identity": dataclasses.asdict(
            attestation.canonical_event_log_identity
        ),
        "canonical_private_schema_identity": dataclasses.asdict(
            attestation.canonical_private_schema_identity
        ),
    }


@dataclass(frozen=True)
class ConfigCollectorReceipt:
    """Strict append-only provenance for one staged configuration generation."""

    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ConfigCollectorReceipt":
        value = _exact_mapping(
            raw,
            _COLLECTOR_RECEIPT_KEYS,
            "config collector receipt",
        )
        if value.get("schema") != COLLECTOR_RECEIPT_SCHEMA:
            raise ValueError("config collector receipt schema is invalid")
        revision = value.get("release_revision")
        if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
            raise ValueError("config collector receipt revision is invalid")
        for name in (
            "release_artifact_sha256",
            "release_manifest_file_sha256",
            "writer_config_sha256",
            "gateway_config_sha256",
            "catalog_attestation_sha256",
            "private_schema_identity_sha256",
            "managed_hba_receipt_sha256",
            "server_certificate_sha256",
            "receipt_sha256",
        ):
            _digest(value.get(name), f"config collector {name}")
        if value.get("release_manifest_path") != str(
            RELEASE_BASE / revision / "release-manifest.json"
        ):
            raise ValueError("config collector release manifest path drifted")
        if value.get("writer_config_path") != str(STAGED_WRITER_CONFIG_PATH):
            raise ValueError("config collector writer config path drifted")
        if value.get("gateway_config_path") != str(STAGED_GATEWAY_CONFIG_PATH):
            raise ValueError("config collector gateway config path drifted")
        database = _exact_mapping(
            value.get("database"),
            _COLLECTOR_DATABASE_KEYS,
            "config collector database",
        )
        if (
            database.get("host") != SQL_PRIVATE_IP
            or database.get("port") != SQL_PORT
            or database.get("database") != SQL_DATABASE
            or database.get("user") != SQL_USER
            or database.get("ca_path") != str(DATABASE_CA_PATH)
        ):
            raise ValueError("config collector database identity drifted")
        _validate_tls_server_name(database.get("tls_server_name"))
        _digest(database.get("ca_sha256"), "config collector database CA")
        credential = _exact_mapping(
            value.get("credential_provenance"),
            _COLLECTOR_CREDENTIAL_KEYS,
            "config collector credential provenance",
        )
        if (
            credential.get("path") != str(DATABASE_CREDENTIAL_PATH)
            or credential.get("owner_uid") != WRITER_UID
            or credential.get("group_gid") != WRITER_GID
            or credential.get("mode") != "0400"
            or credential.get("link_count") != 1
            or credential.get("content_or_digest_recorded") is not False
        ):
            raise ValueError("config collector credential provenance drifted")
        for name in (
            "device",
            "inode",
            "modification_time_ns",
            "change_time_ns",
        ):
            _nonnegative_integer(
                credential.get(name),
                f"config collector credential {name}",
            )
        if (
            value.get("public_routine_count") != len(EXPECTED_ROUTINE_SIGNATURES)
            or value.get("helper_routine_count")
            != len(EXPECTED_HELPER_ROUTINE_SIGNATURES)
            or value.get("discord_edge_enabled") is not False
            or value.get("credential_content_or_digest_recorded") is not False
        ):
            raise ValueError("config collector authority summary drifted")
        observed = _nonnegative_integer(
            value.get("hba_observed_at_unix"),
            "config collector HBA observation time",
        )
        expires = _nonnegative_integer(
            value.get("hba_expires_at_unix"),
            "config collector HBA expiry time",
        )
        collected = _nonnegative_integer(
            value.get("collected_at_unix"),
            "config collector collection time",
        )
        if expires - observed != 300 or not observed <= collected <= expires:
            raise ValueError("config collector freshness window is invalid")
        unsigned = {
            name: copy.deepcopy(item)
            for name, item in value.items()
            if name != "receipt_sha256"
        }
        if _sha256_json(unsigned) != value["receipt_sha256"]:
            raise ValueError("config collector receipt self-digest drifted")
        return cls(json.loads(_canonical_bytes(dict(value)).decode("utf-8")))

    @property
    def sha256(self) -> str:
        return str(self.value["receipt_sha256"])

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(_canonical_bytes(dict(self.value)).decode("utf-8"))

    def require_fresh(self, now_unix: int) -> None:
        if (
            type(now_unix) is not int
            or not self.value["collected_at_unix"]
            <= now_unix
            <= self.value["hba_expires_at_unix"]
        ):
            raise ValueError("config collector receipt is stale or future-dated")

    def require_bindings(
        self,
        *,
        revision: str,
        release_artifact_sha256: str,
        release_manifest_file_sha256: str,
        writer_config_sha256: str,
        gateway_config_sha256: str,
        database_ca_sha256: str,
        sql_private_ip: str,
        sql_tls_server_name: str,
    ) -> None:
        for value, label in (
            (release_artifact_sha256, "collector-bound release artifact"),
            (release_manifest_file_sha256, "collector-bound release manifest"),
            (writer_config_sha256, "collector-bound writer config"),
            (gateway_config_sha256, "collector-bound gateway config"),
            (database_ca_sha256, "collector-bound database CA"),
        ):
            _digest(value, label)
        if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
            raise ValueError("collector-bound revision is invalid")
        _validate_tls_server_name(sql_tls_server_name)
        database = self.value["database"]
        if (
            self.value["release_revision"] != revision
            or self.value["release_artifact_sha256"]
            != release_artifact_sha256
            or self.value["release_manifest_file_sha256"]
            != release_manifest_file_sha256
            or self.value["writer_config_sha256"] != writer_config_sha256
            or self.value["gateway_config_sha256"] != gateway_config_sha256
            or database["ca_sha256"] != database_ca_sha256
            or database["host"] != sql_private_ip
            or database["tls_server_name"] != sql_tls_server_name
        ):
            raise ValueError("config collector receipt binding drifted")


def config_collector_receipt_path(
    *,
    revision: str,
    receipt_sha256: str,
) -> Path:
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("config collector receipt revision is invalid")
    _digest(receipt_sha256, "config collector receipt path")
    return EVIDENCE_ROOT / revision / f"{receipt_sha256}.json"


def load_config_collector_receipt(
    *,
    revision: str,
    receipt_sha256: str,
    require_fresh: bool,
    now_unix: int | None = None,
) -> ConfigCollectorReceipt:
    """Load one canonical receipt only from its derived append-only path."""

    if type(require_fresh) is not bool:
        raise TypeError("config collector freshness requirement must be boolean")
    path = config_collector_receipt_path(
        revision=revision,
        receipt_sha256=receipt_sha256,
    )
    raw = _read_trusted_public_file(
        path,
        uid=0,
        gid=0,
        modes=frozenset({0o400}),
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant:{value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("config collector receipt is not strict JSON") from exc
    if not isinstance(value, Mapping) or raw != _canonical_bytes(value):
        raise ValueError("config collector receipt is not canonical JSON")
    receipt = ConfigCollectorReceipt.from_mapping(value)
    if receipt.sha256 != receipt_sha256:
        raise ValueError("config collector receipt path digest drifted")
    if require_fresh:
        receipt.require_fresh(int(time.time()) if now_unix is None else now_unix)
    return receipt


@dataclass(frozen=True)
class CollectorArtifacts:
    writer_config: bytes
    gateway_config: bytes
    receipt: Mapping[str, Any]


def _build_artifacts(
    *,
    revision: str,
    artifact_sha256: str,
    manifest_file_sha256: str,
    ca_sha256: str,
    credential_provenance: Mapping[str, Any],
    config: WriterDBConfig,
    policy: WriterPrivilegePolicy,
    attestation: PrivilegeAttestation,
    owner_discord_user_ids: Sequence[str],
    collected_at_unix: int,
) -> CollectorArtifacts:
    writer_mapping = _writer_config_mapping(
        config=config,
        policy=policy,
        owner_discord_user_ids=owner_discord_user_ids,
    )
    writer = _canonical_bytes(writer_mapping)
    if len(writer) > _MAX_CONFIG_BYTES:
        raise ValueError("collector writer config is oversized")
    hba = policy.managed_cloudsqladmin_hba_rejection_receipt
    if hba is None:
        raise ValueError("collector HBA receipt is absent")
    if (
        type(collected_at_unix) is not int
        or not hba.observed_at_unix <= collected_at_unix <= hba.expires_at_unix
    ):
        raise ValueError("collector HBA receipt is not fresh at projection time")
    unsigned = {
        "schema": COLLECTOR_RECEIPT_SCHEMA,
        "release_revision": revision,
        "release_artifact_sha256": artifact_sha256,
        "release_manifest_path": str(
            RELEASE_BASE / revision / "release-manifest.json"
        ),
        "release_manifest_file_sha256": manifest_file_sha256,
        "writer_config_path": str(STAGED_WRITER_CONFIG_PATH),
        "writer_config_sha256": _sha256_bytes(writer),
        "gateway_config_path": str(STAGED_GATEWAY_CONFIG_PATH),
        "gateway_config_sha256": _sha256_bytes(GATEWAY_CONFIG_BYTES),
        "database": {
            "host": config.host,
            "tls_server_name": config.tls_server_name,
            "port": config.port,
            "database": config.database,
            "user": config.user,
            "ca_path": str(config.ca_file),
            "ca_sha256": ca_sha256,
        },
        "credential_provenance": dict(credential_provenance),
        "catalog_attestation_sha256": _sha256_json(
            _attestation_projection(attestation)
        ),
        "public_routine_count": len(policy.routine_identities),
        "helper_routine_count": len(policy.dependency_routine_identities),
        "private_schema_identity_sha256": policy.private_schema_identity_sha256,
        "managed_hba_receipt_sha256": hba.sha256,
        "server_certificate_sha256": hba.server_certificate_sha256,
        "hba_observed_at_unix": hba.observed_at_unix,
        "hba_expires_at_unix": hba.expires_at_unix,
        "discord_edge_enabled": False,
        "credential_content_or_digest_recorded": False,
        "collected_at_unix": collected_at_unix,
    }
    receipt_sha256 = _sha256_json(unsigned)
    return CollectorArtifacts(
        writer_config=writer,
        gateway_config=GATEWAY_CONFIG_BYTES,
        receipt={**unsigned, "receipt_sha256": receipt_sha256},
    )


def _atomic_replace_file(
    path: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    _validate_exact_directory(path.parent, uid=uid, gid=gid, mode=0o700)
    if os.path.lexists(path):
        _trusted_regular_stat(
            path,
            uid=uid,
            gid=gid,
            modes=frozenset({mode}),
            maximum_bytes=max(_MAX_CONFIG_BYTES, len(payload)),
            allow_empty=True,
        )
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        os.fchown(descriptor, uid, gid)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("collector staging write made no progress")
            offset += written
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    installed = _read_trusted_public_file(
        path,
        uid=uid,
        gid=gid,
        modes=frozenset({mode}),
        maximum_bytes=max(len(payload), 1),
    )
    if installed != payload:
        raise RuntimeError("collector staged file readback drifted")


def _publish_noreplace(source: Path, target: Path) -> None:
    """Atomically move one complete temp file without replacing a receipt."""

    if sys.platform != "linux":  # exercised only by non-production unit tests
        os.link(source, target, follow_symlinks=False)
        source.unlink()
        return
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("collector requires Linux renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), target)
    raise OSError(error, os.strerror(error), target)


def _write_append_only_receipt(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    uid: int = 0,
    gid: int = 0,
) -> None:
    raw_receipt_sha = receipt.get("receipt_sha256")
    unsigned = {
        name: copy.deepcopy(value)
        for name, value in receipt.items()
        if name != "receipt_sha256"
    }
    if (
        _digest(raw_receipt_sha, "collector append-only receipt")
        != _sha256_json(unsigned)
        or path.name != f"{raw_receipt_sha}.json"
    ):
        raise ValueError("collector append-only receipt identity drifted")
    _validate_exact_directory(path.parent, uid=uid, gid=gid, mode=0o700)
    payload = _canonical_bytes(dict(receipt))
    if os.path.lexists(path):
        existing = _read_trusted_public_file(
            path,
            uid=uid,
            gid=gid,
            modes=frozenset({0o400}),
            maximum_bytes=_MAX_CONFIG_BYTES,
        )
        if existing != payload:
            raise RuntimeError("collector receipt identity collided")
        return
    temporary = path.parent / (
        f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o400)
    completed = False
    try:
        os.fchown(descriptor, uid, gid)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("collector receipt write made no progress")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        completed = True
    finally:
        os.close(descriptor)
        if not completed:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    try:
        try:
            _publish_noreplace(temporary, path)
        except FileExistsError:
            existing = _read_trusted_public_file(
                path,
                uid=uid,
                gid=gid,
                modes=frozenset({0o400}),
                maximum_bytes=_MAX_CONFIG_BYTES,
            )
            if existing != payload:
                raise RuntimeError("collector receipt identity collided")
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    installed = _read_trusted_public_file(
        path,
        uid=uid,
        gid=gid,
        modes=frozenset({0o400}),
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    if installed != payload:
        raise RuntimeError("collector append-only receipt readback drifted")


def collect_and_stage(
    *,
    revision: str,
    release_artifact_sha256: str,
    release_manifest_file_sha256: str,
    tls_server_name: str,
    owner_discord_user_ids: Sequence[str],
    _hba_collector: HBACollector = collect_managed_cloudsqladmin_hba_receipt,
    _session_factory: SessionFactory = _open_postgres_session,
    _clock: Callable[[], float] = time.time,
) -> Mapping[str, Any]:
    """Collect live truth and atomically stage its secret-free projection."""

    _require_root_linux()
    owners = _owner_ids(owner_discord_user_ids)
    _load_release_binding(
        revision=revision,
        expected_artifact_sha256=release_artifact_sha256,
        expected_manifest_file_sha256=release_manifest_file_sha256,
    )
    ca = _read_trusted_public_file(
        DATABASE_CA_PATH,
        uid=0,
        gid=WRITER_GID,
        modes=frozenset({0o440}),
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    ca_identity_before = os.lstat(DATABASE_CA_PATH)
    ca_sha256 = _sha256_bytes(ca)
    with _pinned_credential() as (credential_fd, credential_projection):
        config = _database_config(
            tls_server_name,
            credential_fd=credential_fd,
        )
        policy, attestation, _hba = _collect_live_policy(
            config,
            hba_collector=_hba_collector,
            session_factory=_session_factory,
        )
    ca_after = _read_trusted_public_file(
        DATABASE_CA_PATH,
        uid=0,
        gid=WRITER_GID,
        modes=frozenset({0o440}),
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    if (
        ca_after != ca
        or not _same_file_identity(ca_identity_before, os.lstat(DATABASE_CA_PATH))
    ):
        raise RuntimeError("collector database CA changed during attestation")
    _load_release_binding(
        revision=revision,
        expected_artifact_sha256=release_artifact_sha256,
        expected_manifest_file_sha256=release_manifest_file_sha256,
    )
    if not _hba.is_fresh(int(_clock())):
        raise RuntimeError("collector managed HBA receipt expired during attestation")
    collected_at = int(_clock())
    artifacts = _build_artifacts(
        revision=revision,
        artifact_sha256=release_artifact_sha256,
        manifest_file_sha256=release_manifest_file_sha256,
        ca_sha256=ca_sha256,
        credential_provenance=credential_projection,
        config=config,
        policy=policy,
        attestation=attestation,
        owner_discord_user_ids=owners,
        collected_at_unix=collected_at,
    )
    _ensure_exact_directory(STAGING_ROOT)
    directory_fd = os.open(
        STAGING_ROOT,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        _atomic_replace_file(
            STAGED_WRITER_CONFIG_PATH,
            artifacts.writer_config,
            uid=0,
            gid=0,
            mode=0o400,
        )
        _atomic_replace_file(
            STAGED_GATEWAY_CONFIG_PATH,
            artifacts.gateway_config,
            uid=0,
            gid=0,
            mode=0o400,
        )
        loaded = load_service_config(STAGED_WRITER_CONFIG_PATH)
        loaded_database = loaded.database
        loaded_credential = loaded_database.credential
        if (
            loaded_database.host != config.host
            or loaded_database.tls_server_name != config.tls_server_name
            or loaded_database.port != config.port
            or loaded_database.database != config.database
            or loaded_database.user != config.user
            or loaded_database.ca_file != config.ca_file
            or loaded_database.connect_timeout_seconds
            != config.connect_timeout_seconds
            or loaded_database.io_timeout_seconds != config.io_timeout_seconds
            or loaded_credential.path != DATABASE_CREDENTIAL_PATH
            or loaded_credential.fd is not None
            or loaded_credential.expected_uid != WRITER_UID
            or loaded_credential.expected_gid != WRITER_GID
            or loaded_credential.allowed_modes != frozenset({0o400})
            or loaded.privileges != policy
            or loaded.gateway_uid != GATEWAY_UID
            or loaded.writer_uid != WRITER_UID
            or loaded.writer_gid != WRITER_GID
            or loaded.socket_gid != SOCKET_CLIENT_GID
            or loaded.projector_gid != PROJECTOR_GID
            or loaded.owner_discord_user_ids != frozenset(owners)
            or loaded.discord_edge_authority.enabled is not False
        ):
            raise RuntimeError("collector staged writer config readback drifted")
    finally:
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_UN)
        finally:
            os.close(directory_fd)
    receipt_sha = _digest(artifacts.receipt["receipt_sha256"], "collector receipt")
    receipt_directory = EVIDENCE_ROOT / revision
    _ensure_exact_directory(receipt_directory)
    receipt_path = receipt_directory / f"{receipt_sha}.json"
    _write_append_only_receipt(receipt_path, artifacts.receipt)
    return {
        "ok": True,
        "schema": COLLECTOR_RECEIPT_SCHEMA,
        "release_revision": revision,
        "release_artifact_sha256": release_artifact_sha256,
        "writer_config_path": str(STAGED_WRITER_CONFIG_PATH),
        "writer_config_sha256": artifacts.receipt["writer_config_sha256"],
        "gateway_config_path": str(STAGED_GATEWAY_CONFIG_PATH),
        "gateway_config_sha256": artifacts.receipt["gateway_config_sha256"],
        "catalog_attestation_sha256": artifacts.receipt[
            "catalog_attestation_sha256"
        ],
        "managed_hba_receipt_sha256": artifacts.receipt[
            "managed_hba_receipt_sha256"
        ],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "credential_content_or_digest_recorded": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and stage one trusted writer-only canary config",
    )
    parser.add_argument("action", choices=("collect",))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--release-artifact-sha256", required=True)
    parser.add_argument("--release-manifest-file-sha256", required=True)
    parser.add_argument("--tls-server-name", required=True)
    parser.add_argument(
        "--owner-discord-user-id",
        action="append",
        default=[],
        dest="owner_discord_user_ids",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = collect_and_stage(
            revision=arguments.revision,
            release_artifact_sha256=arguments.release_artifact_sha256,
            release_manifest_file_sha256=(
                arguments.release_manifest_file_sha256
            ),
            tls_server_name=arguments.tls_server_name,
            owner_discord_user_ids=arguments.owner_discord_user_ids,
        )
    except Exception as exc:
        # Error messages and digests are deliberately excluded: an underlying
        # provider or filesystem exception must never reflect credential data.
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "trusted_config_collection_failed",
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return 0


__all__ = [
    "COLLECTOR_RECEIPT_SCHEMA",
    "DATABASE_CA_PATH",
    "DATABASE_CREDENTIAL_PATH",
    "EVIDENCE_ROOT",
    "GATEWAY_CONFIG_BYTES",
    "STAGED_GATEWAY_CONFIG_PATH",
    "STAGED_WRITER_CONFIG_PATH",
    "CollectorArtifacts",
    "ConfigCollectorReceipt",
    "collect_and_stage",
    "config_collector_receipt_path",
    "load_config_collector_receipt",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
