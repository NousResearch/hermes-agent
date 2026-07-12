"""Authoritative same-host collector for the Canonical writer preflight.

The large deployment evaluator remains a pure diagnostic function.  This
module supplies the missing authority boundary: it runs only as UID 0, loads a
separately root-owned approved manifest, obtains the live snapshot through its
fixed same-host collector, actively refreshes the managed-HBA proof, evaluates
the snapshot in-process, and writes only a short-lived root-owned receipt.

There is intentionally no JSON-snapshot CLI and no dynamic collector import.
Deployment code imports :func:`collect_and_evaluate`; callers cannot replace
the reviewed mechanical collector.
"""

from __future__ import annotations

import copy
import argparse
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import ipaddress
import json
import os
import pwd
import re
import socket
import stat
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from gateway.canonical_writer_db import (
    ManagedCloudSQLAdminHBAReceipt,
    collect_managed_cloudsqladmin_hba_receipt,
    managed_cloudsqladmin_hba_receipt_from_mapping,
    validate_tls_server_name,
)
from gateway.canonical_writer_deployment_preflight import (
    _HARDENED_TRUE_PROPERTIES,
    _systemd_checks,
    PreflightCheck,
    PreflightReport,
)
from gateway.canonical_writer_boundary import (
    DEFAULT_DISCORD_EDGE_SOCKET_PATH,
    DEFAULT_DISCORD_EDGE_UNIT,
    DEFAULT_DISCORD_EDGE_USER,
    DEFAULT_GATEWAY_UNIT,
    DEFAULT_SOCKET_PATH,
    DEFAULT_WRITER_UNIT,
)
from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
    PRODUCTION_CATALOG_SHA256,
)


MANIFEST_SCHEMA = "canonical-writer-deployment-manifest.v1"
RECEIPT_SCHEMA = "canonical-writer-root-preflight-receipt.v1"
WRITER_ONLY_MODE = "writer_only"
RECEIPT_TTL_SECONDS = 30
ACTIVE_HBA_MAX_AGE_SECONDS = 30
WRITER_LIVENESS_MAX_AGE_SECONDS = 5
DEFAULT_GATEWAY_MANAGED_CONFIG_PATH = Path("/etc/hermes/config.yaml")
DEFAULT_DISCORD_EDGE_CONFIG_PATH = Path("/etc/muncho/discord-edge.json")
DEFAULT_DISCORD_EDGE_TOKEN_PATH = Path(
    "/etc/muncho/discord-edge-credentials/bot-token"
)
_GATEWAY_RUNTIME_PATH = Path("/run/hermes-cloud-gateway")
_WRITER_CGROUP_PATH = Path(
    "/sys/fs/cgroup/system.slice/muncho-canonical-writer.service"
)
_CGROUP2_SUPER_MAGIC = 0x63677270
_BPF_PROG_QUERY = 16
_BPF_CGROUP_INET_INGRESS = 0
_BPF_CGROUP_INET_EGRESS = 1
_BPF_F_QUERY_EFFECTIVE = 1 << 0
_MAX_BPF_PROGRAM_IDS = 256
_MAX_CGROUP_PROCS_BYTES = 4096
_MAX_TRUSTED_JSON_BYTES = 128 * 1024
_MAX_RELEASE_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_RELEASE_FILE_BYTES = 4 * 1024 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_CODE_INJECTION_ENVIRONMENT_NAMES = frozenset(
    {
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "HERMES_BUNDLED_PLUGINS",
        "HERMES_ENABLE_PROJECT_PLUGINS",
    }
)
_WRITER_ONLY_ALLOWED_ENVIRONMENT_NAMES = frozenset(
    {
        "HOME",
        "INVOCATION_ID",
        "JOURNAL_STREAM",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "MEMORY_PRESSURE_WATCH",
        "MEMORY_PRESSURE_WRITE",
        "NOTIFY_SOCKET",
        "PATH",
        "SHELL",
        "SYSTEMD_EXEC_PID",
        "SYSTEMD_NSS_DYNAMIC_BYPASS",
        "TZ",
        "USER",
    }
)
_FIXED_GATEWAY_ENVIRONMENT = {
    "HOME": "/var/lib/hermes-gateway",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LOGNAME": "muncho-gateway",
    "PATH": "/usr/bin:/bin",
    "SHELL": "/usr/sbin/nologin",
    "TZ": "UTC",
    "USER": "muncho-gateway",
}
_FIXED_WRITER_ENVIRONMENT = {
    "HOME": "/nonexistent",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LOGNAME": "muncho-canonical-writer",
    "PATH": "/usr/bin:/bin",
    "SHELL": "/usr/sbin/nologin",
    "TZ": "UTC",
    "USER": "muncho-canonical-writer",
}
_RELEASE_PATH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
_RELEASE_SCHEMA = "muncho-writer-only-release.v1"
_RELEASE_MANIFEST_NAME = "release-manifest.json"
_RELEASE_INCOMPLETE_MARKER = ".release-build-incomplete"
_RELEASE_MANIFEST_KEYS = frozenset(
    {
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
)
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "mode",
        "approved_plan_sha256",
        "revision",
        "artifact_sha256",
        "snapshot_policy_sha256",
        "host_contract",
        "snapshot_template",
    }
)
_HOST_CONTRACT_KEYS = frozenset(
    {
        "gateway_unit_fragment_path",
        "gateway_unit_fragment_sha256",
        "writer_unit_fragment_path",
        "writer_unit_fragment_sha256",
        "gateway_config_path",
        "gateway_config_sha256",
        "writer_config_sha256",
        "projection_export_path",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "ok",
        "mode",
        "boot_id_sha256",
        "collected_at_unix",
        "collected_at_boottime_ns",
        "expires_at_boottime_ns",
        "manifest_sha256",
        "approved_plan_sha256",
        "revision",
        "artifact_sha256",
        "snapshot_policy_sha256",
        "snapshot_sha256",
        "report_sha256",
        "hba_receipt_sha256",
        "gateway_readiness_sha256",
        "gateway_liveness_sha256",
        "gateway_liveness_generation",
        "writer_runtime_attestation_sha256",
        "gateway_code_closure_sha256",
        "writer_code_closure_sha256",
        "gateway_main_pid",
        "gateway_start_time_ticks",
        "writer_main_pid",
        "writer_start_time_ticks",
        "writer_socket_device",
        "writer_socket_inode",
        "writer_runtime_directory_device",
        "writer_runtime_directory_inode",
        "writer_ip_address_allow_network",
        "writer_cgroup_device",
        "writer_cgroup_inode",
        "writer_cgroup_main_pid",
        "writer_bpf_ingress_direct_program_ids",
        "writer_bpf_ingress_effective_program_ids",
        "writer_bpf_egress_direct_program_ids",
        "writer_bpf_egress_effective_program_ids",
        "failed_checks",
    }
)
_GATEWAY_LIVENESS_RECEIPT_KEYS = frozenset(
    {
        "version",
        "generation",
        "observed_at_unix",
        "observed_at_boottime_ns",
        "boot_id_sha256",
        "gateway_pid",
        "gateway_start_time_ticks",
        "writer_request_id",
        "writer_service",
        "writer_protocol",
        "database_identity",
        "socket_path",
        "socket_device",
        "socket_inode",
        "socket_owner_uid",
        "socket_group_gid",
        "socket_mode",
    }
)
_SYSTEMD_READINESS_KEYS = frozenset(
    {
        "unit_name",
        "unit_type",
        "notify_access",
        "active_state",
        "sub_state",
        "systemd_main_pid",
        "systemd_main_pid_start_time_ticks",
        "status_text",
        "receipt_sha256",
        "receipt",
    }
)
_GATEWAY_READINESS_RECEIPT_KEYS = frozenset(
    {
        "version",
        "observed_at_unix",
        "observed_at_boottime_ns",
        "boot_id_sha256",
        "gateway_pid",
        "gateway_start_time_ticks",
        "writer_request_id",
        "writer_service",
        "writer_protocol",
        "database_identity",
        "gateway_module_origin",
        "gateway_module_sha256",
        "gateway_dumpable",
        "gateway_core_soft_limit",
        "gateway_core_hard_limit",
        "effective_import_paths",
        "unexpected_import_paths",
        "loaded_module_origins",
        "unexpected_import_origins",
        "loaded_module_origins_complete",
        "effective_environment_variable_names",
        "effective_environment_variable_value_sha256",
    }
)
_WRITER_RUNTIME_RECEIPT_KEYS = frozenset(
    {
        "version",
        "observed_at_unix",
        "observed_at_boottime_ns",
        "boot_id_sha256",
        "writer_pid",
        "writer_start_time_ticks",
        "bootstrap_module_origin",
        "bootstrap_module_sha256",
        "service_module_origin",
        "service_module_sha256",
        "statement_catalog_sha256",
        "database_identity",
        "database_role",
        "private_schema_identity_sha256",
        "managed_hba_baseline_sha256",
        "discord_edge_authority_enabled",
        "socket_path",
        "socket_inode",
        "socket_device",
        "socket_owner_uid",
        "socket_group_gid",
        "socket_mode",
        "writer_dumpable",
        "writer_core_soft_limit",
        "writer_core_hard_limit",
        "effective_import_paths",
        "unexpected_import_paths",
        "loaded_module_origins",
        "unexpected_import_origins",
        "loaded_module_origins_complete",
        "effective_environment_variable_names",
        "effective_environment_variable_value_sha256",
    }
)


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_yaml_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found duplicate key",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("collector value is not canonical JSON") from exc
    return encoded.encode("utf-8", errors="strict")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("trusted JSON contains duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"trusted JSON contains non-JSON constant: {value}")


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        return -1
    return int(getter())


def _require_root_linux() -> None:
    if _effective_uid() != 0:
        raise PermissionError("canonical_writer_root_collector_requires_uid_0")
    if sys.platform != "linux":
        raise RuntimeError("canonical_writer_root_collector_requires_linux")


def _absolute_normalized_path(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    path = Path(raw)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(path) != raw
        or any(character in raw for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError("trusted path must be absolute and normalized")
    return path


def _validate_parent_chain(path: Path, *, expected_uid: int) -> None:
    current = path
    while True:
        try:
            item = os.lstat(current)
        except OSError as exc:
            raise ValueError("trusted parent path is unavailable") from exc
        if (
            stat.S_ISLNK(item.st_mode)
            or not stat.S_ISDIR(item.st_mode)
            or item.st_uid != expected_uid
            or stat.S_IMODE(item.st_mode) & 0o022
            or _has_posix_acl(current)
        ):
            raise ValueError("trusted parent path is not root-controlled")
        if current == current.parent:
            return
        current = current.parent


def _mode_access_for_identity(
    path_value: str | os.PathLike[str],
    *,
    uid: int,
    gids: Sequence[int],
) -> dict[str, bool]:
    """Evaluate exact DAC access after rejecting ACL-based exceptions."""

    path = _absolute_normalized_path(path_value)
    try:
        item = os.lstat(path)
    except FileNotFoundError:
        return {"read": False, "write": False, "execute": False}
    if stat.S_ISLNK(item.st_mode):
        raise RuntimeError("access evidence path cannot be a symlink")
    extended = set(_list_xattrs(path))
    if extended & {"system.posix_acl_access", "system.posix_acl_default"}:
        raise RuntimeError("access evidence path has an unreviewed POSIX ACL")
    identities = set(int(gid) for gid in gids)
    mode = stat.S_IMODE(item.st_mode)
    if uid == item.st_uid:
        bits = (mode >> 6) & 0o7
    elif item.st_gid in identities:
        bits = (mode >> 3) & 0o7
    else:
        bits = mode & 0o7
    access = {
        "read": bool(bits & 0o4),
        "write": bool(bits & 0o2),
        "execute": bool(bits & 0o1),
    }
    current = path.parent
    while True:
        parent = os.lstat(current)
        if (
            stat.S_ISLNK(parent.st_mode)
            or not stat.S_ISDIR(parent.st_mode)
            or _has_posix_acl(current)
        ):
            raise RuntimeError("access evidence parent path is invalid")
        parent_mode = stat.S_IMODE(parent.st_mode)
        if uid == parent.st_uid:
            parent_bits = (parent_mode >> 6) & 0o7
        elif parent.st_gid in identities:
            parent_bits = (parent_mode >> 3) & 0o7
        else:
            parent_bits = parent_mode & 0o7
        if not parent_bits & 0o1:
            return {"read": False, "write": False, "execute": False}
        if current == current.parent:
            break
        current = current.parent
    return access


def _list_xattrs(path: Path) -> tuple[str, ...]:
    lister = getattr(os, "listxattr", None)
    if not callable(lister):
        if sys.platform == "linux":
            raise RuntimeError("Linux xattr inspection is unavailable")
        return ()
    try:
        return tuple(lister(path, follow_symlinks=False))
    except OSError as exc:
        raise RuntimeError("filesystem xattrs are unavailable") from exc


def _has_posix_acl(path: Path) -> bool:
    return bool(
        set(_list_xattrs(path))
        & {"system.posix_acl_access", "system.posix_acl_default"}
    )


def _read_fd_bounded(descriptor: int, *, maximum: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total <= maximum:
        chunk = os.read(descriptor, min(4096, maximum + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    raw = b"".join(chunks)
    if not raw or len(raw) > maximum:
        raise ValueError("trusted JSON size is invalid")
    return raw


def _read_trusted_json(
    path_value: str | os.PathLike[str],
    *,
    expected_uid: int,
    expected_gid: int,
    require_trusted_parents: bool = True,
    allowed_modes: frozenset[int] = frozenset({0o400}),
    maximum: int = _MAX_TRUSTED_JSON_BYTES,
) -> Mapping[str, Any]:
    path = _absolute_normalized_path(path_value)
    if require_trusted_parents:
        _validate_parent_chain(path.parent, expected_uid=expected_uid)
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ValueError("trusted JSON is unavailable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != expected_uid
        or before.st_gid != expected_gid
        or stat.S_IMODE(before.st_mode) not in allowed_modes
        or _has_posix_acl(path)
    ):
        raise ValueError("trusted JSON ownership or mode is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("trusted JSON cannot be opened") from exc
    try:
        actual = os.fstat(descriptor)
        if (
            (actual.st_dev, actual.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISREG(actual.st_mode)
            or actual.st_nlink != 1
            or actual.st_uid != expected_uid
            or actual.st_gid != expected_gid
            or stat.S_IMODE(actual.st_mode) not in allowed_modes
        ):
            raise ValueError("trusted JSON identity changed during open")
        raw = _read_fd_bounded(descriptor, maximum=maximum)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("trusted JSON is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("trusted JSON root must be an object")
    return value


@dataclass(frozen=True)
class TrustedDeploymentManifest:
    approved_plan_sha256: str
    revision: str
    artifact_sha256: str
    snapshot_policy_sha256: str
    host_contract: Mapping[str, Any]
    snapshot_template: Mapping[str, Any]
    manifest_sha256: str
    schema: str = MANIFEST_SCHEMA
    mode: str = WRITER_ONLY_MODE

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrustedDeploymentManifest":
        if set(value) != _MANIFEST_KEYS:
            raise ValueError("deployment manifest fields are not exact")
        if value.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("deployment manifest schema is invalid")
        if value.get("mode") != WRITER_ONLY_MODE:
            raise ValueError("deployment manifest mode is not writer-only")
        approved = value.get("approved_plan_sha256")
        revision = value.get("revision")
        artifact = value.get("artifact_sha256")
        policy = value.get("snapshot_policy_sha256")
        host_contract = value.get("host_contract")
        snapshot_template = value.get("snapshot_template")
        if not isinstance(approved, str) or _SHA256_RE.fullmatch(approved) is None:
            raise ValueError("approved plan digest is invalid")
        if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
            raise ValueError("release revision is invalid")
        if not isinstance(artifact, str) or _SHA256_RE.fullmatch(artifact) is None:
            raise ValueError("release artifact digest is invalid")
        if not isinstance(policy, str) or _SHA256_RE.fullmatch(policy) is None:
            raise ValueError("snapshot policy digest is invalid")
        if not isinstance(host_contract, Mapping) or set(host_contract) != (
            _HOST_CONTRACT_KEYS
        ):
            raise ValueError("host contract fields are not exact")
        if not isinstance(snapshot_template, Mapping):
            raise ValueError("snapshot template must be an object")
        for name in (
            "gateway_unit_fragment_path",
            "writer_unit_fragment_path",
            "gateway_config_path",
            "projection_export_path",
        ):
            raw_path = host_contract.get(name)
            if not isinstance(raw_path, str):
                raise ValueError(f"host contract {name} is invalid")
            _absolute_normalized_path(raw_path)
        if (
            host_contract.get("gateway_config_path")
            != str(DEFAULT_GATEWAY_MANAGED_CONFIG_PATH)
        ):
            raise ValueError("gateway managed config path is not pinned")
        for name in (
            "gateway_unit_fragment_sha256",
            "writer_unit_fragment_sha256",
            "gateway_config_sha256",
            "writer_config_sha256",
        ):
            digest = host_contract.get(name)
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"host contract {name} is invalid")
        if snapshot_template.get("deployment_mode") != WRITER_ONLY_MODE:
            raise ValueError("snapshot template is not writer-only")
        database = _mapping(snapshot_template.get("database"))
        connection = _mapping(database.get("connection"))
        if set(connection) != {
            "host",
            "tls_server_name",
            "port",
            "database",
            "user",
        }:
            raise ValueError("snapshot database connection fields are not exact")
        connection_host = connection.get("host")
        try:
            connection_address = ipaddress.ip_address(connection_host)
        except (TypeError, ValueError) as exc:
            raise ValueError("snapshot database host is not exact IPv4") from exc
        if (
            connection_address.version != 4
            or str(connection_address) != connection_host
            or not connection_address.is_private
            or connection_address.is_loopback
            or connection_address.is_link_local
            or connection_address.is_multicast
            or connection_address.is_reserved
            or connection_address.is_unspecified
        ):
            raise ValueError("snapshot database host is not exact private IPv4")
        validate_tls_server_name(connection.get("tls_server_name"))
        if (
            type(connection.get("port")) is not int
            or not 1 <= connection["port"] <= 65535
            or not isinstance(connection.get("database"), str)
            or not connection["database"]
            or not isinstance(connection.get("user"), str)
            or not connection["user"]
            or connection["user"] != database.get("expected_user")
        ):
            raise ValueError("snapshot database connection is invalid")
        discord = _mapping(snapshot_template.get("discord_edge"))
        if (
            discord.get("unit_name") != DEFAULT_DISCORD_EDGE_UNIT
            or discord.get("config_path")
            != str(DEFAULT_DISCORD_EDGE_CONFIG_PATH)
            or discord.get("token_path") != str(DEFAULT_DISCORD_EDGE_TOKEN_PATH)
            or discord.get("socket_path")
            != str(DEFAULT_DISCORD_EDGE_SOCKET_PATH)
        ):
            raise ValueError("Discord edge absence paths are not pinned")
        if snapshot_policy_sha256(snapshot_template) != policy:
            raise ValueError("snapshot template policy digest is invalid")
        for label in ("writer_deployment", "gateway_deployment"):
            deployment_policy = _mapping(
                _mapping(snapshot_template.get(label)).get("policy")
            )
            if (
                deployment_policy.get("revision") != revision
                or deployment_policy.get("artifact_digest_sha256") != artifact
            ):
                raise ValueError(
                    f"snapshot template {label} release is not approved"
                )
            try:
                _preapproved_external_native_mappings(deployment_policy)
            except RuntimeError as exc:
                raise ValueError(
                    f"snapshot template {label} native policy is invalid"
                ) from exc
        gateway_policy = _mapping(
            _mapping(snapshot_template.get("gateway_deployment")).get("policy")
        )
        if (
            gateway_policy.get("module")
            != "gateway.canonical_writer_gateway_bootstrap"
            or gateway_policy.get("read_write_paths")
            != [str(_GATEWAY_RUNTIME_PATH)]
        ):
            raise ValueError("snapshot template gateway policy is not minimal")
        return cls(
            approved_plan_sha256=approved,
            revision=revision,
            artifact_sha256=artifact,
            snapshot_policy_sha256=policy,
            host_contract=copy.deepcopy(dict(host_contract)),
            snapshot_template=copy.deepcopy(dict(snapshot_template)),
            manifest_sha256=_sha256_json(dict(value)),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mode": self.mode,
            "approved_plan_sha256": self.approved_plan_sha256,
            "revision": self.revision,
            "artifact_sha256": self.artifact_sha256,
            "snapshot_policy_sha256": self.snapshot_policy_sha256,
            "host_contract": copy.deepcopy(dict(self.host_contract)),
            "snapshot_template": copy.deepcopy(dict(self.snapshot_template)),
        }


def load_trusted_manifest(
    path: str | os.PathLike[str],
) -> TrustedDeploymentManifest:
    _require_root_linux()
    return TrustedDeploymentManifest.from_mapping(
        _read_trusted_json(path, expected_uid=0, expected_gid=0)
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def snapshot_policy_projection(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return only owner-approved policy, never dynamic host attestations."""

    writer_deployment = _mapping(snapshot.get("writer_deployment"))
    gateway_deployment = _mapping(snapshot.get("gateway_deployment"))
    authority = _mapping(snapshot.get("writer_authority_surface"))
    exporter = _mapping(authority.get("projection_exporter"))
    database = _mapping(snapshot.get("database"))
    writer_socket = _mapping(snapshot.get("socket"))
    discord = _mapping(snapshot.get("discord_edge"))
    return {
        "deployment_mode": snapshot.get("deployment_mode"),
        "identities": {
            name: snapshot.get(name)
            for name in (
                "gateway_uid",
                "gateway_gid",
                "writer_uid",
                "writer_gid",
                "projector_gid",
                "gateway_supplementary_gids",
                "writer_supplementary_gids",
            )
        },
        "socket_expected_group_gid": writer_socket.get("expected_group_gid"),
        "writer_deployment": copy.deepcopy(writer_deployment.get("policy")),
        "gateway_deployment": copy.deepcopy(gateway_deployment.get("policy")),
        "projection_exporter": copy.deepcopy(exporter.get("policy")),
        "database": {
            "expected_user": database.get("expected_user"),
            "connection": copy.deepcopy(database.get("connection")),
            "policy": copy.deepcopy(database.get("policy")),
        },
        "discord_edge": {
            name: discord.get(name)
            for name in (
                "gateway_enabled",
                "writer_authority_enabled",
                "unit_name",
                "config_path",
                "token_path",
                "socket_path",
            )
        },
    }


def snapshot_policy_sha256(snapshot: Mapping[str, Any]) -> str:
    return _sha256_json(snapshot_policy_projection(snapshot))


def _bind_snapshot_to_manifest(
    snapshot: Mapping[str, Any],
    manifest: TrustedDeploymentManifest,
) -> None:
    if snapshot.get("deployment_mode") != WRITER_ONLY_MODE:
        raise ValueError("live snapshot is not writer-only")
    if snapshot_policy_sha256(snapshot) != manifest.snapshot_policy_sha256:
        raise ValueError("live snapshot policy does not match approved manifest")
    for label in ("writer_deployment", "gateway_deployment"):
        policy = _mapping(_mapping(snapshot.get(label)).get("policy"))
        if (
            policy.get("revision") != manifest.revision
            or policy.get("artifact_digest_sha256") != manifest.artifact_sha256
        ):
            raise ValueError(f"{label} release does not match approved manifest")


ActiveHBAProbe = Callable[
    [TrustedDeploymentManifest, Mapping[str, Any]],
    ManagedCloudSQLAdminHBAReceipt | None,
]

_SYSTEMCTL_PATH = "/usr/bin/systemctl"
_SYSTEMD_PROPERTIES = (
    "Type",
    "NotifyAccess",
    "LoadState",
    "ActiveState",
    "SubState",
    "MainPID",
    "StatusText",
    "FragmentPath",
    "DropInPaths",
    "NeedDaemonReload",
    "NoNewPrivileges",
    "PrivateTmp",
    "PrivateDevices",
    "ProtectKernelTunables",
    "ProtectKernelModules",
    "ProtectKernelLogs",
    "ProtectControlGroups",
    "RestrictSUIDSGID",
    "LockPersonality",
    "MemoryDenyWriteExecute",
    "RestrictRealtime",
    "RestrictNamespaces",
    "ProtectSystem",
    "ProtectHome",
    "ProtectProc",
    "ProcSubset",
    "UMask",
    "CapabilityBoundingSet",
    "AmbientCapabilities",
    "LimitCORE",
    "PrivateNetwork",
    "IPAddressDeny",
    "IPAddressAllow",
    "RestrictAddressFamilies",
    "EnvironmentFiles",
    "PassEnvironment",
    "LoadCredential",
    "RootDirectory",
    "RootImage",
    "ReadOnlyPaths",
    "ReadWritePaths",
    "BindPaths",
    "BindReadOnlyPaths",
)


@dataclass(frozen=True)
class RuntimeReadinessBinding:
    gateway_sha256: str
    writer_sha256: str


@dataclass(frozen=True)
class RuntimeLivenessBinding:
    sha256: str
    generation: int


@dataclass(frozen=True)
class RuntimeCodeClosureBinding:
    gateway_sha256: str
    writer_sha256: str


@dataclass(frozen=True)
class RuntimeDirectoryBinding:
    device: int
    inode: int


@dataclass(frozen=True)
class WriterCgroupBPFBinding:
    cgroup_device: int
    cgroup_inode: int
    main_pid: int
    ingress_direct_program_ids: tuple[int, ...]
    ingress_effective_program_ids: tuple[int, ...]
    egress_direct_program_ids: tuple[int, ...]
    egress_effective_program_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cgroup_device": self.cgroup_device,
            "cgroup_inode": self.cgroup_inode,
            "main_pid": self.main_pid,
            "ingress_direct_program_ids": list(
                self.ingress_direct_program_ids
            ),
            "ingress_effective_program_ids": list(
                self.ingress_effective_program_ids
            ),
            "egress_direct_program_ids": list(self.egress_direct_program_ids),
            "egress_effective_program_ids": list(
                self.egress_effective_program_ids
            ),
        }


class _BPFProgQueryAttr(ctypes.Structure):
    """Linux UAPI ``union bpf_attr.query`` through Linux 6.x."""

    _fields_ = (
        ("target_fd", ctypes.c_uint32),
        ("attach_type", ctypes.c_uint32),
        ("query_flags", ctypes.c_uint32),
        ("attach_flags", ctypes.c_uint32),
        ("prog_ids", ctypes.c_uint64),
        ("prog_cnt", ctypes.c_uint32),
        ("padding", ctypes.c_uint32),
        ("prog_attach_flags", ctypes.c_uint64),
        ("link_ids", ctypes.c_uint64),
        ("link_attach_flags", ctypes.c_uint64),
    )


def _parse_bind_read_only_paths(value: str) -> list[str]:
    """Parse the canonical ``systemctl show`` bind-path representation."""

    if not isinstance(value, str):
        raise RuntimeError("systemd BindReadOnlyPaths is invalid")
    if not value:
        return []
    if value != " ".join(value.split()):
        raise RuntimeError("systemd BindReadOnlyPaths is not canonical")
    result: list[str] = []
    for token in value.split():
        if token.count(":") != 2:
            raise RuntimeError("systemd BindReadOnlyPaths is malformed")
        source, destination, options = token.split(":")
        if (
            not source
            or source != destination
            or options != "rbind"
            or any(character.isspace() for character in token)
        ):
            raise RuntimeError("systemd BindReadOnlyPaths is not exact")
        try:
            normalized = _absolute_normalized_path(source)
        except ValueError as exc:
            raise RuntimeError(
                "systemd BindReadOnlyPaths contains an invalid path"
            ) from exc
        if str(normalized) != source or source in result:
            raise RuntimeError("systemd BindReadOnlyPaths contains drift")
        result.append(source)
    return result


def _parse_systemd_ip_networks(
    value: str,
    *,
    allow_any_alias: bool = False,
) -> tuple[str, ...]:
    """Return strict canonical networks from one systemd IP property."""

    if not isinstance(value, str):
        raise RuntimeError("systemd IP network property is invalid")
    if value != " ".join(value.split()):
        raise RuntimeError("systemd IP network property is not canonical")
    if value == "any":
        if not allow_any_alias:
            raise RuntimeError("systemd IP network alias is not allowed")
        return ("0.0.0.0/0", "::/0")
    if not value or "any" in value.split():
        raise RuntimeError("systemd IP network property is incomplete")
    parsed: list[str] = []
    for token in value.split():
        try:
            network = ipaddress.ip_network(token, strict=True)
        except ValueError as exc:
            raise RuntimeError("systemd IP network is invalid") from exc
        canonical = str(network)
        if canonical != token or canonical in parsed:
            raise RuntimeError("systemd IP network is not canonical")
        parsed.append(canonical)
    return tuple(sorted(parsed, key=lambda item: (ipaddress.ip_network(item).version, item)))


def _parse_universal_ip_deny(value: str) -> tuple[str, str]:
    parsed = _parse_systemd_ip_networks(value, allow_any_alias=True)
    if set(parsed) != {"0.0.0.0/0", "::/0"} or len(parsed) != 2:
        raise RuntimeError("systemd IPAddressDeny is not universal")
    return ("0.0.0.0/0", "::/0")


def _validated_program_ids(value: Any, *, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or len(value) > _MAX_BPF_PROGRAM_IDS
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise RuntimeError(f"{label} BPF program IDs are invalid")
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        raise RuntimeError(f"{label} BPF program IDs are not canonical")
    return result


def _bpf_syscall_number() -> int:
    if sys.platform != "linux":
        raise RuntimeError("BPF_PROG_QUERY requires Linux")
    try:
        machine = os.uname().machine
    except (AttributeError, OSError) as exc:
        raise RuntimeError("BPF_PROG_QUERY architecture is unavailable") from exc
    numbers = {"x86_64": 321, "aarch64": 280}
    try:
        return numbers[machine]
    except KeyError as exc:
        raise RuntimeError("BPF_PROG_QUERY architecture is not approved") from exc


def _query_bpf_program_ids(
    cgroup_fd: int,
    *,
    attach_type: int,
    effective: bool,
) -> tuple[int, ...]:
    if attach_type not in {_BPF_CGROUP_INET_INGRESS, _BPF_CGROUP_INET_EGRESS}:
        raise ValueError("BPF cgroup attach type is not pinned")
    if type(cgroup_fd) is not int or cgroup_fd < 0:
        raise ValueError("BPF cgroup descriptor is invalid")
    identifiers = (ctypes.c_uint32 * _MAX_BPF_PROGRAM_IDS)()
    attributes = _BPFProgQueryAttr(
        target_fd=cgroup_fd,
        attach_type=attach_type,
        query_flags=_BPF_F_QUERY_EFFECTIVE if effective else 0,
        prog_ids=ctypes.addressof(identifiers),
        prog_cnt=_MAX_BPF_PROGRAM_IDS,
    )
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = getattr(libc, "syscall", None)
    if syscall is None:
        raise RuntimeError("BPF_PROG_QUERY syscall is unavailable")
    syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = syscall(
        ctypes.c_long(_bpf_syscall_number()),
        ctypes.c_uint(_BPF_PROG_QUERY),
        ctypes.byref(attributes),
        ctypes.c_uint(ctypes.sizeof(attributes)),
    )
    if result != 0:
        error_number = ctypes.get_errno()
        detail = errno.errorcode.get(error_number, "UNKNOWN")
        raise RuntimeError(f"BPF_PROG_QUERY failed closed: {detail}")
    count = int(attributes.prog_cnt)
    if count > _MAX_BPF_PROGRAM_IDS:
        raise RuntimeError("BPF_PROG_QUERY exceeded its program bound")
    raw = tuple(int(identifiers[index]) for index in range(count))
    if not raw:
        return ()
    return _validated_program_ids(sorted(raw), label="queried")


def _fstatfs_magic(descriptor: int) -> int:
    buffer = ctypes.create_string_buffer(256)
    libc = ctypes.CDLL(None, use_errno=True)
    fstatfs = getattr(libc, "fstatfs", None)
    if fstatfs is None:
        raise RuntimeError("cgroup filesystem identity is unavailable")
    fstatfs.argtypes = (ctypes.c_int, ctypes.c_void_p)
    fstatfs.restype = ctypes.c_int
    ctypes.set_errno(0)
    if fstatfs(descriptor, ctypes.byref(buffer)) != 0:
        detail = errno.errorcode.get(ctypes.get_errno(), "UNKNOWN")
        raise RuntimeError(f"cgroup filesystem identity failed: {detail}")
    return int(ctypes.cast(buffer, ctypes.POINTER(ctypes.c_long)).contents.value)


def _required_linux_open_flag(name: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"required Linux open flag {name} is unavailable")
    return value


def _parse_cgroup_procs(raw: bytes) -> tuple[int, ...]:
    if (
        not isinstance(raw, bytes)
        or not raw
        or len(raw) > _MAX_CGROUP_PROCS_BYTES
        or not raw.endswith(b"\n")
    ):
        raise RuntimeError("writer cgroup.procs payload is invalid")
    lines = raw[:-1].split(b"\n")
    if not lines or any(re.fullmatch(rb"[1-9][0-9]*", line) is None for line in lines):
        raise RuntimeError("writer cgroup.procs payload is malformed")
    try:
        values = tuple(int(line) for line in lines)
    except ValueError as exc:
        raise RuntimeError("writer cgroup.procs PID is invalid") from exc
    if any(value <= 1 for value in values) or len(values) != len(set(values)):
        raise RuntimeError("writer cgroup.procs PID set is invalid")
    return tuple(sorted(values))


def _read_cgroup_procs(
    directory_fd: int,
    *,
    expected_device: int,
) -> tuple[int, ...]:
    flags = (
        os.O_RDONLY
        | _required_linux_open_flag("O_CLOEXEC")
        | _required_linux_open_flag("O_NOFOLLOW")
    )
    try:
        descriptor = os.open(
            "cgroup.procs",
            flags,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise RuntimeError("writer cgroup.procs cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != 0
            or before.st_gid != 0
            or stat.S_IMODE(before.st_mode) & 0o022
            or before.st_dev != expected_device
        ):
            raise RuntimeError("writer cgroup.procs identity is invalid")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_CGROUP_PROCS_BYTES:
            try:
                chunk = os.read(
                    descriptor,
                    min(512, _MAX_CGROUP_PROCS_BYTES + 1 - total),
                )
            except OSError as exc:
                raise RuntimeError("writer cgroup.procs read failed") from exc
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or after.st_mode != before.st_mode
            or after.st_uid != before.st_uid
            or after.st_gid != before.st_gid
            or len(raw) > _MAX_CGROUP_PROCS_BYTES
        ):
            raise RuntimeError("writer cgroup.procs rotated during read")
        return _parse_cgroup_procs(raw)
    finally:
        os.close(descriptor)


def _validate_writer_cgroup_stat(item: os.stat_result) -> None:
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISDIR(item.st_mode)
        or item.st_uid != 0
        or item.st_gid != 0
        or stat.S_IMODE(item.st_mode) & 0o022
    ):
        raise RuntimeError("writer cgroup identity is not root-controlled")


@contextlib.contextmanager
def _verified_writer_cgroup_fd():
    path = _absolute_normalized_path(_WRITER_CGROUP_PATH)
    _validate_parent_chain(path.parent, expected_uid=0)
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise RuntimeError("writer cgroup identity is unavailable") from exc
    _validate_writer_cgroup_stat(before)
    if _has_posix_acl(path):
        raise RuntimeError("writer cgroup has an unreviewed POSIX ACL")
    flags = (
        os.O_RDONLY
        | _required_linux_open_flag("O_CLOEXEC")
        | _required_linux_open_flag("O_DIRECTORY")
        | _required_linux_open_flag("O_NOFOLLOW")
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError("writer cgroup cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        _validate_writer_cgroup_stat(opened)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("writer cgroup rotated during open")
        if _fstatfs_magic(descriptor) != _CGROUP2_SUPER_MAGIC:
            raise RuntimeError("writer cgroup is not on cgroup v2")
        yield descriptor, int(opened.st_dev), int(opened.st_ino)
        after_fd = os.fstat(descriptor)
        after_path = os.lstat(path)
        _validate_writer_cgroup_stat(after_fd)
        _validate_writer_cgroup_stat(after_path)
        if (
            (after_fd.st_dev, after_fd.st_ino)
            != (opened.st_dev, opened.st_ino)
            or (after_path.st_dev, after_path.st_ino)
            != (opened.st_dev, opened.st_ino)
            or _has_posix_acl(path)
        ):
            raise RuntimeError("writer cgroup rotated during BPF query")
    finally:
        os.close(descriptor)


def _collect_writer_cgroup_bpf_binding(
    *,
    writer_pid: int,
) -> WriterCgroupBPFBinding:
    if type(writer_pid) is not int or writer_pid <= 1:
        raise ValueError("writer cgroup MainPID is invalid")
    with _verified_writer_cgroup_fd() as (descriptor, device, inode):
        procs_before = _read_cgroup_procs(
            descriptor,
            expected_device=device,
        )
        if procs_before != (writer_pid,):
            raise RuntimeError("writer cgroup does not contain exact MainPID")
        observations: list[dict[tuple[int, bool], tuple[int, ...]]] = []
        for _sample in range(2):
            sample = {
                (attach_type, effective): _query_bpf_program_ids(
                    descriptor,
                    attach_type=attach_type,
                    effective=effective,
                )
                for attach_type in (
                    _BPF_CGROUP_INET_INGRESS,
                    _BPF_CGROUP_INET_EGRESS,
                )
                for effective in (False, True)
            }
            observations.append(sample)
        procs_after = _read_cgroup_procs(
            descriptor,
            expected_device=device,
        )
        if procs_after != procs_before:
            raise RuntimeError("writer cgroup process membership rotated")
    first, second = observations
    if first != second:
        raise RuntimeError("writer cgroup BPF programs rotated during collection")
    ingress_direct = _validated_program_ids(
        first[(_BPF_CGROUP_INET_INGRESS, False)],
        label="ingress direct",
    )
    ingress_effective = _validated_program_ids(
        first[(_BPF_CGROUP_INET_INGRESS, True)],
        label="ingress effective",
    )
    egress_direct = _validated_program_ids(
        first[(_BPF_CGROUP_INET_EGRESS, False)],
        label="egress direct",
    )
    egress_effective = _validated_program_ids(
        first[(_BPF_CGROUP_INET_EGRESS, True)],
        label="egress effective",
    )
    if (
        not set(ingress_direct).issubset(ingress_effective)
        or not set(egress_direct).issubset(egress_effective)
    ):
        raise RuntimeError("writer cgroup direct BPF programs are not effective")
    return WriterCgroupBPFBinding(
        cgroup_device=device,
        cgroup_inode=inode,
        main_pid=writer_pid,
        ingress_direct_program_ids=ingress_direct,
        ingress_effective_program_ids=ingress_effective,
        egress_direct_program_ids=egress_direct,
        egress_effective_program_ids=egress_effective,
    )


def _require_exact_bpf_binding(
    observed: WriterCgroupBPFBinding,
    *,
    cgroup_device: int,
    cgroup_inode: int,
    expected_main_pid: int,
    ingress_direct_program_ids: Sequence[int],
    ingress_effective_program_ids: Sequence[int],
    egress_direct_program_ids: Sequence[int],
    egress_effective_program_ids: Sequence[int],
) -> None:
    expected = WriterCgroupBPFBinding(
        cgroup_device=cgroup_device,
        cgroup_inode=cgroup_inode,
        main_pid=expected_main_pid,
        ingress_direct_program_ids=_validated_program_ids(
            ingress_direct_program_ids,
            label="expected ingress direct",
        ),
        ingress_effective_program_ids=_validated_program_ids(
            ingress_effective_program_ids,
            label="expected ingress effective",
        ),
        egress_direct_program_ids=_validated_program_ids(
            egress_direct_program_ids,
            label="expected egress direct",
        ),
        egress_effective_program_ids=_validated_program_ids(
            egress_effective_program_ids,
            label="expected egress effective",
        ),
    )
    if observed != expected:
        raise RuntimeError("writer cgroup BPF enforcement identity changed")


def _systemctl_show(unit_name: str) -> dict[str, str]:
    if unit_name not in {
        DEFAULT_GATEWAY_UNIT,
        DEFAULT_WRITER_UNIT,
        DEFAULT_DISCORD_EDGE_UNIT,
    }:
        raise ValueError("collector systemd unit is not pinned")
    command = [
        _SYSTEMCTL_PATH,
        "show",
        *(f"--property={name}" for name in _SYSTEMD_PROPERTIES),
        "--",
        unit_name,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=3,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise RuntimeError("bounded systemd evidence collection failed") from exc
    if completed.returncode != 0 or len(completed.stdout.encode("utf-8")) > 64 * 1024:
        raise RuntimeError("bounded systemd evidence collection failed")
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator != "=" or name not in _SYSTEMD_PROPERTIES or name in result:
            raise RuntimeError("systemd evidence fields are invalid")
        result[name] = value
    missing = set(_SYSTEMD_PROPERTIES) - set(result)
    if missing == {"EnvironmentFiles"}:
        # systemd 252 serializes EnvironmentFiles through a custom a(sb)
        # printer. Unlike generic empty properties, an empty array emits no
        # line at all. The live fragment digest and the writer-only unit
        # renderer independently prohibit EnvironmentFile=, so normalize only
        # this one version-specific omission to its exact empty value.
        result["EnvironmentFiles"] = ""
    elif missing:
        raise RuntimeError("systemd evidence is incomplete")
    return result


def _process_start_time_ticks(pid: int) -> int:
    if type(pid) is not int or pid <= 1:
        raise RuntimeError("collector process PID is invalid")
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError as exc:
        raise RuntimeError("collector process identity is unavailable") from exc
    suffix = raw.rsplit(")", 1)
    if len(suffix) != 2:
        raise RuntimeError("collector process identity is invalid")
    try:
        value = int(suffix[1].strip().split()[19])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("collector process start time is invalid") from exc
    if value <= 0:
        raise RuntimeError("collector process start time is invalid")
    return value


def _process_identity(pid: int) -> dict[str, Any]:
    start_before = _process_start_time_ticks(pid)
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
        cmdline_raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        executable = os.readlink(f"/proc/{pid}/exe")
        limits = Path(f"/proc/{pid}/limits").read_text(encoding="ascii")
        maps = Path(f"/proc/{pid}/maps").read_text(encoding="utf-8")
        environ_raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError as exc:
        raise RuntimeError("collector process evidence is unavailable") from exc
    if (
        len(status) > 256 * 1024
        or len(cmdline_raw) > 64 * 1024
        or len(maps) > 16 * 1024 * 1024
        or len(environ_raw) > 1024 * 1024
    ):
        raise RuntimeError("collector process evidence exceeds its bound")
    fields: dict[str, str] = {}
    for line in status.splitlines():
        name, separator, value = line.partition(":")
        if separator:
            fields[name] = value.strip()
    try:
        effective_uid = int(fields["Uid"].split()[1])
        effective_gid = int(fields["Gid"].split()[1])
        supplementary = sorted(int(value) for value in fields["Groups"].split())
    except (IndexError, KeyError, ValueError) as exc:
        raise RuntimeError("collector process credential evidence is invalid") from exc
    core_soft = core_hard = -1
    for line in limits.splitlines():
        if line.startswith("Max core file size"):
            values = line[len("Max core file size") :].split()
            if len(values) >= 2 and values[0].isdigit() and values[1].isdigit():
                core_soft, core_hard = int(values[0]), int(values[1])
            break
    try:
        cmdline = [
            value.decode("utf-8", errors="strict")
            for value in cmdline_raw.rstrip(b"\x00").split(b"\x00")
            if value
        ]
    except UnicodeError as exc:
        raise RuntimeError("collector process argv is invalid") from exc
    environment_values: dict[str, str] = {}
    environment_value_sha256: dict[str, str] = {}
    try:
        for item in environ_raw.rstrip(b"\x00").split(b"\x00"):
            if not item:
                continue
            raw_name, separator, _raw_value = item.partition(b"=")
            name = raw_name.decode("ascii", errors="strict")
            if separator != b"=" or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                raise RuntimeError("collector process environment is invalid")
            if name in environment_values:
                raise RuntimeError("collector process environment contains duplicates")
            environment_values[name] = _raw_value.decode(
                "utf-8",
                errors="surrogateescape",
            )
            environment_value_sha256[name] = hashlib.sha256(_raw_value).hexdigest()
    except UnicodeError as exc:
        raise RuntimeError("collector process environment is invalid") from exc
    notify_socket = environment_values.get("NOTIFY_SOCKET")
    invocation_id = environment_values.get("INVOCATION_ID")
    journal_stream = environment_values.get("JOURNAL_STREAM")
    systemd_exec_pid = environment_values.get("SYSTEMD_EXEC_PID")
    nss_bypass = environment_values.get("SYSTEMD_NSS_DYNAMIC_BYPASS")
    pressure_watch = environment_values.get("MEMORY_PRESSURE_WATCH")
    pressure_write = environment_values.get("MEMORY_PRESSURE_WRITE")
    if (
        notify_socket is not None
        and (
            len(notify_socket.encode("utf-8", errors="surrogateescape")) > 100
            or any(
                character in notify_socket
                for character in ("\x00", "\n", "\r")
            )
            or not notify_socket.startswith(("/", "@"))
        )
    ):
        raise RuntimeError("collector systemd notify environment is invalid")
    if (
        invocation_id is not None
        and re.fullmatch(r"[0-9a-f]{32}", invocation_id) is None
    ):
        raise RuntimeError("collector systemd invocation identity is invalid")
    if (
        journal_stream is not None
        and re.fullmatch(r"[0-9]+:[0-9]+", journal_stream) is None
    ):
        raise RuntimeError("collector journal stream identity is invalid")
    if systemd_exec_pid is not None and systemd_exec_pid != str(pid):
        raise RuntimeError("collector systemd executable PID is invalid")
    if nss_bypass is not None and nss_bypass != "1":
        raise RuntimeError("collector systemd NSS boundary is invalid")
    if pressure_watch is not None and str(
        _absolute_normalized_path(pressure_watch)
    ) != pressure_watch:
        raise RuntimeError("collector memory pressure watch path is invalid")
    if pressure_write is not None and (
        not pressure_write
        or len(pressure_write.encode("utf-8", errors="surrogateescape")) > 4096
        or any(
            character in pressure_write
            for character in ("\x00", "\n", "\r")
        )
    ):
        raise RuntimeError("collector memory pressure payload is invalid")
    mapped_executable_paths: set[str] = set()
    deleted_code_mappings: set[str] = set()
    writable_code_mappings: set[str] = set()
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 5:
            raise RuntimeError("collector process maps evidence is invalid")
        permissions = fields[1]
        if len(fields) != 6 or "x" not in permissions:
            continue
        raw_path = fields[5]
        if not raw_path.startswith("/"):
            continue
        deleted = raw_path.endswith(" (deleted)")
        path = raw_path[: -len(" (deleted)")] if deleted else raw_path
        normalized = _absolute_normalized_path(path)
        mapped_executable_paths.add(str(normalized))
        if deleted:
            deleted_code_mappings.add(str(normalized))
        if "w" in permissions:
            writable_code_mappings.add(str(normalized))
    fd_targets: list[str] = []
    try:
        for entry in sorted(
            Path(f"/proc/{pid}/fd").iterdir(),
            key=lambda value: int(value.name),
        ):
            try:
                target = os.readlink(entry)
            except FileNotFoundError:
                continue
            if "\x00" in target or "\n" in target or "\r" in target:
                raise RuntimeError("collector process descriptor evidence is invalid")
            fd_targets.append(target)
    except OSError as exc:
        raise RuntimeError("collector process descriptor evidence is unavailable") from exc
    socket_inodes = {
        match.group(1)
        for target in fd_targets
        if (match := re.fullmatch(r"socket:\[([0-9]+)\]", target)) is not None
    }
    unix_socket_paths: list[str] = []
    try:
        unix_rows = Path(f"/proc/{pid}/net/unix").read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError("collector Unix socket evidence is unavailable") from exc
    if len(unix_rows) > 4 * 1024 * 1024:
        raise RuntimeError("collector Unix socket evidence exceeds its bound")
    for line in unix_rows.splitlines()[1:]:
        fields = line.split(maxsplit=7)
        if len(fields) >= 7 and fields[6] in socket_inodes and len(fields) == 8:
            unix_socket_paths.append(fields[7])
    start_after = _process_start_time_ticks(pid)
    if start_before != start_after:
        raise RuntimeError("collector process rotated during collection")
    return {
        "pid": pid,
        "start_time_ticks": start_before,
        "effective_uid": effective_uid,
        "effective_gid": effective_gid,
        "supplementary_gids": supplementary,
        "cmdline": cmdline,
        "executable": executable,
        "core_soft_limit": core_soft,
        "core_hard_limit": core_hard,
        "environment_variable_names": sorted(environment_values),
        "environment_variable_value_sha256": {
            name: environment_value_sha256[name]
            for name in sorted(environment_value_sha256)
        },
        "mapped_executable_paths": sorted(mapped_executable_paths),
        "deleted_code_mappings": sorted(deleted_code_mappings),
        "writable_code_mappings": sorted(writable_code_mappings),
        "fd_targets": fd_targets,
        "unix_socket_paths": sorted(set(unix_socket_paths)),
    }


def _unix_listener_paths_for_pid(pid: int) -> list[str]:
    """Return exact AF_UNIX listener paths held by one stable process."""

    start_before = _process_start_time_ticks(pid)
    socket_inodes: set[str] = set()
    try:
        for entry in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(entry)
            except FileNotFoundError:
                continue
            match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
            if match is not None:
                socket_inodes.add(match.group(1))
        raw = Path(f"/proc/{pid}/net/unix").read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError("collector Unix listener evidence is unavailable") from exc
    if len(raw.encode("utf-8")) > 4 * 1024 * 1024:
        raise RuntimeError("collector Unix listener evidence exceeds its bound")
    listeners: set[str] = set()
    for line in raw.splitlines()[1:]:
        fields = line.split(maxsplit=7)
        if len(fields) != 8 or fields[6] not in socket_inodes:
            continue
        try:
            flags = int(fields[3], 16)
        except ValueError as exc:
            raise RuntimeError("collector Unix listener flags are invalid") from exc
        if fields[4] == "0001" and flags & 0x00010000:
            listeners.add(fields[7])
    if _process_start_time_ticks(pid) != start_before:
        raise RuntimeError("collector listener process rotated")
    return sorted(listeners)


def _validate_runtime_directory_stat(
    item: os.stat_result,
    *,
    writer_uid: int,
    socket_gid: int,
) -> None:
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISDIR(item.st_mode)
        or item.st_uid != writer_uid
        or item.st_gid != socket_gid
        or stat.S_IMODE(item.st_mode) != 0o2750
    ):
        raise RuntimeError("writer runtime directory identity is invalid")


def _collect_writer_runtime_directory(
    *,
    writer_uid: int,
    socket_gid: int,
) -> RuntimeDirectoryBinding:
    path = _absolute_normalized_path(DEFAULT_SOCKET_PATH.parent)
    _validate_parent_chain(path.parent, expected_uid=0)
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise RuntimeError("writer runtime directory is unavailable") from exc
    _validate_runtime_directory_stat(
        before,
        writer_uid=writer_uid,
        socket_gid=socket_gid,
    )
    if _has_posix_acl(path):
        raise RuntimeError("writer runtime directory has a POSIX ACL")
    flags = (
        os.O_RDONLY
        | _required_linux_open_flag("O_CLOEXEC")
        | _required_linux_open_flag("O_DIRECTORY")
        | _required_linux_open_flag("O_NOFOLLOW")
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError("writer runtime directory cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        _validate_runtime_directory_stat(
            opened,
            writer_uid=writer_uid,
            socket_gid=socket_gid,
        )
        after = os.lstat(path)
        _validate_runtime_directory_stat(
            after,
            writer_uid=writer_uid,
            socket_gid=socket_gid,
        )
        if (
            (before.st_dev, before.st_ino)
            != (opened.st_dev, opened.st_ino)
            or (after.st_dev, after.st_ino)
            != (opened.st_dev, opened.st_ino)
            or _has_posix_acl(path)
        ):
            raise RuntimeError("writer runtime directory rotated during collection")
        return RuntimeDirectoryBinding(
            device=int(opened.st_dev),
            inode=int(opened.st_ino),
        )
    finally:
        os.close(descriptor)


def _validate_writer_socket_stat(
    item: os.stat_result,
    *,
    writer_uid: int,
    socket_gid: int,
    expected_device: int | None,
    expected_inode: int | None,
) -> None:
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISSOCK(item.st_mode)
        or item.st_uid != writer_uid
        or item.st_gid != socket_gid
        or stat.S_IMODE(item.st_mode) != 0o660
        or (expected_device is not None and item.st_dev != expected_device)
        or (expected_inode is not None and item.st_ino != expected_inode)
    ):
        raise RuntimeError("writer socket pathname identity is invalid")


def _connect_and_validate_writer_socket_peer(
    *,
    writer_pid: int,
    writer_uid: int,
    writer_gid: int,
    socket_gid: int,
    expected_device: int | None = None,
    expected_inode: int | None = None,
) -> os.stat_result:
    path = _absolute_normalized_path(DEFAULT_SOCKET_PATH)
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise RuntimeError("writer socket pathname is unavailable") from exc
    _validate_writer_socket_stat(
        before,
        writer_uid=writer_uid,
        socket_gid=socket_gid,
        expected_device=expected_device,
        expected_inode=expected_inode,
    )
    if _has_posix_acl(path):
        raise RuntimeError("writer socket pathname has a POSIX ACL")
    if str(path) not in _unix_listener_paths_for_pid(writer_pid):
        raise RuntimeError("writer MainPID does not hold the Unix listener")
    peer_option = getattr(socket, "SO_PEERCRED", None)
    if type(peer_option) is not int:
        raise RuntimeError("Linux SO_PEERCRED is unavailable")
    credentials_size = struct.calcsize("=3i")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(1.0)
            client.connect(str(path))
            raw_credentials = client.getsockopt(
                socket.SOL_SOCKET,
                peer_option,
                credentials_size,
            )
    except (OSError, TimeoutError) as exc:
        raise RuntimeError("writer socket peer connection failed") from exc
    if len(raw_credentials) != credentials_size:
        raise RuntimeError("writer socket peer credentials are incomplete")
    peer_pid, peer_uid, peer_gid = struct.unpack("=3i", raw_credentials)
    if (peer_pid, peer_uid, peer_gid) != (writer_pid, writer_uid, writer_gid):
        raise RuntimeError("writer socket SO_PEERCRED identity drifted")
    try:
        after = os.lstat(path)
    except OSError as exc:
        raise RuntimeError("writer socket pathname rotated") from exc
    _validate_writer_socket_stat(
        after,
        writer_uid=writer_uid,
        socket_gid=socket_gid,
        expected_device=before.st_dev,
        expected_inode=before.st_ino,
    )
    if _has_posix_acl(path):
        raise RuntimeError("writer socket pathname acquired a POSIX ACL")
    if str(path) not in _unix_listener_paths_for_pid(writer_pid):
        raise RuntimeError("writer MainPID listener rotated after SO_PEERCRED")
    return after


def _sha256_trusted_file(
    path_value: str,
    *,
    expected_uid: int = 0,
    maximum: int = 64 * 1024 * 1024,
) -> str:
    path = _absolute_normalized_path(path_value)
    _validate_parent_chain(path.parent, expected_uid=expected_uid)
    before = os.lstat(path)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != expected_uid
        or stat.S_IMODE(before.st_mode) & 0o022
        or _has_posix_acl(path)
    ):
        raise RuntimeError("collector protected file identity is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    total = 0
    try:
        actual = os.fstat(descriptor)
        if (
            (actual.st_dev, actual.st_ino) != (before.st_dev, before.st_ino)
            or actual.st_uid != before.st_uid
            or actual.st_gid != before.st_gid
            or actual.st_mode != before.st_mode
            or actual.st_size != before.st_size
            or actual.st_mtime_ns != before.st_mtime_ns
            or actual.st_ctime_ns != before.st_ctime_ns
        ):
            raise RuntimeError("collector protected file rotated during open")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise RuntimeError("collector protected file exceeds its bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_size != actual.st_size
            or after.st_mtime_ns != actual.st_mtime_ns
            or after.st_ctime_ns != actual.st_ctime_ns
            or after.st_mode != actual.st_mode
            or after.st_uid != actual.st_uid
            or after.st_gid != actual.st_gid
        ):
            raise RuntimeError("collector protected file changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _release_relative_path(value: Any) -> str:
    if not isinstance(value, str):
        raise RuntimeError("release manifest path is invalid")
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or value != path.as_posix()
        or value in {".", _RELEASE_MANIFEST_NAME, _RELEASE_INCOMPLETE_MARKER}
        or ".." in path.parts
        or _RELEASE_PATH_RE.fullmatch(value) is None
    ):
        raise RuntimeError("release manifest path is invalid")
    return value


def _live_release_paths(root: Path) -> list[str]:
    paths: list[str] = []
    for current, directories, files in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        directories.sort()
        files.sort()
        current_path = Path(current)
        for name in [*directories, *files]:
            relative = (current_path / name).relative_to(root).as_posix()
            if relative == _RELEASE_MANIFEST_NAME:
                continue
            if relative == _RELEASE_INCOMPLETE_MARKER:
                raise RuntimeError("release artifact is incomplete")
            paths.append(_release_relative_path(relative))
    return sorted(paths)


def _verify_release_artifact(
    snapshot: Mapping[str, Any],
    manifest: TrustedDeploymentManifest,
    *,
    _expected_uid: int = 0,
    _expected_gid: int = 0,
) -> None:
    writer_policy = _mapping(
        _mapping(snapshot.get("writer_deployment")).get("policy")
    )
    gateway_policy = _mapping(
        _mapping(snapshot.get("gateway_deployment")).get("policy")
    )
    root_value = writer_policy.get("artifact_root")
    if (
        not isinstance(root_value, str)
        or gateway_policy.get("artifact_root") != root_value
    ):
        raise RuntimeError("release artifact root is not shared and pinned")
    root = _absolute_normalized_path(root_value)
    _validate_parent_chain(root.parent, expected_uid=_expected_uid)
    root_before = os.lstat(root)
    if (
        stat.S_ISLNK(root_before.st_mode)
        or not stat.S_ISDIR(root_before.st_mode)
        or root_before.st_uid != _expected_uid
        or root_before.st_gid != _expected_gid
        or stat.S_IMODE(root_before.st_mode) != 0o555
        or root.name != manifest.revision
        or _list_xattrs(root)
    ):
        raise RuntimeError("release artifact root identity is invalid")
    release_manifest = _read_trusted_json(
        root / _RELEASE_MANIFEST_NAME,
        expected_uid=_expected_uid,
        expected_gid=_expected_gid,
        maximum=_MAX_RELEASE_MANIFEST_BYTES,
    )
    if set(release_manifest) != _RELEASE_MANIFEST_KEYS:
        raise RuntimeError("release manifest fields are not exact")
    unsigned = {
        name: copy.deepcopy(value)
        for name, value in release_manifest.items()
        if name != "artifact_sha256"
    }
    writer_origin = writer_policy.get("module_origin")
    gateway_origin = gateway_policy.get("module_origin")
    if (
        release_manifest.get("schema") != _RELEASE_SCHEMA
        or release_manifest.get("revision") != manifest.revision
        or release_manifest.get("artifact_root") != str(root)
        or release_manifest.get("artifact_sha256") != manifest.artifact_sha256
        or _sha256_json(unsigned) != manifest.artifact_sha256
        or release_manifest.get("interpreter") != writer_policy.get("interpreter")
        or gateway_policy.get("interpreter") != writer_policy.get("interpreter")
        or release_manifest.get("writer_module")
        != "gateway.canonical_writer_bootstrap"
        or release_manifest.get("gateway_module")
        != "gateway.canonical_writer_gateway_bootstrap"
        or release_manifest.get("writer_module_origin") != writer_origin
        or release_manifest.get("gateway_module_origin") != gateway_origin
    ):
        raise RuntimeError("release manifest does not match approved artifact")
    entries = release_manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("release manifest entries are invalid")
    declared_paths: list[str] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise RuntimeError("release manifest entry is invalid")
        kind = raw_entry.get("kind")
        expected_keys = {
            "file": {"path", "kind", "mode", "size", "sha256"},
            "directory": {"path", "kind", "mode"},
            "symlink": {"path", "kind", "mode", "target"},
        }.get(kind)
        if expected_keys is None or set(raw_entry) != expected_keys:
            raise RuntimeError("release manifest entry fields are invalid")
        relative = _release_relative_path(raw_entry.get("path"))
        declared_paths.append(relative)
        path = root / relative
        item = os.lstat(path)
        mode = f"{stat.S_IMODE(item.st_mode):04o}"
        release_xattrs = set(_list_xattrs(path))
        if (
            item.st_uid != _expected_uid
            or item.st_gid != _expected_gid
            or raw_entry.get("mode") != mode
            or release_xattrs
        ):
            raise RuntimeError("release entry ownership or mode changed")
        if kind == "directory":
            if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
                raise RuntimeError("release directory identity changed")
        elif kind == "file":
            declared_size = raw_entry.get("size")
            declared_sha256 = raw_entry.get("sha256")
            if (
                stat.S_ISLNK(item.st_mode)
                or not stat.S_ISREG(item.st_mode)
                or item.st_nlink != 1
                or type(declared_size) is not int
                or declared_size < 0
                or item.st_size != declared_size
                or not isinstance(declared_sha256, str)
                or _SHA256_RE.fullmatch(declared_sha256) is None
                or _sha256_trusted_file(
                    str(path),
                    expected_uid=_expected_uid,
                    maximum=_MAX_RELEASE_FILE_BYTES,
                )
                != declared_sha256
            ):
                raise RuntimeError("release file identity or digest changed")
        else:
            target = raw_entry.get("target")
            if (
                not stat.S_ISLNK(item.st_mode)
                or not isinstance(target, str)
                or not target
                or os.readlink(path) != target
            ):
                raise RuntimeError("release symlink identity changed")
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise RuntimeError("release symlink target is unavailable") from exc
            if resolved != root and root not in resolved.parents:
                raise RuntimeError("release symlink escapes the artifact")
    if (
        declared_paths != sorted(declared_paths)
        or len(declared_paths) != len(set(declared_paths))
        or declared_paths != _live_release_paths(root)
    ):
        raise RuntimeError("release manifest paths do not match the live artifact")
    root_after = os.lstat(root)
    if (
        root_after.st_dev != root_before.st_dev
        or root_after.st_ino != root_before.st_ino
        or root_after.st_mode != root_before.st_mode
        or root_after.st_uid != root_before.st_uid
        or root_after.st_gid != root_before.st_gid
        or root_after.st_mtime_ns != root_before.st_mtime_ns
        or root_after.st_ctime_ns != root_before.st_ctime_ns
    ):
        raise RuntimeError("release artifact changed during collection")


def _read_runtime_receipt(path: Path, *, expected_uid: int) -> Mapping[str, Any]:
    return _read_trusted_json(
        path,
        expected_uid=expected_uid,
        expected_gid=os.lstat(path).st_gid,
        require_trusted_parents=False,
        allowed_modes=frozenset({0o600}),
        maximum=256 * 1024,
    )


def _environment_value_sha256(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8", errors="surrogateescape")
    ).hexdigest()


def _validate_exact_runtime_environment(
    process: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    fixed_values: Mapping[str, str],
    in_process_only_names: frozenset[str] = frozenset(),
) -> str:
    """Bind exec-time and current in-process environment without values."""

    if not in_process_only_names.issubset(fixed_values):
        raise ValueError("in-process-only environment policy is invalid")
    process_names = process.get("environment_variable_names")
    process_hashes = process.get("environment_variable_value_sha256")
    receipt_names = receipt.get("effective_environment_variable_names")
    receipt_hashes = receipt.get(
        "effective_environment_variable_value_sha256"
    )
    if any(
        not isinstance(names, list)
        or names != sorted(set(names))
        or any(not isinstance(name, str) for name in names)
        for names in (process_names, receipt_names)
    ) or any(
        not isinstance(hashes, Mapping)
        or list(hashes) != names
        or any(
            not isinstance(value, str)
            or _SHA256_RE.fullmatch(value) is None
            for value in hashes.values()
        )
        for names, hashes in (
            (process_names, process_hashes),
            (receipt_names, receipt_hashes),
        )
    ):
        raise RuntimeError("writer-only environment evidence is invalid")
    if (
        not set(process_names).issubset(receipt_names)
        or any(process_hashes[name] != receipt_hashes[name] for name in process_names)
        or any(name not in _WRITER_ONLY_ALLOWED_ENVIRONMENT_NAMES for name in receipt_names)
        or not set(fixed_values).issubset(receipt_names)
        or not (set(fixed_values) - in_process_only_names).issubset(
            process_names
        )
        or "NOTIFY_SOCKET" not in process_names
        or "NOTIFY_SOCKET" not in receipt_names
    ):
        raise RuntimeError("writer-only environment authority drifted")
    for name, expected in fixed_values.items():
        if receipt_hashes.get(name) != _environment_value_sha256(expected):
            raise RuntimeError("writer-only fixed environment value drifted")
    if set(receipt_names) - set(process_names) != set(in_process_only_names):
        raise RuntimeError("writer-only process mutated its environment surface")
    return _sha256_json(
        {
            "names": receipt_names,
            "value_sha256": dict(receipt_hashes),
        }
    )


def _collect_runtime_liveness(
    snapshot: dict[str, Any],
    *,
    gateway_readiness_sha256: str,
    current_boot_id_sha256: str,
    current_boottime_ns: int,
) -> RuntimeLivenessBinding:
    """Collect one fresh MainPID-authenticated writer PING generation."""

    from gateway.canonical_writer_protocol import MAX_SEQUENCE
    from gateway.canonical_writer_readiness import (
        DEFAULT_WRITER_LIVENESS_RECEIPT_PATH,
        WRITER_LIVENESS_RECEIPT_VERSION,
        writer_liveness_status_text,
    )

    if (
        not isinstance(gateway_readiness_sha256, str)
        or _SHA256_RE.fullmatch(gateway_readiness_sha256) is None
    ):
        raise RuntimeError("writer liveness startup digest is invalid")
    gateway_process = _mapping(snapshot.get("gateway_process"))
    writer_socket = _mapping(snapshot.get("socket"))
    receipt: Mapping[str, Any] | None = None
    properties: Mapping[str, str] | None = None
    # Atomic receipt replacement precedes sd_notify.  Sample receipt/status/
    # receipt so a generation transition is either coherent or retried; an
    # unprivileged same-UID sibling can replace neither systemd's MainPID-owned
    # StatusText nor this exact tuple.
    for _attempt in range(3):
        before = _read_runtime_receipt(
            DEFAULT_WRITER_LIVENESS_RECEIPT_PATH,
            expected_uid=int(snapshot["gateway_uid"]),
        )
        observed_properties = _systemctl_show(DEFAULT_GATEWAY_UNIT)
        after = _read_runtime_receipt(
            DEFAULT_WRITER_LIVENESS_RECEIPT_PATH,
            expected_uid=int(snapshot["gateway_uid"]),
        )
        if before != after:
            continue
        candidate_digest = _sha256_json(after)
        candidate_generation = after.get("generation")
        try:
            expected_status = writer_liveness_status_text(
                gateway_readiness_sha256,
                candidate_generation,
                candidate_digest,
            )
            main_pid = int(observed_properties.get("MainPID") or "0")
        except (TypeError, ValueError):
            continue
        if observed_properties.get("StatusText") != expected_status:
            continue
        receipt = after
        properties = observed_properties
        break
    if receipt is None or properties is None:
        raise RuntimeError("writer liveness StatusText is stale or unauthenticated")
    observed = receipt.get("observed_at_boottime_ns")
    generation = receipt.get("generation")
    request_id = receipt.get("writer_request_id")
    try:
        parsed_request_id = uuid.UUID(str(request_id or ""))
    except ValueError as exc:
        raise RuntimeError("writer liveness request identity is invalid") from exc
    if (
        set(receipt) != _GATEWAY_LIVENESS_RECEIPT_KEYS
        or receipt.get("version") != WRITER_LIVENESS_RECEIPT_VERSION
        or receipt.get("boot_id_sha256") != current_boot_id_sha256
        or type(observed) is not int
        or observed < 0
        or current_boottime_ns < observed
        or current_boottime_ns - observed
        > WRITER_LIVENESS_MAX_AGE_SECONDS * 1_000_000_000
        or type(receipt.get("observed_at_unix")) is not int
        or receipt["observed_at_unix"] < 0
        or type(generation) is not int
        or not 1 <= generation <= MAX_SEQUENCE
        or receipt.get("gateway_pid")
        != gateway_process.get("systemd_main_pid")
        or receipt.get("gateway_start_time_ticks")
        != gateway_process.get("systemd_main_pid_start_time_ticks")
        or parsed_request_id.int == 0
        or str(parsed_request_id) != request_id
        or receipt.get("writer_service") != "canonical_writer"
        or receipt.get("writer_protocol") != "v1"
        or receipt.get("database_identity")
        != CANONICAL_WRITER_MIGRATION_OWNER
        or receipt.get("socket_path") != str(DEFAULT_SOCKET_PATH)
        or receipt.get("socket_device") != writer_socket.get("device")
        or receipt.get("socket_inode") != writer_socket.get("inode")
        or receipt.get("socket_owner_uid") != snapshot.get("writer_uid")
        or receipt.get("socket_group_gid")
        != writer_socket.get("expected_group_gid")
        or receipt.get("socket_mode") != "0660"
        or properties.get("Type") != "notify"
        or properties.get("NotifyAccess") != "main"
        or properties.get("ActiveState") != "active"
        or properties.get("SubState") != "running"
        or main_pid != gateway_process.get("systemd_main_pid")
        or _process_start_time_ticks(main_pid)
        != gateway_process.get("systemd_main_pid_start_time_ticks")
    ):
        raise RuntimeError("writer liveness PING receipt is stale or unbound")
    digest = _sha256_json(receipt)
    snapshot["runtime_liveness"] = copy.deepcopy(dict(receipt))
    return RuntimeLivenessBinding(sha256=digest, generation=generation)


def _validate_systemd_receipt(
    evidence: Mapping[str, Any],
    *,
    expected_unit: str,
    expected_version: str,
    expected_pid: int,
    expected_start_ticks: int,
    current_boot_id_sha256: str,
    current_boottime_ns: int,
    receipt_keys: frozenset[str],
    maximum_age_seconds: int | None = RECEIPT_TTL_SECONDS,
    authenticate_status: bool = True,
) -> tuple[Mapping[str, Any], str]:
    if type(authenticate_status) is not bool:
        raise TypeError("runtime readiness status policy is invalid")
    if set(evidence) != _SYSTEMD_READINESS_KEYS:
        raise RuntimeError("runtime readiness systemd evidence is not exact")
    receipt = _mapping(evidence.get("receipt"))
    digest = _sha256_json(receipt)
    observed_boottime = receipt.get("observed_at_boottime_ns")
    if (
        set(receipt) != receipt_keys
        or receipt.get("version") != expected_version
        or receipt.get("boot_id_sha256") != current_boot_id_sha256
        or type(observed_boottime) is not int
        or current_boottime_ns < observed_boottime
        or (
            maximum_age_seconds is not None
            and current_boottime_ns - observed_boottime
            > maximum_age_seconds * 1_000_000_000
        )
        or evidence.get("unit_name") != expected_unit
        or evidence.get("unit_type") != "notify"
        or evidence.get("notify_access") != "main"
        or evidence.get("active_state") != "active"
        or evidence.get("sub_state") != "running"
        or evidence.get("systemd_main_pid") != expected_pid
        or evidence.get("systemd_main_pid_start_time_ticks")
        != expected_start_ticks
        or evidence.get("receipt_sha256") != digest
        or (
            authenticate_status
            and evidence.get("status_text") != f"{expected_version}:{digest}"
        )
    ):
        raise RuntimeError("runtime readiness receipt is stale or unauthenticated")
    return receipt, digest


def _validate_runtime_readiness(
    snapshot: Mapping[str, Any],
    *,
    current_boot_id_sha256: str,
    current_boottime_ns: int,
) -> RuntimeReadinessBinding:
    from gateway.canonical_writer_bootstrap import (
        WRITER_RUNTIME_ATTESTATION_VERSION,
    )
    from gateway.canonical_writer_readiness import READINESS_RECEIPT_VERSION

    readiness = _mapping(snapshot.get("runtime_readiness"))
    if set(readiness) != {"gateway", "writer"}:
        raise RuntimeError("runtime readiness evidence is incomplete")
    gateway_process = _mapping(snapshot.get("gateway_process"))
    writer_process = _mapping(
        _mapping(_mapping(snapshot.get("writer_deployment")).get("attestation")).get(
            "process"
        )
    )
    gateway_pid = gateway_process.get("systemd_main_pid")
    gateway_start = gateway_process.get("systemd_main_pid_start_time_ticks")
    writer_pid = writer_process.get("systemd_main_pid")
    writer_start = writer_process.get("systemd_main_pid_start_time_ticks")
    gateway_receipt, gateway_digest = _validate_systemd_receipt(
        _mapping(readiness.get("gateway")),
        expected_unit=DEFAULT_GATEWAY_UNIT,
        expected_version=READINESS_RECEIPT_VERSION,
        expected_pid=gateway_pid,
        expected_start_ticks=gateway_start,
        current_boot_id_sha256=current_boot_id_sha256,
        current_boottime_ns=current_boottime_ns,
        receipt_keys=_GATEWAY_READINESS_RECEIPT_KEYS,
        maximum_age_seconds=None,
        authenticate_status=False,
    )
    writer_receipt, writer_digest = _validate_systemd_receipt(
        _mapping(readiness.get("writer")),
        expected_unit=DEFAULT_WRITER_UNIT,
        expected_version=WRITER_RUNTIME_ATTESTATION_VERSION,
        expected_pid=writer_pid,
        expected_start_ticks=writer_start,
        current_boot_id_sha256=current_boot_id_sha256,
        current_boottime_ns=current_boottime_ns,
        receipt_keys=_WRITER_RUNTIME_RECEIPT_KEYS,
        maximum_age_seconds=None,
    )
    writer_policy = _mapping(
        _mapping(snapshot.get("writer_deployment")).get("policy")
    )
    gateway_policy = _mapping(
        _mapping(snapshot.get("gateway_deployment")).get("policy")
    )
    artifact_root = writer_policy.get("artifact_root")
    module_origin = writer_policy.get("module_origin")
    database = _mapping(snapshot.get("database"))
    database_policy = _mapping(database.get("policy"))
    writer_socket = _mapping(snapshot.get("socket"))
    gateway_origin = gateway_receipt.get("gateway_module_origin")
    try:
        gateway_module_digest = (
            _sha256_trusted_file(gateway_origin)
            if isinstance(gateway_origin, str)
            else ""
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "runtime readiness gateway module identity is unavailable"
        ) from exc
    gateway_module_ok = (
        isinstance(artifact_root, str)
        and isinstance(gateway_origin, str)
        and Path(artifact_root) in Path(gateway_origin).parents
        and gateway_origin == gateway_policy.get("module_origin")
        and gateway_origin.endswith(
            "/gateway/canonical_writer_gateway_bootstrap.py"
        )
        and gateway_receipt.get("gateway_module_sha256")
        == gateway_module_digest
    )
    try:
        request_id = uuid.UUID(str(gateway_receipt.get("writer_request_id") or ""))
    except ValueError:
        request_id = uuid.UUID(int=0)
    writer_bootstrap_origin = writer_receipt.get("bootstrap_module_origin")
    writer_service_origin = writer_receipt.get("service_module_origin")
    try:
        writer_bootstrap_digest = (
            _sha256_trusted_file(writer_bootstrap_origin)
            if isinstance(writer_bootstrap_origin, str)
            else ""
        )
        writer_service_digest = (
            _sha256_trusted_file(writer_service_origin)
            if isinstance(writer_service_origin, str)
            else ""
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "runtime readiness writer module identity is unavailable"
        ) from exc
    writer_modules_ok = (
        writer_bootstrap_origin == module_origin
        and isinstance(writer_service_origin, str)
        and isinstance(module_origin, str)
        and Path(writer_service_origin).parent == Path(module_origin).parent
        and writer_service_origin.endswith("/gateway/canonical_writer_service.py")
        and writer_receipt.get("bootstrap_module_sha256")
        == writer_bootstrap_digest
        and writer_receipt.get("service_module_sha256") == writer_service_digest
    )
    for label, runtime_receipt in (
        ("gateway", gateway_receipt),
        ("writer", writer_receipt),
    ):
        for name in ("effective_import_paths", "loaded_module_origins"):
            values = runtime_receipt.get(name)
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) for value in values)
                or values != sorted(set(values))
            ):
                raise RuntimeError(
                    f"runtime readiness {label} Python closure is invalid"
                )
        for name in ("unexpected_import_paths", "unexpected_import_origins"):
            if runtime_receipt.get(name) != []:
                raise RuntimeError(
                    f"runtime readiness {label} has an unexpected import surface"
                )
        environment_names = runtime_receipt.get(
            "effective_environment_variable_names"
        )
        environment_hashes = runtime_receipt.get(
            "effective_environment_variable_value_sha256"
        )
        if (
            not isinstance(environment_names, list)
            or environment_names != sorted(set(environment_names))
            or any(
                not isinstance(name, str)
                or name not in _WRITER_ONLY_ALLOWED_ENVIRONMENT_NAMES
                for name in environment_names
            )
            or not isinstance(environment_hashes, Mapping)
            or list(environment_hashes) != environment_names
            or any(
                not isinstance(value, str)
                or _SHA256_RE.fullmatch(value) is None
                for value in environment_hashes.values()
            )
        ):
            raise RuntimeError(
                f"runtime readiness {label} environment is not credential-free"
            )
        if runtime_receipt.get("loaded_module_origins_complete") is not True:
            raise RuntimeError(
                f"runtime readiness {label} Python closure is incomplete"
            )
    if (
        gateway_receipt.get("gateway_pid") != gateway_pid
        or gateway_receipt.get("gateway_start_time_ticks") != gateway_start
        or gateway_receipt.get("writer_service") != "canonical_writer"
        or gateway_receipt.get("writer_protocol") != "v1"
        or gateway_receipt.get("database_identity")
        != CANONICAL_WRITER_MIGRATION_OWNER
        or request_id.int == 0
        or str(request_id) != gateway_receipt.get("writer_request_id")
        or not gateway_module_ok
        or gateway_receipt.get("gateway_dumpable") is not False
        or gateway_receipt.get("gateway_core_soft_limit") != 0
        or gateway_receipt.get("gateway_core_hard_limit") != 0
        or writer_receipt.get("writer_pid") != writer_pid
        or writer_receipt.get("writer_start_time_ticks") != writer_start
        or not writer_modules_ok
        or writer_receipt.get("writer_dumpable") is not False
        or writer_receipt.get("writer_core_soft_limit") != 0
        or writer_receipt.get("writer_core_hard_limit") != 0
        or writer_receipt.get("statement_catalog_sha256")
        != PRODUCTION_CATALOG_SHA256
        or writer_receipt.get("database_identity")
        != CANONICAL_WRITER_MIGRATION_OWNER
        or writer_receipt.get("database_role") != database.get("expected_user")
        or writer_receipt.get("private_schema_identity_sha256")
        != database_policy.get("private_schema_identity_sha256")
        or writer_receipt.get("managed_hba_baseline_sha256")
        != database_policy.get("managed_cloudsqladmin_hba_rejection_sha256")
        or writer_receipt.get("discord_edge_authority_enabled") is not False
        or writer_receipt.get("socket_path") != str(DEFAULT_SOCKET_PATH)
        or writer_receipt.get("socket_owner_uid") != snapshot.get("writer_uid")
        or writer_receipt.get("socket_group_gid")
        != writer_socket.get("expected_group_gid")
        or writer_receipt.get("socket_mode") != "0660"
        or type(writer_receipt.get("socket_inode")) is not int
        or writer_receipt["socket_inode"] <= 0
        or type(writer_receipt.get("socket_device")) is not int
        or writer_receipt["socket_device"] < 0
    ):
        raise RuntimeError("runtime readiness identity binding is invalid")
    return RuntimeReadinessBinding(
        gateway_sha256=gateway_digest,
        writer_sha256=writer_digest,
    )


def _path_within(path: str, root: str) -> bool:
    candidate = Path(path)
    boundary = Path(root)
    return candidate == boundary or boundary in candidate.parents


def _preapproved_external_native_mappings(
    policy: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Validate the exact owner-approved native mapping policy."""

    root_value = policy.get("artifact_root")
    raw = policy.get("preapproved_external_native_executable_mappings")
    if not isinstance(root_value, str):
        raise RuntimeError("native mapping policy lacks an artifact root")
    root = str(_absolute_normalized_path(root_value))
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(item, Mapping) for item in raw)
    ):
        raise RuntimeError("external native mapping policy is absent")
    result: list[dict[str, str]] = []
    for item in raw:
        if set(item) != {"path", "sha256"}:
            raise RuntimeError("external native mapping policy is not exact")
        path = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(path, str)
            or _path_within(path, root)
            or str(_absolute_normalized_path(path)) != path
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise RuntimeError("external native mapping policy is invalid")
        result.append({"path": path, "sha256": digest})
    paths = [item["path"] for item in result]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RuntimeError("external native mapping policy is not canonical")
    return result


def _validate_runtime_code_closure(
    snapshot: Mapping[str, Any],
) -> RuntimeCodeClosureBinding:
    """Validate only live in-process/procfs code identity, never template claims."""

    digests: dict[str, str] = {}
    for deployment_name in ("writer_deployment", "gateway_deployment"):
        deployment = _mapping(snapshot.get(deployment_name))
        policy = _mapping(deployment.get("policy"))
        process = _mapping(_mapping(deployment.get("attestation")).get("process"))
        unit = _mapping(_mapping(deployment.get("attestation")).get("unit"))
        artifact_root = policy.get("artifact_root")
        if not isinstance(artifact_root, str):
            raise RuntimeError("runtime code closure lacks an artifact root")
        root = str(_absolute_normalized_path(artifact_root))
        raw_import_policy = policy.get("import_paths")
        if not isinstance(raw_import_policy, Sequence) or isinstance(
            raw_import_policy, (str, bytes, bytearray)
        ):
            raise RuntimeError("runtime code closure policy is invalid")
        protected_paths: list[str] = []
        for raw in raw_import_policy:
            item = _mapping(raw)
            path = item.get("path")
            digest = item.get("digest_sha256")
            object_type = item.get("object_type")
            if (
                not isinstance(path, str)
                or not _path_within(path, root)
                or not isinstance(digest, str)
                or _SHA256_RE.fullmatch(digest) is None
                or object_type not in {"directory", "regular_file"}
            ):
                raise RuntimeError("runtime code closure policy escapes release")
            live = os.lstat(path)
            if (
                stat.S_ISLNK(live.st_mode)
                or live.st_uid != 0
                or live.st_gid != 0
                or stat.S_IMODE(live.st_mode) & 0o022
                or (
                    object_type == "directory"
                    and not stat.S_ISDIR(live.st_mode)
                )
                or (
                    object_type == "regular_file"
                    and not stat.S_ISREG(live.st_mode)
                )
            ):
                raise RuntimeError("runtime code closure policy path drifted")
            protected_paths.append(path)
        if root not in protected_paths:
            raise RuntimeError("runtime code closure does not cover release root")
        approved_native_mappings = _preapproved_external_native_mappings(policy)
        effective_paths = process.get("effective_import_paths")
        loaded_origins = process.get("loaded_module_origins")
        mapped_paths = process.get("mapped_executable_paths")
        environment_names = process.get("environment_variable_names")
        environment_hashes = process.get("environment_variable_value_sha256")
        environment_digest = process.get("environment_identity_sha256")
        if any(
            not isinstance(values, list)
            or not values
            or values != sorted(set(values))
            or any(not isinstance(value, str) for value in values)
            for values in (effective_paths, loaded_origins, mapped_paths)
        ):
            raise RuntimeError("runtime code closure live lists are invalid")
        if (
            not isinstance(environment_names, list)
            or environment_names != sorted(set(environment_names))
            or not isinstance(environment_hashes, Mapping)
            or list(environment_hashes) != environment_names
            or any(
                not isinstance(value, str)
                or _SHA256_RE.fullmatch(value) is None
                for value in environment_hashes.values()
            )
            or not isinstance(environment_digest, str)
            or _SHA256_RE.fullmatch(environment_digest) is None
        ):
            raise RuntimeError("runtime environment closure is invalid")
        if (
            process.get("loaded_module_origins_complete") is not True
            or process.get("mapped_executable_paths_complete") is not True
            or process.get("unexpected_import_origins") != []
            or process.get("deleted_code_mappings") != []
            or process.get("writable_code_mappings") != []
            or unit.get("code_injection_environment_variable_names") != []
            or unit.get("environment_files") != []
        ):
            raise RuntimeError("runtime code closure contains an injection surface")
        if any(not _path_within(path, root) for path in effective_paths):
            raise RuntimeError("runtime sys.path escapes immutable release")
        for origin in loaded_origins:
            if not _path_within(origin, root):
                raise RuntimeError("runtime Python module escapes immutable release")
            try:
                origin_stat = os.lstat(origin)
            except OSError as exc:
                raise RuntimeError("runtime Python module origin is unavailable") from exc
            if stat.S_ISLNK(origin_stat.st_mode) or not stat.S_ISREG(
                origin_stat.st_mode
            ):
                raise RuntimeError("runtime Python module origin is invalid")
        native_mappings: list[dict[str, str]] = []
        for path in mapped_paths:
            if _path_within(path, root):
                continue
            native_mappings.append(
                {"path": path, "sha256": _sha256_trusted_file(path)}
            )
        if native_mappings != approved_native_mappings:
            raise RuntimeError(
                "live external native mappings differ from approved policy"
            )
        if deployment_name == "gateway_deployment" and (
            policy.get("dynamic_python_loading_mode") != "disabled"
            or policy.get("dynamic_python_discovery_paths") != []
            or process.get("dynamic_python_loading_mode") != "disabled"
            or process.get("dynamic_python_discovery_paths") != []
            or process.get("dynamic_python_loaded_origins") != []
            or process.get("dynamic_python_writable_paths") != []
        ):
            raise RuntimeError("writer-only gateway dynamic Python loading is enabled")
        closure = {
            "deployment": deployment_name,
            "artifact_root": root,
            "effective_import_paths": effective_paths,
            "loaded_module_origins": loaded_origins,
            "mapped_executable_paths": mapped_paths,
            "native_mappings": native_mappings,
            "environment_variable_names": environment_names,
            "environment_variable_value_sha256": dict(environment_hashes),
            "environment_identity_sha256": environment_digest,
        }
        digests[deployment_name] = _sha256_json(closure)
    return RuntimeCodeClosureBinding(
        gateway_sha256=digests["gateway_deployment"],
        writer_sha256=digests["writer_deployment"],
    )


def _pids_for_uid(uid: int) -> list[int]:
    result: list[int] = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            status = (item / "status").read_text(encoding="ascii")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for line in status.splitlines():
            if line.startswith("Uid:"):
                values = line.partition(":")[2].split()
                if len(values) >= 2 and values[1].isdigit() and int(values[1]) == uid:
                    result.append(int(item.name))
                break
    return sorted(result)


def _require_exclusive_service_uids(
    *,
    gateway_uid: int,
    gateway_pid: int,
    writer_uid: int,
    writer_pid: int,
) -> None:
    if (
        _pids_for_uid(gateway_uid) != [gateway_pid]
        or _pids_for_uid(writer_uid) != [writer_pid]
    ):
        raise RuntimeError("writer-only service UID surface is not exclusive")


def _pids_for_exact_python_module(module: str) -> list[int]:
    """Inventory processes whose exact argv activates one pinned module."""

    if module != "scripts.discord_edge_bootstrap":
        raise ValueError("collector process module is not pinned")
    result: list[int] = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            raw = (item / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        try:
            argv = [
                value.decode("utf-8", errors="strict")
                for value in raw.rstrip(b"\x00").split(b"\x00")
                if value
            ]
        except UnicodeError as exc:
            raise RuntimeError("collector process argv is invalid") from exc
        activated = any(
            argv[index] == "-m" and argv[index + 1] == module
            for index in range(max(0, len(argv) - 1))
        )
        direct = any(
            value == "/scripts/discord_edge_bootstrap.py"
            or value.endswith("/scripts/discord_edge_bootstrap.py")
            for value in argv[1:]
        )
        if activated or direct:
            result.append(int(item.name))
    return sorted(result)


def _discord_edge_process_pids() -> list[int]:
    """Prove edge-process absence even when its legacy passwd entry is gone."""

    try:
        edge_uid = pwd.getpwnam(DEFAULT_DISCORD_EDGE_USER).pw_uid
    except KeyError:
        edge_uid = None
    return sorted(
        set(_pids_for_uid(edge_uid) if edge_uid is not None else ())
        | set(_pids_for_exact_python_module("scripts.discord_edge_bootstrap"))
    )


def _child_pids(root_pid: int) -> list[int]:
    parents: dict[int, int] = {}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            raw = (item / "stat").read_text(encoding="ascii")
            suffix = raw.rsplit(")", 1)
            parent = int(suffix[1].strip().split()[1])
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            continue
        parents[int(item.name)] = parent
    descendants: set[int] = set()
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if pid not in descendants and (
                parent == root_pid or parent in descendants
            ):
                descendants.add(pid)
                changed = True
    return sorted(descendants)


def _unit_readiness_evidence(
    unit_name: str,
    properties: Mapping[str, str],
    receipt: Mapping[str, Any],
    *,
    start_time_ticks: int,
) -> dict[str, Any]:
    try:
        main_pid = int(properties.get("MainPID") or "0")
    except ValueError as exc:
        raise RuntimeError("systemd MainPID evidence is invalid") from exc
    digest = _sha256_json(receipt)
    return {
        "unit_name": unit_name,
        "unit_type": properties.get("Type"),
        "notify_access": properties.get("NotifyAccess"),
        "active_state": properties.get("ActiveState"),
        "sub_state": properties.get("SubState"),
        "systemd_main_pid": main_pid,
        "systemd_main_pid_start_time_ticks": start_time_ticks,
        "status_text": properties.get("StatusText"),
        "receipt_sha256": digest,
        "receipt": copy.deepcopy(dict(receipt)),
    }


def _fence_live_activation(
    *,
    gateway_uid: int,
    gateway_pid: int,
    gateway_start_ticks: int,
    gateway_readiness_sha256: str,
    gateway_liveness_sha256: str,
    minimum_gateway_liveness_generation: int,
    writer_uid: int,
    writer_gid: int,
    writer_pid: int,
    writer_start_ticks: int,
    writer_readiness_sha256: str,
    socket_gid: int,
    socket_device: int,
    socket_inode: int,
    runtime_directory_device: int,
    runtime_directory_inode: int,
    ip_address_allow_network: str,
    cgroup_device: int,
    cgroup_inode: int,
    cgroup_main_pid: int,
    ingress_direct_program_ids: Sequence[int],
    ingress_effective_program_ids: Sequence[int],
    egress_direct_program_ids: Sequence[int],
    egress_effective_program_ids: Sequence[int],
) -> None:
    from gateway.canonical_writer_bootstrap import (
        DEFAULT_WRITER_RUNTIME_ATTESTATION_PATH,
        WRITER_RUNTIME_ATTESTATION_VERSION,
    )
    from gateway.canonical_writer_readiness import (
        DEFAULT_READINESS_RECEIPT_PATH,
    )

    _require_exclusive_service_uids(
        gateway_uid=gateway_uid,
        gateway_pid=gateway_pid,
        writer_uid=writer_uid,
        writer_pid=writer_pid,
    )

    gateway = _systemctl_show(DEFAULT_GATEWAY_UNIT)
    writer = _systemctl_show(DEFAULT_WRITER_UNIT)
    _parse_universal_ip_deny(writer.get("IPAddressDeny", ""))
    current_allow = _parse_systemd_ip_networks(
        writer.get("IPAddressAllow", "")
    )
    if current_allow != (ip_address_allow_network,):
        raise RuntimeError("activation fence writer network policy changed")
    runtime_directory = _collect_writer_runtime_directory(
        writer_uid=writer_uid,
        socket_gid=socket_gid,
    )
    if (
        runtime_directory.device != runtime_directory_device
        or runtime_directory.inode != runtime_directory_inode
    ):
        raise RuntimeError("activation fence writer runtime directory changed")
    if cgroup_main_pid != writer_pid:
        raise RuntimeError("activation fence cgroup MainPID binding changed")
    _require_exact_bpf_binding(
        _collect_writer_cgroup_bpf_binding(writer_pid=writer_pid),
        cgroup_device=cgroup_device,
        cgroup_inode=cgroup_inode,
        expected_main_pid=cgroup_main_pid,
        ingress_direct_program_ids=ingress_direct_program_ids,
        ingress_effective_program_ids=ingress_effective_program_ids,
        egress_direct_program_ids=egress_direct_program_ids,
        egress_effective_program_ids=egress_effective_program_ids,
    )
    expected = (
        (
            writer,
            writer_pid,
            writer_start_ticks,
            WRITER_RUNTIME_ATTESTATION_VERSION,
            writer_readiness_sha256,
        ),
    )
    try:
        gateway_main_pid = int(gateway.get("MainPID") or "0")
    except ValueError as exc:
        raise RuntimeError("activation fence MainPID is invalid") from exc
    if (
        gateway.get("Type") != "notify"
        or gateway.get("NotifyAccess") != "main"
        or gateway.get("ActiveState") != "active"
        or gateway.get("SubState") != "running"
        or gateway_main_pid != gateway_pid
        or _process_start_time_ticks(gateway_main_pid) != gateway_start_ticks
    ):
        raise RuntimeError("activation fence gateway identity changed")
    for properties, pid, start_ticks, version, digest in expected:
        try:
            main_pid = int(properties.get("MainPID") or "0")
        except ValueError as exc:
            raise RuntimeError("activation fence MainPID is invalid") from exc
        if (
            properties.get("Type") != "notify"
            or properties.get("NotifyAccess") != "main"
            or properties.get("ActiveState") != "active"
            or properties.get("SubState") != "running"
            or main_pid != pid
            or _process_start_time_ticks(main_pid) != start_ticks
            or properties.get("StatusText") != f"{version}:{digest}"
        ):
            raise RuntimeError("activation fence service identity changed")
    gateway_receipt = _read_runtime_receipt(
        DEFAULT_READINESS_RECEIPT_PATH,
        expected_uid=gateway_uid,
    )
    writer_receipt = _read_runtime_receipt(
        DEFAULT_WRITER_RUNTIME_ATTESTATION_PATH,
        expected_uid=writer_uid,
    )
    if (
        _sha256_json(gateway_receipt) != gateway_readiness_sha256
        or _sha256_json(writer_receipt) != writer_readiness_sha256
    ):
        raise RuntimeError("activation fence readiness receipt changed")
    _connect_and_validate_writer_socket_peer(
        writer_pid=writer_pid,
        writer_uid=writer_uid,
        writer_gid=writer_gid,
        socket_gid=socket_gid,
        expected_device=socket_device,
        expected_inode=socket_inode,
    )
    live_liveness_snapshot: dict[str, Any] = {
        "gateway_uid": gateway_uid,
        "writer_uid": writer_uid,
        "gateway_process": {
            "systemd_main_pid": gateway_pid,
            "systemd_main_pid_start_time_ticks": gateway_start_ticks,
        },
        "socket": {
            "device": socket_device,
            "inode": socket_inode,
            "expected_group_gid": socket_gid,
        },
    }
    current_liveness = _collect_runtime_liveness(
        live_liveness_snapshot,
        gateway_readiness_sha256=gateway_readiness_sha256,
        current_boot_id_sha256=_boot_id_sha256(),
        current_boottime_ns=_current_boottime_ns(),
    )
    if (
        current_liveness.generation < minimum_gateway_liveness_generation
        or (
            current_liveness.generation == minimum_gateway_liveness_generation
            and current_liveness.sha256 != gateway_liveness_sha256
        )
    ):
        raise RuntimeError("activation fence writer liveness generation changed")
    _connect_and_validate_writer_socket_peer(
        writer_pid=writer_pid,
        writer_uid=writer_uid,
        writer_gid=writer_gid,
        socket_gid=socket_gid,
        expected_device=socket_device,
        expected_inode=socket_inode,
    )
    final_runtime_directory = _collect_writer_runtime_directory(
        writer_uid=writer_uid,
        socket_gid=socket_gid,
    )
    if (
        _process_start_time_ticks(writer_pid) != writer_start_ticks
        or final_runtime_directory.device != runtime_directory_device
        or final_runtime_directory.inode != runtime_directory_inode
    ):
        raise RuntimeError("activation fence writer socket changed after PING")
    final_writer = _systemctl_show(DEFAULT_WRITER_UNIT)
    _parse_universal_ip_deny(final_writer.get("IPAddressDeny", ""))
    if _parse_systemd_ip_networks(
        final_writer.get("IPAddressAllow", "")
    ) != (ip_address_allow_network,):
        raise RuntimeError("activation fence writer network policy rotated")
    _require_exact_bpf_binding(
        _collect_writer_cgroup_bpf_binding(writer_pid=writer_pid),
        cgroup_device=cgroup_device,
        cgroup_inode=cgroup_inode,
        expected_main_pid=cgroup_main_pid,
        ingress_direct_program_ids=ingress_direct_program_ids,
        ingress_effective_program_ids=ingress_effective_program_ids,
        egress_direct_program_ids=egress_direct_program_ids,
        egress_effective_program_ids=egress_effective_program_ids,
    )
    _require_exclusive_service_uids(
        gateway_uid=gateway_uid,
        gateway_pid=gateway_pid,
        writer_uid=writer_uid,
        writer_pid=writer_pid,
    )


def _verify_host_contract(
    manifest: TrustedDeploymentManifest,
    *,
    writer_config_path: str,
) -> None:
    contract = manifest.host_contract
    for path_name, digest_name in (
        ("gateway_unit_fragment_path", "gateway_unit_fragment_sha256"),
        ("writer_unit_fragment_path", "writer_unit_fragment_sha256"),
        ("gateway_config_path", "gateway_config_sha256"),
    ):
        if _sha256_trusted_file(str(contract[path_name])) != contract[digest_name]:
            raise RuntimeError(f"host contract {path_name} digest changed")
    if _sha256_trusted_file(writer_config_path) != contract["writer_config_sha256"]:
        raise RuntimeError("host contract writer config digest changed")


def _collect_live_snapshot(
    manifest: TrustedDeploymentManifest,
    *,
    now_unix: int,
) -> dict[str, Any]:
    """Collect fixed local evidence; no caller-supplied collector is allowed."""

    snapshot = copy.deepcopy(dict(manifest.snapshot_template))
    _bind_snapshot_to_manifest(snapshot, manifest)
    writer_policy = _mapping(
        _mapping(snapshot.get("writer_deployment")).get("policy")
    )
    gateway_policy = _mapping(
        _mapping(snapshot.get("gateway_deployment")).get("policy")
    )
    writer_config_path = str(writer_policy.get("config_path") or "")
    _verify_host_contract(manifest, writer_config_path=writer_config_path)
    _verify_release_artifact(snapshot, manifest)

    writer_systemd = _systemctl_show(DEFAULT_WRITER_UNIT)
    gateway_systemd = _systemctl_show(DEFAULT_GATEWAY_UNIT)
    writer_ip_deny = _parse_universal_ip_deny(
        writer_systemd.get("IPAddressDeny", "")
    )
    writer_ip_allow = _parse_systemd_ip_networks(
        writer_systemd.get("IPAddressAllow", "")
    )
    if len(writer_ip_allow) != 1:
        raise RuntimeError("writer systemd IPAddressAllow is not exact")
    for properties, expected_path in (
        (writer_systemd, manifest.host_contract["writer_unit_fragment_path"]),
        (gateway_systemd, manifest.host_contract["gateway_unit_fragment_path"]),
    ):
        if (
            properties.get("LoadState") != "loaded"
            or properties.get("FragmentPath") != expected_path
            or properties.get("DropInPaths") not in {"", "[]"}
            or properties.get("NeedDaemonReload") not in {"no", "false"}
        ):
            raise RuntimeError("systemd unit provenance is not exact")
    if (
        writer_systemd.get("LimitCORE") != "0"
        or gateway_systemd.get("LimitCORE") != "0"
        or writer_systemd.get("PrivateNetwork") not in {"no", "false"}
        or gateway_systemd.get("PrivateNetwork") not in {"yes", "true"}
        or set(writer_systemd.get("RestrictAddressFamilies", "").split())
        != {"AF_UNIX", "AF_INET", "AF_INET6"}
        or set(gateway_systemd.get("RestrictAddressFamilies", "").split())
        != {"AF_UNIX"}
        or any(
            properties.get(name)
            for properties in (writer_systemd, gateway_systemd)
            for name in (
                "EnvironmentFiles",
                "PassEnvironment",
                "LoadCredential",
                "RootDirectory",
                "RootImage",
            )
        )
    ):
        raise RuntimeError("writer-only systemd isolation is not exact")
    try:
        writer_pid = int(writer_systemd["MainPID"])
        gateway_pid = int(gateway_systemd["MainPID"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("systemd MainPID is invalid") from exc
    writer_bpf_binding = _collect_writer_cgroup_bpf_binding(
        writer_pid=writer_pid
    )
    snapshot["writer_kernel_network_enforcement"] = {
        "ip_address_deny": list(writer_ip_deny),
        "ip_address_allow": list(writer_ip_allow),
        **writer_bpf_binding.as_dict(),
    }
    writer_identity = _process_identity(writer_pid)
    gateway_identity = _process_identity(gateway_pid)
    if (
        writer_identity["cmdline"] != list(writer_policy.get("exec_start") or ())
        or writer_identity["executable"] != writer_policy.get("interpreter")
        or gateway_identity["cmdline"] != list(gateway_policy.get("exec_start") or ())
        or gateway_identity["executable"] != gateway_policy.get("interpreter")
    ):
        raise RuntimeError("live service argv does not match approved policy")
    for identity, policy in (
        (writer_identity, writer_policy),
        (gateway_identity, gateway_policy),
    ):
        interpreter_digest = next(
            (
                item.get("digest_sha256")
                for item in policy.get("import_paths", ())
                if isinstance(item, Mapping)
                and item.get("path") == policy.get("interpreter")
            ),
            None,
        )
        if (
            not isinstance(interpreter_digest, str)
            or _sha256_trusted_file(identity["executable"])
            != interpreter_digest
        ):
            raise RuntimeError("live service interpreter digest changed")

    snapshot["collected_at_unix"] = now_unix
    gateway_process = snapshot.get("gateway_process")
    if not isinstance(gateway_process, dict):
        raise ValueError("gateway process snapshot is not mutable")
    gateway_process.update(
        {
            "complete": True,
            "platform": "linux",
            "pid": gateway_pid,
            "systemd_main_pid": gateway_pid,
            "process_start_time_ticks": gateway_identity["start_time_ticks"],
            "systemd_main_pid_start_time_ticks": gateway_identity[
                "start_time_ticks"
            ],
            "observed_at_unix": now_unix,
            "core_soft_limit": gateway_identity["core_soft_limit"],
            "core_hard_limit": gateway_identity["core_hard_limit"],
        }
    )
    for deployment_name, identity, unit_name in (
        ("writer_deployment", writer_identity, DEFAULT_WRITER_UNIT),
        ("gateway_deployment", gateway_identity, DEFAULT_GATEWAY_UNIT),
    ):
        process = _mapping(
            _mapping(_mapping(snapshot[deployment_name]).get("attestation")).get(
                "process"
            )
        )
        if not isinstance(process, dict):
            raise ValueError(f"{deployment_name} process snapshot is not mutable")
        process.update(
            {
                "complete": True,
                "observed_at_unix": now_unix,
                "pid": identity["pid"],
                "systemd_main_pid": identity["pid"],
                "process_start_time_ticks": identity["start_time_ticks"],
                "systemd_main_pid_start_time_ticks": identity["start_time_ticks"],
                "unit_name": unit_name,
                "cmdline": identity["cmdline"],
                "executable_path": identity["executable"],
            }
        )
    for deployment_name, properties, config_path in (
        ("writer_deployment", writer_systemd, writer_config_path),
        (
            "gateway_deployment",
            gateway_systemd,
            str(DEFAULT_GATEWAY_MANAGED_CONFIG_PATH),
        ),
    ):
        deployment = _mapping(snapshot.get(deployment_name))
        policy = _mapping(deployment.get("policy"))
        attestation = _mapping(deployment.get("attestation"))
        mounts = attestation.get("mounts")
        if not isinstance(mounts, dict):
            raise ValueError(f"{deployment_name} mount snapshot is not mutable")
        observed_read_write = properties["ReadWritePaths"].split()
        observed_bind = properties["BindPaths"].split()
        observed_bind_read_only = _parse_bind_read_only_paths(
            properties["BindReadOnlyPaths"]
        )
        observed_read_only = properties["ReadOnlyPaths"].split()
        if (
            observed_read_write != list(policy.get("read_write_paths") or ())
            or observed_bind != list(policy.get("bind_paths") or ())
            or observed_bind_read_only
            != list(policy.get("bind_read_only_paths") or ())
            or observed_read_only != [config_path]
        ):
            raise RuntimeError(
                f"{deployment_name} live filesystem boundary drifted"
            )
        mounts.update(
            {
                "complete": True,
                "read_write_paths": observed_read_write,
                "bind_paths": observed_bind,
                "bind_read_only_paths": observed_bind_read_only,
            }
        )
    snapshot["systemd_properties"] = {
        name: writer_systemd[name]
        for name in _SYSTEMD_PROPERTIES
        if name in {
            *_HARDENED_TRUE_PROPERTIES,
            "ProtectSystem",
            "ProtectHome",
            "ProtectProc",
            "ProcSubset",
            "UMask",
            "CapabilityBoundingSet",
            "AmbientCapabilities",
            "LimitCORE",
            "RestrictAddressFamilies",
        }
    }
    snapshot["gateway_systemd_properties"] = {
        name: gateway_systemd[name] for name in snapshot["systemd_properties"]
    }

    authority = snapshot.get("writer_authority_surface")
    if not isinstance(authority, dict):
        raise ValueError("writer authority snapshot is not mutable")
    gateway_children = _child_pids(gateway_pid)
    if gateway_children:
        raise RuntimeError("writer-only gateway has unexpected child processes")
    authority["complete"] = True
    authority["collected_by_uid"] = 0
    authority["observed_at_unix"] = now_unix
    identities = authority.get("identities")
    if not isinstance(identities, dict):
        raise ValueError("writer authority identities are not mutable")
    identities["gateway"] = {
        "pid": gateway_pid,
        "effective_uid": gateway_identity["effective_uid"],
        "effective_gid": gateway_identity["effective_gid"],
        "supplementary_gids": gateway_identity["supplementary_gids"],
    }
    identities["gateway_children"] = {"complete": True, "processes": []}
    identities["writer"] = {
        "pid": writer_pid,
        "effective_uid": writer_identity["effective_uid"],
        "effective_gid": writer_identity["effective_gid"],
        "supplementary_gids": writer_identity["supplementary_gids"],
    }
    if (
        gateway_identity["effective_gid"] != snapshot.get("gateway_gid")
        or writer_identity["effective_gid"] != snapshot.get("writer_gid")
        or gateway_identity["supplementary_gids"]
        != snapshot.get("gateway_supplementary_gids")
        or writer_identity["supplementary_gids"]
        != snapshot.get("writer_supplementary_gids")
    ):
        raise RuntimeError("writer-only process groups drifted")
    _require_exclusive_service_uids(
        gateway_uid=int(snapshot["gateway_uid"]),
        gateway_pid=gateway_pid,
        writer_uid=int(snapshot["writer_uid"]),
        writer_pid=writer_pid,
    )
    inventory = authority.get("privileged_execution_inventory")
    if not isinstance(inventory, dict):
        raise ValueError("writer process inventory is not mutable")
    inventory["writer_uid_process_executables"] = [writer_identity["executable"]]
    inventory["writer_uid_unattributed_processes"] = []
    inventory["gateway_uid_process_executables"] = [
        gateway_identity["executable"]
    ]
    inventory["gateway_uid_unattributed_processes"] = []

    writer_socket = snapshot.get("socket")
    if not isinstance(writer_socket, dict):
        raise RuntimeError("live writer socket snapshot is invalid")
    runtime_directory = _collect_writer_runtime_directory(
        writer_uid=int(snapshot["writer_uid"]),
        socket_gid=int(writer_socket["expected_group_gid"]),
    )
    socket_stat = _connect_and_validate_writer_socket_peer(
        writer_pid=writer_pid,
        writer_uid=int(snapshot["writer_uid"]),
        writer_gid=int(snapshot["writer_gid"]),
        socket_gid=int(writer_socket["expected_group_gid"]),
    )
    writer_socket.update(
        {
            "owner_uid": socket_stat.st_uid,
            "group_gid": socket_stat.st_gid,
            "mode": f"{stat.S_IMODE(socket_stat.st_mode):04o}",
            "device": socket_stat.st_dev,
            "inode": socket_stat.st_ino,
            "runtime_directory_device": runtime_directory.device,
            "runtime_directory_inode": runtime_directory.inode,
        }
    )
    export_path = _absolute_normalized_path(
        str(manifest.host_contract["projection_export_path"])
    )
    export_stat = os.lstat(export_path)
    projection = snapshot.get("projection_export")
    if not isinstance(projection, dict) or not stat.S_ISREG(export_stat.st_mode):
        raise RuntimeError("projection export identity is invalid")
    projection.update(
        {
            "owner_uid": export_stat.st_uid,
            "group_gid": export_stat.st_gid,
            "mode": f"{stat.S_IMODE(export_stat.st_mode):04o}",
            "gateway_access": _mode_access_for_identity(
                export_path,
                uid=gateway_identity["effective_uid"],
                gids=(
                    gateway_identity["effective_gid"],
                    *gateway_identity["supplementary_gids"],
                ),
            ),
            "projector_access": _mode_access_for_identity(
                export_path,
                uid=-1,
                gids=(int(snapshot["projector_gid"]),),
            ),
        }
    )

    from gateway.canonical_writer_bootstrap import (
        DEFAULT_WRITER_RUNTIME_ATTESTATION_PATH,
        load_service_config,
    )
    from gateway.canonical_writer_readiness import DEFAULT_READINESS_RECEIPT_PATH

    writer_config = load_service_config(writer_config_path)
    if writer_config.discord_edge_authority.enabled:
        raise RuntimeError("writer-only config enables Discord authority")
    try:
        database_address = ipaddress.ip_address(writer_config.database.host)
    except ValueError as exc:
        raise RuntimeError("writer-only database host must be an exact IP") from exc
    if database_address.version != 4:
        raise RuntimeError("writer-only database host must be an IPv4 address")
    expected_database_network = f"{database_address}/32"
    if writer_ip_allow != (expected_database_network,):
        raise RuntimeError("writer network allow-list does not match database host")
    credential_source = writer_config.database.credential
    credential_path = credential_source.path
    if credential_path is None:
        raise RuntimeError("writer credential is not a pinned file")
    try:
        credential_stat = os.lstat(credential_path)
    except OSError as exc:
        raise RuntimeError("writer credential identity is unavailable") from exc
    if (
        stat.S_ISLNK(credential_stat.st_mode)
        or not stat.S_ISREG(credential_stat.st_mode)
        or credential_stat.st_nlink != 1
        or credential_stat.st_uid != credential_source.expected_uid
        or (
            credential_source.expected_gid is not None
            and credential_stat.st_gid != credential_source.expected_gid
        )
        or stat.S_IMODE(credential_stat.st_mode)
        not in credential_source.allowed_modes
    ):
        raise RuntimeError("writer credential ownership or mode is invalid")
    credential = snapshot.get("credential")
    if not isinstance(credential, dict):
        raise ValueError("writer credential snapshot is not mutable")
    credential.update(
        {
            "owner_uid": credential_stat.st_uid,
            "mode": f"{stat.S_IMODE(credential_stat.st_mode):04o}",
            "gateway_access": _mode_access_for_identity(
                credential_path,
                uid=gateway_identity["effective_uid"],
                gids=(
                    gateway_identity["effective_gid"],
                    *gateway_identity["supplementary_gids"],
                ),
            ),
        }
    )
    try:
        gateway_passwd = pwd.getpwuid(gateway_identity["effective_uid"])
    except KeyError as exc:
        raise RuntimeError("gateway passwd identity is unavailable") from exc
    gateway_home = Path(gateway_passwd.pw_dir)
    if gateway_home != Path("/var/lib/hermes-gateway"):
        raise RuntimeError("gateway passwd home is not pinned")
    legacy_env_path = gateway_home / ".hermes/.env"
    op_env_path = gateway_home / ".hermes/.op.env"
    managed_env_path = Path("/etc/hermes/.env")
    pgpass_path = gateway_home / ".pgpass"
    cloud_sql_path = Path("/cloudsql")
    gateway_gids = (
        gateway_identity["effective_gid"],
        *gateway_identity["supplementary_gids"],
    )
    gateway_credential_access = _mode_access_for_identity(
        credential_path,
        uid=gateway_identity["effective_uid"],
        gids=gateway_gids,
    )
    cloud_sql_access = _mode_access_for_identity(
        cloud_sql_path,
        uid=gateway_identity["effective_uid"],
        gids=gateway_gids,
    )
    credential_fd_targets = sorted(
        target
        for target in gateway_identity["fd_targets"]
        if target.removesuffix(" (deleted)") == str(credential_path)
    )
    inherited_cloud_sql_fds = sorted(
        target
        for target in gateway_identity["unix_socket_paths"]
        if target == str(cloud_sql_path)
        or target.startswith(str(cloud_sql_path) + "/")
    )
    writer_credential_fds = sorted(
        target
        for target in writer_identity["fd_targets"]
        if target.removesuffix(" (deleted)") == str(credential_path)
    )
    environment_names = set(gateway_identity["environment_variable_names"])
    database_password_names = sorted(
        name
        for name in environment_names
        if name in {"DATABASE_URL", "PGPASSFILE", "PGPASSWORD"}
    )
    database_connection_names = sorted(
        name
        for name in environment_names
        if name in {"PGDATABASE", "PGHOST", "PGPORT", "PGUSER"}
    )
    runtime_secrets = snapshot.get("runtime_secret_sources")
    if not isinstance(runtime_secrets, dict):
        raise ValueError("runtime secret snapshot is not mutable")
    denied = {"read": False, "write": False, "execute": False}
    runtime_secrets.clear()
    runtime_secrets.update(
        {
            "complete": True,
            "legacy_hermes_env": {
                "path": str(legacy_env_path),
                "exists": legacy_env_path.exists() or legacy_env_path.is_symlink(),
                "gateway_file_access": _mode_access_for_identity(
                    legacy_env_path,
                    uid=gateway_identity["effective_uid"],
                    gids=gateway_gids,
                ),
                "gateway_child_file_access": dict(denied),
            },
            "gateway_readable_secret_files": [],
            "gateway_child_readable_secret_files": [],
            "pgpass": {
                "path": str(pgpass_path),
                "exists": pgpass_path.exists() or pgpass_path.is_symlink(),
                "gateway_file_access": _mode_access_for_identity(
                    pgpass_path,
                    uid=gateway_identity["effective_uid"],
                    gids=gateway_gids,
                ),
                "gateway_child_file_access": dict(denied),
            },
            "gateway_readable_database_credential_files": (
                [str(credential_path)]
                if gateway_credential_access["read"]
                else []
            ),
            "gateway_child_readable_database_credential_files": [],
            "gateway_readable_systemd_environment_files": [],
            "gateway_child_readable_systemd_environment_files": [],
            "open_database_credential_fds": credential_fd_targets,
            "inherited_database_credential_fds": [],
            "cloud_sql_unix_socket": {
                "path": str(cloud_sql_path),
                "gateway_access": cloud_sql_access,
                "gateway_child_access": dict(denied),
                "open_fds": inherited_cloud_sql_fds,
                "inherited_fds": [],
            },
            "effective_gateway_env": {
                "complete": True,
                "values_included": False,
                "database_password_variable_names": database_password_names,
                "database_connection_secret_variable_names": (
                    database_connection_names
                ),
            },
            "sources": [
                {
                    "name": "canonical_writer_database_password",
                    "provisioned_by": "root",
                    "gateway_file_access": gateway_credential_access,
                    "gateway_child_file_access": dict(denied),
                }
            ],
        }
    )
    runtime_secrets["writer_only_forbidden_source_paths_present"] = sorted(
        str(path)
        for path in (legacy_env_path, op_env_path, managed_env_path, pgpass_path)
        if path.exists() or path.is_symlink()
    )
    if writer_credential_fds:
        raise RuntimeError("writer retained its database credential descriptor")
    gateway_config_path = Path(str(manifest.host_contract["gateway_config_path"]))
    if gateway_config_path != DEFAULT_GATEWAY_MANAGED_CONFIG_PATH:
        raise RuntimeError("gateway managed config path is not pinned")
    try:
        gateway_config_value = yaml.load(
            gateway_config_path.read_text(encoding="utf-8"),
            Loader=_UniqueSafeLoader,
        )
    except (OSError, TypeError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError("gateway config is not strict YAML") from exc
    gateway_config_raw = _mapping(gateway_config_value)
    canonical_config = _mapping(gateway_config_raw.get("canonical_brain"))
    gateway_boundary = _mapping(canonical_config.get("writer_boundary"))
    gateway_edge = _mapping(canonical_config.get("discord_edge"))
    plugins_config = _mapping(gateway_config_raw.get("plugins"))
    cron_config = _mapping(gateway_config_raw.get("cron"))
    if (
        set(gateway_config_raw) != {"canonical_brain", "plugins", "cron"}
        or set(canonical_config) != {"writer_boundary", "discord_edge"}
        or gateway_boundary != {"enabled": True}
        or gateway_edge != {"enabled": False}
        or plugins_config != {"enabled": [], "disabled": []}
        or cron_config != {"provider": "builtin"}
    ):
        raise RuntimeError(
            "writer-only gateway managed policy is absent or inconsistent"
        )

    gateway_receipt = _read_runtime_receipt(
        DEFAULT_READINESS_RECEIPT_PATH,
        expected_uid=int(snapshot["gateway_uid"]),
    )
    writer_receipt = _read_runtime_receipt(
        DEFAULT_WRITER_RUNTIME_ATTESTATION_PATH,
        expected_uid=int(snapshot["writer_uid"]),
    )
    gateway_environment_sha256 = _validate_exact_runtime_environment(
        gateway_identity,
        gateway_receipt,
        fixed_values=_FIXED_GATEWAY_ENVIRONMENT,
    )
    writer_environment_sha256 = _validate_exact_runtime_environment(
        writer_identity,
        writer_receipt,
        fixed_values=_FIXED_WRITER_ENVIRONMENT,
    )
    if (
        writer_receipt.get("socket_path") != str(DEFAULT_SOCKET_PATH)
        or writer_receipt.get("socket_inode") != socket_stat.st_ino
        or writer_receipt.get("socket_device") != socket_stat.st_dev
        or writer_receipt.get("socket_owner_uid") != socket_stat.st_uid
        or writer_receipt.get("socket_group_gid") != socket_stat.st_gid
        or writer_receipt.get("socket_mode")
        != f"{stat.S_IMODE(socket_stat.st_mode):04o}"
    ):
        raise RuntimeError("writer readiness socket identity rotated")

    gateway_process.update(
        {
            "dumpable": gateway_receipt.get("gateway_dumpable"),
            "core_soft_limit": gateway_receipt.get("gateway_core_soft_limit"),
            "core_hard_limit": gateway_receipt.get("gateway_core_hard_limit"),
        }
    )
    for deployment_name, identity, runtime_receipt, origin_field in (
        (
            "writer_deployment",
            writer_identity,
            writer_receipt,
            "bootstrap_module_origin",
        ),
        (
            "gateway_deployment",
            gateway_identity,
            gateway_receipt,
            "entry_module_origin",
        ),
    ):
        deployment = snapshot.get(deployment_name)
        if not isinstance(deployment, dict):
            raise ValueError(f"{deployment_name} snapshot is not mutable")
        attestation = deployment.get("attestation")
        if not isinstance(attestation, dict):
            raise ValueError(f"{deployment_name} attestation is not mutable")
        process = attestation.get("process")
        unit = attestation.get("unit")
        policy = deployment.get("policy")
        if not isinstance(process, dict) or not isinstance(unit, dict) or not isinstance(
            policy, Mapping
        ):
            raise ValueError(f"{deployment_name} runtime evidence is not mutable")
        unexpected_imports = sorted(
            {
                *runtime_receipt.get("unexpected_import_paths", []),
                *runtime_receipt.get("unexpected_import_origins", []),
            }
        )
        process.update(
            {
                "effective_import_paths": runtime_receipt.get(
                    "effective_import_paths"
                ),
                "loaded_module_origins": runtime_receipt.get(
                    "loaded_module_origins"
                ),
                "loaded_module_origins_complete": runtime_receipt.get(
                    "loaded_module_origins_complete"
                ),
                "mapped_executable_paths": identity["mapped_executable_paths"],
                "mapped_executable_paths_complete": True,
                "unexpected_import_origins": unexpected_imports,
                "deleted_code_mappings": identity["deleted_code_mappings"],
                "writable_code_mappings": identity["writable_code_mappings"],
                "environment_variable_names": runtime_receipt.get(
                    "effective_environment_variable_names"
                ),
                "environment_variable_value_sha256": runtime_receipt.get(
                    "effective_environment_variable_value_sha256"
                ),
                "environment_identity_sha256": (
                    writer_environment_sha256
                    if deployment_name == "writer_deployment"
                    else gateway_environment_sha256
                ),
                origin_field: runtime_receipt.get(
                    "bootstrap_module_origin"
                    if deployment_name == "writer_deployment"
                    else "gateway_module_origin"
                ),
                "executable_digest_sha256": next(
                    (
                        item.get("digest_sha256")
                        for item in policy.get("import_paths", ())
                        if isinstance(item, Mapping)
                        and item.get("path") == policy.get("interpreter")
                    ),
                    "",
                ),
                "revision": policy.get("revision"),
                "artifact_digest_sha256": policy.get(
                    "artifact_digest_sha256"
                ),
            }
        )
        if deployment_name == "gateway_deployment":
            discovery_paths = list(policy.get("dynamic_python_discovery_paths") or ())
            loaded_origins = list(runtime_receipt.get("loaded_module_origins") or ())
            process.update(
                {
                    "dynamic_python_discovery_complete": True,
                    "dynamic_python_discovery_paths": discovery_paths,
                    "dynamic_python_loaded_origins": sorted(
                        origin
                        for origin in loaded_origins
                        if any(
                            Path(path) == Path(origin)
                            or Path(path) in Path(origin).parents
                            for path in discovery_paths
                        )
                    ),
                    "dynamic_python_loading_mode": policy.get(
                        "dynamic_python_loading_mode"
                    ),
                    "dynamic_python_writable_paths": [],
                }
            )
        injection_names = sorted(
            set(identity["environment_variable_names"])
            & _CODE_INJECTION_ENVIRONMENT_NAMES
        )
        unit.update(
            {
                "alternate_exec_commands": [],
                "environment_files": [],
                "code_injection_environment_variable_names": injection_names,
                "environment_pythonpath": (
                    [] if "PYTHONPATH" not in injection_names else ["present"]
                ),
                "environment_pythonhome": (
                    "" if "PYTHONHOME" not in injection_names else "present"
                ),
                "exec_start": list(policy.get("exec_start") or ()),
                "working_directory": policy.get("working_directory"),
                "interpreter": policy.get("interpreter"),
                "module": policy.get("module"),
                "name": policy.get("unit_name"),
                "revision": policy.get("revision"),
                "artifact_digest_sha256": policy.get(
                    "artifact_digest_sha256"
                ),
                "user_uid": identity["effective_uid"],
                "group_gid": identity["effective_gid"],
            }
        )
        if deployment_name == "writer_deployment":
            unit["config_path"] = policy.get("config_path")
        else:
            unit["module_origin"] = policy.get("module_origin")
    snapshot["runtime_readiness"] = {
        "gateway": _unit_readiness_evidence(
            DEFAULT_GATEWAY_UNIT,
            gateway_systemd,
            gateway_receipt,
            start_time_ticks=gateway_identity["start_time_ticks"],
        ),
        "writer": _unit_readiness_evidence(
            DEFAULT_WRITER_UNIT,
            writer_systemd,
            writer_receipt,
            start_time_ticks=writer_identity["start_time_ticks"],
        ),
    }

    edge_systemd = _systemctl_show(DEFAULT_DISCORD_EDGE_UNIT)
    edge = snapshot.get("discord_edge")
    if not isinstance(edge, dict):
        raise ValueError("Discord edge snapshot is not mutable")
    try:
        edge_pid = int(edge_systemd.get("MainPID") or "0")
    except ValueError as exc:
        raise RuntimeError("Discord edge MainPID is invalid") from exc
    edge.update(
        {
            "complete": True,
            "collected_by_uid": 0,
            "observed_at_unix": now_unix,
            "gateway_enabled": False,
            "writer_authority_enabled": False,
            "unit_exists": edge_systemd.get("LoadState") != "not-found",
            "unit_enabled": False,
            "unit_active": edge_systemd.get("ActiveState") == "active",
            "main_pid": edge_pid,
            "config_exists": (
                DEFAULT_DISCORD_EDGE_CONFIG_PATH.exists()
                or DEFAULT_DISCORD_EDGE_CONFIG_PATH.is_symlink()
            ),
            "token_exists": (
                DEFAULT_DISCORD_EDGE_TOKEN_PATH.exists()
                or DEFAULT_DISCORD_EDGE_TOKEN_PATH.is_symlink()
            ),
            "socket_exists": (
                DEFAULT_DISCORD_EDGE_SOCKET_PATH.exists()
                or DEFAULT_DISCORD_EDGE_SOCKET_PATH.is_symlink()
            ),
            "process_pids": _discord_edge_process_pids(),
        }
    )
    snapshot["authoritative_runtime"] = {
        "gateway_environment_identity_sha256": gateway_environment_sha256,
        "writer_environment_identity_sha256": writer_environment_sha256,
        "gateway_environment_variable_names": sorted(
            set(gateway_identity["environment_variable_names"])
            | set(gateway_receipt["effective_environment_variable_names"])
        ),
        "writer_environment_variable_names": sorted(
            set(writer_identity["environment_variable_names"])
            | set(writer_receipt["effective_environment_variable_names"])
        ),
        "gateway_unexpected_environment_names": sorted(
            name
            for name in (
                set(gateway_identity["environment_variable_names"])
                | set(gateway_receipt["effective_environment_variable_names"])
            )
            if name not in _WRITER_ONLY_ALLOWED_ENVIRONMENT_NAMES
        ),
        "writer_unexpected_environment_names": sorted(
            name
            for name in (
                set(writer_identity["environment_variable_names"])
                | set(writer_receipt["effective_environment_variable_names"])
            )
            if name not in _WRITER_ONLY_ALLOWED_ENVIRONMENT_NAMES
        ),
        "gateway_fd_targets": gateway_identity["fd_targets"],
        "writer_fd_targets": writer_identity["fd_targets"],
        "gateway_unix_socket_paths": gateway_identity["unix_socket_paths"],
        "writer_unix_socket_paths": writer_identity["unix_socket_paths"],
    }
    writer_after = _systemctl_show(DEFAULT_WRITER_UNIT)
    gateway_after = _systemctl_show(DEFAULT_GATEWAY_UNIT)
    try:
        writer_after_pid = int(writer_after["MainPID"])
        gateway_after_pid = int(gateway_after["MainPID"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("systemd MainPID fence is invalid") from exc
    if (
        writer_after_pid != writer_pid
        or gateway_after_pid != gateway_pid
        or _process_start_time_ticks(writer_after_pid)
        != writer_identity["start_time_ticks"]
        or _process_start_time_ticks(gateway_after_pid)
        != gateway_identity["start_time_ticks"]
    ):
        raise RuntimeError("service PID/start-time changed during collection")
    return snapshot


def probe_active_hba_from_writer_config(
    manifest: TrustedDeploymentManifest,
    snapshot: Mapping[str, Any],
) -> ManagedCloudSQLAdminHBAReceipt | None:
    """Actively collect HBA proof through the exact pinned writer config."""

    if manifest.mode != WRITER_ONLY_MODE:
        raise ValueError("active HBA probe requires writer-only manifest")
    writer_policy = _mapping(
        _mapping(snapshot.get("writer_deployment")).get("policy")
    )
    config_path = writer_policy.get("config_path")
    if not isinstance(config_path, str):
        raise ValueError("writer config path is missing from approved policy")
    from gateway.canonical_writer_bootstrap import load_service_config

    config = load_service_config(config_path)
    if config.discord_edge_authority.enabled:
        raise ValueError("writer-only config unexpectedly enables Discord authority")
    database = _mapping(snapshot.get("database"))
    connection = _mapping(database.get("connection"))
    policy = _mapping(database.get("policy"))
    raw_baseline = policy.get("managed_cloudsqladmin_hba_rejection_receipt")
    if raw_baseline is None:
        return None
    if not isinstance(raw_baseline, Mapping):
        raise ValueError("managed HBA baseline is invalid")
    baseline = managed_cloudsqladmin_hba_receipt_from_mapping(raw_baseline)
    configured_baseline = (
        config.privileges.managed_cloudsqladmin_hba_rejection_receipt
    )
    if (
        configured_baseline != baseline
        or config.privileges.managed_cloudsqladmin_hba_rejection_sha256
        != baseline.sha256
        or connection.get("host") != config.database.host
        or connection.get("tls_server_name")
        != config.database.tls_server_name
        or connection.get("port") != config.database.port
        or connection.get("database") != config.database.database
        or connection.get("user") != config.database.user
    ):
        raise ValueError("writer config does not match approved HBA policy")
    return collect_managed_cloudsqladmin_hba_receipt(
        config.database,
        ttl_seconds=ACTIVE_HBA_MAX_AGE_SECONDS,
    )


def _install_active_hba_evidence(
    snapshot: dict[str, Any],
    observed: ManagedCloudSQLAdminHBAReceipt | None,
    *,
    collected_at_unix: int,
) -> str:
    database = _mapping(snapshot.get("database"))
    policy = _mapping(database.get("policy"))
    raw_baseline = policy.get("managed_cloudsqladmin_hba_rejection_receipt")
    if raw_baseline is None:
        if observed is not None:
            raise ValueError("active HBA proof exists without approved baseline")
        if database.get("managed_cloudsqladmin_hba_rejection_evidence") not in (
            None,
            {},
        ):
            raise ValueError("managed HBA evidence exists without approved baseline")
        return ""
    if not isinstance(raw_baseline, Mapping):
        raise ValueError("managed HBA baseline is invalid")
    baseline = managed_cloudsqladmin_hba_receipt_from_mapping(raw_baseline)
    if not isinstance(observed, ManagedCloudSQLAdminHBAReceipt):
        raise ValueError("active managed HBA probe did not return a receipt")
    if (
        observed.host != baseline.host
        or observed.tls_server_name != baseline.tls_server_name
        or observed.port != baseline.port
        or observed.user != baseline.user
        or observed.server_certificate_sha256
        != baseline.server_certificate_sha256
        or not observed.is_fresh(collected_at_unix)
        or not 0
        <= collected_at_unix - observed.observed_at_unix
        <= ACTIVE_HBA_MAX_AGE_SECONDS
    ):
        raise ValueError("active managed HBA proof is stale or not bound")
    evidence = _mapping(
        database.get("managed_cloudsqladmin_hba_rejection_evidence")
    )
    required_metadata = {
        "complete",
        "collector_uid",
        "source_owner_uid",
        "source_mode",
        "source_symlink",
        "same_host",
        "same_tls_server_name",
        "same_port",
        "same_ca",
        "same_user",
        "same_credential",
        "receipt_sha256",
        "receipt",
    }
    if set(evidence) != required_metadata:
        raise ValueError("managed HBA evidence metadata is not exact")
    refreshed = dict(evidence)
    refreshed.update(
        {
            "complete": True,
            "collector_uid": 0,
            "same_host": True,
            "same_tls_server_name": True,
            "same_port": True,
            "same_ca": True,
            "same_user": True,
            "same_credential": True,
            "receipt_sha256": observed.sha256,
            "receipt": observed.as_dict(),
        }
    )
    mutable_database = snapshot.get("database")
    if not isinstance(mutable_database, dict):
        raise ValueError("database snapshot must be mutable object")
    mutable_database["managed_cloudsqladmin_hba_rejection_evidence"] = refreshed
    return observed.sha256


def _authoritative_writer_only_report(
    snapshot: Mapping[str, Any],
    manifest: TrustedDeploymentManifest,
    *,
    readiness: RuntimeReadinessBinding,
    liveness: RuntimeLivenessBinding,
    code_closure: RuntimeCodeClosureBinding,
    hba_sha256: str,
) -> PreflightReport:
    """Evaluate only live or digest-bound writer-only activation facts."""

    gateway_process = _mapping(snapshot.get("gateway_process"))
    writer_process = _mapping(
        _mapping(
            _mapping(snapshot.get("writer_deployment")).get("attestation")
        ).get("process")
    )
    socket_evidence = _mapping(snapshot.get("socket"))
    credential = _mapping(snapshot.get("credential"))
    projection = _mapping(snapshot.get("projection_export"))
    runtime_secrets = _mapping(snapshot.get("runtime_secret_sources"))
    legacy = _mapping(runtime_secrets.get("legacy_hermes_env"))
    pgpass = _mapping(runtime_secrets.get("pgpass"))
    cloud_sql = _mapping(runtime_secrets.get("cloud_sql_unix_socket"))
    runtime = _mapping(snapshot.get("authoritative_runtime"))
    edge = _mapping(snapshot.get("discord_edge"))
    authority = _mapping(snapshot.get("writer_authority_surface"))
    identities = _mapping(authority.get("identities"))
    gateway_identity = _mapping(identities.get("gateway"))
    writer_identity = _mapping(identities.get("writer"))
    gateway_children = _mapping(identities.get("gateway_children"))

    checks: list[PreflightCheck] = [
        PreflightCheck(
            "authority.manifest_release_bound",
            manifest.mode == WRITER_ONLY_MODE
            and snapshot_policy_sha256(snapshot)
            == manifest.snapshot_policy_sha256,
            "root manifest and live release policy must remain digest-bound",
        ),
        PreflightCheck(
            "authority.process_main_pids",
            gateway_process.get("pid") == gateway_process.get("systemd_main_pid")
            and writer_process.get("pid") == writer_process.get("systemd_main_pid")
            and gateway_process.get("process_start_time_ticks")
            == gateway_process.get("systemd_main_pid_start_time_ticks")
            and writer_process.get("process_start_time_ticks")
            == writer_process.get("systemd_main_pid_start_time_ticks"),
            "live service processes must be the exact stable systemd MainPIDs",
        ),
        PreflightCheck(
            "authority.process_hardening",
            gateway_process.get("dumpable") is False
            and gateway_process.get("core_soft_limit") == 0
            and gateway_process.get("core_hard_limit") == 0,
            "in-process readiness must prove PR_GET_DUMPABLE=0 and zero core limits",
        ),
        PreflightCheck(
            "authority.runtime_code_closure",
            all(
                isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
                for value in (
                    code_closure.gateway_sha256,
                    code_closure.writer_sha256,
                )
            ),
            "live Python origins and executable mappings must be immutable and injection-free",
        ),
        PreflightCheck(
            "authority.readiness_receipts",
            all(
                isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
                for value in (readiness.gateway_sha256, readiness.writer_sha256)
            ),
            "both exact MainPIDs must publish fresh digest-bound systemd readiness",
        ),
        PreflightCheck(
            "authority.writer_ping_liveness",
            isinstance(liveness.sha256, str)
            and _SHA256_RE.fullmatch(liveness.sha256) is not None
            and type(liveness.generation) is int
            and liveness.generation > 0
            and _mapping(snapshot.get("runtime_liveness")).get("generation")
            == liveness.generation,
            "the exact gateway MainPID must hold a fresh writer PING generation",
        ),
        PreflightCheck(
            "authority.identities_isolated",
            gateway_identity.get("effective_uid") == snapshot.get("gateway_uid")
            and writer_identity.get("effective_uid") == snapshot.get("writer_uid")
            and snapshot.get("gateway_uid") != snapshot.get("writer_uid")
            and gateway_identity.get("effective_gid")
            == snapshot.get("gateway_gid")
            and writer_identity.get("effective_gid") == snapshot.get("writer_gid")
            and gateway_identity.get("supplementary_gids")
            == snapshot.get("gateway_supplementary_gids")
            and writer_identity.get("supplementary_gids")
            == snapshot.get("writer_supplementary_gids")
            and gateway_children.get("complete") is True
            and gateway_children.get("processes") == [],
            "writer and gateway identities must be distinct with no gateway children",
        ),
        PreflightCheck(
            "authority.writer_socket",
            socket_evidence.get("owner_uid") == snapshot.get("writer_uid")
            and socket_evidence.get("group_gid")
            == socket_evidence.get("expected_group_gid")
            and socket_evidence.get("mode") == "0660",
            "writer socket must have its exact writer/client-group boundary",
        ),
        PreflightCheck(
            "authority.database_credential_isolated",
            credential.get("owner_uid") == snapshot.get("writer_uid")
            and credential.get("mode") in {"0400", "0600"}
            and credential.get("gateway_access")
            == {"read": False, "write": False, "execute": False}
            and runtime_secrets.get("open_database_credential_fds") == []
            and runtime_secrets.get("inherited_database_credential_fds") == [],
            "database credential must be writer-only and absent from gateway descriptors",
        ),
        PreflightCheck(
            "authority.projection_read_boundary",
            projection.get("owner_uid") == snapshot.get("writer_uid")
            and projection.get("group_gid") == snapshot.get("projector_gid")
            and projection.get("mode") == "0640"
            and projection.get("gateway_access")
            == {"read": False, "write": False, "execute": False}
            and projection.get("projector_access")
            == {"read": True, "write": False, "execute": False},
            "projection must be writer-owned, projector-read-only, and gateway-inaccessible",
        ),
        PreflightCheck(
            "authority.runtime_secrets_absent",
            runtime_secrets.get("complete") is True
            and legacy.get("exists") is False
            and pgpass.get("exists") is False
            and runtime.get("gateway_unexpected_environment_names") == []
            and runtime.get("writer_unexpected_environment_names") == []
            and isinstance(
                runtime.get("gateway_environment_identity_sha256"), str
            )
            and _SHA256_RE.fullmatch(
                runtime["gateway_environment_identity_sha256"]
            )
            is not None
            and isinstance(
                runtime.get("writer_environment_identity_sha256"), str
            )
            and _SHA256_RE.fullmatch(
                runtime["writer_environment_identity_sha256"]
            )
            is not None
            and cloud_sql.get("gateway_access")
            == {"read": False, "write": False, "execute": False}
            and cloud_sql.get("open_fds") == []
            and runtime_secrets.get("gateway_readable_systemd_environment_files")
            == []
            and runtime_secrets.get(
                "writer_only_forbidden_source_paths_present"
            )
            == [],
            "writer-only units must have no model, Discord, database, ADC, or environment secret path",
        ),
        PreflightCheck(
            "authority.discord_absent",
            edge.get("complete") is True
            and edge.get("gateway_enabled") is False
            and edge.get("writer_authority_enabled") is False
            and edge.get("unit_exists") is False
            and edge.get("unit_active") is False
            and edge.get("main_pid") == 0
            and edge.get("config_exists") is False
            and edge.get("token_exists") is False
            and edge.get("socket_exists") is False
            and edge.get("process_pids") == [],
            "writer-only canary must contain no Discord unit, credential, socket, or process",
        ),
        PreflightCheck(
            "authority.database_ping_and_hba",
            isinstance(hba_sha256, str)
            and _SHA256_RE.fullmatch(hba_sha256) is not None,
            "fresh immutable writer PING and active peer-bound HBA proof must both pass",
        ),
    ]
    checks.extend(
        PreflightCheck(
            "writer_" + check.name,
            check.passed,
            check.detail,
        )
        for check in _systemd_checks(_mapping(snapshot.get("systemd_properties")))
    )
    checks.extend(
        PreflightCheck(
            "gateway_" + check.name,
            check.passed,
            check.detail,
        )
        for check in _systemd_checks(
            _mapping(snapshot.get("gateway_systemd_properties"))
        )
    )
    return PreflightReport(tuple(checks))


@dataclass(frozen=True)
class CollectorOutcome:
    report: PreflightReport
    receipt: Mapping[str, Any]


def _current_unix() -> int:
    return int(time.time())


def _current_boottime_ns() -> int:
    clock = getattr(time, "CLOCK_BOOTTIME", None)
    getter = getattr(time, "clock_gettime_ns", None)
    if clock is None or not callable(getter):
        raise RuntimeError("linux_boottime_clock_is_unavailable")
    return int(getter(clock))


def _boot_id_sha256() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise RuntimeError("linux_boot_id_is_unavailable") from exc
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise RuntimeError("linux_boot_id_is_invalid") from exc
    if parsed.int == 0 or str(parsed) != value:
        raise RuntimeError("linux_boot_id_is_invalid")
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _positive_int(value: Any) -> int:
    if type(value) is not int or value <= 0:
        return 0
    return value


def collect_and_evaluate(
    manifest_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
) -> CollectorOutcome:
    """Collect, evaluate, and atomically receipt one authoritative preflight."""

    _require_root_linux()
    with _collector_lock(receipt_path):
        _invalidate_root_receipt(receipt_path)
        manifest = load_trusted_manifest(manifest_path)
        return _collect_and_evaluate_locked(manifest, receipt_path)


def _collect_and_evaluate_locked(
    manifest: TrustedDeploymentManifest,
    receipt_path: str | os.PathLike[str],
) -> CollectorOutcome:
    boot_before = _boot_id_sha256()
    snapshot = _collect_live_snapshot(manifest, now_unix=_current_unix())
    observed_hba = probe_active_hba_from_writer_config(manifest, snapshot)
    collected_at_unix = _current_unix()
    snapshot["collected_at_unix"] = collected_at_unix
    hba_sha256 = _install_active_hba_evidence(
        snapshot,
        observed_hba,
        collected_at_unix=collected_at_unix,
    )
    boot_after = _boot_id_sha256()
    if boot_before != boot_after:
        raise RuntimeError("host rebooted during canonical writer preflight")
    collected_at_boottime_ns = _current_boottime_ns()
    readiness = _validate_runtime_readiness(
        snapshot,
        current_boot_id_sha256=boot_after,
        current_boottime_ns=collected_at_boottime_ns,
    )
    liveness = _collect_runtime_liveness(
        snapshot,
        gateway_readiness_sha256=readiness.gateway_sha256,
        current_boot_id_sha256=boot_after,
        current_boottime_ns=collected_at_boottime_ns,
    )
    code_closure = _validate_runtime_code_closure(snapshot)
    report = _authoritative_writer_only_report(
        snapshot,
        manifest,
        readiness=readiness,
        liveness=liveness,
        code_closure=code_closure,
        hba_sha256=hba_sha256,
    )
    if not report.ok:
        failed = sorted(check.name for check in report.checks if not check.passed)
        raise RuntimeError(
            "canonical writer preflight failed: " + ",".join(failed)
        )
    gateway_process = _mapping(snapshot.get("gateway_process"))
    writer_process = _mapping(
        _mapping(_mapping(snapshot.get("writer_deployment")).get("attestation")).get(
            "process"
        )
    )
    report_value = report.to_dict()
    failed_checks = sorted(
        check.name for check in report.checks if not check.passed
    )
    socket_evidence = _mapping(snapshot.get("socket"))
    network_enforcement = _mapping(
        snapshot.get("writer_kernel_network_enforcement")
    )
    allowed_networks = network_enforcement.get("ip_address_allow")
    if (
        not isinstance(allowed_networks, list)
        or len(allowed_networks) != 1
        or not isinstance(allowed_networks[0], str)
    ):
        raise RuntimeError("writer network receipt evidence is invalid")
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "ok": report.ok,
        "mode": manifest.mode,
        "boot_id_sha256": boot_after,
        "collected_at_unix": collected_at_unix,
        "collected_at_boottime_ns": collected_at_boottime_ns,
        "expires_at_boottime_ns": (
            collected_at_boottime_ns + RECEIPT_TTL_SECONDS * 1_000_000_000
        ),
        "manifest_sha256": manifest.manifest_sha256,
        "approved_plan_sha256": manifest.approved_plan_sha256,
        "revision": manifest.revision,
        "artifact_sha256": manifest.artifact_sha256,
        "snapshot_policy_sha256": manifest.snapshot_policy_sha256,
        "snapshot_sha256": _sha256_json(snapshot),
        "report_sha256": _sha256_json(report_value),
        "hba_receipt_sha256": hba_sha256,
        "gateway_readiness_sha256": readiness.gateway_sha256,
        "gateway_liveness_sha256": liveness.sha256,
        "gateway_liveness_generation": liveness.generation,
        "writer_runtime_attestation_sha256": readiness.writer_sha256,
        "gateway_code_closure_sha256": code_closure.gateway_sha256,
        "writer_code_closure_sha256": code_closure.writer_sha256,
        "gateway_main_pid": _positive_int(gateway_process.get("systemd_main_pid")),
        "gateway_start_time_ticks": _positive_int(
            gateway_process.get("systemd_main_pid_start_time_ticks")
        ),
        "writer_main_pid": _positive_int(writer_process.get("systemd_main_pid")),
        "writer_start_time_ticks": _positive_int(
            writer_process.get("systemd_main_pid_start_time_ticks")
        ),
        "writer_socket_device": _positive_int(
            _mapping(snapshot.get("socket")).get("device")
        ),
        "writer_socket_inode": _positive_int(
            socket_evidence.get("inode")
        ),
        "writer_runtime_directory_device": _positive_int(
            socket_evidence.get("runtime_directory_device")
        ),
        "writer_runtime_directory_inode": _positive_int(
            socket_evidence.get("runtime_directory_inode")
        ),
        "writer_ip_address_allow_network": allowed_networks[0],
        "writer_cgroup_device": _positive_int(
            network_enforcement.get("cgroup_device")
        ),
        "writer_cgroup_inode": _positive_int(
            network_enforcement.get("cgroup_inode")
        ),
        "writer_cgroup_main_pid": _positive_int(
            network_enforcement.get("main_pid")
        ),
        "writer_bpf_ingress_direct_program_ids": list(
            network_enforcement.get("ingress_direct_program_ids") or ()
        ),
        "writer_bpf_ingress_effective_program_ids": list(
            network_enforcement.get("ingress_effective_program_ids") or ()
        ),
        "writer_bpf_egress_direct_program_ids": list(
            network_enforcement.get("egress_direct_program_ids") or ()
        ),
        "writer_bpf_egress_effective_program_ids": list(
            network_enforcement.get("egress_effective_program_ids") or ()
        ),
        "failed_checks": failed_checks,
    }
    _fence_live_activation(
        gateway_uid=int(snapshot["gateway_uid"]),
        gateway_pid=receipt["gateway_main_pid"],
        gateway_start_ticks=receipt["gateway_start_time_ticks"],
        gateway_readiness_sha256=readiness.gateway_sha256,
        gateway_liveness_sha256=liveness.sha256,
        minimum_gateway_liveness_generation=liveness.generation,
        writer_uid=int(snapshot["writer_uid"]),
        writer_gid=int(snapshot["writer_gid"]),
        writer_pid=receipt["writer_main_pid"],
        writer_start_ticks=receipt["writer_start_time_ticks"],
        writer_readiness_sha256=readiness.writer_sha256,
        socket_gid=int(_mapping(snapshot.get("socket"))["expected_group_gid"]),
        socket_device=receipt["writer_socket_device"],
        socket_inode=receipt["writer_socket_inode"],
        runtime_directory_device=receipt[
            "writer_runtime_directory_device"
        ],
        runtime_directory_inode=receipt["writer_runtime_directory_inode"],
        ip_address_allow_network=receipt[
            "writer_ip_address_allow_network"
        ],
        cgroup_device=receipt["writer_cgroup_device"],
        cgroup_inode=receipt["writer_cgroup_inode"],
        cgroup_main_pid=receipt["writer_cgroup_main_pid"],
        ingress_direct_program_ids=receipt[
            "writer_bpf_ingress_direct_program_ids"
        ],
        ingress_effective_program_ids=receipt[
            "writer_bpf_ingress_effective_program_ids"
        ],
        egress_direct_program_ids=receipt[
            "writer_bpf_egress_direct_program_ids"
        ],
        egress_effective_program_ids=receipt[
            "writer_bpf_egress_effective_program_ids"
        ],
    )
    if _boot_id_sha256() != boot_after:
        raise RuntimeError("host rebooted before canonical writer receipt commit")
    _write_root_receipt(receipt_path, receipt)
    return CollectorOutcome(report=report, receipt=receipt)


def _atomic_write_json(
    path_value: str | os.PathLike[str],
    value: Mapping[str, Any],
    *,
    owner_uid: int,
    owner_gid: int,
) -> None:
    path = _absolute_normalized_path(path_value)
    parent = path.parent
    _validate_parent_chain(parent.parent, expected_uid=owner_uid)
    parent_stat = os.lstat(parent)
    if (
        stat.S_ISLNK(parent_stat.st_mode)
        or not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != owner_uid
        or parent_stat.st_gid != owner_gid
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
    ):
        raise ValueError("receipt directory must be exact owner-only directory")
    raw = _canonical_bytes(dict(value))
    temporary = parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600)
        os.fchown(descriptor, owner_uid, owner_gid)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("receipt write made no progress")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
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
    result = os.lstat(path)
    if (
        not stat.S_ISREG(result.st_mode)
        or result.st_nlink != 1
        or result.st_uid != owner_uid
        or result.st_gid != owner_gid
        or stat.S_IMODE(result.st_mode) != 0o400
    ):
        raise RuntimeError("atomic receipt ownership verification failed")


def _write_root_receipt(
    path: str | os.PathLike[str],
    value: Mapping[str, Any],
) -> None:
    _atomic_write_json(path, value, owner_uid=0, owner_gid=0)


@contextlib.contextmanager
def _collector_lock(receipt_path_value: str | os.PathLike[str]):
    receipt_path = _absolute_normalized_path(receipt_path_value)
    parent = receipt_path.parent
    _validate_parent_chain(parent.parent, expected_uid=0)
    parent_stat = os.lstat(parent)
    if (
        stat.S_ISLNK(parent_stat.st_mode)
        or not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != 0
        or parent_stat.st_gid != 0
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
    ):
        raise ValueError("collector receipt directory is not root-only")
    lock_path = parent / ".canonical-writer-preflight.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        os.fchown(descriptor, 0, 0)
        os.fchmod(descriptor, 0o600)
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_uid != 0
            or observed.st_gid != 0
            or stat.S_IMODE(observed.st_mode) != 0o600
        ):
            raise RuntimeError("collector lock identity is invalid")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _invalidate_root_receipt(path_value: str | os.PathLike[str]) -> None:
    path = _absolute_normalized_path(path_value)
    parent = path.parent
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(current.st_mode):
        raise RuntimeError("root receipt path is a directory")
    path.unlink()
    descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_receipt_mapping(
    value: Mapping[str, Any],
    manifest: TrustedDeploymentManifest,
    *,
    current_boot_id_sha256: str,
    current_boottime_ns: int,
) -> dict[str, Any]:
    if set(value) != _RECEIPT_KEYS:
        raise ValueError("root preflight receipt fields are not exact")
    failed = value.get("failed_checks")
    if (
        value.get("schema") != RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("mode") != manifest.mode
        or value.get("boot_id_sha256") != current_boot_id_sha256
        or value.get("manifest_sha256") != manifest.manifest_sha256
        or value.get("approved_plan_sha256") != manifest.approved_plan_sha256
        or value.get("revision") != manifest.revision
        or value.get("artifact_sha256") != manifest.artifact_sha256
        or value.get("snapshot_policy_sha256")
        != manifest.snapshot_policy_sha256
        or not isinstance(failed, Sequence)
        or isinstance(failed, (str, bytes, bytearray))
        or failed
    ):
        raise ValueError("root preflight receipt is not an approved success")
    collected = value.get("collected_at_boottime_ns")
    expires = value.get("expires_at_boottime_ns")
    if (
        type(collected) is not int
        or type(expires) is not int
        or type(current_boottime_ns) is not int
        or expires - collected != RECEIPT_TTL_SECONDS * 1_000_000_000
        or not collected <= current_boottime_ns <= expires
    ):
        raise ValueError("root preflight receipt is stale or from the future")
    for name in (
        "boot_id_sha256",
        "manifest_sha256",
        "approved_plan_sha256",
        "artifact_sha256",
        "snapshot_policy_sha256",
        "snapshot_sha256",
        "report_sha256",
        "hba_receipt_sha256",
        "gateway_readiness_sha256",
        "gateway_liveness_sha256",
        "writer_runtime_attestation_sha256",
        "gateway_code_closure_sha256",
        "writer_code_closure_sha256",
    ):
        item = value.get(name)
        if not isinstance(item, str) or _SHA256_RE.fullmatch(item) is None:
            raise ValueError(f"root preflight receipt {name} is invalid")
    if not isinstance(value.get("revision"), str) or _REVISION_RE.fullmatch(
        value["revision"]
    ) is None:
        raise ValueError("root preflight receipt revision is invalid")
    for name in (
        "gateway_main_pid",
        "gateway_start_time_ticks",
        "writer_main_pid",
        "writer_start_time_ticks",
        "writer_socket_device",
        "writer_socket_inode",
        "writer_runtime_directory_device",
        "writer_runtime_directory_inode",
        "writer_cgroup_device",
        "writer_cgroup_inode",
        "writer_cgroup_main_pid",
        "gateway_liveness_generation",
    ):
        if type(value.get(name)) is not int or value[name] <= 0:
            raise ValueError(f"root preflight receipt {name} is invalid")
    if type(value.get("collected_at_unix")) is not int:
        raise ValueError("root preflight receipt wall time is invalid")
    if value["writer_cgroup_main_pid"] != value["writer_main_pid"]:
        raise ValueError("root preflight receipt cgroup MainPID is not exact")
    allowed_network = value.get("writer_ip_address_allow_network")
    try:
        parsed_allowed = ipaddress.ip_network(allowed_network, strict=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("root preflight receipt IP allow network is invalid") from exc
    if (
        parsed_allowed.version != 4
        or parsed_allowed.prefixlen != 32
        or str(parsed_allowed) != allowed_network
    ):
        raise ValueError("root preflight receipt IP allow network is not /32")
    program_fields = {
        name: _validated_program_ids(value.get(name), label=name)
        for name in (
            "writer_bpf_ingress_direct_program_ids",
            "writer_bpf_ingress_effective_program_ids",
            "writer_bpf_egress_direct_program_ids",
            "writer_bpf_egress_effective_program_ids",
        )
    }
    if (
        not set(program_fields["writer_bpf_ingress_direct_program_ids"])
        .issubset(
            program_fields["writer_bpf_ingress_effective_program_ids"]
        )
        or not set(program_fields["writer_bpf_egress_direct_program_ids"])
        .issubset(
            program_fields["writer_bpf_egress_effective_program_ids"]
        )
    ):
        raise ValueError("root preflight receipt BPF programs are not effective")
    return dict(value)


def validate_fresh_receipt(
    receipt_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate the root filesystem authority and same-boot receipt TTL."""

    _require_root_linux()
    with _collector_lock(receipt_path):
        try:
            manifest = load_trusted_manifest(manifest_path)
            receipt = _read_trusted_json(
                receipt_path,
                expected_uid=0,
                expected_gid=0,
            )
            validated = _validate_receipt_mapping(
                receipt,
                manifest,
                current_boot_id_sha256=_boot_id_sha256(),
                current_boottime_ns=_current_boottime_ns(),
            )
            _fence_live_activation(
                gateway_uid=int(manifest.snapshot_template["gateway_uid"]),
                gateway_pid=validated["gateway_main_pid"],
                gateway_start_ticks=validated["gateway_start_time_ticks"],
                gateway_readiness_sha256=validated[
                    "gateway_readiness_sha256"
                ],
                gateway_liveness_sha256=validated[
                    "gateway_liveness_sha256"
                ],
                minimum_gateway_liveness_generation=validated[
                    "gateway_liveness_generation"
                ],
                writer_uid=int(manifest.snapshot_template["writer_uid"]),
                writer_gid=int(manifest.snapshot_template["writer_gid"]),
                writer_pid=validated["writer_main_pid"],
                writer_start_ticks=validated["writer_start_time_ticks"],
                writer_readiness_sha256=validated[
                    "writer_runtime_attestation_sha256"
                ],
                socket_gid=int(
                    _mapping(manifest.snapshot_template["socket"])[
                        "expected_group_gid"
                    ]
                ),
                socket_device=validated["writer_socket_device"],
                socket_inode=validated["writer_socket_inode"],
                runtime_directory_device=validated[
                    "writer_runtime_directory_device"
                ],
                runtime_directory_inode=validated[
                    "writer_runtime_directory_inode"
                ],
                ip_address_allow_network=validated[
                    "writer_ip_address_allow_network"
                ],
                cgroup_device=validated["writer_cgroup_device"],
                cgroup_inode=validated["writer_cgroup_inode"],
                cgroup_main_pid=validated["writer_cgroup_main_pid"],
                ingress_direct_program_ids=validated[
                    "writer_bpf_ingress_direct_program_ids"
                ],
                ingress_effective_program_ids=validated[
                    "writer_bpf_ingress_effective_program_ids"
                ],
                egress_direct_program_ids=validated[
                    "writer_bpf_egress_direct_program_ids"
                ],
                egress_effective_program_ids=validated[
                    "writer_bpf_egress_effective_program_ids"
                ],
            )
            return validated
        except Exception:
            _invalidate_root_receipt(receipt_path)
            raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect or validate the authoritative writer-only preflight",
    )
    parser.add_argument("action", choices=("collect", "validate"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipt", required=True)
    arguments = parser.parse_args(argv)
    if arguments.action == "collect":
        value = dict(
            collect_and_evaluate(
                arguments.manifest,
                arguments.receipt,
            ).receipt
        )
    else:
        value = validate_fresh_receipt(
            arguments.receipt,
            arguments.manifest,
        )
    summary = {
        "ok": True,
        "schema": value["schema"],
        "revision": value["revision"],
        "receipt_sha256": _sha256_json(value),
        "expires_at_boottime_ns": value["expires_at_boottime_ns"],
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "ACTIVE_HBA_MAX_AGE_SECONDS",
    "CollectorOutcome",
    "MANIFEST_SCHEMA",
    "RECEIPT_SCHEMA",
    "RECEIPT_TTL_SECONDS",
    "TrustedDeploymentManifest",
    "WRITER_ONLY_MODE",
    "collect_and_evaluate",
    "load_trusted_manifest",
    "main",
    "probe_active_hba_from_writer_config",
    "snapshot_policy_projection",
    "snapshot_policy_sha256",
    "validate_fresh_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
