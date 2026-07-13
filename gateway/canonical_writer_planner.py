#!/usr/bin/env python3
"""Build and stage digest-bound writer-only canary plans from sealed truth.

This module joins already-reviewed artifacts.  It consumes a sealed release
manifest, the pure systemd unit specification, a loaded secret-free writer
configuration, exact numeric host identities, and independently collected
unit/config digests.  The output is the compact policy template accepted by
the authoritative root collector plus a reviewable unit/install bundle and
fixed collector argv vectors.

Planning is pure.  The only write helpers create root-owned JSON plan/manifest
files in pre-provisioned owner-only directories.  They never install units,
start services, invoke a shell, read a credential value, or mutate Cloud.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from gateway.canonical_writer_bootstrap import (
    CanonicalWriterServiceConfig,
    load_service_config,
)
from gateway.canonical_writer_db import validate_tls_server_name
from gateway.canonical_writer_config_collector import (
    load_config_collector_receipt,
)
from gateway.canonical_writer_boundary import (
    DEFAULT_DISCORD_EDGE_SOCKET_PATH,
    DEFAULT_DISCORD_EDGE_UNIT,
    DEFAULT_GATEWAY_UNIT,
    DEFAULT_SOCKET_PATH,
    DEFAULT_WRITER_UNIT,
)
from gateway.canonical_writer_root_collector import (
    MANIFEST_SCHEMA,
    WRITER_ONLY_MODE,
    TrustedDeploymentManifest,
    snapshot_policy_sha256,
)
from gateway.canonical_writer_host_authority import (
    DEFAULT_NATIVE_OBSERVATION_EVIDENCE_ROOT,
    LEGACY_CLOUD_SQL_HELPER_PATH,
    NATIVE_OBSERVATION_PLAN_SCHEMA,
    NativeObservationPlan,
    NativeObservationReceipt,
    current_host_identity_sha256,
)
from gateway.canonical_writer_activation import (
    ACTIVATION_PLAN_SCHEMA as PACKAGED_ACTIVATION_PLAN_SCHEMA,
    ActivationDigests as PackagedActivationDigests,
    ActivationPaths as PackagedActivationPaths,
    ActivationPlan as PackagedActivationPlan,
    CANARY_GATEWAY_GID,
    CANARY_GATEWAY_UID,
    CANARY_PROJECTOR_GID,
    CANARY_PROJECTOR_UID,
    CANARY_SOCKET_CLIENT_GID,
    CANARY_WRITER_GID,
    CANARY_WRITER_UID,
    DEFAULT_DATABASE_CA_PATH,
    DEFAULT_GATEWAY_CONFIG_SOURCE_PATH,
    DEFAULT_NATIVE_PLAN_PATH,
    DEFAULT_STAGED_PLAN_PATH,
    DEFAULT_STAGED_GATEWAY_UNIT_PATH,
    DEFAULT_STAGED_NATIVE_PLAN_PATH,
    DEFAULT_STAGED_WRITER_UNIT_PATH,
    DEFAULT_WRITER_CONFIG_SOURCE_PATH,
    InstallArtifact,
    NumericIdentities,
    SystemdBundle as PackagedSystemdBundle,
    load_durable_native_observation_receipt,
    load_native_observation_plan,
)
from gateway.canonical_writer_gateway_bootstrap import (
    _StrictConfigLoader,
    _validate_writer_only_policy,
)
from gateway.canonical_writer_release_contract import (
    DEFAULT_RELEASE_BASE,
    GATEWAY_MODULE,
    INCOMPLETE_MARKER_NAME,
    MAX_RELEASE_MANIFEST_BYTES,
    RELEASE_MANIFEST_NAME,
    RELEASE_SCHEMA,
    WRITER_MODULE,
    ReleaseManifest,
    SystemdUnitBundle,
    TreeEntry,
    WriterOnlyUnitSpec,
    render_systemd_units,
)


ACTIVATION_PLAN_SCHEMA = PACKAGED_ACTIVATION_PLAN_SCHEMA
DEFAULT_WRITER_UNIT_PATH = Path("/etc/systemd/system/muncho-canonical-writer.service")
DEFAULT_GATEWAY_UNIT_PATH = Path("/etc/systemd/system/hermes-cloud-gateway.service")
DEFAULT_PROJECTION_FILENAME = "canonical-events.json"
ROOT_COLLECTOR_MODULE = "gateway.canonical_writer_root_collector"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MAX_ROOT_JSON_BYTES = 1024 * 1024
_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_ROOT_JSON_MODE = 0o400
_RELEASE_MANIFEST_KEYS = frozenset({
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
})


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
        raise ValueError("activation value is not canonical JSON") from exc
    return encoded.encode("utf-8", errors="strict")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _positive_int(value: int, label: str) -> int:
    if type(value) is not int or value <= 0 or value >= 1 << 31:
        raise ValueError(f"{label} must be a positive host identity")
    return value


def _absolute_normalized_path(
    value: str | os.PathLike[str],
    label: str,
) -> Path:
    raw = os.fspath(value)
    path = Path(raw)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(path) != raw
        or _CONTROL_RE.search(raw) is not None
    ):
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("activation JSON contains duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"activation JSON contains non-JSON constant: {value}")


def _decode_strict_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_trusted_root_file(
    path_value: str | os.PathLike[str],
    *,
    allowed_modes: frozenset[int],
    maximum: int,
    expected_gid: int = 0,
) -> bytes:
    path = _absolute_normalized_path(path_value, "trusted root file")
    _validate_root_parent_chain(path.parent)
    before = os.lstat(path)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != 0
        or before.st_gid != expected_gid
        or stat.S_IMODE(before.st_mode) not in allowed_modes
        or not 0 < before.st_size <= maximum
        or os.listxattr(path)
    ):
        raise PermissionError("trusted root file identity is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_mode != before.st_mode
            or opened.st_uid != before.st_uid
            or opened.st_gid != before.st_gid
            or opened.st_size != before.st_size
        ):
            raise RuntimeError("trusted root file changed during open")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(raw) != opened.st_size
            or after.st_mode != opened.st_mode
            or after.st_uid != opened.st_uid
            or after.st_gid != opened.st_gid
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
        ):
            raise RuntimeError("trusted root file changed while reading")
    finally:
        os.close(descriptor)
    if not raw or len(raw) > maximum:
        raise ValueError("trusted root file exceeds its bound")
    return raw


def _parse_release_manifest_mapping(
    value: Mapping[str, Any],
    *,
    expected_revision: str,
) -> ReleaseManifest:
    if set(value) != _RELEASE_MANIFEST_KEYS:
        raise ValueError("release manifest fields are not exact")
    if _REVISION_RE.fullmatch(expected_revision) is None:
        raise ValueError("release revision is invalid")
    root = DEFAULT_RELEASE_BASE / expected_revision
    python_version = value.get("python_version")
    if (
        not isinstance(python_version, str)
        or re.fullmatch(r"3\.(?:11|12|13)\.[0-9]+", python_version) is None
    ):
        raise ValueError("release Python version is invalid")
    major, minor, _patch = python_version.split(".")
    site_packages = root / "venv/lib" / f"python{major}.{minor}" / "site-packages"
    if (
        value.get("schema") != RELEASE_SCHEMA
        or value.get("revision") != expected_revision
        or value.get("artifact_root") != str(root)
        or value.get("interpreter") != str(root / "venv/bin/python")
        or value.get("writer_module") != WRITER_MODULE
        or value.get("gateway_module") != GATEWAY_MODULE
        or value.get("writer_module_origin")
        != str(site_packages / "gateway/canonical_writer_bootstrap.py")
        or value.get("gateway_module_origin")
        != str(site_packages / "gateway/canonical_writer_gateway_bootstrap.py")
    ):
        raise ValueError("release manifest identity is not exact")
    _digest(value.get("artifact_sha256"), "release artifact")
    raw_entries = value.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("release manifest entries are invalid")
    entries: list[TreeEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            raise ValueError("release manifest entry is invalid")
        kind = raw.get("kind")
        fields = {
            "file": {"path", "kind", "mode", "size", "sha256"},
            "directory": {"path", "kind", "mode"},
            "symlink": {"path", "kind", "mode", "target"},
        }.get(kind)
        if fields is None or set(raw) != fields:
            raise ValueError("release manifest entry fields are not exact")
        relative = raw.get("path")
        mode = raw.get("mode")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or Path(relative).as_posix() != relative
            or ".." in Path(relative).parts
            or relative in {RELEASE_MANIFEST_NAME, INCOMPLETE_MARKER_NAME}
            or _CONTROL_RE.search(relative) is not None
            or not isinstance(mode, str)
            or re.fullmatch(r"0[0-7]{3}", mode) is None
        ):
            raise ValueError("release manifest entry identity is invalid")
        if kind == "file":
            size = raw.get("size")
            if type(size) is not int or size < 0:
                raise ValueError("release manifest file size is invalid")
            digest = _digest(raw.get("sha256"), "release manifest file")
            entries.append(TreeEntry(relative, kind, mode, size=size, sha256=digest))
        elif kind == "symlink":
            target = raw.get("target")
            if (
                not isinstance(target, str)
                or not target
                or _CONTROL_RE.search(target) is not None
            ):
                raise ValueError("release manifest symlink target is invalid")
            entries.append(TreeEntry(relative, kind, mode, target=target))
        else:
            entries.append(TreeEntry(relative, kind, mode))
    paths = [entry.path for entry in entries]
    if paths != sorted(set(paths)):
        raise ValueError("release manifest entries are not sorted and unique")
    release = ReleaseManifest(
        revision=expected_revision,
        artifact_root=str(root),
        python_version=python_version,
        interpreter=str(value["interpreter"]),
        writer_module_origin=str(value["writer_module_origin"]),
        gateway_module_origin=str(value["gateway_module_origin"]),
        entries=tuple(entries),
        artifact_sha256=str(value["artifact_sha256"]),
    )
    if (
        release.to_mapping() != dict(value)
        or release.computed_artifact_sha256 != release.artifact_sha256
    ):
        raise ValueError("release manifest self-digest is invalid")
    return release


def load_release_manifest(
    revision: str,
) -> tuple[ReleaseManifest, bytes]:
    """Load one exact root-sealed release manifest for native planning."""

    _require_root_linux()
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("release revision is invalid")
    root = DEFAULT_RELEASE_BASE / revision
    root_stat = os.lstat(root)
    if (
        stat.S_ISLNK(root_stat.st_mode)
        or not stat.S_ISDIR(root_stat.st_mode)
        or root_stat.st_uid != 0
        or root_stat.st_gid != 0
        or stat.S_IMODE(root_stat.st_mode) != 0o555
        or os.path.lexists(root / INCOMPLETE_MARKER_NAME)
    ):
        raise PermissionError("release root is not sealed and complete")
    raw = _read_trusted_root_file(
        root / RELEASE_MANIFEST_NAME,
        allowed_modes=frozenset({0o400}),
        maximum=MAX_RELEASE_MANIFEST_BYTES,
    )
    value = _decode_strict_json(raw, label="release manifest")
    release = _parse_release_manifest_mapping(value, expected_revision=revision)
    if raw != _canonical_bytes(release.to_mapping()) + b"\n":
        raise ValueError("release manifest bytes are not canonical")
    return release, raw


@dataclass(frozen=True)
class HostNumericIdentities:
    gateway_uid: int
    gateway_gid: int
    gateway_home: str
    writer_uid: int
    writer_gid: int
    writer_home: str
    socket_client_gid: int
    projector_uid: int
    projector_gid: int
    projector_home: str
    gateway_supplementary_gids: tuple[int, ...]
    writer_supplementary_gids: tuple[int, ...]

    def validate(self) -> None:
        values = {
            name: _positive_int(getattr(self, name), name)
            for name in (
                "gateway_uid",
                "gateway_gid",
                "writer_uid",
                "writer_gid",
                "socket_client_gid",
                "projector_uid",
                "projector_gid",
            )
        }
        if (
            len({
                values["gateway_uid"],
                values["writer_uid"],
                values["projector_uid"],
            })
            != 3
        ):
            raise ValueError("gateway, writer, and projector UIDs must be distinct")
        if (
            len({
                values["gateway_gid"],
                values["writer_gid"],
                values["socket_client_gid"],
                values["projector_gid"],
            })
            != 4
        ):
            raise ValueError("writer-only groups must be distinct")
        if self.gateway_supplementary_gids != tuple(
            sorted((self.gateway_gid, self.socket_client_gid))
        ):
            raise ValueError(
                "gateway must have only its primary and socket-client groups"
            )
        if self.writer_supplementary_gids != tuple(
            sorted((self.writer_gid, self.projector_gid))
        ):
            raise ValueError("writer must have only its primary and projector groups")
        if (
            self.gateway_home != "/var/lib/hermes-gateway"
            or self.writer_home != "/nonexistent"
            or self.projector_home != "/nonexistent"
        ):
            raise ValueError("writer-only identity homes are not exact")

    def to_mapping(self) -> dict[str, Any]:
        self.validate()
        return {
            "gateway_uid": self.gateway_uid,
            "gateway_gid": self.gateway_gid,
            "gateway_home": self.gateway_home,
            "writer_uid": self.writer_uid,
            "writer_gid": self.writer_gid,
            "writer_home": self.writer_home,
            "socket_client_gid": self.socket_client_gid,
            "projector_uid": self.projector_uid,
            "projector_gid": self.projector_gid,
            "projector_home": self.projector_home,
            "gateway_supplementary_gids": list(self.gateway_supplementary_gids),
            "writer_supplementary_gids": list(self.writer_supplementary_gids),
        }


@dataclass(frozen=True)
class NativeObservationDigests:
    """Content digests used to build a deployable native observation plan."""

    writer_unit_sha256: str
    gateway_unit_sha256: str
    exporter_unit_sha256: str
    tmpfiles_sha256: str
    writer_config_sha256: str
    gateway_config_sha256: str

    def validate(self) -> None:
        for name in self.__dataclass_fields__:
            _digest(getattr(self, name), name)


@dataclass(frozen=True)
class ExternalNativeExecutableMapping:
    """One independently reviewed native executable mapping identity."""

    path: str
    sha256: str

    def validate(self, artifact_root: str | os.PathLike[str]) -> None:
        path = _absolute_normalized_path(self.path, "native mapping path")
        root = _absolute_normalized_path(
            artifact_root,
            "native mapping artifact root",
        )
        if path == root or root in path.parents:
            raise ValueError("native mapping path must be outside sealed release")
        _digest(self.sha256, "native mapping sha256")

    def to_mapping(
        self,
        artifact_root: str | os.PathLike[str],
    ) -> dict[str, str]:
        self.validate(artifact_root)
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class PreapprovedNativeExecutablePolicy:
    """Exact reviewed external mappings for each isolated service process."""

    writer: tuple[ExternalNativeExecutableMapping, ...]
    gateway: tuple[ExternalNativeExecutableMapping, ...]

    @staticmethod
    def _validate_service(
        values: tuple[ExternalNativeExecutableMapping, ...],
        *,
        service: str,
        artifact_root: str | os.PathLike[str],
    ) -> None:
        if type(values) is not tuple or not values:
            raise ValueError(
                f"{service} native mapping policy must be a non-empty tuple"
            )
        if any(type(item) is not ExternalNativeExecutableMapping for item in values):
            raise TypeError(
                f"{service} native mapping policy contains an invalid entry"
            )
        paths = [item.path for item in values]
        if paths != sorted(paths):
            raise ValueError(f"{service} native mapping policy must be exactly sorted")
        if len(paths) != len(set(paths)):
            raise ValueError(f"{service} native mapping policy paths must be unique")
        for item in values:
            item.validate(artifact_root)

    def validate(self, artifact_root: str | os.PathLike[str]) -> None:
        self._validate_service(
            self.writer,
            service="writer",
            artifact_root=artifact_root,
        )
        self._validate_service(
            self.gateway,
            service="gateway",
            artifact_root=artifact_root,
        )

    def writer_mapping(
        self,
        artifact_root: str | os.PathLike[str],
    ) -> list[dict[str, str]]:
        self._validate_service(
            self.writer,
            service="writer",
            artifact_root=artifact_root,
        )
        return [item.to_mapping(artifact_root) for item in self.writer]

    def gateway_mapping(
        self,
        artifact_root: str | os.PathLike[str],
    ) -> list[dict[str, str]]:
        self._validate_service(
            self.gateway,
            service="gateway",
            artifact_root=artifact_root,
        )
        return [item.to_mapping(artifact_root) for item in self.gateway]


def _entry_by_path(manifest: ReleaseManifest, relative: str) -> TreeEntry:
    matches = [entry for entry in manifest.entries if entry.path == relative]
    if len(matches) != 1:
        raise ValueError(f"release manifest lacks exact path: {relative}")
    return matches[0]


def _subtree_sha256(manifest: ReleaseManifest, relative: str) -> str:
    prefix = relative.rstrip("/") + "/"
    entries = [
        entry.to_mapping()
        for entry in manifest.entries
        if entry.path == relative or entry.path.startswith(prefix)
    ]
    if not entries or entries[0]["path"] != relative:
        raise ValueError(f"release manifest lacks subtree: {relative}")
    return _sha256_json(entries)


def _release_import_policy(manifest: ReleaseManifest) -> list[dict[str, str]]:
    if (
        manifest.schema != "muncho-writer-only-release.v1"
        or _REVISION_RE.fullmatch(manifest.revision) is None
        or manifest.artifact_sha256 != manifest.computed_artifact_sha256
    ):
        raise ValueError("release manifest identity is invalid")
    root = Path(manifest.artifact_root)
    interpreter = Path(manifest.interpreter)
    try:
        interpreter_relative = interpreter.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("release interpreter escapes artifact") from exc
    interpreter_entry = _entry_by_path(manifest, interpreter_relative)
    if (
        interpreter_entry.kind != "file"
        or _SHA256_RE.fullmatch(interpreter_entry.sha256) is None
    ):
        raise ValueError("release interpreter digest is invalid")
    python_parts = manifest.python_version.split(".")
    if len(python_parts) != 3:
        raise ValueError("release Python version is invalid")
    minor = f"{python_parts[0]}.{python_parts[1]}"
    stdlib_pattern = re.compile(rf"^python/[^/]+/lib/python{re.escape(minor)}$")
    stdlib_candidates = [
        entry.path
        for entry in manifest.entries
        if entry.kind == "directory" and stdlib_pattern.fullmatch(entry.path)
    ]
    if len(stdlib_candidates) != 1:
        raise ValueError("release managed stdlib identity is ambiguous")
    stdlib_relative = stdlib_candidates[0]
    site_relative = f"venv/lib/python{minor}/site-packages"
    site_entry = _entry_by_path(manifest, site_relative)
    if site_entry.kind != "directory":
        raise ValueError("release site-packages identity is invalid")
    return [
        {
            "path": str(root),
            "kind": "application",
            "digest_sha256": manifest.artifact_sha256,
            "object_type": "directory",
        },
        {
            "path": str(interpreter),
            "kind": "interpreter",
            "digest_sha256": interpreter_entry.sha256,
            "object_type": "regular_file",
        },
        {
            "path": str(root / stdlib_relative),
            "kind": "stdlib",
            "digest_sha256": _subtree_sha256(manifest, stdlib_relative),
            "object_type": "directory",
        },
        {
            "path": str(root / site_relative),
            "kind": "site_packages",
            "digest_sha256": _subtree_sha256(manifest, site_relative),
            "object_type": "directory",
        },
    ]


def _writer_config_policy(
    config: CanonicalWriterServiceConfig,
    *,
    sql_ip: str,
    sql_tls_server_name: str,
    identities: HostNumericIdentities,
) -> dict[str, Any]:
    identities.validate()
    if (
        config.gateway_unit != DEFAULT_GATEWAY_UNIT
        or config.socket_path != DEFAULT_SOCKET_PATH
        or config.gateway_uid != identities.gateway_uid
        or config.writer_uid != identities.writer_uid
        or config.writer_gid != identities.writer_gid
        or config.socket_gid != identities.socket_client_gid
        or config.projector_gid != identities.projector_gid
        or config.discord_edge_authority.enabled is not False
    ):
        raise ValueError("loaded writer config does not match host identities")
    database = config.database
    credential = database.credential
    if (
        database.host != sql_ip
        or database.tls_server_name != sql_tls_server_name
        or credential.path is None
        or credential.fd is not None
        or credential.expected_uid != identities.writer_uid
        or credential.expected_gid != identities.writer_gid
    ):
        raise ValueError("writer database authority is not host-pinned")
    receipt = config.privileges.managed_cloudsqladmin_hba_rejection_receipt
    if (
        receipt is None
        or receipt.host != sql_ip
        or receipt.tls_server_name != sql_tls_server_name
        or receipt.port != database.port
        or receipt.user != database.user
        or receipt.sha256
        != config.privileges.managed_cloudsqladmin_hba_rejection_sha256
    ):
        raise ValueError("writer HBA baseline is absent or not SQL-bound")
    return {
        "expected_user": database.user,
        "connection": {
            "host": sql_ip,
            "tls_server_name": sql_tls_server_name,
            "port": database.port,
            "database": database.database,
            "user": database.user,
        },
        "policy": {
            "private_schema_identity_sha256": (
                config.privileges.private_schema_identity_sha256
            ),
            "managed_cloudsqladmin_hba_rejection_receipt": receipt.as_dict(),
            "managed_cloudsqladmin_hba_rejection_sha256": receipt.sha256,
        },
        "managed_cloudsqladmin_hba_rejection_evidence": {
            "complete": False,
            "collector_uid": 0,
            "source_owner_uid": 0,
            "source_mode": "0400",
            "source_symlink": False,
            "same_host": False,
            "same_tls_server_name": False,
            "same_port": False,
            "same_ca": False,
            "same_user": False,
            "same_credential": False,
            "receipt_sha256": receipt.sha256,
            "receipt": receipt.as_dict(),
        },
    }


def _snapshot_template(
    release: ReleaseManifest,
    unit_spec: WriterOnlyUnitSpec,
    writer_config: CanonicalWriterServiceConfig,
    identities: HostNumericIdentities,
    native_executable_policy: PreapprovedNativeExecutablePolicy,
    *,
    sql_ip: str,
    sql_tls_server_name: str,
) -> dict[str, Any]:
    import_paths = _release_import_policy(release)
    root = release.artifact_root
    native_executable_policy.validate(root)
    writer_exec = [
        release.interpreter,
        "-B",
        "-I",
        "-m",
        WRITER_MODULE,
        "--config",
        str(unit_spec.writer_config),
    ]
    gateway_exec = [
        release.interpreter,
        "-B",
        "-I",
        "-m",
        GATEWAY_MODULE,
    ]
    writer_policy = {
        "unit_name": DEFAULT_WRITER_UNIT,
        "artifact_root": root,
        "revision": release.revision,
        "artifact_digest_sha256": release.artifact_sha256,
        "interpreter": release.interpreter,
        "module": WRITER_MODULE,
        "module_origin": release.writer_module_origin,
        "config_path": str(unit_spec.writer_config),
        "exec_start": writer_exec,
        "working_directory": root,
        "import_paths": import_paths,
        "runtime_directory": str(unit_spec.writer_runtime),
        "projection_export_directory": str(unit_spec.projection_directory),
        "read_write_paths": [
            str(unit_spec.writer_runtime),
            str(unit_spec.projection_directory),
        ],
        "bind_paths": [],
        "bind_read_only_paths": [root],
        "preapproved_external_native_executable_mappings": (
            native_executable_policy.writer_mapping(root)
        ),
        "preapproved_kernel_executable_mappings": ["[vdso]", "[vsyscall]"],
    }
    gateway_policy = {
        "unit_name": DEFAULT_GATEWAY_UNIT,
        "artifact_root": root,
        "revision": release.revision,
        "artifact_digest_sha256": release.artifact_sha256,
        "interpreter": release.interpreter,
        "module": GATEWAY_MODULE,
        "module_origin": release.gateway_module_origin,
        "exec_start": gateway_exec,
        "working_directory": root,
        "import_paths": import_paths,
        "read_write_paths": [
            str(unit_spec.gateway_runtime),
        ],
        "bind_paths": [],
        "bind_read_only_paths": [root],
        "dynamic_python_loading_mode": "disabled",
        "dynamic_python_discovery_paths": [],
        "preapproved_external_native_executable_mappings": (
            native_executable_policy.gateway_mapping(root)
        ),
        "preapproved_kernel_executable_mappings": ["[vdso]", "[vsyscall]"],
    }
    export_path = unit_spec.projection_directory / DEFAULT_PROJECTION_FILENAME
    exporter_policy = {
        "enabled": False,
        "unit_name": "muncho-canonical-writer-export.service",
        "timer_name": "muncho-canonical-writer-export.timer",
        "artifact_root": root,
        "revision": release.revision,
        "artifact_digest_sha256": release.artifact_sha256,
        "interpreter": release.interpreter,
        "module": WRITER_MODULE,
        "module_origin": release.writer_module_origin,
        "config_path": str(unit_spec.writer_config),
        "export_path": str(export_path),
        "export_limit": 200_000,
        "exec_start": [
            *writer_exec,
            "--export-events",
            str(export_path),
            "--export-limit",
            "200000",
        ],
        "read_write_paths": [str(unit_spec.projection_directory)],
        "bind_paths": [],
        "bind_read_only_paths": [root],
        "timer_schedule": {
            "OnCalendar": "*-*-* *:*:00",
            "Persistent": False,
        },
    }
    database = _writer_config_policy(
        writer_config,
        sql_ip=sql_ip,
        sql_tls_server_name=sql_tls_server_name,
        identities=identities,
    )
    return {
        "deployment_mode": WRITER_ONLY_MODE,
        "gateway_uid": identities.gateway_uid,
        "gateway_gid": identities.gateway_gid,
        "writer_uid": identities.writer_uid,
        "writer_gid": identities.writer_gid,
        "projector_gid": identities.projector_gid,
        "gateway_supplementary_gids": list(identities.gateway_supplementary_gids),
        "writer_supplementary_gids": list(identities.writer_supplementary_gids),
        "socket": {"expected_group_gid": identities.socket_client_gid},
        "gateway_process": {},
        "writer_deployment": {
            "policy": writer_policy,
            "attestation": {"process": {}, "unit": {}, "mounts": {}},
        },
        "gateway_deployment": {
            "policy": gateway_policy,
            "attestation": {"process": {}, "unit": {}, "mounts": {}},
        },
        "writer_authority_surface": {
            "identities": {},
            "privileged_execution_inventory": {},
            "projection_exporter": {
                "policy": exporter_policy,
                "attestation": {},
            },
        },
        "database": database,
        "credential": {},
        "projection_export": {},
        "runtime_secret_sources": {},
        "discord_edge": {
            "gateway_enabled": False,
            "writer_authority_enabled": False,
            "unit_name": DEFAULT_DISCORD_EDGE_UNIT,
            "config_path": "/etc/muncho/discord-edge.json",
            "token_path": "/etc/muncho/discord-edge-credentials/bot-token",
            "socket_path": str(DEFAULT_DISCORD_EDGE_SOCKET_PATH),
        },
    }


def build_native_observation_plan(
    release: ReleaseManifest,
    unit_spec: WriterOnlyUnitSpec,
    writer_config: CanonicalWriterServiceConfig,
    identities: HostNumericIdentities,
    digests: NativeObservationDigests,
    *,
    observation_id: str,
    boot_id_sha256: str,
    host_identity_sha256: str,
    release_manifest_file_sha256: str,
    config_collector_receipt_sha256: str,
    database_ca_sha256: str,
    external_iam_policy_sha256: str,
    sql_private_ip: str,
    sql_tls_server_name: str,
) -> NativeObservationPlan:
    """Build the separately approved discovery-only first-start plan."""

    if not isinstance(digests, NativeObservationDigests):
        raise TypeError("native observation digests are required")
    identities.validate()
    digests.validate()
    for value, label in (
        (boot_id_sha256, "native boot identity"),
        (host_identity_sha256, "native host identity"),
        (release_manifest_file_sha256, "release manifest file"),
        (config_collector_receipt_sha256, "config collector receipt"),
        (database_ca_sha256, "database CA"),
        (external_iam_policy_sha256, "external IAM policy"),
    ):
        _digest(value, label)
    if (
        release.artifact_sha256 != release.computed_artifact_sha256
        or release.revision != Path(release.artifact_root).name
    ):
        raise ValueError("native observation release identity is invalid")
    try:
        address = ipaddress.ip_address(sql_private_ip)
    except ValueError as exc:
        raise ValueError("native observation SQL endpoint is invalid") from exc
    if (
        address.version != 4
        or str(address) != sql_private_ip
        or not address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or unit_spec.database_ip_allow != (f"{sql_private_ip}/32",)
    ):
        raise ValueError("native observation SQL boundary is not exact")
    validate_tls_server_name(sql_tls_server_name)
    database = writer_config.database
    if (
        database.host != sql_private_ip
        or database.tls_server_name != sql_tls_server_name
        or database.ca_file is None
        or not database.ca_file.is_absolute()
        or ".." in database.ca_file.parts
        or database.ca_file != Path("/etc/muncho/trust/cloudsql-server-ca.pem")
    ):
        raise ValueError("native observation writer database binding drifted")
    bundle = render_systemd_units(release, unit_spec)
    if (
        _sha256_text(bundle.writer_service) != digests.writer_unit_sha256
        or _sha256_text(bundle.gateway_service) != digests.gateway_unit_sha256
        or _sha256_text(bundle.exporter_service) != digests.exporter_unit_sha256
        or _sha256_text(bundle.tmpfiles) != digests.tmpfiles_sha256
    ):
        raise ValueError("native observation unit digest drifted")
    writer_argv = [
        release.interpreter,
        "-B",
        "-I",
        "-m",
        WRITER_MODULE,
        "--config",
        str(unit_spec.writer_config),
    ]
    gateway_argv = [
        release.interpreter,
        "-B",
        "-I",
        "-m",
        GATEWAY_MODULE,
    ]
    return NativeObservationPlan.from_mapping({
        "schema": NATIVE_OBSERVATION_PLAN_SCHEMA,
        "boot_id_sha256": boot_id_sha256,
        "host_identity_sha256": host_identity_sha256,
        "observation_id": observation_id,
        "revision": release.revision,
        "artifact_root": release.artifact_root,
        "artifact_sha256": release.artifact_sha256,
        "release_manifest_file_sha256": release_manifest_file_sha256,
        "config_collector_receipt_sha256": config_collector_receipt_sha256,
        "gateway_unit": {
            "name": DEFAULT_GATEWAY_UNIT,
            "path": str(DEFAULT_GATEWAY_UNIT_PATH),
            "sha256": digests.gateway_unit_sha256,
        },
        "writer_unit": {
            "name": DEFAULT_WRITER_UNIT,
            "path": str(DEFAULT_WRITER_UNIT_PATH),
            "sha256": digests.writer_unit_sha256,
        },
        "gateway_argv": gateway_argv,
        "writer_argv": writer_argv,
        "gateway_config": {
            "path": str(unit_spec.gateway_config),
            "sha256": digests.gateway_config_sha256,
        },
        "writer_config": {
            "path": str(unit_spec.writer_config),
            "sha256": digests.writer_config_sha256,
        },
        "identities": {
            "gateway_uid": identities.gateway_uid,
            "gateway_gid": identities.gateway_gid,
            "gateway_home": identities.gateway_home,
            "gateway_supplementary_gids": list(identities.gateway_supplementary_gids),
            "writer_uid": identities.writer_uid,
            "writer_gid": identities.writer_gid,
            "writer_home": identities.writer_home,
            "writer_supplementary_gids": list(identities.writer_supplementary_gids),
            "socket_group_gid": identities.socket_client_gid,
            "projector_uid": identities.projector_uid,
            "projector_gid": identities.projector_gid,
            "projector_home": identities.projector_home,
        },
        "database": {
            "ip_network": f"{sql_private_ip}/32",
            "tls_server_name": sql_tls_server_name,
            "ca_path": str(database.ca_file),
            "ca_sha256": database_ca_sha256,
        },
        "discord": {
            "unit_name": DEFAULT_DISCORD_EDGE_UNIT,
            "config_path": "/etc/muncho/discord-edge.json",
            "token_path": "/etc/muncho/discord-edge-credentials/bot-token",
            "socket_path": str(DEFAULT_DISCORD_EDGE_SOCKET_PATH),
            "required_absent": True,
        },
        "native_discovery_policy": {
            "allowed_roots": ["/usr/lib"],
            "allowed_kernel_executable_mappings": ["[vdso]", "[vsyscall]"],
            "maximum_mappings": 256,
            "required_owner_uid": 0,
            "required_owner_gid": 0,
            "require_regular": True,
            "require_single_link": True,
            "forbid_symlink": True,
            "forbid_acl": True,
            "forbid_xattrs": True,
            "forbid_writable": True,
            "forbid_deleted": True,
            "exclude_artifact_root": True,
            "digest_algorithm": "sha256",
        },
        "legacy_helper_path": str(LEGACY_CLOUD_SQL_HELPER_PATH),
        "external_iam_policy_sha256": external_iam_policy_sha256,
    })


def build_activation_plan(
    release: ReleaseManifest,
    unit_spec: WriterOnlyUnitSpec,
    writer_config: CanonicalWriterServiceConfig,
    identities: HostNumericIdentities,
    native_observation_receipt: NativeObservationReceipt,
    *,
    writer_config_sha256: str,
    gateway_config_sha256: str,
    release_manifest_file_sha256: str,
    database_ca_sha256: str,
    external_iam_policy_sha256: str,
    external_iam_receipt_path: str | os.PathLike[str],
    sql_private_ip: str,
    sql_tls_server_name: str,
    paths: PackagedActivationPaths,
) -> PackagedActivationPlan:
    """Build the sole deployable v3 plan from a stopped native receipt."""

    if not isinstance(native_observation_receipt, NativeObservationReceipt):
        raise TypeError("stopped native observation receipt is required")
    identities.validate()
    for value, label in (
        (writer_config_sha256, "writer config"),
        (gateway_config_sha256, "gateway config"),
        (release_manifest_file_sha256, "release manifest file"),
        (database_ca_sha256, "database CA"),
        (external_iam_policy_sha256, "external IAM policy"),
    ):
        _digest(value, label)
    native = native_observation_receipt.value
    native_plan = NativeObservationPlan.from_mapping(native["plan"])
    if not isinstance(paths, PackagedActivationPaths):
        raise TypeError("production-pinned activation paths are required")
    identities.validate()
    unit_bundle = render_systemd_units(release, unit_spec)
    try:
        sql_network = ipaddress.ip_network(
            native_plan.value["database"]["ip_network"],
            strict=True,
        )
    except ValueError as exc:
        raise ValueError("native observation SQL network is invalid") from exc
    native_bindings = {
        "revision": release.revision,
        "artifact_root": release.artifact_root,
        "artifact_sha256": release.artifact_sha256,
        "release_manifest_file_sha256": release_manifest_file_sha256,
        "gateway_unit": {
            "name": DEFAULT_GATEWAY_UNIT,
            "path": str(DEFAULT_GATEWAY_UNIT_PATH),
            "sha256": _sha256_text(unit_bundle.gateway_service),
        },
        "writer_unit": {
            "name": DEFAULT_WRITER_UNIT,
            "path": str(DEFAULT_WRITER_UNIT_PATH),
            "sha256": _sha256_text(unit_bundle.writer_service),
        },
        "gateway_config": {
            "path": str(unit_spec.gateway_config),
            "sha256": gateway_config_sha256,
        },
        "writer_config": {
            "path": str(unit_spec.writer_config),
            "sha256": writer_config_sha256,
        },
        "identities": _native_plan_identity_mapping(identities),
        "database": {
            "ip_network": f"{sql_private_ip}/32",
            "tls_server_name": sql_tls_server_name,
            "ca_path": str(paths.database_ca_path),
            "ca_sha256": database_ca_sha256,
        },
        "external_iam_policy_sha256": external_iam_policy_sha256,
    }
    if str(sql_network.network_address) != sql_private_ip:
        raise ValueError("native observation SQL endpoint binding drifted")
    for name, expected in native_bindings.items():
        if native_plan.value[name] != expected:
            raise ValueError(f"native observation {name} binding drifted")
    observed = native["observation"]
    writer_native = tuple(
        ExternalNativeExecutableMapping(**item)
        for item in observed["writer_service"]["external_native_mappings"]
    )
    gateway_native = tuple(
        ExternalNativeExecutableMapping(**item)
        for item in observed["gateway_service"]["external_native_mappings"]
    )
    native_policy = PreapprovedNativeExecutablePolicy(
        writer=writer_native,
        gateway=gateway_native,
    )
    snapshot = _snapshot_template(
        release,
        unit_spec,
        writer_config,
        identities,
        native_policy,
        sql_ip=sql_private_ip,
        sql_tls_server_name=sql_tls_server_name,
    )
    snapshot["writer_deployment"]["policy"][
        "preapproved_kernel_executable_mappings"
    ] = list(observed["writer_service"]["kernel_executable_mappings"])
    snapshot["gateway_deployment"]["policy"][
        "preapproved_kernel_executable_mappings"
    ] = list(observed["gateway_service"]["kernel_executable_mappings"])
    writer_unit_sha256 = _sha256_text(unit_bundle.writer_service)
    gateway_unit_sha256 = _sha256_text(unit_bundle.gateway_service)
    exporter_unit_sha256 = _sha256_text(unit_bundle.exporter_service)
    tmpfiles_sha256 = _sha256_text(unit_bundle.tmpfiles)
    receipt_path = (
        DEFAULT_NATIVE_OBSERVATION_EVIDENCE_ROOT
        / release.revision
        / native_plan.sha256
        / "native-observation.json"
    )
    iam_path = _absolute_normalized_path(
        external_iam_receipt_path,
        "external IAM receipt path",
    )
    host_contract = {
        "gateway_unit_fragment_path": str(paths.gateway_unit_path),
        "gateway_unit_fragment_sha256": gateway_unit_sha256,
        "writer_unit_fragment_path": str(paths.writer_unit_path),
        "writer_unit_fragment_sha256": writer_unit_sha256,
        "gateway_config_path": str(paths.gateway_config_path),
        "gateway_config_sha256": gateway_config_sha256,
        "writer_config_path": str(paths.writer_config_path),
        "writer_config_sha256": writer_config_sha256,
        "projection_export_path": str(paths.projection_export_path),
        "external_iam_policy_sha256": external_iam_policy_sha256,
        "external_iam_receipt_path": str(iam_path),
        "legacy_helper_path": str(LEGACY_CLOUD_SQL_HELPER_PATH),
        "native_observation_plan_sha256": native_plan.sha256,
        "native_observation_receipt_path": str(receipt_path),
        "native_observation_receipt_sha256": native_observation_receipt.sha256,
    }
    trusted = TrustedDeploymentManifest.from_mapping({
        "schema": MANIFEST_SCHEMA,
        "mode": WRITER_ONLY_MODE,
        "revision": release.revision,
        "artifact_sha256": release.artifact_sha256,
        "snapshot_policy_sha256": snapshot_policy_sha256(snapshot),
        "host_contract": host_contract,
        "snapshot_template": snapshot,
    })
    manifest = trusted.to_mapping()
    manifest_sha256 = _sha256_json(manifest)
    numeric = NumericIdentities.from_mapping(identities.to_mapping())
    packaged_bundle = PackagedSystemdBundle.from_mapping(unit_bundle.to_mapping())
    packaged_digests = PackagedActivationDigests(
        native_observation_plan_sha256=native_plan.sha256,
        native_observation_receipt_sha256=native_observation_receipt.sha256,
        release_manifest_file_sha256=release_manifest_file_sha256,
        database_ca_sha256=database_ca_sha256,
        external_iam_policy_sha256=external_iam_policy_sha256,
        deployment_manifest_sha256=manifest_sha256,
        writer_unit_sha256=writer_unit_sha256,
        gateway_unit_sha256=gateway_unit_sha256,
        exporter_unit_sha256=exporter_unit_sha256,
        tmpfiles_sha256=tmpfiles_sha256,
        writer_config_sha256=writer_config_sha256,
        gateway_config_sha256=gateway_config_sha256,
    )
    artifacts = {
        "manifest": InstallArtifact(
            source_path=None,
            target_path=paths.manifest_path,
            sha256=manifest_sha256,
            mode=0o400,
            uid=0,
            gid=0,
            maximum_bytes=8 * 1024 * 1024,
        ),
        "writer_unit": InstallArtifact(
            source_path=None,
            target_path=paths.writer_unit_path,
            sha256=writer_unit_sha256,
            mode=0o644,
            uid=0,
            gid=0,
            maximum_bytes=256 * 1024,
        ),
        "gateway_unit": InstallArtifact(
            source_path=None,
            target_path=paths.gateway_unit_path,
            sha256=gateway_unit_sha256,
            mode=0o644,
            uid=0,
            gid=0,
            maximum_bytes=256 * 1024,
        ),
        "tmpfiles": InstallArtifact(
            source_path=None,
            target_path=paths.tmpfiles_path,
            sha256=tmpfiles_sha256,
            mode=0o644,
            uid=0,
            gid=0,
            maximum_bytes=256 * 1024,
        ),
        "writer_config": InstallArtifact(
            source_path=paths.writer_config_source_path,
            target_path=paths.writer_config_path,
            sha256=writer_config_sha256,
            mode=0o440,
            uid=0,
            gid=numeric.writer_gid,
            maximum_bytes=2 * 1024 * 1024,
        ),
        "gateway_config": InstallArtifact(
            source_path=paths.gateway_config_source_path,
            target_path=paths.gateway_config_path,
            sha256=gateway_config_sha256,
            mode=0o444,
            uid=0,
            gid=0,
            maximum_bytes=2 * 1024 * 1024,
        ),
    }
    base_collector = (
        release.interpreter,
        "-B",
        "-I",
        "-m",
        ROOT_COLLECTOR_MODULE,
        "collect",
        "--manifest",
        str(paths.manifest_path),
        "--receipt",
        str(paths.root_receipt_path),
    )
    base_validator = (
        release.interpreter,
        "-B",
        "-I",
        "-m",
        ROOT_COLLECTOR_MODULE,
        "validate",
        "--manifest",
        str(paths.manifest_path),
        "--receipt",
        str(paths.root_receipt_path),
    )
    unsigned = {
        "schema": PACKAGED_ACTIVATION_PLAN_SCHEMA,
        "revision": release.revision,
        "identities": identities.to_mapping(),
        "paths": paths.to_mapping(),
        "digests": packaged_digests.to_mapping(),
        "deployment_manifest": manifest,
        "native_observation_receipt": native_observation_receipt.to_mapping(),
        "systemd_bundle": packaged_bundle.to_mapping(),
        "install_artifacts": {
            name: item.to_mapping() for name, item in sorted(artifacts.items())
        },
        "collector_argv": list(base_collector),
        "validator_argv": list(base_validator),
    }
    return PackagedActivationPlan.from_mapping({
        **unsigned,
        "activation_plan_sha256": _sha256_json(unsigned),
    })


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    return int(getter()) if callable(getter) else -1


def _require_root_linux() -> None:
    if _effective_uid() != 0:
        raise PermissionError("writer_activation_write_requires_uid_0")
    if sys.platform != "linux":
        raise RuntimeError("writer_activation_write_requires_linux")


def _validate_root_parent_chain(path: Path) -> None:
    current = path
    while True:
        item = os.lstat(current)
        if (
            stat.S_ISLNK(item.st_mode)
            or not stat.S_ISDIR(item.st_mode)
            or item.st_uid != 0
            or item.st_gid != 0
            or stat.S_IMODE(item.st_mode) & 0o022
        ):
            raise PermissionError("activation parent path is not root-controlled")
        if current == current.parent:
            return
        current = current.parent


def _write_atomic_root_staged_file(path: Path, payload: bytes) -> None:
    """Install one immutable staged input without exposing partial bytes."""

    _require_root_linux()
    allowed = frozenset({
        DEFAULT_STAGED_WRITER_UNIT_PATH,
        DEFAULT_STAGED_GATEWAY_UNIT_PATH,
        DEFAULT_STAGED_NATIVE_PLAN_PATH,
        DEFAULT_STAGED_PLAN_PATH,
    })
    path = _absolute_normalized_path(path, "native staged file")
    if path not in allowed:
        raise ValueError("native staged output path is not production-pinned")
    if not payload or len(payload) > _MAX_ROOT_JSON_BYTES:
        raise ValueError("native staged output exceeds its bound")
    _validate_root_parent_chain(path.parent.parent)
    parent = os.lstat(path.parent)
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != 0
        or parent.st_gid != 0
        or stat.S_IMODE(parent.st_mode) != 0o700
        or os.listxattr(path.parent)
    ):
        raise PermissionError("native staging directory must be root-only")
    if os.path.lexists(path):
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    installed = False
    try:
        os.fchown(descriptor, 0, 0)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("native staged write made no progress")
            offset += written
        os.fchmod(descriptor, _ROOT_JSON_MODE)
        os.fsync(descriptor)
        item = os.fstat(descriptor)
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_nlink != 1
            or item.st_uid != 0
            or item.st_gid != 0
            or stat.S_IMODE(item.st_mode) != _ROOT_JSON_MODE
            or item.st_size != len(payload)
        ):
            raise RuntimeError("native staged temporary identity is invalid")
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        installed = True
        temporary.unlink()
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if installed:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    observed = _read_trusted_root_file(
        path,
        allowed_modes=frozenset({_ROOT_JSON_MODE}),
        maximum=max(len(payload), 1),
    )
    if observed != payload:
        raise RuntimeError("native staged output readback drifted")


def _remove_exact_staged_file(path: Path, payload: bytes) -> None:
    observed = _read_trusted_root_file(
        path,
        allowed_modes=frozenset({_ROOT_JSON_MODE}),
        maximum=max(len(payload), 1),
    )
    if observed != payload:
        raise RuntimeError("native staged rollback refuses drifted content")
    path.unlink()
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _validated_preexisting_native_outputs(
    outputs: Sequence[tuple[Path, bytes]],
) -> set[Path]:
    """Classify only exact deterministic unit residue as resumable."""

    preexisting: set[Path] = set()
    for target, payload in outputs:
        if not os.path.lexists(target):
            continue
        # The native plan carries a fresh observation id and therefore cannot
        # be reconstructed after a lost response.  Its existing copy must be
        # handled by the enclosing staged publisher.  Deterministic unit bytes,
        # however, are safe to resume only when their exact protected content
        # matches the bytes derived by the planner.
        if target == DEFAULT_STAGED_NATIVE_PLAN_PATH:
            raise FileExistsError(target)
        observed = _read_trusted_root_file(
            target,
            allowed_modes=frozenset({_ROOT_JSON_MODE}),
            maximum=max(len(payload), 1),
        )
        if observed != payload:
            raise RuntimeError("native staged output collision drifted")
        preexisting.add(target)
    return preexisting


def _current_boot_id_sha256() -> str:
    raw = Path("/proc/sys/kernel/random/boot_id").read_bytes()
    if len(raw) > 128:
        raise RuntimeError("native planning boot identity is oversized")
    try:
        value = raw.decode("ascii", errors="strict").strip()
        parsed = uuid.UUID(value)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("native planning boot identity is invalid") from exc
    if parsed.int == 0 or str(parsed) != value:
        raise RuntimeError("native planning boot identity is invalid")
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _canary_identities() -> HostNumericIdentities:
    identities = HostNumericIdentities(
        gateway_uid=CANARY_GATEWAY_UID,
        gateway_gid=CANARY_GATEWAY_GID,
        gateway_home="/var/lib/hermes-gateway",
        writer_uid=CANARY_WRITER_UID,
        writer_gid=CANARY_WRITER_GID,
        writer_home="/nonexistent",
        socket_client_gid=CANARY_SOCKET_CLIENT_GID,
        projector_uid=CANARY_PROJECTOR_UID,
        projector_gid=CANARY_PROJECTOR_GID,
        projector_home="/nonexistent",
        gateway_supplementary_gids=tuple(sorted((
            CANARY_GATEWAY_GID,
            CANARY_SOCKET_CLIENT_GID,
        ))),
        writer_supplementary_gids=tuple(sorted((
            CANARY_WRITER_GID,
            CANARY_PROJECTOR_GID,
        ))),
    )
    identities.validate()
    return identities


def build_and_stage_native_observation_plan(
    *,
    revision: str,
    external_iam_policy_sha256: str,
    config_collector_receipt_sha256: str,
) -> Mapping[str, str]:
    """Build and stage one exact discovery plan without granting approval."""

    _require_root_linux()
    _digest(external_iam_policy_sha256, "external IAM policy")
    _digest(config_collector_receipt_sha256, "config collector receipt")
    collector_receipt = load_config_collector_receipt(
        revision=revision,
        receipt_sha256=config_collector_receipt_sha256,
        require_fresh=True,
    )
    collector_database = collector_receipt.value["database"]
    sql_private_ip = str(collector_database["host"])
    sql_tls_server_name = str(collector_database["tls_server_name"])
    release, release_manifest_raw = load_release_manifest(revision)
    writer_raw = _read_trusted_root_file(
        DEFAULT_WRITER_CONFIG_SOURCE_PATH,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_CONFIG_BYTES,
    )
    gateway_raw = _read_trusted_root_file(
        DEFAULT_GATEWAY_CONFIG_SOURCE_PATH,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_CONFIG_BYTES,
    )
    writer_mapping = _decode_strict_json(writer_raw, label="staged writer config")
    if writer_raw not in {
        _canonical_bytes(writer_mapping),
        _canonical_bytes(writer_mapping) + b"\n",
    }:
        raise ValueError("staged writer config bytes are not canonical")
    writer_config = load_service_config(DEFAULT_WRITER_CONFIG_SOURCE_PATH)
    if _read_trusted_root_file(
        DEFAULT_WRITER_CONFIG_SOURCE_PATH,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_CONFIG_BYTES,
    ) != writer_raw:
        raise RuntimeError("staged writer config rotated during planning")
    try:
        gateway_value = yaml.load(
            gateway_raw.decode("utf-8", errors="strict"),
            Loader=_StrictConfigLoader,
        )
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError("staged gateway config is not strict YAML") from exc
    _validate_writer_only_policy(gateway_value)
    database_ca_raw = _read_trusted_root_file(
        DEFAULT_DATABASE_CA_PATH,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
        maximum=_MAX_CONFIG_BYTES,
        expected_gid=CANARY_WRITER_GID,
    )
    unit_spec = WriterOnlyUnitSpec(
        database_ip_allow=(f"{sql_private_ip}/32",),
    )
    bundle = render_systemd_units(release, unit_spec)
    digests = NativeObservationDigests(
        writer_unit_sha256=_sha256_text(bundle.writer_service),
        gateway_unit_sha256=_sha256_text(bundle.gateway_service),
        exporter_unit_sha256=_sha256_text(bundle.exporter_service),
        tmpfiles_sha256=_sha256_text(bundle.tmpfiles),
        writer_config_sha256=hashlib.sha256(writer_raw).hexdigest(),
        gateway_config_sha256=hashlib.sha256(gateway_raw).hexdigest(),
    )
    release_manifest_file_sha256 = hashlib.sha256(
        release_manifest_raw
    ).hexdigest()
    database_ca_sha256 = hashlib.sha256(database_ca_raw).hexdigest()
    collector_receipt.require_bindings(
        revision=revision,
        release_artifact_sha256=release.artifact_sha256,
        release_manifest_file_sha256=release_manifest_file_sha256,
        writer_config_sha256=digests.writer_config_sha256,
        gateway_config_sha256=digests.gateway_config_sha256,
        database_ca_sha256=database_ca_sha256,
        sql_private_ip=sql_private_ip,
        sql_tls_server_name=sql_tls_server_name,
    )
    plan = build_native_observation_plan(
        release,
        unit_spec,
        writer_config,
        _canary_identities(),
        digests,
        observation_id=str(uuid.uuid4()),
        boot_id_sha256=_current_boot_id_sha256(),
        host_identity_sha256=current_host_identity_sha256(),
        release_manifest_file_sha256=release_manifest_file_sha256,
        config_collector_receipt_sha256=config_collector_receipt_sha256,
        database_ca_sha256=database_ca_sha256,
        external_iam_policy_sha256=external_iam_policy_sha256,
        sql_private_ip=sql_private_ip,
        sql_tls_server_name=sql_tls_server_name,
    )
    canonical_plan = _canonical_bytes(plan.to_mapping())
    reparsed = NativeObservationPlan.from_mapping(
        _decode_strict_json(canonical_plan, label="native observation plan")
    )
    if reparsed.sha256 != plan.sha256 or reparsed.to_mapping() != plan.to_mapping():
        raise RuntimeError("native observation plan roundtrip drifted")
    for path, expected, modes, expected_gid in (
        (
            DEFAULT_WRITER_CONFIG_SOURCE_PATH,
            writer_raw,
            frozenset({0o400}),
            0,
        ),
        (
            DEFAULT_GATEWAY_CONFIG_SOURCE_PATH,
            gateway_raw,
            frozenset({0o400}),
            0,
        ),
        (
            DEFAULT_DATABASE_CA_PATH,
            database_ca_raw,
            frozenset({0o400, 0o440, 0o444}),
            CANARY_WRITER_GID,
        ),
    ):
        if _read_trusted_root_file(
            path,
            allowed_modes=modes,
            maximum=_MAX_CONFIG_BYTES,
            expected_gid=expected_gid,
        ) != expected:
            raise RuntimeError("native planning input rotated before staging")
    final_collector_receipt = load_config_collector_receipt(
        revision=revision,
        receipt_sha256=config_collector_receipt_sha256,
        require_fresh=True,
    )
    final_collector_receipt.require_bindings(
        revision=revision,
        release_artifact_sha256=release.artifact_sha256,
        release_manifest_file_sha256=release_manifest_file_sha256,
        writer_config_sha256=digests.writer_config_sha256,
        gateway_config_sha256=digests.gateway_config_sha256,
        database_ca_sha256=database_ca_sha256,
        sql_private_ip=sql_private_ip,
        sql_tls_server_name=sql_tls_server_name,
    )
    if final_collector_receipt.to_mapping() != collector_receipt.to_mapping():
        raise RuntimeError("config collector receipt rotated before staging")
    outputs = (
        (
            DEFAULT_STAGED_WRITER_UNIT_PATH,
            bundle.writer_service.encode("utf-8"),
        ),
        (
            DEFAULT_STAGED_GATEWAY_UNIT_PATH,
            bundle.gateway_service.encode("utf-8"),
        ),
        (DEFAULT_STAGED_NATIVE_PLAN_PATH, canonical_plan),
    )
    preexisting = _validated_preexisting_native_outputs(outputs)
    created: list[tuple[Path, bytes]] = []
    try:
        for target, payload in outputs:
            if target in preexisting:
                continue
            _write_atomic_root_staged_file(target, payload)
            created.append((target, payload))
    except BaseException as write_error:
        rollback_errors: list[BaseException] = []
        for target, payload in reversed(created):
            try:
                _remove_exact_staged_file(target, payload)
            except BaseException as exc:
                rollback_errors.append(exc)
        if rollback_errors:
            raise ExceptionGroup(
                "native staging and rollback failed",
                [write_error, *rollback_errors],
            ) from None
        raise
    return {
        "artifact_sha256": release.artifact_sha256,
        "config_collector_receipt_sha256": config_collector_receipt_sha256,
        "database_ca_sha256": database_ca_sha256,
        "gateway_config_sha256": digests.gateway_config_sha256,
        "gateway_unit_sha256": digests.gateway_unit_sha256,
        "native_observation_plan_sha256": plan.sha256,
        "release_manifest_file_sha256": release_manifest_file_sha256,
        "writer_config_sha256": digests.writer_config_sha256,
        "writer_unit_sha256": digests.writer_unit_sha256,
    }


def _native_plan_identity_mapping(
    identities: HostNumericIdentities,
) -> Mapping[str, Any]:
    identities.validate()
    return {
        "gateway_uid": identities.gateway_uid,
        "gateway_gid": identities.gateway_gid,
        "gateway_home": identities.gateway_home,
        "gateway_supplementary_gids": list(
            identities.gateway_supplementary_gids
        ),
        "writer_uid": identities.writer_uid,
        "writer_gid": identities.writer_gid,
        "writer_home": identities.writer_home,
        "writer_supplementary_gids": list(
            identities.writer_supplementary_gids
        ),
        "socket_group_gid": identities.socket_client_gid,
        "projector_uid": identities.projector_uid,
        "projector_gid": identities.projector_gid,
        "projector_home": identities.projector_home,
    }


def build_and_stage_final_activation_plan(
    *,
    native_observation_receipt_sha256: str,
) -> Mapping[str, str]:
    """Build and stage the sole deployable plan from durable stopped evidence."""

    _require_root_linux()
    _digest(
        native_observation_receipt_sha256,
        "expected native observation receipt",
    )
    native_plan = load_native_observation_plan(DEFAULT_NATIVE_PLAN_PATH)
    native_receipt = load_durable_native_observation_receipt(native_plan)
    if native_receipt.sha256 != native_observation_receipt_sha256:
        raise PermissionError("durable native observation receipt digest drifted")

    native_value = native_plan.value
    revision = str(native_value["revision"])
    config_collector_receipt_sha256 = str(
        native_value["config_collector_receipt_sha256"]
    )
    collector_receipt = load_config_collector_receipt(
        revision=revision,
        receipt_sha256=config_collector_receipt_sha256,
        require_fresh=False,
    )
    database_binding = native_value["database"]
    sql_network = ipaddress.ip_network(database_binding["ip_network"], strict=True)
    sql_private_ip = str(sql_network.network_address)
    sql_tls_server_name = str(database_binding["tls_server_name"])

    release, release_manifest_raw = load_release_manifest(revision)
    writer_raw = _read_trusted_root_file(
        DEFAULT_WRITER_CONFIG_SOURCE_PATH,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_CONFIG_BYTES,
    )
    gateway_raw = _read_trusted_root_file(
        DEFAULT_GATEWAY_CONFIG_SOURCE_PATH,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_CONFIG_BYTES,
    )
    writer_mapping = _decode_strict_json(writer_raw, label="staged writer config")
    if writer_raw not in {
        _canonical_bytes(writer_mapping),
        _canonical_bytes(writer_mapping) + b"\n",
    }:
        raise ValueError("staged writer config bytes are not canonical")
    writer_config = load_service_config(DEFAULT_WRITER_CONFIG_SOURCE_PATH)
    try:
        gateway_value = yaml.load(
            gateway_raw.decode("utf-8", errors="strict"),
            Loader=_StrictConfigLoader,
        )
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError("staged gateway config is not strict YAML") from exc
    _validate_writer_only_policy(gateway_value)
    database_ca_raw = _read_trusted_root_file(
        DEFAULT_DATABASE_CA_PATH,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
        maximum=_MAX_CONFIG_BYTES,
        expected_gid=CANARY_WRITER_GID,
    )

    release_manifest_file_sha256 = hashlib.sha256(
        release_manifest_raw
    ).hexdigest()
    writer_config_sha256 = hashlib.sha256(writer_raw).hexdigest()
    gateway_config_sha256 = hashlib.sha256(gateway_raw).hexdigest()
    database_ca_sha256 = hashlib.sha256(database_ca_raw).hexdigest()
    collector_receipt.require_bindings(
        revision=revision,
        release_artifact_sha256=release.artifact_sha256,
        release_manifest_file_sha256=release_manifest_file_sha256,
        writer_config_sha256=writer_config_sha256,
        gateway_config_sha256=gateway_config_sha256,
        database_ca_sha256=database_ca_sha256,
        sql_private_ip=sql_private_ip,
        sql_tls_server_name=sql_tls_server_name,
    )

    identities = _canary_identities()
    if native_value["identities"] != _native_plan_identity_mapping(identities):
        raise RuntimeError("native observation identity binding drifted")
    unit_spec = WriterOnlyUnitSpec(
        database_ip_allow=(str(sql_network),),
    )
    bundle = render_systemd_units(release, unit_spec)
    expected_native_bindings = {
        "artifact_root": release.artifact_root,
        "artifact_sha256": release.artifact_sha256,
        "release_manifest_file_sha256": release_manifest_file_sha256,
        "config_collector_receipt_sha256": collector_receipt.sha256,
        "gateway_unit": {
            "name": DEFAULT_GATEWAY_UNIT,
            "path": str(DEFAULT_GATEWAY_UNIT_PATH),
            "sha256": _sha256_text(bundle.gateway_service),
        },
        "writer_unit": {
            "name": DEFAULT_WRITER_UNIT,
            "path": str(DEFAULT_WRITER_UNIT_PATH),
            "sha256": _sha256_text(bundle.writer_service),
        },
        "gateway_config": {
            "path": str(unit_spec.gateway_config),
            "sha256": gateway_config_sha256,
        },
        "writer_config": {
            "path": str(unit_spec.writer_config),
            "sha256": writer_config_sha256,
        },
        "database": {
            "ip_network": str(sql_network),
            "tls_server_name": sql_tls_server_name,
            "ca_path": str(DEFAULT_DATABASE_CA_PATH),
            "ca_sha256": database_ca_sha256,
        },
    }
    for name, expected in expected_native_bindings.items():
        if native_value[name] != expected:
            raise RuntimeError(f"native observation {name} binding drifted")

    paths = PackagedActivationPaths()
    final_plan = build_activation_plan(
        release,
        unit_spec,
        writer_config,
        identities,
        native_receipt,
        writer_config_sha256=writer_config_sha256,
        gateway_config_sha256=gateway_config_sha256,
        release_manifest_file_sha256=release_manifest_file_sha256,
        database_ca_sha256=database_ca_sha256,
        external_iam_policy_sha256=str(
            native_value["external_iam_policy_sha256"]
        ),
        external_iam_receipt_path=paths.external_iam_receipt_path,
        sql_private_ip=sql_private_ip,
        sql_tls_server_name=sql_tls_server_name,
        paths=paths,
    )
    canonical_plan = _canonical_bytes(final_plan.to_mapping())
    reparsed = PackagedActivationPlan.from_mapping(
        _decode_strict_json(canonical_plan, label="final activation plan")
    )
    if (
        reparsed.sha256 != final_plan.sha256
        or reparsed.to_mapping() != final_plan.to_mapping()
    ):
        raise RuntimeError("final activation plan roundtrip drifted")

    for path, expected, modes, expected_gid in (
        (
            DEFAULT_WRITER_CONFIG_SOURCE_PATH,
            writer_raw,
            frozenset({0o400}),
            0,
        ),
        (
            DEFAULT_GATEWAY_CONFIG_SOURCE_PATH,
            gateway_raw,
            frozenset({0o400}),
            0,
        ),
        (
            DEFAULT_DATABASE_CA_PATH,
            database_ca_raw,
            frozenset({0o400, 0o440, 0o444}),
            CANARY_WRITER_GID,
        ),
    ):
        if _read_trusted_root_file(
            path,
            allowed_modes=modes,
            maximum=_MAX_CONFIG_BYTES,
            expected_gid=expected_gid,
        ) != expected:
            raise RuntimeError("final planning input rotated before staging")
    final_collector_receipt = load_config_collector_receipt(
        revision=revision,
        receipt_sha256=config_collector_receipt_sha256,
        require_fresh=False,
    )
    if final_collector_receipt.to_mapping() != collector_receipt.to_mapping():
        raise RuntimeError("config collector receipt rotated before final staging")
    final_native_plan = load_native_observation_plan(DEFAULT_NATIVE_PLAN_PATH)
    final_native_receipt = load_durable_native_observation_receipt(
        final_native_plan
    )
    if (
        final_native_plan.to_mapping() != native_plan.to_mapping()
        or final_native_receipt.to_mapping() != native_receipt.to_mapping()
    ):
        raise RuntimeError("native stopped evidence rotated before final staging")
    if os.path.lexists(DEFAULT_STAGED_PLAN_PATH):
        raise FileExistsError(DEFAULT_STAGED_PLAN_PATH)
    _write_atomic_root_staged_file(DEFAULT_STAGED_PLAN_PATH, canonical_plan)
    return {
        "activation_plan_sha256": final_plan.sha256,
        "artifact_sha256": release.artifact_sha256,
        "config_collector_receipt_sha256": collector_receipt.sha256,
        "native_observation_plan_sha256": native_plan.sha256,
        "native_observation_receipt_sha256": native_receipt.sha256,
        "release_manifest_file_sha256": release_manifest_file_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build root-pinned writer-only canary plans",
    )
    parser.add_argument(
        "action",
        choices=("build-native-plan", "build-final-plan"),
    )
    parser.add_argument("--revision")
    parser.add_argument("--external-iam-policy-sha256")
    parser.add_argument("--config-collector-receipt-sha256")
    parser.add_argument("--native-observation-receipt-sha256")
    arguments = parser.parse_args(argv)

    def required(value: str | None, option: str) -> str:
        if value is None:
            parser.error(f"{option} is required for {arguments.action}")
        return value

    if arguments.action == "build-native-plan":
        result = build_and_stage_native_observation_plan(
            revision=required(arguments.revision, "--revision"),
            external_iam_policy_sha256=required(
                arguments.external_iam_policy_sha256,
                "--external-iam-policy-sha256",
            ),
            config_collector_receipt_sha256=required(
                arguments.config_collector_receipt_sha256,
                "--config-collector-receipt-sha256",
            ),
        )
    else:
        result = build_and_stage_final_activation_plan(
            native_observation_receipt_sha256=required(
                arguments.native_observation_receipt_sha256,
                "--native-observation-receipt-sha256",
            ),
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


ActivationPlan = PackagedActivationPlan
ActivationDigests = PackagedActivationDigests
ActivationPaths = PackagedActivationPaths


__all__ = [
    "ACTIVATION_PLAN_SCHEMA",
    "ActivationDigests",
    "ActivationPaths",
    "ActivationPlan",
    "ExternalNativeExecutableMapping",
    "HostNumericIdentities",
    "NativeObservationDigests",
    "PreapprovedNativeExecutablePolicy",
    "ROOT_COLLECTOR_MODULE",
    "build_activation_plan",
    "build_and_stage_final_activation_plan",
    "build_and_stage_native_observation_plan",
    "build_native_observation_plan",
    "load_release_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
