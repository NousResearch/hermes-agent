"""Pure sealed-release and systemd contracts for the writer-only canary.

This module is packaged with the runtime wheel because both source-side release
construction and sealed planning must render exactly the same mechanical unit
bytes.  It performs no filesystem writes, subprocess execution, service
mutation, approval, routing, or secret handling.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RELEASE_SCHEMA = "muncho-writer-only-release.v1"
UNIT_BUNDLE_SCHEMA = "muncho-writer-only-systemd-bundle.v2"
RELEASE_MANIFEST_NAME = "release-manifest.json"
INCOMPLETE_MARKER_NAME = ".release-build-incomplete"
WRITER_MODULE = "gateway.canonical_writer_bootstrap"
GATEWAY_MODULE = "gateway.canonical_writer_gateway_bootstrap"
DEFAULT_RELEASE_BASE = Path("/opt/muncho-canary-releases")
WRITER_UNIT_NAME = "muncho-canonical-writer.service"
GATEWAY_UNIT_NAME = "hermes-cloud-gateway.service"
EXPORTER_UNIT_NAME = "muncho-canonical-writer-export.service"
DEFAULT_EXPORT_LIMIT = 200_000
TMPFILES_NAME = "muncho-canonical-writer.conf"

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PYTHON_RE = re.compile(r"^3\.(?:11|12|13)\.[0-9]+$")
_IDENTITY_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,63}$")
_SAFE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]+$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


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
        raise ValueError("release value is not canonical JSON") from exc
    return encoded.encode("utf-8", errors="strict")


def _absolute_normalized_path(value: str | os.PathLike[str], label: str) -> Path:
    raw = os.fspath(value)
    path = Path(raw)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(path) != raw
        or _CONTROL_RE.search(raw) is not None
        or _SAFE_PATH_RE.fullmatch(raw) is None
    ):
        raise ValueError(f"{label} must be an absolute normalized safe path")
    return path


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _identity(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTITY_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


@dataclass(frozen=True)
class TreeEntry:
    path: str
    kind: str
    mode: str
    size: int = 0
    sha256: str = ""
    target: str = ""

    def to_mapping(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "path": self.path,
            "kind": self.kind,
            "mode": self.mode,
        }
        if self.kind == "file":
            value.update({"size": self.size, "sha256": self.sha256})
        elif self.kind == "symlink":
            value["target"] = self.target
        return value


@dataclass(frozen=True)
class ReleaseManifest:
    revision: str
    artifact_root: str
    python_version: str
    interpreter: str
    writer_module_origin: str
    gateway_module_origin: str
    entries: tuple[TreeEntry, ...]
    artifact_sha256: str
    schema: str = RELEASE_SCHEMA
    writer_module: str = WRITER_MODULE
    gateway_module: str = GATEWAY_MODULE

    def unsigned_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "revision": self.revision,
            "artifact_root": self.artifact_root,
            "python_version": self.python_version,
            "interpreter": self.interpreter,
            "writer_module": self.writer_module,
            "writer_module_origin": self.writer_module_origin,
            "gateway_module": self.gateway_module,
            "gateway_module_origin": self.gateway_module_origin,
            "entries": [entry.to_mapping() for entry in self.entries],
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self.unsigned_mapping(), "artifact_sha256": self.artifact_sha256}

    @property
    def computed_artifact_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.unsigned_mapping())).hexdigest()


@dataclass(frozen=True)
class WriterOnlyUnitSpec:
    writer_user: str = "muncho-canonical-writer"
    writer_group: str = "muncho-canonical-writer"
    gateway_user: str = "muncho-gateway"
    gateway_group: str = "muncho-gateway"
    socket_client_group: str = "muncho-writer-client"
    projector_group: str = "muncho-projector"
    writer_config: Path = Path("/etc/muncho-canonical-writer/writer.json")
    gateway_config: Path = Path("/etc/hermes/config.yaml")
    gateway_home: Path = Path("/var/lib/hermes-gateway")
    writer_runtime: Path = Path("/run/muncho-canonical-writer")
    projection_directory: Path = Path("/var/lib/muncho-canonical-writer/projection")
    gateway_runtime: Path = Path("/run/hermes-cloud-gateway")
    gateway_state: Path = Path("/var/lib/hermes-gateway")
    gateway_logs: Path = Path("/var/log/hermes-gateway")
    database_ip_allow: tuple[str, ...] = ()

    def validate(self) -> None:
        identities = (
            _identity(self.writer_user, "writer user"),
            _identity(self.writer_group, "writer group"),
            _identity(self.gateway_user, "gateway user"),
            _identity(self.gateway_group, "gateway group"),
            _identity(self.socket_client_group, "socket client group"),
            _identity(self.projector_group, "projector group"),
        )
        if (
            self.writer_user == self.gateway_user
            or len({
                self.writer_group,
                self.gateway_group,
                self.socket_client_group,
                self.projector_group,
            })
            != 4
        ):
            raise ValueError("writer-only runtime identities must be distinct")
        paths = (
            self.writer_config,
            self.gateway_config,
            self.writer_runtime,
            self.projection_directory,
            self.gateway_runtime,
            self.gateway_state,
            self.gateway_logs,
        )
        normalized = tuple(
            _absolute_normalized_path(path, "unit path") for path in paths
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("writer-only unit paths must be distinct")
        if self.writer_runtime != Path("/run/muncho-canonical-writer"):
            raise ValueError("writer runtime path is protocol-pinned")
        if (
            self.gateway_config != Path("/etc/hermes/config.yaml")
            or self.gateway_home != Path("/var/lib/hermes-gateway")
            or self.gateway_state != self.gateway_home
        ):
            raise ValueError("gateway managed config and passwd home are pinned")
        if len(self.database_ip_allow) != 1:
            raise ValueError("writer-only release requires one exact database IP")
        try:
            network = ipaddress.ip_network(self.database_ip_allow[0], strict=True)
        except ValueError as exc:
            raise ValueError("writer database IP allow-list is invalid") from exc
        if network.num_addresses != 1 or str(network) != self.database_ip_allow[0]:
            raise ValueError("writer database IP allow-list must be an exact host")


@dataclass(frozen=True)
class SystemdUnitBundle:
    writer_service: str
    gateway_service: str
    exporter_service: str
    tmpfiles: str
    contract: tuple[tuple[str, str], ...]
    sha256: str
    schema: str = UNIT_BUNDLE_SCHEMA

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "writer_service": self.writer_service,
            "gateway_service": self.gateway_service,
            "exporter_service": self.exporter_service,
            "tmpfiles": self.tmpfiles,
            "contract": dict(self.contract),
            "sha256": self.sha256,
        }


def _common_hardening(*, address_families: str) -> list[str]:
    return [
        "NoNewPrivileges=yes",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "LockPersonality=yes",
        "MemoryDenyWriteExecute=yes",
        "PrivateDevices=yes",
        "PrivateTmp=yes",
        "ProtectClock=yes",
        "ProtectControlGroups=yes",
        "ProtectHome=yes",
        "ProtectHostname=yes",
        "ProtectKernelLogs=yes",
        "ProtectKernelModules=yes",
        "ProtectKernelTunables=yes",
        "ProtectProc=invisible",
        "ProtectSystem=strict",
        "ProcSubset=pid",
        "RemoveIPC=yes",
        f"RestrictAddressFamilies={address_families}",
        "RestrictNamespaces=yes",
        "RestrictRealtime=yes",
        "RestrictSUIDSGID=yes",
        "SystemCallArchitectures=native",
        "UMask=0077",
    ]


def _fixed_service_environment(*, user: str, home: str) -> list[str]:
    return [
        f"Environment=HOME={home}",
        "Environment=LANG=C.UTF-8",
        "Environment=LC_ALL=C.UTF-8",
        f"Environment=LOGNAME={user}",
        "Environment=PATH=/usr/bin:/bin",
        "Environment=SHELL=/usr/sbin/nologin",
        "Environment=TZ=UTC",
        f"Environment=USER={user}",
    ]


def render_systemd_units(
    manifest: ReleaseManifest,
    spec: WriterOnlyUnitSpec,
) -> SystemdUnitBundle:
    spec.validate()
    if manifest.schema != RELEASE_SCHEMA:
        raise ValueError("release manifest schema is invalid")
    if _REVISION_RE.fullmatch(manifest.revision) is None:
        raise ValueError("release manifest revision is invalid")
    if _SHA256_RE.fullmatch(manifest.artifact_sha256) is None:
        raise ValueError("release artifact digest is invalid")
    if manifest.artifact_sha256 != manifest.computed_artifact_sha256:
        raise ValueError("release artifact digest does not match manifest")
    if _PYTHON_RE.fullmatch(manifest.python_version) is None:
        raise ValueError("release Python identity is invalid")
    release_root = _absolute_normalized_path(
        manifest.artifact_root,
        "manifest artifact root",
    )
    if (
        release_root.parent != DEFAULT_RELEASE_BASE
        or release_root.name != manifest.revision
    ):
        raise ValueError("release path is not revision-addressed")
    interpreter = _absolute_normalized_path(
        manifest.interpreter,
        "manifest interpreter",
    )
    writer_origin = _absolute_normalized_path(
        manifest.writer_module_origin,
        "manifest writer module origin",
    )
    gateway_origin = _absolute_normalized_path(
        manifest.gateway_module_origin,
        "manifest gateway module origin",
    )
    python_parts = manifest.python_version.split(".")
    expected_site_packages = (
        release_root
        / "venv"
        / "lib"
        / f"python{python_parts[0]}.{python_parts[1]}"
        / "site-packages"
    )
    if (
        manifest.writer_module != WRITER_MODULE
        or manifest.gateway_module != GATEWAY_MODULE
        or interpreter != release_root / "venv/bin/python"
        or writer_origin
        != expected_site_packages / "gateway/canonical_writer_bootstrap.py"
        or gateway_origin
        != expected_site_packages / "gateway/canonical_writer_gateway_bootstrap.py"
    ):
        raise ValueError("release module origins are not exact")

    writer_lines = [
        "# Generated from a digest-bound writer-only release; do not edit.",
        f"# ArtifactSHA256={manifest.artifact_sha256}",
        f"# ModuleOrigin={writer_origin}",
        "[Unit]",
        "Description=Muncho privileged Canonical Writer (isolated canary)",
        "After=network-online.target",
        "Wants=network-online.target",
        f"Before={GATEWAY_UNIT_NAME}",
        f"AssertPathIsDirectory={spec.writer_runtime}",
        f"AssertPathIsDirectory={spec.projection_directory}",
        f"AssertPathExists={spec.writer_config}",
        "",
        "[Service]",
        "Type=notify",
        "NotifyAccess=main",
        f"User={spec.writer_user}",
        f"Group={spec.writer_group}",
        f"SupplementaryGroups={spec.projector_group}",
        f"WorkingDirectory={release_root}",
        (
            f"ExecStart={interpreter} -B -I -m {WRITER_MODULE} "
            f"--config {spec.writer_config}"
        ),
        "Restart=on-failure",
        "RestartSec=5s",
        "TimeoutStartSec=60s",
        "TimeoutStopSec=30s",
        "KillMode=mixed",
        "LimitCORE=0",
        *_fixed_service_environment(user=spec.writer_user, home="/nonexistent"),
        *_common_hardening(address_families="AF_UNIX AF_INET AF_INET6"),
        "IPAddressDeny=any",
        f"IPAddressAllow={spec.database_ip_allow[0]}",
        f"BindReadOnlyPaths={release_root}",
        f"ReadOnlyPaths={spec.writer_config}",
        f"ReadWritePaths={spec.writer_runtime}",
        f"ReadWritePaths={spec.projection_directory}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    gateway_lines = [
        "# Generated from a digest-bound writer-only release; do not edit.",
        f"# ArtifactSHA256={manifest.artifact_sha256}",
        f"# ModuleOrigin={gateway_origin}",
        f"# PasswdHome={spec.gateway_home}",
        f"# ManagedConfig={spec.gateway_config}",
        "[Unit]",
        "Description=Muncho credential-free gateway (writer-only canary)",
        f"BindsTo={WRITER_UNIT_NAME}",
        f"After={WRITER_UNIT_NAME}",
        f"AssertPathIsDirectory={spec.gateway_home}",
        f"AssertPathExists={spec.gateway_config}",
        "",
        "[Service]",
        "Type=notify",
        "NotifyAccess=main",
        f"User={spec.gateway_user}",
        f"Group={spec.gateway_group}",
        f"SupplementaryGroups={spec.socket_client_group}",
        f"WorkingDirectory={release_root}",
        f"ExecStart={interpreter} -B -I -m {GATEWAY_MODULE}",
        "Restart=on-failure",
        "RestartSec=5s",
        "TimeoutStartSec=60s",
        "TimeoutStopSec=30s",
        "KillMode=mixed",
        "LimitCORE=0",
        *_fixed_service_environment(
            user=spec.gateway_user,
            home=str(spec.gateway_home),
        ),
        "PrivateNetwork=yes",
        *_common_hardening(address_families="AF_UNIX"),
        f"BindReadOnlyPaths={release_root}",
        f"ReadOnlyPaths={spec.gateway_config}",
        f"ReadWritePaths={spec.gateway_runtime}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    export_path = spec.projection_directory / "canonical-events.json"
    exporter_lines = [
        "# Temporary digest-bound projection export unit; remove after use.",
        f"# ArtifactSHA256={manifest.artifact_sha256}",
        f"# ModuleOrigin={writer_origin}",
        "[Unit]",
        "Description=Muncho Canonical projection export (isolated canary)",
        "After=network-online.target",
        "Wants=network-online.target",
        f"AssertPathIsDirectory={spec.projection_directory}",
        f"AssertPathExists={spec.writer_config}",
        "",
        "[Service]",
        "Type=oneshot",
        f"User={spec.writer_user}",
        f"Group={spec.writer_group}",
        f"SupplementaryGroups={spec.projector_group}",
        f"WorkingDirectory={release_root}",
        (
            f"ExecStart={interpreter} -B -I -m {WRITER_MODULE} "
            f"--config {spec.writer_config} --export-events {export_path} "
            f"--export-limit {DEFAULT_EXPORT_LIMIT}"
        ),
        "TimeoutStartSec=300s",
        "TimeoutStopSec=30s",
        "KillMode=mixed",
        "LimitCORE=0",
        *_fixed_service_environment(user=spec.writer_user, home="/nonexistent"),
        *_common_hardening(address_families="AF_UNIX AF_INET AF_INET6"),
        "IPAddressDeny=any",
        f"IPAddressAllow={spec.database_ip_allow[0]}",
        f"BindReadOnlyPaths={release_root}",
        f"ReadOnlyPaths={spec.writer_config}",
        f"ReadWritePaths={spec.projection_directory}",
        "StandardOutput=journal",
        "StandardError=journal",
    ]
    tmpfiles_lines = [
        "# type path mode user group age argument",
        (
            f"d {spec.writer_runtime} 2750 {spec.writer_user} "
            f"{spec.socket_client_group} - -"
        ),
        (
            f"d {spec.projection_directory} 0750 {spec.writer_user} "
            f"{spec.projector_group} - -"
        ),
        (
            f"d {spec.gateway_runtime} 0700 {spec.gateway_user} "
            f"{spec.gateway_group} - -"
        ),
        (
            f"d {spec.gateway_state} 0700 {spec.gateway_user} "
            f"{spec.gateway_group} - -"
        ),
        (
            f"d {spec.gateway_logs} 0700 {spec.gateway_user} "
            f"{spec.gateway_group} - -"
        ),
    ]
    writer = "\n".join(writer_lines) + "\n"
    gateway = "\n".join(gateway_lines) + "\n"
    exporter = "\n".join(exporter_lines) + "\n"
    tmpfiles = "\n".join(tmpfiles_lines) + "\n"
    forbidden = re.compile(
        r"(?im)^(?:EnvironmentFile|PassEnvironment|LoadCredential)="
    )
    if (
        forbidden.search(writer)
        or forbidden.search(gateway)
        or forbidden.search(exporter)
    ):
        raise RuntimeError("writer-only units cannot inject environment or credentials")
    payload = {
        "schema": UNIT_BUNDLE_SCHEMA,
        "writer_service": writer,
        "gateway_service": gateway,
        "exporter_service": exporter,
        "tmpfiles": tmpfiles,
        "contract": {
            "revision": manifest.revision,
            "artifact_sha256": manifest.artifact_sha256,
            "working_directory": str(release_root),
            "writer_user": spec.writer_user,
            "writer_group": spec.writer_group,
            "gateway_user": spec.gateway_user,
            "gateway_group": spec.gateway_group,
            "gateway_passwd_home": str(spec.gateway_home),
            "gateway_config": str(spec.gateway_config),
            "socket_client_group": spec.socket_client_group,
            "writer_runtime": str(spec.writer_runtime),
            "writer_runtime_mode": "2750",
            "database_ip_allow": spec.database_ip_allow[0],
            "exporter_unit": EXPORTER_UNIT_NAME,
            "projection_export_path": str(export_path),
            "projection_export_limit": str(DEFAULT_EXPORT_LIMIT),
        },
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return SystemdUnitBundle(
        writer_service=writer,
        gateway_service=gateway,
        exporter_service=exporter,
        tmpfiles=tmpfiles,
        contract=tuple(sorted(payload["contract"].items())),
        sha256=digest,
    )


__all__ = [
    "DEFAULT_EXPORT_LIMIT",
    "DEFAULT_RELEASE_BASE",
    "EXPORTER_UNIT_NAME",
    "GATEWAY_MODULE",
    "GATEWAY_UNIT_NAME",
    "INCOMPLETE_MARKER_NAME",
    "RELEASE_MANIFEST_NAME",
    "RELEASE_SCHEMA",
    "ReleaseManifest",
    "SystemdUnitBundle",
    "TMPFILES_NAME",
    "TreeEntry",
    "UNIT_BUNDLE_SCHEMA",
    "WRITER_MODULE",
    "WRITER_UNIT_NAME",
    "WriterOnlyUnitSpec",
    "render_systemd_units",
]
