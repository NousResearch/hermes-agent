"""Release-sealed systemd artifacts for the Canonical alias projection rail.

This module renders bytes only.  It never installs, enables, starts, or stops
anything and accepts no credential value.  The privileged writer export and
the credential-free projector are separate one-shot identities; the timer
triggers the projector, whose exact systemd dependency runs the writer export
first.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scripts.canonical_brain_alias_projector import (
    PRODUCTION_PUBLIC_PROJECTION_PATH,
    PRODUCTION_RUN_RECEIPT_PATH,
    PRODUCTION_WRITER_EXPORT_PATH,
)


PACKAGE_SCHEMA = "muncho-production-alias-projection-package.v1"
PACKAGE_RELATIVE_ROOT = Path("ops/muncho/alias-projection/artifacts")
EXPORTER_UNIT = "muncho-canonical-writer-export.service"
PROJECTOR_UNIT = "muncho-canonical-alias-projector.service"
PROJECTOR_TIMER = "muncho-canonical-alias-projector.timer"

SYSTEMD_ROOT = Path("/etc/systemd/system")
WRITER_CONFIG_PATH = Path("/etc/muncho-canonical-writer/writer.json")
WRITER_CREDENTIAL_PATH = Path(
    "/etc/muncho/credentials/canonical-writer-db-password"
)
PRIVATE_EXPORT_DIRECTORY = PRODUCTION_WRITER_EXPORT_PATH.parent
PROJECTOR_ROOT = PRODUCTION_PUBLIC_PROJECTION_PATH.parent.parent
PUBLIC_PROJECTION_DIRECTORY = PRODUCTION_PUBLIC_PROJECTION_PATH.parent
PRODUCTION_RELEASES = Path(
    "/opt/adventico-ai-platform/hermes-agent-releases"
)
PRODUCTION_HOME = Path("/opt/adventico-ai-platform/hermes-home")

WRITER_MODULE = "gateway.canonical_writer_bootstrap"
PROJECTOR_MODULE = "scripts.canonical_brain_alias_projector"
WRITER_MODULE_RELATIVE = Path("gateway/canonical_writer_bootstrap.py")
PROJECTOR_MODULE_RELATIVE = Path("scripts/canonical_brain_alias_projector.py")
PROJECTION_READER_RELATIVE = Path("gateway/support_ops_alias_projection.py")
TEAM_REGISTRY_RELATIVE = Path("gateway/support_ops_team_registry.py")
CUTOVER_RUNTIME_RELATIVE = Path("gateway/production_alias_projection_cutover.py")
CUTOVER_ENTRYPOINT_RELATIVE = Path(
    "scripts/canary/production_alias_projection_cutover_entrypoint.py"
)

EXPORT_INTERVAL_SECONDS = 300
EXPORT_BOOT_DELAY_SECONDS = 60
EXPORT_RANDOMIZED_DELAY_SECONDS = 15
EXPORT_LIMIT = 1_000_000

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[a-z_][a-z0-9_-]{0,63}$")


class ProductionAliasProjectionUnitError(ValueError):
    """A release-sealed alias projection unit contract is invalid."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProductionAliasProjectionUnitError(
            "alias_projection_package_json_invalid"
        ) from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProductionAliasProjectionUnitError(f"{label}_invalid")
    return value


def _principal(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTITY.fullmatch(value) is None:
        raise ProductionAliasProjectionUnitError(f"{label}_invalid")
    return value


def _positive_id(value: Any, label: str) -> int:
    if type(value) is not int or not 1 <= value < 1 << 31:
        raise ProductionAliasProjectionUnitError(f"{label}_invalid")
    return value


def _unit(lines: list[str]) -> bytes:
    payload = ("\n".join(lines) + "\n").encode("utf-8", errors="strict")
    if b"\x00" in payload or not payload.endswith(b"\n"):
        raise ProductionAliasProjectionUnitError(
            "alias_projection_unit_encoding_invalid"
        )
    return payload


def _fixed_environment(*, home: Path) -> tuple[str, ...]:
    return (
        "Environment=LANG=C.UTF-8",
        "Environment=LC_ALL=C.UTF-8",
        "Environment=PATH=/usr/bin:/bin",
        f"Environment=HOME={home}",
        "Environment=PYTHONNOUSERSITE=1",
        "UnsetEnvironment=PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT",
        (
            "UnsetEnvironment=PGPASSWORD PGSERVICE PGSERVICEFILE PGHOST "
            "PGPORT PGDATABASE PGUSER"
        ),
    )


def _hardening() -> tuple[str, ...]:
    return (
        "NoNewPrivileges=yes",
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
        "LockPersonality=yes",
        "MemoryDenyWriteExecute=yes",
        "RestrictNamespaces=yes",
        "RestrictRealtime=yes",
        "RestrictSUIDSGID=yes",
        "SystemCallArchitectures=native",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "UMask=0027",
    )


@dataclass(frozen=True)
class ProductionAliasProjectionUnitBundle:
    revision: str
    release_root: Path
    interpreter: Path
    writer_uid: int
    writer_gid: int
    projector_uid: int
    projector_gid: int
    gateway_uid: int
    gateway_gid: int
    exporter_service: bytes
    projector_service: bytes
    projector_timer: bytes
    module_digests: Mapping[str, str]
    interpreter_sha256: str
    package_sha256: str

    def unit_payloads(self) -> Mapping[str, bytes]:
        return {
            EXPORTER_UNIT: self.exporter_service,
            PROJECTOR_UNIT: self.projector_service,
            PROJECTOR_TIMER: self.projector_timer,
        }

    def unit_digests(self) -> Mapping[str, str]:
        return {
            name: _sha256(payload) for name, payload in self.unit_payloads().items()
        }

    def _manifest_unsigned(self) -> dict[str, Any]:
        return {
            "schema": PACKAGE_SCHEMA,
            "release_revision": self.revision,
            "release_root": str(self.release_root),
            "interpreter": {
                "path": str(self.interpreter),
                "sha256": self.interpreter_sha256,
            },
            "modules": {
                name: {
                    "path": str(self.release_root / relative),
                    "sha256": self.module_digests[name],
                }
                for name, relative in {
                    "writer_bootstrap": WRITER_MODULE_RELATIVE,
                    "alias_projector": PROJECTOR_MODULE_RELATIVE,
                    "projection_reader": PROJECTION_READER_RELATIVE,
                    "team_registry": TEAM_REGISTRY_RELATIVE,
                    "cutover_runtime": CUTOVER_RUNTIME_RELATIVE,
                    "cutover_entrypoint": CUTOVER_ENTRYPOINT_RELATIVE,
                }.items()
            },
            "units": {
                name: {
                    "path": str(SYSTEMD_ROOT / name),
                    "artifact_path": str(
                        self.release_root / PACKAGE_RELATIVE_ROOT / name
                    ),
                    "sha256": digest,
                    "uid": 0,
                    "gid": 0,
                    "mode": "0644",
                }
                for name, digest in self.unit_digests().items()
            },
            "identities": {
                "writer": {
                    "user": "muncho-canonical-writer",
                    "group": "muncho-canonical-writer",
                    "uid": self.writer_uid,
                    "gid": self.writer_gid,
                },
                "projector": {
                    "user": "muncho-projector",
                    "group": "muncho-projector",
                    "uid": self.projector_uid,
                    "gid": self.projector_gid,
                },
                "gateway": {
                    "user": "ai-platform-brain",
                    "group": "ai-platform-brain",
                    "uid": self.gateway_uid,
                    "gid": self.gateway_gid,
                },
            },
            "directories": {
                str(PRIVATE_EXPORT_DIRECTORY): {
                    "uid": self.writer_uid,
                    "gid": self.projector_gid,
                    "mode": "0750",
                },
                str(PROJECTOR_ROOT): {"uid": 0, "gid": 0, "mode": "0751"},
                str(PUBLIC_PROJECTION_DIRECTORY): {
                    "uid": self.projector_uid,
                    "gid": self.gateway_gid,
                    "mode": "2750",
                },
            },
            "files": {
                "writer_export": {
                    "path": str(PRODUCTION_WRITER_EXPORT_PATH),
                    "uid": self.writer_uid,
                    "gid": self.projector_gid,
                    "mode": "0640",
                    "created_by": EXPORTER_UNIT,
                },
                "public_projection": {
                    "path": str(PRODUCTION_PUBLIC_PROJECTION_PATH),
                    "uid": self.projector_uid,
                    "gid": self.gateway_gid,
                    "mode": "0640",
                    "created_by": PROJECTOR_UNIT,
                },
                "public_run_receipt": {
                    "path": str(PRODUCTION_RUN_RECEIPT_PATH),
                    "uid": self.projector_uid,
                    "gid": self.gateway_gid,
                    "mode": "0640",
                    "created_by": PROJECTOR_UNIT,
                },
            },
            "ordering": {
                "timer_triggers": PROJECTOR_UNIT,
                "projector_requires": EXPORTER_UNIT,
                "exporter_before_projector": True,
                "timer_enabled_before_activation": False,
                "interval_seconds": EXPORT_INTERVAL_SECONDS,
            },
            "credential_boundary": {
                "writer_credential_path": str(WRITER_CREDENTIAL_PATH),
                "projector_credential_paths": [],
                "gateway_credential_paths": [],
                "projector_network_private": True,
            },
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }

    def manifest(self) -> Mapping[str, Any]:
        unsigned = self._manifest_unsigned()
        package_sha256 = _sha256(_canonical(unsigned))
        if package_sha256 != self.package_sha256:
            raise ProductionAliasProjectionUnitError(
                "alias_projection_package_self_digest_invalid"
            )
        return {**unsigned, "package_sha256": package_sha256}


def render_production_alias_projection_units(
    *,
    revision: str,
    database_ip: str,
    writer_user: str,
    writer_group: str,
    writer_uid: int,
    writer_gid: int,
    projector_user: str,
    projector_group: str,
    projector_uid: int,
    projector_gid: int,
    gateway_user: str,
    gateway_group: str,
    gateway_uid: int,
    gateway_gid: int,
    interpreter_sha256: str,
    writer_module_sha256: str,
    projector_module_sha256: str,
    projection_reader_sha256: str,
    team_registry_sha256: str,
    cutover_runtime_sha256: str,
    cutover_entrypoint_sha256: str,
) -> ProductionAliasProjectionUnitBundle:
    """Render the exact disabled-until-authorized recurring projection rail."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ProductionAliasProjectionUnitError("alias_projection_revision_invalid")
    try:
        database = ipaddress.ip_address(database_ip)
    except ValueError as exc:
        raise ProductionAliasProjectionUnitError(
            "alias_projection_database_ip_invalid"
        ) from exc
    if (
        not isinstance(database, ipaddress.IPv4Address)
        or database.is_unspecified
        or database.is_loopback
        or database.is_link_local
        or database.is_multicast
    ):
        raise ProductionAliasProjectionUnitError(
            "alias_projection_database_ip_invalid"
        )
    names = (
        _principal(writer_user, "writer_user"),
        _principal(writer_group, "writer_group"),
        _principal(projector_user, "projector_user"),
        _principal(projector_group, "projector_group"),
        _principal(gateway_user, "gateway_user"),
        _principal(gateway_group, "gateway_group"),
    )
    ids = (
        _positive_id(writer_uid, "writer_uid"),
        _positive_id(writer_gid, "writer_gid"),
        _positive_id(projector_uid, "projector_uid"),
        _positive_id(projector_gid, "projector_gid"),
        _positive_id(gateway_uid, "gateway_uid"),
        _positive_id(gateway_gid, "gateway_gid"),
    )
    if names != (
        "muncho-canonical-writer",
        "muncho-canonical-writer",
        "muncho-projector",
        "muncho-projector",
        "ai-platform-brain",
        "ai-platform-brain",
    ) or len({writer_uid, projector_uid, gateway_uid}) != 3 or len(
        {writer_gid, projector_gid, gateway_gid}
    ) != 3:
        raise ProductionAliasProjectionUnitError(
            "alias_projection_identity_invalid"
        )
    interpreter_digest = _digest(interpreter_sha256, "interpreter_sha256")
    modules = {
        "writer_bootstrap": _digest(
            writer_module_sha256, "writer_module_sha256"
        ),
        "alias_projector": _digest(
            projector_module_sha256, "projector_module_sha256"
        ),
        "projection_reader": _digest(
            projection_reader_sha256, "projection_reader_sha256"
        ),
        "team_registry": _digest(
            team_registry_sha256, "team_registry_sha256"
        ),
        "cutover_runtime": _digest(
            cutover_runtime_sha256, "cutover_runtime_sha256"
        ),
        "cutover_entrypoint": _digest(
            cutover_entrypoint_sha256, "cutover_entrypoint_sha256"
        ),
    }
    release = PRODUCTION_RELEASES / f"hermes-agent-{revision[:12]}"
    interpreter = release / ".venv/bin/python"

    exporter = _unit(
        [
            "# Privileged exact Canonical event export for alias projection.",
            f"# ReleaseRevision={revision}",
            f"# InterpreterSHA256={interpreter_digest}",
            f"# ModuleSHA256={modules['writer_bootstrap']}",
            "[Unit]",
            "Description=Muncho Canonical alias event export",
            "After=network-online.target muncho-canonical-writer.service",
            "Wants=network-online.target",
            f"Before={PROJECTOR_UNIT}",
            f"AssertPathExists={interpreter}",
            f"AssertPathExists={release / WRITER_MODULE_RELATIVE}",
            f"AssertPathExists={WRITER_CONFIG_PATH}",
            f"AssertPathExists={WRITER_CREDENTIAL_PATH}",
            f"AssertPathIsDirectory={PRIVATE_EXPORT_DIRECTORY}",
            "",
            "[Service]",
            "Type=oneshot",
            f"User={writer_user}",
            f"Group={writer_group}",
            f"SupplementaryGroups={projector_group}",
            f"WorkingDirectory={release}",
            (
                f"ExecStart={interpreter} -B -I -m {WRITER_MODULE} "
                f"--config {WRITER_CONFIG_PATH} --export-events "
                f"{PRODUCTION_WRITER_EXPORT_PATH} --export-limit {EXPORT_LIMIT}"
            ),
            "TimeoutStartSec=300s",
            "TimeoutStopSec=30s",
            "KillMode=mixed",
            "LimitCORE=0",
            *_fixed_environment(home=Path("/nonexistent")),
            *_hardening(),
            "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
            "IPAddressDeny=any",
            f"IPAddressAllow={database}/32",
            f"BindReadOnlyPaths={release}",
            f"ReadOnlyPaths={WRITER_CONFIG_PATH}",
            f"ReadOnlyPaths={WRITER_CREDENTIAL_PATH}",
            f"ReadWritePaths={PRIVATE_EXPORT_DIRECTORY}",
            f"InaccessiblePaths={PUBLIC_PROJECTION_DIRECTORY}",
            "StandardOutput=journal",
            "StandardError=journal",
        ]
    )
    projector = _unit(
        [
            "# Credential-free exact person.alias.learned/channel.alias.learned projector.",
            f"# ReleaseRevision={revision}",
            f"# InterpreterSHA256={interpreter_digest}",
            f"# ModuleSHA256={modules['alias_projector']}",
            f"# ProjectionReaderSHA256={modules['projection_reader']}",
            f"# TeamRegistrySHA256={modules['team_registry']}",
            "[Unit]",
            "Description=Muncho Canonical team alias projector",
            f"Requires={EXPORTER_UNIT}",
            f"After={EXPORTER_UNIT}",
            f"AssertPathExists={interpreter}",
            f"AssertPathExists={release / PROJECTOR_MODULE_RELATIVE}",
            f"AssertPathExists={PRODUCTION_WRITER_EXPORT_PATH}",
            f"AssertPathIsDirectory={PUBLIC_PROJECTION_DIRECTORY}",
            "",
            "[Service]",
            "Type=oneshot",
            f"User={projector_user}",
            f"Group={projector_group}",
            f"WorkingDirectory={release}",
            (
                f"ExecStart={interpreter} -B -I -m {PROJECTOR_MODULE} "
                f"--events-json {PRODUCTION_WRITER_EXPORT_PATH} "
                f"--output-json {PRODUCTION_PUBLIC_PROJECTION_PATH} "
                f"--receipt-json {PRODUCTION_RUN_RECEIPT_PATH} "
                "--production-recurring"
            ),
            "TimeoutStartSec=120s",
            "TimeoutStopSec=30s",
            "KillMode=mixed",
            "LimitCORE=0",
            *_fixed_environment(home=Path("/nonexistent")),
            *_hardening(),
            "PrivateNetwork=yes",
            "RestrictAddressFamilies=AF_UNIX",
            f"BindReadOnlyPaths={release}",
            f"ReadOnlyPaths={PRODUCTION_WRITER_EXPORT_PATH}",
            f"ReadWritePaths={PUBLIC_PROJECTION_DIRECTORY}",
            "InaccessiblePaths=/run/credentials",
            "InaccessiblePaths=/etc/muncho",
            f"InaccessiblePaths={PRODUCTION_HOME}",
            "StandardOutput=journal",
            "StandardError=journal",
        ]
    )
    timer = _unit(
        [
            "# Disabled until digest-bound production cutover activation.",
            f"# ReleaseRevision={revision}",
            "[Unit]",
            "Description=Muncho recurring Canonical team alias projection",
            "",
            "[Timer]",
            f"Unit={PROJECTOR_UNIT}",
            f"OnBootSec={EXPORT_BOOT_DELAY_SECONDS}s",
            f"OnUnitInactiveSec={EXPORT_INTERVAL_SECONDS}s",
            f"RandomizedDelaySec={EXPORT_RANDOMIZED_DELAY_SECONDS}s",
            "AccuracySec=1s",
            "Persistent=true",
            "",
            "[Install]",
            "WantedBy=timers.target",
        ]
    )
    forbidden = re.compile(
        rb"(?im)^(?:EnvironmentFile|PassEnvironment|LoadCredential)="
    )
    if (
        forbidden.search(exporter)
        or forbidden.search(projector)
        or forbidden.search(timer)
        or b"canonical-writer-db-password" in projector
        or b"PGPASSWORD" not in projector
        or b"PrivateNetwork=yes" not in projector
        or b"[Install]" in exporter
        or b"[Install]" in projector
    ):
        raise ProductionAliasProjectionUnitError(
            "alias_projection_unit_boundary_invalid"
        )
    provisional = ProductionAliasProjectionUnitBundle(
        revision=revision,
        release_root=release,
        interpreter=interpreter,
        writer_uid=ids[0],
        writer_gid=ids[1],
        projector_uid=ids[2],
        projector_gid=ids[3],
        gateway_uid=ids[4],
        gateway_gid=ids[5],
        exporter_service=exporter,
        projector_service=projector,
        projector_timer=timer,
        module_digests=modules,
        interpreter_sha256=interpreter_digest,
        package_sha256="0" * 64,
    )
    return ProductionAliasProjectionUnitBundle(
        **{
            **provisional.__dict__,
            "package_sha256": _sha256(
                _canonical(provisional._manifest_unsigned())
            ),
        }
    )


def validate_package_manifest(
    value: Any,
    *,
    expected_revision: str | None = None,
    expected_package_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the exact self-digesting release package envelope."""

    fields = {
        "schema",
        "release_revision",
        "release_root",
        "interpreter",
        "modules",
        "units",
        "identities",
        "directories",
        "files",
        "ordering",
        "credential_boundary",
        "secret_material_recorded",
        "secret_digest_recorded",
        "package_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ProductionAliasProjectionUnitError(
            "alias_projection_package_envelope_invalid"
        )
    result = dict(value)
    revision = result.get("release_revision")
    digest = result.get("package_sha256")
    if (
        result.get("schema") != PACKAGE_SCHEMA
        or not isinstance(revision, str)
        or _REVISION.fullmatch(revision) is None
        or (expected_revision is not None and revision != expected_revision)
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or (
            expected_package_sha256 is not None
            and digest != expected_package_sha256
        )
        or result.get("release_root")
        != str(PRODUCTION_RELEASES / f"hermes-agent-{revision[:12]}")
        or result.get("secret_material_recorded") is not False
        or result.get("secret_digest_recorded") is not False
        or _sha256(
            _canonical(
                {
                    key: item
                    for key, item in result.items()
                    if key != "package_sha256"
                }
            )
        )
        != digest
    ):
        raise ProductionAliasProjectionUnitError(
            "alias_projection_package_identity_invalid"
        )
    return result


def _filesystem_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
    )


def _stable_release_marker(path: Path) -> bytes:
    try:
        path_before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ProductionAliasProjectionUnitError(
            "alias_projection_package_release_invalid"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            _filesystem_identity(path_before) != _filesystem_identity(opened)
            or stat.S_ISLNK(path_before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > 256
            or stat.S_IMODE(opened.st_mode) & 0o022
        ):
            raise ProductionAliasProjectionUnitError(
                "alias_projection_package_release_invalid"
            )
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise ProductionAliasProjectionUnitError(
            "alias_projection_package_release_invalid"
        ) from exc
    payload = b"".join(chunks)
    if (
        len(payload) != opened.st_size
        or _filesystem_identity(opened) != _filesystem_identity(after)
        or _filesystem_identity(opened) != _filesystem_identity(path_after)
    ):
        raise ProductionAliasProjectionUnitError(
            "alias_projection_package_release_invalid"
        )
    return payload


def _ensure_package_directory(release: Path) -> Path:
    current = release
    for component in PACKAGE_RELATIVE_ROOT.parts:
        current = current / component
        try:
            item = current.lstat()
        except FileNotFoundError:
            os.mkdir(current, mode=0o755)
            item = current.lstat()
        if (
            stat.S_ISLNK(item.st_mode)
            or not stat.S_ISDIR(item.st_mode)
            or item.st_uid != os.geteuid()
            or stat.S_IMODE(item.st_mode) & 0o022
        ):
            raise ProductionAliasProjectionUnitError(
                "alias_projection_package_directory_untrusted"
            )
    return current


def _atomic_install(path: Path, payload: bytes, *, mode: int) -> None:
    parent_before = path.parent.lstat()
    if (
        stat.S_ISLNK(parent_before.st_mode)
        or not stat.S_ISDIR(parent_before.st_mode)
        or parent_before.st_uid != os.geteuid()
        or stat.S_IMODE(parent_before.st_mode) & 0o022
    ):
        raise ProductionAliasProjectionUnitError(
            "alias_projection_package_directory_untrusted"
        )
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary_name = f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        if _directory_identity(os.fstat(directory_fd)) != _directory_identity(
            parent_before
        ):
            raise ProductionAliasProjectionUnitError(
                "alias_projection_package_directory_changed"
            )
        try:
            existing = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode)
            or existing.st_nlink != 1
            or existing.st_uid != os.geteuid()
        ):
            raise ProductionAliasProjectionUnitError(
                "alias_projection_package_target_untrusted"
            )
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ProductionAliasProjectionUnitError(
                    "alias_projection_package_write_failed"
                )
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
        path_after = path.parent.lstat()
        if _directory_identity(path_after) != _directory_identity(parent_before):
            raise ProductionAliasProjectionUnitError(
                "alias_projection_package_directory_changed"
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def write_release_alias_projection_package(
    release_root: Path,
    bundle: ProductionAliasProjectionUnitBundle,
) -> Mapping[str, Any]:
    """Write exact unit payloads and a canonical manifest into one release."""

    release = Path(release_root)
    if (
        not release.is_absolute()
        or release.is_symlink()
        or not release.is_dir()
        or release.name != f"hermes-agent-{bundle.revision[:12]}"
    ):
        raise ProductionAliasProjectionUnitError(
            "alias_projection_package_release_invalid"
        )
    marker = release / ".codex-source-commit"
    marker_bytes = _stable_release_marker(marker)
    if marker_bytes != (bundle.revision + "\n").encode("ascii"):
        raise ProductionAliasProjectionUnitError(
            "alias_projection_package_release_invalid"
        )
    output = _ensure_package_directory(release)
    for name, payload in bundle.unit_payloads().items():
        _atomic_install(output / name, payload, mode=0o444)
    manifest = bundle.manifest()
    _atomic_install(
        output / "manifest.json",
        _canonical(manifest) + b"\n",
        mode=0o444,
    )
    return manifest


__all__ = [
    "EXPORTER_UNIT",
    "PACKAGE_SCHEMA",
    "PACKAGE_RELATIVE_ROOT",
    "PROJECTOR_TIMER",
    "PROJECTOR_UNIT",
    "ProductionAliasProjectionUnitBundle",
    "ProductionAliasProjectionUnitError",
    "render_production_alias_projection_units",
    "validate_package_manifest",
    "write_release_alias_projection_package",
]
