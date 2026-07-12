#!/usr/bin/env python3
"""Plan the digest-bound activation of one writer-only canary release.

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

import hashlib
import ipaddress
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from gateway.canonical_writer_bootstrap import CanonicalWriterServiceConfig
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
from scripts.canary.writer_release import (
    GATEWAY_MODULE,
    TMPFILES_NAME,
    WRITER_MODULE,
    ReleaseManifest,
    SystemdUnitBundle,
    TreeEntry,
    WriterOnlyUnitSpec,
    render_systemd_units,
)


ACTIVATION_PLAN_SCHEMA = "muncho-writer-only-activation-plan.v1"
DEFAULT_MANIFEST_PATH = Path(
    "/etc/muncho/writer-activation/deployment-manifest.json"
)
DEFAULT_PLAN_PATH = Path("/etc/muncho/writer-activation/activation-plan.json")
DEFAULT_RECEIPT_PATH = Path(
    "/run/muncho-canonical-preflight/root-preflight.json"
)
DEFAULT_WRITER_UNIT_PATH = Path(
    "/etc/systemd/system/muncho-canonical-writer.service"
)
DEFAULT_GATEWAY_UNIT_PATH = Path(
    "/etc/systemd/system/hermes-cloud-gateway.service"
)
DEFAULT_TMPFILES_PATH = Path("/etc/tmpfiles.d") / TMPFILES_NAME
DEFAULT_PROJECTION_FILENAME = "canonical-events.json"
ROOT_COLLECTOR_MODULE = "gateway.canonical_writer_root_collector"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MAX_ROOT_JSON_BYTES = 1024 * 1024
_ROOT_JSON_MODE = 0o400


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


@dataclass(frozen=True)
class HostNumericIdentities:
    gateway_uid: int
    gateway_gid: int
    writer_uid: int
    writer_gid: int
    socket_client_gid: int
    projector_gid: int
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
                "projector_gid",
            )
        }
        if values["gateway_uid"] == values["writer_uid"]:
            raise ValueError("gateway and writer UIDs must be distinct")
        if len(
            {
                values["gateway_gid"],
                values["writer_gid"],
                values["socket_client_gid"],
                values["projector_gid"],
            }
        ) != 4:
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
            raise ValueError(
                "writer must have only its primary and projector groups"
            )

    def to_mapping(self) -> dict[str, Any]:
        self.validate()
        return {
            "gateway_uid": self.gateway_uid,
            "gateway_gid": self.gateway_gid,
            "writer_uid": self.writer_uid,
            "writer_gid": self.writer_gid,
            "socket_client_gid": self.socket_client_gid,
            "projector_gid": self.projector_gid,
            "gateway_supplementary_gids": list(
                self.gateway_supplementary_gids
            ),
            "writer_supplementary_gids": list(
                self.writer_supplementary_gids
            ),
        }


@dataclass(frozen=True)
class ActivationDigests:
    approved_plan_sha256: str
    writer_unit_sha256: str
    gateway_unit_sha256: str
    tmpfiles_sha256: str
    writer_config_sha256: str
    gateway_config_sha256: str

    def validate(self) -> None:
        for name in (
            "approved_plan_sha256",
            "writer_unit_sha256",
            "gateway_unit_sha256",
            "tmpfiles_sha256",
            "writer_config_sha256",
            "gateway_config_sha256",
        ):
            _digest(getattr(self, name), name)

    def to_mapping(self) -> dict[str, str]:
        self.validate()
        return {
            name: getattr(self, name)
            for name in (
                "approved_plan_sha256",
                "writer_unit_sha256",
                "gateway_unit_sha256",
                "tmpfiles_sha256",
                "writer_config_sha256",
                "gateway_config_sha256",
            )
        }


@dataclass(frozen=True)
class ActivationPaths:
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    plan_path: Path = DEFAULT_PLAN_PATH
    receipt_path: Path = DEFAULT_RECEIPT_PATH
    writer_unit_path: Path = DEFAULT_WRITER_UNIT_PATH
    gateway_unit_path: Path = DEFAULT_GATEWAY_UNIT_PATH
    tmpfiles_path: Path = DEFAULT_TMPFILES_PATH

    def validate(self) -> None:
        expected = {
            "manifest_path": DEFAULT_MANIFEST_PATH,
            "plan_path": DEFAULT_PLAN_PATH,
            "receipt_path": DEFAULT_RECEIPT_PATH,
            "writer_unit_path": DEFAULT_WRITER_UNIT_PATH,
            "gateway_unit_path": DEFAULT_GATEWAY_UNIT_PATH,
            "tmpfiles_path": DEFAULT_TMPFILES_PATH,
        }
        for name, pinned in expected.items():
            path = _absolute_normalized_path(getattr(self, name), name)
            if path != pinned:
                raise ValueError(f"activation {name} is production-pinned")

    def to_mapping(self) -> dict[str, str]:
        self.validate()
        return {
            name: str(getattr(self, name))
            for name in (
                "manifest_path",
                "plan_path",
                "receipt_path",
                "writer_unit_path",
                "gateway_unit_path",
                "tmpfiles_path",
            )
        }


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
        if any(
            type(item) is not ExternalNativeExecutableMapping
            for item in values
        ):
            raise TypeError(
                f"{service} native mapping policy contains an invalid entry"
            )
        paths = [item.path for item in values]
        if paths != sorted(paths):
            raise ValueError(
                f"{service} native mapping policy must be exactly sorted"
            )
        if len(paths) != len(set(paths)):
            raise ValueError(
                f"{service} native mapping policy paths must be unique"
            )
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
    if interpreter_entry.kind != "file" or _SHA256_RE.fullmatch(
        interpreter_entry.sha256
    ) is None:
        raise ValueError("release interpreter digest is invalid")
    python_parts = manifest.python_version.split(".")
    if len(python_parts) != 3:
        raise ValueError("release Python version is invalid")
    minor = f"{python_parts[0]}.{python_parts[1]}"
    stdlib_pattern = re.compile(
        rf"^python/[^/]+/lib/python{re.escape(minor)}$"
    )
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
) -> dict[str, Any]:
    import_paths = _release_import_policy(release)
    root = release.artifact_root
    native_executable_policy.validate(root)
    writer_exec = [
        release.interpreter,
        "-I",
        "-m",
        WRITER_MODULE,
        "--config",
        str(unit_spec.writer_config),
    ]
    gateway_exec = [
        release.interpreter,
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
        identities=identities,
    )
    return {
        "deployment_mode": WRITER_ONLY_MODE,
        "gateway_uid": identities.gateway_uid,
        "gateway_gid": identities.gateway_gid,
        "writer_uid": identities.writer_uid,
        "writer_gid": identities.writer_gid,
        "projector_gid": identities.projector_gid,
        "gateway_supplementary_gids": list(
            identities.gateway_supplementary_gids
        ),
        "writer_supplementary_gids": list(
            identities.writer_supplementary_gids
        ),
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


@dataclass(frozen=True)
class ActivationPlan:
    deployment_manifest: Mapping[str, Any]
    unit_bundle: SystemdUnitBundle
    identities: HostNumericIdentities
    digests: ActivationDigests
    paths: ActivationPaths
    sql_private_ip: str
    collector_argv: tuple[str, ...]
    validator_argv: tuple[str, ...]
    sha256: str
    schema: str = ACTIVATION_PLAN_SCHEMA

    def unsigned_mapping(self) -> dict[str, Any]:
        projection_path = (
            Path(
                self.deployment_manifest["host_contract"][
                    "projection_export_path"
                ]
            )
        )
        return {
            "schema": self.schema,
            "sql_private_ip": self.sql_private_ip,
            "identities": self.identities.to_mapping(),
            "digests": self.digests.to_mapping(),
            "paths": self.paths.to_mapping(),
            "deployment_manifest": json.loads(
                json.dumps(self.deployment_manifest)
            ),
            "systemd_bundle": self.unit_bundle.to_mapping(),
            "install_contract": {
                "manifest_mode": "0400",
                "plan_mode": "0400",
                "unit_mode": "0644",
                "tmpfiles_mode": "0644",
                "root_owner_uid": 0,
                "root_owner_gid": 0,
                "projection_export_path": str(projection_path),
                "projection_export_mode": "0640",
                "receipt_parent_mode": "0700",
            },
            "collector_argv": list(self.collector_argv),
            "validator_argv": list(self.validator_argv),
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self.unsigned_mapping(), "plan_sha256": self.sha256}


def build_activation_plan(
    release: ReleaseManifest,
    unit_spec: WriterOnlyUnitSpec,
    writer_config: CanonicalWriterServiceConfig,
    identities: HostNumericIdentities,
    digests: ActivationDigests,
    *,
    native_executable_policy: PreapprovedNativeExecutablePolicy,
    sql_private_ip: str,
    paths: ActivationPaths = ActivationPaths(),
) -> ActivationPlan:
    identities.validate()
    digests.validate()
    paths.validate()
    if type(native_executable_policy) is not PreapprovedNativeExecutablePolicy:
        raise TypeError("preapproved native executable policy is required")
    if not isinstance(sql_private_ip, str) or sql_private_ip != sql_private_ip.strip():
        raise ValueError("SQL private IP is invalid")
    try:
        address = ipaddress.ip_address(sql_private_ip)
    except ValueError as exc:
        raise ValueError("SQL private IP is invalid") from exc
    if (
        address.version != 4
        or not address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise ValueError("SQL endpoint must be exact private IPv4")
    expected_allow = (f"{address}/32",)
    if unit_spec.database_ip_allow != expected_allow:
        raise ValueError("unit SQL allow-list does not match exact private IP")
    unit_bundle = render_systemd_units(release, unit_spec)
    if (
        _sha256_text(unit_bundle.writer_service) != digests.writer_unit_sha256
        or _sha256_text(unit_bundle.gateway_service)
        != digests.gateway_unit_sha256
        or _sha256_text(unit_bundle.tmpfiles) != digests.tmpfiles_sha256
    ):
        raise ValueError("reviewed unit digest does not match rendered bundle")
    snapshot = _snapshot_template(
        release,
        unit_spec,
        writer_config,
        identities,
        native_executable_policy,
        sql_ip=sql_private_ip,
    )
    projection_path = unit_spec.projection_directory / DEFAULT_PROJECTION_FILENAME
    host_contract = {
        "gateway_unit_fragment_path": str(paths.gateway_unit_path),
        "gateway_unit_fragment_sha256": digests.gateway_unit_sha256,
        "writer_unit_fragment_path": str(paths.writer_unit_path),
        "writer_unit_fragment_sha256": digests.writer_unit_sha256,
        "gateway_config_path": str(unit_spec.gateway_config),
        "gateway_config_sha256": digests.gateway_config_sha256,
        "writer_config_sha256": digests.writer_config_sha256,
        "projection_export_path": str(projection_path),
    }
    raw_manifest = {
        "schema": MANIFEST_SCHEMA,
        "mode": WRITER_ONLY_MODE,
        "approved_plan_sha256": digests.approved_plan_sha256,
        "revision": release.revision,
        "artifact_sha256": release.artifact_sha256,
        "snapshot_policy_sha256": snapshot_policy_sha256(snapshot),
        "host_contract": host_contract,
        "snapshot_template": snapshot,
    }
    trusted = TrustedDeploymentManifest.from_mapping(raw_manifest)
    deployment_manifest = trusted.to_mapping()
    collector = (
        release.interpreter,
        "-I",
        "-m",
        ROOT_COLLECTOR_MODULE,
        "collect",
        "--manifest",
        str(paths.manifest_path),
        "--receipt",
        str(paths.receipt_path),
    )
    validator = (
        release.interpreter,
        "-I",
        "-m",
        ROOT_COLLECTOR_MODULE,
        "validate",
        "--manifest",
        str(paths.manifest_path),
        "--receipt",
        str(paths.receipt_path),
    )
    provisional = ActivationPlan(
        deployment_manifest=deployment_manifest,
        unit_bundle=unit_bundle,
        identities=identities,
        digests=digests,
        paths=paths,
        sql_private_ip=sql_private_ip,
        collector_argv=collector,
        validator_argv=validator,
        sha256="",
    )
    plan_sha256 = _sha256_json(provisional.unsigned_mapping())
    return ActivationPlan(
        deployment_manifest=provisional.deployment_manifest,
        unit_bundle=provisional.unit_bundle,
        identities=provisional.identities,
        digests=provisional.digests,
        paths=provisional.paths,
        sql_private_ip=provisional.sql_private_ip,
        collector_argv=provisional.collector_argv,
        validator_argv=provisional.validator_argv,
        sha256=plan_sha256,
    )


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


def _write_root_json(path: Path, value: Mapping[str, Any]) -> None:
    _require_root_linux()
    path = _absolute_normalized_path(path, "root JSON path")
    _validate_root_parent_chain(path.parent.parent)
    parent = os.lstat(path.parent)
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != 0
        or parent.st_gid != 0
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise PermissionError("activation JSON directory must be root-only")
    raw = _canonical_bytes(dict(value)) + b"\n"
    if len(raw) > _MAX_ROOT_JSON_BYTES:
        raise ValueError("activation JSON exceeds its bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    created = False
    try:
        descriptor = os.open(path, flags, 0o600)
        created = True
        os.fchown(descriptor, 0, 0)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("activation JSON write made no progress")
            offset += written
        os.fchmod(descriptor, _ROOT_JSON_MODE)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        if created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    result = os.lstat(path)
    if (
        not stat.S_ISREG(result.st_mode)
        or result.st_nlink != 1
        or result.st_uid != 0
        or result.st_gid != 0
        or stat.S_IMODE(result.st_mode) != _ROOT_JSON_MODE
    ):
        raise RuntimeError("activation JSON was not root-sealed")


def _validate_activation_plan(plan: ActivationPlan) -> None:
    if not isinstance(plan, ActivationPlan):
        raise TypeError("activation plan is required")
    plan.paths.validate()
    plan.identities.validate()
    plan.digests.validate()
    if plan.sha256 != _sha256_json(plan.unsigned_mapping()):
        raise ValueError("activation plan digest is invalid")
    trusted = TrustedDeploymentManifest.from_mapping(plan.deployment_manifest)
    if (
        trusted.approved_plan_sha256 != plan.digests.approved_plan_sha256
        or trusted.host_contract["writer_unit_fragment_sha256"]
        != plan.digests.writer_unit_sha256
        or trusted.host_contract["gateway_unit_fragment_sha256"]
        != plan.digests.gateway_unit_sha256
        or trusted.host_contract["writer_config_sha256"]
        != plan.digests.writer_config_sha256
        or trusted.host_contract["gateway_config_sha256"]
        != plan.digests.gateway_config_sha256
        or _sha256_text(plan.unit_bundle.writer_service)
        != plan.digests.writer_unit_sha256
        or _sha256_text(plan.unit_bundle.gateway_service)
        != plan.digests.gateway_unit_sha256
        or _sha256_text(plan.unit_bundle.tmpfiles)
        != plan.digests.tmpfiles_sha256
    ):
        raise ValueError("activation plan host digests are inconsistent")
    interpreter = str(
        plan.deployment_manifest["snapshot_template"]["writer_deployment"][
            "policy"
        ]["interpreter"]
    )
    collector = (
        interpreter,
        "-I",
        "-m",
        ROOT_COLLECTOR_MODULE,
        "collect",
        "--manifest",
        str(plan.paths.manifest_path),
        "--receipt",
        str(plan.paths.receipt_path),
    )
    validator = (
        interpreter,
        "-I",
        "-m",
        ROOT_COLLECTOR_MODULE,
        "validate",
        "--manifest",
        str(plan.paths.manifest_path),
        "--receipt",
        str(plan.paths.receipt_path),
    )
    if plan.collector_argv != collector or plan.validator_argv != validator:
        raise ValueError("activation root collector argv is inconsistent")


def write_root_collector_manifest(plan: ActivationPlan) -> None:
    _validate_activation_plan(plan)
    _write_root_json(plan.paths.manifest_path, plan.deployment_manifest)


def write_root_activation_plan(plan: ActivationPlan) -> None:
    _validate_activation_plan(plan)
    _write_root_json(plan.paths.plan_path, plan.to_mapping())


__all__ = [
    "ACTIVATION_PLAN_SCHEMA",
    "ActivationDigests",
    "ActivationPaths",
    "ActivationPlan",
    "ExternalNativeExecutableMapping",
    "HostNumericIdentities",
    "PreapprovedNativeExecutablePolicy",
    "ROOT_COLLECTOR_MODULE",
    "build_activation_plan",
    "write_root_activation_plan",
    "write_root_collector_manifest",
]
