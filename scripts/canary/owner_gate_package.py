#!/usr/bin/env python3
"""Build a digest-bound, offline owner-gate package manifest.

This local packager does not install or activate a service.  It verifies every
release payload and every wheel against a caller-supplied complete wheelhouse
manifest, then emits the immutable release layout receipt used by the later
IAP bootstrap gate.  Network dependency installation is deliberately absent.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence, cast

from email import policy
from email.parser import BytesParser
from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement

from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_trust as trust
from scripts.canary import direct_iam_identity_authority as direct_iam


PACKAGE_SCHEMA = "muncho-owner-gate-offline-package.v1"
WHEELHOUSE_SCHEMA = "muncho-owner-gate-offline-wheelhouse.v1"
RUNTIME_LOCK_SCHEMA = "muncho-owner-gate-runtime-wheel-lock.v1"
RUNTIME_LOCK_RELATIVE = "ops/muncho/owner-gate/runtime-wheel-lock.json"
WHEELHOUSE_PLATFORM = "debian_12_x86_64"
_COMPILED_WHEEL_PLATFORM_TAGS = frozenset({
    "manylinux_2_17_x86_64",
    "manylinux2014_x86_64",
    "manylinux_2_28_x86_64",
})
PYTHON_VERSION = "3.11.2"
REQUIRED_ENTRYPOINTS = (
    "bin/muncho-owner-gate-activate-storage",
    "bin/muncho-owner-gate-bootstrap",
    "bin/muncho-owner-gate-install",
    "bin/muncho-owner-gate-intake",
    "bin/muncho-owner-gate-cloud-observation-signer",
    "bin/muncho-owner-gate-stage-activation-evidence",
    "bin/muncho-cloud-trusted-signer-provision",
    "bin/muncho-host-offline-runtime-bootstrap",
    "bin/muncho-host-observation-attestor",
    "bin/muncho-host-trusted-signer-provision",
    "bin/muncho-passkey-v2-web",
    "bin/muncho-passkey-v2-authority",
    "bin/muncho-passkey-v2-executor",
)
ROOT_RUNTIME_FILES = (
    "scripts/canary/owner_gate_activation_evidence_stager.py",
    "scripts/canary/owner_gate_activation_seal.py",
    "scripts/canary/owner_gate_bootstrap.py",
    "scripts/canary/owner_gate_bootstrap_journal.py",
    "scripts/canary/owner_gate_stage0.py",
    "scripts/canary/owner_gate_trust.py",
    "scripts/canary/passkey_v2_protocol.py",
    "scripts/canary/passkey_v2_service.py",
    "scripts/canary/passkey_v2_storage_growth.py",
    "scripts/canary/storage_growth_contract.py",
    "scripts/canary/owner_gate_firewall_readiness.py",
    "scripts/canary/owner_gate_cloud_observation_signer.py",
    "scripts/canary/owner_gate_host_observation_producer.py",
    "scripts/canary/storage_growth_trusted_collector.py",
    "scripts/canary/trusted_signer_provisioning.py",
    "scripts/canary/trusted_signer_stage0.py",
)
FORBIDDEN_RUNTIME_FILES = frozenset({
    "scripts/canary/passkey_v2_store.py",
})
REQUIRED_ASSET_FILES = (
    "ops/muncho/owner-gate/bin/muncho-owner-gate-activate-storage",
    "ops/muncho/owner-gate/bin/muncho-owner-gate-bootstrap",
    "ops/muncho/owner-gate/bin/muncho-owner-gate-install",
    "ops/muncho/owner-gate/bin/muncho-owner-gate-intake",
    "ops/muncho/owner-gate/bin/muncho-owner-gate-cloud-observation-signer",
    "ops/muncho/owner-gate/bin/muncho-owner-gate-stage-activation-evidence",
    "ops/muncho/owner-gate/bin/muncho-cloud-trusted-signer-provision",
    "ops/muncho/owner-gate/bin/muncho-host-offline-runtime-bootstrap",
    "ops/muncho/owner-gate/bin/muncho-host-observation-attestor",
    "ops/muncho/owner-gate/bin/muncho-host-trusted-signer-provision",
    "ops/muncho/owner-gate/bin/muncho-passkey-v2-authority",
    "ops/muncho/owner-gate/bin/muncho-passkey-v2-executor",
    "ops/muncho/owner-gate/bin/muncho-passkey-v2-web",
    "ops/muncho/owner-gate/README.md",
    "ops/muncho/owner-gate/authority.json",
    "ops/muncho/owner-gate/bootstrap-manifest.json",
    "ops/muncho/owner-gate/caddy-private-upstream.Caddyfile.in",
    "ops/muncho/owner-gate/compute-api-hosts.fragment",
    "ops/muncho/owner-gate/cloud-observation-attestor.json",
    "ops/muncho/owner-gate/executor.json",
    "ops/muncho/owner-gate/metadata-firewall.rules",
    "ops/muncho/owner-gate/muncho-owner-gate-metadata-firewall.service",
    "ops/muncho/owner-gate/muncho-owner-gate-firewall-readiness.service",
    "ops/muncho/owner-gate/muncho-owner-gate-firewall-readiness.timer",
    "ops/muncho/owner-gate/muncho-owner-gate.sysusers",
    "ops/muncho/owner-gate/muncho-owner-gate.tmpfiles",
    "ops/muncho/owner-gate/muncho-owner-gate.sudoers",
    "ops/muncho/owner-gate/muncho-owner-gate-provisioning.sudoers.in",
    "ops/muncho/owner-gate/muncho-host-observation-attestor.sudoers.in",
    "ops/muncho/owner-gate/muncho-passkey-authority.service",
    "ops/muncho/owner-gate/muncho-passkey-authority.socket",
    "ops/muncho/owner-gate/muncho-passkey-web.service",
    "ops/muncho/owner-gate/muncho-privileged-executor.service",
    "ops/muncho/owner-gate/muncho-privileged-executor.socket",
    "ops/muncho/owner-gate/python3.sha256.in",
    RUNTIME_LOCK_RELATIVE,
    "ops/muncho/owner-gate/web.json",
)

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WHEEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,254}\.whl$")
_PROJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+!-]{0,127}$")
_MAX_FILE_BYTES = 128 * 1024 * 1024
_MAX_WHEEL_ENTRIES = 10_000
_MAX_WHEEL_ENTRY_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_MAX_WHEEL_TOTAL_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_MAX_WHEEL_COMPRESSION_RATIO = 200
_MAX_WHEEL_METADATA_BYTES = 1024 * 1024
_MAX_WHEEL_DESCRIPTOR_BYTES = 64 * 1024
_GIT = "/usr/bin/git"
_RUNTIME_LOCK_PATH = (
    Path(__file__).resolve(strict=True).parents[2] / RUNTIME_LOCK_RELATIVE
)
_RUNTIME_LOCK_FIELDS = frozenset({
    "bootstrap_pip",
    "schema",
    "python_version",
    "platform",
    "network_required",
    "source_build_allowed",
    "complete_transitive_closure",
    "wheels",
    "lock_sha256",
})
_RUNTIME_LOCK_WHEEL_FIELDS = frozenset({
    "project",
    "version",
    "filename",
    "sha256",
    "size",
    "active_dependencies",
})
PACKAGE_INVENTORY_FIELDS = frozenset({
    "schema",
    "release_revision",
    "source_tree_oid",
    "release_root",
    "release_owner",
    "release_directory_mode",
    "immutable_after_install",
    "offline_bootstrap",
    "network_install_required",
    "interpreter_source",
    "interpreter_version",
    "interpreter_sha256",
    "interpreter_hash_revalidated_before_each_service_start",
    "generic_shell_entrypoint",
    "local_gcloud_runtime_fallback",
    "required_entrypoints",
    "runtime_source_closure",
    "forbidden_runtime_sources",
    "payloads",
    "runtime_lock_sha256",
    "wheelhouse_manifest_sha256",
    "bootstrap_pip",
    "wheels",
    "secret_material_recorded",
    "secret_digest_recorded",
    "activation_performed",
    "cloud_mutation_performed",
    "direct_iam_identity_authority_sha256",
    "pre_foundation_authority_sha256",
    "foundation_apply_receipt_sha256",
    "resource_ancestor_chain",
})
PACKAGE_MANIFEST_FIELDS = PACKAGE_INVENTORY_FIELDS | frozenset({
    "package_inventory_sha256",
    "trust_manifest_sha256",
    "trust_public_key_sha256",
    "interpreter_image",
    "release_supply_chain_attestation",
    "collector_public_key_ids",
    "credential_migration_envelope_sha256",
    "project_ancestry_evidence_sha256",
    "project_ancestry_chain_sha256",
    "caller_self_hash_is_authority",
    "package_sha256",
})
_TARGET_MARKER_ENVIRONMENT = {
    **default_environment(),
    "implementation_name": "cpython",
    "implementation_version": PYTHON_VERSION,
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_release": "",
    "platform_system": "Linux",
    "platform_version": "",
    "python_full_version": PYTHON_VERSION,
    "python_version": "3.11",
    "sys_platform": "linux",
    "extra": "",
}


class OwnerGatePackageError(RuntimeError):
    """Stable, secret-free offline package failure."""


@dataclass(frozen=True)
class PackageSpec:
    source_root: Path
    release_revision: str
    wheelhouse_root: Path
    wheelhouse_manifest: Mapping[str, Any]
    interpreter_sha256: str
    trust_manifest_path: Path | None = None
    trust_public_key_path: Path | None = None
    network_collector_public_key_path: Path | None = None
    cloud_collector_public_key_path: Path | None = None
    host_collector_public_key_path: Path | None = None
    credential_migration_envelope_path: Path | None = None
    direct_iam_identity_authority_path: Path | None = None

    @property
    def release_root(self) -> Path:
        return foundation.RELEASE_BASE / self.release_revision

    def validate(self) -> None:
        if (
            _REVISION.fullmatch(self.release_revision or "") is None
            or not self.source_root.is_absolute()
            or not self.wheelhouse_root.is_absolute()
            or ".." in self.source_root.parts
            or ".." in self.wheelhouse_root.parts
            or _SHA256.fullmatch(self.interpreter_sha256 or "") is None
            or (
                self.trust_manifest_path is not None
                and not self.trust_manifest_path.is_absolute()
            )
            or (
                self.trust_public_key_path is not None
                and not self.trust_public_key_path.is_absolute()
            )
            or any(
                item is not None and not item.is_absolute()
                for item in (
                    self.network_collector_public_key_path,
                    self.cloud_collector_public_key_path,
                    self.host_collector_public_key_path,
                )
            )
            or (
                self.credential_migration_envelope_path is not None
                and not self.credential_migration_envelope_path.is_absolute()
            )
            or (
                self.direct_iam_identity_authority_path is not None
                and not self.direct_iam_identity_authority_path.is_absolute()
            )
        ):
            raise OwnerGatePackageError("owner_gate_package_spec_invalid")


def _verify_clean_git_source(source_root: Path) -> tuple[str, str]:
    def run(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                (_GIT, "-C", str(source_root), *arguments),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            )
        except (OSError, subprocess.SubprocessError):
            raise OwnerGatePackageError(
                "owner_gate_package_git_source_invalid"
            ) from None
        if completed.returncode != 0 or len(completed.stdout) > 4 * 1024 * 1024:
            raise OwnerGatePackageError("owner_gate_package_git_source_invalid")
        try:
            return completed.stdout.decode("ascii", errors="strict").strip()
        except UnicodeError:
            raise OwnerGatePackageError(
                "owner_gate_package_git_source_invalid"
            ) from None

    top_level = run("rev-parse", "--show-toplevel")
    try:
        exact_root = source_root.resolve(strict=True)
        exact_top = Path(top_level).resolve(strict=True)
    except OSError:
        raise OwnerGatePackageError(
            "owner_gate_package_git_source_invalid"
        ) from None
    if exact_root != exact_top:
        raise OwnerGatePackageError("owner_gate_package_git_source_invalid")
    status = run(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    head = run("rev-parse", "--verify", "HEAD")
    tree = run("rev-parse", "--verify", "HEAD^{tree}")
    submodules = run("submodule", "status", "--recursive")
    if (
        status
        or _REVISION.fullmatch(head) is None
        or _REVISION.fullmatch(tree) is None
        or any(line.startswith(("-", "+", "U")) for line in submodules.splitlines())
    ):
        raise OwnerGatePackageError("owner_gate_package_git_source_not_exact")
    return head, tree


def _run_git_raw(
    source_root: Path,
    *arguments: str,
    maximum: int,
) -> bytes:
    try:
        completed = subprocess.run(
            (_GIT, "-C", str(source_root), *arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        raise OwnerGatePackageError(
            "owner_gate_package_git_object_invalid"
        ) from None
    if (
        completed.returncode != 0
        or completed.stderr
        or len(completed.stdout) > maximum
    ):
        raise OwnerGatePackageError("owner_gate_package_git_object_invalid")
    return completed.stdout


def _git_blob(
    source_root: Path,
    release_revision: str,
    relative: str,
    *,
    required: bool,
) -> tuple[bytes, str] | None:
    path = Path(relative)
    if (
        _REVISION.fullmatch(release_revision or "") is None
        or not relative
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != relative
    ):
        raise OwnerGatePackageError("owner_gate_package_git_object_invalid")
    listing = _run_git_raw(
        source_root,
        "ls-tree",
        "--full-tree",
        "-z",
        release_revision,
        "--",
        relative,
        maximum=4 * 1024 * 1024,
    )
    if not listing:
        if required:
            raise OwnerGatePackageError("owner_gate_package_git_object_missing")
        return None
    if listing.count(b"\x00") != 1 or not listing.endswith(b"\x00"):
        raise OwnerGatePackageError("owner_gate_package_git_object_invalid")
    try:
        metadata, listed_path = listing[:-1].split(b"\t", 1)
        mode, object_type, object_id = metadata.split(b" ", 2)
        listed = listed_path.decode("utf-8", errors="strict")
        oid = object_id.decode("ascii", errors="strict")
    except (UnicodeError, ValueError):
        raise OwnerGatePackageError(
            "owner_gate_package_git_object_invalid"
        ) from None
    if (
        listed != relative
        or object_type != b"blob"
        or mode not in {b"100644", b"100755"}
        or _REVISION.fullmatch(oid) is None
    ):
        raise OwnerGatePackageError("owner_gate_package_git_object_invalid")
    size_raw = _run_git_raw(
        source_root,
        "cat-file",
        "-s",
        oid,
        maximum=64,
    )
    try:
        size = int(size_raw.decode("ascii", errors="strict").strip())
    except (UnicodeError, ValueError):
        raise OwnerGatePackageError(
            "owner_gate_package_git_object_invalid"
        ) from None
    if size < 0 or size > _MAX_FILE_BYTES:
        raise OwnerGatePackageError("owner_gate_package_git_object_invalid")
    raw = _run_git_raw(
        source_root,
        "cat-file",
        "blob",
        oid,
        maximum=_MAX_FILE_BYTES,
    )
    if len(raw) != size:
        raise OwnerGatePackageError("owner_gate_package_git_object_invalid")
    return raw, mode.decode("ascii")


def _require_worktree_matches_git_blob(
    path: Path,
    *,
    expected: bytes,
    git_mode: str,
) -> None:
    descriptor: int | None = None
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_nlink < 1
            or opened.st_size != len(expected)
            or bool(stat.S_IMODE(opened.st_mode) & 0o111)
            != (git_mode == "100755")
            or stat.S_IMODE(opened.st_mode) & 0o022
        ):
            raise OwnerGatePackageError(
                "owner_gate_package_worktree_object_mismatch"
            )
        raw = bytearray()
        while len(raw) < opened.st_size:
            chunk = os.read(descriptor, min(1024 * 1024, opened.st_size - len(raw)))
            if not chunk:
                raise OwnerGatePackageError(
                    "owner_gate_package_worktree_object_mismatch"
                )
            raw.extend(chunk)
        after = os.fstat(descriptor)
        if (
            bytes(raw) != expected
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_mode,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_mode,
            )
        ):
            raise OwnerGatePackageError(
                "owner_gate_package_worktree_object_mismatch"
            )
    except OwnerGatePackageError:
        raise
    except OSError:
        raise OwnerGatePackageError(
            "owner_gate_package_worktree_object_mismatch"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _sha256_file(path: Path) -> tuple[str, int]:
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        state = os.fstat(descriptor)
    except OSError:
        raise OwnerGatePackageError(
            "owner_gate_package_file_unavailable"
        ) from None
    if (
        not stat.S_ISREG(state.st_mode)
        or stat.S_ISLNK(state.st_mode)
        or state.st_size > _MAX_FILE_BYTES
        or stat.S_IMODE(state.st_mode) & 0o022
    ):
        raise OwnerGatePackageError("owner_gate_package_file_invalid")
    try:
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest(), state.st_size
    except OSError:
        raise OwnerGatePackageError(
            "owner_gate_package_file_unavailable"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _strict_keys(raw: Mapping[str, Any], fields: frozenset[str], label: str) -> None:
    if not isinstance(raw, Mapping) or frozenset(raw) != fields:
        raise OwnerGatePackageError(f"owner_gate_package_{label}_invalid")


def _normalize_project(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _compiled_wheel_platform_allowed(value: str) -> bool:
    return bool(set(value.lower().split(".")) & _COMPILED_WHEEL_PLATFORM_TAGS)


def _compiled_wheel_tag_allowed(value: str) -> bool:
    parts = value.lower().split("-")
    return (
        len(parts) == 3
        and parts[0] == "cp311"
        and parts[1] in {"cp311", "abi3"}
        and _compiled_wheel_platform_allowed(parts[2])
    )


def _compiled_wheel_filename_allowed(value: str) -> bool:
    if not value.lower().endswith(".whl"):
        return False
    try:
        _prefix, interpreter, abi, platform = value[:-4].rsplit("-", 3)
    except ValueError:
        return False
    return (
        interpreter.lower() == "cp311"
        and abi.lower() in {"cp311", "abi3"}
        and _compiled_wheel_platform_allowed(platform)
    )


def _runtime_lock_artifact_valid(item: object) -> bool:
    if not isinstance(item, Mapping) or frozenset(item) != _RUNTIME_LOCK_WHEEL_FIELDS:
        return False
    artifact = cast(Mapping[str, Any], item)
    project = artifact.get("project")
    version = artifact.get("version")
    filename = artifact.get("filename")
    dependencies = artifact.get("active_dependencies")
    size = artifact.get("size")
    return bool(
        isinstance(project, str)
        and _PROJECT.fullmatch(project) is not None
        and _normalize_project(project) == project
        and isinstance(version, str)
        and _VERSION.fullmatch(version) is not None
        and isinstance(filename, str)
        and _WHEEL_NAME.fullmatch(filename) is not None
        and isinstance(size, int)
        and not isinstance(size, bool)
        and 0 < size <= _MAX_FILE_BYTES
        and _SHA256.fullmatch(str(artifact.get("sha256", ""))) is not None
        and isinstance(dependencies, list)
        and dependencies == sorted(set(dependencies))
        and all(
            isinstance(dependency, str)
            and _PROJECT.fullmatch(dependency) is not None
            and _normalize_project(dependency) == dependency
            and dependency != project
            for dependency in dependencies
        )
        and (
            filename.lower().endswith("-py3-none-any.whl")
            or _compiled_wheel_filename_allowed(filename)
        )
    )


def decode_runtime_lock(raw: bytes) -> Mapping[str, Any]:
    """Validate the canonical, platform-specific owner-gate wheel authority."""

    try:
        if not raw or len(raw) > 4 * 1024 * 1024 or not raw.endswith(b"\n"):
            raise ValueError
        value = json.loads(raw[:-1].decode("ascii", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError):
        raise OwnerGatePackageError(
            "owner_gate_package_runtime_lock_invalid"
        ) from None
    if not isinstance(value, Mapping):
        raise OwnerGatePackageError("owner_gate_package_runtime_lock_invalid")
    _strict_keys(value, _RUNTIME_LOCK_FIELDS, "runtime_lock")
    unsigned = {key: item for key, item in value.items() if key != "lock_sha256"}
    wheels = value.get("wheels")
    bootstrap_pip = value.get("bootstrap_pip")
    if (
        value.get("schema") != RUNTIME_LOCK_SCHEMA
        or value.get("python_version") != PYTHON_VERSION
        or value.get("platform") != WHEELHOUSE_PLATFORM
        or value.get("network_required") is not False
        or value.get("source_build_allowed") is not False
        or value.get("complete_transitive_closure") is not True
        or not isinstance(wheels, list)
        or not wheels
        or len(wheels) > 256
        or not _runtime_lock_artifact_valid(bootstrap_pip)
        or not _SHA256.fullmatch(str(value.get("lock_sha256", "")))
        or foundation.sha256_json(unsigned) != value["lock_sha256"]
        or raw != foundation.canonical_json_bytes(value) + b"\n"
    ):
        raise OwnerGatePackageError("owner_gate_package_runtime_lock_invalid")
    bootstrap_pip = cast(Mapping[str, Any], bootstrap_pip)
    if (
        bootstrap_pip["project"] != "pip"
        or bootstrap_pip["active_dependencies"] != []
        or not bootstrap_pip["filename"].lower().endswith(
            "-py3-none-any.whl"
        )
    ):
        raise OwnerGatePackageError("owner_gate_package_runtime_lock_invalid")
    projects: set[str] = set()
    filenames: set[str] = set()
    normalized_wheels: list[dict[str, Any]] = []
    for item in wheels:
        if not _runtime_lock_artifact_valid(item):
            raise OwnerGatePackageError("owner_gate_package_runtime_lock_invalid")
        artifact = cast(Mapping[str, Any], item)
        project = artifact.get("project")
        filename = artifact.get("filename")
        if (
            project == "pip"
            or project in projects
            or filename in filenames
        ):
            raise OwnerGatePackageError("owner_gate_package_runtime_lock_invalid")
        assert isinstance(project, str)
        assert isinstance(filename, str)
        projects.add(project)
        filenames.add(filename)
        normalized_wheels.append(dict(artifact))
    if (
        bootstrap_pip["filename"] in filenames
        or [item["project"] for item in normalized_wheels] != sorted(projects)
        or any(
            dependency not in projects
            for item in normalized_wheels
            for dependency in item["active_dependencies"]
        )
    ):
        raise OwnerGatePackageError("owner_gate_package_runtime_lock_invalid")
    return dict(value)


def _read_local_runtime_lock() -> Mapping[str, Any]:
    try:
        raw = _RUNTIME_LOCK_PATH.read_bytes()
    except OSError:
        raise OwnerGatePackageError(
            "owner_gate_package_runtime_lock_unavailable"
        ) from None
    return decode_runtime_lock(raw)


LOCAL_RUNTIME_LOCK = _read_local_runtime_lock()
REQUIRED_PROJECTS = {
    item["project"]: item["version"] for item in LOCAL_RUNTIME_LOCK["wheels"]
}
BINARY_PROJECTS = frozenset(
    item["project"]
    for item in LOCAL_RUNTIME_LOCK["wheels"]
    if not item["filename"].lower().endswith("-py3-none-any.whl")
)
EXPECTED_DIRECT_DEPENDENCIES = {
    item["project"]: set(item["active_dependencies"])
    for item in LOCAL_RUNTIME_LOCK["wheels"]
}


def _load_release_runtime_lock(
    source_root: Path,
    release_revision: str,
) -> tuple[Mapping[str, Any], str]:
    selected = _git_blob(
        source_root,
        release_revision,
        RUNTIME_LOCK_RELATIVE,
        required=True,
    )
    assert selected is not None
    raw, git_mode = selected
    if git_mode != "100644":
        raise OwnerGatePackageError("owner_gate_package_runtime_lock_invalid")
    _require_worktree_matches_git_blob(
        source_root / RUNTIME_LOCK_RELATIVE,
        expected=raw,
        git_mode=git_mode,
    )
    return decode_runtime_lock(raw), hashlib.sha256(raw).hexdigest()


def runtime_lock_file_sha256(runtime_lock: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        foundation.canonical_json_bytes(runtime_lock) + b"\n"
    ).hexdigest()


def _active_requirement_names(values: Sequence[str]) -> set[str]:
    active: set[str] = set()
    for value in values:
        try:
            requirement = Requirement(value)
        except InvalidRequirement:
            raise OwnerGatePackageError(
                "owner_gate_package_wheel_metadata_invalid"
            ) from None
        if requirement.url is not None:
            raise OwnerGatePackageError(
                "owner_gate_package_wheel_metadata_invalid"
            )
        if requirement.marker is not None and not requirement.marker.evaluate(
            environment=_TARGET_MARKER_ENVIRONMENT,
        ):
            continue
        active.add(_normalize_project(requirement.name))
    return active


def wheel_site_packages_relative(name: str) -> PurePosixPath | None:
    """Map one wheel member to its eventual site-packages destination."""

    path = PurePosixPath(name)
    parts = path.parts
    if not parts:
        return None
    if parts[0].lower().endswith(".data"):
        if len(parts) < 3 or parts[1].lower() not in {"purelib", "platlib"}:
            return None
        parts = parts[2:]
    return PurePosixPath(*parts) if parts else None


def unsupported_wheel_data_destination(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return bool(
        parts
        and parts[0].casefold().endswith(".data")
        and (len(parts) < 3 or parts[1].casefold() not in {"purelib", "platlib"})
    )


def startup_sensitive_site_packages_path(path: PurePosixPath) -> bool:
    """Return true for import-time hooks that must never enter the venv."""

    parts = path.parts
    if not parts:
        return False
    if any(part.casefold().endswith(".pth") for part in parts):
        return True
    first = parts[0].casefold()
    return first in {"sitecustomize", "usercustomize"} or any(
        first.startswith(f"{name}.")
        for name in ("sitecustomize", "usercustomize")
    )


def _verify_wheel_archive(
    path: Path,
    *,
    project: str,
    version: str,
    compiled_wheel: bool,
    expected_dependencies: set[str] | None = None,
    allowed_projects: set[str] | None = None,
) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if (
                not names
                or len(entries) > _MAX_WHEEL_ENTRIES
                or len(names) != len(set(names))
                or any(
                    not name
                    or name.startswith("/")
                    or ".." in Path(name).parts
                    or "\\" in name
                    or "\x00" in name
                    or unsupported_wheel_data_destination(name)
                    or (
                        (destination := wheel_site_packages_relative(name))
                        is not None
                        and startup_sensitive_site_packages_path(destination)
                    )
                    for name in names
                )
            ):
                raise OwnerGatePackageError(
                    "owner_gate_package_wheel_archive_invalid"
                )
            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            wheel_names = [
                name for name in names if name.endswith(".dist-info/WHEEL")
            ]
            if len(metadata_names) != 1 or len(wheel_names) != 1:
                raise OwnerGatePackageError(
                    "owner_gate_package_wheel_archive_invalid"
                )
            metadata_name = metadata_names[0]
            wheel_name = wheel_names[0]
            total_uncompressed = 0
            for entry in entries:
                mode = entry.external_attr >> 16
                node_type = stat.S_IFMT(mode)
                is_directory = entry.is_dir()
                if (
                    entry.flag_bits & 0x1
                    or entry.compress_type
                    not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or entry.file_size < 0
                    or entry.compress_size < 0
                    or entry.file_size > _MAX_WHEEL_ENTRY_UNCOMPRESSED_BYTES
                    or (
                        entry.filename == metadata_name
                        and entry.file_size > _MAX_WHEEL_METADATA_BYTES
                    )
                    or (
                        entry.filename == wheel_name
                        and entry.file_size > _MAX_WHEEL_DESCRIPTOR_BYTES
                    )
                    or (
                        is_directory
                        and node_type not in {0, stat.S_IFDIR}
                    )
                    or (
                        not is_directory
                        and node_type not in {0, stat.S_IFREG}
                    )
                    or (
                        entry.file_size > 0
                        and (
                            entry.compress_size == 0
                            or entry.file_size
                            > entry.compress_size * _MAX_WHEEL_COMPRESSION_RATIO
                        )
                    )
                ):
                    raise OwnerGatePackageError(
                        "owner_gate_package_wheel_archive_invalid"
                    )
                total_uncompressed += entry.file_size
                if total_uncompressed > _MAX_WHEEL_TOTAL_UNCOMPRESSED_BYTES:
                    raise OwnerGatePackageError(
                        "owner_gate_package_wheel_archive_invalid"
                    )
            # Force decompression and CRC validation for every bounded member;
            # pip never becomes the first parser of signed archive bytes.
            captured: dict[str, bytes] = {}
            for entry in entries:
                observed = 0
                selected = (
                    bytearray()
                    if entry.filename in {metadata_name, wheel_name}
                    else None
                )
                with archive.open(entry, "r") as member:
                    while chunk := member.read(1024 * 1024):
                        observed += len(chunk)
                        if observed > entry.file_size:
                            raise OwnerGatePackageError(
                                "owner_gate_package_wheel_archive_invalid"
                            )
                        if selected is not None:
                            selected.extend(chunk)
                if observed != entry.file_size:
                    raise OwnerGatePackageError(
                        "owner_gate_package_wheel_archive_invalid"
                    )
                if selected is not None:
                    captured[entry.filename] = bytes(selected)
            metadata_raw = captured[metadata_name]
            wheel_raw = captured[wheel_name]
    except (
        OSError,
        EOFError,
        KeyError,
        RuntimeError,
        NotImplementedError,
        zipfile.BadZipFile,
    ):
        raise OwnerGatePackageError(
            "owner_gate_package_wheel_archive_invalid"
        ) from None
    try:
        metadata = BytesParser(policy=policy.compat32).parsebytes(metadata_raw)
        wheel = BytesParser(policy=policy.compat32).parsebytes(wheel_raw)
    except (TypeError, ValueError):
        raise OwnerGatePackageError(
            "owner_gate_package_wheel_metadata_invalid"
        ) from None
    normalized = _normalize_project(project)
    requires = _active_requirement_names(metadata.get_all("Requires-Dist", []))
    required_dependencies = (
        EXPECTED_DIRECT_DEPENDENCIES.get(normalized, set())
        if expected_dependencies is None
        else expected_dependencies
    )
    dependency_projects = (
        set(REQUIRED_PROJECTS)
        if allowed_projects is None
        else allowed_projects
    )
    if (
        _normalize_project(str(metadata.get("Name", ""))) != normalized
        or metadata.get("Version") != version
        or requires != required_dependencies
        or any(item not in dependency_projects for item in requires)
    ):
        raise OwnerGatePackageError("owner_gate_package_wheel_metadata_invalid")
    tags = set(wheel.get_all("Tag", []))
    root_is_pure = str(wheel.get("Root-Is-Purelib", "")).lower()
    if compiled_wheel:
        if root_is_pure != "false" or not any(
            _compiled_wheel_tag_allowed(tag) for tag in tags
        ):
            raise OwnerGatePackageError("owner_gate_package_wheel_tag_invalid")
    elif root_is_pure != "true" or "py3-none-any" not in tags:
        raise OwnerGatePackageError("owner_gate_package_wheel_tag_invalid")


def _module_candidates(module: str, names: Sequence[str]) -> tuple[str, ...]:
    candidates = [module.replace(".", "/") + ".py"]
    candidates.append(module.replace(".", "/") + "/__init__.py")
    for name in names:
        candidates.append((module + "." + name).replace(".", "/") + ".py")
    return tuple(candidates)


def resolve_runtime_source_closure(
    source_root: Path,
    *,
    release_revision: str,
) -> tuple[str, ...]:
    pending = list(ROOT_RUNTIME_FILES)
    resolved: set[str] = set()
    cache: dict[str, tuple[bytes, str] | None] = {}

    def git_blob(relative: str, *, required: bool) -> tuple[bytes, str] | None:
        if relative not in cache:
            cache[relative] = _git_blob(
                source_root,
                release_revision,
                relative,
                required=required,
            )
        value = cache[relative]
        if required and value is None:
            raise OwnerGatePackageError("owner_gate_package_git_object_missing")
        return value

    while pending:
        relative = pending.pop()
        if relative in resolved:
            continue
        if relative in FORBIDDEN_RUNTIME_FILES:
            raise OwnerGatePackageError("owner_gate_package_forbidden_runtime_module")
        try:
            selected = git_blob(relative, required=True)
            assert selected is not None
            source = selected[0].decode("utf-8", errors="strict")
            tree = ast.parse(source, filename=relative)
        except (UnicodeError, SyntaxError):
            raise OwnerGatePackageError(
                "owner_gate_package_runtime_source_invalid"
            ) from None
        resolved.add(relative)
        parent = Path(relative).parent
        while parent != Path("."):
            initializer = str(parent / "__init__.py")
            if git_blob(initializer, required=False) is not None and initializer not in resolved:
                pending.append(initializer)
            parent = parent.parent
        for node in ast.walk(tree):
            candidates: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                candidates = tuple(
                    candidate
                    for name in node.names
                    for candidate in _module_candidates(name.name, ())
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package_parts = list(Path(relative).with_suffix("").parts[:-1])
                    remove = node.level - 1
                    if remove > len(package_parts):
                        raise OwnerGatePackageError(
                            "owner_gate_package_runtime_import_invalid"
                        )
                    if remove:
                        package_parts = package_parts[:-remove]
                    if module:
                        package_parts.extend(module.split("."))
                    module = ".".join(package_parts)
                if module:
                    candidates = _module_candidates(
                        module,
                        tuple(name.name for name in node.names),
                    )
            for candidate in candidates:
                if git_blob(candidate, required=False) is not None and candidate not in resolved:
                    pending.append(candidate)
    if resolved & FORBIDDEN_RUNTIME_FILES:
        raise OwnerGatePackageError("owner_gate_package_forbidden_runtime_module")
    return tuple(sorted(resolved))


def validate_wheelhouse(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    runtime_lock: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    selected_lock = LOCAL_RUNTIME_LOCK if runtime_lock is None else runtime_lock
    # Re-encode through the same independent validator even for in-memory
    # callers so a partially shaped or mutated mapping never becomes authority.
    selected_lock = decode_runtime_lock(
        foundation.canonical_json_bytes(selected_lock) + b"\n"
    )
    lock_file_sha256 = runtime_lock_file_sha256(selected_lock)
    _strict_keys(
        manifest,
        frozenset({
            "schema",
            "python_version",
            "platform",
            "network_required",
            "source_build_allowed",
            "complete_transitive_closure",
            "runtime_lock_sha256",
            "bootstrap_pip",
            "wheels",
            "manifest_sha256",
        }),
        "wheelhouse_manifest",
    )
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    wheels = manifest.get("wheels")
    if (
        manifest.get("schema") != WHEELHOUSE_SCHEMA
        or manifest.get("python_version") != PYTHON_VERSION
        or manifest.get("platform") != WHEELHOUSE_PLATFORM
        or manifest.get("network_required") is not False
        or manifest.get("source_build_allowed") is not False
        or manifest.get("complete_transitive_closure") is not True
        or manifest.get("runtime_lock_sha256") != lock_file_sha256
        or not isinstance(wheels, list)
        or not wheels
        or not isinstance(manifest.get("bootstrap_pip"), Mapping)
        or not _SHA256.fullmatch(str(manifest.get("manifest_sha256", "")))
        or foundation.sha256_json(unsigned) != manifest["manifest_sha256"]
    ):
        raise OwnerGatePackageError(
            "owner_gate_package_wheelhouse_manifest_invalid"
        )
    verified: list[dict[str, Any]] = []
    expected_wheels = [
        {
            key: item[key]
            for key in ("filename", "project", "version", "sha256", "size")
        }
        for item in selected_lock["wheels"]
    ]
    locked_bootstrap = {
        key: selected_lock["bootstrap_pip"][key]
        for key in ("filename", "project", "version", "sha256", "size")
    }
    bootstrap_pip = manifest["bootstrap_pip"]
    _strict_keys(
        bootstrap_pip,
        frozenset({"filename", "project", "version", "sha256", "size"}),
        "bootstrap_pip",
    )
    if dict(bootstrap_pip) != locked_bootstrap:
        raise OwnerGatePackageError("owner_gate_package_wheelhouse_lock_mismatch")
    bootstrap_digest, bootstrap_size = _sha256_file(
        root / locked_bootstrap["filename"]
    )
    if (
        bootstrap_digest != locked_bootstrap["sha256"]
        or bootstrap_size != locked_bootstrap["size"]
    ):
        raise OwnerGatePackageError("owner_gate_package_wheel_digest_invalid")
    _verify_wheel_archive(
        root / locked_bootstrap["filename"],
        project=locked_bootstrap["project"],
        version=locked_bootstrap["version"],
        compiled_wheel=False,
        expected_dependencies=set(),
        allowed_projects={"pip"},
    )
    lock_by_project = {
        item["project"]: item for item in selected_lock["wheels"]
    }
    projects: dict[str, str] = {}
    names: set[str] = set()
    for item in wheels:
        _strict_keys(
            item,
            frozenset({"filename", "project", "version", "sha256", "size"}),
            "wheel",
        )
        filename = item.get("filename")
        project = item.get("project")
        version = item.get("version")
        if (
            not isinstance(filename, str)
            or _WHEEL_NAME.fullmatch(filename) is None
            or filename in names
            or not isinstance(project, str)
            or _PROJECT.fullmatch(project) is None
            or not isinstance(version, str)
            or _VERSION.fullmatch(version) is None
            or not isinstance(item.get("size"), int)
            or isinstance(item.get("size"), bool)
            or item["size"] <= 0
            or item["size"] > _MAX_FILE_BYTES
            or not _SHA256.fullmatch(str(item.get("sha256", "")))
        ):
            raise OwnerGatePackageError("owner_gate_package_wheel_invalid")
        digest, size = _sha256_file(root / filename)
        if digest != item["sha256"] or size != item["size"]:
            raise OwnerGatePackageError("owner_gate_package_wheel_digest_invalid")
        normalized_project = _normalize_project(project)
        if normalized_project in projects:
            raise OwnerGatePackageError("owner_gate_package_wheel_project_invalid")
        projects[normalized_project] = version
        names.add(filename)
        pure_wheel = "-py3-none-any.whl" in filename.lower()
        compiled_wheel = _compiled_wheel_filename_allowed(filename)
        locked_filename = str(
            lock_by_project.get(normalized_project, {}).get("filename", "")
        )
        locked_binary = not locked_filename.lower().endswith(
            "-py3-none-any.whl"
        )
        if not (pure_wheel or compiled_wheel) or (
            locked_binary and not compiled_wheel
        ):
            raise OwnerGatePackageError("owner_gate_package_wheel_platform_invalid")
        _verify_wheel_archive(
            root / filename,
            project=project,
            version=version,
            compiled_wheel=compiled_wheel,
            expected_dependencies=set(
                lock_by_project.get(normalized_project, {}).get(
                    "active_dependencies", ()
                )
            ),
            allowed_projects=set(lock_by_project),
        )
        verified.append(dict(item))
    if (
        projects
        != {
            item["project"]: item["version"]
            for item in selected_lock["wheels"]
        }
        or sorted(verified, key=lambda item: item["filename"])
        != sorted(expected_wheels, key=lambda item: item["filename"])
    ):
        raise OwnerGatePackageError("owner_gate_package_wheelhouse_lock_mismatch")
    return tuple(sorted(verified, key=lambda item: item["filename"]))


def build_inventory(spec: PackageSpec) -> Mapping[str, Any]:
    spec.validate()
    head_revision, source_tree_oid = _verify_clean_git_source(spec.source_root)
    if head_revision != spec.release_revision:
        raise OwnerGatePackageError("owner_gate_package_revision_mismatch")
    runtime_lock, runtime_lock_sha256 = _load_release_runtime_lock(
        spec.source_root,
        spec.release_revision,
    )
    wheels = validate_wheelhouse(
        root=spec.wheelhouse_root,
        manifest=spec.wheelhouse_manifest,
        runtime_lock=runtime_lock,
    )
    if spec.direct_iam_identity_authority_path is None:
        raise OwnerGatePackageError(
            "owner_gate_package_direct_iam_identity_authority_required"
        )
    try:
        direct_iam_raw = trust._read_immutable(
            spec.direct_iam_identity_authority_path,
            maximum=direct_iam.MAX_BYTES,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
        direct_iam_authority = direct_iam.decode_canonical(direct_iam_raw)
    except (
        OSError,
        trust.OwnerGateTrustError,
        direct_iam.DirectIamIdentityAuthorityError,
    ):
        raise OwnerGatePackageError(
            "owner_gate_package_direct_iam_identity_authority_invalid"
        ) from None
    direct_iam_sha256 = hashlib.sha256(direct_iam_raw).hexdigest()
    payloads: list[dict[str, Any]] = []
    runtime_sources = resolve_runtime_source_closure(
        spec.source_root,
        release_revision=spec.release_revision,
    )
    source_relatives: set[str] = set()
    release_relatives: set[str] = set()
    for relative in (*runtime_sources, *REQUIRED_ASSET_FILES):
        executable_name = relative.removeprefix("ops/muncho/owner-gate/bin/")
        is_entrypoint = executable_name != relative
        release_relative = (
            f"bin/{executable_name}" if is_entrypoint else relative
        )
        if (
            relative in source_relatives
            or release_relative in release_relatives
        ):
            raise OwnerGatePackageError(
                "owner_gate_package_payload_path_duplicate"
            )
        source_relatives.add(relative)
        release_relatives.add(release_relative)
        selected = _git_blob(
            spec.source_root,
            spec.release_revision,
            relative,
            required=True,
        )
        assert selected is not None
        git_raw, git_mode = selected
        _require_worktree_matches_git_blob(
            spec.source_root / relative,
            expected=git_raw,
            git_mode=git_mode,
        )
        digest = hashlib.sha256(git_raw).hexdigest()
        size = len(git_raw)
        payloads.append({
            "source_relative": relative,
            "release_relative": release_relative,
            "sha256": digest,
            "size": size,
            "owner": "root:root",
            "mode": "0555" if is_entrypoint else "0444",
        })
    unsigned = {
        "schema": PACKAGE_SCHEMA,
        "release_revision": spec.release_revision,
        "source_tree_oid": source_tree_oid,
        "release_root": str(spec.release_root),
        "release_owner": "root:root",
        "release_directory_mode": "0555",
        "immutable_after_install": True,
        "offline_bootstrap": True,
        "network_install_required": False,
        "interpreter_source": "pinned_debian_image_usr_bin_python3",
        "interpreter_version": PYTHON_VERSION,
        "interpreter_sha256": spec.interpreter_sha256,
        "interpreter_hash_revalidated_before_each_service_start": True,
        "generic_shell_entrypoint": False,
        "local_gcloud_runtime_fallback": False,
        "required_entrypoints": list(REQUIRED_ENTRYPOINTS),
        "runtime_source_closure": list(runtime_sources),
        "forbidden_runtime_sources": sorted(FORBIDDEN_RUNTIME_FILES),
        "payloads": sorted(payloads, key=lambda item: item["source_relative"]),
        "runtime_lock_sha256": runtime_lock_sha256,
        "wheelhouse_manifest_sha256": spec.wheelhouse_manifest["manifest_sha256"],
        "bootstrap_pip": {
            key: runtime_lock["bootstrap_pip"][key]
            for key in ("filename", "project", "version", "sha256", "size")
        },
        "wheels": list(wheels),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "activation_performed": False,
        "cloud_mutation_performed": False,
        "direct_iam_identity_authority_sha256": direct_iam_sha256,
        "pre_foundation_authority_sha256": direct_iam_authority[
            "pre_foundation_authority_sha256"
        ],
        "foundation_apply_receipt_sha256": direct_iam_authority[
            "foundation_apply_receipt_sha256"
        ],
        "resource_ancestor_chain": list(
            direct_iam_authority["resource_ancestor_chain"]
        ),
    }
    if frozenset(unsigned) != PACKAGE_INVENTORY_FIELDS:
        raise OwnerGatePackageError("owner_gate_package_inventory_fields_invalid")
    final_head, final_tree = _verify_clean_git_source(spec.source_root)
    if final_head != head_revision or final_tree != source_tree_oid:
        raise OwnerGatePackageError("owner_gate_package_git_source_changed")
    return unsigned


def build_manifest(spec: PackageSpec) -> Mapping[str, Any]:
    inventory = build_inventory(spec)
    if spec.trust_manifest_path is None or spec.trust_public_key_path is None:
        raise OwnerGatePackageError("owner_gate_package_trust_anchor_required")
    try:
        authority = trust.load_pinned_release_trust(
            manifest_path=spec.trust_manifest_path,
            public_key_path=spec.trust_public_key_path,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
        )
        trust.verify_inventory_authority(authority, inventory=inventory)
    except trust.OwnerGateTrustError:
        raise OwnerGatePackageError(
            "owner_gate_package_trust_invalid"
        ) from None
    unsigned = {
        **inventory,
        "package_inventory_sha256": foundation.sha256_json(inventory),
        "trust_manifest_sha256": authority["trust_manifest_sha256"],
        "trust_public_key_sha256": authority["trust_public_key_sha256"],
        "interpreter_image": authority["interpreter_image"],
        "release_supply_chain_attestation": authority["release_attestation"],
        "collector_public_key_ids": authority["collector_public_key_ids"],
        "credential_migration_envelope_sha256": authority[
            "credential_migration_envelope_sha256"
        ],
        "project_ancestry_evidence_sha256": authority[
            "project_ancestry_evidence_sha256"
        ],
        "project_ancestry_chain_sha256": authority[
            "project_ancestry_chain_sha256"
        ],
        "caller_self_hash_is_authority": False,
    }
    return {**unsigned, "package_sha256": foundation.sha256_json(unsigned)}


def validate_authorized_manifest(
    manifest: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(manifest, Mapping) or frozenset(manifest) != PACKAGE_MANIFEST_FIELDS:
        raise OwnerGatePackageError("owner_gate_package_manifest_fields_invalid")
    inventory = {key: manifest[key] for key in PACKAGE_INVENTORY_FIELDS}
    unsigned = {
        key: item for key, item in manifest.items() if key != "package_sha256"
    }
    if (
        manifest.get("package_sha256") != foundation.sha256_json(unsigned)
        or manifest.get("package_inventory_sha256")
        != foundation.sha256_json(inventory)
        or manifest.get("trust_manifest_sha256")
        != authority.get("trust_manifest_sha256")
        or manifest.get("trust_public_key_sha256")
        != authority.get("trust_public_key_sha256")
        or manifest.get("interpreter_image")
        != authority.get("interpreter_image")
        or manifest.get("release_supply_chain_attestation")
        != authority.get("release_attestation")
        or manifest.get("collector_public_key_ids")
        != authority.get("collector_public_key_ids")
        or manifest.get("credential_migration_envelope_sha256")
        != authority.get("credential_migration_envelope_sha256")
        or manifest.get("project_ancestry_evidence_sha256")
        != authority.get("project_ancestry_evidence_sha256")
        or manifest.get("project_ancestry_chain_sha256")
        != authority.get("project_ancestry_chain_sha256")
        or manifest.get("resource_ancestor_chain")
        != authority.get("resource_ancestor_chain")
        or manifest.get("caller_self_hash_is_authority") is not False
    ):
        raise OwnerGatePackageError("owner_gate_package_manifest_invalid")
    try:
        trust.verify_inventory_authority(authority, inventory=inventory)
    except trust.OwnerGateTrustError:
        raise OwnerGatePackageError(
            "owner_gate_package_trust_invalid"
        ) from None
    return inventory


def _copy_exact(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    mode: int,
) -> None:
    source_fd: int | None = None
    destination_fd: int | None = None
    try:
        source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        source_flags |= getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(source, source_flags)
        source_state = os.fstat(source_fd)
        if (
            not stat.S_ISREG(source_state.st_mode)
            or source_state.st_size != expected_size
            or source_state.st_nlink < 1
        ):
            raise OwnerGatePackageError("owner_gate_package_bundle_source_invalid")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        destination_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        destination_fd = os.open(destination, destination_flags, mode)
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(source_fd, 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError
                view = view[written:]
        if total != expected_size or digest.hexdigest() != expected_sha256:
            raise OwnerGatePackageError("owner_gate_package_bundle_digest_invalid")
        os.fchmod(destination_fd, mode)
        os.fsync(destination_fd)
    except OSError:
        raise OwnerGatePackageError(
            "owner_gate_package_bundle_write_failed"
        ) from None
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)


def _write_exact(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    except OSError:
        raise OwnerGatePackageError(
            "owner_gate_package_bundle_write_failed"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def materialize_bundle(
    spec: PackageSpec,
    *,
    destination: Path,
) -> Mapping[str, Any]:
    if (
        not destination.is_absolute()
        or ".." in destination.parts
        or destination.exists()
        or not destination.parent.is_dir()
    ):
        raise OwnerGatePackageError("owner_gate_package_bundle_path_invalid")
    manifest = build_manifest(spec)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise OwnerGatePackageError("owner_gate_package_bundle_path_invalid")
    temporary.mkdir(mode=0o700)
    try:
        seen: set[str] = set()
        for item in manifest["payloads"]:
            relative = item["release_relative"]
            if (
                not isinstance(relative, str)
                or relative in seen
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise OwnerGatePackageError(
                    "owner_gate_package_bundle_payload_invalid"
                )
            seen.add(relative)
            _copy_exact(
                spec.source_root / item["source_relative"],
                temporary / "payload" / relative,
                expected_sha256=item["sha256"],
                expected_size=item["size"],
                mode=int(item["mode"], 8),
            )
        for item in manifest["wheels"]:
            _copy_exact(
                spec.wheelhouse_root / item["filename"],
                temporary / "wheels" / item["filename"],
                expected_sha256=item["sha256"],
                expected_size=item["size"],
                mode=0o444,
            )
        bootstrap_pip = manifest["bootstrap_pip"]
        _copy_exact(
            spec.wheelhouse_root / bootstrap_pip["filename"],
            temporary / "bootstrap" / bootstrap_pip["filename"],
            expected_sha256=bootstrap_pip["sha256"],
            expected_size=bootstrap_pip["size"],
            mode=0o444,
        )
        trust_manifest_sha, trust_manifest_size = _sha256_file(
            spec.trust_manifest_path  # type: ignore[arg-type]
        )
        _copy_exact(
            spec.trust_manifest_path,  # type: ignore[arg-type]
            temporary / "trust" / "release-trust.json",
            expected_sha256=trust_manifest_sha,
            expected_size=trust_manifest_size,
            mode=0o444,
        )
        trust_key_sha, trust_key_size = _sha256_file(
            spec.trust_public_key_path  # type: ignore[arg-type]
        )
        _copy_exact(
            spec.trust_public_key_path,  # type: ignore[arg-type]
            temporary / "trust" / "release-trust-signing.pub",
            expected_sha256=trust_key_sha,
            expected_size=trust_key_size,
            mode=0o444,
        )
        collector_paths = {
            "network": spec.network_collector_public_key_path,
            "cloud": spec.cloud_collector_public_key_path,
            "host": spec.host_collector_public_key_path,
        }
        if any(item is None for item in collector_paths.values()):
            raise OwnerGatePackageError(
                "owner_gate_package_collector_keys_required"
            )
        for name, path in collector_paths.items():
            digest, size = _sha256_file(path)  # type: ignore[arg-type]
            if size != 32 or digest != manifest["collector_public_key_ids"][name]:
                raise OwnerGatePackageError(
                    "owner_gate_package_collector_key_invalid"
                )
            _copy_exact(
                path,  # type: ignore[arg-type]
                temporary / "trust" / f"{name}-observation-attestation.pub",
                expected_sha256=digest,
                expected_size=size,
                mode=0o444,
            )
        if spec.credential_migration_envelope_path is None:
            raise OwnerGatePackageError(
                "owner_gate_package_migration_envelope_required"
            )
        migration_sha, migration_size = _sha256_file(
            spec.credential_migration_envelope_path
        )
        if migration_sha != manifest["credential_migration_envelope_sha256"]:
            raise OwnerGatePackageError(
                "owner_gate_package_migration_envelope_invalid"
            )
        _copy_exact(
            spec.credential_migration_envelope_path,
            temporary / "migration" / "credential.json",
            expected_sha256=migration_sha,
            expected_size=migration_size,
            mode=0o400,
        )
        if spec.direct_iam_identity_authority_path is None:
            raise OwnerGatePackageError(
                "owner_gate_package_direct_iam_identity_authority_required"
            )
        direct_iam_sha, direct_iam_size = _sha256_file(
            spec.direct_iam_identity_authority_path
        )
        if direct_iam_sha != manifest["direct_iam_identity_authority_sha256"]:
            raise OwnerGatePackageError(
                "owner_gate_package_direct_iam_identity_authority_invalid"
            )
        _copy_exact(
            spec.direct_iam_identity_authority_path,
            temporary / "trust/direct-iam-identity-authority.json",
            expected_sha256=direct_iam_sha,
            expected_size=direct_iam_size,
            mode=0o444,
        )
        _write_exact(
            temporary / "package-manifest.json",
            foundation.canonical_json_bytes(manifest),
            mode=0o444,
        )
        for directory in sorted(
            (item for item in temporary.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        temporary.chmod(0o555)
        os.replace(temporary, destination)
        directory_fd = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # The temporary path is private and was created by this process.  A
        # failed bundle is never renamed into the caller-visible destination.
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        if not raw or len(raw) > 4 * 1024 * 1024:
            raise ValueError
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        raise OwnerGatePackageError(
            "owner_gate_package_manifest_unavailable"
        ) from None
    if not isinstance(value, Mapping):
        raise OwnerGatePackageError("owner_gate_package_manifest_invalid")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-revision", required=True)
    parser.add_argument("--wheelhouse-root", type=Path, required=True)
    parser.add_argument("--wheelhouse-manifest", type=Path, required=True)
    parser.add_argument("--interpreter-sha256", required=True)
    parser.add_argument("--trust-manifest", type=Path, required=True)
    parser.add_argument("--trust-public-key", type=Path, required=True)
    parser.add_argument(
        "--network-collector-public-key", type=Path, required=True
    )
    parser.add_argument(
        "--cloud-collector-public-key", type=Path, required=True
    )
    parser.add_argument(
        "--host-collector-public-key", type=Path, required=True
    )
    parser.add_argument("--bundle-output", type=Path, required=True)
    parser.add_argument(
        "--credential-migration-envelope", type=Path, required=True
    )
    parser.add_argument(
        "--direct-iam-identity-authority", type=Path, required=True
    )
    arguments = parser.parse_args(argv)
    manifest = materialize_bundle(PackageSpec(
        source_root=arguments.source_root,
        release_revision=arguments.release_revision,
        wheelhouse_root=arguments.wheelhouse_root,
        wheelhouse_manifest=_load_json(arguments.wheelhouse_manifest),
        interpreter_sha256=arguments.interpreter_sha256,
        trust_manifest_path=arguments.trust_manifest,
        trust_public_key_path=arguments.trust_public_key,
        network_collector_public_key_path=(
            arguments.network_collector_public_key
        ),
        cloud_collector_public_key_path=arguments.cloud_collector_public_key,
        host_collector_public_key_path=arguments.host_collector_public_key,
        credential_migration_envelope_path=(
            arguments.credential_migration_envelope
        ),
        direct_iam_identity_authority_path=(
            arguments.direct_iam_identity_authority
        ),
    ), destination=arguments.bundle_output)
    print(foundation.canonical_json_bytes(manifest).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
