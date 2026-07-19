#!/usr/bin/env python3
"""Standard-library stage-zero verifier for the offline owner-gate bundle.

This file is the only Python executed before the pinned wheelhouse exists.  It
uses the exact Debian OpenSSL binary to verify the Ed25519 release signature,
then validates every bundle byte and the host capabilities needed to construct
the isolated venv.  It deliberately performs no Cloud, IAM, service, or DNS
mutation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence, cast


TRUST_SCHEMA = "muncho-owner-gate-release-trust.v2"
PACKAGE_SCHEMA = "muncho-owner-gate-offline-package.v2"
RUNTIME_LOCK_SCHEMA = "muncho-owner-gate-runtime-wheel-lock.v1"
RUNTIME_LOCK_RELATIVE = "ops/muncho/owner-gate/runtime-wheel-lock.json"
WHEELHOUSE_PLATFORM = "debian_12_x86_64"
PREFLIGHT_SCHEMA = "muncho-owner-gate-stage0-preflight.v1"
FORK_REPOSITORY = "lomliev/hermes-agent"
ATTESTATION_PURPOSE = "muncho_owner_gate_exact_offline_release_supply_chain"
PYTHON_VERSION = "3.11.2"
OPENSSL_VERSION_PREFIX = "OpenSSL 3.0.20 "
PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256 = (
    "302bd03b449a4f46476d9d2dc8026acedaca17334154ba2cf8ba2a68c72992a0"
)
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_DIRECT_IAM_BYTES = 8 * 1024 * 1024
MAX_PAYLOAD_BYTES = 128 * 1024 * 1024
OPENSSL = Path("/usr/bin/openssl")
PYTHON = Path("/usr/bin/python3")
OS_RELEASE = Path("/etc/os-release")
REQUIRED_EXECUTABLES = (
    Path("/usr/bin/openssl"),
    Path("/usr/bin/python3"),
    Path("/usr/bin/sha256sum"),
    Path("/usr/bin/systemctl"),
    Path("/usr/bin/systemd"),
    Path("/usr/bin/systemd-sysusers"),
    Path("/usr/bin/systemd-tmpfiles"),
    Path("/usr/sbin/iptables-nft"),
    Path("/usr/sbin/iptables-nft-save"),
    Path("/usr/sbin/visudo"),
)
RELEASE_BASE = Path("/opt/muncho-owner-gate/releases")
HOST_TRUSTED_OBSERVATION_RELEASE_BASE = Path(
    "/opt/muncho-trusted-observation/releases"
)
ALLOWED_OFFLINE_RELEASE_BASES = frozenset({
    RELEASE_BASE,
    HOST_TRUSTED_OBSERVATION_RELEASE_BASE,
})
INSTALL_RUNTIME_ENTRYPOINT = "bin/muncho-owner-gate-install"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_PROJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+!-]{0,127}$")
_WHEEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,254}\.whl$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_FOLDER_RESOURCE = re.compile(r"^folders/[1-9][0-9]{5,30}$")
_ORGANIZATION_RESOURCE = re.compile(
    r"^organizations/[1-9][0-9]{5,30}$"
)
_SPKI_ED25519_PREFIX = bytes.fromhex("302a300506032b6570032100")
_RFC8032_PUBLIC_KEY = bytes.fromhex(
    "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"
)
_RFC8032_MESSAGE = bytes.fromhex("72")
_RFC8032_SIGNATURE = bytes.fromhex(
    "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
    "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"
)
_BASE_COMMAND_ENV = {
    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
    "LC_ALL": "C.UTF-8",
}
_PIP_COMMAND_ENV = {
    **_BASE_COMMAND_ENV,
    "HOME": "/nonexistent",
    "PIP_CONFIG_FILE": "/dev/null",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_NO_INDEX": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": "",
}
_SIGNED_PIP_RUNNER = """import os
import runpy
import sys

wheel, venv, *arguments = sys.argv[1:]
stdlib = [
    "/usr/lib/python311.zip",
    "/usr/lib/python3.11",
    "/usr/lib/python3.11/lib-dynload",
]
if [os.path.realpath(item) for item in sys.path] != stdlib:
    raise SystemExit(86)
if not os.path.isabs(wheel) or not os.path.isabs(venv):
    raise SystemExit(87)
sys.path[:] = [wheel, *stdlib]
sys.prefix = venv
sys.exec_prefix = venv
sys.argv = ["pip", *arguments]
runpy.run_module("pip", run_name="__main__", alter_sys=True)
"""

INVENTORY_FIELDS = frozenset({
    "schema",
    "release_revision",
    "source_tree_oid",
    "foundation_source_revision",
    "foundation_source_tree_oid",
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
MANIFEST_FIELDS = INVENTORY_FIELDS | frozenset({
    "package_inventory_sha256",
    "trust_manifest_sha256",
    "trust_public_key_sha256",
    "interpreter_image",
    "release_supply_chain_attestation",
    "collector_public_key_ids",
    "credential_migration_envelope_sha256",
    "direct_iam_identity_authority_sha256",
    "pre_foundation_authority_sha256",
    "foundation_apply_receipt_sha256",
    "project_ancestry_evidence_sha256",
    "project_ancestry_chain_sha256",
    "caller_self_hash_is_authority",
    "package_sha256",
})
TRUST_FIELDS = frozenset({
    "schema",
    "approved_for_offline_install",
    "fork_repository",
    "release_revision",
    "source_tree_oid",
    "foundation_source_revision",
    "foundation_source_tree_oid",
    "package_inventory_sha256",
    "boot_image_self_link",
    "collector_public_key_ids",
    "credential_migration_envelope_sha256",
    "direct_iam_identity_authority_sha256",
    "pre_foundation_authority_sha256",
    "foundation_apply_receipt_sha256",
    "project_ancestry_evidence_sha256",
    "project_ancestry_chain_sha256",
    "resource_ancestor_chain",
    "interpreter_image",
    "release_attestation",
    "signer_key_id",
    "signature_ed25519_b64url",
})
RUNTIME_LOCK_FIELDS = frozenset({
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
RUNTIME_LOCK_WHEEL_FIELDS = frozenset({
    "project",
    "version",
    "filename",
    "sha256",
    "size",
    "active_dependencies",
})
SIGNED_WHEEL_FIELDS = frozenset({
    "filename",
    "project",
    "version",
    "sha256",
    "size",
})


class OwnerGateStage0Error(RuntimeError):
    """Stable, secret-free stage-zero failure."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError):
        raise OwnerGateStage0Error("owner_gate_stage0_json_invalid") from None


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_regular(
    path: Path,
    *,
    maximum: int,
    expected_uid: int,
    expected_gid: int | None = None,
    allowed_modes: frozenset[int],
) -> bytes:
    descriptor: int | None = None
    try:
        before = path.lstat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_nlink != 1
            or opened.st_uid != expected_uid
            or expected_gid is not None
            and opened.st_gid != expected_gid
            or stat.S_IMODE(opened.st_mode) not in allowed_modes
            or opened.st_size < 1
            or opened.st_size > maximum
        ):
            raise OwnerGateStage0Error("owner_gate_stage0_file_invalid")
        raw = bytearray()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise OwnerGateStage0Error("owner_gate_stage0_file_invalid")
            raw.extend(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise OwnerGateStage0Error("owner_gate_stage0_file_changed")
        return bytes(raw)
    except OSError:
        raise OwnerGateStage0Error("owner_gate_stage0_file_unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_exact_python_interpreter(
    python: Path,
    *,
    expected_uid: int = 0,
    expected_gid: int = 0,
    expected_link_mode: int = 0o777,
) -> bytes:
    """Read Debian's exact launcher/target pair without following ambiguity."""

    if not python.is_absolute() or python.name != "python3":
        raise OwnerGateStage0Error("owner_gate_stage0_python_identity_invalid")
    try:
        before = python.lstat()
        link_target = os.readlink(python)
        after = python.lstat()
    except OSError:
        raise OwnerGateStage0Error(
            "owner_gate_stage0_python_identity_invalid"
        ) from None
    if (
        not stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != expected_uid
        or before.st_gid != expected_gid
        or stat.S_IMODE(before.st_mode) != expected_link_mode
        or link_target != "python3.11"
        or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_python_identity_invalid")
    resolved = python.parent / "python3.11"
    try:
        if python.resolve(strict=True) != resolved:
            raise OwnerGateStage0Error(
                "owner_gate_stage0_python_identity_invalid"
            )
    except OSError:
        raise OwnerGateStage0Error(
            "owner_gate_stage0_python_identity_invalid"
        ) from None
    try:
        return _read_regular(
            resolved,
            maximum=MAX_PAYLOAD_BYTES,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            allowed_modes=frozenset({0o755}),
        )
    except OwnerGateStage0Error:
        raise OwnerGateStage0Error(
            "owner_gate_stage0_python_identity_invalid"
        ) from None


def _read_exact_os_release(
    os_release: Path,
    *,
    expected_uid: int = 0,
    expected_gid: int = 0,
) -> Mapping[str, str]:
    """Descriptor-stably parse the exact Debian 12 os-release identity."""

    target = os_release
    if os_release == OS_RELEASE:
        try:
            before = os_release.lstat()
            link_target = os.readlink(os_release)
            after = os_release.lstat()
        except OSError:
            raise OwnerGateStage0Error(
                "owner_gate_stage0_os_invalid"
            ) from None
        if (
            not stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != expected_uid
            or before.st_gid != expected_gid
            or stat.S_IMODE(before.st_mode) != 0o777
            or link_target != "../usr/lib/os-release"
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise OwnerGateStage0Error("owner_gate_stage0_os_invalid")
        target = Path("/usr/lib/os-release")
    try:
        raw = _read_regular(
            target,
            maximum=64 * 1024,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            allowed_modes=frozenset({0o444, 0o644}),
        )
        text = raw.decode("ascii", errors="strict")
    except (OwnerGateStage0Error, UnicodeError):
        raise OwnerGateStage0Error("owner_gate_stage0_os_invalid") from None
    if not text.endswith("\n") or "\x00" in text:
        raise OwnerGateStage0Error("owner_gate_stage0_os_invalid")
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, separator, raw_value = line.partition("=")
        if (
            separator != "="
            or re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is None
            or name in values
            or not raw_value
        ):
            raise OwnerGateStage0Error("owner_gate_stage0_os_invalid")
        if raw_value.startswith('"'):
            if (
                len(raw_value) < 2
                or not raw_value.endswith('"')
                or '"' in raw_value[1:-1]
                or "\\" in raw_value[1:-1]
            ):
                raise OwnerGateStage0Error("owner_gate_stage0_os_invalid")
            value = raw_value[1:-1]
        else:
            if re.fullmatch(r"[A-Za-z0-9._:/+@-]+", raw_value) is None:
                raise OwnerGateStage0Error("owner_gate_stage0_os_invalid")
            value = raw_value
        if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
            raise OwnerGateStage0Error("owner_gate_stage0_os_invalid")
        values[name] = value
    if values.get("ID") != "debian" or values.get("VERSION_ID") != "12":
        raise OwnerGateStage0Error("owner_gate_stage0_os_invalid")
    return values


def _capture_executable_identity(
    path: Path,
    *,
    expected_uid: int = 0,
    expected_gid: int = 0,
) -> Mapping[str, Any]:
    """Capture a stable source-path and opened-target executable identity."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise OwnerGateStage0Error(
            "owner_gate_stage0_executable_identity_invalid"
        )
    descriptor: int | None = None
    try:
        before = path.lstat()
        if (
            not (stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode))
            or before.st_nlink != 1
            or before.st_uid != expected_uid
            or before.st_gid != expected_gid
        ):
            raise OwnerGateStage0Error(
                "owner_gate_stage0_executable_identity_invalid"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        after_path = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != expected_uid
            or opened.st_gid != expected_gid
            or stat.S_IMODE(opened.st_mode) != 0o755
            or opened.st_size < 1
            or opened.st_size > MAX_PAYLOAD_BYTES
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after_path.st_dev,
                after_path.st_ino,
                after_path.st_size,
                after_path.st_mtime_ns,
                after_path.st_ctime_ns,
            )
        ):
            raise OwnerGateStage0Error(
                "owner_gate_stage0_executable_identity_invalid"
            )
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise OwnerGateStage0Error(
                    "owner_gate_stage0_executable_identity_invalid"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        after_open = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            after_open.st_dev,
            after_open.st_ino,
            after_open.st_size,
            after_open.st_mtime_ns,
            after_open.st_ctime_ns,
        ):
            raise OwnerGateStage0Error(
                "owner_gate_stage0_executable_identity_invalid"
            )
        return {
            "path": str(path),
            "source_device": before.st_dev,
            "source_inode": before.st_ino,
            "source_mode": stat.S_IMODE(before.st_mode),
            "target_device": opened.st_dev,
            "target_inode": opened.st_ino,
            "target_size": opened.st_size,
            "target_mtime_ns": opened.st_mtime_ns,
            "target_ctime_ns": opened.st_ctime_ns,
            "target_sha256": digest.hexdigest(),
        }
    except OwnerGateStage0Error:
        raise
    except OSError:
        raise OwnerGateStage0Error(
            "owner_gate_stage0_executable_identity_invalid"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _load_canonical_json(raw: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError):
        raise OwnerGateStage0Error("owner_gate_stage0_json_invalid") from None
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != raw:
        raise OwnerGateStage0Error("owner_gate_stage0_json_invalid")
    return value


def _b64url(value: Any, *, expected_size: int) -> bytes:
    if not isinstance(value, str) or not value or "=" in value:
        raise OwnerGateStage0Error("owner_gate_stage0_signature_invalid")
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError):
        raise OwnerGateStage0Error(
            "owner_gate_stage0_signature_invalid"
        ) from None
    if len(raw) != expected_size:
        raise OwnerGateStage0Error("owner_gate_stage0_signature_invalid")
    if base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value:
        raise OwnerGateStage0Error("owner_gate_stage0_signature_invalid")
    return raw


def _run(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> bytes:
    try:
        completed = subprocess.run(
            tuple(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
            env=dict(_BASE_COMMAND_ENV if env is None else env),
        )
    except (OSError, subprocess.SubprocessError):
        raise OwnerGateStage0Error("owner_gate_stage0_command_failed") from None
    if completed.returncode != 0 or len(completed.stdout) > MAX_JSON_BYTES:
        raise OwnerGateStage0Error("owner_gate_stage0_command_failed")
    return completed.stdout


def _assert_no_symlink_ancestors(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise OwnerGateStage0Error("owner_gate_stage0_staging_failed")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            state = current.lstat()
        except OSError:
            raise OwnerGateStage0Error(
                "owner_gate_stage0_staging_failed"
            ) from None
        if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
            raise OwnerGateStage0Error("owner_gate_stage0_staging_failed")


def _copy_verified_file(
    source: Path,
    destination: Path,
    *,
    sha256: str,
    size: int,
    mode: int,
    expected_uid: int = 0,
    expected_gid: int = 0,
    after_open: Callable[[], None] | None = None,
    after_write_chunk: Callable[[], None] | None = None,
    after_file_fsync: Callable[[], None] | None = None,
    after_publish_link: Callable[[], None] | None = None,
) -> None:
    """Copy one verified bundle file through a recoverable no-clobber stage.

    The final pathname is never opened for writing.  A crash may leave either
    a single-link partial stage or a two-link fully published inode.  Replay
    removes only a byte-for-byte prefix written by this operation, fsyncs a
    complete recovered stage before publishing it, and otherwise fails closed.
    """

    if (
        size < 1
        or size > MAX_PAYLOAD_BYTES
        or _SHA256.fullmatch(sha256) is None
        or mode not in {0o400, 0o440, 0o444, 0o500, 0o550, 0o555}
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_payload_invalid")
    source_descriptor: int | None = None
    try:
        before = source.lstat()
        source_descriptor = os.open(
            source,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(source_descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_nlink != 1
            or opened.st_size != size
        ):
            raise OwnerGateStage0Error("owner_gate_stage0_payload_invalid")
        digest = hashlib.sha256()
        payload = bytearray()
        remaining = size
        while remaining:
            chunk = os.read(source_descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise OwnerGateStage0Error("owner_gate_stage0_payload_invalid")
            digest.update(chunk)
            payload.extend(chunk)
            remaining -= len(chunk)
        after = os.fstat(source_descriptor)
        if (
            digest.hexdigest() != sha256
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise OwnerGateStage0Error("owner_gate_stage0_payload_invalid")
    except OSError:
        raise OwnerGateStage0Error("owner_gate_stage0_payload_invalid") from None
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
    _publish_exact_bytes(
        destination,
        bytes(payload),
        mode=mode,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        after_open=after_open,
        after_write_chunk=after_write_chunk,
        after_file_fsync=after_file_fsync,
        after_publish_link=after_publish_link,
    )


def _read_publish_inode(
    path: Path,
    *,
    maximum: int,
    mode: int,
    expected_uid: int,
    expected_gid: int,
    allowed_nlinks: frozenset[int],
) -> tuple[bytes, os.stat_result]:
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
            or opened.st_uid != expected_uid
            or opened.st_gid != expected_gid
            or stat.S_IMODE(opened.st_mode) != mode
            or opened.st_nlink not in allowed_nlinks
            or opened.st_size > maximum
        ):
            raise OwnerGateStage0Error("owner_gate_stage0_staging_conflict")
        raw = bytearray()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise OwnerGateStage0Error("owner_gate_stage0_staging_conflict")
            raw.extend(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise OwnerGateStage0Error("owner_gate_stage0_staging_conflict")
        return bytes(raw), after
    except OSError:
        raise OwnerGateStage0Error("owner_gate_stage0_staging_failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_exact_bytes(
    destination: Path,
    payload: bytes,
    *,
    mode: int,
    expected_uid: int = 0,
    expected_gid: int = 0,
    after_open: Callable[[], None] | None = None,
    after_write_chunk: Callable[[], None] | None = None,
    after_file_fsync: Callable[[], None] | None = None,
    after_publish_link: Callable[[], None] | None = None,
) -> None:
    """Crash-recoverable exact-byte publication without final-name writes."""

    if (
        not payload
        or len(payload) > MAX_PAYLOAD_BYTES
        or not destination.is_absolute()
        or ".." in destination.parts
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_staging_failed")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _assert_no_symlink_ancestors(destination.parent)
    parent_state = destination.parent.lstat()
    if (
        not stat.S_ISDIR(parent_state.st_mode)
        or parent_state.st_uid != expected_uid
        or parent_state.st_gid != expected_gid
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_staging_failed")
    staging = destination.parent / f".{destination.name}.stage0-staged"
    descriptor: int | None = None
    try:
        if os.path.lexists(destination):
            final_raw, final_state = _read_publish_inode(
                destination,
                maximum=len(payload),
                mode=mode,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
                allowed_nlinks=frozenset({1, 2}),
            )
            if final_raw != payload:
                raise OwnerGateStage0Error("owner_gate_stage0_staging_conflict")
            if os.path.lexists(staging):
                staged_raw, staged_state = _read_publish_inode(
                    staging,
                    maximum=len(payload),
                    mode=mode,
                    expected_uid=expected_uid,
                    expected_gid=expected_gid,
                    allowed_nlinks=frozenset({1, 2}),
                )
                same_inode = (staged_state.st_dev, staged_state.st_ino) == (
                    final_state.st_dev,
                    final_state.st_ino,
                )
                if staged_state.st_nlink == 2 and not same_inode:
                    raise OwnerGateStage0Error(
                        "owner_gate_stage0_staging_conflict"
                    )
                if staged_state.st_nlink == 1 and not payload.startswith(staged_raw):
                    raise OwnerGateStage0Error(
                        "owner_gate_stage0_staging_conflict"
                    )
                staging.unlink()
                _fsync_directory(destination.parent)
            final_raw, final_state = _read_publish_inode(
                destination,
                maximum=len(payload),
                mode=mode,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
                allowed_nlinks=frozenset({1}),
            )
            if final_raw != payload or final_state.st_nlink != 1:
                raise OwnerGateStage0Error("owner_gate_stage0_staging_conflict")
            return

        if os.path.lexists(staging):
            staged_removed = False
            try:
                staged_raw, staged_state = _read_publish_inode(
                    staging,
                    maximum=len(payload),
                    mode=mode,
                    expected_uid=expected_uid,
                    expected_gid=expected_gid,
                    allowed_nlinks=frozenset({1}),
                )
            except OwnerGateStage0Error:
                # open(O_EXCL) is itself crash-visible.  Before fchmod/fchown
                # the protected root-created stage can be empty with a
                # umask-reduced mode and root ownership.  No non-empty or
                # multi-link metadata mismatch is ever auto-repaired.
                state = staging.lstat()
                actual_mode = stat.S_IMODE(state.st_mode)
                if (
                    stat.S_ISLNK(state.st_mode)
                    or not stat.S_ISREG(state.st_mode)
                    or state.st_nlink != 1
                    or state.st_size != 0
                    or state.st_uid not in {0, expected_uid}
                    or state.st_gid not in {0, expected_gid}
                    or actual_mode & ~mode
                ):
                    raise
                staging.unlink()
                _fsync_directory(destination.parent)
                staged_removed = True
                staged_raw = b""
                staged_state = state
            if staged_removed:
                pass
            elif staged_raw == payload:
                recovered = os.open(
                    staging,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    os.fsync(recovered)
                finally:
                    os.close(recovered)
                _fsync_directory(destination.parent)
            elif len(staged_raw) < len(payload) and payload.startswith(staged_raw):
                if staged_state.st_nlink != 1:
                    raise OwnerGateStage0Error(
                        "owner_gate_stage0_staging_conflict"
                    )
                staging.unlink()
                _fsync_directory(destination.parent)
            else:
                raise OwnerGateStage0Error("owner_gate_stage0_staging_conflict")

        if not os.path.lexists(staging):
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(staging, flags, mode)
            if after_open is not None:
                after_open()
            os.fchmod(descriptor, mode)
            os.fchown(descriptor, expected_uid, expected_gid)
            view = memoryview(payload)
            while view:
                chunk = view[: min(len(view), 1024 * 1024)]
                written = os.write(descriptor, chunk)
                if written <= 0:
                    raise OSError
                view = view[written:]
                if after_write_chunk is not None:
                    after_write_chunk()
            os.fsync(descriptor)
            if after_file_fsync is not None:
                after_file_fsync()
            os.close(descriptor)
            descriptor = None
            _fsync_directory(destination.parent)

        try:
            os.link(staging, destination, follow_symlinks=False)
        except FileExistsError:
            pass
        if after_publish_link is not None:
            after_publish_link()
        _fsync_directory(destination.parent)
        final_raw, final_state = _read_publish_inode(
            destination,
            maximum=len(payload),
            mode=mode,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            allowed_nlinks=frozenset({1, 2}),
        )
        staged_raw, staged_state = _read_publish_inode(
            staging,
            maximum=len(payload),
            mode=mode,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            allowed_nlinks=frozenset({1, 2}),
        )
        if (
            final_raw != payload
            or staged_raw != payload
            or (
                staged_state.st_nlink == 2
                and (staged_state.st_dev, staged_state.st_ino)
                != (final_state.st_dev, final_state.st_ino)
            )
        ):
            raise OwnerGateStage0Error("owner_gate_stage0_staging_conflict")
        if staged_state.st_nlink == 1:
            # A concurrent exact publisher won.  Our completed single-link
            # stage is safe to discard only after the final bytes were proven.
            if final_state.st_nlink != 1:
                raise OwnerGateStage0Error("owner_gate_stage0_staging_conflict")
        staging.unlink()
        _fsync_directory(destination.parent)
    except OwnerGateStage0Error:
        raise
    except OSError:
        raise OwnerGateStage0Error("owner_gate_stage0_staging_failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)

    final_raw, final_state = _read_publish_inode(
        destination,
        maximum=len(payload),
        mode=mode,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        allowed_nlinks=frozenset({1}),
    )
    if final_raw != payload or final_state.st_nlink != 1:
        raise OwnerGateStage0Error("owner_gate_stage0_staging_conflict")


def _safe_remove_private_tree(path: Path, *, parent: Path) -> None:
    if path.parent != parent or path.is_symlink():
        raise OwnerGateStage0Error("owner_gate_stage0_staging_invalid")
    if not path.exists():
        return
    state = path.lstat()
    if not stat.S_ISDIR(state.st_mode) or state.st_uid != 0:
        raise OwnerGateStage0Error("owner_gate_stage0_staging_invalid")
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            target = os.readlink(candidate)
            if os.path.isabs(target) or ".." in Path(target).parts:
                raise OwnerGateStage0Error("owner_gate_stage0_staging_invalid")
    shutil.rmtree(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalize_project(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _runtime_lock_artifact_valid(item: object) -> bool:
    if not isinstance(item, Mapping) or set(item) != RUNTIME_LOCK_WHEEL_FIELDS:
        return False
    artifact = cast(Mapping[str, Any], item)
    project = artifact.get("project")
    version = artifact.get("version")
    filename = artifact.get("filename")
    dependencies = artifact.get("active_dependencies")
    size = artifact.get("size")
    pure = isinstance(filename, str) and filename.lower().endswith(
        "-py3-none-any.whl"
    )
    compiled = isinstance(filename, str) and bool(
        re.search(
            r"-cp311-(?:cp311|abi3)-"
            r"(?:manylinux_2_17_x86_64|manylinux2014_x86_64|"
            r"manylinux_2_28_x86_64)(?:\.[A-Za-z0-9_]+)*\.whl$",
            filename,
            flags=re.IGNORECASE,
        )
    )
    return bool(
        isinstance(project, str)
        and _PROJECT.fullmatch(project) is not None
        and _normalize_project(project) == project
        and isinstance(version, str)
        and _VERSION.fullmatch(version) is not None
        and isinstance(filename, str)
        and _WHEEL_NAME.fullmatch(filename) is not None
        and (pure or compiled)
        and type(size) is int
        and 0 < size <= MAX_PAYLOAD_BYTES
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
    )


def decode_runtime_lock(raw: bytes) -> Mapping[str, Any]:
    """Independently validate the release's exact Debian wheel authority."""

    try:
        if not raw or len(raw) > MAX_JSON_BYTES or not raw.endswith(b"\n"):
            raise ValueError
        value = _load_canonical_json(raw[:-1])
    except (OwnerGateStage0Error, ValueError):
        raise OwnerGateStage0Error(
            "owner_gate_stage0_runtime_lock_invalid"
        ) from None
    wheels = value.get("wheels")
    bootstrap_pip = value.get("bootstrap_pip")
    unsigned = {key: item for key, item in value.items() if key != "lock_sha256"}
    if (
        set(value) != RUNTIME_LOCK_FIELDS
        or value.get("schema") != RUNTIME_LOCK_SCHEMA
        or value.get("python_version") != PYTHON_VERSION
        or value.get("platform") != WHEELHOUSE_PLATFORM
        or value.get("network_required") is not False
        or value.get("source_build_allowed") is not False
        or value.get("complete_transitive_closure") is not True
        or not isinstance(wheels, list)
        or not wheels
        or len(wheels) > 256
        or not _runtime_lock_artifact_valid(bootstrap_pip)
        or _SHA256.fullmatch(str(value.get("lock_sha256", ""))) is None
        or value["lock_sha256"] != sha256_json(unsigned)
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_runtime_lock_invalid")
    bootstrap_pip = cast(Mapping[str, Any], bootstrap_pip)
    if (
        bootstrap_pip["project"] != "pip"
        or bootstrap_pip["active_dependencies"] != []
        or not bootstrap_pip["filename"].lower().endswith(
            "-py3-none-any.whl"
        )
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_runtime_lock_invalid")
    projects: set[str] = set()
    filenames: set[str] = set()
    for item in wheels:
        if not _runtime_lock_artifact_valid(item):
            raise OwnerGateStage0Error(
                "owner_gate_stage0_runtime_lock_invalid"
            )
        artifact = cast(Mapping[str, Any], item)
        project = artifact.get("project")
        filename = artifact.get("filename")
        if (
            project == "pip"
            or project in projects
            or filename in filenames
        ):
            raise OwnerGateStage0Error(
                "owner_gate_stage0_runtime_lock_invalid"
            )
        assert isinstance(project, str)
        assert isinstance(filename, str)
        projects.add(project)
        filenames.add(filename)
    if (
        bootstrap_pip["filename"] in filenames
        or [item["project"] for item in wheels] != sorted(projects)
        or any(
            dependency not in projects
            for item in wheels
            for dependency in item["active_dependencies"]
        )
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_runtime_lock_invalid")
    return dict(value)


def _verify_runtime_lock_payload(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_uid: int,
) -> Mapping[str, Any]:
    payload_items = [
        item
        for item in manifest.get("payloads", [])
        if isinstance(item, Mapping)
        and item.get("release_relative") == RUNTIME_LOCK_RELATIVE
    ]
    if len(payload_items) != 1:
        raise OwnerGateStage0Error("owner_gate_stage0_runtime_lock_invalid")
    payload_item = payload_items[0]
    raw = _read_regular(
        root / "payload" / RUNTIME_LOCK_RELATIVE,
        maximum=MAX_JSON_BYTES,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o444}),
    )
    file_digest = hashlib.sha256(raw).hexdigest()
    if (
        file_digest != manifest.get("runtime_lock_sha256")
        or payload_item.get("sha256") != file_digest
        or payload_item.get("size") != len(raw)
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_runtime_lock_invalid")
    runtime_lock = decode_runtime_lock(raw)
    manifest_wheels = manifest.get("wheels")
    if not isinstance(manifest_wheels, list) or any(
        not isinstance(item, Mapping)
        or set(item) != {"filename", "project", "version", "sha256", "size"}
        for item in manifest_wheels
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_runtime_lock_invalid")
    locked_artifacts = [
        {
            key: item[key]
            for key in ("filename", "project", "version", "sha256", "size")
        }
        for item in runtime_lock["wheels"]
    ]
    locked_bootstrap = {
        key: runtime_lock["bootstrap_pip"][key]
        for key in SIGNED_WHEEL_FIELDS
    }
    if sorted(manifest_wheels, key=lambda item: item["filename"]) != sorted(
        locked_artifacts,
        key=lambda item: item["filename"],
    ) or _bootstrap_pip_artifact(manifest) != locked_bootstrap:
        raise OwnerGateStage0Error("owner_gate_stage0_runtime_lock_invalid")
    return runtime_lock


def _bootstrap_pip_artifact(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    item = manifest.get("bootstrap_pip")
    if (
        not isinstance(item, Mapping)
        or set(item) != SIGNED_WHEEL_FIELDS
        or item.get("project") != "pip"
        or not isinstance(item.get("version"), str)
        or _VERSION.fullmatch(item["version"]) is None
        or not isinstance(item.get("filename"), str)
        or _WHEEL_NAME.fullmatch(item["filename"]) is None
        or not item["filename"].lower().endswith("-py3-none-any.whl")
        or _SHA256.fullmatch(str(item.get("sha256", ""))) is None
        or type(item.get("size")) is not int
        or not 0 < item["size"] <= MAX_PAYLOAD_BYTES
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_bootstrap_pip_invalid")
    return dict(item)


def _expected_distribution_inventory(
    manifest: Mapping[str, Any],
) -> Mapping[str, str]:
    wheels = manifest.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        raise OwnerGateStage0Error("owner_gate_stage0_distribution_invalid")
    expected: dict[str, str] = {}
    for item in wheels:
        if not isinstance(item, Mapping):
            raise OwnerGateStage0Error("owner_gate_stage0_distribution_invalid")
        project = item.get("project")
        version = item.get("version")
        if (
            not isinstance(project, str)
            or not project
            or not isinstance(version, str)
            or not version
        ):
            raise OwnerGateStage0Error("owner_gate_stage0_distribution_invalid")
        normalized = _normalize_project(project)
        if normalized in expected or normalized == "pip":
            raise OwnerGateStage0Error("owner_gate_stage0_distribution_invalid")
        expected[normalized] = version
    bootstrap_pip = _bootstrap_pip_artifact(manifest)
    if bootstrap_pip["project"] in expected:
        raise OwnerGateStage0Error("owner_gate_stage0_distribution_invalid")
    expected["pip"] = bootstrap_pip["version"]
    return dict(sorted(expected.items()))


def _expected_runtime_sys_path(venv: Path) -> list[str]:
    return [
        "/usr/lib/python311.zip",
        "/usr/lib/python3.11",
        "/usr/lib/python3.11/lib-dynload",
        str(venv / "lib/python3.11/site-packages"),
    ]


def _pip_command_environment() -> dict[str, str]:
    return dict(_PIP_COMMAND_ENV)


def _signed_pip_argv(
    venv: Path,
    bootstrap_pip_wheel: Path,
    arguments: Sequence[str],
) -> tuple[str, ...]:
    return (
        str(venv / "bin/python"),
        "-I",
        "-S",
        "-B",
        "-c",
        _SIGNED_PIP_RUNNER,
        str(bootstrap_pip_wheel),
        str(venv),
        *arguments,
    )


def _pip_install_argv(
    venv: Path,
    bootstrap_pip_wheel: Path,
    wheels: Sequence[str],
) -> tuple[str, ...]:
    return _signed_pip_argv(
        venv,
        bootstrap_pip_wheel,
        (
            "--isolated",
            "--disable-pip-version-check",
            "install",
            "--no-index",
            "--no-deps",
            "--only-binary=:all:",
            "--no-input",
            "--no-compile",
            str(bootstrap_pip_wheel),
            *wheels,
        ),
    )


def _wheel_site_packages_relative(name: str) -> PurePosixPath | None:
    path = PurePosixPath(name)
    parts = path.parts
    if not parts:
        return None
    if parts[0].casefold().endswith(".data"):
        if len(parts) < 3 or parts[1].casefold() not in {"purelib", "platlib"}:
            return None
        parts = parts[2:]
    return PurePosixPath(*parts) if parts else None


def _unsupported_wheel_data_destination(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return bool(
        parts
        and parts[0].casefold().endswith(".data")
        and (len(parts) < 3 or parts[1].casefold() not in {"purelib", "platlib"})
    )


def _startup_sensitive_site_packages_path(path: PurePosixPath) -> bool:
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


def _wheel_expected_site_packages(
    raw: bytes,
) -> tuple[dict[PurePosixPath, tuple[str, int] | None], set[PurePosixPath]]:
    """Derive installed files without importing or executing the wheel."""

    expected: dict[PurePosixPath, tuple[str, int] | None] = {}
    optional: set[PurePosixPath] = set()
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > 10_000:
                raise OwnerGateStage0Error(
                    "owner_gate_stage0_site_packages_invalid"
                )
            names: set[str] = set()
            total = 0
            dist_info_directories: set[PurePosixPath] = set()
            for entry in entries:
                name = entry.filename
                trimmed = name[:-1] if entry.is_dir() and name.endswith("/") else name
                raw_parts = trimmed.split("/") if trimmed else []
                mode = entry.external_attr >> 16
                node_type = stat.S_IFMT(mode)
                if (
                    not name
                    or name in names
                    or name.startswith("/")
                    or "\\" in name
                    or "\x00" in name
                    or not raw_parts
                    or any(part in {"", ".", ".."} for part in raw_parts)
                    or _unsupported_wheel_data_destination(trimmed)
                    or entry.flag_bits & 0x1
                    or entry.file_size < 0
                    or entry.file_size > MAX_PAYLOAD_BYTES
                    or (
                        entry.is_dir()
                        and node_type not in {0, stat.S_IFDIR}
                    )
                    or (
                        not entry.is_dir()
                        and node_type not in {0, stat.S_IFREG}
                    )
                ):
                    raise OwnerGateStage0Error(
                        "owner_gate_stage0_site_packages_invalid"
                    )
                names.add(name)
                total += entry.file_size
                if total > 512 * 1024 * 1024:
                    raise OwnerGateStage0Error(
                        "owner_gate_stage0_site_packages_invalid"
                    )
                destination = _wheel_site_packages_relative(trimmed)
                if destination is None:
                    continue
                if _startup_sensitive_site_packages_path(destination):
                    raise OwnerGateStage0Error(
                        "owner_gate_stage0_site_packages_invalid"
                    )
                if entry.is_dir():
                    continue
                if destination in expected:
                    raise OwnerGateStage0Error(
                        "owner_gate_stage0_site_packages_invalid"
                    )
                member = archive.read(entry)
                if len(member) != entry.file_size:
                    raise OwnerGateStage0Error(
                        "owner_gate_stage0_site_packages_invalid"
                    )
                mutable = (
                    destination.parent.name.casefold().endswith(".dist-info")
                    and destination.name == "RECORD"
                )
                expected[destination] = (
                    None
                    if mutable
                    else (hashlib.sha256(member).hexdigest(), len(member))
                )
                if (
                    destination.parent.name.casefold().endswith(".dist-info")
                    and destination.name == "METADATA"
                ):
                    dist_info_directories.add(destination.parent)
            if len(dist_info_directories) != 1:
                raise OwnerGateStage0Error(
                    "owner_gate_stage0_site_packages_invalid"
                )
            dist_info = next(iter(dist_info_directories))
            optional.update({
                dist_info / "INSTALLER",
                dist_info / "REQUESTED",
                dist_info / "direct_url.json",
            })
    except OwnerGateStage0Error:
        raise
    except (
        OSError,
        EOFError,
        KeyError,
        RuntimeError,
        NotImplementedError,
        zipfile.BadZipFile,
    ):
        raise OwnerGateStage0Error(
            "owner_gate_stage0_site_packages_invalid"
        ) from None
    return expected, optional


def _read_expected_wheel(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_uid: int,
) -> bytes:
    raw = _read_regular(
        path,
        maximum=MAX_PAYLOAD_BYTES,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o444, 0o644}),
    )
    if (
        len(raw) != expected_size
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_site_packages_invalid")
    return raw


def validate_wheel_archives_for_install(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    expected_uid: int = 0,
) -> None:
    """Parse all install inputs with Stage0 before any venv interpreter runs."""

    wheels = manifest.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        raise OwnerGateStage0Error("owner_gate_stage0_site_packages_invalid")
    for item in wheels:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("filename"), str)
            or _WHEEL_NAME.fullmatch(item["filename"]) is None
            or _SHA256.fullmatch(str(item.get("sha256", ""))) is None
            or type(item.get("size")) is not int
        ):
            raise OwnerGateStage0Error(
                "owner_gate_stage0_site_packages_invalid"
            )
        raw = _read_expected_wheel(
            root / "wheels" / item["filename"],
            expected_sha256=item["sha256"],
            expected_size=item["size"],
            expected_uid=expected_uid,
        )
        _wheel_expected_site_packages(raw)
    bootstrap_pip = _bootstrap_pip_artifact(manifest)
    pip_raw = _read_expected_wheel(
        root / "bootstrap" / bootstrap_pip["filename"],
        expected_sha256=bootstrap_pip["sha256"],
        expected_size=bootstrap_pip["size"],
        expected_uid=expected_uid,
    )
    _wheel_expected_site_packages(pip_raw)


def purge_generated_site_packages_bytecode(
    venv: Path,
    *,
    expected_uid: int = 0,
    expected_gid: int = 0,
) -> None:
    """Remove only verified interpreter-generated caches before tree sealing."""

    site_packages = venv / "lib/python3.11/site-packages"
    try:
        root_state = site_packages.lstat()
        if (
            not stat.S_ISDIR(root_state.st_mode)
            or root_state.st_uid != expected_uid
            or root_state.st_gid != expected_gid
            or stat.S_IMODE(root_state.st_mode) & 0o022
        ):
            raise OwnerGateStage0Error(
                "owner_gate_stage0_generated_bytecode_invalid"
            )
        cache_directories = tuple(sorted(
            (
                candidate
                for candidate in site_packages.rglob("*")
                if candidate.name == "__pycache__"
            ),
            key=lambda candidate: len(candidate.parts),
            reverse=True,
        ))
        cache_files: dict[Path, tuple[Path, ...]] = {}
        for cache in cache_directories:
            cache_state = cache.lstat()
            if (
                not stat.S_ISDIR(cache_state.st_mode)
                or cache_state.st_uid != expected_uid
                or cache_state.st_gid != expected_gid
                or stat.S_IMODE(cache_state.st_mode) & 0o022
            ):
                raise OwnerGateStage0Error(
                    "owner_gate_stage0_generated_bytecode_invalid"
                )
            children = tuple(cache.iterdir())
            for child in children:
                child_state = child.lstat()
                if (
                    not child.name.endswith(".pyc")
                    or not stat.S_ISREG(child_state.st_mode)
                    or child_state.st_nlink != 1
                    or child_state.st_uid != expected_uid
                    or child_state.st_gid != expected_gid
                    or stat.S_IMODE(child_state.st_mode) & 0o022
                ):
                    raise OwnerGateStage0Error(
                        "owner_gate_stage0_generated_bytecode_invalid"
                    )
            cache_files[cache] = children
        for cache in cache_directories:
            for child in cache_files[cache]:
                child.unlink()
            cache.rmdir()
            _fsync_directory(cache.parent)
    except OwnerGateStage0Error:
        raise
    except OSError:
        raise OwnerGateStage0Error(
            "owner_gate_stage0_generated_bytecode_invalid"
        ) from None


def seal_and_validate_venv_executables(
    venv: Path,
    *,
    interpreter_sha256: str,
    purge_generated_bin: bool,
    expected_uid: int = 0,
    expected_gid: int = 0,
) -> None:
    """Remove generated scripts, then prove the sole venv executable."""

    bin_root = venv / "bin"
    try:
        bin_state = bin_root.lstat()
        if (
            not stat.S_ISDIR(bin_state.st_mode)
            or bin_state.st_uid != expected_uid
            or bin_state.st_gid != expected_gid
            or stat.S_IMODE(bin_state.st_mode) & 0o022
        ):
            raise OwnerGateStage0Error(
                "owner_gate_stage0_venv_executable_invalid"
            )
        if purge_generated_bin:
            for candidate in tuple(bin_root.iterdir()):
                if candidate.name == "python":
                    continue
                state = candidate.lstat()
                if (
                    state.st_uid != expected_uid
                    or state.st_gid != expected_gid
                    or not (stat.S_ISREG(state.st_mode) or stat.S_ISLNK(state.st_mode))
                    or stat.S_ISREG(state.st_mode)
                    and state.st_nlink != 1
                ):
                    raise OwnerGateStage0Error(
                        "owner_gate_stage0_venv_executable_invalid"
                    )
                candidate.unlink()
            _fsync_directory(bin_root)
        entries = tuple(bin_root.iterdir())
        if len(entries) != 1 or entries[0].name != "python":
            raise OwnerGateStage0Error(
                "owner_gate_stage0_venv_executable_invalid"
            )
        interpreter = _read_regular(
            bin_root / "python",
            maximum=MAX_PAYLOAD_BYTES,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            allowed_modes=frozenset({0o555, 0o755}),
        )
        if (
            _SHA256.fullmatch(interpreter_sha256 or "") is None
            or hashlib.sha256(interpreter).hexdigest() != interpreter_sha256
        ):
            raise OwnerGateStage0Error(
                "owner_gate_stage0_venv_executable_invalid"
            )
        site_packages = venv / "lib/python3.11/site-packages"
        for candidate in venv.rglob("*"):
            state = candidate.lstat()
            if stat.S_ISLNK(state.st_mode):
                if candidate != venv / "lib64" or os.readlink(candidate) != "lib":
                    raise OwnerGateStage0Error(
                        "owner_gate_stage0_venv_executable_invalid"
                    )
                continue
            if (
                stat.S_ISREG(state.st_mode)
                and stat.S_IMODE(state.st_mode) & 0o111
                and candidate != bin_root / "python"
                and site_packages not in candidate.parents
            ):
                raise OwnerGateStage0Error(
                    "owner_gate_stage0_venv_executable_invalid"
                )
    except OwnerGateStage0Error:
        raise
    except OSError:
        raise OwnerGateStage0Error(
            "owner_gate_stage0_venv_executable_invalid"
        ) from None


def validate_installed_site_packages(
    root: Path,
    *,
    venv: Path,
    manifest: Mapping[str, Any],
    expected_uid: int = 0,
    expected_gid: int = 0,
) -> None:
    """Verify the installed tree with trusted Stage0 Python before launch."""

    expected: dict[PurePosixPath, tuple[str, int] | None] = {}
    optional: set[PurePosixPath] = set()
    wheels = manifest.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        raise OwnerGateStage0Error("owner_gate_stage0_site_packages_invalid")
    for item in wheels:
        if not isinstance(item, Mapping):
            raise OwnerGateStage0Error(
                "owner_gate_stage0_site_packages_invalid"
            )
        filename = item.get("filename")
        if (
            not isinstance(filename, str)
            or _WHEEL_NAME.fullmatch(filename) is None
            or _SHA256.fullmatch(str(item.get("sha256", ""))) is None
            or type(item.get("size")) is not int
        ):
            raise OwnerGateStage0Error(
                "owner_gate_stage0_site_packages_invalid"
            )
        raw = _read_expected_wheel(
            root / "wheels" / filename,
            expected_sha256=item["sha256"],
            expected_size=item["size"],
            expected_uid=expected_uid,
        )
        wheel_expected, wheel_optional = _wheel_expected_site_packages(raw)
        if set(expected) & set(wheel_expected):
            raise OwnerGateStage0Error(
                "owner_gate_stage0_site_packages_invalid"
            )
        expected.update(wheel_expected)
        optional.update(wheel_optional)
    bootstrap_pip = _bootstrap_pip_artifact(manifest)
    pip_raw = _read_expected_wheel(
        root / "bootstrap" / bootstrap_pip["filename"],
        expected_sha256=bootstrap_pip["sha256"],
        expected_size=bootstrap_pip["size"],
        expected_uid=expected_uid,
    )
    pip_expected, pip_optional = _wheel_expected_site_packages(pip_raw)
    if set(expected) & set(pip_expected):
        raise OwnerGateStage0Error("owner_gate_stage0_site_packages_invalid")
    expected.update(pip_expected)
    optional.update(pip_optional)

    site_packages = venv / "lib/python3.11/site-packages"
    try:
        root_state = site_packages.lstat()
    except OSError:
        raise OwnerGateStage0Error(
            "owner_gate_stage0_site_packages_invalid"
        ) from None
    if (
        not stat.S_ISDIR(root_state.st_mode)
        or root_state.st_uid != expected_uid
        or root_state.st_gid != expected_gid
        or stat.S_IMODE(root_state.st_mode) & 0o022
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_site_packages_invalid")
    required_directories: set[PurePosixPath] = set()
    for path in (*expected, *optional):
        parent = path.parent
        while parent != PurePosixPath("."):
            required_directories.add(parent)
            parent = parent.parent
    observed_files: set[PurePosixPath] = set()
    observed_directories: set[PurePosixPath] = set()
    try:
        candidates = sorted(site_packages.rglob("*"), key=lambda item: str(item))
        for candidate in candidates:
            before = candidate.lstat()
            relative = PurePosixPath(candidate.relative_to(site_packages).as_posix())
            if (
                stat.S_ISLNK(before.st_mode)
                or before.st_uid != expected_uid
                or before.st_gid != expected_gid
                or stat.S_IMODE(before.st_mode) & 0o022
                or _startup_sensitive_site_packages_path(relative)
            ):
                raise OwnerGateStage0Error(
                    "owner_gate_stage0_site_packages_invalid"
                )
            if stat.S_ISDIR(before.st_mode):
                observed_directories.add(relative)
                continue
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise OwnerGateStage0Error(
                    "owner_gate_stage0_site_packages_invalid"
                )
            if relative not in expected and relative not in optional:
                raise OwnerGateStage0Error(
                    "owner_gate_stage0_site_packages_invalid"
                )
            descriptor = os.open(
                candidate,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                opened = os.fstat(descriptor)
                digest = hashlib.sha256()
                while chunk := os.read(descriptor, 1024 * 1024):
                    digest.update(chunk)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if (
                (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
                or (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                )
                != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
            ):
                raise OwnerGateStage0Error(
                    "owner_gate_stage0_site_packages_invalid"
                )
            identity = expected.get(relative)
            if identity is not None and (
                opened.st_size != identity[1]
                or digest.hexdigest() != identity[0]
            ):
                raise OwnerGateStage0Error(
                    "owner_gate_stage0_site_packages_invalid"
                )
            observed_files.add(relative)
    except OwnerGateStage0Error:
        raise
    except OSError:
        raise OwnerGateStage0Error(
            "owner_gate_stage0_site_packages_invalid"
        ) from None
    if (
        not set(expected).issubset(observed_files)
        or not observed_files.issubset(set(expected) | optional)
        or observed_directories != required_directories
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_site_packages_invalid")


def _runtime_inventory_probe_code() -> str:
    return """import importlib.metadata as metadata
import json
import os
import re
import sys

def normalize(value):
    return re.sub(r"[-_.]+", "-", value).lower()

distributions = []
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    distributions.append({
        "project": normalize(name) if isinstance(name, str) else "",
        "version": distribution.version,
    })
report = {
    "base_prefix": os.path.realpath(sys.base_prefix),
    "distributions": sorted(
        distributions,
        key=lambda item: (item["project"], item["version"]),
    ),
    "executable": os.path.realpath(sys.executable),
    "flags": {
        "ignore_environment": sys.flags.ignore_environment,
        "isolated": sys.flags.isolated,
        "no_user_site": sys.flags.no_user_site,
        "safe_path": sys.flags.safe_path,
    },
    "prefix": os.path.realpath(sys.prefix),
    "sys_path": [os.path.realpath(item) for item in sys.path],
}
print(json.dumps(
    report,
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
))
"""


def validate_runtime_inventory(
    raw: bytes,
    *,
    venv: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = _load_canonical_json(raw.rstrip(b"\n"))
    if set(value) != {
        "base_prefix",
        "distributions",
        "executable",
        "flags",
        "prefix",
        "sys_path",
    }:
        raise OwnerGateStage0Error("owner_gate_stage0_runtime_inventory_invalid")
    distributions = value.get("distributions")
    if not isinstance(distributions, list):
        raise OwnerGateStage0Error("owner_gate_stage0_runtime_inventory_invalid")
    actual: dict[str, str] = {}
    for item in distributions:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"project", "version"}
            or not isinstance(item.get("project"), str)
            or not item["project"]
            or item["project"] != _normalize_project(item["project"])
            or not isinstance(item.get("version"), str)
            or not item["version"]
            or item["project"] in actual
        ):
            raise OwnerGateStage0Error(
                "owner_gate_stage0_runtime_inventory_invalid"
            )
        actual[item["project"]] = item["version"]
    if (
        dict(sorted(actual.items())) != _expected_distribution_inventory(manifest)
        or value.get("sys_path") != _expected_runtime_sys_path(venv)
        or value.get("executable") != str(venv / "bin/python")
        or value.get("prefix") != str(venv)
        or value.get("base_prefix") != "/usr"
        or value.get("flags") != {
            "ignore_environment": 1,
            "isolated": 1,
            "no_user_site": 1,
            "safe_path": True,
        }
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_runtime_inventory_invalid")
    return value


def _offline_runtime_marker(
    manifest: Mapping[str, Any],
    *,
    venv: Path,
) -> bytes:
    bootstrap_pip = _bootstrap_pip_artifact(manifest)
    return canonical_json_bytes({
        "schema": "muncho-owner-gate-offline-venv.v1",
        "package_sha256": manifest["package_sha256"],
        "python_version": PYTHON_VERSION,
        "bootstrap_pip": bootstrap_pip,
        "expected_distributions": _expected_distribution_inventory(manifest),
        "expected_sys_path": _expected_runtime_sys_path(venv),
        "wheels": [
            {
                "filename": item["filename"],
                "project": item["project"],
                "version": item["version"],
                "sha256": item["sha256"],
            }
            for item in manifest["wheels"]
        ],
    })


def prepare_offline_runtime(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    release_base: Path = RELEASE_BASE,
    runner: Callable[..., bytes] = _run,
) -> Path:
    """Create/replay the exact staging release and its no-network venv."""

    revision = str(manifest["release_revision"])
    if (
        _REVISION.fullmatch(revision) is None
        or release_base not in ALLOWED_OFFLINE_RELEASE_BASES
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_release_path_invalid")
    validate_wheel_archives_for_install(root, manifest=manifest)
    release_base.mkdir(parents=True, exist_ok=True, mode=0o755)
    base_state = release_base.lstat()
    if (
        not stat.S_ISDIR(base_state.st_mode)
        or base_state.st_uid != 0
        or stat.S_IMODE(base_state.st_mode) != 0o755
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_release_path_invalid")
    final = release_base / revision
    staging = release_base / f".{revision}.bootstrap"
    if final.exists():
        state = final.lstat()
        if (
            not stat.S_ISDIR(state.st_mode)
            or state.st_uid != 0
            or stat.S_IMODE(state.st_mode) != 0o555
        ):
            raise OwnerGateStage0Error("owner_gate_stage0_release_conflict")
        seal_and_validate_venv_executables(
            final / "venv",
            interpreter_sha256=manifest["interpreter_sha256"],
            purge_generated_bin=False,
        )
        validate_installed_site_packages(
            root,
            venv=final / "venv",
            manifest=manifest,
        )
        inventory_raw = runner(
            (
                str(final / "venv/bin/python"),
                "-I",
                "-B",
                "-c",
                _runtime_inventory_probe_code(),
            ),
            env=_pip_command_environment(),
        )
        validate_runtime_inventory(
            inventory_raw,
            venv=final / "venv",
            manifest=manifest,
        )
        return final
    if not staging.exists():
        staging.mkdir(mode=0o700)
        os.chown(staging, 0, 0)
        _fsync_directory(release_base)
    else:
        state = staging.lstat()
        if (
            not stat.S_ISDIR(state.st_mode)
            or state.st_uid != 0
            or stat.S_IMODE(state.st_mode) not in {0o700, 0o555}
        ):
            raise OwnerGateStage0Error("owner_gate_stage0_staging_invalid")
        if stat.S_IMODE(state.st_mode) == 0o555:
            staging.chmod(0o700)
    for item in manifest["payloads"]:
        relative = _safe_relative(item["release_relative"])
        _copy_verified_file(
            root / "payload" / relative,
            staging / relative,
            sha256=item["sha256"],
            size=item["size"],
            mode=int(item["mode"], 8),
        )
    # Preserve the exact signed package authority beside the immutable runtime.
    # Provisioning later derives a private key's public identity and compares it
    # to these release-pinned bytes without needing the transferred bundle.
    authority_files = (
        "package-manifest.json",
        "trust/release-trust.json",
        "trust/release-trust-signing.pub",
        "trust/network-observation-attestation.pub",
        "trust/cloud-observation-attestation.pub",
        "trust/host-observation-attestation.pub",
        "trust/direct-iam-identity-authority.json",
    )
    for relative_text in authority_files:
        relative = _safe_relative(relative_text)
        source = root / relative
        raw = _read_regular(
            source,
            maximum=MAX_PAYLOAD_BYTES,
            expected_uid=0,
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
        _copy_verified_file(
            source,
            staging / relative,
            sha256=hashlib.sha256(raw).hexdigest(),
            size=len(raw),
            mode=0o444,
        )
    wheel_stage = staging / ".bootstrap-wheelhouse"
    wheel_stage.mkdir(exist_ok=True, mode=0o700)
    for item in manifest["wheels"]:
        relative = _safe_relative(item["filename"])
        _copy_verified_file(
            root / "wheels" / relative,
            wheel_stage / relative,
            sha256=item["sha256"],
            size=item["size"],
            mode=0o444,
        )
    bootstrap_pip = _bootstrap_pip_artifact(manifest)
    bootstrap_pip_stage = wheel_stage / bootstrap_pip["filename"]
    _copy_verified_file(
        root / "bootstrap" / bootstrap_pip["filename"],
        bootstrap_pip_stage,
        sha256=bootstrap_pip["sha256"],
        size=bootstrap_pip["size"],
        mode=0o444,
    )
    marker = staging / ".bootstrap-wheelhouse-installed.json"
    venv = staging / "venv"
    expected_marker = _offline_runtime_marker(manifest, venv=venv)
    marker_valid = False
    marker_stage = marker.parent / f".{marker.name}.stage0-staged"
    if os.path.lexists(marker) or os.path.lexists(marker_stage):
        _publish_exact_bytes(marker, expected_marker, mode=0o400)
        raw = _read_regular(
            marker,
            maximum=MAX_JSON_BYTES,
            expected_uid=0,
            allowed_modes=frozenset({0o400}),
        )
        marker_valid = raw == expected_marker and (venv / "bin/python").is_file()
        if not marker_valid:
            raise OwnerGateStage0Error("owner_gate_stage0_venv_marker_invalid")
    if not marker_valid:
        _safe_remove_private_tree(venv, parent=staging)
        runner(
            (
                str(PYTHON),
                "-I",
                "-B",
                "-m",
                "venv",
                "--without-pip",
                "--copies",
                str(venv),
            ),
            env=_pip_command_environment(),
        )
        wheels = tuple(str(wheel_stage / item["filename"]) for item in manifest["wheels"])
        runner(
            _pip_install_argv(venv, bootstrap_pip_stage, wheels),
            env=_pip_command_environment(),
        )
        purge_generated_site_packages_bytecode(venv)
    seal_and_validate_venv_executables(
        venv,
        interpreter_sha256=manifest["interpreter_sha256"],
        purge_generated_bin=not marker_valid,
    )
    validate_installed_site_packages(
        root,
        venv=venv,
        manifest=manifest,
    )
    inventory_raw = runner(
        (
            str(venv / "bin/python"),
            "-I",
            "-B",
            "-c",
            _runtime_inventory_probe_code(),
        ),
        env=_pip_command_environment(),
    )
    validate_runtime_inventory(
        inventory_raw,
        venv=venv,
        manifest=manifest,
    )
    if not marker_valid:
        _publish_exact_bytes(
            marker,
            expected_marker,
            mode=0o400,
        )
    return staging


def invoke_runtime_installer(
    staging_or_release: Path,
    bundle: Path,
    *,
    runner: Callable[[Sequence[str]], bytes] = _run,
) -> Mapping[str, Any]:
    entrypoint = staging_or_release / INSTALL_RUNTIME_ENTRYPOINT
    output = runner((
        str(staging_or_release / "venv/bin/python"),
        "-I",
        "-B",
        str(entrypoint),
        "install-after-stage0",
        "--bundle",
        str(bundle),
    ))
    return _load_canonical_json(output.rstrip(b"\n"))


def verify_ed25519_with_openssl(
    *,
    public_key: bytes,
    signature: bytes,
    message: bytes,
    runner: Callable[[Sequence[str]], bytes] = _run,
    temporary_root: Path | None = None,
) -> None:
    if len(public_key) != 32 or len(signature) != 64:
        raise OwnerGateStage0Error("owner_gate_stage0_signature_invalid")
    try:
        with tempfile.TemporaryDirectory(
            prefix=".muncho-owner-gate-stage0-",
            dir=str(temporary_root) if temporary_root is not None else None,
        ) as directory:
            root = Path(directory)
            paths = {
                "key": root / "public.der",
                "signature": root / "signature.bin",
                "message": root / "message.bin",
            }
            payloads = {
                "key": _SPKI_ED25519_PREFIX + public_key,
                "signature": signature,
                "message": message,
            }
            for name, path in paths.items():
                descriptor = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                try:
                    view = memoryview(payloads[name])
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise OSError
                        view = view[written:]
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            runner((
                str(OPENSSL),
                "pkeyutl",
                "-verify",
                "-pubin",
                "-keyform",
                "DER",
                "-inkey",
                str(paths["key"]),
                "-rawin",
                "-in",
                str(paths["message"]),
                "-sigfile",
                str(paths["signature"]),
            ))
    except OSError:
        raise OwnerGateStage0Error("owner_gate_stage0_signature_invalid") from None


def _validate_trust(value: Mapping[str, Any], public_key: bytes) -> Mapping[str, Any]:
    if set(value) != TRUST_FIELDS:
        raise OwnerGateStage0Error("owner_gate_stage0_trust_invalid")
    unsigned = {
        key: item for key, item in value.items() if key != "signature_ed25519_b64url"
    }
    collectors = unsigned.get("collector_public_key_ids")
    image = unsigned.get("interpreter_image")
    release_attestation = unsigned.get("release_attestation")
    resource_ancestor_chain = unsigned.get("resource_ancestor_chain")
    if (
        unsigned.get("schema") != TRUST_SCHEMA
        or unsigned.get("approved_for_offline_install") is not True
        or unsigned.get("fork_repository") != FORK_REPOSITORY
        or _REVISION.fullmatch(str(unsigned.get("release_revision", ""))) is None
        or _REVISION.fullmatch(str(unsigned.get("source_tree_oid", ""))) is None
        or _REVISION.fullmatch(
            str(unsigned.get("foundation_source_revision", ""))
        )
        is None
        or _REVISION.fullmatch(
            str(unsigned.get("foundation_source_tree_oid", ""))
        )
        is None
        or unsigned.get("foundation_source_revision")
        == unsigned.get("release_revision")
        or _SHA256.fullmatch(
            str(unsigned.get("package_inventory_sha256", ""))
        )
        is None
        or not isinstance(unsigned.get("boot_image_self_link"), str)
        or re.fullmatch(
            r"projects/debian-cloud/global/images/debian-12-bookworm-v[0-9]{8}",
            unsigned["boot_image_self_link"],
        )
        is None
        or not isinstance(collectors, Mapping)
        or set(collectors) != {"network", "cloud", "host"}
        or any(_SHA256.fullmatch(str(item)) is None for item in collectors.values())
        or len(set(collectors.values())) != 3
        or _SHA256.fullmatch(
            str(unsigned.get("credential_migration_envelope_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(unsigned.get("direct_iam_identity_authority_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(unsigned.get("pre_foundation_authority_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(unsigned.get("foundation_apply_receipt_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(unsigned.get("project_ancestry_evidence_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(unsigned.get("project_ancestry_chain_sha256", ""))
        )
        is None
        or not isinstance(resource_ancestor_chain, list)
        or not resource_ancestor_chain
        or len(resource_ancestor_chain) > 31
        or any(not isinstance(item, str) for item in resource_ancestor_chain)
        or len(resource_ancestor_chain) != len(set(resource_ancestor_chain))
        or _ORGANIZATION_RESOURCE.fullmatch(resource_ancestor_chain[-1]) is None
        or any(
            _FOLDER_RESOURCE.fullmatch(item) is None
            for item in resource_ancestor_chain[:-1]
        )
        or unsigned.get("signer_key_id") != hashlib.sha256(public_key).hexdigest()
        or not isinstance(image, Mapping)
        or set(image)
        != {
            "project",
            "image_name",
            "image_numeric_id",
            "image_self_link",
            "python_version",
            "interpreter_sha256",
        }
        or image.get("project") != "debian-cloud"
        or not isinstance(image.get("image_name"), str)
        or not image["image_name"]
        or _NUMERIC_ID.fullmatch(str(image.get("image_numeric_id", ""))) is None
        or not isinstance(image.get("image_self_link"), str)
        or image.get("python_version") != PYTHON_VERSION
        or _SHA256.fullmatch(str(image.get("interpreter_sha256", ""))) is None
        or image.get("image_self_link")
        != "https://www.googleapis.com/compute/v1/"
        + str(unsigned.get("boot_image_self_link", ""))
        or image["image_name"]
        != unsigned["boot_image_self_link"].rsplit("/", 1)[-1]
        or not isinstance(release_attestation, Mapping)
        or set(release_attestation) != {"purpose", "attested_at_unix"}
        or release_attestation.get("purpose") != ATTESTATION_PURPOSE
        or type(release_attestation.get("attested_at_unix")) is not int
        or release_attestation["attested_at_unix"] <= 0
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_trust_invalid")
    return unsigned


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str):
        raise OwnerGateStage0Error("owner_gate_stage0_manifest_invalid")
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise OwnerGateStage0Error("owner_gate_stage0_manifest_invalid")
    return path


def _verify_manifest_and_payloads(
    root: Path,
    *,
    manifest_raw: bytes,
    authority: Mapping[str, Any],
    trust_raw: bytes,
    public_key: bytes,
    expected_uid: int,
) -> Mapping[str, Any]:
    manifest = _load_canonical_json(manifest_raw)
    if set(manifest) != MANIFEST_FIELDS:
        raise OwnerGateStage0Error("owner_gate_stage0_manifest_invalid")
    inventory = {key: manifest[key] for key in INVENTORY_FIELDS}
    unsigned = {key: item for key, item in manifest.items() if key != "package_sha256"}
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("package_sha256") != sha256_json(unsigned)
        or manifest.get("package_inventory_sha256") != sha256_json(inventory)
        or manifest["package_inventory_sha256"]
        != authority["package_inventory_sha256"]
        or manifest.get("trust_manifest_sha256")
        != hashlib.sha256(trust_raw).hexdigest()
        or manifest.get("trust_public_key_sha256")
        != hashlib.sha256(public_key).hexdigest()
        or manifest.get("release_revision") != authority["release_revision"]
        or manifest.get("source_tree_oid") != authority["source_tree_oid"]
        or manifest.get("foundation_source_revision")
        != authority["foundation_source_revision"]
        or manifest.get("foundation_source_tree_oid")
        != authority["foundation_source_tree_oid"]
        or manifest.get("foundation_source_revision")
        == manifest.get("release_revision")
        or manifest.get("collector_public_key_ids")
        != authority["collector_public_key_ids"]
        or manifest.get("credential_migration_envelope_sha256")
        != authority["credential_migration_envelope_sha256"]
        or manifest.get("direct_iam_identity_authority_sha256")
        != authority["direct_iam_identity_authority_sha256"]
        or manifest.get("pre_foundation_authority_sha256")
        != authority["pre_foundation_authority_sha256"]
        or manifest.get("foundation_apply_receipt_sha256")
        != authority["foundation_apply_receipt_sha256"]
        or manifest.get("project_ancestry_evidence_sha256")
        != authority["project_ancestry_evidence_sha256"]
        or manifest.get("project_ancestry_chain_sha256")
        != authority["project_ancestry_chain_sha256"]
        or manifest.get("resource_ancestor_chain")
        != authority["resource_ancestor_chain"]
        or manifest.get("interpreter_image") != authority["interpreter_image"]
        or manifest.get("interpreter_sha256")
        != authority["interpreter_image"]["interpreter_sha256"]
        or manifest.get("interpreter_version") != PYTHON_VERSION
        or manifest.get("release_root")
        != f"/opt/muncho-owner-gate/releases/{manifest['release_revision']}"
        or manifest.get("release_owner") != "root:root"
        or manifest.get("release_directory_mode") != "0555"
        or manifest.get("immutable_after_install") is not True
        or manifest.get("offline_bootstrap") is not True
        or manifest.get("network_install_required") is not False
        or manifest.get("generic_shell_entrypoint") is not False
        or manifest.get("local_gcloud_runtime_fallback") is not False
        or manifest.get("secret_material_recorded") is not False
        or manifest.get("secret_digest_recorded") is not False
        or manifest.get("activation_performed") is not False
        or manifest.get("cloud_mutation_performed") is not False
        or manifest.get("caller_self_hash_is_authority") is not False
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_manifest_invalid")
    seen: set[Path] = set()
    for item in (*manifest.get("payloads", []), *manifest.get("wheels", [])):
        if not isinstance(item, Mapping):
            raise OwnerGateStage0Error("owner_gate_stage0_manifest_invalid")
        relative = (
            Path("payload") / _safe_relative(item.get("release_relative"))
            if "release_relative" in item
            else Path("wheels") / _safe_relative(item.get("filename"))
        )
        if relative in seen:
            raise OwnerGateStage0Error("owner_gate_stage0_manifest_invalid")
        seen.add(relative)
        digest = item.get("sha256")
        size = item.get("size")
        if (
            _SHA256.fullmatch(str(digest or "")) is None
            or type(size) is not int
            or size <= 0
            or size > MAX_PAYLOAD_BYTES
        ):
            raise OwnerGateStage0Error("owner_gate_stage0_manifest_invalid")
        allowed_modes = (
            frozenset({int(str(item.get("mode")), 8)})
            if "release_relative" in item
            and str(item.get("mode")) in {"0444", "0555"}
            else frozenset({0o444})
        )
        raw = _read_regular(
            root / relative,
            maximum=MAX_PAYLOAD_BYTES,
            expected_uid=expected_uid,
            allowed_modes=allowed_modes,
        )
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise OwnerGateStage0Error("owner_gate_stage0_payload_invalid")
    bootstrap_pip = _bootstrap_pip_artifact(manifest)
    bootstrap_relative = (
        Path("bootstrap") / _safe_relative(bootstrap_pip["filename"])
    )
    if bootstrap_relative in seen:
        raise OwnerGateStage0Error("owner_gate_stage0_manifest_invalid")
    bootstrap_raw = _read_regular(
        root / bootstrap_relative,
        maximum=MAX_PAYLOAD_BYTES,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o444}),
    )
    if (
        len(bootstrap_raw) != bootstrap_pip["size"]
        or hashlib.sha256(bootstrap_raw).hexdigest()
        != bootstrap_pip["sha256"]
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_payload_invalid")
    _verify_runtime_lock_payload(
        root,
        manifest,
        expected_uid=expected_uid,
    )
    for name, key_id in authority["collector_public_key_ids"].items():
        raw = _read_regular(
            root / "trust" / f"{name}-observation-attestation.pub",
            maximum=32,
            expected_uid=expected_uid,
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
        if len(raw) != 32 or hashlib.sha256(raw).hexdigest() != key_id:
            raise OwnerGateStage0Error("owner_gate_stage0_collector_key_invalid")
    migration = _read_regular(
        root / "migration/credential.json",
        maximum=MAX_JSON_BYTES,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
    )
    if (
        hashlib.sha256(migration).hexdigest()
        != authority["credential_migration_envelope_sha256"]
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_migration_invalid")
    direct_iam = _read_regular(
        root / "trust/direct-iam-identity-authority.json",
        maximum=MAX_DIRECT_IAM_BYTES,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
    )
    if (
        hashlib.sha256(direct_iam).hexdigest()
        != authority["direct_iam_identity_authority_sha256"]
    ):
        raise OwnerGateStage0Error(
            "owner_gate_stage0_direct_iam_identity_authority_invalid"
        )
    direct_iam_value = _load_canonical_json(direct_iam)
    if (
        direct_iam_value.get("schema")
        != "muncho-owner-gate-direct-iam-identity-authority.v1"
        or direct_iam_value.get("release_revision")
        != authority["foundation_source_revision"]
        or direct_iam_value.get("release_revision")
        != manifest["foundation_source_revision"]
        or direct_iam_value.get("pre_foundation_authority_sha256")
        != authority["pre_foundation_authority_sha256"]
        or direct_iam_value.get("pre_foundation_authority_sha256")
        != manifest["pre_foundation_authority_sha256"]
        or direct_iam_value.get("foundation_apply_receipt_sha256")
        != authority["foundation_apply_receipt_sha256"]
        or direct_iam_value.get("foundation_apply_receipt_sha256")
        != manifest["foundation_apply_receipt_sha256"]
        or direct_iam_value.get("resource_ancestor_chain")
        != authority["resource_ancestor_chain"]
        or direct_iam_value.get("resource_ancestor_chain")
        != manifest["resource_ancestor_chain"]
    ):
        raise OwnerGateStage0Error(
            "owner_gate_stage0_direct_iam_identity_authority_invalid"
        )
    return manifest


def verify_bundle_stage0(
    root: Path,
    *,
    expected_uid: int = 0,
    runner: Callable[[Sequence[str]], bytes] = _run,
    temporary_root: Path | None = None,
) -> Mapping[str, Any]:
    if not root.is_absolute() or ".." in root.parts or not root.is_dir():
        raise OwnerGateStage0Error("owner_gate_stage0_bundle_invalid")
    if _SHA256.fullmatch(PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256 or "") is None:
        raise OwnerGateStage0Error("owner_gate_stage0_trust_anchor_unconfigured")
    trust_raw = _read_regular(
        root / "trust/release-trust.json",
        maximum=MAX_JSON_BYTES,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
    )
    public_key = _read_regular(
        root / "trust/release-trust-signing.pub",
        maximum=32,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
    )
    if (
        len(public_key) != 32
        or hashlib.sha256(public_key).hexdigest()
        != PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_trust_anchor_mismatch")
    trust_value = _load_canonical_json(trust_raw)
    authority = _validate_trust(trust_value, public_key)
    verify_ed25519_with_openssl(
        public_key=public_key,
        signature=_b64url(
            trust_value["signature_ed25519_b64url"], expected_size=64
        ),
        message=canonical_json_bytes(authority),
        runner=runner,
        temporary_root=temporary_root,
    )
    manifest_raw = _read_regular(
        root / "package-manifest.json",
        maximum=MAX_JSON_BYTES,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
    )
    return _verify_manifest_and_payloads(
        root,
        manifest_raw=manifest_raw,
        authority=authority,
        trust_raw=trust_raw,
        public_key=public_key,
        expected_uid=expected_uid,
    )


def validate_target_capabilities(
    manifest: Mapping[str, Any],
    *,
    bundle: Path,
    runner: Callable[..., bytes] = _run,
    python: Path = PYTHON,
    os_release: Path = OS_RELEASE,
    required_executables: Sequence[Path] = REQUIRED_EXECUTABLES,
    temporary_root: Path | None = None,
    expected_bundle_uid: int = 0,
) -> Mapping[str, Any]:
    validate_wheel_archives_for_install(
        bundle,
        manifest=manifest,
        expected_uid=expected_bundle_uid,
    )
    bootstrap_pip_artifact = _bootstrap_pip_artifact(manifest)
    bootstrap_pip_wheel = (
        bundle / "bootstrap" / bootstrap_pip_artifact["filename"]
    )
    try:
        executable_identities = tuple(
            _capture_executable_identity(executable)
            for executable in required_executables
        )
    except OwnerGateStage0Error:
        raise OwnerGateStage0Error(
            "owner_gate_stage0_capability_missing"
        ) from None
    python_raw = _read_exact_python_interpreter(python)
    _read_exact_os_release(os_release)
    openssl_version = runner((str(OPENSSL), "version")).decode("ascii").strip()
    python_version = runner((str(python), "--version")).decode("ascii").strip()
    systemd_version = runner(("/usr/bin/systemd", "--version")).decode(
        "ascii"
    ).splitlines()[0]
    try:
        with tempfile.TemporaryDirectory(
            prefix=".muncho-owner-gate-venv-probe-",
            dir=str(temporary_root) if temporary_root is not None else None,
        ) as directory:
            probe = Path(directory) / "venv"
            runner((
                str(python),
                "-I",
                "-B",
                "-m",
                "venv",
                "--without-pip",
                "--copies",
                str(probe),
            ))
            bootstrap_pip = runner(
                _signed_pip_argv(
                    probe,
                    bootstrap_pip_wheel,
                    (
                        "--isolated",
                        "--disable-pip-version-check",
                        "--version",
                    ),
                ),
                env=_pip_command_environment(),
            ).decode("ascii", errors="strict").strip()
    except OSError:
        raise OwnerGateStage0Error(
            "owner_gate_stage0_venv_capability_missing"
        ) from None
    verify_ed25519_with_openssl(
        public_key=_RFC8032_PUBLIC_KEY,
        signature=_RFC8032_SIGNATURE,
        message=_RFC8032_MESSAGE,
        runner=runner,
        temporary_root=temporary_root,
    )
    try:
        executable_identities_after = tuple(
            _capture_executable_identity(executable)
            for executable in required_executables
        )
    except OwnerGateStage0Error:
        raise OwnerGateStage0Error(
            "owner_gate_stage0_capability_changed"
        ) from None
    if (
        hashlib.sha256(python_raw).hexdigest() != manifest["interpreter_sha256"]
        or python_version != f"Python {PYTHON_VERSION}"
        or not openssl_version.startswith(OPENSSL_VERSION_PREFIX)
        or not systemd_version.startswith("systemd 252 ")
        or not bootstrap_pip.startswith(
            f"pip {bootstrap_pip_artifact['version']} from "
        )
        or str(bootstrap_pip_wheel) not in bootstrap_pip
        or platform.machine() != "x86_64"
        or executable_identities_after != executable_identities
    ):
        raise OwnerGateStage0Error("owner_gate_stage0_runtime_mismatch")
    unsigned = {
        "schema": PREFLIGHT_SCHEMA,
        "release_revision": manifest["release_revision"],
        "python_version": python_version,
        "python_sha256": hashlib.sha256(python_raw).hexdigest(),
        "openssl_version": openssl_version,
        "openssl_ed25519_rawin_verified": True,
        "systemd_version": systemd_version,
        "systemd_sysusers_available": True,
        "systemd_tmpfiles_available": True,
        "python_venv_available": True,
        "python_venv_without_pip_available": True,
        "bootstrap_pip_version": bootstrap_pip_artifact["version"],
        "bootstrap_pip_sha256": bootstrap_pip_artifact["sha256"],
        "executable_identities_sha256": sha256_json(
            executable_identities
        ),
        "network_install_required": False,
        "cloud_mutation_performed": False,
        "activation_performed": False,
    }
    return {**unsigned, "preflight_sha256": sha256_json(unsigned)}


def run_verified_bundle_operation(
    operation: str,
    *,
    bundle: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    if operation not in {"preflight", "install"}:
        raise OwnerGateStage0Error("owner_gate_stage0_operation_invalid")
    preflight = validate_target_capabilities(
        manifest,
        bundle=bundle,
        expected_bundle_uid=0,
    )
    if operation == "preflight":
        return preflight
    staging_or_release = prepare_offline_runtime(bundle, manifest)
    return invoke_runtime_installer(staging_or_release, bundle)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("verify", "preflight", "install"))
    parser.add_argument("--bundle", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if os.geteuid() != 0:  # windows-footgun: ok — Debian root boundary
        raise OwnerGateStage0Error("owner_gate_stage0_root_required")
    manifest = verify_bundle_stage0(arguments.bundle, expected_uid=0)
    if arguments.operation == "verify":
        result: Mapping[str, Any] = {
            "schema": "muncho-owner-gate-stage0-bundle-verification.v1",
            "release_revision": manifest["release_revision"],
            "package_sha256": manifest["package_sha256"],
            "verified": True,
        }
    else:
        result = run_verified_bundle_operation(
            arguments.operation,
            bundle=arguments.bundle,
            manifest=manifest,
        )
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
