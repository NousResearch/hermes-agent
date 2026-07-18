#!/usr/bin/env python3
"""Owner-only provisioning and readiness for release-pinned observation signers.

The private Ed25519 seed crosses the boundary only as one canonical JSON frame
on stdin.  It is never accepted in argv or the environment and is never
included, hashed, or otherwise committed to a receipt.  The installed public
identity is derived from the seed and must equal the raw public key authorized
by the signed offline package.

This module deliberately contains no Cloud/IAM client, network fetch, service
start, activation, generic command, or shell execution surface.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator, Mapping, NoReturn, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import owner_gate_foundation as foundation
from scripts.canary.runtime_units import CANARY_RUNTIME_UNITS


OWNER_DISCORD_USER_ID = "1279454038731264061"
PROVISIONING_RECEIPT_SCHEMA = "muncho-trusted-signer-provisioning-receipt.v1"
READINESS_SCHEMA = "muncho-trusted-signer-readiness.v1"
ENVELOPE_SCHEMAS = {
    "cloud": "muncho-cloud-trusted-signer-provisioning-envelope.v1",
    "host": "muncho-host-trusted-signer-provisioning-envelope.v1",
}
MAX_ENVELOPE_BYTES = 16 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
HOST_RELEASE_BASE = Path("/opt/muncho-trusted-observation/releases")
OWNER_GATE_RELEASE_BASE = Path("/opt/muncho-owner-gate/releases")
CLOUD_EXECUTOR_UID = 29103

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLE = frozenset(ENVELOPE_SCHEMAS)
_INERT_UNITS = (
    "muncho-passkey-web.service",
    "muncho-passkey-authority.socket",
    "muncho-passkey-authority.service",
    "muncho-privileged-executor.socket",
    "muncho-privileged-executor.service",
)
_SERVICE_PROPERTIES = frozenset({
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "MainPID",
    "FragmentPath",
    "DropInPaths",
})


class TrustedSignerProvisioningError(RuntimeError):
    """Stable, secret-free provisioning/readiness failure."""


@dataclass(frozen=True)
class SignerLayout:
    """All paths and identities for one fixed signer role.

    Production callers use the constructors below.  Explicit layouts exist so
    the exact filesystem boundary can be adversarially tested without changing
    the production constants.
    """

    role: str
    release_base: Path
    release: Path
    authority_manifest: Path
    pinned_public_key: Path
    private_key: Path
    installed_public_key: Path
    config: Path
    replay_directory: Path
    receipt: Path
    lock: Path
    activation_seal: Path
    current_link: Path
    private_uid: int
    private_gid: int
    config_uid: int
    config_gid: int
    replay_uid: int
    replay_gid: int
    receipt_uid: int
    receipt_gid: int
    runtime_receipt: Path | None = None
    runtime_receipt_uid: int = 0
    runtime_receipt_gid: int = 0
    runtime_receipt_mode: int = 0o440
    release_uid: int = 0
    release_gid: int = 0
    sudoers: Path | None = None
    sudoers_template: Path | None = None
    runtime_entrypoint_name: str = ""
    service_units: tuple[str, ...] = ()
    preexisting_directories: tuple[tuple[Path, int, int, int], ...] = ()

    def validate(self) -> None:
        paths = (
            self.release_base,
            self.release,
            self.authority_manifest,
            self.pinned_public_key,
            self.private_key,
            self.installed_public_key,
            self.config,
            self.replay_directory,
            self.receipt,
            self.lock,
            self.activation_seal,
            self.current_link,
        )
        if (
            self.role not in _ROLE
            or any(not path.is_absolute() or ".." in path.parts for path in paths)
            or self.release.parent != self.release_base
            or _REVISION.fullmatch(self.release.name) is None
            or not self.runtime_entrypoint_name
            or (
                self.role == "cloud" and self.service_units != _INERT_UNITS
            )
            or (
                self.role == "host"
                and self.service_units != tuple(CANARY_RUNTIME_UNITS)
            )
            or len(self.service_units) != len(set(self.service_units))
            or not self.preexisting_directories
            or len({item[0] for item in self.preexisting_directories})
            != len(self.preexisting_directories)
            or any(
                not path.is_absolute()
                or ".." in path.parts
                or mode < 0
                or mode > 0o777
                for path, _uid, _gid, mode in self.preexisting_directories
            )
            or (self.sudoers is None) != (self.sudoers_template is None)
            or (
                self.role == "cloud" and self.runtime_receipt is None
            )
            or (
                self.runtime_receipt is not None
                and (
                    not self.runtime_receipt.is_absolute()
                    or ".." in self.runtime_receipt.parts
                    or self.runtime_receipt_mode not in {0o440, 0o444}
                )
            )
            or (
                self.sudoers is not None
                and (
                    not self.sudoers.is_absolute()
                    or not self.sudoers_template.is_absolute()  # type: ignore[union-attr]
                )
            )
        ):
            raise TrustedSignerProvisioningError(
                "trusted_signer_layout_invalid"
            )


def cloud_layout(release_revision: str) -> SignerLayout:
    if _REVISION.fullmatch(release_revision or "") is None:
        raise TrustedSignerProvisioningError("trusted_signer_release_invalid")
    release = OWNER_GATE_RELEASE_BASE / release_revision
    return SignerLayout(
        role="cloud",
        release_base=OWNER_GATE_RELEASE_BASE,
        release=release,
        authority_manifest=release / "package-manifest.json",
        pinned_public_key=(
            release / "trust/cloud-observation-attestation.pub"
        ),
        private_key=Path(
            "/etc/muncho-owner-gate/executor-keys/"
            "cloud-observation-attestation.key"
        ),
        installed_public_key=Path(
            "/etc/muncho-owner-gate/public/cloud-observation-attestation.pub"
        ),
        config=Path("/etc/muncho-owner-gate/cloud-observation-attestor.json"),
        replay_directory=Path(
            "/var/lib/muncho-owner-gate/executor/cloud-attestations"
        ),
        receipt=Path(
            "/var/lib/muncho-owner-gate/bootstrap-receipts/"
            f"cloud-signer-{release_revision}.json"
        ),
        lock=Path("/run/muncho-owner-gate/cloud-signer-provision.lock"),
        activation_seal=Path("/etc/muncho-owner-gate/storage-executor-enabled"),
        current_link=Path("/opt/muncho-owner-gate/current"),
        private_uid=CLOUD_EXECUTOR_UID,
        private_gid=CLOUD_EXECUTOR_UID,
        config_uid=0,
        config_gid=0,
        replay_uid=CLOUD_EXECUTOR_UID,
        replay_gid=CLOUD_EXECUTOR_UID,
        receipt_uid=0,
        receipt_gid=0,
        runtime_receipt=Path(
            "/etc/muncho-owner-gate/public/"
            "cloud-signer-provisioning-receipt.json"
        ),
        runtime_receipt_uid=0,
        runtime_receipt_gid=CLOUD_EXECUTOR_UID,
        runtime_receipt_mode=0o440,
        sudoers=Path("/etc/sudoers.d/muncho-owner-gate-provisioning"),
        sudoers_template=(
            release
            / "ops/muncho/owner-gate/muncho-owner-gate-provisioning.sudoers.in"
        ),
        runtime_entrypoint_name="muncho-cloud-trusted-signer-provision",
        service_units=_INERT_UNITS,
        preexisting_directories=(
            (Path("/opt/muncho-owner-gate"), 0, 0, 0o755),
            (OWNER_GATE_RELEASE_BASE, 0, 0, 0o755),
            (Path("/etc/muncho-owner-gate"), 0, 0, 0o755),
            (Path("/etc/muncho-owner-gate/executor-keys"), 0, CLOUD_EXECUTOR_UID, 0o710),
            (Path("/etc/muncho-owner-gate/public"), 0, 0, 0o755),
            (Path("/var/lib/muncho-owner-gate"), 0, 0, 0o711),
            (Path("/var/lib/muncho-owner-gate/executor"), CLOUD_EXECUTOR_UID, CLOUD_EXECUTOR_UID, 0o700),
            (Path("/var/lib/muncho-owner-gate/bootstrap-receipts"), 0, 0, 0o700),
            (Path("/run/muncho-owner-gate"), 0, 0, 0o755),
            (Path("/etc/sudoers.d"), 0, 0, 0o755),
        ),
    )


def host_layout(release_revision: str) -> SignerLayout:
    if _REVISION.fullmatch(release_revision or "") is None:
        raise TrustedSignerProvisioningError("trusted_signer_release_invalid")
    release = HOST_RELEASE_BASE / release_revision
    return SignerLayout(
        role="host",
        release_base=HOST_RELEASE_BASE,
        release=release,
        authority_manifest=release / "package-manifest.json",
        pinned_public_key=(
            release / "trust/host-observation-attestation.pub"
        ),
        private_key=Path(
            "/etc/muncho/trusted-observation/"
            "host-observation-attestation.key"
        ),
        installed_public_key=Path(
            "/etc/muncho/trusted-observation/"
            "host-observation-attestation.pub"
        ),
        config=Path("/etc/muncho/trusted-observation/host-attestor.json"),
        replay_directory=Path(
            "/var/lib/muncho/trusted-observation/host-attestations"
        ),
        receipt=Path(
            "/var/lib/muncho/trusted-observation/receipts/"
            f"host-signer-{release_revision}.json"
        ),
        lock=Path(
            "/run/muncho-trusted-observation/host-signer-provision.lock"
        ),
        activation_seal=Path("/etc/muncho/trusted-observation/enabled"),
        current_link=Path("/opt/muncho-trusted-observation/current"),
        private_uid=0,
        private_gid=0,
        config_uid=0,
        config_gid=0,
        replay_uid=0,
        replay_gid=0,
        receipt_uid=0,
        receipt_gid=0,
        sudoers=Path("/etc/sudoers.d/muncho-host-observation-attestor"),
        sudoers_template=(
            release
            / "ops/muncho/owner-gate/muncho-host-observation-attestor.sudoers.in"
        ),
        runtime_entrypoint_name="muncho-host-trusted-signer-provision",
        service_units=tuple(CANARY_RUNTIME_UNITS),
        preexisting_directories=(
            (Path("/opt/muncho-trusted-observation"), 0, 0, 0o755),
            (HOST_RELEASE_BASE, 0, 0, 0o755),
            (Path("/etc/muncho/trusted-observation"), 0, 0, 0o700),
            (Path("/var/lib/muncho/trusted-observation"), 0, 0, 0o700),
            (Path("/var/lib/muncho/trusted-observation/receipts"), 0, 0, 0o700),
            (Path("/run/muncho-trusted-observation"), 0, 0, 0o700),
            (Path("/etc/sudoers.d"), 0, 0, 0o755),
        ),
    )


def _stable_error(code: str, exc: BaseException | None = None) -> NoReturn:
    del exc
    raise TrustedSignerProvisioningError(code) from None


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode_canonical(value: Any, *, size: int) -> bytes:
    if not isinstance(value, str) or not value or "=" in value:
        _stable_error("trusted_signer_envelope_invalid")
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        _stable_error("trusted_signer_envelope_invalid", exc)
    if len(raw) != size or _b64url_encode(raw) != value:
        _stable_error("trusted_signer_envelope_invalid")
    return raw


def _read_regular(
    path: Path,
    *,
    maximum: int,
    uid: int,
    gid: int,
    mode: int,
    allow_empty: bool = False,
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
            or opened.st_uid != uid
            or opened.st_gid != gid
            or stat.S_IMODE(opened.st_mode) != mode
            or (opened.st_size < 1 and not allow_empty)
            or opened.st_size > maximum
        ):
            _stable_error("trusted_signer_file_invalid")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                _stable_error("trusted_signer_file_invalid")
            chunks.append(chunk)
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
            _stable_error("trusted_signer_file_changed")
        return b"".join(chunks)
    except TrustedSignerProvisioningError:
        raise
    except OSError as exc:
        _stable_error("trusted_signer_file_unavailable", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    raise AssertionError("unreachable")


def _canonical_mapping(raw: bytes, *, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _stable_error(code, exc)
    if (
        not isinstance(value, Mapping)
        or foundation.canonical_json_bytes(value) != raw
    ):
        _stable_error(code)
    return dict(value)


def decode_provisioning_envelope(
    raw: bytes,
    *,
    role: str,
    release_revision: str,
    package_sha256: str,
) -> tuple[Mapping[str, Any], bytes]:
    """Decode exactly one LF-terminated canonical envelope."""

    if (
        role not in _ROLE
        or type(raw) is not bytes
        or len(raw) < 2
        or len(raw) > MAX_ENVELOPE_BYTES
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
    ):
        _stable_error("trusted_signer_envelope_invalid")
    payload = raw[:-1]
    value = _canonical_mapping(payload, code="trusted_signer_envelope_invalid")
    fields = {
        "schema",
        "role",
        "release_revision",
        "package_sha256",
        "owner_discord_user_id",
        "owner_authorization_receipt_sha256",
        "private_seed_ed25519_b64url",
        "public_key_id",
    }
    seed = _b64url_decode_canonical(
        value.get("private_seed_ed25519_b64url"), size=32
    )
    try:
        public_raw = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()
    except ValueError as exc:
        _stable_error("trusted_signer_envelope_invalid", exc)
    public_key_id = hashlib.sha256(public_raw).hexdigest()
    if (
        set(value) != fields
        or value.get("schema") != ENVELOPE_SCHEMAS[role]
        or value.get("role") != role
        or value.get("release_revision") != release_revision
        or value.get("package_sha256") != package_sha256
        or value.get("owner_discord_user_id") != OWNER_DISCORD_USER_ID
        or _SHA256.fullmatch(
            str(value.get("owner_authorization_receipt_sha256", ""))
        )
        is None
        or value.get("public_key_id") != public_key_id
    ):
        _stable_error("trusted_signer_envelope_invalid")
    return value, seed


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0),
        )
        os.fsync(descriptor)
    except OSError as exc:
        _stable_error("trusted_signer_directory_sync_failed", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _directory_evidence(path: Path, *, uid: int, gid: int, mode: int) -> Mapping[str, Any]:
    try:
        state = path.lstat()
    except OSError as exc:
        _stable_error("trusted_signer_directory_invalid", exc)
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISDIR(state.st_mode)
        or state.st_uid != uid
        or state.st_gid != gid
        or stat.S_IMODE(state.st_mode) != mode
    ):
        _stable_error("trusted_signer_directory_invalid")
    return {
        "path": str(path),
        "uid": uid,
        "gid": gid,
        "mode": f"{mode:04o}",
    }


def _ensure_directory(path: Path, *, uid: int, gid: int, mode: int) -> Mapping[str, Any]:
    if not path.is_absolute() or ".." in path.parts:
        _stable_error("trusted_signer_directory_invalid")
    if path.exists() or path.is_symlink():
        return _directory_evidence(path, uid=uid, gid=gid, mode=mode)
    try:
        parent = path.parent.lstat()
        if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
            _stable_error("trusted_signer_directory_invalid")
        path.mkdir(mode=mode)
        os.chown(path, uid, gid)
        os.chmod(path, mode)
        _fsync_directory(path.parent)
    except TrustedSignerProvisioningError:
        raise
    except OSError as exc:
        _stable_error("trusted_signer_directory_create_failed", exc)
    return _directory_evidence(path, uid=uid, gid=gid, mode=mode)


def _validate_layout_directories(layout: SignerLayout) -> None:
    layout.validate()
    for path, uid, gid, mode in layout.preexisting_directories:
        _assert_no_symlink_ancestors(path.parent)
        _directory_evidence(path, uid=uid, gid=gid, mode=mode)


def _file_evidence(
    path: Path, *, uid: int, gid: int, mode: int, include_digest: bool
) -> Mapping[str, Any]:
    raw = _read_regular(
        path,
        maximum=MAX_JSON_BYTES,
        uid=uid,
        gid=gid,
        mode=mode,
    )
    value: dict[str, Any] = {
        "path": str(path),
        "uid": uid,
        "gid": gid,
        "mode": f"{mode:04o}",
        "size": len(raw),
    }
    if include_digest:
        value["sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def _install_exclusive(
    path: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
    mode: int,
    include_digest: bool,
    after_open: Callable[[], None] | None = None,
    after_fsync: Callable[[], None] | None = None,
    after_write_chunk: Callable[[], None] | None = None,
    after_publish_link: Callable[[], None] | None = None,
) -> Mapping[str, Any]:
    """Stage with O_EXCL, fsync, then no-clobber publish by hard link.

    A crash may leave only the deterministic protected staging inode.  Replay
    either publishes an exact completed stage or safely removes an incomplete
    single-link stage; an existing final inode is never replaced.
    """

    if not path.is_absolute() or ".." in path.parts or not payload:
        _stable_error("trusted_signer_install_invalid")
    try:
        parent_state = path.parent.lstat()
    except OSError as exc:
        _stable_error("trusted_signer_install_parent_invalid", exc)
    if stat.S_ISLNK(parent_state.st_mode) or not stat.S_ISDIR(parent_state.st_mode):
        _stable_error("trusted_signer_install_parent_invalid")
    _assert_no_symlink_ancestors(path.parent)
    staging = path.parent / f".{path.name}.muncho-staged"
    descriptor: int | None = None
    try:
        if os.path.lexists(path):
            if staging.exists() or staging.is_symlink():
                final_state = path.lstat()
                stage_state = staging.lstat()
                incomplete_open = (
                    stat.S_ISREG(stage_state.st_mode)
                    and stage_state.st_nlink == 1
                    and stage_state.st_size == 0
                    and stage_state.st_uid in {0, uid}
                    and stage_state.st_gid in {0, gid}
                    and not (stat.S_IMODE(stage_state.st_mode) & ~mode)
                )
                if incomplete_open:
                    staging.unlink()
                    _fsync_directory(path.parent)
                    stage_state = None
                if (
                    stat.S_ISLNK(final_state.st_mode)
                    or (
                        stage_state is not None
                        and stat.S_ISLNK(stage_state.st_mode)
                    )
                    or not stat.S_ISREG(final_state.st_mode)
                    or (
                        stage_state is not None
                        and (
                            not stat.S_ISREG(stage_state.st_mode)
                            or stage_state.st_uid != uid
                            or stage_state.st_gid != gid
                            or stat.S_IMODE(stage_state.st_mode) != mode
                            or stage_state.st_nlink not in {1, 2}
                        )
                    )
                ):
                    _stable_error("trusted_signer_staging_invalid")
                if stage_state is not None and stage_state.st_nlink == 2 and (
                    stage_state.st_dev,
                    stage_state.st_ino,
                ) != (final_state.st_dev, final_state.st_ino):
                    _stable_error("trusted_signer_staging_invalid")
                if stage_state is not None and stage_state.st_nlink == 2:
                    staging.unlink()
                    _fsync_directory(path.parent)
            existing = _read_regular(
                path,
                maximum=max(MAX_JSON_BYTES, len(payload)),
                uid=uid,
                gid=gid,
                mode=mode,
            )
            if existing != payload:
                _stable_error("trusted_signer_install_conflict")
            if staging.exists() and not staging.is_symlink():
                staging.unlink()
                _fsync_directory(path.parent)
            return _file_evidence(
                path,
                uid=uid,
                gid=gid,
                mode=mode,
                include_digest=include_digest,
            )
        if staging.exists() or staging.is_symlink():
            try:
                staged = _read_regular(
                    staging,
                    maximum=max(MAX_JSON_BYTES, len(payload)),
                    uid=uid,
                    gid=gid,
                    mode=mode,
                    allow_empty=True,
                )
            except TrustedSignerProvisioningError:
                stage_state = staging.lstat()
                if (
                    stat.S_ISLNK(stage_state.st_mode)
                    or not stat.S_ISREG(stage_state.st_mode)
                    or stage_state.st_nlink != 1
                    or stage_state.st_size != 0
                    or stage_state.st_uid not in {0, uid}
                    or stage_state.st_gid not in {0, gid}
                    or stat.S_IMODE(stage_state.st_mode) & ~mode
                ):
                    raise
                staging.unlink()
                _fsync_directory(path.parent)
                staged = None
            if staged is None:
                pass
            elif staged != payload:
                stage_state = staging.lstat()
                if (
                    not stat.S_ISREG(stage_state.st_mode)
                    or stage_state.st_nlink != 1
                    or stage_state.st_uid != uid
                    or stage_state.st_gid != gid
                    or stat.S_IMODE(stage_state.st_mode) != mode
                    or len(staged) >= len(payload)
                    or not payload.startswith(staged)
                ):
                    _stable_error("trusted_signer_staging_invalid")
                staging.unlink()
                _fsync_directory(path.parent)
            else:
                recovered_descriptor = os.open(
                    staging,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    recovered = os.fstat(recovered_descriptor)
                    if (
                        not stat.S_ISREG(recovered.st_mode)
                        or recovered.st_nlink != 1
                        or recovered.st_uid != uid
                        or recovered.st_gid != gid
                        or stat.S_IMODE(recovered.st_mode) != mode
                    ):
                        _stable_error("trusted_signer_staging_invalid")
                    os.fsync(recovered_descriptor)
                finally:
                    os.close(recovered_descriptor)
                _fsync_directory(path.parent)
        if not staging.exists():
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(staging, flags, mode)
            if after_open is not None:
                after_open()
            os.fchmod(descriptor, mode)
            os.fchown(descriptor, uid, gid)
            view = memoryview(payload)
            while view:
                chunk = view[: min(len(view), 16)]
                written = os.write(descriptor, chunk)
                if written <= 0:
                    raise OSError
                view = view[written:]
                if after_write_chunk is not None:
                    after_write_chunk()
            os.fsync(descriptor)
            if after_fsync is not None:
                after_fsync()
            os.close(descriptor)
            descriptor = None
            _fsync_directory(path.parent)
        try:
            os.link(staging, path, follow_symlinks=False)
        except FileExistsError:
            pass
        if after_publish_link is not None:
            after_publish_link()
        _fsync_directory(path.parent)
        if staging.exists() and not staging.is_symlink():
            stage_state = staging.lstat()
            if (
                not stat.S_ISREG(stage_state.st_mode)
                or stage_state.st_uid != uid
                or stage_state.st_gid != gid
                or stat.S_IMODE(stage_state.st_mode) != mode
                or stage_state.st_nlink not in {1, 2}
            ):
                _stable_error("trusted_signer_staging_invalid")
            staging.unlink()
            _fsync_directory(path.parent)
    except OSError as exc:
        _stable_error("trusted_signer_install_failed", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    existing = _read_regular(
        path,
        maximum=max(MAX_JSON_BYTES, len(payload)),
        uid=uid,
        gid=gid,
        mode=mode,
    )
    if existing != payload:
        _stable_error("trusted_signer_install_conflict")
    return _file_evidence(
        path,
        uid=uid,
        gid=gid,
        mode=mode,
        include_digest=include_digest,
    )


def _assert_no_symlink_ancestors(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        _stable_error("trusted_signer_path_invalid")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            state = current.lstat()
        except OSError as exc:
            _stable_error("trusted_signer_path_invalid", exc)
        if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
            _stable_error("trusted_signer_path_invalid")


@contextmanager
def _exclusive_lock(path: Path, *, uid: int, gid: int) -> Iterator[None]:
    descriptor: int | None = None
    try:
        _assert_no_symlink_ancestors(path.parent)
        common = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        common |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, common | os.O_CREAT | os.O_EXCL, 0o600)
            os.fchmod(descriptor, 0o600)
            os.fchown(descriptor, uid, gid)
            os.fsync(descriptor)
            _fsync_directory(path.parent)
        except FileExistsError:
            descriptor = os.open(path, common)
        state = os.fstat(descriptor)
        if (
            not stat.S_ISREG(state.st_mode)
            or state.st_nlink != 1
            or state.st_uid != uid
            or state.st_gid != gid
            or stat.S_IMODE(state.st_mode) != 0o600
        ):
            _stable_error("trusted_signer_lock_invalid")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    except TrustedSignerProvisioningError:
        raise
    except OSError as exc:
        _stable_error("trusted_signer_lock_failed", exc)
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _payload_record(manifest: Mapping[str, Any], relative: str) -> Mapping[str, Any]:
    matches = [
        item
        for item in manifest.get("payloads", [])
        if isinstance(item, Mapping) and item.get("release_relative") == relative
    ]
    if len(matches) != 1:
        _stable_error("trusted_signer_runtime_closure_invalid")
    return dict(matches[0])


def _validate_release_and_authority(layout: SignerLayout) -> Mapping[str, Any]:
    _validate_layout_directories(layout)
    state = layout.release.lstat()
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISDIR(state.st_mode)
        or state.st_uid != layout.release_uid
        or state.st_gid != layout.release_gid
        or stat.S_IMODE(state.st_mode) != 0o555
    ):
        _stable_error("trusted_signer_release_invalid")
    manifest_raw = _read_regular(
        layout.authority_manifest,
        maximum=MAX_JSON_BYTES,
        uid=layout.release_uid if layout.authority_manifest.is_relative_to(layout.release) else 0,
        gid=layout.release_gid if layout.authority_manifest.is_relative_to(layout.release) else 0,
        mode=0o444,
    )
    manifest = _canonical_mapping(
        manifest_raw, code="trusted_signer_package_manifest_invalid"
    )
    collectors = manifest.get("collector_public_key_ids")
    closure = manifest.get("runtime_source_closure")
    wheels = manifest.get("wheels")
    if (
        manifest.get("release_revision") != layout.release.name
        or _SHA256.fullmatch(str(manifest.get("package_sha256", ""))) is None
        or not isinstance(collectors, Mapping)
        or set(collectors) != {"network", "cloud", "host"}
        or _SHA256.fullmatch(str(collectors.get(layout.role, ""))) is None
        or not isinstance(closure, list)
        or "scripts/canary/trusted_signer_provisioning.py" not in closure
        or "scripts/canary/storage_growth_trusted_collector.py" not in closure
        or not isinstance(wheels, list)
        or len(
            [
                item
                for item in wheels
                if isinstance(item, Mapping)
                and item.get("project") == "cryptography"
                and item.get("version") == "49.0.0"
            ]
        )
        != 1
    ):
        _stable_error("trusted_signer_package_manifest_invalid")
    public_raw = _read_regular(
        layout.pinned_public_key,
        maximum=32,
        uid=layout.release_uid if layout.pinned_public_key.is_relative_to(layout.release) else 0,
        gid=layout.release_gid if layout.pinned_public_key.is_relative_to(layout.release) else 0,
        mode=0o444,
    )
    public_id = hashlib.sha256(public_raw).hexdigest()
    if len(public_raw) != 32 or public_id != collectors[layout.role]:
        _stable_error("trusted_signer_pinned_public_key_invalid")
    try:
        Ed25519PublicKey.from_public_bytes(public_raw)
    except ValueError as exc:
        _stable_error("trusted_signer_pinned_public_key_invalid", exc)

    if (
        (layout.release / ".bootstrap-wheelhouse").exists()
        or (layout.release / ".bootstrap-wheelhouse-installed.json").exists()
        or any(layout.release.rglob("__pycache__"))
    ):
        _stable_error("trusted_signer_release_invalid")
    projection: list[dict[str, Any]] = []
    for path in sorted(
        layout.release.rglob("*"),
        key=lambda item: str(item.relative_to(layout.release)),
    ):
        relative = str(path.relative_to(layout.release))
        item_state = path.lstat()
        if item_state.st_uid != layout.release_uid or item_state.st_gid != layout.release_gid:
            _stable_error("trusted_signer_release_owner_invalid")
        if stat.S_ISLNK(item_state.st_mode):
            target = os.readlink(path)
            if os.path.isabs(target) or ".." in Path(target).parts:
                _stable_error("trusted_signer_release_symlink_invalid")
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                _stable_error("trusted_signer_release_symlink_invalid", exc)
            if not resolved.is_relative_to(layout.release):
                _stable_error("trusted_signer_release_symlink_invalid")
            projection.append({"path": relative, "type": "symlink", "target": target})
        elif stat.S_ISDIR(item_state.st_mode):
            if stat.S_IMODE(item_state.st_mode) != 0o555:
                _stable_error("trusted_signer_release_mode_invalid")
            projection.append({"path": relative, "type": "directory", "mode": "0555"})
        elif stat.S_ISREG(item_state.st_mode):
            mode = stat.S_IMODE(item_state.st_mode)
            if mode not in {0o444, 0o555}:
                _stable_error("trusted_signer_release_mode_invalid")
            raw = _read_regular(
                path,
                maximum=128 * 1024 * 1024,
                uid=layout.release_uid,
                gid=layout.release_gid,
                mode=mode,
            )
            projection.append({
                "path": relative,
                "type": "file",
                "mode": f"{mode:04o}",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            })
        else:
            _stable_error("trusted_signer_release_node_invalid")
    for item in manifest.get("payloads", []):
        if not isinstance(item, Mapping):
            _stable_error("trusted_signer_package_manifest_invalid")
        relative = item.get("release_relative")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or item.get("mode") not in {"0444", "0555"}
        ):
            _stable_error("trusted_signer_package_manifest_invalid")
        raw = _read_regular(
            layout.release / relative,
            maximum=128 * 1024 * 1024,
            uid=layout.release_uid,
            gid=layout.release_gid,
            mode=int(str(item["mode"]), 8),
        )
        if (
            hashlib.sha256(raw).hexdigest() != item.get("sha256")
            or len(raw) != item.get("size")
        ):
            _stable_error("trusted_signer_runtime_closure_invalid")

    runtime_python = layout.release / "venv/bin/python"
    runtime_python3 = runtime_python.resolve(strict=True)
    if (
        not runtime_python3.is_relative_to(layout.release / "venv")
        or not runtime_python3.is_file()
        or not os.access(runtime_python3, os.X_OK)
    ):
        _stable_error("trusted_signer_runtime_invalid")
    interpreter_raw = _read_regular(
        runtime_python3,
        maximum=128 * 1024 * 1024,
        uid=layout.release_uid,
        gid=layout.release_gid,
        mode=0o555,
    )
    if hashlib.sha256(interpreter_raw).hexdigest() != manifest.get(
        "interpreter_sha256"
    ):
        _stable_error("trusted_signer_runtime_invalid")
    entrypoint_relative = f"bin/{layout.runtime_entrypoint_name}"
    entrypoint = layout.release / entrypoint_relative
    entrypoint_record = _payload_record(manifest, entrypoint_relative)
    entrypoint_raw = _read_regular(
        entrypoint,
        maximum=MAX_JSON_BYTES,
        uid=layout.release_uid,
        gid=layout.release_gid,
        mode=0o555,
    )
    if (
        hashlib.sha256(entrypoint_raw).hexdigest() != entrypoint_record.get("sha256")
        or len(entrypoint_raw) != entrypoint_record.get("size")
    ):
        _stable_error("trusted_signer_runtime_invalid")
    required_sources = (
        "scripts/canary/trusted_signer_provisioning.py",
        "scripts/canary/storage_growth_trusted_collector.py",
    )
    for relative in required_sources:
        record = _payload_record(manifest, relative)
        raw = _read_regular(
            layout.release / relative,
            maximum=128 * 1024 * 1024,
            uid=layout.release_uid,
            gid=layout.release_gid,
            mode=0o444,
        )
        if (
            hashlib.sha256(raw).hexdigest() != record.get("sha256")
            or len(raw) != record.get("size")
        ):
            _stable_error("trusted_signer_runtime_closure_invalid")
    return {
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "package_sha256": manifest["package_sha256"],
        "public_raw": public_raw,
        "public_key_id": public_id,
        "runtime": {
            "release": str(layout.release),
            "release_uid": layout.release_uid,
            "release_gid": layout.release_gid,
            "release_mode": "0555",
            "interpreter": str(runtime_python),
            "resolved_interpreter": str(runtime_python3),
            "interpreter_sha256": hashlib.sha256(interpreter_raw).hexdigest(),
            "entrypoint": str(entrypoint),
            "entrypoint_sha256": hashlib.sha256(entrypoint_raw).hexdigest(),
            "runtime_source_closure_sha256": foundation.sha256_json(closure),
            "immutable_release_projection_sha256": foundation.sha256_json(projection),
            "immutable_release_projection_count": len(projection),
            "cryptography_version": "49.0.0",
            "offline_runtime": True,
            "generic_usr_bin_python3_runtime": False,
        },
    }


def _default_unit_probe(name: str) -> Mapping[str, Any]:
    try:
        completed = subprocess.run(
            (
                "/usr/bin/systemctl",
                "show",
                "--no-pager",
                name,
                *(f"--property={item}" for item in sorted(_SERVICE_PROPERTIES)),
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C.UTF-8"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _stable_error("trusted_signer_inert_probe_failed", exc)
    if completed.returncode != 0:
        _stable_error("trusted_signer_inert_probe_failed")
    try:
        lines = completed.stdout.decode("ascii", errors="strict").splitlines()
    except UnicodeError as exc:
        _stable_error("trusted_signer_inert_probe_failed", exc)
    properties: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            _stable_error("trusted_signer_inert_probe_failed")
        key, value = line.split("=", 1)
        if key not in _SERVICE_PROPERTIES or key in properties:
            _stable_error("trusted_signer_inert_probe_failed")
        properties[key] = value
    missing = _SERVICE_PROPERTIES - set(properties)
    if missing == {"MainPID"} and name.endswith((".socket", ".timer")):
        properties["MainPID"] = "0"
    if set(properties) != _SERVICE_PROPERTIES:
        _stable_error("trusted_signer_inert_probe_failed")
    return properties


def _inert_evidence(
    layout: SignerLayout,
    *,
    unit_probe: Callable[[str], Mapping[str, Any]],
) -> Mapping[str, Any]:
    if layout.activation_seal.exists() or layout.activation_seal.is_symlink():
        _stable_error("trusted_signer_activation_not_inert")
    if layout.current_link.exists() or layout.current_link.is_symlink():
        _stable_error("trusted_signer_current_release_selected")
    units: list[dict[str, Any]] = []
    for name in layout.service_units:
        observed = dict(unit_probe(name))
        absent = {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "",
            "MainPID": "0",
            "FragmentPath": "",
            "DropInPaths": "",
        }
        disabled = {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "disabled",
            "MainPID": "0",
            "FragmentPath": f"/etc/systemd/system/{name}",
            "DropInPaths": "",
        }
        if set(observed) != _SERVICE_PROPERTIES or (
            observed != absent and observed != disabled
        ):
            _stable_error("trusted_signer_service_not_inert")
        units.append({
            "name": name,
            "state": "absent" if observed == absent else "disabled_inactive",
            "properties": observed,
        })
    return {
        "activation_seal": str(layout.activation_seal),
        "activation_seal_absent": True,
        "current_link": str(layout.current_link),
        "current_link_absent": True,
        "services": units,
        "services_started": False,
        "service_enablement_mutated": False,
        "iam_mutation_performed": False,
        "cloud_mutation_performed": False,
        "network_fetch_performed": False,
        "generic_shell_executed": False,
    }


def _render_config(layout: SignerLayout, public_key_id: str) -> bytes:
    value = {
        "schema": f"muncho-storage-growth-{layout.role}-attestor-config.v1",
        "role": layout.role,
        "private_key_path": str(layout.private_key),
        "private_key_uid": layout.private_uid,
        "private_key_gid": layout.private_gid,
        "private_key_mode": "0400",
        "public_key_id": public_key_id,
        "replay_directory": str(layout.replay_directory),
        "replay_directory_uid": layout.replay_uid,
        "replay_directory_gid": layout.replay_gid,
        "replay_directory_mode": "0700",
    }
    return foundation.canonical_json_bytes(value)


def _render_sudoers(layout: SignerLayout) -> bytes | None:
    if layout.sudoers_template is None:
        return None
    raw = _read_regular(
        layout.sudoers_template,
        maximum=64 * 1024,
        uid=layout.release_uid,
        gid=layout.release_gid,
        mode=0o444,
    )
    placeholder = b"@RELEASE_SHA@"
    if raw.count(placeholder) < 1:
        _stable_error("trusted_signer_sudoers_template_invalid")
    rendered = raw.replace(placeholder, layout.release.name.encode("ascii"))
    if b"@" in rendered or b"/usr/bin/python3" in rendered or b"\r" in rendered:
        _stable_error("trusted_signer_sudoers_template_invalid")
    interpreter = str(layout.release / "venv/bin/python").encode("ascii")
    if interpreter not in rendered:
        _stable_error("trusted_signer_sudoers_template_invalid")
    return rendered


def _default_visudo(path: Path) -> None:
    try:
        completed = subprocess.run(
            ("/usr/sbin/visudo", "-cf", str(path)),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C.UTF-8"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _stable_error("trusted_signer_sudoers_validation_failed", exc)
    if completed.returncode != 0:
        _stable_error("trusted_signer_sudoers_validation_failed")


def _validate_sudoers_bytes(
    destination: Path,
    payload: bytes,
    *,
    validator: Callable[[Path], None],
) -> None:
    temporary = destination.parent / f".{destination.name}.visudo-staged"
    descriptor: int | None = None
    try:
        if temporary.exists() or temporary.is_symlink():
            staged = _read_regular(
                temporary,
                maximum=64 * 1024,
                uid=0,
                gid=0,
                mode=0o600,
                allow_empty=True,
            )
            if staged != payload:
                staged_state = temporary.lstat()
                if (
                    not stat.S_ISREG(staged_state.st_mode)
                    or staged_state.st_nlink != 1
                    or staged_state.st_uid != 0
                    or staged_state.st_gid != 0
                    or stat.S_IMODE(staged_state.st_mode) != 0o600
                ):
                    _stable_error("trusted_signer_sudoers_validation_failed")
                temporary.unlink()
                _fsync_directory(destination.parent)
        if not temporary.exists():
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            os.fchown(descriptor, 0, 0)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view[: min(len(view), 64)])
                if written <= 0:
                    raise OSError
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            _fsync_directory(destination.parent)
        else:
            descriptor = os.open(
                temporary,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
        validator(temporary)
    except TrustedSignerProvisioningError:
        raise
    except OSError as exc:
        _stable_error("trusted_signer_sudoers_validation_failed", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
        except OSError:
            pass


def _verify_private_matches(
    layout: SignerLayout, *, expected_public_raw: bytes
) -> tuple[Ed25519PrivateKey, Mapping[str, Any]]:
    seed = _read_regular(
        layout.private_key,
        maximum=32,
        uid=layout.private_uid,
        gid=layout.private_gid,
        mode=0o400,
    )
    if len(seed) != 32:
        _stable_error("trusted_signer_private_key_invalid")
    try:
        private_key = Ed25519PrivateKey.from_private_bytes(seed)
    except ValueError as exc:
        _stable_error("trusted_signer_private_key_invalid", exc)
    derived = private_key.public_key().public_bytes_raw()
    if derived != expected_public_raw:
        _stable_error("trusted_signer_private_public_mismatch")
    # Deliberately no private-key digest in this evidence.
    return private_key, _file_evidence(
        layout.private_key,
        uid=layout.private_uid,
        gid=layout.private_gid,
        mode=0o400,
        include_digest=False,
    )


def _receipt_unsigned(
    *,
    layout: SignerLayout,
    authority: Mapping[str, Any],
    owner_authorization_receipt_sha256: str,
    private_evidence: Mapping[str, Any],
    public_evidence: Mapping[str, Any],
    config_evidence: Mapping[str, Any],
    replay_evidence: Mapping[str, Any],
    sudoers_evidence: Mapping[str, Any] | None,
    inert: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "schema": PROVISIONING_RECEIPT_SCHEMA,
        "role": layout.role,
        "release_revision": layout.release.name,
        "package_sha256": authority["package_sha256"],
        "package_manifest_sha256": authority["manifest_sha256"],
        "owner_discord_user_id": OWNER_DISCORD_USER_ID,
        "owner_authorization_receipt_sha256": owner_authorization_receipt_sha256,
        "public_key_id": authority["public_key_id"],
        "pinned_public_key_sha256": authority["public_key_id"],
        "private_key": dict(private_evidence),
        "installed_public_key": dict(public_evidence),
        "config": dict(config_evidence),
        "replay_directory": dict(replay_evidence),
        "sudoers": dict(sudoers_evidence) if sudoers_evidence is not None else None,
        "runtime_receipt": (
            {
                "path": str(layout.runtime_receipt),
                "uid": layout.runtime_receipt_uid,
                "gid": layout.runtime_receipt_gid,
                "mode": f"{layout.runtime_receipt_mode:04o}",
                "authoritative_copy": False,
                "byte_exact_signed_public_projection": True,
            }
            if layout.runtime_receipt is not None
            else None
        ),
        "runtime": dict(authority["runtime"]),
        "inert_boundary": dict(inert),
        "private_key_material_recorded": False,
        "private_key_digest_recorded": False,
        "private_key_argv_source": False,
        "private_key_environment_source": False,
        "private_key_stdin_source": True,
        "activation_performed": False,
        "iam_mutation_performed": False,
        "cloud_mutation_performed": False,
        "service_start_performed": False,
        "network_fetch_performed": False,
    }


def _sign_receipt(
    unsigned: Mapping[str, Any], private_key: Ed25519PrivateKey
) -> Mapping[str, Any]:
    digest = foundation.sha256_json(unsigned)
    signed = {**unsigned, "receipt_sha256": digest}
    signature = private_key.sign(foundation.canonical_json_bytes(signed))
    return {**signed, "signature_ed25519_b64url": _b64url_encode(signature)}


def _verify_receipt(
    receipt: Mapping[str, Any], *, public_key: Ed25519PublicKey
) -> Mapping[str, Any]:
    if not isinstance(receipt, Mapping):
        _stable_error("trusted_signer_receipt_invalid")
    expected_fields = {
        "schema",
        "role",
        "release_revision",
        "package_sha256",
        "package_manifest_sha256",
        "owner_discord_user_id",
        "owner_authorization_receipt_sha256",
        "public_key_id",
        "pinned_public_key_sha256",
        "private_key",
        "installed_public_key",
        "config",
        "replay_directory",
        "sudoers",
        "runtime_receipt",
        "runtime",
        "inert_boundary",
        "private_key_material_recorded",
        "private_key_digest_recorded",
        "private_key_argv_source",
        "private_key_environment_source",
        "private_key_stdin_source",
        "activation_performed",
        "iam_mutation_performed",
        "cloud_mutation_performed",
        "service_start_performed",
        "network_fetch_performed",
        "receipt_sha256",
        "signature_ed25519_b64url",
    }
    unsigned = {
        key: item
        for key, item in receipt.items()
        if key not in {"receipt_sha256", "signature_ed25519_b64url"}
    }
    signed = {**unsigned, "receipt_sha256": receipt.get("receipt_sha256")}
    signature = _b64url_decode_canonical(
        receipt.get("signature_ed25519_b64url"), size=64
    )
    role = receipt.get("role")
    private_evidence = receipt.get("private_key")
    public_evidence = receipt.get("installed_public_key")
    config_evidence = receipt.get("config")
    replay_evidence = receipt.get("replay_directory")
    sudoers_evidence = receipt.get("sudoers")
    runtime_receipt = receipt.get("runtime_receipt")
    runtime = receipt.get("runtime")
    inert = receipt.get("inert_boundary")
    expected_units = _INERT_UNITS if role == "cloud" else tuple(CANARY_RUNTIME_UNITS)
    services = inert.get("services") if isinstance(inert, Mapping) else None
    if (
        set(receipt) != expected_fields
        or receipt.get("schema") != PROVISIONING_RECEIPT_SCHEMA
        or role not in {"cloud", "host"}
        or _REVISION.fullmatch(str(receipt.get("release_revision", ""))) is None
        or _SHA256.fullmatch(str(receipt.get("package_sha256", ""))) is None
        or _SHA256.fullmatch(
            str(receipt.get("package_manifest_sha256", ""))
        )
        is None
        or receipt.get("owner_discord_user_id") != OWNER_DISCORD_USER_ID
        or _SHA256.fullmatch(
            str(receipt.get("owner_authorization_receipt_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(str(receipt.get("public_key_id", ""))) is None
        or receipt.get("pinned_public_key_sha256")
        != receipt.get("public_key_id")
        or not isinstance(private_evidence, Mapping)
        or set(private_evidence) != {"path", "uid", "gid", "mode", "size"}
        or private_evidence.get("mode") != "0400"
        or private_evidence.get("size") != 32
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"path", "uid", "gid", "mode", "size", "sha256"}
            or _SHA256.fullmatch(str(item.get("sha256", ""))) is None
            for item in (public_evidence, config_evidence)
        )
        or not isinstance(replay_evidence, Mapping)
        or set(replay_evidence) != {"path", "uid", "gid", "mode"}
        or replay_evidence.get("mode") != "0700"
        or not isinstance(sudoers_evidence, Mapping)
        or set(sudoers_evidence)
        != {"path", "uid", "gid", "mode", "size", "sha256"}
        or sudoers_evidence.get("uid") != 0
        or sudoers_evidence.get("gid") != 0
        or sudoers_evidence.get("mode") != "0440"
        or _SHA256.fullmatch(str(sudoers_evidence.get("sha256", ""))) is None
        or (
            role == "cloud"
            and (
                not isinstance(runtime_receipt, Mapping)
                or set(runtime_receipt)
                != {
                    "path",
                    "uid",
                    "gid",
                    "mode",
                    "authoritative_copy",
                    "byte_exact_signed_public_projection",
                }
                or runtime_receipt.get("uid") != 0
                or runtime_receipt.get("gid") != CLOUD_EXECUTOR_UID
                or runtime_receipt.get("mode") != "0440"
                or runtime_receipt.get("authoritative_copy") is not False
                or runtime_receipt.get("byte_exact_signed_public_projection")
                is not True
            )
        )
        or (role == "host" and runtime_receipt is not None)
        or not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "release",
            "release_uid",
            "release_gid",
            "release_mode",
            "interpreter",
            "resolved_interpreter",
            "interpreter_sha256",
            "entrypoint",
            "entrypoint_sha256",
            "runtime_source_closure_sha256",
            "immutable_release_projection_sha256",
            "immutable_release_projection_count",
            "cryptography_version",
            "offline_runtime",
            "generic_usr_bin_python3_runtime",
        }
        or runtime.get("release_mode") != "0555"
        or runtime.get("cryptography_version") != "49.0.0"
        or runtime.get("offline_runtime") is not True
        or runtime.get("generic_usr_bin_python3_runtime") is not False
        or not isinstance(inert, Mapping)
        or set(inert)
        != {
            "activation_seal",
            "activation_seal_absent",
            "current_link",
            "current_link_absent",
            "services",
            "services_started",
            "service_enablement_mutated",
            "iam_mutation_performed",
            "cloud_mutation_performed",
            "network_fetch_performed",
            "generic_shell_executed",
        }
        or inert.get("activation_seal_absent") is not True
        or inert.get("current_link_absent") is not True
        or inert.get("services_started") is not False
        or inert.get("service_enablement_mutated") is not False
        or inert.get("iam_mutation_performed") is not False
        or inert.get("cloud_mutation_performed") is not False
        or inert.get("network_fetch_performed") is not False
        or inert.get("generic_shell_executed") is not False
        or not isinstance(services, list)
        or [item.get("name") for item in services if isinstance(item, Mapping)]
        != list(expected_units)
        or len(services) != len(expected_units)
        or receipt.get("receipt_sha256") != foundation.sha256_json(unsigned)
        or receipt.get("private_key_material_recorded") is not False
        or receipt.get("private_key_digest_recorded") is not False
        or receipt.get("private_key_argv_source") is not False
        or receipt.get("private_key_environment_source") is not False
        or receipt.get("private_key_stdin_source") is not True
        or receipt.get("activation_performed") is not False
        or receipt.get("iam_mutation_performed") is not False
        or receipt.get("cloud_mutation_performed") is not False
        or receipt.get("service_start_performed") is not False
        or receipt.get("network_fetch_performed") is not False
    ):
        _stable_error("trusted_signer_receipt_invalid")
    for expected_name, service in zip(expected_units, services, strict=True):
        if not isinstance(service, Mapping) or set(service) != {
            "name",
            "state",
            "properties",
        }:
            _stable_error("trusted_signer_receipt_invalid")
        properties = service.get("properties")
        absent = {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "",
            "MainPID": "0",
            "FragmentPath": "",
            "DropInPaths": "",
        }
        disabled = {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "disabled",
            "MainPID": "0",
            "FragmentPath": f"/etc/systemd/system/{expected_name}",
            "DropInPaths": "",
        }
        if (
            service.get("name") != expected_name
            or service.get("state")
            != ("absent" if properties == absent else "disabled_inactive")
            or (properties != absent and properties != disabled)
        ):
            _stable_error("trusted_signer_receipt_invalid")
    try:
        public_key.verify(signature, foundation.canonical_json_bytes(signed))
    except InvalidSignature as exc:
        _stable_error("trusted_signer_receipt_signature_invalid", exc)
    return dict(receipt)


def _read_stdin_frame(stdin: BinaryIO) -> bytes:
    raw = stdin.read(MAX_ENVELOPE_BYTES + 1)
    if len(raw) > MAX_ENVELOPE_BYTES:
        _stable_error("trusted_signer_envelope_invalid")
    return raw


def provision_signer(
    raw_envelope: bytes,
    *,
    layout: SignerLayout,
    unit_probe: Callable[[str], Mapping[str, Any]] = _default_unit_probe,
    visudo_validator: Callable[[Path], None] = _default_visudo,
) -> Mapping[str, Any]:
    """Provision/replay one fixed signer while the activation boundary is inert."""

    authority = _validate_release_and_authority(layout)
    envelope, seed = decode_provisioning_envelope(
        raw_envelope,
        role=layout.role,
        release_revision=layout.release.name,
        package_sha256=str(authority["package_sha256"]),
    )
    try:
        input_private = Ed25519PrivateKey.from_private_bytes(seed)
    except ValueError as exc:
        _stable_error("trusted_signer_envelope_invalid", exc)
    if input_private.public_key().public_bytes_raw() != authority["public_raw"]:
        _stable_error("trusted_signer_private_public_mismatch")

    _inert_evidence(layout, unit_probe=unit_probe)
    with _exclusive_lock(
        layout.lock,
        uid=layout.receipt_uid,
        gid=layout.receipt_gid,
    ):
        inert = _inert_evidence(layout, unit_probe=unit_probe)
        private_evidence = _install_exclusive(
            layout.private_key,
            seed,
            uid=layout.private_uid,
            gid=layout.private_gid,
            mode=0o400,
            include_digest=False,
        )
        private_key, private_evidence = _verify_private_matches(
            layout, expected_public_raw=authority["public_raw"]
        )
        public_evidence = _install_exclusive(
            layout.installed_public_key,
            authority["public_raw"],
            uid=layout.config_uid,
            gid=layout.config_gid,
            mode=0o444,
            include_digest=True,
        )
        config_evidence = _install_exclusive(
            layout.config,
            _render_config(layout, authority["public_key_id"]),
            uid=layout.config_uid,
            gid=layout.config_gid,
            mode=0o444,
            include_digest=True,
        )
        replay_evidence = _ensure_directory(
            layout.replay_directory,
            uid=layout.replay_uid,
            gid=layout.replay_gid,
            mode=0o700,
        )
        sudoers_evidence: Mapping[str, Any] | None = None
        rendered_sudoers = _render_sudoers(layout)
        if rendered_sudoers is not None:
            assert layout.sudoers is not None
            _validate_sudoers_bytes(
                layout.sudoers, rendered_sudoers, validator=visudo_validator
            )
            sudoers_evidence = _install_exclusive(
                layout.sudoers,
                rendered_sudoers,
                uid=0,
                gid=0,
                mode=0o440,
                include_digest=True,
            )
        inert = _inert_evidence(layout, unit_probe=unit_probe)
        unsigned = _receipt_unsigned(
            layout=layout,
            authority=authority,
            owner_authorization_receipt_sha256=str(
                envelope["owner_authorization_receipt_sha256"]
            ),
            private_evidence=private_evidence,
            public_evidence=public_evidence,
            config_evidence=config_evidence,
            replay_evidence=replay_evidence,
            sudoers_evidence=sudoers_evidence,
            inert=inert,
        )
        receipt = _sign_receipt(unsigned, private_key)
        receipt_raw = foundation.canonical_json_bytes(receipt)
        _install_exclusive(
            layout.receipt,
            receipt_raw,
            uid=layout.receipt_uid,
            gid=layout.receipt_gid,
            mode=0o444,
            include_digest=True,
        )
        if layout.runtime_receipt is not None:
            _install_exclusive(
                layout.runtime_receipt,
                receipt_raw,
                uid=layout.runtime_receipt_uid,
                gid=layout.runtime_receipt_gid,
                mode=layout.runtime_receipt_mode,
                include_digest=True,
            )
        return _verify_receipt(
            receipt, public_key=Ed25519PublicKey.from_public_bytes(authority["public_raw"])
        )


def verify_signer_readiness(
    *,
    layout: SignerLayout,
    unit_probe: Callable[[str], Mapping[str, Any]] = _default_unit_probe,
) -> Mapping[str, Any]:
    """Re-attest every installed signer prerequisite without mutation."""

    authority = _validate_release_and_authority(layout)
    inert = _inert_evidence(layout, unit_probe=unit_probe)
    private_key, private_evidence = _verify_private_matches(
        layout, expected_public_raw=authority["public_raw"]
    )
    public_evidence = _file_evidence(
        layout.installed_public_key,
        uid=layout.config_uid,
        gid=layout.config_gid,
        mode=0o444,
        include_digest=True,
    )
    if _read_regular(
        layout.installed_public_key,
        maximum=32,
        uid=layout.config_uid,
        gid=layout.config_gid,
        mode=0o444,
    ) != authority["public_raw"]:
        _stable_error("trusted_signer_installed_public_key_invalid")
    expected_config = _render_config(layout, authority["public_key_id"])
    config_raw = _read_regular(
        layout.config,
        maximum=MAX_JSON_BYTES,
        uid=layout.config_uid,
        gid=layout.config_gid,
        mode=0o444,
    )
    if config_raw != expected_config:
        _stable_error("trusted_signer_config_invalid")
    config_evidence = _file_evidence(
        layout.config,
        uid=layout.config_uid,
        gid=layout.config_gid,
        mode=0o444,
        include_digest=True,
    )
    replay_evidence = _directory_evidence(
        layout.replay_directory,
        uid=layout.replay_uid,
        gid=layout.replay_gid,
        mode=0o700,
    )
    sudoers_evidence: Mapping[str, Any] | None = None
    rendered_sudoers = _render_sudoers(layout)
    if rendered_sudoers is not None:
        assert layout.sudoers is not None
        existing = _read_regular(
            layout.sudoers,
            maximum=64 * 1024,
            uid=0,
            gid=0,
            mode=0o440,
        )
        if existing != rendered_sudoers:
            _stable_error("trusted_signer_sudoers_invalid")
        sudoers_evidence = _file_evidence(
            layout.sudoers,
            uid=0,
            gid=0,
            mode=0o440,
            include_digest=True,
        )
    receipt_raw = _read_regular(
        layout.receipt,
        maximum=MAX_JSON_BYTES,
        uid=layout.receipt_uid,
        gid=layout.receipt_gid,
        mode=0o444,
    )
    receipt = _verify_receipt(
        _canonical_mapping(receipt_raw, code="trusted_signer_receipt_invalid"),
        public_key=private_key.public_key(),
    )
    if layout.runtime_receipt is not None:
        runtime_receipt_raw = _read_regular(
            layout.runtime_receipt,
            maximum=MAX_JSON_BYTES,
            uid=layout.runtime_receipt_uid,
            gid=layout.runtime_receipt_gid,
            mode=layout.runtime_receipt_mode,
        )
        if runtime_receipt_raw != receipt_raw:
            _stable_error("trusted_signer_runtime_receipt_mismatch")
    expected_unsigned = _receipt_unsigned(
        layout=layout,
        authority=authority,
        owner_authorization_receipt_sha256=str(
            receipt.get("owner_authorization_receipt_sha256", "")
        ),
        private_evidence=private_evidence,
        public_evidence=public_evidence,
        config_evidence=config_evidence,
        replay_evidence=replay_evidence,
        sudoers_evidence=sudoers_evidence,
        inert=inert,
    )
    if (
        _SHA256.fullmatch(
            str(receipt.get("owner_authorization_receipt_sha256", ""))
        )
        is None
        or receipt != _sign_receipt(expected_unsigned, private_key)
    ):
        _stable_error("trusted_signer_receipt_state_mismatch")
    unsigned = {
        "schema": READINESS_SCHEMA,
        "role": layout.role,
        "release_revision": layout.release.name,
        "package_sha256": authority["package_sha256"],
        "public_key_id": authority["public_key_id"],
        "provisioning_receipt_sha256": receipt["receipt_sha256"],
        "private_public_identity_matched": True,
        "config_exact": True,
        "replay_directory_exact": True,
        "sudoers_exact": sudoers_evidence is not None,
        "offline_runtime_exact": True,
        "activation_seal_absent": True,
        "current_link_absent": True,
        "services_inactive_disabled": True,
        "activation_performed": False,
        "iam_mutation_performed": False,
    }
    return {**unsigned, "readiness_sha256": foundation.sha256_json(unsigned)}


def verify_cloud_signer_inert_readiness(
    release_revision: str | None = None,
    *,
    unit_probe: Callable[[str], Mapping[str, Any]] = _default_unit_probe,
) -> Mapping[str, Any]:
    """Root-only post-provisioning gate before release selection/activation."""

    revision = release_revision or _runtime_release_revision(
        expected_base=OWNER_GATE_RELEASE_BASE
    )
    return verify_signer_readiness(
        layout=cloud_layout(revision), unit_probe=unit_probe
    )


def _selected_release_evidence(layout: SignerLayout) -> Mapping[str, Any]:
    try:
        state = layout.current_link.lstat()
        target = os.readlink(layout.current_link)
        resolved = layout.current_link.resolve(strict=True)
    except OSError as exc:
        _stable_error("trusted_signer_current_release_invalid", exc)
    if (
        not stat.S_ISLNK(state.st_mode)
        or state.st_uid != 0
        or state.st_gid != 0
        or resolved != layout.release
        or target != str(layout.release)
    ):
        _stable_error("trusted_signer_current_release_invalid")
    return {
        "path": str(layout.current_link),
        "target": target,
        "resolved_release": str(resolved),
        "uid": state.st_uid,
        "exact_release_selected": True,
    }


def verify_cloud_signer_runtime_readiness(
    release_revision: str | None = None,
) -> Mapping[str, Any]:
    """Executor-UID-safe static signer readiness after exact release selection.

    The historical inert snapshot is verified through the signed provisioning
    receipt; it is not recomputed after activation.  Root-only sudoers remains
    covered by that receipt and by the separate root inert gate.  The mutation
    activation seal is intentionally independent and is checked by the
    executor's mutation authorization path, not this signer prerequisite.
    """

    revision = release_revision or _runtime_release_revision(
        expected_base=OWNER_GATE_RELEASE_BASE
    )
    layout = cloud_layout(revision)
    authority = _validate_release_and_authority(layout)
    selection = _selected_release_evidence(layout)
    private_key, private_evidence = _verify_private_matches(
        layout, expected_public_raw=authority["public_raw"]
    )
    public_raw = _read_regular(
        layout.installed_public_key,
        maximum=32,
        uid=layout.config_uid,
        gid=layout.config_gid,
        mode=0o444,
    )
    if public_raw != authority["public_raw"]:
        _stable_error("trusted_signer_installed_public_key_invalid")
    public_evidence = _file_evidence(
        layout.installed_public_key,
        uid=layout.config_uid,
        gid=layout.config_gid,
        mode=0o444,
        include_digest=True,
    )
    config_raw = _read_regular(
        layout.config,
        maximum=MAX_JSON_BYTES,
        uid=layout.config_uid,
        gid=layout.config_gid,
        mode=0o444,
    )
    if config_raw != _render_config(layout, authority["public_key_id"]):
        _stable_error("trusted_signer_config_invalid")
    config_evidence = _file_evidence(
        layout.config,
        uid=layout.config_uid,
        gid=layout.config_gid,
        mode=0o444,
        include_digest=True,
    )
    replay_evidence = _directory_evidence(
        layout.replay_directory,
        uid=layout.replay_uid,
        gid=layout.replay_gid,
        mode=0o700,
    )
    if layout.runtime_receipt is None:
        _stable_error("trusted_signer_runtime_receipt_unavailable")
    receipt_raw = _read_regular(
        layout.runtime_receipt,
        maximum=MAX_JSON_BYTES,
        uid=layout.runtime_receipt_uid,
        gid=layout.runtime_receipt_gid,
        mode=layout.runtime_receipt_mode,
    )
    receipt = _verify_receipt(
        _canonical_mapping(receipt_raw, code="trusted_signer_receipt_invalid"),
        public_key=private_key.public_key(),
    )
    historical_inert = receipt.get("inert_boundary")
    historical_sudoers = receipt.get("sudoers")
    rendered_sudoers = _render_sudoers(layout)
    if rendered_sudoers is None:
        _stable_error("trusted_signer_sudoers_invalid")
    if (
        receipt.get("role") != "cloud"
        or receipt.get("release_revision") != revision
        or receipt.get("package_sha256") != authority["package_sha256"]
        or receipt.get("package_manifest_sha256") != authority["manifest_sha256"]
        or receipt.get("public_key_id") != authority["public_key_id"]
        or receipt.get("pinned_public_key_sha256") != authority["public_key_id"]
        or receipt.get("private_key") != private_evidence
        or receipt.get("installed_public_key") != public_evidence
        or receipt.get("config") != config_evidence
        or receipt.get("replay_directory") != replay_evidence
        or receipt.get("runtime") != authority["runtime"]
        or receipt.get("runtime_receipt")
        != {
            "path": str(layout.runtime_receipt),
            "uid": layout.runtime_receipt_uid,
            "gid": layout.runtime_receipt_gid,
            "mode": f"{layout.runtime_receipt_mode:04o}",
            "authoritative_copy": False,
            "byte_exact_signed_public_projection": True,
        }
        or not isinstance(historical_sudoers, Mapping)
        or historical_sudoers.get("path") != str(layout.sudoers)
        or historical_sudoers.get("uid") != 0
        or historical_sudoers.get("gid") != 0
        or historical_sudoers.get("mode") != "0440"
        or historical_sudoers.get("size") != len(rendered_sudoers)
        or historical_sudoers.get("sha256")
        != hashlib.sha256(rendered_sudoers).hexdigest()
        or _SHA256.fullmatch(str(historical_sudoers.get("sha256", ""))) is None
        or not isinstance(historical_inert, Mapping)
        or historical_inert.get("activation_seal_absent") is not True
        or historical_inert.get("activation_seal") != str(layout.activation_seal)
        or historical_inert.get("current_link_absent") is not True
        or historical_inert.get("current_link") != str(layout.current_link)
        or historical_inert.get("services_started") is not False
        or receipt.get("activation_performed") is not False
        or receipt.get("iam_mutation_performed") is not False
    ):
        _stable_error("trusted_signer_receipt_state_mismatch")
    unsigned = {
        "schema": "muncho-cloud-trusted-signer-runtime-readiness.v1",
        "role": "cloud",
        "release_revision": revision,
        "package_sha256": authority["package_sha256"],
        "public_key_id": authority["public_key_id"],
        "provisioning_receipt_sha256": receipt["receipt_sha256"],
        "selected_release": selection,
        "private_public_identity_matched": True,
        "config_exact": True,
        "replay_directory_exact": True,
        "offline_runtime_exact": True,
        "historical_root_inert_receipt_verified": True,
        "root_sudoers_digest_verified_via_signed_receipt": True,
        "live_root_sudoers_read_attempted": False,
        "activation_seal_checked_by_mutation_boundary": True,
        "service_state_recomputed_in_executor": False,
        "iam_mutation_performed": False,
    }
    return {**unsigned, "readiness_sha256": foundation.sha256_json(unsigned)}


def verify_cloud_signer_readiness(
    release_revision: str | None = None,
) -> Mapping[str, Any]:
    """Backward-stable name for the executor-safe runtime readiness hook."""

    return verify_cloud_signer_runtime_readiness(release_revision)


def verify_host_signer_runtime_readiness(
    release_revision: str | None = None,
    *,
    unit_probe: Callable[[str], Mapping[str, Any]] = _default_unit_probe,
) -> Mapping[str, Any]:
    """Fixed host pre-collection gate with the full stopped-unit inventory."""

    revision = release_revision or _runtime_release_revision(
        expected_base=HOST_RELEASE_BASE
    )
    return verify_signer_readiness(
        layout=host_layout(revision), unit_probe=unit_probe
    )


def _runtime_release_revision(*, expected_base: Path) -> str:
    executable = Path(sys.executable).resolve(strict=True)
    candidates = [parent for parent in executable.parents if parent.parent == expected_base]
    if len(candidates) != 1 or _REVISION.fullmatch(candidates[0].name) is None:
        _stable_error("trusted_signer_runtime_invalid")
    return candidates[0].name


def _run_fixed_main(
    *,
    role: str,
    layout_builder: Callable[[str], SignerLayout],
    argv: Sequence[str] | None,
    stdin: BinaryIO,
    stdout: BinaryIO,
    expected_release_base: Path,
    expected_euid: int,
) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments or os.geteuid() != expected_euid:  # windows-footgun: ok — Debian runtime boundary
        _stable_error("trusted_signer_entrypoint_invalid")
    revision = _runtime_release_revision(expected_base=expected_release_base)
    receipt = provision_signer(
        _read_stdin_frame(stdin), layout=layout_builder(revision)
    )
    stdout.write(foundation.canonical_json_bytes(receipt) + b"\n")
    stdout.flush()
    return 0


def cloud_provision_main(argv: Sequence[str] | None = None) -> int:
    return _run_fixed_main(
        role="cloud",
        layout_builder=cloud_layout,
        argv=argv,
        stdin=sys.stdin.buffer,
        stdout=sys.stdout.buffer,
        expected_release_base=OWNER_GATE_RELEASE_BASE,
        expected_euid=0,
    )


def host_provision_main(argv: Sequence[str] | None = None) -> int:
    return _run_fixed_main(
        role="host",
        layout_builder=host_layout,
        argv=argv,
        stdin=sys.stdin.buffer,
        stdout=sys.stdout.buffer,
        expected_release_base=HOST_RELEASE_BASE,
        expected_euid=0,
    )


__all__ = [
    "CLOUD_EXECUTOR_UID",
    "ENVELOPE_SCHEMAS",
    "HOST_RELEASE_BASE",
    "PROVISIONING_RECEIPT_SCHEMA",
    "READINESS_SCHEMA",
    "SignerLayout",
    "TrustedSignerProvisioningError",
    "cloud_layout",
    "cloud_provision_main",
    "decode_provisioning_envelope",
    "host_layout",
    "host_provision_main",
    "provision_signer",
    "verify_cloud_signer_readiness",
    "verify_cloud_signer_inert_readiness",
    "verify_cloud_signer_runtime_readiness",
    "verify_host_signer_runtime_readiness",
    "verify_signer_readiness",
]
