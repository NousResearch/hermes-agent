#!/usr/bin/env python3
"""Initialize and use the offline Muncho owner-gate release signing key.

The private Ed25519 seed is created outside the repository, is never printed,
and is accepted only from a caller-owned mode-0600 regular file.  Signing is
possible only after the corresponding public-key digest has been pinned in
``owner_gate_trust.py`` by a reviewed fork commit.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pwd
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import direct_iam_identity_authority as direct_iam
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_trust as trust


PRIVATE_KEY_NAME = "release-trust-signing.key"
PUBLIC_KEY_NAME = "release-trust-signing.pub"
OWNER_HOME = Path(pwd.getpwuid(os.geteuid()).pw_dir)  # windows-footgun: ok — POSIX owner boundary
AUTHORITY_PARENT = OWNER_HOME / ".hermes"
KEY_DIRECTORY = AUTHORITY_PARENT / "owner-gate-release-authority"
MANIFEST_DIRECTORY = KEY_DIRECTORY / "manifests"
PRIVATE_KEY_BYTES = 32
PUBLIC_KEY_BYTES = 32
SIGNATURE_BYTES = 64
_MAX_UNSIGNED_BYTES = 4 * 1024 * 1024


class OwnerGateTrustAuthorError(RuntimeError):
    """Stable, secret-free offline authoring failure."""


_FOUNDATION_CHAIN_MARKER = object()


@dataclass(frozen=True, init=False)
class _ValidatedFoundationChain:
    """Private opaque projection of the journal-backed A lifecycle."""

    final_release_revision: str
    pre_foundation_authority_sha256: str
    foundation_apply_receipt_sha256: str
    bootstrap_network_collector_public_key_id: str
    foundation_source_revision: str
    foundation_source_tree_oid: str
    direct_iam_identity_authority_sha256: str
    project_ancestry_evidence_sha256: str
    project_ancestry_chain_sha256: str
    resource_ancestor_chain: tuple[str, ...]
    interpreter_sha256: str
    interpreter_version: str
    _marker: object

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "_ValidatedFoundationChain":
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_foundation_chain_factory_required"
        )

    @classmethod
    def _create(
        cls,
        *,
        final_release_revision: str,
        pre_foundation_authority_sha256: str,
        foundation_apply_receipt_sha256: str,
        bootstrap_network_collector_public_key_id: str,
        foundation_source_revision: str,
        foundation_source_tree_oid: str,
        direct_iam_identity_authority_sha256: str,
        project_ancestry_evidence_sha256: str,
        project_ancestry_chain_sha256: str,
        resource_ancestor_chain: tuple[str, ...],
        interpreter_sha256: str,
        interpreter_version: str,
    ) -> "_ValidatedFoundationChain":
        value = object.__new__(cls)
        for name, item in {
            "final_release_revision": final_release_revision,
            "pre_foundation_authority_sha256": (
                pre_foundation_authority_sha256
            ),
            "foundation_apply_receipt_sha256": (
                foundation_apply_receipt_sha256
            ),
            "bootstrap_network_collector_public_key_id": (
                bootstrap_network_collector_public_key_id
            ),
            "foundation_source_revision": foundation_source_revision,
            "foundation_source_tree_oid": foundation_source_tree_oid,
            "direct_iam_identity_authority_sha256": (
                direct_iam_identity_authority_sha256
            ),
            "project_ancestry_evidence_sha256": (
                project_ancestry_evidence_sha256
            ),
            "project_ancestry_chain_sha256": project_ancestry_chain_sha256,
            "resource_ancestor_chain": resource_ancestor_chain,
            "interpreter_sha256": interpreter_sha256,
            "interpreter_version": interpreter_version,
        }.items():
            object.__setattr__(value, name, item)
        object.__setattr__(value, "_marker", _FOUNDATION_CHAIN_MARKER)
        return value


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fsync(descriptor)
    except OSError as exc:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_directory_fsync_failed"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _require_owner_directory(
    path: Path,
    *,
    expected: Path,
    parent: Path,
    create: bool,
) -> None:
    if (
        not path.is_absolute()
        or ".." in path.parts
        or path != expected
        or path.parent != parent
    ):
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_directory_invalid"
        )
    try:
        parent_before = parent.lstat()
        parent_resolved = parent.resolve(strict=True)
        parent_after = parent_resolved.stat()
    except OSError as exc:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_parent_invalid"
        ) from None
    if (
        stat.S_ISLNK(parent_before.st_mode)
        or not stat.S_ISDIR(parent_after.st_mode)
        or parent_resolved != parent
        or (parent_before.st_dev, parent_before.st_ino)
        != (parent_after.st_dev, parent_after.st_ino)
        or parent_after.st_uid != os.geteuid()  # windows-footgun: ok — POSIX owner boundary
        or stat.S_IMODE(parent_after.st_mode) != 0o700
    ):
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_parent_invalid"
        )
    if create:
        try:
            path.mkdir(mode=0o700)
            _fsync_directory(path.parent)
        except FileExistsError:
            pass
        except OSError as exc:
            raise OwnerGateTrustAuthorError(
                "owner_gate_trust_author_directory_unavailable"
            ) from None
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
        after = resolved.stat()
    except OSError as exc:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_directory_unavailable"
        ) from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(after.st_mode)
        or resolved != path
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or after.st_uid != os.geteuid()  # windows-footgun: ok — POSIX owner boundary
        or stat.S_IMODE(after.st_mode) != 0o700
    ):
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_directory_invalid"
        )


def _read_exact_regular(
    path: Path,
    *,
    size: int,
    modes: frozenset[int],
    code: str,
) -> bytes:
    descriptor: int | None = None
    try:
        if not path.is_absolute() or ".." in path.parts:
            raise OwnerGateTrustAuthorError(code)
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
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()  # windows-footgun: ok — POSIX owner boundary
            or stat.S_IMODE(opened.st_mode) not in modes
            or opened.st_size != size
        ):
            raise OwnerGateTrustAuthorError(code)
        raw = bytearray()
        while len(raw) < size:
            chunk = os.read(descriptor, size - len(raw))
            if not chunk:
                raise OwnerGateTrustAuthorError(code)
            raw.extend(chunk)
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
            raise OwnerGateTrustAuthorError(code)
        return bytes(raw)
    except OwnerGateTrustAuthorError:
        raise
    except OSError as exc:
        raise OwnerGateTrustAuthorError(code) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_publish_file(
    path: Path,
    *,
    mode: int,
    allowed_nlinks: frozenset[int],
    maximum: int,
    code: str,
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
            or opened.st_nlink not in allowed_nlinks
            or opened.st_uid != os.geteuid()  # windows-footgun: ok — POSIX owner boundary
            or stat.S_IMODE(opened.st_mode) != mode
            or opened.st_size < 0
            or opened.st_size > maximum
        ):
            raise OwnerGateTrustAuthorError(code)
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise OwnerGateTrustAuthorError(code)
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
            opened.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        ):
            raise OwnerGateTrustAuthorError(code)
        return b"".join(chunks), opened
    except OwnerGateTrustAuthorError:
        raise
    except OSError as exc:
        raise OwnerGateTrustAuthorError(code) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _discard_empty_unsealed_stage(path: Path, *, code: str) -> None:
    """Recover only the pre-fchmod empty stage left by an interrupted open."""

    if not (path.exists() or path.is_symlink()):
        return
    descriptor: int | None = None
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()  # windows-footgun: ok — POSIX owner boundary
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_size != 0
        ):
            raise OwnerGateTrustAuthorError(code)
        os.fsync(descriptor)
    except OwnerGateTrustAuthorError:
        raise
    except OSError as exc:
        raise OwnerGateTrustAuthorError(code) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        path.unlink()
    except OSError as exc:
        raise OwnerGateTrustAuthorError(code) from None
    _fsync_directory(path.parent)


def _publish_exclusive(path: Path, raw: bytes, *, mode: int, code: str) -> None:
    """Crash-recoverable no-clobber publication in one private directory."""

    if (
        not path.is_absolute()
        or ".." in path.parts
        or path.parent.resolve(strict=True) != path.parent
        or not raw
    ):
        raise OwnerGateTrustAuthorError(code)
    stage = path.with_name(f".{path.name}.stage")
    if path.exists() or path.is_symlink():
        final_raw, final_state = _read_publish_file(
            path,
            mode=mode,
            allowed_nlinks=frozenset({1, 2}),
            maximum=len(raw),
            code=code,
        )
        if final_raw != raw or final_state.st_size != len(raw):
            raise OwnerGateTrustAuthorError(code)
        if final_state.st_nlink == 2:
            stage_raw, stage_state = _read_publish_file(
                stage,
                mode=mode,
                allowed_nlinks=frozenset({2}),
                maximum=len(raw),
                code=code,
            )
            if (
                stage_raw != raw
                or (stage_state.st_dev, stage_state.st_ino)
                != (final_state.st_dev, final_state.st_ino)
            ):
                raise OwnerGateTrustAuthorError(code)
            try:
                stage.unlink()
            except OSError as exc:
                raise OwnerGateTrustAuthorError(code) from None
            _fsync_directory(path.parent)
        elif stage.exists() or stage.is_symlink():
            raise OwnerGateTrustAuthorError(code)
        _read_exact_regular(
            path,
            size=len(raw),
            modes=frozenset({mode}),
            code=code,
        )
        return

    descriptor: int | None = None
    try:
        if mode != 0o600:
            _discard_empty_unsealed_stage(stage, code=code)
        if stage.exists() or stage.is_symlink():
            prefix, stage_state = _read_publish_file(
                stage,
                mode=mode,
                allowed_nlinks=frozenset({1}),
                maximum=len(raw),
                code=code,
            )
            if prefix != raw[: len(prefix)]:
                raise OwnerGateTrustAuthorError(code)
            descriptor = os.open(
                stage,
                os.O_WRONLY
                | os.O_APPEND
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if (
                (opened.st_dev, opened.st_ino)
                != (stage_state.st_dev, stage_state.st_ino)
                or opened.st_nlink != 1
                or opened.st_size != len(prefix)
            ):
                raise OwnerGateTrustAuthorError(code)
            offset = len(prefix)
        else:
            descriptor = os.open(
                stage,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            os.fchmod(descriptor, mode)
            offset = 0
        view = memoryview(raw)
        try:
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise OSError("short trust-author stage write")
                offset += written
        finally:
            view.release()
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        written_state = os.fstat(descriptor)
        if (
            not stat.S_ISREG(written_state.st_mode)
            or written_state.st_nlink != 1
            or written_state.st_uid != os.geteuid()  # windows-footgun: ok — POSIX owner boundary
            or stat.S_IMODE(written_state.st_mode) != mode
            or written_state.st_size != len(raw)
        ):
            raise OwnerGateTrustAuthorError(code)
    except OwnerGateTrustAuthorError:
        raise
    except OSError as exc:
        raise OwnerGateTrustAuthorError(code) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        os.link(stage, path, follow_symlinks=False)
        _fsync_directory(path.parent)
        stage_state = stage.lstat()
        final_state = path.lstat()
        if (
            not stat.S_ISREG(stage_state.st_mode)
            or not stat.S_ISREG(final_state.st_mode)
            or (stage_state.st_dev, stage_state.st_ino)
            != (final_state.st_dev, final_state.st_ino)
            or final_state.st_nlink != 2
        ):
            raise OwnerGateTrustAuthorError(code)
        stage.unlink()
        _fsync_directory(path.parent)
    except OwnerGateTrustAuthorError:
        raise
    except OSError as exc:
        raise OwnerGateTrustAuthorError(code) from None
    final = _read_exact_regular(
        path,
        size=len(raw),
        modes=frozenset({mode}),
        code=code,
    )
    if final != raw:
        raise OwnerGateTrustAuthorError(code)


def _public_raw(private_raw: bytes) -> bytes:
    try:
        return Ed25519PrivateKey.from_private_bytes(private_raw).public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    except ValueError as exc:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_private_key_invalid"
        ) from None


def _recover_private_stage(path: Path) -> None:
    stage = path.with_name(f".{path.name}.stage")
    if path.exists() or path.is_symlink() or not (stage.exists() or stage.is_symlink()):
        return
    raw, state = _read_publish_file(
        stage,
        mode=0o600,
        allowed_nlinks=frozenset({1}),
        maximum=PRIVATE_KEY_BYTES,
        code="owner_gate_trust_author_private_key_recovery_failed",
    )
    if len(raw) == PRIVATE_KEY_BYTES:
        _publish_exclusive(
            path,
            raw,
            mode=0o600,
            code="owner_gate_trust_author_private_key_recovery_failed",
        )
        return
    descriptor: int | None = None
    try:
        descriptor = os.open(
            stage,
            os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (state.st_dev, state.st_ino)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()  # windows-footgun: ok — POSIX owner boundary
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_size != len(raw)
        ):
            raise OwnerGateTrustAuthorError(
                "owner_gate_trust_author_private_key_recovery_failed"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        remaining = len(raw)
        zeroes = b"\x00" * min(remaining, 4096)
        while remaining:
            count = os.write(descriptor, zeroes[:remaining])
            if count <= 0:
                raise OSError("short private stage wipe")
            remaining -= count
        os.ftruncate(descriptor, 0)
        os.fsync(descriptor)
    except OwnerGateTrustAuthorError:
        raise
    except OSError as exc:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_private_key_recovery_failed"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        stage.unlink()
    except OSError as exc:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_private_key_recovery_failed"
        ) from None
    _fsync_directory(path.parent)


def initialize_keypair(directory: Path | None = None) -> Mapping[str, Any]:
    """Create or verify the one stable keypair without revealing its seed."""

    directory = KEY_DIRECTORY if directory is None else directory
    _require_owner_directory(
        directory,
        expected=KEY_DIRECTORY,
        parent=AUTHORITY_PARENT,
        create=True,
    )
    _require_owner_directory(
        MANIFEST_DIRECTORY,
        expected=MANIFEST_DIRECTORY,
        parent=KEY_DIRECTORY,
        create=True,
    )
    private_path = directory / PRIVATE_KEY_NAME
    public_path = directory / PUBLIC_KEY_NAME
    _recover_private_stage(private_path)
    private_exists = private_path.exists()
    public_exists = public_path.exists()
    if not private_exists:
        if public_exists:
            raise OwnerGateTrustAuthorError(
                "owner_gate_trust_author_keypair_incomplete"
            )
        seed = bytearray(os.urandom(PRIVATE_KEY_BYTES))
        try:
            _publish_exclusive(
                private_path,
                bytes(seed),
                mode=0o600,
                code="owner_gate_trust_author_private_key_write_failed",
            )
        finally:
            for index in range(len(seed)):
                seed[index] = 0
    private_raw = _read_exact_regular(
        private_path,
        size=PRIVATE_KEY_BYTES,
        modes=frozenset({0o600}),
        code="owner_gate_trust_author_private_key_invalid",
    )
    expected_public = _public_raw(private_raw)
    if not public_exists:
        _publish_exclusive(
            public_path,
            expected_public,
            mode=0o444,
            code="owner_gate_trust_author_public_key_write_failed",
        )
    public_raw = _read_exact_regular(
        public_path,
        size=PUBLIC_KEY_BYTES,
        modes=frozenset({0o400, 0o440, 0o444}),
        code="owner_gate_trust_author_public_key_invalid",
    )
    if public_raw != expected_public:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_keypair_mismatch"
        )
    key_id = hashlib.sha256(public_raw).hexdigest()
    return {
        "schema": "muncho-owner-gate-release-key-initialization.v1",
        "key_initialized": True,
        "private_key_material_printed": False,
        "private_key_digest_printed": False,
        "public_key_path": str(public_path),
        "public_key_sha256": key_id,
    }


def _read_unsigned(path: Path) -> Mapping[str, Any]:
    try:
        raw = trust._read_immutable(
            path,
            maximum=_MAX_UNSIGNED_BYTES,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, trust.OwnerGateTrustError) as exc:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_unsigned_manifest_invalid"
        ) from None
    if not isinstance(value, Mapping):
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_unsigned_manifest_invalid"
        )
    try:
        trust._validate_unsigned(value)
    except trust.OwnerGateTrustError as exc:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_unsigned_manifest_invalid"
        ) from None
    if foundation.canonical_json_bytes(value) != raw:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_unsigned_manifest_not_canonical"
        )
    return dict(value)


def _read_chain_bytes(path: Path, *, maximum: int, code: str) -> bytes:
    try:
        return trust._read_immutable(
            path,
            maximum=maximum,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
    except trust.OwnerGateTrustError as exc:
        raise OwnerGateTrustAuthorError(code) from None


def _validate_direct_iam_apply_identity(
    direct_authority: Mapping[str, Any],
    apply_receipt: Mapping[str, Any],
) -> None:
    steps = apply_receipt.get("applied_steps")
    if not isinstance(steps, list):
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_foundation_identity_mismatch"
        )
    identities = {
        item.get("step_name"): item.get("resource_identity")
        for item in steps
        if isinstance(item, Mapping)
    }
    required = {
        "create_dedicated_service_account",
        "create_narrow_iam_observation_reader_role",
        "bind_narrow_iam_observation_reader_to_owner_gate_service_account",
        "create_narrow_storage_executor_role",
        "create_narrow_organization_iam_observation_reader_role",
        "bind_narrow_organization_iam_observation_reader_to_owner_gate_service_account",
        "create_private_owner_gate_vm",
    }
    if not required <= set(identities) or any(
        not isinstance(identities[name], Mapping) for name in required
    ):
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_foundation_identity_mismatch"
        )
    service_account = identities["create_dedicated_service_account"]
    project_role = identities["create_narrow_iam_observation_reader_role"]
    project_binding = identities[
        "bind_narrow_iam_observation_reader_to_owner_gate_service_account"
    ]
    mutation_role = identities["create_narrow_storage_executor_role"]
    ancestor_role = identities[
        "create_narrow_organization_iam_observation_reader_role"
    ]
    ancestor_binding = identities[
        "bind_narrow_organization_iam_observation_reader_to_owner_gate_service_account"
    ]
    vm = identities["create_private_owner_gate_vm"]
    trust_root = direct_authority.get("external_gcp_admin_trust_root")
    generations = (
        trust_root.get("resource_policy_generations")
        if isinstance(trust_root, Mapping)
        else None
    )
    generation_by_resource = {
        item.get("resource"): item
        for item in generations
        if isinstance(item, Mapping)
    } if isinstance(generations, list) else {}
    project_generation = generation_by_resource.get(
        f"projects/{foundation.PROJECT}"
    )
    organization_generation = generation_by_resource.get(
        ancestor_binding.get("resource_name")
    )
    if (
        direct_authority.get("owner_gate_vm_numeric_id") != vm.get("numeric_id")
        or direct_authority.get("owner_gate_vm_name") != vm.get("name")
        or direct_authority.get("owner_gate_service_account_email")
        != service_account.get("email")
        or direct_authority.get("owner_gate_service_account_email")
        != vm.get("service_account_email")
        or direct_authority.get("owner_gate_service_account_unique_id")
        != service_account.get("unique_id")
        or direct_authority.get("project_read_role") != project_role.get("name")
        or direct_authority.get("project_read_role_etag")
        != project_role.get("etag")
        or direct_authority.get("project_read_binding_member")
        != project_binding.get("member")
        or direct_authority.get("ancestor_read_role")
        != ancestor_role.get("name")
        or direct_authority.get("ancestor_read_role_etag")
        != ancestor_role.get("etag")
        or direct_authority.get("ancestor_binding_member")
        != ancestor_binding.get("member")
        or direct_authority.get("mutation_role") != mutation_role.get("name")
        or direct_authority.get("mutation_role_etag")
        != mutation_role.get("etag")
        or not isinstance(project_generation, Mapping)
        or project_generation.get("etag") != project_binding.get("policy_etag")
        or not isinstance(organization_generation, Mapping)
        or organization_generation.get("etag")
        != ancestor_binding.get("policy_etag")
    ):
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_foundation_identity_mismatch"
        )


def _validate_foundation_chain_files(
    *,
    pre_foundation_authority_path: Path,
    owner_reauthentication_receipt_path: Path,
    network_evidence_path: Path,
    network_collector_public_key_path: Path,
    project_ancestry_evidence_path: Path,
    project_ancestry_collector_public_key_path: Path,
    direct_iam_identity_authority_path: Path,
    release_public_key: Ed25519PublicKey,
    final_release_revision: str,
) -> _ValidatedFoundationChain:
    """Validate the canonical signed A lifecycle before authoring B trust."""

    # Lazy imports break the intentional authoring cycle: ancestry collection
    # loads the release-author paths, while this author validates ancestry only
    # after its own immutable authority paths have been initialized.
    from scripts.canary import owner_gate_foundation_apply as foundation_apply
    from scripts.canary import owner_gate_pre_foundation as pre_foundation
    from scripts.canary import owner_gate_project_ancestry as project_ancestry

    try:
        if (
            network_collector_public_key_path
            == project_ancestry_collector_public_key_path
        ):
            raise ValueError
        reauth_raw = _read_chain_bytes(
            owner_reauthentication_receipt_path,
            maximum=owner_reauth.MAX_CAPTURE_BYTES,
            code="owner_gate_trust_author_foundation_chain_invalid",
        )
        authority_raw = _read_chain_bytes(
            pre_foundation_authority_path,
            maximum=pre_foundation.MAX_JSON_BYTES,
            code="owner_gate_trust_author_foundation_chain_invalid",
        )
        network_evidence_raw = _read_chain_bytes(
            network_evidence_path,
            maximum=pre_foundation.MAX_JSON_BYTES,
            code="owner_gate_trust_author_foundation_chain_invalid",
        )
        network_key_raw = _read_chain_bytes(
            network_collector_public_key_path,
            maximum=32,
            code="owner_gate_trust_author_foundation_chain_invalid",
        )
        ancestry_key_raw = _read_chain_bytes(
            project_ancestry_collector_public_key_path,
            maximum=32,
            code="owner_gate_trust_author_foundation_chain_invalid",
        )
        project_ancestry_raw = _read_chain_bytes(
            project_ancestry_evidence_path,
            maximum=project_ancestry.MAX_JSON_BYTES,
            code="owner_gate_trust_author_foundation_chain_invalid",
        )
        direct_iam_raw = _read_chain_bytes(
            direct_iam_identity_authority_path,
            maximum=direct_iam.MAX_BYTES,
            code="owner_gate_trust_author_foundation_chain_invalid",
        )
        loaded_network_key = Ed25519PublicKey.from_public_bytes(network_key_raw)
        loaded_ancestry_key = Ed25519PublicKey.from_public_bytes(
            ancestry_key_raw
        )
        apply_chain = (
            foundation_apply._load_validated_foundation_apply_chain_for_source_recovery(
                pre_foundation_authority_raw=authority_raw,
                owner_reauthentication_receipt_raw=reauth_raw,
                network_evidence_raw=network_evidence_raw,
                project_ancestry_evidence_raw=project_ancestry_raw,
                release_public_key=release_public_key,
                network_collector_public_key=loaded_network_key,
                project_ancestry_collector_public_key=loaded_ancestry_key,
            )
        )
        foundation_a = apply_chain.foundation_a
        reauthentication = foundation_a.owner_reauthentication_receipt
        authority = foundation_a.authority
        ancestry = foundation_a.ancestry_evidence
        apply_receipt = apply_chain.apply_receipt
        direct_authority = direct_iam.decode_canonical(
            direct_iam_raw,
            release_revision=apply_chain.foundation_source_revision,
        )
    except (
        KeyError,
        OSError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
        foundation.OwnerGateFoundationError,
        owner_reauth.OwnerGateOwnerReauthError,
        foundation_apply.OwnerGateFoundationApplyError,
        pre_foundation.OwnerGatePreFoundationError,
        project_ancestry.OwnerGateProjectAncestryError,
        direct_iam.DirectIamIdentityAuthorityError,
    ) as exc:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_foundation_chain_invalid"
        ) from None
    resource_ancestor_chain = tuple(
        str(node["resource_name"])
        for node in ancestry.ordered_chain[1:]
    )
    if (
        apply_chain.foundation_source_revision == final_release_revision
        or authority["foundation_source_revision"]
        != apply_chain.foundation_source_revision
        or authority["foundation_source_tree_oid"]
        != apply_chain.foundation_source_tree_oid
        or authority["ancestry_evidence_sha256"]
        != ancestry.signed_evidence_sha256
        or authority["ancestry_chain_sha256"]
        != ancestry.value["stable_chain_sha256"]
        or direct_authority["resource_ancestor_chain"]
        != list(resource_ancestor_chain)
        or direct_authority["pre_foundation_authority_sha256"]
        != authority["pre_foundation_authority_sha256"]
        or direct_authority["foundation_apply_receipt_sha256"]
        != apply_receipt["foundation_apply_receipt_sha256"]
        or direct_authority["owner_reauthentication_receipt_sha256"]
        != reauthentication["owner_reauthentication_receipt_sha256"]
        or direct_authority["collected_at_unix"]
        < apply_receipt["completed_at_unix"]
        or direct_authority["collected_at_unix"]
        < reauthentication["issued_at_unix"]
        or direct_authority["collected_at_unix"]
        > reauthentication["expires_at_unix"]
    ):
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_foundation_chain_invalid"
        )
    _validate_direct_iam_apply_identity(direct_authority, apply_receipt)
    image = authority["interpreter_image"]
    return _ValidatedFoundationChain._create(
        final_release_revision=final_release_revision,
        pre_foundation_authority_sha256=authority[
            "pre_foundation_authority_sha256"
        ],
        foundation_apply_receipt_sha256=apply_receipt[
            "foundation_apply_receipt_sha256"
        ],
        bootstrap_network_collector_public_key_id=authority[
            "network_collector_public_key_id"
        ],
        foundation_source_revision=apply_chain.foundation_source_revision,
        foundation_source_tree_oid=apply_chain.foundation_source_tree_oid,
        direct_iam_identity_authority_sha256=hashlib.sha256(
            direct_iam_raw
        ).hexdigest(),
        project_ancestry_evidence_sha256=ancestry.signed_evidence_sha256,
        project_ancestry_chain_sha256=str(
            ancestry.value["stable_chain_sha256"]
        ),
        resource_ancestor_chain=resource_ancestor_chain,
        interpreter_sha256=image["interpreter_sha256"],
        interpreter_version=image["python_version"],
    )


def sign_manifest(
    *,
    unsigned_path: Path,
    private_key_path: Path,
    public_key_path: Path,
    output_path: Path,
    pre_foundation_authority_path: Path,
    owner_reauthentication_receipt_path: Path,
    network_evidence_path: Path,
    network_collector_public_key_path: Path,
    project_ancestry_evidence_path: Path,
    project_ancestry_collector_public_key_path: Path,
    direct_iam_identity_authority_path: Path,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Sign one exact canonical manifest after its public key is fork-pinned."""

    _require_owner_directory(
        KEY_DIRECTORY,
        expected=KEY_DIRECTORY,
        parent=AUTHORITY_PARENT,
        create=False,
    )
    _require_owner_directory(
        MANIFEST_DIRECTORY,
        expected=MANIFEST_DIRECTORY,
        parent=KEY_DIRECTORY,
        create=False,
    )
    if (
        private_key_path != KEY_DIRECTORY / PRIVATE_KEY_NAME
        or public_key_path != KEY_DIRECTORY / PUBLIC_KEY_NAME
    ):
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_key_path_invalid"
        )
    private_raw = _read_exact_regular(
        private_key_path,
        size=PRIVATE_KEY_BYTES,
        modes=frozenset({0o600}),
        code="owner_gate_trust_author_private_key_invalid",
    )
    public_raw = _read_exact_regular(
        public_key_path,
        size=PUBLIC_KEY_BYTES,
        modes=frozenset({0o400, 0o440, 0o444}),
        code="owner_gate_trust_author_public_key_invalid",
    )
    key_id = hashlib.sha256(public_raw).hexdigest()
    if (
        _public_raw(private_raw) != public_raw
        or key_id != trust.PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256
    ):
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_key_not_pinned"
        )
    unsigned = _read_unsigned(unsigned_path)
    chain = _validate_foundation_chain_files(
        pre_foundation_authority_path=pre_foundation_authority_path,
        owner_reauthentication_receipt_path=owner_reauthentication_receipt_path,
        network_evidence_path=network_evidence_path,
        network_collector_public_key_path=network_collector_public_key_path,
        project_ancestry_evidence_path=project_ancestry_evidence_path,
        project_ancestry_collector_public_key_path=(
            project_ancestry_collector_public_key_path
        ),
        direct_iam_identity_authority_path=direct_iam_identity_authority_path,
        release_public_key=Ed25519PublicKey.from_public_bytes(public_raw),
        final_release_revision=unsigned["release_revision"],
    )
    if (
        type(chain) is not _ValidatedFoundationChain
        or chain._marker is not _FOUNDATION_CHAIN_MARKER
    ):
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_foundation_chain_invalid"
        )
    final_collectors = unsigned["collector_public_key_ids"]
    if (
        type(now_unix) is bool
        or (now_unix is not None and (type(now_unix) is not int or now_unix <= 0))
        or chain.final_release_revision != unsigned["release_revision"]
        or chain.foundation_source_revision
        != unsigned["foundation_source_revision"]
        or chain.foundation_source_tree_oid
        != unsigned["foundation_source_tree_oid"]
        or chain.foundation_source_revision == chain.final_release_revision
        or unsigned["pre_foundation_authority_sha256"]
        != chain.pre_foundation_authority_sha256
        or unsigned["foundation_apply_receipt_sha256"]
        != chain.foundation_apply_receipt_sha256
        or unsigned["direct_iam_identity_authority_sha256"]
        != chain.direct_iam_identity_authority_sha256
        or unsigned["project_ancestry_evidence_sha256"]
        != chain.project_ancestry_evidence_sha256
        or unsigned["project_ancestry_chain_sha256"]
        != chain.project_ancestry_chain_sha256
        or unsigned["resource_ancestor_chain"]
        != list(chain.resource_ancestor_chain)
        or chain.bootstrap_network_collector_public_key_id
        in set(final_collectors.values())
    ):
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_foundation_chain_mismatch"
        )
    expected_output = MANIFEST_DIRECTORY / (
        f"{unsigned['release_revision']}.trust.json"
    )
    if output_path != expected_output:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_manifest_path_invalid"
        )
    if unsigned.get("signer_key_id") != key_id:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_signer_mismatch"
        )
    try:
        signature = Ed25519PrivateKey.from_private_bytes(private_raw).sign(
            foundation.canonical_json_bytes(unsigned)
        )
    except ValueError as exc:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_private_key_invalid"
        ) from None
    if len(signature) != SIGNATURE_BYTES:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_signature_invalid"
        )
    signed = {
        **unsigned,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode("ascii"),
    }
    raw = foundation.canonical_json_bytes(signed)
    _publish_exclusive(
        output_path,
        raw,
        mode=0o444,
        code="owner_gate_trust_author_manifest_write_failed",
    )
    try:
        verified = trust.load_pinned_release_trust(
            manifest_path=output_path,
            public_key_path=public_key_path,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
        )
    except trust.OwnerGateTrustError as exc:
        raise OwnerGateTrustAuthorError(
            "owner_gate_trust_author_postwrite_verification_failed"
        ) from None
    return {
        "schema": "muncho-owner-gate-release-trust-authoring-receipt.v1",
        "manifest_authored": True,
        "release_revision": verified["release_revision"],
        "source_tree_oid": verified["source_tree_oid"],
        "foundation_source_revision": verified[
            "foundation_source_revision"
        ],
        "foundation_source_tree_oid": verified[
            "foundation_source_tree_oid"
        ],
        "manifest_path": str(output_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "public_key_sha256": key_id,
        "project_ancestry_evidence_sha256": verified[
            "project_ancestry_evidence_sha256"
        ],
        "project_ancestry_chain_sha256": verified[
            "project_ancestry_chain_sha256"
        ],
        "resource_ancestor_chain": list(verified["resource_ancestor_chain"]),
        "private_key_material_printed": False,
        "private_key_digest_printed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-key")
    sign = subparsers.add_parser("sign")
    sign.add_argument("--unsigned", type=Path, required=True)
    sign.add_argument("--output", type=Path, required=True)
    sign.add_argument("--pre-foundation-authority", type=Path, required=True)
    sign.add_argument("--owner-reauth-receipt", type=Path, required=True)
    sign.add_argument("--network-evidence", type=Path, required=True)
    sign.add_argument("--network-collector-public-key", type=Path, required=True)
    sign.add_argument("--project-ancestry-evidence", type=Path, required=True)
    sign.add_argument(
        "--project-ancestry-collector-public-key",
        type=Path,
        required=True,
    )
    sign.add_argument("--direct-iam-identity-authority", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "init-key":
        receipt = initialize_keypair()
    else:
        receipt = sign_manifest(
            unsigned_path=arguments.unsigned,
            private_key_path=KEY_DIRECTORY / PRIVATE_KEY_NAME,
            public_key_path=KEY_DIRECTORY / PUBLIC_KEY_NAME,
            output_path=arguments.output,
            pre_foundation_authority_path=arguments.pre_foundation_authority,
            owner_reauthentication_receipt_path=arguments.owner_reauth_receipt,
            network_evidence_path=arguments.network_evidence,
            network_collector_public_key_path=(
                arguments.network_collector_public_key
            ),
            project_ancestry_evidence_path=(
                arguments.project_ancestry_evidence
            ),
            project_ancestry_collector_public_key_path=(
                arguments.project_ancestry_collector_public_key
            ),
            direct_iam_identity_authority_path=(
                arguments.direct_iam_identity_authority
            ),
        )
    print(foundation.canonical_json_bytes(receipt).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
