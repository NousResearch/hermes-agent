#!/usr/bin/env python3
"""Stage the exact owner-gate activation evidence bundle without activating.

The root entrypoint accepts one canonical public frame on stdin and no
arguments.  It derives every destination from its installed release, validates
the eight documents through the activation-seal contract with strict
freshness, and publishes one immutable evidence directory plus one immutable
self-hashed staging receipt.  It has no Cloud client, IAM writer, service
starter, Caddy capability, or activation-seal publication path.

An interrupted first publication is intentionally not cleaned up or resumed.
Any deterministic scratch object or one-sided final publication is durable
evidence of partial state and requires manual reconciliation.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.canary import owner_gate_activation_seal as activation
from scripts.canary import passkey_v2_protocol as protocol


FRAME_SCHEMA = "muncho-owner-gate-activation-evidence-staging-frame.v1"
RECEIPT_SCHEMA = (
    "muncho-owner-gate-activation-evidence-staging-receipt.v1"
)
RESPONSE_SCHEMA = (
    "muncho-owner-gate-activation-evidence-staging-response.v1"
)

EVIDENCE_BASE = activation.ACTIVATION_EVIDENCE_BASE
STAGING_RECEIPT_BASE = Path(
    "/var/lib/muncho-owner-gate/activation-evidence-staging-receipts"
)
ACTIVATION_SEAL_PATH = activation.ACTIVATION_SEAL_PATH
LOCK_PATH = activation.ACTIVATION_LOCK_PATH
RELEASE_BASE = activation.RELEASE_BASE

ROOT_UID = 0
ROOT_GID = 0
BASE_MODE = 0o700
EVIDENCE_DIRECTORY_MODE = 0o500
EVIDENCE_FILE_MODE = 0o444
RECEIPT_FILE_MODE = 0o444
LOCK_PARENT_MODE = 0o755
LOCK_FILE_MODE = 0o600
MAX_FILE_BYTES = activation.MAX_JSON_BYTES
MAX_BUNDLE_BYTES = activation.MAX_PAYLOAD_BYTES
MAX_FRAME_BYTES = activation.MAX_PAYLOAD_BYTES

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FRAME_FIELDS = frozenset({
    "schema",
    "release_revision",
    "evidence",
    "evidence_file_sha256",
    "bundle_sha256",
})
_VALIDATION_FIELDS = frozenset({
    "schema",
    "release_revision",
    "evidence_file_sha256",
    "prospective_activation_seal_sha256",
    "freshness_enforced",
    "fresh_through_unix",
    "activation_seal_published",
    "cloud_mutation_performed",
    "validation_sha256",
})
_PROCESS_LOCK = threading.Lock()
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_MAX_SAFE_UNIX_TIME = (1 << 53) - 1
_MODULE_RELATIVE = Path(
    "scripts/canary/owner_gate_activation_evidence_stager.py"
)


class OwnerGateActivationEvidenceStagingError(RuntimeError):
    """Stable, secret-free evidence staging failure."""


def _canonical(value: Any) -> bytes:
    try:
        return protocol.canonical_json_bytes(value)
    except (
        protocol.PasskeyV2ProtocolError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_json_invalid"
        ) from None


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _decode_canonical(raw: bytes) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in items:
            if name in value:
                raise OwnerGateActivationEvidenceStagingError(
                    "owner_gate_activation_evidence_staging_duplicate_key"
                )
            value[name] = item
        return value

    def constant(_value: str) -> None:
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_nonfinite_number"
        )

    if not raw or len(raw) > MAX_FRAME_BYTES or raw.endswith(b"\n"):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_input_invalid"
        )
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except OwnerGateActivationEvidenceStagingError:
        raise
    except (RecursionError, UnicodeError, ValueError, json.JSONDecodeError):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_json_invalid"
        ) from None
    if not isinstance(value, Mapping) or _canonical(value) != raw:
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_json_not_canonical"
        )
    return dict(value)


def _validated_frame(
    value: Mapping[str, Any],
) -> tuple[str, dict[str, bytes], dict[str, str], str]:
    if set(value) != _FRAME_FIELDS:
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_fields_invalid"
        )
    revision = value.get("release_revision")
    evidence = value.get("evidence")
    digests = value.get("evidence_file_sha256")
    if (
        value.get("schema") != FRAME_SCHEMA
        or not isinstance(revision, str)
        or _REVISION.fullmatch(revision) is None
        or not isinstance(evidence, Mapping)
        or set(evidence) != set(activation.EVIDENCE_NAMES)
        or not isinstance(digests, Mapping)
        or set(digests) != set(activation.EVIDENCE_NAMES)
    ):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_identity_invalid"
        )
    payloads: dict[str, bytes] = {}
    checked_digests: dict[str, str] = {}
    total = 0
    for name in activation.EVIDENCE_NAMES:
        document = evidence.get(name)
        expected = digests.get(name)
        if not isinstance(document, Mapping):
            raise OwnerGateActivationEvidenceStagingError(
                "owner_gate_activation_evidence_staging_document_invalid"
            )
        raw = _canonical(document)
        total += len(raw)
        if (
            not raw
            or len(raw) > MAX_FILE_BYTES
            or total > MAX_BUNDLE_BYTES
            or not isinstance(expected, str)
            or _SHA256.fullmatch(expected) is None
            or _sha256(raw) != expected
        ):
            raise OwnerGateActivationEvidenceStagingError(
                "owner_gate_activation_evidence_staging_digest_invalid"
            )
        payloads[name] = raw
        checked_digests[name] = expected
    unsigned = {
        "schema": FRAME_SCHEMA,
        "release_revision": revision,
        "evidence": dict(evidence),
        "evidence_file_sha256": dict(digests),
    }
    expected_bundle = _sha256(_canonical(unsigned))
    if (
        not isinstance(value.get("bundle_sha256"), str)
        or _SHA256.fullmatch(value["bundle_sha256"]) is None
        or value["bundle_sha256"] != expected_bundle
    ):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_bundle_invalid"
        )
    return revision, payloads, checked_digests, expected_bundle


def build_staging_frame(
    *,
    release_revision: str,
    evidence: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Build the canonical-frame value; this function accepts no paths."""

    documents = {name: dict(value) for name, value in evidence.items()}
    digests = {
        name: _sha256(_canonical(value)) for name, value in documents.items()
    }
    unsigned = {
        "schema": FRAME_SCHEMA,
        "release_revision": release_revision,
        "evidence": documents,
        "evidence_file_sha256": digests,
    }
    frame = {**unsigned, "bundle_sha256": _sha256(_canonical(unsigned))}
    _validated_frame(frame)
    return frame


def _require_directory(
    path: Path,
    *,
    parent: Path,
    uid: int,
    gid: int,
    mode: int,
    code: str,
) -> os.stat_result:
    if not path.is_absolute() or ".." in path.parts or path.parent != parent:
        raise OwnerGateActivationEvidenceStagingError(code)
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
        after = resolved.stat()
    except (OSError, RuntimeError):
        raise OwnerGateActivationEvidenceStagingError(code) from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(after.st_mode)
        or resolved != path
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or after.st_uid != uid
        or after.st_gid != gid
        or stat.S_IMODE(after.st_mode) != mode
    ):
        raise OwnerGateActivationEvidenceStagingError(code)
    return after


def _fsync_directory(path: Path, *, code: str) -> None:
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
    except OSError:
        raise OwnerGateActivationEvidenceStagingError(code) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                raise OwnerGateActivationEvidenceStagingError(code) from None


def _read_exact_file(
    path: Path,
    *,
    expected: bytes,
    uid: int,
    gid: int,
    mode: int,
    allowed_links: frozenset[int] = frozenset({1}),
    code: str,
) -> os.stat_result:
    descriptor: int | None = None
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or resolved != path
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino)
            != (opened.st_dev, opened.st_ino)
            or opened.st_uid != uid
            or opened.st_gid != gid
            or stat.S_IMODE(opened.st_mode) != mode
            or opened.st_nlink not in allowed_links
            or opened.st_size != len(expected)
        ):
            raise OwnerGateActivationEvidenceStagingError(code)
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise OwnerGateActivationEvidenceStagingError(code)
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
            opened.st_nlink,
        )
        if (
            b"".join(chunks) != expected
            or identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
            )
        ):
            raise OwnerGateActivationEvidenceStagingError(code)
        return opened
    except OwnerGateActivationEvidenceStagingError:
        raise
    except (OSError, RuntimeError):
        raise OwnerGateActivationEvidenceStagingError(code) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                raise OwnerGateActivationEvidenceStagingError(code) from None


def _write_exact_file(
    path: Path,
    *,
    payload: bytes,
    uid: int,
    gid: int,
    mode: int,
    code: str,
) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchown(descriptor, uid, gid)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    except (FileExistsError, OSError):
        raise OwnerGateActivationEvidenceStagingError(code) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                raise OwnerGateActivationEvidenceStagingError(code) from None
    _read_exact_file(
        path,
        expected=payload,
        uid=uid,
        gid=gid,
        mode=mode,
        code=code,
    )


def _verify_evidence_directory(
    root: Path,
    *,
    payloads: Mapping[str, bytes],
    uid: int,
    gid: int,
) -> None:
    _require_directory(
        root,
        parent=root.parent,
        uid=uid,
        gid=gid,
        mode=EVIDENCE_DIRECTORY_MODE,
        code="owner_gate_activation_evidence_staging_directory_invalid",
    )
    try:
        names = {entry.name for entry in os.scandir(root)}
    except OSError:
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_directory_invalid"
        ) from None
    if names != set(activation.EVIDENCE_NAMES):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_directory_invalid"
        )
    for name in activation.EVIDENCE_NAMES:
        _read_exact_file(
            root / name,
            expected=payloads[name],
            uid=uid,
            gid=gid,
            mode=EVIDENCE_FILE_MODE,
            code="owner_gate_activation_evidence_staging_file_invalid",
        )


def _create_scratch_directory(
    root: Path,
    *,
    payloads: Mapping[str, bytes],
    uid: int,
    gid: int,
) -> None:
    descriptor: int | None = None
    try:
        os.mkdir(root, 0o700)
        descriptor = os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fchown(descriptor, uid, gid)
        for name in activation.EVIDENCE_NAMES:
            _write_exact_file(
                root / name,
                payload=payloads[name],
                uid=uid,
                gid=gid,
                mode=EVIDENCE_FILE_MODE,
                code="owner_gate_activation_evidence_staging_write_failed",
            )
        os.fchmod(descriptor, EVIDENCE_DIRECTORY_MODE)
        os.fsync(descriptor)
    except (FileExistsError, OSError):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_scratch_conflict"
        ) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                raise OwnerGateActivationEvidenceStagingError(
                    "owner_gate_activation_evidence_staging_write_failed"
                ) from None
    _fsync_directory(
        root.parent,
        code="owner_gate_activation_evidence_staging_fsync_failed",
    )
    _verify_evidence_directory(
        root,
        payloads=payloads,
        uid=uid,
        gid=gid,
    )


def _rename_noreplace(source: Path, destination: Path) -> bool:
    """Use Linux renameat2(RENAME_NOREPLACE); never emulate in production."""

    if not sys.platform.startswith("linux"):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_linux_required"
        )
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_noreplace_unavailable"
        ) from None
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return True
    number = ctypes.get_errno()
    if number == errno.EEXIST:
        return False
    raise OwnerGateActivationEvidenceStagingError(
        "owner_gate_activation_evidence_staging_publication_failed"
    ) from None


def _publish_receipt_noreplace(
    scratch: Path,
    final: Path,
    *,
    raw: bytes,
    uid: int,
    gid: int,
) -> None:
    _read_exact_file(
        scratch,
        expected=raw,
        uid=uid,
        gid=gid,
        mode=RECEIPT_FILE_MODE,
        code="owner_gate_activation_evidence_staging_receipt_write_failed",
    )
    _fsync_directory(
        scratch.parent,
        code="owner_gate_activation_evidence_staging_fsync_failed",
    )
    try:
        os.link(scratch, final, follow_symlinks=False)
    except FileExistsError:
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_receipt_conflict"
        ) from None
    except OSError:
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_receipt_publish_failed"
        ) from None
    _fsync_directory(
        final.parent,
        code="owner_gate_activation_evidence_staging_fsync_failed",
    )
    _read_exact_file(
        final,
        expected=raw,
        uid=uid,
        gid=gid,
        mode=RECEIPT_FILE_MODE,
        allowed_links=frozenset({2}),
        code="owner_gate_activation_evidence_staging_receipt_invalid",
    )
    try:
        os.unlink(scratch)
    except OSError:
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_receipt_publish_failed"
        ) from None
    _fsync_directory(
        final.parent,
        code="owner_gate_activation_evidence_staging_fsync_failed",
    )
    _read_exact_file(
        final,
        expected=raw,
        uid=uid,
        gid=gid,
        mode=RECEIPT_FILE_MODE,
        code="owner_gate_activation_evidence_staging_receipt_invalid",
    )


def _open_lock(path: Path, *, uid: int, gid: int) -> int:
    descriptor: int | None = None
    created = False
    try:
        flags = (
            os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                path,
                flags | os.O_CREAT | os.O_EXCL,
                LOCK_FILE_MODE,
            )
            created = True
        except FileExistsError:
            descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino)
            != (opened.st_dev, opened.st_ino)
            or opened.st_nlink != 1
            or (
                not created
                and (
                    opened.st_uid != uid
                    or opened.st_gid != gid
                    or stat.S_IMODE(opened.st_mode) != LOCK_FILE_MODE
                )
            )
        ):
            raise OSError("lock metadata drift")
        if created:
            os.fchown(descriptor, uid, gid)
            os.fchmod(descriptor, LOCK_FILE_MODE)
        opened = os.fstat(descriptor)
        if (
            opened.st_uid != uid
            or opened.st_gid != gid
            or stat.S_IMODE(opened.st_mode) != LOCK_FILE_MODE
            or opened.st_nlink != 1
        ):
            raise OSError("lock metadata drift")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = os.fstat(descriptor)
        after = path.lstat()
        if (
            stat.S_ISLNK(after.st_mode)
            or (locked.st_dev, locked.st_ino)
            != (opened.st_dev, opened.st_ino)
            or (after.st_dev, after.st_ino)
            != (locked.st_dev, locked.st_ino)
            or locked.st_nlink != 1
        ):
            raise OSError("lock metadata drift")
        return descriptor
    except OSError:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_lock_invalid"
        ) from None


def _release_lock(descriptor: int) -> None:
    failed = False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError:
        failed = True
    try:
        os.close(descriptor)
    except OSError:
        failed = True
    if failed:
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_lock_release_failed"
        ) from None


def _require_linux_root() -> None:
    flags = sys.flags
    if (
        not sys.platform.startswith("linux")
        or os.getuid() != ROOT_UID  # windows-footgun: ok — POSIX root boundary
        or os.geteuid() != ROOT_UID  # windows-footgun: ok — POSIX root boundary
        or os.getgid() != ROOT_GID  # windows-footgun: ok — POSIX root boundary
        or os.getegid() != ROOT_GID  # windows-footgun: ok — POSIX root boundary
        or flags.isolated != 1
        or flags.ignore_environment != 1
        or flags.no_user_site != 1
        or flags.safe_path is not True
        or flags.dont_write_bytecode != 1
    ):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_linux_root_required"
        )


def _current_unix_time() -> int:
    try:
        raw = time.time()
    except (OSError, OverflowError, ValueError):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_time_invalid"
        ) from None
    if (
        type(raw) not in {int, float}
        or not math.isfinite(raw)
        or raw <= 0
        or raw > _MAX_SAFE_UNIX_TIME
    ):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_time_invalid"
        )
    return int(raw)


def _validate_fresh_evidence(
    *,
    release: Path,
    evidence_root: Path,
) -> Mapping[str, Any]:
    """Derive once at start and prove completion stayed in its fresh window."""

    started_at_unix = _current_unix_time()
    raw_validation = activation.validate_activation_evidence_strict(
        release=release,
        evidence_root=evidence_root,
        now_unix=started_at_unix,
    )
    if (
        not isinstance(raw_validation, Mapping)
        or set(raw_validation) != _VALIDATION_FIELDS
    ):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_validation_invalid"
        )
    validation = dict(raw_validation)
    digests = validation.get("evidence_file_sha256")
    fresh_through_unix = validation.get("fresh_through_unix")
    unsigned = {
        name: item
        for name, item in validation.items()
        if name != "validation_sha256"
    }
    if (
        validation.get("schema")
        != activation.ACTIVATION_EVIDENCE_VALIDATION_SCHEMA
        or validation.get("release_revision") != release.name
        or not isinstance(digests, Mapping)
        or set(digests) != set(activation.EVIDENCE_NAMES)
        or any(
            not isinstance(digests.get(name), str)
            or _SHA256.fullmatch(digests[name]) is None
            for name in activation.EVIDENCE_NAMES
        )
        or not isinstance(
            validation.get("prospective_activation_seal_sha256"), str
        )
        or _SHA256.fullmatch(
            validation["prospective_activation_seal_sha256"]
        )
        is None
        or validation.get("freshness_enforced") is not True
        or type(fresh_through_unix) is not int
        or fresh_through_unix <= 0
        or fresh_through_unix > _MAX_SAFE_UNIX_TIME
        or validation.get("activation_seal_published") is not False
        or validation.get("cloud_mutation_performed") is not False
        or not isinstance(validation.get("validation_sha256"), str)
        or _SHA256.fullmatch(validation["validation_sha256"]) is None
        or validation["validation_sha256"]
        != protocol.sha256_json(unsigned)
    ):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_validation_invalid"
        )
    completed_at_unix = _current_unix_time()
    if completed_at_unix < started_at_unix:
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_time_invalid"
        )
    if (
        started_at_unix > fresh_through_unix
        or completed_at_unix > fresh_through_unix
    ):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_freshness_expired"
        )
    return validation


def _installed_release() -> Path:
    try:
        module = Path(__file__)
        module_state = module.lstat()
        resolved_module = module.resolve(strict=True)
        release = resolved_module.parents[2]
        interpreter = release / "venv/bin/python"
        interpreter_state = interpreter.lstat()
        resolved_interpreter = interpreter.resolve(strict=True)
        executable = Path(sys.executable)
        resolved_executable = executable.resolve(strict=True)
    except (OSError, RuntimeError, IndexError):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_release_invalid"
        ) from None
    if (
        release.parent != RELEASE_BASE
        or _REVISION.fullmatch(release.name) is None
        or not module.is_absolute()
        or module != resolved_module
        or resolved_module != release / _MODULE_RELATIVE
        or stat.S_ISLNK(module_state.st_mode)
        or not stat.S_ISREG(module_state.st_mode)
        or module_state.st_uid != ROOT_UID
        or module_state.st_gid != ROOT_GID
        or stat.S_IMODE(module_state.st_mode) != EVIDENCE_FILE_MODE
        or module_state.st_nlink != 1
        or not executable.is_absolute()
        or resolved_executable != resolved_interpreter
        or not stat.S_ISREG(interpreter_state.st_mode)
        or interpreter_state.st_uid != ROOT_UID
        or interpreter_state.st_gid != ROOT_GID
        or stat.S_IMODE(interpreter_state.st_mode) != 0o755
        or interpreter_state.st_nlink != 1
    ):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_release_invalid"
        )
    return release


def _require_activation_absent() -> None:
    if os.path.lexists(ACTIVATION_SEAL_PATH):
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_activation_present"
        )


def _staging_receipt(
    *,
    revision: str,
    bundle_sha256: str,
    digests: Mapping[str, str],
    validation: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = {
        "schema": RECEIPT_SCHEMA,
        "release_revision": revision,
        "bundle_sha256": bundle_sha256,
        "evidence_file_sha256": dict(digests),
        "activation_evidence_validation_sha256": validation[
            "validation_sha256"
        ],
        "activation_evidence_fresh_through_unix": validation[
            "fresh_through_unix"
        ],
        "prospective_activation_seal_sha256": validation[
            "prospective_activation_seal_sha256"
        ],
        "staging_state": "complete",
        "evidence_directory_mode": "0500",
        "evidence_file_mode": "0444",
        "activation_seal_present": False,
        "activation_performed": False,
        "runtime_started": False,
        "cloud_mutation_performed": False,
        "storage_mutation_performed": False,
        "iam_mutation_performed": False,
        "caddy_mutation_performed": False,
    }
    return {**unsigned, "receipt_sha256": protocol.sha256_json(unsigned)}


def _response(
    *,
    receipt: Mapping[str, Any],
    disposition: str,
) -> Mapping[str, Any]:
    unsigned = {
        "schema": RESPONSE_SCHEMA,
        "release_revision": receipt["release_revision"],
        "bundle_sha256": receipt["bundle_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "activation_evidence_fresh_through_unix": receipt[
            "activation_evidence_fresh_through_unix"
        ],
        "disposition": disposition,
        "staging_state": "complete",
        "activation_seal_present": False,
        "activation_performed": False,
        "runtime_started": False,
        "cloud_mutation_performed": False,
        "storage_mutation_performed": False,
        "iam_mutation_performed": False,
        "caddy_mutation_performed": False,
    }
    return {**unsigned, "response_sha256": protocol.sha256_json(unsigned)}


def _stage_activation_evidence(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require_linux_root()
    revision, payloads, digests, bundle_sha256 = _validated_frame(value)
    release = _installed_release()
    if release.name != revision:
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_revision_invalid"
        )
    uid = ROOT_UID
    gid = ROOT_GID
    _require_directory(
        EVIDENCE_BASE,
        parent=EVIDENCE_BASE.parent,
        uid=uid,
        gid=gid,
        mode=BASE_MODE,
        code="owner_gate_activation_evidence_staging_base_invalid",
    )
    _require_directory(
        STAGING_RECEIPT_BASE,
        parent=STAGING_RECEIPT_BASE.parent,
        uid=uid,
        gid=gid,
        mode=BASE_MODE,
        code="owner_gate_activation_evidence_staging_receipt_base_invalid",
    )
    _require_directory(
        LOCK_PATH.parent,
        parent=LOCK_PATH.parent.parent,
        uid=uid,
        gid=gid,
        mode=LOCK_PARENT_MODE,
        code="owner_gate_activation_evidence_staging_lock_parent_invalid",
    )
    final_root = EVIDENCE_BASE / revision
    scratch_root = EVIDENCE_BASE / f".{revision}.staged"
    receipt_path = STAGING_RECEIPT_BASE / f"{revision}.json"
    receipt_scratch = STAGING_RECEIPT_BASE / f".{revision}.json.staged"

    with _PROCESS_LOCK:
        lock_descriptor = _open_lock(LOCK_PATH, uid=uid, gid=gid)
        try:
            _require_activation_absent()
            final_exists = os.path.lexists(final_root)
            receipt_exists = os.path.lexists(receipt_path)
            if os.path.lexists(scratch_root) or os.path.lexists(
                receipt_scratch
            ):
                raise OwnerGateActivationEvidenceStagingError(
                    "owner_gate_activation_evidence_staging_partial_state"
                )
            if final_exists != receipt_exists:
                raise OwnerGateActivationEvidenceStagingError(
                    "owner_gate_activation_evidence_staging_partial_state"
                )

            if final_exists:
                _verify_evidence_directory(
                    final_root,
                    payloads=payloads,
                    uid=uid,
                    gid=gid,
                )
                validation = _validate_fresh_evidence(
                    release=release,
                    evidence_root=final_root,
                )
                if validation["evidence_file_sha256"] != digests:
                    raise OwnerGateActivationEvidenceStagingError(
                        "owner_gate_activation_evidence_staging_validation_drift"
                    )
                receipt = _staging_receipt(
                    revision=revision,
                    bundle_sha256=bundle_sha256,
                    digests=digests,
                    validation=validation,
                )
                receipt_raw = _canonical(receipt)
                _read_exact_file(
                    receipt_path,
                    expected=receipt_raw,
                    uid=uid,
                    gid=gid,
                    mode=RECEIPT_FILE_MODE,
                    code="owner_gate_activation_evidence_staging_receipt_invalid",
                )
                _require_activation_absent()
                _verify_evidence_directory(
                    final_root,
                    payloads=payloads,
                    uid=uid,
                    gid=gid,
                )
                _read_exact_file(
                    receipt_path,
                    expected=receipt_raw,
                    uid=uid,
                    gid=gid,
                    mode=RECEIPT_FILE_MODE,
                    code=(
                        "owner_gate_activation_evidence_staging_"
                        "receipt_invalid"
                    ),
                )
                replay_validation = _validate_fresh_evidence(
                    release=release,
                    evidence_root=final_root,
                )
                if replay_validation != validation:
                    raise OwnerGateActivationEvidenceStagingError(
                        "owner_gate_activation_evidence_staging_validation_drift"
                    )
                _require_activation_absent()
                return _response(
                    receipt=receipt,
                    disposition="exact_replay",
                )

            _create_scratch_directory(
                scratch_root,
                payloads=payloads,
                uid=uid,
                gid=gid,
            )
            validation = _validate_fresh_evidence(
                release=release,
                evidence_root=scratch_root,
            )
            if validation["evidence_file_sha256"] != digests:
                raise OwnerGateActivationEvidenceStagingError(
                    "owner_gate_activation_evidence_staging_validation_drift"
                )
            receipt = _staging_receipt(
                revision=revision,
                bundle_sha256=bundle_sha256,
                digests=digests,
                validation=validation,
            )
            receipt_raw = _canonical(receipt)
            _write_exact_file(
                receipt_scratch,
                payload=receipt_raw,
                uid=uid,
                gid=gid,
                mode=RECEIPT_FILE_MODE,
                code="owner_gate_activation_evidence_staging_receipt_write_failed",
            )
            _fsync_directory(
                STAGING_RECEIPT_BASE,
                code="owner_gate_activation_evidence_staging_fsync_failed",
            )
            _require_activation_absent()
            _verify_evidence_directory(
                scratch_root,
                payloads=payloads,
                uid=uid,
                gid=gid,
            )
            publication_validation = _validate_fresh_evidence(
                release=release,
                evidence_root=scratch_root,
            )
            if publication_validation != validation:
                raise OwnerGateActivationEvidenceStagingError(
                    "owner_gate_activation_evidence_staging_validation_drift"
                )
            _require_activation_absent()
            if not _rename_noreplace(scratch_root, final_root):
                raise OwnerGateActivationEvidenceStagingError(
                    "owner_gate_activation_evidence_staging_publication_conflict"
                )
            _fsync_directory(
                EVIDENCE_BASE,
                code="owner_gate_activation_evidence_staging_fsync_failed",
            )
            _verify_evidence_directory(
                final_root,
                payloads=payloads,
                uid=uid,
                gid=gid,
            )
            receipt_validation = _validate_fresh_evidence(
                release=release,
                evidence_root=final_root,
            )
            if receipt_validation != validation:
                raise OwnerGateActivationEvidenceStagingError(
                    "owner_gate_activation_evidence_staging_validation_drift"
                )
            _require_activation_absent()
            _publish_receipt_noreplace(
                receipt_scratch,
                receipt_path,
                raw=receipt_raw,
                uid=uid,
                gid=gid,
            )
            _require_activation_absent()
            _verify_evidence_directory(
                final_root,
                payloads=payloads,
                uid=uid,
                gid=gid,
            )
            _read_exact_file(
                receipt_path,
                expected=receipt_raw,
                uid=uid,
                gid=gid,
                mode=RECEIPT_FILE_MODE,
                code="owner_gate_activation_evidence_staging_receipt_invalid",
            )
            final_validation = _validate_fresh_evidence(
                release=release,
                evidence_root=final_root,
            )
            if final_validation != validation:
                raise OwnerGateActivationEvidenceStagingError(
                    "owner_gate_activation_evidence_staging_validation_drift"
                )
            _require_activation_absent()
            return _response(receipt=receipt, disposition="installed")
        finally:
            _release_lock(lock_descriptor)


def stage_activation_evidence(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Stage one exact fixed-path bundle behind a stable public boundary."""

    try:
        frozen = _decode_canonical(_canonical(value))
        return _stage_activation_evidence(frozen)
    except (
        OwnerGateActivationEvidenceStagingError,
        activation.OwnerGateActivationSealError,
    ):
        raise
    except Exception:
        raise OwnerGateActivationEvidenceStagingError(
            "owner_gate_activation_evidence_staging_internal_failure"
        ) from None


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(argv or ())
    try:
        if arguments:
            raise OwnerGateActivationEvidenceStagingError(
                "owner_gate_activation_evidence_staging_command_invalid"
            )
        raw = sys.stdin.buffer.read(MAX_FRAME_BYTES + 1)
        response = stage_activation_evidence(_decode_canonical(raw))
    except (
        OSError,
        RecursionError,
        OwnerGateActivationEvidenceStagingError,
        activation.OwnerGateActivationSealError,
    ):
        print(
            '{"error_code":"owner_gate_activation_evidence_staging_failed",'
            '"ok":false}',
            file=sys.stderr,
        )
        return 2
    print(_canonical(response).decode("utf-8"))
    return 0


__all__ = [
    "ACTIVATION_SEAL_PATH",
    "EVIDENCE_BASE",
    "FRAME_SCHEMA",
    "LOCK_PATH",
    "OwnerGateActivationEvidenceStagingError",
    "RECEIPT_SCHEMA",
    "RESPONSE_SCHEMA",
    "STAGING_RECEIPT_BASE",
    "build_staging_frame",
    "main",
    "stage_activation_evidence",
]


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
