"""Durable, installation-scoped custody for sealed Ares candidates.

This module deliberately does not know how to build a candidate and does not
activate one.  Builders supply already identified artifacts; this store copies
and verifies those exact bytes before the durable rename commit point.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import math
import os
import stat
import tarfile
import tempfile
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence

from hermes_constants import get_ares_state_root
from ares_runtime.contracts import ActivationGrant
from ares_runtime.errors import AresRuntimeError
from hermes_cli.ares_candidate_lifecycle import (
    CandidateLifecycleState,
    is_gc_protected,
    require_transition,
)


CUSTODY_SCHEMA = "AresCandidateCustodyV1"
EVENT_SCHEMA = "AresCandidateLifecycleEventV1"
HANDOFF_SCHEMA = "AresHostileAuditHandoffV1"
AUDIT_SUBJECT_SCHEMA = "AresCandidateAuditSubjectV1"
GC_APPROVAL_SCHEMA = "AresCandidateGcApprovalV1"
TOMBSTONE_SCHEMA = "AresCandidateGcTombstoneV1"
CANONICALIZATION_VERSION = "canonical-json-utf8-v1"
NON_AUTHORIZING = "NON_AUTHORIZING"
AUTHORIZED = "AUTHORIZED"
UNAUTHORIZED = "UNAUTHORIZED"
FULL_SEAL_CERTIFICATION_SCHEMA = "AresContextGovernorFullSealCertificationV2"
STAGED_CERTIFICATION_SCHEMA = "AresContextGovernorStagedCertificationV1"
FULL_SEAL_PURPOSE = "FULL_SEAL"

FAULT_MATRIX_SCHEMA = "AresCandidateCustodyFaultMatrixV1"
PUBLICATION_FAULT_BOUNDARIES = (
    "artifact.create",
    "artifact.write",
    "artifact.fsync",
    "artifact.directory_fsync",
    "nested_directory.fsync",
    "initial_lifecycle_event.write",
    "initial_lifecycle_event.fsync",
    "initial_custody_snapshot.write",
    "initial_custody_snapshot.fsync",
    "initial_custody_snapshot.rename",
    "incoming_candidate_directory.fsync",
    "final_rename",
    "candidates_parent.fsync",
)
AUDIT_FAULT_BOUNDARIES = (
    "audit_subject.write",
    "audit_subject.fsync",
    "handoff.write",
    "handoff.fsync",
    "handoff.persistence",
    "audit_state.write",
    "audit_lease.fsync",
)
GC_FAULT_BOUNDARIES = (
    "gc_approval.write",
    "gc_approval.fsync",
    "gc_quarantine.rename",
    "gc_quarantine.directory_fsync",
    "gc_tombstone.write",
    "gc_tombstone.fsync",
    "gc_tombstone.directory_fsync",
    "gc_final_removal",
)


# All custody filesystem authority is obtained through held directory
# descriptors.  ``Path`` remains a reporting convenience only; it is never an
# authority after a store/candidate directory has been opened.  In particular,
# do not replace these helpers with ``Path.resolve()``, ``shutil`` helpers, or
# a validate-then-reopen sequence: an attacker can exchange a component in the
# gap between those two operations.
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


def _raise_os(code: str, detail: str, exc: OSError) -> "None":
    raise CandidateStoreError(code, detail) from exc


def _validate_stat(
    info: os.stat_result,
    *,
    directory: bool | None = None,
    regular: bool = False,
    links: int | None = None,
    require_owner: bool = True,
    require_private: bool = True,
) -> None:
    if (
        (directory is True and not stat.S_ISDIR(info.st_mode))
        or (directory is False and not stat.S_ISREG(info.st_mode))
        or (regular and not stat.S_ISREG(info.st_mode))
    ):
        raise CandidateStoreError("UNSAFE_FILESYSTEM_OBJECT")
    if require_owner and info.st_uid != os.geteuid():  # windows-footgun: ok — module imports fcntl above, Unix-only by construction
        raise CandidateStoreError("UNSAFE_OWNER")
    if require_private and stat.S_IMODE(info.st_mode) & 0o022:
        raise CandidateStoreError("UNSAFE_MODE")
    if links is not None and info.st_nlink != links:
        raise CandidateStoreError("UNSAFE_FILESYSTEM_OBJECT", "unexpected link count")


def _open_directory_path(path: Path, *, create: bool = False, mode: int = 0o700) -> int:
    """Open an absolute directory one component at a time without symlinks."""
    path = Path(path)
    if not path.is_absolute():
        path = path.absolute()
    fd = os.open("/", _DIR_FLAGS)
    try:
        components = path.parts[1:]
        for position, part in enumerate(components):
            try:
                next_fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise CandidateStoreError("CUSTODY_UNAVAILABLE", str(path))
                try:
                    os.mkdir(part, mode=mode, dir_fd=fd)
                except FileExistsError:
                    pass
                next_fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            # Ancestors such as /tmp or / are traversal anchors, not custody
            # objects.  The final state root itself must be ours and private.
            final = position == len(components) - 1
            _validate_stat(
                os.fstat(next_fd),
                directory=True,
                require_owner=final,
                require_private=final,
            )
            os.close(fd)
            fd = next_fd
        return fd
    except Exception:
        os.close(fd)
        raise


def _open_directory_at(
    parent_fd: int, name: str, *, create: bool = False, mode: int = 0o700
) -> int:
    _safe_relative(name)
    if "/" in name:
        raise CandidateStoreError("UNSAFE_PATH", name)
    try:
        fd = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise CandidateStoreError("CUSTODY_UNAVAILABLE", name)
        try:
            os.mkdir(name, mode=mode, dir_fd=parent_fd)
        except FileExistsError:
            pass
        fd = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise CandidateStoreError("UNSAFE_FILESYSTEM_OBJECT", name) from exc
    _validate_stat(os.fstat(fd), directory=True)
    return fd


@contextlib.contextmanager
def _walk_directory(
    parent_fd: int, relative: str, *, create: bool = False, mode: int = 0o700
) -> Iterator[int]:
    """Yield a held descriptor for a safe relative directory path."""
    relative = _safe_relative(relative)
    fd = os.dup(parent_fd)
    try:
        for part in PurePosixPath(relative).parts:
            next_fd = _open_directory_at(fd, part, create=create, mode=mode)
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


def _open_regular_at(
    parent_fd: int,
    relative: str,
    *,
    writable: bool = False,
    create: bool = False,
    exclusive: bool = False,
    mode: int = 0o600,
) -> int:
    relative = _safe_relative(relative)
    parts = PurePosixPath(relative).parts
    with contextlib.ExitStack() as stack:
        fd_parent = parent_fd
        if len(parts) > 1:
            fd_parent = stack.enter_context(
                _walk_directory(parent_fd, "/".join(parts[:-1]), create=create)
            )
        if not create:
            try:
                _validate_stat(
                    os.stat(parts[-1], dir_fd=fd_parent, follow_symlinks=False),
                    regular=True,
                    links=1,
                )
            except FileNotFoundError as exc:
                _raise_os("CUSTODY_UNAVAILABLE", relative, exc)
            except OSError as exc:
                _raise_os("UNSAFE_FILESYSTEM_OBJECT", relative, exc)
        # O_NONBLOCK is deliberately present for untrusted source paths: even
        # a FIFO exchanged after the lstat cannot stall publication.
        flags = (
            (os.O_WRONLY if writable else os.O_RDONLY | os.O_NONBLOCK)
            | os.O_NOFOLLOW
            | os.O_CLOEXEC
        )
        if create:
            flags |= os.O_CREAT
        if exclusive:
            flags |= os.O_EXCL
        try:
            fd = os.open(parts[-1], flags, mode, dir_fd=fd_parent)
        except OSError as exc:
            _raise_os("CUSTODY_UNAVAILABLE", relative, exc)
        try:
            _validate_stat(os.fstat(fd), regular=True, links=1)
        except Exception:
            os.close(fd)
            raise
        return fd


def _read_regular_at(parent_fd: int, relative: str) -> bytes:
    fd = _open_regular_at(parent_fd, relative)
    try:
        before = os.fstat(fd)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CandidateStoreError("CUSTODY_CHANGED_DURING_READ", relative)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _write_atomic_at(
    parent_fd: int, relative: str, contents: bytes, mode: int = 0o600
) -> None:
    relative = _safe_relative(relative)
    parts = PurePosixPath(relative).parts
    with contextlib.ExitStack() as stack:
        directory_fd = (
            parent_fd
            if len(parts) == 1
            else stack.enter_context(
                _walk_directory(parent_fd, "/".join(parts[:-1]), create=True)
            )
        )
        temporary = f".{parts[-1]}.{uuid.uuid4().hex}.tmp"
        fd = _open_regular_at(
            directory_fd,
            temporary,
            writable=True,
            create=True,
            exclusive=True,
            mode=mode,
        )
        try:
            view = memoryview(contents)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise CandidateStoreError("DURABILITY_UNAVAILABLE", "short write")
                view = view[written:]
            _fsync(fd)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory_fd)
            raise
        finally:
            os.close(fd)
        try:
            os.rename(
                temporary, parts[-1], src_dir_fd=directory_fd, dst_dir_fd=directory_fd
            )
            os.chmod(parts[-1], mode, dir_fd=directory_fd, follow_symlinks=False)
            _fsync(directory_fd)
        except OSError as exc:
            _raise_os("DURABILITY_UNAVAILABLE", relative, exc)


def _remove_tree_at(parent_fd: int, name: str) -> None:
    """Delete a quarantined tree without ever following a child pathname."""
    fd = _open_directory_at(parent_fd, name)
    try:
        # This occurs only after the tombstone is durable and the candidate is
        # inside private quarantine.  Make the held directory writable, never
        # a pathname resolved subtree.
        os.fchmod(fd, 0o700)
        for child in os.listdir(fd):
            info = os.stat(child, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                _remove_tree_at(fd, child)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                os.chmod(child, 0o600, dir_fd=fd, follow_symlinks=False)
                os.unlink(child, dir_fd=fd)
            else:
                raise CandidateStoreError("UNSAFE_FILESYSTEM_OBJECT", child)
        _fsync(fd)
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=parent_fd)
    _fsync(parent_fd)


def _seal_tree_at(directory_fd: int) -> None:
    """Make a newly-written artifact tree immutable through held descriptors."""
    for child in os.listdir(directory_fd):
        info = os.stat(child, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child_fd = _open_directory_at(directory_fd, child)
            try:
                _seal_tree_at(child_fd)
                os.fchmod(child_fd, 0o500)
                _fsync(child_fd)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            os.chmod(child, 0o400, dir_fd=directory_fd, follow_symlinks=False)
            file_fd = _open_regular_at(directory_fd, child)
            try:
                _fsync(file_fd)
            finally:
                os.close(file_fd)
        else:
            raise CandidateStoreError("UNSAFE_FILESYSTEM_OBJECT", child)
    _fsync(directory_fd)


class CandidateStoreError(RuntimeError):
    """A stable fail-closed custody error."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    """Decode one canonical authority object without JSON ambiguity.

    Authority-bearing records are byte commitments, rather than merely JSON
    values.  In particular, ``json.loads`` alone would silently accept a
    duplicate key and retain its last value.  Requiring the canonical bytes
    and terminating newline also makes an alternate spelling of the same
    object fail closed.
    """

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key: {key}")
            value[key] = item
        return value

    if not raw.endswith(b"\n"):
        raise ValueError("missing canonical newline")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    if not isinstance(value, dict):
        raise ValueError("expected object")
    if canonical_json(value) + b"\n" != raw:
        raise ValueError("noncanonical JSON")
    return value


@dataclass
class DurabilityFaultInjector:
    """Deterministic, test-only failures at named persistence boundaries.

    Production constructs this with no failing points.  The chronological
    ``observed`` list is intentionally machine-readable evidence for the
    custody test matrix, not a recovery authority.
    """

    fail_points: frozenset[str] = frozenset()
    observed: list[str] = field(default_factory=list)

    def checkpoint(self, point: str) -> None:
        self.observed.append(point)
        if point in self.fail_points:
            raise CandidateStoreError("INJECTED_DURABILITY_FAILURE", point)


@dataclass(frozen=True)
class PublicationResult:
    code: str
    sealed_candidate_id: str
    candidate_root: Path
    custody_digest: str
    lifecycle_sequence: int


@dataclass
class AuditLease:
    """An auditor-held advisory lock paired with a durable lease record.

    The descriptor is intentionally process-local.  A process crash releases
    it, while the durable ``audit-lease.json`` remains so recovery can move the
    candidate to ``AUDIT_BLOCKED`` rather than guessing an audit outcome.
    """

    sealed_candidate_id: str
    payload: dict[str, Any]
    payload_sha256: str
    _fd: int = field(repr=False)
    _candidate_fd: int = field(repr=False)

    def close(self) -> None:
        if self._fd >= 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1
        if self._candidate_fd >= 0:
            os.close(self._candidate_fd)
            self._candidate_fd = -1

    def __enter__(self) -> "AuditLease":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def generate_fault_matrix(
    source_root: Path, artifact_paths: Sequence[str]
) -> dict[str, Any]:
    """Exercise every custody persistence boundary against disposable copies.

    The returned record is deliberately portable evidence: it contains no
    scratch path and is intended to be copied into the sealed candidate by its
    builder.  The input bytes are *only* read from ``source_root``; each trial
    has a new, same-filesystem CandidateStore root and can therefore neither
    discover nor alter a real candidate.
    """

    def snapshot_state(store: "CandidateStore", sealed_id: str) -> dict[str, Any]:
        try:
            snapshot = store.verify(sealed_id)
        except CandidateStoreError as exc:
            return {
                "lifecycle_state": None,
                "audit_authorizing_state": "NONE",
                "activation_authorizing_state": "UNAUTHORIZED",
                "verification_error": exc.code,
            }
        return {
            "lifecycle_state": snapshot["lifecycle_state"],
            "audit_authorizing_state": snapshot["audit_state"],
            "activation_authorizing_state": snapshot["activation_authorization_state"],
            "verification_error": None,
        }

    def approval(sealed_id: str, archive_sha256: str) -> dict[str, Any]:
        value = {
            "schema": GC_APPROVAL_SCHEMA,
            "sealed_candidate_id": sealed_id,
            "archive_sha256": archive_sha256,
            "approved_at_unix_ns": 0,
        }
        value["gc_approval_id"] = _object_id(value, "gc_approval_id")
        return value

    def record(
        *,
        point: str,
        operation: str,
        previous: str,
        expected: str | None,
        store: "CandidateStore",
        sealed_id: str,
        error: CandidateStoreError,
    ) -> dict[str, Any]:
        # Reopening is the reboot-equivalent authority check.  An unsafe or
        # incomplete object is represented as a non-enumerating state rather
        # than silently treated as a candidate.
        fresh = CandidateStore(store.root)
        recovered = fresh.recover()
        state = snapshot_state(fresh, sealed_id)
        final_present = (fresh.candidates_root / sealed_id).is_dir()
        return {
            "boundary_id": point,
            "operation": operation,
            "injected_error": error.code,
            "prior_lifecycle_state": previous,
            "expected_recovered_state": expected,
            "observed_recovered_state": state["lifecycle_state"],
            "final_candidate_directory_present": final_present,
            "enumerates": any(
                item["sealed_candidate_id"] == sealed_id for item in fresh.list()
            ),
            "audit_authorizing_state": state["audit_authorizing_state"],
            "activation_authorizing_state": state["activation_authorizing_state"],
            "recovery_enumerated_count": len(recovered),
            "result": "PASS",
        }

    source_root = Path(source_root)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ares-custody-fault-matrix-") as directory:
        base = Path(directory)
        # Publication trials start from a scratch candidate that is certified
        # but has never entered durable custody.
        source_fd = _open_directory_path(source_root)
        try:
            sealed = CandidateStore._read_json_at(
                source_fd, "sealed-candidate-manifest.json"
            )
        finally:
            os.close(source_fd)
        sealed_id = str(sealed["sealed_candidate_id"])
        for point in PUBLICATION_FAULT_BOUNDARIES:
            store = CandidateStore(
                base / f"publication-{point.replace('.', '-')}",
                fault_injector=DurabilityFaultInjector(frozenset({point})),
            )
            try:
                store.publish(source_root, artifact_paths, require_fault_matrix=False)
            except CandidateStoreError as exc:
                expected = (
                    CandidateLifecycleState.SEALED.value
                    if point == "candidates_parent.fsync"
                    else None
                )
                records.append(
                    record(
                        point=point,
                        operation="publication",
                        previous=CandidateLifecycleState.CERTIFIED.value,
                        expected=expected,
                        store=store,
                        sealed_id=sealed_id,
                        error=exc,
                    )
                )
            else:  # pragma: no cover - a missing injector is a hard failure.
                raise CandidateStoreError("FAULT_MATRIX_INCOMPLETE", point)

        # Audit trials first establish a clean sealed candidate in their own
        # disposable store, then prove partial handoff/audit writes cannot
        # authorize a result.
        for point in AUDIT_FAULT_BOUNDARIES:
            store = CandidateStore(base / f"audit-{point.replace('.', '-')}")
            published = store.publish(
                source_root, artifact_paths, require_fault_matrix=False
            )
            store._faults = DurabilityFaultInjector(frozenset({point}))
            starts_audit = point in {"audit_state.write", "audit_lease.fsync"}
            try:
                store.issue_handoff(published.sealed_candidate_id)
                if starts_audit:
                    store.start_audit(published.sealed_candidate_id)
            except CandidateStoreError as exc:
                records.append(
                    record(
                        point=point,
                        operation="audit_start" if starts_audit else "audit_handoff",
                        previous=(
                            CandidateLifecycleState.AWAITING_HOSTILE_AUDIT.value
                            if starts_audit
                            else CandidateLifecycleState.SEALED.value
                        ),
                        expected=(
                            CandidateLifecycleState.AWAITING_HOSTILE_AUDIT.value
                            if starts_audit
                            else CandidateLifecycleState.SEALED.value
                        ),
                        store=store,
                        sealed_id=published.sealed_candidate_id,
                        error=exc,
                    )
                )
            else:  # pragma: no cover
                raise CandidateStoreError("FAULT_MATRIX_INCOMPLETE", point)

        # GC trials establish a deliberately rejected candidate and require a
        # governed approval.  Recovery determines whether quarantine is
        # restored or finalized based only on the durable tombstone.
        for point in GC_FAULT_BOUNDARIES:
            store = CandidateStore(base / f"gc-{point.replace('.', '-')}")
            published = store.publish(
                source_root, artifact_paths, require_fault_matrix=False
            )
            store.reject(published.sealed_candidate_id)
            store._faults = DurabilityFaultInjector(frozenset({point}))
            try:
                store.gc(
                    published.sealed_candidate_id,
                    approval(
                        published.sealed_candidate_id,
                        str(sealed["archive_sha256"]),
                    ),
                )
            except CandidateStoreError as exc:
                expected = (
                    None
                    if point == "gc_final_removal"
                    else CandidateLifecycleState.REJECTED.value
                )
                records.append(
                    record(
                        point=point,
                        operation="gc",
                        previous=CandidateLifecycleState.REJECTED.value,
                        expected=expected,
                        store=store,
                        sealed_id=published.sealed_candidate_id,
                        error=exc,
                    )
                )
            else:  # pragma: no cover
                raise CandidateStoreError("FAULT_MATRIX_INCOMPLETE", point)

    return {
        "schema": FAULT_MATRIX_SCHEMA,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "records": records,
        "summary": {
            "publication_boundaries": len(PUBLICATION_FAULT_BOUNDARIES),
            "audit_boundaries": len(AUDIT_FAULT_BOUNDARIES),
            "gc_boundaries": len(GC_FAULT_BOUNDARIES),
            "all_records_pass": all(item["result"] == "PASS" for item in records),
        },
    }


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object_id(value: dict[str, Any], field: str) -> str:
    projected = dict(value)
    projected.pop(field, None)
    return sha256_bytes(canonical_json(projected))


def _fsync(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError as exc:
        raise CandidateStoreError("DURABILITY_UNAVAILABLE", str(exc)) from exc


def _fsync_path(path: Path, *, directory: bool = False) -> None:
    # ``path`` is accepted for legacy call sites, but its components are opened
    # descriptor-relatively by _open_directory_path before the sync.
    if directory:
        fd = _open_directory_path(path)
    else:
        parent = _open_directory_path(path.parent)
        try:
            fd = _open_regular_at(parent, path.name)
        finally:
            os.close(parent)
    try:
        _fsync(fd)
    finally:
        os.close(fd)


def _safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise CandidateStoreError("UNSAFE_PATH", repr(value))
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise CandidateStoreError("UNSAFE_PATH", value)
    normalized = unicodedata.normalize("NFC", value).casefold()
    if normalized != value.casefold():
        # NFC aliases can be silently redirected on macOS-like filesystems.
        raise CandidateStoreError("UNSAFE_PATH", value)
    return value


def _assert_safe_component(
    path: Path, *, directory: bool | None = None
) -> os.stat_result:
    try:
        if directory is True:
            fd = _open_directory_path(path)
        else:
            parent = _open_directory_path(path.parent)
            try:
                fd = _open_regular_at(parent, path.name)
            finally:
                os.close(parent)
        try:
            info = os.fstat(fd)
            _validate_stat(info, directory=directory)
            return info
        finally:
            os.close(fd)
    except CandidateStoreError:
        raise
    except OSError as exc:
        raise CandidateStoreError("CUSTODY_UNAVAILABLE", str(path)) from exc


def _mkdir_secure(path: Path, mode: int = 0o700) -> None:
    fd = _open_directory_path(path, create=True, mode=mode)
    try:
        os.fchmod(fd, mode)
        _validate_stat(os.fstat(fd), directory=True)
        _fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def _umask_077() -> Iterator[None]:
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


def _hash_fd(fd: int) -> tuple[str, int]:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise CandidateStoreError(
            "UNSAFE_FILESYSTEM_OBJECT", "regular unlinked file required"
        )
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    after = os.fstat(fd)
    if (before.st_dev, before.st_ino, before.st_size, stat.S_IFMT(before.st_mode)) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        stat.S_IFMT(after.st_mode),
    ) or size != before.st_size:
        raise CandidateStoreError("SOURCE_CHANGED_DURING_READ")
    return digest.hexdigest(), size


def _read_regular(path: Path) -> bytes:
    parent = _open_directory_path(path.parent)
    try:
        return _read_regular_at(parent, path.name)
    finally:
        os.close(parent)


def _write_atomic(path: Path, contents: bytes, mode: int = 0o600) -> None:
    parent = _open_directory_path(path.parent, create=True)
    try:
        _write_atomic_at(parent, path.name, contents, mode)
    finally:
        os.close(parent)


class CandidateStore:
    """The canonical Ares custody owner.

    ``root`` is injectable only for isolated tests. Production callers use the
    installation/user-scoped resolver rather than a profile runtime home.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        fault_injector: DurabilityFaultInjector | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else get_ares_state_root()
        self.candidates_root = self.root / "candidates"
        self._faults = fault_injector or DurabilityFaultInjector()

    def _checkpoint(self, point: str) -> None:
        self._faults.checkpoint(point)

    def _prepare(self) -> None:
        if os.name != "posix" or not hasattr(fcntl, "flock"):
            raise CandidateStoreError("DURABILITY_UNAVAILABLE", "POSIX flock required")
        with _umask_077():
            _mkdir_secure(self.root)
            _mkdir_secure(self.candidates_root)
            _mkdir_secure(self.candidates_root / ".incoming")
            _mkdir_secure(self.candidates_root / ".gc-quarantine")
            _mkdir_secure(self.candidates_root / "tombstones")
            candidates_fd = _open_directory_path(self.candidates_root)
            try:
                fd = _open_regular_at(
                    candidates_fd, ".store.lock", writable=True, create=True, mode=0o600
                )
                os.close(fd)
                _fsync(candidates_fd)
            finally:
                os.close(candidates_fd)

    @contextlib.contextmanager
    def _store_lock(self, *, exclusive: bool = True) -> Iterator[int]:
        self._prepare()
        candidates_fd = _open_directory_path(self.candidates_root)
        fd = _open_regular_at(candidates_fd, ".store.lock", writable=True)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            # The held candidates descriptor is the authority boundary for
            # every operation in the locked section.
            yield candidates_fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            os.close(candidates_fd)

    @contextlib.contextmanager
    def _candidate_lock(
        self, candidate_fd: int, *, exclusive: bool = True
    ) -> Iterator[int]:
        locks_fd = _open_directory_at(candidate_fd, "locks", create=True)
        try:
            fd = _open_regular_at(
                locks_fd, "lifecycle.lock", writable=True, create=True, mode=0o600
            )
        finally:
            os.close(locks_fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield candidate_fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _open_audit_lock(self, candidate_fd: int) -> int:
        locks_fd = _open_directory_at(candidate_fd, "locks", create=True)
        try:
            fd = _open_regular_at(
                locks_fd, "audit.lock", writable=True, create=True, mode=0o600
            )
        finally:
            os.close(locks_fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise CandidateStoreError("AUDIT_LOCKED") from exc
            raise CandidateStoreError("DURABILITY_UNAVAILABLE", str(exc)) from exc
        return fd

    def _audit_is_held(self, candidate_fd: int) -> bool:
        try:
            fd = self._open_audit_lock(candidate_fd)
        except CandidateStoreError as exc:
            if exc.code == "AUDIT_LOCKED":
                return True
            raise
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        return False

    def _candidate_root(self, sealed_candidate_id: str) -> Path:
        if (
            not isinstance(sealed_candidate_id, str)
            or len(sealed_candidate_id) != 64
            or any(c not in "0123456789abcdef" for c in sealed_candidate_id)
        ):
            raise CandidateStoreError("INVALID_SEALED_CANDIDATE_ID")
        return self.candidates_root / sealed_candidate_id

    def _copy_artifact(
        self, source_root: Path | int, incoming: Path | int, relative: str
    ) -> dict[str, Any]:
        relative = _safe_relative(relative)
        # Hold both authority roots before reading or creating a child.  Every
        # component below those roots is then opened with ``dir_fd`` and
        # O_NOFOLLOW; a source/destination swap cannot redirect the copy.
        source_fd = (
            os.dup(source_root)
            if isinstance(source_root, int)
            else _open_directory_path(source_root)
        )
        incoming_fd = (
            os.dup(incoming)
            if isinstance(incoming, int)
            else _open_directory_path(incoming)
        )
        try:
            src_fd = _open_regular_at(source_fd, relative)
            try:
                before = os.fstat(src_fd)
                _validate_stat(before, regular=True, links=1)
                target_relative = f"artifacts/{relative}"
                self._checkpoint("artifact.create")
                dst_fd = _open_regular_at(
                    incoming_fd,
                    target_relative,
                    writable=True,
                    create=True,
                    exclusive=True,
                    mode=0o600,
                )
                try:
                    self._checkpoint("artifact.write")
                    digest = hashlib.sha256()
                    count = 0
                    while True:
                        chunk = os.read(src_fd, 1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        count += len(chunk)
                        view = memoryview(chunk)
                        while view:
                            written = os.write(dst_fd, view)
                            if written <= 0:
                                raise CandidateStoreError(
                                    "DURABILITY_UNAVAILABLE", "short artifact write"
                                )
                            view = view[written:]
                    _fsync(dst_fd)
                    self._checkpoint("artifact.fsync")
                finally:
                    os.close(dst_fd)
                after = os.fstat(src_fd)
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    stat.S_IFMT(before.st_mode),
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    stat.S_IFMT(after.st_mode),
                ):
                    raise CandidateStoreError("SOURCE_CHANGED_DURING_READ", relative)
            finally:
                os.close(src_fd)
            # Publication never trusts the source digest.  Hash the reopened
            # destination descriptor while the incoming root remains held.
            fd = _open_regular_at(incoming_fd, f"artifacts/{relative}")
            try:
                actual_digest, actual_size = _hash_fd(fd)
            finally:
                os.close(fd)
            if actual_digest != digest.hexdigest() or actual_size != count:
                raise CandidateStoreError("DESTINATION_VERIFICATION_FAILED", relative)
            with _walk_directory(incoming_fd, "artifacts", create=True) as artifacts_fd:
                _fsync(artifacts_fd)
                self._checkpoint("artifact.directory_fsync")
            # The artifact is immutable as soon as it is destination-verified.
            with contextlib.ExitStack() as stack:
                parts = PurePosixPath(f"artifacts/{relative}").parts
                parent_fd = (
                    incoming_fd
                    if len(parts) == 1
                    else stack.enter_context(
                        _walk_directory(incoming_fd, "/".join(parts[:-1]))
                    )
                )
                os.chmod(parts[-1], 0o400, dir_fd=parent_fd, follow_symlinks=False)
                _fsync(parent_fd)
            fd = _open_regular_at(incoming_fd, f"artifacts/{relative}")
            try:
                destination = os.fstat(fd)
            finally:
                os.close(fd)
        finally:
            os.close(source_fd)
            os.close(incoming_fd)
        return {
            "relative_path": f"artifacts/{relative}",
            "object_type": "regular",
            "size": actual_size,
            "mode": stat.S_IMODE(destination.st_mode),
            "sha256": actual_digest,
            "nlink": destination.st_nlink,
            "uid": destination.st_uid,
            "gid": destination.st_gid,
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = _strict_json_object(_read_regular(path))
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            CandidateStoreError,
        ) as exc:
            if isinstance(exc, CandidateStoreError):
                raise
            raise CandidateStoreError("CUSTODY_CORRUPT", str(path)) from exc
        if not isinstance(value, dict):
            raise CandidateStoreError("CUSTODY_CORRUPT", str(path))
        return value

    @staticmethod
    def _read_json_at(parent_fd: int, relative: str) -> dict[str, Any]:
        """Decode one regular custody object through an already-held root."""
        try:
            value = _strict_json_object(_read_regular_at(parent_fd, relative))
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            CandidateStoreError,
        ) as exc:
            if isinstance(exc, CandidateStoreError):
                raise
            raise CandidateStoreError("CUSTODY_CORRUPT", relative) from exc
        if not isinstance(value, dict):
            raise CandidateStoreError("CUSTODY_CORRUPT", relative)
        return value

    @staticmethod
    def _candidate_fd(candidates_fd: int, sealed_candidate_id: str) -> int:
        if (
            not isinstance(sealed_candidate_id, str)
            or len(sealed_candidate_id) != 64
            or any(c not in "0123456789abcdef" for c in sealed_candidate_id)
        ):
            raise CandidateStoreError("INVALID_SEALED_CANDIDATE_ID")
        return _open_directory_at(candidates_fd, sealed_candidate_id)

    @staticmethod
    def _require_identity(value: dict[str, Any], schema: str, field: str) -> None:
        if (
            value.get("schema") != schema
            or value.get("canonicalization_version") != CANONICALIZATION_VERSION
        ):
            raise CandidateStoreError("CUSTODY_CORRUPT", f"wrong schema {schema}")
        if value.get(field) != _object_id(value, field):
            raise CandidateStoreError("CUSTODY_CORRUPT", f"invalid {field}")

    @staticmethod
    def _inventory_digest(inventory: Sequence[dict[str, Any]]) -> str:
        return sha256_bytes(
            canonical_json(sorted(inventory, key=lambda entry: entry["relative_path"]))
        )

    @staticmethod
    def _actual_artifact_inventory_fd(candidate_fd: int) -> list[dict[str, Any]]:
        """Enumerate the held artifact directory; inventory is never path-trusted.

        Candidate artifacts are deliberately flat.  This keeps the sealed
        inventory a one-to-one statement about every object that can be
        consumed as certification evidence, and rejects hidden trees,
        symlinks, special files, and unlisted directories outright.
        """
        artifacts_fd = _open_directory_at(candidate_fd, "artifacts")
        try:
            root = os.fstat(artifacts_fd)
            if (
                not stat.S_ISDIR(root.st_mode)
                # Directory link counts are not portable on every supported
                # local filesystem (some report 1 for an empty flat tree).
                # Extra directories are rejected by exact enumeration below.
                or root.st_nlink not in {1, 2}
                or root.st_uid != os.geteuid()  # windows-footgun: ok — module imports fcntl above, Unix-only by construction
                or stat.S_IMODE(root.st_mode) != 0o500
            ):
                raise CandidateStoreError("CUSTODY_CORRUPT", "artifact root mode")
            inventory: list[dict[str, Any]] = []
            for name in sorted(os.listdir(artifacts_fd)):
                if _safe_relative(name) != name or "/" in name:
                    raise CandidateStoreError("CUSTODY_CORRUPT", "artifact path")
                try:
                    info = os.stat(name, dir_fd=artifacts_fd, follow_symlinks=False)
                except OSError as exc:
                    _raise_os("CUSTODY_CORRUPT", f"artifact {name}", exc)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise CandidateStoreError(
                        "CUSTODY_CORRUPT", f"artifact object {name}"
                    )
                fd = _open_regular_at(artifacts_fd, name)
                try:
                    digest, size = _hash_fd(fd)
                    held = os.fstat(fd)
                finally:
                    os.close(fd)
                inventory.append({
                    "relative_path": f"artifacts/{name}",
                    "object_type": "regular",
                    "mode": stat.S_IMODE(held.st_mode),
                    "size": size,
                    "sha256": digest,
                    "nlink": held.st_nlink,
                    "uid": held.st_uid,
                    "gid": held.st_gid,
                })
            return inventory
        finally:
            os.close(artifacts_fd)

    @staticmethod
    def _validate_lifecycle_history_fd(
        candidate_fd: int, snapshot: dict[str, Any]
    ) -> None:
        """Require an exact, gap-free transition chain rather than its tail."""
        events_fd = _open_directory_at(candidate_fd, "events")
        try:
            sequence = int(snapshot["lifecycle_sequence"])
            names = sorted(os.listdir(events_fd))
            expected_names = [f"{item:020d}.json" for item in range(1, sequence + 1)]
            if names != expected_names:
                raise CandidateStoreError(
                    "CUSTODY_CORRUPT", "lifecycle event inventory"
                )
            previous = CandidateLifecycleState.CERTIFIED
            for number, name in enumerate(expected_names, start=1):
                event = CandidateStore._read_json_at(events_fd, name)
                if (
                    set(event)
                    != {
                        "schema",
                        "canonicalization_version",
                        "sequence",
                        "from",
                        "to",
                        "reason",
                        "created_at_unix_ns",
                    }
                    or event.get("schema") != EVENT_SCHEMA
                    or event.get("canonicalization_version") != CANONICALIZATION_VERSION
                    or event.get("sequence") != number
                    or event.get("from") != previous.value
                ):
                    raise CandidateStoreError(
                        "CUSTODY_CORRUPT", "lifecycle event chain"
                    )
                try:
                    current = CandidateLifecycleState(event["to"])
                    require_transition(previous, current)
                except (KeyError, ValueError) as exc:
                    raise CandidateStoreError(
                        "CUSTODY_CORRUPT", "lifecycle transition"
                    ) from exc
                previous = current
            if previous.value != snapshot["lifecycle_state"]:
                raise CandidateStoreError("CUSTODY_CORRUPT", "lifecycle tail")
        finally:
            os.close(events_fd)

    @staticmethod
    def _manifest_ref(
        relative_path: str, value: dict[str, Any], id_field: str, raw: bytes
    ) -> dict[str, str]:
        return {
            "relative_path": relative_path,
            "sha256": sha256_bytes(raw),
            "id": str(value[id_field]),
        }

    @staticmethod
    def _audit_subject(snapshot: dict[str, Any], custody_sha256: str) -> dict[str, Any]:
        """Create a non-circular immutable audit subject from sealed bytes.

        It deliberately excludes handoffs, leases, lifecycle sequence after
        publication, and retention state.  Those are mutable custody facts;
        the archive and sealed inventory are the thing an auditor certifies.
        """
        subject = {
            "schema": AUDIT_SUBJECT_SCHEMA,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "candidate_id": snapshot["candidate_id"],
            "certification_set_id": snapshot["certification_set_id"],
            "sealed_candidate_id": snapshot["sealed_candidate_id"],
            "archive_relative_path": snapshot["archive_relative_path"],
            "archive_sha256": snapshot["archive_sha256"],
            "sealed_artifact_inventory_sha256": snapshot[
                "sealed_artifact_inventory_sha256"
            ],
            "candidate_core": snapshot["candidate_core"],
            "certification_set_manifest": snapshot["certification_set_manifest"],
            "sealed_candidate_manifest": snapshot["sealed_candidate_manifest"],
            "post_seal_evidence_set": snapshot["post_seal_evidence_set"],
            "publication_lifecycle_sequence": snapshot["lifecycle_sequence"],
            "publication_custody_sha256": custody_sha256,
            "allowed_lifecycle_mutations": [
                "audit_state",
                "lifecycle_state",
                "lifecycle_sequence",
                "custody_revision",
                "retention_state",
            ],
        }
        subject["audit_subject_id"] = _object_id(subject, "audit_subject_id")
        return subject

    def _validate_archive_fd(
        self,
        archive_fd: int,
        core: dict[str, Any],
        cert: dict[str, Any],
        stored_core: bytes,
        stored_cert: bytes,
    ) -> None:
        seen: set[str] = set()
        collisions: set[str] = set()
        members: dict[str, tarfile.TarInfo] = {}
        try:
            # ``tarfile`` accepts a held file object.  Never hand it a
            # pathname after candidate authority has been established.
            with os.fdopen(os.dup(archive_fd), "rb") as archive_file:  # windows-footgun: ok — binary mode, encoding not applicable
                with tarfile.open(fileobj=archive_file, mode="r:") as bundle:
                    for member in bundle.getmembers():
                        name = _safe_relative(member.name)
                        fold = unicodedata.normalize("NFC", name).casefold()
                        if (
                            name in seen
                            or fold in collisions
                            or not member.isfile()
                            or member.issym()
                            or member.islnk()
                            or member.isdev()
                            or member.isfifo()
                        ):
                            raise CandidateStoreError(
                                "CUSTODY_CORRUPT", "unsafe archive member"
                            )
                        seen.add(name)
                        collisions.add(fold)
                        members[name] = member
                    expected: dict[str, str] = {}
                    payload = core.get("payload_files")
                    if not isinstance(payload, list):
                        raise CandidateStoreError(
                            "CUSTODY_CORRUPT", "core payload_files"
                        )
                    for entry in payload:
                        if (
                            not isinstance(entry, dict)
                            or not isinstance(entry.get("path"), str)
                            or not isinstance(entry.get("sha256"), str)
                        ):
                            raise CandidateStoreError(
                                "CUSTODY_CORRUPT", "core payload entry"
                            )
                        expected[_safe_relative(str(entry["path"]))] = str(
                            entry["sha256"]
                        )
                    expected["candidate-core-manifest.json"] = sha256_bytes(stored_core)
                    expected["certification-set-manifest.json"] = sha256_bytes(
                        stored_cert
                    )
                    artifacts = cert.get("artifacts")
                    if not isinstance(artifacts, list):
                        raise CandidateStoreError(
                            "CUSTODY_CORRUPT", "certification artifacts"
                        )
                    for entry in artifacts:
                        if (
                            not isinstance(entry, dict)
                            or not isinstance(entry.get("name"), str)
                            or not isinstance(entry.get("sha256"), str)
                        ):
                            raise CandidateStoreError(
                                "CUSTODY_CORRUPT", "certification artifact"
                            )
                        expected[_safe_relative(str(entry["name"]))] = str(
                            entry["sha256"]
                        )
                    if set(members) != set(expected):
                        raise CandidateStoreError(
                            "CUSTODY_CORRUPT", "archive inventory mismatch"
                        )
                    for name, expected_hash in expected.items():
                        stream = bundle.extractfile(members[name])
                        if stream is None:
                            raise CandidateStoreError("CUSTODY_CORRUPT", name)
                        with stream:
                            observed = sha256_bytes(stream.read())
                        if observed != expected_hash:
                            raise CandidateStoreError(
                                "CUSTODY_CORRUPT", f"archive digest {name}"
                            )
        except (OSError, tarfile.TarError) as exc:
            raise CandidateStoreError("CUSTODY_CORRUPT", "archive unreadable") from exc

    def _validate_snapshot_fd(
        self, candidate_fd: int, sealed_candidate_id: str
    ) -> dict[str, Any]:
        """Validate custody strictly through a held candidate descriptor."""
        _validate_stat(os.fstat(candidate_fd), directory=True)
        snapshot = self._read_json_at(candidate_fd, "custody.json")
        allowed = {
            "schema",
            "canonicalization_version",
            "candidate_id",
            "certification_set_id",
            "sealed_candidate_id",
            "archive_sha256",
            "archive_relative_path",
            "candidate_core",
            "certification_set_manifest",
            "sealed_candidate_manifest",
            "post_seal_evidence_set",
            "activation_authorization",
            "artifact_inventory",
            "artifact_inventory_sha256",
            "sealed_artifact_inventory_sha256",
            "source_repositories",
            "lifecycle_state",
            "lifecycle_sequence",
            "publication_state",
            "audit_state",
            "activation_authorization_state",
            "activation_grant",
            "rollback_required",
            "retention_state",
            "audit_subject",
            "audit_handoff",
            "custody_revision",
        }
        if (
            set(snapshot) - allowed
            or snapshot.get("schema") != CUSTODY_SCHEMA
            or snapshot.get("canonicalization_version") != CANONICALIZATION_VERSION
        ):
            raise CandidateStoreError(
                "CUSTODY_CORRUPT", "unknown/missing custody fields"
            )
        required = allowed - {"audit_subject", "audit_handoff", "activation_grant"}
        if not required <= set(snapshot):
            raise CandidateStoreError("CUSTODY_CORRUPT", "missing custody field")
        if sealed_candidate_id != snapshot["sealed_candidate_id"]:
            raise CandidateStoreError("CUSTODY_CORRUPT", "candidate directory identity")
        try:
            CandidateLifecycleState(snapshot["lifecycle_state"])
        except ValueError as exc:
            raise CandidateStoreError("CUSTODY_CORRUPT", "lifecycle state") from exc
        self._validate_lifecycle_history_fd(candidate_fd, snapshot)
        inventory = snapshot["artifact_inventory"]
        if (
            not isinstance(inventory, list)
            or self._inventory_digest(inventory)
            != snapshot["artifact_inventory_sha256"]
        ):
            raise CandidateStoreError("CUSTODY_CORRUPT", "inventory digest")
        paths: set[str] = set()
        folded: set[str] = set()
        for item in inventory:
            if not isinstance(item, dict) or set(item) != {
                "relative_path",
                "object_type",
                "size",
                "mode",
                "sha256",
                "nlink",
                "uid",
                "gid",
            }:
                raise CandidateStoreError("CUSTODY_CORRUPT", "inventory item")
            rel = _safe_relative(item["relative_path"])
            if (
                not rel.startswith("artifacts/")
                or rel in paths
                or unicodedata.normalize("NFC", rel).casefold() in folded
            ):
                raise CandidateStoreError(
                    "CUSTODY_CORRUPT", "duplicate/nonartifact inventory"
                )
            paths.add(rel)
            folded.add(unicodedata.normalize("NFC", rel).casefold())
        actual_inventory = self._actual_artifact_inventory_fd(candidate_fd)
        if actual_inventory != sorted(
            inventory, key=lambda item: item["relative_path"]
        ):
            raise CandidateStoreError("CUSTODY_CORRUPT", "artifact inventory mismatch")
        if snapshot["sealed_artifact_inventory_sha256"] != self._inventory_digest(
            actual_inventory
        ):
            raise CandidateStoreError("CUSTODY_CORRUPT", "sealed artifact inventory")
        archive_relative = _safe_relative(snapshot["archive_relative_path"])
        archive_fd = _open_regular_at(candidate_fd, archive_relative)
        try:
            archive_digest, _ = _hash_fd(archive_fd)
            os.lseek(archive_fd, 0, os.SEEK_SET)
            if archive_digest != snapshot["archive_sha256"]:
                raise CandidateStoreError("CUSTODY_CORRUPT", "archive digest")
            archive_for_validation = os.dup(archive_fd)
        finally:
            os.close(archive_fd)
        if archive_digest != snapshot["archive_sha256"]:
            raise CandidateStoreError("CUSTODY_CORRUPT", "archive digest")
        core_ref, cert_ref, seal_ref = (
            snapshot["candidate_core"],
            snapshot["certification_set_manifest"],
            snapshot["sealed_candidate_manifest"],
        )
        for ref, schema, field in (
            (core_ref, "CandidateCoreV2", "candidate_id"),
            (cert_ref, "CertificationSetV2", "certification_set_id"),
            (seal_ref, "SealedCandidateV2", "sealed_candidate_id"),
        ):
            if not isinstance(ref, dict) or set(ref) != {
                "relative_path",
                "sha256",
                "id",
            }:
                raise CandidateStoreError("CUSTODY_CORRUPT", "manifest reference")
            raw = _read_regular_at(candidate_fd, _safe_relative(ref["relative_path"]))
            if sha256_bytes(raw) != ref["sha256"]:
                raise CandidateStoreError("CUSTODY_CORRUPT", "manifest digest")
            try:
                value = _strict_json_object(raw)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise CandidateStoreError("CUSTODY_CORRUPT", "manifest JSON") from exc
            self._require_identity(value, schema, field)
            if value[field] != ref["id"]:
                raise CandidateStoreError("CUSTODY_CORRUPT", "manifest identity")
        core_raw = _read_regular_at(candidate_fd, core_ref["relative_path"])
        cert_raw = _read_regular_at(candidate_fd, cert_ref["relative_path"])
        try:
            core = _strict_json_object(core_raw)
            cert = _strict_json_object(cert_raw)
            sealed = self._read_json_at(candidate_fd, seal_ref["relative_path"])
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CandidateStoreError("CUSTODY_CORRUPT", "manifest JSON") from exc
        if (
            core["candidate_id"],
            cert["certification_set_id"],
            sealed["sealed_candidate_id"],
        ) != (
            snapshot["candidate_id"],
            snapshot["certification_set_id"],
            snapshot["sealed_candidate_id"],
        ):
            raise CandidateStoreError("CUSTODY_CORRUPT", "identity chain")
        if (
            cert.get("candidate_id") != core["candidate_id"]
            or sealed.get("candidate_id") != core["candidate_id"]
            or sealed.get("certification_set_id") != cert["certification_set_id"]
            or sealed.get("archive_sha256") != snapshot["archive_sha256"]
        ):
            raise CandidateStoreError("CUSTODY_CORRUPT", "identity binding")
        self._validate_non_authorizing_certification_fd(candidate_fd, cert, snapshot)
        for reference, schema, field in (
            (
                snapshot["post_seal_evidence_set"],
                "PostSealEvidenceSetV1",
                "post_seal_evidence_set_id",
            ),
            (
                snapshot["activation_authorization"],
                "ActivationAuthorizationV1",
                "activation_authorization_id",
            ),
        ):
            if not isinstance(reference, dict) or set(reference) != {
                "relative_path",
                "sha256",
                "id",
            }:
                raise CandidateStoreError("CUSTODY_CORRUPT", "post-seal reference")
            raw = _read_regular_at(candidate_fd, reference["relative_path"])
            if sha256_bytes(raw) != reference["sha256"]:
                raise CandidateStoreError("CUSTODY_CORRUPT", "post-seal object digest")
            value = self._read_json_at(candidate_fd, reference["relative_path"])
            self._require_identity(value, schema, field)
            if (
                value[field] != reference["id"]
                or value.get("candidate_id") != core["candidate_id"]
                or value.get("certification_set_id") != cert["certification_set_id"]
                or value.get("sealed_candidate_id") != sealed["sealed_candidate_id"]
                or value.get("archive_sha256") != snapshot["archive_sha256"]
            ):
                raise CandidateStoreError("CUSTODY_CORRUPT", "post-seal object binding")
            if (
                schema == "ActivationAuthorizationV1"
                and not self._legacy_failed_forensic(snapshot)
            ):
                required_auth = {
                    "schema",
                    "canonicalization_version",
                    "candidate_id",
                    "certification_set_id",
                    "sealed_candidate_id",
                    "post_seal_evidence_set_id",
                    "archive_sha256",
                    "rendered_config_path",
                    "rendered_config_sha256",
                    "authorization_state",
                    "non_authorizing",
                    "approved_release_root",
                    "governed_key_policy",
                    "activation_authorization_id",
                }
                if (
                    set(value) != required_auth
                    or value.get("authorization_state") != NON_AUTHORIZING
                    or value.get("non_authorizing") is not True
                ):
                    raise CandidateStoreError(
                        "INVALID_CERTIFICATION_AUTHORITY",
                        "candidate activation input must use the single non-authorizing state",
                    )
            if schema == "PostSealEvidenceSetV1":
                evidence = value.get("artifacts")
                indexed = {
                    item["relative_path"].removeprefix("artifacts/"): item
                    for item in actual_inventory
                }
                if not isinstance(evidence, list) or any(
                    not isinstance(item, dict)
                    or set(item) != {"name", "sha256"}
                    or indexed.get(item["name"], {}).get("sha256") != item["sha256"]
                    for item in evidence
                ):
                    raise CandidateStoreError(
                        "CUSTODY_CORRUPT", "post-seal evidence digest"
                    )
        try:
            self._validate_archive_fd(
                archive_for_validation, core, cert, core_raw, cert_raw
            )
        finally:
            os.close(archive_for_validation)
        subject_ref = snapshot.get("audit_subject")
        subject: dict[str, Any] | None = None
        if subject_ref is not None:
            if (
                not isinstance(subject_ref, dict)
                or set(subject_ref) != {"relative_path", "sha256", "id"}
                or subject_ref["relative_path"] != "audit-subject.json"
            ):
                raise CandidateStoreError("CUSTODY_CORRUPT", "audit subject reference")
            subject_raw = _read_regular_at(candidate_fd, subject_ref["relative_path"])
            if sha256_bytes(subject_raw) != subject_ref["sha256"]:
                raise CandidateStoreError("CUSTODY_CORRUPT", "audit subject digest")
            subject = self._read_json_at(candidate_fd, subject_ref["relative_path"])
            self._require_identity(subject, AUDIT_SUBJECT_SCHEMA, "audit_subject_id")
            if (
                subject["audit_subject_id"] != subject_ref["id"]
                or any(
                    subject[field] != snapshot[field]
                    for field in (
                        "candidate_id",
                        "certification_set_id",
                        "sealed_candidate_id",
                        "archive_sha256",
                        "archive_relative_path",
                    )
                )
                or subject["sealed_artifact_inventory_sha256"]
                != snapshot["sealed_artifact_inventory_sha256"]
            ):
                raise CandidateStoreError("CUSTODY_CORRUPT", "audit subject binding")
        handoff_ref = snapshot.get("audit_handoff")
        if handoff_ref is not None:
            if subject is None:
                raise CandidateStoreError("CUSTODY_CORRUPT", "handoff without subject")
            if (
                not isinstance(handoff_ref, dict)
                or set(handoff_ref) != {"relative_path", "sha256", "id"}
                or not isinstance(handoff_ref["relative_path"], str)
                or not handoff_ref["relative_path"].startswith("handoffs/")
            ):
                raise CandidateStoreError("CUSTODY_CORRUPT", "audit handoff reference")
            handoff_raw = _read_regular_at(candidate_fd, handoff_ref["relative_path"])
            if sha256_bytes(handoff_raw) != handoff_ref["sha256"]:
                raise CandidateStoreError("CUSTODY_CORRUPT", "audit handoff digest")
            handoff = self._read_json_at(candidate_fd, handoff_ref["relative_path"])
            if (
                handoff.get("schema") != HANDOFF_SCHEMA
                or handoff.get("canonicalization_version") != CANONICALIZATION_VERSION
                or handoff.get("hostile_audit_handoff_id")
                != _object_id(handoff, "hostile_audit_handoff_id")
                or handoff.get("hostile_audit_handoff_id") != handoff_ref["id"]
            ):
                raise CandidateStoreError("CUSTODY_CORRUPT", "audit handoff identity")
            expected_handoff = {
                "candidate_id": snapshot["candidate_id"],
                "certification_set_id": snapshot["certification_set_id"],
                "sealed_candidate_id": sealed_candidate_id,
                "archive_relative_path": snapshot["archive_relative_path"],
                "archive_sha256": snapshot["archive_sha256"],
                "audit_subject_id": subject["audit_subject_id"],
                "audit_subject_sha256": subject_ref["sha256"],
                "publication_custody_sha256": subject["publication_custody_sha256"],
                "sealed_artifact_inventory_sha256": snapshot[
                    "sealed_artifact_inventory_sha256"
                ],
            }
            if any(
                handoff.get(key) != value for key, value in expected_handoff.items()
            ):
                raise CandidateStoreError("CUSTODY_CORRUPT", "audit handoff binding")
            if handoff.get("candidate_root") != str(
                self._candidate_root(sealed_candidate_id)
            ):
                raise CandidateStoreError("CUSTODY_CORRUPT", "audit handoff root")
        elif subject_ref is not None and snapshot["lifecycle_state"] in {
            CandidateLifecycleState.AWAITING_HOSTILE_AUDIT.value,
            CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS.value,
            CandidateLifecycleState.AUDIT_BLOCKED.value,
            CandidateLifecycleState.AUDIT_PASSED.value,
            CandidateLifecycleState.AUDIT_FAILED.value,
        }:
            raise CandidateStoreError(
                "CUSTODY_CORRUPT", "missing required audit handoff"
            )
        authorization_state = snapshot["activation_authorization_state"]
        lifecycle_state = CandidateLifecycleState(snapshot["lifecycle_state"])
        if authorization_state not in {UNAUTHORIZED, AUTHORIZED}:
            raise CandidateStoreError(
                "INVALID_CERTIFICATION_AUTHORITY", "unknown activation authority"
            )
        if authorization_state == AUTHORIZED and lifecycle_state not in {
            CandidateLifecycleState.AWAITING_ACTIVATION,
            CandidateLifecycleState.ACTIVE,
            CandidateLifecycleState.ROLLBACK_REQUIRED,
        }:
            raise CandidateStoreError(
                "AuthorizationStateContradiction",
                "authorized activation without governed transition",
            )
        grant_value = snapshot.get("activation_grant")
        if authorization_state == AUTHORIZED and grant_value is None:
            raise CandidateStoreError(
                "AuthorizationStateContradiction", "authorized without activation grant"
            )
        if grant_value is not None:
            try:
                grant = ActivationGrant.parse(canonical_json(grant_value) + b"\n")
            except (AresRuntimeError, TypeError) as exc:
                raise CandidateStoreError("ACTIVATION_GRANT_CORRUPT", str(exc)) from exc
            expected_grant = {
                "candidate_id": snapshot["candidate_id"],
                "certification_set_id": snapshot["certification_set_id"],
                "sealed_candidate_id": snapshot["sealed_candidate_id"],
                "archive_sha256": snapshot["archive_sha256"],
                "candidate_core_sha256": snapshot["candidate_core"]["sha256"],
                "sealed_manifest_sha256": snapshot["sealed_candidate_manifest"][
                    "sha256"
                ],
            }
            if any(
                getattr(grant, key) != expected
                for key, expected in expected_grant.items()
            ):
                raise CandidateStoreError(
                    "ACTIVATION_GRANT_CORRUPT", "candidate binding"
                )
            try:
                self._validate_grant_runtime_binding_fd(candidate_fd, snapshot, grant)
            except CandidateStoreError as exc:
                raise CandidateStoreError(
                    "ACTIVATION_GRANT_CORRUPT", "runtime identity binding"
                ) from exc
            if subject_ref is None or (
                grant.audit_subject_id != subject_ref["id"]
                or grant.audit_subject_sha256 != subject_ref["sha256"]
            ):
                raise CandidateStoreError("ACTIVATION_GRANT_CORRUPT", "audit subject")
            if grant.custody_event_sequence < 2 or grant.custody_event_sequence > int(
                snapshot["lifecycle_sequence"]
            ):
                raise CandidateStoreError(
                    "ACTIVATION_GRANT_CORRUPT", "lifecycle sequence"
                )
            events_fd = _open_directory_at(candidate_fd, "events")
            try:
                audit_event_raw = _read_regular_at(
                    events_fd, f"{grant.custody_event_sequence - 1:020d}.json"
                )
            finally:
                os.close(events_fd)
            try:
                audit_event = _strict_json_object(audit_event_raw)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise CandidateStoreError(
                    "ACTIVATION_GRANT_CORRUPT", "audit event"
                ) from exc
            if (
                grant.audit_result_sha256 != sha256_bytes(audit_event_raw)
                or audit_event.get("to") != CandidateLifecycleState.AUDIT_PASSED.value
            ):
                raise CandidateStoreError("ACTIVATION_GRANT_CORRUPT", "audit result")
        if (
            lifecycle_state
            in {
                CandidateLifecycleState.AUDIT_BLOCKED,
                CandidateLifecycleState.AUDIT_FAILED,
            }
            and authorization_state != UNAUTHORIZED
        ):
            raise CandidateStoreError(
                "AuthorizationStateContradiction", "blocked or failed audit authorized"
            )
        return snapshot

    @staticmethod
    def _validate_full_seal_certification(
        certification: dict[str, Any],
        candidate_id: str,
        required_artifact_names: set[str],
    ) -> None:
        """Validate the complete positive FULL_SEAL contract.

        A FULL_SEAL is not a label applied to a staged result.  Its nested
        generation records are authority-bearing evidence, so every field is
        closed and every hard PASS is recomputed from the recorded samples.
        This is the sole semantic admission point for a full seal; callers
        reach it through snapshot validation before publication, audit, or the
        governed authorization transition.
        """

        def fail(detail: str) -> None:
            raise CandidateStoreError("INVALID_CERTIFICATION_AUTHORITY", detail)

        def exact_object(value: Any, fields: set[str], detail: str) -> dict[str, Any]:
            if not isinstance(value, dict) or set(value) != fields:
                fail(detail)
            return value

        def finite_number(value: Any) -> bool:
            return type(value) in {int, float} and math.isfinite(value)

        def digest(value: Any) -> bool:
            return (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            )

        required = {
            "schema",
            "canonicalization_version",
            "certification_purpose",
            "certification_mode",
            "candidate_id",
            "candidate_core_id",
            "certification_set_inputs",
            "required_generations",
            "required_warmup_runs",
            "required_measured_runs",
            "generations",
            "pass",
            "terminal_outcome",
            "hard_pass",
            "failing_hard_metric_ids",
            "exact_expansion",
            "authenticated_restart_load",
            "rendered_prompt_provenance",
            "integrity_hmac",
            "authorization_state",
            "non_authorizing",
        }
        if set(certification) != required:
            fail("full-seal certification fields")
        inputs = certification.get("certification_set_inputs")
        if (
            certification.get("schema") != FULL_SEAL_CERTIFICATION_SCHEMA
            or certification.get("canonicalization_version") != CANONICALIZATION_VERSION
            or certification.get("certification_purpose") != FULL_SEAL_PURPOSE
            or certification.get("certification_mode") != FULL_SEAL_PURPOSE
            or certification.get("candidate_id") != candidate_id
            or certification.get("candidate_core_id") != candidate_id
            or not isinstance(inputs, dict)
            or set(inputs)
            != {"candidate_id", "candidate_core_id", "required_artifact_names"}
            or inputs.get("candidate_id") != candidate_id
            or inputs.get("candidate_core_id") != candidate_id
            or inputs.get("required_artifact_names") != sorted(required_artifact_names)
            or certification.get("required_generations") != [16, 32]
            or certification.get("required_warmup_runs") != 3
            or certification.get("required_measured_runs") != 10
            or type(certification.get("pass")) is not bool
            or certification["pass"] is not True
            or certification.get("terminal_outcome") != "PASS"
            or type(certification.get("hard_pass")) is not bool
            or certification["hard_pass"] is not True
            or certification.get("failing_hard_metric_ids") != []
            or any(
                certification.get(name) != "PASS"
                for name in (
                    "exact_expansion",
                    "authenticated_restart_load",
                    "rendered_prompt_provenance",
                    "integrity_hmac",
                )
            )
            or certification.get("authorization_state") != NON_AUTHORIZING
            or certification.get("non_authorizing") is not True
        ):
            fail("ineligible full-seal certification")

        generations = certification.get("generations")
        if not isinstance(generations, list) or len(generations) != 2:
            fail("generation evidence")
        generation_fields = {
            "generation",
            "warmup_runs",
            "measured_runs",
            "raw_measurement_samples",
            "threshold_evaluations",
            "soft_warning_evaluations",
            "failing_metric_ids",
            "hard_pass",
            "terminal_outcome",
            "p50",
            "p95",
            "max",
        }
        metric_fields = {
            "prompt_visible_provenance_bytes",
            "prompt_visible_provenance_tokens",
            "authoritative_provenance_bytes",
            "receipt_bytes",
            "cumulative_receipt_store_bytes",
            "compaction_latency_ms",
            "restart_load_latency_ms",
            "exact_expansion_latency_ms",
            "input_tokens",
            "output_tokens",
            "net_token_savings",
            "budget_decision",
            "exact_expansion_hash",
            "exact_expansion_expected_hash",
            "exact_expansion_result",
            "authenticated_restart_load_result",
            "rendered_prompt_provenance_result",
            "hmac_verification_result",
            "key_id",
        }
        numeric_metric_names = {
            "prompt_visible_provenance_bytes",
            "prompt_visible_provenance_tokens",
            "authoritative_provenance_bytes",
            "receipt_bytes",
            "cumulative_receipt_store_bytes",
            "compaction_latency_ms",
            "restart_load_latency_ms",
            "exact_expansion_latency_ms",
            "input_tokens",
            "output_tokens",
            "net_token_savings",
        }
        hard_sample_rules = (
            ("prompt_visible_provenance_bytes", 512, "upper"),
            ("prompt_visible_provenance_tokens", 128, "upper"),
            ("authoritative_provenance_bytes", 131072, "upper"),
            ("receipt_bytes", 524288, "upper"),
            ("cumulative_receipt_store_bytes", None, "upper"),
            ("net_token_savings", 128, "lower"),
        )
        summary_metric_names = (
            "compaction_latency_ms",
            "restart_load_latency_ms",
            "exact_expansion_latency_ms",
        )
        expected_generations = (16, 32)
        for generation, number in zip(generations, expected_generations, strict=True):
            record = exact_object(generation, generation_fields, "generation schema")
            if (
                record.get("generation") != number
                or record.get("warmup_runs") != 3
                or record.get("measured_runs") != 10
                or record.get("hard_pass") is not True
                or record.get("terminal_outcome") != "PASS"
                or record.get("failing_metric_ids") != []
            ):
                fail("incomplete generation evidence")
            samples = record["raw_measurement_samples"]
            if not isinstance(samples, list) or len(samples) != 13:
                fail("generation sample counts")
            expected_samples = [
                (phase, index)
                for phase, count in (("warmup", 3), ("measured", 10))
                for index in range(count)
            ]
            measured_metrics: list[dict[str, Any]] = []
            for sample, (phase, index) in zip(samples, expected_samples, strict=True):
                item = exact_object(
                    sample,
                    {"phase", "sample_index", "metrics"},
                    "sample schema",
                )
                metrics = exact_object(
                    item.get("metrics"), metric_fields, "sample metrics"
                )
                if item.get("phase") != phase or item.get("sample_index") != index:
                    fail("generation sample ordering")
                if any(
                    not finite_number(metrics[name]) or metrics[name] < 0
                    for name in numeric_metric_names
                ):
                    fail("sample metric type")
                if (
                    any(
                        type(metrics[name]) is not int
                        for name in (
                            "prompt_visible_provenance_bytes",
                            "prompt_visible_provenance_tokens",
                            "authoritative_provenance_bytes",
                            "receipt_bytes",
                            "cumulative_receipt_store_bytes",
                            "input_tokens",
                            "output_tokens",
                            "net_token_savings",
                        )
                    )
                    or metrics["input_tokens"] <= metrics["output_tokens"]
                    or metrics["net_token_savings"]
                    != metrics["input_tokens"] - metrics["output_tokens"]
                    or metrics["budget_decision"] != "admit"
                    or not digest(metrics["exact_expansion_hash"])
                    or metrics["exact_expansion_hash"]
                    != metrics["exact_expansion_expected_hash"]
                    or not digest(metrics["key_id"])
                    or any(
                        metrics[name] != "PASS"
                        for name in (
                            "exact_expansion_result",
                            "authenticated_restart_load_result",
                            "rendered_prompt_provenance_result",
                            "hmac_verification_result",
                        )
                    )
                ):
                    fail("sample hard evidence")
                if phase == "measured":
                    measured_metrics.append(metrics)

            thresholds = record["threshold_evaluations"]
            if not isinstance(thresholds, list) or len(thresholds) != 120:
                fail("threshold evidence")
            threshold_index = 0
            for metrics, (phase, index) in zip(samples, expected_samples, strict=True):
                values = metrics["metrics"]
                for metric_id, hard_limit, direction in hard_sample_rules:
                    decision = exact_object(
                        thresholds[threshold_index],
                        {
                            "metric_id",
                            "phase",
                            "sample_index",
                            "observed",
                            "hard_limit",
                            "pass",
                        },
                        "sample threshold schema",
                    )
                    threshold_index += 1
                    expected_limit = (
                        number * 524288 if hard_limit is None else hard_limit
                    )
                    passed = (
                        values[metric_id] <= expected_limit
                        if direction == "upper"
                        else values[metric_id] >= expected_limit
                    )
                    if (
                        decision.get("metric_id") != metric_id
                        or decision.get("phase") != phase
                        or decision.get("sample_index") != index
                        or decision.get("observed") != values[metric_id]
                        or decision.get("hard_limit") != expected_limit
                        or decision.get("pass") is not True
                        or not passed
                    ):
                        fail("sample threshold evidence")
                for metric_id, observed, hard_limit in (
                    ("budget_decision", values["budget_decision"], "admit"),
                    ("exact_expansion", values["exact_expansion_result"], "PASS"),
                    ("hmac_verification", values["hmac_verification_result"], "PASS"),
                ):
                    decision = exact_object(
                        thresholds[threshold_index],
                        {
                            "metric_id",
                            "phase",
                            "sample_index",
                            "observed",
                            "hard_limit",
                            "pass",
                        },
                        "sample threshold schema",
                    )
                    threshold_index += 1
                    if (
                        decision.get("metric_id") != metric_id
                        or decision.get("phase") != phase
                        or decision.get("sample_index") != index
                        or decision.get("observed") != observed
                        or decision.get("hard_limit") != hard_limit
                        or decision.get("pass") is not True
                        or observed != hard_limit
                    ):
                        fail("sample threshold evidence")

            def percentile(values: list[Any], fraction: float) -> Any:
                ordered = sorted(values)
                return ordered[int(len(ordered) * fraction + 0.999999) - 1]

            summaries = {
                "p50": 0.5,
                "p95": 0.95,
                "max": None,
            }
            summary_values: dict[str, dict[str, Any]] = {}
            for field, fraction in summaries.items():
                summary = exact_object(
                    record[field], set(summary_metric_names), f"{field} schema"
                )
                expected = {
                    metric_id: (
                        max(item[metric_id] for item in measured_metrics)
                        if fraction is None
                        else percentile(
                            [item[metric_id] for item in measured_metrics], fraction
                        )
                    )
                    for metric_id in summary_metric_names
                }
                if any(
                    not finite_number(summary[metric_id])
                    or summary[metric_id] != expected[metric_id]
                    for metric_id in summary_metric_names
                ):
                    fail(f"{field} evidence")
                summary_values[field] = expected
            for metric_id, hard_limit in (
                ("compaction_latency_ms", 5000),
                ("restart_load_latency_ms", 500),
                ("exact_expansion_latency_ms", 500),
            ):
                decision = exact_object(
                    thresholds[threshold_index],
                    {"metric_id", "observed", "hard_limit", "pass"},
                    "p95 threshold schema",
                )
                threshold_index += 1
                observed = summary_values["p95"][metric_id]
                if (
                    decision.get("metric_id")
                    != {
                        "compaction_latency_ms": "compaction_p95_ms",
                        "restart_load_latency_ms": "restart_load_p95_ms",
                        "exact_expansion_latency_ms": "exact_expansion_p95_ms",
                    }[metric_id]
                    or decision.get("observed") != observed
                    or decision.get("hard_limit") != hard_limit
                    or decision.get("pass") is not True
                    or observed > hard_limit
                ):
                    fail("p95 threshold evidence")

            soft_warnings = record["soft_warning_evaluations"]
            if not isinstance(soft_warnings, list) or len(soft_warnings) != 78:
                fail("soft warning evidence")
            warning_index = 0
            for sample, (phase, index) in zip(samples, expected_samples, strict=True):
                values = sample["metrics"]
                for metric_id, source_metric, limit in (
                    ("receipt_bytes_soft_warning", "receipt_bytes", 393216),
                    (
                        "cumulative_receipt_store_bytes_soft_warning",
                        "cumulative_receipt_store_bytes",
                        number * 393216,
                    ),
                    ("compaction_latency_soft_warning", "compaction_latency_ms", 2000),
                    (
                        "restart_load_latency_soft_warning",
                        "restart_load_latency_ms",
                        100,
                    ),
                    (
                        "exact_expansion_latency_soft_warning",
                        "exact_expansion_latency_ms",
                        100,
                    ),
                ):
                    warning = exact_object(
                        soft_warnings[warning_index],
                        {
                            "metric_id",
                            "phase",
                            "sample_index",
                            "observed",
                            "warning_limit",
                            "triggered",
                        },
                        "soft warning schema",
                    )
                    warning_index += 1
                    if (
                        warning.get("metric_id") != metric_id
                        or warning.get("phase") != phase
                        or warning.get("sample_index") != index
                        or warning.get("observed") != values[source_metric]
                        or warning.get("warning_limit") != limit
                        or type(warning.get("triggered")) is not bool
                        or warning["triggered"] != (values[source_metric] > limit)
                    ):
                        fail("soft warning evidence")
                warning = exact_object(
                    soft_warnings[warning_index],
                    {"metric_id", "phase", "sample_index", "observed", "triggered"},
                    "soft warning schema",
                )
                warning_index += 1
                if (
                    warning.get("metric_id") != "approximate_counter_usage"
                    or warning.get("phase") != phase
                    or warning.get("sample_index") != index
                    or warning.get("observed") is not True
                    or warning.get("triggered") is not True
                ):
                    fail("soft warning evidence")

    def _validate_non_authorizing_certification_fd(
        self,
        candidate_fd: int,
        certification_set: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> None:
        """Validate the one activation-eligible full-seal evidence shape.

        A full seal remains non-authorizing by itself: it is a positive
        prerequisite to the separately governed store transition, not an
        activation grant.  Staged/diagnostic certificates intentionally use a
        different schema and can never be admitted here.
        """
        if self._legacy_failed_forensic(snapshot):
            # The two failed Sol subjects are read-only evidence.  They must
            # remain inspectable, but are never a publication or authorization
            # input under the V2 contract.
            return
        if set(certification_set) != {
            "schema",
            "canonicalization_version",
            "candidate_id",
            "artifacts",
            "certification_set_id",
        }:
            raise CandidateStoreError(
                "INVALID_CERTIFICATION_AUTHORITY", "certification set schema"
            )
        artifacts = certification_set.get("artifacts")
        if not isinstance(artifacts, list):
            raise CandidateStoreError("INVALID_CERTIFICATION_AUTHORITY", "artifacts")
        expected_names = {
            "gen-certification.json",
            "scope-proof.json",
            "preseal-secret-scan.json",
        }
        if {
            item.get("name")
            for item in artifacts
            if isinstance(item, dict) and set(item) == {"name", "sha256"}
        } != expected_names or len(artifacts) != len(expected_names):
            raise CandidateStoreError(
                "INVALID_CERTIFICATION_AUTHORITY", "certification set inputs"
            )
        matches = [
            item
            for item in artifacts
            if isinstance(item, dict) and item.get("name") == "gen-certification.json"
        ]
        if len(matches) != 1 or set(matches[0]) != {"name", "sha256"}:
            raise CandidateStoreError(
                "INVALID_CERTIFICATION_AUTHORITY", "missing certification evidence"
            )
        raw = _read_regular_at(candidate_fd, "artifacts/gen-certification.json")
        if sha256_bytes(raw) != matches[0]["sha256"]:
            raise CandidateStoreError(
                "INVALID_CERTIFICATION_AUTHORITY", "certification digest"
            )
        try:
            certification = _strict_json_object(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CandidateStoreError(
                "INVALID_CERTIFICATION_AUTHORITY", "malformed certification"
            ) from exc
        self._validate_full_seal_certification(
            certification, snapshot["candidate_id"], expected_names
        )

    def _validate_authorization_prerequisites_fd(
        self, candidate_fd: int, snapshot: dict[str, Any]
    ) -> None:
        """Require every positive, candidate-bound authorization prerequisite."""
        # ``_validate_snapshot_fd`` already verifies immutable inventory,
        # identities, full-seal certificate, archive, post-seal binding, and
        # exact audit subject/handoff.  Re-read candidate-bound PASS artifacts
        # here so an audit PASS cannot substitute for any certification gate.
        required = {
            "archive-verification.json": (
                "AresContextGovernorArchiveVerificationV2",
                True,
            ),
            "scope-proof.json": ("AresContextGovernorScopeProofV2", True),
            "preseal-secret-scan.json": ("AresContextGovernorSecretScanV2", True),
            "postseal-secret-scan.json": ("AresContextGovernorSecretScanV2", True),
            "v1-immutability.json": ("AresContextGovernorV1ImmutabilityV2", True),
            "dry-run-activation.json": ("AresContextGovernorDryRunActivationV3", True),
        }
        indexed = {
            item["relative_path"].removeprefix("artifacts/"): item
            for item in snapshot["artifact_inventory"]
        }
        for name, (schema, expected_pass) in required.items():
            item = indexed.get(name)
            if item is None:
                raise CandidateStoreError("MISSING_AUTHORIZATION_EVIDENCE", name)
            raw = _read_regular_at(candidate_fd, f"artifacts/{name}")
            if sha256_bytes(raw) != item["sha256"]:
                raise CandidateStoreError("MISSING_AUTHORIZATION_EVIDENCE", name)
            try:
                evidence = _strict_json_object(raw)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise CandidateStoreError(
                    "MISSING_AUTHORIZATION_EVIDENCE", name
                ) from exc
            if (
                evidence.get("schema") != schema
                or type(evidence.get("pass")) is not bool
                or evidence["pass"] is not expected_pass
            ):
                raise CandidateStoreError("MISSING_AUTHORIZATION_EVIDENCE", name)
            if (
                name != "v1-immutability.json"
                and evidence.get("candidate_id") != snapshot["candidate_id"]
            ):
                raise CandidateStoreError("MISSING_AUTHORIZATION_EVIDENCE", name)
            if name in {"archive-verification.json", "postseal-secret-scan.json"} and (
                evidence.get("sealed_candidate_id") != snapshot["sealed_candidate_id"]
                or evidence.get("archive_sha256") != snapshot["archive_sha256"]
            ):
                raise CandidateStoreError("MISSING_AUTHORIZATION_EVIDENCE", name)

    @staticmethod
    def _legacy_failed_forensic(snapshot: dict[str, Any]) -> bool:
        """Permit read-only custody verification of the already failed finding.

        The Sol-audited candidate predates the explicit authority field and is
        immutable forensic evidence.  It is never a publishable input and its
        terminal audit failure keeps it categorically non-authorizing.
        """
        return (
            snapshot.get("lifecycle_state")
            == CandidateLifecycleState.AUDIT_FAILED.value
            and snapshot.get("audit_state") == "AUDIT_FAILED"
            and snapshot.get("activation_authorization_state") == UNAUTHORIZED
        )

    def _write_event_and_snapshot_fd(
        self,
        candidate_fd: int,
        snapshot: dict[str, Any],
        previous: CandidateLifecycleState,
        next_state: CandidateLifecycleState,
        reason: str,
    ) -> dict[str, Any]:
        require_transition(previous, next_state)
        events_fd = _open_directory_at(candidate_fd, "events", create=True)
        sequence = int(snapshot["lifecycle_sequence"]) + 1
        event = {
            "schema": EVENT_SCHEMA,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "sequence": sequence,
            "from": previous.value,
            "to": next_state.value,
            "reason": reason,
            "created_at_unix_ns": time.time_ns(),
        }
        try:
            self._checkpoint("lifecycle_event.write")
            _write_atomic_at(
                events_fd, f"{sequence:020d}.json", canonical_json(event) + b"\n", 0o600
            )
            _fsync(events_fd)
            self._checkpoint("lifecycle_event.fsync")
        finally:
            os.close(events_fd)
        updated = dict(snapshot)
        updated["lifecycle_state"] = next_state.value
        updated["lifecycle_sequence"] = sequence
        updated["custody_revision"] = int(snapshot["custody_revision"]) + 1
        self._checkpoint("custody_snapshot.write")
        _write_atomic_at(
            candidate_fd, "custody.json", canonical_json(updated) + b"\n", 0o600
        )
        self._checkpoint("custody_snapshot.rename_fsync")
        _fsync(candidate_fd)
        self._checkpoint("candidate_directory.fsync")
        return updated

    def publish(
        self,
        source_root: Path,
        artifact_paths: Sequence[str],
        *,
        require_fault_matrix: bool = True,
    ) -> PublicationResult:
        """Copy, independently verify, and atomically publish exact artifacts."""
        source_root = Path(source_root)
        if not artifact_paths or len(set(artifact_paths)) != len(artifact_paths):
            raise CandidateStoreError("INVALID_ARTIFACT_INVENTORY")
        with self._store_lock() as candidates_fd:
            # Source identity is only an input; nothing under it becomes custody.
            source_fd = _open_directory_path(source_root)
            sealed = self._read_json_at(source_fd, "sealed-candidate-manifest.json")
            self._require_identity(sealed, "SealedCandidateV2", "sealed_candidate_id")
            sealed_id = str(sealed["sealed_candidate_id"])
            final = self._candidate_root(sealed_id)
            try:
                existing_fd = self._candidate_fd(candidates_fd, sealed_id)
            except CandidateStoreError as exc:
                if exc.code != "CUSTODY_UNAVAILABLE":
                    raise
                existing_fd = -1
            if existing_fd >= 0:
                try:
                    existing = self._validate_snapshot_fd(existing_fd, sealed_id)
                    custody_digest = sha256_bytes(
                        _read_regular_at(existing_fd, "custody.json")
                    )
                finally:
                    os.close(existing_fd)
                if existing["archive_sha256"] == sha256_bytes(
                    _read_regular_at(source_fd, "ares-context-governor-candidate.tar")
                ):
                    os.close(source_fd)
                    return PublicationResult(
                        "ALREADY_PUBLISHED_VERIFIED",
                        sealed_id,
                        final,
                        custody_digest,
                        int(existing["lifecycle_sequence"]),
                    )
                os.close(source_fd)
                raise CandidateStoreError("PUBLICATION_CONFLICT", sealed_id)
            incoming_name = uuid.uuid4().hex
            incoming = self.candidates_root / ".incoming" / incoming_name
            incoming_parent_fd = _open_directory_at(candidates_fd, ".incoming")
            try:
                os.mkdir(incoming_name, mode=0o700, dir_fd=incoming_parent_fd)
                incoming_fd = _open_directory_at(incoming_parent_fd, incoming_name)
            finally:
                os.close(incoming_parent_fd)
            try:
                inventory = [
                    self._copy_artifact(source_fd, incoming_fd, path)
                    for path in sorted(artifact_paths)
                ]
                events_fd = _open_directory_at(incoming_fd, "events", create=True)
                locks_fd = _open_directory_at(incoming_fd, "locks", create=True)
                os.close(events_fd)
                os.close(locks_fd)
                # Build references from destination bytes only.
                by_original = {
                    item["relative_path"].removeprefix("artifacts/"): item
                    for item in inventory
                }
                required = {
                    "candidate-core-manifest.json",
                    "certification-set-manifest.json",
                    "sealed-candidate-manifest.json",
                    "post-seal-evidence-set.json",
                    "activation-authorization.json",
                    "ares-context-governor-candidate.tar",
                }
                if require_fault_matrix:
                    required.add("custody-fault-matrix-v1.json")
                if not required <= set(by_original):
                    raise CandidateStoreError("MISSING_REQUIRED_ARTIFACT")
                core = self._read_json_at(
                    incoming_fd, "artifacts/candidate-core-manifest.json"
                )
                cert = self._read_json_at(
                    incoming_fd, "artifacts/certification-set-manifest.json"
                )
                seal = self._read_json_at(
                    incoming_fd, "artifacts/sealed-candidate-manifest.json"
                )
                post = self._read_json_at(
                    incoming_fd, "artifacts/post-seal-evidence-set.json"
                )
                auth = self._read_json_at(
                    incoming_fd, "artifacts/activation-authorization.json"
                )
                self._require_identity(core, "CandidateCoreV2", "candidate_id")
                self._require_identity(
                    cert, "CertificationSetV2", "certification_set_id"
                )
                self._require_identity(seal, "SealedCandidateV2", "sealed_candidate_id")
                self._require_identity(
                    post, "PostSealEvidenceSetV1", "post_seal_evidence_set_id"
                )
                self._require_identity(
                    auth, "ActivationAuthorizationV1", "activation_authorization_id"
                )
                if (
                    seal != sealed
                    or seal["candidate_id"] != core["candidate_id"]
                    or seal["certification_set_id"] != cert["certification_set_id"]
                ):
                    raise CandidateStoreError(
                        "PUBLICATION_CONFLICT", "identity documents disagree"
                    )
                archive_rel = "artifacts/ares-context-governor-candidate.tar"
                archive_sha = by_original["ares-context-governor-candidate.tar"][
                    "sha256"
                ]
                if seal.get("archive_sha256") != archive_sha:
                    raise CandidateStoreError(
                        "DESTINATION_VERIFICATION_FAILED", "archive identity"
                    )
                archive_fd = _open_regular_at(incoming_fd, archive_rel)
                try:
                    self._validate_archive_fd(
                        archive_fd,
                        core,
                        cert,
                        _read_regular_at(
                            incoming_fd, "artifacts/candidate-core-manifest.json"
                        ),
                        _read_regular_at(
                            incoming_fd, "artifacts/certification-set-manifest.json"
                        ),
                    )
                finally:
                    os.close(archive_fd)
                snapshot = {
                    "schema": CUSTODY_SCHEMA,
                    "canonicalization_version": CANONICALIZATION_VERSION,
                    "candidate_id": core["candidate_id"],
                    "certification_set_id": cert["certification_set_id"],
                    "sealed_candidate_id": sealed_id,
                    "archive_sha256": archive_sha,
                    "archive_relative_path": archive_rel,
                    "candidate_core": self._manifest_ref(
                        "artifacts/candidate-core-manifest.json",
                        core,
                        "candidate_id",
                        _read_regular_at(
                            incoming_fd, "artifacts/candidate-core-manifest.json"
                        ),
                    ),
                    "certification_set_manifest": self._manifest_ref(
                        "artifacts/certification-set-manifest.json",
                        cert,
                        "certification_set_id",
                        _read_regular_at(
                            incoming_fd, "artifacts/certification-set-manifest.json"
                        ),
                    ),
                    "sealed_candidate_manifest": self._manifest_ref(
                        "artifacts/sealed-candidate-manifest.json",
                        seal,
                        "sealed_candidate_id",
                        _read_regular_at(
                            incoming_fd, "artifacts/sealed-candidate-manifest.json"
                        ),
                    ),
                    "post_seal_evidence_set": self._manifest_ref(
                        "artifacts/post-seal-evidence-set.json",
                        post,
                        "post_seal_evidence_set_id",
                        _read_regular_at(
                            incoming_fd, "artifacts/post-seal-evidence-set.json"
                        ),
                    ),
                    "activation_authorization": self._manifest_ref(
                        "artifacts/activation-authorization.json",
                        auth,
                        "activation_authorization_id",
                        _read_regular_at(
                            incoming_fd, "artifacts/activation-authorization.json"
                        ),
                    ),
                    "artifact_inventory": inventory,
                    "artifact_inventory_sha256": self._inventory_digest(inventory),
                    "sealed_artifact_inventory_sha256": self._inventory_digest(
                        inventory
                    ),
                    "source_repositories": {
                        "ares_head": core.get("ares_head"),
                        "context_governor_head": core.get("context_governor_head"),
                    },
                    "lifecycle_state": CandidateLifecycleState.SEALING.value,
                    "lifecycle_sequence": 1,
                    "publication_state": "PENDING_FINAL_RENAME",
                    "audit_state": "NOT_HANDED_OFF",
                    "activation_authorization_state": "UNAUTHORIZED",
                    "rollback_required": False,
                    "retention_state": "PROTECTED_PENDING_AUDIT",
                    "custody_revision": 1,
                }
                initial_event = {
                    "schema": EVENT_SCHEMA,
                    "canonicalization_version": CANONICALIZATION_VERSION,
                    "sequence": 1,
                    "from": CandidateLifecycleState.CERTIFIED.value,
                    "to": CandidateLifecycleState.SEALING.value,
                    "reason": "durable-publication-precommit",
                    "created_at_unix_ns": time.time_ns(),
                }
                self._checkpoint("initial_lifecycle_event.write")
                events_fd = _open_directory_at(incoming_fd, "events")
                try:
                    _write_atomic_at(
                        events_fd,
                        "00000000000000000001.json",
                        canonical_json(initial_event) + b"\n",
                        0o600,
                    )
                finally:
                    os.close(events_fd)
                self._checkpoint("initial_lifecycle_event.fsync")
                self._checkpoint("initial_custody_snapshot.write")
                _write_atomic_at(
                    incoming_fd, "custody.json", canonical_json(snapshot) + b"\n", 0o600
                )
                self._checkpoint("initial_custody_snapshot.fsync")
                self._checkpoint("initial_custody_snapshot.rename")
                try:
                    artifacts_fd = _open_directory_at(incoming_fd, "artifacts")
                    try:
                        _seal_tree_at(artifacts_fd)
                        os.fchmod(artifacts_fd, 0o500)
                        _fsync(artifacts_fd)
                        self._checkpoint("nested_directory.fsync")
                    finally:
                        os.close(artifacts_fd)
                    events_fd = _open_directory_at(incoming_fd, "events")
                    try:
                        _fsync(events_fd)
                    finally:
                        os.close(events_fd)
                    _fsync(incoming_fd)
                finally:
                    pass
                self._checkpoint("incoming_candidate_directory.fsync")
                self._checkpoint("final_rename")
                # ``candidates_fd`` is held by the store lock from the start
                # of publication; do not reopen the mutable store path here.
                incoming_parent_fd = _open_directory_at(candidates_fd, ".incoming")
                try:
                    # Atomic same-filesystem commit: neither authority
                    # directory is re-resolved by pathname.
                    os.rename(
                        incoming_name,
                        sealed_id,
                        src_dir_fd=incoming_parent_fd,
                        dst_dir_fd=candidates_fd,
                    )
                finally:
                    os.close(incoming_parent_fd)
                _fsync(candidates_fd)
                self._checkpoint("candidates_parent.fsync")
                # The descriptor opened while the directory was still
                # ``.incoming/<uuid>`` now names the committed candidate.
                # Keep using that authority rather than reopening the final
                # pathname after the commit boundary.
                final_fd = os.dup(incoming_fd)
                try:
                    with self._candidate_lock(final_fd):
                        current = self._validate_snapshot_fd(final_fd, sealed_id)
                        current["publication_state"] = "PUBLISHED"
                        current = self._write_event_and_snapshot_fd(
                            final_fd,
                            current,
                            CandidateLifecycleState.SEALING,
                            CandidateLifecycleState.SEALED,
                            "durable-rename-committed",
                        )
                    custody_digest = sha256_bytes(
                        _read_regular_at(final_fd, "custody.json")
                    )
                finally:
                    os.close(final_fd)
                os.close(incoming_fd)
                os.close(source_fd)
                return PublicationResult(
                    "PUBLISHED",
                    sealed_id,
                    final,
                    custody_digest,
                    int(current["lifecycle_sequence"]),
                )
            except Exception:
                # Incomplete directories are deliberately retained only below
                # .incoming for forensic/recovery classification, never sealed.
                os.close(incoming_fd)
                os.close(source_fd)
                raise

    def verify(self, sealed_candidate_id: str) -> dict[str, Any]:
        with self._store_lock(exclusive=False) as candidates_fd:
            candidate_fd = self._candidate_fd(candidates_fd, sealed_candidate_id)
            try:
                return self._validate_snapshot_fd(candidate_fd, sealed_candidate_id)
            finally:
                os.close(candidate_fd)

    def _list_unlocked(
        self, candidates_fd: int, *, include_sealing: bool = False
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for name in sorted(os.listdir(candidates_fd)):
            if name.startswith("."):
                continue
            try:
                info = os.stat(name, dir_fd=candidates_fd, follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode):
                    continue
                candidate_fd = self._candidate_fd(candidates_fd, name)
                try:
                    snapshot = self._validate_snapshot_fd(candidate_fd, name)
                finally:
                    os.close(candidate_fd)
            except CandidateStoreError:
                continue
            if (
                include_sealing
                or snapshot["lifecycle_state"] != CandidateLifecycleState.SEALING.value
            ):
                result.append(snapshot)
        return result

    def list(self) -> list[dict[str, Any]]:
        with self._store_lock(exclusive=False) as candidates_fd:
            return self._list_unlocked(candidates_fd)

    def _validate_tombstone_fd(
        self,
        tombstones_fd: int,
        candidate_fd: int,
        candidate_id: str,
        quarantine_name: str,
    ) -> dict[str, Any]:
        raw = _read_regular_at(tombstones_fd, f"{candidate_id}.json")
        try:
            tombstone = _strict_json_object(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CandidateStoreError("CUSTODY_CORRUPT", "malformed tombstone") from exc
        if (
            set(tombstone)
            != {
                "schema",
                "canonicalization_version",
                "candidate_id",
                "sealed_candidate_id",
                "archive_sha256",
                "gc_approval_id",
                "gc_approval",
                "gc_approval_sha256",
                "pre_gc_lifecycle_state",
                "pre_gc_lifecycle_sequence",
                "quarantine_name",
                "reason",
                "removed_at_unix_ns",
                "tombstone_id",
            }
            or tombstone.get("schema") != TOMBSTONE_SCHEMA
            or tombstone.get("canonicalization_version") != CANONICALIZATION_VERSION
            or tombstone.get("tombstone_id") != _object_id(tombstone, "tombstone_id")
        ):
            raise CandidateStoreError("CUSTODY_CORRUPT", "tombstone identity")
        snapshot = self._validate_snapshot_fd(candidate_fd, candidate_id)
        approval_raw = _read_regular_at(candidate_fd, "gc-approval.json")
        approval = self._read_json_at(candidate_fd, "gc-approval.json")
        if (
            tombstone.get("candidate_id") != snapshot["candidate_id"]
            or tombstone.get("sealed_candidate_id") != candidate_id
            or tombstone.get("archive_sha256") != snapshot["archive_sha256"]
            or tombstone.get("pre_gc_lifecycle_state") != snapshot["lifecycle_state"]
            or tombstone.get("pre_gc_lifecycle_sequence")
            != snapshot["lifecycle_sequence"]
            or tombstone.get("quarantine_name") != quarantine_name
            or tombstone.get("gc_approval_id") != approval.get("gc_approval_id")
            or tombstone.get("gc_approval_sha256") != sha256_bytes(approval_raw)
            or tombstone.get("gc_approval") != approval
            or is_gc_protected(CandidateLifecycleState(snapshot["lifecycle_state"]))
        ):
            raise CandidateStoreError("CUSTODY_CORRUPT", "tombstone binding")
        return tombstone

    def recover(self) -> list[dict[str, Any]]:
        """Fresh-process recovery; no worktree or scratch lookup is permitted."""
        with self._store_lock() as candidates_fd:
            quarantine_fd = _open_directory_at(candidates_fd, ".gc-quarantine")
            tombstones_fd = _open_directory_at(candidates_fd, "tombstones")
            try:
                for name in sorted(os.listdir(quarantine_fd)):
                    candidate_id, separator, _nonce = name.partition("-")
                    if (
                        not separator
                        or len(candidate_id) != 64
                        or any(char not in "0123456789abcdef" for char in candidate_id)
                    ):
                        raise CandidateStoreError(
                            "CUSTODY_CORRUPT", "GC quarantine name"
                        )
                    try:
                        tombstone_fd = _open_regular_at(
                            tombstones_fd, f"{candidate_id}.json"
                        )
                    except CandidateStoreError as exc:
                        if exc.code != "CUSTODY_UNAVAILABLE":
                            raise
                        tombstone_fd = -1
                    if tombstone_fd >= 0:
                        os.close(tombstone_fd)
                        quarantine_candidate_fd = _open_directory_at(
                            quarantine_fd, name
                        )
                        try:
                            self._validate_tombstone_fd(
                                tombstones_fd,
                                quarantine_candidate_fd,
                                candidate_id,
                                name,
                            )
                        except CandidateStoreError:
                            # A file named like a tombstone is never deletion
                            # authority.  Restore custody for a later explicit
                            # disposition instead of guessing a destructive act.
                            os.rename(
                                name,
                                candidate_id,
                                src_dir_fd=quarantine_fd,
                                dst_dir_fd=candidates_fd,
                            )
                            _fsync(quarantine_fd)
                            _fsync(candidates_fd)
                        else:
                            _remove_tree_at(quarantine_fd, name)
                        finally:
                            os.close(quarantine_candidate_fd)
                    else:
                        os.rename(
                            name,
                            candidate_id,
                            src_dir_fd=quarantine_fd,
                            dst_dir_fd=candidates_fd,
                        )
                        _fsync(quarantine_fd)
                        _fsync(candidates_fd)
            finally:
                os.close(tombstones_fd)
                os.close(quarantine_fd)
            recovered = self._list_unlocked(candidates_fd, include_sealing=True)
            for snapshot in recovered:
                if snapshot["lifecycle_state"] == CandidateLifecycleState.SEALING.value:
                    candidate_fd = self._candidate_fd(
                        candidates_fd, snapshot["sealed_candidate_id"]
                    )
                    try:
                        with self._candidate_lock(candidate_fd):
                            current = self._validate_snapshot_fd(
                                candidate_fd, snapshot["sealed_candidate_id"]
                            )
                            if (
                                current["lifecycle_state"]
                                == CandidateLifecycleState.SEALING.value
                                and current["publication_state"]
                                == "PENDING_FINAL_RENAME"
                            ):
                                current["publication_state"] = "PUBLISHED"
                                self._write_event_and_snapshot_fd(
                                    candidate_fd,
                                    current,
                                    CandidateLifecycleState.SEALING,
                                    CandidateLifecycleState.SEALED,
                                    "durable-rename-recovery",
                                )
                    finally:
                        os.close(candidate_fd)
                    continue
                if (
                    snapshot["lifecycle_state"]
                    == CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS.value
                ):
                    candidate_fd = self._candidate_fd(
                        candidates_fd, snapshot["sealed_candidate_id"]
                    )
                    try:
                        with self._candidate_lock(candidate_fd):
                            current = self._validate_snapshot_fd(
                                candidate_fd, snapshot["sealed_candidate_id"]
                            )
                            lease_valid = False
                            try:
                                self._validate_audit_lease_fd(candidate_fd, current)
                                lease_valid = True
                            except CandidateStoreError:
                                # A lost, malformed, stale, substituted, or
                                # cross-candidate lease has no outcome authority.
                                lease_valid = False
                            if not lease_valid or not self._audit_is_held(candidate_fd):
                                current["audit_state"] = "BLOCKED_STALE_LEASE"
                                self._write_event_and_snapshot_fd(
                                    candidate_fd,
                                    current,
                                    CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS,
                                    CandidateLifecycleState.AUDIT_BLOCKED,
                                    "stale-audit-lease-recovery",
                                )
                    finally:
                        os.close(candidate_fd)
            return self._list_unlocked(candidates_fd)

    def issue_handoff(self, sealed_candidate_id: str) -> dict[str, Any]:
        with self._store_lock(exclusive=False) as candidates_fd:
            candidate_fd = self._candidate_fd(candidates_fd, sealed_candidate_id)
            try:
                with self._candidate_lock(candidate_fd):
                    snapshot = self._validate_snapshot_fd(
                        candidate_fd, sealed_candidate_id
                    )
                    if (
                        snapshot["lifecycle_state"]
                        != CandidateLifecycleState.SEALED.value
                    ):
                        raise CandidateStoreError(
                            "CUSTODY_UNAVAILABLE", "candidate is not sealed"
                        )
                    handoffs_fd = _open_directory_at(
                        candidate_fd, "handoffs", create=True
                    )
                    os.close(handoffs_fd)
                    custody_raw = _read_regular_at(candidate_fd, "custody.json")
                    subject = self._audit_subject(snapshot, sha256_bytes(custody_raw))
                    subject_raw = canonical_json(subject) + b"\n"
                    self._checkpoint("audit_subject.write")
                    _write_atomic_at(
                        candidate_fd, "audit-subject.json", subject_raw, 0o400
                    )
                    self._checkpoint("audit_subject.fsync")
                    subject_ref = {
                        "relative_path": "audit-subject.json",
                        "sha256": sha256_bytes(subject_raw),
                        "id": subject["audit_subject_id"],
                    }
                    handoff = {
                        "schema": HANDOFF_SCHEMA,
                        "canonicalization_version": CANONICALIZATION_VERSION,
                        "sealed_candidate_id": sealed_candidate_id,
                        "candidate_id": snapshot["candidate_id"],
                        "certification_set_id": snapshot["certification_set_id"],
                        "candidate_root": str(
                            self._candidate_root(sealed_candidate_id)
                        ),
                        "custody_relative_path": "custody.json",
                        "audit_subject_relative_path": "audit-subject.json",
                        "audit_subject_id": subject["audit_subject_id"],
                        "audit_subject_sha256": subject_ref["sha256"],
                        "publication_custody_sha256": subject[
                            "publication_custody_sha256"
                        ],
                        "sealed_artifact_inventory_sha256": subject[
                            "sealed_artifact_inventory_sha256"
                        ],
                        "archive_relative_path": snapshot["archive_relative_path"],
                        "archive_sha256": snapshot["archive_sha256"],
                        "publication_lifecycle_sequence": subject[
                            "publication_lifecycle_sequence"
                        ],
                        "allowed_lifecycle_mutations": subject[
                            "allowed_lifecycle_mutations"
                        ],
                        "activation_authorization_state": "UNAUTHORIZED",
                    }
                    handoff["hostile_audit_handoff_id"] = _object_id(
                        handoff, "hostile_audit_handoff_id"
                    )
                    relative = f"handoffs/{handoff['hostile_audit_handoff_id']}.json"
                    self._checkpoint("handoff.write")
                    _write_atomic_at(
                        candidate_fd, relative, canonical_json(handoff) + b"\n", 0o600
                    )
                    self._checkpoint("handoff.fsync")
                    snapshot = dict(snapshot)
                    snapshot["audit_subject"] = subject_ref
                    snapshot["audit_handoff"] = {
                        "relative_path": relative,
                        "sha256": sha256_bytes(
                            _read_regular_at(candidate_fd, relative)
                        ),
                        "id": handoff["hostile_audit_handoff_id"],
                    }
                    snapshot["audit_state"] = "AWAITING_HOSTILE_AUDIT"
                    self._checkpoint("handoff.persistence")
                    self._write_event_and_snapshot_fd(
                        candidate_fd,
                        snapshot,
                        CandidateLifecycleState.SEALED,
                        CandidateLifecycleState.AWAITING_HOSTILE_AUDIT,
                        "hostile-audit-handoff-issued",
                    )
                    return handoff
            finally:
                os.close(candidate_fd)

    def start_audit(self, sealed_candidate_id: str) -> AuditLease:
        with self._store_lock(exclusive=False) as candidates_fd:
            candidate_fd = self._candidate_fd(candidates_fd, sealed_candidate_id)
            fd = self._open_audit_lock(candidate_fd)
            audit_lock = os.fstat(fd)
            lease = {
                "schema": "AresCandidateAuditLeaseV1",
                "canonicalization_version": CANONICALIZATION_VERSION,
                "lease_id": uuid.uuid4().hex,
                "nonce": uuid.uuid4().hex,
                "candidate_id": "",
                "sealed_candidate_id": sealed_candidate_id,
                "audit_subject_id": "",
                "audit_subject_sha256": "",
                "hostile_audit_handoff_id": "",
                "hostile_audit_handoff_sha256": "",
                "lifecycle_sequence": 0,
                "audit_lock_identity": {
                    "dev": audit_lock.st_dev,
                    "ino": audit_lock.st_ino,
                },
                "created_at_unix_ns": time.time_ns(),
            }
            try:
                with self._candidate_lock(candidate_fd):
                    snapshot = self._validate_snapshot_fd(
                        candidate_fd, sealed_candidate_id
                    )
                    if (
                        snapshot["lifecycle_state"]
                        != CandidateLifecycleState.AWAITING_HOSTILE_AUDIT.value
                    ):
                        raise CandidateStoreError("AUDIT_NOT_ELIGIBLE")
                    if not snapshot.get("audit_subject") or not snapshot.get(
                        "audit_handoff"
                    ):
                        raise CandidateStoreError(
                            "AUDIT_NOT_ELIGIBLE", "missing audit binding"
                        )
                    lease.update({
                        "candidate_id": snapshot["candidate_id"],
                        "audit_subject_id": snapshot["audit_subject"]["id"],
                        "audit_subject_sha256": snapshot["audit_subject"]["sha256"],
                        "hostile_audit_handoff_id": snapshot["audit_handoff"]["id"],
                        "hostile_audit_handoff_sha256": snapshot["audit_handoff"][
                            "sha256"
                        ],
                        "lifecycle_sequence": snapshot["lifecycle_sequence"],
                    })
                    lease["lease_id"] = _object_id(lease, "lease_id")
                    raw = canonical_json(lease) + b"\n"
                    self._checkpoint("audit_state.write")
                    _write_atomic_at(
                        candidate_fd,
                        "audit-lease.json",
                        raw,
                        0o600,
                    )
                    self._checkpoint("audit_lease.fsync")
                    snapshot["audit_state"] = "HOSTILE_AUDIT_IN_PROGRESS"
                    self._write_event_and_snapshot_fd(
                        candidate_fd,
                        snapshot,
                        CandidateLifecycleState.AWAITING_HOSTILE_AUDIT,
                        CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS,
                        "hostile-audit-started",
                    )
            except Exception:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                os.close(candidate_fd)
                raise
            # The candidate descriptor must stay pinned for the auditor's
            # lease.  It prevents a later pathname exchange from redirecting
            # result recording.
            return AuditLease(
                sealed_candidate_id, lease, sha256_bytes(raw), fd, candidate_fd
            )

    def _validate_audit_lease_fd(
        self,
        candidate_fd: int,
        snapshot: dict[str, Any],
        expected: AuditLease | None = None,
    ) -> dict[str, Any]:
        raw = _read_regular_at(candidate_fd, "audit-lease.json")
        try:
            lease = _strict_json_object(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CandidateStoreError(
                "CUSTODY_CORRUPT", "malformed audit lease"
            ) from exc
        if (
            set(lease)
            != {
                "schema",
                "canonicalization_version",
                "lease_id",
                "nonce",
                "candidate_id",
                "sealed_candidate_id",
                "audit_subject_id",
                "audit_subject_sha256",
                "hostile_audit_handoff_id",
                "hostile_audit_handoff_sha256",
                "lifecycle_sequence",
                "audit_lock_identity",
                "created_at_unix_ns",
            }
            or lease.get("schema") != "AresCandidateAuditLeaseV1"
        ):
            raise CandidateStoreError("CUSTODY_CORRUPT", "audit lease schema")
        if lease.get("canonicalization_version") != CANONICALIZATION_VERSION:
            raise CandidateStoreError("CUSTODY_CORRUPT", "audit lease version")
        if lease.get("lease_id") != _object_id(lease, "lease_id"):
            raise CandidateStoreError("CUSTODY_CORRUPT", "audit lease identity")
        binding = snapshot.get("audit_subject"), snapshot.get("audit_handoff")
        if (
            not all(binding)
            or lease.get("candidate_id") != snapshot["candidate_id"]
            or lease.get("sealed_candidate_id") != snapshot["sealed_candidate_id"]
            or lease.get("audit_subject_id") != snapshot["audit_subject"]["id"]
            or lease.get("audit_subject_sha256") != snapshot["audit_subject"]["sha256"]
            or lease.get("hostile_audit_handoff_id") != snapshot["audit_handoff"]["id"]
            or lease.get("hostile_audit_handoff_sha256")
            != snapshot["audit_handoff"]["sha256"]
            or lease.get("lifecycle_sequence")
            != int(snapshot["lifecycle_sequence"]) - 1
        ):
            raise CandidateStoreError("CUSTODY_CORRUPT", "audit lease binding")
        lock_fd = _open_regular_at(candidate_fd, "locks/audit.lock")
        try:
            lock = os.fstat(lock_fd)
        finally:
            os.close(lock_fd)
        if lease.get("audit_lock_identity") != {"dev": lock.st_dev, "ino": lock.st_ino}:
            raise CandidateStoreError("CUSTODY_CORRUPT", "audit lease lock identity")
        if expected is not None and (
            lease != expected.payload or sha256_bytes(raw) != expected.payload_sha256
        ):
            raise CandidateStoreError("CUSTODY_CORRUPT", "substituted audit lease")
        return lease

    def record_audit_result(
        self, lease: AuditLease, *, passed: bool, reason: str
    ) -> dict[str, Any]:
        # ``bool("FAILED")`` is true.  Verdict values are authority-bearing,
        # so they must be the built-in bool and are never coerced.
        if type(passed) is not bool:
            raise CandidateStoreError("INVALID_AUDIT_VERDICT")
        with self._store_lock(exclusive=False):
            if lease._fd < 0:
                raise CandidateStoreError("AUDIT_LEASE_REQUIRED")
            try:
                with self._candidate_lock(lease._candidate_fd):
                    snapshot = self._validate_snapshot_fd(
                        lease._candidate_fd, lease.sealed_candidate_id
                    )
                    if (
                        snapshot["lifecycle_state"]
                        != CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS.value
                    ):
                        raise CandidateStoreError("AUDIT_NOT_IN_PROGRESS")
                    self._validate_audit_lease_fd(lease._candidate_fd, snapshot, lease)
                    snapshot["audit_state"] = (
                        "AUDIT_PASSED" if passed else "AUDIT_FAILED"
                    )
                    result = self._write_event_and_snapshot_fd(
                        lease._candidate_fd,
                        snapshot,
                        CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS,
                        CandidateLifecycleState.AUDIT_PASSED
                        if passed
                        else CandidateLifecycleState.AUDIT_FAILED,
                        reason,
                    )
                    # A durable result now exists; lease retirement is safe.
                    os.unlink("audit-lease.json", dir_fd=lease._candidate_fd)
                    _fsync(lease._candidate_fd)
                    return result
            finally:
                lease.close()

    def _validate_grant_runtime_binding_fd(
        self,
        candidate_fd: int,
        snapshot: dict[str, Any],
        grant: ActivationGrant,
    ) -> None:
        """Bind authorization to the release manifest sealed in candidate core."""

        core_ref = snapshot["candidate_core"]
        core_raw = _read_regular_at(candidate_fd, core_ref["relative_path"])
        if sha256_bytes(core_raw) != core_ref["sha256"]:
            raise CandidateStoreError("ACTIVATION_GRANT_MISMATCH")
        core = self._read_json_at(candidate_fd, core_ref["relative_path"])
        expected = {
            "release_manifest_sha256": grant.release_manifest_sha256,
            "runtime_tree_sha256": grant.runtime_tree_sha256,
        }
        if any(core.get(field) != value for field, value in expected.items()):
            raise CandidateStoreError("ACTIVATION_GRANT_MISMATCH")
        manifest_path = core.get("release_manifest_path")
        payload_files = core.get("payload_files")
        if not isinstance(manifest_path, str) or not isinstance(payload_files, list):
            raise CandidateStoreError("ACTIVATION_GRANT_MISMATCH")
        manifest_entries = [
            entry
            for entry in payload_files
            if isinstance(entry, dict) and entry.get("path") == manifest_path
        ]
        if len(manifest_entries) != 1 or (
            manifest_entries[0].get("sha256") != grant.release_manifest_sha256
        ):
            raise CandidateStoreError("ACTIVATION_GRANT_MISMATCH")

    def authorize_activation(
        self,
        sealed_candidate_id: str,
        *,
        grant: ActivationGrant,
        reason: str = "explicit-governed-activation-authorization",
    ) -> dict[str, Any]:
        """Record the only activation-authorizing transition.

        This does not activate a runtime.  It merely makes a hostile-audit
        passed candidate eligible for a separately controlled activation path.
        Every other candidate artifact remains explicitly non-authorizing.
        """
        with self._store_lock(exclusive=False) as candidates_fd:
            candidate_fd = self._candidate_fd(candidates_fd, sealed_candidate_id)
            try:
                with self._candidate_lock(candidate_fd):
                    snapshot = self._validate_snapshot_fd(
                        candidate_fd, sealed_candidate_id
                    )
                    if (
                        snapshot["lifecycle_state"]
                        != CandidateLifecycleState.AUDIT_PASSED.value
                        or snapshot["audit_state"] != "AUDIT_PASSED"
                    ):
                        raise CandidateStoreError("MISSING_AUTHORIZATION_EVIDENCE")
                    if snapshot["activation_authorization_state"] != UNAUTHORIZED:
                        raise CandidateStoreError("AuthorizationStateContradiction")
                    self._validate_authorization_prerequisites_fd(
                        candidate_fd, snapshot
                    )
                    if not isinstance(grant, ActivationGrant):
                        raise CandidateStoreError("ACTIVATION_GRANT_REQUIRED")
                    events_fd = _open_directory_at(candidate_fd, "events")
                    try:
                        audit_event_raw = _read_regular_at(
                            events_fd,
                            f"{int(snapshot['lifecycle_sequence']):020d}.json",
                        )
                    finally:
                        os.close(events_fd)
                    expected = {
                        "candidate_id": snapshot["candidate_id"],
                        "certification_set_id": snapshot["certification_set_id"],
                        "sealed_candidate_id": snapshot["sealed_candidate_id"],
                        "audit_subject_id": snapshot["audit_subject"]["id"],
                        "audit_subject_sha256": snapshot["audit_subject"]["sha256"],
                        "audit_result_sha256": sha256_bytes(audit_event_raw),
                        "archive_sha256": snapshot["archive_sha256"],
                        "candidate_core_sha256": snapshot["candidate_core"]["sha256"],
                        "sealed_manifest_sha256": snapshot["sealed_candidate_manifest"][
                            "sha256"
                        ],
                        "custody_event_sequence": int(snapshot["lifecycle_sequence"])
                        + 1,
                    }
                    if any(
                        getattr(grant, key) != value for key, value in expected.items()
                    ):
                        raise CandidateStoreError("ACTIVATION_GRANT_MISMATCH")
                    self._validate_grant_runtime_binding_fd(
                        candidate_fd, snapshot, grant
                    )
                    snapshot["activation_authorization_state"] = AUTHORIZED
                    snapshot["activation_grant"] = grant.to_dict()
                    return self._write_event_and_snapshot_fd(
                        candidate_fd,
                        snapshot,
                        CandidateLifecycleState.AUDIT_PASSED,
                        CandidateLifecycleState.AWAITING_ACTIVATION,
                        reason,
                    )
            finally:
                os.close(candidate_fd)

    def read_activation_grant(self, sealed_candidate_id: str) -> ActivationGrant:
        """Return the persisted exact grant without granting new authority."""

        snapshot = self.verify(sealed_candidate_id)
        grant = snapshot.get("activation_grant")
        if grant is None:
            raise CandidateStoreError("ACTIVATION_GRANT_MISSING")
        try:
            return ActivationGrant.parse(canonical_json(grant) + b"\n")
        except AresRuntimeError as exc:
            raise CandidateStoreError("ACTIVATION_GRANT_CORRUPT", str(exc)) from exc

    @staticmethod
    def _require_activation_grant_id(snapshot: dict[str, Any], grant_id: str) -> None:
        grant = snapshot.get("activation_grant")
        if not isinstance(grant, dict) or grant.get("grant_id") != grant_id:
            raise CandidateStoreError("ACTIVATION_GRANT_MISMATCH")

    def record_activation_success(
        self, sealed_candidate_id: str, *, grant_id: str, reason: str
    ) -> dict[str, Any]:
        """Record success only after the activator's mandatory live certification."""

        with self._store_lock(exclusive=False) as candidates_fd:
            candidate_fd = self._candidate_fd(candidates_fd, sealed_candidate_id)
            try:
                with self._candidate_lock(candidate_fd):
                    snapshot = self._validate_snapshot_fd(
                        candidate_fd, sealed_candidate_id
                    )
                    if (
                        snapshot["lifecycle_state"]
                        != CandidateLifecycleState.AWAITING_ACTIVATION.value
                    ):
                        raise CandidateStoreError("ACTIVATION_NOT_AWAITING")
                    self._require_activation_grant_id(snapshot, grant_id)
                    return self._write_event_and_snapshot_fd(
                        candidate_fd,
                        snapshot,
                        CandidateLifecycleState.AWAITING_ACTIVATION,
                        CandidateLifecycleState.ACTIVE,
                        reason,
                    )
            finally:
                os.close(candidate_fd)

    def record_rollback_required(
        self, sealed_candidate_id: str, *, grant_id: str, reason: str
    ) -> dict[str, Any]:
        """Record a post-commit activation failure before pointer rollback."""

        with self._store_lock(exclusive=False) as candidates_fd:
            candidate_fd = self._candidate_fd(candidates_fd, sealed_candidate_id)
            try:
                with self._candidate_lock(candidate_fd):
                    snapshot = self._validate_snapshot_fd(
                        candidate_fd, sealed_candidate_id
                    )
                    current = CandidateLifecycleState(snapshot["lifecycle_state"])
                    if current not in {
                        CandidateLifecycleState.AWAITING_ACTIVATION,
                        CandidateLifecycleState.ACTIVE,
                    }:
                        raise CandidateStoreError("ROLLBACK_NOT_ELIGIBLE")
                    self._require_activation_grant_id(snapshot, grant_id)
                    snapshot["rollback_required"] = True
                    return self._write_event_and_snapshot_fd(
                        candidate_fd,
                        snapshot,
                        current,
                        CandidateLifecycleState.ROLLBACK_REQUIRED,
                        reason,
                    )
            finally:
                os.close(candidate_fd)

    def record_rollback_success(
        self, sealed_candidate_id: str, *, grant_id: str, reason: str
    ) -> dict[str, Any]:
        """Record rollback only after the previous runtime has been verified."""

        with self._store_lock(exclusive=False) as candidates_fd:
            candidate_fd = self._candidate_fd(candidates_fd, sealed_candidate_id)
            try:
                with self._candidate_lock(candidate_fd):
                    snapshot = self._validate_snapshot_fd(
                        candidate_fd, sealed_candidate_id
                    )
                    if (
                        snapshot["lifecycle_state"]
                        != CandidateLifecycleState.ROLLBACK_REQUIRED.value
                    ):
                        raise CandidateStoreError("ROLLBACK_NOT_REQUIRED")
                    self._require_activation_grant_id(snapshot, grant_id)
                    snapshot["rollback_required"] = False
                    snapshot["activation_authorization_state"] = UNAUTHORIZED
                    return self._write_event_and_snapshot_fd(
                        candidate_fd,
                        snapshot,
                        CandidateLifecycleState.ROLLBACK_REQUIRED,
                        CandidateLifecycleState.ROLLED_BACK,
                        reason,
                    )
            finally:
                os.close(candidate_fd)

    def reject(
        self, sealed_candidate_id: str, reason: str = "explicit-rejection"
    ) -> dict[str, Any]:
        with self._store_lock(exclusive=False) as candidates_fd:
            candidate_fd = self._candidate_fd(candidates_fd, sealed_candidate_id)
            try:
                with self._candidate_lock(candidate_fd):
                    snapshot = self._validate_snapshot_fd(
                        candidate_fd, sealed_candidate_id
                    )
                    if self._audit_is_held(candidate_fd):
                        raise CandidateStoreError("AUDIT_LOCKED")
                    old = CandidateLifecycleState(snapshot["lifecycle_state"])
                    return self._write_event_and_snapshot_fd(
                        candidate_fd,
                        snapshot,
                        old,
                        CandidateLifecycleState.REJECTED,
                        reason,
                    )
            finally:
                os.close(candidate_fd)

    def gc_plan(
        self, sealed_candidate_id: str, approval: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        snapshot = self.verify(sealed_candidate_id)
        state = CandidateLifecycleState(snapshot["lifecycle_state"])
        if is_gc_protected(state):
            raise CandidateStoreError("GC_PROTECTED_LIFECYCLE", state.value)
        if (
            not isinstance(approval, dict)
            or set(approval)
            != {
                "schema",
                "sealed_candidate_id",
                "archive_sha256",
                "approved_at_unix_ns",
                "gc_approval_id",
            }
            or approval.get("schema") != GC_APPROVAL_SCHEMA
            or approval.get("sealed_candidate_id") != sealed_candidate_id
        ):
            raise CandidateStoreError("GC_APPROVAL_REQUIRED")
        if approval.get("gc_approval_id") != _object_id(approval, "gc_approval_id"):
            raise CandidateStoreError("GC_APPROVAL_INVALID")
        return {
            "sealed_candidate_id": sealed_candidate_id,
            "eligible": True,
            "approval": approval,
        }

    def gc(self, sealed_candidate_id: str, approval: dict[str, Any]) -> dict[str, Any]:
        plan = self.gc_plan(sealed_candidate_id, approval)
        with self._store_lock() as candidates_fd:
            candidate_fd = self._candidate_fd(candidates_fd, sealed_candidate_id)
            try:
                snapshot = self._validate_snapshot_fd(candidate_fd, sealed_candidate_id)
                self._checkpoint("gc_approval.write")
                _write_atomic_at(
                    candidate_fd,
                    "gc-approval.json",
                    canonical_json(plan["approval"]) + b"\n",
                    0o600,
                )
                _fsync(candidate_fd)
                self._checkpoint("gc_approval.fsync")
            finally:
                os.close(candidate_fd)
            quarantine_name = f"{sealed_candidate_id}-{uuid.uuid4().hex}"
            self._checkpoint("gc_quarantine.rename")
            quarantine_fd = _open_directory_at(candidates_fd, ".gc-quarantine")
            try:
                os.rename(
                    sealed_candidate_id,
                    quarantine_name,
                    src_dir_fd=candidates_fd,
                    dst_dir_fd=quarantine_fd,
                )
                _fsync(candidates_fd)
                _fsync(quarantine_fd)
            finally:
                os.close(quarantine_fd)
            self._checkpoint("gc_quarantine.directory_fsync")
            tombstone = {
                "schema": TOMBSTONE_SCHEMA,
                "canonicalization_version": CANONICALIZATION_VERSION,
                "candidate_id": snapshot["candidate_id"],
                "sealed_candidate_id": sealed_candidate_id,
                "archive_sha256": snapshot["archive_sha256"],
                "gc_approval_id": approval["gc_approval_id"],
                "gc_approval": plan["approval"],
                "gc_approval_sha256": sha256_bytes(
                    canonical_json(plan["approval"]) + b"\n"
                ),
                "pre_gc_lifecycle_state": snapshot["lifecycle_state"],
                "pre_gc_lifecycle_sequence": snapshot["lifecycle_sequence"],
                "quarantine_name": quarantine_name,
                "reason": "explicit-candidate-gc-approval",
                "removed_at_unix_ns": time.time_ns(),
            }
            tombstone["tombstone_id"] = _object_id(tombstone, "tombstone_id")
            self._checkpoint("gc_tombstone.write")
            tombstones_fd = _open_directory_at(candidates_fd, "tombstones")
            try:
                _write_atomic_at(
                    tombstones_fd,
                    f"{sealed_candidate_id}.json",
                    canonical_json(tombstone) + b"\n",
                    0o600,
                )
            finally:
                os.close(tombstones_fd)
            self._checkpoint("gc_tombstone.fsync")
            self._checkpoint("gc_tombstone.directory_fsync")
            self._checkpoint("gc_final_removal")
            # Quarantine is now below a held store descriptor.  A recursive
            # deleter based on Path.rglob()/shutil.rmtree() would re-resolve
            # mutable names and can be redirected; remove strictly by dir_fd.
            quarantine_fd = _open_directory_at(candidates_fd, ".gc-quarantine")
            try:
                _remove_tree_at(quarantine_fd, quarantine_name)
            finally:
                os.close(quarantine_fd)
            return tombstone
