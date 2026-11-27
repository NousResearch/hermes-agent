"""Owner-only append-only receipt storage for the frozen Phase 1 seam."""

from __future__ import annotations

import errno
import fcntl
import os
import secrets
import stat
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import (
    MAX_DIRECTORY_ENTRIES_PER_KEY,
    MAX_RECORD_BYTES,
    MAX_RECORDS_PER_KEY,
    MAX_RECEIPT_BYTES,
    ContractValidationError,
    JSONValue,
    Operation,
    build_record,
    canonical_bytes,
    parse_receipt_bytes,
    parse_record_bytes,
    record_filename,
    sealed_receipt,
    transient_conflict,
    validate_record_chain,
)

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC
_WRITE_FLAGS = os.O_WRONLY | _O_NOFOLLOW | _O_CLOEXEC
_LOCK_FLAGS = os.O_RDWR | _O_NOFOLLOW | _O_CLOEXEC
_EUID = os.geteuid()
_PUBLICATION_STAGE_PREFIX = ".phase1-stage-"

HERMES_CHECKOUT = Path(__file__).resolve().parents[1]
OLYMPUS_CHECKOUT = Path("/Users/macmini/Hermes-Handoff/olympus-engine")


class RootSafetyError(RuntimeError):
    """A supplied root is outside the owner-controlled test profile."""

    def __init__(self, reason_code: str) -> None:
        if reason_code not in {
            "UNSAFE_RECEIPT_ROOT",
            "ROOTS_OVERLAP",
            "UNSAFE_EVIDENCE_ROOT",
            "UNSAFE_REPOSITORY_ROOT",
        }:
            raise ValueError("invalid root safety reason")
        self.reason_code = reason_code
        super().__init__(reason_code)


class PersistenceError(RuntimeError):
    """A no-retry storage operation failed before safe completion."""


class StoreIndeterminate(RuntimeError):
    """Existing storage cannot safely prove whether invocation occurred."""

    def __init__(self, operation: Operation) -> None:
        self.idempotency_key_digest = operation.idempotency_key_digest
        self.operation_digest = operation.operation_digest
        super().__init__("unclassifiable existing Phase 1 key")


class StorePreInvokeUnavailable(PersistenceError):
    """A clean pre-invocation prefix could not publish a sealed rejection."""

    def __init__(self, operation: Operation) -> None:
        self.idempotency_key_digest = operation.idempotency_key_digest
        self.operation_digest = operation.operation_digest
        super().__init__("pre-invocation receipt is unavailable")


class StoreIndeterminateUnavailable(PersistenceError):
    """A clean consumed-attempt prefix could not publish a sealed receipt."""

    def __init__(self, operation: Operation) -> None:
        self.idempotency_key_digest = operation.idempotency_key_digest
        self.operation_digest = operation.operation_digest
        super().__init__("indeterminate receipt is unavailable")


class StoreSealedUnavailable(RuntimeError):
    """Sealing is proven, but the exact receipt cannot be returned safely."""

    def __init__(
        self,
        operation: Operation,
        *,
        receipt_digest: str,
        receipt_state: str,
    ) -> None:
        self.idempotency_key_digest = operation.idempotency_key_digest
        self.operation_digest = operation.operation_digest
        self.receipt_digest = receipt_digest
        self.receipt_state = receipt_state
        super().__init__("sealed Phase 1 receipt is unavailable")


class _FinalReceiptUnavailable(PersistenceError):
    """A validated final record proves sealing without replayable bytes."""

    def __init__(self, *, receipt_digest: str, predecessor_state: str) -> None:
        self.receipt_digest = receipt_digest
        self.predecessor_state = predecessor_state
        super().__init__("final record proves an unavailable receipt")


@dataclass(slots=True)
class _PublicationAttempt:
    """Side-effect state shared with a publication caller during interruption."""

    link_attempted: bool = False
    target_observed: bool = False
    target_parent_fsync_attempted: bool = False
    target_parent_synced: bool = False
    unlink_attempted: bool = False
    stage_absent: bool = False
    stage_parent_fsync_attempted: bool = False
    stage_parent_synced: bool = False
    final_revalidated: bool = False

    @property
    def committed(self) -> bool:
        return self.target_observed and self.target_parent_synced

    @property
    def complete(self) -> bool:
        return (
            self.committed
            and self.stage_absent
            and self.stage_parent_synced
            and self.final_revalidated
        )


@dataclass(frozen=True, slots=True)
class RootCapability:
    name: str
    path: Path
    fd: int
    device: int
    inode: int
    mutable: bool

    @property
    def identity(self) -> tuple[int, int]:
        return (self.device, self.inode)

    def revalidate(self) -> None:
        try:
            held = os.fstat(self.fd)
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise PersistenceError(f"{self.name} root vanished") from exc
        if (
            not stat.S_ISDIR(held.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or (held.st_dev, held.st_ino) != self.identity
            or (current.st_dev, current.st_ino) != self.identity
            or held.st_uid != _EUID
            or current.st_uid != _EUID
        ):
            raise PersistenceError(f"{self.name} root identity changed")
        mode = stat.S_IMODE(current.st_mode)
        if self.mutable:
            if mode != 0o700:
                raise PersistenceError(f"{self.name} root mode changed")
        elif mode & 0o022:
            raise PersistenceError(f"{self.name} root became writable by others")


@dataclass(slots=True)
class RootSet:
    repository: RootCapability
    receipt: RootCapability
    evidence: RootCapability
    hermes: RootCapability
    olympus: RootCapability
    _closed: bool = False

    def __enter__(self) -> "RootSet":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for capability in (
            self.olympus,
            self.hermes,
            self.evidence,
            self.receipt,
            self.repository,
        ):
            try:
                os.close(capability.fd)
            except OSError:
                pass

    def revalidate_all(self) -> None:
        for capability in (
            self.repository,
            self.receipt,
            self.evidence,
            self.hermes,
            self.olympus,
        ):
            capability.revalidate()


def _mode(entry: os.stat_result) -> int:
    return stat.S_IMODE(entry.st_mode)


def _path_components(path: Path) -> list[Path]:
    components: list[Path] = []
    current = path
    while True:
        components.append(current)
        if current.parent == current:
            break
        current = current.parent
    components.reverse()
    return components


def _open_root(
    value: str | Path,
    *,
    name: str,
    mutable: bool,
    reason_code: str,
) -> RootCapability:
    if not isinstance(value, (str, Path)):
        raise RootSafetyError(reason_code)
    raw = os.fspath(value)
    if not os.path.isabs(raw) or os.path.normpath(raw) != raw:
        raise RootSafetyError(reason_code)
    path = Path(raw)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RootSafetyError(reason_code) from exc
    if resolved != path:
        raise RootSafetyError(reason_code)
    try:
        for component in _path_components(path):
            entry = os.lstat(component)
            if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
                raise RootSafetyError(reason_code)
            writable_by_others = bool(_mode(entry) & 0o022)
            protected_sticky_ancestor = (
                component != path
                and entry.st_uid == 0
                and bool(entry.st_mode & stat.S_ISVTX)
            )
            if writable_by_others and not protected_sticky_ancestor:
                raise RootSafetyError(reason_code)
            if entry.st_uid not in {0, _EUID}:
                raise RootSafetyError(reason_code)
        entry = os.lstat(path)
        if entry.st_uid != _EUID:
            raise RootSafetyError(reason_code)
        if mutable:
            if _mode(entry) != 0o700 or not os.access(path, os.W_OK | os.X_OK):
                raise RootSafetyError(reason_code)
        elif _mode(entry) & 0o022:
            raise RootSafetyError(reason_code)
        fd = os.open(path, _DIR_FLAGS)
        held = os.fstat(fd)
    except RootSafetyError:
        raise
    except OSError as exc:
        raise RootSafetyError(reason_code) from exc
    if (
        not stat.S_ISDIR(held.st_mode)
        or held.st_uid != _EUID
        or (held.st_dev, held.st_ino) != (entry.st_dev, entry.st_ino)
    ):
        os.close(fd)
        raise RootSafetyError(reason_code)
    return RootCapability(
        name=name,
        path=path,
        fd=fd,
        device=held.st_dev,
        inode=held.st_ino,
        mutable=mutable,
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        common = Path(os.path.commonpath((left, right)))
    except ValueError:
        return False
    return common == left or common == right


def _lexical_root_path(value: str | Path) -> Path | None:
    """Return an absolute normalized path without consulting the filesystem."""

    if not isinstance(value, (str, Path)):
        return None
    raw = os.fspath(value)
    if not os.path.isabs(raw):
        return None
    return Path(os.path.normpath(raw))


def _root_path_representations(value: str | Path) -> tuple[Path, ...]:
    """Return lexical and best-effort resolved paths for reason classification."""

    lexical = _lexical_root_path(value)
    if lexical is None:
        return ()
    representations = [lexical]
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError):
        return tuple(representations)
    if resolved != lexical:
        representations.append(resolved)
    return tuple(representations)


def open_root_set(
    *,
    repository_root: str | Path,
    receipt_root: str | Path,
    evidence_root: str | Path,
) -> RootSet:
    opened: list[RootCapability] = []
    try:
        # Receipt-root safety has the highest frozen precedence and therefore
        # must be decided before any lower-priority condition.
        receipt = _open_root(
            receipt_root,
            name="receipt",
            mutable=True,
            reason_code="UNSAFE_RECEIPT_ROOT",
        )
        opened.append(receipt)

        # Lexical containment and best-effort strict resolution are used only
        # to classify overlap before a lower-priority safety failure.  These
        # representations never become capabilities; _open_root still applies
        # the stricter exact-path contract before any root is accepted.
        root_representations = [
            _root_path_representations(value)
            for value in (
                receipt_root,
                evidence_root,
                repository_root,
                HERMES_CHECKOUT,
                OLYMPUS_CHECKOUT,
            )
        ]
        for index, left_paths in enumerate(root_representations):
            for right_paths in root_representations[index + 1 :]:
                if any(
                    _paths_overlap(left, right)
                    for left in left_paths
                    for right in right_paths
                ):
                    raise RootSafetyError("ROOTS_OVERLAP")

        evidence: RootCapability | None = None
        repository: RootCapability | None = None
        hermes: RootCapability | None = None
        olympus: RootCapability | None = None
        evidence_unsafe = False
        repository_unsafe = False

        try:
            evidence = _open_root(
                evidence_root,
                name="evidence",
                mutable=True,
                reason_code="UNSAFE_EVIDENCE_ROOT",
            )
            opened.append(evidence)
        except RootSafetyError:
            evidence_unsafe = True

        repository_specs = (
            (repository_root, "repository"),
            (HERMES_CHECKOUT, "hermes"),
            (OLYMPUS_CHECKOUT, "olympus"),
        )
        repository_capabilities: list[RootCapability | None] = []
        for value, name in repository_specs:
            try:
                capability = _open_root(
                    value,
                    name=name,
                    mutable=False,
                    reason_code="UNSAFE_REPOSITORY_ROOT",
                )
                opened.append(capability)
                repository_capabilities.append(capability)
            except RootSafetyError:
                repository_unsafe = True
                repository_capabilities.append(None)
        repository, hermes, olympus = repository_capabilities

        # Physical-identity overlap is available only for roots that passed
        # their individual safety checks.  It still outranks any collected
        # evidence/repository safety failure.
        for index, left in enumerate(opened):
            for right in opened[index + 1 :]:
                if left.identity == right.identity or _paths_overlap(
                    left.path, right.path
                ):
                    raise RootSafetyError("ROOTS_OVERLAP")
        if evidence_unsafe:
            raise RootSafetyError("UNSAFE_EVIDENCE_ROOT")
        if repository_unsafe:
            raise RootSafetyError("UNSAFE_REPOSITORY_ROOT")
        if (
            repository is None
            or evidence is None
            or hermes is None
            or olympus is None
        ):
            raise AssertionError("safe root capability is unavailable")

        roots = RootSet(
            repository=repository,
            receipt=receipt,
            evidence=evidence,
            hermes=hermes,
            olympus=olympus,
        )
        roots.revalidate_all()
        return roots
    except BaseException:
        for capability in reversed(opened):
            try:
                os.close(capability.fd)
            except OSError:
                pass
        raise


def _open_directory(path: Path, accepted_modes: set[int]) -> int:
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != _EUID
            or _mode(before) not in accepted_modes
            or _mode(before) & 0o022
        ):
            raise PersistenceError(f"unsafe directory: {path.name}")
        fd = os.open(path, _DIR_FLAGS)
        held = os.fstat(fd)
        after = os.stat(path, follow_symlinks=False)
    except PersistenceError:
        raise
    except OSError as exc:
        raise PersistenceError(f"cannot open directory: {path.name}") from exc
    if (
        len(
            {
                (before.st_dev, before.st_ino),
                (held.st_dev, held.st_ino),
                (after.st_dev, after.st_ino),
            }
        )
        != 1
        or not stat.S_ISDIR(held.st_mode)
        or held.st_uid != _EUID
    ):
        os.close(fd)
        raise PersistenceError(f"directory identity changed: {path.name}")
    return fd


def _fsync_directory(path: Path, accepted_modes: set[int]) -> None:
    fd = _open_directory(path, accepted_modes)
    try:
        os.fsync(fd)
    except OSError as exc:
        raise PersistenceError(f"directory fsync failed: {path.name}") from exc
    finally:
        os.close(fd)


def _fsync_parent(path: Path) -> None:
    _fsync_directory(path.parent, {0o700, 0o500})


def _create_directory(parent: Path, name: str) -> Path:
    if not name or "/" in name or name in {".", ".."}:
        raise PersistenceError("invalid directory component")
    parent_fd = _open_directory(parent, {0o700})
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except OSError as exc:
        raise PersistenceError(f"directory creation failed: {name}") from exc
    finally:
        os.close(parent_fd)
    child = parent / name
    try:
        _fsync_directory(child, {0o700})
        _fsync_directory(parent, {0o700})
    except BaseException:
        raise
    return child


def _ensure_directory(parent: Path, name: str) -> Path:
    child = parent / name
    try:
        entry = os.lstat(child)
    except FileNotFoundError:
        return _create_directory(parent, name)
    except OSError as exc:
        raise PersistenceError(f"directory lookup failed: {name}") from exc
    if (
        not stat.S_ISDIR(entry.st_mode)
        or stat.S_ISLNK(entry.st_mode)
        or entry.st_uid != _EUID
        or _mode(entry) != 0o700
    ):
        raise PersistenceError(f"existing directory is unsafe: {name}")
    # A visible directory may have survived an earlier failed child/parent
    # fsync.  Re-establish both durability edges before permitting any later
    # topology or ownership operation to depend on it.
    _fsync_directory(child, {0o700})
    _fsync_directory(parent, {0o700})
    return child


def _validate_regular_entry(
    entry: os.stat_result,
    *,
    modes: set[int],
    label: str,
) -> None:
    if (
        not stat.S_ISREG(entry.st_mode)
        or stat.S_ISLNK(entry.st_mode)
        or entry.st_uid != _EUID
        or entry.st_nlink != 1
        or _mode(entry) not in modes
    ):
        raise PersistenceError(f"unsafe regular file: {label}")


def _is_publication_stage_name(name: str) -> bool:
    suffix = name.removeprefix(_PUBLICATION_STAGE_PREFIX)
    return (
        name.startswith(_PUBLICATION_STAGE_PREFIX)
        and len(suffix) == 32
        and all(character in "0123456789abcdef" for character in suffix)
    )


def _validate_receipt_alias_entry(entry: os.stat_result, *, label: str) -> None:
    if (
        not stat.S_ISREG(entry.st_mode)
        or stat.S_ISLNK(entry.st_mode)
        or entry.st_uid != _EUID
        or entry.st_nlink != 2
        or _mode(entry) != 0o600
    ):
        raise PersistenceError(f"unsafe receipt publication alias: {label}")


def _read_regular(path: Path, *, modes: set[int], maximum: int) -> bytes:
    try:
        before = os.lstat(path)
        _validate_regular_entry(before, modes=modes, label=path.name)
        fd = os.open(path, _READ_FLAGS)
        held = os.fstat(fd)
        _validate_regular_entry(held, modes=modes, label=path.name)
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after_fd = os.fstat(fd)
        after_path = os.stat(path, follow_symlinks=False)
    except PersistenceError:
        raise
    except OSError as exc:
        raise PersistenceError(f"file read failed: {path.name}") from exc
    finally:
        if "fd" in locals():
            os.close(fd)
    raw = b"".join(chunks)
    if len(raw) > maximum:
        raise PersistenceError(f"file exceeds byte limit: {path.name}")
    identities = {
        (before.st_dev, before.st_ino),
        (held.st_dev, held.st_ino),
        (after_fd.st_dev, after_fd.st_ino),
        (after_path.st_dev, after_path.st_ino),
    }
    if len(identities) != 1 or before.st_size != after_path.st_size:
        raise PersistenceError(f"file identity changed: {path.name}")
    _validate_regular_entry(after_path, modes=modes, label=path.name)
    return raw


def _revalidate_named_lock(root: RootCapability, fd: int) -> None:
    name = ".phase1-store.lock"
    try:
        held = os.fstat(fd)
        named = os.stat(
            name,
            dir_fd=root.fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise PersistenceError("global lock identity is unavailable") from exc
    _validate_regular_entry(held, modes={0o600}, label=name)
    _validate_regular_entry(named, modes={0o600}, label=name)
    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
        raise PersistenceError("global lock identity changed")


def _create_lock_file(root: RootCapability, *, allow_create: bool) -> int:
    name = ".phase1-store.lock"
    flags = _LOCK_FLAGS
    if allow_create:
        flags |= os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(name, flags, 0o600, dir_fd=root.fd)
    except FileExistsError:
        if not allow_create:
            raise PersistenceError("global lock unexpectedly exists") from None
        try:
            fd = os.open(name, _LOCK_FLAGS, dir_fd=root.fd)
        except OSError as exc:
            raise PersistenceError("global lock open failed") from exc
    except OSError as exc:
        action = "creation" if allow_create else "open"
        raise PersistenceError(f"global lock {action} failed") from exc
    try:
        _revalidate_named_lock(root, fd)
        os.fsync(fd)
        os.fsync(root.fd)
        _revalidate_named_lock(root, fd)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _create_permanent_lock(key_directory: Path) -> int:
    key_fd = _open_directory(key_directory, {0o700})
    try:
        fd = os.open(
            "lock",
            _LOCK_FLAGS | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=key_fd,
        )
        entry = os.fstat(fd)
        _validate_regular_entry(entry, modes={0o600}, label="lock")
        os.fsync(fd)
        os.fsync(key_fd)
    except BaseException:
        if "fd" in locals():
            os.close(fd)
        raise
    finally:
        os.close(key_fd)
    return fd


def _open_permanent_lock(key_directory: Path) -> int:
    key_fd = _open_directory(key_directory, {0o700, 0o500})
    try:
        fd = os.open("lock", _LOCK_FLAGS, dir_fd=key_fd)
        entry = os.fstat(fd)
        _validate_regular_entry(entry, modes={0o600}, label="lock")
    except BaseException:
        if "fd" in locals():
            os.close(fd)
        raise
    finally:
        os.close(key_fd)
    return fd


def _publication_target_matches_stage(
    *,
    key_fd: int,
    parent_fd: int,
    stage_name: str,
    stage_fd: int,
    target_name: str,
    raw: bytes,
) -> tuple[bool, bool]:
    """Return (exact target, exact stage name still present)."""

    stage_entry = os.fstat(stage_fd)
    if (
        not stat.S_ISREG(stage_entry.st_mode)
        or stage_entry.st_uid != _EUID
        or _mode(stage_entry) != 0o600
        or stage_entry.st_size != len(raw)
        or stage_entry.st_nlink not in {1, 2}
    ):
        return (False, False)
    try:
        target_fd = os.open(target_name, _READ_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        return (False, False)
    primary: BaseException | None = None
    try:
        target_entry = os.fstat(target_fd)
        target_named = os.stat(
            target_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        identities = {
            (stage_entry.st_dev, stage_entry.st_ino),
            (target_entry.st_dev, target_entry.st_ino),
            (target_named.st_dev, target_named.st_ino),
        }
        if len(identities) != 1:
            return (False, False)
        initial_attributes = {
            (
                entry.st_size,
                stat.S_IMODE(entry.st_mode),
                entry.st_uid,
                entry.st_nlink,
            )
            for entry in (stage_entry, target_entry, target_named)
        }
        if (
            len(initial_attributes) != 1
            or not stat.S_ISREG(target_entry.st_mode)
            or not stat.S_ISREG(target_named.st_mode)
            or target_entry.st_uid != _EUID
            or target_named.st_uid != _EUID
            or _mode(target_entry) != 0o600
            or _mode(target_named) != 0o600
            or target_entry.st_size != len(raw)
            or target_entry.st_nlink not in {1, 2}
        ):
            return (False, False)
        chunks: list[bytes] = []
        remaining = len(raw) + 1
        while remaining:
            chunk = os.read(target_fd, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if b"".join(chunks) != raw:
            return (False, False)
        after_fd = os.fstat(target_fd)
        after_named = os.stat(
            target_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        identities.update(
            {
                (after_fd.st_dev, after_fd.st_ino),
                (after_named.st_dev, after_named.st_ino),
            }
        )
        final_attributes = {
            (
                entry.st_size,
                stat.S_IMODE(entry.st_mode),
                entry.st_uid,
                entry.st_nlink,
            )
            for entry in (stage_entry, target_entry, target_named, after_fd, after_named)
        }
        if (
            len(identities) != 1
            or len(final_attributes) != 1
            or not stat.S_ISREG(after_fd.st_mode)
            or not stat.S_ISREG(after_named.st_mode)
            or after_fd.st_uid != _EUID
            or after_named.st_uid != _EUID
            or _mode(after_fd) != 0o600
            or _mode(after_named) != 0o600
            or after_fd.st_size != len(raw)
            or after_fd.st_nlink not in {1, 2}
        ):
            return (False, False)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            os.close(target_fd)
        except BaseException:
            if primary is None:
                raise

    try:
        stage_named = os.stat(
            stage_name,
            dir_fd=key_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return (after_fd.st_nlink == 1, False)
    if (
        (stage_named.st_dev, stage_named.st_ino)
        != (stage_entry.st_dev, stage_entry.st_ino)
        or not stat.S_ISREG(stage_named.st_mode)
        or stage_named.st_uid != _EUID
        or _mode(stage_named) != 0o600
        or after_fd.st_nlink != 2
    ):
        return (False, False)
    return (True, True)


def _reconcile_publication(
    *,
    key_fd: int,
    parent_fd: int,
    stage_name: str,
    stage_fd: int,
    target_name: str,
    raw: bytes,
    attempt: _PublicationAttempt,
) -> None:
    target_matches, stage_named = _publication_target_matches_stage(
        key_fd=key_fd,
        parent_fd=parent_fd,
        stage_name=stage_name,
        stage_fd=stage_fd,
        target_name=target_name,
        raw=raw,
    )
    if target_matches:
        attempt.target_observed = True
        if not attempt.target_parent_fsync_attempted:
            attempt.target_parent_fsync_attempted = True
            os.fsync(parent_fd)
            attempt.target_parent_synced = True
        if not attempt.target_parent_synced:
            return
        if stage_named:
            if attempt.unlink_attempted:
                return
            attempt.unlink_attempted = True
            os.unlink(stage_name, dir_fd=key_fd)
            attempt.stage_absent = True
        else:
            attempt.stage_absent = True
        if not attempt.stage_parent_fsync_attempted:
            attempt.stage_parent_fsync_attempted = True
            os.fsync(key_fd)
            attempt.stage_parent_synced = True
        if not attempt.stage_parent_synced:
            return
        target_matches, stage_named = _publication_target_matches_stage(
            key_fd=key_fd,
            parent_fd=parent_fd,
            stage_name=stage_name,
            stage_fd=stage_fd,
            target_name=target_name,
            raw=raw,
        )
        if target_matches and not stage_named:
            attempt.final_revalidated = True
        return

    # A target that is absent or belongs to another inode was never published
    # by this attempt. Remove only this attempt's still-named staging inode.
    try:
        stage_named_entry = os.stat(
            stage_name,
            dir_fd=key_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    held = os.fstat(stage_fd)
    if (stage_named_entry.st_dev, stage_named_entry.st_ino) == (
        held.st_dev,
        held.st_ino,
    ):
        if not attempt.unlink_attempted:
            attempt.unlink_attempted = True
            os.unlink(stage_name, dir_fd=key_fd)
            attempt.stage_absent = True
        if attempt.stage_absent and not attempt.stage_parent_fsync_attempted:
            attempt.stage_parent_fsync_attempted = True
            os.fsync(key_fd)
            attempt.stage_parent_synced = True


def _publish_no_replace(
    *,
    key_directory: Path,
    target_parent: Path,
    target_name: str,
    raw: bytes,
    attempt: _PublicationAttempt | None = None,
) -> None:
    if not target_name or "/" in target_name or target_name in {".", ".."}:
        raise PersistenceError("invalid publication target")
    publication = attempt if attempt is not None else _PublicationAttempt()
    stage_name = f".phase1-stage-{secrets.token_hex(16)}"
    key_fd: int | None = None
    parent_fd: int | None = None
    stage_fd: int | None = None
    primary: BaseException | None = None
    try:
        key_fd = _open_directory(key_directory, {0o700})
        parent_fd = _open_directory(target_parent, {0o700})
        stage_fd = os.open(
            stage_name,
            _WRITE_FLAGS | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=key_fd,
        )
        view = memoryview(raw)
        while view:
            written = os.write(stage_fd, view)
            if written <= 0:
                raise PersistenceError("short publication write")
            view = view[written:]
        entry = os.fstat(stage_fd)
        _validate_regular_entry(entry, modes={0o600}, label=stage_name)
        os.fsync(stage_fd)
        publication.link_attempted = True
        os.link(
            stage_name,
            target_name,
            src_dir_fd=key_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        publication.target_observed = True
        publication.target_parent_fsync_attempted = True
        os.fsync(parent_fd)
        publication.target_parent_synced = True
        publication.unlink_attempted = True
        os.unlink(stage_name, dir_fd=key_fd)
        publication.stage_absent = True
        publication.stage_parent_fsync_attempted = True
        os.fsync(key_fd)
        publication.stage_parent_synced = True
        target_matches, stage_named = _publication_target_matches_stage(
            key_fd=key_fd,
            parent_fd=parent_fd,
            stage_name=stage_name,
            stage_fd=stage_fd,
            target_name=target_name,
            raw=raw,
        )
        if not target_matches or stage_named:
            raise PersistenceError("published file revalidation failed")
        publication.final_revalidated = True
    except BaseException as original:
        primary = original
        reconciliation_error: BaseException | None = None
        if stage_fd is not None and key_fd is not None and parent_fd is not None:
            try:
                _reconcile_publication(
                    key_fd=key_fd,
                    parent_fd=parent_fd,
                    stage_name=stage_name,
                    stage_fd=stage_fd,
                    target_name=target_name,
                    raw=raw,
                    attempt=publication,
                )
            except BaseException as exc:
                reconciliation_error = exc
        if publication.complete and isinstance(original, Exception):
            return
        if not isinstance(original, Exception):
            raise
        if reconciliation_error is not None and not isinstance(
            reconciliation_error, Exception
        ):
            raise reconciliation_error
        if isinstance(original, FileExistsError):
            raise PersistenceError(
                f"publication target already exists: {target_name}"
            ) from original
        if isinstance(original, PersistenceError):
            raise
        if isinstance(original, OSError):
            raise PersistenceError(f"publication failed: {target_name}") from original
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if stage_fd is not None:
            try:
                os.close(stage_fd)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if key_fd is not None:
            try:
                os.close(key_fd)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors and primary is None:
            raise cleanup_errors[0]


def _directory_names(path: Path, *, maximum: int) -> list[str]:
    fd = _open_directory(path, {0o700, 0o500})
    names: list[str] = []
    try:
        with os.scandir(fd) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > maximum:
                    raise PersistenceError(
                        f"directory entry limit exceeded: {path.name}"
                    )
    except OSError as exc:
        raise PersistenceError(f"directory enumeration failed: {path.name}") from exc
    finally:
        os.close(fd)
    return sorted(names)


def _directory_has_matching_name(
    path: Path,
    predicate: Callable[[str], bool],
) -> bool:
    fd = _open_directory(path, {0o700, 0o500})
    try:
        with os.scandir(fd) as entries:
            return any(predicate(entry.name) for entry in entries)
    except OSError as exc:
        raise PersistenceError(f"directory enumeration failed: {path.name}") from exc
    finally:
        os.close(fd)


def _read_committed_receipt_alias(
    key_directory: Path,
    stage_name: str,
) -> bytes:
    """Read one exact two-name receipt publication without repairing it."""

    if not _is_publication_stage_name(stage_name):
        raise PersistenceError("invalid receipt publication stage name")
    key_fd: int | None = None
    receipt_fd: int | None = None
    stage_fd: int | None = None
    primary: BaseException | None = None
    try:
        key_fd = _open_directory(key_directory, {0o700, 0o500})
        receipt_before = os.stat(
            "receipt.json", dir_fd=key_fd, follow_symlinks=False
        )
        stage_before = os.stat(
            stage_name, dir_fd=key_fd, follow_symlinks=False
        )
        receipt_fd = os.open("receipt.json", _READ_FLAGS, dir_fd=key_fd)
        stage_fd = os.open(stage_name, _READ_FLAGS, dir_fd=key_fd)
        receipt_held = os.fstat(receipt_fd)
        stage_held = os.fstat(stage_fd)
        initial_entries = (
            receipt_before,
            stage_before,
            receipt_held,
            stage_held,
        )
        for entry in initial_entries:
            _validate_receipt_alias_entry(entry, label=stage_name)
        if len({(entry.st_dev, entry.st_ino) for entry in initial_entries}) != 1:
            raise PersistenceError("receipt publication aliases differ")
        initial_attributes = {
            (
                entry.st_size,
                stat.S_IMODE(entry.st_mode),
                entry.st_uid,
                entry.st_nlink,
            )
            for entry in initial_entries
        }
        if len(initial_attributes) != 1 or receipt_held.st_size > MAX_RECEIPT_BYTES:
            raise PersistenceError("receipt publication alias changed")

        chunks: list[bytes] = []
        remaining = MAX_RECEIPT_BYTES + 1
        while remaining:
            chunk = os.read(receipt_fd, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)

        receipt_after_fd = os.fstat(receipt_fd)
        stage_after_fd = os.fstat(stage_fd)
        receipt_after_name = os.stat(
            "receipt.json", dir_fd=key_fd, follow_symlinks=False
        )
        stage_after_name = os.stat(
            stage_name, dir_fd=key_fd, follow_symlinks=False
        )
        final_entries = (
            *initial_entries,
            receipt_after_fd,
            stage_after_fd,
            receipt_after_name,
            stage_after_name,
        )
        for entry in final_entries:
            _validate_receipt_alias_entry(entry, label=stage_name)
        if (
            len({(entry.st_dev, entry.st_ino) for entry in final_entries}) != 1
            or len(
                {
                    (
                        entry.st_size,
                        stat.S_IMODE(entry.st_mode),
                        entry.st_uid,
                        entry.st_nlink,
                    )
                    for entry in final_entries
                }
            )
            != 1
            or len(raw) != receipt_after_fd.st_size
        ):
            raise PersistenceError("receipt publication alias changed while read")
        return raw
    except BaseException as exc:
        primary = exc
        if isinstance(exc, OSError):
            raise PersistenceError("receipt publication alias is unavailable") from exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        for fd in (stage_fd, receipt_fd, key_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if cleanup_errors and primary is None:
            raise PersistenceError("receipt publication alias close failed") from cleanup_errors[0]


def _read_committed_final_record_alias(
    key_directory: Path,
    records_directory: Path,
    stage_name: str,
    target_name: str,
) -> bytes:
    """Read one exact staged alias of the canonical final ownership record."""

    if (
        not _is_publication_stage_name(stage_name)
        or not target_name
        or "/" in target_name
        or not target_name.endswith("-RECEIPT_FINALIZED.json")
    ):
        raise PersistenceError("invalid final-record publication alias")
    key_fd: int | None = None
    records_fd: int | None = None
    target_fd: int | None = None
    stage_fd: int | None = None
    primary: BaseException | None = None
    try:
        key_fd = _open_directory(key_directory, {0o700, 0o500})
        records_fd = _open_directory(records_directory, {0o700, 0o500})
        target_before = os.stat(
            target_name,
            dir_fd=records_fd,
            follow_symlinks=False,
        )
        stage_before = os.stat(
            stage_name,
            dir_fd=key_fd,
            follow_symlinks=False,
        )
        target_fd = os.open(target_name, _READ_FLAGS, dir_fd=records_fd)
        stage_fd = os.open(stage_name, _READ_FLAGS, dir_fd=key_fd)
        target_held = os.fstat(target_fd)
        stage_held = os.fstat(stage_fd)
        initial_entries = (target_before, stage_before, target_held, stage_held)
        for entry in initial_entries:
            _validate_receipt_alias_entry(entry, label=target_name)
        if len({(entry.st_dev, entry.st_ino) for entry in initial_entries}) != 1:
            raise PersistenceError("final-record publication aliases differ")
        if (
            len(
                {
                    (
                        entry.st_size,
                        stat.S_IMODE(entry.st_mode),
                        entry.st_uid,
                        entry.st_nlink,
                    )
                    for entry in initial_entries
                }
            )
            != 1
            or target_held.st_size > MAX_RECORD_BYTES
        ):
            raise PersistenceError("final-record publication alias changed")

        chunks: list[bytes] = []
        remaining = MAX_RECORD_BYTES + 1
        while remaining:
            chunk = os.read(target_fd, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)

        target_after_fd = os.fstat(target_fd)
        stage_after_fd = os.fstat(stage_fd)
        target_after_name = os.stat(
            target_name,
            dir_fd=records_fd,
            follow_symlinks=False,
        )
        stage_after_name = os.stat(
            stage_name,
            dir_fd=key_fd,
            follow_symlinks=False,
        )
        final_entries = (
            *initial_entries,
            target_after_fd,
            stage_after_fd,
            target_after_name,
            stage_after_name,
        )
        for entry in final_entries:
            _validate_receipt_alias_entry(entry, label=target_name)
        if (
            len({(entry.st_dev, entry.st_ino) for entry in final_entries}) != 1
            or len(
                {
                    (
                        entry.st_size,
                        stat.S_IMODE(entry.st_mode),
                        entry.st_uid,
                        entry.st_nlink,
                    )
                    for entry in final_entries
                }
            )
            != 1
            or len(raw) != target_after_fd.st_size
        ):
            raise PersistenceError("final-record alias changed while read")
        return raw
    except BaseException as exc:
        primary = exc
        if isinstance(exc, OSError):
            raise PersistenceError("final-record alias is unavailable") from exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        for fd in (stage_fd, target_fd, records_fd, key_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if cleanup_errors and primary is None:
            raise PersistenceError("final-record alias close failed") from cleanup_errors[0]


@dataclass(frozen=True, slots=True)
class _RegistryEntry:
    token: object
    operation_digest: str
    ownership_record_digest: str
    global_lock_identity: tuple[int, int]


_REGISTRY_PID = os.getpid()
_REGISTRY_GUARD = threading.Lock()
_ACTIVE_KEYS: dict[tuple[int, int, str], _RegistryEntry] = {}
_ROOT_MUTEXES: dict[tuple[int, int], tuple[threading.Lock, int]] = {}


def _ensure_registry_pid() -> None:
    global _REGISTRY_PID, _REGISTRY_GUARD, _ACTIVE_KEYS, _ROOT_MUTEXES
    pid = os.getpid()
    if pid != _REGISTRY_PID:
        _REGISTRY_PID = pid
        _REGISTRY_GUARD = threading.Lock()
        _ACTIVE_KEYS = {}
        _ROOT_MUTEXES = {}


def _registry_lookup(key: tuple[int, int, str]) -> _RegistryEntry | None:
    _ensure_registry_pid()
    with _REGISTRY_GUARD:
        return _ACTIVE_KEYS.get(key)


def _registry_insert(
    key: tuple[int, int, str],
    *,
    token: object,
    operation_digest: str,
    ownership_record_digest: str,
    global_lock_identity: tuple[int, int],
) -> None:
    _ensure_registry_pid()
    with _REGISTRY_GUARD:
        if key in _ACTIVE_KEYS:
            raise PersistenceError("active-key registry collision")
        _ACTIVE_KEYS[key] = _RegistryEntry(
            token,
            operation_digest,
            ownership_record_digest,
            global_lock_identity,
        )


def _registry_remove(key: tuple[int, int, str], token: object) -> None:
    _ensure_registry_pid()
    with _REGISTRY_GUARD:
        current = _ACTIVE_KEYS.get(key)
        if current is not None and current.token is token:
            del _ACTIVE_KEYS[key]


def _root_mutex_acquire(identity: tuple[int, int]) -> threading.Lock:
    _ensure_registry_pid()
    with _REGISTRY_GUARD:
        current = _ROOT_MUTEXES.get(identity)
        if current is None:
            lock = threading.Lock()
            _ROOT_MUTEXES[identity] = (lock, 1)
        else:
            lock, references = current
            _ROOT_MUTEXES[identity] = (lock, references + 1)
    try:
        lock.acquire()
    except BaseException:
        with _REGISTRY_GUARD:
            current = _ROOT_MUTEXES.get(identity)
            if current is not None and current[0] is lock:
                if current[1] == 1:
                    del _ROOT_MUTEXES[identity]
                else:
                    _ROOT_MUTEXES[identity] = (lock, current[1] - 1)
        raise
    return lock


def _root_mutex_release(identity: tuple[int, int], lock: threading.Lock) -> None:
    lock.release()
    with _REGISTRY_GUARD:
        current = _ROOT_MUTEXES.get(identity)
        if current is None or current[0] is not lock:
            raise PersistenceError("root mutex registry mismatch")
        if current[1] == 1:
            del _ROOT_MUTEXES[identity]
        else:
            _ROOT_MUTEXES[identity] = (lock, current[1] - 1)


@dataclass(slots=True)
class _StoredState:
    records: list[dict[str, JSONValue]]
    record_names: list[str]
    record_modes: list[int]
    receipt_raw: bytes | None
    receipt: dict[str, JSONValue] | None
    key_mode: int
    records_mode: int


@dataclass(slots=True)
class OwnerHandle:
    roots: RootSet
    operation: Operation
    key_directory: Path
    records_directory: Path
    evidence_destination: Path
    key_lock_fd: int
    registry_key: tuple[int, int, str]
    registry_token: object
    records: list[dict[str, JSONValue]]
    preflight_result: Any = None
    sealed_raw: bytes | None = None
    sealed_receipt: dict[str, JSONValue] | None = None
    _closed: bool = False
    _finalization_attempted: bool = False
    _finalization_complete: bool = False
    _receipt_publication_complete: bool = False
    _store_ambiguous: bool = False

    def __enter__(self) -> "OwnerHandle":
        return self

    def __exit__(
        self,
        _exc_type: object,
        exc: object,
        _traceback: object,
    ) -> None:
        self.close(_primary=exc if isinstance(exc, BaseException) else None)

    def close(self, *, _primary: BaseException | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        released = False
        try:
            fcntl.flock(self.key_lock_fd, fcntl.LOCK_UN)
            released = True
        except BaseException as exc:
            errors.append(exc)
        try:
            os.close(self.key_lock_fd)
            released = True
        except BaseException as exc:
            errors.append(exc)
        if released:
            try:
                _registry_remove(self.registry_key, self.registry_token)
            except BaseException as exc:
                errors.append(exc)
        if errors and _primary is None:
            raise errors[0]

    @property
    def last_record(self) -> dict[str, JSONValue]:
        return self.records[-1]

    def append(
        self,
        state: str,
        *,
        reason_code: str | None = None,
        phase0_terminal_report_digest: str | None = None,
        phase0_evidence_manifest_digest: str | None = None,
        phase0_evidence_directory_name: str | None = None,
        receipt_digest: str | None = None,
    ) -> dict[str, JSONValue]:
        if self._closed:
            raise PersistenceError("owner handle is closed")
        if self._store_ambiguous:
            raise StoreIndeterminate(self.operation)
        previous = self.last_record
        record, raw = build_record(
            self.operation,
            sequence=len(self.records) + 1,
            state=state,
            previous_record_digest=str(previous["record_digest"]),
            reason_code=reason_code,
            phase0_terminal_report_digest=phase0_terminal_report_digest,
            phase0_evidence_manifest_digest=phase0_evidence_manifest_digest,
            phase0_evidence_directory_name=phase0_evidence_directory_name,
            receipt_digest=receipt_digest,
        )
        validate_record_chain([*self.records, record])
        publication = _PublicationAttempt()
        try:
            _publish_no_replace(
                key_directory=self.key_directory,
                target_parent=self.records_directory,
                target_name=record_filename(record),
                raw=raw,
                attempt=publication,
            )
        except BaseException:
            if publication.committed:
                self.records.append(record)
                self._store_ambiguous = not publication.complete
            raise
        self.records.append(record)
        return record

    def refresh_active(self) -> list[dict[str, JSONValue]]:
        """Reload a clean, receipt-free active prefix under the held key lock."""

        if self._store_ambiguous:
            raise StoreIndeterminate(self.operation)
        try:
            state = _load_stored_state(
                self.key_directory,
                self.records_directory,
                self.operation,
            )
            _validate_active_permissions(state)
        except (PersistenceError, ContractValidationError, OSError):
            raise StoreIndeterminate(self.operation) from None
        self.records = state.records
        return self.records

    def seal(
        self,
        *,
        receipt_state: str,
        reason_code: str,
        phase0_terminal_report: Mapping[str, JSONValue] | None = None,
        phase0_evidence_manifest: Mapping[str, JSONValue] | None = None,
    ) -> bytes:
        if self._store_ambiguous:
            if self.sealed_receipt is not None:
                raise StoreSealedUnavailable(
                    self.operation,
                    receipt_digest=str(self.sealed_receipt["receipt_digest"]),
                    receipt_state=str(self.sealed_receipt["receipt_state"]),
                ) from None
            raise StoreIndeterminate(self.operation)
        receipt, raw = sealed_receipt(
            self.operation,
            receipt_state=receipt_state,
            predecessor=self.last_record,
            reason_code=reason_code,
            phase0_terminal_report=phase0_terminal_report,
            phase0_evidence_manifest=phase0_evidence_manifest,
        )
        publication = _PublicationAttempt()
        try:
            _publish_no_replace(
                key_directory=self.key_directory,
                target_parent=self.key_directory,
                target_name="receipt.json",
                raw=raw,
                attempt=publication,
            )
        except BaseException as error:
            if publication.committed:
                self.sealed_raw = raw
                self.sealed_receipt = receipt
                self._receipt_publication_complete = publication.complete
                self._store_ambiguous = not publication.complete
            if self.sealed_raw is not None:
                if isinstance(error, Exception):
                    if not publication.complete:
                        if self._safe_revalidate_sealed():
                            return raw
                        raise StoreSealedUnavailable(
                            self.operation,
                            receipt_digest=str(receipt["receipt_digest"]),
                            receipt_state=receipt_state,
                        ) from None
                    try:
                        self._attempt_finalize_once()
                    except Exception:
                        if self._safe_revalidate_sealed():
                            return raw
                        raise StoreSealedUnavailable(
                            self.operation,
                            receipt_digest=str(receipt["receipt_digest"]),
                            receipt_state=receipt_state,
                        ) from None
                    return raw
                self.best_effort_finalize()
            raise
        self.sealed_raw = raw
        self.sealed_receipt = receipt
        self._receipt_publication_complete = publication.complete
        if not publication.complete:
            raise StoreSealedUnavailable(
                self.operation,
                receipt_digest=str(receipt["receipt_digest"]),
                receipt_state=receipt_state,
            ) from None
        try:
            self._attempt_finalize_once()
        except Exception:
            if self._safe_revalidate_sealed():
                return raw
            raise StoreSealedUnavailable(
                self.operation,
                receipt_digest=str(receipt["receipt_digest"]),
                receipt_state=receipt_state,
            ) from None
        return raw

    def _attempt_finalize_once(self) -> None:
        if self._finalization_attempted:
            if self._finalization_complete:
                return
            raise PersistenceError("finalization was already attempted in this call")
        if not self._receipt_publication_complete or self._store_ambiguous:
            raise PersistenceError("receipt publication is not cleanly finalizable")
        self._finalization_attempted = True
        self._finalization_complete = self._finalize_once()

    def _finalize_once(self) -> bool:
        if self.sealed_raw is None or self.sealed_receipt is None:
            raise PersistenceError("receipt is not sealed")
        if self.last_record["state"] != "RECEIPT_FINALIZED":
            try:
                self.append(
                    "RECEIPT_FINALIZED",
                    receipt_digest=str(self.sealed_receipt["receipt_digest"]),
                )
            except Exception:
                # The receipt is already durably sealed. A final-record
                # publication fault must not relabel those exact bytes, and
                # must not trigger another mutation in this owner call.
                if self._safe_revalidate_sealed():
                    return False
                raise
        _finalize_metadata(
            key_directory=self.key_directory,
            records_directory=self.records_directory,
            records=self.records,
            operation=self.operation,
        )
        return True

    def best_effort_finalize(self) -> None:
        if (
            self.sealed_raw is None
            or self._finalization_attempted
            or self._store_ambiguous
        ):
            return
        try:
            self._attempt_finalize_once()
        except BaseException:
            pass

    def _safe_revalidate_sealed(self) -> bool:
        if self.sealed_raw is None or self.sealed_receipt is None:
            return False
        try:
            try:
                state = _load_stored_state(
                    self.key_directory,
                    self.records_directory,
                    self.operation,
                )
            except (PersistenceError, ContractValidationError, OSError):
                receipt_alias = _load_receipt_alias_state(
                    key_directory=self.key_directory,
                    records_directory=self.records_directory,
                    operation=self.operation,
                    require_operation_match=True,
                )
                if receipt_alias is not None:
                    state = receipt_alias[0]
                else:
                    final_alias = _load_final_record_alias_state(
                        key_directory=self.key_directory,
                        records_directory=self.records_directory,
                        operation=self.operation,
                        require_operation_match=True,
                    )
                    if final_alias is None:
                        return False
                    state = final_alias[0]
            if state.receipt_raw != self.sealed_raw:
                return False
            _validate_permission_prefix(
                key_directory=self.key_directory,
                records_directory=self.records_directory,
                state=state,
            )
            self.records = state.records
            return True
        except (PersistenceError, ContractValidationError, OSError):
            return False


@dataclass(frozen=True, slots=True)
class ClaimResult:
    response: bytes | None = None
    owner: OwnerHandle | None = None

    def __post_init__(self) -> None:
        if (self.response is None) == (self.owner is None):
            raise ValueError("claim result must contain exactly one outcome")


def _validate_receipt_root_entries(root: RootCapability) -> str:
    names: list[str] = []
    try:
        with os.scandir(root.fd) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > 2:
                    raise RootSafetyError("UNSAFE_RECEIPT_ROOT")
    except OSError as exc:
        raise RootSafetyError("UNSAFE_RECEIPT_ROOT") from exc
    found = set(names)
    if len(found) != len(names) or not found <= {".phase1-store.lock", "v1"}:
        raise RootSafetyError("UNSAFE_RECEIPT_ROOT")
    if found == set():
        return "virgin"
    if found == {".phase1-store.lock"}:
        return "lock-only"
    if found == {".phase1-store.lock", "v1"}:
        return "established"
    # v1 cannot be made durable before the permanent global lock.
    if found == {"v1"}:
        return "missing-lock"
    raise RootSafetyError("UNSAFE_RECEIPT_ROOT")


def _validate_active_registry_store(
    roots: RootSet,
    expected_global_lock_identity: tuple[int, int],
) -> None:
    """Read-only validation before trusting an in-process active-key entry."""

    lock_fd: int | None = None
    primary: BaseException | None = None
    try:
        roots.revalidate_all()
        if _validate_receipt_root_entries(roots.receipt) != "established":
            raise PersistenceError("active registry lacks an established store")
        v1_entry = os.stat(
            "v1",
            dir_fd=roots.receipt.fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(v1_entry.st_mode)
            or stat.S_ISLNK(v1_entry.st_mode)
            or v1_entry.st_uid != _EUID
            or _mode(v1_entry) != 0o700
        ):
            raise PersistenceError("active registry store topology is unsafe")
        lock_fd = os.open(
            ".phase1-store.lock",
            _READ_FLAGS,
            dir_fd=roots.receipt.fd,
        )
        _revalidate_named_lock(roots.receipt, lock_fd)
        held_lock = os.fstat(lock_fd)
        if (held_lock.st_dev, held_lock.st_ino) != expected_global_lock_identity:
            raise PersistenceError("active registry global lock was replaced")
        roots.revalidate_all()
        if _validate_receipt_root_entries(roots.receipt) != "established":
            raise PersistenceError("active registry store topology changed")
        v1_after = os.stat(
            "v1",
            dir_fd=roots.receipt.fd,
            follow_symlinks=False,
        )
        if (
            (v1_after.st_dev, v1_after.st_ino)
            != (v1_entry.st_dev, v1_entry.st_ino)
            or not stat.S_ISDIR(v1_after.st_mode)
            or stat.S_ISLNK(v1_after.st_mode)
            or v1_after.st_uid != _EUID
            or _mode(v1_after) != 0o700
        ):
            raise PersistenceError("active registry store topology changed")
        _revalidate_named_lock(roots.receipt, lock_fd)
        held_lock_after = os.fstat(lock_fd)
        if (
            (held_lock_after.st_dev, held_lock_after.st_ino)
            != expected_global_lock_identity
        ):
            raise PersistenceError("active registry global lock was replaced")
    except BaseException as exc:
        primary = exc
        if isinstance(exc, OSError):
            raise PersistenceError("active registry store is unavailable") from exc
        raise
    finally:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except BaseException as exc:
                if primary is None:
                    if isinstance(exc, OSError):
                        raise PersistenceError(
                            "active registry lock close failed"
                        ) from exc
                    raise


def _prepare_evidence_destination(
    roots: RootSet,
    operation: Operation,
) -> Path:
    v1 = _ensure_directory(roots.evidence.path, "v1")
    shard = _ensure_directory(v1, operation.idempotency_key_digest[:2])
    destination_name = f"phase0-{operation.idempotency_key_digest}"
    destination = shard / destination_name
    prefix = f".{destination_name}."
    if _directory_has_matching_name(
        shard,
        lambda name: name == destination_name
        or (name.startswith(prefix) and name.endswith(".staging")),
    ):
        raise RootSafetyError("UNSAFE_EVIDENCE_ROOT")
    roots.revalidate_all()
    return destination


def _load_anchor(records_directory: Path) -> dict[str, JSONValue]:
    expected = "000001-OWNERSHIP_ACQUIRED.json"
    raw = _read_regular(
        records_directory / expected,
        modes={0o600, 0o400},
        maximum=MAX_RECORD_BYTES,
    )
    record = parse_record_bytes(raw)
    validate_record_chain([record], filenames=[expected])
    return record


def _load_stored_state(
    key_directory: Path,
    records_directory: Path,
    operation: Operation,
    *,
    require_operation_match: bool = True,
    receipt_stage_name: str | None = None,
    final_record_alias: tuple[str, str] | None = None,
) -> _StoredState:
    names = _directory_names(
        key_directory, maximum=MAX_DIRECTORY_ENTRIES_PER_KEY
    )
    allowed_names = {"lock", "records", "receipt.json"}
    if receipt_stage_name is not None and final_record_alias is not None:
        raise PersistenceError("multiple publication aliases are not recoverable")
    staged_name = receipt_stage_name
    if final_record_alias is not None:
        staged_name = final_record_alias[0]
    if staged_name is not None:
        if not _is_publication_stage_name(staged_name):
            raise PersistenceError("invalid receipt publication stage name")
        allowed_names.add(staged_name)
    if (
        not {"lock", "records"} <= set(names)
        or not set(names) <= allowed_names
        or (
            staged_name is not None
            and staged_name not in set(names)
        )
        or (
            receipt_stage_name is not None
            and "receipt.json" not in set(names)
        )
    ):
        raise PersistenceError("unexpected key-directory entry")
    lock_entry = os.lstat(key_directory / "lock")
    _validate_regular_entry(lock_entry, modes={0o600}, label="lock")
    record_names = _directory_names(
        records_directory, maximum=MAX_RECORDS_PER_KEY
    )
    if not record_names:
        raise PersistenceError("ownership chain is empty")
    records: list[dict[str, JSONValue]] = []
    modes: list[int] = []
    for name in record_names:
        path = records_directory / name
        entry = os.lstat(path)
        modes.append(_mode(entry))
        if final_record_alias is not None and name == final_record_alias[1]:
            raw = _read_committed_final_record_alias(
                key_directory,
                records_directory,
                final_record_alias[0],
                final_record_alias[1],
            )
        else:
            raw = _read_regular(
                path,
                modes={0o600, 0o400},
                maximum=MAX_RECORD_BYTES,
            )
        records.append(parse_record_bytes(raw))
    if final_record_alias is not None and final_record_alias[1] not in record_names:
        raise PersistenceError("final-record publication target is absent")
    records = validate_record_chain(records, filenames=record_names)
    anchor = records[0]
    if (
        anchor["idempotency_key_digest"] != operation.idempotency_key_digest
        or (
            require_operation_match
            and anchor["operation_digest"] != operation.operation_digest
        )
    ):
        raise ContractValidationError("stored state is bound to another operation")

    key_entry = os.lstat(key_directory)
    records_entry = os.lstat(records_directory)
    receipt_raw: bytes | None = None
    receipt: dict[str, JSONValue] | None = None
    if "receipt.json" in names:
        try:
            if receipt_stage_name is None:
                receipt_raw = _read_regular(
                    key_directory / "receipt.json",
                    modes={0o600, 0o400},
                    maximum=MAX_RECEIPT_BYTES,
                )
            else:
                receipt_raw = _read_committed_receipt_alias(
                    key_directory,
                    receipt_stage_name,
                )
            receipt = parse_receipt_bytes(
                receipt_raw,
                chain=records,
                operation=operation if require_operation_match else None,
                require_sealed=True,
            )
        except (PersistenceError, ContractValidationError, OSError):
            if records[-1]["state"] == "RECEIPT_FINALIZED":
                raise _FinalReceiptUnavailable(
                    receipt_digest=str(records[-1]["receipt_digest"]),
                    predecessor_state=str(records[-2]["state"]),
                ) from None
            raise
    elif records[-1]["state"] == "RECEIPT_FINALIZED":
        raise _FinalReceiptUnavailable(
            receipt_digest=str(records[-1]["receipt_digest"]),
            predecessor_state=str(records[-2]["state"]),
        )
    return _StoredState(
        records=records,
        record_names=record_names,
        record_modes=modes,
        receipt_raw=receipt_raw,
        receipt=receipt,
        key_mode=_mode(key_entry),
        records_mode=_mode(records_entry),
    )


def _validate_active_permissions(state: _StoredState) -> None:
    if (
        state.key_mode != 0o700
        or state.records_mode != 0o700
        or any(mode != 0o600 for mode in state.record_modes)
        or state.receipt_raw is not None
    ):
        raise PersistenceError("active permission state is invalid")


def _validate_permission_prefix(
    *,
    key_directory: Path,
    records_directory: Path,
    state: _StoredState,
) -> None:
    if state.receipt_raw is None:
        raise PersistenceError("sealed permission check lacks a receipt")
    receipt_mode = _mode(os.lstat(key_directory / "receipt.json"))
    modes = state.record_modes
    first_active = next((index for index, mode in enumerate(modes) if mode == 0o600), len(modes))
    if any(mode != 0o400 for mode in modes[:first_active]) or any(
        mode != 0o600 for mode in modes[first_active:]
    ):
        raise PersistenceError("record modes are not an ordered prefix")
    all_records_final = first_active == len(modes)
    if receipt_mode not in {0o600, 0o400}:
        raise PersistenceError("receipt mode is invalid")
    if receipt_mode == 0o400 and not all_records_final:
        raise PersistenceError("receipt was restricted before every record")
    if state.records_mode not in {0o700, 0o500}:
        raise PersistenceError("records directory mode is invalid")
    if state.records_mode == 0o500 and receipt_mode != 0o400:
        raise PersistenceError("records directory was restricted out of order")
    if state.key_mode not in {0o700, 0o500}:
        raise PersistenceError("key directory mode is invalid")
    if state.key_mode == 0o500 and state.records_mode != 0o500:
        raise PersistenceError("key directory was restricted out of order")
    has_final = state.records[-1]["state"] == "RECEIPT_FINALIZED"
    any_restricted = (
        first_active > 0
        or receipt_mode == 0o400
        or state.records_mode == 0o500
        or state.key_mode == 0o500
    )
    if any_restricted and not has_final:
        raise PersistenceError("metadata restricted before final record")


def _chmod_file_and_sync(path: Path, mode: int, parent: Path) -> None:
    fd = os.open(path, _READ_FLAGS)
    try:
        entry = os.fstat(fd)
        _validate_regular_entry(entry, modes={0o600, 0o400}, label=path.name)
        os.fchmod(fd, mode)
        os.fsync(fd)
    except OSError as exc:
        raise PersistenceError(f"file finalization failed: {path.name}") from exc
    finally:
        os.close(fd)
    _fsync_directory(parent, {0o700, 0o500})


def _chmod_directory_and_sync(path: Path, mode: int, parent: Path) -> None:
    fd = _open_directory(path, {0o700, 0o500})
    try:
        os.fchmod(fd, mode)
        os.fsync(fd)
    except OSError as exc:
        raise PersistenceError(f"directory finalization failed: {path.name}") from exc
    finally:
        os.close(fd)
    _fsync_directory(parent, {0o700, 0o500})


def _finalize_metadata(
    *,
    key_directory: Path,
    records_directory: Path,
    records: Sequence[Mapping[str, JSONValue]],
    operation: Operation,
) -> None:
    state = _load_stored_state(
        key_directory,
        records_directory,
        operation,
    )
    _validate_permission_prefix(
        key_directory=key_directory,
        records_directory=records_directory,
        state=state,
    )
    for name, mode in zip(state.record_names, state.record_modes, strict=True):
        path = records_directory / name
        if mode == 0o400:
            _fsync_file_and_parent(path, records_directory, {0o400})
        else:
            _chmod_file_and_sync(path, 0o400, records_directory)
    receipt_path = key_directory / "receipt.json"
    receipt_mode = _mode(os.lstat(receipt_path))
    if receipt_mode == 0o400:
        _fsync_file_and_parent(receipt_path, key_directory, {0o400})
    else:
        _chmod_file_and_sync(receipt_path, 0o400, key_directory)
    records_mode = _mode(os.lstat(records_directory))
    if records_mode == 0o500:
        _fsync_directory(records_directory, {0o500})
        _fsync_directory(key_directory, {0o700, 0o500})
    else:
        _chmod_directory_and_sync(records_directory, 0o500, key_directory)
    key_mode = _mode(os.lstat(key_directory))
    if key_mode == 0o500:
        _fsync_directory(key_directory, {0o500})
        _fsync_directory(key_directory.parent, {0o700})
    else:
        _chmod_directory_and_sync(key_directory, 0o500, key_directory.parent)


def _fsync_file_and_parent(path: Path, parent: Path, modes: set[int]) -> None:
    fd = os.open(path, _READ_FLAGS)
    try:
        entry = os.fstat(fd)
        _validate_regular_entry(entry, modes=modes, label=path.name)
        os.fsync(fd)
    except OSError as exc:
        raise PersistenceError(f"file fsync failed: {path.name}") from exc
    finally:
        os.close(fd)
    _fsync_directory(parent, {0o700, 0o500})


def _receipt_state_from_predecessor(state: str) -> str:
    mapping = {
        "PREINVOKE_REJECTED": "REJECTED_PRE_INVOKE",
        "ENGINE_EVIDENCE_VERIFIED": "ENGINE_TERMINAL",
        "INDETERMINATE_NO_RETRY": "INDETERMINATE_NO_RETRY",
    }
    try:
        return mapping[state]
    except KeyError as exc:
        raise PersistenceError("final record has an invalid predecessor") from exc


def _load_receipt_alias_state(
    *,
    key_directory: Path,
    records_directory: Path,
    operation: Operation,
    require_operation_match: bool,
) -> tuple[_StoredState, str] | None:
    """Validate, without mutation, one staged alias of receipt.json."""

    names = _directory_names(
        key_directory,
        maximum=MAX_DIRECTORY_ENTRIES_PER_KEY,
    )
    stage_names = [name for name in names if _is_publication_stage_name(name)]
    if len(stage_names) != 1:
        return None
    stage_name = stage_names[0]
    if set(names) != {"lock", "records", "receipt.json", stage_name}:
        return None
    stage_entry = os.lstat(key_directory / stage_name)
    receipt_entry = os.lstat(key_directory / "receipt.json")
    if (stage_entry.st_dev, stage_entry.st_ino) != (
        receipt_entry.st_dev,
        receipt_entry.st_ino,
    ):
        return None
    state = _load_stored_state(
        key_directory,
        records_directory,
        operation,
        require_operation_match=require_operation_match,
        receipt_stage_name=stage_name,
    )
    if state.receipt is None or state.receipt_raw is None:
        raise PersistenceError("receipt alias lacks a sealed canonical receipt")
    return (state, stage_name)


def _load_final_record_alias_state(
    *,
    key_directory: Path,
    records_directory: Path,
    operation: Operation,
    require_operation_match: bool,
) -> tuple[_StoredState, str, str] | None:
    """Validate, without mutation, one staged alias of RECEIPT_FINALIZED."""

    names = _directory_names(
        key_directory,
        maximum=MAX_DIRECTORY_ENTRIES_PER_KEY,
    )
    stage_names = [name for name in names if _is_publication_stage_name(name)]
    if len(stage_names) != 1:
        return None
    stage_name = stage_names[0]
    if set(names) != {"lock", "records", "receipt.json", stage_name}:
        return None
    record_names = _directory_names(
        records_directory,
        maximum=MAX_RECORDS_PER_KEY,
    )
    targets = [
        name for name in record_names if name.endswith("-RECEIPT_FINALIZED.json")
    ]
    if len(targets) != 1:
        return None
    target_name = targets[0]
    state = _load_stored_state(
        key_directory,
        records_directory,
        operation,
        require_operation_match=require_operation_match,
        final_record_alias=(stage_name, target_name),
    )
    if (
        state.receipt is None
        or state.receipt_raw is None
        or state.records[-1]["state"] != "RECEIPT_FINALIZED"
        or state.record_names[-1] != target_name
    ):
        raise PersistenceError("final-record alias lacks a sealed canonical chain")
    return (state, stage_name, target_name)


def _repair_committed_receipt_stage(
    *,
    key_directory: Path,
    records_directory: Path,
    operation: Operation,
    require_operation_match: bool,
) -> bool:
    """Repair one canonical sealed receipt/final alias under the held key flock."""

    names = _directory_names(
        key_directory,
        maximum=MAX_DIRECTORY_ENTRIES_PER_KEY,
    )
    stage_names = [name for name in names if _is_publication_stage_name(name)]
    if not stage_names:
        return False
    if len(stage_names) != 1:
        return False
    stage_name = stage_names[0]
    if set(names) != {"lock", "records", "receipt.json", stage_name}:
        return False

    stage_entry = os.lstat(key_directory / stage_name)
    receipt_entry = os.lstat(key_directory / "receipt.json")
    receipt_alias = (stage_entry.st_dev, stage_entry.st_ino) == (
        receipt_entry.st_dev,
        receipt_entry.st_ino,
    )
    final_target_name: str | None = None
    if not receipt_alias:
        record_names = _directory_names(
            records_directory,
            maximum=MAX_RECORDS_PER_KEY,
        )
        matching_final_targets: list[str] = []
        for name in record_names:
            if not name.endswith("-RECEIPT_FINALIZED.json"):
                continue
            entry = os.lstat(records_directory / name)
            if (entry.st_dev, entry.st_ino) == (
                stage_entry.st_dev,
                stage_entry.st_ino,
            ):
                matching_final_targets.append(name)
        if len(matching_final_targets) != 1:
            return False
        final_target_name = matching_final_targets[0]

    try:
        if receipt_alias:
            state = _load_stored_state(
                key_directory,
                records_directory,
                operation,
                require_operation_match=require_operation_match,
                receipt_stage_name=stage_name,
            )
        else:
            if final_target_name is None:
                raise PersistenceError("final-record alias target is unavailable")
            state = _load_stored_state(
                key_directory,
                records_directory,
                operation,
                require_operation_match=require_operation_match,
                final_record_alias=(stage_name, final_target_name),
            )
    except _FinalReceiptUnavailable as exc:
        raise StoreSealedUnavailable(
            operation,
            receipt_digest=exc.receipt_digest,
            receipt_state=_receipt_state_from_predecessor(exc.predecessor_state),
        ) from None
    if (
        state.receipt is None
        or state.receipt_raw is None
        or (
            not receipt_alias
            and (
                state.records[-1]["state"] != "RECEIPT_FINALIZED"
                or state.record_names[-1] != final_target_name
            )
        )
    ):
        raise PersistenceError("publication alias lacks a canonical sealed chain")

    receipt_digest = str(state.receipt["receipt_digest"])
    receipt_state = str(state.receipt["receipt_state"])
    try:
        _validate_permission_prefix(
            key_directory=key_directory,
            records_directory=records_directory,
            state=state,
        )
    except PersistenceError:
        raise StoreSealedUnavailable(
            operation,
            receipt_digest=receipt_digest,
            receipt_state=receipt_state,
        ) from None

    target_parent = key_directory if receipt_alias else records_directory
    target_name = "receipt.json" if receipt_alias else final_target_name
    if target_name is None:
        raise PersistenceError("publication alias target is unavailable")
    target_size = (
        len(state.receipt_raw)
        if receipt_alias
        else len(canonical_bytes(state.records[-1]))
    )

    key_fd: int | None = None
    target_parent_fd: int | None = None
    primary: BaseException | None = None
    try:
        key_fd = _open_directory(key_directory, {0o700, 0o500})
        target_parent_fd = _open_directory(target_parent, {0o700, 0o500})
        target_before = os.stat(
            target_name,
            dir_fd=target_parent_fd,
            follow_symlinks=False,
        )
        stage_before = os.stat(
            stage_name, dir_fd=key_fd, follow_symlinks=False
        )
        _validate_receipt_alias_entry(target_before, label=target_name)
        _validate_receipt_alias_entry(stage_before, label=stage_name)
        if (
            (target_before.st_dev, target_before.st_ino)
            != (stage_before.st_dev, stage_before.st_ino)
            or target_before.st_size != target_size
        ):
            raise PersistenceError("sealed publication aliases changed")

        # Establish the canonical target name durably before removing its alias.
        os.fsync(target_parent_fd)
        target_synced = os.stat(
            target_name,
            dir_fd=target_parent_fd,
            follow_symlinks=False,
        )
        stage_synced = os.stat(
            stage_name, dir_fd=key_fd, follow_symlinks=False
        )
        _validate_receipt_alias_entry(target_synced, label=target_name)
        _validate_receipt_alias_entry(stage_synced, label=stage_name)
        if len(
            {
                (target_before.st_dev, target_before.st_ino),
                (stage_before.st_dev, stage_before.st_ino),
                (target_synced.st_dev, target_synced.st_ino),
                (stage_synced.st_dev, stage_synced.st_ino),
            }
        ) != 1:
            raise PersistenceError("sealed publication aliases changed")

        os.unlink(stage_name, dir_fd=key_fd)
        os.fsync(key_fd)
        target_after = os.stat(
            target_name,
            dir_fd=target_parent_fd,
            follow_symlinks=False,
        )
        _validate_regular_entry(
            target_after,
            modes={0o600},
            label=target_name,
        )
        if (
            (target_after.st_dev, target_after.st_ino)
            != (target_before.st_dev, target_before.st_ino)
            or target_after.st_size != target_size
        ):
            raise PersistenceError("repaired sealed target changed")
        try:
            os.stat(stage_name, dir_fd=key_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise PersistenceError("receipt publication alias still exists")
    except BaseException as exc:
        primary = exc
        if not isinstance(exc, Exception):
            raise
        raise StoreSealedUnavailable(
            operation,
            receipt_digest=receipt_digest,
            receipt_state=receipt_state,
        ) from None
    finally:
        cleanup_errors: list[BaseException] = []
        for fd in (target_parent_fd, key_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if cleanup_errors and primary is None:
            if not isinstance(cleanup_errors[0], Exception):
                raise cleanup_errors[0]
            raise StoreSealedUnavailable(
                operation,
                receipt_digest=receipt_digest,
                receipt_state=receipt_state,
            ) from None

    try:
        repaired = _load_stored_state(
            key_directory,
            records_directory,
            operation,
            require_operation_match=require_operation_match,
        )
    except _FinalReceiptUnavailable as exc:
        raise StoreSealedUnavailable(
            operation,
            receipt_digest=exc.receipt_digest,
            receipt_state=_receipt_state_from_predecessor(exc.predecessor_state),
        ) from None
    except BaseException as exc:
        if not isinstance(exc, Exception):
            raise
        raise StoreSealedUnavailable(
            operation,
            receipt_digest=receipt_digest,
            receipt_state=receipt_state,
        ) from None
    if (
        repaired.receipt_raw != state.receipt_raw
        or repaired.receipt is None
        or repaired.receipt["receipt_digest"] != receipt_digest
    ):
        raise StoreSealedUnavailable(
            operation,
            receipt_digest=receipt_digest,
            receipt_state=receipt_state,
        ) from None
    return True


def _recover_existing(owner: OwnerHandle) -> bytes:
    try:
        state = _load_stored_state(
            owner.key_directory,
            owner.records_directory,
            owner.operation,
        )
    except _FinalReceiptUnavailable as exc:
        raise StoreSealedUnavailable(
            owner.operation,
            receipt_digest=exc.receipt_digest,
            receipt_state=_receipt_state_from_predecessor(exc.predecessor_state),
        ) from None
    except (PersistenceError, ContractValidationError, OSError):
        raise StoreIndeterminate(owner.operation) from None
    owner.records = state.records
    if state.receipt_raw is not None and state.receipt is not None:
        try:
            _validate_permission_prefix(
                key_directory=owner.key_directory,
                records_directory=owner.records_directory,
                state=state,
            )
        except PersistenceError:
            raise StoreSealedUnavailable(
                owner.operation,
                receipt_digest=str(state.receipt["receipt_digest"]),
                receipt_state=str(state.receipt["receipt_state"]),
            ) from None
        owner.sealed_raw = state.receipt_raw
        owner.sealed_receipt = state.receipt
        owner._receipt_publication_complete = True
        try:
            owner._attempt_finalize_once()
        except Exception:
            if not owner._safe_revalidate_sealed():
                raise StoreSealedUnavailable(
                    owner.operation,
                    receipt_digest=str(state.receipt["receipt_digest"]),
                    receipt_state=str(state.receipt["receipt_state"]),
                ) from None
        return state.receipt_raw

    if state.records[-1]["state"] == "RECEIPT_FINALIZED":
        predecessor = state.records[-2]
        raise StoreSealedUnavailable(
            owner.operation,
            receipt_digest=str(state.records[-1]["receipt_digest"]),
            receipt_state=_receipt_state_from_predecessor(str(predecessor["state"])),
        )
    try:
        _validate_active_permissions(state)
    except PersistenceError:
        raise StoreIndeterminate(owner.operation) from None

    last = owner.last_record
    last_state = str(last["state"])
    if last_state in {"OWNERSHIP_ACQUIRED", "PREINVOKE_REJECTED"} and (
        _matching_evidence_exists(owner.evidence_destination)
    ):
        raise StoreIndeterminate(owner.operation) from None
    if last_state == "OWNERSHIP_ACQUIRED":
        try:
            owner.append(
                "PREINVOKE_REJECTED",
                reason_code="RECOVERED_BEFORE_INVOCATION_CLAIM",
            )
        except Exception:
            try:
                owner.refresh_active()
            except StoreIndeterminate:
                raise StoreIndeterminate(owner.operation) from None
            if owner.last_record["state"] != "PREINVOKE_REJECTED":
                raise StorePreInvokeUnavailable(owner.operation) from None
        try:
            return owner.seal(
                receipt_state="REJECTED_PRE_INVOKE",
                reason_code="RECOVERED_BEFORE_INVOCATION_CLAIM",
            )
        except StoreSealedUnavailable:
            raise
        except Exception:
            raise StorePreInvokeUnavailable(owner.operation) from None
    if last_state == "PREINVOKE_REJECTED":
        try:
            return owner.seal(
                receipt_state="REJECTED_PRE_INVOKE",
                reason_code=str(last["reason_code"]),
            )
        except StoreSealedUnavailable:
            raise
        except Exception:
            raise StorePreInvokeUnavailable(owner.operation) from None
    if last_state in {"INVOCATION_CLAIMED", "ENGINE_EVIDENCE_VERIFIED"}:
        try:
            owner.append(
                "INDETERMINATE_NO_RETRY",
                reason_code="RECOVERED_AFTER_INVOCATION_CLAIM",
            )
        except Exception:
            try:
                owner.refresh_active()
            except StoreIndeterminate:
                raise StoreIndeterminateUnavailable(owner.operation) from None
            if owner.last_record["state"] != "INDETERMINATE_NO_RETRY":
                raise StoreIndeterminateUnavailable(owner.operation) from None
        try:
            return owner.seal(
                receipt_state="INDETERMINATE_NO_RETRY",
                reason_code="RECOVERED_AFTER_INVOCATION_CLAIM",
            )
        except StoreSealedUnavailable:
            raise
        except Exception:
            raise StoreIndeterminateUnavailable(owner.operation) from None
    if last_state == "INDETERMINATE_NO_RETRY":
        try:
            return owner.seal(
                receipt_state="INDETERMINATE_NO_RETRY",
                reason_code=str(last["reason_code"]),
            )
        except StoreSealedUnavailable:
            raise
        except Exception:
            raise StoreIndeterminateUnavailable(owner.operation) from None
    raise StoreIndeterminate(owner.operation)


def _matching_evidence_exists(destination: Path) -> bool:
    shard = destination.parent
    prefix = f".{destination.name}."
    try:
        return _directory_has_matching_name(
            shard,
            lambda name: name == destination.name
            or (name.startswith(prefix) and name.endswith(".staging")),
        )
    except (PersistenceError, OSError):
        return True


def _release_flock_fd(fd: int) -> BaseException | None:
    """Release one flock/fd once and return, rather than raise, cleanup error."""

    first_error: BaseException | None = None
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except BaseException as exc:
        first_error = exc
    try:
        os.close(fd)
    except BaseException as exc:
        if first_error is None:
            first_error = exc
    return first_error


def _seal_known_preclaim(owner: OwnerHandle, reason_code: str) -> bytes:
    if owner._store_ambiguous:
        raise StoreIndeterminate(owner.operation)
    state = str(owner.last_record["state"])
    if state == "OWNERSHIP_ACQUIRED":
        owner.append("PREINVOKE_REJECTED", reason_code=reason_code)
    elif state != "PREINVOKE_REJECTED":
        raise StoreIndeterminate(owner.operation)
    persisted_reason = str(owner.last_record["reason_code"])
    return owner.seal(
        receipt_state="REJECTED_PRE_INVOKE",
        reason_code=persisted_reason,
    )


def _best_effort_known_preclaim(owner: OwnerHandle, reason_code: str) -> None:
    try:
        _seal_known_preclaim(owner, reason_code)
    except BaseException:
        pass


class ReceiptStore:
    """Acquire one idempotency key or return a replay/conflict response."""

    def __init__(self, roots: RootSet) -> None:
        self.roots = roots

    def claim(
        self,
        operation: Operation,
        *,
        fresh_preflight: Callable[[], Any],
    ) -> ClaimResult:
        registry_key = (
            self.roots.receipt.device,
            self.roots.receipt.inode,
            operation.idempotency_key_digest,
        )
        active = _registry_lookup(registry_key)
        if active is not None:
            try:
                _validate_active_registry_store(
                    self.roots,
                    active.global_lock_identity,
                )
            except (RootSafetyError, PersistenceError, OSError):
                raise StoreIndeterminate(operation) from None
            return ClaimResult(
                response=transient_conflict(
                    operation,
                    bound_operation_digest=active.operation_digest,
                    ownership_record_digest=active.ownership_record_digest,
                )
            )

        root_identity = self.roots.receipt.identity
        root_mutex = _root_mutex_acquire(root_identity)
        global_fd: int | None = None
        global_lock_identity: tuple[int, int] | None = None
        key_lock_fd: int | None = None
        release_root = True

        def release_outer_locks() -> BaseException | None:
            nonlocal global_fd, release_root
            first_error: BaseException | None = None
            if global_fd is not None:
                releasing_global = global_fd
                global_fd = None
                error = _release_flock_fd(releasing_global)
                if error is not None:
                    first_error = error
            if release_root:
                release_root = False
                try:
                    _root_mutex_release(root_identity, root_mutex)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
            return first_error

        try:
            self.roots.revalidate_all()
            root_state = _validate_receipt_root_entries(self.roots.receipt)
            if root_state == "missing-lock":
                raise StoreIndeterminate(operation)
            try:
                global_fd = _create_lock_file(
                    self.roots.receipt,
                    allow_create=root_state == "virgin",
                )
                fcntl.flock(global_fd, fcntl.LOCK_EX)
                _revalidate_named_lock(self.roots.receipt, global_fd)
            except (PersistenceError, OSError):
                if root_state == "established":
                    raise StoreIndeterminate(operation) from None
                raise
            self.roots.revalidate_all()
            post_lock_state = _validate_receipt_root_entries(self.roots.receipt)
            if post_lock_state not in {"lock-only", "established"}:
                raise StoreIndeterminate(operation)
            _revalidate_named_lock(self.roots.receipt, global_fd)
            global_lock_entry = os.fstat(global_fd)
            global_lock_identity = (
                global_lock_entry.st_dev,
                global_lock_entry.st_ino,
            )
            active = _registry_lookup(registry_key)
            if active is not None:
                if active.global_lock_identity != global_lock_identity:
                    raise StoreIndeterminate(operation)
                return ClaimResult(
                    response=transient_conflict(
                        operation,
                        bound_operation_digest=active.operation_digest,
                        ownership_record_digest=active.ownership_record_digest,
                    )
                )

            v1 = _ensure_directory(self.roots.receipt.path, "v1")
            shard = _ensure_directory(v1, operation.idempotency_key_digest[:2])
            key_directory = shard / operation.idempotency_key_digest
            try:
                key_entry = os.lstat(key_directory)
            except FileNotFoundError:
                preflight_result = fresh_preflight()
                evidence_destination = _prepare_evidence_destination(
                    self.roots, operation
                )
                self.roots.revalidate_all()
                key_directory = _create_directory(
                    shard, operation.idempotency_key_digest
                )
                key_lock_fd = _create_permanent_lock(key_directory)
                records_directory = _create_directory(key_directory, "records")
                try:
                    fcntl.flock(
                        key_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                except OSError as exc:
                    raise PersistenceError("fresh per-key flock failed") from exc
                record, raw = build_record(
                    operation,
                    sequence=1,
                    state="OWNERSHIP_ACQUIRED",
                    previous_record_digest=None,
                )
                publication = _PublicationAttempt()
                publication_error: BaseException | None = None
                try:
                    _publish_no_replace(
                        key_directory=key_directory,
                        target_parent=records_directory,
                        target_name=record_filename(record),
                        raw=raw,
                        attempt=publication,
                    )
                except BaseException as exc:
                    if not publication.complete:
                        if publication.target_observed:
                            if isinstance(exc, Exception):
                                raise StoreIndeterminate(operation) from None
                            raise
                        raise
                    publication_error = exc
                token = object()
                owner = OwnerHandle(
                    roots=self.roots,
                    operation=operation,
                    key_directory=key_directory,
                    records_directory=records_directory,
                    evidence_destination=evidence_destination,
                    key_lock_fd=key_lock_fd,
                    registry_key=registry_key,
                    registry_token=token,
                    records=[record],
                    preflight_result=preflight_result,
                )
                key_lock_fd = None
                try:
                    _registry_insert(
                        registry_key,
                        token=token,
                        operation_digest=operation.operation_digest,
                        ownership_record_digest=str(record["record_digest"]),
                        global_lock_identity=global_lock_identity,
                    )
                except BaseException as exc:
                    if publication_error is None or isinstance(
                        publication_error, Exception
                    ):
                        publication_error = exc
                if publication_error is not None:
                    if not isinstance(publication_error, Exception):
                        _best_effort_known_preclaim(
                            owner, "CANCELLED_BEFORE_INVOCATION_CLAIM"
                        )
                        release_outer_locks()
                        owner.close(_primary=publication_error)
                        raise publication_error
                    try:
                        response = _seal_known_preclaim(
                            owner, "PERSISTENCE_CORRUPTION"
                        )
                    except StoreSealedUnavailable as exc:
                        owner.close(_primary=exc)
                        raise
                    except BaseException as exc:
                        owner.close(_primary=exc)
                        if not isinstance(exc, Exception):
                            raise
                        raise StorePreInvokeUnavailable(operation) from None
                    release_error = release_outer_locks()
                    owner.close(_primary=release_error or publication_error)
                    if release_error is not None and not isinstance(
                        release_error, Exception
                    ):
                        raise release_error
                    return ClaimResult(response=response)
                release_error = release_outer_locks()
                if release_error is None:
                    return ClaimResult(owner=owner)
                if not isinstance(release_error, Exception):
                    _best_effort_known_preclaim(
                        owner, "CANCELLED_BEFORE_INVOCATION_CLAIM"
                    )
                    owner.close(_primary=release_error)
                    raise release_error
                try:
                    response = _seal_known_preclaim(
                        owner, "PERSISTENCE_CORRUPTION"
                    )
                except StoreSealedUnavailable as exc:
                    owner.close(_primary=exc)
                    raise
                except BaseException as exc:
                    owner.close(_primary=exc)
                    if not isinstance(exc, Exception):
                        raise
                    raise StorePreInvokeUnavailable(operation) from None
                owner.close(_primary=release_error)
                return ClaimResult(response=response)
            except OSError as exc:
                raise PersistenceError("key lookup failed") from exc

            if (
                not stat.S_ISDIR(key_entry.st_mode)
                or stat.S_ISLNK(key_entry.st_mode)
                or key_entry.st_uid != _EUID
                or _mode(key_entry) not in {0o700, 0o500}
            ):
                raise StoreIndeterminate(operation)
            records_directory = key_directory / "records"
            try:
                records_fd = _open_directory(
                    records_directory, {0o700, 0o500}
                )
            except PersistenceError:
                raise StoreIndeterminate(operation) from None
            else:
                os.close(records_fd)
            try:
                anchor = _load_anchor(records_directory)
                if anchor["idempotency_key_digest"] != operation.idempotency_key_digest:
                    raise ContractValidationError("anchor key digest mismatch")
                key_lock_fd = _open_permanent_lock(key_directory)
            except (PersistenceError, ContractValidationError, OSError):
                raise StoreIndeterminate(operation) from None
            try:
                fcntl.flock(key_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK}:
                    return ClaimResult(
                        response=transient_conflict(
                            operation,
                            bound_operation_digest=str(anchor["operation_digest"]),
                            ownership_record_digest=str(anchor["record_digest"]),
                        )
                    )
                raise StoreIndeterminate(operation) from None
            try:
                _repair_committed_receipt_stage(
                    key_directory=key_directory,
                    records_directory=records_directory,
                    operation=operation,
                    require_operation_match=(
                        anchor["operation_digest"] == operation.operation_digest
                    ),
                )
            except StoreSealedUnavailable:
                raise
            except (PersistenceError, ContractValidationError, OSError):
                raise StoreIndeterminate(operation) from None
            if anchor["operation_digest"] != operation.operation_digest:
                try:
                    stored = _load_stored_state(
                        key_directory,
                        records_directory,
                        operation,
                        require_operation_match=False,
                    )
                except _FinalReceiptUnavailable as exc:
                    raise StoreSealedUnavailable(
                        operation,
                        receipt_digest=exc.receipt_digest,
                        receipt_state=_receipt_state_from_predecessor(
                            exc.predecessor_state
                        ),
                    ) from None
                except (PersistenceError, ContractValidationError, OSError):
                    raise StoreIndeterminate(operation) from None
                if stored.receipt is None:
                    try:
                        _validate_active_permissions(stored)
                    except PersistenceError:
                        raise StoreIndeterminate(operation) from None
                else:
                    try:
                        _validate_permission_prefix(
                            key_directory=key_directory,
                            records_directory=records_directory,
                            state=stored,
                        )
                    except PersistenceError:
                        raise StoreSealedUnavailable(
                            operation,
                            receipt_digest=str(stored.receipt["receipt_digest"]),
                            receipt_state=str(stored.receipt["receipt_state"]),
                        ) from None
                if str(stored.records[-1]["state"]) in {
                    "OWNERSHIP_ACQUIRED",
                    "PREINVOKE_REJECTED",
                } and _matching_evidence_exists(
                    self.roots.evidence.path
                    / "v1"
                    / operation.idempotency_key_digest[:2]
                    / f"phase0-{operation.idempotency_key_digest}"
                ):
                    raise StoreIndeterminate(operation) from None
                return ClaimResult(
                    response=transient_conflict(
                        operation,
                        bound_operation_digest=str(anchor["operation_digest"]),
                        ownership_record_digest=str(anchor["record_digest"]),
                    )
                )
            token = object()
            owner = OwnerHandle(
                roots=self.roots,
                operation=operation,
                key_directory=key_directory,
                records_directory=records_directory,
                evidence_destination=(
                    self.roots.evidence.path
                    / "v1"
                    / operation.idempotency_key_digest[:2]
                    / f"phase0-{operation.idempotency_key_digest}"
                ),
                key_lock_fd=key_lock_fd,
                registry_key=registry_key,
                registry_token=token,
                records=[anchor],
            )
            key_lock_fd = None
            try:
                _registry_insert(
                    registry_key,
                    token=token,
                    operation_digest=str(anchor["operation_digest"]),
                    ownership_record_digest=str(anchor["record_digest"]),
                    global_lock_identity=global_lock_identity,
                )
            except BaseException as exc:
                owner.close(_primary=exc)
                if isinstance(exc, Exception):
                    raise StoreIndeterminate(operation) from None
                raise
            release_error = release_outer_locks()
            if release_error is not None:
                owner.close(_primary=release_error)
                if isinstance(release_error, Exception):
                    raise StoreIndeterminate(operation) from None
                raise release_error
            with owner:
                response = _recover_existing(owner)
            return ClaimResult(response=response)
        finally:
            primary = sys.exc_info()[1]
            cleanup_errors: list[BaseException] = []
            if key_lock_fd is not None:
                releasing_key = key_lock_fd
                key_lock_fd = None
                error = _release_flock_fd(releasing_key)
                if error is not None:
                    cleanup_errors.append(error)
            error = release_outer_locks()
            if error is not None:
                cleanup_errors.append(error)
            if cleanup_errors and primary is None:
                raise cleanup_errors[0]
