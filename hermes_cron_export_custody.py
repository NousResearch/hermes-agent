"""Profile-local cron custody for bounded control-store capture.

Cron writers take a shared lease while they update their own durable state.  A
capture takes an exclusive lease for the whole read and publication window.
The lock is a coordination boundary only: SQLite writer fences and the outer
cron drain still prove the actual export composition.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass
import ctypes
import errno
import math
import os
from pathlib import Path
import stat
import threading
import time
from typing import Final, Iterator, Literal

from hermes_constants import get_hermes_home


COORDINATION_FILENAME: Final = ".cron-export-custody.lock"
DEFAULT_TIMEOUT_SECONDS: Final = 5.0
MAX_TIMEOUT_SECONDS: Final = 120.0
_POLL_INTERVAL_SECONDS: Final = 0.02
_FILE_ATTRIBUTE_REPARSE_POINT: Final = getattr(
    stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
)


class CronCustodyRefused(RuntimeError):
    """A profile custody lease could not be acquired, used, or released."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class _FilesystemIdentity:
    device: int
    inode: int
    file_type: int
    permissions: int
    owner: int
    group: int
    attributes: int
    reparse: bool
    generation: int = 0

    @property
    def stable(self) -> tuple[int, int, int, int, int, bool]:
        return (
            self.device,
            self.inode,
            self.file_type,
            self.owner,
            self.group,
            self.reparse,
        )


@dataclass(frozen=True, slots=True)
class _ProfileSnapshot:
    root: Path
    ancestry: tuple[tuple[str, _FilesystemIdentity], ...]


@dataclass(slots=True)
class _OpenCoordinationFile:
    profile: _ProfileSnapshot
    path: Path
    descriptor: int
    identity: _FilesystemIdentity
    native_aux: object | None = None
    locked: bool = False


class _LockBusy(Exception):
    """The nonblocking native lock attempt found another holder."""


@dataclass(slots=True)
class _WindowsOverlapped:
    """Opaque holder for a LockFileEx byte-range operation."""

    struct: ctypes.Structure
    kernel32: object
    handle: int


if os.name == "nt":

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", ctypes.c_uint32),
            ("OffsetHigh", ctypes.c_uint32),
            ("hEvent", ctypes.c_void_p),
        ]

else:
    _OVERLAPPED = None


def _timeout(value: object) -> float:
    if type(value) not in (int, float):
        raise CronCustodyRefused("cron_custody_timeout_invalid")
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= MAX_TIMEOUT_SECONDS:
        raise CronCustodyRefused("cron_custody_timeout_invalid")
    return result


def _identity(
    metadata: os.stat_result, *, include_generation: bool = False
) -> _FilesystemIdentity:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return _FilesystemIdentity(
        device=int(getattr(metadata, "st_dev", -1)),
        inode=int(getattr(metadata, "st_ino", -1)),
        file_type=stat.S_IFMT(metadata.st_mode),
        permissions=stat.S_IMODE(metadata.st_mode),
        owner=int(getattr(metadata, "st_uid", -1)),
        group=int(getattr(metadata, "st_gid", -1)),
        attributes=attributes,
        reparse=stat.S_ISLNK(metadata.st_mode)
        or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT),
        generation=(
            int(getattr(metadata, "st_ctime_ns", 0)) if include_generation else 0
        ),
    )


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _lexical_ancestors(path: Path) -> tuple[Path, ...]:
    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    result = [current]
    for component in absolute.parts:
        if component == absolute.anchor:
            continue
        current = current / component
        result.append(current)
    return tuple(result)


def _directory_entries(
    path: Path, *, code: str, allow_alias: bool
) -> tuple[tuple[str, _FilesystemIdentity], ...]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CronCustodyRefused("cron_custody_profile_unavailable") from exc
    identity = _identity(metadata)
    if identity.reparse:
        if not allow_alias:
            raise CronCustodyRefused("cron_custody_profile_symlink")
        identity = _identity(metadata, include_generation=True)
        try:
            target = path.resolve(strict=True)
            target_metadata = target.stat()
        except (OSError, RuntimeError) as exc:
            raise CronCustodyRefused("cron_custody_profile_ancestry_invalid") from exc
        target_identity = _identity(target_metadata)
        if target_identity.reparse or target_identity.file_type != stat.S_IFDIR:
            raise CronCustodyRefused("cron_custody_profile_ancestry_invalid")
        physical_entries: list[tuple[str, _FilesystemIdentity]] = []
        for candidate in _lexical_ancestors(target):
            if candidate == target:
                physical_entries.append((str(candidate), target_identity))
                continue
            try:
                candidate_metadata = candidate.lstat()
            except OSError as exc:
                raise CronCustodyRefused(
                    "cron_custody_profile_ancestry_invalid"
                ) from exc
            candidate_identity = _identity(candidate_metadata)
            if (
                candidate_identity.reparse
                or candidate_identity.file_type != stat.S_IFDIR
            ):
                raise CronCustodyRefused("cron_custody_profile_ancestry_invalid")
            physical_entries.append((str(candidate), candidate_identity))
        return ((str(path), identity), *physical_entries)
    if identity.file_type != stat.S_IFDIR:
        raise CronCustodyRefused(code)
    return ((str(path), identity),)


def _profile_ancestry(root: Path) -> tuple[tuple[str, _FilesystemIdentity], ...]:
    ancestry: list[tuple[str, _FilesystemIdentity]] = []
    for candidate in _lexical_ancestors(root):
        ancestry.extend(
            _directory_entries(
                candidate,
                code=(
                    "cron_custody_profile_not_directory"
                    if candidate == root
                    else "cron_custody_profile_ancestry_invalid"
                ),
                allow_alias=candidate != root,
            )
        )
    return tuple(ancestry)


def _profile_snapshot() -> _ProfileSnapshot:
    try:
        configured = get_hermes_home()
        root = Path(configured)
    except (TypeError, ValueError, OSError) as exc:
        raise CronCustodyRefused("cron_custody_profile_unavailable") from exc
    if not root.is_absolute():
        raise CronCustodyRefused("cron_custody_profile_invalid")
    root = _absolute_path(root)
    return _ProfileSnapshot(root=root, ancestry=_profile_ancestry(root))


def _assert_profile_current(profile: _ProfileSnapshot) -> None:
    if _profile_ancestry(profile.root) != profile.ancestry:
        raise CronCustodyRefused("cron_custody_profile_changed")


def _coordination_identity(path: Path) -> _FilesystemIdentity | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CronCustodyRefused("cron_custody_lock_unavailable") from exc
    identity = _identity(metadata)
    if identity.reparse:
        raise CronCustodyRefused("cron_custody_lock_symlink")
    if identity.file_type != stat.S_IFREG:
        raise CronCustodyRefused("cron_custody_lock_not_regular")
    return identity


def _same_inode_identity(
    left: _FilesystemIdentity, right: _FilesystemIdentity
) -> bool:
    return left.stable == right.stable


def _same_open_identity(
    left: _FilesystemIdentity, right: _FilesystemIdentity
) -> bool:
    return left == right


def _open_coordination_file(profile: _ProfileSnapshot) -> _OpenCoordinationFile:
    _assert_profile_current(profile)
    path = profile.root / COORDINATION_FILENAME
    before = _coordination_identity(path)
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    created_by_us = False
    try:
        if before is None:
            try:
                descriptor = os.open(
                    path,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                created_by_us = True
            except FileExistsError:
                descriptor = os.open(path, flags)
        else:
            descriptor = os.open(path, flags)
        os.set_inheritable(descriptor, False)
        if os.name != "nt" and created_by_us:
            os.fchmod(descriptor, 0o600)
        after = _coordination_identity(path)
        descriptor_identity = _identity(os.fstat(descriptor))
        if after is None or not _same_open_identity(after, descriptor_identity):
            raise CronCustodyRefused("cron_custody_lock_changed")
        if before is not None and not _same_inode_identity(before, after):
            raise CronCustodyRefused("cron_custody_lock_changed")
        if os.name != "nt" and after.permissions & 0o077:
            raise CronCustodyRefused("cron_custody_lock_permissions")
        _assert_profile_current(profile)
        return _OpenCoordinationFile(
            profile=profile,
            path=path,
            descriptor=descriptor,
            identity=after,
        )
    except CronCustodyRefused:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        raise CronCustodyRefused("cron_custody_lock_unavailable") from exc


def _assert_open_file_current(handle: _OpenCoordinationFile) -> None:
    _assert_profile_current(handle.profile)
    path_identity = _coordination_identity(handle.path)
    if path_identity is None or not _same_open_identity(path_identity, handle.identity):
        raise CronCustodyRefused("cron_custody_lock_changed")
    try:
        descriptor_identity = _identity(os.fstat(handle.descriptor))
    except OSError as exc:
        raise CronCustodyRefused("cron_custody_lock_unavailable") from exc
    if not _same_open_identity(descriptor_identity, handle.identity):
        raise CronCustodyRefused("cron_custody_lock_changed")


def _try_posix_lock(descriptor: int, *, exclusive: bool) -> None:
    try:
        import fcntl

        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
    except ImportError as exc:
        raise CronCustodyRefused("cron_custody_backend_unavailable") from exc
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
            raise _LockBusy from exc
        raise CronCustodyRefused("cron_custody_lock_failed") from exc


def _windows_api() -> object:
    if _OVERLAPPED is None:
        raise CronCustodyRefused("cron_custody_backend_unavailable")
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        from ctypes import wintypes

        kernel32.LockFileEx.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_OVERLAPPED),
        ]
        kernel32.LockFileEx.restype = wintypes.BOOL
        kernel32.UnlockFileEx.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_OVERLAPPED),
        ]
        kernel32.UnlockFileEx.restype = wintypes.BOOL
        return kernel32
    except (AttributeError, OSError) as exc:
        raise CronCustodyRefused("cron_custody_backend_unavailable") from exc


def _try_windows_lock(
    descriptor: int, *, exclusive: bool
) -> _WindowsOverlapped:
    import msvcrt

    kernel32 = _windows_api()
    handle = msvcrt.get_osfhandle(descriptor)
    if handle == -1:
        raise CronCustodyRefused("cron_custody_lock_failed")
    overlapped = _OVERLAPPED()
    flags = 0x00000001
    if exclusive:
        flags |= 0x00000002
    if not kernel32.LockFileEx(
        ctypes.c_void_p(handle), flags, 0, 1, 0, ctypes.byref(overlapped)
    ):
        error = ctypes.get_last_error()
        if error in {32, 33}:
            raise _LockBusy
        raise CronCustodyRefused("cron_custody_lock_failed")
    return _WindowsOverlapped(overlapped, kernel32, handle)


def _try_lock(
    descriptor: int, *, exclusive: bool
) -> object | None:
    if os.name == "nt":
        return _try_windows_lock(descriptor, exclusive=exclusive)
    if os.name == "posix":
        _try_posix_lock(descriptor, exclusive=exclusive)
        return None
    raise CronCustodyRefused("cron_custody_backend_unavailable")


def _unlock(handle: _OpenCoordinationFile) -> None:
    if not handle.locked:
        return
    if os.name == "nt":
        aux = handle.native_aux
        if not isinstance(aux, _WindowsOverlapped):
            return
        if not aux.kernel32.UnlockFileEx(
            ctypes.c_void_p(aux.handle),
            0,
            1,
            0,
            ctypes.byref(aux.struct),
        ):
            raise OSError(ctypes.get_last_error(), "UnlockFileEx failed")
        return
    try:
        import fcntl

        fcntl.flock(handle.descriptor, fcntl.LOCK_UN)
    except ImportError as exc:
        raise OSError("fcntl unavailable during unlock") from exc


def _acquire_native(
    handle: _OpenCoordinationFile, *, exclusive: bool, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    wake = threading.Event()
    while True:
        _assert_open_file_current(handle)
        try:
            handle.native_aux = _try_lock(handle.descriptor, exclusive=exclusive)
            handle.locked = True
            _assert_open_file_current(handle)
            return
        except _LockBusy:
            if time.monotonic() >= deadline:
                raise CronCustodyRefused("cron_custody_busy")
            wake.wait(min(_POLL_INTERVAL_SECONDS, deadline - time.monotonic()))
        except CronCustodyRefused:
            raise
        except OSError as exc:
            raise CronCustodyRefused("cron_custody_lock_failed") from exc


def _close_open_file(handle: _OpenCoordinationFile) -> None:
    try:
        with suppress(OSError):
            _unlock(handle)
    finally:
        handle.locked = False
        with suppress(OSError):
            os.close(handle.descriptor)


@dataclass(slots=True)
class CronCustodyLease:
    """An owner-bound shared or exclusive profile custody lease."""

    _handle: _OpenCoordinationFile
    mode: Literal["shared", "exclusive"]
    _closed: bool = False

    @property
    def profile_root(self) -> Path:
        return self._handle.profile.root

    @property
    def lock_path(self) -> Path:
        return self._handle.path

    def assert_current(self) -> None:
        if self._closed:
            raise CronCustodyRefused("cron_custody_released")
        _assert_open_file_current(self._handle)

    def check_current(self) -> None:
        """Alias for consumers that use a check-style custody seam."""

        self.assert_current()

    def release(self) -> None:
        self._release(verify=True)

    def _release(self, *, verify: bool) -> None:
        if self._closed:
            return
        error: CronCustodyRefused | None = None
        if verify:
            try:
                self.assert_current()
            except CronCustodyRefused as exc:
                error = exc
        _close_open_file(self._handle)
        self._closed = True
        if error is not None:
            raise error


def _acquire_lease(
    *, mode: Literal["shared", "exclusive"], timeout_seconds: object
) -> CronCustodyLease:
    timeout = _timeout(timeout_seconds)
    profile = _profile_snapshot()
    handle: _OpenCoordinationFile | None = None
    try:
        handle = _open_coordination_file(profile)
        _acquire_native(
            handle,
            exclusive=mode == "exclusive",
            timeout=timeout,
        )
        return CronCustodyLease(handle, mode)
    except CronCustodyRefused:
        if handle is not None:
            _close_open_file(handle)
        raise
    except (OSError, TypeError, ValueError) as exc:
        if handle is not None:
            _close_open_file(handle)
        raise CronCustodyRefused("cron_custody_unavailable") from exc


@contextmanager
def cron_writer_custody(
    *, timeout_seconds: object = DEFAULT_TIMEOUT_SECONDS
) -> Iterator[CronCustodyLease]:
    """Hold a bounded shared custody lease for one cron writer operation."""

    lease = _acquire_lease(mode="shared", timeout_seconds=timeout_seconds)
    try:
        yield lease
    except BaseException:
        lease._release(verify=False)
        raise
    else:
        lease._release(verify=True)


@contextmanager
def cron_capture_custody(
    *, timeout_seconds: object = DEFAULT_TIMEOUT_SECONDS
) -> Iterator[CronCustodyLease]:
    """Hold a bounded exclusive custody lease through capture consumption."""

    lease = _acquire_lease(mode="exclusive", timeout_seconds=timeout_seconds)
    try:
        yield lease
    except BaseException:
        lease._release(verify=False)
        raise
    else:
        lease._release(verify=True)


__all__ = [
    "COORDINATION_FILENAME",
    "CronCustodyLease",
    "CronCustodyRefused",
    "cron_capture_custody",
    "cron_writer_custody",
]
