"""Cross-process shared update lock for Hermes update lanes.

The lock identity is derived from the canonical repository identity, while the
lock file lives in a private per-user namespace outside the repository.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
import errno
import hashlib
import math
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
from types import ModuleType

_fcntl: ModuleType | None
try:
    import fcntl as _fcntl_module
except ModuleNotFoundError:  # pragma: no cover - exercised in import subprocess
    _fcntl = None
else:
    _fcntl = _fcntl_module

__all__ = [
    "UpdateLockError",
    "UpdateLockErrorCode",
    "acquire_shared_update_lock",
    "mark_shared_update_lock_post_mutation",
    "shared_update_lock_identity",
    "shared_update_lock_path",
    "shared_update_lock_supported",
]


class UpdateLockErrorCode(str, Enum):
    """Closed failure classes for the shared update lock."""

    INVALID_IDENTITY = "invalid_identity"
    INVALID_TIMEOUT = "invalid_timeout"
    CONTENDED = "contended"
    ACQUISITION_FAILED = "acquisition_failed"
    RELEASE_FAILED = "release_failed"


class UpdateLockError(RuntimeError):
    """Sanitized lock failure with an explicit mutation boundary."""

    def __init__(self, code: UpdateLockErrorCode, message: str, *, boundary=None) -> None:
        from hermes_cli.update_engine import UpdateMutationBoundary

        self.code = code
        self.boundary = boundary or UpdateMutationBoundary.PRE_MUTATION
        super().__init__(message)


_LOCK_STATE = threading.local()
_RUNTIME_UID = os.getuid() if hasattr(os, "getuid") else 0
_LOCK_ROOT = Path(tempfile.gettempdir()) / f"hermes-agent-update-locks-{_RUNTIME_UID}"
_MAX_TIMEOUT_SECONDS = 300.0


def shared_update_lock_identity(repository_identity: os.PathLike[str] | str) -> str:
    """Return the canonical identity used to key the shared update lock."""

    if type(repository_identity) is not str:
        repository_identity = os.fspath(repository_identity)
    if type(repository_identity) is not str or not repository_identity:
        raise UpdateLockError(
            UpdateLockErrorCode.INVALID_IDENTITY,
            "shared update lock requires a repository identity",
        )

    resolved = Path(os.path.realpath(repository_identity))
    try:
        st = resolved.stat()
    except OSError:
        raise UpdateLockError(
            UpdateLockErrorCode.INVALID_IDENTITY,
            "shared update lock requires an existing repository directory",
        ) from None
    if not stat.S_ISDIR(st.st_mode):
        raise UpdateLockError(
            UpdateLockErrorCode.INVALID_IDENTITY,
            "shared update lock requires an existing repository directory",
        )
    return f"dir:{st.st_dev}:{st.st_ino}"


def shared_update_lock_path(repository_identity: os.PathLike[str] | str) -> Path:
    """Return the stable lock-file path for a canonical repository identity."""

    identity = shared_update_lock_identity(repository_identity)
    return _lock_path_for_identity(identity)


def _lock_path_for_identity(identity: str) -> Path:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return _lock_namespace_root() / f"{digest}.lock"


def shared_update_lock_supported() -> bool:
    """Return whether the hardened POSIX lock backend is available."""

    return _fcntl is not None and all(
        hasattr(os, name) for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    )


@contextmanager
def acquire_shared_update_lock(
    repository_identity: os.PathLike[str] | str,
    *,
    timeout_seconds: float = 30.0,
):
    """Acquire the shared update lock for one canonical repository identity."""

    if (
        type(timeout_seconds) not in {int, float}
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds < 0.0
        or timeout_seconds > _MAX_TIMEOUT_SECONDS
    ):
        raise UpdateLockError(
            UpdateLockErrorCode.INVALID_TIMEOUT,
            "shared update lock timeout is invalid",
        )
    if not shared_update_lock_supported():
        raise UpdateLockError(
            UpdateLockErrorCode.ACQUISITION_FAILED,
            "shared update lock is unavailable on this platform",
        )

    identity = shared_update_lock_identity(repository_identity)
    depth = getattr(_LOCK_STATE, "depth", 0)
    if depth > 0 and getattr(_LOCK_STATE, "identity", None) == identity:
        _LOCK_STATE.depth = depth + 1
        try:
            yield
        finally:
            _LOCK_STATE.depth -= 1
        return

    lock_path = _lock_path_for_identity(identity)
    namespace_fd = _open_lock_namespace(lock_path.parent)
    try:
        try:
            fd = os.open(
                lock_path.name,
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=namespace_fd,
            )
        except OSError:
            raise UpdateLockError(
                UpdateLockErrorCode.ACQUISITION_FAILED,
                "shared update lock could not be acquired",
            ) from None
    finally:
        os.close(namespace_fd)

    try:
        _validate_lock_file(fd)
        _lock_file_exclusive(fd, timeout_seconds=float(timeout_seconds))
    except Exception:
        os.close(fd)
        raise

    _LOCK_STATE.identity = identity
    _LOCK_STATE.path = lock_path
    _LOCK_STATE.fd = fd
    _LOCK_STATE.depth = 1
    _LOCK_STATE.boundary = None
    try:
        yield
    finally:
        boundary = getattr(_LOCK_STATE, "boundary", None)
        _LOCK_STATE.depth = 0
        _LOCK_STATE.identity = None
        _LOCK_STATE.path = None
        _LOCK_STATE.fd = None
        _LOCK_STATE.boundary = None
        try:
            _close_lock_fd(fd)
        except OSError:
            from hermes_cli.update_engine import UpdateMutationBoundary

            raise UpdateLockError(
                UpdateLockErrorCode.RELEASE_FAILED,
                "shared update lock release is uncertain",
                boundary=boundary or UpdateMutationBoundary.PRE_MUTATION,
            ) from None


def mark_shared_update_lock_post_mutation() -> None:
    """Mark a held lock as past checkout intent for release error mapping."""

    if getattr(_LOCK_STATE, "depth", 0) <= 0:
        return
    from hermes_cli.update_engine import UpdateMutationBoundary

    _LOCK_STATE.boundary = UpdateMutationBoundary.POST_MUTATION_UNCERTAIN


def _lock_file_exclusive(fd: int, *, timeout_seconds: float) -> None:
    if _fcntl is None:  # Defensive: acquisition rejects unsupported platforms.
        raise UpdateLockError(
            UpdateLockErrorCode.ACQUISITION_FAILED,
            "shared update lock is unavailable on this platform",
        )
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            return
        except (BlockingIOError, OSError) as exc:
            if isinstance(exc, OSError) and exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise UpdateLockError(
                    UpdateLockErrorCode.ACQUISITION_FAILED,
                    "shared update lock could not be acquired",
                ) from None
            if time.monotonic() >= deadline:
                raise UpdateLockError(
                    UpdateLockErrorCode.CONTENDED,
                    "shared update lock is busy",
                ) from None
            time.sleep(0.05)


def _lock_namespace_root() -> Path:
    _LOCK_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    _validate_private_directory(_LOCK_ROOT)
    return _LOCK_ROOT


def _open_lock_namespace(path: Path) -> int:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    expected = _validate_private_directory(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError:
        raise UpdateLockError(
            UpdateLockErrorCode.ACQUISITION_FAILED,
            "shared update lock namespace is unavailable",
        ) from None
    try:
        opened = _validate_private_directory_fd(fd)
        if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise UpdateLockError(
                UpdateLockErrorCode.ACQUISITION_FAILED,
                "shared update lock namespace is unavailable",
            )
    except Exception:
        os.close(fd)
        raise
    return fd


def _validate_private_directory(path: Path) -> os.stat_result:
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError:
        raise UpdateLockError(
            UpdateLockErrorCode.ACQUISITION_FAILED,
            "shared update lock namespace is unavailable",
        ) from None
    if (
        not stat.S_ISDIR(st.st_mode)
        or st.st_uid != _current_uid()
        or st.st_mode & 0o077
    ):
        raise UpdateLockError(
            UpdateLockErrorCode.ACQUISITION_FAILED,
            "shared update lock namespace is unavailable",
        )
    return st


def _validate_private_directory_fd(fd: int) -> os.stat_result:
    try:
        st = os.fstat(fd)
    except OSError:
        raise UpdateLockError(
            UpdateLockErrorCode.ACQUISITION_FAILED,
            "shared update lock namespace is unavailable",
        ) from None
    if (
        not stat.S_ISDIR(st.st_mode)
        or st.st_uid != _current_uid()
        or st.st_mode & 0o077
    ):
        raise UpdateLockError(
            UpdateLockErrorCode.ACQUISITION_FAILED,
            "shared update lock namespace is unavailable",
        )
    return st


def _validate_lock_file(fd: int) -> None:
    try:
        st = os.fstat(fd)
    except OSError:
        raise UpdateLockError(
            UpdateLockErrorCode.ACQUISITION_FAILED,
            "shared update lock could not be acquired",
        ) from None
    if (
        not stat.S_ISREG(st.st_mode)
        or st.st_uid != _current_uid()
        or st.st_mode & 0o077
    ):
        raise UpdateLockError(
            UpdateLockErrorCode.ACQUISITION_FAILED,
            "shared update lock could not be acquired",
        )


def _current_uid() -> int:
    return _RUNTIME_UID


def _close_lock_fd(fd: int) -> None:
    os.close(fd)
