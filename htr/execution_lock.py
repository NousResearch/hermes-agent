"""Run-scoped durable write marker and mutation boundary (Task 23–26C).

Lock-order contract (non-recursive, no bypass flags):
- run_write_barrier holds marker ownership for its lifetime but acquires directory
  coordination (fcntl.flock) only briefly around marker create/release/cleanup.
- disposition execution acquires coordination once on its pinned lock_root_fd,
  calls disposition_unlink_marker and related helpers that require coordination
  already held, and releases in finally on all paths.
- No helper re-acquires coordination on the same directory identity in-process.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import inspect
import json
import os
import socket
import stat
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

_coordination_state = threading.local()
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from htr import paths
from htr.ids import validate_id

F = TypeVar("F", bound=Callable[..., Any])

LOCKS_DIR_NAME = ".execution_locks"
MARKER_SUFFIX = ".marker"
MARKER_MODE = 0o600
LOCK_ROOT_MODE = 0o700

ERROR_OCCUPIED_UNKNOWN = "RUN_EXEC_LOCK_OCCUPIED_UNKNOWN"
ERROR_PATH_UNSAFE = "RUN_EXEC_LOCK_PATH_UNSAFE"
ERROR_UNSUPPORTED = "RUN_EXEC_LOCK_UNSUPPORTED"
ERROR_INIT_FAILED = "RUN_EXEC_LOCK_INIT_FAILED"
ERROR_OWNERSHIP_MISMATCH = "RUN_EXEC_LOCK_OWNERSHIP_MISMATCH"
ERROR_BOUNDARY_VIOLATION = "RUN_EXEC_LOCK_BOUNDARY_VIOLATION"
ERROR_RELEASE_CONFLICT = "RUN_EXEC_LOCK_RELEASE_CONFLICT"
ERROR_INDETERMINATE = "RUN_EXEC_LOCK_INDETERMINATE"
ERROR_DURABILITY_FAILED = "RUN_EXEC_LOCK_DURABILITY_FAILED"

_O_DIRECTORY = os.O_DIRECTORY
_O_RDONLY = os.O_RDONLY
_O_WRONLY = os.O_WRONLY
_O_CREAT = os.O_CREAT
_O_EXCL = os.O_EXCL
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)

_registry_lock = threading.Lock()
_registry: dict[tuple[int, int, str], "_RegistryEntry"] = {}
_active_write_context = threading.local()
_closure_append_local = threading.local()


class RunExecutionLockError(Exception):
    """Base class for Task 23 execution-lock failures."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        run_id: str | None = None,
        mutation_may_have_committed: bool = False,
        safe_to_retry: bool = True,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.run_id = run_id
        self.mutation_may_have_committed = mutation_may_have_committed
        self.safe_to_retry = safe_to_retry


class RunExecutionLockOccupiedError(RunExecutionLockError):
    def __init__(self, message: str = "Run execution marker is occupied.", **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code=kwargs.pop("error_code", ERROR_OCCUPIED_UNKNOWN),
            safe_to_retry=kwargs.pop("safe_to_retry", True),
            **kwargs,
        )


class RunExecutionLockPathUnsafeError(RunExecutionLockError):
    def __init__(self, message: str = "Run execution lock path is unsafe.", **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code=kwargs.pop("error_code", ERROR_PATH_UNSAFE),
            safe_to_retry=kwargs.pop("safe_to_retry", False),
            **kwargs,
        )


class RunExecutionLockUnsupportedError(RunExecutionLockError):
    def __init__(
        self, message: str = "Execution lock unsupported on this platform.", **kwargs: Any
    ) -> None:
        super().__init__(
            message,
            error_code=kwargs.pop("error_code", ERROR_UNSUPPORTED),
            safe_to_retry=kwargs.pop("safe_to_retry", False),
            **kwargs,
        )


class RunExecutionLockIndeterminateError(RunExecutionLockError):
    def __init__(self, message: str = "Execution lock state is indeterminate.", **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code=kwargs.pop("error_code", ERROR_INDETERMINATE),
            safe_to_retry=kwargs.pop("safe_to_retry", False),
            **kwargs,
        )


class RunExecutionLockBoundaryViolationError(RunExecutionLockError):
    def __init__(self, message: str = "Mutation boundary violation.", **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code=kwargs.pop("error_code", ERROR_BOUNDARY_VIOLATION),
            safe_to_retry=kwargs.pop("safe_to_retry", False),
            **kwargs,
        )


class RunExecutionLockReleaseConflictError(RunExecutionLockError):
    def __init__(self, message: str = "Marker release conflict.", **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code=kwargs.pop("error_code", ERROR_RELEASE_CONFLICT),
            safe_to_retry=kwargs.pop("safe_to_retry", False),
            **kwargs,
        )


class RunExecutionLockDurabilityError(RunExecutionLockError):
    def __init__(
        self,
        message: str = "Run mutation may have committed but marker durability failed.",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            error_code=kwargs.pop("error_code", ERROR_DURABILITY_FAILED),
            mutation_may_have_committed=True,
            safe_to_retry=False,
            **kwargs,
        )


@dataclass
class _RegistryEntry:
    key: tuple[int, int, str]
    token: str
    marker_fd: int
    lock_root_fd: int
    owner_pid: int
    owner_thread_id: int
    depth: int = 1
    run_write_started: bool = False
    marker_name: str = ""


@dataclass
class RunWriteContext:
    run_id: str
    base_dir: Path | None
    key: tuple[int, int, str]
    token: str
    is_outermost: bool
    run_write_started: bool = False

    def revalidate_mutation_allowed(self) -> None:
        from htr.finalization import assert_run_mutation_allowed

        assert_run_mutation_allowed(self.run_id, self.base_dir)

    def mark_run_write_started(self) -> None:
        entry = _require_entry(self.key, self.token)
        entry.run_write_started = True
        self.run_write_started = True

    def activate_closure_append_marker(self) -> None:
        _closure_append_local.active = True

    def deactivate_closure_append_marker(self) -> None:
        _closure_append_local.active = False


def _platform_supported() -> bool:
    return os.name == "posix" and bool(_O_NOFOLLOW) and bool(_O_CLOEXEC) and hasattr(os, "O_EXCL")


def _require_platform() -> None:
    if not _platform_supported():
        raise RunExecutionLockUnsupportedError()


def _canonical_runs_root_path(base_dir: Path | None) -> Path:
    if base_dir is not None:
        return Path(os.path.normpath(os.path.abspath(str(base_dir))))
    return Path(os.path.normpath(os.path.abspath(str(paths.default_runs_root()))))


def _registry_key_from_fds(runs_root_fd: int, run_id: str) -> tuple[int, int, str]:
    st = os.fstat(runs_root_fd)
    return (st.st_dev, st.st_ino, run_id)


def _marker_name(run_id: str) -> str:
    validate_id(run_id, "run")
    return f"{run_id}{MARKER_SUFFIX}"


def _open_dir_no_follow(path: Path) -> int:
    return os.open(str(path), _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)


def _open_dirat_no_follow(dir_fd: int, name: str) -> int:
    return os.open(name, _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC, dir_fd=dir_fd)


def _mkdirat(dir_fd: int, name: str, mode: int) -> None:
    try:
        os.mkdir(name, mode, dir_fd=dir_fd)
    except FileExistsError:
        return
    except FileNotFoundError as exc:
        raise RunExecutionLockPathUnsafeError(
            f"cannot create directory component {name!r}"
        ) from exc


def _fsync_dir_fd(dir_fd: int) -> None:
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        raise RunExecutionLockIndeterminateError(f"directory fsync failed: {exc}") from exc


def _fsync_file_fd(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError as exc:
        raise RunExecutionLockIndeterminateError(f"marker fsync failed: {exc}") from exc


def _close_fd_once(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass




def _coordination_identity(lock_root_fd: int) -> tuple[int, int]:
    return lock_directory_identity(lock_root_fd)


def _assert_marker_directory_coordination_held(lock_root_fd: int) -> None:
    identity = _coordination_identity(lock_root_fd)
    holder = getattr(_coordination_state, "holder", None)
    if holder != identity:
        raise RunExecutionLockBoundaryViolationError(
            "marker directory coordination required"
        )


def _release_marker_success_locked(entry: _RegistryEntry) -> None:
    """Release marker after successful body; coordination must already be held."""
    _verify_entry_owner(entry)
    _assert_marker_directory_coordination_held(entry.lock_root_fd)
    if _fstat_identity(entry.marker_fd) != _stat_entry_identity(
        entry.lock_root_fd, entry.marker_name
    ):
        raise RunExecutionLockReleaseConflictError(
            "marker replaced before success release"
        )
    os.unlink(entry.marker_name, dir_fd=entry.lock_root_fd)
    _fsync_dir_fd(entry.lock_root_fd)


def _cleanup_owned_marker_locked(
    lock_root_fd: int, marker_name: str, marker_fd: int
) -> None:
    """Pre-write cleanup; coordination must already be held."""
    _assert_marker_directory_coordination_held(lock_root_fd)
    if _fstat_identity(marker_fd) != _stat_entry_identity(lock_root_fd, marker_name):
        raise RunExecutionLockReleaseConflictError(
            "marker path identity mismatch during pre-write cleanup"
        )
    os.unlink(marker_name, dir_fd=lock_root_fd)
    _fsync_dir_fd(lock_root_fd)


def _fstat_identity(fd: int) -> tuple[int, int]:
    st = os.fstat(fd)
    return st.st_dev, st.st_ino


def _stat_entry_identity(lock_root_fd: int, name: str) -> tuple[int, int]:
    st = os.stat(name, dir_fd=lock_root_fd, follow_symlinks=False)
    return st.st_dev, st.st_ino


def _require_flock_support() -> None:
    if not hasattr(fcntl, "flock"):
        raise RunExecutionLockUnsupportedError(
            "fcntl.flock is required for marker directory-entry coordination"
        )


@contextlib.contextmanager
def _marker_directory_entry_coordination(lock_root_fd: int) -> Iterator[None]:
    """Exclusive flock on pinned .execution_locks directory fd (Task 26C)."""
    acquire_marker_directory_entry_coordination(lock_root_fd)
    try:
        yield
    finally:
        release_marker_directory_entry_coordination(lock_root_fd)


def acquire_marker_directory_entry_coordination(lock_root_fd: int) -> None:
    """Acquire exclusive marker-directory coordination (for disposition execution)."""
    _require_flock_support()
    identity = _coordination_identity(lock_root_fd)
    holder = getattr(_coordination_state, "holder", None)
    if holder is not None:
        if holder == identity:
            raise RunExecutionLockBoundaryViolationError(
                "recursive marker directory coordination acquire"
            )
        raise RunExecutionLockBoundaryViolationError(
            "marker directory coordination already held on different directory"
        )
    try:
        fcntl.flock(lock_root_fd, fcntl.LOCK_EX)
    except OSError as exc:
        raise RunExecutionLockIndeterminateError(
            f"marker directory coordination lock failed: {exc}"
        ) from exc
    _coordination_state.holder = identity


def release_marker_directory_entry_coordination(lock_root_fd: int) -> None:
    """Release marker-directory coordination after unlink + dir fsync."""
    _require_flock_support()
    identity = _coordination_identity(lock_root_fd)
    holder = getattr(_coordination_state, "holder", None)
    if holder != identity:
        raise RunExecutionLockBoundaryViolationError(
            "marker directory coordination release without matching acquire"
        )
    try:
        fcntl.flock(lock_root_fd, fcntl.LOCK_UN)
    except OSError as exc:
        raise RunExecutionLockIndeterminateError(
            f"marker directory coordination unlock failed: {exc}"
        ) from exc
    _coordination_state.holder = None


def pin_lock_directory(base_dir: Path | None) -> tuple[int, int]:
    """Return (runs_root_fd, lock_root_fd) for disposition; caller must close both."""
    return _bootstrap_and_pin(base_dir)


def lock_directory_identity(lock_root_fd: int) -> tuple[int, int]:
    return _fstat_identity(lock_root_fd)


def read_marker_metadata_at(
    lock_root_fd: int,
    run_id: str,
) -> tuple[dict[str, Any], tuple[int, int]]:
    """Read marker JSON and (st_dev, st_ino) via no-follow dir-fd access."""
    marker_name = _marker_name(run_id)
    try:
        st = os.stat(marker_name, dir_fd=lock_root_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise RunExecutionLockIndeterminateError("marker not present") from exc
    if not stat.S_ISREG(st.st_mode):
        raise RunExecutionLockPathUnsafeError("marker is not a regular file")
    flags = _O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC
    fd = os.open(marker_name, flags, dir_fd=lock_root_fd)
    try:
        identity = _fstat_identity(fd)
        if identity != (st.st_dev, st.st_ino):
            raise RunExecutionLockReleaseConflictError("marker identity mismatch on open")
        payload = os.read(fd, 65536)
        metadata = json.loads(payload.decode("utf-8"))
        if not isinstance(metadata, dict):
            raise RunExecutionLockPathUnsafeError("marker metadata is not an object")
        return metadata, identity
    finally:
        _close_fd_once(fd)


def disposition_unlink_marker(
    lock_root_fd: int,
    run_id: str,
    *,
    expected_identity: tuple[int, int],
    expected_acquisition_id: str,
) -> None:
    """Identity-checked unlink under coordination; caller must hold coordination flock."""
    _assert_marker_directory_coordination_held(lock_root_fd)
    marker_name = _marker_name(run_id)
    metadata, identity = read_marker_metadata_at(lock_root_fd, run_id)
    if identity != expected_identity:
        raise RunExecutionLockReleaseConflictError("marker identity changed before unlink")
    if metadata.get("acquisition_id") != expected_acquisition_id:
        raise RunExecutionLockReleaseConflictError("marker acquisition_id mismatch")
    if metadata.get("run_id") != run_id:
        raise RunExecutionLockReleaseConflictError("marker run_id mismatch")
    os.unlink(marker_name, dir_fd=lock_root_fd)
    _fsync_dir_fd(lock_root_fd)


def _thread_active_entry() -> _RegistryEntry | None:
    tid = threading.get_ident()
    pid = os.getpid()
    with _registry_lock:
        for entry in _registry.values():
            if entry.owner_pid == pid and entry.owner_thread_id == tid:
                return entry
    return None


def _find_nested_entry(run_id: str) -> _RegistryEntry | None:
    entry = _thread_active_entry()
    if entry is None:
        return None
    if entry.key[2] != run_id:
        raise RunExecutionLockBoundaryViolationError(
            "cross-key nested mutation is not allowed"
        )
    return entry


def get_active_write_context() -> RunWriteContext | None:
    return getattr(_active_write_context, "current", None)


@contextmanager
def bind_active_write_context(ctx: RunWriteContext) -> Iterator[None]:
    prior = getattr(_active_write_context, "current", None)
    _active_write_context.current = ctx
    try:
        yield
    finally:
        _active_write_context.current = prior


def begin_run_write() -> None:
    """Mark immediately before the first possibly-writing helper."""
    ctx = get_active_write_context()
    if ctx is None:
        raise RunExecutionLockBoundaryViolationError(
            "begin_run_write called outside run_write_barrier"
        )
    ctx.mark_run_write_started()


def preliminary_terminal_seal_check(run_id: str, base_dir: Path | None) -> None:
    from htr.finalization import SealState, evaluate_run_seal
    from htr.state import RunFinalizedError, RunSealBlockedError

    evaluation = evaluate_run_seal(run_id, base_dir)
    if evaluation.state == SealState.FINALIZED_VALID:
        raise RunFinalizedError(run_id=run_id)
    if evaluation.state in (SealState.CLOSURE_PRESENT_UNTRUSTED, SealState.INDETERMINATE):
        raise RunSealBlockedError(
            run_id=run_id,
            reason_codes=evaluation.reason_codes,
        )


def _bootstrap_and_pin(base_dir: Path | None) -> tuple[int, int]:
    _require_platform()
    runs_path = _canonical_runs_root_path(base_dir)
    parts = runs_path.parts
    if not parts:
        raise RunExecutionLockPathUnsafeError("empty runs root path")

    index = len(parts) - 1
    while index > 0 and not Path(*parts[: index + 1]).is_dir():
        index -= 1
    anchor = Path(*parts[: index + 1])
    if not anchor.is_dir():
        raise RunExecutionLockPathUnsafeError(
            f"no existing ancestor for runs root {runs_path}"
        )

    parent_fd = _open_dir_no_follow(anchor)
    current_fd = parent_fd
    try:
        for component in parts[index + 1 :]:
            _mkdirat(current_fd, component, LOCK_ROOT_MODE)
            next_fd = _open_dirat_no_follow(current_fd, component)
            if current_fd != parent_fd:
                _close_fd_once(current_fd)
            current_fd = next_fd
        runs_root_fd = current_fd
        _mkdirat(runs_root_fd, LOCKS_DIR_NAME, LOCK_ROOT_MODE)
        lock_root_fd = _open_dirat_no_follow(runs_root_fd, LOCKS_DIR_NAME)
        _fsync_dir_fd(runs_root_fd)
        return runs_root_fd, lock_root_fd
    except Exception:
        if current_fd != parent_fd:
            _close_fd_once(current_fd)
        _close_fd_once(parent_fd)
        raise


def _stat_marker_present(lock_root_fd: int, marker_name: str) -> bool:
    try:
        os.stat(marker_name, dir_fd=lock_root_fd, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RunExecutionLockIndeterminateError(f"marker lookup failed: {exc}") from exc


def marker_present_noncreating(base_dir: Path | None, run_id: str) -> bool:
    runs_path = _canonical_runs_root_path(base_dir)
    lock_root_path = runs_path / LOCKS_DIR_NAME
    if not lock_root_path.is_dir():
        return False
    lock_root_fd = _open_dir_no_follow(lock_root_path)
    try:
        return _stat_marker_present(lock_root_fd, _marker_name(run_id))
    finally:
        _close_fd_once(lock_root_fd)


def _write_marker_metadata(fd: int, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    view = memoryview(encoded)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            raise RunExecutionLockIndeterminateError("marker metadata write stalled")
        offset += written


def _acquire_marker(lock_root_fd: int, run_id: str) -> tuple[int, str, str]:
    marker_name = _marker_name(run_id)
    with _marker_directory_entry_coordination(lock_root_fd):
        if _stat_marker_present(lock_root_fd, marker_name):
            raise RunExecutionLockOccupiedError(run_id=run_id)

        flags = _O_CREAT | _O_EXCL | _O_WRONLY | _O_NOFOLLOW | _O_CLOEXEC
        try:
            marker_fd = os.open(marker_name, flags, MARKER_MODE, dir_fd=lock_root_fd)
        except FileExistsError as exc:
            raise RunExecutionLockOccupiedError(run_id=run_id) from exc
        except OSError as exc:
            raise RunExecutionLockIndeterminateError(
                f"marker acquisition failed: {exc}", run_id=run_id
            ) from exc

        token = str(uuid.uuid4())
        metadata = {
            "schema_version": "1",
            "acquisition_id": token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "run_id": run_id,
        }
        try:
            _write_marker_metadata(marker_fd, metadata)
            _fsync_file_fd(marker_fd)
            _fsync_dir_fd(lock_root_fd)
        except Exception:
            _cleanup_owned_marker(lock_root_fd, marker_name, marker_fd)
            raise
        return marker_fd, token, marker_name


def _cleanup_owned_marker(lock_root_fd: int, marker_name: str, marker_fd: int) -> None:
    try:
        with _marker_directory_entry_coordination(lock_root_fd):
            _cleanup_owned_marker_locked(lock_root_fd, marker_name, marker_fd)
    except RunExecutionLockError:
        raise
    except OSError as exc:
        raise RunExecutionLockIndeterminateError(
            f"pre-write marker cleanup failed: {exc}"
        ) from exc
    finally:
        _close_fd_once(marker_fd)


def _require_entry(key: tuple[int, int, str], token: str) -> _RegistryEntry:
    with _registry_lock:
        entry = _registry.get(key)
        if entry is None:
            raise RunExecutionLockBoundaryViolationError("no active marker context")
        if entry.token != token:
            raise RunExecutionLockBoundaryViolationError("marker token mismatch")
        if entry.owner_pid != os.getpid():
            raise RunExecutionLockBoundaryViolationError("marker pid mismatch")
        if entry.owner_thread_id != threading.get_ident():
            raise RunExecutionLockBoundaryViolationError("marker thread mismatch")
        return entry


def _verify_entry_owner(entry: _RegistryEntry) -> None:
    if entry.owner_pid != os.getpid():
        raise RunExecutionLockBoundaryViolationError("marker pid mismatch")
    if entry.owner_thread_id != threading.get_ident():
        raise RunExecutionLockBoundaryViolationError("marker thread mismatch")


def _release_marker_success(entry: _RegistryEntry) -> None:
    _verify_entry_owner(entry)
    try:
        with _marker_directory_entry_coordination(entry.lock_root_fd):
            _release_marker_success_locked(entry)
    except RunExecutionLockError:
        raise
    except OSError as exc:
        raise RunExecutionLockDurabilityError(
            f"marker removal failed after successful body: {exc}",
            run_id=entry.key[2],
        ) from exc
    finally:
        _close_fd_once(entry.marker_fd)
        _close_fd_once(entry.lock_root_fd)


def _resolve_barrier_args(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    run_id_param: str,
    base_dir_param: str,
    project_as_base: bool,
) -> tuple[str, Path | None]:
    bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    bound.apply_defaults()
    run_id = bound.arguments[run_id_param]
    if project_as_base:
        raw = bound.arguments["project_dir"]
        base_dir = Path(raw) if raw is not None else None
    else:
        base_dir = bound.arguments.get(base_dir_param)
        if base_dir is not None and not isinstance(base_dir, Path):
            base_dir = Path(base_dir)
    return run_id, base_dir


def run_mutation_boundary(
    fn: F | None = None,
    *,
    run_id_param: str = "run_id",
    base_dir_param: str = "base_dir",
    project_as_base: bool = False,
) -> F | Callable[[F], F]:
    """Decorator: enter durable write barrier for run-aware public mutators."""

    def decorate(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            run_id, base_dir = _resolve_barrier_args(
                f,
                args,
                kwargs,
                run_id_param=run_id_param,
                base_dir_param=base_dir_param,
                project_as_base=project_as_base,
            )
            with run_write_barrier(run_id, base_dir) as wb:
                wb.revalidate_mutation_allowed()
                prior = getattr(_active_write_context, "current", None)
                _active_write_context.current = wb
                try:
                    return f(*args, **kwargs)
                finally:
                    _active_write_context.current = prior

        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorate(fn)
    return decorate


def _acquire_outer_run_marker(
    run_id: str,
    base_dir: Path | None,
) -> tuple[RunWriteContext, _RegistryEntry]:
    """Acquire run marker and register ownership. No lifecycle seal policy."""
    runs_root_fd, lock_root_fd = _bootstrap_and_pin(base_dir)
    key = _registry_key_from_fds(runs_root_fd, run_id)
    _close_fd_once(runs_root_fd)

    with _registry_lock:
        if key in _registry:
            _close_fd_once(lock_root_fd)
            raise RunExecutionLockOccupiedError(run_id=run_id)

    marker_fd, token, marker_name = _acquire_marker(lock_root_fd, run_id)
    entry = _RegistryEntry(
        key=key,
        token=token,
        marker_fd=marker_fd,
        lock_root_fd=lock_root_fd,
        owner_pid=os.getpid(),
        owner_thread_id=threading.get_ident(),
        marker_name=marker_name,
    )
    with _registry_lock:
        _registry[key] = entry
    ctx = RunWriteContext(
        run_id=run_id,
        base_dir=base_dir,
        key=key,
        token=token,
        is_outermost=True,
    )
    return ctx, entry


@contextmanager
def run_write_barrier(run_id: str, base_dir: Path | None = None) -> Iterator[RunWriteContext]:
    validate_id(run_id, "run")
    _require_platform()

    nested = _find_nested_entry(run_id)
    if nested is not None:
        nested.depth += 1
        ctx = RunWriteContext(
            run_id=run_id,
            base_dir=base_dir,
            key=nested.key,
            token=nested.token,
            is_outermost=False,
            run_write_started=nested.run_write_started,
        )
        try:
            from htr.finalization import assert_run_mutation_allowed

            assert_run_mutation_allowed(run_id, base_dir)
            yield ctx
        finally:
            nested.depth -= 1
            ctx.run_write_started = nested.run_write_started
        return

    active_other = _thread_active_entry()
    if active_other is not None and active_other.key[2] != run_id:
        raise RunExecutionLockBoundaryViolationError(
            "cross-key nested mutation is not allowed"
        )

    ctx: RunWriteContext | None = None
    entry: _RegistryEntry | None = None
    exc_info: BaseException | None = None
    try:
        preliminary_terminal_seal_check(run_id, base_dir)
        ctx, entry = _acquire_outer_run_marker(run_id, base_dir)
        from htr.finalization import assert_run_mutation_allowed

        assert_run_mutation_allowed(run_id, base_dir)
        yield ctx
    except BaseException as exc:
        exc_info = exc
        raise
    finally:
        if ctx is not None and entry is not None:
            _exit_outer_barrier(ctx, entry, exc=exc_info)


def _exit_outer_barrier(
    ctx: RunWriteContext,
    entry: _RegistryEntry,
    *,
    exc: BaseException | None,
) -> None:
    try:
        _verify_entry_owner(entry)
        if exc is not None:
            if not entry.run_write_started:
                _cleanup_owned_marker(entry.lock_root_fd, entry.marker_name, entry.marker_fd)
            return
        try:
            _release_marker_success(entry)
        except RunExecutionLockDurabilityError:
            raise
        except RunExecutionLockError as lock_exc:
            if entry.run_write_started:
                raise RunExecutionLockDurabilityError(
                    str(lock_exc), run_id=ctx.run_id
                ) from lock_exc
            raise
    finally:
        with _registry_lock:
            _registry.pop(ctx.key, None)


def require_closure_append_context(run_id: str, base_dir: Path | None) -> None:
    if not getattr(_closure_append_local, "active", False):
        raise RunExecutionLockBoundaryViolationError(
            "closure append outside first-final-closure marker"
        )
    entry = _thread_active_entry()
    if entry is None or entry.key[2] != run_id:
        raise RunExecutionLockBoundaryViolationError(
            "closure append requires active marker context"
        )
    if entry.owner_thread_id != threading.get_ident():
        raise RunExecutionLockBoundaryViolationError("closure append thread mismatch")
    if entry.depth <= 0:
        raise RunExecutionLockBoundaryViolationError("closure append depth invalid")
    if not entry.run_write_started:
        raise RunExecutionLockBoundaryViolationError(
            "closure append before run_write_started"
        )
