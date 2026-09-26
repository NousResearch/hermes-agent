"""Cross-process lifecycle authority for reusable named-profile paths.

Named profile names can be deleted and recreated, so a pathname alone cannot
identify the generation a deferred operation originally resolved. This module
owns the mutation lease, durable deletion tombstones, in-process retirement,
and external-holder proof used by profile create/delete/import/rename flows.
It deliberately does not import ``hermes_cli.profiles``; command wiring may
depend on this owner, never the reverse.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import wraps
import hashlib
import inspect
import logging
import os
from pathlib import Path
import secrets
import shutil
import stat
import sys
import threading
import time
from typing import Callable, Iterator

from hermes_constants import get_default_hermes_root, profile_deletion_marker_path
from pm.filesystem import lock_fd

logger = logging.getLogger(__name__)

# Each reusable name owns a stable lock outside its directory. Never unlink
# these files: waiters must keep locking the same inode across delete/recreate.
_PROFILE_MUTATION_LOCAL = threading.local()
_PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS = 120.0
_PROFILE_DB_RELEASE_TIMEOUT_SECONDS = 5.0


def _profiles_root() -> Path:
    return get_default_hermes_root() / "profiles"


def _profile_lifecycle_modes(profiles_root: Path | None = None) -> tuple[int, int]:
    """Return directory/file modes compatible with managed shared roots."""
    try:
        mode = (profiles_root or _profiles_root()).stat().st_mode
    except OSError:
        mode = 0
    directory, file = (0o770, 0o660) if mode & stat.S_IWGRP else (0o700, 0o600)
    return directory | (mode & stat.S_ISGID), file


def _profile_lock_path(profile_home: Path | str) -> Path | None:
    marker = profile_deletion_marker_path(Path(profile_home))
    if marker is None:
        return None
    # Resolve the parent, not the reusable profile directory (which can move).
    return marker.parent.parent.resolve() / ".locks" / (marker.name + ".lock")


@contextmanager
def _cross_process_profile_mutation_lock(lock_path: Path, *, deadline: float) -> Iterator[None]:
    """Acquire one stable lock file without any process-wide mutex held."""
    directory_mode, file_mode = _profile_lifecycle_modes(lock_path.parent.parent)
    lock_path.parent.mkdir(mode=directory_mode, parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        # Every lease (read paths included) lands here: stat first so a settled
        # lock touches no metadata.
        for path, mode in ((lock_path, file_mode), (lock_path.parent, directory_mode)):
            if stat.S_IMODE(os.stat(path).st_mode) != mode:
                os.chmod(path, mode)
    except OSError:
        pass
    acquired = False
    try:
        if not lock_fd(handle.fileno(), wait=True, timeout=max(0.0, deadline - time.monotonic())):
            raise TimeoutError(
                f"Timed out waiting for profile lifecycle lock: {lock_path}; retry the operation."
            )
        acquired = True
        yield
    finally:
        if acquired:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


@contextmanager
def _lifecycle_leases(lock_paths: list[Path], timeout: float | None = None) -> Iterator[None]:
    deadline = time.monotonic() + (
        _PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS if timeout is None else timeout
    )
    # Bookkeeping is thread-local; a global RLock would hide the deadline while
    # another thread waits on the OS lock. Nested calls reuse only their OWN names.
    if getattr(_PROFILE_MUTATION_LOCAL, "pid", None) != os.getpid():
        _PROFILE_MUTATION_LOCAL.pid = os.getpid()
        _PROFILE_MUTATION_LOCAL.held = set()
    held = _PROFILE_MUTATION_LOCAL.held
    paths = {os.path.normcase(str(path)): path for path in lock_paths}
    acquiring = sorted(paths.keys() - held)
    if held and acquiring and acquiring[0] < max(held):
        raise RuntimeError("Profile lifecycle lock order inversion; acquire all profile homes together.")
    with ExitStack() as stack:
        for key in acquiring:
            stack.enter_context(_cross_process_profile_mutation_lock(paths[key], deadline=deadline))
            held.add(key)
            stack.callback(held.remove, key)
        yield


@contextmanager
def profile_lifecycle_lease(
    profile_home: Path | str, *other_homes: Path | str, timeout: float | None = None,
) -> Iterator[None]:
    """Exclude mutation of these names; default/custom/staging homes stay lock-free.

    Acquire all rename homes together in sorted order, then gateway
    session/resource locks. Re-entry for already held homes is allowed; adding
    a lower-ordered name is rejected instead of deadlocking. One deadline covers
    the entire acquisition, including contention with threads in this process.
    Every cold SessionDB open takes its name's lease, so never hold one across
    unbounded work: a clone leases only its target and pins the source by
    incarnation, re-checked before publication.
    """
    paths = [path for home in (profile_home, *other_homes)
             if (path := _profile_lock_path(home)) is not None]
    with _lifecycle_leases(paths, timeout):
        yield


@contextmanager
def profile_shared_file_lease(path: Path) -> Iterator[None]:
    """Protect profile-mutation RMWs whose file is shared across names.

    These short leases sort after names and before sticky selection. They
    never enclose new profile acquisitions or expensive copy/retirement work.
    """
    key = hashlib.sha256(os.path.normcase(str(path.resolve())).encode()).hexdigest()
    with _lifecycle_leases([_profiles_root().resolve() / ".locks" / f"~file-{key}.lock"]):
        yield


@contextmanager
def profile_selection_lease() -> Iterator[None]:
    """Serialize only the shared sticky-selection read/modify/write.

    This sorts after named-profile locks and must never enclose a new profile
    acquisition. It is not held across copy, retirement or database binding.
    """
    with _lifecycle_leases([_profiles_root().resolve() / ".locks" / "~selection.lock"]):
        yield


def serialized_profile_mutation(*path_parameters: str):
    """Lease the explicit path arguments of a profile mutation."""
    def decorate(func):
        signature = inspect.signature(func)

        @wraps(func)
        def wrapped(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            with profile_lifecycle_lease(*(arguments[name] for name in path_parameters)):
                return func(*args, **kwargs)

        return wrapped
    return decorate


def profile_deletion_marker(profile_dir: Path | str) -> Path:
    """Return the durable tombstone path for a named profile home."""
    marker = profile_deletion_marker_path(Path(profile_dir))
    if marker is None:
        raise ValueError(f"Not a named profile home: {profile_dir}")
    return marker


def mark_profile_deleting(
    profile_dir: Path | str,
    profile_incarnation: str | None = None,
) -> Path:
    """Publish a durable cross-process guard before tearing a profile down."""
    profile_dir = Path(profile_dir)
    marker = profile_deletion_marker(profile_dir)
    directory_mode, file_mode = _profile_lifecycle_modes()
    marker.parent.mkdir(mode=directory_mode, parents=True, exist_ok=True)
    try:
        marker.parent.chmod(directory_mode)
    except OSError:
        pass
    if profile_incarnation is None:
        marker.touch(exist_ok=True)
    else:
        temp = marker.with_name(
            f".{marker.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
        )
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, file_mode)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(profile_incarnation + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, marker)
        finally:
            temp.unlink(missing_ok=True)
    try:
        marker.chmod(file_mode)
    except OSError:
        pass
    return marker


def clear_profile_deletion_marker(profile_dir: Path | str) -> None:
    """Remove a named profile's durable tombstone after publication/rollback."""
    marker = profile_deletion_marker(profile_dir)
    marker.unlink(missing_ok=True)
    # Other names publish tombstones independently. Removing their shared
    # parent can race another publisher between mkdir and atomic replace.


def profile_home_is_tombstoned(profile_dir: Path | str) -> bool:
    """Return whether profile deletion has been committed for this home."""
    marker = profile_deletion_marker_path(Path(profile_dir))
    return marker is not None and marker.is_file()


def retire_in_process_profile_resources(
    profile_dir: Path | str,
    profile_incarnation: str | None = None,
) -> int:
    """Close current-process sessions/caches retaining a profile home."""
    profile_dir = Path(profile_dir)
    retired = 0
    retire_error: Exception | None = None
    gateway_server = sys.modules.get("tui_gateway.server")
    retire_sessions = getattr(gateway_server, "retire_profile_home", None)
    if callable(retire_sessions):
        try:
            result = retire_sessions(
                profile_dir,
                profile_incarnation=profile_incarnation,
            )
            if isinstance(result, int) and not isinstance(result, bool):
                retired += max(0, result)
        except Exception as exc:
            logger.debug("Failed to retire in-process profile sessions", exc_info=True)
            retire_error = exc

    goals_module = sys.modules.get("hermes_cli.goals")
    release_goals_db = getattr(goals_module, "release_session_db_for_home", None)
    if callable(release_goals_db):
        try:
            retired += int(bool(release_goals_db(profile_dir)))
        except Exception:
            logger.debug("Failed to release cached profile SessionDB", exc_info=True)

    try:
        from plugins.memory import import_provider_module

        memory_store = import_provider_module("holographic", "store").MemoryStore
        retired += max(0, int(memory_store.release_all_under(profile_dir) or 0))
    except Exception:
        logger.debug("Failed to release profile memory-store connections", exc_info=True)
    if retire_error is not None:
        raise retire_error
    return retired


def allow_in_process_profile_resources(
    profile_dir: Path | str,
    profile_incarnation: str | None = None,
) -> None:
    """Admit a newly published profile generation to process-local owners."""
    profile_dir = Path(profile_dir)
    gateway_server = sys.modules.get("tui_gateway.server")
    allow_profile = getattr(gateway_server, "allow_profile_home", None)
    if callable(allow_profile):
        try:
            allow_profile(
                profile_dir,
                profile_incarnation=profile_incarnation,
            )
        except Exception:
            logger.debug("Failed to admit recreated profile home", exc_info=True)


def wait_for_profile_state_db_release(profile_dir: Path | str) -> bool:
    """Wait briefly for tracked in-process state.db handles to close."""
    from hermes_cli.sqlite_safe_read import has_live_connection

    db_path = Path(profile_dir) / "state.db"
    deadline = time.monotonic() + _PROFILE_DB_RELEASE_TIMEOUT_SECONDS
    while has_live_connection(db_path):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def external_profile_file_holders(profile_dir: Path | str) -> list[int]:
    """Same-user external PIDs with an open file under ``profile_dir``."""
    profile_dir = Path(profile_dir)
    try:
        import psutil  # type: ignore
    except Exception:
        return []
    try:
        root = profile_dir.resolve()
    except OSError:
        root = profile_dir
    try:
        current_user = psutil.Process(os.getpid()).username()
    except Exception:
        current_user = None

    holders: list[int] = []
    for proc in psutil.process_iter(["pid", "username", "open_files"]):
        try:
            info = proc.info
            pid = info.get("pid")
            if not isinstance(pid, int) or pid == os.getpid():
                continue
            process_user = info.get("username")
            if (
                current_user is not None
                and process_user is not None
                and process_user != current_user
            ):
                continue
            files = info.get("open_files")
            if files is None:
                files = proc.open_files()
            for opened in files or []:
                raw = getattr(opened, "path", "")
                if not raw:
                    continue
                try:
                    path = Path(raw).resolve()
                except OSError:
                    path = Path(raw)
                if path == root or root in path.parents:
                    holders.append(pid)
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            continue
    return holders


def wait_for_external_profile_file_release(profile_dir: Path | str) -> list[int]:
    """Return remaining external holders after a bounded release grace."""
    deadline = time.monotonic() + _PROFILE_DB_RELEASE_TIMEOUT_SECONDS
    while True:
        holders = external_profile_file_holders(profile_dir)
        if not holders or time.monotonic() >= deadline:
            return holders
        time.sleep(0.05)


def publish_profile_generation(
    profile_dir: Path | str,
    incarnation: str | None,
) -> None:
    """Publish a fully initialized named generation to in-process readers."""
    if incarnation is None:
        raise RuntimeError(f"Published profile has no incarnation: {profile_dir}")
    clear_profile_deletion_marker(profile_dir)
    allow_in_process_profile_resources(profile_dir, incarnation)


def create_profile_generation(
    canon: str,
    profile_dir: Path | str,
    profiles_root: Path | str,
    initialize: Callable[[Path], None],
) -> Path:
    """Build a generation behind a tombstone and atomically publish it."""
    profile_dir = Path(profile_dir)
    profiles_root = Path(profiles_root)
    if profile_dir.exists():
        raise FileExistsError(f"Profile '{canon}' already exists at {profile_dir}")

    prior_tombstone = profile_home_is_tombstoned(profile_dir)
    mark_profile_deleting(profile_dir)
    staging_root = profiles_root / ".profile-creating"
    staging_parent = staging_root / f"{canon}-{os.getpid()}-{secrets.token_hex(6)}"
    staging_dir = staging_parent / canon
    moved_to_final = False
    incarnation: str | None = None
    try:
        staging_parent.mkdir(parents=True, exist_ok=False)
        initialize(staging_dir)
        if profile_dir.exists():
            raise FileExistsError(f"Profile '{canon}' appeared during initialization")
        os.replace(staging_dir, profile_dir)
        moved_to_final = True
        from hermes_cli.profile_incarnation import read_profile_incarnation

        incarnation = read_profile_incarnation(profile_dir)
        publish_profile_generation(profile_dir, incarnation)
    except Exception:
        cleanup_complete = not moved_to_final and not profile_dir.exists()
        if moved_to_final and profile_dir.exists():
            try:
                shutil.rmtree(profile_dir)
                cleanup_complete = True
            except Exception:
                logger.exception("Could not remove unpublished profile %s", profile_dir)
        if not prior_tombstone and cleanup_complete:
            clear_profile_deletion_marker(profile_dir)
        raise
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)
        # Keep the shared staging root: another name may be between its
        # parent mkdir and attempt-directory mkdir.
    return profile_dir


def import_profile_generation(
    canon: str,
    profile_dir: Path | str,
    build_and_move: Callable[[], str],
) -> Path:
    """Publish an imported generation while preserving prior tombstone state."""
    profile_dir = Path(profile_dir)
    if profile_dir.exists():
        raise FileExistsError(f"Profile '{canon}' already exists at {profile_dir}")
    had_tombstone = profile_home_is_tombstoned(profile_dir)
    mark_profile_deleting(profile_dir)
    try:
        incarnation = build_and_move()
    except Exception:
        if not had_tombstone and not profile_dir.exists():
            clear_profile_deletion_marker(profile_dir)
        raise
    publish_profile_generation(profile_dir, incarnation)
    return profile_dir


def rollback_profile_retirement(
    profile_dir: Path | str,
    profile_incarnation: str | None,
) -> None:
    """Restore admission when retirement fails before pathname mutation."""
    clear_profile_deletion_marker(profile_dir)
    allow_in_process_profile_resources(profile_dir, profile_incarnation)


def begin_profile_retirement(
    profile_dir: Path | str,
    profile_incarnation: str | None,
    *,
    rollback_on_failure: bool = True,
) -> int:
    """Tombstone a generation and retire every process-local owner."""
    mark_profile_deleting(profile_dir, profile_incarnation)
    try:
        return retire_in_process_profile_resources(profile_dir, profile_incarnation)
    except Exception:
        if rollback_on_failure:
            rollback_profile_retirement(profile_dir, profile_incarnation)
        raise


def verify_profile_resources_released(
    profile_dir: Path | str,
    profile_incarnation: str | None,
    *,
    subject: str,
    retry_action: str,
    rollback_on_failure: bool = True,
) -> None:
    """Prove holders drained, optionally rolling back this attempt's fence."""
    if not wait_for_profile_state_db_release(profile_dir):
        if rollback_on_failure:
            rollback_profile_retirement(profile_dir, profile_incarnation)
        raise RuntimeError(
            f"{subject} is still in use by this Hermes process; retry {retry_action}."
        )
    external_holders = wait_for_external_profile_file_release(profile_dir)
    if external_holders:
        if rollback_on_failure:
            rollback_profile_retirement(profile_dir, profile_incarnation)
        raise RuntimeError(
            f"{subject} is still in use by external process(es) "
            f"{', '.join(str(pid) for pid in external_holders)}; retry {retry_action}."
        )


def move_profile_generation(
    old_dir: Path | str,
    new_dir: Path | str,
    profile_incarnation: str | None,
    after_move: Callable[[], None],
) -> Path:
    """Move a retired generation and publish only its new pathname."""
    old_dir = Path(old_dir)
    new_dir = Path(new_dir)
    new_had_tombstone = profile_home_is_tombstoned(new_dir)
    mark_profile_deleting(new_dir)
    try:
        old_dir.rename(new_dir)
    except Exception:
        rollback_profile_retirement(old_dir, profile_incarnation)
        if not new_had_tombstone:
            clear_profile_deletion_marker(new_dir)
        raise
    try:
        after_move()
    finally:
        publish_profile_generation(new_dir, profile_incarnation)
    return new_dir
