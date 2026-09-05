"""Cross-process lifecycle authority for reusable named-profile paths.

Named profile names can be deleted and recreated, so a pathname alone cannot
identify the generation a deferred operation originally resolved. This module
owns the mutation lease, durable deletion tombstones, in-process retirement,
and external-holder proof used by profile create/delete/import/rename flows.
It deliberately does not import ``hermes_cli.profiles``; command wiring may
depend on this owner, never the reverse.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
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

logger = logging.getLogger(__name__)

# Named-profile resource binders take the same lock only while turning a
# checked pathname into an open handle or file write. Ordinary observations
# remain lock-free.
_PROFILE_LIFECYCLE_LOCK = threading.RLock()
_PROFILE_MUTATION_LOCAL = threading.local()
_PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS = 120.0
_PROFILE_DB_RELEASE_TIMEOUT_SECONDS = 5.0


def _profiles_root() -> Path:
    return get_default_hermes_root() / "profiles"


def _profile_lifecycle_modes() -> tuple[int, int]:
    """Return directory/file modes compatible with managed shared roots."""
    try:
        group_shared = bool(_profiles_root().stat().st_mode & stat.S_IWGRP)
    except OSError:
        group_shared = False
    return (0o770, 0o660) if group_shared else (0o700, 0o600)


@contextmanager
def _cross_process_profile_mutation_lock() -> Iterator[None]:
    """Serialize create/delete/rename across Hermes processes on this host."""
    lock_path = _profiles_root() / ".profile-lifecycle.lock"
    _, file_mode = _profile_lifecycle_modes()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        os.chmod(lock_path, file_mode)
    except OSError:
        pass
    acquired = False
    try:
        deadline = time.monotonic() + _PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    handle.seek(0, os.SEEK_END)
                    if handle.tell() == 0:
                        handle.write(b"\0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for profile lifecycle lock: {lock_path}"
                    )
                time.sleep(0.05)
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
def profile_lifecycle_lease() -> Iterator[None]:
    """Exclude profile mutations while binding one named-profile resource.

    Lock order is lifecycle first, then any gateway session/resource lock.
    Nested profile operations on the same thread reuse the outer cross-process
    lease so create/import/delete helpers remain re-entrant.
    """
    with _PROFILE_LIFECYCLE_LOCK:
        depth = int(getattr(_PROFILE_MUTATION_LOCAL, "depth", 0))
        if depth:
            _PROFILE_MUTATION_LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _PROFILE_MUTATION_LOCAL.depth = depth
            return
        with _cross_process_profile_mutation_lock():
            _PROFILE_MUTATION_LOCAL.depth = 1
            try:
                yield
            finally:
                _PROFILE_MUTATION_LOCAL.depth = 0


def serialized_profile_mutation(func):
    """Serialize a profile mutation through the shared lifecycle lease."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        with profile_lifecycle_lease():
            return func(*args, **kwargs)

    return wrapped


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
    try:
        marker.parent.rmdir()
    except OSError:
        pass


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
        from plugins.memory.holographic.store import MemoryStore

        retired += max(0, int(MemoryStore.release_all_under(profile_dir) or 0))
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
    initialize: Callable[[Path], Path],
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
        created = initialize(staging_dir)
        if created != staging_dir:
            raise RuntimeError(f"Profile initialization targeted unexpected path: {created}")
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
        try:
            staging_root.rmdir()
        except OSError:
            pass
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
) -> None:
    """Prove local/external holders drained or roll retirement back."""
    if not wait_for_profile_state_db_release(profile_dir):
        rollback_profile_retirement(profile_dir, profile_incarnation)
        raise RuntimeError(
            f"{subject} is still in use by this Hermes process; retry {retry_action}."
        )
    external_holders = wait_for_external_profile_file_release(profile_dir)
    if external_holders:
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
