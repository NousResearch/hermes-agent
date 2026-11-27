"""Profile directory removal: wait out the live owner, then rmtree with a short retry."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import shutil
import stat
import time

import hermes_constants as _hermes_constants


def _delete_incomplete_on_old_constants(profile_home: str | Path) -> bool:
    """``hermes_constants.named_profile_delete_incomplete`` built on the names v0.23.3 already has: a
    current process may have left an ``incomplete`` record that an old one must not rmtree as a shell."""
    home = Path(profile_home)
    if not home.is_dir() or not _hermes_constants.named_profile_is_deleted(home):
        return False
    if _hermes_constants.named_profile_has_identity(home):
        return True
    try:
        return _hermes_constants.profile_tombstone_path(home).read_bytes().strip() != b"deleted"
    except OSError:
        return True


def _mark_incomplete(profile_home: str | Path) -> None:
    _hermes_constants.mark_named_profile_deleted(profile_home, incomplete=True)


def _mark_incomplete_on_old_constants(profile_home: str | Path) -> None:
    """v0.23.3's ``mark_named_profile_deleted`` has no ``incomplete`` and records a finished delete,
    which would let a later create rmtree what a failed delete left; write the ``incomplete`` record."""
    marker = _hermes_constants.profile_tombstone_path(Path(profile_home))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(b"incomplete\n")


# A v0.23.3 process keeps its hermes_constants and imports this module lazily after `hermes update`
# moved the checkout (#158723): these names resolve on its constants too.
named_profile_delete_incomplete = getattr(
    _hermes_constants, "named_profile_delete_incomplete", _delete_incomplete_on_old_constants)
mark_delete_incomplete = (_mark_incomplete if hasattr(_hermes_constants, "TOMBSTONE_INCOMPLETE")
                          else _mark_incomplete_on_old_constants)
_TOMBSTONE_DELETED = getattr(_hermes_constants, "TOMBSTONE_DELETED", b"deleted\n")

# A multiplexer answers a delete's rescan ``pending`` after 5 s and keeps tearing the profile down;
# its gateway.lock reservation is released last, once the profile's writers are gone.
OWNER_RELEASE_WAIT_SECONDS = 60.0


def delete_incomplete_message(canon: str, profile_dir: Path | None) -> str | None:
    """The one refusal every surface gives for a profile whose delete did not finish
    (``named_profile_delete_incomplete``), else None."""
    if profile_dir is None or not named_profile_delete_incomplete(profile_dir):
        return None
    return (f"Deletion of profile '{canon}' did not finish: {profile_dir} still holds its files. "
            f"Finish it with `hermes profile delete {canon}`, or keep it with "
            f"`hermes profile rename {canon} <new-name>`.")


def delete_failed_error(canon: str, profile_dir: Path, error: Exception) -> RuntimeError:
    """A delete whose rmtree was refused or failed; names the state it left and how to finish."""
    # delete_profile recorded the delete as incomplete before its rmtree, so files it left (even with
    # their identity files already gone) are never mistaken for a shell re-made after a finished delete.
    left = delete_incomplete_message(canon, profile_dir)
    return RuntimeError(f"Could not remove profile directory {profile_dir}: {error}" + (f". {left}" if left else ""))


def record_delete_finished(profile_dir: Path) -> None:
    """After a successful rmtree: a plain delete, so a shell a stale writer re-makes is replaceable.
    Rewrites the existing record only (no ``O_CREAT``): a same-name create in another process may already
    have cleared it, and recreating it would tombstone the new profile. Any other failure leaves the record
    ``incomplete``: a shell a stale writer re-makes then lists as delete incomplete, so say how to finish."""
    try:
        fd = os.open(_hermes_constants.profile_tombstone_path(profile_dir), os.O_WRONLY | os.O_TRUNC)
        try:
            os.write(fd, _TOMBSTONE_DELETED)
        finally:
            os.close(fd)
    except FileNotFoundError:
        pass  # cleared by a same-name create: the name is that profile's now
    except OSError as e:
        print(f"⚠ Could not record the delete of {profile_dir} as finished: {e}. If "
              f"'{profile_dir.name}' later lists as delete incomplete, finish it with "
              f"`hermes profile delete {profile_dir.name}`.")


def wait_profile_unowned(profile_dir: Path, *, timeout: float | None = None) -> bool:
    """True once no gateway runtime holds ``profile_dir``'s ``gateway.lock`` (taken and released
    through ``exclusive_maintenance``); False when it is still held after ``timeout`` seconds."""
    from gateway.runtime_ownership import OwnershipConflict, exclusive_maintenance
    deadline = time.monotonic() + (OWNER_RELEASE_WAIT_SECONDS if timeout is None else timeout)
    while True:
        try:
            with exclusive_maintenance([profile_dir]):
                return True
        except OwnershipConflict:
            pass
        except OSError:
            # A read-only copy (0555 home, 0444 lock) cannot host a reservation: probe it instead.
            if _lock_unheld_read_only(profile_dir / 'gateway.lock'):
                return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _lock_unheld_read_only(path: Path) -> bool:
    """A missing lock, or one our own flock through a read-only descriptor wins, has no holder."""
    if os.name == 'nt':
        return False  # a sharing violation there is a live holder
    import fcntl
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | os.O_NONBLOCK)
    except FileNotFoundError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def _rmtree_make_writable(func, path, exc):
    """onexc/onerror handler: add +w on PermissionError so rmtree can proceed. Covers NixOS-
    style read-only copies where the path itself (0444) or its parent (0555) isn't writable."""
    # onexc(func, path, exc_instance) on 3.12+; onerror(func, path, exc_info_tuple) on 3.11.
    if isinstance(exc, tuple):
        exc = exc[1]
    if not isinstance(exc, PermissionError):
        raise
    for target in (path, os.path.dirname(path)):  # parent needed for unlink/rmdir
        if target:
            with contextlib.suppress(OSError):
                os.chmod(target, os.stat(target).st_mode | stat.S_IWUSR)
    func(path)


def _rmtree_with_retry(profile_dir: Path, onexc_handler) -> None:
    """``shutil.rmtree`` with a short retry loop: a just-terminated process can leave in-flight
    writes (SQLite -wal/-shm checkpoints, sandbox temp files) landing after rmtree walked
    past a directory — ENOTEMPTY on POSIX, transient PermissionError on Windows."""
    attempts = 3
    last_exc: OSError | None = None
    for attempt in range(attempts):
        try:
            try:
                shutil.rmtree(profile_dir, onexc=onexc_handler)
            except TypeError:  # ``onexc`` is 3.12+; 3.11 has ``onerror``
                shutil.rmtree(profile_dir, onerror=onexc_handler)
            return
        except OSError as e:
            last_exc = e
            if not profile_dir.exists():
                return
            if attempt < attempts - 1:
                time.sleep(0.3 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
