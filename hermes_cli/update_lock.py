"""Cross-process mutual exclusion for in-flight Hermes updates.

The marker file the Tauri updater writes (``UpdateMarkerGuard`` in
``apps/bootstrap-installer/src-tauri/src/update.rs``) and the Electron desktop reads
(``electron/update-marker.ts``) is the single lock for **all** update entrypoints.
Format and location are byte-compatible with both readers.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from hermes_mcp_update_gate import (
    MARKER_NAME,
    UPDATE_MARKER_MAX_AGE_SECONDS,
    UpdateMarkerError,
    UpdateMarkerUnhealthyError,
    live_update_marker_owner,
    pid_alive,
    pid_create_time as _stdlib_pid_create_time,
)

logger = logging.getLogger(__name__)

__all__ = ["MARKER_NAME", "UPDATE_MARKER_MAX_AGE_SECONDS", "UpdateMarkerError", "UpdateLock",
           "UpdateHolder", "UPDATE_EXIT_CONCURRENT", "HANDOFF_PID_ENV", "describe_holder",
           "read_live_update", "update_marker_path"]

# A body that names nobody may be a rewrite caught in flight. Observe the exact
# same bytes across this dwell before reclaiming, so a real writer always wins.
UPDATE_MARKER_DWELL_SECONDS = 2.0

# Set by an orchestrating updater (Tauri `hermes-setup --update`) to its own pid before
# spawning `hermes update` as a child stage; the parent holds the marker for its whole run,
# so without this the child would refuse its own parent's lock. The writer is
# `update_child_env` in apps/bootstrap-installer/src-tauri/src/update.rs (with its own test,
# `update_child_env_names_our_pid_for_the_lock_handoff`) -- this is a live production
# channel, not a vestigial one, and removing the adoption path here dead-ends every GUI
# update on exit 2. Adoption still requires the named pid to be the marker's `matching`
# owner, so naming a pid grants nothing on its own.
HANDOFF_PID_ENV = "HERMES_UPDATE_HANDOFF_PID"

# Exit code meaning "another updater/instance owns this install right now" — the same
# contract as the Windows shim / venv-holder guards in _cmd_update_impl, matched by the
# Tauri updater (UPDATE_EXIT_CONCURRENT in update.rs) to show "Hermes is still running".
UPDATE_EXIT_CONCURRENT = 2


def update_marker_path() -> Path:
    """Path of the shared update marker.

    Uses the *process* Hermes home (never the context-local profile override): the Rust
    updater resolves ``$HERMES_HOME`` or the platform default and the desktop pins that same
    value into the updater's env, so a profile-scoped path would be one the other owners never look at.
    """
    from hermes_constants import get_process_hermes_home
    return get_process_hermes_home() / MARKER_NAME


def _pid_create_time(pid: int) -> float | None:
    if os.name == "nt":
        return _stdlib_pid_create_time(pid)
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _handoff_pid() -> int | None:
    """Pid of the orchestrating updater that spawned us (:data:`HANDOFF_PID_ENV`); malformed
    values count as absent so a broken handoff falls back to the normal refusal."""
    try:
        pid = int(os.environ.get(HANDOFF_PID_ENV, "").strip())
    except ValueError:
        return None
    return pid if pid > 0 else None


def _is_ancestor_pid(pid: int) -> bool:
    """True when ``pid`` is a live ancestor of this process.

    The orchestrating updater spawns ``hermes update`` as a (grand)child, so a live marker
    owned by an ancestor can only be the claim we already run under — an unrelated concurrent
    updater is never in our parent chain. Never our own pid; any failure is "not an ancestor".
    """
    if pid <= 0:
        return False
    try:
        import psutil
        return any(parent.pid == pid for parent in psutil.Process().parents())
    except Exception as exc:
        logger.debug("Could not walk process ancestry for pid %s: %s", pid, exc)
        return False


@dataclass(frozen=True)
class UpdateHolder:
    """A live or unprovable owner currently holding the lock."""

    pid: int
    age_seconds: float
    identity: str = "unknown"


def _restore_marker(marker: Path, isolated: Path) -> None:
    """Restore without overwriting a successor; unresolved artifacts stay blocking."""
    try:
        os.link(isolated, marker)
    except FileExistsError:
        pass
    except OSError as exc:
        raise UpdateMarkerError("Cannot restore the isolated update marker") from exc
    isolated.unlink()


def _remove_exact_marker(marker: Path, expected: str) -> None:
    isolated = marker.with_name(f"{marker.name}.cas-release-{os.getpid()}-{uuid.uuid4()}")
    try:
        marker.rename(isolated)
    except FileNotFoundError:
        return
    try:
        if isolated.read_text(encoding="utf-8") != expected:
            _restore_marker(marker, isolated)
            return
        isolated.unlink()
    except (OSError, UnicodeError) as exc:
        if isolated.exists():
            _restore_marker(marker, isolated)
        raise UpdateMarkerError("Cannot reclaim the update marker safely") from exc


def read_live_update(
    *,
    path: Path | None = None,
    dwell_seconds: float = UPDATE_MARKER_DWELL_SECONDS,
    sleep=None,
) -> UpdateHolder | None:
    """Read the shared claim, reclaiming a proven stale or unowned exact marker.

    A marker we cannot READ still raises UpdateMarkerError and callers must
    stop. A marker we read fine and found to name nobody -- empty, malformed,
    future-dated, the residue of a torn write -- is reclaimed after a dwell
    instead of blocking every update from then on (upstream main unlinks these
    outright at :125; the PR removed that and left no way back).
    """
    marker = path or update_marker_path()
    _sleep = sleep or time.sleep
    for _ in range(3):
        try:
            artifacts = list(marker.parent.glob(marker.name + ".cas-*"))
            if artifacts:
                if len(artifacts) == 1 and artifacts[0].name.startswith(marker.name + ".cas-release-") and not marker.exists():
                    releaser = artifacts[0].name[len(marker.name + ".cas-release-"):].split("-", 1)[0]
                    # A live releaser owns the rename/read/unlink transaction.
                    # Recover only an abandoned release, never resurrect one
                    # that another process is still removing.
                    if releaser.isdecimal() and int(releaser) > 0 and not pid_alive(int(releaser)):
                        _restore_marker(marker, artifacts[0])
                        continue
                raise UpdateMarkerError(
                    f"Update marker cleanup is unresolved; retry after the updater finishes ({artifacts[0]})"
                )
            raw = marker.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError) as exc:
            raise UpdateMarkerError(f"Cannot read the update marker safely: {marker}") from exc
        try:
            owner = live_update_marker_owner(marker, pid_alive=pid_alive, pid_create_time=_pid_create_time)
        except UpdateMarkerUnhealthyError:
            # Dwell, then re-read. _remove_exact_marker is already an exact
            # content CAS, so an owner that finished writing inside the dwell
            # keeps its claim and we simply loop again.
            _sleep(dwell_seconds)
            try:
                if marker.read_text(encoding="utf-8") != raw:
                    continue
            except FileNotFoundError:
                return None
            except (OSError, UnicodeError) as exc:
                raise UpdateMarkerError(f"Cannot read the update marker safely: {marker}") from exc
            logger.warning("Reclaiming an update marker that names no owner: %s", marker)
            _remove_exact_marker(marker, raw)
            continue
        if owner is not None:
            return UpdateHolder(owner.pid, owner.age_seconds, owner.identity)
        _remove_exact_marker(marker, raw)
    raise UpdateMarkerError(f"Update marker changed during reclamation; retry ({marker})")


def describe_holder(holder: UpdateHolder) -> str:
    """One-line, user-facing explanation of who holds the update lock."""
    minutes, seconds = divmod(int(max(holder.age_seconds, 0)), 60)
    elapsed = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    return (
        f"✗ Another Hermes update is already running (PID {holder.pid}, "
        f"started {elapsed} ago).\n"
        "\n"
        "  Two updates mutating the same checkout corrupt it: one rewrites\n"
        "  source while the other is mid-install. Wait for it to finish, or\n"
        "  close the window/dashboard tab that started it, then retry."
    )


class UpdateLock:
    """Context manager owning the shared update marker for this process.

    ``acquired`` is False when another live update holds it; callers decide between hard
    refusal (CLI/dashboard) and waiting. Release only removes the marker when *we* still own
    it, so a marker rewritten by a handoff partner (the Tauri updater writes its own pid) is
    never deleted from under its new owner.
    """

    def __init__(self, *, path: Path | None = None) -> None:
        self.path = path or update_marker_path()
        self.acquired = False
        self.holder: UpdateHolder | None = None

    def acquire(self) -> bool:
        """Claim the lock. Returns False (and sets ``holder``) if it's taken.

        A live holder whose pid matches :data:`HANDOFF_PID_ENV` — or is an ancestor of ours —
        is our own orchestrating parent: run under ITS claim and leave its marker untouched on
        release. The ancestry path covers staged updaters older than the env-var export.
        """
        existing = read_live_update(path=self.path)
        if existing is not None:
            if existing.identity == "matching" and (existing.pid == _handoff_pid() or _is_ancestor_pid(existing.pid)):
                return True
            self.holder = existing
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise UpdateMarkerError("Cannot create the update marker directory") from exc
        try:
            with self.path.open("x", encoding="utf-8") as claim:
                self._claim = f"{os.getpid()}\n{int(time.time())}\n"
                claim.write(self._claim)
                claim.flush()
                os.fsync(claim.fileno())
        except FileExistsError:
            self.holder = read_live_update(path=self.path)
            if self.holder is None:
                raise UpdateMarkerError("Update marker changed during acquisition; retry")
            return False
        except OSError as exc:
            raise UpdateMarkerError("Cannot claim the update marker; installation was not changed") from exc
        if list(self.path.parent.glob(self.path.name + ".cas-*")):
            _remove_exact_marker(self.path, self._claim)
            raise UpdateMarkerError("Update marker cleanup started during acquisition; retry")
        self.acquired = True
        return True

    def release(self) -> None:
        """Drop the marker if this process still owns it. Never raises."""
        if not self.acquired:
            return
        self.acquired = False
        try:
            _remove_exact_marker(self.path, self._claim)
        except (OSError, UpdateMarkerError):
            logger.warning("Could not release update marker %s", self.path)

    def __enter__(self) -> "UpdateLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
